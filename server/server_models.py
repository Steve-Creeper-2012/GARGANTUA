# ===========================================================================
# GARGANTUA v1 — server_models.py：模型注册表 + 异步任务池
# ---------------------------------------------------------------------------
# 2026-10-24 重构（steve：服务按需启动，省内存）：
#   ModelManager 从「权重加载器」降级为「注册表」——只跟踪哪个模型是活动
#   模型、它的 config.json 里有什么。模型权重（单个 safetensors 可达数 GB）
#   只存在于 infer/train 服务进程里（见 server_services.py）；server 本体
#   永不加载权重，切模型从「读 2.1GB」变成「改个名字」。
#
#   ModelManager   同一时刻一个活动模型（登记簿，不含权重）
#   TaskPool       有界异步任务池（202 + task_id 长轮询的执行体）
#   step_to_json   codec 产物 Step → JSON（bytes 字段 base64）
#   ConflictError  状态冲突（训练中切模型/覆盖活动模型）→ HTTP 409
# server.py 继续 re-export 这些符号，老 import 路径全部保持可用。
# ===========================================================================
import base64
import glob
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import spec
from common import (CKPT_NAME, model_exists, read_config, read_ctx_config,
                    valid_model_name)


class ConflictError(RuntimeError):
    """状态冲突：训练中切换/创建模型，或加载被更新的加载请求取代。"""


class ModelManager:
    """同一时刻一个活动模型——但只登记「名字 + config」，不持有权重。

    权重在服务进程里（infer/train 按需拉起）。因此：
      load()      即时返回（读 config.json，不读 safetensors）；
      self.model  恒为 None（保留属性是为兼容旧代码/测试打桩）；
      切模型不再与正在跑的服务抢内存。
    """

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.lock = threading.RLock()
        self.model = None                 # 恒为 None：权重不在本进程
        self.name = None                  # 活动模型名（登记簿）
        self.params = 0                   # 来自 config.json（非实测）
        self.ctx = None                   # 活动模型的上下文长度配置
        os.makedirs(model_dir, exist_ok=True)

    def ctx_note(self) -> str:
        """活动模型的 KV 配置摘要（无活动模型时回退 spec 默认）。
        2026-10-23：上下文长度建模时可配，status 必须反映真实值。 """
        c = self.ctx or {"ctx_full": spec.CTX_FULL,
                         "ctx_compress": spec.CTX_COMPRESS,
                         "csa_ratio": spec.CSA_RATIO,
                         "hca_ratio": spec.HCA_RATIO}
        return (f"KV: {c['ctx_full']} full + {c['ctx_compress']} compress = "
                f"{c['ctx_full'] + c['ctx_compress']}；"
                f"CSA {c['csa_ratio']}:1（检索视图）· "
                f"HCA {c['hca_ratio']}:1（全局概括）；两视图都读完整区；"
                f"时间步为原子单位"
                f"（文本 1 token/步，图片 1 步，音频 533 采样/步，视频 1 帧/步）")

    def list_models(self):
        out = []
        for d in sorted(glob.glob(os.path.join(self.model_dir, "*"))):
            if not os.path.isdir(d):
                continue
            # 2026-10-27：manifest 化发布后权重文件名带 uuid，存在性判断
            # 改用 model_exists（config.json 或 model*.safetensors 任一）
            if not model_exists(d):
                continue
            info = {"name": os.path.basename(d), "params": None}
            cfg = os.path.join(d, "config.json")
            if os.path.exists(cfg):
                try:
                    with open(cfg, encoding="utf-8") as f:
                        info["params"] = json.load(f).get("params")
                except Exception:
                    pass
            out.append(info)
        return out

    def load(self, name):
        # 2026-10-20：模型名必须过统一校验——此前 load 不校验名字，
        # "../x" 可穿越到 model_dir 外（并可经 _save_ckpt 把检查点写到
        # 任意路径），HTTP 可直接触发
        bad = valid_model_name(name)
        if bad:
            raise ValueError(f"非法模型名: {bad}")
        d = os.path.join(self.model_dir, name)
        if not model_exists(d):
            raise FileNotFoundError(f"模型不存在: {name}")
        # 只读 config.json（block 数/上下文/params 都在里面）——权重留给
        # 服务进程按需加载。config 缺失/损坏时 read_config 返回 {}，
        # params=None、ctx 回退 spec 默认（与旧行为一致）。
        cfg = read_config(d)
        params = cfg.get("params")
        if not isinstance(params, int):
            params = None
        with self.lock:
            self.name = name
            self.params = params
            self.ctx = read_ctx_config(d)
        return {"model": name, "params": params}

    def auto_load_unique(self):
        """models/ 下唯一模型 → 直接设为活动模型（登记，秒回）。"""
        models = self.list_models()
        if len(models) != 1:
            return
        try:
            self.load(models[0]["name"])
            print(f"[server] 活动模型：{self.name}（权重按需加载）")
        except Exception as e:
            print(f"[server] 自动加载失败: {e}")


# ---------------------------------------------------------------------------
# 异步任务池
# ---------------------------------------------------------------------------
class TaskPool:
    """ 有界任务池：最多保留最近 MAX_TASKS 个任务，防 _tasks 无限增长。 """

    MAX_TASKS = 256

    def __init__(self, workers=4):
        self._ex = ThreadPoolExecutor(max_workers=workers)
        self._tasks = {}
        self._lock = threading.Lock()

    def submit(self, fn):
        tid = uuid.uuid4().hex
        with self._lock:
            self._tasks[tid] = {"status": "running", "result": None,
                                "error": None}
            # 只淘汰已完成/已失败的任务；运行中的任务永不淘汰
            # （旧实现按插入序 pop，可能丢掉运行中任务 → 前端永久 polling）
            done_ids = [k for k, v in self._tasks.items()
                        if v["status"] != "running"]
            while (len(self._tasks) > self.MAX_TASKS and done_ids):
                self._tasks.pop(done_ids.pop(0))

        def run():
            try:
                res = fn()
                self._finish(tid, status="done", result=res)
            except Exception as e:
                self._finish(tid, status="error", error=str(e))

        self._ex.submit(run)
        return tid

    def _finish(self, tid, **kw):
        with self._lock:
            t = self._tasks.get(tid)
            if t is not None:
                t.update(kw)

    def get(self, tid):
        with self._lock:
            t = self._tasks.get(tid)
            if t is None:
                return None
            out = {"status": t["status"]}
            if t["status"] == "done":
                out["result"] = t["result"]
            elif t["status"] == "error":
                out["error"] = t["error"]
            return out


# ---------------------------------------------------------------------------
# codec 辅助：Step → JSON（bytes 字段 base64）
# ---------------------------------------------------------------------------
def step_to_json(st):
    out = {"token_ids": st["token_ids"], "type": st["type"],
           "modality": st["modality"], "meta": {}}
    meta = st.get("meta", {})
    if st["modality"] == "visual":
        patches = meta.get("patches", [])
        out["meta"] = {
            "frame_idx": meta.get("frame_idx"),
            "full": meta.get("full"),
            "grid": meta.get("grid"),
            "n_patches": len(patches),
            "patches": [{"x": p["x"], "y": p["y"],
                         "vec_b64": base64.b64encode(p["vec"]).decode()}
                        for p in patches],
        }
    elif st["modality"] == "audio":
        samples = meta.get("samples", b"")
        out["meta"] = {"n": meta.get("n"),
                       "samples_b64": base64.b64encode(samples).decode()
                       if samples else ""}
    else:
        out["meta"] = {k: v for k, v in meta.items()
                       if not isinstance(v, (bytes, bytearray))}
    return out
