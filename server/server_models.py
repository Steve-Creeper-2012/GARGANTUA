# ===========================================================================
# GARGANTUA v1 — server_models.py：模型生命周期 + 异步任务池
# ---------------------------------------------------------------------------
# 从臃肿的 server.py（986 行）拆出，HTTP 层不再夹带这两块职责：
#   ModelManager   同一时刻一个活动模型；所有模型操作走 self.lock（训练/推理互斥）
#   TaskPool       有界异步任务池（202 + task_id 长轮询的执行体）
#   step_to_json   codec 产物 Step → JSON（bytes 字段 base64）
#   ConflictError  状态冲突（训练中切模型 / 加载被更新的请求取代）→ HTTP 409
# server.py 继续 re-export 这些符号，老 import 路径全部保持可用。
# ===========================================================================
import base64
import glob
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import spec
from common import (CKPT_NAME, build_model, save_checkpoint,
                    valid_model_name, validate_ctx_config)
from model import Gargantua


class ConflictError(RuntimeError):
    """状态冲突：训练中切换/创建模型，或加载被更新的加载请求取代。"""


class ModelManager:
    """同一时刻一个活动模型；所有模型操作走 self.lock（训练/推理互斥）。"""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.lock = threading.RLock()
        self.model = None
        self.name = None
        self.params = 0
        self.ctx = None                   # 活动模型的上下文长度配置
        self.loading = None                 # 正在后台加载的模型名
        self._load_gen = 0                  # 加载代号：并发加载时仅最新者生效
        os.makedirs(model_dir, exist_ok=True)

    def ctx_note(self) -> str:
        """活动模型的 KV 配置摘要（无活动模型时回退 spec 默认）。
        2026-10-23：上下文长度建模时可配，status 必须反映真实值。 """
        c = self.ctx or {"ctx_full": spec.CTX_FULL,
                         "ctx_compress": spec.CTX_COMPRESS,
                         "csa_ratio": spec.CSA_RATIO,
                         "hca_ratio": spec.HCA_RATIO,
                         "csa_local_window": spec.CSA_LOCAL_WINDOW}
        return (f"KV: {c['ctx_full']} full + {c['ctx_compress']} compress = "
                f"{c['ctx_full'] + c['ctx_compress']}；"
                f"CSA {c['csa_ratio']}:1(窗 {c['csa_local_window']}) · "
                f"HCA {c['hca_ratio']}:1；时间步为原子单位"
                f"（文本 1 token/步，图片 1 步，音频 533 采样/步，视频 1 帧/步）")

    def list_models(self):
        out = []
        for d in sorted(glob.glob(os.path.join(self.model_dir, "*"))):
            if not os.path.isdir(d):
                continue
            if not os.path.exists(os.path.join(d, CKPT_NAME)):
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
        ckpt = os.path.join(d, CKPT_NAME)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"模型不存在: {name}")
        with self.lock:
            self._load_gen += 1              # 占号：构造期间若有更新的
            gen = self._load_gen             # 加载请求，本次结果作废
        # 统一入口：按 config.json 的 block 数构造后加载权重（锁外构造）
        m = build_model(d)
        m.eval()
        params = sum(p.numel() for p in m.parameters())
        ctx = {k: getattr(m, k) for k in
               ("ctx_full", "ctx_compress", "csa_ratio", "hca_ratio",
                "csa_local_window")}
        with self.lock:
            if gen != self._load_gen:
                raise ConflictError(
                    f"加载 {name} 被更新的加载请求取代，请重试")
            self.model, self.name, self.params = m, name, params
            self.ctx = ctx
        return {"model": name, "params": params}

    def init_new(self, name, n_enc_blocks=None, n_dec_blocks=None,
                 overwrite=False, ctx=None):
        """新建随机初始化模型并落盘（config.json + safetensors，原子写）。

        2026-10-20：同名模型默认拒绝覆盖（防误抹训练成果），overwrite=True
        才允许；模型名走 common.valid_model_name 统一校验（含 Windows
        反斜杠/保留设备名）。
        2026-10-23：ctx 为 common.validate_ctx_config 的产物（或 None →
        spec 默认），记录进 config.json——上下文长度（完整/压缩容量 +
        CSA/HCA 压缩比）从此是建模期可选项。 """
        bad = valid_model_name(name)
        if bad:
            raise ValueError(f"非法模型名: {bad}")
        ctx = ctx or validate_ctx_config()
        d = os.path.join(self.model_dir, name)
        if (os.path.exists(os.path.join(d, CKPT_NAME))
                and not overwrite):
            raise FileExistsError(f"模型已存在: {name}（ overwrite 才可覆盖）")
        m = Gargantua(n_enc_blocks=n_enc_blocks, n_dec_blocks=n_dec_blocks,
                      **ctx)
        params = sum(p.numel() for p in m.parameters())
        config = {
            "name": name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": params,
            "d_model": spec.D_MODEL, "head_dim": spec.HEAD_DIM,
            "n_enc_blocks": (n_enc_blocks if n_enc_blocks is not None
                             else spec.N_ENC_BLOCKS),
            "n_dec_blocks": (n_dec_blocks if n_dec_blocks is not None
                             else spec.N_DEC_BLOCKS),
            "vocab_size_padded": spec.VOCAB_SIZE_PADDED,
            "ctx_full": ctx["ctx_full"],
            "ctx_compress": ctx["ctx_compress"],
            "csa_ratio": ctx["csa_ratio"],
            "hca_ratio": ctx["hca_ratio"],
            "csa_local_window": ctx["csa_local_window"],
            "csa_ratio_note": "CSA/HCA 是同一段压缩区内容的两个视图",
            "optimizer": spec.OPTIMIZER,
        }
        with self.lock:      # 与训练落盘/模型加载互斥（覆盖同名模型时）
            save_checkpoint(d, m, config)
        return {"name": name, "params": params}

    def auto_load_unique(self):
        """models/ 下唯一模型 → 后台线程自动加载。"""
        models = self.list_models()
        if len(models) != 1:
            return
        name = models[0]["name"]

        def work():
            self.loading = name
            try:
                self.load(name)
                print(f"[server] 自动加载模型 {name}")
            except Exception as e:
                print(f"[server] 自动加载失败: {e}")
            finally:
                self.loading = None

        threading.Thread(target=work, daemon=True).start()


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
