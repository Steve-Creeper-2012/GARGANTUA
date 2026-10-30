# ===========================================================================
# GARGANTUA v1 — server.py：HTTP 层 + 启动编排（仅标准库 http.server）
# ---------------------------------------------------------------------------
# 用法：python server.py [--port 8712] [--host 127.0.0.1] [--model-dir models]
#                     [--data-dir data] [--log-file logs/auto_train.jsonl]
#                     [--no-window] [--idle-timeout 300]
#
# 2026-10-24 重构（steve 的启动链要求）——server 是唯一入口之后的「管家」：
#   1. 环境预检（解释器版本 + 主依赖）——server_env.check_environment；
#   2. 拉起独立桌面窗口 service/window/window.py（--no-window 可关）；
#   3. HTTP 服务（本文件：路由 + Handler + App 装配）；
#   4. 按需调度 infer / train 服务子进程（server_services.ServiceSupervisor）：
#      有请求才起进程，空闲超时自行退出，模型权重不常驻 server。
#
# 本进程永不 import torch 加载权重——重活全在服务子进程里，server 本身
# 只路由 + 记账，常驻内存 ~百 MB 级（codec 用的 numpy/PIL 除外）。
#
# 2026-10-23 拆分（每层只干一件事，server.py 主体只留装配与路由）：
#   server_env.py      环境预检 + 子进程管理小工具
#   server_services.py 服务监督器（按需子进程 + JSON-lines 协议）
#   server_models.py   ModelManager（注册表，不含权重）+ TaskPool
#   server_media.py    上传媒体解码与尺寸守卫（b64_to_pil / ffmpeg / 抽帧）
#   server_codecs.py   CodecApi   —— /api/codec/* 全部端点
#   server_train.py    TrainApi   —— /api/train/* 全部端点（走训练服务）
#   server_tokens.py   TokensApi  —— /api/tokens/* 与 /api/vocab/add
#   server_util.py     num_param / pip 源 / 可选依赖安装命令
# App 以 mixin 组合上面几个 Api；原 `from server import X` 的路径全部通过
# 文件末尾的 re-export 保持可用。
#
# HTTP 错误契约（README「HTTP contract」）：
#   400 入参非法   404 路径/任务不存在   409 状态冲突
#   411 需 Content-Length（拒 chunked）   413 请求体 > 512MB
# ===========================================================================
import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# ---- GARGANTUA 路径引导（目录树重构：把 server/ 与 service/* 放上 path）----
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "start" / "boot.py").is_file():
        _sys.path.insert(0, str(_anc / "start"))
        break
else:
    raise SystemExit("找不到 GARGANTUA 项目根（start/boot.py）")
import boot
boot.bootstrap()
from common import (MAX_BLOCKS, MAX_NEW_TOKENS, model_init_lock,
                    valid_model_name, validate_ctx_config)
from server_env import ROOT, check_environment, kill_proc
from server_models import (ConflictError, ModelManager, TaskPool,
                           step_to_json)
from server_media import (MAX_IMAGE_DIM, MAX_PATCHES_TOTAL, b64_to_pil,
                          decode_audio_any, extract_video_audio,
                          extract_video_frames, size_guard_pil)
from server_codecs import CodecApi
from server_monitor import MonitorApi
from server_train import TrainApi
from server_tokens import TokensApi
from server_services import ServiceError, ServiceSupervisor
from server_util import num_param, optional_install_cmd, pip_index_args

WINDOW_PY = os.path.join(ROOT, "service", "window", "window.py")
WEB_INDEX = os.path.join(ROOT, "service", "web", "index.html")

# 请求体硬上限（base64 后的视频/音频可能上百 MB，512MB 是宽松保护值）
MAX_BODY_BYTES = 512 * 1024 * 1024
# prompt 字符上限（2026-10-27 H10）：编码前的第一道闸门，防超长文本在
# 服务进程里先完整 token 化（内存尖峰）才被拒
MAX_PROMPT_CHARS = 1_000_000

# 多模态推理的媒体闸门（2026-10-28）：单次最多 4 个附件、合计 64MB。
# 真正把媒体解成 patch/采样后的规模上限在 infer 服务里再查一遍
# （MAX_PATCHES_TOTAL / MAX_AUDIO_TOKENS）——这里只挡明显的滥用。
MAX_MEDIA_ITEMS = 4
MAX_MEDIA_BYTES = 64 * 1024 * 1024


# ---------------------------------------------------------------------------
# 训练状态代理：app.trainer.running 反映训练服务进程是否在训练
# ---------------------------------------------------------------------------
class _TrainStatusProxy:
    """App.trainer 的轻量替身（2026-10-24：训练搬到服务子进程后，
    server 不再持有 Trainer 实例）。只有 .running 被 409 冲突检查与
    /api/status 读取；查询走 supervisor（服务没起 = 没在训练）。
    测试仍可整体替换 app.trainer 打桩。"""

    def __init__(self, supervisor: ServiceSupervisor):
        self._sup = supervisor

    @property
    def running(self) -> bool:
        st = self._sup.query("train", "status", timeout=10)
        return bool(st and st.get("running"))


# ---------------------------------------------------------------------------
# 应用装配：注册表 + 任务池 + 服务监督器 + 三个 API mixin
# ---------------------------------------------------------------------------
class App(TrainApi, CodecApi, TokensApi, MonitorApi):
    def __init__(self, model_dir, data_dir, log_path, idle_timeout=300):
        self.mgr = ModelManager(model_dir)
        self.pool = TaskPool()
        self.data_dir = data_dir
        self.queue_dir = os.path.join(data_dir, "queue")
        os.makedirs(self.queue_dir, exist_ok=True)
        self.log_path = log_path
        self.supervisor = ServiceSupervisor(ROOT, idle_timeout=idle_timeout)
        self.trainer = _TrainStatusProxy(self.supervisor)
        # 2026-10-27 审计修复（H11）：活动模型「检查+快照」与「切换」共用
        # 的状态锁，消除模型切换与推理提交之间的 TOCTOU
        self._state_lock = threading.RLock()

    def apply_model_vocab(self) -> None:
        """活动模型的词表覆盖（每模型可有自己的 vocab.txt，2026-10-24）。
        没有 vocab.txt 就用全局词表（service/tokens/vocab.json）。

        2026-10-27 审计修复（M7）：词表损坏时清退到全局词表，而不是沿用
        上一模型的词表（旧模型的映射对新模型是错的）。"""
        if self.mgr.name is None:
            return
        try:
            from tokens.vocab import set_model_vocab
            set_model_vocab(os.path.join(self.mgr.model_dir, self.mgr.name))
        except Exception as e:
            from tokens.vocab import clear_model_vocab
            clear_model_vocab()
            print(f"[server] 模型词表加载失败，已清退到全局词表: {e}")

    # ---------------- status / models / infer ----------------
    def api_status(self):
        n_queued = sum(1 for p in glob.glob(os.path.join(self.queue_dir, "*"))
                       if os.path.isfile(p))
        train_st = self.supervisor.query("train", "status", timeout=10)
        return {
            "model": self.mgr.name,
            "loading": False,               # 注册表模式：切模型是瞬时的
            "params": self.mgr.params,
            "training": bool(train_st and train_st.get("running")),
            "queue": n_queued,
            "ctx": self.mgr.ctx,
            "kv_note": self.mgr.ctx_note(),
        }

    def api_models(self):
        return {"models": self.mgr.list_models(), "active": self.mgr.name}

    def api_model_init(self, body):
        name = body.get("name")
        if not name:
            raise ValueError("缺少 name")
        # 2026-10-23 审计修复（P2）：训练中覆盖【活动模型】会把磁盘上的
        # 训练成果抹掉再被下一次落盘盖回，状态混乱。api_model_load 早有
        # 409 保护，init 此前没有——状态契约不对称，这里补齐。
        if (self.trainer.running and self.mgr.name is not None
                and name == self.mgr.name):
            raise ConflictError("训练进行中，请先停止训练再覆盖活动模型")
        # 2026-10-23：block 数与上下文长度统一走严格整数校验——
        # int(True)==1 的类型混淆、"abc" 的裸 ValueError 都变成可读 400
        enc = (self._clamp_int(body["n_enc_blocks"], 1, MAX_BLOCKS,
                               "n_enc_blocks")
               if body.get("n_enc_blocks") is not None else None)
        dec = (self._clamp_int(body["n_dec_blocks"], 1, MAX_BLOCKS,
                               "n_dec_blocks")
               if body.get("n_dec_blocks") is not None else None)
        # 上下文长度（完整/压缩容量 + CSA/HCA 压缩比）；
        # 缺省项取 spec 默认，非法值 → ValueError（→400）
        ctx = validate_ctx_config(
            ctx_full=body.get("ctx_full"),
            ctx_compress=body.get("ctx_compress"),
            csa_ratio=body.get("csa_ratio"),
            hca_ratio=body.get("hca_ratio"))

        # 2026-10-25：词表预设 + 精度提前校验（非法 → 400，避免提交失败任务）
        if body.get("vocab"):
            from vocab_presets import validate_preset
            try:
                validate_preset(body["vocab"])
            except ValueError as e:
                raise ValueError(f"vocab: {e}")
        if body.get("dtype"):
            from precision import validate_dtype
            try:
                validate_dtype(body["dtype"])
            except ValueError as e:
                raise ValueError(f"dtype: {e}")

        def work():
            # 建模是罕见的重活（默认 4+6 blocks ≈ 527M 参数、2.1GB
            # safetensors）：一次性子进程跑完即退，server 内存零增长。
            argv = ["--name", name, "--out", self.mgr.model_dir]
            if enc is not None:
                argv += ["--enc-blocks", str(enc)]
            if dec is not None:
                argv += ["--dec-blocks", str(dec)]
            if body.get("overwrite"):
                argv += ["--force"]
            # 2026-10-25：词表预设 + 精度下发给 init 子进程
            if body.get("vocab"):
                argv += ["--vocab", str(body["vocab"])]
            if body.get("dtype"):
                argv += ["--dtype", str(body["dtype"])]
            # ctx 已在本进程校验过，原样下发（服务端不再解释默认值）
            argv += ["--ctx-full", str(ctx["ctx_full"]),
                     "--ctx-compress", str(ctx["ctx_compress"]),
                     "--csa-ratio", str(ctx["csa_ratio"]),
                     "--hca-ratio", str(ctx["hca_ratio"])]
            return self.supervisor.run_oneshot(
                os.path.join("service", "init", "init.py"), argv,
                timeout=900)

        tid = self.pool.submit(work)
        return 202, {"task_id": tid}

    @staticmethod
    def _clamp_int(v, lo, hi, name):
        """严格整数入参：拒 bool / 非整数值 / 非数字字符串，再钳范围。"""
        if isinstance(v, bool):
            raise ValueError(f"{name} 必须是整数，不能是布尔值: {v!r}")
        if isinstance(v, float):
            if not v.is_integer():
                raise ValueError(f"{name} 必须是整数: {v!r}")
            v = int(v)
        elif isinstance(v, str):
            s = v.strip()
            if not s.lstrip("+-").isdigit():
                raise ValueError(f"{name} 必须是整数: {v!r}")
            v = int(s)
        elif not isinstance(v, int):
            raise ValueError(f"{name} 必须是整数: {v!r}")
        if not (lo <= v <= hi):
            raise ValueError(f"{name} 必须在 [{lo}, {hi}]: {v}")
        return v

    def api_model_load(self, body):
        name = body.get("name")
        if not name:
            raise ValueError("缺少 name")
        # 先做入参合法性（路径穿越）再做状态互斥：非法输入不该因为训练
        # 正在跑就换个错误码蒙混过去，校验顺序也是契约的一部分
        bad = valid_model_name(name)
        if bad:
            raise ValueError(bad)
        if self.trainer.running:
            # 训练中热替换模型会令 Trainer 的优化器绑定旧参数（已防，但
            # 从源头禁止更稳）：加载与训练互斥
            raise ConflictError("训练进行中，请先停止训练再切换模型")
        d = os.path.join(self.mgr.model_dir, name)
        # 2026-10-27 审计修复（M7）：先验证词表可完整加载，失败直接 400
        # 且【不切换活动模型】——旧实现先切活动模型再捕获词表异常继续
        # 成功，加载词表损坏的 B 后，B 的 token API 仍按旧模型 A 编解码。
        from tokens.vocab import set_model_vocab
        try:
            set_model_vocab(d)
        except Exception as e:
            raise ValueError(f"模型词表加载失败: {e}")
        # 状态锁内「再查训练状态 + 切换」原子化（H11）：与 api_infer 的
        # 快照互斥，切换不可能插在检查与快照之间
        with self._state_lock:
            if self.trainer.running:
                raise ConflictError("训练进行中，请先停止训练再切换模型")
            res = self.mgr.load(name)      # 注册表登记：不读权重，瞬时完成
        return 200, res

    def api_infer(self, body):
        prompt = body.get("prompt")
        if not prompt:
            raise ValueError("缺少 prompt")
        if not isinstance(prompt, str):
            raise ValueError(f"prompt 必须是字符串: {type(prompt).__name__}")
        # 2026-10-27 审计修复（H10 的一部分）：请求体虽已有 512MB 硬上限，
        # prompt 文本本身再限一道字符上限，防止超大字符串在服务进程里
        # 先被完整编码成 token（内存尖峰）才被拒绝
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(
                f"prompt 过长: {len(prompt)} > {MAX_PROMPT_CHARS} 字符")
        # 2026-10-28：多模态推理已接线——媒体作为 prompt 前缀参与编码
        # （视觉走 visual_proj、音频走 audio_proj，统一投影到 d_model），
        # 生成仍然是文本。media 是 [{kind, b64, filename?}]。
        # 旧字段 media_b64/media_name 已废弃：留着不报错等于静默丢弃附件，
        # 用户会以为模型「看到了」——显式拒绝并给出新格式。
        if body.get("media_b64") or body.get("media_name"):
            raise ValueError(
                "media_b64/media_name 已废弃：请改用 "
                "media=[{kind: image|audio|video, b64: ..., filename: ...}]")
        media = body.get("media")
        if media is None:
            media = []
        if not isinstance(media, list):
            raise ValueError("media 必须是数组")
        if len(media) > MAX_MEDIA_ITEMS:
            raise ValueError(
                f"媒体数量超限: {len(media)} > {MAX_MEDIA_ITEMS}")
        total_bytes = 0
        clean_media = []
        for it in media:
            if not isinstance(it, dict):
                raise ValueError(f"每个 media 项必须是对象: {it!r}")
            kind = it.get("kind")
            if kind not in ("image", "audio", "video"):
                raise ValueError(
                    f"media.kind 必须是 image/audio/video: {kind!r}")
            b64 = it.get("b64")
            if not isinstance(b64, str) or not b64:
                raise ValueError("media.b64 必须是非空 base64 字符串")
            total_bytes += len(b64) * 3 // 4
            if total_bytes > MAX_MEDIA_BYTES:
                raise ValueError(
                    f"媒体总量超限: > {MAX_MEDIA_BYTES // 1048576}MB")
            clean_media.append({"kind": kind, "b64": b64,
                                "filename": it.get("filename") or ""})
        # 2026-10-27 审计修复（H11）：在状态锁内完成「无活动模型检查 +
        # model_dir 快照」，后台任务只使用快照——旧实现在线程真正执行
        # 时才读 self.mgr.name，A 模型推理排队后切换到 B，任务会在 B 上跑。
        with self._state_lock:
            if self.mgr.name is None:
                raise ValueError("无活动模型，请先加载模型")
            model_dir = os.path.join(self.mgr.model_dir, self.mgr.name)
        max_new = num_param(body, "max_new_tokens", 32, int)
        if not (1 <= max_new <= MAX_NEW_TOKENS):
            raise ValueError(
                f"max_new_tokens 必须在 [1, {MAX_NEW_TOKENS}]: {max_new}")
        temperature = num_param(body, "temperature", 0.8, float)
        top_p = num_param(body, "top_p", 0.95, float)
        if not (temperature > 0):
            raise ValueError(f"temperature 必须 > 0: {temperature}")
        if not (0 < top_p <= 1):
            raise ValueError(f"top_p 必须在 (0, 1]: {top_p}")

        def work():
            # 推理服务按需拉起（首次含建模加载，之后复用；空闲自行退出）。
            # 服务返回的 text 已用该模型的词表解码；archive 是本次推理的
            # token id 存档（.tokens，随 logs 目录走）。model_dir 是提交时
            # 的快照：期间切换模型不影响本次任务。
            return self.supervisor.call(
                "infer", "infer", timeout=600,
                model_dir=model_dir,
                prompt=prompt, max_new=max_new,
                temperature=temperature, top_p=top_p,
                media=clean_media,
                archive_dir=os.path.join(os.path.dirname(self.log_path),
                                         "infer_archive"))

        tid = self.pool.submit(work)
        return 202, {"task_id": tid}

    # ---------------- system / vocab ----------------
    # 模块名 → 发行包名（importlib.metadata 认的是后者，PIL 装在 pillow 里）
    _DIST_NAME = {"PIL": "pillow", "cv2": "opencv-python"}

    # 可选加速依赖：缺了主流程照跑（Pillow + ffmpeg 兜底），所以不当错误报，
    # 而是告诉用户「它有什么用、要不要装」。不要把 str(e) 倒给前端——用户看到
    # "No module named 'cv2'" 并不知道该不该管它。
    _OPTIONAL_DEPS = {
        "cv2": {"pkg": "opencv-python",
                "why": "图像/视频解码加速（缺失时用 Pillow + ffmpeg）"},
        "soundfile": {"pkg": "soundfile",
                      "why": "音频解码加速（缺失时用 ffmpeg）"},
    }

    def api_system(self):
        import importlib.metadata as _md
        # 延迟 import torch：server 本体不依赖 torch（重活都在服务进程），
        # 只有这个「环境与依赖」端点需要报版本号
        try:
            import torch
        except Exception:
            torch = None
        deps = []
        for mod in ["torch", "numpy", "PIL", "safetensors", "cv2", "soundfile"]:
            entry = {"name": mod, "ok": False}
            try:
                __import__(mod)
                entry["ok"] = True
                try:                       # 已安装 → 带上版本号，前端直接展示
                    entry["version"] = _md.version(
                        self._DIST_NAME.get(mod, mod))
                except Exception:
                    pass
            except Exception:
                opt = self._OPTIONAL_DEPS.get(mod)
                if opt:
                    # 可选依赖：给一句人话 + pip 包名，前端渲染成「未安装 · 可选」
                    # 并提供一键安装
                    entry["optional"] = True
                    entry["pkg"] = opt["pkg"]
                    entry["why"] = opt["why"]
                else:                      # 主依赖缺失是真问题，保留原因
                    entry["note"] = str(sys.exc_info()[1])
            deps.append(entry)
        device = "unknown"
        if torch is not None:
            device = "cpu"
            if torch.cuda.is_available():
                device = f"cuda ({torch.cuda.get_device_name(0)})"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        return 200, {
            "torch": getattr(torch, "__version__", None),
            "device": device,
            "dependencies": deps,
            "optional_missing": [d["name"] for d in deps
                                 if d.get("optional") and not d["ok"]],
            "ctx": self.mgr.ctx,
            "kv_note": self.mgr.ctx_note(),
        }

    def api_system_install_optional(self, body):
        """ 在 UI 里补装可选依赖（cv2 / soundfile），异步任务。

        装进的是当前跑 server 的解释器；已 import 的模块不会热更新，所以返回
        里明确要求重启。主依赖不在这里装——那是 setup 脚本的职责。 """
        names = body.get("names") or []
        if isinstance(names, str):
            names = [names]
        pkgs = []
        for n in names:
            opt = self._OPTIONAL_DEPS.get(n)
            if opt is None:
                raise ValueError(
                    f"不支持安装: {n!r}（可选: {sorted(self._OPTIONAL_DEPS)}）")
            if not any(d["name"] == n and d["ok"]
                       for d in self.api_system()[1]["dependencies"]):
                pkgs.append(opt["pkg"])
        if not pkgs:
            return 200, {"installed": [], "note": "都已安装，无需操作"}

        def work():
            req = os.path.join(boot.project_root(), "start",
                               "requirements", "core.txt")
            cmd = optional_install_cmd(sys.executable, pkgs, req)
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=900)
            if r.returncode != 0:
                raise RuntimeError("pip 安装失败: "
                                   + (r.stderr or r.stdout)[-400:])
            return {"installed": pkgs,
                    "note": "已安装；Python 不会热更新已 import 的模块，"
                            "重启 server 后生效"}

        tid = self.pool.submit(work)
        return 202, {"task_id": tid}

    def api_vocab(self, query):
        from tokens.vocab import get_vocab
        q = (query.get("q", [""])[0] or "").lower()
        v = get_vocab()
        hits = []
        if q:
            for i, tok in enumerate(v.id_to_token):
                if tok.startswith("<reserved:"):
                    continue
                if q in tok.lower():
                    hits.append({"id": i, "token": tok,
                                 "category": v.category_of(i)})
                if len(hits) >= 50:
                    break
        from tokens.vocab import CATEGORIES
        return 200, {"q": q, "hits": hits, "vocab_size": len(v),
                     "categories": list(CATEGORIES)}


# ---------------------------------------------------------------------------
# HTTP 层
# ---------------------------------------------------------------------------
class BodyTooLarge(ValueError):
    """ 请求体超过 MAX_BODY_BYTES → 413。 """


class Handler(BaseHTTPRequestHandler):
    server_version = "GargantuaHTTP/1.0"
    protocol_version = "HTTP/1.1"
    timeout = 30                          # socket 超时：慢速恶意连接不占死线程

    def log_message(self, fmt, *args):
        pass                                    # 静默访问日志
    # ---------------- 基础工具 ----------------
    def _send_json(self, obj, code=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error_json(self, code, msg):
        try:
            self._send_json({"error": msg}, code)
        except Exception:
            pass

    def _read_body(self):
        # chunked 传输编码：Content-Length 解析为 0，后续 read 会错乱 → 411
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            raise BodyTooLarge("不支持 chunked 传输编码，请携带 Content-Length")
        raw_cl = (self.headers.get("Content-Length") or "").strip()
        # 2026-10-23 审计修复（P0）：仅接受纯 ASCII 数字。int() 会接受
        # "-1"（负值 → read(-1) 无界读取）、"1_0"（下划线分隔 = 10）、
        # "１２３"（全角数字）等非 RFC 7230 形式，必须在校验阶段就拒掉
        if raw_cl and not (raw_cl.isascii() and raw_cl.isdigit()):
            raise ValueError(f"Content-Length 非法: {raw_cl!r}")
        n = int(raw_cl) if raw_cl else 0
        # 2026-10-23 审计修复（P0）：负 Content-Length 曾让
        # `n > MAX_BODY_BYTES` 判断失效，read(-1) 变为「读到 EOF 为止」
        # 的无界读取——客户端保持连接持续发送即可耗尽服务端内存
        if n < 0:
            raise ValueError(f"Content-Length 不能为负: {n}")
        if n > MAX_BODY_BYTES:
            raise BodyTooLarge(
                f"请求体过大: {n} > {MAX_BODY_BYTES} 字节")
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            raise ValueError("请求体不是合法 JSON")
        if body is None:                   # "null" → 按空对象处理
            return {}
        if not isinstance(body, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return body

    # ---------------- 路由 ----------------
    def do_GET(self):
        app = self.server.app
        u = urlparse(self.path)
        path, query = u.path, parse_qs(u.query)
        try:
            if path == "/":
                with open(WEB_INDEX, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/status":
                self._send_json(app.api_status())
            elif path == "/api/models":
                self._send_json(app.api_models())
            elif path.startswith("/api/task/"):
                tid = path.rsplit("/", 1)[-1]
                t = app.pool.get(tid)
                if t is None:
                    self._send_error_json(404, f"任务不存在: {tid}")
                else:
                    self._send_json(t)
            elif path == "/api/train/status":
                self._send_json(app.api_train_status())
            elif path == "/api/train/queue":
                _code, obj = app.api_train_queue()
                self._send_json(obj, _code)
            elif path == "/api/system":
                _code, obj = app.api_system()
                self._send_json(obj, _code)
            elif path == "/api/system/devices":
                self._send_json(app.api_system_devices()[1])
            elif path == "/api/system/processes":
                self._send_json(app.api_system_processes()[1])
            elif path == "/api/system/updates":
                self._send_json(app.api_system_updates(query)[1])
            elif path == "/api/vocab":
                _code, obj = app.api_vocab(query)
                self._send_json(obj, _code)
            elif path == "/api/vocab/ranges":
                _code, obj = app.api_vocab_ranges(query)
                self._send_json(obj, _code)
            elif path == "/api/vocab/presets":
                _code, obj = app.api_vocab_presets({})
                self._send_json(obj, _code)
            else:
                self._send_error_json(404, f"未知路径: {path}")
        except BrokenPipeError:
            pass
        except Exception as e:
            self._send_error_json(500, str(e))

    def do_POST(self):
        app = self.server.app
        path = urlparse(self.path).path
        routes = {
            "/api/model/init": app.api_model_init,
            "/api/model/load": app.api_model_load,
            "/api/infer": app.api_infer,
            "/api/train/enqueue": app.api_train_enqueue,
            "/api/train/start": app.api_train_start,
            "/api/train/stop": app.api_train_stop,
            "/api/train/delete": app.api_train_delete,
            "/api/codec/image": app.api_codec_image,
            "/api/codec/frames": app.api_codec_frames,
            "/api/codec/audio": app.api_codec_audio,
            "/api/codec/draw": app.api_codec_draw,
            "/api/codec/text": app.api_codec_text,
            "/api/codec/video": app.api_codec_video,
            "/api/codec/image2prims": app.api_codec_image2prims,
            "/api/tokens/encode": app.api_tokens_encode,
            "/api/tokens/decode": app.api_tokens_decode,
            "/api/tokens/export": app.api_tokens_export,
            "/api/vocab/add": app.api_vocab_add,
            "/api/vocab/presets": app.api_vocab_presets,
            "/api/vocab/preset/token": app.api_vocab_preset_token,
            "/api/system/install-optional": app.api_system_install_optional,
            "/api/system/install": app.api_system_install,
            "/api/system/kill": app.api_system_kill,
        }
        fn = routes.get(path)
        if fn is None:
            self._send_error_json(404, f"未知路径: {path}")
            return
        try:
            body = self._read_body()
            ret = fn(body)
            code, obj = ret if isinstance(ret, tuple) else (200, ret)
            self._send_json(obj, code)
        except BrokenPipeError:
            pass
        except BodyTooLarge as e:
            self._send_error_json(413, str(e))
        except (ConflictError, FileExistsError) as e:
            self._send_error_json(409, str(e))
        except (ValueError, FileNotFoundError, KeyError) as e:
            self._send_error_json(400, str(e))
        except RuntimeError as e:
            self._send_error_json(500, str(e))
        except Exception as e:
            self._send_error_json(500, str(e))


# ---------------------------------------------------------------------------
# 窗口：拉起 + 看护（窗口崩了不影响 server；server 退出带走窗口）
# ---------------------------------------------------------------------------
def _window_env() -> dict:
    env = dict(os.environ)
    try:
        env["PYTHONPATH"] = boot.pythonpath()
    except Exception:
        pass
    return env


def spawn_window(port: int, host: str):
    cmd = [sys.executable, WINDOW_PY, "--port", str(port), "--host", host]
    print(f"[server] 启动窗口：{' '.join(cmd)}")
    return subprocess.Popen(cmd, env=_window_env())


def _watch_window(proc, port: int, host: str) -> None:
    """窗口退出只打印重开提示——server 是老大，不跟着退（Steam 语义）。"""
    proc.wait()
    print(f"[server] 窗口已退出（server 继续运行）。重开界面：")
    print(f"  {sys.executable} {WINDOW_PY} --port {port} --host {host}")


# ---------------------------------------------------------------------------
# 主流程：环境检查 → 窗口校验/拉起 → HTTP → 收尸
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="GARGANTUA server")
    ap.add_argument("--port", type=int, default=8712)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model-dir", default=os.path.join(ROOT, "models"))
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--log-file",
                    default=os.path.join(ROOT, "logs", "auto_train.jsonl"))
    ap.add_argument("--no-window", action="store_true",
                    help="不起独立窗口（仅 HTTP server，SSH/CI 用）")
    ap.add_argument("--idle-timeout", type=int, default=300,
                    help="服务子进程空闲多少秒后自行退出（默认 300）")
    args = ap.parse_args(argv)

    # 1) 环境预检：解释器版本 + 主依赖（缺了给可执行提示，不黑屏）
    #    经由 launcher（启动.sh / 启动.command）启动时，它已做轻量预检
    #    （只确认隔离 Python + pywebview）并设置 GARGANTUA_NO_DEPCHECK=1，
    #    故此处跳过——torch 导入在弱网/首次会拖很久，不应阻塞启动。
    #    直接 `python server.py` 时仍会做完整预检作为兜底。
    if os.environ.get("GARGANTUA_NO_DEPCHECK") == "1":
        print("[server] 跳过主依赖预检（由 launcher 轻量预检接管："
              "GARGANTUA_NO_DEPCHECK=1）")
    elif not check_environment(sys.executable):
        return 1

    app = App(args.model_dir, args.data_dir, args.log_file,
              idle_timeout=args.idle_timeout)
    app.mgr.auto_load_unique()
    app.apply_model_vocab()

    # 2) HTTP 服务
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.app = app
    srv.daemon_threads = True
    print(f"[server] http://{args.host}:{args.port}  "
          f"model_dir={args.model_dir}  data_dir={args.data_dir}")

    window_proc = None
    if not args.no_window:
        try:
            window_proc = spawn_window(args.port, args.host)
            threading.Thread(target=_watch_window,
                             args=(window_proc, args.port, args.host),
                             daemon=True).start()
        except Exception as e:
            print(f"[server] 窗口启动失败（不影响 server）：{e}")

    # SIGTERM 也走 finally 收尸（默认动作会直接杀掉进程，不给清理机会）
    def _sigterm(sig, frame):
        raise SystemExit(0)
    try:
        signal.signal(signal.SIGTERM, _sigterm)
    except Exception:
        pass

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("[server] 正在关闭：服务子进程 → 窗口 → HTTP ……")
        app.supervisor.shutdown_all()
        kill_proc(window_proc)
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
