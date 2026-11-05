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
#   400 入参非法   404 路径/任务不存在   409 状态冲突   403 非法来源
#   413 请求体 > 512MB / 拒 chunked（缺 Content-Length）
# ===========================================================================
import argparse
import binascii
import glob
import json
import math
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
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
import spec
from common import (MAX_BLOCKS, MAX_NEW_TOKENS, MAX_PROMPT_TOKENS,
                    model_init_lock, normalize_shared_blocks,
                    valid_model_name, validate_ctx_config)
from server_env import ROOT, check_environment, kill_proc
from server_models import (ConflictError, ModelManager, TaskPool,
                           step_to_json)
from server_media import (MAX_IMAGE_DIM, MAX_PATCHES_TOTAL, b64_to_pil,
                          decode_audio_any, extract_video_audio,
                          extract_video_frames, size_guard_pil)
from server_codecs import CodecApi
from server_monitor import MonitorApi
from server_prims import PrimsApi
from server_train import TrainApi
from server_tokens import TokensApi, resolve_vocab_query
from server_services import ServiceError, ServiceSupervisor
from server_term import TermError, TermManager
from server_util import num_param, optional_install_cmd, pip_index_args

WINDOW_PY = os.path.join(ROOT, "service", "window", "window.py")
WEB_INDEX = os.path.join(ROOT, "service", "web", "index.html")
WEB_DIR = os.path.dirname(WEB_INDEX)            # 前端静态资源（app.css/app.js）所在目录

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

# ---- 推理显示插件：导入注册表（2026-11-04） -------------------------------
# 注册表只记录「导入插件」；内置 tokens/base 由前端 PLUGINS 静态声明，不在此列。
# 注意：_serve_static 白名单只放行 plugins/*.js|css|md，.json 不会被静态服务，
# 注册表只能走 /api/plugin/*，不会通过 HTTP 直接泄漏。路径一律用 WEB_DIR 拼。
_PLUGIN_DIR = os.path.join(WEB_DIR, "plugins")
_PLUGIN_REGISTRY = os.path.join(_PLUGIN_DIR, "_imported.json")
_PLUGIN_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
_PLUGIN_RESERVED = {"tokens", "base", "_imported"}
_PLUGIN_BUILTIN_FILES = {"tokens.js", "tokens.css", "README.md"}


def _plugin_check_id(pid):
    """id 白名单：正则 + 保留名 + 禁 .. / \\ 等路径穿越（非法一律 ValueError）。"""
    if not isinstance(pid, str):
        raise ValueError("插件 id 必须是字符串")
    if not _PLUGIN_ID_RE.fullmatch(pid):
        raise ValueError("插件 id 非法：须匹配 ^[a-z0-9][a-z0-9_-]{0,31}$")
    if pid in _PLUGIN_RESERVED:
        raise ValueError(f"插件 id 为保留名，禁止导入: {pid}")
    if pid.startswith("_") or ".." in pid or "/" in pid or "\\" in pid:
        raise ValueError(f"插件 id 含非法字符: {pid}")
    return pid


def _plugin_target(pid, ext):
    """拼 plugins/<id>.<ext>，realpath 必须落在 plugins/ 内且非内置文件。"""
    base = os.path.realpath(_PLUGIN_DIR)
    fname = pid + ext
    target = os.path.realpath(os.path.join(base, fname))
    # 目标必须是 base 下直系文件：符号链接/穿越导致 realpath 越界即拒绝
    if (target != os.path.join(base, fname)
            or not target.startswith(base + os.sep)):
        raise ValueError(f"插件路径越界: {fname}")
    if fname in _PLUGIN_BUILTIN_FILES:
        raise ValueError(f"禁止覆盖内置文件: {fname}")
    return target


def _plugin_read_registry():
    """读导入插件注册表；文件缺失 / 损坏一律按空列表处理。"""
    try:
        with open(_PLUGIN_REGISTRY, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError, TypeError):
        return []
    if isinstance(data, dict):                  # 也兼容 {"plugins": [...]} 形态
        data = data.get("plugins")
    if not isinstance(data, list):
        return []
    out = []
    for it in data:
        if not isinstance(it, dict) or not isinstance(it.get("id"), str):
            continue
        pid = it["id"]
        out.append({"id": pid,
                    "name": it.get("name") or pid,
                    "desc": it.get("desc") or "",
                    "entry": it.get("entry") or f"plugins/{pid}.js",
                    "css": it.get("css") or ""})
    return out


def _plugin_write_registry(items):
    """原子写注册表：同目录临时文件 + os.replace，避免半截文件。"""
    d = os.path.dirname(_PLUGIN_REGISTRY)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="._imported-", suffix=".tmp", dir=d)
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False)
        os.replace(tmp, _PLUGIN_REGISTRY)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _plugin_write_text(path, text):
    """原子写插件 .js/.css：同目录临时文件 + os.replace，杜绝半截模块。"""
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="._plugin-", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _plugin_meta_from_code(code, pid):
    """从 code 前 40 行解析 // @name / // @desc；缺省 name=id、desc=""。"""
    name, desc = pid, ""
    got_name = got_desc = False
    for line in code.splitlines()[:40]:
        s = line.strip()
        if not got_name:
            m = re.match(r"//\s*@name\s+(.+?)\s*$", s)
            if m:
                name, got_name = m.group(1), True
        if not got_desc:
            # 要求 @desc 后是空白或行尾，避免误把 @description 当成 @desc
            m = re.match(r"//\s*@desc(?:\s+(.*))?$", s)
            if m:
                desc, got_desc = (m.group(1) or "").strip(), True
    return name, desc


def _read_version() -> str:
    """项目版本（ROOT/VERSION，一行）。读不到就返回空串。

    只给「关于」页的版本徽章用——不参与任何逻辑判断。
    """
    try:
        with open(os.path.join(ROOT, "VERSION"), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _draw_digit_ids():
    """文本数字 token（"0".."9"）的 ID 表——画布声明的「数字」就是它们。

    走进程当前词表（词表不存在时退回 spec 的可见区布局兜底）。
    """
    from tokens.draw_codec import canvas_digit_ids
    try:
        from tokens.vocab import get_vocab
        return canvas_digit_ids(get_vocab())
    except Exception:
        return canvas_digit_ids()


def draw_layout_payload() -> dict:
    """图元 token 布局 = /api/status.draw_layout 的载荷（唯一构造处）。

    前端「图元编辑器」要实时算 token ID，但它不许自己抄一份常量——把
    spec 的 offset/bin 表发下去，前端只做「值 → bin」的算术（公式与
    draw_codec 逐行对应）。test_pdraw_js 也 import 这个函数做 JS/Python
    对拍——测试与线上永远是同一张表，改 spec 不会出现「测试用旧表」。
    """
    return {
        "begin": spec.DRAW_BEGIN,
        "tokens_per": spec.DRAW_TOKENS_PER_PRIMITIVE,
        "canvas": {"offset": spec.DRAW_CANVAS_OFFSET,
                   "bins": spec.DRAW_CANVAS_BINS,
                   "markers": list(spec.DRAW_CANVAS_MARKERS)},
        # 画布声明里的数字 = 可见区的文本数字 token 的 ID
        "digits": _draw_digit_ids(),
        "special_offset": spec.DRAW_SPECIAL_OFFSET,
        "color_max": spec.DRAW_COLOR_MAX,
        "fields": [
            {"name": n, "offset": off, "bins": bins, "per": per}
            for n, off, bins, per in (
                ("x", spec.DRAW_X_OFFSET, spec.DRAW_X_BINS,
                 spec.XY_BINS_PER_PIXEL),
                ("y", spec.DRAW_Y_OFFSET, spec.DRAW_Y_BINS,
                 spec.XY_BINS_PER_PIXEL),
                ("theta", spec.DRAW_ANGLE_OFFSET,
                 spec.DRAW_ANGLE_BINS, spec.ANGLE_BINS_PER_DEGREE),
                ("w", spec.DRAW_W_OFFSET, spec.DRAW_W_BINS,
                 spec.SIZE_BINS_PER_PIXEL),
                ("h", spec.DRAW_H_OFFSET, spec.DRAW_H_BINS,
                 spec.SIZE_BINS_PER_PIXEL),
                ("shape", spec.DRAW_SHAPE_OFFSET,
                 spec.DRAW_SHAPE_BINS, 1),
                # 颜色 10 bit：1 bin = 1 个值（不再是 8bit//2 的 per=0.5）
                ("r", spec.DRAW_R_OFFSET, spec.DRAW_COLOR_BINS, 1),
                ("g", spec.DRAW_G_OFFSET, spec.DRAW_COLOR_BINS, 1),
                ("b", spec.DRAW_B_OFFSET, spec.DRAW_COLOR_BINS, 1),
            )
        ],
    }


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
class App(TrainApi, CodecApi, TokensApi, MonitorApi, PrimsApi):
    def __init__(self, model_dir, data_dir, log_path, idle_timeout=300):
        self.mgr = ModelManager(model_dir)
        self.pool = TaskPool()
        self.data_dir = data_dir
        self.queue_dir = os.path.join(data_dir, "queue")
        os.makedirs(self.queue_dir, exist_ok=True)
        self.log_path = log_path
        self.supervisor = ServiceSupervisor(ROOT, idle_timeout=idle_timeout)
        self.trainer = _TrainStatusProxy(self.supervisor)
        # 内置终端：默认 cwd = ROOT，cd 出界会被拒绝（见 server_term.py）
        self.term = TermManager(ROOT)
        # 2026-10-27 审计修复（H11）：活动模型「检查+快照」与「切换」共用
        # 的状态锁，消除模型切换与推理提交之间的 TOCTOU
        self._state_lock = threading.RLock()
        # 流式推理的增量缓冲（2026-11-04）：key = stream_id，
        # value = {tokens, done, error, result, t0, ...}。前端按 from 游标
        # 拉增量，不重传已看过的 token。只保留最近 N 条，防长会话堆积。
        self._streams = {}
        self._streams_lock = threading.Lock()
        self._streams_max = 32

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

    # ---------------- 内置终端（项目目录内） ----------------
    def api_term_open(self, body):
        s = self.term.open()
        return 200, {"ok": True, "cwd": s.cwd, "alive": s.alive,
                     "root": s.root}

    def api_term_poll(self, query):
        s = self.term.get()
        return 200, {"text": s.snapshot(), "alive": s.alive,
                     "cwd": s.cwd, "root": s.root,
                     # 密码提示（sudo / ssh / passwd）：前端据此把输入框
                     # 切成 password、且不把这一行记进命令历史
                     "awaiting_secret": s.awaiting_secret()}

    def api_term_input(self, body):
        line = body.get("line")
        if not isinstance(line, str):
            raise ValueError("line 必须是字符串")
        s = self.term.get()
        # 前端标记之外再看一眼屏幕：用户可能在 250ms 轮询发现提示之前
        # 就已经把密码敲进去了，这一行同样不能走 cd 守卫/历史。
        secret = (self._bool_arg(body.get("secret"), "secret")
                  or s.awaiting_secret())
        s.submit(line, secret=secret)
        return 200, {"ok": True}

    def api_term_interrupt(self, body):
        # force=true：除了写 \x03，再直接给前台进程组补一发 SIGINT
        self.term.get().interrupt(
            force=self._bool_arg(body.get("force"), "force"))
        return 200, {"ok": True}

    def api_term_kill(self, body):
        """强制结束当前前台进程（SIGKILL）。"""
        self.term.get().kill()
        return 200, {"ok": True}

    def api_term_clear(self, body):
        self.term.get().clear()
        return 200, {"ok": True}

    def api_term_close(self, body):
        self.term.close()
        return 200, {"ok": True}

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
            "version": _read_version(),
            # 共享契约：前端推理页 max_new_tokens 输入框上限
            "max_new_tokens": MAX_NEW_TOKENS,
            # d_model 的合法档位：唯一真源在 spec.check_d_model，前端建模
            # 表单据此生成「128 的倍数」下拉，不再维护第二套硬编码范围。
            "d_model_range": {
                "min": spec.D_MODEL_MIN, "max": spec.D_MODEL_MAX,
                "step": spec.HEAD_DIM, "default": spec.D_MODEL,
            },
            # 图元画布档位：唯一真源在 spec，前端据此填下拉框
            "draw_canvas": {
                "default": list(spec.DRAW_DEFAULT_CANVAS),
                "sizes": [list(s) for s in spec.DRAW_CANVAS_SIZES],
            },
            # 图元 token 布局：唯一构造处在 draw_layout_payload()（test_pdraw_js
            # 也用它对拍，保证测试与线上同表）
            "draw_layout": draw_layout_payload(),
        }

    def api_models(self):
        return {"models": self.mgr.list_models(), "active": self.mgr.name}

    def api_model_estimate(self, body):
        """建模前的参数量 / 存储占用预估（不建模型、不碰 torch）。

        入参与 /api/model/init 同源：d_model / n_enc_blocks / n_dec_blocks /
        share_enc_dec / csa_ratio / hca_ratio / vocab / dtype。校验也走同一套
        规则（spec.check_d_model、normalize_shared_blocks、validate_ctx_config），
        避免「预估能过、真建报 400」的两套口径。
        """
        from param_estimate import estimate
        from vocab_presets import BASE, list_vocabs

        d_model = None
        if body.get("d_model") is not None:
            try:
                d_model = spec.check_d_model(body["d_model"])
            except ValueError as e:
                raise ValueError(f"d_model: {e}")
        enc = (self._clamp_int(body["n_enc_blocks"], 1, MAX_BLOCKS,
                               "n_enc_blocks")
               if body.get("n_enc_blocks") is not None else None)
        dec = (self._clamp_int(body["n_dec_blocks"], 1, MAX_BLOCKS,
                               "n_dec_blocks")
               if body.get("n_dec_blocks") is not None else None)
        share = self._bool_arg(body.get("share_enc_dec"), "share_enc_dec")
        enc, dec = normalize_shared_blocks(enc, dec, share)
        # 压缩比只影响压缩器位置嵌入行数；复用建模的校验口径
        ctx = validate_ctx_config(
            ctx_full=body.get("ctx_full"),
            ctx_compress=body.get("ctx_compress"),
            csa_ratio=body.get("csa_ratio"),
            hca_ratio=body.get("hca_ratio"))
        # 词表长度以 vocabs/<name>/config.json（正文条数为准）为事实源，
        # 与 init 子进程的 install_model_vocab 保持一致。
        preset = body.get("vocab") or BASE
        v = None
        for item in list_vocabs():
            if item["name"] == preset:
                v = item["vocab_size"]
                break
        if v is None:
            raise ValueError(f"词表不存在: {preset}")
        dtype = body.get("dtype") or "fp32"
        return 200, estimate(
            d_model=spec.D_MODEL if d_model is None else d_model,
            n_enc_blocks=spec.N_ENC_BLOCKS if enc is None else enc,
            n_dec_blocks=spec.N_DEC_BLOCKS if dec is None else dec,
            share_enc_dec=share,
            csa_ratio=ctx["csa_ratio"],
            hca_ratio=ctx["hca_ratio"],
            vocab_size=v,
            dtype=dtype,
        )

    def api_model_init(self, body):
        name = body.get("name")
        if not name:
            raise ValueError("缺少 name")
        # 审计 L3：init 此前不校验模型名（load 早已校验），"../" 会拼进
        # service/init/init.py 的 --name / --out 路径。统一走 valid_model_name。
        bad = valid_model_name(name)
        if bad:
            raise ValueError(f"非法模型名: {bad}")
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
        # 2026-11-04：enc/dec 权重模式。共享时两侧等深，归一化规则与
        # init.py CLI 共用 common.normalize_shared_blocks（不等 → 400）。
        share = self._bool_arg(body.get("share_enc_dec"), "share_enc_dec")
        # 不等深直接 ValueError（HTTP 层映射 400），不当场悄悄改用户的值
        enc, dec = normalize_shared_blocks(enc, dec, share)
        # 2026-11-04：模型宽度建模可选（规则唯一真源在 spec.check_d_model）
        d_model = None
        if body.get("d_model") is not None:
            try:
                d_model = spec.check_d_model(body["d_model"])
            except ValueError as e:
                raise ValueError(f"d_model: {e}")
        # 上下文长度（完整/压缩容量 + CSA/HCA 压缩比）；
        # 缺省项取 spec 默认，非法值 → ValueError（→400）
        ctx = validate_ctx_config(
            ctx_full=body.get("ctx_full"),
            ctx_compress=body.get("ctx_compress"),
            csa_ratio=body.get("csa_ratio"),
            hca_ratio=body.get("hca_ratio"))

        # 2026-10-25：词表预设 + 精度提前校验（非法 → 400，避免提交失败任务）
        if body.get("vocab"):
            from vocab_presets import validate_vocab
            try:
                validate_vocab(body["vocab"])
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
            if share:
                argv += ["--share-enc-dec"]
            if d_model is not None:
                argv += ["--d-model", str(d_model)]
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

    @staticmethod
    def _bool_arg(v, name):
        """严格布尔入参：bool('false') 为 True，绝不能用 bool() 转换。

        与 _clamp_int 同一套口径：认真布尔、0/1、常见字符串；其余一律
        400，避免把 "0"/"no" 之类解释成共享权重。"""
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, int) and v in (0, 1):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("true", "1", "on", "yes"):
                return True
            if s in ("false", "0", "off", "no", ""):
                return False
        raise ValueError(f"{name} 必须是布尔值: {v!r}")

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

    def api_model_delete(self, body):
        """删除一个模型目录（config.json + weights/ + vocab/ 一并移除）。

        路径穿越防护：realpath 后必须仍在 model_dir 内。删除活动模型时先
        杀 infer 服务（清空 fast weight / 上下文）再清空登记簿；训练中删除
        活动模型 → 409。返回是否成功删除。"""
        name = body.get("name")
        if not name:
            raise ValueError("缺少 name")
        bad = valid_model_name(name)
        if bad:
            raise ValueError(bad)
        d = os.path.realpath(os.path.join(self.mgr.model_dir, name))
        base = os.path.realpath(self.mgr.model_dir)
        if d != base and not d.startswith(base + os.sep):
            raise ValueError("非法模型路径")
        # 只要是 models/ 下的真实目录就允许删——**包括没建完的半成品**
        # （model_exists 要求 config.json + 权重 manifest，半成品不满足，
        # 恰恰是最该被删掉的那批）。路径穿越在上一段已挡住。
        if not os.path.isdir(d):
            raise FileNotFoundError(f"模型不存在: {name}")
        if self.trainer.running and self.mgr.name == name:
            raise ConflictError("训练进行中，请先停止训练再删除活动模型")
        with self.mgr.lock:
            if self.mgr.name == name:
                self.supervisor.kill_service("infer")
                self.mgr.name = None
                self.mgr.params = 0
                self.mgr.ctx = None
        shutil.rmtree(d)
        return 200, {"ok": True, "deleted": name}

    def api_infer_reset(self, body=None):
        """杀掉 infer 服务进程 → 下次推理重新加载模型。

        当前实现每次 generate 从零开始（state/kv 都是本次局部量），跨轮
        持久状态尚未接线，所以杀进程并不等于「清空记忆 / 重置 fast weight」，
        只是让下一次推理重新走一遍模型加载。返回是否真有进程被终止。
        """
        killed = self.supervisor.kill_service("infer")
        return {"ok": True, "killed_infer": killed,
                "note": "已清空界面会话并终止 infer 服务，下次推理会重新加载模型"}

    def api_infer(self, body):
        # 两种输入入口（2026-11-04）：
        #   prompt     —— 文本，走词表编码器（旧路径，媒体也走这条）
        #   token_ids  —— 纯 token 输入（默认 plugin 用），ID 直接进 embedding
        token_ids = body.get("token_ids")
        prompt = body.get("prompt")
        if token_ids is not None:
            if not isinstance(token_ids, list) or not token_ids:
                raise ValueError("token_ids 必须是非空整数数组")
            if len(token_ids) > MAX_PROMPT_TOKENS:
                raise ValueError(
                    f"token_ids 过长: {len(token_ids)} > {MAX_PROMPT_TOKENS}")
            if any(not isinstance(i, int) or isinstance(i, bool)
                   for i in token_ids):
                raise ValueError("token_ids 只能包含整数")
            prompt = ""
        else:
            if not prompt:
                raise ValueError("缺少 prompt 或 token_ids")
            if not isinstance(prompt, str):
                raise ValueError(
                    f"prompt 必须是字符串: {type(prompt).__name__}")
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
        # 补上界 100：与 infer.generate 同一口径，错误信息写清 [1e-5, 100]
        if not (1e-5 <= temperature <= 100):
            raise ValueError(f"temperature 必须在 [1e-5, 100]: {temperature}")
        if not (0 < top_p <= 1):
            raise ValueError(f"top_p 必须在 (0, 1]: {top_p}")
        archive_dir = os.path.join(os.path.dirname(self.log_path),
                                   "infer_archive")

        if self._bool_arg(body.get("stream"), "stream"):
            return 202, self._start_infer_stream(
                model_dir=model_dir, prompt=prompt, token_ids=token_ids,
                media=clean_media, max_new=max_new,
                temperature=temperature, top_p=top_p,
                archive_dir=archive_dir)

        def work():
            # 推理服务按需拉起（首次含建模加载，之后复用；空闲自行退出）。
            # 服务返回的 text 已用该模型的词表解码；archive 是本次推理的
            # token id 存档（.tokens，随 logs 目录走）。model_dir 是提交时
            # 的快照：期间切换模型不影响本次任务。
            return self.supervisor.call(
                "infer", "infer", timeout=600,
                model_dir=model_dir,
                prompt=prompt, token_ids=token_ids, max_new=max_new,
                temperature=temperature, top_p=top_p,
                media=clean_media,
                archive_dir=archive_dir)

        tid = self.pool.submit(work)
        return 202, {"task_id": tid}

    # ---------------- 流式推理（2026-11-04） ----------------
    def _start_infer_stream(self, *, model_dir, prompt, token_ids, media,
                            max_new, temperature, top_p, archive_dir):
        """起一条流式推理，立刻返回 stream_id；token 边产边进缓冲区。

        为什么不复用 TaskPool 的结果：任务池只有「跑完/失败」两种状态，
        而流式要的是「跑到第 3 个 token 了」。这里单独一份缓冲，前端按
        from 游标增量拉取；缓冲满 MAX 条后先淘汰已结束的，仍超限就按 t0
        淘汰最旧的一条（含 running），避免 32 条全 running 时突破上限。
        """
        sid = uuid.uuid4().hex
        buf = {"tokens": [], "done": False, "error": None, "result": None,
               "t0": time.time(), "ttft_ms": None, "stats": None,
               "kv_steps": 0, "fw": None, "writes": []}
        with self._streams_lock:
            # 无 done 可淘汰时按 t0 淘汰最旧（含 running），保证不超 _streams_max
            while len(self._streams) >= self._streams_max:
                done_ids = [k for k, v in self._streams.items() if v["done"]]
                if done_ids:
                    self._streams.pop(done_ids[0], None)
                else:
                    oldest = min(self._streams,
                                 key=lambda k: self._streams[k]["t0"])
                    self._streams.pop(oldest, None)
            self._streams[sid] = buf

        def work():
            gen = None
            try:
                gen = self.supervisor.stream(
                    "infer", "infer", timeout=600,
                    model_dir=model_dir, prompt=prompt, token_ids=token_ids,
                    media=media, max_new=max_new,
                    temperature=temperature, top_p=top_p,
                    archive_dir=archive_dir, stream=True,
                    # 墙钟起点：让首 token 的 t_ms / TTFT 从「收到请求」算，
                    # 含模型加载与排队，而不是从 generate() 内部算。
                    t0_wall=buf["t0"])
                for frame in gen:
                    part = frame.get("partial")
                    if part is not None:
                        # 最新 fast weight 快照 / 写入幅度 / KV 步数单独进 buf：
                        # 生成期 query 拿不到句柄锁，stats 直接读这里实时合成
                        if part.get("fw") is not None:
                            buf["fw"] = part["fw"]
                            buf["stats"] = part["fw"]
                        w = part.get("write")
                        if (w is not None and isinstance(w, (int, float))
                                and math.isfinite(float(w))):
                            buf["writes"].append(w)
                            if len(buf["writes"]) > spec.FW_WRITE_HISTORY:
                                del buf["writes"][:-spec.FW_WRITE_HISTORY]
                        if part.get("kv_steps") is not None:
                            buf["kv_steps"] = part["kv_steps"]
                        # token 字典里把 fw 置 None（保留键以符合「可能为 null」
                        # 协议），否则最多 2048 个 token 各带一份 24×24 矩阵吃穿内存
                        tok = dict(part)
                        tok["fw"] = None
                        buf["tokens"].append(tok)
                        if buf["ttft_ms"] is None:
                            buf["ttft_ms"] = part.get("t_ms")
                        continue
                    buf["result"] = frame.get("result")
            except Exception as e:
                buf["error"] = str(e)
            finally:
                # 显式关闭：否则句柄锁要等 GC 才释放，下一次推理会一直等
                if gen is not None:
                    try:
                        gen.close()
                    except Exception:
                        pass
                buf["done"] = True
                buf["t1"] = time.time()

        self.pool.submit(work)
        return {"task_id": sid, "stream": True}

    def api_infer_stream(self, query):
        """GET /api/infer/stream?task_id=…&from=N —— 拉增量 token。

        from 是前端已消费的 token 数（不是时间戳），返回 next 作为下一次
        的游标。这样断线重连/慢轮询都不会漏 token，也不会重复渲染。
        """
        sid = (query.get("task_id") or [""])[0]
        try:
            frm = max(0, int((query.get("from") or ["0"])[0]))
        except ValueError:
            raise ValueError("from 必须是整数")
        with self._streams_lock:
            buf = self._streams.get(sid)
            if buf is None:
                raise ValueError(f"流不存在或已过期: {sid}")
            toks = buf["tokens"][frm:]
            out = {"status": ("error" if buf["error"] else
                              "done" if buf["done"] else "running"),
                   "tokens": toks, "next": frm + len(toks),
                   "ttft_ms": buf["ttft_ms"],
                   "error": buf["error"],
                   "result": buf["result"] if buf["done"] else None}
            if buf["done"]:
                out["elapsed_ms"] = int((buf.get("t1", time.time())
                                         - buf["t0"]) * 1000)
            else:
                out["elapsed_ms"] = int((time.time() - buf["t0"]) * 1000)
        return 200, out

    def api_infer_stats(self):
        """推理侧实时状态：上下文占用 / KV cache / fast weight 可视化。

        生成期 stream() 持有句柄锁，query() 拿不到锁会立刻返回 None，所以
        优先读本地流缓冲（每帧 partial 已把 fw/writes/kv_steps 写进 buf）；
        只有一条流都没有时才回退到 query 取上一轮非流式统计。
        """
        cfg = self.mgr.ctx or {"ctx_full": spec.CTX_FULL,
                               "ctx_compress": spec.CTX_COMPRESS}
        model_cfg = {}
        if self.mgr.name:
            try:
                from common import read_config
                model_cfg = read_config(
                    os.path.join(self.mgr.model_dir, self.mgr.name)) or {}
            except Exception:
                model_cfg = {}

        # 锁内取 t0 最大的流（最新一次推理）并快照实时字段；有缓冲就绝不去
        # query（生成期 query 抢不到锁、也不该被阻塞），避免拿到过期统计
        with self._streams_lock:
            buf = max(self._streams.values(), key=lambda b: b["t0"],
                      default=None)
            snap = None
            if buf is not None:
                snap = {"fw": buf["fw"], "writes": list(buf["writes"]),
                        "kv_steps": buf["kv_steps"],
                        "n_generated": len(buf["tokens"]),
                        "ttft_ms": buf["ttft_ms"],
                        "streaming": not buf["done"]}

        st = {}
        if snap is None:
            # 没有任何流缓冲：取上一轮非流式推理留下的统计（超时 2s，不阻塞）
            st = self.supervisor.query("infer", "stats", timeout=2) or {}
            streaming = False
            fw = st.get("fw")
            fw_writes = st.get("fw_writes") or []
            kv_steps = int(st.get("kv_steps") or 0)
            n_generated = int(st.get("n_generated") or 0)
            ttft_ms = st.get("ttft_ms")
            tps = st.get("tps") or 0
        else:
            streaming = snap["streaming"]
            # 用 buf 的实时值覆盖统计（生成中每收到一个 partial 都在更新）
            fw = snap["fw"]
            fw_writes = snap["writes"]
            kv_steps = snap["kv_steps"]
            n_generated = snap["n_generated"]
            ttft_ms = snap["ttft_ms"]
            # 粗略 tps：只有首 token 时间 ttft_ms 可用，无首 token 时记 0
            tps = (round(n_generated * 1000.0 / ttft_ms, 2)
                   if ttft_ms else 0)

        return {
            "active": self.mgr.name,
            # 有流缓冲时 st 为空，不能再拿 st["loaded"] 判断；直接查句柄表
            "running": self.supervisor.running("infer") or streaming,
            "streaming": streaming,
            "ctx_full": int(st.get("ctx_full") or cfg.get("ctx_full")
                            or spec.CTX_FULL),
            "ctx_compress": int(st.get("ctx_compress")
                                or cfg.get("ctx_compress")
                                or spec.CTX_COMPRESS),
            "kv_steps": int(kv_steps),
            "kv_bytes": int(st.get("kv_bytes") or 0),
            "d_model": int(st.get("d_model") or model_cfg.get("d_model")
                           or spec.D_MODEL),
            "vocab_size": int(st.get("vocab_size")
                              or model_cfg.get("vocab_size") or 0),
            "n_generated": int(n_generated),
            "ttft_ms": ttft_ms,
            "tps": tps,
            "fw": fw,
            "fw_writes": fw_writes,
            "fw_bins": spec.FW_VIZ_BINS,
            "fw_write_history": spec.FW_WRITE_HISTORY,
        }

    # ---------------- system / vocab ----------------
    # 模块名 → 发行包名（importlib.metadata 认的是后者，PIL 装在 pillow 里）
    _DIST_NAME = {"PIL": "pillow", "cv2": "opencv-python"}

    # 主页「Python 依赖」区的完整清单由 start/requirements/{must,necessary}.txt
    # 声明（server/reqs.py 解析）——**所有依赖都列出来**，不分轻重：
    # must 是启动必备（轻量），necessary 是正常工作需要（启动不检查）。
    # 以前只列 6 个 Python 模块，pywebview / pytest / ffmpeg / rust 装了也
    # 看不见，用户没法判断「我这个环境到底缺什么」。

    def api_system(self):
        import importlib.metadata as _md
        # 延迟 import torch：server 本体不依赖 torch（重活都在服务进程），
        # 只有这个「环境与依赖」端点需要报版本号
        try:
            import torch
        except Exception:
            torch = None
        # 依赖清单是唯一真源：start/requirements/{must,necessary}.txt
        # （server/reqs.py 负责解析）。已装的、没装的、需要但没装的，全在这里。
        from reqs import all_requirements, dep_state
        deps = []
        for d in all_requirements():
            st = dep_state(d)
            deps.append({
                "name": d["name"],
                "title": d.get("title") or d["name"],
                "group": d["group"],                 # must | necessary
                "kind": d.get("kind") or "pip",       # pip|runtime|system|toolchain
                "min": d.get("min") or "",
                "constraints": d.get("constraints") or [],
                "raw": d.get("raw") or "",
                "ok": st["ok"],
                "version": st["version"],
                "satisfied": st["satisfied"],
                "install": d.get("install") or "",
                "pack": d.get("pack") or "",
                "module": d.get("module") or "",
                "dist": d.get("dist") or "",
            })
        device = "unknown"
        if torch is not None:
            device = "cpu"
            if torch.cuda.is_available():
                device = f"cuda ({torch.cuda.get_device_name(0)})"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        # environment/ 下每个文件夹 = 一个依赖包，主页「环境包」区就是它；
        # 系统级外部命令（ffmpeg / 编译器）也在这里——它们装不进
        # environment/，容器化时得在基础镜像里解决。
        from server_env import detect_tools, list_env_packs
        return 200, {
            "torch": getattr(torch, "__version__", None),
            "device": device,
            "dependencies": deps,
            "packs": list_env_packs(),
            "tools": detect_tools(),
            "optional_missing": [d["name"] for d in deps
                                 if d["group"] == "necessary" and not d["ok"]],
            "ctx": self.mgr.ctx,
            "kv_note": self.mgr.ctx_note(),
        }

    def api_system_install_optional(self, body):
        """ 在 UI 里补装依赖（旧的「只装可选依赖」入口，保留兼容）。

        2026-11-01：所有依赖都走同一个安装端点（/api/system/install），
        这个方法只是转发过去——「可选/必需」的分级现在由清单决定
        （must.txt vs necessary.txt），不再由端点决定。
        """
        return self.api_system_install(body)

    def api_vocab(self, query):
        q = (query.get("q", [""])[0] or "").lower()
        # 2026-10-30：支持 ?name=<词表>，否则沿用进程当前词表。
        v = resolve_vocab_query(query)
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
                     "name": v.name,
                     "categories": list(CATEGORIES)}

    # ---------------- 推理显示插件（导入 / 注册表） ----------------
    def api_plugin_list(self):
        # 只返回注册表里的导入插件；内置 tokens/base 由前端静态声明，不在此列
        return {"plugins": _plugin_read_registry()}

    def api_plugin_import(self, body):
        pid = _plugin_check_id(body.get("id"))
        code = body.get("code")
        if not isinstance(code, str) or not code:
            raise ValueError("code 必须是非空字符串")
        if "export default" not in code:
            raise ValueError("code 必须包含 export default")
        css = body.get("css")
        if css is None:
            css = ""
        if not isinstance(css, str):
            raise ValueError("css 必须是字符串")
        # 体积按 UTF-8 字节算；孤立代理项等非法 Unicode 一并拒绝
        for label, val, limit in (("code", code, 512 * 1024),
                                  ("css", css, 256 * 1024)):
            try:
                n = len(val.encode("utf-8"))
            except UnicodeEncodeError:
                raise ValueError(f"{label} 含非法 Unicode 字符")
            if n > limit:
                raise ValueError(f"{label} 超过 {limit // 1024}KiB")
        force = self._bool_arg(body.get("force"), "force")

        js_path = _plugin_target(pid, ".js")     # 先校验路径安全，再落任何文件
        css_path = _plugin_target(pid, ".css")
        reg = _plugin_read_registry()
        # 同名且未带 force:true → FileExistsError（do_POST 已映射 409）
        if any(p["id"] == pid for p in reg) and not force:
            raise FileExistsError(f"插件已存在: {pid}（可带 force:true 覆盖）")

        name, desc = _plugin_meta_from_code(code, pid)
        plugin = {"id": pid, "name": name, "desc": desc,
                  "entry": f"plugins/{pid}.js",
                  "css": f"plugins/{pid}.css" if css else ""}
        # 只写 .js / .css；路径已过 realpath 白名单，内置文件被挡住
        _plugin_write_text(js_path, code)
        if css:
            _plugin_write_text(css_path, css)
        else:
            # 覆盖导入时旧版带 CSS 而新版不带：删掉遗留 css，避免幽灵样式
            try:
                os.unlink(css_path)
            except FileNotFoundError:
                pass
        # 注册表最后写（原子替换），保证它只指向已完整落盘的插件
        reg = [p for p in reg if p["id"] != pid] + [plugin]
        _plugin_write_registry(reg)
        return {"ok": True, "plugin": plugin}

    def api_plugin_remove(self, body):
        pid = _plugin_check_id(body.get("id"))
        reg = _plugin_read_registry()
        # 只允许删注册表里存在的导入插件（内置插件不在注册表，天然不可删）
        if not any(p["id"] == pid for p in reg):
            raise ValueError(f"导入插件不存在: {pid}")
        js_path = _plugin_target(pid, ".js")
        css_path = _plugin_target(pid, ".css")
        for p in (js_path, css_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass            # 文件不存在则忽略
        _plugin_write_registry([p for p in reg if p["id"] != pid])
        return {"ok": True, "removed": pid}


# ---------------------------------------------------------------------------
# HTTP 层
# ---------------------------------------------------------------------------
class BodyTooLarge(ValueError):
    """ 请求体超过 MAX_BODY_BYTES → 413。 """


def _finite(obj):
    """递归把非有限 float 换成 0（_send_json 的全端点兜底消毒）。"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else 0
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


class Handler(BaseHTTPRequestHandler):
    server_version = "GargantuaHTTP/1.0"
    protocol_version = "HTTP/1.1"
    timeout = 30                          # socket 超时：慢速恶意连接不占死线程

    # ---------------- 来源防护（2026-10-29 审计 S2） ----------------
    # 默认绑定 127.0.0.1，但 HTTP 层此前完全信任 Host / Origin：
    #   * DNS rebinding：恶意域名解析到 127.0.0.1，浏览器视其同源，
    #     可读 /api/vocab 等状态、可用简单请求盲打状态变更端点；
    #   * 任意网页可用 text/plain 简单 POST 盲打 /api/system/install。
    # 三道闸门：Host 白名单 + Origin/Sec-Fetch-Site 校验 + 状态变更 POST
    # 必须 application/json（非 CORS-safelisted，跨站会触发预检，而本服务
    # 不支持 OPTIONS/CORS，预检必然失败）。本机 pywebview 同源请求不受影响。
    _LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}

    @staticmethod
    def _host_only(value):
        """从 'host:port' / '[::1]:8712' 里取出 host 部分。"""
        v = (value or "").strip()
        if v.startswith("["):
            return (v.split("]", 1)[0] + "]").lower()
        return (v.rsplit(":", 1)[0] if ":" in v else v).lower()

    def _guard_request(self, is_post=False):
        bound = ""
        try:
            bound = self.server.server_address[0]
        except Exception:
            pass
        # 显式绑定非回环（--host 0.0.0.0 供局域网访问）时跳过 Host 校验，
        # 但下面的 Origin / Content-Type 校验仍然生效。
        strict_host = bound in ("127.0.0.1", "::1", "localhost", "")
        host = (self.headers.get("Host") or "").strip()
        if strict_host and self._host_only(host) not in self._LOCAL_HOSTS:
            self._send_error_json(403, f"非法 Host: {host!r}")
            return False
        if not is_post:
            return True
        origin = (self.headers.get("Origin") or "").strip()
        if origin and origin.lower() != "null":
            oh = origin.split("://", 1)[-1].split("/", 1)[0]
            if self._host_only(oh) not in self._LOCAL_HOSTS:
                self._send_error_json(403, f"跨站来源被拒绝: {origin!r}")
                return False
        if (self.headers.get("Sec-Fetch-Site") or "").lower() == "cross-site":
            self._send_error_json(403, "跨站请求被拒绝")
            return False
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype != "application/json":
            self._send_error_json(403, "状态变更请求必须使用 application/json")
            return False
        return True

    def do_OPTIONS(self):
        # 明确拒绝预检：不返回任何 CORS 头，跨站 application/json 必然失败
        self._send_error_json(403, "不支持跨源预检")

    def log_message(self, fmt, *args):
        pass                                    # 静默访问日志
    # ---------------- 基础工具 ----------------
    def _send_json(self, obj, code=200):
        # allow_nan=False：默认会输出非法 JSON 的 NaN/Infinity，浏览器
        # JSON.parse 直接炸；出现异常再递归消毒后重 dump，作为全端点兜底
        try:
            data = json.dumps(obj, ensure_ascii=False,
                              allow_nan=False).encode("utf-8")
        except ValueError:
            data = json.dumps(_finite(obj), ensure_ascii=False,
                              allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error_json(self, code, msg, extra=None):
        try:
            obj = {"error": msg}
            if extra:
                obj.update(extra)
            self._send_json(obj, code)
        except Exception:
            pass

    def _send_file(self, abspath, ctype):
        """按绝对路径发一个文件（调用方负责校验路径来源）。"""
        if not os.path.isfile(abspath):
            self._send_error_json(404, "文件不存在")
            return
        with open(abspath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, name):
        # 只放白名单里的前端资源：app.css / app.js，以及 plugins/ 下的
        # .js/.css/.md（推理页显示插件是独立 ES 模块，按目录服务）。
        # 其它路径一律 403，杜绝任意文件读取与穿越；注册表 _imported.json
        # 是 .json，不在白名单内，永远不会被静态服务，只能走 /api/plugin/*。
        ok = name in ("app.css", "app.js") or (
            name.startswith("plugins/")
            and name.endswith((".js", ".css", ".md"))
            and ".." not in name)
        if not ok:
            self._send_error_json(403, f"禁止访问: {name}")
            return
        base = os.path.realpath(WEB_DIR)
        abspath = os.path.realpath(os.path.join(base, name))
        if abspath != os.path.join(base, name) and not abspath.startswith(base + os.sep):
            self._send_error_json(403, "禁止访问")
            return
        if not os.path.isfile(abspath):
            self.send_error(404, "Not Found")
            return
        ctype = ("text/css; charset=utf-8" if name.endswith(".css")
                 else "text/markdown; charset=utf-8" if name.endswith(".md")
                 else "application/javascript; charset=utf-8")
        with open(abspath, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(data)

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
        if not self._guard_request(False):
            return
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
            elif path == "/api/term/poll":
                _code, obj = app.api_term_poll(query)
                self._send_json(obj, _code)
            elif path == "/api/prims/log":
                _code, obj = app.api_prims_log(query)
                self._send_json(obj, _code)
            elif path == "/api/prims/jobs":
                _code, obj = app.api_prims_jobs(query)
                self._send_json(obj, _code)
            elif path == "/api/prims/job/live":
                _code, obj = app.api_prims_job_live(query)
                self._send_json(obj, _code)
            elif path == "/api/prims/job/result":
                _code, obj = app.api_prims_job_result(query)
                self._send_json(obj, _code)
            elif path == "/api/prims/job/source":
                _code, obj = app.api_prims_job_source(query)
                self._send_file(obj["path"], "image/png")
            elif path == "/api/system/updates":
                self._send_json(app.api_system_updates(query)[1])
            elif path == "/api/system/versions":
                # 某个 pip 依赖在镜像上的全部版本（用户挑版本用）
                self._send_json(app.api_system_versions(query)[1])
            elif path == "/api/infer/stats":
                self._send_json(app.api_infer_stats())
            elif path == "/api/infer/stream":
                _code, obj = app.api_infer_stream(query)
                self._send_json(obj, _code)
            elif path == "/api/plugin/list":
                self._send_json(app.api_plugin_list())
            elif path == "/api/vocab":
                _code, obj = app.api_vocab(query)
                self._send_json(obj, _code)
            elif path == "/api/vocab/list":
                _code, obj = app.api_vocab_list(query)
                self._send_json(obj, _code)
            elif path == "/api/vocab/regions":
                _code, obj = app.api_vocab_regions(query)
                self._send_json(obj, _code)
            elif path == "/api/vocab/source":
                _code, obj = app.api_vocab_source_get(query)
                self._send_json(obj, _code)
            elif path in ("/app.css", "/app.js") or path.startswith("/plugins/"):
                self._serve_static(path.lstrip("/"))
            else:
                self._send_error_json(404, f"未知路径: {path}")
        except BrokenPipeError:
            pass
        except ValueError as e:
            # 与 do_POST 一致：api_infer_stream 的非法 from / 流过期都是入参问题
            self._send_error_json(400, str(e))
        except Exception as e:
            self._send_error_json(500, str(e))

    def do_POST(self):
        app = self.server.app
        if not self._guard_request(True):
            return
        path = urlparse(self.path).path
        routes = {
            "/api/model/estimate": app.api_model_estimate,
            "/api/model/init": app.api_model_init,
            "/api/model/load": app.api_model_load,
            "/api/model/delete": app.api_model_delete,
            "/api/infer": app.api_infer,
            "/api/infer/reset": app.api_infer_reset,
            "/api/plugin/import": app.api_plugin_import,
            "/api/plugin/remove": app.api_plugin_remove,
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
            "/api/vocab/token": app.api_vocab_token,
            "/api/vocab/source": app.api_vocab_source,
            "/api/vocab/fork": app.api_vocab_fork,
            "/api/vocab/copy": app.api_vocab_fork,
            "/api/vocab/create": app.api_vocab_create,
            "/api/vocab/delete": app.api_vocab_delete,
            "/api/system/install-optional": app.api_system_install_optional,
            "/api/system/install": app.api_system_install,
            "/api/system/kill": app.api_system_kill,
            "/api/term/open": app.api_term_open,
            "/api/term/input": app.api_term_input,
            "/api/term/interrupt": app.api_term_interrupt,
            "/api/term/kill": app.api_term_kill,
            "/api/term/clear": app.api_term_clear,
            "/api/term/close": app.api_term_close,
            "/api/prims/job/create": app.api_prims_job_create,
            "/api/prims/job/update": app.api_prims_job_update,
            "/api/prims/job/delete": app.api_prims_job_delete,
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
        except (ValueError, FileNotFoundError, KeyError, TypeError,
                binascii.Error) as e:
            # 审计 L2：base64 错值抛 binascii.Error、类型混淆抛 TypeError，
            # 都属入参问题，应 400 而不是 500（5xx 会误报成服务端故障）
            # .vfc 校验失败（VfcListError）带 errors[] 逐行列表，一并透传
            # 给前端做行级高亮——只回一句拼接字符串，用户没法定位。
            extra = ({"errors": e.errors}
                     if isinstance(getattr(e, "errors", None), list) else None)
            self._send_error_json(400, str(e), extra)
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
    # start_new_session=True → 窗口 setsid 到自己的会话，脱离启动它的那个终端。
    # 2026-10-29 真实事故：launcher 把 server 后台化后，双击 启动.command 的
    # 终端窗口会在交接完成后自动关闭；窗口子进程当时和终端同一个会话，
    # 终端一关就收到 SIGHUP/会话销毁，pywebview 触发 closed、窗口凭空消失
    # （日志里只剩一句「窗口已关闭（server 继续运行）」）。setsid 之后终端
    # 怎么关都碰不到它；GUI 权限来自 launchd 的 bootstrap 命名空间，与 tty 无关。
    return subprocess.Popen(cmd, env=_window_env(), start_new_session=True)


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
        app.term.shutdown()
        app.supervisor.shutdown_all()
        kill_proc(window_proc)
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
