# ===========================================================================
# GARGANTUA v1 — server.py：HTTP 层（仅标准库 http.server）
# ---------------------------------------------------------------------------
# 用法：python server.py [--port 8712] [--model-dir models] [--data-dir data]
# 绑定 127.0.0.1；内存中同一时刻一个活动模型；models/ 下唯一模型自动加载
# （后台线程）；所有模型操作走一把全局锁，训练与推理互斥。
# 重计算（推理 / 新建模型）一律异步任务：202 + task_id，GET /api/task/{id}
# 长轮询取结果。
#
# 2026-10-23 拆分（server.py 从 986 行降到 ~300 行，每层只干一件事）：
#   server_models.py   ModelManager / TaskPool / step_to_json / ConflictError
#   server_media.py    上传媒体解码与尺寸守卫（b64_to_pil / ffmpeg / 抽帧）
#   server_codecs.py   CodecApi   —— /api/codec/* 全部端点
#   server_train.py    TrainApi   —— /api/train/* 全部端点
#   server_tokens.py   TokensApi  —— /api/tokens/* 与 /api/vocab/add
#                        （tok-32 离散 token 编解码与词表扩充）
#   server_util.py     num_param  —— 数值入参取值（零值不被默认值吞掉）
# App 以 mixin 组合上面几个 Api；本文件只留：应用装配 + 路由 + HTTP 细节。
# 原 `from server import X` 的路径全部通过文件末尾的 re-export 保持可用。
#
# HTTP 错误契约（README「HTTP contract」）：
#   400 入参非法   404 路径/任务不存在   409 状态冲突
#   411 需 Content-Length（拒 chunked）   413 请求体 > 512MB
# ===========================================================================
import argparse
import base64
import glob
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import torch

# ---- GARGANTUA 路径引导（目录树重构：把 server/ 与 service/* 放上 path）----
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "start" / "boot.py").is_file():
        _sys.path.insert(0, str(_anc / "start"))
        break
else:
    raise SystemExit("找不到 GARGANTUA 项目根（start/boot.py）")
import boot as _boot
_boot.bootstrap()
from common import (CKPT_NAME, MAX_BLOCKS, read_config, save_checkpoint,
                    valid_model_name, validate_ctx_config)
from infer import MAX_NEW_TOKENS, generate
from tokens.text_codec import decode as text_decode, encode as text_encode
from tokens.vocab import CATEGORIES, get_vocab

# 拆出去的模块在这里 re-export：调用方无需知道内部怎么分文件
from server_models import (ConflictError, ModelManager, TaskPool,
                           step_to_json)
from server_media import (MAX_IMAGE_DIM, MAX_PATCHES_TOTAL, b64_to_pil,
                          decode_audio_any, extract_video_audio,
                          extract_video_frames, size_guard_pil)
from server_codecs import CodecApi
from server_train import TrainApi
from server_tokens import TokensApi
from server_util import num_param, optional_install_cmd, pip_index_args

WEB_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "web", "index.html")

# 请求体硬上限（base64 后的视频/音频可能上百 MB，512MB 是宽松保护值）
MAX_BODY_BYTES = 512 * 1024 * 1024


# ---------------------------------------------------------------------------
# 应用装配：把模型/任务/训练/编解码四块拼成可路由的对象
# ---------------------------------------------------------------------------
class App(TrainApi, CodecApi, TokensApi):
    def __init__(self, model_dir, data_dir, log_path):
        self.mgr = ModelManager(model_dir)
        self.pool = TaskPool()
        self.data_dir = data_dir
        self.queue_dir = os.path.join(data_dir, "queue")
        os.makedirs(self.queue_dir, exist_ok=True)
        from auto_train import Trainer
        self.trainer = Trainer(
            get_model=lambda: self.mgr.model,
            save_fn=self._save_ckpt,
            lock=self.mgr.lock,
            queue_dir=self.queue_dir,
            done_dir=os.path.join(data_dir, "done"),
            log_path=log_path,
        )

    def _save_ckpt(self, model):
        if self.mgr.name is None:
            raise RuntimeError("无活动模型，无法落盘")
        # 原子写 + 保留既有 config.json（block 数等），仅刷新 params
        d = os.path.join(self.mgr.model_dir, self.mgr.name)
        # 纵深防御：即便名字校验被绕过，落盘路径也不得越出 model_dir
        real_d = os.path.realpath(d)
        real_root = os.path.realpath(self.mgr.model_dir)
        if os.path.commonpath([real_d, real_root]) != real_root:
            raise RuntimeError(f"拒绝越权落盘: {d}")
        cfg = dict(read_config(d))
        cfg["params"] = sum(p.numel() for p in model.parameters())
        save_checkpoint(d, model, cfg)

    # ---------------- status / models / infer ----------------
    def api_status(self):
        n_queued = sum(1 for p in glob.glob(os.path.join(self.queue_dir, "*"))
                       if os.path.isfile(p))
        return {
            "model": self.mgr.name,
            "loading": self.mgr.loading,
            "params": self.mgr.params,
            "training": self.trainer.running,
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
        # 上下文长度（完整/压缩容量 + CSA/HCA 压缩比与局部窗）；
        # 缺省项取 spec 默认，非法值 → ValueError（→400）
        ctx = validate_ctx_config(
            ctx_full=body.get("ctx_full"),
            ctx_compress=body.get("ctx_compress"),
            csa_ratio=body.get("csa_ratio"),
            hca_ratio=body.get("hca_ratio"),
            csa_local_window=body.get("csa_local_window"))
        tid = self.pool.submit(
            lambda: self.mgr.init_new(name, enc, dec,
                                      overwrite=bool(body.get("overwrite")),
                                      ctx=ctx))
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
        return 200, self.mgr.load(name)

    def api_infer(self, body):
        prompt = body.get("prompt")
        if not prompt:
            raise ValueError("缺少 prompt")
        # 2026-10-23 审计修复（P1）：聊天窗会上传 media_b64/media_name，
        # 但 generate() 只有文本路径——此前媒体被静默丢弃，用户以为模型
        # 「看到了」附件。多模态生成尚未接线，显式拒绝好过静默成功。
        if body.get("media_b64") or body.get("media_name"):
            raise ValueError("当前版本推理仅支持文本 prompt；媒体输入"
                             "（image/audio/video）尚未接入生成路径")
        if self.mgr.model is None:
            raise ValueError("无活动模型，请先加载模型")
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
            with self.mgr.lock:                     # 推理与训练互斥
                steps = generate(self.mgr.model, prompt, max_new,
                                 temperature, top_p)
            return {"text": text_decode(steps),
                    "token_ids": [s["token_ids"][0] for s in steps]}

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
        import torch
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
        device = "cpu"
        if torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        return 200, {
            "torch": torch.__version__,
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
        里明确要求重启。主依赖不在这里装——那是 run.py 的职责。 """
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
            import subprocess
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
            elif path == "/api/vocab":
                _code, obj = app.api_vocab(query)
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
            "/api/vocab/add": app.api_vocab_add,
            "/api/system/install-optional": app.api_system_install_optional,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8712)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--log-file", default="logs/auto_train.jsonl")
    args = ap.parse_args()

    torch.manual_seed(0)
    app = App(args.model_dir, args.data_dir, args.log_file)
    app.mgr.auto_load_unique()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.app = app
    srv.daemon_threads = True
    print(f"[server] http://{args.host}:{args.port}  "
          f"model_dir={args.model_dir}  data_dir={args.data_dir}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.trainer.stop()
        srv.server_close()


if __name__ == "__main__":
    main()
