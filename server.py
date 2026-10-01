# ===========================================================================
# GARGANTUA v1 — server.py：HTTP 后端（仅标准库 http.server）
# ---------------------------------------------------------------------------
# 用法：python server.py [--port 8712] [--model-dir models] [--data-dir data]
# 绑定 127.0.0.1；内存中同一时刻一个活动模型；models/ 下唯一模型自动加载
# （后台线程）；所有模型操作走一把全局锁，训练与推理互斥。
# 重计算（推理 / 新建模型）一律异步任务：202 + task_id，GET /api/task/{id}
# 长轮询取结果。
# ===========================================================================
import argparse
import base64
import glob
import io
import json
import os
import tempfile
import threading
import time
import uuid

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import torch
from safetensors.torch import load_file, save_file

import spec
from model import Gargantua
from infer import generate
from tokens.text_codec import decode as text_decode, encode as text_encode
from tokens.vocab import get_vocab
from tokens import image_codec, audio_codec, midi_codec, draw_codec
from tokens.image_to_primitives import image_to_primitives

CKPT_NAME = "model.safetensors"
WEB_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "web", "index.html")


# ---------------------------------------------------------------------------
# 模型管理
# ---------------------------------------------------------------------------
class ModelManager:
    """同一时刻一个活动模型；所有模型操作走 self.lock（训练/推理互斥）。"""

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.lock = threading.RLock()
        self.model = None
        self.name = None
        self.params = 0
        self.loading = None                 # 正在后台加载的模型名
        os.makedirs(model_dir, exist_ok=True)

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
        d = os.path.join(self.model_dir, name)
        ckpt = os.path.join(d, CKPT_NAME)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"模型不存在: {name}")
        enc_b = dec_b = None
        cfg_path = os.path.join(d, "config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                enc_b = cfg.get("n_enc_blocks")
                dec_b = cfg.get("n_dec_blocks")
            except Exception:
                pass
        m = Gargantua(n_enc_blocks=enc_b, n_dec_blocks=dec_b)
        m.load_state_dict(load_file(ckpt))
        m.eval()
        params = sum(p.numel() for p in m.parameters())
        with self.lock:
            self.model, self.name, self.params = m, name, params
        return {"model": name, "params": params}

    def init_new(self, name, n_enc_blocks=None, n_dec_blocks=None):
        """新建随机初始化模型并落盘（与 init.py 相同的 save_file 方式）。"""
        if not name or os.sep in name or name in (".", ".."):
            raise ValueError(f"非法模型名: {name!r}")
        d = os.path.join(self.model_dir, name)
        os.makedirs(d, exist_ok=True)
        m = Gargantua(n_enc_blocks=n_enc_blocks, n_dec_blocks=n_dec_blocks)
        params = sum(p.numel() for p in m.parameters())
        save_file(m.state_dict(), os.path.join(d, CKPT_NAME))
        config = {
            "name": name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": params,
            "d_model": spec.D_MODEL, "head_dim": spec.HEAD_DIM,
            "n_enc_blocks": n_enc_blocks or spec.N_ENC_BLOCKS,
            "n_dec_blocks": n_dec_blocks or spec.N_DEC_BLOCKS,
            "vocab_size_padded": spec.VOCAB_SIZE_PADDED,
            "ctx_full": spec.CTX_FULL, "ctx_compress": spec.CTX_COMPRESS,
            "csa_ratio": spec.CSA_RATIO, "hca_ratio": spec.HCA_RATIO,
            "optimizer": spec.OPTIMIZER,
        }
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
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
    def __init__(self, workers=4):
        self._ex = ThreadPoolExecutor(max_workers=workers)
        self._tasks = {}
        self._lock = threading.Lock()

    def submit(self, fn):
        tid = uuid.uuid4().hex
        with self._lock:
            self._tasks[tid] = {"status": "running", "result": None,
                                "error": None}

        def run():
            try:
                res = fn()
                with self._lock:
                    self._tasks[tid].update(status="done", result=res)
            except Exception as e:
                with self._lock:
                    self._tasks[tid].update(status="error", error=str(e))

        self._ex.submit(run)
        return tid

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


def b64_to_pil(b64):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")


_FFMPEG_CACHE = []


def find_ffmpeg():
    """定位可用的 ffmpeg：缓存命中 → PATH → 常见安装路径 → imageio-ffmpeg。
    对候选做 `-version` 冒烟测试，过滤架构/格式不对的二进制。"""
    if _FFMPEG_CACHE:
        return _FFMPEG_CACHE[0]
    import shutil
    import subprocess
    candidates = []
    exe = shutil.which("ffmpeg")
    if exe:
        candidates.append(exe)
    candidates += [
        "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg",
        os.path.expanduser(
            "~/Library/Application Support/TRAE SOLO CN/ModularData/"
            "ai-agent/vm/tools/bin/ffmpeg"),
    ]
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    for cand in candidates:
        if not cand or not os.path.exists(cand):
            continue
        try:
            r = subprocess.run([cand, "-version"], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=5)
            if r.returncode == 0:
                _FFMPEG_CACHE.append(cand)
                return cand
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def decode_audio_with_ffmpeg(path, sr=16000):
    """ffmpeg 解码任意音频（wav/mp3/flac/ogg…）为 int16 单声道采样。"""
    import subprocess
    exe = find_ffmpeg()
    if exe is None:
        raise RuntimeError("未找到可用 ffmpeg，无法解码非 wav 音频")
    proc = subprocess.run(
        [exe, "-v", "error", "-i", path, "-f", "s16le", "-acodec",
         "pcm_s16le", "-ar", str(sr), "-ac", "1", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg 解码失败: " +
                           proc.stderr.decode(errors="replace")[:200])
    return np.frombuffer(proc.stdout, dtype=np.int16).copy()


def decode_audio_any(path, filename="", sr=16000):
    """优先 WAV 快速路径；其余格式走 ffmpeg 强兼容解码。"""
    if os.path.splitext(filename.lower())[1] == ".wav":
        try:
            samples, in_sr = audio_codec.read_wav(path)
            return audio_codec.resample_linear(samples, in_sr, sr)
        except Exception:
            pass                       # wav 头损坏也回退 ffmpeg
    return decode_audio_with_ffmpeg(path, sr)


def extract_video_frames(path, fps=30, max_frames=300):
    """ffmpeg 抽帧（mp4/mov/webm/mkv/gif 通吃）；失败回退 Pillow 动画帧。"""
    exe = find_ffmpeg()
    if exe is not None:
        import subprocess
        outdir = tempfile.mkdtemp(prefix="garg_vframes_")
        try:
            proc = subprocess.run(
                [exe, "-v", "error", "-i", path, "-vf",
                 f"fps={fps}", "-frames:v", str(max_frames),
                 os.path.join(outdir, "f_%06d.png")],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            files = sorted(glob.glob(os.path.join(outdir, "f_*.png")))
            if proc.returncode == 0 and files:
                from PIL import Image
                return [Image.open(f).convert("RGBA") for f in files]
        finally:
            import shutil as _sh
            _sh.rmtree(outdir, ignore_errors=True)
    try:
        from PIL import Image, ImageSequence
        im = Image.open(path)
        return [f.convert("RGBA") for f in ImageSequence.Iterator(im)][:max_frames]
    except Exception as e:
        raise RuntimeError(f"无法解析视频（无 ffmpeg 且不是动画图）: {e}")


def extract_video_audio(path, sr=16000):
    """ffmpeg 提取音轨；无声轨返回空数组。"""
    try:
        return decode_audio_with_ffmpeg(path, sr)
    except Exception:
        return np.zeros(0, dtype=np.int16)


# ---------------------------------------------------------------------------
# 应用层（路由处理函数）
# ---------------------------------------------------------------------------
class App:
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
        save_file(model.state_dict(),
                  os.path.join(self.mgr.model_dir, self.mgr.name, CKPT_NAME))

    # ---------------- status / models ----------------
    def api_status(self):
        return {
            "model": self.mgr.name,
            "loading": self.mgr.loading,
            "params": self.mgr.params,
            "training": self.trainer.running,
            "queue": len(glob.glob(os.path.join(self.queue_dir, "*.txt"))),
            "kv_note": (f"KV: {spec.CTX_FULL} full + {spec.CTX_COMPRESS} "
                        f"compress = {spec.CTX_TOTAL}；{spec.TS_PAD_NOTE}"),
        }

    def api_models(self):
        return {"models": self.mgr.list_models(), "active": self.mgr.name}

    def api_model_init(self, body):
        name = body.get("name")
        if not name:
            raise ValueError("缺少 name")
        enc = body.get("n_enc_blocks")
        dec = body.get("n_dec_blocks")
        if enc is not None:
            enc = max(1, min(64, int(enc)))
        if dec is not None:
            dec = max(1, min(64, int(dec)))
        tid = self.pool.submit(lambda: self.mgr.init_new(name, enc, dec))
        return 202, {"task_id": tid}

    def api_model_load(self, body):
        name = body.get("name")
        if not name:
            raise ValueError("缺少 name")
        return 200, self.mgr.load(name)

    # ---------------- infer ----------------
    def api_infer(self, body):
        prompt = body.get("prompt")
        if not prompt:
            raise ValueError("缺少 prompt")
        if self.mgr.model is None:
            raise RuntimeError("无活动模型")
        max_new = int(body.get("max_new_tokens", 32))
        temperature = float(body.get("temperature", 0.8))
        top_p = float(body.get("top_p", 0.95))

        def work():
            with self.mgr.lock:                     # 推理与训练互斥
                steps = generate(self.mgr.model, prompt, max_new,
                                 temperature, top_p)
            return {"text": text_decode(steps),
                    "token_ids": [s["token_ids"][0] for s in steps]}

        tid = self.pool.submit(work)
        return 202, {"task_id": tid}

    # ---------------- train ----------------
    def api_train_enqueue(self, body):
        """训练数据：json_b64 上传。内容必须是
        {"text": "..."} 对象 / 对象数组 / JSONL 每行一个对象。"""
        json_b64 = body.get("json_b64")
        if not json_b64:
            raise ValueError("缺少 json_b64（仅支持上传 JSON/JSONL 训练数据）")
        raw = base64.b64decode(json_b64).decode("utf-8")
        filename = body.get("filename", "")
        samples = []
        if filename.endswith(".jsonl"):
            for i, line in enumerate(raw.splitlines()):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict) or "text" not in obj:
                    raise ValueError(f"第 {i+1} 行缺少 text 字段，"
                                     f"训练数据必须为 {{\"text\": \"...\"}}")
                samples.append(obj["text"])
        else:
            obj = json.loads(raw)
            items = obj if isinstance(obj, list) else [obj]
            for it in items:
                if not isinstance(it, dict) or "text" not in it:
                    raise ValueError("每条训练数据必须为 "
                                     "{\"text\": \"...\"} 对象")
                samples.append(it["text"])
        if not samples:
            raise ValueError("训练数据为空")
        optimizer = body.get("optimizer", spec.OPTIMIZER)
        if optimizer not in ("muon", "adamw"):
            raise ValueError(f"未知优化器: {optimizer}")
        queued = []
        for text in samples:
            fn = f"{time.strftime('%Y%m%d_%H%M%S')}_{optimizer}_" \
                 f"{uuid.uuid4().hex[:8]}.txt"
            path = os.path.join(self.queue_dir, fn)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            queued.append(fn)
        return 200, {"queued": queued, "n_samples": len(samples)}

    def api_train_start(self, body):
        if self.mgr.model is None:
            raise RuntimeError("无活动模型，请先加载模型")
        if body.get("block_size"):
            self.trainer.block_size = int(body["block_size"])
        if body.get("steps"):
            self.trainer.steps_per_file = int(body["steps"])
        if body.get("optimizer"):
            opt = body["optimizer"]
            if opt not in ("muon", "adamw"):
                raise ValueError(f"未知优化器: {opt}")
            self.trainer.optimizer = opt
        started = self.trainer.start()
        return 200, {"started": started, **self.trainer.status()}

    def api_train_stop(self, _body):
        self.trainer.stop()
        return 200, {"stopping": True, **self.trainer.status()}

    def api_train_status(self):
        return self.trainer.status()

    # ---------------- codecs ----------------
    def api_codec_image(self, body):
        b64 = body.get("image_b64")
        if not b64:
            raise ValueError("缺少 image_b64")
        step = image_codec.image_to_step(b64_to_pil(b64))
        return 200, {"steps": [step_to_json(step)],
                     "archive": image_codec.step_to_archive(step)}

    def api_codec_frames(self, body):
        frames_b64 = body.get("frames_b64")
        if not frames_b64:
            raise ValueError("缺少 frames_b64")
        frames = [b64_to_pil(b) for b in frames_b64]
        steps = image_codec.frames_to_steps(frames)
        stats = [{"frame_idx": s["meta"]["frame_idx"],
                  "full": s["meta"]["full"],
                  "grid": s["meta"]["grid"],
                  "n_patches": len(s["meta"]["patches"])} for s in steps]
        return 200, {"frames": stats,
                     "delta_rule": ("第 1 帧全量；2..30 帧只送有变化 patch；"
                                    "第 31 帧（每秒首帧）全量；全变帧全量并重新"
                                    "计数；最后一帧强制全量"),
                     "steps": [step_to_json(s) for s in steps]}

    def api_codec_audio(self, body):
        """音频 → 音元 token 序列。支持 wav/mp3/flac/ogg（ffmpeg 强兼容）。"""
        b64 = body.get("audio_b64")
        if not b64:
            raise ValueError("缺少 audio_b64")
        data = base64.b64decode(b64)
        filename = body.get("filename", "audio.wav")
        suffix = os.path.splitext(filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            apath = f.name
        try:
            samples = decode_audio_any(apath, filename, spec.AUDIO_SR)
        finally:
            os.unlink(apath)
        steps = audio_codec.waveform_to_steps(samples)
        n = len(steps)
        return 200, {
            "n_tokens": n,
            "duration_s": round(n * spec.AUDIO_SAMPLES_PER_TOKEN
                                / spec.AUDIO_SR, 3),
            "filename": filename,
            "steps": [step_to_json(s) for s in steps],
        }

    def api_codec_midi(self, body):
        if "midi_b64" in body:
            data = base64.b64decode(body["midi_b64"])
            with tempfile.NamedTemporaryFile(suffix=".mid",
                                             delete=False) as f:
                f.write(data)
                tmp = f.name
            try:
                tokens = midi_codec.midi_to_tokens(tmp)
            finally:
                os.unlink(tmp)
            return 200, {"tokens": tokens, "n": len(tokens)}
        if "tokens" in body:
            tokens = [int(t) for t in body["tokens"]]
            with tempfile.NamedTemporaryFile(suffix=".mid",
                                             delete=False) as f:
                tmp = f.name
            try:
                midi_codec.tokens_to_midi(tokens, tmp)
                with open(tmp, "rb") as f:
                    data = f.read()
            finally:
                os.unlink(tmp)
            return 200, {"midi_b64": base64.b64encode(data).decode(),
                         "n": len(tokens)}
        raise ValueError("缺少 midi_b64 或 tokens")

    def api_codec_draw(self, body):
        if "primitives" in body:
            prims = [draw_codec.Primitive(
                x=p.get("x", 0), y=p.get("y", 0),
                shape=p.get("shape", "rectangle"),
                width=p.get("width", 0), length=p.get("length", 0),
                rot=p.get("rot", 0), r=p.get("r", 0),
                g=p.get("g", 0), b=p.get("b", 0))
                for p in body["primitives"]]
        elif "tokens" in body:
            prims = draw_codec.tokens_to_primitives(
                [int(t) for t in body["tokens"]])
        else:
            raise ValueError("缺少 primitives 或 tokens")
        tokens = draw_codec.primitives_to_tokens(prims)
        buf = draw_codec.render_rgba(prims)
        png = draw_codec.encode_png(bytes(buf), spec.CANVAS_W, spec.CANVAS_H)
        return 200, {
            "png_b64": base64.b64encode(png).decode(),
            "svg": draw_codec.primitives_to_svg(prims),
            "tokens": tokens,
            "n_primitives": len(prims),
        }

    # ---------------- system ----------------
    def api_system(self):
        import torch
        deps = []
        for mod in ["torch", "numpy", "PIL", "safetensors", "cv2", "soundfile"]:
            try:
                __import__(mod)
                deps.append({"name": mod, "ok": True})
            except Exception as e:
                deps.append({"name": mod, "ok": False, "note": str(e)})
        device = "cpu"
        if torch.cuda.is_available():
            device = f"cuda ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        return 200, {
            "torch": torch.__version__,
            "device": device,
            "dependencies": deps,
            "kv_note": (f"KV: {spec.CTX_FULL} full + {spec.CTX_COMPRESS} "
                        f"compress = {spec.CTX_TOTAL}；{spec.TS_PAD_NOTE}"),
        }

    # ---------------- train queue management ----------------
    def api_train_queue(self):
        items = []
        for path in sorted(glob.glob(os.path.join(self.queue_dir, "*"))):
            name = os.path.basename(path)
            # 文件名形如 20260928_153000_muon_ab12cd34.txt，中段为优化器
            opt = ""
            parts = name.split("_")
            if len(parts) >= 3 and parts[2] in ("muon", "adamw"):
                opt = parts[2]
            items.append({"name": name, "size": os.path.getsize(path),
                          "optimizer": opt})
        return 200, {"items": items}

    def api_train_delete(self, body):
        name = body.get("name")
        if not name:
            raise ValueError("缺少 name")
        path = os.path.join(self.queue_dir, os.path.basename(name))
        if not os.path.exists(path):
            raise FileNotFoundError(f"队列项不存在: {name}")
        os.unlink(path)
        return 200, {"deleted": name}

    # ---------------- text codec ----------------
    def api_codec_text(self, body):
        text = body.get("text")
        if text is None:
            raise ValueError("缺少 text")
        steps = text_encode(text)
        return 200, {"steps": [step_to_json(s) for s in steps]}

    # ---------------- video codec ----------------
    def api_codec_video(self, body):
        b64 = body.get("video_b64")
        if not b64:
            raise ValueError("缺少 video_b64")
        data = base64.b64decode(b64)
        suffix = os.path.splitext(body.get("filename", ".mp4"))[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            vpath = f.name
        try:
            frames = extract_video_frames(vpath)
            audio_samples = extract_video_audio(vpath)
            visual_steps = image_codec.frames_to_steps(frames)
            audio_steps = audio_codec.waveform_to_steps(audio_samples)
            aligned = audio_codec.align_audio_to_frames(audio_steps, len(frames))
            # 合并：每一帧的 visual step + 对应音频 steps 作为同一时间步列表
            combined = []
            total_patches = 0
            for vstep, asteps in zip(visual_steps, aligned):
                total_patches += len(vstep["meta"].get("patches", []))
                combined.append({
                    "time_step": [step_to_json(vstep)] +
                                 [step_to_json(a) for a in asteps]
                })
            return 200, {
                "n_frames": len(frames),
                "n_audio_tokens": len(audio_steps),
                "total_patches": total_patches,
                "steps": combined,
            }
        finally:
            os.unlink(vpath)

    # ---------------- image -> primitives ----------------
    def api_codec_image2prims(self, body):
        b64 = body.get("image_b64")
        if not b64:
            raise ValueError("缺少 image_b64")
        max_steps = int(body.get("max_steps", 100))
        quality = body.get("quality", "fast")
        if quality not in ("fast", "slow"):
            quality = "fast"
        tid = self.pool.submit(
            lambda: self._image2prims(b64, max_steps, quality))
        return 202, {"task_id": tid}

    def _image2prims(self, b64, max_steps, quality):
        prims, note = image_to_primitives(
            b64_to_pil(b64), max_steps=max_steps, quality=quality)
        tokens = draw_codec.primitives_to_tokens(prims)
        return {
            "primitives": [{"x": p.x, "y": p.y, "shape": p.shape,
                            "width": p.width, "length": p.length,
                            "rot": p.rot, "r": p.r, "g": p.g, "b": p.b}
                           for p in prims],
            "n_primitives": len(prims),
            "tokens": tokens,
            "svg": draw_codec.primitives_to_svg(prims),
            "note": note,
        }

    # ---------------- vocab ----------------
    def api_vocab(self, query):
        q = (query.get("q", [""])[0] or "").lower()
        v = get_vocab()
        hits = []
        if q:
            for i, tok in enumerate(v.id_to_token):
                if tok.startswith("<reserved:"):
                    continue
                if q in tok.lower():
                    hits.append({"id": i, "token": tok})
                    if len(hits) >= 50:
                        break
        return 200, {"q": q, "hits": hits, "vocab_size": len(v)}


# ---------------------------------------------------------------------------
# HTTP 层
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "GargantuaHTTP/1.0"
    protocol_version = "HTTP/1.1"

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
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            raise ValueError("请求体不是合法 JSON")

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
            "/api/codec/midi": app.api_codec_midi,
            "/api/codec/draw": app.api_codec_draw,
            "/api/codec/text": app.api_codec_text,
            "/api/codec/video": app.api_codec_video,
            "/api/codec/image2prims": app.api_codec_image2prims,
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
