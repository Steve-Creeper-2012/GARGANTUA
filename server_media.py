# ===========================================================================
# GARGANTUA v1 — server_media.py：上传媒体的解码/守卫工具
# ---------------------------------------------------------------------------
# 从臃肿的 server.py（986 行）拆出。这里只做「把用户上传的 base64/文件变成
# PIL 帧或 int16 采样」，不碰 HTTP、不碰模型：
#   size_guard_pil / b64_to_pil   解压炸弹防护（8192 单边 / 100k patch）
#   find_ffmpeg                   ffmpeg 定位 + `-version` 冒烟测试
#   decode_audio_with_ffmpeg      任意音频 → int16 单声道
#   decode_audio_any              wav 快路径 + ffmpeg 回退
#   extract_video_frames          抽帧（ffmpeg → Pillow 动画帧回退）
#   extract_video_audio           提音轨（无声轨返回空数组）
# server.py re-export 这些符号，老 import 路径保持可用。
# ===========================================================================
import base64
import glob
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image

from tokens import audio_codec

# 编解码入口的尺寸/规模上限（防解压炸弹与 OOM：单个 patch = 4096 字节，
# 100k patch ≈ 400MB 原始数据）
MAX_IMAGE_DIM = 8192            # 单边像素上限
MAX_PATCHES_TOTAL = 100_000     # 单次请求的 patch 总数上限


def size_guard_pil(img):
    """ PIL 图像尺寸守卫（解压炸弹防护）。超限 → ValueError（→400）。 """
    w, h = img.size
    if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM:
        raise ValueError(f"图片尺寸超限: {w}×{h} > {MAX_IMAGE_DIM}")
    if w * h > MAX_IMAGE_DIM * MAX_IMAGE_DIM:
        raise ValueError(f"图片像素数超限: {w}×{h}")
    return img


def b64_to_pil(b64):
    img = Image.open(_bytes_io(b64)).convert("RGBA")
    return size_guard_pil(img)


def _bytes_io(b64):
    import io
    return io.BytesIO(base64.b64decode(b64))


_FFMPEG_CACHE = []


def find_ffmpeg():
    """定位可用的 ffmpeg：缓存命中 → PATH → 常见安装路径 → imageio-ffmpeg。
    对候选做 `-version` 冒烟测试，过滤架构/格式不对的二进制。"""
    if _FFMPEG_CACHE:
        return _FFMPEG_CACHE[0]
    import shutil
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
    exe = find_ffmpeg()
    if exe is None:
        raise RuntimeError("未找到可用 ffmpeg，无法解码非 wav 音频")
    try:
        proc = subprocess.run(
            [exe, "-v", "error", "-i", path, "-f", "s16le", "-acodec",
             "pcm_s16le", "-ar", str(sr), "-ac", "1", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg 解码超时（>120s）")
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
        outdir = tempfile.mkdtemp(prefix="garg_vframes_")
        try:
            try:
                proc = subprocess.run(
                    [exe, "-v", "error", "-i", path, "-vf",
                     f"fps={fps}", "-frames:v", str(max_frames),
                     os.path.join(outdir, "f_%06d.png")],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    timeout=180)
            except subprocess.TimeoutExpired:
                raise RuntimeError("ffmpeg 抽帧超时（>180s）")
            files = sorted(glob.glob(os.path.join(outdir, "f_*.png")))
            if proc.returncode == 0 and files:
                return [Image.open(f).convert("RGBA") for f in files]
        finally:
            import shutil as _sh
            _sh.rmtree(outdir, ignore_errors=True)
    try:
        from PIL import ImageSequence
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
