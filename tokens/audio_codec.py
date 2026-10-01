# ===========================================================================
# GARGANTUA v1 — 音频输入 codec（16kHz 单声道 → 音元时间步序列）
# ---------------------------------------------------------------------------
# 契约来源：ROOT/spec.py（冻结，禁止修改）。
#   - AUDIO_SR=16000，单声道；AUDIO_SAMPLES_PER_TOKEN=533（≈30 tok/s）。
#   - 音频 token 不入词表：Step["token_ids"] 恒为空；
#     meta = {"samples": bytes（533 个 float32 小端）, "n": 实际采样数}；
#     末尾不足 533 补零并记录真实长度 n。
#   - 一帧内全部 patch + 该帧音频 token 处于同一时间步
#     → align_audio_to_frames 按 30fps 把音元均分到各帧。
# 纯标准库 + numpy 实现：wave 读 wav，线性插值重采样，不装 librosa。
# ===========================================================================
"""audio_codec: wav / int16 数组 → Step 序列（音频输入，token 不入词表）。"""

from __future__ import annotations

import io
import os
import wave
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

import spec

SR = spec.AUDIO_SR                              # 16000
N = spec.AUDIO_SAMPLES_PER_TOKEN                # 533

Step = Dict
AudioInput = Union[str, os.PathLike, io.IOBase, bytes, np.ndarray,
                   Sequence[int]]


# ---------------------------------------------------------------------------
# 读取与重采样
# ---------------------------------------------------------------------------
def _decode_pcm(raw: bytes, sampwidth: int) -> np.ndarray:
    """PCM 字节流 → int16 单声道样本（未分声道）。"""
    if sampwidth == 1:                          # 8bit unsigned
        a = np.frombuffer(raw, dtype=np.uint8).astype(np.int32)
        return ((a - 128) << 8).astype(np.int16)
    if sampwidth == 2:                          # 16bit
        return np.frombuffer(raw, dtype="<i2").copy()
    if sampwidth == 3:                          # 24bit little-endian
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32)
             | (b[:, 1].astype(np.int32) << 8)
             | (b[:, 2].astype(np.int32) << 16))
        v = np.where(v & 0x800000, v - 0x1000000, v)   # 符号扩展
        return (v >> 8).astype(np.int16)
    if sampwidth == 4:                          # 32bit
        v = np.frombuffer(raw, dtype="<i4")
        return (v >> 16).astype(np.int16)
    raise ValueError(f"unsupported sample width: {sampwidth}")


def read_wav(fileobj: Union[str, os.PathLike, io.IOBase, bytes]
             ) -> Tuple[np.ndarray, int]:
    """读 wav → (int16 单声道数组, 采样率)。立体声混成单声道（均值）。"""
    if isinstance(fileobj, (bytes, bytearray)):
        fileobj = io.BytesIO(fileobj)
    with wave.open(fileobj, "rb") as wf:
        nch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    samples = _decode_pcm(raw, sampwidth)
    if nch > 1:
        samples = samples.reshape(-1, nch).mean(axis=1)
        samples = np.clip(np.rint(samples), -32768, 32767).astype(np.int16)
    return samples, sr


def resample_linear(samples: np.ndarray, sr_in: int,
                    sr_out: int = SR) -> np.ndarray:
    """纯 numpy 线性插值重采样（int16 → int16）。"""
    if sr_in == sr_out:
        return samples.astype(np.int16)
    n_in = len(samples)
    if n_in == 0:
        return samples.astype(np.int16)
    n_out = int(round(n_in * sr_out / sr_in))
    pos = np.arange(n_out) * (sr_in / sr_out)
    out = np.interp(pos, np.arange(n_in), samples.astype(np.float64))
    return np.clip(np.rint(out), -32768, 32767).astype(np.int16)


def load_audio(source: AudioInput) -> np.ndarray:
    """统一入口 → 16kHz 单声道 int16 numpy 数组。

    - wav 文件路径 / 文件对象 / wav 字节：读取、混单声道、必要时重采样；
    - numpy int16 一维数组（或等效序列）：假定已是 16kHz 单声道；
      形状 (n, 2) 视为立体声并混成单声道；float 数组按 [-1,1] 转 int16。
    """
    if isinstance(source, np.ndarray) or (
            isinstance(source, (list, tuple))
            and not isinstance(source, (str, bytes, bytearray))):
        arr = np.asarray(source)
        if arr.dtype.kind == "f":
            arr = np.clip(arr, -1.0, 1.0)
            arr = np.rint(arr * 32767.0).astype(np.int16)
        if arr.ndim == 2:
            arr = np.clip(np.rint(arr.mean(axis=1)), -32768,
                          32767).astype(np.int16)
        return arr.astype(np.int16)
    samples, sr = read_wav(source)
    return resample_linear(samples, sr, SR)


# ---------------------------------------------------------------------------
# Step 构造
# ---------------------------------------------------------------------------
def waveform_to_steps(samples: Union[np.ndarray, Sequence[int]]) -> List[Step]:
    """16kHz 单声道 int16 波形 → 音元 Step 序列。

    每 533 采样一个音元；末尾不足补零，meta["n"] 记录真实采样数。
    samples 字段 = 533 个 float32 小端（int16/32768 归一化到 [-1,1]）。
    """
    arr = np.asarray(samples, dtype=np.int16).reshape(-1)
    steps: List[Step] = []
    for start in range(0, len(arr), N):
        chunk = arr[start:start + N]
        n = len(chunk)
        buf = np.zeros(N, dtype="<f4")
        buf[:n] = chunk.astype(np.float32) / 32768.0
        steps.append({
            "token_ids": [],                  # 音频 token 不入词表
            "type": spec.TYPE_USER,
            "modality": "audio",
            "meta": {"samples": buf.tobytes(), "n": n},
        })
    return steps


def audio_to_steps(source: AudioInput) -> List[Step]:
    """load_audio + waveform_to_steps 便捷入口。"""
    return waveform_to_steps(load_audio(source))


def steps_to_waveform(steps: Sequence[Step]) -> np.ndarray:
    """Step 序列 → float32 波形（逐音元取前 n 个采样拼接，剥掉补零）。

    供 roundtrip 测试与调试使用。
    """
    parts = []
    for step in steps:
        meta = step["meta"]
        buf = np.frombuffer(meta["samples"], dtype="<f4")
        parts.append(buf[:meta["n"]])
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------------
# 音画对齐
# ---------------------------------------------------------------------------
def align_audio_to_frames(audio_steps: Sequence[Step],
                          n_frames: int) -> List[List[Step]]:
    """把音元 Step 按 30fps 均分到 n_frames 帧，返回每帧的音元列表。

    均分策略：base = total // n_frames，前 rem = total % n_frames 帧
    各多分 1 个；顺序保持，总数守恒。视频同一时间步内容纳该帧的
    patch + 对应音频 token。
    """
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    total = len(audio_steps)
    base, rem = divmod(total, n_frames)
    out: List[List[Step]] = []
    idx = 0
    for f in range(n_frames):
        k = base + (1 if f < rem else 0)
        out.append(list(audio_steps[idx:idx + k]))
        idx += k
    return out
