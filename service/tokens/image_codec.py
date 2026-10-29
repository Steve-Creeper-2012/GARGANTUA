# ===========================================================================
# GARGANTUA v1 — 视觉输入 codec（图片/视频 → patch 时间步序列）
# ---------------------------------------------------------------------------
# 契约来源：ROOT/spec.py（冻结，禁止修改）。
#   - PATCH_SIZE=32，非 32 倍数宽高向上 pad，边缘补透明像素 (0,0,0,0)。
#   - 视觉输入 token 不入词表：Step["token_ids"] 恒为空，全部信息在 meta。
#   - meta = {"frame_idx": int, "full": bool, "grid": [W, H],
#             "patches": [{"x": gx, "y": gy, "vec": bytes}, ...]}
#     vec = 该 patch 1024 像素 RGBA 行优先平铺的 4096 字节。
#   - delta 规则（帧计数从 1 起）：
#       第 1 帧全量；第 2..30 帧只送有变化 patch（逐像素 RGBA 任一通道
#       不等即变）；第 31 帧（每秒首帧）全量；某帧全部 patch 都变 → 全量
#       并以该帧为新第 1 帧重新计数；输入序列最后一帧强制全量。
#       图片 = 单帧视频（永远 full）。
#
# 2026-10-24 tok-32 迁移：视觉 token 没有离散 token ID，向量存档改用
# .tokenvector 格式（steps_to_tokenvector）。每帧 = 一个时间步，帧内
# 每个 patch = 一个 token 向量；与音频一起投影到同一 d_model 后可混存
# 同一文件（视频帧时间步 = 全部 patch 向量 + 该帧音频向量）。
# ===========================================================================
"""image_codec: 图片/视频帧序列 → Step 序列（视觉输入，token 不入词表）。"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageSequence

import spec
from .tokenvector import dumps as tv_dumps
from .tokenvector import project as tv_project

PATCH = spec.PATCH_SIZE                  # 32
_VEC_BYTES = spec.PATCH_VEC_DIM          # 4096

Step = Dict
ImageInput = Union[str, os.PathLike, Image.Image]


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def load_rgba(img: ImageInput) -> Image.Image:
    """接受 PIL Image 或图片路径，统一转为 RGBA。"""
    if isinstance(img, (str, os.PathLike)):
        img = Image.open(img)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img


def extract_frames(source: Union[str, os.PathLike, Sequence[Image.Image]],
                   max_frames: int | None = None) -> List[Image.Image]:
    """demo 级抽帧辅助（不依赖 ffmpeg）：

    - source 为 List[Image]：原样返回（调用方已按 30fps 抽好）；
    - source 为动图路径（GIF/WebP 等多帧格式，Pillow 可解码）：逐帧取出；
    - source 为目录：按文件名排序读取其中的图片作为帧序列。
    真实视频文件（mp4 等）请在外部用 ffmpeg 抽帧后以帧序列/目录传入。
    """
    frames: List[Image.Image] = []
    if isinstance(source, (list, tuple)):
        frames = [load_rgba(f) for f in source]
    elif isinstance(source, (str, os.PathLike)) and os.path.isdir(source):
        names = sorted(
            n for n in os.listdir(source)
            if os.path.splitext(n)[1].lower()
            in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif")
        )
        frames = [load_rgba(os.path.join(source, n)) for n in names]
    elif isinstance(source, (str, os.PathLike)):
        im = Image.open(source)
        for frame in ImageSequence.Iterator(im):
            frames.append(frame.convert("RGBA"))
    else:
        raise TypeError(f"unsupported frame source: {type(source)!r}")
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


def _pad_to_multiple(img: Image.Image) -> Image.Image:
    """宽高向上 pad 到 PATCH 的倍数，边缘补透明像素。"""
    w, h = img.size
    pw = (w + PATCH - 1) // PATCH * PATCH
    ph = (h + PATCH - 1) // PATCH * PATCH
    if (pw, ph) == (w, h):
        return img
    canvas = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    canvas.paste(img, (0, 0))
    return canvas


def extract_patches(img: ImageInput
                    ) -> Tuple[Tuple[int, int], List[Tuple[int, int, bytes]]]:
    """RGBA 图片 → (grid=(W,H), [(gx, gy, vec_bytes), ...])，行优先排列。"""
    img = _pad_to_multiple(load_rgba(img))
    arr = np.asarray(img, dtype=np.uint8)                    # (ph, pw, 4)
    ph, pw = arr.shape[:2]
    gx_n, gy_n = pw // PATCH, ph // PATCH
    patches: List[Tuple[int, int, bytes]] = []
    for gy in range(gy_n):
        for gx in range(gx_n):
            block = arr[gy * PATCH:(gy + 1) * PATCH,
                        gx * PATCH:(gx + 1) * PATCH, :]
            patches.append((gx, gy, block.tobytes()))        # 行优先 4096 字节
    return (gx_n, gy_n), patches


# ---------------------------------------------------------------------------
# Step 构造
# ---------------------------------------------------------------------------
def _make_step(frame_idx: int, full: bool, grid: Tuple[int, int],
               patches: List[Tuple[int, int, bytes]],
               static: bool = False) -> Step:
    return {
        "token_ids": [],                      # 视觉输入 token 不入词表
        "type": spec.TYPE_USER,
        "modality": "visual",
        "meta": {
            "frame_idx": frame_idx,           # 输入帧序列内索引，从 0 起
            "full": full,
            "grid": [grid[0], grid[1]],
            # static=True：与上一帧逐像素相同（delta 帧本应为空，用首 patch
            # 作非空占位）。解码端按「重放同值 patch」处理，画面不变。
            "static": static,
            "patches": [{"x": gx, "y": gy, "vec": vec}
                        for gx, gy, vec in patches],
        },
    }


def image_to_step(img: ImageInput, frame_idx: int = 0) -> Step:
    """单张图片 → 一个全量 Step（图片 = 单帧视频，永远 full）。"""
    grid, patches = extract_patches(img)
    return _make_step(frame_idx, True, grid, patches)


def frames_to_steps(frames: Sequence[ImageInput]) -> List[Step]:
    """帧序列（30fps 已抽好）→ Step 序列，实现 spec 冻结的 delta 规则。

    帧计数从 1 起：
      - 第 1 帧全量；
      - 第 2..30 帧只送与上一帧相比有变化像素的 patch（逐像素 RGBA
        任一通道不等即变）；
      - 第 31 帧（每秒首帧）全量并重新计数；
      - 某帧全部 patch 都变 → 全量并以该帧为新第 1 帧重新计数；
      - 输入序列最后一帧强制全量；
      - 帧尺寸（patch 网格）发生变化时无法比较，按全量并重新计数；
      - 与上一帧逐像素相同的帧（静态画面）：delta 本应为空，但时间步
        必须非空，改用首 patch 作占位并置 meta["static"]=True。
    """
    steps: List[Step] = []
    prev: Dict[Tuple[int, int], bytes] | None = None
    prev_grid: Tuple[int, int] | None = None
    counter = 0
    n = len(frames)
    for i, frame in enumerate(frames):
        grid, patches = extract_patches(frame)
        cur: Dict[Tuple[int, int], bytes] = {(gx, gy): vec
                                             for gx, gy, vec in patches}
        counter += 1
        full = False
        # 2026-10-27 审计修复：网格比较用完整 (gw, gh)——旧实现只比 patch
        # 数（len(prev) != len(cur)），64×32 → 32×64 都是 2 个 patch 会被
        # 当成同网格继续 delta，随后访问不存在的 prev[(0,1)] 直接 KeyError
        if prev is None or prev_grid != grid:
            full = True                                   # 首帧 / 网格变化
        elif counter > 30:                                # 第 31 帧：每秒首帧
            full = True
        else:
            changed = [k for k in cur if cur[k] != prev[k]]
            if len(changed) == len(cur):                  # 全变 → 重置计数
                full = True
        if i == n - 1:                                    # 末帧强制全量
            full = True
        static = False
        if full:
            counter = 1
            send = patches
        else:
            send = [(gx, gy, cur[(gx, gy)]) for gx, gy in changed]
            if not send:
                # 2026-10-27 审计修复（M3）：静态视频（三帧同画面即可触发）
                # 的 delta 帧没有任何变化 patch。时间步是原子单位、必须非空
                # （tokenvector.dumps 直接拒绝空步），而跳帧会丢失时序，
                # 所以用首 patch 作「无变化占位」：值与上一帧逐像素相同，
                # 解码端重放后画面不变，只是给这一步一个 token。
                send = [patches[0]] if patches else []
                static = True
        steps.append(_make_step(i, full, grid, send, static=static))
        prev = cur
        prev_grid = grid
    return steps


# ---------------------------------------------------------------------------
# 存档 / 调试
# ---------------------------------------------------------------------------
def _quantize(mean_val: float, levels: int) -> int:
    """0..255 均值 → [0, levels-1] 整数档。"""
    q = int(round(mean_val / 255.0 * (levels - 1)))
    return max(0, min(levels - 1, q))


def step_to_archive(step: Step) -> Dict:
    """Step → 存档 json 形式：{"visual": [{"x","y","R","G","B","A"}, ...]}

    对 patch 内 1024 像素逐通道取均值后量化：
    R/G/B = spec.VISUAL_R_QUANT(1024) 档，A = spec.VISUAL_A_QUANT(1000) 档。
    """
    out = []
    rq, aq = spec.VISUAL_R_QUANT, spec.VISUAL_A_QUANT
    for p in step["meta"]["patches"]:
        arr = np.frombuffer(p["vec"], dtype=np.uint8).reshape(-1, 4)
        means = arr.mean(axis=0)
        out.append({
            "x": p["x"], "y": p["y"],
            "R": _quantize(means[0], rq),
            "G": _quantize(means[1], rq),
            "B": _quantize(means[2], rq),
            "A": _quantize(means[3], aq),
        })
    return {"visual": out}


# ---------------------------------------------------------------------------
# .tokenvector 导出（2026-10-24：视觉向量存档格式迁移）
# ---------------------------------------------------------------------------
def steps_to_tokenvector(steps: Sequence[Step], weight=None,
                         precision: str = "f32") -> str:
    """视觉 Step 序列 → .tokenvector 文本。

    每帧 = 一个时间步，帧内每个 patch = 一个 token 向量（行优先）。

    weight: 可选 [D, 4096] 投影矩阵（通常取自 model.visual_proj.weight）。
      给定时输出投影后的 D 维 patch 向量；坐标加性位置嵌入不含在内
      （需要时由调用方叠加，或直接使用模型侧 assemble_embeddings）。
      缺省输出原始 4096 维 patch 向量（uint8/255 归一化，dim=4096）。
      视频混合帧：把音频步（audio_codec.steps_to_tokenvector 同参）得到
      的向量按帧拼接进同一时间步，即可产出单文件混合 .tokenvector。
    """
    timesteps: List[List[np.ndarray]] = []
    for step in steps:
        vecs: List[np.ndarray] = []
        for p in step["meta"]["patches"]:
            arr = np.frombuffer(p["vec"], dtype=np.uint8).astype(np.float32)
            vec = arr / 255.0
            if weight is not None:
                vec = tv_project(vec.reshape(1, -1), weight)[0]
            vecs.append(vec)
        timesteps.append(vecs)
    if timesteps and timesteps[0]:
        dim = int(timesteps[0][0].size)
    elif weight is not None:
        dim = int(np.asarray(weight).shape[0])
    else:
        dim = spec.PATCH_VEC_DIM
    return tv_dumps(timesteps, dim=dim, precision=precision)


def patches_to_rgba_array(step: Step,
                          prev_frame: np.ndarray | None = None
                          ) -> np.ndarray:
    """把 Step 的 patch 重建为 (H*32, W*32, 4) uint8 数组（含 pad 区域）。

    full 帧：直接由 patches 拼出；delta 帧：未包含的 patch 从 prev_frame
    （上一帧重建结果，同形状）拷贝。prev_frame 缺失的 patch 补透明。
    仅供测试与前端调试，解码侧不需要重建图像。
    """
    gw, gh = step["meta"]["grid"]
    out = np.zeros((gh * PATCH, gw * PATCH, 4), dtype=np.uint8)
    if prev_frame is not None:
        out[:, :, :] = prev_frame[:gh * PATCH, :gw * PATCH, :]
    for p in step["meta"]["patches"]:
        block = np.frombuffer(p["vec"], dtype=np.uint8).reshape(PATCH, PATCH, 4)
        y0, x0 = p["y"] * PATCH, p["x"] * PATCH
        out[y0:y0 + PATCH, x0:x0 + PATCH, :] = block
    return out
