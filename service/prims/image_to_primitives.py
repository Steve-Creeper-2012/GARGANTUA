# GARGANTUA v1 — 图像 → 图元序列转换器（仅输出 / 可视化）
# ---------------------------------------------------------------------------
# 所属服务：service/prims/（primitive 管线）。同目录的 fast_shape_render/
# 是逐字照搬的第三方 Rust/wgpu 转换器（wonderfulearth/primitive-operation-
# painter，未改动一行）；upstream/ 是该仓库的完整留档。
#
# 两条路径，输出统一的仓库 schema 图元序列（draw_codec.Primitive）：
#
#   GPU 路径（照搬第三方代码，未改动）：
#     service/prims/fast_shape_render —— Rust/wgpu 实现的 image-to-sequence
#     converter（275 步搜索，输出 CSV）。由 convert_with_renderer() 以
#     子进程驱动；未构建时给出构建指引（sh service/prims/setup.sh，
#     Rust 工具链装在 environment/rust/，与系统隔离）。
#
#   CPU 兜底路径（本模块原有拟合器，输出改到仓库 schema）：
#     与 fast_shape_render 算法对齐的纯 numpy 贪心拟合——海选候选 + 可选
#     爬山，fast 模式 20 候选/步无爬山（几秒），slow 模式 50 候选 + 短爬山。
#
# 与 fast_shape_render 对齐的细节：
#   - 画布 256×256；背景色 = 全图平均 RGB（整除）；Alpha = 0.5
#   - SDF 覆盖率：smoothstep(-0.5, 0.5, d) 三次 Hermite
#   - 最优颜色 = Σ((target − canvas)/α + canvas)·cov / Σcov，先 clamp 再截断
#   - 误差 = 颜色代入后的二次式 SSD；越界惩罚 1.8
#   - 采纳条件：error < 0；无改善记 dummy 步
#   - 早停：环形误差历史窗口，改善 ≤ 0.0005 停止
#
# 输出图元序列（仓库 CSV schema）：
#   Primitive(cx, cy, w, h, shape_type, theta, r, g, b)
#   画布 256×256；theta 弧度 [0,π)；shape_type -1 背景 / 0 矩形 / 1 椭圆；
#   颜色 8bit。首项恒为背景图元。
# ===========================================================================
"""image_to_primitives: 图像 → 图元序列（GPU 转换器驱动 + CPU 兜底拟合）。"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

import spec
from tokens.draw_codec import Primitive

# ---------------------------------------------------------------------------
# GPU 转换器（vendored Rust/wgpu，照搬第三方代码）定位与驱动
# ---------------------------------------------------------------------------
# 转换器与驱动同目录（service/prims/），不再依赖项目根路径——整个目录
# 拷到哪儿都能跑。
_RENDERER_BIN = (Path(__file__).resolve().parent / "fast_shape_render"
                 / "target" / "release" / "gpu_anime_vectorizer")
BUILD_HINT = (
    "GPU 转换器未构建。构建一次即可长期使用：\n"
    "  sh service/prims/setup.sh\n"
    "（Rust 工具链自动装进 environment/rust/，与系统隔离；macOS 走 Metal）"
)


def renderer_binary() -> Optional[Path]:
    """定位 GPU 转换器可执行文件；未找到返回 None。

    优先环境变量 GARGANTUA_SHAPE_RENDERER，否则找同目录 vendored 构建产物。
    """
    env = os.environ.get("GARGANTUA_SHAPE_RENDERER")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    return _RENDERER_BIN if _RENDERER_BIN.is_file() else None


def convert_with_renderer(img_path: str, out_dir: str,
                          max_steps: Optional[int] = None,
                          num_versions: int = 1) -> str:
    """驱动 GPU 转换器：图片目录 → CSV 文件路径（照搬第三方代码，未改动）。

    环境变量与仓库一致：SHAPE_RENDERER_INPUT_DIR（必填）、
    SHAPE_RENDERER_OUTPUT_DIR、SHAPE_RENDERER_NUM_VERSIONS、
    SHAPE_RENDERER_MAX_OUTPUT_STEPS。
    """
    binary = renderer_binary()
    if binary is None:
        raise RuntimeError(BUILD_HINT)
    src_dir = tempfile.mkdtemp(prefix="garg_render_in_")
    dst_dir = tempfile.mkdtemp(prefix="garg_render_out_")
    try:
        shutil.copy(img_path, os.path.join(src_dir,
                                           os.path.basename(img_path)))
        env = dict(os.environ)
        env["SHAPE_RENDERER_INPUT_DIR"] = src_dir
        env["SHAPE_RENDERER_OUTPUT_DIR"] = dst_dir
        env["SHAPE_RENDERER_NUM_VERSIONS"] = str(num_versions)
        if max_steps is not None:
            env["SHAPE_RENDERER_MAX_OUTPUT_STEPS"] = str(max_steps)
        r = subprocess.run([str(binary)], env=env, capture_output=True,
                           text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(
                f"GPU 转换器失败（退出码 {r.returncode}）: "
                + (r.stderr or r.stdout or "")[-800:])
        csvs = sorted(Path(dst_dir).rglob("*.csv"))
        if not csvs:
            raise RuntimeError(f"GPU 转换器未产出 CSV: {dst_dir}")
        out = os.path.join(out_dir, os.path.basename(csvs[0]))
        shutil.copy(csvs[0], out)
        return out
    finally:
        shutil.rmtree(src_dir, ignore_errors=True)
        shutil.rmtree(dst_dir, ignore_errors=True)

# ---------------------------------------------------------------------------
# 默认常量
# ---------------------------------------------------------------------------
W = 256
H = 256
ALPHA = 0.5
PENALTY_FACTOR = 1.8

# fast 模式：纯随机海选，无爬山，速度优先
DEFAULT_MAX_STEPS = 100
DEFAULT_CANDIDATES = 20
DEFAULT_SURVIVORS = 1
DEFAULT_MAX_MUTATIONS = 0          # 0 表示不爬山
DEFAULT_MUTATION_STAGNATION = 0

# slow 模式：50 候选 + 短爬山，质量优先
SLOW_MAX_STEPS = 100
SLOW_CANDIDATES = 50
SLOW_SURVIVORS = 3
SLOW_MAX_MUTATIONS = 100
SLOW_MUTATION_STAGNATION = 30

POSITION_MUTATION_RANGE = 28.0
WIDTH_MUTATION_RANGE = 28.0
HEIGHT_MUTATION_RANGE = 28.0
ANGLE_MUTATION_RANGE_DEG = 16.0
MIN_HALF_SIZE = 1.0
MAX_HALF_SIZE = 128.0
HISTORY_WINDOW = 50           # 早停比较窗口（步）：必须 < max_steps 才可能触发
                              # （2026-10-20：原 1000 > max_steps=100，早停是死代码）
EARLY_STOP_THRESHOLD = 0.0005

_EPS = 1e-6


# ---------------------------------------------------------------------------
# RNG（与 WGSL 的 pcg_hash 一致）
# ---------------------------------------------------------------------------
class _Rng:
    __slots__ = ("state",)

    def __init__(self, seed: int = 0):
        self.state = int(seed) & 0xFFFFFFFF

    def _pcg(self) -> int:
        s = self.state
        s = (s * 747796405 + 2891336453) & 0xFFFFFFFF
        self.state = s
        word = (((s >> ((s >> 28) + 4)) ^ s) * 277803737) & 0xFFFFFFFF
        return (word >> 22) ^ word

    def rand_f32(self) -> float:
        return float(self._pcg()) / 4294967296.0

    def rand_range(self, lo: float, hi: float) -> float:
        return lo + self.rand_f32() * (hi - lo)

    def rand_int(self, lo: int, hi: int) -> int:
        return lo + self._pcg() % (hi - lo + 1)


# ---------------------------------------------------------------------------
# Shape 结构（与 GpuShape 对齐）
# ---------------------------------------------------------------------------
class _Shape:
    __slots__ = ("cx", "cy", "hw", "hh", "theta", "shape_type", "r", "g", "b")

    def __init__(self, cx=0.0, cy=0.0, hw=1.0, hh=1.0,
                 theta=0.0, shape_type=0):
        self.cx = float(cx)
        self.cy = float(cy)
        self.hw = float(hw)
        self.hh = float(hh)
        self.theta = float(theta)
        self.shape_type = int(shape_type)
        self.r = 0
        self.g = 0
        self.b = 0

    def copy(self) -> "_Shape":
        s = _Shape(self.cx, self.cy, self.hw, self.hh,
                   self.theta, self.shape_type)
        s.r, s.g, s.b = self.r, self.g, self.b
        return s

    def clone_mutated(self, rng: _Rng) -> "_Shape":
        s = self.copy()
        choice = rng.rand_int(0, 4)
        if choice == 0:          # position
            s.cx = float(np.clip(s.cx + rng.rand_range(-POSITION_MUTATION_RANGE,
                                                        POSITION_MUTATION_RANGE),
                                 0.0, W - 1))
            s.cy = float(np.clip(s.cy + rng.rand_range(-POSITION_MUTATION_RANGE,
                                                        POSITION_MUTATION_RANGE),
                                 0.0, H - 1))
        elif choice == 1:        # width
            s.hw = float(np.clip(s.hw + rng.rand_range(-WIDTH_MUTATION_RANGE,
                                                        WIDTH_MUTATION_RANGE),
                                 MIN_HALF_SIZE, MAX_HALF_SIZE))
        elif choice == 2:        # height
            s.hh = float(np.clip(s.hh + rng.rand_range(-HEIGHT_MUTATION_RANGE,
                                                        HEIGHT_MUTATION_RANGE),
                                 MIN_HALF_SIZE, MAX_HALF_SIZE))
        elif choice == 3:        # angle
            pi = math.pi
            mut = s.theta + rng.rand_range(
                -math.radians(ANGLE_MUTATION_RANGE_DEG),
                math.radians(ANGLE_MUTATION_RANGE_DEG))
            s.theta = mut - pi * math.floor(mut / pi)
        else:                    # shape_type flip
            s.shape_type = 1 - s.shape_type
        return s

    @staticmethod
    def random(rng: _Rng) -> "_Shape":
        return _Shape(
            cx=rng.rand_range(0.0, W - 1),
            cy=rng.rand_range(0.0, H - 1),
            hw=rng.rand_range(MIN_HALF_SIZE, MAX_HALF_SIZE),
            hh=rng.rand_range(MIN_HALF_SIZE, MAX_HALF_SIZE),
            theta=rng.rand_range(0.0, math.pi),
            shape_type=rng.rand_int(0, 1),
        )


# ---------------------------------------------------------------------------
# 预计算的全局像素网格（避免每步重复 np.mgrid）
# ---------------------------------------------------------------------------
_YY, _XX = np.mgrid[0:H, 0:W]
_YY = _YY.astype(np.float32)
_XX = _XX.astype(np.float32)


def _bbox(shape: _Shape) -> Tuple[int, int, int, int]:
    """返回 (min_x, max_x, min_y, max_y)，已裁剪到画布内。"""
    cos_t = math.cos(shape.theta)
    sin_t = math.sin(shape.theta)
    abs_cos = abs(cos_t)
    abs_sin = abs(sin_t)
    if shape.shape_type == 0:
        ext_x = abs_cos * shape.hw + abs_sin * shape.hh
        ext_y = abs_sin * shape.hw + abs_cos * shape.hh
    else:
        hw_cos = shape.hw * cos_t
        hh_sin = shape.hh * sin_t
        hw_sin = shape.hw * sin_t
        hh_cos = shape.hh * cos_t
        ext_x = math.sqrt(hw_cos * hw_cos + hh_sin * hh_sin)
        ext_y = math.sqrt(hw_sin * hw_sin + hh_cos * hh_cos)
    aa = 1.0
    min_x = max(0, int(math.floor(shape.cx - ext_x - aa)))
    max_x = min(W - 1, int(math.ceil(shape.cx + ext_x + aa)))
    min_y = max(0, int(math.floor(shape.cy - ext_y - aa)))
    max_y = min(H - 1, int(math.ceil(shape.cy + ext_y + aa)))
    return min_x, max_x, min_y, max_y


def _get_coverage(shape_type: int, hw: float, hh: float,
                  dx: np.ndarray, dy: np.ndarray,
                  cos_t: float, sin_t: float) -> np.ndarray:
    """向量化 SDF 覆盖率计算，返回 [0,1] float32。"""
    inv_hw2 = 1.0 / max(hw * hw, 0.0001)
    inv_hh2 = 1.0 / max(hh * hh, 0.0001)
    lx = dx * cos_t + dy * sin_t
    ly = -dx * sin_t + dy * cos_t
    if shape_type == 0:          # rectangle
        d_vec_x = np.abs(lx) - hw
        d_vec_y = np.abs(ly) - hh
        d = np.sqrt(np.maximum(d_vec_x, 0.0) ** 2 +
                    np.maximum(d_vec_y, 0.0) ** 2) + \
            np.minimum(np.maximum(d_vec_x, d_vec_y), 0.0)
    else:                         # ellipse
        f = (lx * lx) * inv_hw2 + (ly * ly) * inv_hh2 - 1.0
        gx = 2.0 * lx * inv_hw2
        gy = 2.0 * ly * inv_hh2
        g_len = np.sqrt(gx * gx + gy * gy)
        with np.errstate(divide="ignore", invalid="ignore"):
            d = np.where(g_len > 1e-5, f / g_len, f)
    t = np.clip(d + 0.5, 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def _eval_shape(shape: _Shape,
                target: np.ndarray,
                canvas: np.ndarray,
                ) -> float:
    """返回 error（越小越好，<0 表示可以采纳）。同时把最优颜色写入 shape.r/g/b。"""
    cos_t = math.cos(shape.theta)
    sin_t = math.sin(shape.theta)
    min_x, max_x, min_y, max_y = _bbox(shape)
    if min_x > max_x or min_y > max_y:
        shape.r = shape.g = shape.b = 0
        return 0.0

    dx = _XX[min_y:max_y + 1, min_x:max_x + 1] - shape.cx
    dy = _YY[min_y:max_y + 1, min_x:max_x + 1] - shape.cy
    cov = _get_coverage(shape.shape_type, shape.hw, shape.hh,
                        dx, dy, cos_t, sin_t)
    mask = cov > 0.0
    if not mask.any():
        shape.r = shape.g = shape.b = 0
        return 0.0

    tr = target[min_y:max_y + 1, min_x:max_x + 1, 0][mask]
    tg = target[min_y:max_y + 1, min_x:max_x + 1, 1][mask]
    tb = target[min_y:max_y + 1, min_x:max_x + 1, 2][mask]
    cr = canvas[min_y:max_y + 1, min_x:max_x + 1, 0][mask]
    cg = canvas[min_y:max_y + 1, min_x:max_x + 1, 1][mask]
    cb = canvas[min_y:max_y + 1, min_x:max_x + 1, 2][mask]
    c = cov[mask]

    k = ALPHA * c
    k2 = k * k
    inv_a = 1.0 / ALPHA
    sum_w = max(float(np.sum(c)), _EPS)

    dr = tr - cr
    dg = tg - cg
    db = tb - cb

    opt_r = float(int(np.clip(np.sum((dr * inv_a + cr) * c) / sum_w, 0.0, 255.0)))
    opt_g = float(int(np.clip(np.sum((dg * inv_a + cg) * c) / sum_w, 0.0, 255.0)))
    opt_b = float(int(np.clip(np.sum((db * inv_a + cb) * c) / sum_w, 0.0, 255.0)))

    sum_k2 = float(np.sum(k2))
    sum_b_r = float(np.sum(-2.0 * k * dr - 2.0 * k2 * cr))
    sum_b_g = float(np.sum(-2.0 * k * dg - 2.0 * k2 * cg))
    sum_b_b = float(np.sum(-2.0 * k * db - 2.0 * k2 * cb))
    sum_c_r = float(np.sum(2.0 * k * dr * cr + k2 * cr * cr))
    sum_c_g = float(np.sum(2.0 * k * dg * cg + k2 * cg * cg))
    sum_c_b = float(np.sum(2.0 * k * db * cb + k2 * cb * cb))

    raw_err = (sum_k2 * (opt_r * opt_r + opt_g * opt_g + opt_b * opt_b)
               + sum_b_r * opt_r + sum_b_g * opt_g + sum_b_b * opt_b
               + sum_c_r + sum_c_g + sum_c_b)

    inside_area = float(np.sum(c))
    if shape.shape_type == 0:
        total_area = 4.0 * shape.hw * shape.hh
    else:
        total_area = math.pi * shape.hw * shape.hh
    oob_ratio = max(0.0, 1.0 - inside_area / max(total_area, _EPS))
    eff_oob = oob_ratio * PENALTY_FACTOR
    final_ratio = np.clip(1.0 - eff_oob, 0.001, 1.0)
    final_err = raw_err * final_ratio if raw_err < 0.0 else raw_err / final_ratio

    shape.r = int(round(opt_r))
    shape.g = int(round(opt_g))
    shape.b = int(round(opt_b))
    return final_err


def _hill_climb(seed_shape: _Shape, target: np.ndarray, canvas: np.ndarray,
                rng: _Rng,
                max_mutations: int,
                stagnation_limit: int) -> Tuple[_Shape, float]:
    """单候选 greedy 爬山。max_mutations=0 时直接返回 seed。"""
    best = seed_shape.copy()
    best_err = _eval_shape(best, target, canvas)
    if max_mutations <= 0:
        return best, best_err
    age = 0
    iters = 0
    while iters < max_mutations:
        if age >= stagnation_limit:
            break
        cand = best.clone_mutated(rng)
        err = _eval_shape(cand, target, canvas)
        if err < best_err:
            best_err = err
            best = cand.copy()
            age = 0
        else:
            age += 1
        iters += 1
    return best, best_err


def _apply_shape(shape: _Shape, canvas: np.ndarray) -> None:
    """将采纳的图元用 alpha 混合涂抹到画布（u8 截断，与原作一致）。"""
    cos_t = math.cos(shape.theta)
    sin_t = math.sin(shape.theta)
    min_x, max_x, min_y, max_y = _bbox(shape)
    if min_x > max_x or min_y > max_y:
        return

    dx = _XX[min_y:max_y + 1, min_x:max_x + 1] - shape.cx
    dy = _YY[min_y:max_y + 1, min_x:max_x + 1] - shape.cy
    cov = _get_coverage(shape.shape_type, shape.hw, shape.hh,
                        dx, dy, cos_t, sin_t)
    mask = cov > 0.0
    if not mask.any():
        return

    eff = (ALPHA * cov[mask]).astype(np.float32)
    sr, sg, sb = float(shape.r), float(shape.g), float(shape.b)
    for ch, sc in enumerate((sr, sg, sb)):
        old = canvas[min_y:max_y + 1, min_x:max_x + 1, ch][mask].astype(np.float32)
        canvas[min_y:max_y + 1, min_x:max_x + 1, ch][mask] = np.floor(
            sc * eff + old * (1.0 - eff))


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def image_to_primitives(img: Image.Image,
                        max_steps: int = DEFAULT_MAX_STEPS,
                        seed: int = 42,
                        quality: str = "fast",
                        progress_cb: Optional[Callable[[int, int], None]] = None,
                        backend: str = "auto",
                        ) -> Tuple[List[Primitive], str]:
    """图像 → 图元序列 + 说明。

    Args:
        img: 输入 PIL 图像（任意尺寸；GPU 路径原图落盘由转换器缩放到
             256×256，CPU 路径本地缩放）。
        max_steps: 最大迭代步数。fast 默认 100，slow 默认 100。
        seed: 随机种子（CPU 路径）。
        quality: "fast"（默认，20 候选/步，无爬山）或 "slow"
                （50 候选/步 + 3 幸存者短爬山）。
        progress_cb: 可选回调 progress_cb(step, max_steps)（CPU 路径）。
        backend: "auto"（默认，GPU 转换器可用则用之，否则 CPU 兜底）、
                 "gpu"（强制 GPU，未构建则报错）、"cpu"（强制 CPU）。

    Returns:
        (primitives, note)；primitives 首项恒为背景图元。
    """
    if backend not in ("auto", "gpu", "cpu"):
        raise ValueError(f"backend 必须是 auto/gpu/cpu: {backend!r}")
    if backend in ("auto", "gpu"):
        binary = renderer_binary()
        if binary is not None:
            return _image_to_primitives_gpu(img, max_steps, binary)
        if backend == "gpu":
            raise RuntimeError(BUILD_HINT)
        note = "GPU 转换器未构建，回退 CPU 拟合（" + BUILD_HINT.splitlines()[1].strip() + "）"
        prims, cpu_note = _image_to_primitives_cpu(
            img, max_steps, seed, quality, progress_cb)
        return prims, note + "；" + cpu_note
    return _image_to_primitives_cpu(img, max_steps, seed, quality,
                                    progress_cb)


def _image_to_primitives_gpu(img: Image.Image, max_steps: int,
                             binary: Path) -> Tuple[List[Primitive], str]:
    """GPU 路径：原图 → 转换器 → CSV → 图元序列（照搬第三方转换器）。"""
    from tokens.draw_codec import csv_to_primitives

    with tempfile.TemporaryDirectory(prefix="garg_i2p_") as d:
        src = os.path.join(d, "input.png")
        img.save(src)
        csv_path = convert_with_renderer(src, d, max_steps=max_steps)
        prims = csv_to_primitives(csv_path)
    if not prims:
        raise RuntimeError("GPU 转换器产出的 CSV 为空")
    note = (f"GPU 转换器（{binary.name}）：{len(prims)} 个图元"
            f"（含背景），max_steps={max_steps}")
    return prims, note


def _image_to_primitives_cpu(img: Image.Image,
                             max_steps: int = DEFAULT_MAX_STEPS,
                             seed: int = 42,
                             quality: str = "fast",
                             progress_cb: Optional[Callable[[int, int], None]] = None,
                             ) -> Tuple[List[Primitive], str]:
    """CPU 兜底：纯 numpy 贪心拟合（算法与 fast_shape_render 对齐）。"""
    if quality == "slow":
        candidate_count = SLOW_CANDIDATES
        survivor_count = SLOW_SURVIVORS
        max_mutations = SLOW_MAX_MUTATIONS
        stagnation = SLOW_MUTATION_STAGNATION
        if max_steps == DEFAULT_MAX_STEPS:
            max_steps = SLOW_MAX_STEPS
    else:
        candidate_count = DEFAULT_CANDIDATES
        survivor_count = DEFAULT_SURVIVORS
        max_mutations = DEFAULT_MAX_MUTATIONS
        stagnation = DEFAULT_MUTATION_STAGNATION

    # 预处理
    img = img.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    raw = np.asarray(img, dtype=np.uint64)
    target = raw.astype(np.float32)

    count = W * H
    bg_r = int(raw[:, :, 0].sum() // count)
    bg_g = int(raw[:, :, 1].sum() // count)
    bg_b = int(raw[:, :, 2].sum() // count)
    canvas = np.broadcast_to(
        np.array([bg_r, bg_g, bg_b], dtype=np.float32), (H, W, 3)).copy()

    initial_ssd = float(np.sum((target - canvas) ** 2))
    if initial_ssd == 0.0:
        initial_ssd = 1.0

    rng = _Rng(seed)
    prims: List[Primitive] = []
    # 背景行（仓库 CSV schema：几何清零、shape_type=-1、theta=0）
    prims.append(Primitive(cx=0.0, cy=0.0, w=0.0, h=0.0, shape_type=-1,
                           theta=0.0, r=bg_r, g=bg_g, b=bg_b))

    err_history = np.zeros(HISTORY_WINDOW, dtype=np.float32)
    accepted = 0
    steps_run = 0
    start_t = time.time()

    for step in range(max_steps):
        steps_run = step + 1
        if progress_cb is not None:
            progress_cb(step, max_steps)

        # ---------- 海选：candidate_count 个随机候选 ----------
        SENTINEL = 99999999.0
        top_errs = [SENTINEL] * survivor_count
        top_shapes: List[Optional[_Shape]] = [None] * survivor_count
        for _ in range(candidate_count):
            cand = _Shape.random(rng)
            err = _eval_shape(cand, target, canvas)
            for j in range(survivor_count):
                if err < top_errs[j]:
                    for k in range(survivor_count - 1, j, -1):
                        top_errs[k] = top_errs[k - 1]
                        top_shapes[k] = top_shapes[k - 1]
                    top_errs[j] = err
                    top_shapes[j] = cand
                    break

        # ---------- 爬山（fast 模式 max_mutations=0，跳过） ----------
        global_best: Optional[Tuple[_Shape, float]] = None
        for s in top_shapes:
            if s is None:
                continue
            local_best, local_err = _hill_climb(
                s, target, canvas, rng, max_mutations, stagnation)
            if global_best is None or local_err < global_best[1]:
                global_best = (local_best, local_err)

        # ---------- 采纳 / dummy ----------
        if global_best is not None and global_best[1] < 0.0:
            _apply_shape(global_best[0], canvas)
            accepted += 1
            prims.append(_to_primitive(global_best[0]))

        # ---------- 早停 ----------
        current_ssd = float(np.sum((target - canvas) ** 2))
        current_err = current_ssd / initial_ssd
        if HISTORY_WINDOW > 0:
            slot = step % HISTORY_WINDOW
            if step >= HISTORY_WINDOW:
                old_err = err_history[slot]
                if old_err - current_err <= EARLY_STOP_THRESHOLD:
                    break
            err_history[slot] = current_err

    elapsed = time.time() - start_t
    final_mse = float(np.mean((target - canvas) ** 2))
    note = (f"quality={quality}：{accepted} 个采纳图元（含背景共 {len(prims)}）"
            f"，{steps_run} 步迭代，残差 MSE={final_mse:.3f}，耗时 {elapsed:.1f}s")
    return prims, note


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _to_primitive(s: _Shape) -> Primitive:
    """将 fast_shape_render 的 GpuShape 转成仓库 schema 的 Primitive。

    画布即 256×256（与仓库一致），无需缩放；半尺寸 → 全尺寸；角度保持
    弧度；颜色直接 8bit（仓库 bin = value // 2）。
    """
    return Primitive(
        cx=float(s.cx),
        cy=float(s.cy),
        w=max(0.0, float(s.hw) * 2.0),
        h=max(0.0, float(s.hh) * 2.0),
        shape_type=int(s.shape_type),
        theta=float(s.theta),
        r=int(s.r), g=int(s.g), b=int(s.b),
    )
