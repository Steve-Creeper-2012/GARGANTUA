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
#   - 背景色 = 全图平均 RGB（整除）；Alpha = 0.5
#   - SDF 覆盖率：smoothstep(-0.5, 0.5, d) 三次 Hermite
#   - 最优颜色 = Σ((target − canvas)/α + canvas)·cov / Σcov，先 clamp 再截断
#   - 误差 = 颜色代入后的二次式 SSD；越界惩罚 1.8
#   - 采纳条件：error < 0；无改善记 dummy 步
#   - 早停：环形误差历史窗口，改善 ≤ 0.0005 停止
#
# 画布（2026-10-31，2026-11-02 改为「按目标比例拟合」）：
#   - GPU 路径由 SHAPE_RENDERER_CANVAS_WIDTH/HEIGHT 指定拟合画布（2026-11-02
#     起转换器支持运行时尺寸），调用方按目标画布的**长宽比**给一个长边 256
#     的小画布，拟合全程比例正确，输出只需等比放大；
#   - CPU 路径按调用方给的 canvas 拟合，1..1920 × 1..1080 任意尺寸
#     （默认 256×256）；
#   - 步数无上限，调用方给多少步就跑多少步（早停仍按误差窗口生效）。
#
# 输出图元序列（仓库 CSV schema）：
#   Primitive(cx, cy, w, h, shape_type, theta, r, g, b)
#   theta 弧度 [0,π)；shape_type -1 背景 / 0 矩形 / 1 椭圆；颜色 8bit。
#   首项恒为背景图元。
# ===========================================================================
"""image_to_primitives: 图像 → 图元序列（GPU 转换器驱动 + CPU 兜底拟合）。"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image

import spec
from paths import project_tmp_dir
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
                          num_versions: int = 1,
                          canvas: Optional[Tuple[int, int]] = None,
                          early_stop: bool = True,
                          stream_path: Optional[str] = None) -> str:
    """驱动 GPU 转换器：图片目录 → CSV 文件路径（照搬第三方代码，未改动）。

    环境变量与仓库一致：SHAPE_RENDERER_INPUT_DIR（必填）、
    SHAPE_RENDERER_OUTPUT_DIR、SHAPE_RENDERER_NUM_VERSIONS；画布尺寸与
    搜索轮数用本项目扩展的三个（2026-11-02 / 11-03 起支持）：
    SHAPE_RENDERER_CANVAS_WIDTH/HEIGHT、SHAPE_RENDERER_MAX_ITERATIONS。

    max_steps 现在映射到**搜索轮数**（MAX_ITERATIONS），不再用
    SHAPE_RENDERER_MAX_OUTPUT_STEPS——那个变量是「必须产出这么多行，
    不够就报错」，图片一旦产不出就会整个失败（2026-11-03 实测）。改成
    「跑多少轮就写多少行」，再由调用方截断，行为更像「步数上限」。
    """
    binary = renderer_binary()
    if binary is None:
        raise RuntimeError(BUILD_HINT)
    src_dir = tempfile.mkdtemp(prefix="garg_render_in_",
                               dir=project_tmp_dir())
    dst_dir = tempfile.mkdtemp(prefix="garg_render_out_",
                               dir=project_tmp_dir())
    try:
        shutil.copy(img_path, os.path.join(src_dir,
                                           os.path.basename(img_path)))
        env = dict(os.environ)
        env["SHAPE_RENDERER_INPUT_DIR"] = src_dir
        env["SHAPE_RENDERER_OUTPUT_DIR"] = dst_dir
        env["SHAPE_RENDERER_NUM_VERSIONS"] = str(num_versions)
        if canvas is not None:
            env["SHAPE_RENDERER_CANVAS_WIDTH"] = str(int(canvas[0]))
            env["SHAPE_RENDERER_CANVAS_HEIGHT"] = str(int(canvas[1]))
        if max_steps is not None:
            env["SHAPE_RENDERER_MAX_ITERATIONS"] = str(int(max_steps))
        # 明确不设 MAX_OUTPUT_STEPS：那是「必须产出 N 行」的硬要求
        env.pop("SHAPE_RENDERER_MAX_OUTPUT_STEPS", None)
        if not early_stop:
            env["SHAPE_RENDERER_EARLY_STOP"] = "0"
        if stream_path:
            env["SHAPE_RENDERER_STREAM_FILE"] = stream_path
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
# 2026-10-31：画布不再钉死 256×256。默认画布取自 spec（256×256，与仓库
# geometrize_256_v1 一致），调用方可用 canvas=(宽, 高) 指定 spec 预设表
# 里的任意一档（最大 1920×1080）。W/H 只作「默认画布」的别名，拟合过程
# 一律用传进来的 canvas，不再读写这两个全局。
W, H = spec.DRAW_DEFAULT_CANVAS
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
# 半尺寸上限：**取画布的一半**，再被 token 可表示范围兜住。
#   2026-10-27：曾把上限钉成 token 上限（127.75/2=63.875），因为当时 w/h
#     只能表达到 128 px；那是为了「预览 == token」的权宜之计，代价是
#     256 画布上一半宽的形状根本长不出来。
#   2026-11-02：w/h 改成 1 token = 1 像素（上限 1920/1080），限制解除，
#     回到上游转换器的做法——半尺寸上限 = 画布短边 / 2（256 画布 → 128，
#     与 fast_shape_render 的 MAX_HALF_SIZE=128 完全一致）。
MAX_HALF_SIZE = (spec.DRAW_W_BINS - 1) / spec.SIZE_BINS_PER_PIXEL / 2.0


def _max_half(w: int, h: int) -> float:
    """当前画布下的半尺寸上限（画布短边的一半，再受 token 上限约束）。"""
    return max(MIN_HALF_SIZE, min(MAX_HALF_SIZE, min(w, h) / 2.0))
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

    def clone_mutated(self, rng: _Rng, w: int = W, h: int = H) -> "_Shape":
        s = self.copy()
        mh = _max_half(w, h)
        choice = rng.rand_int(0, 4)
        if choice == 0:          # position
            s.cx = float(np.clip(s.cx + rng.rand_range(-POSITION_MUTATION_RANGE,
                                                        POSITION_MUTATION_RANGE),
                                 0.0, w - 1))
            s.cy = float(np.clip(s.cy + rng.rand_range(-POSITION_MUTATION_RANGE,
                                                        POSITION_MUTATION_RANGE),
                                 0.0, h - 1))
        elif choice == 1:        # width
            s.hw = float(np.clip(s.hw + rng.rand_range(-WIDTH_MUTATION_RANGE,
                                                        WIDTH_MUTATION_RANGE),
                                 MIN_HALF_SIZE, mh))
        elif choice == 2:        # height
            s.hh = float(np.clip(s.hh + rng.rand_range(-HEIGHT_MUTATION_RANGE,
                                                        HEIGHT_MUTATION_RANGE),
                                 MIN_HALF_SIZE, mh))
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
    def random(rng: _Rng, w: int = W, h: int = H) -> "_Shape":
        mh = _max_half(w, h)
        return _Shape(
            cx=rng.rand_range(0.0, w - 1),
            cy=rng.rand_range(0.0, h - 1),
            hw=rng.rand_range(MIN_HALF_SIZE, mh),
            hh=rng.rand_range(MIN_HALF_SIZE, mh),
            theta=rng.rand_range(0.0, math.pi),
            shape_type=rng.rand_int(0, 1),
        )


# ---------------------------------------------------------------------------
# 预计算的像素网格（按画布尺寸缓存，避免每步重复 np.mgrid）
# ---------------------------------------------------------------------------
# 一张 1920×1080 的网格是 2×8.3MB（float32）。只缓存最近两档：一次转换
# 只会用到一档画布，两档足够覆盖「刚切过尺寸」的连点场景，再多就是白占
# 内存（且这是常驻进程里的模块级缓存）。
_GRID_CACHE: dict = {}
_GRID_CACHE_MAX = 2


def _grid(w: int, h: int):
    """(宽, 高) → (XX, YY) float32 网格（列坐标 / 行坐标）。"""
    key = (int(w), int(h))
    hit = _GRID_CACHE.get(key)
    if hit is not None:
        return hit
    yy, xx = np.mgrid[0:key[1], 0:key[0]]
    g = (xx.astype(np.float32), yy.astype(np.float32))
    if len(_GRID_CACHE) >= _GRID_CACHE_MAX:
        _GRID_CACHE.clear()
    _GRID_CACHE[key] = g
    return g


def _bbox(shape: _Shape, w: int = W, h: int = H) -> Tuple[int, int, int, int]:
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
    max_x = min(w - 1, int(math.ceil(shape.cx + ext_x + aa)))
    min_y = max(0, int(math.floor(shape.cy - ext_y - aa)))
    max_y = min(h - 1, int(math.ceil(shape.cy + ext_y + aa)))
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
                grid=None,
                w: int = W,
                h: int = H,
                ) -> float:
    """返回 error（越小越好，<0 表示可以采纳）。同时把最优颜色写入 shape.r/g/b。

    grid 是 (XX, YY) 像素网格，不传就按 (w, h) 从缓存取。
    """
    if grid is None:
        grid = _grid(w, h)
    xx, yy = grid
    cos_t = math.cos(shape.theta)
    sin_t = math.sin(shape.theta)
    min_x, max_x, min_y, max_y = _bbox(shape, w, h)
    if min_x > max_x or min_y > max_y:
        shape.r = shape.g = shape.b = 0
        return 0.0

    dx = xx[min_y:max_y + 1, min_x:max_x + 1] - shape.cx
    dy = yy[min_y:max_y + 1, min_x:max_x + 1] - shape.cy
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
                stagnation_limit: int,
                grid=None, w: int = W, h: int = H,
                ) -> Tuple[_Shape, float]:
    """单候选 greedy 爬山。max_mutations=0 时直接返回 seed。"""
    best = seed_shape.copy()
    best_err = _eval_shape(best, target, canvas, grid, w, h)
    if max_mutations <= 0:
        return best, best_err
    age = 0
    iters = 0
    while iters < max_mutations:
        if age >= stagnation_limit:
            break
        cand = best.clone_mutated(rng, w, h)
        err = _eval_shape(cand, target, canvas, grid, w, h)
        if err < best_err:
            best_err = err
            best = cand.copy()
            age = 0
        else:
            age += 1
        iters += 1
    return best, best_err


def _apply_shape(shape: _Shape, canvas: np.ndarray, grid=None,
                 w: int = W, h: int = H) -> None:
    """将采纳的图元用 alpha 混合涂抹到画布（u8 截断，与原作一致）。"""
    if grid is None:
        grid = _grid(w, h)
    xx, yy = grid
    cos_t = math.cos(shape.theta)
    sin_t = math.sin(shape.theta)
    min_x, max_x, min_y, max_y = _bbox(shape, w, h)
    if min_x > max_x or min_y > max_y:
        return

    dx = xx[min_y:max_y + 1, min_x:max_x + 1] - shape.cx
    dy = yy[min_y:max_y + 1, min_x:max_x + 1] - shape.cy
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
                        canvas: Optional[Tuple[int, int]] = None,
                        early_stop: bool = True,
                        on_primitive: Optional[Callable[[Primitive, int], None]] = None,
                        ) -> Tuple[List[Primitive], str]:
    """图像 → 图元序列 + 说明。

    Args:
        img: 输入 PIL 图像（任意尺寸；按 canvas 缩放）。
        max_steps: 最大迭代步数（**无上限**，由调用方决定；fast/slow 的
                   默认值都是 100）。
        seed: 随机种子（CPU 路径）。
        quality: "fast"（默认，20 候选/步，无爬山）或 "slow"
                （50 候选/步 + 3 幸存者短爬山）。
        progress_cb: 可选回调 progress_cb(step, max_steps)（CPU 路径）。
        backend: "auto"（默认，GPU 转换器可用则用之，否则 CPU 兜底）、
                 "gpu"（强制 GPU，未构建则报错）、"cpu"（强制 CPU）。
        early_stop: 早停（连续 50 步没有改善就收工）。关掉 = 跑满 max_steps，
                代价是后面那些步大多只能采纳「几乎没有改善」的图元。
        on_primitive: 可选回调 on_primitive(primitive, index)，每采纳一个
                图元立刻回调（含 index=0 的背景）。GPU 路径靠转换器的
                stream 文件边跑边喂，CPU 路径直接在主循环里回调。
        canvas: 画布 (宽, 高)，1..1920 × 1..1080 内任意整数
                （spec.draw_canvas_valid 校验；默认 256×256）。GPU 转换器按
                `gpu_fit_canvas()` 算出的同比例小画布拟合（长边 256），结果
                等比放大到 canvas（见 `_remap_canvas`）。

    Returns:
        (primitives, note)；primitives 首项恒为背景图元。
    """
    if backend not in ("auto", "gpu", "cpu"):
        raise ValueError(f"backend 必须是 auto/gpu/cpu: {backend!r}")
    cw, ch = (spec.DRAW_DEFAULT_CANVAS if canvas is None
              else (int(canvas[0]), int(canvas[1])))
    if not spec.draw_canvas_valid(cw, ch):
        raise ValueError(
            f"画布 {cw}×{ch} 超出契约范围（1..{spec.DRAW_CANVAS_MAX_W} × "
            f"1..{spec.DRAW_CANVAS_MAX_H}）")

    if backend in ("auto", "gpu"):
        binary = renderer_binary()
        if binary is not None:
            return _image_to_primitives_gpu(
                img, max_steps, binary, (cw, ch),
                early_stop=early_stop, on_primitive=on_primitive)
        if backend == "gpu":
            raise RuntimeError(BUILD_HINT)
        note = ("GPU 转换器未构建，回退 CPU 拟合（"
                + BUILD_HINT.splitlines()[1].strip() + "）")
        prims, cpu_note = _image_to_primitives_cpu(
            img, max_steps, seed, quality, progress_cb, (cw, ch),
            early_stop=early_stop, on_primitive=on_primitive)
        return prims, note + "；" + cpu_note
    return _image_to_primitives_cpu(
        img, max_steps, seed, quality, progress_cb, (cw, ch),
        early_stop=early_stop, on_primitive=on_primitive)


# 第三方转换器默认 256×256；2026-11-02 起画布尺寸可由环境变量指定，
# 所以这里按**目标画布的长宽比**给一个长边 256 的小画布去拟合——
# 拟合全程比例正确，输出只要等比放大，不需要任何拉伸。
GPU_FIT_LONG_SIDE = 256


def gpu_fit_canvas(canvas: Tuple[int, int]) -> Tuple[int, int]:
    """目标画布 → GPU 拟合用的同比例小画布（长边 = GPU_FIT_LONG_SIDE）。

    例：1920×1080 → 256×144；640×480 → 256×192；256×256 → 256×256。
    用长边 256 而不是直接用目标尺寸：像素数 ∝ 画布面积，1920×1080 直接
    拟合是 256² 的 31 倍慢（单步要扫更大的包围盒），而形状本身也会等比
    变大，多出来的精度对最终观感没有帮助。
    """
    cw, ch = int(canvas[0]), int(canvas[1])
    if cw <= 0 or ch <= 0:
        return (GPU_FIT_LONG_SIDE, GPU_FIT_LONG_SIDE)
    s = float(GPU_FIT_LONG_SIDE) / float(max(cw, ch))
    gw = max(1, min(GPU_FIT_LONG_SIDE, int(round(cw * s))))
    gh = max(1, min(GPU_FIT_LONG_SIDE, int(round(ch * s))))
    return (gw, gh)


def _remap_canvas(prims: List[Primitive], src: Tuple[int, int],
                  dst: Tuple[int, int]) -> List[Primitive]:
    """把 src 画布下的图元序列映射到 dst 画布（GPU 转换器专用）。

    2026-11-02 起 GPU 是按目标长宽比拟合的（见 gpu_fit_canvas），src 与
    dst 比例相同，所以这里实际上只是**等比放大**（两个轴的系数差 ≤1%，
    来自取整）。仍按轴各自缩放，是为了让「恰好落在画布边缘」的形状不
    因为取整而越界。

    旋转形状在非等比缩放下严格来说会变成平行四边形，没有等价矩形；
    这里按「局部轴向量长度」近似：角度取 atan2(sy·sinθ, sx·cosθ)，
    宽取 |(sx·cosθ, sy·sinθ)|·w、高取 |(−sx·sinθ, sy·cosθ)|·h。
    sx == sy 时退化为恒等映射。
    """
    if tuple(src) == tuple(dst):
        return prims
    sx = float(dst[0]) / float(src[0])
    sy = float(dst[1]) / float(src[1])
    out: List[Primitive] = []
    for p in prims:
        if p.shape_type == -1:
            # 背景：整幅填充，几何参数无意义（与仓库 CSV 首行约定一致）
            out.append(Primitive(cx=0.0, cy=0.0, w=0.0, h=0.0, shape_type=-1,
                                 theta=0.0, r=p.r, g=p.g, b=p.b))
            continue
        cos_t, sin_t = math.cos(p.theta), math.sin(p.theta)
        w_dir = math.hypot(sx * cos_t, sy * sin_t)
        h_dir = math.hypot(-sx * sin_t, sy * cos_t)
        theta = math.atan2(sy * sin_t, sx * cos_t) % math.pi
        # 映射后可能正好等于画布边长（256 满宽 → 1920）：bin 上界是
        # 边长 − 1，多出来的 1px 直接吃掉，免得 token 化时告警。
        out.append(Primitive(
            cx=min(p.cx * sx, float(dst[0] - 1)),
            cy=min(p.cy * sy, float(dst[1] - 1)),
            w=min(p.w * w_dir, float(dst[0] - 1)),
            h=min(p.h * h_dir, float(dst[1] - 1)),
            shape_type=p.shape_type, theta=theta,
            r=p.r, g=p.g, b=p.b))
    return out


class _StreamTail:
    """边跑边读：轮询转换器写的 stream 文件，把新图元喂给回调。

    转换器每 5 步 flush 一次（见 main.rs 的 sync_interval），所以延迟上限
    约「5 步 + 一个轮询周期」。只读完整行——半行留着下次拼，绝不半截解析。
    """

    def __init__(self, path: str, fit, canvas, on_primitive):
        self.path = path
        self.fit = fit
        self.canvas = canvas
        self.cb = on_primitive
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._pos = 0          # 已读字节数（只增量读，不整文件重读）
        self._buf = ""         # 未收尾的半行
        self.count = 0

    def start(self):
        self._th.start()
        return self

    def stop(self):
        self._stop.set()
        self._th.join(timeout=2.0)
        self._pump()          # 收尾：把最后几行也读掉

    def _pump(self):
        try:
            with open(self.path, "rb") as f:
                f.seek(self._pos)
                raw = f.read()
        except OSError:
            return
        if not raw:
            return
        self._pos += len(raw)
        self._buf += raw.decode("utf-8", "replace")
        lines = self._buf.split("\n")
        self._buf = lines.pop()          # 最后一段可能没写完，留到下次
        for line in lines:
            self._emit(line.strip())

    def _emit(self, line: str):
        if not line:
            return
        parts = line.split(",")
        if len(parts) != 10:
            return
        try:
            # ⚠ 流文件里的颜色是 **8 bit**（和 CSV 一样），内部 Primitive 是
            # 10 bit。少了这一步，实时预览会暗 4 倍（2026-11-03 的 bug：
            # 渲染页边跑边画时整张图发黑，跑完换成 result 又正常）。
            from tokens.draw_codec import color_to10
            p = Primitive(cx=float(parts[1]), cy=float(parts[2]),
                          w=float(parts[3]), h=float(parts[4]),
                          shape_type=int(parts[5]), theta=float(parts[6]),
                          r=color_to10(int(parts[7])),
                          g=color_to10(int(parts[8])),
                          b=color_to10(int(parts[9])))
        except (TypeError, ValueError):
            return
        mapped = _remap_canvas([p], self.fit, self.canvas)[0]
        self.count += 1
        self.cb(mapped, self.count)

    def _run(self):
        while not self._stop.wait(0.15):
            self._pump()


def _gpu_background(img: Image.Image, fit) -> Primitive:
    """转换器的背景色 = 输入图 resize_exact 到拟合画布后的平均色。"""
    from tokens.draw_codec import color_to10
    small = img.convert("RGB").resize((int(fit[0]), int(fit[1])),
                                      Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.uint16)
    n = arr.shape[0] * arr.shape[1]
    return Primitive(cx=0.0, cy=0.0, w=0.0, h=0.0, shape_type=-1, theta=0.0,
                     r=color_to10(int(arr[:, :, 0].sum() // n)),
                     g=color_to10(int(arr[:, :, 1].sum() // n)),
                     b=color_to10(int(arr[:, :, 2].sum() // n)))


def _image_to_primitives_gpu(img: Image.Image, max_steps: int,
                             binary: Path,
                             canvas: Tuple[int, int] = (GPU_FIT_LONG_SIDE,
                                                        GPU_FIT_LONG_SIDE),
                             early_stop: bool = True,
                             on_primitive=None,
                             ) -> Tuple[List[Primitive], str]:
    """GPU 路径：原图 → 转换器（按目标长宽比的小画布）→ CSV → 图元序列。"""
    from tokens.draw_codec import csv_to_primitives

    cw, ch = int(canvas[0]), int(canvas[1])
    fit = gpu_fit_canvas((cw, ch))
    asked = max(1, int(max_steps))
    with tempfile.TemporaryDirectory(prefix="garg_i2p_",
                                     dir=project_tmp_dir()) as d:
        src = os.path.join(d, "input.png")
        img.save(src)
        tail = None
        if on_primitive is not None:
            on_primitive(_gpu_background(img, fit), 0)
            tail = _StreamTail(os.path.join(d, "stream.csv"), fit, (cw, ch),
                               on_primitive).start()
        try:
            # max_steps = 搜索轮数（转换器每轮最多采纳 1 个图元，另有早停）
            csv_path = convert_with_renderer(
                src, d, max_steps=asked, canvas=fit,
                early_stop=early_stop,
                stream_path=(os.path.join(d, "stream.csv") if tail else None))
            prims = csv_to_primitives(csv_path)
        finally:
            if tail is not None:
                tail.stop()
    if not prims:
        raise RuntimeError("GPU 转换器产出的 CSV 为空")
    # 转换器每轮最多采纳 1 个 → 行数天然 ≤ asked + 1（背景行恒在第一行）。
    # 截断留 +1：CPU 路径也是「N 步 → N 个图元 + 背景」，两条路径口径一致。
    if len(prims) > asked + 1:
        prims = prims[:asked + 1]
    prims = _remap_canvas(prims, fit, (cw, ch))
    mapped = "" if fit == (cw, ch) else f"，{fit[0]}×{fit[1]} 拟合后放大到 {cw}×{ch}"
    note = (f"GPU 转换器（{binary.name}）：{len(prims)} 个图元"
            f"（含背景），搜索 {asked} 轮{mapped}")
    return prims, note


def _fit_image(img: Image.Image, w: int, h: int) -> Image.Image:
    """把图**按比例**缩放到画布内并居中，空余用图像平均色补齐。

    旧实现是 `img.resize((w, h))`——直接拉伸成画布尺寸，1920×1080 的图
    塞进 256×256 会被压扁成正方形（steve 2026-10-31 指出的问题）。
    这里改成 contain：等比缩放到「刚好装得下」，剩余区域填平均色，
    既不变形，也不让画布边缘出现突兀的纯黑边框（平均色与背景图元同源）。
    """
    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        return img.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    scale = min(w / iw, h / ih)
    nw = max(1, min(w, int(round(iw * scale))))
    nh = max(1, min(h, int(round(ih * scale))))
    scaled = img.convert("RGB").resize((nw, nh), Image.Resampling.LANCZOS)
    if (nw, nh) == (w, h):
        return scaled
    arr = np.asarray(scaled, dtype=np.uint16)
    fill = tuple(int(arr[:, :, c].mean()) for c in range(3))
    out = Image.new("RGB", (w, h), fill)
    out.paste(scaled, ((w - nw) // 2, (h - nh) // 2))
    return out


def _image_to_primitives_cpu(img: Image.Image,
                             max_steps: int = DEFAULT_MAX_STEPS,
                             seed: int = 42,
                             quality: str = "fast",
                             progress_cb: Optional[Callable[[int, int], None]] = None,
                             canvas: Tuple[int, int] = (W, H),
                             early_stop: bool = True,
                             on_primitive=None,
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

    # 预处理（等比缩放 + 居中，绝不拉伸变形）
    w, h = int(canvas[0]), int(canvas[1])
    img = _fit_image(img, w, h)
    raw = np.asarray(img, dtype=np.uint64)
    target = raw.astype(np.float32)
    grid = _grid(w, h)

    count = w * h
    bg_r = int(raw[:, :, 0].sum() // count)
    bg_g = int(raw[:, :, 1].sum() // count)
    bg_b = int(raw[:, :, 2].sum() // count)
    canvas_buf = np.broadcast_to(
        np.array([bg_r, bg_g, bg_b], dtype=np.float32), (h, w, 3)).copy()

    initial_ssd = float(np.sum((target - canvas_buf) ** 2))
    if initial_ssd == 0.0:
        initial_ssd = 1.0

    rng = _Rng(seed)
    prims: List[Primitive] = []
    # 背景行（仓库 CSV schema：几何清零、shape_type=-1、theta=0）
    from tokens.draw_codec import color_to10
    prims.append(Primitive(cx=0.0, cy=0.0, w=0.0, h=0.0, shape_type=-1,
                           theta=0.0, r=color_to10(bg_r), g=color_to10(bg_g),
                           b=color_to10(bg_b)))
    if on_primitive is not None:
        on_primitive(prims[0], 0)

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
            cand = _Shape.random(rng, w, h)
            err = _eval_shape(cand, target, canvas_buf, grid, w, h)
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
                s, target, canvas_buf, rng, max_mutations, stagnation,
                grid, w, h)
            if global_best is None or local_err < global_best[1]:
                global_best = (local_best, local_err)

        # ---------- 采纳 / dummy ----------
        if global_best is not None and global_best[1] < 0.0:
            _apply_shape(global_best[0], canvas_buf, grid, w, h)
            accepted += 1
            prims.append(_to_primitive(global_best[0]))
            if on_primitive is not None:
                on_primitive(prims[-1], accepted)

        # ---------- 早停 ----------
        current_ssd = float(np.sum((target - canvas_buf) ** 2))
        current_err = current_ssd / initial_ssd
        if early_stop and HISTORY_WINDOW > 0:
            slot = step % HISTORY_WINDOW
            if step >= HISTORY_WINDOW:
                old_err = err_history[slot]
                if old_err - current_err <= EARLY_STOP_THRESHOLD:
                    break
            err_history[slot] = current_err

    elapsed = time.time() - start_t
    final_mse = float(np.mean((target - canvas_buf) ** 2))
    note = (f"quality={quality}，画布 {w}×{h}：{accepted} 个采纳图元"
            f"（含背景共 {len(prims)}），{steps_run} 步迭代，"
            f"残差 MSE={final_mse:.3f}，耗时 {elapsed:.1f}s")
    return prims, note


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _to_primitive(s: _Shape) -> Primitive:
    """将 fast_shape_render 的 GpuShape 转成仓库 schema 的 Primitive。

    坐标就是画布像素（CPU 路径按所选画布拟合，无需缩放）；半尺寸 → 全尺寸；
    角度保持弧度。颜色：拟合器内部按 8 bit 算（和原图像一致），落到
    Primitive 时升到 **10 bit**（词表里的图元颜色是 10 bit，见 spec）。
    """
    from tokens.draw_codec import color_to10
    return Primitive(
        cx=float(s.cx),
        cy=float(s.cy),
        w=max(0.0, float(s.hw) * 2.0),
        h=max(0.0, float(s.hh) * 2.0),
        shape_type=int(s.shape_type),
        theta=float(s.theta),
        r=color_to10(s.r), g=color_to10(s.g), b=color_to10(s.b),
    )
