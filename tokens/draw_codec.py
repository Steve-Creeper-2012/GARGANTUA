# ===========================================================================
# GARGANTUA v1 — 绘图输出 codec（图元序列 ↔ 绘图 token + 纯 Python 渲染）
# ---------------------------------------------------------------------------
# 契约来源：ROOT/spec.py（冻结，禁止修改）。
#   - shape ∈ spec.DRAW_SHAPES = background / rectangle / ellipse / triangle
#   - 每个图元 9 个 token：x→y→shape→width→length→rot→R→G→B，
#     ID 用 spec 的 DRAW_*_BEGIN 区间换算；width/length 越界 clamp 并记录；
#     rot 0–359。
#   - 颜色 1024 档 → 8bit（>>2）；rot 单位度，绕图元中心；
#     逐个图元覆盖绘制；画布默认 1920×1080。
#   - PNG 写出：zlib + struct 手写 encoder（filter 0），不依赖 Pillow 画图。
# ===========================================================================
"""draw_codec: 图元序列 ↔ 绘图 token；纯 Python PNG 渲染器与 SVG 导出。"""

from __future__ import annotations

import logging
import math
import struct
import zlib
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import spec

log = logging.getLogger(__name__)

SHAPES = spec.DRAW_SHAPES
_TOKENS_PER = spec.DRAW_TOKENS_PER_PRIMITIVE          # 9


@dataclass
class Primitive:
    x: int
    y: int
    shape: str
    width: int
    length: int
    rot: int
    r: int
    g: int
    b: int


def _clamp(v: int, lo: int, hi: int, name: str) -> int:
    v = int(v)
    if v < lo or v > hi:
        log.warning("draw_codec: %s=%r 越界，clamp 到 [%d, %d]",
                    name, v, lo, hi)
        v = max(lo, min(hi, v))
    return v


def clamp_primitive(p: Primitive) -> Primitive:
    """各字段 clamp 到 spec 词表区间；越界记录 warning。"""
    shape = p.shape if p.shape in SHAPES else SHAPES[0]
    if p.shape not in SHAPES:
        log.warning("draw_codec: 未知 shape=%r，回退为 %r", p.shape, shape)
    return Primitive(
        x=_clamp(p.x, 0, spec.DRAW_X_SIZE - 1, "x"),
        y=_clamp(p.y, 0, spec.DRAW_Y_SIZE - 1, "y"),
        shape=shape,
        width=_clamp(p.width, 0, spec.DRAW_W_SIZE - 1, "width"),
        length=_clamp(p.length, 0, spec.DRAW_L_SIZE - 1, "length"),
        rot=_clamp(p.rot, 0, spec.DRAW_ROT_SIZE - 1, "rot"),
        r=_clamp(p.r, 0, spec.DRAW_R_SIZE - 1, "r"),
        g=_clamp(p.g, 0, spec.DRAW_G_SIZE - 1, "g"),
        b=_clamp(p.b, 0, spec.DRAW_B_SIZE - 1, "b"),
    )


# ---------------------------------------------------------------------------
# 图元 ↔ token（每图元 9 个，顺序 x y shape width length rot R G B）
# ---------------------------------------------------------------------------
def primitives_to_tokens(prims: Sequence[Primitive]) -> List[int]:
    ids: List[int] = []
    for prim in prims:
        p = clamp_primitive(prim)
        ids += [
            spec.DRAW_X_BEGIN + p.x,
            spec.DRAW_Y_BEGIN + p.y,
            spec.DRAW_SHAPE_BEGIN + SHAPES.index(p.shape),
            spec.DRAW_W_BEGIN + p.width,
            spec.DRAW_L_BEGIN + p.length,
            spec.DRAW_ROT_BEGIN + p.rot,
            spec.DRAW_R_BEGIN + p.r,
            spec.DRAW_G_BEGIN + p.g,
            spec.DRAW_B_BEGIN + p.b,
        ]
    return ids


def tokens_to_primitives(ids: Sequence[int]) -> List[Primitive]:
    if len(ids) % _TOKENS_PER != 0:
        log.warning("draw_codec: token 数 %d 不是 %d 的倍数，截断尾部",
                    len(ids), _TOKENS_PER)
    prims: List[Primitive] = []
    for i in range(0, len(ids) - _TOKENS_PER + 1, _TOKENS_PER):
        t = ids[i:i + _TOKENS_PER]
        shape_idx = _clamp(t[2] - spec.DRAW_SHAPE_BEGIN, 0,
                           spec.DRAW_SHAPE_SIZE - 1, "shape")
        if shape_idx >= len(SHAPES):
            log.warning("draw_codec: shape 槽位 %d 为预留，回退 %r",
                        shape_idx, SHAPES[0])
            shape_idx = 0
        prims.append(clamp_primitive(Primitive(
            x=t[0] - spec.DRAW_X_BEGIN,
            y=t[1] - spec.DRAW_Y_BEGIN,
            shape=SHAPES[shape_idx],
            width=t[3] - spec.DRAW_W_BEGIN,
            length=t[4] - spec.DRAW_L_BEGIN,
            rot=t[5] - spec.DRAW_ROT_BEGIN,
            r=t[6] - spec.DRAW_R_BEGIN,
            g=t[7] - spec.DRAW_G_BEGIN,
            b=t[8] - spec.DRAW_B_BEGIN,
        )))
    return prims


# ---------------------------------------------------------------------------
# 纯 Python RGBA 渲染（逐图元覆盖绘制，不依赖 Pillow 画图）
# ---------------------------------------------------------------------------
def _rgba8(p: Primitive) -> bytes:
    return bytes((p.r >> 2, p.g >> 2, p.b >> 2, 255))


def _fill_span(buf: bytearray, w: int, h: int,
               x0: int, y: int, x1: int, px: bytes) -> None:
    """第 y 行 [x0, x1) 填充（自动裁剪画布）。"""
    if y < 0 or y >= h:
        return
    x0, x1 = max(0, x0), min(w, x1)
    if x0 >= x1:
        return
    row = y * w * 4
    buf[row + x0 * 4:row + x1 * 4] = px * (x1 - x0)


def render_rgba(prims: Sequence[Primitive],
                w: int = spec.CANVAS_W,
                h: int = spec.CANVAS_H) -> bytearray:
    """图元序列 → RGBA bytearray（w*h*4），初始透明，逐个覆盖绘制。

    - background：整幅填充（忽略几何参数）；
    - rectangle：轴对齐矩形，(x, y) 为中心，width/length 为宽高；
    - ellipse：轴对齐椭圆（含圆），(x, y) 为中心，width/length 为轴径；
    - triangle：等腰三角形，(x, y) 为外接框中心，width/length 为外接框，
      顶点朝 -y 方向，rot（度）绕图元中心旋转。
    """
    buf = bytearray(w * h * 4)
    for prim in prims:
        p = clamp_primitive(prim)
        px = _rgba8(p)
        if p.shape == "background":
            buf[:] = px * (w * h)
        elif p.shape in ("rectangle", "ellipse", "triangle"):
            hw, hl = p.width / 2.0, p.length / 2.0
            if hw <= 0 or hl <= 0:
                continue
            rad = math.radians(p.rot)
            cos_t, sin_t = math.cos(rad), math.sin(rad)
            # 旋转后外接圆半径（保守包围盒）
            ext = math.hypot(hw, hl)
            x0 = int(math.floor(p.x - ext))
            x1 = int(math.ceil(p.x + ext))
            y0 = int(math.floor(p.y - ext))
            y1 = int(math.ceil(p.y + ext))
            inv_a2 = 1.0 / (hw * hw) if hw > 0 else 1e9
            inv_b2 = 1.0 / (hl * hl) if hl > 0 else 1e9
            for y in range(max(0, y0), min(h - 1, y1) + 1):
                for x in range(max(0, x0), min(w - 1, x1) + 1):
                    dx, dy = x - p.x, y - p.y
                    # 逆旋转到图元局部坐标
                    u = dx * cos_t + dy * sin_t
                    v = -dx * sin_t + dy * cos_t
                    inside = False
                    if p.shape == "rectangle":
                        inside = abs(u) <= hw and abs(v) <= hl
                    elif p.shape == "ellipse":
                        inside = u * u * inv_a2 + v * v * inv_b2 <= 1.0
                    else:  # triangle：顶点朝 -y
                        if -hl <= v <= hl:
                            inside = abs(u) <= (v + hl) * hw / p.length
                    if inside:
                        off = (y * w + x) * 4
                        buf[off:off + 4] = px
    return buf


# ---------------------------------------------------------------------------
# PNG 写出（zlib + struct 手写 encoder，filter 0）
# ---------------------------------------------------------------------------
def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def encode_png(buf: bytes, w: int, h: int) -> bytes:
    """RGBA 字节流 → PNG 文件字节（8bit、color type 6、filter 0）。"""
    stride = w * 4
    raw = bytearray((stride + 1) * h)
    for y in range(h):
        row = y * (stride + 1)
        raw[row] = 0                                # filter type 0
        raw[row + 1:row + 1 + stride] = buf[y * stride:(y + 1) * stride]
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", zlib.compress(bytes(raw)))
            + _png_chunk(b"IEND", b""))


def primitives_to_png(prims: Sequence[Primitive], path: str,
                      w: int = spec.CANVAS_W,
                      h: int = spec.CANVAS_H) -> None:
    """渲染图元并写出 PNG 文件。"""
    buf = render_rgba(prims, w, h)
    with open(path, "wb") as f:
        f.write(encode_png(bytes(buf), w, h))


# ---------------------------------------------------------------------------
# SVG 导出（前端可视化用）
# ---------------------------------------------------------------------------
def _rgb8(p: Primitive) -> str:
    return f"rgb({p.r >> 2},{p.g >> 2},{p.b >> 2})"


def _triangle_points(p: Primitive) -> str:
    hw, hl = p.width / 2.0, p.length / 2.0
    rad = math.radians(p.rot)
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    pts = []
    for u, v in ((0.0, -hl), (-hw, hl), (hw, hl)):
        x = p.x + u * cos_t - v * sin_t
        y = p.y + u * sin_t + v * cos_t
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def primitives_to_svg(prims: Sequence[Primitive],
                      w: int = spec.CANVAS_W,
                      h: int = spec.CANVAS_H) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
    ]
    for prim in prims:
        p = clamp_primitive(prim)
        fill = _rgb8(p)
        if p.shape == "background":
            parts.append(f'<rect width="{w}" height="{h}" fill="{fill}"/>')
        elif p.shape == "rectangle":
            rot = f' transform="rotate({p.rot} {p.x} {p.y})"' if p.rot else ""
            parts.append(
                f'<rect x="{p.x - p.width / 2:g}" y="{p.y - p.length / 2:g}" '
                f'width="{p.width}" height="{p.length}" fill="{fill}"{rot}/>')
        elif p.shape == "ellipse":
            rot = f' transform="rotate({p.rot} {p.x} {p.y})"' if p.rot else ""
            parts.append(
                f'<ellipse cx="{p.x}" cy="{p.y}" rx="{p.width / 2:g}" '
                f'ry="{p.length / 2:g}" fill="{fill}"{rot}/>')
        elif p.shape == "triangle":
            parts.append(
                f'<polygon points="{_triangle_points(p)}" fill="{fill}"/>')
    parts.append("</svg>")
    return "".join(parts)
