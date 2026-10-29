# ===========================================================================
# GARGANTUA v1 — 绘图输出 codec（图元序列 ↔ 绘图 token + 纯 Python 渲染）
# ---------------------------------------------------------------------------
# 契约来源：ROOT/spec.py（冻结）+ service/prims/fast_shape_render
# （逐字照搬第三方布局，geometrize_256_v1）
# （wonderfulearth，geometrize_256_v1 布局照搬）。
#   - 每个图元 9 个 token，字段序 cx, cy, theta, w, h, shape, r, g, b；
#     各字段独立 bin 区间（token ID = spec.DRAW_BEGIN + offset + bin）。
#   - bin 数学与仓库 visualize.py 的 encode_image_group /
#     decode_tokens_to_render_data 逐行一致：
#       cx/cy = px * 2（512 bins，[0,256) px）
#       theta = deg * 3（270 bins，[0,90) 度；CSV 存弧度，≥90° 交换宽高）
#       w/h   = px * 4（512 bins，[0,128) px）
#       shape = shape_type + 1（256 bins；-1 背景 / 0 矩形 / 1 椭圆）
#       r/g/b = value // 2（各 128 bins）
#   - CSV schema 照搬：image_name,cx,cy,w,h,shape_type,theta,r,g,b
#     （首行背景：cx=cy=w=h=0，shape_type=-1，theta=0，rgb=背景色）。
#   - 渲染：256×256 画布，背景整幅填充，矩形/椭圆绕中心旋转，逐图元覆盖
#     绘制（仓库 visualize 用 alpha 0.5 叠加，此处保持 GARGANTUA 既有
#     不透明覆盖语义，供 PNG/SVG 预览）。
#   - PNG 写出：zlib + struct 手写 encoder（filter 0），不依赖 Pillow 画图。
# ===========================================================================
"""draw_codec: 图元序列 ↔ 绘图 token；CSV 归档；PNG/SVG 渲染。"""

from __future__ import annotations

import csv
import logging
import math
import struct
import zlib
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

import spec

log = logging.getLogger(__name__)

SHAPES = spec.DRAW_SHAPES                      # background/rectangle/ellipse
_TOKENS_PER = spec.DRAW_TOKENS_PER_PRIMITIVE   # 9
_ALPHA = 0.5                                    # 仓库渲染语义（预览不用）


@dataclass
class Primitive:
    """一个图元操作（字段与仓库 CSV 一致）。

    cx/cy/w/h 单位 px（画布 256×256，w/h ∈ [0,128)）；
    shape_type：-1 背景 / 0 矩形 / 1 椭圆；
    theta 单位弧度（[0,π)，token 化时归一到 [0,90) 度）；
    r/g/b 为 8bit。
    """

    cx: float
    cy: float
    w: float
    h: float
    shape_type: int
    theta: float
    r: int
    g: int
    b: int


def shape_name(shape_type: int) -> str:
    """shape_type → 名称（-1/0/1 → background/rectangle/ellipse）。"""
    if shape_type == -1:
        return "background"
    if shape_type == 1:
        return "ellipse"
    return "rectangle"


def shape_type_of(name: str) -> int:
    """名称 → shape_type（未知名按矩形 + warning）。"""
    if name == "background":
        return -1
    if name == "ellipse":
        return 1
    if name != "rectangle":
        log.warning("draw_codec: 未知 shape=%r，按 rectangle 处理", name)
    return 0


def _clamp(v: float, lo: float, hi: float, name: str) -> float:
    v = float(v)
    if v < lo or v > hi:
        log.warning("draw_codec: %s=%r 越界，clamp 到 [%g, %g]", name, v, lo, hi)
        v = max(lo, min(hi, v))
    return v


def clamp_primitive(p: Primitive) -> Primitive:
    """各字段 clamp 到 spec 词表区间；越界记录 warning。"""
    st = p.shape_type
    if st not in (-1, 0, 1):
        log.warning("draw_codec: 未知 shape_type=%r，按矩形处理", st)
        st = 0
    theta = math.fmod(float(p.theta), math.pi)
    if theta < 0:
        theta += math.pi
    return Primitive(
        cx=_clamp(p.cx, 0.0, spec.CANVAS_W - 1.0, "cx"),
        cy=_clamp(p.cy, 0.0, spec.CANVAS_H - 1.0, "cy"),
        w=_clamp(p.w, 0.0, (spec.DRAW_W_BINS - 1) / spec.SIZE_BINS_PER_PIXEL,
                 "w"),
        h=_clamp(p.h, 0.0, (spec.DRAW_H_BINS - 1) / spec.SIZE_BINS_PER_PIXEL,
                 "h"),
        shape_type=st,
        theta=theta,
        r=int(_clamp(p.r, 0, 255, "r")),
        g=int(_clamp(p.g, 0, 255, "g")),
        b=int(_clamp(p.b, 0, 255, "b")),
    )


# ---------------------------------------------------------------------------
# 图元 ↔ token（每图元 9 个，顺序 cx cy theta w h shape r g b；bin 数学照搬仓库）
# ---------------------------------------------------------------------------
def _theta_w_h(p: Primitive) -> Tuple[float, float, float]:
    """theta(rad) → (deg∈[0,90), w, h)；≥90° 交换宽高（与仓库一致）。"""
    deg = math.degrees(p.theta) % 180.0
    w, h = p.w, p.h
    if deg >= 90.0:
        deg -= 90.0
        w, h = h, w
    return deg, w, h


def _bin(value: float, per: float, bins: int, name: str) -> int:
    return int(_clamp(value * per, 0, bins - 1, name))


def primitive_to_tokens(p: Primitive) -> List[int]:
    """单个图元 → 9 个 token ID（照搬仓库 encode_image_group 的 bin 数学）。"""
    p = clamp_primitive(p)
    deg, w, h = _theta_w_h(p)
    b = spec.DRAW_BEGIN
    return [
        b + spec.DRAW_X_OFFSET + _bin(p.cx, spec.XY_BINS_PER_PIXEL,
                                      spec.DRAW_X_BINS, "cx"),
        b + spec.DRAW_Y_OFFSET + _bin(p.cy, spec.XY_BINS_PER_PIXEL,
                                      spec.DRAW_Y_BINS, "cy"),
        b + spec.DRAW_ANGLE_OFFSET + _bin(deg, spec.ANGLE_BINS_PER_DEGREE,
                                          spec.DRAW_ANGLE_BINS, "theta"),
        b + spec.DRAW_W_OFFSET + _bin(w, spec.SIZE_BINS_PER_PIXEL,
                                      spec.DRAW_W_BINS, "w"),
        b + spec.DRAW_H_OFFSET + _bin(h, spec.SIZE_BINS_PER_PIXEL,
                                      spec.DRAW_H_BINS, "h"),
        b + spec.DRAW_SHAPE_OFFSET + int(_clamp(p.shape_type + 1, 0,
                                                spec.DRAW_SHAPE_BINS - 1,
                                                "shape")),
        b + spec.DRAW_R_OFFSET + int(_clamp(p.r // 2, 0,
                                            spec.DRAW_COLOR_BINS - 1, "r")),
        b + spec.DRAW_G_OFFSET + int(_clamp(p.g // 2, 0,
                                            spec.DRAW_COLOR_BINS - 1, "g")),
        b + spec.DRAW_B_OFFSET + int(_clamp(p.b // 2, 0,
                                            spec.DRAW_COLOR_BINS - 1, "b")),
    ]


def primitives_to_tokens(prims: Sequence[Primitive]) -> List[int]:
    ids: List[int] = []
    for prim in prims:
        ids.extend(primitive_to_tokens(prim))
    return ids


def tokens_to_primitives(ids: Sequence[int]) -> List[Primitive]:
    """token ID 流 → 图元序列（照搬仓库 decode_tokens_to_render_data）。

    步首 token 落在 special bin 区（含仓库自带 pad）即停止——
    模型输出特殊 bin 表示序列结束。总数非 9 倍数时截断尾部并 warning。

    2026-10-27 审计修复：逐字段校验 bin 区间——旧实现不检查九个字段的
    取值范围，9 个 0 也能「解出」cx=-48、负宽高、负颜色等荒谬图元；
    畸形 token 现在一律 ValueError（HTTP 400），而不是静默产出坏图。
    """
    if len(ids) % _TOKENS_PER != 0:
        log.warning("draw_codec: token 数 %d 不是 %d 的倍数，截断尾部",
                    len(ids), _TOKENS_PER)
    b = spec.DRAW_BEGIN
    # 九个字段的 (offset, bins) 表——与 primitive_to_tokens 的编码表同源
    _FIELDS = (
        (spec.DRAW_X_OFFSET, spec.DRAW_X_BINS),
        (spec.DRAW_Y_OFFSET, spec.DRAW_Y_BINS),
        (spec.DRAW_ANGLE_OFFSET, spec.DRAW_ANGLE_BINS),
        (spec.DRAW_W_OFFSET, spec.DRAW_W_BINS),
        (spec.DRAW_H_OFFSET, spec.DRAW_H_BINS),
        (spec.DRAW_SHAPE_OFFSET, spec.DRAW_SHAPE_BINS),
        (spec.DRAW_R_OFFSET, spec.DRAW_COLOR_BINS),
        (spec.DRAW_G_OFFSET, spec.DRAW_COLOR_BINS),
        (spec.DRAW_B_OFFSET, spec.DRAW_COLOR_BINS),
    )

    def _bin_of(tok: int, field: int) -> int:
        off, bins = _FIELDS[field]
        v = tok - b
        if not (off <= v < off + bins):
            raise ValueError(
                f"draw token 字段 {field} 越界: id={tok}（期望 "
                f"[{b + off}, {b + off + bins})）")
        return v - off

    prims: List[Primitive] = []
    for i in range(0, len(ids) - _TOKENS_PER + 1, _TOKENS_PER):
        t = ids[i:i + _TOKENS_PER]
        if not all(isinstance(x, int) for x in t):
            raise ValueError(f"draw token 必须是整数: {t!r}")
        if t[0] - b >= spec.DRAW_SPECIAL_OFFSET:
            break
        prims.append(Primitive(
            cx=_bin_of(t[0], 0) / spec.XY_BINS_PER_PIXEL,
            cy=_bin_of(t[1], 1) / spec.XY_BINS_PER_PIXEL,
            w=_bin_of(t[3], 3) / spec.SIZE_BINS_PER_PIXEL,
            h=_bin_of(t[4], 4) / spec.SIZE_BINS_PER_PIXEL,
            shape_type=_bin_of(t[5], 5) - 1,
            theta=math.radians(_bin_of(t[2], 2)
                               / spec.ANGLE_BINS_PER_DEGREE),
            r=_bin_of(t[6], 6) * 2,
            g=_bin_of(t[7], 7) * 2,
            b=_bin_of(t[8], 8) * 2,
        ))
    return prims


# ---------------------------------------------------------------------------
# CSV 归档（schema 照搬仓库：image_name,cx,cy,w,h,shape_type,theta,r,g,b）
# ---------------------------------------------------------------------------
def primitives_to_csv(prims: Sequence[Primitive], path: str,
                      image_name: str = "image") -> None:
    """图元序列 → CSV 文件（首行必须是背景图元）。

    浮点用最短精确回环表示（repr），保证 CSV → token 的 bin 量化不被
    写入精度污染（%.8f 会让 radians(20°) 少一位尾数，角度 bin 差 1）。
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(spec.DRAW_CSV_COLUMNS)
        for p in prims:
            w.writerow([image_name, repr(float(p.cx)), repr(float(p.cy)),
                        repr(float(p.w)), repr(float(p.h)),
                        int(p.shape_type), repr(float(p.theta)),
                        int(p.r), int(p.g), int(p.b)])


def csv_to_primitives(path: str) -> List[Primitive]:
    """CSV 文件 → 图元序列（按 image_name 分组只取第一组，行为与仓库
    iter_complete_image_groups 一致：每组首行背景）。"""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    first = rows[0]["image_name"]
    prims: List[Primitive] = []
    for row in rows:
        if row["image_name"] != first:
            break
        prims.append(Primitive(
            cx=float(row["cx"]), cy=float(row["cy"]),
            w=float(row["w"]), h=float(row["h"]),
            shape_type=int(row["shape_type"]),
            theta=float(row["theta"]),
            r=int(row["r"]), g=int(row["g"]), b=int(row["b"]),
        ))
    return prims


# ---------------------------------------------------------------------------
# 纯 Python RGBA 渲染（逐图元覆盖绘制，不依赖 Pillow 画图）
# ---------------------------------------------------------------------------
def _rgba8(p: Primitive) -> bytes:
    return bytes((p.r, p.g, p.b, 255))


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

    - background（shape_type=-1）：整幅填充（忽略几何参数）；
    - rectangle：(cx, cy) 为中心，w/h 为全宽/全高，theta 绕中心旋转；
    - ellipse：轴对齐椭圆（含圆），(cx, cy) 为中心，w/h 为轴径。

    2026-10-20：逐像素 Python 双重循环改为 numpy 掩码赋值（逐图元一次
    向量化计算，输出逐字节一致）。100 图元从 ~35s 降到 <1s。
    2026-10-24：字段迁移到仓库 schema（theta 弧度、w/h 全尺寸、无 triangle）。
    """
    buf = bytearray(w * h * 4)
    for prim in prims:
        p = clamp_primitive(prim)
        px = _rgba8(p)
        if p.shape_type == -1:
            buf[:] = px * (w * h)
            continue
        if p.shape_type not in (0, 1):
            continue
        hw, hl = p.w / 2.0, p.h / 2.0
        if hw <= 0 or hl <= 0:
            continue
        rad = p.theta
        cos_t, sin_t = math.cos(rad), math.sin(rad)
        # 旋转后外接圆半径（保守包围盒）
        ext = math.hypot(hw, hl)
        x0 = max(0, int(math.floor(p.cx - ext)))
        x1 = min(w - 1, int(math.ceil(p.cx + ext)))
        y0 = max(0, int(math.floor(p.cy - ext)))
        y1 = min(h - 1, int(math.ceil(p.cy + ext)))
        if x0 > x1 or y0 > y1:
            continue
        xs = np.arange(x0, x1 + 1, dtype=np.float64)
        ys = np.arange(y0, y1 + 1, dtype=np.float64)
        dx = xs[None, :] - float(p.cx)
        dy = ys[:, None] - float(p.cy)
        # 逆旋转到图元局部坐标
        u = dx * cos_t + dy * sin_t
        v = -dx * sin_t + dy * cos_t
        if p.shape_type == 0:
            inside = (np.abs(u) <= hw) & (np.abs(v) <= hl)
        else:
            inv_a2 = 1.0 / (hw * hw)
            inv_b2 = 1.0 / (hl * hl)
            inside = u * u * inv_a2 + v * v * inv_b2 <= 1.0
        region = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        region[y0:y1 + 1, x0:x1 + 1][inside] = np.frombuffer(px, dtype=np.uint8)
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
    return f"rgb({p.r},{p.g},{p.b})"


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
        if p.shape_type == -1:
            parts.append(f'<rect width="{w}" height="{h}" fill="{fill}"/>')
            continue
        deg = math.degrees(p.theta)
        rot = (f' transform="rotate({deg:g} {p.cx:g} {p.cy:g})"'
               if abs(deg) > 1e-9 else "")
        if p.shape_type == 0:
            parts.append(
                f'<rect x="{p.cx - p.w / 2:g}" y="{p.cy - p.h / 2:g}" '
                f'width="{p.w:g}" height="{p.h:g}" fill="{fill}"{rot}/>')
        else:
            parts.append(
                f'<ellipse cx="{p.cx:g}" cy="{p.cy:g}" rx="{p.w / 2:g}" '
                f'ry="{p.h / 2:g}" fill="{fill}"{rot}/>')
    parts.append("</svg>")
    return "".join(parts)
