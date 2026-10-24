# ===========================================================================
# GARGANTUA v1 — .tokenvector 向量存档格式（UTF-8 文本）
# ---------------------------------------------------------------------------
# 用途（2026-10-24 冻结）：音频 / 视觉 token 没有离散 token ID（它们经
# 线性投影直接成为向量），所以不走 .tokens 格式，改用本格式存档——
# 直接投影过的向量序列（同一文件内所有 token 向量同维度，例如视频一帧
# 内的 patch token 与音频 token 全部投影到同一个 d_model）。
#
# 格式（UTF-8 文本）：
#   # 注释行（可选，仅允许出现在头部）
#   dim=<向量维度> precision=<f32|f16|bf16>
#   { 0.123 -0.5 ... | 1.0 2.0 ... }{ ... }
#
#   - 头部第一行声明词向量的维度与精度；
#   - 时间步之间用 { } 分隔（UTF-8 字符，不是 tok-32 系统 token）；
#   - 一个时间步内的多个 token 向量用 | 分隔；
#   - {} 必须含有 token，不能为空集（空时间步读写均报错）；
#   - 每个 token 向量 = dim 个数字，仅用空白分隔，没有任何其他符号。
#
# 数值精度与 roundtrip 保证：
#   f32  → float32，%.9g（9 位有效数字保证 roundtrip）
#   f16  → float16，%.5g
#   bf16 → bfloat16（f32 高位截断，RNE），%.4g
#
# 用法：
#   from tokens.tokenvector import dumps, loads
#   text = dumps([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6]]], dim=2)
#   dim, precision, steps = loads(text)
#   CLI： python tokens/tokenvector.py in.tokenvector   （解析并回显统计）
# ===========================================================================
"""tokenvector: .tokenvector 向量存档格式（dim+precision 头，{} | 结构）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

class TokenVectorError(ValueError):
    """tokenvector 格式错误（头部缺失 / 维度不符 / 结构非法 / 数值非法）。"""


# precision → 解析后 dtype（bf16 以 f32 视图保存其值）
PRECISIONS = {
    "f32": np.float32,
    "f16": np.float16,
    "bf16": np.float32,   # bf16 值以 f32 视图保存
}

# ---------------------------------------------------------------------------
# bfloat16 支持（numpy 无原生 bf16：f32 高位截断，round-to-nearest-even）
# ---------------------------------------------------------------------------

def _f32_to_bf16_bits(bits: np.ndarray) -> np.ndarray:
    """f32 位模式（uint32）→ bf16 位模式（uint16，RNE）。"""
    bits = bits.astype(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16))
                                         & np.uint32(1))
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _bf16_to_f32_bits(bits16: np.ndarray) -> np.ndarray:
    """bf16 位模式（uint16）→ f32 位模式（uint32，低位补零）。"""
    return bits16.astype(np.uint32) << np.uint32(16)


def _cast(values: np.ndarray, precision: str) -> np.ndarray:
    """任意数值数组 → 目标精度数组（bf16 以 f32 视图保存其值）。"""
    if precision not in PRECISIONS:
        raise TokenVectorError(f"未知精度: {precision!r}"
                               f"（可选 {sorted(PRECISIONS)}）")
    if precision == "bf16":
        f32 = np.asarray(values, dtype=np.float32)
        bits = _f32_to_bf16_bits(f32.view(np.uint32))
        return _bf16_to_f32_bits(bits).view(np.float32)
    return np.asarray(values, dtype=PRECISIONS[precision])


# ---------------------------------------------------------------------------
# 序列化 / 反序列化
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 数值格式化（最短 roundtrip：从 1 位有效数字起找第一个能原样转回
# 目标精度的十进制表示；f32 典型输出如 "0.1" 而非 "0.100000001"）
# ---------------------------------------------------------------------------
def _same_bits(a, b) -> bool:
    return np.asarray(a).tobytes() == np.asarray(b).tobytes()


def format_number(x, precision: str = "f32") -> str:
    """单值 → 目标精度下的最短十进制文本（roundtrip 保证）。"""
    if precision not in PRECISIONS:
        raise TokenVectorError(f"未知精度: {precision!r}"
                               f"（可选 {sorted(PRECISIONS)}）")
    target = _cast(np.asarray(x, dtype=np.float64), precision)
    tv = float(target)
    for p in range(1, 10):
        s = "%.*g" % (p, tv)
        if _same_bits(_cast(np.asarray(float(s), dtype=np.float64),
                            precision), target):
            return s
    return "%.9g" % tv


def parse_number(s: str, precision: str = "f32") -> float:
    """十进制文本 → 目标精度数值（bf16 返回其 f32 视图值）。"""
    if precision not in PRECISIONS:
        raise TokenVectorError(f"未知精度: {precision!r}"
                               f"（可选 {sorted(PRECISIONS)}）")
    try:
        return float(_cast(np.asarray(float(s), dtype=np.float64),
                           precision))
    except ValueError as e:
        raise TokenVectorError(f"非法数值: {s!r}") from e


def dumps(timesteps, dim: int, precision: str = "f32") -> str:
    """时间步向量序列 → .tokenvector 文本。

    timesteps: List[List[vector]]，一步一列表，一步内一个 token 向量
    （长度必须为 dim）。vector 为任意可迭代数值序列。
    """
    if precision not in PRECISIONS:
        raise TokenVectorError(f"未知精度: {precision!r}"
                               f"（可选 {sorted(PRECISIONS)}）")
    dim = int(dim)
    if dim <= 0:
        raise TokenVectorError(f"维度必须为正整数: {dim}")
    lines = [f"dim={dim} precision={precision}"]
    for step in timesteps:
        if not step:
            raise TokenVectorError(
                "空时间步非法：{} 必须至少包含一个 token 向量")
        parts = []
        for vec in step:
            arr = _cast(np.asarray(list(vec), dtype=np.float64), precision)
            if arr.size != dim:
                raise TokenVectorError(
                    f"向量长度 {arr.size} != 声明的维度 {dim}")
            parts.append(" ".join(format_number(x, precision) for x in arr))
        lines.append("{" + "|".join(parts) + "}")
    return "\n".join(lines) + "\n"


def loads(text: str):
    """.tokenvector 文本 → (dim, precision, timesteps)。

    timesteps[i][j] = 第 i 个时间步第 j 个 token 的 numpy 向量
    （f32/bf16 为 float32，f16 为 float16）。
    """
    if not isinstance(text, str):
        raise TokenVectorError(f"输入必须是字符串，得到 {type(text).__name__}")

    # ---- 头部 ----
    lines = text.splitlines()
    idx = 0
    header = None
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith("#"):
            continue
        header = line
        break
    if header is None:
        raise TokenVectorError("缺少头部（dim=... precision=...）")
    fields = {}
    for tok in header.split():
        if "=" not in tok:
            raise TokenVectorError(f"头部字段格式错误: {tok!r}")
        k, v = tok.split("=", 1)
        fields[k] = v
    if "dim" not in fields or "precision" not in fields:
        raise TokenVectorError(f"头部缺少 dim/precision: {header!r}")
    try:
        dim = int(fields["dim"])
    except ValueError as e:
        raise TokenVectorError(f"非法维度: {fields['dim']!r}") from e
    if dim <= 0:
        raise TokenVectorError(f"维度必须为正整数: {dim}")
    precision = fields["precision"]
    if precision not in PRECISIONS:
        raise TokenVectorError(f"未知精度: {precision!r}"
                               f"（可选 {sorted(PRECISIONS)}）")

    # ---- 体：词法扫描（数字 / { / } / |）----
    body = "\n".join(lines[idx:])
    tokens: list = []
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c.isspace():
            i += 1
            continue
        if c in "{}|":
            tokens.append(c)
            i += 1
            continue
        j = i
        while j < n and not body[j].isspace() and body[j] not in "{}|":
            j += 1
        tokens.append(body[i:j])
        i = j

    # ---- 语法分析 ----
    timesteps: list = []
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    while pos < len(tokens):
        if peek() != "{":
            raise TokenVectorError(f"期望时间步起始 {{，得到 {peek()!r}")
        pos += 1
        step: list = []
        if peek() == "}":                     # 空时间步：非法
            raise TokenVectorError(
                "空时间步非法：{} 必须至少包含一个 token 向量")
        while True:
            # 一个 token 向量：恰好 dim 个数字
            vec_vals = []
            while peek() is not None and peek() not in ("|", "}", "{"):
                vec_vals.append(parse_number(tokens[pos], precision))
                pos += 1
            if len(vec_vals) != dim:
                raise TokenVectorError(
                    f"向量长度 {len(vec_vals)} != 声明的维度 {dim}")
            step.append(np.asarray(vec_vals, dtype=PRECISIONS[precision]))
            nxt = peek()
            if nxt == "|":
                pos += 1
                if peek() in ("}", "{", None):
                    raise TokenVectorError("分隔符 | 后缺少 token 向量")
                continue
            if nxt == "}":
                pos += 1
                break
            raise TokenVectorError(f"期望 | 或 }}，得到 {nxt!r}")
        timesteps.append(step)
    return dim, precision, timesteps


# ---------------------------------------------------------------------------
# 文件 IO
# ---------------------------------------------------------------------------
def write_tokenvector(path, timesteps, dim: int,
                      precision: str = "f32") -> None:
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(dumps(timesteps, dim, precision), encoding="utf-8")
    tmp.replace(p)


def read_tokenvector(path):
    p = Path(path)
    try:
        return loads(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise TokenVectorError(f"无法读取 {p}: {e}") from e


# ---------------------------------------------------------------------------
# 投影工具（音频/视觉 codec 导出 .tokenvector 时用：原始向量 → d_model）
# ---------------------------------------------------------------------------
def project(vecs, weight) -> np.ndarray:
    """vecs: [N, in] → weight: [D, in] → [N, D]（vecs @ weight.T）。

    weight 通常直接取自模型的模态投影层（如 model.audio_proj.weight），
    保证存档向量与模型实际消费的向量一致。
    """
    v = np.asarray(vecs, dtype=np.float32).reshape(-1, np.asarray(weight).shape[-1])
    w = np.asarray(weight, dtype=np.float32)
    return v @ w.T


# ---------------------------------------------------------------------------
# CLI：解析 .tokenvector 并打印统计
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="tokenvector：解析 .tokenvector 文件并打印统计")
    ap.add_argument("path", help=".tokenvector 文件路径")
    args = ap.parse_args(argv)

    dim, precision, steps = read_tokenvector(args.path)
    n_vec = sum(len(s) for s in steps)
    lens = sorted({len(v) for s in steps for v in s})
    print(f"dim={dim} precision={precision}")
    print(f"时间步={len(steps)} token 向量={n_vec} 向量长度集合={lens}")


if __name__ == "__main__":
    main()
