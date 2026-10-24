# ===========================================================================
# GARGANTUA v1 — tok32 解释器（.tokens 二进制格式）
# ---------------------------------------------------------------------------
# tok-32 编码（2026-10-24 发明，spec.py 第 1 节冻结）：
#   - token ID 为 32-bit 无符号整数（0 .. 2^32-1）。
#   - .tokens 文件 = 纯粹的大端（big-endian）uint32 流，无文件头、无
#     分隔符、无长度字段：每 4 个字节构成一个「字符」，一个字符对应
#     一个 token。用文本编辑器打开即乱码（二进制格式）。
#   - 十六进制文本形式：每个 token 固定 8 位十六进制（不足左侧补零），
#     顺序拼接，无分隔符（定长所以可无歧义切分）。
#
# 解释器职责（本模块）：
#   ids ↔ 二进制（.tokens 文件 / bytes）
#   ids ↔ 十六进制文本
#   越界 ID / 非 4 倍数长度 / 非法十六进制 一律抛 ValueError 子类
#   （Tok32Error），不让 struct.error 之类的异常逃逸。
#
# 用法：
#   from tokens.tok32 import encode_ids, decode_ids, ids_to_hex, hex_to_ids
#   data = encode_ids([64, 12384])          # → bytes（16 字节）
#   ids  = decode_ids(data)                 # → [64, 12384]
#   hexs = ids_to_hex(ids)                  # → "0000004000003060"
#   CLI： python tokens/tok32.py in.tokens out.hex   （二进制→十六进制）
#         python tokens/tok32.py in.hex out.tokens   （十六进制→二进制）
# ===========================================================================
"""tok32: token ID 流 ↔ .tokens 二进制 / 十六进制文本 的双向解释器。"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

# 词表上界（ID 必须 < VOCAB_SIZE_PADDED；reserved 尾段是合法 ID 位，
# 但正常流中不应出现 reserved 占位 token——由调用方按需检查）
_ID_LIMIT = 1 << 32


class Tok32Error(ValueError):
    """tok-32 编解码错误（非法长度 / 越界 ID / 非法十六进制）。"""


# ---------------------------------------------------------------------------
# 二进制（.tokens）
# ---------------------------------------------------------------------------
def encode_ids(ids) -> bytes:
    """token ID 序列 → .tokens 二进制（大端 uint32 流，4 字节/token）。"""
    out = bytearray()
    for i in ids:
        i = _check_id(i)
        out += struct.pack(">I", i)
    return bytes(out)


def decode_ids(data: bytes) -> list:
    """.tokens 二进制 → token ID 序列。长度非 4 的倍数 → Tok32Error。"""
    if isinstance(data, (bytearray, memoryview)):
        data = bytes(data)
    if not isinstance(data, bytes):
        raise Tok32Error(f"tokens 数据必须是 bytes，得到 {type(data).__name__}")
    if len(data) % 4 != 0:
        raise Tok32Error(
            f"tok-32 数据长度 {len(data)} 不是 4 的倍数（损坏的 .tokens 文件）")
    return list(struct.unpack(">%dI" % (len(data) // 4), data))


def read_tokens_file(path) -> list:
    """读 .tokens 二进制文件 → token ID 序列。"""
    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError as e:
        raise Tok32Error(f"无法读取 .tokens 文件 {p}: {e}") from e
    return decode_ids(data)


def write_tokens_file(path, ids) -> None:
    """token ID 序列 → 写 .tokens 二进制文件（先写临时文件再原子替换）。"""
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(encode_ids(ids))
    tmp.replace(p)


# ---------------------------------------------------------------------------
# 十六进制文本
# ---------------------------------------------------------------------------
def ids_to_hex(ids) -> str:
    """token ID 序列 → 8 位/个的十六进制文本（定长、无分隔符）。"""
    return "".join("%08x" % _check_id(i) for i in ids)


def hex_to_ids(text: str) -> list:
    """8 位/个的十六进制文本 → token ID 序列。

    允许空白（换行/空格）便于折行传输；其余字符必须是 [0-9a-fA-F]。
    """
    if not isinstance(text, str):
        raise Tok32Error(f"十六进制输入必须是字符串，得到 {type(text).__name__}")
    compact = "".join(text.split())
    if len(compact) % 8 != 0:
        raise Tok32Error(
            f"十六进制长度 {len(compact)} 不是 8 的倍数"
            f"（每个 token 固定 8 位十六进制）")
    try:
        return [int(compact[i:i + 8], 16) for i in range(0, len(compact), 8)]
    except ValueError as e:
        raise Tok32Error(f"非法十六进制字符: {e}") from e


# ---------------------------------------------------------------------------
# 入参校验
# ---------------------------------------------------------------------------
def _check_id(i) -> int:
    if isinstance(i, bool) or not isinstance(i, int):
        raise Tok32Error(f"token ID 必须是整数，得到 {i!r}")
    if not (0 <= i < _ID_LIMIT):
        raise Tok32Error(f"token ID 越界（32-bit）: {i}")
    return i


def check_vocab_id(i, vocab_size_padded: int) -> int:
    """词表边界校验：ID 必须 < VOCAB_SIZE_PADDED（模型输出维度）。

    与 _check_id 分离：tok-32 本身只约束 32-bit；是否落在词表内
    由调用方按 spec.VOCAB_SIZE_PADDED 校验。
    """
    i = _check_id(i)
    if i >= vocab_size_padded:
        raise Tok32Error(
            f"token ID {i} 超出词表（VOCAB_SIZE_PADDED={vocab_size_padded}）")
    return i


# ---------------------------------------------------------------------------
# CLI：二进制 ↔ 十六进制 互转（扩展名 .tokens/.hex 自动判别方向）
# ---------------------------------------------------------------------------
def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="tok-32 解释器：.tokens ↔ hex")
    ap.add_argument("src", help="输入文件（.tokens 二进制或 .hex 文本）")
    ap.add_argument("dst", help="输出文件")
    args = ap.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_file():
        raise SystemExit(f"输入文件不存在: {src}")
    if dst.suffix.lower() == ".tokens":
        # 文本 → 二进制
        ids = hex_to_ids(src.read_text(encoding="utf-8"))
        write_tokens_file(dst, ids)
        print(f"{src} → {dst}：{len(ids)} 个 token（{len(ids) * 4} 字节）")
    else:
        # 二进制 → 文本
        ids = read_tokens_file(src)
        dst.write_text(ids_to_hex(ids), encoding="utf-8")
        print(f"{src} → {dst}：{len(ids)} 个 token")


if __name__ == "__main__":
    main()
