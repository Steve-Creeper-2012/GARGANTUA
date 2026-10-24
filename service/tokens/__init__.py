# ===========================================================================
# GARGANTUA v1 tokens 包：tok-32 词表、离散/向量编解码与时间步结构
# ---------------------------------------------------------------------------
# 四大类 token（spec.py 第 1 节）：
#   regular  常规可见（数字/空白/符号/emoji/中文/英文词）
#   fallback 可见（a–z）
#   special  不可见特殊（<caps>/<user>/<model>/<stop> 等）
#   system   不可见系统（{ } | 时间步边界与分隔）
#
# 格式：
#   .tokens       tok-32 二进制（大端 uint32 流，4 字节/token）——tok32.py
#   .tokenvector  向量存档（UTF-8，dim+precision 头，{} | 结构）——tokenvector.py
#
# 脚本（均可 `python tokens/X.py` 或 `python -m tokens.X`）：
#   enctok  文本 → token ID 流（hex / .tokens）
#   dectok  token ID 流 → 可读形式（{ } | + hex / 表面文本）
#   build_vocab / add_token（Vocab.add_token）  词表构建与扩充
#
# 子模块按需自动导入（`from tokens import tok32` 即可），避免
# `python -m tokens.X` 的双重导入警告。
# ===========================================================================
"""tokens 包：词表、tok-32 编解码、时间步结构与 .tokenvector 格式。"""

from .vocab import (CATEGORIES, Vocab, get_vocab,  # noqa: F401
                    reset_vocab_cache)
from .text_codec import decode, decode_ids, encode  # noqa: F401

__all__ = [
    "CATEGORIES", "Vocab", "get_vocab", "reset_vocab_cache",
    "encode", "decode", "decode_ids",
    # 子模块（按需导入）：tok32 / timestep / tokenvector /
    # enctok / dectok / text_codec / build_vocab
]
