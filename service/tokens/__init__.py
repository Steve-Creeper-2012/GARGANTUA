# ===========================================================================
# GARGANTUA v1 tokens 包：token-5 词表、离散/向量编解码与时间步结构
# ---------------------------------------------------------------------------
# 四大类 token（spec.py 第 1 节）：
#   regular  常规可见（数字/空白/符号/emoji/中文/英文词）
#   fallback 可见（a–z）
#   special  不可见特殊（<caps>/<user>/<model>/<stop> 等）
#   system   不可见系统（{ } | 时间步边界与分隔）
#
# 格式：
#   .tokens       token-5 位流（27 个符号 × 5 bit，终止符码 27）——tok5.py
#                 文本 {t:NNNNN} / 音频 {w:HEX} / 视觉 {v:HEXx:..y:..}
#                 扩展系统 token <NNNNN>；{ } | 是结构字符不是 token
#   .tokenvector  向量存档（UTF-8，dim+precision 头，{} | 结构）——tokenvector.py
#   tok-32       历史格式（大端 uint32 流）——tok32.py，仅作旧数据读取工具
#
# 脚本（均可 `python tokens/X.py` 或 `python -m tokens.X`）：
#   enctok  文本 → token ID 流（hex / token-5 .tokens）
#   dectok  token-5 .tokens → 可读文本（{text: C732} …）
#   tok5    token-5 编解码（字符 / 位流 / 元素 / 时间步流）
#   gts     历史编辑器格式（.gts，tok-32 时代）→ token ID 流，保留只读
#   tokenset .tokens 训练流 → (ids, step_ids)
#   build_vocab / add_token（Vocab.add_token）  词表构建与扩充
#
# 子模块按需自动导入（`from tokens import tok32` 即可），避免
# `python -m tokens.X` 的双重导入警告。
# ===========================================================================
"""tokens 包：词表、token-5 编解码、时间步结构与 .tokenvector 格式。"""

from .vocab import (CATEGORIES, Vocab, get_vocab,  # noqa: F401
                    reset_vocab_cache)
from .text_codec import decode, decode_ids, encode  # noqa: F401

__all__ = [
    "CATEGORIES", "Vocab", "get_vocab", "reset_vocab_cache",
    "encode", "decode", "decode_ids",
    # 子模块（按需导入）：tok5 / tok32(历史) / timestep / tokenvector /
    # enctok / dectok / text_codec / build_vocab
]
