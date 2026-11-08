# ===========================================================================
# GARGANTUA v1 — tok5：token-6 的兼容导入别名（**不是** 5-bit 解码器）
# ---------------------------------------------------------------------------
# 2026-11-05：.tokens 升级为 token-6（6 bit/字符），实现全部搬到
# tokens/tok6.py。本次不做向后兼容——旧的 5 bit 位流不再可读，本文件
# 也不提供任何 5-bit 解码能力。
#
# 保留这个模块名的唯一理由：还有一批历史测试 / 脚本写的是
# `from tokens import tok5`。它们通过本模块拿到的是 **token-6 的实现**，
# 词表契约头、位宽、字符集一律以 spec.py 为准。新代码请直接
# `from tokens import tok6`。
# ===========================================================================
"""tok5: token-6 的兼容别名；实际实现见 tokens/tok6.py。"""

from __future__ import annotations

from tokens import tok6 as _impl
from tokens.tok6 import *  # noqa: F401,F403  （公开 API 全量转发）

# 旧异常名 → 新异常名（isinstance  catching 仍然有效，是同一个类对象）
Tok5Error = _impl.Tok6Error

# 少数调用方会取私有工具函数，显式转发，免得别名层出现 AttributeError
_iter_codes = _impl._iter_codes
_parse_element = _impl._parse_element
_parse_hex_id = _impl._parse_hex_id
_parse_uint = _impl._parse_uint
_check_id = _impl._check_id
_check_coord = _impl._check_coord
_check_hex = _impl._check_hex
_id_hex = _impl._id_hex
