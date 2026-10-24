# ===========================================================================
# GARGANTUA v1 — 文本 ↔ token 时间步序列 编解码器（tok-32 体系）
# ---------------------------------------------------------------------------
# Step 结构（见 spec.py 第 8 节）：
#     {"token_ids": [int], "type": spec.TYPE_NONE,
#      "modality": "text", "meta": {}}
# 文本一步一个 token（{ } | 系统 token 不显式出现，步边界由 Step 列表
# 承载；需要显式结构时用 tokens/timestep.py 的 wrap/wrap_steps）。
#
# 编码规则 = tokens/enctok.py（常规可见优先、fallback 兜底、英文词段
# + <caps>）；解码规则：
#   - <stop> 静默跳过（生成终止标记，不产生输出）；
#   - <caps> 置「首字母大写」模式，作用于其后第一个英文词/字母 token；
#   - 其余特殊 token（<user> 等）与系统 token（{ } |）原样输出字面量；
#   - 未收录字符在编码时被跳过并记入最后一个 Step 的 meta["skipped"]
#     （decode 无法还原，可接受）。
#
# 2026-10-24 重构：编码核心移到 enctok.encode_text；本模块只负责
# List[Step] 装配与 <caps> 解码，供 train / infer / server 使用。
# ===========================================================================
"""text_codec: 文本 ↔ List[Step]（每 token 一个时间步）。"""

from __future__ import annotations

import spec
from .vocab import get_vocab

_CAPS_ID = spec.SPECIAL_BEGIN + spec.SPECIAL_TOKENS.index("<caps>")


def _step(token_id: int) -> dict:
    return {
        "token_ids": [token_id],
        "type": spec.TYPE_NONE,
        "modality": "text",
        "meta": {},
    }


def encode(text: str) -> list:
    """文本 → List[Step]（每 token 独占一个时间步）。"""
    from . import enctok          # 懒加载：保持 `python -m tokens.enctok` 干净
    ids, skipped = enctok.encode_text(text)
    steps = [_step(i) for i in ids]
    if skipped and steps:
        steps[-1]["meta"]["skipped"] = skipped
    return steps


def _apply_case(token: str, mode: int) -> str:
    if mode == 1 and token:
        return token[0].upper() + token[1:]
    return token


def _is_wordish(v, tid: int) -> bool:
    """英文词 / fallback 字母 token（可被 <caps> 作用）。"""
    tok = v.id_to_token[tid]
    if not tok or not tok.isascii() or not tok.isalpha():
        return False
    if spec.ENGLISH_BEGIN <= tid < spec.ENGLISH_BEGIN + spec.ENGLISH_SIZE:
        return True
    if spec.FALLBACK_BEGIN <= tid < spec.FALLBACK_BEGIN + spec.FALLBACK_SIZE:
        return True
    return False


def decode_ids(ids) -> str:
    """token ID 流 → 文本（<caps> 消费、<stop> 跳过、特殊/系统原样）。

    reserved 占位或越界 ID 输出 8 位十六进制（不让 IndexError 逃逸）。
    """
    v = get_vocab()
    out: list = []
    case_mode = 0
    for tid in ids:
        if not isinstance(tid, int) or isinstance(tid, bool):
            raise ValueError(f"token ID 必须是整数，得到 {tid!r}")
        if not (0 <= tid < len(v)):
            raise ValueError(f"token ID 越界: {tid}")
        if tid == spec.STOP_ID:            # 停止 token：静默跳过
            continue
        if tid == _CAPS_ID:
            case_mode = max(case_mode, 1)
            continue
        tok = v.id_to_token[tid]
        if tok.startswith("<reserved:"):   # 占位槽：输出十六进制 ID
            out.append("%08x" % tid)
            continue
        if _is_wordish(v, tid):
            out.append(_apply_case(tok, case_mode))
            case_mode = 0
            continue
        # 特殊 token / 系统 token / 中文 / 数字 / 符号：原样拼接，
        # 不消费大小写标记
        out.append(tok)
    return "".join(out)


def decode(steps: list) -> str:
    """List[Step] → 文本。"""
    ids = [t for st in steps for t in st["token_ids"]]
    return decode_ids(ids)
