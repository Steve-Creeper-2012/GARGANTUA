# ===========================================================================
# GARGANTUA v1 — gts：Token 编辑器文本（.gts）→ token ID 流
# ---------------------------------------------------------------------------
# .gts 是 Token 编辑器里的可读形态（dectok 的逆过程）：
#   {  时间步开始        }  时间步结束        |  步内分隔
#   <user> / <stop> 等  特殊/系统 token（整词匹配词表，否则按普通文本）
#   其余字符            可见 token 文本（走 enctok：常规词表 → fallback 字母）
#
# 解析结果必须能通过 timestep.unwrap（{ } 配对、| 位置合法、无空步），
# 否则抛 GtsError——导出 .tokens 前就把结构错误挡在编辑器里。
#
# 空白规则：换行/回车/制表符视为排版空白直接忽略；**空格是合法 token**
# （WHITESPACE 区），原样保留。
# ===========================================================================
"""gts: Token 编辑器文本 ↔ token ID 流（导出 .tokens 用）。"""

from __future__ import annotations

import spec
from tokens import enctok, timestep
from tokens.vocab import get_vocab

_LAYOUT_WS = "\n\r\t"


class GtsError(ValueError):
    """编辑器文本无法解析成合法 token 流。"""


def _special_id(v, cand: str):
    """<...> 整词命中 special/system 才当特殊 token（否则是普通文本）。"""
    tid = v.special_ids.get(cand)
    if tid is None:
        tid = v.system_ids.get(cand)
    return tid


def parse_gts(text: str, vocab=None) -> list:
    """Token 编辑器文本 → 扁平 token ID 流（含 { } | 系统 token）。

    结构非法（括号不配对 / 空步 / 文本含未收录字符）抛 GtsError。
    """
    if not isinstance(text, str):
        raise GtsError(f"token 文本必须是字符串，得到 {type(text).__name__}")
    v = vocab if vocab is not None else get_vocab()
    ids: list = []
    buf: list = []
    in_step = False
    need_sep = False          # 上一个输出是步内内容 token

    def emit(tid: int):
        nonlocal need_sep
        if in_step and need_sep:
            ids.append(spec.TS_SEP_ID)
        ids.append(tid)
        need_sep = in_step

    def flush():
        if not buf:
            return
        s = "".join(buf)
        buf.clear()
        got, skipped = enctok.encode_text(s)
        if skipped:
            raise GtsError(
                f"以下字符未收录进词表，无法导出: {''.join(skipped)!r}"
                f"（可先在「词表」页添加，或改用 <...> 特殊 token）")
        for t in got:
            emit(t)

    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _LAYOUT_WS:
            i += 1
            continue
        if ch == spec.TS_OPEN:
            flush()
            ids.append(spec.TS_OPEN_ID)
            in_step, need_sep = True, False
            i += 1
            continue
        if ch == spec.TS_CLOSE:
            flush()
            ids.append(spec.TS_CLOSE_ID)
            in_step, need_sep = False, False
            i += 1
            continue
        if ch == spec.TS_SEP:
            flush()
            ids.append(spec.TS_SEP_ID)
            need_sep = False
            i += 1
            continue
        if ch == "<":
            j = text.find(">", i + 1)
            if j != -1:
                cand = text[i:j + 1]
                tid = _special_id(v, cand)
                if tid is not None:
                    flush()
                    emit(tid)
                    i = j + 1
                    continue
        buf.append(ch)
        i += 1
    flush()
    if not ids:
        raise GtsError("token 流为空")
    try:
        timestep.unwrap(ids)
    except timestep.TimestepError as e:
        raise GtsError(f"时间步结构非法: {e}") from e
    return ids
