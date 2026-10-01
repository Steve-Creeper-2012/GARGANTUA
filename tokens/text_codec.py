"""文本 ↔ token 时间步序列 编解码器。

Step 结构（见 spec.py 第 8 节）：
    {"token_ids": [int], "type": spec.TYPE_NONE,
     "modality": "text", "meta": {}}
文本一步一个 token。

编码规则（严格按 spec / 任务约定）：
  - 中文单字逐字查 CJK 区；ASCII 可打印字符（含空格、a–z）逐字符查
    SYMBOL 区；数字逐字符查 DIGIT 区。
  - 英文单词：[A-Za-z]+ 连续段 → 全小写查 ENGLISH 区。命中时按大小写
    规则前置 <caps>（首字母大写，含 McDonald 这类内部大写不保留）或
    <allcaps>（全大写且长度>1）；未命中则 fallback 逐字母输出，大写
    字母前加 <caps>。
  - 字面 "<bold>" 等 12 个格式化 token 与 14 个边界 token 字符串识别为
    对应特殊 token 时间步（decode 还原为原字符串）；其余 "<...>" 按
    普通字符逐字编码。
  - 未收录字符（CJK 扩展 B 及以后、emoji 等）：跳过，记入返回序列
    最后一个 Step 的 meta["skipped"]（decode 无法还原，可接受）。
"""
from __future__ import annotations

import spec
from .vocab import get_vocab

# 输入文本中被识别为特殊 token 的字面字符串（12 格式化 + 14 边界，不含 <pad>）
_RECOGNIZED_SPECIALS = tuple(spec.SPECIAL_TOKENS[1:])

_CAPS = "<caps>"
_ALLCAPS = "<allcaps>"


def _step(token_id: int) -> dict:
    return {
        "token_ids": [token_id],
        "type": spec.TYPE_NONE,
        "modality": "text",
        "meta": {},
    }


def _emit_word(word: str, steps: list, v) -> None:
    """处理一个 [A-Za-z]+ 连续段：ENGLISH 命中走词 token，否则逐字母。"""
    lower = word.lower()
    wid = v.english_ids.get(lower)
    if wid is not None:
        if len(word) > 1 and word.isupper():
            steps.append(_step(v.special_ids[_ALLCAPS]))
        elif not word.islower():
            # 首字母大写（含 "McDonald" 等内部大写，统一按首字母规则）
            steps.append(_step(v.special_ids[_CAPS]))
        steps.append(_step(wid))
        return
    # fallback：逐字母拼写（a–z 位于 SYMBOL 区），大写字母前加 <caps>
    for c in word:
        if c.isupper():
            steps.append(_step(v.special_ids[_CAPS]))
        steps.append(_step(v.symbol_ids[c.lower()]))


def encode(text: str) -> list:
    """文本 → List[Step]。"""
    v = get_vocab()
    steps: list = []
    skipped: list[str] = []
    i, n = 0, len(text)
    while i < n:
        # 1) 字面特殊 token（"<bold>" / "<think>" 等 26 个）
        if text[i] == "<":
            hit = None
            for s in _RECOGNIZED_SPECIALS:
                if text.startswith(s, i):
                    hit = s
                    break
            if hit is not None:
                steps.append(_step(v.special_ids[hit]))
                i += len(hit)
                continue
        ch = text[i]
        # 2) 英文单词段
        if "A" <= ch <= "Z" or "a" <= ch <= "z":
            j = i + 1
            while j < n and ("A" <= text[j] <= "Z" or "a" <= text[j] <= "z"):
                j += 1
            _emit_word(text[i:j], steps, v)
            i = j
            continue
        # 3) ASCII 数字 → DIGIT 区
        if "0" <= ch <= "9":
            steps.append(_step(spec.DIGIT_BEGIN + (ord(ch) - 0x30)))
            i += 1
            continue
        # 4) 单字符查 SYMBOL/CJK 区（含空格、半角/全角符号、全角数字、中文）
        tid = v.char_ids.get(ch)
        if tid is not None:
            steps.append(_step(tid))
            i += 1
            continue
        # 5) 未收录字符：跳过并记录
        skipped.append(ch)
        i += 1
    if skipped and steps:
        steps[-1]["meta"]["skipped"] = skipped
    return steps


def _apply_case(token: str, mode: int) -> str:
    if mode == 2:
        return token.upper()
    if mode == 1:
        return token[0].upper() + token[1:]
    return token


def decode(steps: list) -> str:
    """List[Step] → 文本。<caps>/<allcaps> 作用于其后第一个字母类 token。"""
    v = get_vocab()
    caps_id = v.special_ids[_CAPS]
    allcaps_id = v.special_ids[_ALLCAPS]
    special_literal_ids = set(v.special_ids.values())

    out: list[str] = []
    case_mode = 0  # 0 无 / 1 首字母大写 / 2 全大写
    for st in steps:
        for tid in st["token_ids"]:
            if tid == spec.STOP_ID:          # 停止 token：静默跳过，不输出
                continue
            if tid == caps_id:
                case_mode = max(case_mode, 1)
                continue
            if tid == allcaps_id:
                case_mode = 2
                continue
            tok = v.id_to_token[tid]
            # 格式化/边界特殊 token：还原为原字面字符串
            if tid in special_literal_ids:
                out.append(tok)
                continue
            # 英文词 token（跳过 reserved 占位）
            if spec.ENGLISH_BEGIN <= tid < spec.ENGLISH_BEGIN + spec.ENGLISH_SIZE:
                if not tok.startswith("<"):
                    out.append(_apply_case(tok, case_mode))
                    case_mode = 0
                    continue
            # SYMBOL 区字母（fallback 拼写）：同样消费大小写标记
            if (
                spec.SYMBOL_BEGIN <= tid < spec.SYMBOL_BEGIN + spec.SYMBOL_SIZE
                and len(tok) == 1
                and ("a" <= tok <= "z")
            ):
                out.append(_apply_case(tok, case_mode))
                case_mode = 0
                continue
            # 其余（CJK/数字/符号/reserved）：原样拼接，不消费大小写标记
            out.append(tok)
    return "".join(out)
