# ===========================================================================
# GARGANTUA v1 — tok5：token-5 编码（.tokens 的现行格式）
# ---------------------------------------------------------------------------
# 2026-10-28 定案，取代 tok-32（spec.py 第 5b 节是唯一事实来源）。
#
# 三层表示，本模块负责全部互转：
#   1) 字符流（人类可读，27 个符号）：
#        {t:C732}                  文本 token（ID 大写十六进制、自界定）
#        {w:<2132 hex>}            音频 token（533 采样 × 16 bit）
#        {v:<10240 hex>x:10y:24}   视觉 patch（32×32 × RGBA 各 10 bit）
#        {a|b|c}                   一个时间步内多个元素，| 分隔
#        <5>                       扩展系统 token（有嵌入、不进 loss）
#   2) 位流（.tokens 落盘）：5 bit/字符，MSB-first；末字符后跟终止符码
#      27，再补零到字节边界。
#   3) 元素模型（进程内）：
#        {"kind":"text",   "id": int}
#        {"kind":"wave",   "hex": str}                 # 定长 2132
#        {"kind":"visual", "hex": str, "x": int, "y": int}
#        {"kind":"ext",    "id": int}
#      流 = [{"kind":"step","elements":[...]} | {"kind":"ext","id":N}]
#
# 语义（spec 第 5b 节）：{t:}/{w:}/{v:} 是可见 token，有向量坐标、进 loss；
# <id> 是扩展系统 token，有嵌入但没有位置坐标、不进 loss、不显示为内容；
# { } | 是结构字符，本身不是 token。
#
# 用法：
#   from tokens.tok5 import dumps, loads, read_file, write_file, to_display
#   data  = dumps(items)              # → bytes（.tokens）
#   items = loads(data)               # → 流
#   print(to_display(items))          # → "{text: C732}{visual: ... x: 0 y: 0}"
#   CLI：python tokens/tok5.py in.tokens out.txt   （二进制 → 可读文本）
#        python tokens/tok5.py in.txt out.tokens   （可读文本 → 二进制）
# ===========================================================================
"""tok5: token-5 编码 —— 字符 / 位流 / 元素 / 时间步流 的双向编解码。"""

from __future__ import annotations

import sys
from pathlib import Path

import spec

# 词表上界：文本/扩展 token 的 ID 必须 < MAX_VOCAB_SIZE。
# 注意这不是「词表有多长」——词表长度每个词表不同，由各自的 config.json
# 声明（2026-10-30 契约重构）；ID 在字符流里是自界定十六进制，宽度不受
# 格式限制，这个数只是与 spec 对齐的防呆上界（RAM，见 spec.py）。
ID_LIMIT = spec.MAX_VOCAB_SIZE

_HEX = frozenset(spec.T5_HEX_CHARS)         # 波形/视觉载荷：定长大写
_HEX_CHARS = frozenset("0123456789ABCDEFabcdef")   # ID 解析：大小写都收
_DIGITS = frozenset("0123456789")


class Tok5Error(ValueError):
    """token-5 编解码错误（非法字符 / 载荷长度不符 / 结构非法）。"""


# ---------------------------------------------------------------------------
# 1. 字符 ↔ 位流
# ---------------------------------------------------------------------------
def encode_chars(text: str) -> bytes:
    """字符流 → 5 bit/字符 的位流（末尾终止符 + 零填充）。"""
    if not isinstance(text, str):
        raise Tok5Error(f"字符流必须是字符串，得到 {type(text).__name__}")
    acc = 0
    nbits = 0
    out = bytearray()
    code_of = spec.T5_CHAR_CODE
    for pos, ch in enumerate(text):
        code = code_of.get(ch)
        if code is None:
            raise Tok5Error(
                f"位置 {pos} 出现字符集外字符 {ch!r}"
                f"（token-5 只允许 {spec.T5_CHARS}）")
        acc = (acc << spec.T5_BITS_PER_CHAR) | code
        nbits += spec.T5_BITS_PER_CHAR
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
        acc &= (1 << nbits) - 1
    # 终止符：让解码端不必依赖文件长度取整
    acc = (acc << spec.T5_BITS_PER_CHAR) | spec.T5_END_CODE
    nbits += spec.T5_BITS_PER_CHAR
    while nbits >= 8:
        nbits -= 8
        out.append((acc >> nbits) & 0xFF)
    acc &= (1 << nbits) - 1
    if nbits:
        out.append((acc << (8 - nbits)) & 0xFF)
    return bytes(out)


class BitWriter:
    """增量写 token-5 位流（推理边生成边落盘用）。

    用法：w = BitWriter(fh); w.write_chars("{t:C732}"); w.close()
    close() 写终止符并补零到字节边界；重复 close 安全。
    """

    def __init__(self, fh):
        self._fh = fh
        self._acc = 0
        self._nbits = 0
        self._closed = False

    def write_chars(self, text: str) -> None:
        if self._closed:
            raise Tok5Error("BitWriter 已关闭")
        code_of = spec.T5_CHAR_CODE
        buf = bytearray()
        acc, nbits = self._acc, self._nbits
        for pos, ch in enumerate(text):
            code = code_of.get(ch)
            if code is None:
                raise Tok5Error(
                    f"位置 {pos} 出现字符集外字符 {ch!r}"
                    f"（token-5 只允许 {spec.T5_CHARS}）")
            acc = (acc << spec.T5_BITS_PER_CHAR) | code
            nbits += spec.T5_BITS_PER_CHAR
            while nbits >= 8:
                nbits -= 8
                buf.append((acc >> nbits) & 0xFF)
            acc &= (1 << nbits) - 1
        self._acc, self._nbits = acc, nbits
        if buf:
            self._fh.write(bytes(buf))

    def close(self) -> None:
        if self._closed:
            return
        acc, nbits = self._acc, self._nbits
        acc = (acc << spec.T5_BITS_PER_CHAR) | spec.T5_END_CODE
        nbits += spec.T5_BITS_PER_CHAR
        buf = bytearray()
        while nbits >= 8:
            nbits -= 8
            buf.append((acc >> nbits) & 0xFF)
        acc &= (1 << nbits) - 1
        if nbits:
            buf.append((acc << (8 - nbits)) & 0xFF)
        self._fh.write(bytes(buf))
        self._acc = self._nbits = 0
        self._closed = True


def decode_chars(data: bytes) -> str:
    """位流 → 字符流（遇到终止符码 27 停止；缺终止符视为截断）。"""
    if isinstance(data, (bytearray, memoryview)):
        data = bytes(data)
    if not isinstance(data, bytes):
        raise Tok5Error(f"tokens 数据必须是 bytes，得到 {type(data).__name__}")
    chars = spec.T5_CHARS
    acc = 0
    nbits = 0
    out: list = []
    for byte in data:
        acc = (acc << 8) | byte
        nbits += 8
        while nbits >= spec.T5_BITS_PER_CHAR:
            nbits -= spec.T5_BITS_PER_CHAR
            code = (acc >> nbits) & 0x1F
            acc &= (1 << nbits) - 1
            if code == spec.T5_END_CODE:
                return "".join(out)
            if code >= len(chars):
                raise Tok5Error(f"未定义的字符码 {code}（文件损坏或不是 token-5）")
            out.append(chars[code])
    raise Tok5Error(
        "缺少终止符：文件被截断，或这不是 token-5 格式"
        "（旧 tok-32 的 .tokens 请先用 tokens/tok32.py 读取）")


# ---------------------------------------------------------------------------
# 2. 元素构造
# ---------------------------------------------------------------------------
def text(token_id: int) -> dict:
    return {"kind": "text", "id": _check_id(token_id, "文本 token")}


def ext(token_id: int) -> dict:
    return {"kind": "ext", "id": _check_id(token_id, "扩展系统 token")}


def wave(payload) -> dict:
    """音频元素。payload 可以是 2132 位 hex 串、533 个采样值，或
    audio_codec 的 meta["samples"]（533 个 float32 小端 bytes）。"""
    if isinstance(payload, str):
        h = payload.upper()
        if len(h) != spec.T5_WAVE_HEX:
            raise Tok5Error(
                f"音频载荷必须 {spec.T5_WAVE_HEX} 个 hex 字符"
                f"（{spec.AUDIO_SAMPLES_PER_TOKEN} 采样 × "
                f"{spec.AUDIO_SAMPLE_BITS} bit），得到 {len(h)}")
        _check_hex(h, "音频载荷")
        return {"kind": "wave", "hex": h}
    if isinstance(payload, (bytes, bytearray, memoryview)):
        payload = _samples_to_int16(payload)
    return {"kind": "wave", "hex": samples_to_hex(payload)}


def visual(payload, x: int, y: int) -> dict:
    """视觉 patch 元素。payload 可以是 10240 位 hex 串，或 4096 个 8-bit 通道值。"""
    if isinstance(payload, str):
        h = payload.upper()
        if len(h) != spec.T5_PATCH_HEX:
            raise Tok5Error(
                f"视觉载荷必须 {spec.T5_PATCH_HEX} 个 hex 字符"
                f"（{spec.PATCH_PIXELS} 像素 × {spec.T5_PIXEL_BITS} bit），"
                f"得到 {len(h)}")
        _check_hex(h, "视觉载荷")
    else:
        h = rgba8_to_hex(payload)
    return {"kind": "visual", "hex": h,
            "x": _check_coord(x, "x"), "y": _check_coord(y, "y")}


def step(*elements) -> dict:
    """一个时间步。elements 也可直接传一个列表。"""
    if len(elements) == 1 and isinstance(elements[0], (list, tuple)):
        elements = tuple(elements[0])
    if not elements:
        raise Tok5Error("空时间步非法：{} 必须至少包含一个元素")
    return {"kind": "step", "elements": list(elements)}


def _check_id(v, what: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise Tok5Error(f"{what} ID 必须是整数，得到 {v!r}")
    if not (0 <= v < ID_LIMIT):
        raise Tok5Error(f"{what} ID {v} 越界 [0, {ID_LIMIT})")
    return v


def _check_coord(v, axis: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise Tok5Error(f"坐标 {axis} 必须是整数，得到 {v!r}")
    if not (0 <= v < spec.T5_COORD_LIMIT):
        raise Tok5Error(f"坐标 {axis}={v} 越界 [0, {spec.T5_COORD_LIMIT})")
    return v


def _check_hex(h: str, what: str) -> None:
    bad = next((c for c in h if c not in _HEX), None)
    if bad is not None:
        raise Tok5Error(f"{what} 含非十六进制字符 {bad!r}")


def _id_hex(v: int) -> str:
    """ID → 大写十六进制文本（不定宽：长度随 ID 值/词表自然变化）。"""
    return format(v, "X")


# ---------------------------------------------------------------------------
# 3. 元素 ↔ 字符
# ---------------------------------------------------------------------------
def element_to_chars(el: dict) -> str:
    kind = el.get("kind")
    if kind == "text":
        return "t:" + _id_hex(_check_id(el["id"], "文本 token"))
    if kind == "ext":
        return "<" + _id_hex(_check_id(el["id"], "扩展系统 token")) + ">"
    if kind == "wave":
        return "w:" + el["hex"].upper()
    if kind == "visual":
        return ("v:%sx:%dy:%d" % (el["hex"].upper(),
                                  _check_coord(el["x"], "x"),
                                  _check_coord(el["y"], "y")))
    raise Tok5Error(f"未知元素类型: {kind!r}")


def _parse_element(s: str, i: int):
    """从位置 i 解析一个元素 → (element, 新位置)。"""
    n = len(s)
    if i >= n:
        raise Tok5Error("时间步意外结束（缺少元素）")
    c = s[i]
    if c == "t":
        if i + 1 >= n or s[i + 1] != ":":
            raise Tok5Error(f"位置 {i}: 文本元素缺少 ':'")
        tid, j = _parse_hex_id(s, i + 2, "文本 token")
        return text(tid), j
    if c == "w":
        if i + 1 >= n or s[i + 1] != ":":
            raise Tok5Error(f"位置 {i}: 音频元素缺少 ':'")
        j = i + 2 + spec.T5_WAVE_HEX
        if j > n:
            raise Tok5Error(f"位置 {i}: 音频载荷长度不足"
                            f"（需要 {spec.T5_WAVE_HEX} 个 hex 字符）")
        return wave(s[i + 2:j]), j
    if c == "v":
        if i + 1 >= n or s[i + 1] != ":":
            raise Tok5Error(f"位置 {i}: 视觉元素缺少 ':'")
        j = i + 2 + spec.T5_PATCH_HEX
        if j > n:
            raise Tok5Error(f"位置 {i}: 视觉载荷长度不足"
                            f"（需要 {spec.T5_PATCH_HEX} 个 hex 字符）")
        h = s[i + 2:j]
        if j >= n or s[j] != "x" or j + 1 >= n or s[j + 1] != ":":
            raise Tok5Error(f"位置 {j}: 视觉元素缺少 'x:' 坐标")
        x, j = _parse_uint(s, j + 2, "x")
        if j >= n or s[j] != "y" or j + 1 >= n or s[j + 1] != ":":
            raise Tok5Error(f"位置 {j}: 视觉元素缺少 'y:' 坐标")
        y, j = _parse_uint(s, j + 2, "y")
        return visual(h, x, y), j
    raise Tok5Error(
        f"位置 {i}: 时间步内出现非法元素起始字符 {c!r}"
        f"（只能是 t / w / v）")


def _parse_hex_id(s: str, i: int, what: str):
    """自界定十六进制 ID：从位置 i 读到第一个非 hex 字符。

    {t:...} 里以 '}' / '|' 收尾、<...> 里以 '>' 收尾——这些结构字符都不
    是 hex 字符，贪心读取天然自界定，ID 宽度随词表自然变化。至少一位。
    解析大小写都收（用户手敲小写很常见），写出统一大写（见 _id_hex）。
    """
    j = i
    while j < len(s) and s[j] in _HEX_CHARS:
        j += 1
    if j == i:
        raise Tok5Error(f"位置 {i}: {what} 缺少十六进制 ID")
    return int(s[i:j], 16), j


def _parse_uint(s: str, i: int, what: str):
    j = i
    while j < len(s) and s[j] in _DIGITS:
        j += 1
    if j == i:
        raise Tok5Error(f"位置 {i}: 缺少{what}坐标数字")
    return int(s[i:j]), j


# ---------------------------------------------------------------------------
# 4. 时间步流 ↔ 字符
# ---------------------------------------------------------------------------
def items_to_chars(items) -> str:
    """流 → 字符流。"""
    out: list = []
    for it in items:
        kind = it.get("kind")
        if kind == "step":
            els = it.get("elements") or []
            if not els:
                raise Tok5Error("空时间步非法：{} 必须至少包含一个元素")
            out.append("{")
            out.append("|".join(element_to_chars(e) for e in els))
            out.append("}")
        elif kind == "ext":
            out.append(element_to_chars({"kind": "ext", "id": it["id"]}))
        else:
            raise Tok5Error(f"未知流元素: {kind!r}")
    return "".join(out)


def parse_items(text: str) -> list:
    """字符流 → 流（结构校验；非法抛 Tok5Error）。"""
    if not isinstance(text, str):
        raise Tok5Error(f"字符流必须是字符串，得到 {type(text).__name__}")
    items: list = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "{":
            i += 1
            elements: list = []
            while True:
                el, i = _parse_element(text, i)
                elements.append(el)
                if i >= n:
                    raise Tok5Error("时间步括号不配对：流结束时 { 未闭合")
                nxt = text[i]
                if nxt == "|":
                    i += 1
                    continue
                if nxt == "}":
                    i += 1
                    break
                raise Tok5Error(
                    f"位置 {i}: 元素后只能跟 '|' 或 '}}'，得到 {nxt!r}")
            items.append({"kind": "step", "elements": elements})
        elif c == "<":
            j = text.find(">", i + 1)
            if j < 0:
                raise Tok5Error(f"位置 {i}: '<' 未闭合")
            body = text[i + 1:j]
            if not body or any(ch not in _HEX_CHARS for ch in body):
                raise Tok5Error(
                    f"位置 {i}: 扩展 token 必须是 <十六进制 ID>，"
                    f"得到 {text[i:j + 1]!r}")
            items.append({"kind": "ext", "id": int(body, 16)})
            i = j + 1
        elif c == "}":
            raise Tok5Error(f"位置 {i}: '}}' 出现在时间步之外")
        elif c == "|":
            raise Tok5Error(f"位置 {i}: '|' 出现在时间步之外")
        else:
            raise Tok5Error(
                f"位置 {i}: 流中出现游离字符 {c!r}"
                f"（token 必须包在 {{ }} 或 < > 内）")
    return items


# ---------------------------------------------------------------------------
# 5. .tokens 文件
# ---------------------------------------------------------------------------
def dumps(items) -> bytes:
    """流 → .tokens 位流。"""
    return encode_chars(items_to_chars(items))


def loads(data: bytes) -> list:
    """.tokens 位流 → 流。"""
    return parse_items(decode_chars(data))


def read_file(path) -> list:
    p = Path(path)
    try:
        return loads(p.read_bytes())
    except OSError as e:
        raise Tok5Error(f"无法读取 .tokens 文件 {p}: {e}") from e


def write_file(path, items) -> None:
    """写 .tokens（唯一临时文件 + fsync + 原子替换）。"""
    import os
    import uuid
    p = Path(path)
    data = dumps(items)
    tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 6. 载荷编解码
# ---------------------------------------------------------------------------
def samples_to_hex(samples) -> str:
    """533 个采样（int16，-32768..32767）→ 2132 位 hex（无符号 16 bit）。"""
    vals = list(samples)
    if len(vals) != spec.AUDIO_SAMPLES_PER_TOKEN:
        raise Tok5Error(
            f"音频载荷必须恰好 {spec.AUDIO_SAMPLES_PER_TOKEN} 个采样，"
            f"得到 {len(vals)}")
    out = []
    for v in vals:
        iv = int(round(v))
        if not (-32768 <= iv <= 32767):
            raise Tok5Error(f"采样值越界 int16: {v}")
        out.append("%04X" % (iv & 0xFFFF))
    return "".join(out)


def hex_to_samples(h: str) -> list:
    """2132 位 hex → 533 个 int16 采样。"""
    h = h.upper()
    if len(h) != spec.T5_WAVE_HEX:
        raise Tok5Error(f"音频载荷长度必须是 {spec.T5_WAVE_HEX}，得到 {len(h)}")
    _check_hex(h, "音频载荷")
    out = []
    for i in range(0, len(h), 4):
        v = int(h[i:i + 4], 16)
        out.append(v - 0x10000 if v >= 0x8000 else v)
    return out


def rgba8_to_hex(values) -> str:
    """4096 个 8-bit 通道值（RGBA 平铺）→ 10240 位 hex。

    8 bit → 10 bit 用 round(v * 1023 / 255)，保证 0 与 255 精确落在
    0 / 1023（往返无损到 8 bit）。
    """
    vals = list(values)
    if len(vals) != spec.PATCH_VEC_DIM:
        raise Tok5Error(
            f"视觉载荷必须 {spec.PATCH_VEC_DIM} 个通道值（"
            f"{spec.PATCH_PIXELS} 像素 × 4），得到 {len(vals)}")
    acc = 0
    nbits = 0
    out = bytearray()
    for v in vals:
        iv = int(v)
        if not (0 <= iv <= 255):
            raise Tok5Error(f"通道值越界 8-bit: {v}")
        q = (iv * spec.T5_CHANNEL_MAX + 127) // 255
        acc = (acc << 10) | q
        nbits += 10
        while nbits >= 8:
            nbits -= 8
            out.append((acc >> nbits) & 0xFF)
        acc &= (1 << nbits) - 1
    return bytes(out).hex().upper()


def hex_to_pixels10(h: str) -> list:
    """10240 位 hex → 4096 个 10-bit 通道值（精确值，供检查/调试）。"""
    h = h.upper()
    if len(h) != spec.T5_PATCH_HEX:
        raise Tok5Error(f"视觉载荷长度必须是 {spec.T5_PATCH_HEX}，得到 {len(h)}")
    _check_hex(h, "视觉载荷")
    data = bytes.fromhex(h)
    acc = 0
    nbits = 0
    out: list = []
    for byte in data:
        acc = (acc << 8) | byte
        nbits += 8
        while nbits >= 10:
            nbits -= 10
            out.append((acc >> nbits) & 0x3FF)
        acc &= (1 << nbits) - 1
    return out


def hex_to_rgba8(h: str) -> list:
    """10240 位 hex → 4096 个 8-bit 通道值（回量化，供模型投影用）。"""
    return [(v * 255 + 511) // 1023 for v in hex_to_pixels10(h)]


# ---------------------------------------------------------------------------
# 7. 显示形式（编辑器 / 人读）
# ---------------------------------------------------------------------------
_DISPLAY_MARK = {"t": "text", "w": "wave", "v": "visual"}
_SHORT_MARK = {v: k for k, v in _DISPLAY_MARK.items()}


def to_display(items, name_of=None) -> str:
    """流 → 可读文本：{text: C732} / {wave: HEX} / {visual: HEX x: 1 y: 2}。

    ID 是大写十六进制（不定宽）。name_of: 可选 callable(id) -> str，把
    扩展 token 的 ID 映射回名字（如 5 → "<stop>"）；缺省显示 <5>。
    """
    out: list = []
    for it in items:
        if it.get("kind") == "ext":
            tid = it["id"]
            nm = name_of(tid) if name_of else None
            out.append(nm if nm else "<%s>" % _id_hex(tid))
            continue
        parts = []
        for el in it.get("elements", []):
            k = el["kind"]
            if k == "text":
                parts.append("text: " + _id_hex(el["id"]))
            elif k == "wave":
                parts.append("wave: " + el["hex"].upper())
            elif k == "visual":
                parts.append("visual: %s x: %d y: %d"
                             % (el["hex"].upper(), el["x"], el["y"]))
            else:
                parts.append(element_to_chars(el))
        out.append("{" + " | ".join(parts) + "}")
    return "".join(out)


def vocab_name_of(v):
    """构造 name_of 回调：扩展 token 的 ID → 词表里的名字（<stop> 等）。

    可见区 ID 返回 None（那些是普通文本 token，显示成 {text: NNNNN}）。
    """
    def f(tid: int):
        # 2026-10-30：不可显示不再靠「ID < HEAD_SIZE」这个固定边界
        # 判断——用户可以把新的不可显示 token 追加到任意 ID。以词表自身
        # 记录的形态为准。
        if v.is_invisible(tid):
            name = v.id_to_token[tid]
            return None if name.startswith("<reserved:") else name
        return None
    return f


def vocab_id_of(v):
    """构造 id_of 回调：名字（"<stop>"）→ 扩展 token 的 ID；查不到 None。"""
    def f(name: str):
        tid = v.special_ids.get(name)
        if tid is None:
            tid = v.system_ids.get(name)
        return tid
    return f


def parse_display(s: str, id_of=None) -> list:
    """可读文本 → 流。同时接受紧凑形式（{t:C732}）与显示形式。

    ID 是十六进制（大小写都收，不定宽）。id_of: 可选 callable(name) -> id，
    把 "<stop>" 这类名字解析回 ID；解析不出则报错（缺省只接受 <5> 这类
    纯十六进制数字形式）。
    """
    if not isinstance(s, str):
        raise Tok5Error(f"文本必须是字符串，得到 {type(s).__name__}")
    t = "".join(s.split())              # token-5 没有空格，排版空白一律丢弃
    for long_name, short in _SHORT_MARK.items():
        t = t.replace(long_name + ":", short + ":")
    # <name> → <id>
    if id_of is not None and "<" in t:
        out_chars: list = []
        i, n = 0, len(t)
        while i < n:
            if t[i] == "<":
                j = t.find(">", i + 1)
                if j < 0:
                    raise Tok5Error(f"位置 {i}: '<' 未闭合")
                body = t[i + 1:j]
                if body and all(ch in _HEX_CHARS for ch in body):
                    out_chars.append(t[i:j + 1])
                else:
                    tid = id_of("<" + body + ">")
                    if tid is None:
                        raise Tok5Error(
                            f"未知扩展 token <{body}>（词表里没有这个名字）")
                    out_chars.append("<" + _id_hex(tid) + ">")
                i = j + 1
            else:
                out_chars.append(t[i])
                i += 1
        t = "".join(out_chars)
    return parse_items(t)


# ---------------------------------------------------------------------------
# 8. 与旧 Step 结构互转（模型管线兼容层）
# ---------------------------------------------------------------------------
def _samples_to_int16(payload) -> list:
    """Step 里的音频载荷 → 533 个 int16。

    接受三种形态：bytes（533 个 float32 小端，归一化到 [-1,1]，
    audio_codec 的产物）、numpy 数组、普通数值序列。
    """
    if isinstance(payload, (bytes, bytearray, memoryview)):
        import struct
        raw = bytes(payload)
        n = len(raw) // 4
        vals = struct.unpack("<%df" % n, raw)
        return [max(-32768, min(32767, int(round(v * 32768.0))))
                for v in vals]
    if hasattr(payload, "tolist"):
        payload = payload.tolist()
    return [int(v) for v in payload]


def from_steps(steps, vocab=None) -> list:
    """旧 List[Step] → token-5 流。

    vocab 缺省用进程词表；它决定「哪些 ID 是不可显示 token」（见
    vocab_name_of）——不要用固定边界判断。

    Step = {"token_ids", "type", "modality", "meta"}：
      - text：token_ids → {t:}；不可显示 ID 按扩展系统 token <id> 写出
      - audio：meta["samples"]（533 个采样）→ {w:}；末尾不足 533 的
        残段直接丢弃（spec 5b：每个音频 token 必须恰好 533 采样）
      - visual：meta["patches"]（x/y/vec 4096 字节）→ {v:...x:..y:..}
      - mixed：同一步内视觉 + 音频，视觉在前、音频在后
    """
    if vocab is None:
        from tokens.vocab import get_vocab
        vocab = get_vocab()
    items: list = []
    for st in steps:
        modality = st.get("modality", "text")
        meta = st.get("meta") or {}
        if modality == "text":
            for tid in st.get("token_ids") or []:
                if vocab.is_invisible(int(tid)):
                    items.append({"kind": "ext", "id": int(tid)})
                else:
                    items.append(step(text(int(tid))))
            continue
        if modality in ("visual", "audio", "mixed"):
            els: list = []
            for p in meta.get("patches") or []:
                els.append(visual(p["vec"], int(p["x"]), int(p["y"])))
            if meta.get("samples") is not None:
                n = meta.get("n")
                if n is None or n >= spec.AUDIO_SAMPLES_PER_TOKEN:
                    els.append(wave(_samples_to_int16(meta["samples"])))
            for a in meta.get("audio") or []:          # 视频帧内的多个音元
                n = a.get("n")
                if n is None or n >= spec.AUDIO_SAMPLES_PER_TOKEN:
                    els.append(wave(_samples_to_int16(a["samples"])))
            if not els:
                # 只有「末尾不足 533 采样的残段」→ 整步丢弃
                continue
            items.append(step(els))
            continue
        raise Tok5Error(f"未知 modality: {modality!r}")
    return items


def to_steps(items) -> list:
    """token-5 流 → 旧 List[Step]（供模型前向使用）。

    ext 元素单独成一步（token_ids=[id], modality="text", meta.ext=True），
    与旧体系「边界 token 独占一步」一致。同一时间步内视觉 + 音频按
    modality="mixed" 输出。
    """
    steps: list = []
    for it in items:
        if it.get("kind") == "ext":
            steps.append({"token_ids": [int(it["id"])], "type": spec.TYPE_NONE,
                          "modality": "text", "meta": {"ext": True}})
            continue
        els = it.get("elements") or []
        kinds = {e["kind"] for e in els}
        if kinds <= {"text"}:
            steps.append({"token_ids": [int(e["id"]) for e in els],
                          "type": spec.TYPE_USER, "modality": "text",
                          "meta": {}})
            continue
        if kinds <= {"visual", "wave"}:
            patches = [{"x": e["x"], "y": e["y"],
                        "vec": bytes(hex_to_rgba8(e["hex"]))}
                       for e in els if e["kind"] == "visual"]
            waves = [e for e in els if e["kind"] == "wave"]
            meta = {}
            if patches:
                meta["patches"] = patches
                meta["grid"] = _grid_of(patches)
            if len(waves) == 1:
                meta["samples"] = _int16_to_bytes(
                    hex_to_samples(waves[0]["hex"]))
                meta["n"] = spec.AUDIO_SAMPLES_PER_TOKEN
            elif len(waves) > 1:
                # 视频帧：同一时间步里多个音元 → meta["audio"] 列表
                # （与 audio_codec.align_audio_to_frames 的形态一致）
                meta["audio"] = [
                    {"samples": _int16_to_bytes(hex_to_samples(w["hex"])),
                     "n": spec.AUDIO_SAMPLES_PER_TOKEN} for w in waves]
            modality = ("mixed" if patches and waves
                        else ("visual" if patches else "audio"))
            steps.append({"token_ids": [], "type": spec.TYPE_USER,
                          "modality": modality, "meta": meta})
            continue
        raise Tok5Error(
            f"同一时间步内混用了文本与模态元素（{sorted(kinds)}）："
            f"文本必须独占一步")
    return steps


def _grid_of(patches) -> list:
    return [max(p["x"] for p in patches) + 1,
            max(p["y"] for p in patches) + 1]


def _int16_to_bytes(values) -> bytes:
    """533 个 int16 → 533 个 float32 小端（与 audio_codec 的 meta 同构）。"""
    import struct
    vals = [max(-32768, min(32767, int(v))) / 32768.0 for v in values]
    return struct.pack("<%df" % len(vals), *vals)


# ---------------------------------------------------------------------------
# 9. CLI：.tokens ↔ 可读文本
# ---------------------------------------------------------------------------
def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="token-5 解释器：.tokens ↔ 可读文本")
    ap.add_argument("src", help="输入文件（.tokens 二进制，或文本形式）")
    ap.add_argument("dst", help="输出文件（扩展名 .tokens 表示写二进制）")
    ap.add_argument("--stats", action="store_true", help="只打印统计信息")
    args = ap.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_file():
        raise SystemExit(f"输入文件不存在: {src}")

    if src.suffix.lower() == ".tokens":
        items = read_file(src)
    else:
        items = parse_display(src.read_text(encoding="utf-8"))

    n_steps = sum(1 for it in items if it["kind"] == "step")
    n_ext = sum(1 for it in items if it["kind"] == "ext")
    n_text = sum(1 for it in items if it["kind"] == "step"
                 for e in it["elements"] if e["kind"] == "text")
    n_wave = sum(1 for it in items if it["kind"] == "step"
                 for e in it["elements"] if e["kind"] == "wave")
    n_vis = sum(1 for it in items if it["kind"] == "step"
                for e in it["elements"] if e["kind"] == "visual")

    if args.stats:
        print(f"时间步 {n_steps}｜文本 {n_text}｜音频 {n_wave}｜"
              f"视觉 {n_vis}｜扩展 {n_ext}")
        return 0

    if dst.suffix.lower() == ".tokens":
        write_file(dst, items)
        print(f"{src} → {dst}：{dst.stat().st_size} 字节"
              f"（{n_steps} 步 / {n_text + n_wave + n_vis} 元素）")
    else:
        dst.write_text(to_display(items), encoding="utf-8")
        print(f"{src} → {dst}：{n_steps} 步 / {n_text + n_wave + n_vis} 元素")
    return 0


if __name__ == "__main__":
    sys.exit(main())
