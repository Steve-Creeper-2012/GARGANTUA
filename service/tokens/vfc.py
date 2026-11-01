# ===========================================================================
# GARGANTUA v1 — .vfc 词表文件格式（vocabulary for converting）
# ---------------------------------------------------------------------------
# .vfc 是 GARGANTUA 唯一的词表文件格式（取代早期的 vocab.txt / vocab.json
# 内部表示）。
# 编码 .tokens 文件时要读词表，所以词表的可读性直接影响调试效率：.vfc
# 用显式 ID + 非转义内容 + 最小必要转义。
#
# —— 格式 ——
# 编码 UTF-8，一行一个 token，每行严格是 `内容:ID`：
#   "hello":A1B2         可显示 token（双引号包裹）
#   'hello':A1B3         可显示 token（单引号包裹，与上行语义相同）
#   <model>:3            不可显示标记（尖括号包裹）
#   <cx:0>:9             不可显示标记（draw 图元；尖括号内允许 ':'）
# ID 是**十六进制**（2026-10-31 steve 定案：所有 token 的 ID 都是 16 进
# 制，长度因不同词表而异），解析大小写都收、写出统一大写。冒号必须半
# 角，且后面直接跟十六进制数字（不允许空格）。
#
# —— 为什么要用两种包裹形态 ——
# 词表里同时存在「文本内容」与「不可显示标记」两类条目：`<stop>` 是一个
# 不可见的控制标记，而用户文本里也可能真的出现 `<stop>` 这 6 个字符。
# 靠包裹形态区分二者，往返不丢信息：
#   <stop>:5        → 不可显示标记（text = '<stop>'，visible=False）
#   "<stop>":40000  → 文本内容恰好是 "<stop>" 这 6 个字符的可显示 token
#                     （text = '<stop>'，visible=True）
#
# 【不可显示条目的 text **自带尖括号**】
#   与 spec.SPECIAL_TOKENS / vocab.json 的 id_to_token 完全一致，例如
#   '<caps>'、'<cx:0>'。写出时原样输出、再加 `:ID`——不二次加括号（否则
#   会写成 '<<caps>>'）。
#
#   为什么不用裸名（'caps'）：基础词表里 stop / model / user / caps 全都
#   是收录的英文单词（vocab.json 里分别是 ID 50948 / 52544 / 62187 /
#   61131）。若不可显示条目取裸名，它们就会和这些可见英文词撞在「文本
#   全表唯一」这条规则上——基础词表根本无法用 .vfc 表达。带尖括号则天然
#   与英文单词区分开。
#
#   代价：<stop>:5 与 "<stop>":N 的文本相同，受唯一性规则约束不能同时出现
#   在同一个文件里。可接受——两者的区别由 visible 标记保留，且反查映射里
#   同文本两个 ID 本来就是歧义。
#
# —— 引号内的转义（JSON 兼容子集，严格模式）——
#   \\ \' \" \n \r \t \uXXXX（含代理对）
# 其它任何 `\x` 组合一律视为语法错误：宽松解析会让「写出去再读回来」产生
# 静默漂移（例如 \d 被原样保留，下次再转义就变成 \\d），宁可在保存时报错。
#
# —— 为什么 ID 必须严格连续 0..N-1 ——
# 每个 ID 都占一行 embedding（哪怕该行是空位），空位白吃 RAM。.vfc 要求
# 全部 ID 恰好构成 0,1,...,N-1，不允许任何空位——这与旧扁平词表「空位 =
# 文件里没有这一行」的语义不同：.vfc 里没有空位这个概念，删 token 就是
# 重排。需要空位的场景（模型权重对齐）不通过 .vfc 表达。
#
# —— 架构契约头 ——
# 文件开头 spec.HEAD_SIZE（2970）行是架构契约，名字与顺序全部从 spec.py
# 推导（禁止硬编码）：
#   [0, 6)     SPECIAL：spec.SPECIAL_TOKENS 的 6 个不可显示标记
#   [6, 9)     SYSTEM ：spec.SYSTEM_CHARS（{ } |）
#   [9, 2970)  DRAW   ：图元输出 token（spec.DRAW_VOCAB_SIZE = 2961 个）
# 2026-10-30 契约重构后 spec 的头也是紧凑的（不再留预留空位），所以 .vfc
# 里的 ID **就是** spec 运行时 ID，二者一一对应，不需要任何换算。
#
# 所有区间常数一律取自 spec.py（冻结契约），禁止硬编码。
# ===========================================================================
""".vfc 词表文件格式的解析、校验与序列化（含架构契约头校验）。"""

from __future__ import annotations

import os
import re
import uuid

import spec

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
# 写出时固定写在第一行；读取时只当普通注释行跳过（宽容读取：老文件/手工
# 编辑的文件没有这一行也能读）。
MAGIC = "#gargantua-vfc-v1"

# ID 上界：spec.MAX_VOCAB_SIZE = 100000。它不是 spec 规定的词表长度
# （长度由每个词表的 config.json 声明），也不是格式约束（ID 是自界定
# 十六进制，宽度不限）——纯粹是 RAM 防呆上界（每 ID 约 8KB embedding，
# 见 spec.py），与 tok5 的 ID_LIMIT 对齐。
MAX_VOCAB_SIZE = spec.MAX_VOCAB_SIZE

# system 区：spec 有 SYSTEM_CHARS；旧版没有，退回三个 TS_* 常量
SYSTEM_CHARS = list(getattr(spec, "SYSTEM_CHARS", None)
                    or [spec.TS_OPEN, spec.TS_CLOSE, spec.TS_SEP])

# 尖括号内允许的名字：一个标识符式名字，不含空白、不含 '>'；允许 ':' 是
# 因为 draw token 的名字形如 'cx:0'、'draw:special:0'——'>' 是唯一终止符，
# 所以含冒号并不产生歧义。
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./:-]*\Z")
# 不可显示条目的 text 形态：必须是 `<名字>`（含尖括号）
_ANGLED_TEXT_RE = re.compile(r"<[A-Za-z_][A-Za-z0-9_./:-]*>\Z")

# 行尾 `:<ID>`：冒号后紧跟十六进制数字（0-9/A-F/a-f），直到行尾（不允许
# 空格/符号）；解析大小写都收，写出统一大写。
_TAIL_RE = re.compile(r"^(.*):([0-9A-Fa-f]+)\Z")

# 头部契约错误最多报这么多条：ID 错位会让后续行全部对不上，逐行报 2970 条
# 没有任何排查价值，截断即可（其余错误类型不受此限制）。
_MAX_HEAD_ERRORS = 5


class VfcError(ValueError):
    """.vfc 语法错误；携带 line（行号，1 起）与 message。

    line=None 表示错误不落在具体某一行（ID 空位、条目数不足等全局问题）。
    """

    def __init__(self, message: str, line: int | None = None,
                 source: str | None = None):
        self.message = message
        self.line = line
        self.source = source
        super().__init__(message)

    def __str__(self) -> str:                       # pragma: no cover
        return self.message


# 内部错误条目（带行号，供 validate_vfc 批量收集）
class _Err:
    __slots__ = ("line", "message")

    def __init__(self, line, message):
        self.line = line
        self.message = message


# ---------------------------------------------------------------------------
# 架构契约头（从 spec 推导，禁止硬编码）
# ---------------------------------------------------------------------------
# draw 区的字段分段：用相邻 offset 的差值求段长，spec 改 bins 数量时这里
# 自动跟随（build_vocab.py 是显式列 size 的，等价但会重复一遍常量）。
_DRAW_OFFSETS = (
    spec.DRAW_X_OFFSET,
    spec.DRAW_Y_OFFSET,
    spec.DRAW_ANGLE_OFFSET,
    spec.DRAW_W_OFFSET,
    spec.DRAW_H_OFFSET,
    spec.DRAW_SHAPE_OFFSET,
    spec.DRAW_R_OFFSET,
    spec.DRAW_G_OFFSET,
    spec.DRAW_B_OFFSET,
    spec.DRAW_SPECIAL_OFFSET,
    spec.DRAW_VOCAB_SIZE,
)
# 最后一段（special bins）的字段前缀名，与 build_vocab.py 保持一致
_DRAW_SPECIAL_PREFIX = "draw:special"


def _draw_head_names() -> list:
    """从 spec 推导 2961 个 draw token 的名字（含尖括号），顺序即 DRAW 区布局。

    名字带尖括号，与 spec.SPECIAL_TOKENS / vocab.json 的 id_to_token 拼写
    完全一致（'<cx:0>'、'<draw:special:0>'），加载侧可直接装进 id_to_token。
    """
    names = []
    for i, field in enumerate(spec.DRAW_FIELD_ORDER):
        size = _DRAW_OFFSETS[i + 1] - _DRAW_OFFSETS[i]
        names.extend(f"<{field}:{n}>" for n in range(size))
    # 末尾 special bins：仓库布局自带的 pad + 2 预留
    size = _DRAW_OFFSETS[-1] - _DRAW_OFFSETS[-2]
    names.extend(f"<{_DRAW_SPECIAL_PREFIX}:{n}>" for n in range(size))
    return names


def head_entries() -> list:
    """架构契约头条目：[(id, text, visible), ...]，id 从 0 起严格连续。

    三段的名字与顺序全部取自 spec（spec 一改这里自动跟随，无硬编码）：
      [0, 6)    SPECIAL：spec.SPECIAL_TOKENS（'<caps>' '<user>' '<user/>'
                '<model>' '<model/>' '<stop>'），不可显示 → 尖括号形态；
      [6, 9)    SYSTEM ：SYSTEM_CHARS（'{' '}' '|'）；
      [9, 2970) DRAW   ：图元输出 token，名字由 spec 的字段布局推导。

    SYSTEM 区是一个例外：spec 把 { } | 归类为「不可显示」，但它们不是合法
    标识符（尖括号内的名字必须匹配 [A-Za-z_][A-Za-z0-9_./:-]*），无法写成
    <{> 这种形态，只能落引号 → 条目的 visible=True。也就是说 **visible 标记
    只描述文件里的包裹形态，不完全等同 spec 的「可显示/不可显示」分类**。
    加载侧按 ID 落在 [SYSTEM_BEGIN, SYSTEM_BEGIN + SYSTEM_SIZE) 判定，不受
    包裹形态影响。

    头部编号与 spec 运行时 ID **是一致的**（spec 的头部区间已压缩为
    SPECIAL 0–5 / SYSTEM 6–8 / DRAW 9–2969，中间无空位）。
    """
    entries = []
    tid = 0
    for text in spec.SPECIAL_TOKENS:
        entries.append((tid, text, False))
        tid += 1
    for text in SYSTEM_CHARS:
        entries.append((tid, text, True))
        tid += 1
    for text in _draw_head_names():
        entries.append((tid, text, False))
        tid += 1
    assert len(entries) == spec.HEAD_SIZE, (
        f"vfc 头部条目数 {len(entries)} != spec.HEAD_SIZE {spec.HEAD_SIZE}"
        f"（spec 与 vfc 的契约头推导不一致）")
    return entries


def head_size() -> int:
    """架构契约头的长度（= 头部条目数，当前 6 + 3 + 2961 = 2970）。

    直接取 spec.HEAD_SIZE（spec 里由 SPECIAL_SIZE + SYSTEM_SIZE +
    MODAL_SIZE 算出）；head_entries 里的 assert 保证这两个推导不会因为
    spec 变更而悄悄分叉。
    """
    return spec.HEAD_SIZE


# ---------------------------------------------------------------------------
# 转义 / 反转义（手工实现，不用 eval / 不用 json，拒绝非法转义）
# ---------------------------------------------------------------------------
# 控制字符必须转义，否则「一行一个 token」的前提就没了（\n 会把一行劈成
# 两行）。U+2028/U+2029 虽然不是 \n，但很多编辑器和 JS 把二者当行分隔符，
# 一并转义以免跨工具搬运时静默断行。
_MUST_ESCAPE = frozenset([0x2028, 0x2029])


def _escape(text: str, quote: str) -> str:
    """把可显示 token 的文本转成引号内的字面量（不含外层引号）。"""
    out = []
    for ch in text:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == quote:
            out.append("\\" + quote)
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif o < 0x20 or o == 0x7F or o in _MUST_ESCAPE:
            out.append("\\u%04x" % o)
        else:
            out.append(ch)
    return "".join(out)


def _unescape(body: str, lineno: int, source: str) -> str:
    """反转义引号内的字面量；任何非法转义一律抛 VfcError。

    手写扫描而不用 json.loads / codecs 的原因：json 接受 \\b \\f \\/ 等本
    格式不接受的转义，且错误信息里没有行号。
    """
    out = []
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        i += 1
        if i >= n:
            raise VfcError(f"{source}:{lineno} 行尾有孤立反斜杠",
                           lineno, source)
        esc = body[i]
        i += 1
        if esc == "\\":
            out.append("\\")
        elif esc == "'":
            out.append("'")
        elif esc == '"':
            out.append('"')
        elif esc == "n":
            out.append("\n")
        elif esc == "r":
            out.append("\r")
        elif esc == "t":
            out.append("\t")
        elif esc == "u":
            hex4 = body[i:i + 4]
            if len(hex4) != 4 or any(c not in "0123456789abcdefABCDEF"
                                     for c in hex4):
                raise VfcError(
                    f"{source}:{lineno} \\u 转义后必须跟 4 位十六进制数字",
                    lineno, source)
            i += 4
            cp = int(hex4, 16)
            if 0xD800 <= cp <= 0xDBFF:
                # 高代理：必须紧跟一个 \uXXXX 低代理凑成一对
                if not body.startswith("\\u", i) or len(body) < i + 6:
                    raise VfcError(
                        f"{source}:{lineno} \\u{hex4} 是高代理项，"
                        f"后面必须紧跟 \\u 低代理项", lineno, source)
                hex4b = body[i + 2:i + 6]
                if any(c not in "0123456789abcdefABCDEF" for c in hex4b):
                    raise VfcError(
                        f"{source}:{lineno} 代理对后半部分不是 4 位十六进制数字",
                        lineno, source)
                i += 6
                low = int(hex4b, 16)
                if not (0xDC00 <= low <= 0xDFFF):
                    raise VfcError(
                        f"{source}:{lineno} \\u{hex4} 后面的 \\u{hex4b} "
                        f"不是合法低代理项", lineno, source)
                cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00)
            elif 0xDC00 <= cp <= 0xDFFF:
                raise VfcError(
                    f"{source}:{lineno} \\u{hex4} 是孤立低代理项",
                    lineno, source)
            out.append(chr(cp))
        else:
            raise VfcError(
                f"{source}:{lineno} 非法转义 \\{esc}"
                f"（只允许 \\\\ \\' \\\" \\n \\r \\t \\uXXXX）",
                lineno, source)
    return "".join(out)


# ---------------------------------------------------------------------------
# 序列化
# ---------------------------------------------------------------------------
def _pick_quote(text: str) -> str:
    """引号选择：默认双引号；含 " 但不含 ' 改用单引号（减少转义）。"""
    if '"' not in text:
        return '"'
    if "'" not in text:
        return "'"
    return '"'                      # 两者都含 → 双引号并转义内部的 "


def _dump_entry(entry) -> str:
    """单条目 → 一行文本（不含换行）。"""
    try:
        tid, text, visible = entry
    except (TypeError, ValueError):
        raise VfcError(f"条目必须是 (id, text, visible) 三元组: {entry!r}")
    if isinstance(tid, bool) or not isinstance(tid, int):
        raise VfcError(f"token ID 必须是整数: {tid!r}")
    if not (0 <= tid < MAX_VOCAB_SIZE):
        raise VfcError(
            f"token ID {tid} 越界 [0, {MAX_VOCAB_SIZE})")
    if not isinstance(text, str):
        raise VfcError(f"token 文本必须是字符串: {text!r}")
    if not text:
        raise VfcError(f"token 文本不得为空（ID {tid}）")
    # 孤立代理项无法编码成 UTF-8，必须在写出前拦下（否则 os.replace 之后
    # 留下一个半截或不可读的文件）
    if any(0xD800 <= ord(c) <= 0xDFFF for c in text):
        raise VfcError(
            f"token 文本含孤立代理项，无法编码为 UTF-8（ID {tid}）")
    if visible:
        quote = _pick_quote(text)
        return quote + _escape(text, quote) + quote + f":{tid:X}"
    # 不可显示：text 自带尖括号，原样写出后接 :ID（不二次加括号）
    if not _ANGLED_TEXT_RE.match(text):
        raise VfcError(
            f"不可显示 token 的 text 必须是 <名字> 形态（名字匹配 "
            f"[A-Za-z_][A-Za-z0-9_./:-]*）: {text!r}")
    return f"{text}:{tid:X}"


def dumps_vfc(entries) -> str:
    """[(id, text, visible), ...] → .vfc 文本（含魔数第一行，末尾换行）。

    条目按 id 升序输出（保证幂等：同一批条目无论输入顺序如何，输出的
    字节完全一致）。空条目列表只输出魔数行。
    """
    items = sorted(entries, key=lambda e: e[0])
    lines = [MAGIC]
    for e in items:
        lines.append(_dump_entry(e))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------
def _scan(text: str, source: str):
    """逐行扫描 → (内部条目 [(id, text, visible, lineno)], 行级错误 [_Err])。

    行语法错误只报该行，不中断——该行拿不到 id/文本，语义校验无从谈起，
    其余行照常校验（这样编辑器能一次看到全部问题）。
    """
    entries = []
    errors = []
    for lineno, raw in enumerate(text.split("\n"), 1):
        line = raw.rstrip("\r")                 # 宽容 CRLF
        if not line.strip():
            continue
        if line.startswith("#"):
            continue                            # 注释行 / 魔数行
        m = _TAIL_RE.match(line)
        if not m:
            if ":" in line:
                msg = ("冒号后必须紧跟十六进制数字 ID"
                       "（0-9/A-F/a-f，不允许空格、正负号或其它字符）")
            else:
                msg = "缺少 `:<ID>` 后缀（半角冒号 + 十六进制数字）"
            errors.append(_Err(lineno, f"{source}:{lineno} {msg}"))
            continue
        body, id_s = m.group(1), m.group(2)

        # —— 形态判定：先看引号（"<x>" 以引号开头，不会被误判成尖括号）——
        if len(body) >= 2 and body[0] in "\"'" and body[-1] == body[0]:
            quote = body[0]
            try:
                text_tok = _unescape(body[1:-1], lineno, source)
            except VfcError as e:
                errors.append(_Err(lineno, str(e)))
                continue
            visible = True
        elif len(body) >= 2 and body[0] == "<" and body[-1] == ">":
            name = body[1:-1]
            if not _NAME_RE.match(name):
                errors.append(_Err(
                    lineno,
                    f"{source}:{lineno} 尖括号内的名字非法: <{name}>"
                    f"（必须匹配 [A-Za-z_][A-Za-z0-9_./:-]*）"))
                continue
            # text 自带尖括号：与 spec.SPECIAL_TOKENS / vocab.json 一致
            text_tok = body
            visible = False
        elif body and body[0] in "\"'":
            errors.append(_Err(
                lineno, f"{source}:{lineno} 引号未闭合: {line}"))
            continue
        elif body and body[0] == "<":
            errors.append(_Err(
                lineno, f"{source}:{lineno} 尖括号未闭合（必须以 '>' 结尾）: "
                        f"{line}"))
            continue
        else:
            errors.append(_Err(
                lineno,
                f"{source}:{lineno} 内容必须用单引号/双引号（可显示 token）"
                f"或尖括号（不可显示标记）包裹: {line}"))
            continue

        tid = int(id_s, 16)
        if tid >= MAX_VOCAB_SIZE:
            errors.append(_Err(
                lineno,
                f"{source}:{lineno} token ID {tid} 越界 "
                f"[0, {MAX_VOCAB_SIZE})"))
            continue
        if not text_tok:
            errors.append(_Err(
                lineno, f"{source}:{lineno} token 文本不得为空"))
            continue
        entries.append((tid, text_tok, visible, lineno))
    entries.sort(key=lambda e: e[0])
    return entries, errors


def _is_system_id(tid: int) -> bool:
    """该 ID 是否落在 system 区（`{` `}` `|`）。

    system 字符与可见符号区里的同名 token 是**允许共存**的唯一例外：
    它们既是时间步结构字符（按 ID 定位，不进文本反查），又是用户文本里
    真实存在的符号（按文本查表）。没有这条豁免，基础词表根本无法用
    .vfc 表达——符号区必然收录 `{` `}` `|`。
    """
    return spec.SYSTEM_BEGIN <= tid < spec.SYSTEM_BEGIN + spec.SYSTEM_SIZE


def _check_semantics(entries) -> list:
    """行级之上的全局校验：ID 唯一、文本唯一、ID 严格连续。

    文本唯一性有一处豁免：system 区的 `{` `}` `|` 可以与可见符号区的同名
    token 共存（理由见 _is_system_id）。除此之外任何重复都拒绝——反查
    歧义会让 enctok 的编码结果依赖于文件行序。
    """
    errors = []
    seen_id = {}                                # id -> lineno
    seen_vis = {}                               # 可显示 text -> lineno
    seen_sys = {}                               # system text -> lineno
    for tid, text, _vis, lineno in entries:
        if tid in seen_id:
            errors.append(_Err(
                lineno,
                f"token ID 重复: {tid}"
                f"（第 {seen_id[tid]} 行已使用该 ID）"))
        else:
            seen_id[tid] = lineno
        bucket = seen_sys if _is_system_id(tid) else seen_vis
        if text in bucket:
            errors.append(_Err(
                lineno,
                f"token 文本重复: {text!r}"
                f"（第 {bucket[text]} 行已使用该文本，反查歧义）"))
        else:
            bucket[text] = lineno

    # ID 必须恰好构成 0..N-1：任何一个空位都白占一行 embedding（吃 RAM）。
    # N 取去重后的 ID 数——ID 重复的行数不代表真实条目数，用 len(entries)
    # 会在「有重复 ID」时凭空多报一个空位错误。
    n = len(seen_id)
    expected = set(range(n))
    actual = set(seen_id)
    for missing in sorted(expected - actual):
        errors.append(_Err(
            None, f"ID 不连续：缺少 ID {missing}"
                  f"（全部 ID 必须恰好构成 0..{n - 1}，不允许空位）"))
    for extra in sorted(actual - expected):
        errors.append(_Err(
            seen_id[extra],
            f"token ID {extra} 超出范围：共 {n} 个不同 ID，"
            f"合法 ID 为 0..{n - 1}"))
    return errors


def _check_head(entries) -> list:
    """架构契约头校验：前 head_size() 个条目的 ID、文本、形态必须逐一对上。"""
    errors = []
    expected = head_entries()
    n = min(len(entries), len(expected))
    for i in range(n):
        tid, text, visible, lineno = entries[i]
        eid, etext, evisible = expected[i]
        if tid != eid:
            errors.append(_Err(
                lineno,
                f"架构契约头第 {i + 1} 条 ID 应为 {eid}，实际为 {tid}"))
        elif text != etext or visible != evisible:
            errors.append(_Err(
                lineno,
                f"架构契约头第 {i + 1} 条应为 "
                f"{'<尖括号>' if not evisible else '<引号>'}"
                f"{etext!r}，实际为 {text!r}"))
        if len(errors) >= _MAX_HEAD_ERRORS:
            errors.append(_Err(None, "架构契约头错误过多，已截断后续报告"))
            return errors
    if len(entries) < len(expected):
        errors.append(_Err(
            None,
            f"架构契约头不完整：只有 {len(entries)} 个条目，"
            f"至少需要 {len(expected)} 个（头部条目不可省略）"))
    return errors


def _collect(text: str, source: str):
    """扫描 + 全部校验 → (条目, 错误列表)。"""
    if not isinstance(text, str):
        raise VfcError(f"{source}: 内容必须是字符串，实际为 {type(text).__name__}")
    entries, errors = _scan(text, source)
    return entries, errors + _check_semantics(entries) + _check_head(entries)


def validate_vfc(text: str, source: str = "vocab.vfc") -> list:
    """只校验不返回数据：[{"line": int|None, "message": str}]，空列表 = 合法。

    一次性收集多条错误（重复 ID / 重复文本 / ID 空位逐条报），供编辑器在
    保存前批量展示；行语法解析失败的行只报语法错误（拿不到 id/文本）。
    """
    _, errors = _collect(text, source)
    return [{"line": e.line, "message": e.message} for e in errors]


def parse_vfc(text: str, source: str = "vocab.vfc") -> list:
    """解析 .vfc 文本 → [(id, text, visible), ...]（按 id 升序）。

    任一校验失败抛 VfcError（带行号）。
    """
    entries, errors = _collect(text, source)
    if errors:
        e = errors[0]
        raise VfcError(e.message, e.line, source)
    return [(tid, text, visible) for tid, text, visible, _l in entries]


# ---------------------------------------------------------------------------
# 文件读写
# ---------------------------------------------------------------------------
def read_vfc(path: str) -> list:
    """读 .vfc 文件 → [(id, text, visible), ...]（按 id 升序）。"""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return parse_vfc(text, source=str(path))


def write_vfc(path: str, entries) -> None:
    """原子写 .vfc：唯一临时文件 + flush + fsync + os.replace。

    唯一临时文件名（uuid4）让并发写不会互相踩；fsync 保证断电后不会留一个
    内容为空的新文件（旧文件已 replace 掉、新内容还没落盘的中间态）。
    """
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(dumps_vfc(entries))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
