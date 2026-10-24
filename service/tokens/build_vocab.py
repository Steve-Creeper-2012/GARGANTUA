#!/usr/bin/env python3
# ===========================================================================
# GARGANTUA v1 — 构建 tokens/vocab.json（tok-32 四类词表）
# ---------------------------------------------------------------------------
# 布局严格依照 ROOT/spec.py（冻结契约），禁止硬编码区间。
# 生成结构：{"id_to_token": [...], "token_to_id": {...}, "meta": {...}}
# 总长度 = spec.VOCAB_SIZE_PADDED。
#
# 内容来源：
#   - emoji：内嵌 EMOJI_RANGES（Unicode 16.0.0 emoji-data.txt 的
#     Emoji=Yes 码点，非 ASCII；构建离线可复现）。
#   - 符号：运行时 unicodedata 的全部 P*/S* 类码点（排除 emoji 区与
#     CJK 区）+ 全角字母数字 62 个；构建结果随 UCD 版本可能微调，
#     meta 记录 unidata_version 与实际数量。
#   - 英文词：tokens/words_en.txt（bundled，hermitdave/FrequencyWords
#     词频序前 30000，单字母词除外）；fallback 链见 load_english_words。
#   - 用户新增 token：旧 vocab.json 的 meta["user_added"] 自动重放
#     （原 ID 可用则原样占用，否则按类别重新分配），重建不丢数据。
#
# 用法：
#   cd ROOT && python tokens/build_vocab.py [--words-file PATH]
# ===========================================================================
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---- GARGANTUA 路径引导（目录树重构：把 server/ 与 service/* 放上 path）----
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "start" / "boot.py").is_file():
        _sys.path.insert(0, str(_anc / "start"))
        break
else:
    raise SystemExit("找不到 GARGANTUA 项目根（start/boot.py）")
import boot
boot.bootstrap()
import spec  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "vocab.json"
WORDS_PATH = Path(__file__).resolve().parent / "words_en.txt"

EN_FREQ_URL = (
    "https://raw.githubusercontent.com/hermitdave/FrequencyWords/"
    "master/content/2016/en/en_full.txt"
)
EN_FALLBACK_URL = (
    "https://raw.githubusercontent.com/first20hours/"
    "google-10000-english/master/google-10000-english.txt"
)
# 英文词形：全小写 [a-z]+，允许一个撇号后缀（don't / it's 等原形收录）
EN_WORD_RE = re.compile(r"^[a-z]+(?:'[a-z]+)?$")

# Unicode 16.0.0 emoji-data.txt 中 Emoji=Yes 的码点（已排除 ASCII 键帽
# 基字符 # * 0–9——它们属于 SYMBOL/DIGIT 区），压缩为闭区间列表。
EMOJI_RANGES = [
    (0xA9, 0xA9), (0xAE, 0xAE), (0x203C, 0x203C), (0x2049, 0x2049), (0x2122, 0x2122), (0x2139, 0x2139), (0x2194, 0x2199), (0x21A9, 0x21AA), (0x231A, 0x231B), (0x2328, 0x2328), (0x23CF, 0x23CF), (0x23E9, 0x23F3), (0x23F8, 0x23FA), (0x24C2, 0x24C2), (0x25AA, 0x25AB), (0x25B6, 0x25B6), (0x25C0, 0x25C0), (0x25FB, 0x25FE), (0x2600, 0x2604), (0x260E, 0x260E), (0x2611, 0x2611), (0x2614, 0x2615), (0x2618, 0x2618), (0x261D, 0x261D), (0x2620, 0x2620), (0x2622, 0x2623), (0x2626, 0x2626), (0x262A, 0x262A), (0x262E, 0x262F), (0x2638, 0x263A), (0x2640, 0x2640), (0x2642, 0x2642), (0x2648, 0x2653), (0x265F, 0x2660), (0x2663, 0x2663), (0x2665, 0x2666), (0x2668, 0x2668), (0x267B, 0x267B), (0x267E, 0x267F), (0x2692, 0x2697), (0x2699, 0x2699), (0x269B, 0x269C), (0x26A0, 0x26A1), (0x26A7, 0x26A7), (0x26AA, 0x26AB), (0x26B0, 0x26B1), (0x26BD, 0x26BE), (0x26C4, 0x26C5), (0x26C8, 0x26C8), (0x26CE, 0x26CF), (0x26D1, 0x26D1), (0x26D3, 0x26D4), (0x26E9, 0x26EA), (0x26F0, 0x26F5), (0x26F7, 0x26FA), (0x26FD, 0x26FD), (0x2702, 0x2702), (0x2705, 0x2705), (0x2708, 0x270D), (0x270F, 0x270F), (0x2712, 0x2712), (0x2714, 0x2714), (0x2716, 0x2716), (0x271D, 0x271D), (0x2721, 0x2721), (0x2728, 0x2728), (0x2733, 0x2734), (0x2744, 0x2744), (0x2747, 0x2747), (0x274C, 0x274C), (0x274E, 0x274E), (0x2753, 0x2755), (0x2757, 0x2757), (0x2763, 0x2764), (0x2795, 0x2797), (0x27A1, 0x27A1), (0x27B0, 0x27B0), (0x27BF, 0x27BF), (0x2934, 0x2935), (0x2B05, 0x2B07), (0x2B1B, 0x2B1C), (0x2B50, 0x2B50), (0x2B55, 0x2B55), (0x3030, 0x3030), (0x303D, 0x303D), (0x3297, 0x3297), (0x3299, 0x3299), (0x1F004, 0x1F004), (0x1F0CF, 0x1F0CF), (0x1F170, 0x1F171), (0x1F17E, 0x1F17F), (0x1F18E, 0x1F18E), (0x1F191, 0x1F19A), (0x1F1E6, 0x1F1FF), (0x1F201, 0x1F202), (0x1F21A, 0x1F21A), (0x1F22F, 0x1F22F), (0x1F232, 0x1F23A), (0x1F250, 0x1F251), (0x1F300, 0x1F321), (0x1F324, 0x1F393), (0x1F396, 0x1F397), (0x1F399, 0x1F39B), (0x1F39E, 0x1F3F0), (0x1F3F3, 0x1F3F5), (0x1F3F7, 0x1F4FD), (0x1F4FF, 0x1F53D), (0x1F549, 0x1F54E), (0x1F550, 0x1F567), (0x1F56F, 0x1F570), (0x1F573, 0x1F57A), (0x1F587, 0x1F587), (0x1F58A, 0x1F58D), (0x1F590, 0x1F590), (0x1F595, 0x1F596), (0x1F5A4, 0x1F5A5), (0x1F5A8, 0x1F5A8), (0x1F5B1, 0x1F5B2), (0x1F5BC, 0x1F5BC), (0x1F5C2, 0x1F5C4), (0x1F5D1, 0x1F5D3), (0x1F5DC, 0x1F5DE), (0x1F5E1, 0x1F5E1), (0x1F5E3, 0x1F5E3), (0x1F5E8, 0x1F5E8), (0x1F5EF, 0x1F5EF), (0x1F5F3, 0x1F5F3), (0x1F5FA, 0x1F64F), (0x1F680, 0x1F6C5), (0x1F6CB, 0x1F6D2), (0x1F6D5, 0x1F6D7), (0x1F6DC, 0x1F6E5), (0x1F6E9, 0x1F6E9), (0x1F6EB, 0x1F6EC), (0x1F6F0, 0x1F6F0), (0x1F6F3, 0x1F6FC), (0x1F7E0, 0x1F7EB), (0x1F7F0, 0x1F7F0), (0x1F90C, 0x1F93A), (0x1F93C, 0x1F945), (0x1F947, 0x1F9FF), (0x1FA70, 0x1FA7C), (0x1FA80, 0x1FA89), (0x1FA8F, 0x1FAC6), (0x1FACE, 0x1FADC), (0x1FADF, 0x1FAE9), (0x1FAF0, 0x1FAF8),
]

# emoji 组件字符（emoji-data.txt 的 Emoji_Component 属性中不属于 Emoji
# 属性、且未被其他区收录的码点）：ZWJ / VS16 / 分区旗帜标签（E0020–E007F）。
# ASCII 键帽基字符（# * 0–9）已在 SYMBOL/DIGIT 区；肤色修饰与区域指示符
# 已在 Emoji 属性内。不收录这些组件会导致 ZWJ 序列 / 分区旗帜有损。
EMOJI_COMPONENT_EXTRA = [0x200D, 0xFE0F] + list(range(0xE0020, 0xE0080))

# 全角字母数字（U+FF10–FF19 / U+FF21–FF3A / U+FF41–FF5A）：不属于 P*/S*
# 类，但为保证全角文本无损 roundtrip 显式收入 SYMBOL 区。
FULLWIDTH_ALNUM = (
    list(range(0xFF10, 0xFF1A))
    + list(range(0xFF21, 0xFF3B))
    + list(range(0xFF41, 0xFF5B))
)


def _reserved(i: int) -> str:
    return f"<reserved:{i}>"


# ---------------------------------------------------------------------------
# emoji / 符号 码点集
# ---------------------------------------------------------------------------
def emoji_chars() -> list:
    """内嵌 emoji 区间 + 组件字符 → 字符列表（按码点升序）。"""
    cps = {cp for a, b in EMOJI_RANGES for cp in range(a, b + 1)}
    cps.update(EMOJI_COMPONENT_EXTRA)
    return [chr(cp) for cp in sorted(cps)]


def symbol_chars() -> list:
    """全部 Unicode P*/S* 类码点（排除 emoji 区）+ 全角字母数字。"""
    emoji = {ord(c) for c in emoji_chars()}
    out = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:          # 代理区，无字符
            continue
        if cp in emoji:
            continue
        if unicodedata.category(chr(cp))[0] in ("P", "S"):
            out.append(chr(cp))
    out.extend(chr(cp) for cp in FULLWIDTH_ALNUM)
    return sorted(set(out), key=ord)


# ---------------------------------------------------------------------------
# 英文词加载（bundled 文件优先，fallback 链与旧版一致）
# ---------------------------------------------------------------------------
def _iter_words(lines):
    """按出现顺序产出合法去重英文词（词频表每行 'word freq' 或纯词表）。"""
    seen = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        w = line.split()[0].lower()
        if EN_WORD_RE.match(w) and len(w) > 1 and w not in seen:
            # 单字母词（a / i）不收录：由 fallback 字母区承担，
            # 保证同一字符串在全词表内唯一
            seen.add(w)
            yield w


def _take(gen, limit: int):
    out = []
    for w in gen:
        out.append(w)
        if len(out) >= limit:
            break
    return out


ENGLISH_TARGET = 30000        # 用户规格：3 万个常用英文单词


def load_english_words(words_file: str | None = None,
                       source_desc: str | None = None):
    """返回 (words, source_desc)。按 fallback 链依次尝试。"""
    errors = []
    limit = ENGLISH_TARGET

    # (a) 显式指定的本地文件 / bundled 词表
    local = words_file or os.environ.get("GARGANTUA_WORDS_FILE")
    candidates = ([Path(local)] if local else []) + [WORDS_PATH]
    for p in candidates:
        if p.is_file():
            words = _take(
                _iter_words(
                    p.read_text(encoding="utf-8", errors="replace").splitlines()
                ),
                limit,
            )
            if words:
                return words, source_desc or f"file:{p}"
            errors.append(f"local file {p} yielded no valid words")
        else:
            errors.append(f"local file not found: {p}")

    # (b) hermitdave/FrequencyWords（在线）
    try:
        with urllib.request.urlopen(EN_FREQ_URL, timeout=120) as r:
            text = r.read().decode("utf-8", "replace")
        words = _take(_iter_words(text.splitlines()), limit)
        if words:
            return words, EN_FREQ_URL
        errors.append(f"{EN_FREQ_URL}: no valid words parsed")
    except Exception as e:  # noqa: BLE001
        errors.append(f"{EN_FREQ_URL}: {e!r}")

    # (c) wordfreq 包
    try:
        from wordfreq import top_n_list  # type: ignore

        words = _take(iter(top_n_list("en", 60000)), limit)
        if words:
            return words, "wordfreq:top_n_list('en', 60000)"
        errors.append("wordfreq: no valid words after filtering")
    except Exception as e:  # noqa: BLE001
        errors.append(f"wordfreq: {e!r}")

    # (d) google-10000-english
    try:
        with urllib.request.urlopen(EN_FALLBACK_URL, timeout=120) as r:
            text = r.read().decode("utf-8", "replace")
        words = _take(_iter_words(text.splitlines()), limit)
        if words:
            return words, EN_FALLBACK_URL
        errors.append(f"{EN_FALLBACK_URL}: no valid words parsed")
    except Exception as e:  # noqa: BLE001
        errors.append(f"{EN_FALLBACK_URL}: {e!r}")

    raise RuntimeError("所有英文词源均不可用：\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------------------
# 用户新增 token 重放（重建不丢失）
# ---------------------------------------------------------------------------
def load_user_added() -> list:
    """旧 vocab.json 的 meta["user_added"] → [{text, category, id}, ...]。"""
    if not OUT_PATH.is_file():
        return []
    try:
        meta = json.loads(OUT_PATH.read_text(encoding="utf-8")).get("meta", {})
    except (ValueError, OSError):
        return []
    return list(meta.get("user_added", []))


def apply_user_added(tokens: list, entries: list) -> list:
    """把用户新增 token 放回新构建的词表；返回最终生效的条目列表。"""
    applied = []
    alloc = {
        "regular": (spec.EXTRA_BEGIN, spec.EXTRA_BEGIN + spec.EXTRA_SIZE),
        "fallback": (spec.FALLBACK_BEGIN + len(spec.FALLBACK_LETTERS),
                     spec.FALLBACK_BEGIN + spec.FALLBACK_SIZE),
        "special": (spec.SPECIAL_BEGIN + len(spec.SPECIAL_TOKENS),
                    spec.SPECIAL_BEGIN + spec.SPECIAL_SIZE),
        "system": (spec.SYSTEM_BEGIN + 3,
                   spec.SYSTEM_BEGIN + spec.SYSTEM_SIZE),
    }
    for e in entries:
        text, cat = e.get("text"), e.get("category")
        if cat not in alloc or not isinstance(text, str) or not text:
            print(f"警告：跳过无效 user_added 条目: {e!r}", file=sys.stderr)
            continue
        if text in tokens:
            print(f"警告：user_added {text!r} 与基础词表撞名，跳过",
                  file=sys.stderr)
            continue
        old_id = e.get("id")
        begin, end = alloc[cat]
        slot = None
        if isinstance(old_id, int) and begin <= old_id < end \
                and tokens[old_id].startswith("<reserved:"):
            slot = old_id
        else:
            slot = next((i for i in range(begin, end)
                         if tokens[i].startswith("<reserved:")), None)
        if slot is None:
            print(f"警告：{cat} 类预留槽位已满，{text!r} 无法重放",
                  file=sys.stderr)
            continue
        tokens[slot] = text
        applied.append({"text": text, "category": cat, "id": slot})
    return applied


# ---------------------------------------------------------------------------
# 词表拼装（纯函数，便于幂等性测试）
# ---------------------------------------------------------------------------
def build_tokens(english_words, user_added: list | None = None,
                 with_applied: bool = False):
    """按 spec 布局拼装 id_to_token 列表（长度 = VOCAB_SIZE_PADDED）。

    with_applied=True 时返回 (tokens, applied_user_added)，供 meta 记录。
    """
    tokens: list[str] = []

    def pad_to(end: int):
        while len(tokens) < end:
            tokens.append(_reserved(len(tokens)))

    # 1. SPECIAL 区（0–63）：7 个特殊 token + 预留
    assert len(tokens) == spec.SPECIAL_BEGIN
    tokens.extend(spec.SPECIAL_TOKENS)
    pad_to(spec.SPECIAL_BEGIN + spec.SPECIAL_SIZE)

    # 2. SYSTEM 区（64–95）：{ } | + 预留
    assert len(tokens) == spec.SYSTEM_BEGIN
    tokens.extend([spec.TS_OPEN, spec.TS_CLOSE, spec.TS_SEP])
    pad_to(spec.SYSTEM_BEGIN + spec.SYSTEM_SIZE)

    # 3. MODAL 区：图元输出 token（照搬 primitive-operation-painter
    #    geometrize_256_v1：字段序 cx cy theta w h shape r g b + special 3）
    assert len(tokens) == spec.MODAL_BEGIN
    b = spec.DRAW_BEGIN
    fields = (
        ("cx", spec.DRAW_X_BINS), ("cy", spec.DRAW_Y_BINS),
        ("theta", spec.DRAW_ANGLE_BINS), ("w", spec.DRAW_W_BINS),
        ("h", spec.DRAW_H_BINS), ("shape", spec.DRAW_SHAPE_BINS),
        ("r", spec.DRAW_COLOR_BINS), ("g", spec.DRAW_COLOR_BINS),
        ("b", spec.DRAW_COLOR_BINS),
    )
    offsets = (spec.DRAW_X_OFFSET, spec.DRAW_Y_OFFSET,
               spec.DRAW_ANGLE_OFFSET, spec.DRAW_W_OFFSET,
               spec.DRAW_H_OFFSET, spec.DRAW_SHAPE_OFFSET,
               spec.DRAW_R_OFFSET, spec.DRAW_G_OFFSET,
               spec.DRAW_B_OFFSET)
    for (name, size), off in zip(fields, offsets):
        assert len(tokens) == b + off, f"draw {name}: {len(tokens)} != {b + off}"
        tokens.extend(f"<{name}:{n}>" for n in range(size))
    assert len(tokens) == b + spec.DRAW_SPECIAL_OFFSET
    # special 3 bins（仓库布局自带：pad + 2 预留）
    tokens.extend(f"<draw:special:{n}>" for n in range(spec.DRAW_SPECIAL_BINS))
    assert len(tokens) == b + spec.DRAW_VOCAB_SIZE
    pad_to(spec.MODAL_BEGIN + spec.MODAL_SIZE)   # 剩余预留

    # 4. DIGIT 区
    assert len(tokens) == spec.DIGIT_BEGIN
    tokens.extend(str(d) for d in range(10))
    assert len(tokens) == spec.DIGIT_BEGIN + spec.DIGIT_SIZE

    # 5. WHITESPACE 区
    assert len(tokens) == spec.WHITESPACE_BEGIN
    tokens.extend(spec.WHITESPACE_CHARS)
    pad_to(spec.WHITESPACE_BEGIN + spec.WHITESPACE_SIZE)

    # 6. SYMBOL 区
    assert len(tokens) == spec.SYMBOL_BEGIN
    symbols = symbol_chars()
    assert len(symbols) <= spec.SYMBOL_SIZE, (
        f"符号数 {len(symbols)} 超出 SYMBOL_SIZE {spec.SYMBOL_SIZE}")
    tokens.extend(symbols)
    pad_to(spec.SYMBOL_BEGIN + spec.SYMBOL_SIZE)

    # 7. EMOJI 区
    assert len(tokens) == spec.EMOJI_BEGIN
    emojis = emoji_chars()
    assert len(emojis) <= spec.EMOJI_SIZE, (
        f"emoji 数 {len(emojis)} 超出 EMOJI_SIZE {spec.EMOJI_SIZE}")
    tokens.extend(emojis)
    pad_to(spec.EMOJI_BEGIN + spec.EMOJI_SIZE)

    # 8. CJK 区：U+4E00–U+9FFF 接 U+3400–U+4DBF（各按码点升序）
    assert len(tokens) == spec.CJK_BEGIN
    tokens.extend(chr(cp) for cp in range(0x4E00, 0x9FFF + 1))
    tokens.extend(chr(cp) for cp in range(0x3400, 0x4DBF + 1))
    pad_to(spec.CJK_BEGIN + spec.CJK_SIZE)

    # 9. ENGLISH 区
    assert len(tokens) == spec.ENGLISH_BEGIN
    assert len(english_words) <= spec.ENGLISH_SIZE
    for w in english_words:
        assert EN_WORD_RE.match(w), f"非法英文词形: {w!r}"
    tokens.extend(english_words)
    pad_to(spec.ENGLISH_BEGIN + spec.ENGLISH_SIZE)

    # 10. EXTRA 区（用户新增常规 token 的 reserved 尾段）
    assert len(tokens) == spec.EXTRA_BEGIN
    pad_to(spec.EXTRA_BEGIN + spec.EXTRA_SIZE)

    # 11. FALLBACK 区：a–z + 预留
    assert len(tokens) == spec.FALLBACK_BEGIN
    tokens.extend(spec.FALLBACK_LETTERS)
    pad_to(spec.FALLBACK_BEGIN + spec.FALLBACK_SIZE)

    # 12. 尾部补齐到 64 的倍数
    assert len(tokens) == spec.VOCAB_SIZE
    pad_to(spec.VOCAB_SIZE_PADDED)
    assert len(tokens) == spec.VOCAB_SIZE_PADDED

    # 13. 用户新增 token 重放（整个基础词表构建完成后，只写 reserved 槽）
    applied = apply_user_added(tokens, user_added or [])
    if with_applied:
        return tokens, applied
    return tokens


def build_vocab(words_file: str | None = None,
                source_desc: str | None = None) -> dict:
    """完整构建（含英文词加载、用户 token 重放与 meta）。"""
    english_words, source = load_english_words(words_file, source_desc)
    user_added = load_user_added()
    id_to_token, applied = build_tokens(english_words, user_added,
                                        with_applied=True)
    # 同一字符串在全词表内唯一，唯一例外是 3 个系统 token 的显示字符
    # （{ } |）与可见符号区撞名——设计使然：text→id 经 regular 查表
    # 永远解析到可见符号 token；id→显示时两者同为字面量，无歧义。
    token_to_id = {t: i for i, t in enumerate(id_to_token)}
    seen: dict = {}
    for i, t in enumerate(id_to_token):
        if t in seen and t not in (spec.TS_OPEN, spec.TS_CLOSE, spec.TS_SEP):
            raise AssertionError(
                f"词表存在同文本双 ID: {t!r}（{seen[t]} 与 {i}）")
        seen.setdefault(t, i)
    meta = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "format": "tok-32",
        "id_bits": 32,
        "english_source": source,
        "english_word_count": len(english_words),
        "emoji_source": ("Unicode 16.0.0 emoji-data.txt (Emoji=Yes, 非 ASCII)"
                         " + Emoji_Component 组件字符"),
        "emoji_count": len(emoji_chars()),
        "symbol_source": (f"Unicode P*/S* 类码点（UCD "
                          f"{unicodedata.unidata_version}）排除 emoji + "
                          f"全角字母数字"),
        "symbol_count": len(symbol_chars()),
        "unidata_version": unicodedata.unidata_version,
        "vocab_size": spec.VOCAB_SIZE,
        "vocab_size_padded": spec.VOCAB_SIZE_PADDED,
        "user_added": applied,
        "categories": {
            "regular": "常规可见（数字/空白/符号/emoji/中文/英文词）",
            "fallback": "可见 fallback（a–z）",
            "special": "不可见特殊（<caps>/<user>/<model>/<stop> 等）",
            "system": "不可见系统（{ } | 时间步结构）",
        },
        "ranges": {
            "special": {"begin": spec.SPECIAL_BEGIN, "size": spec.SPECIAL_SIZE},
            "system": {"begin": spec.SYSTEM_BEGIN, "size": spec.SYSTEM_SIZE},
            "modal": {"begin": spec.MODAL_BEGIN, "size": spec.MODAL_SIZE},
            "digit": {"begin": spec.DIGIT_BEGIN, "size": spec.DIGIT_SIZE},
            "whitespace": {"begin": spec.WHITESPACE_BEGIN,
                           "size": spec.WHITESPACE_SIZE},
            "symbol": {"begin": spec.SYMBOL_BEGIN, "size": spec.SYMBOL_SIZE},
            "emoji": {"begin": spec.EMOJI_BEGIN, "size": spec.EMOJI_SIZE},
            "cjk": {"begin": spec.CJK_BEGIN, "size": spec.CJK_SIZE},
            "english": {"begin": spec.ENGLISH_BEGIN, "size": spec.ENGLISH_SIZE},
            "extra": {"begin": spec.EXTRA_BEGIN, "size": spec.EXTRA_SIZE},
            "fallback": {"begin": spec.FALLBACK_BEGIN,
                         "size": spec.FALLBACK_SIZE},
            "draw": {"begin": spec.DRAW_BEGIN, "size": spec.DRAW_VOCAB_SIZE,
                     "tokens_per_primitive": spec.DRAW_TOKENS_PER_PRIMITIVE,
                     "field_order": list(spec.DRAW_FIELD_ORDER),
                     "bins": {"cx": spec.DRAW_X_BINS, "cy": spec.DRAW_Y_BINS,
                              "theta": spec.DRAW_ANGLE_BINS,
                              "w": spec.DRAW_W_BINS, "h": spec.DRAW_H_BINS,
                              "shape": spec.DRAW_SHAPE_BINS,
                              "r": spec.DRAW_COLOR_BINS,
                              "g": spec.DRAW_COLOR_BINS,
                              "b": spec.DRAW_COLOR_BINS,
                              "special": spec.DRAW_SPECIAL_BINS},
                     "source": ("wonderfulearth/primitive-operation-painter "
                                "geometrize_256_v1")},
        },
    }
    return {
        "id_to_token": id_to_token,
        "token_to_id": token_to_id,
        "meta": meta,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="构建 tokens/vocab.json")
    ap.add_argument("--words-file", default=None, help="本地英文词频文件路径")
    ap.add_argument(
        "--source-desc",
        default=None,
        help="meta 中记录的英文词来源描述（默认按实际获取途径记录）",
    )
    ap.add_argument(
        "--export-txt",
        default=None,
        help="顺便把词表导出为 vocab.txt（一行一个 token，行号即 ID；"
             "可直接当作 models/NAME/vocab.txt 的起点）",
    )
    args = ap.parse_args(argv)
    vocab = build_vocab(args.words_file, args.source_desc)
    OUT_PATH.write_text(
        json.dumps(vocab, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    if args.export_txt:
        out = Path(args.export_txt)
        out.write_text(
            "\n".join(vocab["id_to_token"]) + "\n", encoding="utf-8")
        print(f"  导出 vocab.txt : {out}（{len(vocab['id_to_token'])} 行）")
    m = vocab["meta"]
    print(f"已写出 {OUT_PATH}")
    print(f"  总槽位        : {len(vocab['id_to_token'])}")
    print(f"  英文词实收    : {m['english_word_count']}（{m['english_source']}）")
    print(f"  emoji         : {m['emoji_count']}")
    print(f"  符号          : {m['symbol_count']}")
    print(f"  用户新增重放  : {len(m['user_added'])}")
    print(f"  文件大小      : {OUT_PATH.stat().st_size} 字节")


if __name__ == "__main__":
    main()
