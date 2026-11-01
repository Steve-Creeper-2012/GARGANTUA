#!/usr/bin/env python3
# ===========================================================================
# GARGANTUA v1 — 构建全局基础词表 tokens/{vocab.vfc, config.json}
# ---------------------------------------------------------------------------
# 布局严格依照 ROOT/spec.py（冻结契约），禁止硬编码区间。
# 产物（词表文件夹契约，与 vocabs/<name>/ 完全同构）：
#   vocab.vfc    词表正文（一行一个 token，见 tokens/vfc.py）
#   config.json  vocab_size / regions 分区表 / user_added 重放源
# 总长度 = 架构契约头 + 各可见子区实际收录数（不 pad、不留空位）。
#
# 内容来源：
#   - emoji：内嵌 EMOJI_RANGES（Unicode 16.0.0 emoji-data.txt 的
#     Emoji=Yes 码点，非 ASCII；构建离线可复现）。
#   - 符号：运行时 unicodedata 的全部 P*/S* 类码点（排除 emoji 区与
#     CJK 区）+ 全角字母数字 62 个；构建结果随 UCD 版本可能微调，
#     meta 记录 unidata_version 与实际数量。
#   - 英文词：tokens/words_en.txt（bundled，hermitdave/FrequencyWords
#     词频序前 30000，单字母词除外）；fallback 链见 load_english_words。
#   - 用户新增 token：config.json 的 "user_added" 自动重放，重建不丢数据。
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

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
VFC_PATH = Path(__file__).resolve().parent / spec.VOCAB_FILE_NAME

# 大类标签（与 tokens/vocab.py 同源，这里只用来标注分区表）
CAT_REGULAR = "regular"
CAT_FALLBACK = "fallback"
CAT_SPECIAL = "special"
CAT_SYSTEM = "system"
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
    """config.json 的 "user_added" → [{text, ...}, ...]（重建重放源）。"""
    if not CONFIG_PATH.is_file():
        return []
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    got = cfg.get("user_added")
    return list(got) if isinstance(got, list) else []


# 图元 DRAW 区各字段的 bin 数（顺序与 spec.DRAW_FIELD_ORDER 一致）。
# 从 spec 取，不硬编码——spec 调整 bin 数时这里自动跟随。
_DRAW_FIELD_SIZES = (
    spec.DRAW_X_BINS, spec.DRAW_Y_BINS, spec.DRAW_ANGLE_BINS,
    spec.DRAW_W_BINS, spec.DRAW_H_BINS, spec.DRAW_SHAPE_BINS,
    spec.DRAW_COLOR_BINS, spec.DRAW_COLOR_BINS, spec.DRAW_COLOR_BINS,
)


def build_entries(english_words, user_added: list | None = None):
    """按 spec 布局拼装 id_to_token 列表（**长度随内容变化，不再 pad**）。

    2026-10-30 契约重构：删掉所有 `pad_to(...)` 预留。可见区各子区**连续**
    排布，长度就是它实际收录的条数，一个空位都不留——空位同样要占一个
    embedding 行（wte 与 lm_head 各一份 1024 维），白吃 RAM，还会在
    logits 里留下一堆永远不该被采样到的死行。
    于是**每个词表的总长度可以不同**，真值记在词表目录的 config.json。

    返回 (entries, regions, applied)：
      entries  [(id, text, visible)]  连续编号，可直接落 .vfc
      regions  [{region, category, size}]  各子区**大小**（begin 由累加得出）
      applied  实际生效的 user_added 条目
    """
    entries: list[tuple] = []
    tokens: list[str] = []
    regions: list[dict] = []

    def emit(seq, region: str, category: str):
        for _t in seq:
            entries.append((len(entries), _t, category != CAT_SPECIAL))
            tokens.append(_t)
        regions.append({"region": region, "category": category,
                        "size": len(seq)})

    # ---- 架构契约头：special / system / draw（内容与顺序由 spec 冻结）----
    emit(list(spec.SPECIAL_TOKENS), "special", CAT_SPECIAL)
    emit(list(getattr(spec, "SYSTEM_CHARS",
                      (spec.TS_OPEN, spec.TS_CLOSE, spec.TS_SEP))),
         "system", CAT_SYSTEM)
    draw: list[str] = []
    for _f, _sz in zip(spec.DRAW_FIELD_ORDER, _DRAW_FIELD_SIZES):
        draw.extend(f"<{_f}:{n}>" for n in range(_sz))
    draw.extend(f"<draw:special:{n}>"
                for n in range(spec.DRAW_SPECIAL_BINS))
    emit(draw, "draw", CAT_SPECIAL)
    assert len(tokens) == spec.HEAD_SIZE, (
        f"架构契约头长度 {len(tokens)} != HEAD_SIZE {spec.HEAD_SIZE}")

    # ---- 可见区：各子区按 spec.VISIBLE_REGION_ORDER 连续排布 ----
    # 用户新增 token 统一落进 extra 区（config["append_region"]）。
    # 先按「已收录文本」去重：user_added 里若混进了基础词表本来就有的词
    # （历史上 meta 被写脏过一次），直接重放会造出同文本双 ID，整张词表
    # 会被 Vocab 的唯一性校验拒绝。
    known = {t for _i, t, _v in entries}
    extra = []
    dropped = 0
    for e in (user_added or []):
        if not isinstance(e, dict):
            continue
        t = e.get("text")
        if not isinstance(t, str) or not t or t in known:
            dropped += 1
            continue
        known.add(t)
        extra.append(t)
    emit([str(d) for d in range(10)], "digit", CAT_REGULAR)
    emit(list(spec.WHITESPACE_CHARS), "whitespace", CAT_REGULAR)
    emit(symbol_chars(), "symbol", CAT_REGULAR)
    emit(emoji_chars(), "emoji", CAT_REGULAR)
    emit([chr(cp) for cp in range(0x4E00, 0x9FFF + 1)] +
         [chr(cp) for cp in range(0x3400, 0x4DBF + 1)],
         "cjk", CAT_REGULAR)
    emit([str(w) for w in english_words], "english", CAT_REGULAR)
    emit(extra, "extra", CAT_REGULAR)
    emit(list(spec.FALLBACK_LETTERS), "fallback", CAT_FALLBACK)

    if len(entries) > spec.MAX_VOCAB_SIZE:
        raise ValueError(
            f"词表长度 {len(entries)} 超过安全上界 {spec.MAX_VOCAB_SIZE}")

    # applied 只能取 extra 区里的条目。旧实现用 `i >= sum(regions[:7])`
    # 当边界，而 regions[:7] 到 emoji 为止——cjk/english 全被误记成
    # 「用户新增」，下次重建就会把它们当 extra 重放一遍（文本双 ID）。
    extra_begin = 0
    extra_size = 0
    for r in regions:
        if r["region"] == "extra":
            extra_size = r["size"]
            break
        extra_begin += r["size"]
    applied = [{"text": t, "category": "regular", "id": i}
               for i, t, _v in entries[extra_begin:extra_begin + extra_size]]
    if dropped:
        print(f"[build_vocab] user_added 中 {dropped} 条与基础词表重复，"
              f"已跳过", file=sys.stderr)
    return entries, regions, applied


def build_vocab(words_file: str | None = None,
                source_desc: str | None = None) -> dict:
    """完整构建：entries + regions + config.json 内容。"""
    english_words, source = load_english_words(words_file, source_desc)
    user_added = load_user_added()
    entries, regions, applied = build_entries(english_words, user_added)
    id_to_token = [t for _i, t, _v in entries]
    # 同一字符串在全词表内唯一，唯一例外是 3 个系统 token 的显示字符
    # （{ } |）与可见符号区撞名——设计使然：text→id 经 regular 查表
    # 永远解析到可见符号 token；id→显示时两者同为字面量，无歧义。
    seen: dict = {}
    for i, t in enumerate(id_to_token):
        if t in seen and t not in (spec.TS_OPEN, spec.TS_CLOSE, spec.TS_SEP):
            raise AssertionError(
                f"词表存在同文本双 ID: {t!r}（{seen[t]} 与 {i}）")
        seen.setdefault(t, i)
    config = {
        "name": "base",
        "format": "gargantua-vocab-folder-v1",
        "is_main": True,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vocab_size": len(id_to_token),
        "head_size": spec.HEAD_SIZE,
        "regions": regions,
        "append_region": "extra",
        "user_added": applied,
        "sources": {
            "english": source,
            "english_word_count": len(english_words),
            "emoji": ("Unicode 16.0.0 emoji-data.txt (Emoji=Yes, 非 ASCII)"
                      " + Emoji_Component 组件字符"),
            "emoji_count": len(emoji_chars()),
            "symbol": (f"Unicode P*/S* 类码点（UCD "
                       f"{unicodedata.unidata_version}）排除 emoji + "
                       f"全角字母数字"),
            "symbol_count": len(symbol_chars()),
            "unidata_version": unicodedata.unidata_version,
        },
    }
    return {"entries": entries, "regions": regions, "config": config}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="构建全局基础词表 tokens/{vocab.vfc, config.json}")
    ap.add_argument("--words-file", default=None, help="本地英文词频文件路径")
    ap.add_argument(
        "--source-desc",
        default=None,
        help="config 中记录的英文词来源描述（默认按实际获取途径记录）",
    )
    args = ap.parse_args(argv)
    vocab = build_vocab(args.words_file, args.source_desc)

    from tokens import vfc as _vfc
    # 直接用 build_entries 给出的 (id, text, visible)：visible 标记决定
    # .vfc 里是引号还是尖括号形态，system 的 { } | 必须是引号（尖括号
    # 里放不下非标识符），不能按「ID < HEAD_SIZE」重新推导。
    _vfc.write_vfc(str(VFC_PATH), vocab["entries"])
    cfg = vocab["config"]
    CONFIG_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    src = cfg["sources"]
    print(f"已写出 {VFC_PATH}")
    print(f"已写出 {CONFIG_PATH}")
    print(f"  词表长度      : {cfg['vocab_size']}")
    print(f"  分区数        : {len(cfg['regions'])}")
    print(f"  英文词实收    : {src['english_word_count']}（{src['english']}）")
    print(f"  emoji         : {src['emoji_count']}")
    print(f"  符号          : {src['symbol_count']}")
    print(f"  用户新增重放  : {len(cfg['user_added'])}")


if __name__ == "__main__":
    main()
