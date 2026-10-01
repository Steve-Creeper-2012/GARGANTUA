#!/usr/bin/env python3
"""构建 tokens/vocab.json —— GARGANTUA v1 词表生成脚本。

布局严格依照 ROOT/spec.py（冻结契约），禁止硬编码区间。
生成结构：{"id_to_token": [...], "token_to_id": {...}, "meta": {...}}
总长度 = spec.VOCAB_SIZE_PADDED（78656）。

用法：
    cd ROOT && python tokens/build_vocab.py [--words-file PATH]

英文词源 fallback 链：
    a. --words-file / 环境变量 GARGANTUA_WORDS_FILE 指定的本地词频文件
    b. hermitdave/FrequencyWords en_full.txt（在线）
    c. wordfreq 包 top_n_list('en', 60000)
    d. first20hours/google-10000-english（在线，凑不满时剩余槽位 reserved）
"""
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

import spec  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / "vocab.json"

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


def _reserved(i: int) -> str:
    return f"<reserved:{i}>"


# ---------------------------------------------------------------------------
# 英文词加载（fallback 链）
# ---------------------------------------------------------------------------
def _iter_words(lines):
    """按出现顺序产出合法去重英文词（词频表每行 'word freq' 或纯词表）。"""
    seen = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        w = line.split()[0].lower()
        if EN_WORD_RE.match(w) and w not in seen:
            seen.add(w)
            yield w


def _take(gen, limit: int):
    out = []
    for w in gen:
        out.append(w)
        if len(out) >= limit:
            break
    return out


def load_english_words(words_file: str | None = None, source_desc: str | None = None):
    """返回 (words, source_desc)。按 fallback 链依次尝试。"""
    errors = []

    # (a) 本地文件（CLI 参数或环境变量）
    local = words_file or os.environ.get("GARGANTUA_WORDS_FILE")
    if local:
        p = Path(local)
        if p.is_file():
            words = _take(
                _iter_words(
                    p.read_text(encoding="utf-8", errors="replace").splitlines()
                ),
                spec.ENGLISH_SIZE,
            )
            if words:
                return words, source_desc or f"file:{p}"
            errors.append(f"local file {p} yielded no valid words")
        else:
            errors.append(f"local file not found: {p}")

    # (b) hermitdave/FrequencyWords
    try:
        with urllib.request.urlopen(EN_FREQ_URL, timeout=120) as r:
            text = r.read().decode("utf-8", "replace")
        words = _take(_iter_words(text.splitlines()), spec.ENGLISH_SIZE)
        if words:
            return words, EN_FREQ_URL
        errors.append(f"{EN_FREQ_URL}: no valid words parsed")
    except Exception as e:  # noqa: BLE001
        errors.append(f"{EN_FREQ_URL}: {e!r}")

    # (c) wordfreq 包
    try:
        from wordfreq import top_n_list  # type: ignore

        words = _take(iter(top_n_list("en", 60000)), spec.ENGLISH_SIZE)
        if words:
            return words, "wordfreq:top_n_list('en', 60000)"
        errors.append("wordfreq: no valid words after filtering")
    except Exception as e:  # noqa: BLE001
        errors.append(f"wordfreq: {e!r}")

    # (d) google-10000-english
    try:
        with urllib.request.urlopen(EN_FALLBACK_URL, timeout=120) as r:
            text = r.read().decode("utf-8", "replace")
        words = _take(_iter_words(text.splitlines()), spec.ENGLISH_SIZE)
        if words:
            return words, EN_FALLBACK_URL
        errors.append(f"{EN_FALLBACK_URL}: no valid words parsed")
    except Exception as e:  # noqa: BLE001
        errors.append(f"{EN_FALLBACK_URL}: {e!r}")

    raise RuntimeError("所有英文词源均不可用：\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------------------
# 词表拼装（纯函数，便于幂等性测试）
# ---------------------------------------------------------------------------
def _fullwidth_symbols():
    """U+FF01–U+FF5E 全角符号 + U+3000–U+303F 中实际定义的字符。"""
    chars = [chr(cp) for cp in range(0xFF01, 0xFF5E + 1)]
    chars += [
        chr(cp)
        for cp in range(0x3000, 0x303F + 1)
        if unicodedata.category(chr(cp)) != "Cn"  # 跳过未分配码点
    ]
    return chars


def build_tokens(english_words):
    """按 spec 布局拼装 id_to_token 列表（长度 = VOCAB_SIZE_PADDED）。"""
    tokens: list[str] = []

    def pad_to(end: int):
        while len(tokens) < end:
            tokens.append(_reserved(len(tokens)))

    # 1. SPECIAL 0–63
    tokens.extend(spec.SPECIAL_TOKENS)
    pad_to(spec.SPECIAL_BEGIN + spec.SPECIAL_SIZE)

    # 2. DIGIT 64–73
    assert len(tokens) == spec.DIGIT_BEGIN
    tokens.extend(str(d) for d in range(10))
    assert len(tokens) == spec.DIGIT_BEGIN + spec.DIGIT_SIZE

    # 3. SYMBOL 74–585：ASCII 可打印 + 全角符号，剩余 reserved
    assert len(tokens) == spec.SYMBOL_BEGIN
    tokens.extend(chr(cp) for cp in range(0x20, 0x7E + 1))  # 95 个
    tokens.extend(_fullwidth_symbols())
    pad_to(spec.SYMBOL_BEGIN + spec.SYMBOL_SIZE)

    # 4. CJK 586–28169：U+4E00–U+9FFF 接 U+3400–U+4DBF（各按码点升序）
    assert len(tokens) == spec.CJK_BEGIN
    tokens.extend(chr(cp) for cp in range(0x4E00, 0x9FFF + 1))
    tokens.extend(chr(cp) for cp in range(0x3400, 0x4DBF + 1))
    assert len(tokens) == spec.CJK_BEGIN + spec.CJK_SIZE

    # 5. ENGLISH 28170–68169
    assert len(tokens) == spec.ENGLISH_BEGIN
    assert len(english_words) <= spec.ENGLISH_SIZE
    for w in english_words:
        assert EN_WORD_RE.match(w), f"非法英文词形: {w!r}"
    tokens.extend(english_words)
    pad_to(spec.ENGLISH_BEGIN + spec.ENGLISH_SIZE)

    # 6. DRAW 区间 68170–77609
    def numbered(prefix, begin, size):
        assert len(tokens) == begin, f"{prefix}: begin {len(tokens)} != {begin}"
        tokens.extend(f"<{prefix}:{n}>" for n in range(size))

    numbered("x-axis", spec.DRAW_X_BEGIN, spec.DRAW_X_SIZE)
    numbered("y-axis", spec.DRAW_Y_BEGIN, spec.DRAW_Y_SIZE)
    numbered("width", spec.DRAW_W_BEGIN, spec.DRAW_W_SIZE)
    numbered("length", spec.DRAW_L_BEGIN, spec.DRAW_L_SIZE)
    numbered("rot", spec.DRAW_ROT_BEGIN, spec.DRAW_ROT_SIZE)
    assert len(tokens) == spec.DRAW_R_BEGIN
    tokens.extend(f"<R{n}>" for n in range(spec.DRAW_R_SIZE))
    assert len(tokens) == spec.DRAW_G_BEGIN
    tokens.extend(f"<G{n}>" for n in range(spec.DRAW_G_SIZE))
    assert len(tokens) == spec.DRAW_B_BEGIN
    tokens.extend(f"<B{n}>" for n in range(spec.DRAW_B_SIZE))
    assert len(tokens) == spec.DRAW_SHAPE_BEGIN
    tokens.extend(f"<shape:{name}>" for name in spec.DRAW_SHAPES)
    pad_to(spec.DRAW_SHAPE_BEGIN + spec.DRAW_SHAPE_SIZE)

    # 7. MIDI 区间 77610–78633（子布局见 spec.py 注释）
    assert len(tokens) == spec.MIDI_BEGIN
    tokens.extend(f"<midi:note_on:{p}>" for p in range(128))       # 0–127
    tokens.extend(f"<midi:note_off:{p}>" for p in range(128))      # 128–255
    tokens.extend(f"<midi:vel:{v}>" for v in range(32))            # 256–287
    tokens.extend(f"<midi:prog:{p}>" for p in range(128))          # 288–415
    tokens.extend(f"<midi:tempo:{t}>" for t in range(32))          # 416–447
    tokens.extend(f"<midi:shift:{s}>" for s in range(100))         # 448–547
    pad_to(spec.MIDI_BEGIN + spec.MIDI_SIZE)                       # 548–1023

    # 8. 尾部补齐到 64 的倍数
    assert len(tokens) == spec.VOCAB_SIZE
    pad_to(spec.VOCAB_SIZE_PADDED)
    assert len(tokens) == spec.VOCAB_SIZE_PADDED
    return tokens


def build_vocab(words_file: str | None = None, source_desc: str | None = None) -> dict:
    """完整构建（含英文词加载与 meta），返回可 JSON 序列化的 dict。"""
    english_words, source = load_english_words(words_file, source_desc)
    id_to_token = build_tokens(english_words)
    # 注：DIGIT 区 "0".."9" 与 SYMBOL 区同名字符、ENGLISH 区单字母词
    # （"a"/"i" 等）存在文本撞名，token_to_id 后写覆盖先写；
    # 编解码器一律按区间专用映射查表，不依赖 token_to_id 的歧义项。
    token_to_id = {t: i for i, t in enumerate(id_to_token)}
    meta = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "english_source": source,
        "english_word_count": len(english_words),
        "vocab_size": spec.VOCAB_SIZE,
        "vocab_size_padded": spec.VOCAB_SIZE_PADDED,
        "ranges": {
            "special": {"begin": spec.SPECIAL_BEGIN, "size": spec.SPECIAL_SIZE},
            "digit": {"begin": spec.DIGIT_BEGIN, "size": spec.DIGIT_SIZE},
            "symbol": {"begin": spec.SYMBOL_BEGIN, "size": spec.SYMBOL_SIZE},
            "cjk": {"begin": spec.CJK_BEGIN, "size": spec.CJK_SIZE},
            "english": {"begin": spec.ENGLISH_BEGIN, "size": spec.ENGLISH_SIZE},
            "draw_x": {"begin": spec.DRAW_X_BEGIN, "size": spec.DRAW_X_SIZE},
            "draw_y": {"begin": spec.DRAW_Y_BEGIN, "size": spec.DRAW_Y_SIZE},
            "draw_w": {"begin": spec.DRAW_W_BEGIN, "size": spec.DRAW_W_SIZE},
            "draw_l": {"begin": spec.DRAW_L_BEGIN, "size": spec.DRAW_L_SIZE},
            "draw_rot": {"begin": spec.DRAW_ROT_BEGIN, "size": spec.DRAW_ROT_SIZE},
            "draw_r": {"begin": spec.DRAW_R_BEGIN, "size": spec.DRAW_R_SIZE},
            "draw_g": {"begin": spec.DRAW_G_BEGIN, "size": spec.DRAW_G_SIZE},
            "draw_b": {"begin": spec.DRAW_B_BEGIN, "size": spec.DRAW_B_SIZE},
            "draw_shape": {
                "begin": spec.DRAW_SHAPE_BEGIN,
                "size": spec.DRAW_SHAPE_SIZE,
            },
            "midi": {"begin": spec.MIDI_BEGIN, "size": spec.MIDI_SIZE},
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
    args = ap.parse_args(argv)
    vocab = build_vocab(args.words_file, args.source_desc)
    OUT_PATH.write_text(
        json.dumps(vocab, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    m = vocab["meta"]
    print(f"已写出 {OUT_PATH}")
    print(f"  总槽位        : {len(vocab['id_to_token'])}")
    print(f"  英文词实收    : {m['english_word_count']}")
    print(f"  英文词来源    : {m['english_source']}")
    print(f"  文件大小      : {OUT_PATH.stat().st_size} 字节")


if __name__ == "__main__":
    main()
