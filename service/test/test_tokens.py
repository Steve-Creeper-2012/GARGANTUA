"""tokens 包测试：tok-32 词表布局、构建幂等、编解码 roundtrip、
时间步结构、.tokenvector 格式与添加 token。

运行：cd ROOT && python -m unittest service.test.test_tokens -v
前置：tokens/vocab.json 已由 tokens/build_vocab.py 生成。
"""
from __future__ import annotations

# ---- GARGANTUA 路径引导（2026-10-24 目录树重构：server/ service/ start/）----
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

import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(boot.project_root())   # 2026-10-24：项目根

import spec  # noqa: E402
from tokens import dectok, enctok, text_codec, timestep, tok32, tokenvector
from tokens.build_vocab import EN_WORD_RE, build_tokens  # noqa: E402
from tokens.vocab import (CATEGORIES, VOCAB_PATH, Vocab,  # noqa: E402
                          get_vocab)


def setUpModule():
    if not VOCAB_PATH.is_file():
        raise RuntimeError(
            f"{VOCAB_PATH} 不存在，请先运行：python tokens/build_vocab.py"
        )


class VocabLayoutTest(unittest.TestCase):
    """词表区间与 spec 常数一致性（tok-32 四类布局）。"""

    @classmethod
    def setUpClass(cls):
        cls.v: Vocab = get_vocab()
        cls.t = cls.v.id_to_token

    def test_total_size(self):
        self.assertEqual(len(self.t), spec.VOCAB_SIZE_PADDED)
        self.assertEqual(spec.VOCAB_SIZE_PADDED % 64, 0)
        self.assertEqual(spec.VOCAB_SIZE, spec.FALLBACK_BEGIN
                         + spec.FALLBACK_SIZE)
        self.assertEqual(spec.INVISIBLE_END, spec.DIGIT_BEGIN)

    def test_special_region(self):
        # <pad> 已删除（2026-10-24，steve 拍板）
        self.assertNotIn("<pad>", spec.SPECIAL_TOKENS)
        self.assertFalse(hasattr(spec, "PAD_ID"))
        self.assertEqual(spec.SPECIAL_TOKENS,
                         ["<caps>", "<user>", "<user/>",
                          "<model>", "<model/>", "<stop>"])
        for i, tok in enumerate(spec.SPECIAL_TOKENS):
            self.assertEqual(self.t[i], tok)
        self.assertEqual(self.t[spec.STOP_ID], spec.STOP_TOKEN)
        for i in range(len(spec.SPECIAL_TOKENS), spec.SPECIAL_SIZE):
            self.assertEqual(self.t[i], f"<reserved:{i}>")

    def test_system_region(self):
        # 系统 token：{ } | （不可见，时间步结构）
        self.assertEqual(self.t[spec.TS_OPEN_ID], spec.TS_OPEN)
        self.assertEqual(self.t[spec.TS_CLOSE_ID], spec.TS_CLOSE)
        self.assertEqual(self.t[spec.TS_SEP_ID], spec.TS_SEP)
        self.assertEqual(self.v.system_ids[spec.TS_OPEN], spec.TS_OPEN_ID)
        self.assertEqual(self.v.category_of(spec.TS_OPEN_ID), "system")
        self.assertEqual(self.v.category_of(spec.TS_SEP_ID), "system")
        for i in range(3, spec.SYSTEM_SIZE):
            self.assertEqual(self.t[spec.SYSTEM_BEGIN + i],
                             f"<reserved:{spec.SYSTEM_BEGIN + i}>")

    def test_modal_region_draw_layout(self):
        # 图元 token 照搬 geometrize_256_v1：字段 bin 区间 + special 3
        b = spec.DRAW_BEGIN
        self.assertEqual(self.t[b + spec.DRAW_X_OFFSET], "<cx:0>")
        self.assertEqual(self.t[b + spec.DRAW_X_OFFSET + 511], "<cx:511>")
        self.assertEqual(self.t[b + spec.DRAW_ANGLE_OFFSET], "<theta:0>")
        self.assertEqual(self.t[b + spec.DRAW_ANGLE_OFFSET + 269],
                         "<theta:269>")
        self.assertEqual(self.t[b + spec.DRAW_W_OFFSET + 511], "<w:511>")
        self.assertEqual(self.t[b + spec.DRAW_SHAPE_OFFSET], "<shape:0>")
        self.assertEqual(self.t[b + spec.DRAW_R_OFFSET], "<r:0>")
        self.assertEqual(self.t[b + spec.DRAW_R_OFFSET + 127], "<r:127>")
        self.assertEqual(self.t[b + spec.DRAW_SPECIAL_OFFSET],
                         "<draw:special:0>")
        self.assertEqual(self.t[b + spec.DRAW_SPECIAL_OFFSET + 2],
                         "<draw:special:2>")
        # 之后是 reserved，直到 MODAL 区尾
        self.assertEqual(self.t[b + spec.DRAW_VOCAB_SIZE],
                         f"<reserved:{b + spec.DRAW_VOCAB_SIZE}>")
        # MIDI 已暂时删除
        self.assertFalse(hasattr(spec, "MIDI_BEGIN"))
        self.assertEqual(self.v.category_of(b + spec.DRAW_X_OFFSET),
                         "special")
        # 布局自洽：offset 链 + 总数
        self.assertEqual(spec.DRAW_VOCAB_SIZE, 2961)
        self.assertEqual(spec.DRAW_TOKENS_PER_PRIMITIVE, 9)

    def test_digit_region(self):
        for d in range(10):
            self.assertEqual(self.t[spec.DIGIT_BEGIN + d], str(d))

    def test_whitespace_region(self):
        for i, ch in enumerate(spec.WHITESPACE_CHARS):
            self.assertEqual(self.t[spec.WHITESPACE_BEGIN + i], ch)

    def test_symbol_region(self):
        b = spec.SYMBOL_BEGIN
        # 半角 ASCII 标点（P*/S* 类码点按码点升序，字母数字不在其中）
        for ch in ("!", "~", "@", "#", "$", "%", "^", "&", "*", "(", ")",
                   "-", "_", "=", "+", "[", "]", "{", "}", "\\", "|",
                   ";", ":", "'", "\"", ",", ".", "/", "<", ">", "?"):
            self.assertIn(ch, self.v.regular_ids)
        # 全角符号与全角字母数字
        for ch in ("！", "，", "。", "、", "「", "」", "　", "１", "Ａ", "ａ"):
            self.assertIn(ch, self.v.regular_ids)
        # 区域尾部是 reserved
        self.assertTrue(
            self.t[spec.SYMBOL_BEGIN + spec.SYMBOL_SIZE - 1]
            .startswith("<reserved:"))

    def test_emoji_region(self):
        b = spec.EMOJI_BEGIN
        self.assertEqual(self.t[b], "\u00a9")            # ©
        self.assertIn("\U0001F600", self.v.regular_ids)   # 😀
        self.assertIn("\U0001F1E6", self.v.regular_ids)   # 🇦 区域指示符
        self.assertIn("\u200D", self.v.regular_ids)       # ZWJ
        self.assertIn("\uFE0F", self.v.regular_ids)       # VS16
        # ASCII 键帽基字符归属 SYMBOL/DIGIT 区，不进 emoji 区
        self.assertLess(self.v.regular_ids["#"], spec.EMOJI_BEGIN)
        self.assertLess(self.v.regular_ids["*"], spec.EMOJI_BEGIN)

    def test_cjk_boundaries(self):
        b = spec.CJK_BEGIN
        self.assertEqual(self.t[b], "\u4e00")             # 一
        self.assertEqual(self.t[b + 20991], "\u9fff")
        self.assertEqual(self.t[b + 20992], "\u3400")     # 㐀
        self.assertEqual(self.t[b + 27584 - 1], "\u4dbf")   # 最后一个实字
        self.assertTrue(self.t[b + 27584].startswith("<reserved:"))
        self.assertEqual(self.v.regular_ids["\u4e16"], b + (0x4E16 - 0x4E00))
        self.assertEqual(b + spec.CJK_SIZE, spec.ENGLISH_BEGIN)

    def test_english_region(self):
        n = self.v.meta["english_word_count"]
        self.assertEqual(n, 30000)
        self.assertLessEqual(n, spec.ENGLISH_SIZE)
        self.assertIn("hello", self.v.regular_ids)
        self.assertIn("world", self.v.regular_ids)
        # 单字母词不收录（由 fallback 承担，保证全表文本唯一）
        for w in ("a", "i"):
            self.assertNotIn(w, self.v.regular_ids)
            self.assertIn(w, self.v.fallback_ids)

    def test_fallback_region(self):
        b = spec.FALLBACK_BEGIN
        for i, ch in enumerate(spec.FALLBACK_LETTERS):
            self.assertEqual(self.t[b + i], ch)
            self.assertEqual(self.v.fallback_ids[ch], b + i)
        self.assertEqual(self.v.category_of(b), "fallback")

    def test_global_text_uniqueness(self):
        # 全词表同一字符串唯一（唯一例外：3 个系统显示字符，见 build_vocab）
        seen = {}
        for i, tok in enumerate(self.t):
            if tok in seen and tok not in (spec.TS_OPEN, spec.TS_CLOSE,
                                           spec.TS_SEP):
                self.fail(f"同文本双 ID: {tok!r}（{seen[tok]} 与 {i}）")
            seen.setdefault(tok, i)
        # 系统显示字符的 text→id 必须解析到可见符号 token
        for ch in ("{", "}", "|"):
            tid = self.v.lookup(ch)
            self.assertIsNotNone(tid)
            self.assertEqual(self.v.category_of(tid), "regular")

    def test_lookup_priority(self):
        # 常规优先、fallback 兜底；不可见两类不参与查表
        self.assertEqual(self.v.lookup("hello"), self.v.regular_ids["hello"])
        self.assertEqual(self.v.lookup("q"), self.v.fallback_ids["q"])
        self.assertIsNone(self.v.lookup("<caps>"))
        self.assertIsNone(self.v.lookup("\u4dbf\u4dbf\u4dbf"))  # 未收录

    def test_category_of(self):
        self.assertEqual(self.v.category_of(spec.DIGIT_BEGIN), "regular")
        self.assertEqual(self.v.category_of(spec.CJK_BEGIN), "regular")
        self.assertEqual(self.v.category_of(spec.FALLBACK_BEGIN), "fallback")
        self.assertEqual(self.v.category_of(spec.TS_OPEN_ID), "system")
        self.assertEqual(self.v.category_of(spec.SPECIAL_BEGIN + 1),
                         "special")
        self.assertEqual(self.v.category_of(spec.DRAW_BEGIN), "special")

    def test_meta_consistency(self):
        m = self.v.meta
        self.assertEqual(m["format"], "tok-32")
        self.assertEqual(m["id_bits"], 32)
        self.assertEqual(m["vocab_size"], spec.VOCAB_SIZE)
        self.assertEqual(m["vocab_size_padded"], spec.VOCAB_SIZE_PADDED)
        self.assertEqual(m["english_word_count"], 30000)
        self.assertEqual(m["emoji_count"], 1517)   # 1419 Emoji + 98 组件
        self.assertEqual(m["symbol_count"], 7263)
        r = m["ranges"]
        for key in ("special", "system", "modal", "digit", "whitespace",
                    "symbol", "emoji", "cjk", "english", "extra", "fallback",
                    "draw"):
            self.assertIn(key, r)
        self.assertNotIn("midi", r)
        self.assertEqual(r["draw"]["tokens_per_primitive"], 9)
        self.assertEqual(r["emoji"]["begin"], spec.EMOJI_BEGIN)
        self.assertEqual(r["fallback"]["begin"], spec.FALLBACK_BEGIN)


class BuildIdempotentTest(unittest.TestCase):
    """构建可重跑幂等：同一输入 → 同一 id_to_token；user_added 重放。"""

    def test_build_tokens_idempotent(self):
        v = get_vocab()
        words = [v.id_to_token[spec.ENGLISH_BEGIN + i]
                 for i in range(v.meta["english_word_count"])]
        t1 = build_tokens(words)
        t2 = build_tokens(words)
        self.assertEqual(t1, t2)
        self.assertEqual(t1, v.id_to_token)

    def test_build_tokens_small_wordlist(self):
        words = ["hello", "world", "don't"]
        t1 = build_tokens(words)
        t2 = build_tokens(words)
        self.assertEqual(t1, t2)
        b = spec.ENGLISH_BEGIN
        self.assertEqual(t1[b:b + 3], words)
        self.assertEqual(t1[b + 3], f"<reserved:{b + 3}>")
        self.assertEqual(len(t1), spec.VOCAB_SIZE_PADDED)

    def test_user_added_replay(self):
        # 重建时 user_added 条目按原 ID 重放
        v = get_vocab()
        words = [v.id_to_token[spec.ENGLISH_BEGIN + i]
                 for i in range(v.meta["english_word_count"])]
        entries = [{"text": "\u4dbf\u4dbf", "category": "regular",
                    "id": spec.EXTRA_BEGIN},
                   {"text": "<mood>", "category": "special",
                    "id": spec.SPECIAL_BEGIN + len(spec.SPECIAL_TOKENS)}]
        t = build_tokens(words, entries)
        self.assertEqual(t[spec.EXTRA_BEGIN], "\u4dbf\u4dbf")
        self.assertEqual(t[spec.SPECIAL_BEGIN + len(spec.SPECIAL_TOKENS)],
                         "<mood>")
        self.assertEqual(len(t), spec.VOCAB_SIZE_PADDED)


CAPS_ID = spec.SPECIAL_BEGIN + spec.SPECIAL_TOKENS.index("<caps>")


class CodecTest(unittest.TestCase):
    """文本编解码（enctok 规则 + text_codec Steps API）。"""

    def assertRoundtrip(self, text: str):
        steps = text_codec.encode(text)
        self.assertEqual(text_codec.decode(steps), text)
        return steps

    def test_step_structure(self):
        steps = text_codec.encode("\u4f60\u597d world 123")
        for st in steps:
            self.assertEqual(set(st), {"token_ids", "type", "modality",
                                       "meta"})
            self.assertEqual(len(st["token_ids"]), 1)
            self.assertIsInstance(st["token_ids"][0], int)
            self.assertEqual(st["type"], spec.TYPE_NONE)
            self.assertEqual(st["modality"], "text")
            self.assertIsInstance(st["meta"], dict)

    def test_roundtrip_mixed_cjk_english_digit(self):
        steps = self.assertRoundtrip("\u4f60\u597d world 123")
        v = get_vocab()
        ids = [st["token_ids"][0] for st in steps]
        self.assertEqual(
            ids,
            [
                v.regular_ids["\u4f60"], v.regular_ids["\u597d"],
                v.regular_ids[" "], v.regular_ids["world"],
                v.regular_ids[" "],
                spec.DIGIT_BEGIN + 1, spec.DIGIT_BEGIN + 2,
                spec.DIGIT_BEGIN + 3,
            ],
        )

    def test_caps_roundtrip(self):
        steps = self.assertRoundtrip("Hello")
        v = get_vocab()
        ids = [st["token_ids"][0] for st in steps]
        self.assertEqual(ids, [CAPS_ID, v.regular_ids["hello"]])
        self.assertRoundtrip("Hello world")

    def test_allcaps_letter_spelling(self):
        # 无 <allcaps>：全大写逐字母 fallback + 每字母前 <caps>（无损）
        steps = self.assertRoundtrip("HELLO")
        v = get_vocab()
        ids = [st["token_ids"][0] for st in steps]
        self.assertEqual(ids, [CAPS_ID, v.fallback_ids["h"], CAPS_ID,
                               v.fallback_ids["e"], CAPS_ID,
                               v.fallback_ids["l"], CAPS_ID,
                               v.fallback_ids["l"], CAPS_ID,
                               v.fallback_ids["o"]])

    def test_internal_caps_word(self):
        # 混合大小写同样逐字母，精确还原（旧版有损归一为 Mcdonald）
        self.assertRoundtrip("McDonald")

    def test_fallback_spelling_roundtrip(self):
        v = get_vocab()
        self.assertNotIn("asdfqwer", v.regular_ids)
        steps = self.assertRoundtrip("asdfqwer")
        self.assertEqual(len(steps), 8)  # 逐字母，全小写无 caps

    def test_fallback_spelling_with_caps(self):
        steps = self.assertRoundtrip("AsdfQwer")
        v = get_vocab()
        ids = [st["token_ids"][0] for st in steps]
        self.assertEqual(ids[0], CAPS_ID)          # A
        self.assertEqual(ids[5], CAPS_ID)          # Q

    def test_literal_special_strings_roundtrip(self):
        # 字面 "<caps>" 等不再被特殊识别，按普通字符编码并可还原
        self.assertRoundtrip("<caps> hello")
        self.assertRoundtrip("<user>\u5b57<user/>")

    def test_empty_string(self):
        self.assertEqual(text_codec.encode(""), [])
        self.assertEqual(text_codec.decode([]), "")

    def test_fullwidth_halfwidth_punct(self):
        self.assertRoundtrip("\u534a\u89d2!\u5168\u89d2\uff01\u3000\u3002\uff0c\u3001\u300c\u300d\uff1b\uff1f\uff11\uff12\uff13\uff21\uff22")
        self.assertRoundtrip("a~b@c#d$e%f^g&h*i(j)k_l+m=n|o")

    def test_emoji_roundtrip(self):
        self.assertRoundtrip("\U0001F600\U0001F44D\u4e2d\u56fd\U0001F1E8\U0001F1F3")
        self.assertRoundtrip("\U0001F468\u200D\U0001F469\u200D\U0001F467\u200D\U0001F466")  # ZWJ 序列

    def test_skipped_unmapped_chars(self):
        # CJK 扩展B 及以后未收录：跳过并记录
        steps = text_codec.encode("\u4f60\u597d\U00020000")
        self.assertEqual(text_codec.decode(steps), "\u4f60\u597d")
        self.assertEqual(steps[-1]["meta"]["skipped"], ["\U00020000"])

    def test_digits_in_cjk_context(self):
        self.assertRoundtrip("2026\u5e749\u670827\u65e5")

    def test_single_letter_words(self):
        self.assertRoundtrip("I am a cat")

    def test_apostrophe_word(self):
        self.assertRoundtrip("don't stop")

    def test_control_chars(self):
        self.assertRoundtrip("a\nb\tc\rd")

    def test_stop_token_skipped_in_decode(self):
        v = get_vocab()
        stop_only = [{"token_ids": [spec.STOP_ID], "type": spec.TYPE_NONE,
                      "modality": "text", "meta": {}}]
        self.assertEqual(text_codec.decode(stop_only), "")
        mixed = [
            {"token_ids": [CAPS_ID], "type": spec.TYPE_NONE,
             "modality": "text", "meta": {}},
            {"token_ids": [spec.STOP_ID], "type": spec.TYPE_NONE,
             "modality": "text", "meta": {}},
            {"token_ids": [v.regular_ids["of"]], "type": spec.TYPE_NONE,
             "modality": "text", "meta": {}},
        ]
        self.assertEqual(text_codec.decode(mixed), "Of")

    def test_decode_ids_errors(self):
        for bad in (["x"], [True], [-1], [spec.VOCAB_SIZE_PADDED]):
            with self.assertRaises(ValueError):
                text_codec.decode_ids(bad)


class EncDecScriptTest(unittest.TestCase):
    """tok32 / enctok / dectok 脚本行为。"""

    def test_tok32_roundtrip(self):
        ids = [0, 1, 64, 12384, 88689, spec.VOCAB_SIZE_PADDED - 1]
        data = tok32.encode_ids(ids)
        self.assertEqual(len(data), len(ids) * 4)
        self.assertEqual(tok32.decode_ids(data), ids)
        h = tok32.ids_to_hex(ids)
        self.assertEqual(len(h), len(ids) * 8)
        self.assertEqual(tok32.hex_to_ids(h), ids)

    def test_tok32_errors(self):
        with self.assertRaises(ValueError):
            tok32.decode_ids(b"\x00\x00\x00")          # 非 4 倍数
        with self.assertRaises(ValueError):
            tok32.encode_ids([1 << 32])                    # 越界
        with self.assertRaises(ValueError):
            tok32.encode_ids(["x"])                        # 非整数
        with self.assertRaises(ValueError):
            tok32.hex_to_ids("0000")                       # 非 8 倍数
        with self.assertRaises(ValueError):
            tok32.hex_to_ids("zzzzzzzz")                   # 非法十六进制

    def test_tok32_file_roundtrip(self):
        ids = [1, 2, 3, 12384]
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.tokens"
            tok32.write_tokens_file(f, ids)
            self.assertEqual(tok32.read_tokens_file(f), ids)
            self.assertEqual(f.stat().st_size, 16)

    def test_enctok_hex_output(self):
        ids, skipped = enctok.encode_text("Hello World")
        self.assertEqual(skipped, [])
        v = get_vocab()
        self.assertEqual(ids, [CAPS_ID, v.regular_ids["hello"],
                               v.regular_ids[" "], CAPS_ID,
                               v.regular_ids["world"]])

    def test_enctok_input_type_error(self):
        with self.assertRaises(ValueError):
            enctok.encode_text(123)

    def test_dectok_ids_mode(self):
        v = get_vocab()
        ids = [spec.TS_OPEN_ID, CAPS_ID, v.regular_ids["hi"],
               spec.TS_SEP_ID, spec.TS_CLOSE_ID]
        self.assertEqual(dectok.decode_ids(ids), "{%08x%08x|}"
                         % (CAPS_ID, v.regular_ids["hi"]))

    def test_dectok_text_mode(self):
        ids, _ = enctok.encode_text("Hello \u4e16\u754c")
        self.assertEqual(dectok.decode_ids(ids, "text"),
                         "Hello \u4e16\u754c")

    def test_dectok_hex_roundtrip(self):
        ids, _ = enctok.encode_text("abc 123")
        self.assertEqual(dectok.decode_hex(tok32.ids_to_hex(ids), "text"),
                         "abc 123")


class TimestepTest(unittest.TestCase):
    """时间步结构：{ } 包裹、| 分隔。"""

    def test_wrap_unwrap(self):
        self.assertEqual(timestep.wrap([1, 2, 3]),
                         [spec.TS_OPEN_ID, 1, spec.TS_SEP_ID, 2,
                          spec.TS_SEP_ID, 3, spec.TS_CLOSE_ID])

    def test_empty_step_rejected(self):
        # steve 设计约束：{} 必须含有 token，不能为空集
        with self.assertRaises(ValueError):
            timestep.wrap([])
        with self.assertRaises(ValueError):
            timestep.unwrap([spec.TS_OPEN_ID, spec.TS_CLOSE_ID])
        flat = timestep.wrap_steps([[1, 2], [3]])
        self.assertEqual(flat, [spec.TS_OPEN_ID, 1, spec.TS_SEP_ID, 2,
                                spec.TS_CLOSE_ID, spec.TS_OPEN_ID, 3,
                                spec.TS_CLOSE_ID])
        self.assertEqual(timestep.unwrap(flat), [[1, 2], [3]])

    def test_unwrap_errors(self):
        O, C, S = spec.TS_OPEN_ID, spec.TS_CLOSE_ID, spec.TS_SEP_ID
        for bad in ([1], [O, 1], [C], [O, S, C], [O, C], [O, 1, S, S, 2, C],
                    [O, O, 1, C], [O, 1, C, 2], [O, 1, S]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                timestep.unwrap(bad)

    def test_render(self):
        O, C, S = spec.TS_OPEN_ID, spec.TS_CLOSE_ID, spec.TS_SEP_ID
        self.assertEqual(timestep.render([O, 1, S, 2, C]),
                         "{%08x|%08x}" % (1, 2))
        self.assertEqual(timestep.render([O, 1, C]),
                         "{%08x}" % 1)
        self.assertEqual(
            timestep.render([O, 1, C], surface=lambda i: "t%d" % i),
            "{t1}")
        self.assertEqual(timestep.count_steps([O, C, O, 1, C]), 2)

    def test_text_is_one_token_per_step(self):
        # 文本每 token 独占一步：Step 数 == token 数
        steps = text_codec.encode("hello \u4e16\u754c")
        self.assertEqual(len(steps), len(enctok.encode_text_ids(
            "hello \u4e16\u754c")))


class TokenVectorTest(unittest.TestCase):
    """.tokenvector 格式：dim+precision 头，{} | 结构，纯数字体。"""

    def test_dumps_loads_roundtrip(self):
        tv = tokenvector.dumps(
            [[[0.1, -0.2, 0.3], [1.0, 2.0, 3.0]], [[0.5, 0.5, 0.5]]],
            dim=3)
        self.assertTrue(tv.startswith("dim=3 precision=f32\n"))
        self.assertIn("{0.1 -0.2 0.3|1 2 3}", tv)
        self.assertIn("{0.5 0.5 0.5}", tv)
        dim, prec, steps = tokenvector.loads(tv)
        self.assertEqual((dim, prec), (3, "f32"))
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0][0].tobytes(),
                         np.asarray([0.1, -0.2, 0.3],
                                    dtype=np.float32).tobytes())

    def test_precisions_roundtrip(self):
        rng = np.random.default_rng(7)
        for prec in ("f32", "f16", "bf16"):
            vals = rng.normal(size=64).astype(np.float32)
            tv = tokenvector.dumps([[vals]], dim=64, precision=prec)
            d, p, steps = tokenvector.loads(tv)
            self.assertEqual((d, p), (64, prec))
            self.assertEqual(steps[0][0].tobytes(),
                             tokenvector._cast(vals, prec).tobytes())

    def test_empty_sequence_ok_empty_step_rejected(self):
        # 零时间步（空序列）合法；{} 内无 token 非法（steve 设计约束）
        tv = tokenvector.dumps([], dim=4)
        self.assertEqual(tokenvector.loads(tv), (4, "f32", []))
        with self.assertRaises(ValueError):
            tokenvector.dumps([[]], dim=4)
        with self.assertRaises(ValueError):
            tokenvector.loads("dim=4 precision=f32\n{}\n")

    def test_dim_mismatch(self):
        with self.assertRaises(ValueError):
            tokenvector.dumps([[[1.0, 2.0]]], dim=3)

    def test_loads_errors(self):
        for bad in ("", "dim=3\n", "{1 2 3}\n", "dim=0 precision=f32\n",
                    "dim=3 precision=f32\n{}\n",
                    "dim=3 precision=f64\n{1 2 3}\n",
                    "dim=3 precision=f32\n{1 2\n",
                    "dim=3 precision=f32\n{1 2 x}\n",
                    "dim=3 precision=f32\n{1 2 3 4}\n"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                tokenvector.loads(bad)

    def test_comments_in_header(self):
        tv = "# note\ndim=2 precision=f32\n{1 2}\n"
        dim, prec, steps = tokenvector.loads(tv)
        self.assertEqual((dim, prec, steps[0][0].tolist()), (2, "f32", [1.0, 2.0]))

    def test_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.tokenvector"
            tokenvector.write_tokenvector(f, [[[1.0, 2.0]]], dim=2)
            dim, prec, steps = tokenvector.read_tokenvector(f)
            self.assertEqual((dim, prec), (2, "f32"))
            self.assertEqual(steps[0][0].tolist(), [1.0, 2.0])


class AddTokenTest(unittest.TestCase):
    """添加 token：四类任选、全局查重、reserved 槽分配、持久化。

    全部在临时副本上进行，不碰仓库里的 tokens/vocab.json。
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="garg_vocab_"))
        self.copy = self.tmp / "vocab.json"
        shutil.copy(VOCAB_PATH, self.copy)
        self.v = Vocab(self.copy)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_add_all_categories(self):
        cases = [("\U00020000", "regular"),   # CJK 扩展B 单字
                 ("\u00df", "fallback"),                   # ß
                 ("<mood>", "special"),
                 ("<ts>", "system")]
        for text, cat in cases:
            tid = self.v.add_token(text, cat)
            self.assertEqual(self.v.category_of(tid), cat)
            self.assertEqual(self.v.token(tid), text)
        self.assertEqual(self.v.lookup("\U00020000"),
                         self.v.regular_ids["\U00020000"])
        self.assertEqual(self.v.fallback_ids["\u00df"],
                         self.v.fallback_ids["\u00df"])

    def test_duplicate_rejected(self):
        self.v.add_token("\U00020000", "regular")
        with self.assertRaises(ValueError):
            self.v.add_token("\U00020000", "regular")   # 同类重复
        with self.assertRaises(ValueError):
            self.v.add_token("hello", "regular")          # 与基础词表撞名
        with self.assertRaises(ValueError):
            self.v.add_token("<caps>", "special")         # 与特殊 token 撞名
        with self.assertRaises(ValueError):
            self.v.add_token("a", "regular")              # 与 fallback 字母撞名

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.v.add_token("x", "bogus")
        with self.assertRaises(ValueError):
            self.v.add_token("", "regular")
        with self.assertRaises(ValueError):
            self.v.add_token("<reserved:9>", "regular")

    def test_persist_and_reload(self):
        tid = self.v.add_token("<mood>", "special")
        v2 = Vocab(self.copy)
        self.assertEqual(v2.special_ids["<mood>"], tid)
        import json
        meta = json.loads(self.copy.read_text(encoding="utf-8"))["meta"]
        self.assertEqual(meta["user_added"],
                         [{"text": "<mood>", "category": "special",
                           "id": tid}])

    def test_free_slots(self):
        before = self.v.free_slots("regular")
        self.v.add_token("\U00020000", "regular")
        self.assertEqual(self.v.free_slots("regular"), before - 1)

    def test_category_list(self):
        self.assertEqual(set(CATEGORIES),
                         {"regular", "fallback", "special", "system"})


if __name__ == "__main__":
    unittest.main()
