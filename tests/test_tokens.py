"""tokens 包测试：词表布局一致性、构建幂等、文本编解码 roundtrip。

运行：cd ROOT && python -m unittest tests.test_tokens -v
前置：tokens/vocab.json 已由 tokens/build_vocab.py 生成。
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import spec  # noqa: E402
from tokens.build_vocab import EN_WORD_RE, build_tokens  # noqa: E402
from tokens.text_codec import decode, encode  # noqa: E402
from tokens.vocab import VOCAB_PATH, Vocab, get_vocab  # noqa: E402


def setUpModule():
    if not VOCAB_PATH.is_file():
        raise RuntimeError(
            f"{VOCAB_PATH} 不存在，请先运行：python tokens/build_vocab.py"
        )


class VocabLayoutTest(unittest.TestCase):
    """词表区间与 spec 常数一致性。"""

    @classmethod
    def setUpClass(cls):
        cls.v: Vocab = get_vocab()
        cls.t = cls.v.id_to_token

    def test_total_size(self):
        self.assertEqual(len(self.t), spec.VOCAB_SIZE_PADDED)
        self.assertEqual(spec.VOCAB_SIZE_PADDED % 64, 0)
        self.assertEqual(
            spec.VOCAB_SIZE, spec.MIDI_BEGIN + spec.MIDI_SIZE
        )

    def test_special_region(self):
        self.assertEqual(self.t[spec.PAD_ID], "<pad>")
        for i, tok in enumerate(spec.SPECIAL_TOKENS):
            self.assertEqual(self.t[i], tok)
        self.assertEqual(len(spec.SPECIAL_TOKENS), 28)
        self.assertEqual(self.t[spec.STOP_ID], spec.STOP_TOKEN)
        self.assertEqual(self.v.special_ids[spec.STOP_TOKEN], spec.STOP_ID)
        for i in range(len(spec.SPECIAL_TOKENS), spec.SPECIAL_SIZE):
            self.assertEqual(self.t[i], f"<reserved:{i}>")

    def test_digit_region(self):
        self.assertEqual(spec.DIGIT_BEGIN, 64)
        for d in range(10):
            self.assertEqual(self.t[spec.DIGIT_BEGIN + d], str(d))

    def test_symbol_ascii(self):
        self.assertEqual(spec.SYMBOL_BEGIN, 74)
        self.assertEqual(self.t[spec.SYMBOL_BEGIN], " ")          # 0x20
        self.assertEqual(self.t[spec.SYMBOL_BEGIN + 94], "~")     # 0x7E
        self.assertEqual(self.v.symbol_ids["a"], spec.SYMBOL_BEGIN + (0x61 - 0x20))
        self.assertEqual(self.v.symbol_ids["z"], spec.SYMBOL_BEGIN + (0x7A - 0x20))

    def test_symbol_fullwidth(self):
        # 全角符号 U+FF01 起紧跟在 95 个 ASCII 之后
        base = spec.SYMBOL_BEGIN + 95
        self.assertEqual(self.t[base], "！")                       # U+FF01
        self.assertEqual(self.t[base + (0xFF0C - 0xFF01)], "，")
        self.assertEqual(self.t[base + 94 - 1], "～")              # U+FF5E
        for ch in ("。", "、", "　", "「", "」", "１", "Ａ"):
            self.assertIn(ch, self.v.symbol_ids)
        # 区间内无越界 reserved 之外的非法 token
        self.assertTrue(
            self.t[spec.SYMBOL_BEGIN + spec.SYMBOL_SIZE - 1].startswith("<reserved:")
        )

    def test_cjk_boundaries(self):
        b = spec.CJK_BEGIN
        self.assertEqual(b, 586)
        self.assertEqual(self.t[b], "一")                          # U+4E00
        self.assertEqual(self.t[b + 20991], "鿿")                  # U+9FFF
        self.assertEqual(self.t[b + 20992], "㐀")                  # U+3400
        self.assertEqual(self.t[b + spec.CJK_SIZE - 1], "䶿")      # U+4DBF
        self.assertEqual(self.v.cjk_ids["世"], b + (0x4E16 - 0x4E00))
        self.assertEqual(b + spec.CJK_SIZE, spec.ENGLISH_BEGIN)

    def test_english_region(self):
        b = spec.ENGLISH_BEGIN
        self.assertEqual(b, 28170)
        n = self.v.meta["english_word_count"]
        self.assertGreaterEqual(n, 39000)
        self.assertLessEqual(n, spec.ENGLISH_SIZE)
        # 词形合法、全小写、无重复
        words = [w for w in self.v.english_ids]
        self.assertEqual(len(words), n)
        for w in words[:5000]:
            self.assertRegex(w, EN_WORD_RE)
        # 高频词必在
        for w in ("the", "hello", "world", "i", "a"):
            self.assertIn(w, self.v.english_ids)
        # 边界：最后一个实词之后是 reserved（若未收满）
        if n < spec.ENGLISH_SIZE:
            self.assertEqual(self.t[b + n], f"<reserved:{b + n}>")
        self.assertEqual(self.t[b + spec.ENGLISH_SIZE - 1], self.t[68169])

    def test_draw_regions(self):
        t = self.t
        self.assertEqual(t[spec.DRAW_X_BEGIN], "<x-axis:0>")
        self.assertEqual(t[spec.DRAW_X_BEGIN + 1919], "<x-axis:1919>")
        self.assertEqual(t[spec.DRAW_Y_BEGIN], "<y-axis:0>")
        self.assertEqual(t[spec.DRAW_Y_BEGIN + 1079], "<y-axis:1079>")
        self.assertEqual(t[spec.DRAW_W_BEGIN + 1919], "<width:1919>")
        self.assertEqual(t[spec.DRAW_L_BEGIN + 1079], "<length:1079>")
        self.assertEqual(t[spec.DRAW_ROT_BEGIN + 359], "<rot:359>")
        self.assertEqual(t[spec.DRAW_R_BEGIN], "<R0>")
        self.assertEqual(t[spec.DRAW_R_BEGIN + 1023], "<R1023>")
        self.assertEqual(t[spec.DRAW_G_BEGIN + 1023], "<G1023>")
        self.assertEqual(t[spec.DRAW_B_BEGIN + 1023], "<B1023>")
        for i, name in enumerate(spec.DRAW_SHAPES):
            self.assertEqual(t[spec.DRAW_SHAPE_BEGIN + i], f"<shape:{name}>")
        self.assertEqual(
            t[spec.DRAW_SHAPE_BEGIN + 4],
            f"<reserved:{spec.DRAW_SHAPE_BEGIN + 4}>",
        )

    def test_midi_layout(self):
        t, m = self.t, spec.MIDI_BEGIN
        self.assertEqual(m, 77610)
        self.assertEqual(t[m + 0], "<midi:note_on:0>")
        self.assertEqual(t[m + 127], "<midi:note_on:127>")
        self.assertEqual(t[m + 128], "<midi:note_off:0>")
        self.assertEqual(t[m + 255], "<midi:note_off:127>")
        self.assertEqual(t[m + 256], "<midi:vel:0>")
        self.assertEqual(t[m + 287], "<midi:vel:31>")
        self.assertEqual(t[m + 288], "<midi:prog:0>")
        self.assertEqual(t[m + 415], "<midi:prog:127>")
        self.assertEqual(t[m + 416], "<midi:tempo:0>")
        self.assertEqual(t[m + 447], "<midi:tempo:31>")
        self.assertEqual(t[m + 448], "<midi:shift:0>")
        self.assertEqual(t[m + 547], "<midi:shift:99>")
        self.assertEqual(t[m + 548], f"<reserved:{m + 548}>")
        self.assertEqual(t[m + 1023], f"<reserved:{m + 1023}>")
        self.assertEqual(m + spec.MIDI_SIZE, spec.VOCAB_SIZE)

    def test_tail_reserved(self):
        for i in range(spec.VOCAB_SIZE, spec.VOCAB_SIZE_PADDED):
            self.assertEqual(self.t[i], f"<reserved:{i}>")

    def test_meta_consistency(self):
        m = self.v.meta
        self.assertEqual(m["vocab_size"], spec.VOCAB_SIZE)
        self.assertEqual(m["vocab_size_padded"], spec.VOCAB_SIZE_PADDED)
        r = m["ranges"]
        self.assertEqual(r["special"], {"begin": 0, "size": spec.SPECIAL_SIZE})
        self.assertEqual(r["cjk"]["begin"], spec.CJK_BEGIN)
        self.assertEqual(r["english"]["size"], spec.ENGLISH_SIZE)
        self.assertEqual(r["midi"]["begin"], spec.MIDI_BEGIN)
        self.assertTrue(m["english_source"])
        self.assertTrue(m["built_at"])


class BuildIdempotentTest(unittest.TestCase):
    """构建可重跑幂等：同一英文词表输入 → 同一 id_to_token。"""

    def test_build_tokens_idempotent(self):
        # 用已构建词表中的实收英文词（原顺序）重放
        v = get_vocab()
        b = spec.ENGLISH_BEGIN
        ordered = [
            v.id_to_token[b + i]
            for i in range(v.meta["english_word_count"])
        ]
        t1 = build_tokens(ordered)
        t2 = build_tokens(ordered)
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


class CodecTest(unittest.TestCase):
    """文本编解码。"""

    def assertRoundtrip(self, text: str):
        steps = encode(text)
        self.assertEqual(decode(steps), text)
        return steps

    def test_step_structure(self):
        steps = encode("你好 world 123")
        for st in steps:
            self.assertEqual(set(st), {"token_ids", "type", "modality", "meta"})
            self.assertEqual(len(st["token_ids"]), 1)
            self.assertIsInstance(st["token_ids"][0], int)
            self.assertEqual(st["type"], spec.TYPE_NONE)
            self.assertEqual(st["modality"], "text")
            self.assertIsInstance(st["meta"], dict)

    def test_roundtrip_mixed_cjk_english_digit(self):
        steps = self.assertRoundtrip("你好 world 123")
        v = get_vocab()
        ids = [st["token_ids"][0] for st in steps]
        self.assertEqual(
            ids,
            [
                v.cjk_ids["你"], v.cjk_ids["好"],
                v.symbol_ids[" "], v.english_ids["world"],
                v.symbol_ids[" "],
                spec.DIGIT_BEGIN + 1, spec.DIGIT_BEGIN + 2, spec.DIGIT_BEGIN + 3,
            ],
        )

    def test_caps_roundtrip(self):
        steps = self.assertRoundtrip("Hello")
        v = get_vocab()
        ids = [st["token_ids"][0] for st in steps]
        self.assertEqual(ids, [v.special_ids["<caps>"], v.english_ids["hello"]])
        self.assertRoundtrip("Hello world")

    def test_allcaps_roundtrip(self):
        steps = self.assertRoundtrip("HELLO")
        v = get_vocab()
        ids = [st["token_ids"][0] for st in steps]
        self.assertEqual(
            ids, [v.special_ids["<allcaps>"], v.english_ids["hello"]]
        )

    def test_internal_caps_word(self):
        # 内部大写不保留：按首字母大写规则 → <caps> + mcdonald
        steps = encode("McDonald")
        self.assertEqual(decode(steps), "Mcdonald")

    def test_fallback_spelling_roundtrip(self):
        v = get_vocab()
        self.assertNotIn("asdfqwer", v.english_ids)
        steps = self.assertRoundtrip("asdfqwer")
        self.assertEqual(len(steps), 8)  # 逐字母，全小写无 caps

    def test_fallback_spelling_with_caps(self):
        steps = encode("AsdfQwer")
        self.assertEqual(decode(steps), "AsdfQwer")
        v = get_vocab()
        caps = v.special_ids["<caps>"]
        ids = [st["token_ids"][0] for st in steps]
        self.assertEqual(ids[0], caps)          # A
        self.assertEqual(ids[5], caps)          # Q

    def test_format_tags_roundtrip(self):
        self.assertRoundtrip("<bold>加粗<bold/>普通<italic>斜<italic/>")
        self.assertRoundtrip("<think>嗯<think/><answer>好<answer/>")
        self.assertRoundtrip("<user><text>你好<text/><user/>")

    def test_unknown_angle_brackets(self):
        # 未注册的 "<...>" 形式按普通字符逐字编码
        self.assertRoundtrip("<foo> a <b123>")

    def test_empty_string(self):
        self.assertEqual(encode(""), [])
        self.assertEqual(decode([]), "")

    def test_fullwidth_halfwidth_punct(self):
        self.assertRoundtrip("半角!全角！　。，、「」；？１２３ＡＢ")
        self.assertRoundtrip("a~b@c#d$e%f^g&h*i(j)k_l+m=n|o")

    def test_skipped_unmapped_chars(self):
        steps = encode("你好😀𠀀")  # emoji + CJK 扩展 B 均未收录
        self.assertEqual(decode(steps), "你好")
        self.assertEqual(steps[-1]["meta"]["skipped"], ["😀", "𠀀"])

    def test_digits_in_cjk_context(self):
        self.assertRoundtrip("2026年9月27日")

    def test_single_letter_words(self):
        self.assertRoundtrip("I am a cat")

    def test_apostrophe_word(self):
        # 撇号不属于 [A-Za-z]，词被拆开但仍精确还原
        self.assertRoundtrip("don't stop")

    def test_stop_token_skipped_in_decode(self):
        # 停止 token 静默跳过，不产生任何输出
        stop_only = [{"token_ids": [spec.STOP_ID], "type": spec.TYPE_NONE,
                      "modality": "text", "meta": {}}]
        self.assertEqual(decode(stop_only), "")
        # 与大小写/词 token 混合时同样被跳过（caps+of → "Of"）
        v = get_vocab()
        mixed = [
            {"token_ids": [v.special_ids["<caps>"]], "type": spec.TYPE_NONE,
             "modality": "text", "meta": {}},
            {"token_ids": [spec.STOP_ID], "type": spec.TYPE_NONE,
             "modality": "text", "meta": {}},
            {"token_ids": [v.english_ids["of"]], "type": spec.TYPE_NONE,
             "modality": "text", "meta": {}},
        ]
        self.assertEqual(decode(mixed), "Of")

    def test_sample_for_report(self):
        # 汇报样例（非断言内容正确性之外的硬编码，仅验证可解码还原）
        self.assertRoundtrip("Hello，世界 123")


if __name__ == "__main__":
    unittest.main()
