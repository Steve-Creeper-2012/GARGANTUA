"""tokens 包测试：tok-32 词表布局、构建幂等、编解码 roundtrip、
时间步结构、.tokenvector 格式与添加 token。

运行：cd ROOT && python -m unittest service.test.test_tokens -v
前置：tokens/vocab.vfc 已由 tokens/build_vocab.py 生成。
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
from tokens import vfc  # noqa: E402
from tokens.build_vocab import build_entries  # noqa: E402
from tokens.vocab import (CATEGORIES, GLOBAL_VOCAB_VFC, Vocab,  # noqa: E402
                          get_vocab)


def _tokens(words, user_added=None):
    """build_entries 的 id_to_token 视图（旧 build_tokens 的等价替身）。"""
    entries, _regions, _applied = build_entries(words, user_added)
    return [t for _i, t, _v in entries]


def setUpModule():
    if not GLOBAL_VOCAB_VFC.is_file():
        raise RuntimeError(
            f"{GLOBAL_VOCAB_VFC} 不存在，请先运行：python tokens/build_vocab.py"
        )


class VocabLayoutTest(unittest.TestCase):
    """词表布局一致性（2026-10-30 契约：分区表住在词表 config 里）。

    架构契约头（special / system / draw）的位置与大小仍由 spec 冻结；
    可见区各子区的边界不再是 spec 常数，从词表自身的分区表取。
    """

    @classmethod
    def setUpClass(cls):
        cls.v: Vocab = get_vocab()
        cls.t = cls.v.id_to_token
        cls.reg = {r["region"]: r for r in cls.v.regions}

    def test_total_size(self):
        # 新契约：不再有全局 padding；分区表必须严丝合缝盖满全表
        covered = 0
        for r in self.v.regions:
            self.assertEqual(r["begin"], covered)
            covered += r["size"]
        self.assertEqual(covered, len(self.t))
        self.assertEqual(self.v.config.get("vocab_size"), len(self.t))
        # 头部仍是架构契约
        self.assertEqual(covered - spec.HEAD_SIZE, len(self.t)
                         - spec.HEAD_SIZE)

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
        # 紧凑头部：special 区之后直接是 system，没有 reserved 空位
        self.assertNotIn("<reserved:6>", self.t[:spec.HEAD_SIZE])

    def test_system_region(self):
        # 系统 token：{ } | （时间步结构字符，按 ID 定位）
        self.assertEqual(self.t[spec.TS_OPEN_ID], spec.TS_OPEN)
        self.assertEqual(self.t[spec.TS_CLOSE_ID], spec.TS_CLOSE)
        self.assertEqual(self.t[spec.TS_SEP_ID], spec.TS_SEP)
        self.assertEqual(self.v.system_ids[spec.TS_OPEN], spec.TS_OPEN_ID)
        self.assertEqual(self.v.category_of(spec.TS_OPEN_ID), "system")
        self.assertEqual(self.v.category_of(spec.TS_SEP_ID), "system")

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
        # 紧凑头部：draw 区之后直接是可见区，没有 reserved 空位
        # MIDI 已暂时删除
        self.assertFalse(hasattr(spec, "MIDI_BEGIN"))
        self.assertEqual(self.v.category_of(b + spec.DRAW_X_OFFSET),
                         "special")
        # 布局自洽：offset 链 + 总数
        self.assertEqual(spec.DRAW_VOCAB_SIZE, 2961)
        self.assertEqual(spec.DRAW_TOKENS_PER_PRIMITIVE, 9)

    def test_digit_region(self):
        b = self.reg["digit"]["begin"]
        for d in range(10):
            self.assertEqual(self.t[b + d], str(d))

    def test_whitespace_region(self):
        b = self.reg["whitespace"]["begin"]
        for i, ch in enumerate(spec.WHITESPACE_CHARS):
            self.assertEqual(self.t[b + i], ch)

    def test_symbol_region(self):
        # 半角 ASCII 标点（P*/S* 类码点按码点升序，字母数字不在其中）
        for ch in ("!", "~", "@", "#", "$", "%", "^", "&", "*", "(", ")",
                   "-", "_", "=", "+", "[", "]", "{", "}", "\\", "|",
                   ";", ":", "'", "\"", ",", ".", "/", "<", ">", "?"):
            self.assertIn(ch, self.v.regular_ids)
        # 全角符号与全角字母数字
        for ch in ("！", "，", "。", "、", "「", "」", "　", "１", "Ａ", "ａ"):
            self.assertIn(ch, self.v.regular_ids)

    def test_emoji_region(self):
        b = self.reg["emoji"]["begin"]
        self.assertEqual(self.t[b], "\u00a9")            # ©
        self.assertIn("\U0001F600", self.v.regular_ids)   # 😀
        self.assertIn("\U0001F1E6", self.v.regular_ids)   # 🇦 区域指示符
        self.assertIn("\u200D", self.v.regular_ids)       # ZWJ
        self.assertIn("\uFE0F", self.v.regular_ids)       # VS16
        # ASCII 键帽基字符归属 SYMBOL/DIGIT 区，不进 emoji 区
        self.assertLess(self.v.regular_ids["#"], b)
        self.assertLess(self.v.regular_ids["*"], b)

    def test_cjk_boundaries(self):
        b = self.reg["cjk"]["begin"]
        self.assertEqual(self.t[b], "\u4e00")             # 一
        self.assertEqual(self.t[b + 20991], "\u9fff")
        self.assertEqual(self.t[b + 20992], "\u3400")     # 㐀
        self.assertEqual(self.t[b + 27584 - 1], "\u4dbf")   # 最后一个实字
        self.assertEqual(self.v.regular_ids["\u4e16"], b + (0x4E16 - 0x4E00))

    def test_english_region(self):
        reg = self.reg["english"]
        self.assertEqual(reg["size"], 30000)
        self.assertIn("hello", self.v.regular_ids)
        self.assertIn("world", self.v.regular_ids)
        # 单字母词不单独收录（由 fallback 字母承担）：全表唯一的 "a"/"i"
        # token 就是 fallback 字母本身，而不是 english 区的词条
        for w in ("a", "i"):
            self.assertIsNotNone(self.v.fallback_ids.get(w))
            self.assertEqual(self.v.regular_ids.get(w),
                             self.v.fallback_ids.get(w))

    def test_fallback_region(self):
        b = self.reg["fallback"]["begin"]
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
        self.assertEqual(self.v.category_of(self.reg["digit"]["begin"]),
                         "regular")
        self.assertEqual(self.v.category_of(self.reg["cjk"]["begin"]),
                         "regular")
        self.assertEqual(self.v.category_of(self.reg["fallback"]["begin"]),
                         "fallback")
        self.assertEqual(self.v.category_of(spec.TS_OPEN_ID), "system")
        self.assertEqual(self.v.category_of(spec.SPECIAL_BEGIN + 1),
                         "special")
        self.assertEqual(self.v.category_of(spec.DRAW_BEGIN), "special")

    def test_config_consistency(self):
        """新契约：config.json 是分区表的唯一事实源，必须与正文一致。"""
        cfg = self.v.config
        self.assertEqual(cfg.get("vocab_size"), len(self.t))
        regions = cfg.get("regions") or []
        self.assertEqual(sum(r["size"] for r in regions), len(self.t))


class BuildIdempotentTest(unittest.TestCase):
    """构建可重跑幂等：同一输入 → 同一 id_to_token；user_added 重放。"""

    @classmethod
    def setUpClass(cls):
        cls.reg = {r["region"]: r for r in get_vocab().regions}

    def test_build_tokens_idempotent(self):
        v = get_vocab()
        reg = self.reg["english"]
        words = [v.id_to_token[reg["begin"] + i]
                 for i in range(reg["size"])]
        t1 = _tokens(words)
        t2 = _tokens(words)
        self.assertEqual(t1, t2)
        self.assertEqual(t1, v.id_to_token)

    def test_build_tokens_small_wordlist(self):
        words = ["hello", "world", "don't"]
        t1 = _tokens(words)
        t2 = _tokens(words)
        self.assertEqual(t1, t2)
        # 新契约：分区按内容收紧，不再预留空位——全表没有 reserved
        self.assertFalse(any(x.startswith("<reserved:") for x in t1))
        # 3 个词恰好占满 english 区（各 1 行），紧跟着就是 fallback 字母
        self.assertIn("hello", t1)
        self.assertIn("don't", t1)
        self.assertEqual(t1[-26:], list(spec.FALLBACK_LETTERS))
        # 头部契约原样保留
        self.assertEqual(t1[spec.DRAW_BEGIN], "<cx:0>")
        self.assertEqual(t1[spec.HEAD_SIZE], "0")

    def test_user_added_replay(self):
        # 重建时 user_added 条目统一落进 extra 区（ID 由构建时分配）
        v = get_vocab()
        reg = self.reg["english"]
        words = [v.id_to_token[reg["begin"] + i]
                 for i in range(reg["size"])]
        t = _tokens(words, [{"text": "\u4dbf\u4dbf"},
                            {"text": "<mood>"}])
        base = _tokens(words)
        self.assertEqual(len(t), len(base) + 2)
        self.assertIn("\u4dbf\u4dbf", t)
        self.assertIn("<mood>", t)
        # 重放稳定：再建一次结果一致
        self.assertEqual(t, _tokens(
            words, [{"text": "\u4dbf\u4dbf"}, {"text": "<mood>"}]))


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
        db = {r["region"]: r for r in v.regions}["digit"]["begin"]
        ids = [st["token_ids"][0] for st in steps]
        self.assertEqual(
            ids,
            [
                v.regular_ids["\u4f60"], v.regular_ids["\u597d"],
                v.regular_ids[" "], v.regular_ids["world"],
                v.regular_ids[" "],
                db + 1, db + 2, db + 3,
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
        n = len(get_vocab())
        for bad in (["x"], [True], [-1], [n]):
            with self.assertRaises(ValueError):
                text_codec.decode_ids(bad)


class EncDecScriptTest(unittest.TestCase):
    """tok32 / enctok / dectok 脚本行为。"""

    def test_tok32_roundtrip(self):
        # tok32 是历史读取工具：32 位裸容器，上界是格式硬上限而非词表长度
        ids = [0, 1, 64, 12384, 88689, spec.MAX_VOCAB_SIZE - 1]
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
    """添加 token：两档形态、全局查重、分区同步、持久化。

    2026-10-30 契约：category 只有 regular（可显示）/ special（不可显示）
    两档——fallback 由「可显示的单个 a–z 字母」自动派生，system 属架构
    契约头不接受用户添加。全部在临时副本上进行，不碰仓库里的词表。
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="garg_vocab_"))
        self.copy = self.tmp / "vocab.vfc"
        shutil.copy(GLOBAL_VOCAB_VFC, self.copy)
        self.v = Vocab.load(self.copy)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_add_regular_and_special(self):
        cases = [("\U00020000", "regular"),   # CJK 扩展B 单字
                 ("<mood>", "special")]
        for text, cat in cases:
            tid = self.v.add_token(text, cat)
            self.assertEqual(self.v.category_of(tid), cat)
            self.assertEqual(self.v.token(tid), text)
        self.assertEqual(self.v.lookup("\U00020000"),
                         self.v.regular_ids["\U00020000"])

    def test_fallback_and_system_not_user_selectable(self):
        # fallback 是派生类别、system 是架构契约头：都不接受用户添加
        with self.assertRaises(ValueError):
            self.v.add_token("ß", "fallback")
        with self.assertRaises(ValueError):
            self.v.add_token("<ts>", "system")

    def test_single_ascii_letter_derives_fallback(self):
        # 新增可显示的单个 a–z 字母自动兼作 fallback——但 26 个字母全在
        # 基础词表里，这里借一个临时小词表验证派生规则本身
        entries = vfc.head_entries() + [(vfc.head_size() + i, t, True)
                                        for i, t in enumerate(["xy"])]
        v = Vocab.from_entries(entries, path=self.tmp / "mini.vfc")
        tid = v.add_token("z", "regular")
        self.assertEqual(v.fallback_ids["z"], tid)

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
        v2 = Vocab.load(self.copy)
        self.assertEqual(v2.special_ids["<mood>"], tid)
        # .vfc 正文里确实多了一行（尖括号形态、行号即新 ID）
        lines = self.copy.read_text(encoding="utf-8").splitlines()
        self.assertIn(f"<mood>:{tid}", lines)

    def test_len_grows_and_stays_dense(self):
        """新契约：词表必须占满——加 token 长度 +1、ID 恒连续。"""
        before = len(self.v)
        tid = self.v.add_token("\U00020000", "regular")
        self.assertEqual(len(self.v), before + 1)
        self.assertEqual([e[0] for e in self.v.to_entries()],
                         list(range(len(self.v))))
        self.assertEqual(self.v.token_to_id["\U00020000"], tid)
        # 持久化后重读，新 token 仍在同一 ID 上
        v2 = Vocab.load(self.copy)
        self.assertEqual(v2.token_to_id["\U00020000"], tid)

    def test_category_list(self):
        self.assertEqual(set(CATEGORIES),
                         {"regular", "fallback", "special", "system"})


class GtsAndTokensetTest(unittest.TestCase):
    """2026-10-25：Token 编辑器文本 ↔ .tokens 导出 + 训练流读取。"""

    def test_gts_roundtrip(self):
        from tokens import gts
        src = "{<user>}{hello}{ }{world}{<user/>}"
        ids = gts.parse_gts(src)
        # 结构：每步都被 { } 包裹，步内 | 分隔
        self.assertEqual(timestep.count_steps(ids), 5)
        steps = timestep.unwrap(ids)
        self.assertEqual(len(steps), 5)
        self.assertEqual(steps[0], [spec.SPECIAL_TOKENS.index("<user>")])
        # 导出二进制再读回，ID 完全一致
        data = tok32.encode_ids(ids)
        self.assertEqual(tok32.decode_ids(data), ids)

    def test_gts_rejects_bad_structure(self):
        from tokens import gts
        for bad in ("{a", "{}{a}", "{a}}", "a{b}"):
            with self.assertRaises(gts.GtsError, msg=bad):
                gts.parse_gts(bad)

    def test_gts_rejects_unmapped_chars(self):
        from tokens import gts
        with self.assertRaises(gts.GtsError):
            gts.parse_gts("{\U00020000}")     # 扩展 B 未收录

    def test_tokenset_read_stream(self):
        from tokens import tokenset, tok5
        items = [{"kind": "ext", "id": 1}, tok5.step(tok5.text(50994)),
                 {"kind": "ext", "id": 2}]
        tmp = tempfile.NamedTemporaryFile(suffix=".tokens", delete=False)
        try:
            tmp.write(tok5.dumps(items))
            tmp.close()
            flat, steps = tokenset.read_stream(tmp.name)
            # read_stream 只返回 token（{} 与结构字符由 step_ids 承载）；
            # read_doc 会在文档末尾追加 <stop> 时间步（教模型学会终止）
            self.assertEqual(flat, [1, 50994, 2, spec.STOP_ID])
            self.assertEqual(steps, [0, 1, 2, 3])   # 四步，一步一 token
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def test_tokenset_requires_structure(self):
        from tokens import tokenset
        tmp = tempfile.NamedTemporaryFile(suffix=".tokens", delete=False)
        try:
            tmp.write(b"\x00\x01\x02")        # 不是 token-5 位流
            tmp.close()
            with self.assertRaises(tokenset.TokensetError):
                tokenset.read_stream(tmp.name)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
