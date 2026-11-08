# ===========================================================================
# token-6 / base1.3.0 契约回归（2026-11-05 建，11-07 重写）
# ---------------------------------------------------------------------------
# 取代 test_tok5.py。token-6 = 6 bit/字符、29 单字符 + 6 原子边界 + 码 63
# 终止；[] 是段边界/扩展 token，<> 是 core 机制单标记、slot 编号槽位与
# 1.3.0 新增的 typst 结构 token（开/闭对）。
# 1.2.0：53 个预定义工具全部退役；**预留槽位数由词表 config 决定**，
# base 零预留（slot_count=0）；spec.head_layout(n) 是任意 N 的布局真源。
# 1.3.0：契约头在 slot 之后、draw 之前插入 typst 段（114 构造 × 开/闭
# = 228 token），故 DRAW_BEGIN = 240+N、HEAD_SIZE = 9845+N。
# ===========================================================================
"""token-6 编码 + base1.3.0 契约头 / 动态预留槽位 的契约测试。"""

from __future__ import annotations

# ---- GARGANTUA 路径引导（与其它测试一致，独立进程跑也能 import spec）----
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

import json
import tempfile
import unittest
from pathlib import Path

import spec  # noqa: E402
from tokens import tok5, tok6, vfc  # noqa: E402
from tokens.vocab import Vocab, get_vocab  # noqa: E402

ROOT = Path(boot.project_root())


class SpecContractTest(unittest.TestCase):
    """spec.py 是唯一事实来源：位宽、码位、头部区间、ID 全部对得上。"""

    def test_bits_and_alphabet(self):
        self.assertEqual(spec.T6_BITS_PER_CHAR, 6)
        self.assertEqual(spec.T6_END_CODE, 63)
        self.assertEqual(spec.T6_ALPHABET_SIZE, 29)
        self.assertEqual(len(spec.T6_CHARS), 29)
        self.assertEqual(len(set(spec.T6_CHARS)), 29)
        for ch in "<>[]":
            self.assertIn(ch, spec.T6_CHARS)
        self.assertEqual([spec.T6_CHAR_CODE[c] for c in spec.T6_CHARS],
                         list(range(29)))
        used = set(spec.T6_CHAR_CODE.values()) | set(spec.T6_BOUND_TOKENS)
        self.assertFalse(used & set(spec.T6_RESERVED_CODES))
        self.assertNotIn(spec.T6_END_CODE, used)
        self.assertEqual(len(spec.T6_RESERVED_CODES), 28)

    def test_bound_tokens_are_bracketed(self):
        self.assertEqual(spec.T6_BOUND_TOKENS, {
            29: "[model]", 30: "[user]", 31: "[model/]", 32: "[user/]",
            33: "[sys]", 34: "[sys/]",
        })
        self.assertEqual(spec.T6_BOUND_NAME_TO_CODE,
                         {v: k for k, v in spec.T6_BOUND_TOKENS.items()})
        self.assertEqual(spec.T6_NAME_TO_BOUND_ID, {
            "[model]": spec.MODEL_ID, "[user]": spec.USER_ID,
            "[model/]": spec.MODEL_END_ID, "[user/]": spec.USER_END_ID,
            "[sys]": spec.SYS_ID, "[sys/]": spec.SYS_END_ID,
        })
        self.assertEqual(spec.T6_BOUND_ID_TO_NAME,
                         {v: k for k, v in spec.T6_NAME_TO_BOUND_ID.items()})
        for name in spec.T6_BOUND_TOKENS.values():
            self.assertTrue(name.startswith("[") and name.endswith("]"))

    def test_special_and_head_layout(self):
        # core 三个机制单标记单开一类，其后是 6 个 [] boundary
        self.assertEqual(spec.CORE_TOKENS,
                         ["<caps>", "<stop>", "<draw:end>"])
        self.assertEqual(spec.BOUNDARY_TOKENS,
                         ["[user]", "[user/]", "[model]", "[model/]",
                          "[sys]", "[sys/]"])
        self.assertEqual(spec.SPECIAL_TOKENS,
                         spec.CORE_TOKENS + spec.BOUNDARY_TOKENS)
        self.assertEqual(spec.STOP_ID, 1)
        self.assertEqual(spec.CAPS_ID, 0)
        self.assertEqual(spec.DRAW_END_ID, 2)
        self.assertEqual((spec.USER_ID, spec.USER_END_ID,
                          spec.MODEL_ID, spec.MODEL_END_ID), (3, 4, 5, 6))
        self.assertEqual((spec.SYS_ID, spec.SYS_END_ID), (7, 8))
        self.assertEqual(spec.SYSTEM_BEGIN, 9)
        self.assertEqual(spec.SLOT_BEGIN, 12)
        # 1.3.0：slot 与 draw 之间插入 228 个 typst 结构 token
        self.assertEqual(spec.DRAW_BEGIN,
                         spec.AGENT_END + spec.TYPST_SIZE)
        self.assertEqual(spec.HEAD_SIZE,
                         spec.CORE_SIZE + spec.BOUNDARY_SIZE
                         + spec.SYSTEM_SIZE + spec.SLOT_SIZE
                         + spec.TYPST_SIZE + spec.MODAL_SIZE)
        self.assertEqual(spec.NUM_TYPES, 4)
        self.assertEqual(spec.TYPE_SYS, 3)

    def test_vocab_version_constants(self):
        self.assertEqual(spec.BASE_VOCAB_NAME, "base1.3.0")
        self.assertEqual(spec.VOCAB_VERSION, "1.3.0")
        self.assertEqual(spec.TOKEN_ENCODING, "token-6")
        self.assertFalse((ROOT / "vocabs" / "base").exists())
        self.assertFalse((ROOT / "vocabs" / "base1.0.0").exists())
        self.assertFalse((ROOT / "vocabs" / "base1.1.0").exists())
        # 1.2.0 已升级归档，vocabs/ 下不再有 base1.2.0/
        self.assertFalse((ROOT / "vocabs" / "base1.2.0").exists())

    def test_code_region_registered(self):
        # 1.3.0：末尾追加 unicode_ext（3537 字符，regular）；前一区仍是 code
        self.assertEqual(spec.VISIBLE_REGION_ORDER[-1], "unicode_ext")
        self.assertEqual(spec.VISIBLE_REGION_ORDER[-2], "code")
        self.assertEqual(spec.REGION_CATEGORY["code"], "regular")
        self.assertEqual(spec.REGION_CATEGORY["unicode_ext"], "regular")
        self.assertEqual(spec.VISIBLE_REGION_ORDER[:8],
                         ("digit", "whitespace", "symbol", "emoji",
                          "cjk", "english", "extra", "fallback"))

    def test_segment_type_of(self):
        self.assertEqual(spec.segment_type_of(spec.CAPS_ID), spec.TYPE_NONE)
        self.assertEqual(spec.segment_type_of(spec.STOP_ID), spec.TYPE_NONE)
        self.assertEqual(spec.segment_type_of(spec.DRAW_END_ID),
                         spec.TYPE_NONE)
        self.assertEqual(spec.segment_type_of(spec.USER_ID), spec.TYPE_USER)
        self.assertEqual(spec.segment_type_of(spec.USER_END_ID),
                         spec.TYPE_USER)
        self.assertEqual(spec.segment_type_of(spec.MODEL_ID), spec.TYPE_MODEL)
        self.assertEqual(spec.segment_type_of(spec.SYS_ID), spec.TYPE_SYS)
        self.assertEqual(spec.segment_type_of(spec.SYS_END_ID), spec.TYPE_SYS)
        self.assertEqual(spec.segment_type_of(spec.HEAD_SIZE + 10),
                         spec.TYPE_NONE)
        self.assertEqual(spec.segment_type_of(-1), spec.TYPE_NONE)
        self.assertEqual(spec.segment_type_of("11"), spec.TYPE_NONE)

    def test_layout_is_dynamic(self):
        """spec 不钉死槽位数：head_layout(n) 是任意 N 的纯函数。"""
        self.assertEqual(spec.BASE_SLOT_COUNT, 0)
        self.assertEqual(spec.SLOT_COUNT, 0)          # 全局 config = base
        for n in (0, 1, 7, 128, 4096):
            lay = spec.head_layout(n)
            self.assertEqual(lay["slot_count"], n)
            self.assertEqual(lay["slot_begin"], 12)
            # 1.3.0：typst 段（228）夹在 slot 与 draw 之间
            self.assertEqual(lay["typst_begin"], 12 + n)
            self.assertEqual(lay["typst_size"], spec.TYPST_SIZE)
            self.assertEqual(lay["draw_begin"], 240 + n)
            self.assertEqual(lay["head_size"], 9845 + n)
            self.assertEqual(len(lay["slot_tokens"]), n)
            if n:
                self.assertEqual(lay["slot_tokens"][0], "<s0>")
                self.assertEqual(lay["slot_tokens"][-1], f"<s{n - 1}>")
        self.assertEqual(spec.HEAD_SIZE,
                         spec.head_layout(spec.SLOT_COUNT)["head_size"])
        with self.assertRaises(ValueError):
            spec.head_layout(-1)
        with self.assertRaises(ValueError):
            spec.head_layout(spec.MAX_SLOT_COUNT + 1)


class SlotLayoutTest(unittest.TestCase):
    """base 零预留；派生词表按 config 自带 N 个槽位。"""

    def test_base_has_no_slots(self):
        self.assertEqual(spec.SLOT_COUNT, 0)
        self.assertEqual(spec.SLOT_TOKENS, ())
        self.assertEqual(spec.AGENT_TOKENS, ())
        self.assertEqual(spec.AGENT_ID, {})
        self.assertEqual(spec.AGENT_NAME_BY_ID, {})
        self.assertEqual(spec.AGENT_BEGIN, spec.AGENT_END)
        self.assertEqual(spec.DRAW_BEGIN, 240)
        self.assertEqual(spec.HEAD_SIZE, 9845)
        self.assertEqual(spec.TOOL_SPECS, {})
        self.assertEqual(spec.TOOL_EXECUTABLE, frozenset())

    def test_slot_names_avoid_hex(self):
        """<sN> 的 s 不是十六进制字符，不会被 parse_items 当成 ID。"""
        names = spec.slot_tokens(8)
        self.assertEqual(names[:3], ("<s0>", "<s1>", "<s2>"))
        for name in names:
            body = name[1:-1]
            self.assertFalse(all(c in "0123456789ABCDEFabcdef" for c in body))

    def test_head_entries_parameterized(self):
        self.assertEqual(len(vfc.head_entries(0)), 9845)
        self.assertEqual(len(vfc.head_entries(128)), 9973)
        h = vfc.head_entries(4)
        self.assertEqual([e[1] for e in h[12:16]],
                         ["<s0>", "<s1>", "<s2>", "<s3>"])
        # 1.3.0：N=4 时 typst 段从 16 起（<DOC>），draw 从 244 起（<x>）
        self.assertEqual(h[16][1], "<DOC>")
        self.assertEqual(h[244][1], "<x>")
        self.assertEqual(vfc.head_size(4), 9849)
        self.assertEqual(vfc.infer_slot_count(h), 4)
        self.assertEqual(vfc.infer_slot_count(vfc.head_entries()), 0)

    def test_gen_vocab_derived_layout(self):
        """gen_vocab 生成带 N 个槽位的词表，且能被 Vocab.load 独立加载。"""
        from tokens import gen_vocab
        with tempfile.TemporaryDirectory() as d:
            old = gen_vocab.VOCABS
            gen_vocab.VOCABS = Path(d)
            try:
                r = gen_vocab.generate("derived", 8,
                                       ROOT / "vocabs" / spec.BASE_VOCAB_NAME)
                self.assertEqual(r["slot_count"], 8)
                # 1.3.0：head_size = 9845+8、draw_begin = 240+8
                self.assertEqual(r["head_size"], 9853)
                self.assertEqual(r["draw_begin"], 248)
                v = Vocab.load(Path(d) / "derived" / "vocab.vfc")
                self.assertEqual(v.slot_count, 8)
                self.assertEqual(v.head_size, 9853)
                self.assertEqual(v.draw_begin, 248)
                self.assertEqual(v.id_to_token[12], "<s0>")
                # N=8：typst 段从 20 起（<DOC>），draw 段从 248 起（<x>）
                self.assertEqual(v.id_to_token[20], "<DOC>")
                self.assertEqual(v.id_to_token[248], "<x>")
                self.assertEqual(v.region_of(248), "draw")
                cfg = json.loads((Path(d) / "derived" / "config.json")
                                 .read_text("utf-8"))
                self.assertEqual(cfg["slot_count"], 8)
                self.assertEqual(cfg["head_size"], 9853)
                # 派生词表的 .vfc 自描述：校验不依赖全局 spec 布局
                self.assertEqual(vfc.validate_vfc(
                    (Path(d) / "derived" / "vocab.vfc").read_text("utf-8")),
                    [])
            finally:
                gen_vocab.VOCABS = old

    def test_gen_vocab_rejects_bad_input(self):
        from tokens import gen_vocab
        with tempfile.TemporaryDirectory() as d:
            old = gen_vocab.VOCABS
            gen_vocab.VOCABS = Path(d)
            try:
                base = ROOT / "vocabs" / spec.BASE_VOCAB_NAME
                with self.assertRaises(ValueError):
                    gen_vocab.generate("bad/name", 4, base)
                with self.assertRaises(ValueError):
                    gen_vocab.generate("x", -1, base)
                with self.assertRaises(ValueError):
                    gen_vocab.generate(spec.BASE_VOCAB_NAME, 4, base)
            finally:
                gen_vocab.VOCABS = old


class BitstreamTest(unittest.TestCase):
    """位流：6 bit/字符、终止符、原子边界、非法输入。"""

    def test_six_bit_width(self):
        self.assertEqual(len(tok6.encode_chars("0" * 9)), 8)
        self.assertEqual(len(tok6.encode_chars("0" * 3)), 3)
        self.assertEqual(tok6.decode_chars(tok6.encode_chars("0" * 9)), "0" * 9)

    def test_all_chars_roundtrip(self):
        s = spec.T6_CHARS + "".join(spec.T6_BOUND_TOKENS.values())
        self.assertEqual(tok6.decode_chars(tok6.encode_chars(s)), s)

    def test_bound_tokens_are_atomic(self):
        names = list(spec.T6_BOUND_NAME_TO_CODE)
        s = "".join(names)
        self.assertEqual(len(tok6.encode_chars(s)), 6)
        self.assertEqual(tok6.decode_chars(tok6.encode_chars(s)), s)

    def test_terminator_and_truncation(self):
        data = tok6.encode_chars("0")
        self.assertEqual(tok6.decode_chars(data + b"\xff\xff"), "0")
        with self.assertRaises(tok6.Tok6Error):
            tok6.decode_chars(b"\x00\x01\x02")
        with self.assertRaises(tok6.Tok6Error):
            tok6.encode_chars("hello")

    def test_bitwriter(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.bin"
            with open(p, "wb") as f:
                w = tok6.BitWriter(f)
                w.write_chars("[user]{t:C732}")
                w.close()
                w.close()
            self.assertEqual(tok6.decode_chars(p.read_bytes()),
                             "[user]{t:C732}")


class GrammarTest(unittest.TestCase):
    """[] = 段边界/扩展，<> = core 单标记 / slot 槽位；双向回环。"""

    def test_ext_bracket_and_tool_angle(self):
        self.assertEqual(tok6.element_to_chars(tok6.ext(spec.STOP_ID)),
                         "[%X]" % spec.STOP_ID)
        self.assertEqual(tok6.element_to_chars(tok6.ext(spec.USER_ID)),
                         "[user]")
        self.assertEqual(tok6.element_to_chars(tok6.tool(spec.CAPS_ID)), "<0>")
        self.assertEqual(tok6.parse_items("[1]"),
                         [{"kind": "ext", "id": spec.STOP_ID}])
        self.assertEqual(tok6.parse_items("[user][sys][sys/]"), [
            {"kind": "ext", "id": spec.USER_ID},
            {"kind": "ext", "id": spec.SYS_ID},
            {"kind": "ext", "id": spec.SYS_END_ID}])
        # core 名字与 <HEX> 都解析成 tool 元素
        self.assertEqual(tok6.parse_items("<caps>"),
                         [{"kind": "tool", "id": spec.CAPS_ID}])
        self.assertEqual(tok6.parse_items("<0>"),
                         [{"kind": "tool", "id": spec.CAPS_ID}])
        self.assertEqual(tok6.parse_items("<stop>"),
                         [{"kind": "tool", "id": spec.STOP_ID}])

    def test_invalid_wrappers(self):
        for bad in ("[user", "[zzz]", "<nosuchtool>", "[5", "<5", "]",
                    "{t:GG}", "{}", "{t:1", "t:1"):
            with self.assertRaises(tok6.Tok6Error, msg=bad):
                tok6.parse_items(bad)

    def test_full_roundtrip(self):
        items = [
            tok6.ext(spec.USER_ID),
            tok6.step(tok6.text(0xABC)),
            {"kind": "tool", "id": spec.CAPS_ID},
            tok6.step(tok6.wave("0" * spec.T6_WAVE_HEX), tok6.text(0x10)),
            {"kind": "tool", "id": spec.DRAW_END_ID},
            tok6.ext(spec.USER_END_ID),
        ]
        data = tok6.dumps(items)
        self.assertEqual(tok6.loads(data), items)
        self.assertEqual(tok6.parse_display(tok6.to_display(items)), items)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.tokens"
            tok6.write_file(p, items)
            self.assertEqual(tok6.read_file(p), items)

    def test_parse_display_resolves_names(self):
        v = get_vocab()
        got = tok6.parse_display("[user]{t:C732}<caps>[stop][sys]",
                                 id_of=tok6.vocab_id_of(v))
        # [user] / {t:...} / <caps> / [stop] / [sys] 共 5 个元素
        self.assertEqual(got[0], {"kind": "ext", "id": spec.USER_ID})
        self.assertEqual(got[2], {"kind": "tool", "id": spec.CAPS_ID})
        self.assertEqual(got[3], {"kind": "ext", "id": spec.STOP_ID})
        self.assertEqual(got[4], {"kind": "ext", "id": spec.SYS_ID})
        # 1.2.0：core 不再折成 []，保持 <> 的 tool 形态（单开一类）
        core = tok6.parse_display("<caps><stop><draw:end>",
                                  id_of=tok6.vocab_id_of(v))
        self.assertEqual([it["kind"] for it in core],
                         ["tool", "tool", "tool"])
        self.assertEqual([it["id"] for it in core],
                         [spec.CAPS_ID, spec.STOP_ID, spec.DRAW_END_ID])

    def test_step_bridge_core_has_loss(self):
        """core ID 走 tool 元素（进 loss、有 KV），定型 TYPE_NONE。"""
        v = get_vocab()
        steps = [{"token_ids": [spec.CAPS_ID, spec.HEAD_SIZE + 10],
                  "type": spec.TYPE_USER, "modality": "text", "meta": {}}]
        items = tok6.from_steps(steps, vocab=v)
        self.assertEqual(items[0]["kind"], "tool")
        self.assertEqual(items[1]["kind"], "step")
        back = tok6.to_steps(tok6.loads(tok6.dumps(items)))
        self.assertEqual(back[0]["token_ids"], [spec.CAPS_ID])
        self.assertNotIn("ext", back[0]["meta"])
        self.assertTrue(back[0]["meta"].get("tool"))
        self.assertEqual(back[0]["type"], spec.TYPE_NONE)

    def test_to_steps_type_by_segment(self):
        """ext → NONE、core tool → NONE（单开一类）、普通文本步 → USER。"""
        items = [
            tok6.ext(spec.USER_ID),
            tok6.tool(spec.CAPS_ID),
            tok6.step(tok6.text(spec.HEAD_SIZE + 10)),
        ]
        steps = tok6.to_steps(tok6.loads(tok6.dumps(items)))
        self.assertEqual([s["type"] for s in steps],
                         [spec.TYPE_NONE, spec.TYPE_NONE, spec.TYPE_USER])
        self.assertTrue(steps[1]["meta"].get("tool"))
        self.assertTrue(steps[0]["meta"].get("ext"))


class VocabSyncTest(unittest.TestCase):
    """内部同步：三份 base 词表、config、.vfc 头全部一致且零预留。"""

    def _load(self, rel):
        return Vocab.load(ROOT / rel)

    def test_all_vocabs_share_new_head(self):
        head = vfc.head_entries()
        self.assertEqual(len(head), spec.HEAD_SIZE)
        # 2026-11-08：vocabs/test/ 是被 .gitignore 排除的本地可选 fixture，
        # 1.3.0 升级后未重建；存在才校验，不能硬依赖（否则干净检出必红）。
        for rel in ("service/tokens/vocab.vfc",
                    "vocabs/base1.3.0/vocab.vfc", "vocabs/test/vocab.vfc"):
            if not (ROOT / rel).is_file():
                continue
            v = self._load(rel)
            self.assertEqual(len(v), v.config.get("vocab_size", len(v)))
            for i, (_hid, text, _vis) in enumerate(head):
                self.assertEqual(v.id_to_token[i], text, (rel, i))
            self.assertEqual(v.id_to_token[spec.CAPS_ID], "<caps>")
            self.assertEqual(v.id_to_token[spec.DRAW_BEGIN], "<x>")
            self.assertEqual(v.id_to_token[spec.SYS_ID], "[sys]")
            self.assertEqual(v.slot_count, 0)
            self.assertEqual(v.head_size, spec.HEAD_SIZE)
            self.assertEqual(v.draw_begin, spec.DRAW_BEGIN)

    def test_versioned_config(self):
        g = json.loads((ROOT / "service/tokens/config.json").read_text("utf-8"))
        b = json.loads((ROOT / "vocabs/base1.3.0/config.json").read_text("utf-8"))
        for cfg in (g, b):
            self.assertEqual(cfg["name"], spec.BASE_VOCAB_NAME)
            self.assertEqual(cfg["vocab_version"], spec.VOCAB_VERSION)
            self.assertEqual(cfg["token_encoding"], spec.TOKEN_ENCODING)
            self.assertEqual(cfg["head_size"], spec.HEAD_SIZE)
            self.assertEqual(cfg["slot_count"], 0)
            self.assertEqual(sum(r["size"] for r in cfg["regions"]),
                             cfg["vocab_size"])
        self.assertEqual([r["region"] for r in g["regions"][:6]],
                         ["core", "boundary", "system", "slot", "typst",
                          "draw"])
        self.assertEqual(g["regions"][3]["size"], 0)
        self.assertEqual(g["regions"][-1]["region"], "unicode_ext")
        self.assertEqual(g["regions"][-1]["category"], "regular")
        self.assertGreater(g["regions"][-1]["size"], 300)
        self.assertEqual(g["regions"], b["regions"])
        self.assertEqual(g["vocab_size"], b["vocab_size"])

    def test_code_tokens_usable(self):
        from tokens import enctok
        v = get_vocab()
        for t in ("==", "->", "::", "?.", "??", "nullptr", "constexpr",
                  "useState", "None", "SELECT", "HashMap", "rgba",
                  "__init__", "background-color"):
            self.assertIn(t, v.regular_ids, t)
            tid = v.regular_ids[t]
            self.assertGreaterEqual(tid, spec.HEAD_SIZE)
            self.assertEqual(v.region_of(tid), "code", (t, tid))
            got, skipped = enctok.encode_text(t, v)
            self.assertFalse(skipped, (t, skipped))
            self.assertIn(tid, got, (t, got))
        self.assertEqual(enctok.encode_text("Hello", v)[0],
                         [spec.CAPS_ID, v.regular_ids["hello"]])

    def test_vfc_bracket_roundtrip(self):
        entries = vfc.head_entries()
        text = vfc.dumps_vfc(entries)
        self.assertIn(f"[user]:{spec.USER_ID:X}", text)
        self.assertIn("<caps>:0", text)
        self.assertEqual(vfc.validate_vfc(text), [])

    def test_global_vocab_name(self):
        self.assertEqual(get_vocab().name, spec.BASE_VOCAB_NAME)
        self.assertEqual(tok5.Tok5Error, tok6.Tok6Error)

    def test_core_not_in_draw_region(self):
        self.assertTrue(spec.CORE_BEGIN <= spec.DRAW_END_ID
                        < spec.CORE_END)
        self.assertNotIn(spec.DRAW_END_ID,
                         range(spec.DRAW_BEGIN, spec.HEAD_SIZE))


if __name__ == "__main__":
    unittest.main(verbosity=2)
