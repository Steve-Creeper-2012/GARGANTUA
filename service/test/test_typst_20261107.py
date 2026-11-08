# ===========================================================================
# base1.3.0 typst 段 / unicode_ext 区 契约测试（2026-11-07 建）
# ---------------------------------------------------------------------------
# 1.3.0 在 slot 之后、draw 之前插入 typst 结构段（114 构造 × 开/闭对
# = 228 token，<DOC>…</ATTR>），并在可见区末尾追加 unicode_ext 区
# （3537 字符 / 31 个 Unicode 区块，category=regular）。本套件把这些
# 升级事实钉成契约：布局、typst 段、字符覆盖、三份词表同步、gen_vocab
# 派生对拍、typst_map.json、token-6 往返、agent 载荷/元数据。
# 风格仿 test_tok6.py：unittest + spec 为唯一事实来源。
# ===========================================================================
"""base1.3.0：typst 结构段 + unicode_ext 可见区 的契约测试。"""

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

import agent_tools  # noqa: E402
import spec  # noqa: E402
from tokens import enctok, gen_vocab, tok6, vfc  # noqa: E402
from tokens.vocab import Vocab, get_vocab  # noqa: E402

ROOT = Path(boot.project_root())

# 同步词表（旧的 vocabs/base1.2.0/ 已归档，目录不存在）。
# 2026-11-08：vocabs/test/ 是被 .gitignore 排除的**本地可选 fixture**，
# 1.3.0 升级后没有重建。它存在就一起校验，不存在不能算失败——硬依赖会让
# 套件在干净检出上永远红（前两份已覆盖 global + preset 的核心同步契约）。
_VOCAB_CANDIDATES = ("service/tokens/vocab.vfc",
                     "vocabs/base1.3.0/vocab.vfc",
                     "vocabs/test/vocab.vfc")
VOCAB_PATHS = tuple(p for p in _VOCAB_CANDIDATES if (ROOT / p).is_file())


class VersionLayoutTest(unittest.TestCase):
    """1) 版本常量与 head_layout(n) 的 typst 相关布局。"""

    def test_version_constants(self):
        self.assertEqual(spec.BASE_VOCAB_NAME, "base1.3.0")
        self.assertEqual(spec.VOCAB_VERSION, "1.3.0")
        self.assertEqual(spec.TOKEN_ENCODING, "token-6")
        self.assertEqual(spec.HEAD_SIZE, 9845)
        self.assertEqual(spec.DRAW_BEGIN, 240)
        self.assertEqual(spec.TYPST_BEGIN, 12)
        self.assertEqual(spec.TYPST_END, 240)
        self.assertEqual(spec.TYPST_SIZE, 228)
        self.assertEqual(spec.TYPST_CONSTRUCT_COUNT, 114)
        self.assertEqual(spec.MAX_VOCAB_SIZE, 100000)

    def test_head_layout_parameterized(self):
        # typst 段固定 228，draw_begin / head_size 随 N 平移
        for n in (0, 1, 7, 128, 4096):
            lay = spec.head_layout(n)
            self.assertEqual(lay["typst_begin"], 12 + n, n)
            self.assertEqual(lay["typst_size"], 228, n)
            self.assertEqual(lay["draw_begin"], 240 + n, n)
            self.assertEqual(lay["head_size"], 9845 + n, n)
            self.assertEqual(len(lay["typst_tokens"]), 228, n)
            self.assertEqual(lay["typst_tokens"][0], "<DOC>")
            self.assertEqual(lay["typst_tokens"][-1], "</ATTR>")


class TypstSegmentTest(unittest.TestCase):
    """2) typst 段的 114 构造 / 228 token 契约。"""

    def test_constructs_have_open_close_pairs(self):
        names = [c[0] for c in spec.TYPST_CONSTRUCTS]
        self.assertEqual(spec.TYPST_CONSTRUCT_COUNT, 114)
        self.assertEqual(len(names), 114)
        for name in names:
            self.assertIn(f"<{name}>", spec.TYPST_TOKENS)
            self.assertIn(f"</{name}>", spec.TYPST_TOKENS)
            # close_id == open_id + 1
            self.assertEqual(spec.TYPST_CLOSE_ID[name],
                             spec.TYPST_ID[name] + 1, name)
            self.assertEqual(spec.TYPST_NAME_BY_ID[spec.TYPST_ID[name]],
                             f"<{name}>", name)
            self.assertEqual(spec.TYPST_NAME_BY_ID[spec.TYPST_CLOSE_ID[name]],
                             f"</{name}>", name)

    def test_all_tokens_unique_and_sized(self):
        self.assertEqual(len(spec.TYPST_TOKENS), 228)
        self.assertEqual(len(set(spec.TYPST_TOKENS)), 228)
        self.assertEqual(spec.TYPST_SIZE, 228)
        self.assertEqual(spec.TYPST_BEGIN + spec.TYPST_SIZE,
                         spec.TYPST_END)
        self.assertEqual(spec.TYPST_TOKENS[0], "<DOC>")
        self.assertEqual(spec.TYPST_TOKENS[-1], "</ATTR>")

    def test_name_id_maps_are_inverse(self):
        self.assertEqual(len(spec.TYPST_TOKEN_ID), 228)
        self.assertEqual(len(spec.TYPST_NAME_BY_ID), 228)
        for tok, tid in spec.TYPST_TOKEN_ID.items():
            self.assertEqual(spec.TYPST_NAME_BY_ID[tid], tok)
        for tid, tok in spec.TYPST_NAME_BY_ID.items():
            self.assertEqual(spec.TYPST_TOKEN_ID[tok], tid)

    def test_no_name_conflict_with_other_head_segments(self):
        head = vfc.head_entries()
        typst = set(spec.TYPST_TOKENS)
        # typst 之前的 core / boundary / system / slot，之后的 draw
        pre = {e[1] for e in head[:spec.TYPST_BEGIN]}
        post = {e[1] for e in head[spec.TYPST_END:]}
        self.assertFalse(typst & pre)
        self.assertFalse(typst & post)
        self.assertFalse(typst & set(spec.CORE_TOKENS))
        self.assertFalse(typst & set(spec.BOUNDARY_TOKENS))
        # 任意 N 的 slot 名字也不与 typst 撞名
        self.assertFalse(typst & set(spec.slot_tokens(16)))

    def test_typst_by_name_fields(self):
        self.assertEqual(set(spec.TYPST_BY_NAME),
                         {c[0] for c in spec.TYPST_CONSTRUCTS})
        self.assertEqual(spec.TYPST_BY_NAME["DOC"]["name"], "DOC")
        self.assertEqual(spec.TYPST_BY_NAME["ATTR"]["name"], "ATTR")
        self.assertEqual(spec.TYPST_CONSTRUCT_FIELDS,
                         ("name", "section", "typst", "kind", "empty",
                          "attrs", "note"))

    def test_segment_type_and_vocab_visibility(self):
        for tid in (spec.TYPST_BEGIN,
                    spec.TYPST_BEGIN + spec.TYPST_SIZE // 2,
                    spec.TYPST_END - 1):
            self.assertEqual(spec.segment_type_of(tid), spec.TYPE_NONE, tid)
        v = get_vocab()
        # typst ID 全部不可显示、且绝不进 regular_ids
        for tid in range(spec.TYPST_BEGIN, spec.TYPST_END):
            self.assertTrue(v.is_invisible(tid), tid)
        for tid in (spec.TYPST_BEGIN, spec.TYPST_END - 1):
            self.assertEqual(v.region_of(tid), "typst", tid)
        self.assertFalse(set(range(spec.TYPST_BEGIN, spec.TYPST_END))
                         & set(v.regular_ids.values()))


class UnicodeExtTest(unittest.TestCase):
    """3) unicode_ext 区（3537 字符）与新表总长。"""

    @classmethod
    def setUpClass(cls):
        cls.v = get_vocab()
        cls.ue = next(r for r in cls.v.regions
                      if r["region"] == "unicode_ext")

    def test_region_and_total_size(self):
        self.assertEqual(self.ue["size"], 3537)
        self.assertEqual(self.ue["category"], "regular")
        self.assertEqual(len(self.v), 80262)
        self.assertEqual(spec.VISIBLE_REGION_ORDER[-1], "unicode_ext")
        self.assertEqual(spec.VISIBLE_REGION_ORDER[-2], "code")
        self.assertEqual(spec.REGION_CATEGORY["unicode_ext"], "regular")

    def test_sample_chars_encode_into_unicode_ext(self):
        # 用 codepoint 构造，避免源码编码问题；这些字符在旧 base1.2.0
        # 会被 enctok 跳过，1.3.0 起收录进 unicode_ext。
        # 注：≠ (U+2260) 属旧 symbol 区，不在 unicode_ext，故不列入。
        cps = [0x00E9,   # é  拉丁扩展
               0x0416,   # Ж  西里尔
               0x03B1,   # α  希腊
               0x3042,   # あ 平假名
               0x30A2,   # ア 片假名
               0x1D53D,  # 𝔽 数学字母符号
               0x00B2]   # ²  上标
        lo = self.ue["begin"]
        hi = lo + self.ue["size"]
        for cp in cps:
            ch = chr(cp)
            self.assertIn(ch, self.v.regular_ids, hex(cp))
            tid = self.v.regular_ids[ch]
            self.assertTrue(lo <= tid < hi, (hex(cp), tid))
            self.assertEqual(self.v.region_of(tid), "unicode_ext", hex(cp))
            got, skipped = enctok.encode_text(ch, self.v)
            self.assertFalse(skipped, (hex(cp), skipped))
            self.assertEqual(got, [tid], hex(cp))


class VocabSyncTest(unittest.TestCase):
    """4) 三份词表同步 + config 自洽。"""

    @staticmethod
    def _entries(rel):
        return vfc.read_vfc(str(ROOT / rel))

    def test_first_9845_head_identical(self):
        heads = [self._entries(p)[:9845] for p in VOCAB_PATHS]
        self.assertEqual(len(heads[0]), 9845)
        for i in range(1, len(heads)):
            self.assertEqual(heads[0], heads[i], VOCAB_PATHS[i])

    def test_global_and_base_config_self_consistent(self):
        for rel in ("service/tokens/config.json",
                    "vocabs/base1.3.0/config.json"):
            c = json.loads((ROOT / rel).read_text("utf-8"))
            self.assertEqual(c["name"], "base1.3.0")
            self.assertEqual(c["vocab_version"], "1.3.0")
            self.assertEqual(c["head_size"], 9845)
            self.assertEqual(c["vocab_size"], 80262)
            self.assertEqual(sum(r["size"] for r in c["regions"]),
                             c["vocab_size"])
            self.assertEqual(c["regions"][-1]["region"], "unicode_ext")
            self.assertEqual(c["regions"][-1]["size"], 3537)

    def test_global_and_base_full_table_identical(self):
        g = self._entries("service/tokens/vocab.vfc")
        b = self._entries("vocabs/base1.3.0/vocab.vfc")
        self.assertEqual(g, b)
        self.assertEqual(len(g), 80262)


class GenVocabDerivedTest(unittest.TestCase):
    """5) gen_vocab 派生对拍：typst 整体 +N、unicode_ext 仍在末尾。"""

    def test_derived_layout_shifts_typst_and_keeps_unicode_ext(self):
        n = 8
        with tempfile.TemporaryDirectory() as d:
            old = gen_vocab.VOCABS
            gen_vocab.VOCABS = Path(d)
            try:
                r = gen_vocab.generate("derived_typst", n,
                                       ROOT / "vocabs" / spec.BASE_VOCAB_NAME)
                self.assertEqual(r["slot_count"], n)
                self.assertEqual(r["head_size"], 9845 + n)
                self.assertEqual(r["draw_begin"], 240 + n)
                self.assertEqual(r["vocab_size"], 80262 + n)
                self.assertFalse(r.get("activated", True))
                v = Vocab.load(Path(d) / "derived_typst" / "vocab.vfc")
                self.assertEqual(v.slot_count, n)
                self.assertEqual(v.head_size, 9845 + n)
                self.assertEqual(v.draw_begin, 240 + n)
                # typst 段整体 +N：起点 12+N、终点 240+N
                tb = 12 + n
                self.assertEqual(v.id_to_token[tb], "<DOC>")
                self.assertEqual(v.id_to_token[tb + 1], "</DOC>")
                self.assertEqual(v.id_to_token[240 + n - 1], "</ATTR>")
                self.assertEqual(v.region_of(tb), "typst")
                self.assertEqual(v.region_of(240 + n), "draw")
                self.assertEqual(v.id_to_token[240 + n], "<x>")
                # unicode_ext 仍在末尾且 size 3537
                self.assertEqual(v.regions[-1]["region"], "unicode_ext")
                self.assertEqual(v.regions[-1]["size"], 3537)
                self.assertEqual(len(v), 80262 + n)
                cfg = json.loads(
                    (Path(d) / "derived_typst" / "config.json")
                    .read_text("utf-8"))
                self.assertEqual(cfg["head_size"], 9845 + n)
                self.assertEqual(cfg["vocab_size"], 80262 + n)
                # 派生 .vfc 自描述校验不依赖全局 spec 布局
                self.assertEqual(vfc.validate_vfc(
                    (Path(d) / "derived_typst" / "vocab.vfc")
                    .read_text("utf-8")), [])
            finally:
                gen_vocab.VOCABS = old


class TypstMapTest(unittest.TestCase):
    """6) typst_map.json 与人可读文档。"""

    def test_map_json(self):
        p = ROOT / "vocabs/base1.3.0/typst_map.json"
        self.assertTrue(p.exists())
        d = json.loads(p.read_text("utf-8"))
        self.assertEqual(d["vocab_name"], "base1.3.0")
        self.assertEqual(d["construct_count"], 114)
        self.assertEqual(d["token_count"], 228)
        toks = d["tokens"]
        self.assertEqual(len(toks), 228)
        # ID 严格覆盖 [12, 240)
        self.assertEqual([t["id"] for t in toks], list(range(12, 240)))
        by_id = {t["id"]: t for t in toks}
        for t in toks:
            # pair_id 互指
            self.assertEqual(by_id[t["pair_id"]]["pair_id"], t["id"])
            want = ("<%s>" if t["kind"] == "open" else "</%s>") \
                % t["construct"]
            self.assertEqual(t["token"], want)
            self.assertEqual(t["hex"], format(t["id"], "X"))
        lay = d["layout"]
        self.assertEqual(lay["typst_begin"], spec.TYPST_BEGIN)
        self.assertEqual(lay["typst_end"], spec.TYPST_END)
        self.assertEqual(lay["typst_size"], spec.TYPST_SIZE)
        self.assertEqual(lay["head_size"], spec.HEAD_SIZE)
        self.assertEqual(lay["draw_begin"], spec.DRAW_BEGIN)
        self.assertEqual(lay["vocab_size"], 80262)

    def test_readable_tokens_doc(self):
        md = ROOT / "service/web/plugins/TYPST_TOKENS.md"
        self.assertTrue(md.exists())
        text = md.read_text("utf-8")
        self.assertIn("<DOC>", text)
        self.assertIn("</DOC>", text)


class Token6RoundTripTest(unittest.TestCase):
    """7) token-6 往返：typst 走 tool 元素、<B> 名字优先、闭标签。"""

    def test_from_steps_tool_element_and_type_none(self):
        v = get_vocab()
        tid = spec.TYPST_ID["B"]
        steps = [{"token_ids": [tid], "type": spec.TYPE_USER,
                  "modality": "text", "meta": {}}]
        items = tok6.from_steps(steps, vocab=v)
        self.assertEqual(items[0]["kind"], "tool")
        self.assertEqual(items[0]["id"], tid)
        back = tok6.to_steps(tok6.loads(tok6.dumps(items)))
        self.assertEqual(back[0]["token_ids"], [tid])
        self.assertEqual(back[0]["type"], spec.TYPE_NONE)
        self.assertTrue(back[0]["meta"].get("tool"))

    def test_bold_name_not_parsed_as_hex(self):
        # <B>（加粗）全由十六进制字符组成，名字优先于 hex 解析，
        # 否则会被误当成 ID 0xB。
        tid = spec.TYPST_ID["B"]
        self.assertEqual(tok6.to_display([{"kind": "tool", "id": tid}]),
                         "<B>")
        self.assertEqual(tok6.parse_display("<B>"),
                         [{"kind": "tool", "id": tid}])

    def test_closed_tag_display_parse_roundtrip(self):
        open_id = spec.TYPST_ID["H1"]
        close_id = spec.TYPST_CLOSE_ID["H1"]
        pair = [{"kind": "tool", "id": open_id},
                {"kind": "tool", "id": close_id}]
        self.assertEqual(tok6.to_display(pair), "<H1></H1>")
        self.assertEqual(tok6.parse_display("</H1>"),
                         [{"kind": "tool", "id": close_id}])
        self.assertEqual(tok6.parse_display("<H1></H1>"), pair)


class AgentSpecMetaTest(unittest.TestCase):
    """8) /api/agent/spec 的 typst 字段与 token_meta 分类。"""

    def test_spec_payload_has_typst_field(self):
        p = agent_tools.agent_spec_payload()
        self.assertIn("typst", p)
        self.assertEqual(p["typst"], {
            "begin": 12, "end": 240, "size": 228,
            "construct_count": 114,
            "pair_rule": "close_id = open_id + 1",
        })

    def test_token_meta_typst(self):
        v = get_vocab()
        for tid in (spec.TYPST_ID["B"], spec.TYPST_CLOSE_ID["H1"]):
            m = agent_tools.token_meta(tid, v)
            self.assertEqual(m["char"], spec.TYPST_NAME_BY_ID[tid])
            self.assertIsNotNone(m["tool"])
            self.assertEqual(m["tool"]["category"], "typst")


if __name__ == "__main__":
    unittest.main(verbosity=2)
