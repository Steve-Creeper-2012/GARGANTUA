"""TokensApi 词表文件夹端点集成测试（无 HTTP，直接调 TokensApi 方法）。

运行：cd ROOT && python -m unittest service.test.test_vocab_api -v
说明：把 vocab_presets.VOCABS_DIR 重定向到临时目录，不污染项目真实
      vocabs/。端点清单（2026-10-30 词表契约）：
        GET  /api/vocab/list     全量词表（＋号下拉数据源）
        GET  /api/vocab/regions  单词表分区占用
        GET  /api/vocab/query    按文本查 token
        GET  /api/vocab/source   读 .vfc 原文（编辑器）
        POST /api/vocab/token    添加 / 删除 / 查询（表单模式）
        POST /api/vocab/source   .vfc 全文校验后写回（文本模式）
        POST /api/vocab/fork     Fork / 复制词表
        POST /api/vocab/delete   删除词表（main 不可删）
"""
from __future__ import annotations

# ---- GARGANTUA 路径引导 ----
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

import shutil
import tempfile
import unittest
from pathlib import Path

import spec
import vocab_presets as vp
from server_tokens import TokensApi

_ORIG_VOCABS_DIR = vp.VOCABS_DIR


class VocabApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="garg_api_"))
        vp.VOCABS_DIR = self.tmp / "vocabs"
        self.api = TokensApi()

    def tearDown(self):
        vp.VOCABS_DIR = _ORIG_VOCABS_DIR
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---------------- list / regions ----------------
    def test_list(self):
        code, data = self.api.api_vocab_list({})
        self.assertEqual(code, 200)
        self.assertIn("vocabs", data)
        self.assertIn("max_vocab_size", data)
        names = {v["name"] for v in data["vocabs"]}
        self.assertIn(vp.BASE, names)
        base = next(v for v in data["vocabs"] if v["name"] == vp.BASE)
        self.assertTrue(base["is_main"])

    def test_regions(self):
        code, data = self.api.api_vocab_regions({"name": [vp.BASE]})
        self.assertEqual(code, 200)
        self.assertEqual(data["name"], vp.BASE)
        self.assertGreater(len(data["regions"]), 0)
        self.assertEqual(data["vocab_size"],
                         data["visible_size"] + data["invisible_size"])

    # ---------------- token 增删查（表单模式）----------------
    def test_token_add_remove_query(self):
        vp.ensure_base()
        vp.create_vocab("tokp")
        tok = "zzz_api_tok_unique_555"
        code, data = self.api.api_vocab_token(
            {"action": "add", "name": "tokp", "token": tok,
             "category": "regular"})
        self.assertEqual(code, 200)
        self.assertEqual(data["token"], tok)

        # 再 add 同名应冲突
        with self.assertRaises(ValueError):
            self.api.api_vocab_token(
                {"action": "add", "name": "tokp", "token": tok})

        # query（GET 形态）
        code, data = self.api.api_vocab_query(
            {"name": ["tokp"], "token": [tok]})
        self.assertEqual(code, 200)
        self.assertTrue(data["found"])

        # remove
        code, data = self.api.api_vocab_token(
            {"action": "remove", "name": "tokp", "token": tok})
        self.assertEqual(code, 200)
        self.assertEqual(data["vocab_size"],
                         len(vp.load_vocab("tokp")))

    def test_token_missing_fields(self):
        with self.assertRaises(ValueError):
            self.api.api_vocab_token({"action": "add"})
        with self.assertRaises(ValueError):
            self.api.api_vocab_token({"action": "add", "name": "x"})
        with self.assertRaises(ValueError):
            self.api.api_vocab_token(
                {"action": "bogus", "name": vp.BASE, "token": "x"})

    # ---------------- .vfc 编辑器（文本模式）----------------
    def test_source_get_and_write(self):
        vp.ensure_base()
        vp.create_vocab("srcp")
        code, got = self.api.api_vocab_source_get({"name": ["srcp"]})
        self.assertEqual(code, 200)
        self.assertIn("#gargantua-vfc-v1", got["text"])
        n0 = got["vocab_size"]
        text = got["text"]

        # 原文原样写回（mode=this）→ 合法
        code, data = self.api.api_vocab_source(
            {"name": "srcp", "text": text, "mode": "this"})
        self.assertEqual(code, 200)
        self.assertEqual(data["vocab_size"], n0)

        # 语法错误 → ValueError（错误体带逐条行号，由 api() 包装）
        bad = text.replace('"hello"', 'hello', 1)
        if bad == text:                 # 该词表没有 hello 就造一个坏行
            bad = text + "no-colon-line\n"
        with self.assertRaises(ValueError):
            self.api.api_vocab_source(
                {"name": "srcp", "text": bad, "mode": "this"})

    # ---------------- 按子区添加（2026-11-07）----------------
    def test_region_add_policy_and_slot_roundtrip(self):
        """下拉给全 16 个子区 + 策略；契约头锁定、slot 强制 <>、文本无要求。"""
        vp.ensure_base()
        vp.create_vocab("regp")
        code, occ = self.api.api_vocab_regions({"name": ["regp"]})
        self.assertEqual(code, 200)
        self.assertEqual(len(occ["add_regions"]), len(spec.REGION_ORDER))
        self.assertEqual(set(occ["add_policy_hints"]),
                         {spec.POLICY_LOCKED, spec.POLICY_ANGLED,
                          spec.POLICY_TEXT})
        by_name = {r["region"]: r["add_policy"]
                   for r in occ["add_regions"]}
        self.assertEqual(by_name["system"], spec.POLICY_LOCKED)
        self.assertEqual(by_name["slot"], spec.POLICY_ANGLED)
        self.assertEqual(by_name["unicode_ext"], spec.POLICY_TEXT)

        # 锁定 / 未知子区：后端也要拦，不能只靠前端 disabled
        for bad in ("core", "boundary", "system", "typst", "draw", "nope"):
            with self.assertRaises(ValueError):
                self.api.api_vocab_token(
                    {"action": "add", "name": "regp", "token": "x",
                     "region": bad})

        # slot：裸名自动补 <>，名字任意；头部长度与 config 同步 +1
        code, r = self.api.api_vocab_token(
            {"action": "add", "name": "regp", "token": "mood",
             "region": "slot"})
        self.assertEqual(code, 200)
        self.assertEqual((r["token"], r["id"], r["region"]),
                         ("<mood>", 12, "slot"))
        v = vp.load_vocab("regp", fresh=True)
        self.assertEqual((v.slot_count, v.head_size),
                         (1, spec.HEAD_SIZE + 1))
        self.assertEqual((v.token(12), v.region_of(12)), ("<mood>", "slot"))
        occ2 = vp.vocab_occupancy("regp")
        self.assertEqual(occ2["invisible_size"], spec.HEAD_SIZE + 1)
        self.assertEqual(occ2["vocab_size"],
                         occ2["visible_size"] + occ2["invisible_size"])

        # 重名 / 非法形态一律拒绝
        for bad in ("mood", "<DOC>", "[user]", "</H1>", "<caps>"):
            with self.assertRaises(ValueError):
                self.api.api_vocab_token(
                    {"action": "add", "name": "regp", "token": bad,
                     "region": "slot"})

        # 文本子区：无格式要求
        for tok, rg in (("zzz_region_plain", "unicode_ext"),
                        ("<looks_like_tool>", "extra")):
            code, r = self.api.api_vocab_token(
                {"action": "add", "name": "regp", "token": tok,
                 "region": rg})
            self.assertEqual(code, 200)
            self.assertEqual((r["token"], r["region"]), (tok, rg))

        # slot token 可删，头部长度回落；旧 category 路径仍可用
        code, _ = self.api.api_vocab_token(
            {"action": "remove", "name": "regp", "token": "<mood>"})
        self.assertEqual(code, 200)
        v = vp.load_vocab("regp", fresh=True)
        self.assertEqual((v.slot_count, v.head_size), (0, spec.HEAD_SIZE))
        self.assertEqual(v.token(12), spec.TYPST_TOKENS[0])
        code, r = self.api.api_vocab_token(
            {"action": "add", "name": "regp", "token": "zzz_region_legacy",
             "category": "regular"})
        self.assertEqual(code, 200)
        self.assertEqual(r["region"], "extra")

    def test_empty_vocab_keeps_all_add_regions(self):
        """空词表的 0 长度子区不能丢：否则第二次添加会落回契约头里。"""
        vp.ensure_base()
        vp.create_empty_vocab("emptyreg")
        occ = vp.vocab_occupancy("emptyreg")
        self.assertEqual(len(occ["add_regions"]), len(spec.REGION_ORDER))
        for tok, rg in (("mood", "slot"), ("hello", "unicode_ext"),
                        ("42", "digit")):
            code, r = self.api.api_vocab_token(
                {"action": "add", "name": "emptyreg", "token": tok,
                 "region": rg})
            self.assertEqual(code, 200)
            self.assertEqual(r["region"], rg)
        v = vp.load_vocab("emptyreg", fresh=True)
        self.assertEqual((v.slot_count, v.head_size),
                         (1, spec.HEAD_SIZE + 1))
        self.assertEqual(v.token(12), "<mood>")
        # digit 在 unicode_ext 之前，新 token 紧凑插到 digit 区末尾
        self.assertEqual(v.token(v.head_size), "42")
        self.assertEqual(len(v), spec.HEAD_SIZE + 3)

    # ---------------- fork / delete ----------------
    def test_fork_and_delete(self):
        vp.ensure_base()
        code, data = self.api.api_vocab_fork(
            {"name": vp.BASE, "new_name": "api_fork"})
        self.assertEqual(code, 200)
        self.assertEqual(data["name"], "api_fork")

        code, data = self.api.api_vocab_delete({"name": "api_fork"})
        self.assertEqual(code, 200)
        self.assertNotIn("api_fork", {v["name"] for v in data["vocabs"]})

        # main 不可删
        with self.assertRaises(ValueError):
            self.api.api_vocab_delete({"name": vp.BASE})


if __name__ == "__main__":
    unittest.main()
