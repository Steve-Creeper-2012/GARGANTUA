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
        self.assertIn("base", names)
        base = next(v for v in data["vocabs"] if v["name"] == "base")
        self.assertTrue(base["is_main"])

    def test_regions(self):
        code, data = self.api.api_vocab_regions({"name": ["base"]})
        self.assertEqual(code, 200)
        self.assertEqual(data["name"], "base")
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
                {"action": "bogus", "name": "base", "token": "x"})

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

    # ---------------- fork / delete ----------------
    def test_fork_and_delete(self):
        vp.ensure_base()
        code, data = self.api.api_vocab_fork(
            {"name": "base", "new_name": "api_fork"})
        self.assertEqual(code, 200)
        self.assertEqual(data["name"], "api_fork")

        code, data = self.api.api_vocab_delete({"name": "api_fork"})
        self.assertEqual(code, 200)
        self.assertNotIn("api_fork", {v["name"] for v in data["vocabs"]})

        # main 不可删
        with self.assertRaises(ValueError):
            self.api.api_vocab_delete({"name": "base"})


if __name__ == "__main__":
    unittest.main()
