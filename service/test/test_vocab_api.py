"""TokensApi 词表/预设端点集成测试（无 HTTP，直接调 TokensApi 方法）。

运行：cd ROOT && python -m unittest service.test.test_vocab_api -v
说明：把 vocab_presets.VOCABS_DIR 重定向到临时目录，不污染项目真实
      vocabs/；global_ranges 只读全局 vocab.json（安全）。
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

    def test_ranges(self):
        code, data = self.api.api_vocab_ranges({})
        self.assertEqual(code, 200)
        self.assertIn("regions", data)
        self.assertIn("presets", data)
        self.assertIn("category_totals", data)
        names = {p["name"] for p in data["presets"]}
        self.assertIn("base", names)

    def test_ranges_with_selected(self):
        vp.ensure_base()
        vp.create_preset("sel")
        code, data = self.api.api_vocab_ranges({"preset": ["sel"]})
        self.assertEqual(code, 200)
        self.assertIn("selected_preset", data)
        self.assertEqual(data["selected_preset"]["name"], "sel")

    def test_presets_create_list_delete(self):
        # 默认列出
        code, data = self.api.api_vocab_presets({})
        self.assertEqual(code, 200)
        self.assertIn("base", {p["name"] for p in data["presets"]})

        # 创建
        code, data = self.api.api_vocab_presets(
            {"action": "create", "name": "api_p", "source": "base"})
        self.assertEqual(code, 200)
        self.assertEqual(data["name"], "api_p")

        # 创建后能在列表里看到
        code, data = self.api.api_vocab_presets({})
        self.assertIn("api_p", {p["name"] for p in data["presets"]})

        # 删除
        code, data = self.api.api_vocab_presets(
            {"action": "delete", "name": "api_p"})
        self.assertEqual(code, 200)
        self.assertNotIn("api_p", {p["name"] for p in data["presets"]})

    def test_presets_missing_name(self):
        with self.assertRaises(ValueError):
            self.api.api_vocab_presets({"action": "create"})

    def test_preset_token_add_remove(self):
        vp.ensure_base()
        vp.create_preset("tokp")
        tok = "zzz_api_tok_unique_555"
        code, data = self.api.api_vocab_preset_token(
            {"action": "add", "name": "tokp", "token": tok})
        self.assertEqual(code, 200)
        self.assertEqual(data["token"], tok)

        # 再 add 同名应冲突
        with self.assertRaises(ValueError):
            self.api.api_vocab_preset_token(
                {"action": "add", "name": "tokp", "token": tok})

        # remove
        code, data = self.api.api_vocab_preset_token(
            {"action": "remove", "name": "tokp", "token": tok})
        self.assertEqual(code, 200)
        self.assertEqual(data["id"] > 0, True)

    def test_preset_token_missing_fields(self):
        with self.assertRaises(ValueError):
            self.api.api_vocab_preset_token({"action": "add"})
        with self.assertRaises(ValueError):
            self.api.api_vocab_preset_token(
                {"action": "add", "name": "x"})


if __name__ == "__main__":
    unittest.main()
