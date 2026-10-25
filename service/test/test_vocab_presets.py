"""vocab_presets 模块测试：预设 CRUD、软删、seed 模型、全局区间占用。

运行：cd ROOT && python -m unittest service.test.test_vocab_presets -v
前置：tokens/vocab.json 已生成；测试把 VOCABS_DIR 重定向到临时目录，
      不会污染项目真实的 vocabs/。
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
from tokens.vocab import DEL_MARKER, Vocab

# 把预设存储目录重定向到临时目录，避免污染项目真实 vocabs/
_ORIG_VOCABS_DIR = vp.VOCABS_DIR
_MODULE_TMP = None


def setUpModule():
    if not (Path(__file__).resolve().parents[2] / "service" / "tokens"
            / "vocab.json").is_file():
        raise RuntimeError("tokens/vocab.json 不存在，请先运行 build_vocab.py")
    # 模块级重定向：确保所有测试类（含 GlobalRangesTest，无自身 setUp）
    # 都写到临时目录，绝不污染项目真实的 vocabs/。
    global _MODULE_TMP
    _MODULE_TMP = Path(tempfile.mkdtemp(prefix="garg_vp_mod_"))
    vp.VOCABS_DIR = _MODULE_TMP / "vocabs"


def tearDownModule():
    global _MODULE_TMP
    vp.VOCABS_DIR = _ORIG_VOCABS_DIR
    if _MODULE_TMP is not None:
        shutil.rmtree(_MODULE_TMP, ignore_errors=True)
        _MODULE_TMP = None


class PresetLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="garg_vp_"))
        vp.VOCABS_DIR = self.tmp / "vocabs"

    def tearDown(self):
        vp.VOCABS_DIR = _ORIG_VOCABS_DIR
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- base ----
    def test_ensure_base_idempotent(self):
        vp.ensure_base()
        base_txt = vp.vocab_txt_path(vp.BASE)
        self.assertTrue(base_txt.is_file())
        n1 = len(vp.read_flat_tokens(vp.BASE))
        vp.ensure_base()                       # 二次调用不应重建/增长
        n2 = len(vp.read_flat_tokens(vp.BASE))
        self.assertEqual(n1, n2)
        self.assertGreater(n1, 0)

    # ---- create / list ----
    def test_create_and_list(self):
        vp.ensure_base()
        info = vp.create_preset("p1", source=vp.BASE)
        self.assertEqual(info["name"], "p1")
        self.assertEqual(info["source"], vp.BASE)
        names = {p["name"] for p in vp.list_presets()}
        self.assertIn("base", names)
        self.assertIn("p1", names)
        # 新预设与 base 同容量
        p1 = next(p for p in vp.list_presets() if p["name"] == "p1")
        base = next(p for p in vp.list_presets() if p["name"] == "base")
        self.assertEqual(p1["capacity"], base["capacity"])
        self.assertTrue(p1["is_base"] is False)
        self.assertTrue(base["is_base"] is True)

    def test_create_invalid_name(self):
        with self.assertRaises(ValueError):
            vp.create_preset("../evil")
        with self.assertRaises(ValueError):
            vp.create_preset("")

    def test_create_duplicate(self):
        vp.ensure_base()
        vp.create_preset("dup")
        with self.assertRaises(ValueError):
            vp.create_preset("dup")

    def test_create_from_missing_source(self):
        vp.ensure_base()
        with self.assertRaises(ValueError):
            vp.create_preset("x", source="nope")

    # ---- delete ----
    def test_delete_preset(self):
        vp.ensure_base()
        vp.create_preset("delme")
        vp.delete_preset("delme")
        names = {p["name"] for p in vp.list_presets()}
        self.assertNotIn("delme", names)

    def test_delete_base_forbidden(self):
        vp.ensure_base()
        with self.assertRaises(ValueError):
            vp.delete_preset(vp.BASE)

    # ---- add / remove (soft delete) ----
    def test_add_token(self):
        vp.ensure_base()
        vp.create_preset("addp")
        tok = "zzz_add_token_unique_987"
        res = vp.add_token("addp", tok, category="regular")
        self.assertEqual(res["token"], tok)
        self.assertEqual(res["vocab"], "addp")
        self.assertGreaterEqual(res["id"], spec.INVISIBLE_END)
        # 文件里确实多了这一行
        self.assertIn(tok, vp.read_flat_tokens("addp"))

    def test_remove_token_soft_delete(self):
        vp.ensure_base()
        vp.create_preset("remp")
        tok = "zzz_rem_token_unique_654"
        res = vp.add_token("remp", tok)
        tid = res["id"]
        used_before = vp.preset_occupancy("remp")["used"]

        rem = vp.remove_token("remp", tok)
        self.assertEqual(rem["id"], tid)               # 返回原 ID
        used_after = vp.preset_occupancy("remp")["used"]
        self.assertEqual(used_after, used_before - 1)

        # 文件里该 token 被替换为 <del:N> 占位符（保留 ID 位置）
        flat = vp.read_flat_tokens("remp")
        self.assertNotIn(tok, flat)
        self.assertTrue(any(t.startswith(DEL_MARKER) for t in flat))

        # 重新构建 Vocab：该 token 不可被查表
        v = Vocab.from_tokens(flat, vp.vocab_txt_path("remp"))
        self.assertIsNone(v.lookup(tok))
        self.assertNotIn(tok, v.token_to_id)
        # 但 ID 位置被占位符占着，后续 token 不漂移
        self.assertEqual(v.id_to_token[tid], f"{DEL_MARKER}{tid - spec.INVISIBLE_END}>")

    def test_remove_missing_token_raises(self):
        vp.ensure_base()
        vp.create_preset("remp2")
        with self.assertRaises(ValueError):
            vp.remove_token("remp2", "not_present_token")

    # ---- validate_preset ----
    def test_validate_preset(self):
        vp.ensure_base()
        self.assertEqual(vp.validate_preset(None), vp.BASE)
        self.assertEqual(vp.validate_preset(vp.BASE), vp.BASE)
        with self.assertRaises(ValueError):
            vp.validate_preset("does_not_exist")
        vp.create_preset("okp")
        self.assertEqual(vp.validate_preset("okp"), "okp")

    # ---- seed_model_vocab ----
    def test_seed_model_vocab(self):
        vp.ensure_base()
        vp.create_preset("seedp")
        tok = "zzz_seed_token_unique_321"
        vp.add_token("seedp", tok)
        model_dir = self.tmp / "models" / "M1"
        vp.seed_model_vocab(str(model_dir), "seedp")
        copied = model_dir / "vocab.txt"
        self.assertTrue(copied.is_file())
        # seed 合同：逐字节复制预设 vocab.txt（含可能的空行占位，与模型加载
        # 路径 read_model_vocab 的空行跳过逻辑一致）。
        self.assertEqual(
            vp.vocab_txt_path("seedp").read_text(encoding="utf-8"),
            copied.read_text(encoding="utf-8"))
        self.assertIn(tok, copied.read_text(encoding="utf-8").splitlines())

    def test_seed_default_is_base(self):
        vp.ensure_base()
        model_dir = self.tmp / "models" / "M2"
        vp.seed_model_vocab(str(model_dir), None)     # 缺省/base
        copied = model_dir / "vocab.txt"
        self.assertTrue(copied.is_file())
        self.assertEqual(
            vp.vocab_txt_path(vp.BASE).read_text(encoding="utf-8"),
            copied.read_text(encoding="utf-8"))


class GlobalRangesTest(unittest.TestCase):
    """全局词表区间占用（只读，不依赖 VOCABS_DIR）。"""

    def test_global_ranges(self):
        r = vp.global_ranges()
        self.assertEqual(r["vocab_size_padded"], spec.VOCAB_SIZE_PADDED)
        self.assertEqual(r["invisible_end"], spec.INVISIBLE_END)
        self.assertEqual(set(r["categories"]),
                         {"regular", "fallback", "special", "system"})
        # 12 个子区
        self.assertEqual(len(r["regions"]), 11)  # 见全局布局（无独立 fallback 子区重复）
        for reg in r["regions"]:
            self.assertIn("region", reg)
            self.assertIn("category", reg)
            self.assertEqual(reg["padded"], reg["size"])
            self.assertGreaterEqual(reg["used"], 0)
            self.assertLessEqual(reg["used"], reg["size"])
        # 分类合计 = 各子区 used 之和
        tot = r["category_totals"]
        self.assertEqual(
            tot["regular"],
            sum(x["used"] for x in r["regions"] if x["category"] == "regular"))

    def test_preset_occupancy(self):
        vp.ensure_base()
        occ = vp.preset_occupancy(vp.BASE)
        self.assertEqual(occ["capacity"],
                         spec.VOCAB_SIZE_PADDED - spec.INVISIBLE_END)
        self.assertEqual(occ["free"], occ["capacity"] - occ["used"])


if __name__ == "__main__":
    unittest.main()
