"""vocab_presets 模块测试：词表 CRUD、增删 token、seed 模型、占用统计。

2026-10-30 词表契约重构后的形态：
  - 词表 = vocabs/<name>/{vocab.vfc, config.json} 文件夹（不再是无上限
    的「预留槽位 + 软删占位符」——词表必须占满，删除即前移，恒无空位）；
  - base（main）禁止直接编辑/删除，改动必须先 Fork；
  - seed = vocab.vfc + config.json 整目录复制进 models/NAME/vocab/。

运行：cd ROOT && python -m unittest service.test.test_vocab_presets -v
前置：全局基础词表（service/tokens/vocab.vfc，或可迁移的 vocab.json）
      已生成；测试把 VOCABS_DIR 重定向到临时目录，不污染真实 vocabs/。
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
from tokens import vfc

# 把词表存储目录重定向到临时目录，避免污染项目真实 vocabs/
_ORIG_VOCABS_DIR = vp.VOCABS_DIR
_MODULE_TMP = None


def setUpModule():
    have_global = (
        (Path(__file__).resolve().parents[2] / "service" / "tokens"
         / "vocab.vfc").is_file()
        or (Path(__file__).resolve().parents[2] / "service" / "tokens"
            / "vocab.json").is_file())
    if not have_global:
        raise RuntimeError(
            "全局基础词表不存在，请先运行 build_vocab.py")
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
        base_txt = vp.vocab_vfc_path(vp.BASE)
        self.assertTrue(base_txt.is_file())
        self.assertTrue(vp.config_path(vp.BASE).is_file(),
                        "词表文件夹必须同时有 config.json")
        n1 = len(vp.load_vocab(vp.BASE))
        vp.ensure_base()                       # 二次调用不应重建/增长
        n2 = len(vp.load_vocab(vp.BASE))
        self.assertEqual(n1, n2)
        self.assertGreater(n1, spec.HEAD_SIZE)

    # ---- create / list ----
    def test_create_and_list(self):
        vp.ensure_base()
        info = vp.create_vocab("p1", source=vp.BASE)
        self.assertEqual(info["name"], "p1")
        self.assertEqual(info["source"], vp.BASE)
        self.assertEqual(info["vocab_size"],
                         len(vp.load_vocab(vp.BASE)))
        names = {p["name"] for p in vp.list_vocabs()}
        self.assertIn("base", names)
        self.assertIn("p1", names)
        # fork 出来的新词表与源同长（整目录复制）
        p1 = next(p for p in vp.list_vocabs() if p["name"] == "p1")
        base = next(p for p in vp.list_vocabs() if p["name"] == "base")
        self.assertEqual(p1["vocab_size"], base["vocab_size"])
        self.assertTrue(p1["is_main"] is False)
        self.assertTrue(base["is_main"] is True)

    def test_create_invalid_name(self):
        with self.assertRaises(ValueError):
            vp.create_vocab("../evil")
        with self.assertRaises(ValueError):
            vp.create_vocab("")

    def test_create_duplicate(self):
        vp.ensure_base()
        vp.create_vocab("dup")
        with self.assertRaises(ValueError):
            vp.create_vocab("dup")

    def test_create_from_missing_source(self):
        vp.ensure_base()
        with self.assertRaises(ValueError):
            vp.create_vocab("x", source="nope")

    # ---- delete ----
    def test_delete_preset(self):
        vp.ensure_base()
        vp.create_vocab("delme")
        vp.delete_vocab("delme")
        names = {p["name"] for p in vp.list_vocabs()}
        self.assertNotIn("delme", names)

    def test_delete_base_forbidden(self):
        vp.ensure_base()
        with self.assertRaises(ValueError):
            vp.delete_vocab(vp.BASE)

    # ---- add / remove（新契约：删除即前移，恒无空位）----
    def test_add_token(self):
        vp.ensure_base()
        vp.create_vocab("addp")
        tok = "zzz_add_token_unique_987"
        base_size = len(vp.load_vocab(vp.BASE))
        res = vp.add_token_to("addp", tok, category="regular")
        self.assertEqual(res["token"], tok)
        self.assertEqual(res["vocab"], "addp")
        self.assertGreaterEqual(res["id"], spec.HEAD_SIZE)
        # 词表确实长了 1，且查表能反查到同一 ID
        v = vp.load_vocab("addp")
        self.assertEqual(res["vocab_size"], base_size + 1)
        self.assertEqual(v.token_to_id[tok], res["id"])
        # 正文文件里也确实多了这一条
        entries = vfc.read_vfc(str(vp.vocab_vfc_path("addp")))
        self.assertIn(res["id"], [e[0] for e in entries])

    def test_add_token_to_base_forbidden(self):
        """main 词表不可直接编辑：加 token 必须先 Fork。"""
        vp.ensure_base()
        with self.assertRaises(ValueError):
            vp.add_token_to(vp.BASE, "zzz_should_not_land")

    def test_remove_token_no_hole(self):
        vp.ensure_base()
        vp.create_vocab("remp")
        tok = "zzz_rem_token_unique_654"
        fork_size = len(vp.load_vocab("remp"))
        res = vp.add_token_to("remp", tok)
        tid = res["id"]

        rem = vp.remove_token_from("remp", tok)
        self.assertEqual(rem["id"], tid)               # 返回被删 ID
        self.assertEqual(rem["vocab_size"], fork_size)  # 长度回落

        # 新契约：不留 <del:N> 占位符——ID 整体前移，恒无空位
        v = vp.load_vocab("remp")
        self.assertIsNone(v.token_to_id.get(tok))
        self.assertEqual(len(v), fork_size)
        entries = vfc.read_vfc(str(vp.vocab_vfc_path("remp")))
        self.assertEqual([e[0] for e in entries],
                         list(range(len(entries))), "ID 必须严格连续")
        self.assertFalse(any(e[1].startswith("<del:") for e in entries),
                         "新契约不允许软删占位符残留在正文里")

    def test_remove_missing_token_raises(self):
        vp.ensure_base()
        vp.create_vocab("remp2")
        with self.assertRaises(ValueError):
            vp.remove_token_from("remp2", "not_present_token")

    # ---- query ----
    def test_query_token(self):
        vp.ensure_base()
        vp.create_vocab("qp")
        res = vp.add_token_to("qp", "zzz_query_token_123")
        q = vp.query_token("qp", "zzz_query_token_123")
        self.assertTrue(q["found"])
        self.assertEqual(q["id"], res["id"])
        miss = vp.query_token("qp", "zzz_never_added")
        self.assertFalse(miss["found"])
        self.assertIsNone(miss["id"])

    # ---- validate_vocab ----
    def test_validate_vocab(self):
        vp.ensure_base()
        self.assertEqual(vp.validate_vocab(None), vp.BASE)
        self.assertEqual(vp.validate_vocab(vp.BASE), vp.BASE)
        with self.assertRaises(ValueError):
            vp.validate_vocab("does_not_exist")
        vp.create_vocab("okp")
        self.assertEqual(vp.validate_vocab("okp"), "okp")

    # ---- seed_model_vocab ----
    def test_seed_model_vocab(self):
        vp.ensure_base()
        vp.create_vocab("seedp")
        tok = "zzz_seed_token_unique_321"
        res = vp.add_token_to("seedp", tok)
        model_dir = self.tmp / "models" / "M1"
        vp.seed_model_vocab(str(model_dir), "seedp")
        # 新契约：vocab.vfc + config.json 两个文件都复制进 vocab/ 子目录
        vdir = model_dir / "vocab"
        self.assertTrue((vdir / "vocab.vfc").is_file())
        self.assertTrue((vdir / "config.json").is_file())
        # seed 合同：逐字节复制词表文件夹（含 .vfc 魔数头）
        self.assertEqual(
            vp.vocab_vfc_path("seedp").read_text(encoding="utf-8"),
            (vdir / "vocab.vfc").read_text(encoding="utf-8"))
        seeded = [e[1] for e in
                  vfc.read_vfc(str(vdir / "vocab.vfc"))]
        self.assertIn(tok, seeded)
        self.assertEqual(seeded.index(tok), res["id"])

    def test_seed_default_is_base(self):
        vp.ensure_base()
        model_dir = self.tmp / "models" / "M2"
        vp.seed_model_vocab(str(model_dir), None)     # 缺省/base
        vdir = model_dir / "vocab"
        self.assertTrue((vdir / "vocab.vfc").is_file())
        self.assertEqual(
            vp.vocab_vfc_path(vp.BASE).read_text(encoding="utf-8"),
            (vdir / "vocab.vfc").read_text(encoding="utf-8"))

    def test_seed_replaces_stale_copy(self):
        """重复 seed：旧副本整体替换，不残留上一轮的词表内容。"""
        vp.ensure_base()
        model_dir = self.tmp / "models" / "M3"
        vp.seed_model_vocab(str(model_dir), vp.BASE)
        vp.create_vocab("other")
        tok = "zzz_other_token_77"
        vp.add_token_to("other", tok)
        vp.seed_model_vocab(str(model_dir), "other")
        seeded = [e[1] for e in vfc.read_vfc(
            str(model_dir / "vocab" / "vocab.vfc"))]
        self.assertIn(tok, seeded)
        # base 独有、other 改过的内容不该残留——用长度差异验证整体替换
        self.assertEqual(len(seeded), len(vp.load_vocab("other")))


class GlobalVocabTest(unittest.TestCase):
    """全局词表区间占用（只读，不依赖 VOCABS_DIR）。"""

    def test_global_vocab_config(self):
        """全局词表 = service/tokens/{vocab.vfc, config.json}，与 base 同构。"""
        from tokens.vocab import Vocab
        g = Vocab.global_vocab()
        self.assertEqual(g.name, "base")
        self.assertEqual(g.config.get("vocab_size"), len(g))
        regions = g.config.get("regions") or []
        self.assertEqual(sum(r["size"] for r in regions), len(g))
        # 分区表覆盖全表且首尾相接
        covered = 0
        for r in g.regions:
            self.assertEqual(r["begin"], covered)
            covered += r["size"]
        self.assertEqual(covered, len(g))

    def test_vocab_occupancy(self):
        vp.ensure_base()
        occ = vp.vocab_occupancy(vp.BASE)
        self.assertEqual(occ["name"], vp.BASE)
        self.assertTrue(occ["is_main"])
        # 新契约：容量 = 可见 + 不可见（头部），不再有 free/capacity
        self.assertEqual(occ["vocab_size"],
                         occ["visible_size"] + occ["invisible_size"])
        self.assertEqual(occ["invisible_size"], spec.HEAD_SIZE)
        for reg in occ["regions"]:
            self.assertIn("samples", reg)


if __name__ == "__main__":
    unittest.main()
