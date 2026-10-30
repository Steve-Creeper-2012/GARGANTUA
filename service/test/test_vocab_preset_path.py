# ===========================================================================
# GARGANTUA — service/test/test_vocab_preset_path.py：词表预设路径穿越回归
# ---------------------------------------------------------------------------
# 2026-10-30 安全修复（真实任意文件写漏洞）：本地 HTTP 服务
#   POST /api/vocab/preset/token {"action":"add","name":"../models/alpha",...}
# 此前可覆盖任意模型的 vocab.txt——name / source / --vocab 只在
# create_preset / delete_preset 校验，其余入口（add_token / remove_token /
# validate_preset / seed_model_vocab / read_flat_entries ...）直接拼路径写出。
# 修复：在 preset_dir / vocab_txt_path 最底层统一强制校验 name + realpath
# 前缀，堵住全部调用者。本测试用 tempfile 隔离 VOCABS_DIR，不污染项目
# 真实 vocabs/ 与 models/。
# 运行：cd <ROOT> && environment/venv/bin/python3 \
#           service/test/test_vocab_preset_path.py -v
# ===========================================================================
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

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import vocab_presets as vp

# 各类非法预设名：相对穿越 / 绝对路径 / . / .. / 非法字符 / 超长 / 非字符串
_BAD_NAMES = [
    "../models/alpha",
    "..",
    ".",
    "/tmp/evil",
    "a/b",
    "a.b",
    "a b",
    "a\\b",
    "",
    "x" * 65,
    None,
    123,
]


class TestPathPrimitives(unittest.TestCase):
    """vocab_txt_path / preset_dir 底层校验：非法名一律 ValueError。"""

    def test_vocab_txt_path_rejects_all_bad_names(self):
        for bad in _BAD_NAMES:
            with self.subTest(name=bad):
                with self.assertRaises(ValueError):
                    vp.vocab_txt_path(bad)

    def test_preset_dir_rejects_all_bad_names(self):
        # validate_preset 走 preset_dir 而非 vocab_txt_path，必须在这一层也拦
        for bad in _BAD_NAMES:
            with self.subTest(name=bad):
                with self.assertRaises(ValueError):
                    vp.preset_dir(bad)

    def test_non_string_raises_valueerror_not_typeerror(self):
        for bad in (None, 123, 3.14, [], {}):
            with self.subTest(name=bad), self.assertRaises(ValueError):
                vp.vocab_txt_path(bad)

    def test_legal_names_return_path_within_vocabs(self):
        base = os.path.realpath(vp.VOCABS_DIR)
        for good in ("base-1", "abc_2", "base", "A1_z-9"):
            p = vp.vocab_txt_path(good)
            self.assertEqual(p.name, "vocab.txt")
            self.assertEqual(
                os.path.commonpath([base, os.path.realpath(p)]), base,
                f"{good} 应落在 vocabs/ 内")
            d = vp.preset_dir(good)
            self.assertEqual(
                os.path.commonpath([base, os.path.realpath(d)]), base)


class TestCallersRejectTraversal(unittest.TestCase):
    """所有上层调用者传穿越名都抛 ValueError，且不触发任何写盘。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="garg_path_"))
        self._orig = vp.VOCABS_DIR
        vp.VOCABS_DIR = self.tmp / "vocabs"
        # 真实预建 base（v2，首行 magic）：之后各调用者里的 ensure_base()
        # 走 _upgrade_preset 快路径只读不写，避免干扰"未写盘"断言。
        vp.ensure_base()

    def tearDown(self):
        vp.VOCABS_DIR = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _assert_no_write(self, fn, *args):
        # 同时堵住两处写盘出口：_write_preset 用的 vp.write_flat_vocab_file，
        # 以及 Vocab._persist 用的 tokens.vocab.write_flat_vocab_file。
        with mock.patch.object(vp, "write_flat_vocab_file") as m_vp, \
                mock.patch("tokens.vocab.write_flat_vocab_file") as m_tv:
            with self.assertRaises(ValueError):
                fn(*args)
            m_vp.assert_not_called()
            m_tv.assert_not_called()

    def test_add_token(self):
        self._assert_no_write(vp.add_token, "../models/alpha", "x")

    def test_remove_token(self):
        self._assert_no_write(vp.remove_token, "../models/alpha", "x")

    def test_validate_preset(self):
        self._assert_no_write(vp.validate_preset, "../models/alpha")

    def test_preset_occupancy(self):
        self._assert_no_write(vp.preset_occupancy, "../models/alpha")

    def test_seed_model_vocab(self):
        self._assert_no_write(vp.seed_model_vocab,
                              str(self.tmp / "m"), "../models/alpha")

    def test_read_flat_entries(self):
        self._assert_no_write(vp.read_flat_entries, "../models/alpha")

    def test_read_flat_tokens(self):
        self._assert_no_write(vp.read_flat_tokens, "../models/alpha")

    def test_create_preset_with_traversal_source(self):
        self._assert_no_write(vp.create_preset, "good", "../models/alpha")

    def test_delete_preset_keeps_rejecting(self):
        # delete_preset 本就有 _valid_name，确认保留后仍拒绝穿越名
        self._assert_no_write(vp.delete_preset, "../models/alpha")

    def test_no_external_file_written(self):
        # 不依赖 mock 的真实证据：穿越落点 models/alpha/vocab.txt 不存在
        target = self.tmp / "models" / "alpha" / "vocab.txt"
        with self.assertRaises(ValueError):
            vp.add_token("../models/alpha", "x")
        self.assertFalse(target.exists())

    def test_list_presets_skips_illegal_dirs(self):
        # 配套回归防护：底层 preset_dir/vocab_txt_path 会拒非法名；vocabs/
        # 下若混入非法名目录（如手动备份 bad.name），list_presets 不应因此
        # 抛 ValueError 让列预设接口 500，而应跳过它、只返回合法预设。
        (vp.VOCABS_DIR / "bad.name").mkdir()
        (vp.VOCABS_DIR / "bad.name" / "vocab.txt").write_text(
            "#gargantua-vocab-v2\n", encoding="utf-8")
        (vp.VOCABS_DIR / "ok-1").mkdir()
        (vp.VOCABS_DIR / "ok-1" / "vocab.txt").write_text(
            "#gargantua-vocab-v2\n", encoding="utf-8")
        names = [p["name"] for p in vp.list_presets()]
        self.assertIn("ok-1", names)
        self.assertNotIn("bad.name", names)


class TestSymlinkEscape(unittest.TestCase):
    """符号链接绕过：vocabs/link -> 外部目录，name=link 必须被拒。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="garg_symlink_"))
        self._orig = vp.VOCABS_DIR
        self.vocabs = self.tmp / "vocabs"
        self.vocabs.mkdir()
        self.outside = self.tmp / "outside"
        self.outside.mkdir()
        (self.outside / "vocab.txt").write_text(
            "#gargantua-vocab-v2\n", encoding="utf-8")
        os.symlink(str(self.outside), str(self.vocabs / "link"))
        vp.VOCABS_DIR = self.vocabs

    def tearDown(self):
        vp.VOCABS_DIR = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_symlink_dir_name_rejected(self):
        with self.assertRaises(ValueError):
            vp.vocab_txt_path("link")
        with self.assertRaises(ValueError):
            vp.preset_dir("link")


if __name__ == "__main__":
    unittest.main(verbosity=2)
