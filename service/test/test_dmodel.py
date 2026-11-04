# ===========================================================================
# GARGANTUA v1 — service/test/test_dmodel.py：建模可选 d_model 回归
# ---------------------------------------------------------------------------
# 2026-11-04 新功能：创建模型时可选模型宽度。只改语义向量宽度，注意力
# 头数/头维、MLA 低秩秩、FFN 隐藏宽度仍按 spec 冻结。覆盖：
#   1. spec.check_d_model 的严格校验（唯一规则，所有入口共用）；
#   2. config 读取的容错与旧目录回退；
#   3. 真实小模型的形状（含 block 内部投影）与 config ↔ 权重 roundtrip；
#   4. 非法宽度在构造期就被拒绝。
# 运行：cd ROOT && python service/test/test_dmodel.py -v
# ===========================================================================
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

import os
import shutil
import tempfile
import unittest

import torch

import model as model_mod
import spec
from common import (build_model, read_config, read_d_model,
                    save_checkpoint)

_VS = 16
_D = 256


class TestDModelRules(unittest.TestCase):
    def test_check_d_model(self):
        self.assertEqual(spec.check_d_model(1024), 1024)
        self.assertEqual(spec.check_d_model(256), 256)
        self.assertEqual(spec.check_d_model("512"), 512)
        self.assertEqual(spec.check_d_model(512.0), 512)
        for bad in (True, False, 0, 127, 129, 2049, 100000,
                    1.5, "abc", "-128", None, [], {}):
            with self.assertRaises(ValueError, msg=f"应拒绝 {bad!r}"):
                spec.check_d_model(bad)
        for d in spec.D_MODEL_CHOICES:
            self.assertEqual(spec.check_d_model(d), d)

    def test_read_d_model_fallback(self):
        self.assertEqual(read_d_model({}), spec.D_MODEL)
        self.assertEqual(read_d_model({"d_model": 512}), 512)
        # 手改坏值只告警回退，不让目录报废
        self.assertEqual(read_d_model({"d_model": "abc"}), spec.D_MODEL)
        self.assertEqual(read_d_model({"d_model": 9999}), spec.D_MODEL)


class TestDModelShapes(unittest.TestCase):
    def test_shapes_follow_d_model(self):
        m = model_mod.Gargantua(n_enc_blocks=1, n_dec_blocks=1,
                                vocab_size=_VS, d_model=_D)
        self.assertEqual(m.d_model, _D)
        self.assertEqual(tuple(m.wte.weight.shape), (_VS, _D))
        self.assertEqual(tuple(m.lm_head.weight.shape), (_VS, _D))
        self.assertEqual(tuple(m.type_emb.weight.shape),
                         (spec.NUM_TYPES, _D))
        self.assertEqual(m.visual_proj.out_features, _D)
        self.assertEqual(m.visual_proj.in_features, spec.PATCH_VEC_DIM)
        self.assertEqual(m.audio_proj.out_features, _D)
        for blk in list(m.enc_blocks) + list(m.dec_blocks):
            self.assertEqual(blk.d_model, _D)
            self.assertEqual(blk.kda1.w_q.in_features, _D)
            self.assertEqual(blk.kda1.w_o.out_features, _D)
            self.assertEqual(blk.mla_csa.w_dq.in_features, _D)
            self.assertEqual(blk.mla_csa.w_o.out_features, _D)
            self.assertEqual(blk.ffn_up.in_features, _D)
            self.assertEqual(blk.ffn_down.out_features, _D)
            # 冻结部分不随 d_model 变
            self.assertEqual(blk.mla_csa.w_uq.out_features,
                             spec.N_HEADS * spec.HEAD_DIM)
            self.assertEqual(blk.ffn_down.in_features, spec.FFN_HIDDEN)
        del m

    def test_default_is_1024(self):
        m = model_mod.Gargantua(n_enc_blocks=1, n_dec_blocks=1,
                                vocab_size=_VS)
        self.assertEqual(m.d_model, spec.D_MODEL)
        self.assertEqual(tuple(m.wte.weight.shape), (_VS, spec.D_MODEL))
        del m

    def test_invalid_rejected(self):
        with self.assertRaises(ValueError):
            model_mod.Gargantua(n_enc_blocks=1, n_dec_blocks=1,
                                vocab_size=_VS, d_model=9999)
        with self.assertRaises(ValueError):
            model_mod.Block(is_decoder=False, d_model=127)


class TestDModelCheckpoint(unittest.TestCase):
    """config.json 的 d_model 是 build_model 构造形状的唯一依据。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="garg-dmodel-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_roundtrip(self):
        torch.manual_seed(7)
        m = model_mod.Gargantua(n_enc_blocks=1, n_dec_blocks=1,
                                vocab_size=_VS, d_model=_D)
        save_checkpoint(self.tmp, m, {
            "name": "narrow", "vocab_size": _VS, "d_model": m.d_model,
            "n_enc_blocks": 1, "n_dec_blocks": 1,
        })
        self.assertEqual(read_config(self.tmp)["d_model"], _D)
        got = build_model(self.tmp)
        self.assertEqual(got.d_model, _D)
        self.assertEqual(tuple(got.wte.weight.shape), (_VS, _D))
        for k, v in m.state_dict().items():
            self.assertTrue(torch.equal(got.state_dict()[k], v), k)
        del m, got

    def test_missing_field_uses_default(self):
        # 旧目录：只写 vocab 与 block 数，不带 d_model
        with open(os.path.join(self.tmp, "config.json"), "w",
                  encoding="utf-8") as f:
            f.write('{"vocab_size": 16}')
        self.assertEqual(read_d_model(read_config(self.tmp)), spec.D_MODEL)


if __name__ == "__main__":
    unittest.main()
