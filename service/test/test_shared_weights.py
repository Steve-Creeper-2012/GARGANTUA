# ===========================================================================
# GARGANTUA v1 — service/test/test_shared_weights.py：enc/dec 权重模式回归
# ---------------------------------------------------------------------------
# 2026-11-04 新功能：创建模型时可选「分离权重」（默认）或「共享权重」。
# 共享 = enc_blocks[i] / dec_blocks[i] 逐层共用同一批 nn.Parameter（含两侧
# final norm），所以两侧必须等深。本文件覆盖：
#   1. config 解析 / block 数归一化 / HTTP 布尔入参（纯函数，秒级）；
#   2. 真实 Gargantua 的参数绑定、行为差异保留、参数去重、梯度累加；
#   3. 权重落盘只存 enc 侧一份 + 加载时镜像回完整键集合 + build_model
#      按 config 构造共享模型、不等深配置给出明确错误。
#
# 真 Gargantua 只构造一次（1+1 层、极小词表）；与规模无关的目录/IO
# 契约用同名 key 的极小替身验证，不往磁盘写几百 MB。
#
# 运行：cd ROOT && python -m unittest service.test.test_shared_weights -v
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

import inspect
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn
from safetensors.torch import load_file

import common
import model as model_mod
import server as srv
import spec
from common import (CONFIG_NAME, build_model, expand_tied_dec_keys,
                    load_model_state_dict, normalize_shared_blocks,
                    read_share_enc_dec, read_config, resolve_weights_path,
                    save_checkpoint, strip_tied_dec_keys)

_VS = 16                       # 极小词表：真模型只用来验参数绑定


class _TinyNet(nn.Module):
    """绑定区 key 与真实模型同名的极小替身（几 KB）。

    common 的分组/镜像规则按 key 前缀匹配，替身必须用
    enc_blocks. / dec_blocks. / enc_final_norm. / dec_final_norm. 这些真名。
    """

    def __init__(self, tied: bool = True):
        super().__init__()
        self.enc_blocks = nn.ModuleList([nn.Linear(4, 4)])
        self.dec_blocks = nn.ModuleList([nn.Linear(4, 4)])
        self.enc_final_norm = nn.LayerNorm(4)
        self.dec_final_norm = nn.LayerNorm(4)
        self.wte = nn.Embedding(8, 4)
        self.share_enc_dec = tied
        self.retie_calls = 0
        if tied:
            self.retie_shared_weights()

    def retie_shared_weights(self):
        self.retie_calls += 1
        if not self.share_enc_dec:
            return
        model_mod.tie_module_parameters(self.dec_blocks[0], self.enc_blocks[0])
        model_mod.tie_module_parameters(self.dec_final_norm,
                                        self.enc_final_norm)


class TestConfigRules(unittest.TestCase):
    """纯函数：配置解析、block 数归一化、HTTP 布尔入参。"""

    def test_read_share_enc_dec(self):
        self.assertFalse(read_share_enc_dec({}))            # 旧目录 = 分离
        self.assertFalse(read_share_enc_dec({"share_enc_dec": False}))
        self.assertTrue(read_share_enc_dec({"share_enc_dec": True}))
        self.assertTrue(read_share_enc_dec({"share_enc_dec": 1}))
        self.assertFalse(read_share_enc_dec({"share_enc_dec": 0}))
        self.assertTrue(read_share_enc_dec({"share_enc_dec": "TRUE"}))
        self.assertFalse(read_share_enc_dec({"share_enc_dec": "false"}))
        # 手改的怪值不能静默当成共享（否则加载侧键集合全错）
        self.assertFalse(read_share_enc_dec({"share_enc_dec": "yes"}))
        self.assertFalse(read_share_enc_dec({"share_enc_dec": 2}))
        self.assertFalse(read_share_enc_dec({"share_enc_dec": {}}))

    def test_normalize_separate_keeps_inputs(self):
        self.assertEqual(normalize_shared_blocks(None, None, False),
                         (None, None))
        self.assertEqual(normalize_shared_blocks(4, 6, False), (4, 6))
        self.assertEqual(normalize_shared_blocks(None, 9, False), (None, 9))

    def test_normalize_shared_forces_equal_depth(self):
        self.assertEqual(normalize_shared_blocks(4, 4, True), (4, 4))
        self.assertEqual(normalize_shared_blocks(3, None, True), (3, 3))
        self.assertEqual(normalize_shared_blocks(None, 5, True), (5, 5))
        self.assertEqual(normalize_shared_blocks(None, None, True),
                         (spec.SHARED_DEFAULT_BLOCKS,
                          spec.SHARED_DEFAULT_BLOCKS))
        with self.assertRaisesRegex(ValueError, "enc blocks == dec blocks"):
            normalize_shared_blocks(4, 6, True)
        with self.assertRaises(ValueError):
            normalize_shared_blocks(2, 3, True)

    def test_bool_arg_rejects_truthy_string(self):
        # bool("false") 为 True——这正是必须走严格解析的原因
        self.assertFalse(srv.App._bool_arg(None, "share_enc_dec"))
        self.assertTrue(srv.App._bool_arg(True, "share_enc_dec"))
        self.assertFalse(srv.App._bool_arg(False, "share_enc_dec"))
        self.assertTrue(srv.App._bool_arg(1, "share_enc_dec"))
        self.assertFalse(srv.App._bool_arg("false", "share_enc_dec"))
        self.assertTrue(srv.App._bool_arg(" yes ", "share_enc_dec"))
        self.assertFalse(srv.App._bool_arg("", "share_enc_dec"))
        with self.assertRaises(ValueError):
            srv.App._bool_arg("maybe", "share_enc_dec")
        with self.assertRaises(ValueError):
            srv.App._bool_arg([], "share_enc_dec")

    def test_default_is_separate(self):
        self.assertFalse(spec.SHARE_ENC_DEC)
        sig = inspect.signature(model_mod.Gargantua.__init__)
        self.assertEqual(sig.parameters["share_enc_dec"].default,
                         spec.SHARE_ENC_DEC)


class TestParameterTying(unittest.TestCase):
    """真实模型：一次构造覆盖绑定、行为、去重、梯度、retie。"""

    def test_tied_model(self):
        m = model_mod.Gargantua(n_enc_blocks=2, n_dec_blocks=2,
                                vocab_size=_VS, share_enc_dec=True)
        self.assertTrue(m.share_enc_dec)
        self.assertEqual(len(m.enc_blocks), len(m.dec_blocks))
        for enc, dec in zip(m.enc_blocks, m.dec_blocks):
            # 参数对象是同一个，不是值相等的两份
            self.assertIs(enc.kda1.w_q.weight, dec.kda1.w_q.weight)
            self.assertIs(enc.mla_csa.w_dq.weight, dec.mla_csa.w_dq.weight)
            self.assertIs(enc.ffn_up.weight, dec.ffn_up.weight)
            self.assertIs(enc.ffns[4].ffn_down.weight,
                          dec.ffns[4].ffn_down.weight)
            self.assertIs(enc.norms[0].weight, dec.norms[0].weight)
            self.assertIs(enc.norms[-1].weight, dec.norms[-1].weight)
            self.assertIs(enc.attn_res_gate, dec.attn_res_gate)
        self.assertIs(m.enc_final_norm.weight, m.dec_final_norm.weight)

        # 共享的是参数、不是 Block 实例：双向 / 因果行为必须原样保留
        for enc, dec in zip(m.enc_blocks, m.dec_blocks):
            self.assertIsNot(enc, dec)
            self.assertFalse(enc.is_decoder)
            self.assertTrue(dec.is_decoder)
            self.assertFalse(enc.mla_csa.causal)
            self.assertTrue(dec.mla_csa.causal)

        # state_dict 仍有两个前缀的 key（strict load 需要），但
        # parameters() 由 torch 去重 → 参数量只算一份
        sd = m.state_dict()
        self.assertIn("enc_blocks.0.kda1.w_q.weight", sd)
        self.assertIn("dec_blocks.0.kda1.w_q.weight", sd)
        ids = [id(p) for p in m.parameters()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLess(len(dict(m.named_parameters())), len(sd))

        # 两次使用同一个参数 → autograd 把梯度累加到同一张量
        p = m.enc_blocks[0].kda1.w_q.weight
        (p.sum() + m.dec_blocks[0].kda1.w_q.weight.sum()).backward()
        self.assertTrue(torch.equal(p.grad, torch.full_like(p, 2.0)))
        del m

    def test_unequal_depth_rejected(self):
        with self.assertRaisesRegex(ValueError, "共享权重要求"):
            model_mod.Gargantua(n_enc_blocks=2, n_dec_blocks=3,
                                vocab_size=_VS, share_enc_dec=True)
        with self.assertRaises(ValueError):
            model_mod.Gargantua(n_enc_blocks=1, n_dec_blocks=2,
                                vocab_size=_VS, share_enc_dec="false")

    def test_retie_restores_binding_after_dtype_conversion(self):
        # nn.Module._apply（model.to(dtype)）可能用新 Parameter 替换各模块
        # 的 _parameters，逐层绑定会被静默打断；retie 必须能重建。
        a = nn.Sequential(nn.Linear(4, 4))
        b = nn.Sequential(nn.Linear(4, 4))
        model_mod.tie_module_parameters(b, a)
        self.assertIs(a[0].weight, b[0].weight)
        net = nn.Sequential(a, b)
        net.to(torch.float16)
        model_mod.tie_module_parameters(b, a)
        self.assertIs(a[0].weight, b[0].weight)
        self.assertEqual(a[0].weight.dtype, torch.float16)


class TestCheckpointContract(unittest.TestCase):
    """共享模型只落一份 enc 权重、加载时镜像；旧目录行为不变。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="garg-share-")
        self.cfg = {"name": "tiny", "vocab_size": 8,
                    "n_enc_blocks": 1, "n_dec_blocks": 1}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _main_file(self):
        return resolve_weights_path(self.tmp)

    def test_shared_roundtrip(self):
        m = _TinyNet(tied=True)
        save_checkpoint(self.tmp, m, self.cfg)
        cfg = read_config(self.tmp)
        self.assertIs(cfg["share_enc_dec"], True)

        on_disk = load_file(self._main_file())
        self.assertIn("enc_blocks.0.weight", on_disk)
        self.assertIn("enc_final_norm.weight", on_disk)
        # dec 侧一个 key 都不落：safetensors 不允许共享 storage，
        # 存两份也等于把共享省下的磁盘又吃回去
        self.assertFalse(any(k.startswith("dec_") for k in on_disk),
                         f"共享模型不应落 dec 侧权重: {list(on_disk)}")

        sd = load_model_state_dict(self.tmp)
        self.assertIn("dec_blocks.0.weight", sd)
        self.assertIn("dec_final_norm.weight", sd)
        self.assertIs(sd["dec_blocks.0.weight"], sd["enc_blocks.0.weight"])
        self.assertEqual(set(sd), set(m.state_dict()))
        for k, v in m.state_dict().items():
            self.assertTrue(torch.equal(sd[k], v), k)

    def test_separate_model_unchanged(self):
        m = _TinyNet(tied=False)
        save_checkpoint(self.tmp, m, self.cfg)
        # 旧契约逐字不变：config 不凭空多字段、权重两侧齐全
        self.assertNotIn("share_enc_dec", read_config(self.tmp))
        self.assertIn("dec_blocks.0.weight", load_file(self._main_file()))
        sd = load_model_state_dict(self.tmp)
        self.assertEqual(set(sd), set(m.state_dict()))
        self.assertIsNot(sd["dec_blocks.0.weight"],
                         sd["enc_blocks.0.weight"])

    def test_strip_expand_and_conflict(self):
        m = _TinyNet(tied=True)
        stripped = strip_tied_dec_keys(m.state_dict())
        self.assertFalse(any(k.startswith("dec_") for k in stripped))
        restored = expand_tied_dec_keys(stripped)
        self.assertEqual(set(restored), set(m.state_dict()))
        # config 说共享、权重里却已有 dec 侧 → 明确报错，不静默覆盖
        with self.assertRaisesRegex(ValueError, "冲突"):
            expand_tied_dec_keys(m.state_dict())

    def test_build_model_uses_config_flag(self):
        save_checkpoint(self.tmp, _TinyNet(tied=True), self.cfg)
        made = {}

        def factory(**kw):
            made.update(kw)
            return _TinyNet(tied=bool(kw.get("share_enc_dec")))

        with mock.patch.object(model_mod, "Gargantua", factory):
            m = build_model(self.tmp)
        self.assertTrue(made["share_enc_dec"])
        self.assertEqual(made["n_enc_blocks"], made["n_dec_blocks"])
        self.assertTrue(m.share_enc_dec)
        # strict load 通过 = expand 补齐了 dec 侧 key
        self.assertGreater(m.retie_calls, 1)  # 构造时 + 精度转换后的重建

    def test_build_model_rejects_unequal_config(self):
        cfg = dict(self.cfg, share_enc_dec=True,
                   n_enc_blocks=2, n_dec_blocks=3)
        with open(os.path.join(self.tmp, CONFIG_NAME), "w",
                  encoding="utf-8") as f:
            json.dump(cfg, f)
        with self.assertRaisesRegex(ValueError, "n_enc_blocks == n_dec_blocks"):
            build_model(self.tmp)


if __name__ == "__main__":
    unittest.main()
