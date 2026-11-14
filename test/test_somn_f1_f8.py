#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===========================================================================
# SOMN V4 抢救性手术 · F1-F8 专项回归用例
# 分支 feat/somn-stress-test —— 与 test/somn_stress.py 的验收压测互补：
#   压测脚本负责「极端条件下的实测指标」，本文件负责「每个补丁的单元断言」。
# 运行：
#   /usr/local/bin/python3 test/test_somn_f1_f8.py -v
# ===========================================================================
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

import torch
import torch.nn as nn
import somn

D = 64


def W(seed, B=2, T=4, dim=D, dtype=torch.float32):
    return torch.randn(B, T, dim,
                       generator=torch.Generator().manual_seed(seed)).to(dtype)


class TestF1InputContract(unittest.TestCase):
    """F1：统一输入契约，非法输入立刻 raise 清晰错误。"""

    def test_01_nan_inf_rejected_and_state_untouched(self):
        m = somn.SOMNModule(D, min_score=0.0, warmup_steps=0)
        with torch.no_grad():
            m(W(1))
        before = m.layers[0].S.clone()
        for bad in (float("nan"), float("inf"), float("-inf")):
            H = W(2).clone()
            H[0, 0, 0] = bad
            with self.assertRaises(FloatingPointError) as ctx:
                with torch.no_grad():
                    m(H)
            self.assertIn("NaN/Inf", str(ctx.exception))
        self.assertTrue(torch.equal(before, m.layers[0].S))
        self.assertTrue(torch.isfinite(m.layers[0].S).all())

    def test_02_shape_and_dtype_rejected_with_clear_message(self):
        m = somn.SOMNModule(D)
        cases = [
            ("2D", torch.randn(2, D)),
            ("1D", torch.randn(D)),
            ("4D", torch.randn(1, 4, 2, D)),
            ("dim_mismatch", torch.randn(1, 4, D + 1)),
            ("T=0", torch.randn(1, 0, D)),
            ("B=0", torch.randn(0, 4, D)),
            ("int64", torch.randint(0, 3, (1, 4, D))),
        ]
        for name, H in cases:
            with self.subTest(name=name):
                with self.assertRaises((ValueError, TypeError)) as ctx:
                    with torch.no_grad():
                        m(H)
                msg = str(ctx.exception)
                self.assertTrue(any(k in msg for k in
                                   ("[B,T,D]", "维度", "空序列", "浮点")),
                                msg=f"{name}: 异常消息不清晰: {msg}")

    def test_03_read_only_also_validates(self):
        layer = somn.IndependentFWLayer(D)
        layer.S = torch.eye(D) * 0.1
        with self.assertRaises(ValueError):
            layer.read_only(torch.randn(2, D))
        H = W(3).clone()
        H[0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            layer.read_only(H)

    def test_04_constructor_asserts(self):
        with self.assertRaises(ValueError):
            somn.IndependentFWLayer(0)
        with self.assertRaises(ValueError):
            somn.IndependentFWLayer(D, eta=-0.1)
        with self.assertRaises(ValueError):
            somn.IndependentFWLayer(D, lam=1.5)
        with self.assertRaises(ValueError):
            somn.SOMNModule(-1)
        with self.assertRaises(ValueError):
            somn.EnergyElbowRouter(elbow_alpha=1.5)


class TestF2NumericalSafety(unittest.TestCase):
    """F2：除零 eps / 状态有限性 / eps 按 dtype / 近零重建 / 秩亏拦截。"""

    def test_05_empty_sequence_rejected_and_div_eps_exists(self):
        self.assertGreater(somn._DIV_EPS, 0.0)
        layer = somn.IndependentFWLayer(D)
        for shape in ((1, 0, D), (0, 4, D)):
            with self.assertRaises(ValueError):
                with torch.no_grad():
                    layer.write(torch.randn(*shape))
        self.assertIsNone(layer.S)

    def test_06_state_finite_after_many_writes(self):
        m = somn.SOMNModule(D, min_score=0.0, warmup_steps=0,
                            max_write_count=10 ** 9)
        with torch.no_grad():
            for i in range(60):
                m(W(100 + i) * 3.0)
        for l in m.layers:
            self.assertTrue(torch.isfinite(l.S).all())
            self.assertLessEqual(float(l.S.abs().max().item()), 10.0 + 1e-6)

    def test_07_eps_by_dtype_is_representable(self):
        for dt in (torch.float16, torch.bfloat16,
                   torch.float32, torch.float64):
            eps = somn._cos_eps(dt)
            self.assertGreater(float(torch.tensor(eps, dtype=dt).item()), 0.0,
                               msg=f"eps={eps} 在 {dt} 下下溢")
        self.assertGreater(somn._cos_eps(torch.float16), 1e-8)

    def test_08_near_zero_rebuild_scores_zero(self):
        r = somn.EnergyElbowRouter()
        lz = somn.IndependentFWLayer(D)
        lz.S = torch.zeros(D, D)
        self.assertEqual(r.scores(W(7), [lz]), [0.0])
        lg = somn.IndependentFWLayer(D)
        lg.S = torch.eye(D) * 1e-9
        self.assertLess(max(abs(x) for x in r.scores(W(8), [lg])), 0.05)

    def test_09_rank_deficient_split_returns_none_and_keeps_parent(self):
        u = torch.zeros(D); u[0] = 1.0
        v = torch.zeros(D); v[0] = 1.0
        layer = somn.IndependentFWLayer(D)
        layer.S = torch.outer(u, v)
        before = layer.S.clone()
        self.assertIsNone(layer.svd_split())
        self.assertTrue(torch.equal(before, layer.S))
        self.assertEqual(layer.write_count, 0)

    def test_10_full_rank_split_still_lossless(self):
        g = torch.Generator().manual_seed(7)
        S = torch.randn(D, D, generator=g) * 0.1
        layer = somn.IndependentFWLayer(D)
        layer.S = S.clone()
        child = layer.svd_split()
        self.assertIsNotNone(child)
        self.assertLess(float((S - layer.S - child.S).abs().max().item()), 1e-5)


class TestF3Bounds(unittest.TestCase):
    """F3：层池硬上限 + 每步分裂预算。"""

    def test_11_hard_max_layers_cap(self):
        self.assertEqual(somn.SOMNModule(D, max_layers=0).max_layers, 8)
        self.assertEqual(somn.SOMNModule(D, max_layers=64).max_layers, 8)
        self.assertEqual(somn.SOMNModule(D, max_layers=-1).max_layers, 8)
        self.assertEqual(somn.SOMNModule(D, max_layers=4).max_layers, 4)
        self.assertEqual(somn.SOMNModule(D).max_layers, 8)

    def test_12_one_split_per_step(self):
        m = somn.SOMNModule(D, warmup_steps=0, max_write_count=0,
                            expand_growth_threshold=0.0, min_score=0.0)
        m.layers.clear()
        for k in range(4):
            l = somn.IndependentFWLayer(D)
            l.S = torch.eye(D) * (0.1 * (k + 1))
            l.write_count = 1
            m.layers.append(l)
        before = len(m.layers)
        m._check_expand()
        self.assertEqual(len(m.layers) - before, 1)
        self.assertEqual(m.max_splits_per_step, 1)

    def test_13_1000_steps_cannot_exceed_cap(self):
        m = somn.SOMNModule(D, warmup_steps=0, max_write_count=1,
                            expand_growth_threshold=1e-12,
                            min_score=0.0, max_layers=0)
        with torch.no_grad():
            for i in range(1000):
                m(W(1000 + i, B=1, T=1))
        self.assertLessEqual(len(m.layers), 8)
        self.assertEqual(m.stats()["steps"], 1000)


class TestF4Reset(unittest.TestCase):
    """F4：reset 彻底清理缓存与旧引用。"""

    def test_14_reset_clears_router_cache_and_old_S(self):
        m = somn.SOMNModule(D, warmup_steps=0, min_score=0.0)
        with torch.no_grad():
            for i in range(6):
                m(W(200 + i))
        old_layers = m.layers
        old_S = old_layers[0].S
        self.assertIsNotNone(old_S)
        m.reset()
        self.assertEqual(m.stats()["scores"], [])
        self.assertEqual(m.stats()["boosted"], [])
        self.assertEqual(m.stats()["steps"], 0)
        self.assertEqual(len(m.layers), 1)
        self.assertIsNone(m.layers[0].S)
        self.assertIsNone(old_layers[0].S)      # 旧引用已断开
        self.assertFalse(m.layers is old_layers)


class TestF5Attach(unittest.TestCase):
    """F5：attach_somn 幂等。"""

    def test_15_idempotent_same_dim(self):
        class Root:
            pass
        root = Root()
        m1 = somn.attach_somn(root, D, warmup_steps=0)
        m2 = somn.attach_somn(root, D, warmup_steps=0)
        self.assertIs(m1, m2)
        self.assertIs(m1.router, m2.router)
        self.assertEqual(len([k for k in root._rt["somn"]
                               if k == "module"]), 1)

    def test_16_different_dim_raises(self):
        class Root:
            pass
        root = Root()
        somn.attach_somn(root, D)
        with self.assertRaises(ValueError) as ctx:
            somn.attach_somn(root, 128)
        self.assertIn("dim", str(ctx.exception).lower())


class TestF6Exploration(unittest.TestCase):
    """F6：探索机制打破永久零激活死锁。"""

    def test_17_exploration_breaks_deadlock(self):
        m = somn.SOMNModule(D, min_score=0.99, warmup_steps=0,
                            explore_every=10, max_write_count=10 ** 9)
        with torch.no_grad():
            for i in range(60):
                m(W(300 + i))
        st = m.stats()
        self.assertGreater(st["explorations"], 0)
        self.assertGreaterEqual(st["writes"][0], st["explorations"])
        self.assertLessEqual(st["idle_steps"], 10)

    def test_18_explore_can_be_disabled(self):
        m = somn.SOMNModule(D, min_score=0.99, warmup_steps=0,
                            explore_every=0)
        with torch.no_grad():
            for i in range(30):
                m(W(400 + i))
        self.assertEqual(m.stats()["explorations"], 0)


class TestF7SvdFallback(unittest.TestCase):
    """F7：SVD CPU 降级二次保护，异常信息指明污染来源。"""

    def test_19_second_failure_raises_clear_error(self):
        layer = somn.IndependentFWLayer(D)
        layer.S = torch.eye(D) * 0.1
        orig = torch.linalg.svd

        def always_fail(*args, **kwargs):
            raise RuntimeError("mock linalg failure")

        torch.linalg.svd = always_fail
        try:
            with self.assertRaises(RuntimeError) as ctx:
                layer.svd_split()
            self.assertIn("SVD 失败", str(ctx.exception))
        finally:
            torch.linalg.svd = orig
        self.assertTrue(torch.isfinite(layer.S).all())


class TestF8Precision(unittest.TestCase):
    """F8：低精度输入升 fp32 状态，输出 dtype 跟随输入。"""

    def test_20_fp16_input_keeps_fp32_state(self):
        m = somn.SOMNModule(D, min_score=0.0, warmup_steps=0)
        with torch.no_grad():
            out = m(W(500).half())
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(m.layers[0].S.dtype, torch.float32)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(m.layers[0].S).all())

    def test_21_zero_input_fp16_scores_finite(self):
        r = somn.EnergyElbowRouter()
        lz = somn.IndependentFWLayer(D)
        lz.S = torch.zeros(D, D, dtype=torch.float16)
        scores = r.scores(torch.zeros(2, 4, D, dtype=torch.float16), [lz])
        self.assertEqual(len(scores), 1)
        self.assertTrue(all(abs(x) < 1e-6 for x in scores))

    def test_22_read_only_never_downcasts(self):
        l = somn.IndependentFWLayer(D)
        l.S = torch.eye(D) * 0.1                       # fp32
        ro = l.read_only(W(600).half())
        self.assertEqual(ro.dtype, torch.float16)
        self.assertEqual(l.S.dtype, torch.float32)
        l2 = somn.IndependentFWLayer(D)
        l2.S = (torch.eye(D) * 0.1).half()            # 手工 fp16 S
        ro2 = l2.read_only(W(601))                    # fp32 输入
        self.assertEqual(ro2.dtype, torch.float32)   # 不降到 fp16 计算
        self.assertTrue(torch.isfinite(ro2).all())


class TestZeroPollution(unittest.TestCase):
    """强制验收：补丁后 state_dict 仍零 Parameter / 零 Buffer、逐位一致。"""

    def test_23_state_dict_byte_identical_after_split_and_reset(self):
        model = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        mgr = somn.attach_somn(model, D, warmup_steps=0,
                               max_write_count=2,
                               expand_growth_threshold=1e-9,
                               min_score=0.0)
        with torch.no_grad():
            for i in range(30):
                mgr(W(700 + i))
        self.assertGreater(len(mgr.layers), 1)
        mgr.reset()
        with torch.no_grad():
            for i in range(10):
                mgr(W(800 + i))
        after = model.state_dict()
        self.assertEqual(list(before), list(after))
        for k in before:
            self.assertTrue(torch.equal(before[k], after[k]), msg=k)

    def test_24_manager_has_no_registered_state(self):
        m = somn.SOMNModule(D)
        self.assertEqual(list(m.state_dict().keys()), [])
        self.assertEqual(list(m.parameters()), [])
        self.assertEqual(list(m.buffers()), [])
        self.assertEqual(len(dict(m.named_modules())), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
