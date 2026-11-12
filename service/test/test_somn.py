# ===========================================================================
# GARGANTUA v4-somn — service/test/test_somn.py
# ---------------------------------------------------------------------------
# 覆盖规格「验证与调优清单」5 项 + 规格不自洽处的回归证据：
#   #1 SVD 跨平台   → CPU 侧 [1024,1024] 实测 + MPS 降级路径对拍 + fp16 升精度
#   #2 冷启动       → 100 条数据，第一层 S 成型
#   #3 分裂         → max_write_count=10 触发 SVD 拆分，层数增加
#   #4 能量截断     → Elbow / min_energy / 相关 vs 随机输入的实测行为
#   #5 零污染       → 两种挂载方式下 state_dict 逐键比对
#   额外：read_only 与 write 一致性、空激活不崩、dtype 对齐、梯度回传、reset
#
# 运行：
#   cd ROOT && /usr/local/bin/python3 service/test/test_somn.py -v
#   cd ROOT && /usr/local/bin/python3 service/test/test_somn.py --demo
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

import time
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

import somn

DIM = 64


def _batch(seed, B=2, T=4, dim=DIM, dtype=torch.float32,
           correlated=False, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    if correlated:
        base = torch.randn(1, 1, dim, generator=g)
        x = base.expand(B, T, dim).clone() \
            + 0.01 * torch.randn(B, T, dim, generator=g)
    else:
        x = torch.randn(B, T, dim, generator=g)
    return (x * scale).to(dtype)


class TestSVDSplit(unittest.TestCase):
    """验证清单 #1：SVD 无损性、跨平台降级、大矩阵、低精度。"""

    def test_lossless_split(self):
        g = torch.Generator().manual_seed(7)
        layer = somn.IndependentFWLayer(DIM, eta=0.02, lam=0.95)
        layer.S = torch.randn(DIM, DIM, generator=g) * 0.1
        layer.write_count = 37
        orig = layer.S.clone()

        new = layer.svd_split()

        err = (orig - layer.S - new.S).abs().max().item()
        self.assertLess(err, 1e-5, msg=f"无损拆分误差 {err}")
        self.assertEqual(tuple(layer.S.shape), (DIM, DIM))
        self.assertEqual(tuple(new.S.shape), (DIM, DIM))
        self.assertEqual(layer.write_count, 0)
        self.assertEqual(new.write_count, 0)
        self.assertLessEqual(torch.linalg.matrix_rank(layer.S).item(), DIM // 2)
        # 新层应继承父层超参（规格里 new_layer(dim) 用默认值，是缺陷）
        self.assertEqual(new.eta, 0.02)
        self.assertEqual(new.lam, 0.95)
        with self.assertRaises(RuntimeError):
            somn.IndependentFWLayer(DIM).svd_split()

    def test_1024_svd_on_cpu(self):
        g = torch.Generator().manual_seed(11)
        a = torch.randn(1024, 32, generator=g)
        b = torch.randn(32, 1024, generator=g)
        S = a @ b
        t0 = time.perf_counter()
        U, Sigma, Vh = torch.linalg.svd(S, full_matrices=False)
        dt = time.perf_counter() - t0
        recon = U @ torch.diag(Sigma) @ Vh
        err = (S - recon).abs().max().item()
        self.assertLess(err, 1e-3)
        print(f"\n    [1024x1024 SVD / CPU] {dt:.2f}s  max recon err={err:.2e}")
        # 本机无 MPS/CUDA，只能验证降级判定与路径，不谎称硬件实测
        self.assertTrue(somn._needs_cpu_svd(torch.device("mps")))
        self.assertFalse(somn._needs_cpu_svd(torch.device("cpu")))
        self.assertFalse(torch.cuda.is_available())
        self.assertFalse(torch.backends.mps.is_available())

    def test_mps_fallback_path_equivalence(self):
        g = torch.Generator().manual_seed(13)
        S = torch.randn(DIM, DIM, generator=g)
        a = somn.IndependentFWLayer(DIM)
        b = somn.IndependentFWLayer(DIM)
        a.S, b.S = S.clone(), S.clone()
        orig_fn = somn._needs_cpu_svd
        try:
            somn._needs_cpu_svd = lambda dev: True   # 强制走 CPU 暂存分支
            nb = b.svd_split()
        finally:
            somn._needs_cpu_svd = orig_fn
        na = a.svd_split()
        self.assertTrue(torch.allclose(a.S, b.S, atol=1e-5))
        self.assertTrue(torch.allclose(na.S, nb.S, atol=1e-5))

    def test_low_precision_svd(self):
        layer = somn.IndependentFWLayer(DIM)
        layer.S = (torch.randn(DIM, DIM) * 0.1).to(torch.float16)
        new = layer.svd_split()
        self.assertEqual(layer.S.dtype, torch.float16)
        self.assertEqual(new.S.dtype, torch.float16)


class TestColdStart(unittest.TestCase):
    """验证清单 #2：初始 1 层且 S=None，灌 100 条后第一层成型。"""

    def test_100_batches(self):
        m = somn.SOMNModule(DIM)
        self.assertEqual(len(m.layers), 1)
        self.assertIsNone(m.layers[0].S)
        with torch.no_grad():
            for i in range(100):
                out = m(_batch(i))
        self.assertIsNotNone(m.layers[0].S)
        self.assertEqual(tuple(m.layers[0].S.shape), (DIM, DIM))
        self.assertEqual(m.stats()["steps"], 100)
        self.assertEqual(m.stats()["writes"][0], 100)
        self.assertEqual(len(m.layers), 1)  # 默认 max_write_count=100 不分裂
        self.assertEqual(tuple(out.shape), (2, 4, DIM))
        self.assertTrue(torch.isfinite(out).all())
        self.assertLessEqual(m.layers[0].S.abs().max().item(), 10.0 + 1e-6)


class TestSplitGrowth(unittest.TestCase):
    """验证清单 #3：max_write_count=10 触发 SVD 拆分，层数增加。"""

    def test_split_increases_layers(self):
        m = somn.SOMNModule(DIM, max_write_count=10,
                            expand_growth_threshold=0.05, max_layers=8)
        with torch.no_grad():
            for i in range(60):
                m(_batch(i))
        st = m.stats()
        self.assertGreaterEqual(st["layers"], 2)
        self.assertGreaterEqual(st["splits"], 1)
        self.assertEqual(st["layers"], 1 + st["splits"])
        self.assertTrue(all(l.S is not None for l in m.layers))
        print(f"\n    [split] layers={st['layers']} splits={st['splits']} "
              f"writes={st['writes']}")

    def test_max_layers_cap(self):
        m = somn.SOMNModule(DIM, max_write_count=3,
                            expand_growth_threshold=0.001, max_layers=4)
        with torch.no_grad():
            for i in range(60):
                m(_batch(i))
        self.assertLessEqual(len(m.layers), 4)


class TestEnergyElbow(unittest.TestCase):
    """验证清单 #4：Elbow 截断、min_energy 跳过、两组输入的实测行为。"""

    @staticmethod
    def _make_layers(scales):
        layers = []
        for s in scales:
            layer = somn.IndependentFWLayer(DIM)
            layer.S = torch.eye(DIM) * s
            layers.append(layer)
        return layers

    def test_elbow_cutoff(self):
        layers = self._make_layers([1.0, 0.9, 0.01])
        r = somn.EnergyElbowRouter(elbow_alpha=0.6, min_energy=1e-4)
        active = r.route(_batch(3), layers)
        self.assertIn(0, active)
        self.assertIn(1, active)
        self.assertNotIn(2, active)  # 0.01 < 0.9*0.6 → 截断

    def test_min_energy_skip(self):
        layers = self._make_layers([1.0])
        r = somn.EnergyElbowRouter()
        self.assertEqual(r.route(torch.zeros(2, 4, DIM), layers), [])

    def test_correlated_vs_random(self):
        layers = self._make_layers([1.0, 0.95, 0.5, 0.2])
        r = somn.EnergyElbowRouter()
        corr = _batch(5, correlated=True)
        rand = _batch(6, correlated=False)

        def energies(H):
            return [round(torch.norm(l.read_only(H), p=2, dim=-1).mean().item(), 4)
                    for l in layers]

        e_c, e_r = energies(corr), energies(rand)
        a_c, a_r = r.route(corr, layers), r.route(rand, layers)
        print(f"\n    [correlated] energies={e_c} active={a_c}")
        print(f"    [random    ] energies={e_r} active={a_r}")
        self.assertTrue(all(0 <= i < len(layers) for i in a_c + a_r))
        # 实测结论：能量是响应范数 ||H@S||，随机输入 norm 更大，
        # 反而不会比相关输入更早被截断（规格清单 #4 的预期与实现相反）

    def test_router_does_not_measure_structure(self):
        """硬证据：同 norm、不同结构的输入，Router 给出完全相同的激活。"""
        layers = self._make_layers([1.0, 0.95, 0.5, 0.2])
        r = somn.EnergyElbowRouter()
        g = torch.Generator().manual_seed(9)
        structured = torch.randn(1, 1, DIM, generator=g).expand(2, 4, DIM).clone()
        white = torch.randn(2, 4, DIM, generator=g)
        structured = structured / structured.norm() * white.norm()
        a_s = r.route(structured, layers)
        a_w = r.route(white, layers)
        print(f"\n    [same-norm structured] active={a_s}")
        print(f"    [same-norm white     ] active={a_w}")
        self.assertEqual(a_s, a_w)


class TestZeroPollution(unittest.TestCase):
    """验证清单 #5：state_dict 里绝无 somn / layers / S 开头的键。"""

    @staticmethod
    def _baseline():
        return nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(),
                             nn.Linear(DIM, DIM))

    def test_rt_attach_keeps_state_dict(self):
        model = self._baseline()
        before = list(model.state_dict().keys())
        mgr = somn.attach_somn(model, DIM)
        with torch.no_grad():
            for i in range(3):
                mgr(_batch(i))
        self.assertEqual(list(model.state_dict().keys()), before)
        self.assertEqual(list(mgr.parameters()), [])
        self.assertEqual(list(mgr.buffers()), [])
        self.assertEqual(len(dict(mgr.named_modules())), 1)
        self.assertIs(model._rt["somn"]["layers"], mgr.layers)

    def test_even_wrong_attach_is_clean(self):
        model = self._baseline()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        model.somn = somn.SOMNModule(DIM)  # 故意误挂成 submodule
        with torch.no_grad():
            for i in range(3):
                model.somn(_batch(i))
        keys = list(model.state_dict().keys())
        self.assertFalse(any("somn" in k or "layers" in k for k in keys))
        self.assertEqual(len(keys), len(before))
        after = model.state_dict()
        for k, v in before.items():
            self.assertTrue(torch.equal(v, after[k]), msg=k)
        print(f"\n    [state_dict keys] {keys}")

    def test_split_still_no_new_keys(self):
        model = self._baseline()
        before = list(model.state_dict().keys())
        mgr = somn.attach_somn(model, DIM, max_write_count=3,
                               expand_growth_threshold=0.001, max_layers=5)
        with torch.no_grad():
            for i in range(40):
                mgr(_batch(i))
        self.assertGreater(len(mgr.layers), 1)
        self.assertEqual(list(model.state_dict().keys()), before)


class TestConsistencyAndRobustness(unittest.TestCase):
    """规格不自洽处的回归证据 + 边界条件。"""

    def test_read_matches_write_and_spec_gap(self):
        g = torch.Generator().manual_seed(21)
        layer = somn.IndependentFWLayer(DIM)
        layer.S = torch.randn(DIM, DIM, generator=g) * 0.1
        S0 = layer.S.clone()
        H = _batch(4)
        ro = layer.read_only(H)           # H @ S0
        spec_ro = F.linear(H, S0)         # 规格写法 == H @ S0.T
        out = layer.write(H)              # 写入前输出 == H @ S0
        self.assertTrue(torch.allclose(ro, out, atol=1e-6))
        gap = (spec_ro - out).abs().max().item()
        print(f"\n    [F.linear vs H@S] max gap={gap:.4f}")
        self.assertGreater(gap, 1e-3)     # 证明规格两者确实不是一回事

    def test_empty_active_no_crash(self):
        m = somn.SOMNModule(DIM)
        with torch.no_grad():
            m(_batch(1))                              # 让 S 成型
            out = m(torch.zeros(2, 4, DIM))           # route → []
        self.assertEqual(m.stats()["active_last"], [])
        self.assertTrue(torch.isfinite(out).all())

    def test_dtype_alignment(self):
        m = somn.SOMNModule(DIM)
        with torch.no_grad():
            m(_batch(1, dtype=torch.float32))
            self.assertEqual(m.layers[0].S.dtype, torch.float32)
            m(_batch(2, dtype=torch.float64))
            self.assertEqual(m.layers[0].S.dtype, torch.float64)
            out = m(_batch(3, dtype=torch.float64))
        self.assertEqual(out.dtype, torch.float64)

    def test_gradient_passthrough(self):
        m = somn.SOMNModule(DIM)
        H = _batch(8).requires_grad_(True)
        out = m(H)
        self.assertTrue(out.requires_grad)
        out.sum().backward()
        self.assertIsNotNone(H.grad)
        self.assertTrue(torch.isfinite(H.grad).all())

    def test_reset(self):
        m = somn.SOMNModule(DIM)
        with torch.no_grad():
            for i in range(5):
                m(_batch(i))
        m.reset()
        self.assertEqual(len(m.layers), 1)
        self.assertIsNone(m.layers[0].S)
        self.assertEqual(m.stats()["steps"], 0)


def run_growth_demo():
    """第一轮 100 条数据的层数增长日志（规格默认参数 + 两组对照）。"""
    print("=" * 92)
    print("SOMN v4 · 第一轮 100 条数据 · 层数增长日志")
    print(f"torch={torch.__version__}  device=cpu  "
          f"(MPS={torch.backends.mps.is_available()} / "
          f"CUDA={torch.cuda.is_available()})")
    print("=" * 92)

    def run(title, **cfg):
        m = somn.SOMNModule(256, **cfg)
        print(f"\n--- {title} ---")
        print(f"dim=256  max_write_count={m.max_write_count}  "
              f"threshold={m.expand_growth_threshold}  "
              f"elbow_alpha={m.router.elbow_alpha}  max_layers={m.max_layers}")
        print(f"{'step':>5} {'layers':>7} {'splits':>7} {'n_active':>9} "
              f"{'writes':>28} {'max||S||':>10}")
        last = -1
        with torch.no_grad():
            for i in range(1, 101):
                m(_batch(1000 + i, dim=256, B=1, T=8))
                st = m.stats()
                if len(m.layers) != last or i in (1, 10, 25, 50, 75, 100):
                    print(f"{i:>5} {st['layers']:>7} {st['splits']:>7} "
                          f"{len(st['active_last']):>9} "
                          f"{str(st['writes'][:6]):>28} "
                          f"{max(st['energies']):>10.4f}")
                    last = st['layers']
        st = m.stats()
        print(f"  => 结束 layers={st['layers']} splits={st['splits']} "
              f"writes={st['writes']}")
        return st

    a = run("A 组：规格默认参数（原样）")
    b = run("B 组：max_write_count=10（验证清单 #3 的改法）",
            max_write_count=10)
    c = run("C 组：max_write_count=10 + threshold=0.5（放大观察增长）",
            max_write_count=10, expand_growth_threshold=0.5, max_layers=8)
    print("\n" + "=" * 92)
    print("结论：A 组 100 条内不分裂（write_count 需 > 100 且 ||S|| > 5.0）；"
          "B/C 组给出真实增长曲线。")
    print("=" * 92)
    return a, b, c


if __name__ == "__main__":
    if "--demo" in _sys.argv:
        run_growth_demo()
    else:
        unittest.main(verbosity=2)
