# ===========================================================================
# GARGANTUA v4-somn — service/test/test_somn.py  (第二版：动态演化)
# ---------------------------------------------------------------------------
# 验证清单（架构师 2026-11-12 批复）：
#   #1 SVD 跨平台 / 无损 / 交替分配能量对等
#   #2 冷启动 100 条第一层成型
#   #3 分裂 + 层数受控增长
#   #4 结构区分：弃用 ||H@S||，改 cos(H, H@S)，代码段 vs 白噪声
#   #5 零污染：state_dict 逐键逐值一致
#   #6 Warm-up 出生特权：新层拿到 writes、bonus 1→0、分裂豁免
#   #7 规格缺陷回归：x@S 一致性 / 空集保护 / dtype / 梯度 / reset
#   #8 主干接入：默认关闭零侵入、显式 attach 生效、reset_memory 联动
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
_N_CODE_DIRS = 6  # 「代码段」共享的低秩基向量数


def _basis(dim):
    g = torch.Generator().manual_seed(20261112)
    return torch.randn(_N_CODE_DIRS, dim, generator=g)


def _code(seed, B=2, T=4, dim=DIM, dtype=torch.float32):
    """高度结构化的「代码段」：所有 token 落在 6 维共享子空间内。"""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(B, T, _N_CODE_DIRS, generator=g)
    return torch.matmul(w, _basis(dim)).to(dtype)


def _noise(seed, B=2, T=4, dim=DIM, dtype=torch.float32):
    """「随机文本」：各维独立白噪声。"""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, dim, generator=g).to(dtype)


class TestSVDSplit(unittest.TestCase):
    """验证清单 #1。"""

    def _trained_S(self, steps=40):
        m = somn.SOMNModule(DIM, warmup_steps=0, max_write_count=10 ** 6)
        with torch.no_grad():
            for i in range(steps):
                m(_code(1000 + i))
        return m.layers[0].S.clone()

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
        self.assertLessEqual(torch.linalg.matrix_rank(layer.S).item(),
                             (DIM + 1) // 2)
        self.assertEqual(new.eta, 0.02)
        self.assertEqual(new.lam, 0.95)
        with self.assertRaises(RuntimeError):
            somn.IndependentFWLayer(DIM).svd_split()

    def test_alternating_split_balances_energy(self):
        """新分配：父/子能量接近 1:1；旧 Top/Bottom 分配会严重失衡。"""
        S = self._trained_S()
        layer = somn.IndependentFWLayer(DIM)
        layer.S = S.clone()
        new = layer.svd_split()
        ep, ec = layer.get_energy(), new.get_energy()
        ratio = ep / (ec + 1e-9)
        print(f"\n    [alternating] parent={ep:.4f} child={ec:.4f} "
              f"ratio={ratio:.4f}")
        self.assertGreater(ratio, 0.6)
        self.assertLess(ratio, 1.7)
        self.assertLess((S - layer.S - new.S).abs().max().item(), 1e-5)
        U, Sigma, Vh = torch.linalg.svd(S, full_matrices=False)
        k = DIM // 2
        top = torch.matmul(torch.matmul(U[:, :k], torch.diag(Sigma[:k])),
                           Vh[:k, :])
        bot = torch.matmul(torch.matmul(U[:, k:], torch.diag(Sigma[k:])),
                           Vh[k:, :])
        old_ratio = top.norm(p="fro").item() / (bot.norm(p="fro").item() + 1e-9)
        print(f"    [old top/bottom] ratio={old_ratio:.4f}")
        self.assertGreater(old_ratio, ratio)

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
        print(f"    [hardware] MPS={torch.backends.mps.is_available()} "
              f"CUDA={torch.cuda.is_available()} (本轮只实测 CPU)")
        self.assertTrue(somn._needs_cpu_svd(torch.device("mps")))
        self.assertFalse(somn._needs_cpu_svd(torch.device("cpu")))

    def test_mps_fallback_path_equivalence(self):
        g = torch.Generator().manual_seed(13)
        S = torch.randn(DIM, DIM, generator=g)
        a, b = somn.IndependentFWLayer(DIM), somn.IndependentFWLayer(DIM)
        a.S, b.S = S.clone(), S.clone()
        orig_fn = somn._needs_cpu_svd
        try:
            somn._needs_cpu_svd = lambda dev: True
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
    def test_100_batches(self):
        m = somn.SOMNModule(DIM, min_score=0.0)  # 隔离准入阈值，只测冷启动
        self.assertEqual(len(m.layers), 1)
        self.assertIsNone(m.layers[0].S)
        with torch.no_grad():
            for i in range(100):
                out = m(_noise(i))
        self.assertIsNotNone(m.layers[0].S)
        self.assertEqual(tuple(m.layers[0].S.shape), (DIM, DIM))
        self.assertEqual(m.stats()["steps"], 100)
        self.assertEqual(m.stats()["writes"][0], 100)
        self.assertEqual(len(m.layers), 1)
        self.assertEqual(tuple(out.shape), (2, 4, DIM))
        self.assertTrue(torch.isfinite(out).all())
        self.assertLessEqual(m.layers[0].S.abs().max().item(), 10.0 + 1e-6)


class TestSplitGrowth(unittest.TestCase):
    def test_split_increases_layers(self):
        m = somn.SOMNModule(DIM, warmup_steps=0, max_write_count=10,
                            expand_growth_threshold=0.05, max_layers=8,
                            min_score=0.0)
        with torch.no_grad():
            for i in range(60):
                m(_noise(i))
        st = m.stats()
        self.assertGreaterEqual(st["layers"], 2)
        self.assertGreaterEqual(st["splits"], 1)
        self.assertEqual(st["layers"], 1 + st["splits"])
        print(f"\n    [split] layers={st['layers']} splits={st['splits']} "
              f"writes={st['writes']}")

    def test_max_layers_cap(self):
        m = somn.SOMNModule(DIM, warmup_steps=0, max_write_count=3,
                            expand_growth_threshold=0.001, max_layers=4,
                            min_score=0.0)
        with torch.no_grad():
            for i in range(60):
                m(_noise(i))
        self.assertLessEqual(len(m.layers), 4)


class TestEnergyElbow(unittest.TestCase):
    """验证清单 #4：余弦重建打分 + Elbow + min_score。"""

    @staticmethod
    def _projection_layers(ranks):
        """diag 投影：cos(H, H@S) ≈ sqrt(rank/DIM)，可控且确定。"""
        layers = []
        for r in ranks:
            layer = somn.IndependentFWLayer(DIM)
            mask = torch.zeros(DIM)
            mask[:r] = 1.0
            layer.S = torch.diag(mask)
            layers.append(layer)
        return layers

    @staticmethod
    def _trained_layer(steps=40):
        m = somn.SOMNModule(DIM, warmup_steps=0, max_write_count=10 ** 6)
        with torch.no_grad():
            for i in range(steps):
                m(_code(1000 + i))
        return m.layers[0]

    def test_cosine_scores_are_structure_sensitive(self):
        layer = self._trained_layer()
        r = somn.EnergyElbowRouter()
        c = r.scores(_code(999), [layer])[0]
        n = r.scores(_noise(999), [layer])[0]
        print(f"\n    [cosine] code={c:.4f} noise={n:.4f} gap={c - n:.4f}")
        self.assertGreater(c, 0.9)
        self.assertLess(n, 0.5)
        self.assertGreater(c - n, 0.4)

    def test_same_norm_different_structure_now_differs(self):
        """v1 的硬证据反转：同 norm 不同结构，打分不再相同。"""
        layer = self._trained_layer()
        r = somn.EnergyElbowRouter()
        structured = _code(31)
        white = _noise(31)
        structured = structured / structured.norm() * white.norm()
        s1 = r.scores(structured, [layer])[0]
        s2 = r.scores(white, [layer])[0]
        print(f"\n    [same-norm] structured={s1:.4f} white={s2:.4f}")
        self.assertGreater(abs(s1 - s2), 0.3)

    def test_activation_separates_code_from_noise(self):
        layer = self._trained_layer()
        r = somn.EnergyElbowRouter(min_score=0.5)
        a_code = r.route(_code(999), [layer])
        a_noise = r.route(_noise(999), [layer])
        print(f"\n    [activation] code={a_code} noise={a_noise}")
        self.assertGreaterEqual(len(a_code), 1)
        self.assertEqual(a_noise, [])

    def test_elbow_cutoff(self):
        layers = self._projection_layers([DIM, 40, 4])
        r = somn.EnergyElbowRouter(elbow_alpha=0.6, min_score=0.5)
        scores = r.scores(_noise(3), layers)
        active = r.route(_noise(3), layers)
        print(f"\n    [elbow] scores={[round(s, 3) for s in scores]} "
              f"active={active}")
        self.assertIn(0, active)
        self.assertIn(1, active)
        self.assertNotIn(2, active)

    def test_min_score_skip(self):
        layers = self._projection_layers([4])
        r = somn.EnergyElbowRouter(min_score=0.5)
        self.assertEqual(r.route(_noise(1), layers), [])
        self.assertEqual(r.route(torch.zeros(2, 4, DIM), layers), [])

    def test_min_energy_alias_still_works(self):
        r = somn.EnergyElbowRouter(min_energy=0.7)
        self.assertEqual(r.min_score, 0.7)

    def test_empty_layer_neutral_score(self):
        layer = somn.IndependentFWLayer(DIM)
        r = somn.EnergyElbowRouter()
        self.assertEqual(r.scores(_noise(1), [layer]), [0.5])


class TestWarmup(unittest.TestCase):
    """验证清单 #6：出生特权。"""

    def test_bonus_decays_1_to_0(self):
        layer = somn.IndependentFWLayer(DIM)
        layer.begin_warmup(10)
        first = layer.birth_bonus(1.0)
        for _ in range(5):
            layer.tick_warmup()
        mid = layer.birth_bonus(1.0)
        for _ in range(5):
            layer.tick_warmup()
        self.assertAlmostEqual(first, 1.0)
        self.assertAlmostEqual(mid, 0.5)
        self.assertFalse(layer.in_warmup)
        self.assertEqual(layer.birth_bonus(1.0), 0.0)

    def test_new_layer_gets_writes_within_warmup(self):
        # 在 warm-up 窗口内检查：毕业后二次分裂会把 write_count 归零，
        # 用长窗口会掩盖「新层确实被喂过」这一事实。
        m = somn.SOMNModule(DIM, warmup_steps=20, max_write_count=3,
                            expand_growth_threshold=0.001, max_layers=3,
                            min_score=0.0)
        with torch.no_grad():
            for i in range(12):
                m(_noise(800 + i))
        st = m.stats()
        print(f"\n    [warmup] layers={st['layers']} writes={st['writes']} "
              f"warmups={st['warmups']} ages={st['warmup_ages']}")
        self.assertGreaterEqual(st["layers"], 2)
        self.assertGreater(st["writes"][1], 0)
        self.assertTrue(m.layers[1].in_warmup)
        self.assertGreaterEqual(st["warmups"], 1)

    def test_warmup_bypasses_min_score(self):
        m = somn.SOMNModule(DIM, warmup_steps=30, min_score=0.99)
        with torch.no_grad():
            m(_noise(1))
        new = somn.IndependentFWLayer(DIM)
        new.S = torch.eye(DIM) * 0.01
        new.begin_warmup(30)
        m.layers.append(new)
        with torch.no_grad():
            m(torch.zeros(2, 4, DIM))
        self.assertIn(1, m.stats()["active_last"])
        self.assertGreater(new.write_count, 0)

    def test_warmup_exempt_from_split(self):
        m = somn.SOMNModule(DIM, warmup_steps=50, max_write_count=1,
                            expand_growth_threshold=0.0001, min_score=0.0)
        with torch.no_grad():
            m(_noise(1))
        new = somn.IndependentFWLayer(DIM)
        new.S = torch.eye(DIM) * 0.1
        new.write_count = 1000
        new.begin_warmup(50)
        m.layers.append(new)
        m._check_expand()
        self.assertEqual(m.stats()["splits"], 0)
        self.assertEqual(len(m.layers), 2)
        new.warmup_age = None
        m._check_expand()
        self.assertGreater(m.stats()["splits"], 0)


class TestZeroPollution(unittest.TestCase):
    """验证清单 #5。"""

    @staticmethod
    def _baseline():
        return nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(),
                             nn.Linear(DIM, DIM))

    def test_rt_attach_keeps_state_dict(self):
        model = self._baseline()
        before = list(model.state_dict().keys())
        mgr = somn.attach_somn(model, DIM)
        self.assertIs(model._rt["somn"]["module"], mgr)
        with torch.no_grad():
            for i in range(3):
                mgr(_noise(i))
        self.assertEqual(list(model.state_dict().keys()), before)
        self.assertEqual(list(mgr.parameters()), [])
        self.assertEqual(list(mgr.buffers()), [])
        self.assertEqual(len(dict(mgr.named_modules())), 1)
        self.assertIs(model._rt["somn"]["layers"], mgr.layers)

    def test_even_wrong_attach_is_clean(self):
        model = self._baseline()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        model.somn = somn.SOMNModule(DIM)
        with torch.no_grad():
            for i in range(3):
                model.somn(_noise(i))
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
        mgr = somn.attach_somn(model, DIM, warmup_steps=5, max_write_count=3,
                               expand_growth_threshold=0.001, max_layers=5,
                               min_score=0.0)
        with torch.no_grad():
            for i in range(40):
                mgr(_noise(i))
        self.assertGreater(len(mgr.layers), 1)
        self.assertEqual(list(model.state_dict().keys()), before)


class TestConsistencyAndRobustness(unittest.TestCase):
    def test_read_matches_write_and_spec_gap(self):
        g = torch.Generator().manual_seed(21)
        layer = somn.IndependentFWLayer(DIM)
        layer.S = torch.randn(DIM, DIM, generator=g) * 0.1
        S0 = layer.S.clone()
        H = _noise(4)
        ro = layer.read_only(H)
        spec_ro = F.linear(H, S0)
        out = layer.write(H)
        self.assertTrue(torch.allclose(ro, out, atol=1e-6))
        gap = (spec_ro - out).abs().max().item()
        print(f"\n    [F.linear vs H@S] max gap={gap:.4f}")
        self.assertGreater(gap, 1e-3)

    def test_empty_active_no_crash(self):
        m = somn.SOMNModule(DIM, min_score=0.5)
        with torch.no_grad():
            m(_noise(1))
            out = m(torch.zeros(2, 4, DIM))
        self.assertEqual(m.stats()["active_last"], [])
        self.assertTrue(torch.isfinite(out).all())

    def test_dtype_alignment(self):
        m = somn.SOMNModule(DIM, min_score=0.0)
        with torch.no_grad():
            m(_noise(1, dtype=torch.float32))
            self.assertEqual(m.layers[0].S.dtype, torch.float32)
            m(_noise(2, dtype=torch.float64))
            self.assertEqual(m.layers[0].S.dtype, torch.float64)
            out = m(_noise(3, dtype=torch.float64))
        self.assertEqual(out.dtype, torch.float64)

    def test_gradient_passthrough(self):
        m = somn.SOMNModule(DIM)
        H = _noise(8).requires_grad_(True)
        out = m(H)
        self.assertTrue(out.requires_grad)
        out.sum().backward()
        self.assertIsNotNone(H.grad)
        self.assertTrue(torch.isfinite(H.grad).all())

    def test_reset(self):
        m = somn.SOMNModule(DIM)
        with torch.no_grad():
            for i in range(5):
                m(_noise(i))
        m.reset()
        self.assertEqual(len(m.layers), 1)
        self.assertIsNone(m.layers[0].S)
        self.assertEqual(m.stats()["steps"], 0)


class TestModelIntegration(unittest.TestCase):
    """验证清单 #8：主干接入。"""

    @staticmethod
    def _tiny_model():
        import model as model_mod
        return model_mod.Gargantua(
            n_enc_blocks=1, n_dec_blocks=1, d_model=128, vocab_size=256,
        )

    def test_default_disabled_no_state_dict_change(self):
        m = self._tiny_model()
        self.assertIsNone(m._somn_module())
        self.assertFalse(any("somn" in k for k in m.state_dict().keys()))

    def test_attached_somn_forward_and_state_dict(self):
        m = self._tiny_model()
        before = {k: v.clone() for k, v in m.state_dict().items()}
        mgr = somn.attach_somn(m, m.d_model, warmup_steps=0,
                               max_write_count=10 ** 6)
        self.assertIs(m._somn_module(), mgr)
        torch.manual_seed(0)
        ids = torch.randint(0, 256, (1, 8))
        types = torch.zeros_like(ids)
        steps = torch.arange(8).unsqueeze(0)
        x = m.embed_text(ids, types)
        out = m.encode(x, steps)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(mgr.layers[0].write_count, 0)
        self.assertEqual(list(m.state_dict().keys()), list(before.keys()))
        for k, v in before.items():
            self.assertTrue(torch.equal(v, m.state_dict()[k]), msg=k)

    def test_reset_memory_resets_somn_by_default(self):
        m = self._tiny_model()
        mgr = somn.attach_somn(m, m.d_model, warmup_steps=0,
                               max_write_count=10 ** 6)
        ids = torch.randint(0, 256, (1, 8))
        types = torch.zeros_like(ids)
        steps = torch.arange(8).unsqueeze(0)
        with torch.no_grad():
            m.encode(m.embed_text(ids, types), steps)
        self.assertGreater(mgr.layers[0].write_count, 0)
        m.reset_memory()
        self.assertIsNone(mgr.layers[0].S)
        self.assertEqual(len(mgr.layers), 1)


def run_growth_demo():
    """100 条混合数据的层数增长 + 三项验收指标。"""
    D = 256
    print("=" * 100)
    print("SOMN v4 动态演化 · 第一轮 100 条数据（代码段 60% / 随机文本 40%）")
    print(f"torch={torch.__version__}  device=cpu  "
          f"(MPS={torch.backends.mps.is_available()} / "
          f"CUDA={torch.cuda.is_available()})")
    print("=" * 100)

    # eta / 分裂阈值走维度自适应默认（D=256 → eta=0.0025, thr=1.25）
    m = somn.SOMNModule(
        D, max_write_count=10, warmup_steps=50, warmup_bonus=1.0,
        max_layers=5,
    )
    print(f"dim={D} eta={m.eta:.6g} threshold={m.expand_growth_threshold:.4g} "
          f"max_write_count={m.max_write_count} warmup={m.warmup_steps} "
          f"max_layers={m.max_layers} min_score={m.router.min_score}")
    print(f"{'step':>5} {'kind':>6} {'layers':>7} {'splits':>7} "
          f"{'n_active':>9} {'writes':>30} {'scores':>34}")

    def batch(i):
        kind = "code" if i % 5 < 3 else "rand"
        src = _code if kind == "code" else _noise
        return kind, src(1000 + i, B=1, T=8, dim=D)

    last = -1
    with torch.no_grad():
        for i in range(1, 101):
            kind, H = batch(i)
            m(H)
            st = m.stats()
            if st["layers"] != last or i in (1, 10, 25, 50, 75, 100):
                print(f"{i:>5} {kind:>6} {st['layers']:>7} "
                      f"{st['splits']:>7} {len(st['active_last']):>9} "
                      f"{str(st['writes'][:5]):>30} "
                      f"{str(st['scores'][:5]):>34}")
                last = st["layers"]

    st = m.stats()
    print(f"\n  => 结束 layers={st['layers']} splits={st['splits']} "
          f"warmups={st['warmups']} writes={st['writes']} "
          f"warmup_ages={st['warmup_ages']}")

    # 冻结对比：让所有层毕业，单独喂代码 / 随机，看打分与激活
    for layer in m.layers:
        layer.warmup_age = None
    frozen = somn.EnergyElbowRouter(min_score=0.5, elbow_alpha=0.6)
    c_scores = frozen.scores(_code(7777, B=1, T=8, dim=D), m.layers)
    c_active = frozen.route(_code(7777, B=1, T=8, dim=D), m.layers)
    n_scores = frozen.scores(_noise(7777, B=1, T=8, dim=D), m.layers)
    n_active = frozen.route(_noise(7777, B=1, T=8, dim=D), m.layers)
    print("\n--- 冻结对比（warm-up 全部毕业） ---")
    print(f"  代码段 scores={[round(s, 3) for s in c_scores]}  active={c_active}")
    print(f"  随机   scores={[round(s, 3) for s in n_scores]}  active={n_active}")

    print("\n--- 三项验收指标 ---")
    ok_layers = 3 <= st["layers"] <= 5
    ok_writes = (all(w > 0 for w in st["writes"][1:])
                 if st["layers"] > 1 else False)
    ok_diff = (max(c_scores) - max(n_scores) > 0.2) and (c_active != n_active)
    print(f"  1) 层数 3-5                 : {st['layers']}  "
          f"{'PASS' if ok_layers else 'FAIL'}")
    print(f"  2) 新层 50 步内非零 writes  : {st['writes'][1:]}  "
          f"{'PASS' if ok_writes else 'FAIL'}")
    print(f"  3) 代码/随机激活明显区分    : {c_active} vs {n_active}  "
          f"{'PASS' if ok_diff else 'FAIL'}")
    print("=" * 100)
    return st


if __name__ == "__main__":
    if "--demo" in _sys.argv:
        run_growth_demo()
    else:
        unittest.main(verbosity=2)
