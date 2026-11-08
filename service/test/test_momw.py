# ===========================================================================
# GARGANTUA v1 — service/test/test_momw.py：MoMW 三阶段记忆阀门回归
# ---------------------------------------------------------------------------
# 2026-11-08 第三版（数据驱动阈值 / Oja / 持久化 / 门控 / 异步写）。
# 不能破的契约：
#   1. **state_dict 零新增 key**：默认关闭 / 显式关闭 / 启用，三者与同 seed
#      基线逐值一致（build_model 是 strict load，多一个 key 旧目录全废）。
#   2. **阶段一**：锚点惰性首次 forward 计算、provider 可注入、γ=0 时打分
#      就是 cos(H_t, P)。
#   3. **阶段二**：Oja 每步更新（oja_steps 递增）、E 是 D 维单位向量、
#      γ→1 时打分由状态特征主导。
#   4. **阶段三**：空层分裂绝不 clone 状态、派生锚点继承 E、指数 bonus、
#      冷却/上限、同类软负载均衡、相对量过载判定。
#   5. **v3**：分位数阈值（历史不足退回 cold）、快照 roundtrip/损坏容错、
#      cost-aware 门控、单写者异步写 + flush。
#   6. **读写隔离 / 无训练 / 单一出口 / 静默 / 会话隔离 / config 校验**。
# 运行：cd ROOT && /usr/local/bin/python3 service/test/test_momw.py -v
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

import math
import os
import shutil
import tempfile
import unittest

import torch
import torch.nn.functional as F

import momw
import momw_state
import model as model_mod
import spec
from common import (build_model, read_config, read_momw_config,
                    save_checkpoint, validate_momw_config)

_VS = 16
_D = 256
_K = len(spec.MOMW_ANCHOR_GROUPS)


def _kda():
    return model_mod.KDAAttention(d_model=_D)


def _cfg(**kw) -> dict:
    """直接改默认配置（绕过校验，用来测模块内部行为）。"""
    c = spec.momw_default_config()
    c.update(kw)
    return c


def _x(T: int = 4, B: int = 1) -> torch.Tensor:
    return torch.randn(B, T, _D)


def _anchors() -> torch.Tensor:
    return F.normalize(torch.randn(_K, _D), dim=-1)


def _small_model(momw_cfg=None):
    return model_mod.Gargantua(n_enc_blocks=1, n_dec_blocks=1,
                               vocab_size=_VS, d_model=_D, momw=momw_cfg)


# 关闭过载判定 + 门控的「纯路由」配置，避免测试被扩展/门控干扰
_CALM = dict(expand_delta=0.0, saturation_limit=0.0,
             saturation_ratio=0.0, growth_ratio=0.0, fw_gate_c=0.0)


class TestMoMWConfig(unittest.TestCase):
    def test_defaults(self):
        c = spec.momw_default_config()
        self.assertFalse(c["enabled"])
        self.assertEqual(set(c), set(spec.MOMW_CONFIG_FIELDS))
        self.assertEqual(len(c), 25)
        self.assertGreater(c["B0"], c["threshold"])
        self.assertLess(c["threshold"], 1.0)
        self.assertEqual(c["threshold_mode"], "fixed")
        self.assertFalse(c["async_write"])
        self.assertTrue(c["persist"])

    def test_validate(self):
        self.assertFalse(validate_momw_config(None)["enabled"])
        self.assertTrue(validate_momw_config(True)["enabled"])
        self.assertFalse(validate_momw_config(False)["enabled"])
        c = validate_momw_config({
            "enabled": True, "N": 3, "tau": "0.5", "M": 5,
            "eigen_interval": 8, "gamma_k": 16.0, "load_lambda": 0.1,
            "anchor_sample": 16, "eigen_min_writes": 2,
            "expand_delta": 4.0,
            "threshold_mode": "quantile", "quantile_p": "0.8",
            "quantile_window": 32, "saturation_ratio": 1.2,
            "growth_ratio": 0.3, "oja_eta": 0.05, "fw_gate_c": 3.0,
            "async_write": True, "persist": False,
        })
        self.assertEqual(c["N"], 3)
        self.assertEqual(c["tau"], 0.5)
        self.assertEqual(c["threshold_mode"], "quantile")
        self.assertEqual(c["quantile_p"], 0.8)
        self.assertEqual(c["quantile_window"], 32)
        self.assertEqual(c["saturation_ratio"], 1.2)
        self.assertEqual(c["oja_eta"], 0.05)
        self.assertTrue(c["async_write"])
        self.assertFalse(c["persist"])
        bad = (
            {"enabled": "yes"}, {"N": True}, {"N": 0},
            {"tau": 0}, {"tau": float("nan")},
            {"B0": 0.1},                      # B0 <= threshold → 死锁
            {"initial_layers": 10, "max_layers": 2},
            {"eigen_interval": 0}, {"gamma_k": 0}, {"load_lambda": 2.0},
            {"anchor_sample": 0}, {"eigen_min_writes": -1},
            {"expand_delta": -1.0},
            {"threshold_mode": "auto"},
            {"quantile_p": 0.0}, {"quantile_p": 1.5},
            {"quantile_window": 2}, {"quantile_window": 99999},
            {"saturation_ratio": 0.5}, {"growth_ratio": -0.1},
            {"oja_eta": 0.0}, {"oja_eta": 2.0},
            {"fw_gate_c": -1.0}, {"fw_gate_c": 99.0},
            {"async_write": "yes"}, {"persist": 2},
            {"nope": 1}, 42, "abc",
        )
        for v in bad:
            with self.assertRaises(ValueError, msg=f"应拒绝 {v!r}"):
                validate_momw_config(v)

    def test_read_fallback(self):
        self.assertFalse(read_momw_config({})["enabled"])
        self.assertTrue(read_momw_config(
            {"momw": {"enabled": True}})["enabled"])
        self.assertFalse(read_momw_config(
            {"momw": {"enabled": True, "B0": 0.1}})["enabled"])
        self.assertFalse(read_momw_config(
            {"momw": {"enabled": True, "N": True}})["enabled"])
        got = read_momw_config({"momw": {"enabled": True}})
        self.assertEqual(got["threshold"], spec.MOMW_THRESHOLD)

    def test_v1_config_migration(self):
        old = {"enabled": True, "N": 2, "tau": 1.0,
               "threshold": 1.0, "M": 8, "B0": 2.0,
               "starve_threshold": 0.1, "saturation_limit": 100.0,
               "initial_layers": 1, "max_layers": 8}
        got = read_momw_config({"momw": old})
        self.assertTrue(got["enabled"])
        self.assertEqual(got["threshold"], spec.MOMW_THRESHOLD)
        self.assertEqual(got["B0"], spec.MOMW_BONUS)
        self.assertIn("oja_eta", got)


class TestMoMWStateDict(unittest.TestCase):
    def test_zero_new_keys_and_identical_weights(self):
        torch.manual_seed(3)
        base = _small_model()
        torch.manual_seed(3)
        off = _small_model(_cfg(enabled=False))
        torch.manual_seed(3)
        on = _small_model(_cfg(enabled=True))
        self.assertIsNone(base.momw)
        self.assertIsNone(off.momw)
        self.assertIsNotNone(on.momw)
        keys = set(base.state_dict())
        self.assertEqual(keys, set(off.state_dict()))
        self.assertEqual(keys, set(on.state_dict()))
        self.assertEqual(len(list(on.momw.parameters())), 0)
        self.assertEqual(len(list(on.momw.buffers())), 1)
        self.assertFalse(any("momw" in k for k in keys))
        for k, v in base.state_dict().items():
            self.assertTrue(torch.equal(v, on.state_dict()[k]), k)

    def test_expansion_does_not_change_state_dict(self):
        c = _cfg(enabled=True, B0=0.0, threshold=-1e9,
                 saturation_limit=1e-6, max_layers=5, M=2)
        m = _small_model(c)
        before = set(m.state_dict())
        for _ in range(8):
            m.momw(torch.randn(1, 3, _D))
        self.assertGreater(len(m.momw.pool.kda_layers), 1)
        self.assertEqual(before, set(m.state_dict()))

    def test_disabled_or_no_activation_is_identity(self):
        ids = torch.tensor([[1, 2, 3, 4]])
        si = torch.arange(4).unsqueeze(0)
        dt = torch.tensor([[5, 6, 7, 8]])
        sd = torch.arange(4, 8).unsqueeze(0)
        torch.manual_seed(7)
        base = _small_model()
        ref = base.forward_train(ids, si, dt, sd)
        torch.manual_seed(7)
        on = _small_model(_cfg(enabled=True, B0=0.0, threshold=1e9,
                               **_CALM))
        got = on.forward_train(ids, si, dt, sd)
        self.assertTrue(torch.allclose(ref, got))
        self.assertEqual(on.momw.last_indices, [])


class TestStageOne(unittest.TestCase):
    def test_lazy_anchors_and_provider(self):
        c = _cfg(B0=0.0, threshold=1e9, **_CALM)
        m = momw.MoMW(_kda(), c)
        self.assertFalse(m._anchors_ready)
        self.assertEqual(m.stats()["anchor_source"], "sample")
        m(_x())
        self.assertTrue(m._anchors_ready)
        self.assertEqual(tuple(m._anchors.shape), (_K, _D))
        self.assertTrue(torch.allclose(
            m._anchors.norm(dim=-1), torch.ones(_K), atol=1e-4))
        custom = _anchors()
        m.set_anchor_provider(lambda K, D: custom)
        self.assertFalse(m._anchors_ready)
        self.assertEqual(m.stats()["anchor_source"], "provider")
        m(_x())
        self.assertTrue(torch.allclose(m._anchors, custom, atol=1e-5))

    def test_anchor_from_wte(self):
        wte = torch.nn.Embedding(64, _D)
        with torch.no_grad():
            wte.weight.fill_(0.0)
            wte.weight[:, 0] = 1.0
        m = momw.MoMW(_kda(), _cfg(**_CALM), wte=wte)
        m(_x())
        self.assertTrue(torch.allclose(
            m._anchors[:, 0], torch.ones(_K), atol=1e-4))
        self.assertTrue((m._anchors[:, 1:].abs() < 1e-5).all())

    def test_bad_provider_falls_back(self):
        m = momw.MoMW(_kda(), _cfg(**_CALM))
        m.set_anchor_provider(lambda K, D: (_ for _ in ()).throw(
            RuntimeError("boom")))
        m(_x())
        self.assertTrue(m._anchors_ready)
        self.assertTrue(torch.allclose(
            m._anchors.norm(dim=-1), torch.ones(_K), atol=1e-4))

    def test_cold_start_score_is_cosine(self):
        c = _cfg(B0=0.0, threshold=-1e9, load_lambda=0.0,
                 initial_layers=1, **_CALM)
        A = _anchors()
        pool = momw.MoMWPool(_kda(), c, anchors=A)
        r = momw.MoMWRouter(_D)
        x = _x()
        raw, scores = r.score_all(x, pool.kda_layers, A, c)
        h = F.normalize(x.mean(dim=(0, 1)), dim=-1)
        expected = torch.dot(h, A[0])
        self.assertAlmostEqual(raw[0].item(), expected.item(), places=4)
        self.assertAlmostEqual(scores[0].item(), expected.item(), places=4)
        self.assertEqual(r.last_gamma[0], 0.0)

    def test_dimension_checks(self):
        c = _cfg(**_CALM)
        m = momw.MoMW(_kda(), c)
        with self.assertRaises(RuntimeError):
            m(torch.randn(1, 4, _D + 1))
        with self.assertRaises(RuntimeError):
            m(torch.randn(1, _D))
        with self.assertRaises(RuntimeError):
            momw.MoMW(_kda(), dim=_D + 1)


class TestStageTwo(unittest.TestCase):
    def test_gamma_monotonic(self):
        c = _cfg(B0=0.6, threshold=-1e9, **_CALM)
        m = momw.MoMW(_kda(), c)
        l = m.pool.kda_layers[0]
        self.assertEqual(l.gamma, 0.0)
        prev = 0.0
        for _ in range(6):
            m(_x())
            self.assertGreater(l.gamma, prev)
            prev = l.gamma
        self.assertLess(l.gamma, 1.0)

    def test_oja_updates_every_write(self):
        c = _cfg(B0=0.6, threshold=-1e9, **_CALM)
        m = momw.MoMW(_kda(), c)
        l = m.pool.kda_layers[0]
        self.assertIsNone(l._rt["E"])
        m(_x())
        self.assertEqual(l._rt["oja_steps"], 1)
        self.assertIsNotNone(l._rt["E"])
        self.assertEqual(tuple(l.get_eigenvector().shape), (_D,))
        m(_x())
        self.assertEqual(l._rt["oja_steps"], 2)
        self.assertAlmostEqual(l.get_eigenvector().norm().item(), 1.0,
                               places=3)

    def test_oja_converges_to_principal_direction(self):
        """凝固一个低秩 S，Oja 应收敛到 SᵀS 的主特征向量（与 eigh 对拍）。

        不经过 forward：流式输入下 S 每步都在变，主方向本身是非平稳的，
        那种场景只能验证「跟得上」，验证不了「收敛到」。
        """
        torch.manual_seed(21)
        l = momw.FastWeightLayer(_kda(), _cfg(oja_eta=0.05), 0, seed=1)
        kda = l._refs["kda"]
        H, dk, dv = l.H, kda.dk, l.dv
        S = 0.01 * torch.randn(1, H, dk, dv)
        S[0, :, :, 0] += 1.0                      # 主方向 = e0
        l._rt["state"] = (S, {})
        for _ in range(40):
            l._oja_step()
        G = torch.einsum('hkv,hkl->hvl', S[0], S[0])
        _w, V = torch.linalg.eigh(G)
        ref = F.normalize(V[..., -1], dim=-1)
        got = F.normalize(l._rt["eig_v"], dim=-1)
        cos = (ref * got).sum(dim=-1).abs().mean().item()
        self.assertGreater(cos, 0.95, f"Oja 与主特征方向余弦仅 {cos:.3f}")

    def test_score_drifts_to_state_when_gamma_near_one(self):
        c = _cfg(B0=0.0, threshold=-1e9, load_lambda=0.0,
                 initial_layers=1, **_CALM)
        A = _anchors()
        pool = momw.MoMWPool(_kda(), c, anchors=A)
        l = pool.kda_layers[0]
        l._rt["writes"] = 10 ** 6
        E = F.normalize(torch.randn(_D), dim=-1)
        l._rt["E"] = E
        x = _x()
        raw, _ = momw.MoMWRouter(_D).score_all(x, pool.kda_layers, A, c)
        h = F.normalize(x.mean(dim=(0, 1)), dim=-1)
        self.assertAlmostEqual(raw[0].item(), torch.dot(h, E).item(),
                               places=3)


class TestStageThree(unittest.TestCase):
    def test_split_does_not_clone_parent_state(self):
        c = _cfg(B0=0.0, threshold=-1e9, saturation_limit=1e-6,
                 max_layers=4, M=3)
        m = momw.MoMW(_kda(), c)
        parent = m.pool.kda_layers[0]
        for _ in range(3):
            m(_x())
        self.assertEqual(len(m.pool.kda_layers), 2)
        self.assertIsNotNone(parent.S)
        new = m.pool.kda_layers[1]
        self.assertIsNone(new.S, "新层 S_new 必须是全零")
        self.assertEqual(new._rt["writes"], 0)
        self.assertEqual(new.gamma, 0.0)
        self.assertEqual(new._rt["bonus_age"], 0)
        self.assertIsNone(new._rt["E"], "Oja 方向不许复制父层")
        self.assertFalse(torch.equal(new._rt["eig_v"],
                                     parent._rt["eig_v"]))

    def test_derived_anchor_inherits_parent_eigenvector(self):
        c = _cfg(B0=0.0, threshold=-1e9, saturation_limit=1e-6,
                 eigen_min_writes=0, max_layers=3, M=3)
        m = momw.MoMW(_kda(), c)
        for _ in range(3):
            m(_x())
        new = m.pool.kda_layers[-1]
        self.assertIsNotNone(new._rt["P_override"])
        self.assertTrue(new.describe()["derived_anchor"])
        self.assertEqual(tuple(new._rt["P_override"].shape), (_D,))

    def test_exponential_bonus(self):
        l = momw.FastWeightLayer(_kda(), _cfg(), 0, seed=1)
        B0, M = 0.6, 8
        self.assertAlmostEqual(l.bonus(B0, M), B0)
        l.tick_bonus()
        self.assertAlmostEqual(l.bonus(B0, M), B0 * math.exp(-1 / M))
        for _ in range(M):
            l.tick_bonus()
        self.assertEqual(l.bonus(B0, M), 0.0)

    def test_cooldown_and_max_layers(self):
        c = _cfg(B0=0.0, threshold=-1e9, saturation_limit=1e-6,
                 max_layers=4, M=2)
        m = momw.MoMW(_kda(), c)
        for _ in range(20):
            m(_x())
        self.assertEqual(len(m.pool.kda_layers), c["max_layers"])
        self.assertGreaterEqual(m.pool.expansions, 3)

    def test_relative_saturation_ratio(self):
        """相对量：范数超同类中位数 ratio 倍才判过载（绝对值不再必需）。"""
        c = _cfg(B0=0.0, threshold=-1e9, saturation_ratio=1.5,
                 growth_ratio=0.0, saturation_limit=0.0,
                 expand_delta=0.0, initial_layers=2, max_layers=4, M=2)
        pool = momw.MoMWPool(_kda(), c)
        l0, l1 = pool.kda_layers[0], pool.kda_layers[1]
        l1.type_idx = l0.type_idx = 0
        l0.get_norm = lambda: 10.0
        l1.get_norm = lambda: 1.0
        l0._rt["writes"] = 10
        idx = torch.tensor([0])
        self.assertIs(pool._overload_parent(idx), l0)   # 10 > 1.5*median(5.5)
        l0.get_norm = lambda: 2.0
        self.assertIsNone(pool._overload_parent(idx))   # 2 未超过 1.5*1.5

    def test_soft_load_balance_favours_lower_norm(self):
        c = _cfg(B0=0.0, threshold=-1e9, load_lambda=0.5,
                 initial_layers=2, **_CALM)
        A = _anchors()
        pool = momw.MoMWPool(_kda(), c, anchors=A)
        l0, l1 = pool.kda_layers[0], pool.kda_layers[1]
        l1.type_idx = l0.type_idx = 0
        l0.get_norm = lambda: 10.0
        l1.get_norm = lambda: 1.0
        _raw, scores = momw.MoMWRouter(_D).score_all(
            _x(), pool.kda_layers, A, c)
        self.assertGreater(scores[1].item(), scores[0].item())


class TestV3Features(unittest.TestCase):
    """分位数阈值 / 快照 / 门控 / 异步写 / 校准。"""

    def test_quantile_threshold(self):
        c = _cfg(threshold_mode="quantile", quantile_p=0.9,
                 quantile_window=32, threshold=0.0, **_CALM)
        pool = momw.MoMWPool(_kda(), c)
        t, mode = pool.effective_threshold()
        self.assertEqual(mode, "cold")            # 历史不足 → 退回固定值
        self.assertEqual(t, 0.0)
        for i in range(20):
            pool.score_history.append(i / 100.0)
        t, mode = pool.effective_threshold()
        self.assertEqual(mode, "quantile")
        self.assertGreater(t, 0.15)
        self.assertLess(t, 0.19)

    def test_quantile_window_capped(self):
        c = _cfg(threshold_mode="quantile", quantile_window=8, **_CALM)
        pool = momw.MoMWPool(_kda(), c)
        for i in range(100):
            pool.record_scores(torch.tensor([float(i)]))
        self.assertEqual(len(pool.score_history), 8)
        self.assertAlmostEqual(pool.score_history[-1], 99.0)

    def test_state_roundtrip_and_corrupt(self):
        tmp = tempfile.mkdtemp(prefix="garg-momw-st-")
        try:
            c = _cfg(B0=0.6, threshold=-1e9, oja_eta=0.05, **_CALM)
            m = momw.MoMW(_kda(), c)
            for _ in range(3):
                m(_x())
            l = m.pool.kda_layers[0]
            writes, norm = l._rt["writes"], l.get_norm()
            path = m.save_state(tmp)
            self.assertIsNotNone(path)
            self.assertTrue(momw_state.has_state(tmp))
            m2 = momw.MoMW(_kda(), c)
            self.assertTrue(m2.load_state(tmp))
            l2 = m2.pool.kda_layers[0]
            self.assertEqual(l2._rt["writes"], writes)
            self.assertAlmostEqual(l2.get_norm(), norm, places=4)
            self.assertEqual(l2.type_idx, l.type_idx)
            # 快照损坏 → 视为无快照，从零启动
            with open(os.path.join(path, momw_state.LAYOUT_NAME), "w",
                      encoding="utf-8") as f:
                f.write("{ this is not json")
            m3 = momw.MoMW(_kda(), c)
            self.assertFalse(m3.load_state(tmp))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_persist_false_writes_nothing(self):
        tmp = tempfile.mkdtemp(prefix="garg-momw-np-")
        try:
            m = momw.MoMW(_kda(), _cfg(persist=False, **_CALM))
            m(_x())
            self.assertIsNone(m.save_state(tmp))
            self.assertFalse(momw_state.has_state(tmp))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_fw_energy_gate(self):
        c = _cfg(B0=0.6, threshold=0.2, **{**_CALM, "fw_gate_c": 2.0})
        m = momw.MoMW(_kda(), c)
        x = _x()
        out = m(x, fw_energy=0.0)
        self.assertGreater(m.last_gate, 0.99)
        self.assertFalse(torch.equal(out, x))
        x2 = _x()
        out2 = m(x2, fw_energy=50.0)
        self.assertLess(m.last_gate, 1e-6)
        self.assertTrue(torch.equal(out2, x2), "门控为 0 时不注入 ΔH")

    def test_async_write_and_flush(self):
        c = _cfg(B0=0.6, threshold=-1e9, async_write=True, **_CALM)
        m = momw.MoMW(_kda(), c)
        l = m.pool.kda_layers[0]
        for _ in range(3):
            m(_x())
        waited = m.flush()
        self.assertGreaterEqual(waited, 1)
        self.assertEqual(l._rt["writes"], 3)
        self.assertIsNotNone(l.S)
        self.assertEqual(m.pool.pending_writes(), 0)

    def test_sync_and_async_agree(self):
        """同步/异步两条路径最终状态应一致（同一 seed、同一输入序列）。"""
        torch.manual_seed(31)
        xs = [_x() for _ in range(3)]
        torch.manual_seed(41)
        a = momw.MoMW(_kda(), _cfg(B0=0.6, threshold=-1e9, **_CALM))
        for x in xs:
            a(x)
        torch.manual_seed(41)
        b = momw.MoMW(_kda(), _cfg(B0=0.6, threshold=-1e9,
                                   async_write=True, **_CALM))
        for x in xs:
            b(x)
        b.flush()
        self.assertAlmostEqual(a.pool.kda_layers[0].get_norm(),
                               b.pool.kda_layers[0].get_norm(), places=3)
        self.assertEqual(a.pool.kda_layers[0]._rt["oja_steps"],
                         b.pool.kda_layers[0]._rt["oja_steps"])

    def test_calibrate_returns_suggestion(self):
        m = momw.MoMW(_kda(), _cfg(**_CALM))
        r = m.calibrate([_x() for _ in range(4)])
        self.assertEqual(r["samples"], 4 * 1)     # initial_layers=1
        for key in ("all_p5", "all_p50", "all_p95", "suggest_threshold"):
            self.assertIsNotNone(r[key], key)
        self.assertIn("per_type", r)
        self.assertEqual(set(r["per_type"]), set(spec.MOMW_ANCHOR_GROUPS))


class TestIsolation(unittest.TestCase):
    def test_unrecalled_layers_do_not_advance(self):
        c = _cfg(B0=0.6, threshold=0.2, N=1, initial_layers=3,
                 M=8, **_CALM)
        m = momw.MoMW(_kda(), c)
        x = _x()
        out = m(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(len(m.pool.last_indices), 1)
        recalled = m.pool.last_indices[0]
        self.assertIsNotNone(m.pool.kda_layers[recalled].S)
        for i, l in enumerate(m.pool.kda_layers):
            if i != recalled:
                self.assertIsNone(l.S, "未被召回的层状态不该推进")
        self.assertFalse(torch.equal(out, x))

    def test_pool_returns_zero_when_no_activation(self):
        pool = momw.MoMWPool(_kda(), _cfg(B0=0.0, threshold=1e9, **_CALM))
        self.assertEqual(pool.forward(_x(), None, None), 0.0)

    def test_delta_has_no_grad(self):
        m = momw.MoMW(_kda(), _cfg(B0=0.6, threshold=0.2, **_CALM))
        x = _x().requires_grad_(True)
        out = m(x)
        self.assertTrue(out.requires_grad)
        out.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))
        self.assertEqual(len(list(m.parameters())), 0)

    def test_silent_writes_but_does_not_inject(self):
        m = momw.MoMW(_kda(), _cfg(B0=0.6, threshold=0.2, **_CALM))
        st = {momw.MoMW.SILENT_KEY: True}
        x = _x()
        out = m(x, state=st)
        self.assertTrue(torch.equal(out, x))
        s = m.stats(st)
        self.assertTrue(s["silent"])
        self.assertGreater(s["layers"][0]["writes"], 0)

    def test_external_state_isolated_and_reset(self):
        c = _cfg(B0=0.6, threshold=0.2, M=4, **_CALM)
        m = momw.MoMW(_kda(), c)
        s1, s2 = {}, {}
        m(_x(), state=s1)
        p1 = s1[momw.MoMW.STATE_KEY]
        m(_x(), state=s1)
        self.assertIs(s1[momw.MoMW.STATE_KEY], p1)
        m(_x(), state=s2)
        self.assertIsNot(s1[momw.MoMW.STATE_KEY],
                         s2[momw.MoMW.STATE_KEY])
        m.reset()
        self.assertIs(s1[momw.MoMW.STATE_KEY], p1)
        self.assertEqual(m.pool.kda_layers[0]._rt["bonus_age"], 0)

    def test_stats_contract_for_frontend(self):
        m = momw.MoMW(_kda(), _cfg(B0=0.6, threshold=0.2, **_CALM))
        m(_x())
        s = m.stats()
        for key in ("enabled", "silent", "steps", "layers", "layer_count",
                    "expansions", "last_indices", "last_raw", "last_scores",
                    "last_cos_anchor", "last_cos_state", "last_gamma",
                    "config", "types", "threshold", "threshold_mode",
                    "score_window", "queue_pending", "persisted",
                    "async_write", "anchor_source", "fw_energy", "fw_gate"):
            self.assertIn(key, s, key)
        for key in ("type", "gamma", "writes", "norm", "active",
                    "has_state", "derived_anchor", "oja_steps"):
            self.assertIn(key, s["layers"][0], key)


class TestMoMWCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="garg-momw-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _manifest(self, extra=None):
        cfg = {"name": "m", "vocab_size": _VS, "d_model": _D,
               "n_enc_blocks": 1, "n_dec_blocks": 1}
        if extra:
            cfg.update(extra)
        return cfg

    def test_roundtrip_enabled(self):
        torch.manual_seed(11)
        c = validate_momw_config({"enabled": True, "N": 3, "M": 5})
        m = _small_model(c)
        save_checkpoint(self.tmp, m, self._manifest({"momw": c}))
        self.assertEqual(read_config(self.tmp)["momw"], c)
        got = build_model(self.tmp)
        self.assertIsNotNone(got.momw)
        self.assertEqual(got.momw.cfg, c)
        for k, v in m.state_dict().items():
            self.assertTrue(torch.equal(got.state_dict()[k], v), k)

    def test_old_dir_without_field(self):
        torch.manual_seed(12)
        m = _small_model()
        save_checkpoint(self.tmp, m, self._manifest())
        self.assertNotIn("momw", read_config(self.tmp))
        self.assertIsNone(build_model(self.tmp).momw)

    def test_v1_config_dir_still_loads(self):
        torch.manual_seed(13)
        v1 = {"enabled": True, "N": 2, "tau": 1.0, "threshold": 1.0,
              "M": 8, "B0": 2.0, "starve_threshold": 0.1,
              "saturation_limit": 100.0, "initial_layers": 1,
              "max_layers": 8}
        m = _small_model(v1)
        save_checkpoint(self.tmp, m, self._manifest({"momw": v1}))
        got = build_model(self.tmp)
        self.assertIsNotNone(got.momw)
        self.assertEqual(got.momw.cfg["threshold"], spec.MOMW_THRESHOLD)

    def test_build_model_restores_snapshot(self):
        torch.manual_seed(14)
        c = validate_momw_config({"enabled": True, "persist": True})
        m = _small_model(c)
        save_checkpoint(self.tmp, m, self._manifest({"momw": c}))
        for _ in range(2):
            m.momw(torch.randn(1, 3, _D))
        writes = m.momw.pool.kda_layers[0]._rt["writes"]
        m.momw.save_state(self.tmp)
        got = build_model(self.tmp)
        self.assertEqual(got.momw.pool.kda_layers[0]._rt["writes"], writes)

    def test_forward_train_with_momw_backward(self):
        c = _cfg(enabled=True, B0=0.6, threshold=0.2, **_CALM)
        m = _small_model(c)
        ids = torch.tensor([[1, 2, 3, 4]])
        si = torch.arange(4).unsqueeze(0)
        dt = torch.tensor([[5, 6, 7, 8]])
        sd = torch.arange(4, 8).unsqueeze(0)
        logits = m.forward_train(ids, si, dt, sd)
        logits.float().mean().backward()
        self.assertIsNotNone(m.fast_weight.layer.w_o.weight.grad)


if __name__ == "__main__":
    unittest.main()
