# ===========================================================================
# GARGANTUA v1 — service/test/test_momw.py：MoMW 三阶段记忆阀门回归
# ---------------------------------------------------------------------------
# 2026-11-08 第二版（三阶段路由）。重点覆盖这些不能破的契约：
#   1. **state_dict 零新增 key**：默认关闭 / 显式关闭 / 启用，三者与同 seed
#      基线逐值一致（build_model 是 strict load，多一个 key 旧目录全废）。
#      锚点是 persistent=False buffer，幂迭代向量在普通 dict 里，都不落盘。
#   2. **阶段一**：锚点惰性首次 forward 计算、可用 provider 注入、
#      γ=0 时打分就是 cos(H_t, P)。
#   3. **阶段二**：γ 随写入次数单调递增、增量幂迭代填充 D 维 E、
#      γ→1 时打分由状态特征主导。
#   4. **阶段三**：空层分裂绝不 clone 状态、派生锚点继承 E、
#      指数衰减 bonus、冷却/上限、同类软负载均衡。
#   5. **读写隔离 / 无训练 / 单一出口 / 静默激活 / 会话隔离**。
#   6. **config**：16 字段校验、v1 配置迁移、roundtrip、旧目录零影响。
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


class TestMoMWConfig(unittest.TestCase):
    """spec 默认值 + common 校验/容错/v1 迁移（创建与加载共用一套规则）。"""

    def test_defaults(self):
        c = spec.momw_default_config()
        self.assertFalse(c["enabled"])
        self.assertEqual(set(c), set(spec.MOMW_CONFIG_FIELDS))
        self.assertEqual(len(c), 16)
        self.assertGreater(c["B0"], c["threshold"])
        self.assertLess(c["threshold"], 1.0,
                        "v2 是余弦打分，不能用 v1 的能量范数量级")

    def test_validate(self):
        self.assertFalse(validate_momw_config(None)["enabled"])
        self.assertTrue(validate_momw_config(True)["enabled"])
        self.assertFalse(validate_momw_config(False)["enabled"])
        c = validate_momw_config({"enabled": True, "N": 3, "tau": "0.5",
                                  "M": 5, "eigen_interval": 8,
                                  "gamma_k": 16.0, "load_lambda": 0.1,
                                  "anchor_sample": 16,
                                  "eigen_min_writes": 2,
                                  "expand_delta": 4.0})
        self.assertEqual(c["N"], 3)
        self.assertEqual(c["tau"], 0.5)
        self.assertEqual(c["eigen_interval"], 8)
        self.assertEqual(c["gamma_k"], 16.0)
        self.assertEqual(c["load_lambda"], 0.1)
        self.assertEqual(c["expand_delta"], 4.0)
        self.assertEqual(c["threshold"], spec.MOMW_THRESHOLD)
        bad = (
            {"enabled": "yes"}, {"N": True}, {"N": 0},
            {"N": spec.MOMW_TOP_N_RANGE[1] + 1},
            {"tau": 0}, {"tau": float("nan")},
            {"B0": 0.1},                      # B0 <= threshold → 死锁
            {"initial_layers": 10, "max_layers": 2},
            {"eigen_interval": 0},
            {"eigen_interval": spec.MOMW_EIGEN_INTERVAL_RANGE[1] + 1},
            {"gamma_k": 0}, {"load_lambda": 2.0},
            {"anchor_sample": 0}, {"eigen_min_writes": -1},
            {"expand_delta": -1.0},
            {"nope": 1}, 42, "abc",
        )
        for v in bad:
            with self.assertRaises(ValueError, msg=f"应拒绝 {v!r}"):
                validate_momw_config(v)

    def test_read_fallback(self):
        self.assertFalse(read_momw_config({})["enabled"])
        self.assertTrue(read_momw_config(
            {"momw": {"enabled": True}})["enabled"])
        # 显式启用 + 非法 B0：迁移逻辑不能把它悄悄覆盖成默认值
        self.assertFalse(read_momw_config(
            {"momw": {"enabled": True, "B0": 0.1}})["enabled"])
        self.assertFalse(read_momw_config(
            {"momw": {"enabled": True, "N": True}})["enabled"])
        # 新配置只写 enabled 时不应被误判成 v1、也不该重标阈值
        got = read_momw_config({"momw": {"enabled": True}})
        self.assertTrue(got["enabled"])
        self.assertEqual(got["threshold"], spec.MOMW_THRESHOLD)

    def test_v1_config_migration(self):
        """v1 config（无 v2 独有字段）→ 阈值重标到余弦量级，不静默失效。"""
        old = {"enabled": True, "N": 2, "tau": 1.0,
               "threshold": 1.0, "M": 8, "B0": 2.0,
               "starve_threshold": 0.1, "saturation_limit": 100.0,
               "initial_layers": 1, "max_layers": 8}
        got = read_momw_config({"momw": old})
        self.assertTrue(got["enabled"])
        self.assertEqual(got["threshold"], spec.MOMW_THRESHOLD)
        self.assertEqual(got["B0"], spec.MOMW_BONUS)
        self.assertIn("gamma_k", got)
        self.assertIn("eigen_interval", got)


class TestMoMWStateDict(unittest.TestCase):
    """最重要的契约：零新增 parameter / persistent buffer。"""

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
        # 锚点是 persistent=False：会随 .to() 迁移，但绝不进 state_dict
        self.assertEqual(len(list(on.momw.buffers())), 1)
        self.assertNotIn("momw._anchors", keys)
        self.assertFalse(any("momw" in k for k in keys))
        # MoMW 构造不能消耗全局随机序列，否则同 seed 权重会分叉
        for k, v in base.state_dict().items():
            self.assertTrue(torch.equal(v, on.state_dict()[k]), k)

    def test_expansion_does_not_change_state_dict(self):
        c = _cfg(enabled=True, B0=0.0, threshold=-1e9,
                 expand_delta=0.0, max_layers=5, M=2)
        m = _small_model(c)
        before = set(m.state_dict())
        for _ in range(8):
            m.momw(torch.randn(1, 3, _D))   # 直接驱动阀门，Gargantua 无 forward
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
        # 启用但阈值极高、bonus=0 → 没有层被激活 → H_out = H_1
        torch.manual_seed(7)
        on = _small_model(_cfg(enabled=True, B0=0.0, threshold=1e9,
                               expand_delta=1e9, saturation_limit=1e9))
        got = on.forward_train(ids, si, dt, sd)
        self.assertTrue(torch.allclose(ref, got))
        self.assertEqual(on.momw.last_indices, [])


class TestStageOne(unittest.TestCase):
    """阶段一：锚点播种。"""

    def test_lazy_anchors_and_provider(self):
        c = _cfg(B0=0.0, threshold=1e9)
        m = momw.MoMW(_kda(), c)
        self.assertFalse(m._anchors_ready)
        m(_x())
        self.assertTrue(m._anchors_ready)
        self.assertEqual(tuple(m._anchors.shape), (_K, _D))
        # 每行是单位向量（余弦打分的前提）
        self.assertTrue(torch.allclose(
            m._anchors.norm(dim=-1), torch.ones(_K), atol=1e-4))
        custom = _anchors()
        m.set_anchor_provider(lambda K, D: custom)
        self.assertFalse(m._anchors_ready)
        m(_x())
        self.assertTrue(torch.allclose(m._anchors, custom, atol=1e-5))

    def test_anchor_from_wte(self):
        """有 wte 时锚点来自代表性 token Embedding 均值，随权重走。"""
        wte = torch.nn.Embedding(64, _D)
        with torch.no_grad():
            wte.weight.fill_(0.0)
            wte.weight[:, 0] = 1.0        # 所有行同向 → 锚点应指向 e0
        m = momw.MoMW(_kda(), _cfg(), wte=wte)
        m(_x())
        self.assertTrue(torch.allclose(
            m._anchors[:, 0], torch.ones(_K), atol=1e-4))
        self.assertTrue((m._anchors[:, 1:].abs() < 1e-5).all())

    def test_cold_start_score_is_cosine(self):
        """γ=0 时 raw = cos(H_t, P)；balance/bonus 只加在 scores 上。"""
        c = _cfg(B0=0.0, threshold=-1e9, load_lambda=0.0,
                 initial_layers=1)
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
        self.assertAlmostEqual(r.last_cos_anchor[0], expected.item(),
                               places=4)

    def test_dimension_checks(self):
        m = momw.MoMW(_kda(), _cfg())
        with self.assertRaises(RuntimeError):
            m(torch.randn(1, 4, _D + 1))
        with self.assertRaises(RuntimeError):
            m(torch.randn(1, _D))
        with self.assertRaises(RuntimeError):
            momw.MoMW(_kda(), dim=_D + 1)


class TestStageTwo(unittest.TestCase):
    """阶段二：γ 漂移与增量幂迭代。"""

    def test_gamma_monotonic(self):
        c = _cfg(B0=0.6, threshold=-1e9, eigen_interval=2,
                 expand_delta=1e9, saturation_limit=1e9)
        m = momw.MoMW(_kda(), c)
        l = m.pool.kda_layers[0]
        self.assertEqual(l.gamma, 0.0)
        prev = 0.0
        for _ in range(6):
            m(_x())
            self.assertGreater(l.gamma, prev)
            prev = l.gamma
        self.assertLess(l.gamma, 1.0)

    def test_power_iteration_fills_eigenvector(self):
        c = _cfg(B0=0.6, threshold=-1e9, eigen_interval=2,
                 expand_delta=1e9, saturation_limit=1e9)
        m = momw.MoMW(_kda(), c)
        l = m.pool.kda_layers[0]
        self.assertIsNone(l._rt["E"])
        for _ in range(3):
            m(_x())
        self.assertIsNotNone(l._rt["E"])
        self.assertEqual(tuple(l.get_eigenvector().shape), (_D,))
        self.assertAlmostEqual(l.get_eigenvector().norm().item(), 1.0,
                               places=3)

    def test_score_drifts_to_state_when_gamma_near_one(self):
        c = _cfg(B0=0.0, threshold=-1e9, load_lambda=0.0,
                 initial_layers=1)
        A = _anchors()
        pool = momw.MoMWPool(_kda(), c, anchors=A)
        l = pool.kda_layers[0]
        l._rt["writes"] = 10 ** 6            # γ ≈ 1
        E = F.normalize(torch.randn(_D), dim=-1)
        l._rt["E"] = E
        x = _x()
        raw, _ = momw.MoMWRouter(_D).score_all(x, pool.kda_layers, A, c)
        h = F.normalize(x.mean(dim=(0, 1)), dim=-1)
        self.assertAlmostEqual(raw[0].item(), torch.dot(h, E).item(),
                               places=3)


class TestStageThree(unittest.TestCase):
    """阶段三：空层分裂、派生锚点、指数 bonus、冷却、软均衡。"""

    def test_split_does_not_clone_parent_state(self):
        c = _cfg(B0=0.0, threshold=-1e9, expand_delta=0.0,
                 max_layers=4, M=3, eigen_interval=100)
        m = momw.MoMW(_kda(), c)
        parent = m.pool.kda_layers[0]
        m(_x())
        self.assertEqual(len(m.pool.kda_layers), 2)
        self.assertIsNotNone(parent.S)        # 父层确实被写入
        new = m.pool.kda_layers[1]
        self.assertIsNone(new.S, "新层 S_new 必须是全零")
        self.assertEqual(new._rt["writes"], 0)
        self.assertEqual(new.gamma, 0.0)
        self.assertEqual(new._rt["bonus_age"], 0)
        self.assertIsNone(new._rt["E"], "幂迭代向量不许复制父层")
        self.assertFalse(torch.equal(new._rt["eig_v"],
                                     parent._rt["eig_v"]))

    def test_derived_anchor_inherits_parent_eigenvector(self):
        c = _cfg(B0=0.0, threshold=-1e9, expand_delta=0.0,
                 eigen_min_writes=0, eigen_interval=1,
                 max_layers=3, M=5)
        m = momw.MoMW(_kda(), c)
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
        c = _cfg(B0=0.0, threshold=-1e9, expand_delta=0.0,
                 max_layers=4, M=2)
        m = momw.MoMW(_kda(), c)
        for _ in range(20):
            m(_x())
        self.assertEqual(len(m.pool.kda_layers), c["max_layers"])
        self.assertGreaterEqual(m.pool.expansions, 3)

    def test_soft_load_balance_favours_lower_norm(self):
        c = _cfg(B0=0.0, threshold=-1e9, load_lambda=0.5,
                 initial_layers=2)
        A = _anchors()
        pool = momw.MoMWPool(_kda(), c, anchors=A)
        l0, l1 = pool.kda_layers[0], pool.kda_layers[1]
        l1.type_idx = l0.type_idx = 0         # 强制同类才触发均衡
        l0.get_norm = lambda: 10.0
        l1.get_norm = lambda: 1.0
        r = momw.MoMWRouter(_D)
        x = _x()
        _raw, scores = r.score_all(x, pool.kda_layers, A, c)
        self.assertGreater(scores[1].item(), scores[0].item(),
                           "同类层里范数低（余量大）的层应被加分")


class TestIsolation(unittest.TestCase):
    """读写隔离、无训练、静默激活、单一出口、会话复位。"""

    def test_unrecalled_layers_do_not_advance(self):
        c = _cfg(B0=0.6, threshold=0.2, N=1, initial_layers=3,
                 expand_delta=1e9, saturation_limit=1e9, M=8)
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
        c = _cfg(B0=0.0, threshold=1e9)
        pool = momw.MoMWPool(_kda(), c)
        self.assertEqual(pool.forward(_x(), None, None), 0.0)

    def test_delta_has_no_grad(self):
        c = _cfg(B0=0.6, threshold=0.2, expand_delta=1e9,
                 saturation_limit=1e9)
        m = momw.MoMW(_kda(), c)
        x = _x().requires_grad_(True)
        out = m(x)
        self.assertTrue(out.requires_grad, "主残差路径梯度必须保留")
        out.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))
        self.assertEqual(len(list(m.parameters())), 0)

    def test_silent_writes_but_does_not_inject(self):
        c = _cfg(B0=0.6, threshold=0.2, expand_delta=1e9,
                 saturation_limit=1e9)
        m = momw.MoMW(_kda(), c)
        st = {momw.MoMW.SILENT_KEY: True}
        x = _x()
        out = m(x, state=st)
        self.assertTrue(torch.equal(out, x), "静默时 ΔH 不注入输出")
        s = m.stats(st)
        self.assertTrue(s["silent"])
        self.assertGreater(s["layers"][0]["writes"], 0,
                           "静默仍要后台实时写入状态")

    def test_external_state_isolated_and_reset(self):
        c = _cfg(B0=0.6, threshold=0.2, expand_delta=1e9,
                 saturation_limit=1e9, M=4)
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
        self.assertIs(s1[momw.MoMW.STATE_KEY], p1,
                      "reset 不能清掉外部会话状态")
        self.assertEqual(m.pool.kda_layers[0]._rt["bonus_age"], 0)

    def test_stats_contract_for_frontend(self):
        c = _cfg(B0=0.6, threshold=0.2, expand_delta=1e9,
                 saturation_limit=1e9)
        m = momw.MoMW(_kda(), c)
        m(_x())
        s = m.stats()
        for key in ("enabled", "silent", "steps", "layers", "layer_count",
                    "expansions", "last_indices", "last_raw", "last_scores",
                    "last_cos_anchor", "last_cos_state", "last_gamma",
                    "config", "types"):
            self.assertIn(key, s, key)
        for key in ("type", "gamma", "writes", "norm", "active",
                    "has_state", "derived_anchor"):
            self.assertIn(key, s["layers"][0], key)
        self.assertEqual(s["steps"], 1)


class TestMoMWCheckpoint(unittest.TestCase):
    """config.json 是唯一开关；旧目录缺字段 → 关闭，strict load 不受影响。"""

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
        c = validate_momw_config({"enabled": True, "N": 3, "M": 5,
                                  "eigen_interval": 8})
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
        got = build_model(self.tmp)
        self.assertIsNone(got.momw)

    def test_v1_config_dir_still_loads(self):
        """上一版创建的 config（10 字段）能被 v2 加载且阈值已迁移。"""
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

    def test_forward_train_with_momw_backward(self):
        c = _cfg(enabled=True, B0=0.6, threshold=0.2,
                 expand_delta=1e9, saturation_limit=1e9)
        m = _small_model(c)
        ids = torch.tensor([[1, 2, 3, 4]])
        si = torch.arange(4).unsqueeze(0)
        dt = torch.tensor([[5, 6, 7, 8]])
        sd = torch.arange(4, 8).unsqueeze(0)
        logits = m.forward_train(ids, si, dt, sd)
        logits.float().mean().backward()
        self.assertIsNotNone(m.fast_weight.layer.w_o.weight.grad,
                             "主模型梯度不该被 no_grad 阀门吃掉")


if __name__ == "__main__":
    unittest.main()
