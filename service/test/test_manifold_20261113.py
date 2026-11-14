#!/usr/bin/env python3
# ===========================================================================
# GARGANTUA — manifold / Muon 落地验证（2026-11-13）
# ---------------------------------------------------------------------------
# 对应 docs/analysis/manifold_muon_mhc_feasibility.md 的阶段 0 / 1 / 2。
#
# 验证清单：
#   A 谱诊断指标正确性（wide→tall、单位/对角/随机/非有限）
#   B 裁剪前 grad norm / attention / KDA 状态 / 协方差指标
#   C 诊断零侵入：state_dict、forward 数值、attach/detach 复原
#   D 参数分组：秩 1 投影移出 Muon、分组不重不漏、白名单
#   E 样本边界复位统一（train.py / clip.py 不再绕过 reset_memory）
#   F Manifold Muon 数学：msign / 切空间投影 / retraction / 对偶上升
#   G 优化器集成：默认逐位兼容、启用后不重不漏、Parameter 身份保持
#   H config 契约：默认值、容错、build_model 注入、roundtrip 保留
#
# 运行：
#   cd ROOT && /usr/local/bin/python3 service/test/test_manifold_20261113.py -v
#   cd ROOT && /usr/local/bin/python3 -m unittest service.test.test_manifold_20261113 -v
#
# 约束：不建 2GB 全尺寸模型，全部用轻量构造（1+1 blocks / d_model=128 /
# vocab_size=256）；不写系统临时目录，临时文件落 data/tmp/。
# ===========================================================================
from __future__ import annotations

import json
import math
import os
import sys as _sys
import unittest
from pathlib import Path as _Path

for _anc in _Path(__file__).resolve().parents:
    if (_anc / "start" / "boot.py").is_file():
        _sys.path.insert(0, str(_anc / "start"))
        break
else:
    raise SystemExit("找不到 GARGANTUA 项目根（start/boot.py）")
import boot
boot.bootstrap()

import torch
import torch.nn as nn

import common
import manifold_diag
import manifold_muon
import model as model_mod
import muon
import paths
import spec

ROOT = boot.project_root()
_TOL = 1e-4
_MODEL = None


def _small_model():
    """进程内共享一个轻量模型（构造一次，避免重复分配几十 MB）。"""
    global _MODEL
    if _MODEL is None:
        torch.manual_seed(20261113)
        _MODEL = model_mod.Gargantua(
            n_enc_blocks=1, n_dec_blocks=1,
            d_model=128, vocab_size=256)
    return _MODEL


def _orthonormal(m, n, seed=0):
    """返回一个 [m, n] 列正交矩阵（m >= n）。"""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(m, n, generator=g)
    q, _ = torch.linalg.qr(a)
    return q[:, :n]


def _offgram_fro(w):
    """tall/方阵视角下 ‖Off(WᵀW)‖_F。"""
    wt, _ = manifold_muon.to_tall(w)
    gram = wt.T @ wt
    return float((gram - torch.diag_embed(torch.diagonal(gram))).norm().item())


# ---------------------------------------------------------------------------
# A. 谱诊断指标
# ---------------------------------------------------------------------------
class SpectrumTests(unittest.TestCase):

    def test_01_orthonormal_matrix_is_perfect_stiefel(self):
        w = _orthonormal(32, 8, seed=1)
        st = manifold_diag.spectrum_stats(w)
        self.assertAlmostEqual(st["sigma_max"], 1.0, delta=1e-4)
        self.assertAlmostEqual(st["sigma_min"], 1.0, delta=1e-4)
        self.assertLess(st["offgram_fro"], 1e-4)
        self.assertEqual(st["gram_dim"], 8)
        self.assertFalse(st["transposed"])

    def test_02_diagonal_matrix_has_zero_offgram(self):
        w = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        st = manifold_diag.spectrum_stats(w)
        self.assertLess(st["offgram_fro"], 1e-5)
        self.assertAlmostEqual(st["sigma_max"], 4.0, delta=1e-4)
        self.assertAlmostEqual(st["gram_diag_max"], 16.0, delta=1e-3)

    def test_03_wide_matrix_is_transposed_to_min_gram(self):
        w = torch.randn(8, 32, generator=torch.Generator().manual_seed(7))
        st = manifold_diag.spectrum_stats(w)
        self.assertTrue(st["transposed"])
        self.assertEqual(st["gram_dim"], 8)
        self.assertEqual(st["shape"], [8, 32])

    def test_04_tall_matrix_not_transposed(self):
        w = torch.randn(32, 8, generator=torch.Generator().manual_seed(8))
        st = manifold_diag.spectrum_stats(w)
        self.assertFalse(st["transposed"])
        self.assertEqual(st["gram_dim"], 8)

    def test_05_random_dense_matrix_has_positive_offgram(self):
        w = torch.randn(24, 6, generator=torch.Generator().manual_seed(9))
        st = manifold_diag.spectrum_stats(w)
        self.assertGreater(st["offgram_fro"], 0.0)
        self.assertGreater(st["sigma_max"], st["sigma_min"])

    def test_06_nonfinite_input_is_skipped_not_raised(self):
        w = torch.randn(6, 4)
        w[0, 0] = float("nan")
        st = manifold_diag.spectrum_stats(w)
        self.assertTrue(st.get("skipped"))

    def test_07_over_max_dim_is_skipped_not_raised(self):
        w = torch.randn(12, 12)
        st = manifold_diag.spectrum_stats(w, max_dim=4)
        self.assertTrue(st.get("skipped"))

    def test_08_spectrum_does_not_mutate_input(self):
        w = torch.randn(10, 5, requires_grad=True)
        before = w.detach().clone()
        manifold_diag.spectrum_stats(w)
        self.assertTrue(torch.equal(w.detach(), before))
        self.assertTrue(w.requires_grad)

    def test_09_frobenius_matches_direct_norm(self):
        w = torch.randn(16, 6, generator=torch.Generator().manual_seed(11))
        st = manifold_diag.spectrum_stats(w)
        self.assertAlmostEqual(st["fro"], float(w.norm().item()), delta=1e-4)


# ---------------------------------------------------------------------------
# B. 其它诊断指标
# ---------------------------------------------------------------------------
class DiagStatsTests(unittest.TestCase):

    def test_10_grad_norm_before_clip_matches_formula(self):
        a = nn.Parameter(torch.tensor([3.0, 4.0]))
        b = nn.Parameter(torch.tensor([0.0, 0.0]))
        a.grad = torch.tensor([3.0, 4.0])
        b.grad = torch.tensor([0.0, 0.0])
        before = a.grad.clone()
        norm = manifold_diag.grad_norm_before_clip([a, b, nn.Parameter(torch.ones(2))])
        self.assertAlmostEqual(norm, 5.0, delta=1e-6)
        self.assertTrue(torch.equal(a.grad, before))

    def test_11_grad_norm_skips_none_grads(self):
        a = nn.Parameter(torch.tensor([1.0, 2.0]))
        b = nn.Parameter(torch.tensor([3.0, 4.0]))
        a.grad = torch.tensor([3.0, 4.0])
        self.assertAlmostEqual(
            manifold_diag.grad_norm_before_clip([a, b]), 5.0, delta=1e-6)

    def test_12_attention_masked_inf_produces_no_nan(self):
        logits = torch.tensor([[[1.0, 2.0, float("-inf"), float("-inf")]]])
        st = manifold_diag.attention_stats(logits)
        self.assertTrue(math.isfinite(st["logits_absmax"]))
        self.assertTrue(math.isfinite(st["entropy_mean"]))
        self.assertAlmostEqual(st["finite_frac"], 0.5, delta=1e-6)

    def test_13_uniform_logits_entropy_is_log_n(self):
        logits = torch.zeros(1, 1, 8)
        st = manifold_diag.attention_stats(logits)
        self.assertAlmostEqual(st["entropy_mean"], math.log(8), delta=1e-5)

    def test_14_all_inf_logits_are_safe(self):
        logits = torch.full((1, 1, 4), float("-inf"))
        st = manifold_diag.attention_stats(logits)
        self.assertEqual(st["finite_frac"], 0.0)
        self.assertTrue(math.isfinite(st["entropy_mean"]))

    def test_15_attention_jacobian_optional(self):
        logits = torch.zeros(1, 1, 4)
        st = manifold_diag.attention_stats(logits, with_jacobian=True)
        self.assertIsNotNone(st.get("jacobian_fro_mean"))
        self.assertGreater(st["jacobian_fro_mean"], 0.0)

    def test_16_kda_state_stats_shape_and_values(self):
        s = torch.randn(2, 8, 16, 16, generator=torch.Generator().manual_seed(3))
        st = manifold_diag.kda_state_stats(s)
        self.assertTrue(st.get("present", True))
        self.assertGreater(st["fro"], 0.0)
        self.assertGreater(st["spectral_radius"], 0.0)
        self.assertEqual(st["blocks"], 16)

    def test_17_kda_state_none_is_safe(self):
        st = manifold_diag.kda_state_stats(None)
        self.assertFalse(st.get("present", True))

    def test_18_cov_lambda_max_known_diagonal(self):
        # 两个正交方向，第二维方差是第一维的 4 倍
        x = torch.zeros(64, 2)
        x[:, 0] = torch.randn(64)
        x[:, 1] = 2.0 * torch.randn(64)
        lam = manifold_diag.cov_lambda_max(x)
        self.assertGreater(lam, 0.0)
        self.assertTrue(math.isfinite(lam))

    def test_19_cov_lambda_max_empty_is_safe(self):
        lam = manifold_diag.cov_lambda_max(torch.zeros(0, 4))
        self.assertTrue(math.isnan(lam) or lam == 0.0)


# ---------------------------------------------------------------------------
# C. 诊断零侵入
# ---------------------------------------------------------------------------
class DiagInvasionTests(unittest.TestCase):

    def test_20_attach_detach_keeps_state_dict_byte_identical(self):
        m = _small_model()
        before = {k: v.detach().clone() for k, v in m.state_dict().items()}
        log_path = os.path.join(paths.project_tmp_dir(),
                                "manifold_diag_test.jsonl")
        d = manifold_diag.ModelDiagnostics(m, opts=None, every=1,
                                           log_path=log_path)
        d.attach()
        after_attach = m.state_dict()
        self.assertEqual(set(before), set(after_attach))
        for k in before:
            self.assertTrue(torch.equal(before[k], after_attach[k]), k)
        d.detach()
        after_detach = m.state_dict()
        for k in before:
            self.assertTrue(torch.equal(before[k], after_detach[k]), k)

    def test_21_probe_is_restored_after_detach(self):
        m = _small_model()
        blk = m.enc_blocks[0]
        self.assertIsNone(blk.mla_csa.diag_probe)
        d = manifold_diag.ModelDiagnostics(m, every=1)
        d.attach()
        self.assertIsNotNone(blk.mla_csa.diag_probe)
        d.detach()
        self.assertIsNone(blk.mla_csa.diag_probe)

    def test_22_forward_logits_identical_with_and_without_diag(self):
        m = _small_model()
        torch.manual_seed(4242)
        ids = torch.randint(0, 256, (1, 8))
        steps = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
        m.eval()
        with torch.no_grad():
            base = m.forward_train(ids, steps, ids, steps + 2).clone()
            d = manifold_diag.ModelDiagnostics(m, every=1)
            d.attach()
            with_diag = m.forward_train(ids, steps, ids, steps + 2).clone()
            d.detach()
            after = m.forward_train(ids, steps, ids, steps + 2).clone()
        self.assertTrue(torch.equal(base, with_diag))
        self.assertTrue(torch.equal(base, after))

    def test_23_observe_writes_parseable_jsonl(self):
        m = _small_model()
        log_path = os.path.join(paths.project_tmp_dir(),
                                "manifold_diag_observe_test.jsonl")
        if os.path.isfile(log_path):
            os.unlink(log_path)
        d = manifold_diag.ModelDiagnostics(m, every=1, log_path=log_path)
        d.attach()
        d.observe(1, loss=2.5, preclip_grad_norm=7.25)
        d.detach()
        self.assertTrue(os.path.isfile(log_path))
        with open(log_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.assertGreaterEqual(len(rows), 1)
        self.assertEqual(rows[0]["step"], 1)
        self.assertAlmostEqual(rows[0]["preclip_grad_norm"], 7.25)
        self.assertIn("spectra", rows[0])

    def test_24_observe_never_breaks_training_on_bad_input(self):
        m = _small_model()
        d = manifold_diag.ModelDiagnostics(m, every=1)
        d.attach()
        # 故意传坏值：诊断实现必须内部吞掉异常
        d.observe(1, loss=float("nan"), preclip_grad_norm=None,
                  extra={"bad": object()})
        d.detach()

    def test_25_summarize_log_returns_series(self):
        log_path = os.path.join(paths.project_tmp_dir(),
                                "manifold_diag_observe_test.jsonl")
        if not os.path.isfile(log_path):
            self.skipTest("依赖 test_23 产生的日志")
        out = manifold_diag.summarize_log(log_path, last_n=10)
        self.assertIsInstance(out, dict)
        self.assertGreater(out.get("n_records", 0), 0)
        self.assertEqual(len(out["steps"]), out["n_records"])
        self.assertEqual(len(out["sigma_max"]), out["n_records"])

    def test_26_summarize_missing_log_is_empty(self):
        out = manifold_diag.summarize_log(
            os.path.join(paths.project_tmp_dir(), "definitely_absent.jsonl"))
        self.assertEqual(out, {})


# ---------------------------------------------------------------------------
# D. 参数分组
# ---------------------------------------------------------------------------
class ParamGroupTests(unittest.TestCase):

    def test_27_rank1_projections_excluded_from_muon(self):
        m = _small_model()
        groups = model_mod.param_groups_for_optimizers(m)
        rank1 = [n for n, p in m.named_parameters()
                 if p.ndim == 2 and min(p.shape) == 1]
        self.assertGreater(len(rank1), 0)
        muon_ids = {id(p) for p in groups["muon"]}
        for n, p in m.named_parameters():
            if n in rank1:
                self.assertNotIn(id(p), muon_ids, n)

    def test_28_groups_partition_all_trainable_params(self):
        m = _small_model()
        groups = model_mod.param_groups_for_optimizers(m)
        all_ids = [id(p) for p in m.parameters() if p.requires_grad]
        grouped = [id(p) for p in groups["muon"] + groups["adamw"]]
        self.assertEqual(len(grouped), len(set(grouped)), "分组有重复参数")
        self.assertEqual(sorted(all_ids), sorted(grouped))

    def test_29_embedding_head_and_norm_go_to_adamw(self):
        m = _small_model()
        groups = model_mod.param_groups_for_optimizers(m)
        adamw_ids = {id(p) for p in groups["adamw"]}
        for n, p in m.named_parameters():
            if any(k in n for k in ("wte", "lm_head", "coord", "type_emb",
                                    "pos_emb", "norm.weight")):
                self.assertIn(id(p), adamw_ids, n)

    def test_30_param_group_report_counts_are_consistent(self):
        m = _small_model()
        rep = model_mod.param_group_report(m)
        groups = model_mod.param_groups_for_optimizers(m)
        self.assertEqual(rep["muon"], len(groups["muon"]))
        self.assertEqual(rep["adamw"], len(groups["adamw"]))
        self.assertEqual(rep["total"], rep["muon"] + rep["adamw"])
        self.assertGreater(len(rep["rank1_excluded"]), 0)

    def test_31_hidden_2d_matrices_still_in_muon(self):
        m = _small_model()
        groups = model_mod.param_groups_for_optimizers(m)
        muon_ids = {id(p) for p in groups["muon"]}
        hit = 0
        for n, p in m.named_parameters():
            if "ffns." in n and "weight" in n and min(p.shape) > 1:
                self.assertIn(id(p), muon_ids, n)
                hit += 1
        self.assertGreater(hit, 0)

    def test_32_all_muon_group_matrices_have_min_dim_ge_2(self):
        m = _small_model()
        for p in model_mod.param_groups_for_optimizers(m)["muon"]:
            self.assertEqual(p.ndim, 2)
            self.assertGreaterEqual(min(p.shape), 2)


# ---------------------------------------------------------------------------
# E. 复位统一（静态契约 + 功能）
# ---------------------------------------------------------------------------
class ResetContractTests(unittest.TestCase):

    @staticmethod
    def _src(rel):
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            return f.read()

    def test_33_train_py_no_direct_fast_weight_reset(self):
        src = self._src("service/train/train.py")
        self.assertNotIn("m.fast_weight.reset()", src)
        self.assertIn("m.reset_memory()", src)

    def test_34_clip_py_no_direct_fast_weight_reset(self):
        src = self._src("service/train/clip.py")
        self.assertNotIn("m.fast_weight.reset()", src)
        self.assertIn("m.reset_memory()", src)

    def test_35_auto_train_already_uses_reset_memory(self):
        self.assertIn("m.reset_memory()",
                      self._src("service/train/auto_train.py"))

    def test_36_reset_memory_is_the_single_owner(self):
        src = self._src("server/model.py")
        self.assertIn("self.fast_weight.reset()", src)
        self.assertIn("self.momw.reset()", src)
        self.assertIn("somn_mgr.reset()", src)

    def test_37_reset_memory_clears_fw_state(self):
        m = _small_model()
        m.fast_weight._state = (torch.zeros(1, 8, 128, 128), {})
        m.reset_memory()
        self.assertIsNone(m.fast_weight._state)


# ---------------------------------------------------------------------------
# F. Manifold Muon 数学
# ---------------------------------------------------------------------------
class ManifoldMathTests(unittest.TestCase):

    def test_38_msign_exact_is_semi_orthogonal(self):
        x = torch.randn(32, 6, generator=torch.Generator().manual_seed(21))
        q = manifold_muon.msign(x, exact=True)
        self.assertEqual(tuple(q.shape), tuple(x.shape))
        gram = q.T @ q
        diff = float((gram - torch.eye(6)).norm().item())
        self.assertLess(diff, 1e-4)

    def test_39_msign_ns5_improves_orthogonality(self):
        """NS5 五次迭代对随机高斯输入不保证绝对误差很小（既有
        test_fixes.P2-1 已记录 ~0.4 的残差，条件数差时更大），所以这里
        沿用同一不变性判据：比归一化输入更接近半正交 + 奇异值量级合理。"""
        x = torch.randn(32, 6, generator=torch.Generator().manual_seed(22))
        q = manifold_muon.msign(x, steps=5, exact=False)
        err = float((q.T @ q - torch.eye(6)).norm().item())
        xn = x / (x.norm() + 1e-7)
        err_g = float((xn.T @ xn - torch.eye(6)).norm().item())
        self.assertLess(err, err_g)
        sv = torch.linalg.svdvals(q)
        self.assertGreater(float(sv.mean().item()), 0.6)
        self.assertLess(float(sv.max().item()), 1.6)

    def test_40_tangent_project_dgram_removes_diagonal(self):
        m = torch.randn(6, 6, generator=torch.Generator().manual_seed(23))
        out = manifold_muon.tangent_project(m, "dgram")
        self.assertAlmostEqual(float(torch.diagonal(out).abs().max().item()),
                               0.0, delta=1e-6)
        off_src = m - torch.diag_embed(torch.diagonal(m))
        self.assertTrue(torch.allclose(out, off_src, atol=1e-6))

    def test_41_tangent_project_stiefel_is_identity(self):
        m = torch.randn(6, 6)
        self.assertTrue(torch.equal(manifold_muon.tangent_project(m, "stiefel"), m))

    def test_42_tangent_project_rejects_unknown_kind(self):
        with self.assertRaises(ValueError):
            manifold_muon.tangent_project(torch.eye(3), "bogus")

    def test_43_to_tall_roundtrip_preserves_values(self):
        w = torch.randn(6, 24, generator=torch.Generator().manual_seed(24))
        wt, tr = manifold_muon.to_tall(w)
        self.assertTrue(tr)
        self.assertEqual(tuple(wt.shape), (24, 6))
        self.assertTrue(torch.equal(manifold_muon.from_tall(wt, tr), w))

    def test_44_to_tall_keeps_tall_unchanged(self):
        w = torch.randn(24, 6)
        wt, tr = manifold_muon.to_tall(w)
        self.assertFalse(tr)
        self.assertTrue(torch.equal(wt, w))

    def test_45_retract_stiefel_gram_is_identity(self):
        w = _orthonormal(32, 6, seed=25) @ torch.diag(
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        r = manifold_muon.retract(w, "stiefel", exact_msign=True)
        gram = r.T @ r
        self.assertLess(float((gram - torch.eye(6)).norm().item()), 1e-4)

    def test_46_retract_dgram_gram_is_diagonal(self):
        w = torch.randn(24, 6, generator=torch.Generator().manual_seed(26))
        r = manifold_muon.retract(w, "dgram", exact_msign=True)
        self.assertLess(_offgram_fro(r), 1e-3)

    def test_47_retract_dgram_preserves_singular_values(self):
        w = torch.randn(24, 6, generator=torch.Generator().manual_seed(27))
        s_in = torch.linalg.svdvals(w)
        r = manifold_muon.retract(w, "dgram", exact_msign=True)
        s_out = torch.linalg.svdvals(r)
        self.assertTrue(torch.allclose(s_in, s_out, atol=1e-4))

    def test_48_retract_rejects_wide_input(self):
        w = torch.randn(6, 24)
        with self.assertRaises(ValueError):
            manifold_muon.retract(w, "dgram")

    def test_49_direction_shape_matches_wide_weight(self):
        w = torch.randn(6, 24, generator=torch.Generator().manual_seed(28))
        g = torch.randn(6, 24, generator=torch.Generator().manual_seed(29))
        a = manifold_muon.manifold_muon_direction(
            w, g, "dgram", K=2, alpha=0.1, ns_steps=3, exact_msign=True)
        self.assertEqual(tuple(a.shape), tuple(w.shape))
        self.assertTrue(bool(torch.isfinite(a).all()))

    def test_50_direction_K0_degenerates_to_plain_muon(self):
        w = torch.randn(12, 6, generator=torch.Generator().manual_seed(30))
        g = torch.randn(12, 6, generator=torch.Generator().manual_seed(31))
        a = manifold_muon.manifold_muon_direction(
            w, g, "stiefel", K=0, exact_msign=True)
        plain = manifold_muon.msign(g, exact=True)
        self.assertTrue(torch.allclose(a, plain, atol=1e-4))

    def test_51_direction_stiefel_and_dgram_are_finite(self):
        w = _orthonormal(24, 6, seed=32)
        g = torch.randn(24, 6, generator=torch.Generator().manual_seed(33))
        for kind in ("stiefel", "dgram"):
            a = manifold_muon.manifold_muon_direction(
                w, g, kind, K=3, alpha=0.1, ns_steps=3, exact_msign=True)
            self.assertTrue(bool(torch.isfinite(a).all()), kind)

    def test_52_direction_rejects_nonfinite_gradient(self):
        w = torch.randn(12, 6)
        g = torch.randn(12, 6)
        g[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            manifold_muon.manifold_muon_direction(w, g, "stiefel", K=1)

    def test_53_dual_ascent_changes_direction_vs_K0(self):
        """K>0 的对偶上升必须真的改变方向（不是把 K 忽略掉）。"""
        w = torch.randn(12, 6, generator=torch.Generator().manual_seed(34))
        g = torch.randn(12, 6, generator=torch.Generator().manual_seed(35))
        a0 = manifold_muon.manifold_muon_direction(
            w, g, "dgram", K=0, exact_msign=True)
        a3 = manifold_muon.manifold_muon_direction(
            w, g, "dgram", K=3, alpha=0.1, ns_steps=3, exact_msign=True)
        self.assertGreater(float((a0 - a3).norm().item()), 1e-6)


# ---------------------------------------------------------------------------
# G. 优化器集成
# ---------------------------------------------------------------------------
class ManifoldOptimizerTests(unittest.TestCase):

    def test_54_step_keeps_parameter_identity(self):
        layer = nn.Linear(4, 8)
        p = layer.weight
        pid = id(p)
        p.grad = torch.randn_like(p)
        opt = manifold_muon.ManifoldMuon(
            [{"params": [p], "kind": "stiefel"}],
            lr=1e-2, K=1, ns_steps=3, exact_msign=True)
        opt.step()
        self.assertEqual(id(layer.weight), pid)
        self.assertIsInstance(layer.weight, nn.Parameter)

    def test_55_step_stiefel_result_satisfies_constraint(self):
        layer = nn.Linear(4, 8)       # weight [8,4] tall
        layer.weight.grad = torch.randn_like(layer.weight)
        opt = manifold_muon.ManifoldMuon(
            [{"params": [layer.weight], "kind": "stiefel"}],
            lr=1e-2, K=1, ns_steps=3, exact_msign=True)
        opt.step()
        w = layer.weight.detach()
        gram = w.T @ w
        self.assertLess(float((gram - torch.eye(4)).norm().item()), 1e-3)

    def test_56_step_dgram_result_has_diagonal_gram(self):
        layer = nn.Linear(8, 4)       # weight [4,8] wide → 内部转置
        layer.weight.grad = torch.randn_like(layer.weight)
        opt = manifold_muon.ManifoldMuon(
            [{"params": [layer.weight], "kind": "dgram"}],
            lr=1e-2, K=1, ns_steps=3, exact_msign=True)
        opt.step()
        self.assertLess(_offgram_fro(layer.weight.detach()), 1e-3)

    def test_57_select_only_mla_and_ffn(self):
        m = _small_model()
        cfg = dict(spec.manifold_default_config())
        cfg["enabled"] = True
        picked = manifold_muon.select_manifold_params(m, cfg)
        self.assertGreater(len(picked), 0)
        for n, kind in picked.items():
            self.assertTrue(("mla_csa." in n or "mla_hca." in n
                             or ".ffns." in n), n)
            self.assertIn(kind, ("stiefel", "dgram"))

    def test_58_select_excludes_kda(self):
        m = _small_model()
        cfg = dict(spec.manifold_default_config())
        cfg["enabled"] = True
        picked = manifold_muon.select_manifold_params(m, cfg)
        self.assertFalse(any("kda" in n for n in picked))

    def test_59_select_honours_none_kind_and_max_matrices(self):
        m = _small_model()
        cfg = dict(spec.manifold_default_config())
        cfg["enabled"] = True
        cfg["attention"] = "none"
        cfg["mlp"] = "none"
        self.assertEqual(manifold_muon.select_manifold_params(m, cfg), {})
        cfg["mlp"] = "dgram"
        cfg["max_matrices"] = 3
        self.assertLessEqual(len(manifold_muon.select_manifold_params(m, cfg)), 3)

    def test_60_select_deduplicates_tied_parameters(self):
        torch.manual_seed(6060)
        m = model_mod.Gargantua(n_enc_blocks=1, n_dec_blocks=1,
                                d_model=128, vocab_size=256,
                                share_enc_dec=True)
        cfg = dict(spec.manifold_default_config())
        cfg["enabled"] = True
        picked = manifold_muon.select_manifold_params(m, cfg)
        named = dict(m.named_parameters())
        ids = [id(named[n]) for n in picked]
        self.assertEqual(len(ids), len(set(ids)), "共享 Parameter 被重复选中")

    def test_61_build_optimizers_default_is_backward_compatible(self):
        m = _small_model()
        opts = muon.build_optimizers(m, "muon")
        types = [type(o).__name__ for o in opts]
        self.assertIn("Muon", types)
        self.assertIn("AdamW", types)
        self.assertNotIn("ManifoldMuon", types)
        n_params = sum(len(g["params"]) for o in opts for g in o.param_groups)
        self.assertEqual(n_params, len([p for p in m.parameters()
                                        if p.requires_grad]))

    def test_62_build_optimizers_enabled_splits_without_overlap(self):
        m = _small_model()
        cfg = dict(spec.manifold_default_config())
        cfg["enabled"] = True
        cfg["K"] = 1
        opts = muon.build_optimizers(m, "muon", manifold=cfg)
        names = [type(o).__name__ for o in opts]
        self.assertIn("ManifoldMuon", names)
        ids = []
        for o in opts:
            for g in o.param_groups:
                ids += [id(p) for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)), "参数被两个优化器重复持有")
        self.assertEqual(len(ids), len([p for p in m.parameters()
                                        if p.requires_grad]))

    def test_63_adamw_mode_ignores_manifold_config(self):
        m = _small_model()
        cfg = dict(spec.manifold_default_config())
        cfg["enabled"] = True
        opts = muon.build_optimizers(m, "adamw", manifold=cfg)
        self.assertNotIn("ManifoldMuon", [type(o).__name__ for o in opts])

    def test_64_state_dict_has_no_new_keys(self):
        m = _small_model()
        keys_before = set(m.state_dict().keys())
        m.manifold_config = spec.manifold_default_config()
        d = manifold_diag.ModelDiagnostics(m, every=1)
        d.attach()
        d.detach()
        keys_after = set(m.state_dict().keys())
        self.assertEqual(keys_before, keys_after)
        self.assertFalse(any(("manifold" in k or "diag" in k)
                             for k in keys_after))

    def test_65_muon_profile_default_is_off(self):
        p = nn.Parameter(torch.randn(8, 4))
        p.grad = torch.randn_like(p)
        opt = muon.Muon([p], lr=1e-2)
        self.assertFalse(getattr(opt, "profile", False))
        opt.step()
        self.assertEqual(getattr(opt, "ns5_calls", 0), 0)

    def test_66_muon_profile_counts_calls_when_enabled(self):
        p = nn.Parameter(torch.randn(8, 4))
        p.grad = torch.randn_like(p)
        opt = muon.Muon([p], lr=1e-2, profile=True)
        self.assertTrue(opt.profile)
        opt.step()
        self.assertGreaterEqual(opt.ns5_calls, 1)
        self.assertGreater(opt.ns5_seconds, 0.0)


# ---------------------------------------------------------------------------
# H. config 契约
# ---------------------------------------------------------------------------
class ConfigContractTests(unittest.TestCase):

    def setUp(self):
        self.cfg_default = spec.manifold_default_config()
        # 每次从干净 config 起步：test_76 会写出带权重指针的 config，
        # 若不清掉，下一轮 test_74 的 build_model 会去读不存在的权重文件，
        # 测试就变成「只能跑一次」。
        d = os.path.join(paths.project_tmp_dir(), "manifold_test_model")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"name": "manifold_test_model", "vocab_size": 256,
                       "n_enc_blocks": 1, "n_dec_blocks": 1,
                       "d_model": 128}, f)

    def test_67_default_config_keys_and_disabled(self):
        expected = {"enabled", "attention", "mlp", "K", "alpha", "lr",
                    "momentum", "ns_steps", "exclude_kda", "max_matrices",
                    "exact_msign"}
        self.assertEqual(set(self.cfg_default), expected)
        self.assertIs(self.cfg_default["enabled"], False)
        self.assertEqual(self.cfg_default["K"], 20)
        self.assertEqual(self.cfg_default["alpha"], 0.1)

    def test_68_default_lr_momentum_track_spec_constants(self):
        self.assertEqual(self.cfg_default["lr"], spec.MUON_LR)
        self.assertEqual(self.cfg_default["momentum"], spec.MUON_MOMENTUM)
        self.assertEqual(self.cfg_default["ns_steps"], spec.MUON_NS_STEPS)

    def test_69_read_missing_field_returns_disabled_default(self):
        out = common.read_manifold_config({})
        self.assertEqual(out, self.cfg_default)
        self.assertFalse(out["enabled"])

    def test_70_read_partial_config_is_filled(self):
        out = common.read_manifold_config(
            {spec.MANIFOLD_CONFIG_KEY: {"enabled": True,
                                        "attention": "dgram"}})
        self.assertTrue(out["enabled"])
        self.assertEqual(out["attention"], "dgram")
        self.assertEqual(out["mlp"], "dgram")
        self.assertEqual(out["K"], 20)

    def test_71_bad_values_fall_back_to_disabled(self):
        bad = [
            {"enabled": "false"},
            {"enabled": 1},
            {"K": -1},
            {"attention": "bogus"},
            {"alpha": 0},
            {"momentum": 1.5},
            {"max_matrices": -3},
            {"exclude_kda": "yes"},
        ]
        for v in bad:
            with self.subTest(v=v):
                out = common.read_manifold_config(
                    {spec.MANIFOLD_CONFIG_KEY: v})
                self.assertFalse(out["enabled"], v)

    def test_72_validate_rejects_string_bool_explicitly(self):
        with self.assertRaises(ValueError):
            common.validate_manifold_config({"enabled": "false"})

    def test_73_validate_does_not_mutate_input(self):
        v = {"enabled": True}
        snapshot = dict(v)
        out = common.validate_manifold_config(v)
        self.assertEqual(v, snapshot)
        self.assertIsNot(out, v)

    def test_74_build_model_injects_manifold_config(self):
        d = os.path.join(paths.project_tmp_dir(), "manifold_test_model")
        os.makedirs(d, exist_ok=True)
        cfg = {"name": "manifold_test_model", "vocab_size": 256,
               "n_enc_blocks": 1, "n_dec_blocks": 1, "d_model": 128,
               spec.MANIFOLD_CONFIG_KEY: {"enabled": True, "K": 3}}
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        m = common.build_model(d)
        self.assertTrue(m.manifold_config["enabled"])
        self.assertEqual(m.manifold_config["K"], 3)

    def test_75_build_model_old_dir_is_disabled_and_state_dict_equal(self):
        d = os.path.join(paths.project_tmp_dir(), "manifold_test_model")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"name": "manifold_test_model", "vocab_size": 256,
                       "n_enc_blocks": 1, "n_dec_blocks": 1,
                       "d_model": 128}, f)
        m_old = common.build_model(d)
        self.assertFalse(m_old.manifold_config["enabled"])
        m_ref = _small_model()
        self.assertEqual(set(m_old.state_dict().keys()),
                         set(m_ref.state_dict().keys()))

    def test_76_checkpoint_roundtrip_preserves_manifold_field(self):
        """只验 config 透传：patch 掉权重落盘与旧权重回收，避免测试写
        几十 MB safetensors（本机磁盘长期紧张），也不触发批量删除守卫。"""
        from unittest import mock
        d = os.path.join(paths.project_tmp_dir(), "manifold_test_model")
        os.makedirs(d, exist_ok=True)
        cfg = {"name": "manifold_test_model", "vocab_size": 256,
               "n_enc_blocks": 1, "n_dec_blocks": 1, "d_model": 128,
               spec.MANIFOLD_CONFIG_KEY: {"enabled": True, "K": 7}}
        m = _small_model()
        with mock.patch.object(common, "_atomic_save_file",
                               lambda sd, path: None), \
             mock.patch.object(common, "_recycle_old_weights",
                               lambda model_dir, c: None):
            common.save_checkpoint(d, m, cfg)
        saved = common.read_config(d)
        self.assertEqual(saved[spec.MANIFOLD_CONFIG_KEY]["K"], 7)
        self.assertTrue(saved[spec.MANIFOLD_CONFIG_KEY]["enabled"])

    def test_77_config_key_is_stable_string(self):
        self.assertEqual(spec.MANIFOLD_CONFIG_KEY, "manifold_muon")


# ---------------------------------------------------------------------------
# I. 架构师批复补齐（2026-11-13）：top-10 奇异值 / 条件数 / 采集覆盖上限
# ---------------------------------------------------------------------------
class SpectrumContractTests(unittest.TestCase):

    def test_78_top10_is_descending_and_capped_at_10(self):
        w = torch.diag(torch.arange(1.0, 13.0))
        st = manifold_diag.spectrum_stats(w)
        top = st["sigma_top10"]
        self.assertEqual(len(top), 10)
        self.assertEqual(top, sorted(top, reverse=True))
        self.assertAlmostEqual(top[0], 12.0, delta=1e-3)
        self.assertAlmostEqual(top[-1], 3.0, delta=1e-3)

    def test_79_condition_number_equals_sigma_ratio(self):
        w = torch.diag(torch.arange(1.0, 7.0))
        st = manifold_diag.spectrum_stats(w)
        self.assertAlmostEqual(st["cond"], 6.0, delta=1e-3)
        self.assertFalse(st["cond_clamped"])
        self.assertAlmostEqual(st["sigma_max"] / st["sigma_min"], st["cond"],
                               delta=1e-3)

    def test_80_rank_deficient_cond_is_finite_and_flagged(self):
        w = torch.zeros(6, 6)
        w[0, 0] = 2.0
        st = manifold_diag.spectrum_stats(w)
        self.assertTrue(st["cond_clamped"])
        self.assertTrue(math.isfinite(st["cond"]))

    def test_81_max_matrices_zero_covers_all_mla_and_ffn(self):
        m = _small_model()
        full = manifold_diag.ModelDiagnostics(
            m, every=1, max_matrices=0)._collect_spectra()
        capped = manifold_diag.ModelDiagnostics(
            m, every=1, max_matrices=24)._collect_spectra()
        # 1 enc + 1 dec block，每 block 2×MLA(6) + 5×FFN(2) = 22 个矩阵
        self.assertEqual(len(full), 44)
        self.assertEqual(len(capped), 24)

    def test_82_summarize_log_exposes_cond_max_series(self):
        m = _small_model()
        log_path = os.path.join(paths.project_tmp_dir(),
                                "manifold_diag_cond_test.jsonl")
        if os.path.isfile(log_path):
            os.unlink(log_path)
        d = manifold_diag.ModelDiagnostics(m, every=1, log_path=log_path,
                                           max_matrices=0)
        d.observe(1, loss=1.0, preclip_grad_norm=2.0)
        d.detach()
        out = manifold_diag.summarize_log(log_path, last_n=10)
        self.assertIn("cond_max", out)
        self.assertEqual(len(out["cond_max"]), out["n_records"])
        self.assertIsNotNone(out["cond_max"][0])


# ---------------------------------------------------------------------------
# J. 交叉审计修复（2026-11-13）：诊断顺序 / 缓存死角 / Muon 校准开关
# ---------------------------------------------------------------------------
class AuditFixContractTests(unittest.TestCase):

    @staticmethod
    def _src(rel):
        with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
            return f.read()

    def test_83_train_py_observes_before_reset(self):
        src = self._src("service/train/train.py")
        i_obs = src.index("diag.observe(step")
        i_reset = src.index("m.reset_memory()", i_obs)
        self.assertLess(i_obs, i_reset,
                        "observe 必须在 reset_memory 之前，否则 KDA 状态恒为空")
        self.assertIn("KDA 状态恒为 present=False", src)

    def test_84_auto_train_observes_before_reset_and_is_opt_in(self):
        src = self._src("service/train/auto_train.py")
        self.assertIn("GARGANTUA_DIAG_EVERY", src)
        self.assertIn("def _open_diag", src)
        self.assertIn("def _rebind_diag", src)
        i_obs = src.index("diag.observe(")
        i_reset = src.index("m.reset_memory()", i_obs)
        self.assertLess(i_obs, i_reset)
        self.assertIn("finally:", src[:src.index("def _train_clip_batch")])

    def test_85_somn_router_cache_is_cleared_by_reset_memory(self):
        # 独立实例，避免污染进程内共享的 _small_model()
        m = model_mod.Gargantua(n_enc_blocks=1, n_dec_blocks=1,
                                d_model=128, vocab_size=256)
        import somn
        somn_mgr = somn.attach_somn(m, m.d_model)
        somn_mgr.router.last_base_scores = [0.9, 0.8]
        somn_mgr.router.last_boosted_scores = [1.9, 1.8]
        m.reset_memory()
        self.assertEqual(somn_mgr.router.last_base_scores, [])
        self.assertEqual(somn_mgr.router.last_boosted_scores, [])
        self.assertEqual(somn_mgr.stats()["scores"], [])

    def test_86_momw_router_reset_cache_clears_all_last_fields(self):
        import momw
        r = momw.MoMWRouter(64)
        r.last_raw = [0.1]
        r.last_scores = [0.2]
        r.last_cos_anchor = [0.3]
        r.last_cos_state = [0.4]
        r.last_gamma = [0.5]
        r.reset_cache()
        for field in ("last_raw", "last_scores", "last_cos_anchor",
                      "last_cos_state", "last_gamma"):
            self.assertEqual(getattr(r, field), [], field)
        # MoMW.reset() 必须真的调用它（router 不随 pool 重建）
        self.assertIn("self.router.reset_cache()",
                      self._src("server/momw.py"))

    def test_87_build_optimizers_default_keeps_muon_defaults(self):
        m = _small_model()
        opts = muon.build_optimizers(m, "muon")
        muons = [o for o in opts if type(o).__name__ == "Muon"]
        self.assertEqual(len(muons), 1)
        g = muons[0].param_groups[0]
        self.assertIs(g["rms_align"], True)
        self.assertEqual(g["weight_decay"], 0.0)

    def test_88_build_optimizers_honours_calibration_overrides(self):
        m = _small_model()
        opts = muon.build_optimizers(
            m, "muon", muon_rms_align=False, muon_weight_decay=0.01)
        muons = [o for o in opts if type(o).__name__ == "Muon"]
        self.assertEqual(len(muons), 1)
        g = muons[0].param_groups[0]
        self.assertIs(g["rms_align"], False)
        self.assertAlmostEqual(g["weight_decay"], 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
