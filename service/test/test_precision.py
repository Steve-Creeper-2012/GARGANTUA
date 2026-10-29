"""precision 模块测试：dtype 校验、原生 cast、低比特模拟的量化误差界。

运行：cd ROOT && python -m unittest service.test.test_precision -v
前置：torch 可用（项目 environment/venv）。
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

import unittest

import torch

from precision import (ALL_NAMES, DEFAULT, apply_precision, dtype_summary,
                       parse_precision, validate_dtype)


def _small_model():
    """一个极小的 nn.Module，便于快速施加精度。"""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)
    )


def _max_rel_err(orig: torch.Tensor, quant: torch.Tensor) -> float:
    denom = orig.abs().amax()
    if denom == 0:
        return 0.0
    return float(((orig - quant).abs().amax() / denom).item())


def _max_abs_err(orig: torch.Tensor, quant: torch.Tensor) -> float:
    return float((orig - quant).abs().amax().item())


class ValidateTest(unittest.TestCase):
    def test_all_names_present(self):
        self.assertEqual(set(ALL_NAMES),
                         {"fp64", "fp32", "fp16", "bf16",
                          "fp8", "fp4", "int16", "int8", "int4", "int2"})

    def test_valid_names_ok(self):
        for n in ALL_NAMES:
            validate_dtype(n)          # 不抛即过

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            validate_dtype("fp7")
        with self.assertRaises(ValueError):
            validate_dtype(None)

    def test_default_is_fp32(self):
        self.assertEqual(DEFAULT, "fp32")


class ParseTest(unittest.TestCase):
    def test_native_kind(self):
        for n in ("fp64", "fp32", "fp16", "bf16"):
            info = parse_precision(n)
            self.assertEqual(info["kind"], "native")
            self.assertIn("torch_dtype", info)

    def test_sim_kind(self):
        info = parse_precision("int8")
        self.assertEqual(info["kind"], "sim")
        self.assertEqual(info["spec"], 8)
        info = parse_precision("fp4")
        self.assertEqual(info["spec"], "e2m1")


class NativeApplyTest(unittest.TestCase):
    def test_fp32_default_noop(self):
        m = _small_model()
        before = [p.detach().clone() for p in m.parameters()]
        apply_precision(m, "fp32")
        for b, p in zip(before, m.parameters()):
            self.assertTrue(torch.allclose(b, p))

    def test_fp16_cast(self):
        m = _small_model()
        apply_precision(m, "fp16")
        for p in m.parameters():
            self.assertEqual(p.dtype, torch.float16)

    def test_bf16_cast(self):
        m = _small_model()
        apply_precision(m, "bf16")
        for p in m.parameters():
            self.assertEqual(p.dtype, torch.bfloat16)

    def test_fp64_cast(self):
        m = _small_model()
        apply_precision(m, "fp64")
        for p in m.parameters():
            self.assertEqual(p.dtype, torch.float64)


class SimApplyTest(unittest.TestCase):
    """低比特模拟：容器仍是 fp32（CPU 可跑），参数被压到目标网格。"""

    def test_container_stays_fp32(self):
        for n in ("fp8", "fp4", "int16", "int8", "int4", "int2"):
            m = _small_model()
            apply_precision(m, n)
            for p in m.parameters():
                self.assertEqual(p.dtype, torch.float32,
                                 f"{n} 应保持 fp32 容器")

    def test_values_finite(self):
        for n in ("fp8", "fp4", "int16", "int8", "int4", "int2"):
            m = _small_model()
            apply_precision(m, n)
            for p in m.parameters():
                self.assertTrue(torch.isfinite(p).all(),
                                f"{n} 产生非有限值")

    def test_int_quant_error_bound(self):
        # 对称整数量化相对误差上限 ≈ 1/(2*maxq)。
        rng = torch.manual_seed(0)
        x = torch.randn(1024, generator=rng).clamp(-3, 3)
        for bits in (16, 8, 4, 2):
            name = f"int{bits}"
            q = _sim_quant(name, x)
            maxq = (1 << (bits - 1)) - 1
            self.assertLessEqual(
                _max_rel_err(x, q), 1.0 / (2 * maxq) + 1e-5,
                f"{name} 相对误差超出 {1.0/(2*maxq):.4f} 界")

    def test_int_quant_step_is_grid(self):
        # 反量化结果应落在量化网格上（整数倍 scale）。
        x = torch.linspace(-1, 1, 50)
        for bits in (8, 4, 2):
            name = f"int{bits}"
            q = _sim_quant(name, x)
            maxq = (1 << (bits - 1)) - 1
            # 取一个非零元素，验证其是 scale 的整数倍（scale = max|x|/maxq）
            s = x.abs().amax()
            scale = s / maxq
            nz = q[q.abs() > 1e-9]
            if nz.numel():
                ratios = (nz / scale).abs()
                self.assertTrue(torch.allclose(
                    ratios, ratios.round(), atol=1e-5),
                    f"{name} 反量化值未落在整数倍网格上")

    def test_fp8_lossy_but_bounded(self):
        # fp8 e4m3：相对误差在有量级的数值上应较小（≤0.2）。
        x = (torch.rand(2048) + 0.05) * 1.5     # 避开近零亚正规区
        q = _sim_quant("fp8", x)
        self.assertLessEqual(_max_rel_err(x, q), 0.2)

    def test_fp4_lossy_bounded_abs(self):
        # fp4 e2m1：最大相邻网格间距=2（4→6），半距=1 个网格单位=scale。
        x = torch.linspace(-6, 6, 1000)
        q = _sim_quant("fp4", x)
        s = x.abs().amax()
        scale = s / 6.0
        self.assertLessEqual(_max_abs_err(x, q), scale + 1e-5)

    def test_zero_preserved(self):
        x = torch.zeros(16)
        for n in ("fp8", "fp4", "int16", "int8", "int4", "int2"):
            q = _sim_quant(n, x.clone())
            self.assertTrue(torch.allclose(q, torch.zeros_like(q)))

    def test_idempotent(self):
        # 对模拟格式二次施加应基本不变。
        x = torch.randn(256)
        for n in ("fp8", "fp4", "int8", "int4", "int2"):
            q1 = _sim_quant(n, x)
            q2 = _sim_quant(n, q1)
            self.assertLessEqual(_max_abs_err(q1, q2), 1e-4)

    def test_fp4_small_std_weights_not_all_zero(self):
        # 2026-10-27 审计回归：初始权重 std≈0.02，旧实现（无 scale 归一化）
        # 把 100% 的权重压成 0，模型输出近似均匀随机。
        x = torch.randn(10000) * 0.02
        q = _sim_fp4_direct(x)
        zero_rate = (q == 0).float().mean().item()
        self.assertLess(zero_rate, 0.9)          # 绝大多数权重必须非零
        # 重建误差有界：|err| ≤ 半个最大网格间距 = scale（amax/6）
        scale = x.abs().amax().item() / 6.0
        self.assertLessEqual(_max_abs_err(x, q), scale + 1e-7)
        # 量级保持：量化后 amax 应与原 amax 同数量级（网格含 6 → 精确命中）
        self.assertAlmostEqual(q.abs().amax().item(),
                               x.abs().amax().item(), delta=scale)

    def test_fp4_grid_values_exact(self):
        # scale 归一化后，amax 应精确落在网格最大值 6 上（重建精确）。
        x = torch.tensor([0.02, -0.02, 0.01, 0.0, 0.03])
        q = _sim_fp4_direct(x)
        amax = x.abs().amax().item()
        self.assertAlmostEqual(q.abs().amax().item(), amax, places=7)


def _sim_fp4_direct(x: torch.Tensor) -> torch.Tensor:
    from precision import _sim_fp4
    return _sim_fp4(x)


def _sim_quant(name: str, x: torch.Tensor) -> torch.Tensor:
    """直接用 precision 内部的量化函数（绕过 model，便于误差检验）。"""
    from precision import _sim_fp4, _sim_int, _to_fp8_grid
    if name == "fp8":
        return _to_fp8_grid(x, "e4m3fn")
    if name == "fp4":
        return _sim_fp4(x)
    return _sim_int(x, int(name[3:]))


class SummaryTest(unittest.TestCase):
    def test_summary_structure(self):
        for n in ALL_NAMES:
            s = dtype_summary(n)
            self.assertEqual(s["name"], n)
            self.assertIn(s["kind"], ("native", "sim"))
            self.assertIn("bits", s)
            self.assertIn("note", s)
            # 验证 SIMULATED 项的 bits 与声明一致
            if n == "fp8":
                self.assertEqual(s["bits"], 8)
            elif n == "fp4":
                self.assertEqual(s["bits"], 4)
            elif n.startswith("int"):
                self.assertEqual(s["bits"], int(n[3:]))


if __name__ == "__main__":
    unittest.main()
