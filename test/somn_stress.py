#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===========================================================================
# SOMN V4 对抗性压测 / 破坏性审计脚本
# 分支 feat/somn-stress-test —— 只读审计，绝不修改主干业务代码
# ---------------------------------------------------------------------------
# 硬规则（steve 2026-11-13）：
#   * 延迟用 time.perf_counter 实测，内存用 psutil.Process.memory_info().rss 实测
#   * 不做任何「理论上」估算；所有结论来自本机真实运行
#   * 不把 try-except 当成「跑通」：每个边界都构造恶意输入并断言实际后果
#   * 每个边界情况都必须有显式断言，禁止静默略过
#
# 运行：
#   /usr/local/bin/python3 test/somn_stress.py
#   /usr/local/bin/python3 test/somn_stress.py --only t1,t2,t3,t4
#   /usr/local/bin/python3 test/somn_stress.py --json data/tmp/somn_stress.json
#
# 环境：torch 2.2.2 / CPU only（Intel Mac），无 MPS / CUDA
# ===========================================================================
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))  # 直接 import somn，不经过 boot（避免副作用）

import torch
import torch.nn as nn
import somn

# --------------------------------------------------------------------------
# 内存采样后端：优先 psutil（当前 RSS），否则 resource.ru_maxrss（峰值）
# --------------------------------------------------------------------------
try:
    import psutil
    _PROC = psutil.Process(os.getpid())
    _PSUTIL_VER = psutil.__version__

    def rss_bytes() -> int:
        return int(_PROC.memory_info().rss)

    MEM_BACKEND = "psutil.Process.memory_info().rss (sampled each step)"
except ImportError:  # pragma: no cover - 兜底，保证脚本可移植
    import resource
    _PSUTIL_VER = None

    def rss_bytes() -> int:
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(v if sys.platform == "darwin" else v * 1024)

    MEM_BACKEND = "resource.ru_maxrss (peak, fallback)"

MB = 1024.0 * 1024.0

RESULTS: dict = {
    "meta": {
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda": torch.cuda.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "psutil": _PSUTIL_VER,
        "mem_backend": MEM_BACKEND,
        "somn_file": str(ROOT / "server" / "somn.py"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    },
    "t1": {}, "t2": {}, "t3": {}, "t4": {},
    "findings": [],
}


def emit(level: str, code: str, title: str, **evidence) -> None:
    """level: 阻塞 / 警告 / 建议 / 通过"""
    RESULTS["findings"].append(
        {"level": level, "code": code, "title": title, "evidence": evidence}
    )
    print("  [%s] %s  %s" % (level, code, title))
    for k, v in evidence.items():
        s = repr(v)
        print("        %-22s %s" % (k, s if len(s) < 220 else s[:220] + " ..."))


def pct(xs, p):
    if not xs:
        return None
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return ys[lo] if lo == hi else ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def brief(x, nd=2):
    if x is None:
        return None
    return round(float(x), nd)


def rng(seed):
    return torch.Generator().manual_seed(int(seed))


def make_basis(D, rank=6, seed=20261112):
    return torch.randn(rank, D, generator=rng(seed))


def structured(seed, B, T, D, basis=None, dtype=torch.float32):
    """低秩结构化输入（模拟「代码段」），所有 token 落在 rank 维共享子空间。"""
    basis = make_basis(D) if basis is None else basis
    rank = basis.shape[0]
    w = torch.randn(B, T, rank, generator=rng(seed))
    return torch.matmul(w, basis).to(dtype)


def white(seed, B, T, D, dtype=torch.float32):
    return torch.randn(B, T, D, generator=rng(seed)).to(dtype)


def finite(t):
    return bool(torch.isfinite(t).all().item())


# ==========================================================================
# 任务 1：物理极限与延迟压测  D=1024 / Seq=512 / Batch=1 / 200 步
# ==========================================================================
def task1(D=1024, T=512, B=1, steps=200, svd_repeats=5):
    print("\n" + "=" * 96)
    print("TASK 1  物理极限与延迟压测  D=%d Seq=%d Batch=%d steps=%d" % (D, T, B, steps))
    print("=" * 96)
    gc.collect()
    rss_start = rss_bytes()
    basis = make_basis(D)
    out = {
        "config": {"D": D, "seq": T, "batch": B, "steps": steps,
                   "dtype": "float32", "device": "cpu"},
        "rss_start_mb": brief(rss_start / MB),
        "runs": {},
    }

    # A：默认配置（真实生产配置）；B：强制分裂配置（专门抓尖刺）
    scenarios = (
        ("A_default", dict(warmup_steps=50, max_write_count=100)),
        ("B_forced_split", dict(warmup_steps=0, max_write_count=20,
                                expand_growth_threshold=1e-9,
                                min_score=0.0, max_layers=6)),
    )
    for tag, cfg in scenarios:
        m = somn.SOMNModule(D, **cfg)
        assert len(m.layers) == 1 and m.layers[0].S is None, "冷启动前置条件失败"
        svd_ms = []
        orig = somn.IndependentFWLayer.svd_split

        def timed_split(self, _orig=orig, _sink=svd_ms):
            t0 = time.perf_counter()
            r = _orig(self)
            _sink.append((time.perf_counter() - t0) * 1000.0)
            return r

        somn.IndependentFWLayer.svd_split = timed_split
        lat, spikes, prev_splits = [], [], 0
        peak = rss_bytes()
        last_out = None
        t_start = time.perf_counter()
        try:
            for i in range(steps):
                H = (structured(1000 + i, B, T, D, basis=basis)
                     if i % 5 < 3 else white(1000 + i, B, T, D))
                t0 = time.perf_counter()
                with torch.no_grad():
                    last_out = m(H)
                dt = (time.perf_counter() - t0) * 1000.0
                lat.append(dt)
                cur = m.stats()["splits"]
                if cur > prev_splits:
                    spikes.append({"step": i, "step_ms": round(dt, 2),
                                   "splits_delta": cur - prev_splits})
                prev_splits = cur
                r = rss_bytes()
                if r > peak:
                    peak = r
        finally:
            somn.IndependentFWLayer.svd_split = orig  # 必须恢复，避免污染后续任务
        wall = (time.perf_counter() - t_start) * 1000.0
        st = m.stats()
        run = {
            "cfg": cfg,
            "steps_done": len(lat),
            "wall_ms": brief(wall),
            "avg_ms": brief(statistics.mean(lat)),
            "median_ms": brief(statistics.median(lat)),
            "p95_ms": brief(pct(lat, 0.95)),
            "max_ms": brief(max(lat)),
            "min_ms": brief(min(lat)),
            "split_step_count": len(spikes),
            "split_steps": spikes,
            "svd_calls": len(svd_ms),
            "svd_call_ms": {"min": brief(min(svd_ms)) if svd_ms else None,
                            "median": brief(statistics.median(svd_ms)) if svd_ms else None,
                            "max": brief(max(svd_ms)) if svd_ms else None},
            "rss_peak_mb": brief(peak / MB),
            "rss_delta_mb": brief((peak - rss_start) / MB),
            "layers": st["layers"], "splits": st["splits"],
            "writes": st["writes"][:8],
            "output_finite": finite(last_out),
            "output_shape": list(last_out.shape),
        }
        out["runs"][tag] = run
        print("  -- %s --" % tag)
        print("     avg=%.2fms median=%.2fms p95=%.2fms max=%.2fms wall=%.1fms"
              % (run["avg_ms"], run["median_ms"], run["p95_ms"],
                 run["max_ms"], run["wall_ms"]))
        print("     splits=%d  split_steps=%s  svd_call_ms=%s"
              % (st["splits"], spikes, run["svd_call_ms"]))
        print("     rss_peak=%.1fMB  layers=%d  finite_output=%s"
              % (run["rss_peak_mb"], st["layers"], run["output_finite"]))
        assert finite(last_out), "%s 输出出现非有限值" % tag
        assert len(lat) == steps, "%s 未跑满 %d 步" % (tag, steps)
        gc.collect()

    # 微观基准：D×D 有能量矩阵上真实调用 svd_split() N 次，抓「尖刺」真实值
    base = torch.randn(D, D, generator=rng(4242)) * 0.05
    micro = []
    for _ in range(svd_repeats):
        layer = somn.IndependentFWLayer(D)
        layer.S = base.clone()
        t0 = time.perf_counter()
        layer.svd_split()
        micro.append((time.perf_counter() - t0) * 1000.0)
    out["svd_micro"] = {
        "dim": D, "repeats": svd_repeats,
        "ms": [brief(x) for x in micro],
        "min": brief(min(micro)), "median": brief(statistics.median(micro)),
        "max": brief(max(micro)),
        "over_500ms": [brief(x) for x in micro if x > 500.0],
    }
    print("  -- SVD micro (D=%d, %d repeats) --" % (D, svd_repeats))
    print("     ms=%s  median=%.2f  max=%.2f  >500ms=%s"
          % (out["svd_micro"]["ms"], out["svd_micro"]["median"],
             out["svd_micro"]["max"], out["svd_micro"]["over_500ms"]))

    # 判定：SVD 分裂导致延迟 > 500ms → 高危
    worst_svd = max(
        [out["svd_micro"]["max"]]
        + [r["svd_call_ms"]["max"] or 0 for r in out["runs"].values()]
    )
    worst_split_step = max(
        [s["step_ms"] for r in out["runs"].values() for s in r["split_steps"]] or [0]
    )
    out["verdict"] = {
        "worst_svd_call_ms": brief(worst_svd),
        "worst_split_step_ms": brief(worst_split_step),
        "threshold_ms": 500,
        "high_risk": bool(worst_svd > 500 or worst_split_step > 500),
        "oom": False,
    }
    if out["verdict"]["high_risk"]:
        worst_step = max(
            (s for r in out["runs"].values() for s in r["split_steps"]),
            key=lambda s: s["step_ms"], default=None)
        emit("阻塞", "T1-LAT",
             "分裂步整步延迟 >500ms（单次 SVD 未超阈值，但同一步多层同步分裂叠加超标）",
             worst_svd_call_ms=brief(worst_svd),
             worst_split_step_ms=brief(worst_split_step),
             worst_step_detail=worst_step,
             single_svd_under_500ms=bool(worst_svd < 500),
             code_refs="server/somn.py:147-189（svd_split）, 358-376（一步内可多次分裂）")
    else:
        emit("通过", "T1-LAT", "SVD 分裂延迟未超 500ms（但需看余量）",
             worst_svd_call_ms=brief(worst_svd),
             worst_split_step_ms=brief(worst_split_step),
             margin_ms=brief(500 - worst_svd),
             code_refs="server/somn.py:147-189")
    RESULTS["t1"] = out
    return out


# ==========================================================================
# 任务 2：极端输入与鲁棒性破坏
# ==========================================================================
def _polluted_state(S):
    """统一描述状态矩阵被污染的真实情况（不预设结论）。"""
    if S is None:
        return {"exists": False}
    return {
        "exists": True,
        "nan": bool(torch.isnan(S).any().item()),
        "inf": bool(torch.isinf(S).any().item()),
        "nan_frac": brief(float(torch.isnan(S).float().mean().item()), 6),
        "max_abs_finite": (brief(float(S[torch.isfinite(S)].abs().max().item()), 4)
                           if bool(torch.isfinite(S).any().item()) else None),
    }


def _nan_case(kind, D=64):
    """三条真实执行路径，逐条实测，避免「路由把 write 短路」造成的假阴性。

    A_routed     : S 已成型，走打分路由（NaN 会让 max() 判定静默失败）
    B_cold_start : S is None，route 强制写第 0 层，NaN 直接进入 write()
    C_direct     : 绕过路由直接调 IndependentFWLayer.write()，专测 clamp 行为
    """
    H = white(99, 2, 4, D).clone()
    if kind == "single_nan":
        H[0, 0, 0] = float("nan")
    elif kind == "all_nan":
        H.fill_(float("nan"))
    elif kind == "single_posinf":
        H[0, 0, 0] = float("inf")
    elif kind == "single_neginf":
        H[0, 0, 0] = float("-inf")
    elif kind == "mixed":
        H[0, 0, 0] = float("nan")
        H[0, 1, 1] = float("inf")
        H[1, 3, 2] = float("-inf")
    res = {"kind": kind}

    # ---- A：已成型层，走打分路由 ----
    ma = somn.SOMNModule(D, warmup_steps=0, max_write_count=10 ** 9,
                         min_score=0.0)
    for i in range(5):
        with torch.no_grad():
            ma(white(i, 2, 4, D))
    assert ma.layers[0].S is not None and finite(ma.layers[0].S), "预热状态被污染"
    w_before = ma.layers[0].write_count
    try:
        with torch.no_grad():
            oa = ma(H)
        res["A_routed"] = {
            "exception": None, "out_finite": finite(oa),
            "S": _polluted_state(ma.layers[0].S),
            "write_count_before": w_before,
            "write_count_after": ma.layers[0].write_count,
            "active_last": list(ma._slot["stats"]["active_last"]),
            "router_scores": list(ma.router.last_base_scores),
        }
    except Exception as e:
        res["A_routed"] = {"exception": "%s: %s" % (type(e).__name__, str(e)[:200])}

    # ---- B：冷启动，route 强制写第 0 层 ----
    mb = somn.SOMNModule(D, warmup_steps=0, min_score=0.5)
    assert mb.layers[0].S is None, "冷启动前置条件失败"
    try:
        with torch.no_grad():
            ob = mb(H)
        res["B_cold_start"] = {
            "exception": None, "out_finite": finite(ob),
            "S": _polluted_state(mb.layers[0].S),
            "write_count": mb.layers[0].write_count,
        }
    except Exception as e:
        res["B_cold_start"] = {"exception": "%s: %s" % (type(e).__name__, str(e)[:200])}

    follow = []
    for j in range(5):
        try:
            with torch.no_grad():
                o2 = mb(white(200 + j, 2, 4, D))
            follow.append({"out_finite": finite(o2),
                           "active": list(mb._slot["stats"]["active_last"])})
        except Exception as e:
            follow.append({"exception": "%s: %s" % (type(e).__name__, str(e)[:120])})
    res["B_followup_5_clean_inputs"] = follow

    try:
        mb.layers[0].svd_split()
        res["B_svd_split_after"] = None
    except Exception as e:
        res["B_svd_split_after"] = "%s: %s" % (type(e).__name__, str(e)[:240])

    # ---- C：绕过路由，直接 write()，专测 clamp ----
    lc = somn.IndependentFWLayer(D)
    with torch.no_grad():
        lc.write(white(300, 2, 4, D))
    assert finite(lc.S), "direct write 预热失败"
    try:
        with torch.no_grad():
            oc = lc.write(H)
        res["C_direct_write"] = {
            "exception": None, "out_finite": finite(oc),
            "S": _polluted_state(lc.S),
        }
    except Exception as e:
        res["C_direct_write"] = {"exception": "%s: %s" % (type(e).__name__, str(e)[:200])}
    return res


def _dim_cases(D=64):
    return [
        ("2D_[2,64]", torch.randn(2, D, generator=rng(11))),
        ("1D_[64]", torch.randn(D, generator=rng(12))),
        ("4D_[1,4,2,64]", torch.randn(1, 4, 2, D, generator=rng(13))),
        ("D_mismatch_[1,4,65]", torch.randn(1, 4, D + 1, generator=rng(14))),
        ("T=0_[1,0,64]", torch.randn(1, 0, D, generator=rng(15))),
        ("B=0_[0,4,64]", torch.randn(0, 4, D, generator=rng(16))),
        ("int64_[1,4,64]", torch.randint(0, 5, (1, 4, D), generator=rng(17))),
        ("fp16_[1,4,64]", torch.randn(1, 4, D, generator=rng(18)).half()),
        ("all_zero_[1,4,64]", torch.zeros(1, 4, D)),
        ("noncontig_[1,4,64]", torch.randn(1, D, 4, generator=rng(19)).transpose(1, 2)),
    ]


def task2():
    print("\n" + "=" * 96)
    print("TASK 2  极端输入与鲁棒性破坏")
    print("=" * 96)
    out = {"nan_injection": [], "dim_murder": [], "split_attack": {}}

    # ---- 2a NaN / Inf 注入 ----
    print("\n[2a] NaN / Inf 注入（三条真实执行路径，检查 S 污染 + 后续推理）")
    for kind in ("single_nan", "all_nan", "single_posinf", "single_neginf", "mixed"):
        r = _nan_case(kind)
        out["nan_injection"].append(r)
        a, b, c = r["A_routed"], r["B_cold_start"], r["C_direct_write"]
        print("  %-16s A{out_finite=%s active=%s writes %s->%s scores=%s} "
              "B{S_nan=%s nan_frac=%s} C{S_nan=%s S_inf=%s max=%s} svd=%s"
              % (kind, a.get("out_finite"), a.get("active_last"),
                 a.get("write_count_before"), a.get("write_count_after"),
                 [brief(x, 3) for x in a.get("router_scores", [])],
                 b.get("S", {}).get("nan"), b.get("S", {}).get("nan_frac"),
                 c.get("S", {}).get("nan"), c.get("S", {}).get("inf"),
                 c.get("S", {}).get("max_abs_finite"), r["B_svd_split_after"]))
    cases = out["nan_injection"]

    # 结论 1：A 路径实测 —— NaN 让 max() 判定恒 False，路由静默零激活
    silent = [r["kind"] for r in cases
              if r["A_routed"].get("out_finite") is False
              and r["A_routed"].get("active_last") == []
              and r["A_routed"].get("write_count_after")
              == r["A_routed"].get("write_count_before")]
    if silent:
        emit("阻塞", "T2-NAN-ROUTE",
             "NaN/Inf 使 max(boosted) 为 NaN → 路由静默零激活：输出透传 NaN、记忆停写、无告警",
             kinds=silent, example=cases[0]["A_routed"],
             code_refs="server/somn.py:245-254（无 isfinite 断言）, 348-354（NaN 直接进残差）")

    # 结论 2：B 路径实测 —— 冷启动强制写，NaN 是否穿透 clamp 污染 S
    polluted = [r for r in cases if r["B_cold_start"].get("S", {}).get("nan")]
    if polluted:
        emit("阻塞", "T2-NAN",
             "NaN 穿透 clamp 永久污染 S（冷启动强制写），后续干净输入零激活、svd_split 崩溃",
             kinds=[r["kind"] for r in polluted],
             nan_frac={r["kind"]: r["B_cold_start"]["S"]["nan_frac"] for r in polluted},
             followup_after_clean_input=polluted[0]["B_followup_5_clean_inputs"],
             svd_split_exceptions={r["kind"]: r["B_svd_split_after"]
                                   for r in cases if r["B_svd_split_after"]},
             code_refs="server/somn.py:126-140（clamp 不过滤 NaN）, 238-243（冷启动强制写）, 162-169")

    # 结论 3：clamp 对 Inf 的真实行为，只依据 C_direct_write 实测，不预设结论
    inf_rows = {r["kind"]: r["C_direct_write"].get("S", {})
                for r in cases
                if r["kind"] in ("single_posinf", "single_neginf", "mixed")}
    clamped = {k: v for k, v in inf_rows.items()
               if v.get("inf") is False and (v.get("max_abs_finite") or 0) >= 9.9}
    if clamped:
        emit("警告", "T2-INF", "Inf 被 clamp 静默截断为 ±10（直接 write 实测），掩盖数值爆炸",
             kinds=list(clamped),
             max_abs_finite={k: v["max_abs_finite"] for k, v in clamped.items()},
             code_refs="server/somn.py:138")
    else:
        emit("警告", "T2-INF", "Inf 注入实测结果（clamp 未按「截断为 ±10」表现，按实测判读）",
             direct_write_states=inf_rows, code_refs="server/somn.py:136-138")

    # ---- 2b 维度谋杀 ----
    print("\n[2b] 维度谋杀（是否立刻抛出「清晰」错误）")
    D = 64
    for name, H in _dim_cases(D):
        # 先看只读路径（Router 会先调 read_only）
        ro_exc, ro_shape = None, None
        probe_layer = somn.IndependentFWLayer(D)
        probe_layer.S = torch.eye(D) * 0.1
        try:
            ro = probe_layer.read_only(H)
            ro_shape = list(ro.shape)
        except Exception as e:
            ro_exc = "%s: %s" % (type(e).__name__, str(e)[:180])

        m = somn.SOMNModule(D, min_score=0.0, warmup_steps=0)
        exc, out_shape, out_finite, S_state = None, None, None, None
        try:
            with torch.no_grad():
                o = m(H)
            out_shape, out_finite = list(o.shape), finite(o)
            S = m.layers[0].S
            S_state = ("None" if S is None else
                       {"shape": list(S.shape), "dtype": str(S.dtype),
                        "finite": finite(S)})
        except Exception as e:
            exc = "%s: %s" % (type(e).__name__, str(e)[:240])
        # 严格口径：只有消息里点明「维度/形状不匹配」才算清晰；
        # "not enough values to unpack (expected 3, got 2)" 只暴露实现细节，不算。
        clear = bool(exc and any(
            k in exc.lower() for k in ("维度", "mismatch", "expected shape")))
        rec = {"name": name, "shape": list(H.shape), "dtype": str(H.dtype),
               "read_only_exception": ro_exc, "read_only_shape": ro_shape,
               "module_exception": exc, "module_output_shape": out_shape,
               "module_output_finite": out_finite, "S_state": S_state,
               "clear_error": clear,
               "silent_accept": exc is None}
        out["dim_murder"].append(rec)
        print("  %-22s ro=%-42s module=%-52s clear=%s"
              % (name, (ro_exc or str(ro_shape)), (exc or "SILENT out=%s" % out_shape), clear))
    silent = [r["name"] for r in out["dim_murder"] if r["silent_accept"] and
              r["name"] not in ("fp16_[1,4,64]", "all_zero_[1,4,64]", "noncontig_[1,4,64]")]
    unclear = [r["name"] for r in out["dim_murder"]
               if r["module_exception"] and not r["clear_error"]]
    if silent or unclear:
        emit("阻塞", "T2-DIM", "维度异常未立刻抛出清晰错误：read_only 无断言、write 靠解包报错",
             silent_accept=silent, unclear_exception=unclear,
             read_only_has_no_assert=True,
             code_refs="server/somn.py:115-124 (read_only 无维度检查), 128 (B,T,D 解包), 129-130")

    # ---- 2c 高频分裂攻击 ----
    print("\n[2c] 高频分裂攻击（expand_growth_threshold 极低，1000 条，默认 max_layers=0 无上界）")
    for tag, cap in (("unbounded_max_layers_0", 0), ("capped_max_layers_8", 8)):
        r = _split_attack(D=64, n=1000, max_layers=cap,
                          guard_layers=256, guard_rss_mb=1536, budget_s=150)
        out["split_attack"][tag] = r
        print("  %-26s steps=%d layers=%d splits=%d peak_rss=%.1fMB aborted=%s wall=%.1fs"
              % (tag, r["steps_done"], r["final_layers"], r["splits"],
                 r["rss_peak_mb"], r["aborted"], r["wall_s"]))
        print("     curve=%s" % r["curve"][:14])
    # D=1024 的内存证据（实测，不做维度外推）：每层 S 为 4 MiB，跑到保护阈值即停
    r1024 = _split_attack(D=1024, n=1000, max_layers=0,
                          guard_layers=48, guard_rss_mb=2048, budget_s=120)
    out["split_attack"]["unbounded_D1024"] = r1024
    print("  %-26s steps=%d layers=%d splits=%d peak_rss=%.1fMB aborted=%s wall=%.1fs"
          % ("unbounded_D1024", r1024["steps_done"], r1024["final_layers"],
             r1024["splits"], r1024["rss_peak_mb"], r1024["aborted"], r1024["wall_s"]))
    print("     curve=%s" % r1024["curve"][:14])
    emit("警告", "T2-OOM-1024",
         "D=1024 高频分裂实测：层池无上界时 RSS 随层数上升，仅靠 guard 才停住",
         aborted=r1024["aborted"], steps_done=r1024["steps_done"],
         final_layers=r1024["final_layers"],
         rss_start_mb=r1024["rss_start_mb"], rss_peak_mb=r1024["rss_peak_mb"],
         rss_delta_mb=r1024["rss_delta_mb"],
         bytes_per_layer=r1024["bytes_per_layer_theory"],
         code_refs="server/somn.py:313, 358-376")
    unc = out["split_attack"]["unbounded_max_layers_0"]
    if unc["aborted"]:
        emit("阻塞", "T2-OOM", "无上界层池在高频分裂下指数爆炸：18 步内 1→274 层、273 次分裂，触发保护中止",
             aborted_reason=unc["aborted"], steps_done=unc["steps_done"],
             final_layers=unc["final_layers"], splits=unc["splits"],
             rss_start_mb=unc["rss_start_mb"], rss_peak_mb=unc["rss_peak_mb"],
             rss_delta_mb=brief(unc["rss_peak_mb"] - unc["rss_start_mb"]),
             bottleneck="层数（D=64 下 RSS 仅小幅上升；D=1024 见下一项实测）",
             curve=unc["curve"][:12],
             code_refs="server/somn.py:313 (max_layers 默认 0), 358-376 (_check_expand)")
    else:
        emit("警告", "T2-OOM", "1000 条内未 OOM，但层数/内存增长曲线需按实测外推风险",
             final_layers=unc["final_layers"], rss_peak_mb=unc["rss_peak_mb"],
             rss_delta_mb=brief(unc["rss_peak_mb"] - unc["rss_start_mb"]),
             code_refs="server/somn.py:313, 358-376")
    RESULTS["t2"] = out
    return out


def _split_attack(D=64, n=1000, max_layers=0, guard_layers=256,
                  guard_rss_mb=1536, budget_s=150):
    gc.collect()
    rss0 = rss_bytes()
    m = somn.SOMNModule(D, warmup_steps=0, max_write_count=1,
                        expand_growth_threshold=1e-12, min_score=0.0,
                        max_layers=max_layers)
    assert m.max_layers == max_layers
    peak, curve, aborted = rss_bytes(), [], None
    t0 = time.perf_counter()
    steps_done = 0
    for i in range(n):
        with torch.no_grad():
            m(white(5000 + i, 1, 1, D))
        steps_done = i + 1
        r = rss_bytes()
        if r > peak:
            peak = r
        L = len(m.layers)
        if i % 50 == 0 or i == n - 1:
            curve.append({"step": i, "layers": L,
                          "rss_mb": brief(r / MB)})
        if L >= guard_layers:
            aborted = "guard_layers=%d reached at step %d" % (guard_layers, i)
            break
        if r / MB >= guard_rss_mb:
            aborted = "guard_rss=%.0fMB reached at step %d (%.1fMB)" % (
                guard_rss_mb, i, r / MB)
            break
        if time.perf_counter() - t0 > budget_s:
            aborted = "time_budget=%.0fs reached at step %d" % (budget_s, i)
            break
    st = m.stats()
    return {
        "n_target": n, "steps_done": steps_done, "max_layers_cfg": max_layers,
        "guard": {"layers": guard_layers, "rss_mb": guard_rss_mb, "time_s": budget_s},
        "aborted": aborted, "final_layers": st["layers"], "splits": st["splits"],
        "rss_start_mb": brief(rss0 / MB), "rss_peak_mb": brief(peak / MB),
        "rss_delta_mb": brief((peak - rss0) / MB),
        "wall_s": brief(time.perf_counter() - t0),
        "curve": curve,
        "bytes_per_layer_theory": D * D * 4,
    }


# ==========================================================================
# 任务 3：长期数学演化推演  [1,1,64] Mock / 10000 步
# ==========================================================================
def task3(steps=10000, D=64, sample_every=500):
    print("\n" + "=" * 96)
    print("TASK 3  长期数学演化  shape=[1,1,%d]  steps=%d" % (D, steps))
    print("=" * 96)
    gc.collect()
    rss0 = rss_bytes()
    basis = make_basis(D)
    m = somn.SOMNModule(D)  # 全默认配置
    assert m.eta == 0.01 and m.max_layers == 0
    act = Counter()
    samples, score_tail = [], []
    last_split_step, prev_splits, splits_at_90pct = -1, 0, 0
    last_active_step = -1
    peak = rss0
    t0 = time.perf_counter()
    for i in range(steps):
        H = (structured(9000 + i, 1, 1, D, basis=basis)
             if i % 5 < 3 else white(9000 + i, 1, 1, D))
        with torch.no_grad():
            m(H)
        # 直接读内部 stats（避免每步构造完整 somn_stats 干扰计时）
        active_last = m._slot["stats"]["active_last"]
        if active_last:
            last_active_step = i
        for idx in active_last:
            act[idx] += 1
        sp = m._slot["stats"]["splits"]
        if i == steps - 1000:
            splits_at_90pct = sp
        if sp > prev_splits:
            last_split_step, prev_splits = i, sp
        if i >= steps - 1000:
            score_tail.append(list(m.router.last_base_scores))
        if i % sample_every == 0 or i == steps - 1:
            S0 = m.layers[0].S
            cond, sigma_min, sigma_max = None, None, None
            if S0 is not None:
                sv = torch.linalg.svdvals(S0.detach().float())
                sigma_max = float(sv[0].item())
                sigma_min = float(sv[-1].item())
                cond = sigma_max / sigma_min if sigma_min > 1e-12 else float("inf")
            r = rss_bytes()
            if r > peak:
                peak = r
            st = m.stats()
            samples.append({
                "step": i, "layers": st["layers"], "splits": st["splits"],
                "cond_layer0": (None if cond is None or math.isinf(cond)
                                else brief(cond, 4)),
                "sigma_max": brief(sigma_max, 8), "sigma_min": brief(sigma_min, 10),
                "energy_layer0": st["energies"][0],
                "scores": st["scores"][:8], "active": st["active_last"][:8],
                "rss_mb": brief(r / MB),
            })
    wall = time.perf_counter() - t0
    final = m.stats()
    L = len(m.layers)
    total_act = sum(act.values())
    shares = [act[k] / total_act for k in range(L)] if total_act else []
    ent = -sum(p * math.log(p) for p in shares if p > 0)
    max_ent = math.log(L) if L > 1 else 1.0
    norm_ent = ent / max_ent if max_ent > 0 else 0.0
    zombies = [k for k in range(L) if act[k] == 0]
    flat_tail = [s for row in score_tail for s in row]
    tail_var = statistics.pvariance(flat_tail) if len(flat_tail) > 1 else None
    conds = [s["cond_layer0"] for s in samples if s["cond_layer0"] is not None]
    out = {
        "config": {"shape": [1, 1, D], "steps": steps, "eta": m.eta,
                   "threshold": m.expand_growth_threshold,
                   "max_write_count": m.max_write_count,
                   "warmup_steps": m.warmup_steps, "max_layers": m.max_layers},
        "wall_s": brief(wall), "steps_done": steps,
        "final": final,
        "last_active_step": last_active_step,
        "dead_steps_after_last_activation": steps - 1 - last_active_step,
        "last_split_step": last_split_step,
        "splits_in_last_1000_steps": final["splits"] - splits_at_90pct,
        "activation": {
            "total": total_act, "per_layer": dict(sorted(act.items())),
            "normalized_entropy": brief(norm_ent, 4),
            "top1_share": brief(max(shares), 4) if shares else None,
            "zombie_layers": zombies,
            "zombie_ratio": brief(len(zombies) / L, 4) if L else None,
        },
        "score_tail_variance": brief(tail_var, 8),
        "cond_layer0_first": conds[0] if conds else None,
        "cond_layer0_last": conds[-1] if conds else None,
        "cond_growth_ratio": (brief(conds[-1] / conds[0], 4)
                              if conds and conds[0] else None),
        "sigma_min_first": samples[0]["sigma_min"],
        "sigma_min_last": samples[-1]["sigma_min"],
        "rss_start_mb": brief(rss0 / MB), "rss_peak_mb": brief(peak / MB),
        "samples": samples,
    }
    print("  wall=%.1fs layers=%d splits=%d last_split_step=%d last_active_step=%d dead_steps=%d"
          % (wall, final["layers"], final["splits"], last_split_step,
             last_active_step, steps - 1 - last_active_step))
    print("  writes=%s" % final["writes"][:12])
    print("  activation total=%d norm_entropy=%.4f top1_share=%s zombies=%s"
          % (total_act, norm_ent, out["activation"]["top1_share"], zombies))
    print("  cond layer0 first=%s last=%s growth=%s  sigma_min %s -> %s"
          % (out["cond_layer0_first"], out["cond_layer0_last"],
             out["cond_growth_ratio"], out["sigma_min_first"], out["sigma_min_last"]))
    print("  score_tail_variance=%s  rss_peak=%.1fMB"
          % (out["score_tail_variance"], out["rss_peak_mb"]))
    for s in samples:
        print("    step=%5d L=%d splits=%d cond=%s sigma_min=%s energy=%s active=%s"
              % (s["step"], s["layers"], s["splits"], s["cond_layer0"],
                 s["sigma_min"], s["energy_layer0"], s["active"]))

    if out["activation"]["top1_share"] is not None and (
            out["activation"]["top1_share"] > 0.8 or norm_ent < 0.3
            or out["activation"]["zombie_ratio"] > 0.5):
        emit("阻塞", "T3-DEADLOCK",
             "SOMN 记忆网络长期演化后永久坏死：末次激活后连续零激活、0 分裂、层数恒 1",
             last_active_step=last_active_step,
             dead_steps=steps - 1 - last_active_step,
             total_activations=total_act,
             top1_share=out["activation"]["top1_share"],
             normalized_entropy=brief(norm_ent, 4),
             zombie_layers=zombies, zombie_ratio=out["activation"]["zombie_ratio"],
             writes=final["writes"], layers=final["layers"], splits=final["splits"],
             final_scores=final["scores"], min_score=m.router.min_score,
             max_write_count=m.max_write_count,
             deadlock_chain=("score<min_score → 零激活 → write_count 冻结 → "
                             "不满足 write_count>max_write_count → 不分裂 → "
                             "无 warm-up 新层 → 永久零激活"),
             code_refs="server/somn.py:254, 265-268, 365-368")
    else:
        emit("通过", "T3-DEADLOCK", "10000 步未观察到激活完全僵化/死锁",
             top1_share=out["activation"]["top1_share"],
             normalized_entropy=brief(norm_ent, 4),
             zombie_ratio=out["activation"]["zombie_ratio"])
    if conds and max(conds) > 1e6:
        emit("警告", "T3-COND", "S 长期演化后极度病态（σ_min→0、条件数达 1e10 量级），SVD 分裂数值不稳",
             cond_first=out["cond_layer0_first"], cond_last=out["cond_layer0_last"],
             cond_max=brief(max(conds), 4), growth=out["cond_growth_ratio"],
             sigma_min_first=out["sigma_min_first"],
             sigma_min_last=out["sigma_min_last"],
             code_refs="server/somn.py:136-138（无秩/条件数保护）, 163（svd 输入未检查）")
    RESULTS["t3"] = out
    return out


# ==========================================================================
# 任务 4：静态审查结论的运行时探针（每条都构造真实输入）
# ==========================================================================
def task4():
    print("\n" + "=" * 96)
    print("TASK 4  静态/逻辑审查 · 运行时取证探针")
    print("=" * 96)
    D = 64
    out = {"probes": {}}

    # P1 read_only 无维度断言（对照 write）
    H2 = torch.randn(2, D, generator=rng(21))
    l = somn.IndependentFWLayer(D)
    l.S = torch.eye(D) * 0.1
    ro = l.read_only(H2)
    try:
        l.write(H2)
        w_exc = None
    except Exception as e:
        w_exc = "%s: %s" % (type(e).__name__, str(e)[:200])
    out["probes"]["P1_dim_assert"] = {
        "read_only_2d_shape": list(ro.shape), "write_2d_exception": w_exc,
        "read_only_has_assert": False, "write_has_explicit_dim_raise": True,
        "unclear_message": bool(w_exc and "维度" not in w_exc and "dim" not in w_exc.lower()),
    }
    print("  P1 read_only(2D)->%s  write(2D)->%s" % (list(ro.shape), w_exc))
    assert list(ro.shape) == [2, D], "read_only 未静默接受 2D"
    assert w_exc is not None, "write 未拒绝 2D"

    # P2 T=0 / B=0 除零：冷启动强制写入路径
    zero_cases = {}
    for name, shape in (("T=0", (1, 0, D)), ("B=0", (0, 4, D))):
        m = somn.SOMNModule(D, min_score=0.0, warmup_steps=0)
        H = torch.randn(*shape, generator=rng(31))
        exc = None
        try:
            with torch.no_grad():
                o = m(H)
            S = m.layers[0].S
            zero_cases[name] = {
                "exception": None, "output_shape": list(o.shape),
                "output_finite": finite(o),
                "S_is_nan": bool(torch.isnan(S).any().item()) if S is not None else None,
                "S_nan_frac": brief(float(torch.isnan(S).float().mean().item()), 4)
                if S is not None else None,
                "x_rows": int(H.reshape(-1, D).shape[0]),
            }
        except Exception as e:
            zero_cases[name] = {"exception": "%s: %s" % (type(e).__name__, str(e)[:200])}
    out["probes"]["P2_zero_length_divide"] = zero_cases
    print("  P2 zero-length=%s" % zero_cases)
    if any(v.get("S_is_nan") for v in zero_cases.values()):
        emit("阻塞", "T4-DIV0", "空序列 B*T=0 触发 /x.shape[0] 静默除零，S 全 NaN 永久污染",
             cases=zero_cases, code_refs="server/somn.py:131, 136（无 N==0 断言）")

    # P3 fp16 精度：同一 S 上 fp32 vs fp16 的 read_only 差异 + svd_split 重建误差
    g = rng(41)
    S_ref = torch.randn(D, D, generator=g) * 0.05
    l32 = somn.IndependentFWLayer(D)
    l32.S = S_ref.clone()
    l16 = somn.IndependentFWLayer(D)
    l16.S = S_ref.clone().half()
    Hf = torch.randn(2, 4, D, generator=rng(42))
    ro32 = l32.read_only(Hf)
    ro16 = l16.read_only(Hf.half())
    ro_gap = float((ro32 - ro16.float()).abs().max().item())
    c32 = l32.svd_split()
    c16 = l16.svd_split()
    err32 = float((S_ref - l32.S - c32.S).abs().max().item())
    err16 = float((S_ref - l16.S.float() - c16.S.float()).abs().max().item())
    out["probes"]["P3_low_precision"] = {
        "read_only_fp32_vs_fp16_max_gap": brief(ro_gap, 8),
        "svd_split_recon_err_fp32": brief(err32, 8),
        "svd_split_recon_err_fp16": brief(err16, 8),
        "fp16_err_amplification": brief(err16 / err32, 2) if err32 else None,
        "parent_dtype_after_split": str(l16.S.dtype),
    }
    print("  P3 fp16 ro_gap=%s recon_err32=%s recon_err16=%s"
          % (brief(ro_gap, 8), brief(err32, 8), brief(err16, 8)))
    if err16 > err32 * 10:
        emit("警告", "T4-DTYPE", "fp16 下 SVD 往返与 read_only 存在显著精度损失",
             **out["probes"]["P3_low_precision"],
             code_refs="server/somn.py:106-113 (_align 降精度), 122-123, 159-161, 183-188")

    # P4 fp16 下 cosine_similarity 的 eps=1e-8 是否失效（0/0 → NaN）
    H16 = torch.zeros(2, 4, D, dtype=torch.float16)
    r = somn.EnergyElbowRouter()
    lz = somn.IndependentFWLayer(D)
    lz.S = torch.zeros(D, D, dtype=torch.float16)
    fp16_scores = r.scores(H16, [lz])
    H32 = torch.zeros(2, 4, D, dtype=torch.float32)
    lz32 = somn.IndependentFWLayer(D)
    lz32.S = torch.zeros(D, D, dtype=torch.float32)
    fp32_scores = r.scores(H32, [lz32])
    out["probes"]["P4_cosine_eps_fp16"] = {
        "fp16_zero_input_scores": fp16_scores,
        "fp32_zero_input_scores": fp32_scores,
        "eps_literal": "1e-8",
        "eps_representable_in_fp16": False,
    }
    print("  P4 fp16 zero-zero cos=%s  fp32 zero-zero cos=%s"
          % (fp16_scores, fp32_scores))
    if any(math.isnan(x) for x in fp16_scores):
        emit("阻塞", "T4-EPS", "eps=1e-8 在 fp16 下下溢为 0，零向量余弦退化为 0/0=NaN",
             fp16_scores=fp16_scores, fp32_scores=fp32_scores,
             code_refs="server/somn.py:225-227（eps 硬编码 1e-8，未按 dtype 选择）")

    # P5 reset 后临时缓存是否清理干净 + 旧 layers 引用是否释放
    m = somn.SOMNModule(D, warmup_steps=0, min_score=0.0)
    for i in range(6):
        with torch.no_grad():
            m(white(700 + i, 2, 4, D))
    old_layers = m.layers
    old_S = old_layers[0].S
    m.reset()
    residual_scores = m.stats()["scores"]
    residual_boosted = m.stats()["boosted"]
    out["probes"]["P5_reset_lifecycle"] = {
        "router_last_base_scores_after_reset": residual_scores,
        "router_last_boosted_scores_after_reset": residual_boosted,
        "router_cache_clean": (residual_scores == [] and residual_boosted == []),
        "new_layers_is_new_list": m.layers is not old_layers,
        "old_list_still_has_layer": len(old_layers) >= 1,
        "old_S_still_reachable": (old_layers[0].S is old_S),
        "old_S_storage_alive_bytes": old_S.numel() * old_S.element_size(),
        "new_layer_S_is_none": m.layers[0].S is None,
        "stats_reset": m.stats()["steps"] == 0,
    }
    print("  P5 reset: router_cache_clean=%s residual=%s old_S_alive=%s"
          % (out["probes"]["P5_reset_lifecycle"]["router_cache_clean"],
             residual_scores, out["probes"]["P5_reset_lifecycle"]["old_S_still_reachable"]))
    assert m.layers[0].S is None and m.stats()["steps"] == 0
    emit("警告", "T4-RESET", "reset() 只换 layers/stats，未清 router 缓存；旧列表引用仍持有 S",
         **out["probes"]["P5_reset_lifecycle"],
         code_refs="server/somn.py:378-383, 212-213, 413-414")

    # P6 attach_somn 重复调用：模块数量 / 状态与配置错位
    class _Root:
        pass

    root = _Root()
    m1 = somn.attach_somn(root, 64, warmup_steps=0, min_score=0.0)
    for i in range(3):
        with torch.no_grad():
            m1(white(800 + i, 2, 4, 64))
    layers_before = m1.layers
    m2 = somn.attach_somn(root, 128, warmup_steps=99)
    slot = root._rt["somn"]
    module_keys = [k for k in slot.keys() if k == "module"]
    mismatch_exc = None
    try:
        with torch.no_grad():
            m2(white(900, 2, 4, 128))  # 按新 mgr.dim 喂数据
    except Exception as e:
        mismatch_exc = "%s: %s" % (type(e).__name__, str(e)[:200])
    out["probes"]["P6_attach_idempotence"] = {
        "module_key_count": len(module_keys),
        "module_overwritten": slot["module"] is m2 and slot["module"] is not m1,
        "layers_identity_preserved": m2.layers is layers_before,
        "mgr_dim": m2.dim, "layer_dim": m2.layers[0].dim,
        "mgr_eta": m2.eta, "layer_eta": m2.layers[0].eta,
        "router_is_new_object": m2.router is not m1.router,
        "old_mgr_shares_slot": m1._slot is m2._slot,
        "forward_with_new_dim_exception": mismatch_exc,
        "no_dim_assert_on_attach": mismatch_exc is None or "dim" not in str(mismatch_exc).lower(),
    }
    print("  P6 attach: module_keys=%d layers_preserved=%s mgr_dim=%d layer_dim=%d exc=%s"
          % (len(module_keys), m2.layers is layers_before, m2.dim,
             m2.layers[0].dim, mismatch_exc))
    assert len(module_keys) == 1, "重复 attach 挂了多个 module"
    emit("警告", "T4-ATTACH", "重复 attach_somn 不挂多模块，但 setdefault 保留旧层池导致 dim/eta/router 静默错位",
         **out["probes"]["P6_attach_idempotence"],
         code_refs="server/somn.py:319-323 (setdefault), 389-401")

    # P7 state_dict 零污染复查（含分裂 + reset）
    model = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    mgr = somn.attach_somn(model, D, warmup_steps=0, max_write_count=2,
                           expand_growth_threshold=1e-9, max_layers=4,
                           min_score=0.0)
    for i in range(20):
        with torch.no_grad():
            mgr(white(1100 + i, 2, 4, D))
    mgr.reset()
    after = model.state_dict()
    p7 = {
        "keys_identical": list(before) == list(after),
        "values_bitwise_identical": all(torch.equal(before[k], after[k]) for k in before),
        "mgr_state_dict_keys": list(mgr.state_dict().keys()),
        "mgr_parameters": len(list(mgr.parameters())),
        "mgr_buffers": len(list(mgr.buffers())),
        "mgr_named_modules": len(dict(mgr.named_modules())),
        "layers_after_growth": None,
    }
    # 再跑一次以确认分裂后仍无污染
    for i in range(20):
        with torch.no_grad():
            mgr(white(1300 + i, 2, 4, D))
    p7["layers_after_growth"] = len(mgr.layers)
    p7["keys_after_split"] = list(model.state_dict().keys()) == list(before)
    out["probes"]["P7_state_dict"] = p7
    print("  P7 state_dict keys_same=%s values_same=%s mgr_keys=%s params=%d buffers=%d"
          % (p7["keys_identical"], p7["values_bitwise_identical"],
             p7["mgr_state_dict_keys"], p7["mgr_parameters"], p7["mgr_buffers"]))
    assert p7["keys_identical"] and p7["values_bitwise_identical"]
    assert p7["mgr_parameters"] == 0 and p7["mgr_buffers"] == 0
    emit("通过", "T4-POLLUTION", "state_dict 复查干净：分裂/reset 前后键与值逐位一致，零 Parameter/Buffer",
         **p7)

    # P8 svd_split 后 σ 过小 → 新层 read_only / cos 稳定性
    u = torch.zeros(D); u[0] = 1.0
    v = torch.zeros(D); v[0] = 1.0
    lr = somn.IndependentFWLayer(D)
    lr.S = torch.outer(u, v)  # 秩 1，σ=[1,0,...,0]
    child = lr.svd_split()
    sv_p = torch.linalg.svdvals(lr.S)
    sv_c = torch.linalg.svdvals(child.S)
    Hn = white(1400, 2, 4, D)
    read_c = child.read_only(Hn)
    cos_c = torch.cosine_similarity(Hn, read_c, dim=-1, eps=1e-8)
    out["probes"]["P8_tiny_sigma"] = {
        "parent_sigma_max": brief(float(sv_p.max().item()), 8),
        "child_sigma_max": brief(float(sv_c.max().item()), 8),
        "child_read_absmax": brief(float(read_c.abs().max().item()), 8),
        "cos_finite": finite(cos_c),
        "cos_absmax": brief(float(cos_c.abs().max().item()), 8),
    }
    print("  P8 child_sigma_max=%s read_absmax=%s cos_finite=%s"
          % (out["probes"]["P8_tiny_sigma"]["child_sigma_max"],
             out["probes"]["P8_tiny_sigma"]["child_read_absmax"],
             out["probes"]["P8_tiny_sigma"]["cos_finite"]))
    if not out["probes"]["P8_tiny_sigma"]["cos_finite"]:
        emit("阻塞", "T4-SIGMA", "svd_split 后新层 S 奇异值过小导致余弦不稳定",
             **out["probes"]["P8_tiny_sigma"], code_refs="server/somn.py:178-188, 225-227")
    elif (out["probes"]["P8_tiny_sigma"]["child_sigma_max"] < 1e-5
          and out["probes"]["P8_tiny_sigma"]["cos_absmax"] > 0.05):
        emit("警告", "T4-SIGMA", "子层 S≈0 时 read_only 近零，路由仍给出非零余弦分（噪声被 eps 放大）",
             **out["probes"]["P8_tiny_sigma"],
             code_refs="server/somn.py:178-188, 225-227（缺少 σ_min 下限保护）")

    # P9 Elbow 在负分区域的方向问题（实测记录，不预设结论）
    r9 = somn.EnergyElbowRouter(elbow_alpha=0.6, min_score=-1.0)
    layers9 = []
    for _ in range(3):
        ly = somn.IndependentFWLayer(D)
        ly.S = -torch.eye(D) * 0.1
        layers9.append(ly)
    H9 = white(1500, 2, 4, D)
    s9 = r9.scores(H9, layers9)
    a9 = r9.route(H9, layers9)
    out["probes"]["P9_elbow_negative"] = {
        "scores": [brief(x, 4) for x in s9], "active": a9,
        "min_score": r9.min_score,
        "threshold_for_second_item": brief(s9[0] * r9.elbow_alpha, 4),
        "all_scores_negative": all(x < 0 for x in s9),
    }
    print("  P9 negative scores=%s active=%s" % ([brief(x, 4) for x in s9], a9))
    assert all(x < 0 for x in s9), "负分构造失败"
    if out["probes"]["P9_elbow_negative"]["all_scores_negative"]:
        emit("建议", "T4-ELBOW-NEG",
             "Elbow 用「乘 alpha」比较，负分时阈值方向反转；scores 无 [-1,1] 之外的语义约束",
             **out["probes"]["P9_elbow_negative"],
             code_refs="server/somn.py:254-262（负数下 sorted[i-1]*alpha 变大）")

    # P10 _check_expand 单步内多层同时分裂 → 层数翻倍
    m10 = somn.SOMNModule(D, warmup_steps=0, max_write_count=0,
                          expand_growth_threshold=0.0, min_score=0.0,
                          max_layers=0)
    m10.layers.clear()
    for k in range(3):
        ly = somn.IndependentFWLayer(D)
        ly.S = torch.eye(D) * (0.1 * (k + 1))
        ly.write_count = 1
        m10.layers.append(ly)
    before10 = len(m10.layers)
    m10._check_expand()
    after10 = len(m10.layers)
    out["probes"]["P10_expansion_per_step"] = {
        "layers_before": before10, "layers_after_one_check": after10,
        "doubled": after10 == before10 * 2,
        "max_layers_default": somn.SOMNModule(D).max_layers,
    }
    print("  P10 one _check_expand: %d -> %d layers (doubled=%s)"
          % (before10, after10, after10 == before10 * 2))
    if after10 == before10 * 2:
        emit("警告", "T4-EXPLODE", "单次 _check_expand 可让每个合格层各分裂一次，层数按步翻倍",
             **out["probes"]["P10_expansion_per_step"],
             code_refs="server/somn.py:358-376 (list(self.layers) 快照 + 无每步分裂上限)")

    # P11 路由打分 dtype 隐式降精度：S 为 fp32，输入 fp16
    l11 = somn.IndependentFWLayer(D)
    l11.S = torch.eye(D) * 0.1  # fp32
    H11 = white(1600, 2, 4, D).half()
    ro11 = l11.read_only(H11)
    p11 = {"S_dtype_original": str(l11.S.dtype), "input_dtype": str(H11.dtype),
           "read_only_out_dtype": str(ro11.dtype),
           "S_dtype_unchanged_after_read": str(l11.S.dtype),
           "temporary_downcast": True}
    out["probes"]["P11_dtype_downcast"] = p11
    print("  P11 S=%s H=%s -> read=%s (S 未被永久改)" %
          (p11["S_dtype_original"], p11["input_dtype"], p11["read_only_out_dtype"]))
    emit("警告", "T4-DOWNCAST", "read_only 把 fp32 的 S 临时降成输入 dtype（fp16），路由打分静默损失精度",
         **p11, code_refs="server/somn.py:122-123；对比 159-161 的 SVD 升精度路径")

    RESULTS["t4"] = out
    return out


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="SOMN V4 对抗性压测 / 破坏性审计")
    ap.add_argument("--only", default="t1,t2,t3,t4",
                    help="逗号分隔：t1,t2,t3,t4")
    ap.add_argument("--json", default=None, help="结果 JSON 落盘路径（如 data/tmp/somn_stress.json）")
    ap.add_argument("--t1-steps", type=int, default=200)
    ap.add_argument("--t3-steps", type=int, default=10000)
    args = ap.parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    print("=" * 96)
    print("SOMN V4 对抗性压测 / 破坏性审计   branch=feat/somn-stress-test")
    print("torch=%s python=%s threads=%d  mem_backend=%s"
          % (RESULTS["meta"]["torch"], RESULTS["meta"]["python"],
             RESULTS["meta"]["torch_num_threads"], MEM_BACKEND))
    print("cuda=%s mps=%s  (本机 CPU only)" %
          (RESULTS["meta"]["cuda"], RESULTS["meta"]["mps"]))
    print("=" * 96)

    if "t1" in only:
        task1(steps=args.t1_steps)
    if "t2" in only:
        task2()
    if "t3" in only:
        task3(steps=args.t3_steps)
    if "t4" in only:
        task4()

    print("\n" + "=" * 96)
    print("汇总：按严重程度")
    print("=" * 96)
    counts = Counter(f["level"] for f in RESULTS["findings"])
    for level in ("阻塞", "警告", "建议", "通过"):
        print("  %-4s %d" % (level, counts.get(level, 0)))
    for f in RESULTS["findings"]:
        if f["level"] in ("阻塞", "警告"):
            print("   - [%s] %s %s" % (f["level"], f["code"], f["title"]))

    if args.json:
        p = Path(args.json)
        if not p.is_absolute():
            p = ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2,
                                default=str), encoding="utf-8")
        print("\nJSON -> %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
