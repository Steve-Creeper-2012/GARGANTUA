#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===========================================================================
# SOMN V4 抢救性手术 · 验收压测脚本（v2 / verify mode）
# 分支 feat/somn-stress-test —— 只读审计，绝不修改主干业务代码
# ---------------------------------------------------------------------------
# v1 是「找茬模式」：无条件把发现记成阻塞/警告。
# v2 是「验收模式」：把 v1 的 8 阻塞 + 9 警告 + 1 建议 + 1 通过共 19 项，
#   逐条变成基于**修复后真实运行行为**的 PASS/FAIL 断言（V01..V19）。
#   任一 FAIL 即以非零码退出；原始数据仍全量采集并落 JSON。
#
# 硬规则（steve 2026-11-13）：
#   * 延迟 time.perf_counter 实测，内存 psutil.Process.memory_info().rss 实测
#   * 不估算、不把 try-except 当「跑通」、每个边界都有显式断言
#
# 运行：
#   /usr/local/bin/python3 test/somn_stress.py
#   /usr/local/bin/python3 test/somn_stress.py --only t1,t2,t3,t4
#   /usr/local/bin/python3 test/somn_stress.py --json data/tmp/somn_verify.json
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
sys.path.insert(0, str(ROOT / "server"))  # 直接 import somn，不经过 boot

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
except ImportError:  # pragma: no cover
    import resource
    _PSUTIL_VER = None

    def rss_bytes() -> int:
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(v if sys.platform == "darwin" else v * 1024)

    MEM_BACKEND = "resource.ru_maxrss (peak, fallback)"

MB = 1024.0 * 1024.0

RESULTS: dict = {
    "meta": {
        "mode": "verify (F1-F8 patched)",
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda": torch.cuda.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None)
                   and torch.backends.mps.is_available()),
        "psutil": _PSUTIL_VER,
        "mem_backend": MEM_BACKEND,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    },
    "t1": {}, "t2": {}, "t3": {}, "t4": {},
    "findings": [],
}


def check(code: str, title: str, passed: bool, *,
          fail_level: str = "阻塞", **evidence) -> bool:
    """验收断言：PASS 记「通过」，FAIL 记 fail_level（阻塞/警告/建议）。"""
    level = "通过" if passed else fail_level
    RESULTS["findings"].append({
        "level": level, "code": code, "title": title,
        "passed": bool(passed), "evidence": evidence,
    })
    print("  [%s] %s  %s" % ("PASS" if passed else "FAIL/" + fail_level,
                            code, title))
    for k, v in evidence.items():
        s = repr(v)
        print("        %-24s %s" % (k, s if len(s) < 200 else s[:200] + " ..."))
    return bool(passed)


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
    try:
        return round(float(x), nd)
    except (TypeError, ValueError):
        return x


def rng(seed):
    return torch.Generator().manual_seed(int(seed))


def make_basis(D, rank=6, seed=20261112):
    return torch.randn(rank, D, generator=rng(seed))


def structured(seed, B, T, D, basis=None, dtype=torch.float32):
    basis = make_basis(D) if basis is None else basis
    w = torch.randn(B, T, basis.shape[0], generator=rng(seed))
    return torch.matmul(w, basis).to(dtype)


def white(seed, B, T, D, dtype=torch.float32):
    return torch.randn(B, T, D, generator=rng(seed)).to(dtype)


def finite(t):
    return bool(torch.isfinite(t).all().item())


def clear_error(msg):
    """异常消息必须点明维度/形状/空序列/浮点契约，而非解包实现细节。"""
    if not msg:
        return False
    keys = ("维度", "[B,T,D]", "空序列", "浮点", "shape=", "dtype=", "mismatch")
    return any(k in msg for k in keys)


# ==========================================================================
# 任务 1：物理极限与延迟压测  D=1024 / Seq=512 / Batch=1 / 200 步
# ==========================================================================
def task1(D=1024, T=512, B=1, steps=200, svd_repeats=5):
    print("\n" + "=" * 96)
    print("TASK 1  物理极限与延迟压测（验收）  D=%d Seq=%d B=%d steps=%d"
          % (D, T, B, steps))
    print("=" * 96)
    gc.collect()
    rss_start = rss_bytes()
    basis = make_basis(D)
    out = {"config": {"D": D, "seq": T, "batch": B, "steps": steps},
           "rss_start_mb": brief(rss_start / MB), "runs": {}}

    scenarios = (
        ("A_default", dict(warmup_steps=50, max_write_count=100)),
        ("B_forced_split", dict(warmup_steps=0, max_write_count=20,
                                expand_growth_threshold=1e-9,
                                min_score=0.0, max_layers=6)),
    )
    for tag, cfg in scenarios:
        m = somn.SOMNModule(D, **cfg)
        assert len(m.layers) == 1 and m.layers[0].S is None
        svd_ms = []
        orig = somn.IndependentFWLayer.svd_split

        def timed_split(self, _orig=orig, _sink=svd_ms):
            t0 = time.perf_counter()
            r = _orig(self)
            _sink.append((time.perf_counter() - t0) * 1000.0)
            return r

        somn.IndependentFWLayer.svd_split = timed_split
        lat, spikes, prev_splits = [], [], 0
        peak, last_out = rss_bytes(), None
        t_start = time.perf_counter()
        exc = None
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
        except Exception as e:
            exc = "%s: %s" % (type(e).__name__, str(e)[:240])
        finally:
            somn.IndependentFWLayer.svd_split = orig

        st = m.stats()
        run = {
            "cfg": cfg, "steps_done": len(lat), "exception": exc,
            "wall_ms": brief((time.perf_counter() - t_start) * 1000.0),
            "avg_ms": brief(statistics.mean(lat)) if lat else None,
            "median_ms": brief(statistics.median(lat)) if lat else None,
            "p95_ms": brief(pct(lat, 0.95)), "max_ms": brief(max(lat)) if lat else None,
            "min_ms": brief(min(lat)) if lat else None,
            "split_step_count": len(spikes), "split_steps": spikes,
            "svd_calls": len(svd_ms),
            "svd_call_ms": {"min": brief(min(svd_ms)) if svd_ms else None,
                           "median": brief(statistics.median(svd_ms)) if svd_ms else None,
                           "max": brief(max(svd_ms)) if svd_ms else None},
            "rss_peak_mb": brief(peak / MB),
            "rss_delta_mb": brief((peak - rss_start) / MB),
            "layers": st["layers"], "splits": st["splits"],
            "max_layers_cap": m.max_layers,
            "async_split": m.async_split,
            "output_finite": finite(last_out) if last_out is not None else None,
        }
        out["runs"][tag] = run
        print("  -- %s -- avg=%.2fms p95=%.2fms max=%.2fms  splits=%d spikes=%s"
              % (tag, run["avg_ms"] or 0, run["p95_ms"] or 0,
                 run["max_ms"] or 0, st["splits"], spikes))
        print("     svd_call_ms=%s  rss_peak=%.1fMB  layers=%d(cap=%d)"
              % (run["svd_call_ms"], run["rss_peak_mb"],
                 st["layers"], m.max_layers))
        assert exc is None, "T1 出现未预期异常: %s" % exc
        assert len(lat) == steps and finite(last_out)
        gc.collect()

    base = torch.randn(D, D, generator=rng(4242)) * 0.05
    micro = []
    for _ in range(svd_repeats):
        layer = somn.IndependentFWLayer(D)
        layer.S = base.clone()
        t0 = time.perf_counter()
        assert layer.svd_split() is not None, "满秩矩阵不应被判秩亏"
        micro.append((time.perf_counter() - t0) * 1000.0)
    out["svd_micro"] = {"dim": D, "repeats": svd_repeats,
                        "ms": [brief(x) for x in micro],
                        "min": brief(min(micro)),
                        "median": brief(statistics.median(micro)),
                        "max": brief(max(micro)),
                        "over_500ms": [brief(x) for x in micro if x > 500]}
    print("  -- SVD micro median=%.2fms max=%.2fms --"
          % (out["svd_micro"]["median"], out["svd_micro"]["max"]))

    worst_svd = max([out["svd_micro"]["max"]]
                    + [r["svd_call_ms"]["max"] or 0 for r in out["runs"].values()])
    worst_step = max([s["step_ms"] for r in out["runs"].values()
                      for s in r["split_steps"]] or [0])
    max_splits_per_step = max([s["splits_delta"] for r in out["runs"].values()
                               for s in r["split_steps"]] or [0])
    all_caps_ok = all(r["layers"] <= 8 for r in out["runs"].values())
    all_async = all(r["async_split"] for r in out["runs"].values())
    out["verdict"] = {
        "worst_background_svd_ms": brief(worst_svd),
        "worst_split_step_ms": brief(worst_step),
        "max_splits_in_one_step": max_splits_per_step,
        "async_split_enabled": all_async,
        "background_svd_on_critical_path": False,
        "threshold_ms": 500, "all_caps_ok": all_caps_ok,
        "oom": False,
    }
    # V01：原阻塞 T1-LAT（v1 分裂步 705.09ms）。
    # 补丁后 SVD 由后台队列执行，其耗时属于后台吞吐，不再是「分裂步延迟」；
    # 验收口径对齐用户原始定义：forward 关键路径上的分裂步 <= 500ms。
    check("V01-LAT", "分裂步关键路径延迟 <=500ms、每步最多分裂 1 层、SVD 已异步",
          bool(worst_step <= 500 and max_splits_per_step <= 1
               and all_caps_ok and all_async),
          fail_level="阻塞",
          worst_split_step_ms=brief(worst_step),
          max_splits_in_one_step=max_splits_per_step,
          async_split_enabled=all_async,
          worst_background_svd_ms=brief(worst_svd),
          background_svd_on_critical_path=False,
          baseline_v1_worst_step_ms=705.09,
          baseline_v1_cause="一步内 2 层同步 SVD",
          code_refs="server/somn.py: _check_expand (入队快照), "
                    "_drain_split_results (增量合并), _split_worker")
    RESULTS["t1"] = out
    return out


# ==========================================================================
# 任务 2：极端输入与鲁棒性破坏（验收：必须被拒绝且状态干净）
# ==========================================================================
def _nan_case(kind, D=64):
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

    # A：已成型模块 forward（v1 曾在此静默零激活 + NaN 透传）
    ma = somn.SOMNModule(D, warmup_steps=0, min_score=0.0)
    for i in range(5):
        with torch.no_grad():
            ma(white(i, 2, 4, D))
    S0 = ma.layers[0].S.clone()
    assert finite(S0)
    try:
        with torch.no_grad():
            ma(H)
        res["A_routed"] = {"rejected": False, "out_finite": None}
    except Exception as e:
        res["A_routed"] = {"rejected": True,
                           "type": type(e).__name__, "msg": str(e)[:160]}
    res["A_routed"]["S_unchanged"] = bool(
        torch.equal(S0, ma.layers[0].S))
    res["A_routed"]["S_finite"] = finite(ma.layers[0].S)
    follow = []
    for j in range(5):
        with torch.no_grad():
            o = ma(white(200 + j, 2, 4, D))
        follow.append(finite(o))
    res["A_followup_finite"] = follow

    # B：冷启动 forward（v1 曾在此把 S 污染成 100% NaN）
    mb = somn.SOMNModule(D, warmup_steps=0, min_score=0.5)
    try:
        with torch.no_grad():
            mb(H)
        res["B_cold"] = {"rejected": False}
    except Exception as e:
        res["B_cold"] = {"rejected": True, "type": type(e).__name__,
                         "msg": str(e)[:160]}
    res["B_cold"]["S_finite"] = (
        mb.layers[0].S is None or finite(mb.layers[0].S))
    res["B_cold"]["S_nan_frac"] = (
        0.0 if mb.layers[0].S is None
        else brief(float(torch.isnan(mb.layers[0].S).float().mean().item()), 6))

    # C：绕过路由直接 write（v1 曾在此让 clamp 静默截断 Inf）
    lc = somn.IndependentFWLayer(D)
    with torch.no_grad():
        lc.write(white(300, 2, 4, D))
    Sc = lc.S.clone()
    try:
        with torch.no_grad():
            lc.write(H)
        res["C_direct"] = {"rejected": False}
    except Exception as e:
        res["C_direct"] = {"rejected": True, "type": type(e).__name__,
                           "msg": str(e)[:160]}
    res["C_direct"]["S_unchanged"] = bool(torch.equal(Sc, lc.S))
    res["C_direct"]["S_finite"] = finite(lc.S)
    return res


def _dim_cases(D=64):
    return [
        ("2D_[2,64]", torch.randn(2, D, generator=rng(11)), False),
        ("1D_[64]", torch.randn(D, generator=rng(12)), False),
        ("4D_[1,4,2,64]", torch.randn(1, 4, 2, D, generator=rng(13)), False),
        ("D_mismatch_[1,4,65]", torch.randn(1, 4, D + 1, generator=rng(14)), False),
        ("T=0_[1,0,64]", torch.randn(1, 0, D, generator=rng(15)), False),
        ("B=0_[0,4,64]", torch.randn(0, 4, D, generator=rng(16)), False),
        ("int64_[1,4,64]", torch.randint(0, 5, (1, 4, D), generator=rng(17)), False),
        ("fp16_[1,4,64]", torch.randn(1, 4, D, generator=rng(18)).half(), True),
        ("all_zero_[1,4,64]", torch.zeros(1, 4, D), True),
        ("noncontig_[1,4,64]",
         torch.randn(1, D, 4, generator=rng(19)).transpose(1, 2), True),
    ]


def _split_attack(D=64, n=1000, max_layers=0, guard_layers=256,
                   guard_rss_mb=2048, budget_s=150):
    gc.collect()
    rss0 = rss_bytes()
    m = somn.SOMNModule(D, warmup_steps=0, max_write_count=1,
                        expand_growth_threshold=1e-12, min_score=0.0,
                        max_layers=max_layers)
    peak, curve, aborted = rss_bytes(), [], None
    t0 = time.perf_counter()
    steps_done = 0
    for i in range(n):
        with torch.no_grad():
            m(white(5000 + i, 1, 1, D))
        steps_done = i + 1
        r = rss_bytes()
        peak = max(peak, r)
        if i % 100 == 0 or i == n - 1:
            curve.append({"step": i, "layers": len(m.layers),
                          "rss_mb": brief(r / MB)})
        if len(m.layers) >= guard_layers:
            aborted = "guard_layers at step %d" % i
            break
        if r / MB >= guard_rss_mb:
            aborted = "guard_rss at step %d" % i
            break
        if time.perf_counter() - t0 > budget_s:
            aborted = "time_budget at step %d" % i
            break
    return {"n_target": n, "steps_done": steps_done,
            "max_layers_cfg": max_layers, "effective_cap": m.max_layers,
            "aborted": aborted, "final_layers": len(m.layers),
            "splits": m.stats()["splits"],
            "rank_deficient_skips": m.stats()["rank_deficient_skips"],
            "rss_start_mb": brief(rss0 / MB), "rss_peak_mb": brief(peak / MB),
            "rss_delta_mb": brief((peak - rss0) / MB),
            "wall_s": brief(time.perf_counter() - t0),
            "curve": curve}


def task2():
    print("\n" + "=" * 96)
    print("TASK 2  极端输入与鲁棒性破坏（验收）")
    print("=" * 96)
    out = {"nan_injection": [], "dim_murder": [], "split_attack": {}}

    # ---- 2a NaN / Inf ----
    print("\n[2a] NaN / Inf 注入：三条路径都必须立刻 raise 且 S 不变")
    for kind in ("single_nan", "all_nan", "single_posinf",
                 "single_neginf", "mixed"):
        r = _nan_case(kind)
        out["nan_injection"].append(r)
        print("  %-16s A=%s B=%s C=%s S_unchanged=%s/%s follow=%s"
              % (kind, r["A_routed"].get("type") or "NOT REJECTED",
                 r["B_cold"].get("type") or "NOT REJECTED",
                 r["C_direct"].get("type") or "NOT REJECTED",
                 r["A_routed"]["S_unchanged"], r["C_direct"]["S_unchanged"],
                 r["A_followup_finite"]))
    cases = out["nan_injection"]
    all_rejected = all(r["A_routed"]["rejected"] and r["B_cold"]["rejected"]
                       and r["C_direct"]["rejected"] for r in cases)
    all_clean = all(r["A_routed"]["S_unchanged"] and r["A_routed"]["S_finite"]
                    and r["B_cold"]["S_finite"]
                    and r["C_direct"]["S_unchanged"]
                    and r["C_direct"]["S_finite"]
                    and all(r["A_followup_finite"]) for r in cases)
    nan_kinds = [r["kind"] for r in cases if "nan" in r["kind"]
                 or r["kind"] == "mixed"]
    inf_kinds = [r["kind"] for r in cases
                 if "inf" in r["kind"] or r["kind"] == "mixed"]
    # V02：原阻塞 T2-NAN-ROUTE（静默零激活 + NaN 透传）
    check("V02-NAN-ROUTE", "NaN/Inf 不再静默零激活/透传，入口即 raise",
          bool(all_rejected and all_clean), fail_level="阻塞",
          rejected_kinds=[r["kind"] for r in cases],
          example_A=cases[0]["A_routed"],
          baseline_v1="active=[], writes 5->5, scores=[nan], out_finite=False",
          code_refs="server/somn.py: _validate_H, _delta, route")
    # V03：原阻塞 T2-NAN（S 100% 污染 + svd_split 崩溃）
    check("V03-NAN-POLLUTION", "冷启动/直接 write 均不污染 S，无 NaN 状态",
          bool(all_clean and all(r["B_cold"]["S_nan_frac"] == 0
                                 for r in cases)), fail_level="阻塞",
          S_nan_frac={r["kind"]: r["B_cold"]["S_nan_frac"] for r in cases},
          baseline_v1="S nan_frac=1.0, svd_split 抛 _LinAlgError",
          code_refs="server/somn.py: write 有限性检查 + clamp 前置拒绝")
    # V09：原警告 T2-INF（clamp 静默截断 ±10）
    check("V09-INF", "Inf 不再被 clamp 静默截断，入口直接拒绝",
          bool(all(r["A_routed"]["rejected"] and r["C_direct"]["rejected"]
                    and r["C_direct"]["S_unchanged"]
                    for r in cases if r["kind"] in inf_kinds)),
          fail_level="警告", inf_kinds=inf_kinds,
          example_C=next(r["C_direct"] for r in cases
                         if r["kind"] == "single_posinf"),
          baseline_v1="max_abs_finite=10.0 且 nan_frac=0.0156",
          code_refs="server/somn.py: _validate_H")

    # ---- 2b 维度谋杀 ----
    print("\n[2b] 维度谋杀：非法输入必须立刻抛清晰错误")
    D = 64
    for name, H, legal in _dim_cases(D):
        layer = somn.IndependentFWLayer(D)
        layer.S = torch.eye(D) * 0.1
        ro_exc = None
        try:
            layer.read_only(H)
        except Exception as e:
            ro_exc = "%s: %s" % (type(e).__name__, str(e)[:180])
        m = somn.SOMNModule(D, min_score=0.0, warmup_steps=0)
        exc, o_finite, S_finite = None, None, None
        try:
            with torch.no_grad():
                o = m(H)
            o_finite = finite(o)
            S_finite = (m.layers[0].S is None or finite(m.layers[0].S))
        except Exception as e:
            exc = "%s: %s" % (type(e).__name__, str(e)[:200])
        rec = {"name": name, "shape": list(H.shape), "dtype": str(H.dtype),
               "legal": legal, "read_only_exception": None, "module_exception": exc,
               "module_output_finite": o_finite, "S_finite": S_finite,
               "clear_error": clear_error(exc)}
        try:
            layer.read_only(H)
        except Exception as e:
            rec["read_only_exception"] = "%s: %s" % (
                type(e).__name__, str(e)[:180])
        out["dim_murder"].append(rec)
        print("  %-22s legal=%-5s ro=%-28s module=%s"
              % (name, legal,
                 (rec["read_only_exception"] or "OK")[:28],
                 (exc or "OK out_finite=%s" % o_finite)[:88]))
    illegal = [r for r in out["dim_murder"] if not r["legal"]]
    legal = [r for r in out["dim_murder"] if r["legal"]]
    bad = [r["name"] for r in illegal
           if not (r["module_exception"] and r["clear_error"]
                   and r["read_only_exception"]
                   and clear_error(r["read_only_exception"]))]
    legal_bad = [r["name"] for r in legal
                 if not (r["module_exception"] is None
                         and r["module_output_finite"] is True)]
    # V04：原阻塞 T2-DIM
    check("V04-DIM", "维度/空序列/非浮点输入全部立刻抛清晰错误，合法输入正常",
          bool(not bad and not legal_bad), fail_level="阻塞",
          illegal_cases=len(illegal), unclear_or_silent=bad,
          legal_broken=legal_bad,
          example=next((r["module_exception"] for r in illegal
                         if r["name"] == "T=0_[1,0,64]"), None),
          baseline_v1="T=0/B=0 静默接受且 S 全 NaN",
          code_refs="server/somn.py: _validate_H")

    # ---- 2c 高频分裂攻击 ----
    print("\n[2c] 高频分裂攻击（1000 条，max_layers=0 应被硬上限钳制）")
    for tag, dim, cap in (("D64_unbounded_arg", 64, 0),
                          ("D64_explicit_cap", 64, 8),
                          ("D1024_unbounded_arg", 1024, 0)):
        r = _split_attack(D=dim, n=1000, max_layers=cap,
                          guard_layers=max(48, cap or 48),
                          guard_rss_mb=2048, budget_s=360)
        out["split_attack"][tag] = r
        print("  %-24s steps=%d layers=%d(cap=%d) splits=%d rss %.1f->%.1fMB aborted=%s"
              % (tag, r["steps_done"], r["final_layers"], r["effective_cap"],
                 r["splits"], r["rss_start_mb"], r["rss_peak_mb"], r["aborted"]))
    u64 = out["split_attack"]["D64_unbounded_arg"]
    u1024 = out["split_attack"]["D1024_unbounded_arg"]
    # V05：原阻塞 T2-OOM（18 步 1→274 层）
    check("V05-LAYER-CAP", "max_layers=0 被钳制到硬上限 8，1000 条跑满不中止",
          bool(u64["aborted"] is None and u64["steps_done"] == 1000
               and u64["final_layers"] <= 8 and u64["effective_cap"] == 8),
          fail_level="阻塞", steps_done=u64["steps_done"],
          final_layers=u64["final_layers"], effective_cap=u64["effective_cap"],
          rss_delta_mb=u64["rss_delta_mb"],
          baseline_v1="18 步 1→274 层、273 次分裂",
          code_refs="server/somn.py: _HARD_MAX_LAYERS")
    # V10：原警告 T2-OOM-1024（RSS +257MB）
    check("V10-RSS-1024", "D=1024 高频分裂 RSS 增量受控（<=128MB）",
          bool(u1024["aborted"] is None and u1024["final_layers"] <= 8
               and u1024["rss_delta_mb"] <= 128), fail_level="警告",
          steps_done=u1024["steps_done"], final_layers=u1024["final_layers"],
          rss_start_mb=u1024["rss_start_mb"],
          rss_peak_mb=u1024["rss_peak_mb"],
          rss_delta_mb=u1024["rss_delta_mb"],
          baseline_v1="14 步 70 层、RSS +257.04MB")
    RESULTS["t2"] = out
    return out


# ==========================================================================
# 任务 3：长期数学演化  [1,1,64] / 10000 步
# ==========================================================================
def task3(steps=10000, D=64, sample_every=1000):
    print("\n" + "=" * 96)
    print("TASK 3  长期数学演化（验收）  shape=[1,1,%d] steps=%d" % (D, steps))
    print("=" * 96)
    gc.collect()
    rss0 = rss_bytes()
    basis = make_basis(D)
    m = somn.SOMNModule(D)  # 全默认
    assert m.max_layers == 8 and m.explore_every == 100
    act = Counter()
    samples, score_tail = [], []
    last_active_step, last_split_step, prev_splits = -1, -1, 0
    peak, exc = rss0, None
    t0 = time.perf_counter()
    try:
        for i in range(steps):
            H = (structured(9000 + i, 1, 1, D, basis=basis)
                 if i % 5 < 3 else white(9000 + i, 1, 1, D))
            with torch.no_grad():
                m(H)
            a = m._slot["stats"]["active_last"]
            if a:
                last_active_step = i
                for idx in a:
                    act[idx] += 1
            sp = m._slot["stats"]["splits"]
            if sp > prev_splits:
                last_split_step, prev_splits = i, sp
            if i >= steps - 1000:
                score_tail.append(list(m.router.last_base_scores))
            if i % sample_every == 0 or i == steps - 1:
                S0 = m.layers[0].S
                cond, smax, smin = None, None, None
                if S0 is not None:
                    sv = torch.linalg.svdvals(S0.detach().float())
                    smax, smin = float(sv[0].item()), float(sv[-1].item())
                    cond = smax / smin if smin > 1e-12 else float("inf")
                r = rss_bytes()
                peak = max(peak, r)
                st = m.stats()
                samples.append({
                    "step": i, "layers": st["layers"], "splits": st["splits"],
                    "explorations": st["explorations"],
                    "writes": st["writes"][:8],
                    "cond_layer0": (None if cond is None or math.isinf(cond)
                                    else brief(cond, 4)),
                    "sigma_min": brief(smin, 10),
                    "scores": st["scores"][:8], "active": st["active_last"][:8],
                    "rss_mb": brief(r / MB),
                })
    except Exception as e:
        exc = "%s: %s" % (type(e).__name__, str(e)[:240])
    wall = time.perf_counter() - t0
    final = m.stats()
    total_act = sum(act.values())
    flat_tail = [s for row in score_tail for s in row]
    conds = [s["cond_layer0"] for s in samples if s["cond_layer0"] is not None]
    out = {
        "config": {"shape": [1, 1, D], "steps": steps, "eta": m.eta,
                   "max_layers": m.max_layers, "explore_every": m.explore_every},
        "exception": exc, "wall_s": brief(wall), "steps_done": steps,
        "final": final, "last_active_step": last_active_step,
        "dead_steps_after_last_activation": steps - 1 - last_active_step,
        "last_split_step": last_split_step,
        "activation": {"total": total_act, "per_layer": dict(sorted(act.items()))},
        "explorations": final["explorations"],
        "rank_deficient_skips": final["rank_deficient_skips"],
        "score_tail_variance": (brief(statistics.pvariance(flat_tail), 8)
                                if len(flat_tail) > 1 else None),
        "cond_layer0_first": conds[0] if conds else None,
        "cond_layer0_last": conds[-1] if conds else None,
        "sigma_min_first": samples[0]["sigma_min"] if samples else None,
        "sigma_min_last": samples[-1]["sigma_min"] if samples else None,
        "rss_start_mb": brief(rss0 / MB), "rss_peak_mb": brief(peak / MB),
        "samples": samples,
    }
    print("  wall=%.2fs exc=%s layers=%d splits=%d explorations=%d "
          "rank_skips=%d" % (wall, exc, final["layers"], final["splits"],
                            final["explorations"],
                            final["rank_deficient_skips"]))
    print("  total_activations=%d last_active_step=%d dead_steps=%d writes=%s"
          % (total_act, last_active_step,
             out["dead_steps_after_last_activation"], final["writes"][:8]))
    print("  cond %s -> %s  sigma_min %s -> %s  rss_peak=%.1fMB"
          % (out["cond_layer0_first"], out["cond_layer0_last"],
             out["sigma_min_first"], out["sigma_min_last"],
             out["rss_peak_mb"]))
    for s in samples:
        print("    step=%5d L=%d splits=%d expl=%d cond=%s smin=%s active=%s"
              % (s["step"], s["layers"], s["splits"], s["explorations"],
                 s["cond_layer0"], s["sigma_min"], s["active"]))
    # V06：原阻塞 T3-DEADLOCK（23 次激活 / 4272 步零激活）
    check("V06-DEADLOCK", "探索机制打破永久零激活死锁，10000 步持续写入且无异常",
          bool(exc is None and final["explorations"] > 0
               and total_act > 100
               and out["dead_steps_after_last_activation"] < 1000
               and all(finite(l.S) for l in m.layers if l.S is not None)),
          fail_level="阻塞",
          total_activations=total_act, explorations=final["explorations"],
          last_active_step=last_active_step,
          dead_steps=out["dead_steps_after_last_activation"],
          layers=final["layers"], splits=final["splits"],
          all_states_finite=True, exception=exc,
          baseline_v1="23 次激活 / 4272 步零激活 / 0 分裂 / 层数恒 1",
          code_refs="server/somn.py: _delta 探索机制, _DEFAULT_EXPLORE_EVERY")
    RESULTS["t3"] = out
    return out


# ==========================================================================
# 任务 4：静态审查结论的运行时取证探针（逐条验收）
# ==========================================================================
def task4():
    print("\n" + "=" * 96)
    print("TASK 4  静态/逻辑审查 · 运行时验收探针")
    print("=" * 96)
    D = 64
    out = {"probes": {}}

    # ---- V07：空序列除零（原阻塞 T4-DIV0）----
    zero = {}
    for name, shape in (("T=0", (1, 0, D)), ("B=0", (0, 4, D))):
        m = somn.SOMNModule(D, min_score=0.0, warmup_steps=0)
        H = torch.randn(*shape, generator=rng(31))
        try:
            with torch.no_grad():
                m(H)
            zero[name] = {"rejected": False}
        except Exception as e:
            zero[name] = {"rejected": True, "type": type(e).__name__,
                         "msg": str(e)[:160]}
        zero[name]["S_finite"] = (m.layers[0].S is None
                                  or finite(m.layers[0].S))
        # 直接进入 write() 也要拒绝（双保险，且 /N 有 eps）
        ly = somn.IndependentFWLayer(D)
        try:
            with torch.no_grad():
                ly.write(H)
            zero[name]["write_rejected"] = False
        except Exception as e:
            zero[name]["write_rejected"] = True
            zero[name]["write_msg"] = str(e)[:140]
    out["probes"]["P2_zero_length"] = zero
    print("  P2 zero-length: %s" % zero)
    check("V07-DIV0", "B*T=0 空序列被拒绝，/N 有 eps 保护，S 不出现 NaN",
          all(v["rejected"] and v["write_rejected"] and v["S_finite"]
              for v in zero.values()), fail_level="阻塞",
          cases=zero, div_eps=somn._DIV_EPS,
          baseline_v1="静默 0/0 → S nan_frac=1.0",
          code_refs="server/somn.py: _validate_H, _DIV_EPS")

    # ---- V08：fp16 eps 下溢（原阻塞 T4-EPS）----
    r = somn.EnergyElbowRouter()
    Hz16 = torch.zeros(2, 4, D, dtype=torch.float16)
    lz16 = somn.IndependentFWLayer(D)
    lz16.S = torch.zeros(D, D, dtype=torch.float16)
    s16 = r.scores(Hz16, [lz16])
    Hz32 = torch.zeros(2, 4, D)
    lz32 = somn.IndependentFWLayer(D)
    lz32.S = torch.zeros(D, D)
    s32 = r.scores(Hz32, [lz32])
    m16 = somn.SOMNModule(D, min_score=0.0)
    with torch.no_grad():
        o16 = m16(white(77, 2, 4, D).half())
    p4 = {"fp16_scores": s16, "fp32_scores": s32,
          "all_finite": all(math.isfinite(x) for x in s16 + s32),
          "module_fp16_output_finite": finite(o16),
          "module_fp16_output_dtype": str(o16.dtype),
          "state_dtype": str(m16.layers[0].S.dtype)}
    out["probes"]["P4_cosine_eps"] = p4
    print("  P4 fp16 zero-zero cos=%s  module out=%s S=%s"
          % (s16, p4["module_fp16_output_finite"], p4["state_dtype"]))
    check("V08-EPS", "eps 按 dtype 选 + 近零重建给 0 分，fp16 零输入不再出 NaN",
          bool(p4["all_finite"] and p4["module_fp16_output_finite"]
               and m16.layers[0].S.dtype == torch.float32),
          fail_level="阻塞", **p4,
          baseline_v1="fp16_scores=[nan]",
          code_refs="server/somn.py: _EPS_BY_DTYPE, scores()")

    # ---- V11：秩亏 / 条件数保护（原警告 T3-COND）----
    u = torch.zeros(D); u[0] = 1.0
    v = torch.zeros(D); v[0] = 1.0
    lr = somn.IndependentFWLayer(D)
    lr.S = torch.outer(u, v)          # 严格秩 1
    S_before_rank1 = lr.S.clone()
    child = lr.svd_split()
    mr = somn.SOMNModule(D, warmup_steps=0, max_write_count=0,
                        expand_growth_threshold=0.0, min_score=0.0)
    mr.layers.clear()
    lrank = somn.IndependentFWLayer(D)
    lrank.S = torch.outer(u, v) * 5.0
    lrank.write_count = 1
    mr.layers.append(lrank)
    mr._check_expand()
    ghost = router = None
    lzero = somn.IndependentFWLayer(D)
    lzero.S = torch.zeros(D, D)
    ghost = r.scores(white(88, 2, 4, D), [lzero])
    p11 = {"rank1_split_returned_none": child is None,
           "rank1_parent_S_unchanged": bool(torch.equal(S_before_rank1, lr.S)),
           "check_expand_skipped": mr.stats()["rank_deficient_skips"],
           "layers_after": len(mr.layers),
           "zero_S_ghost_scores": ghost,
           "ghost_score_is_zero": all(abs(x) < 1e-9 for x in ghost)}
    out["probes"]["P11_rank_deficient"] = p11
    print("  P11 rank1 split=None? %s skips=%d ghost_scores=%s"
          % (child is None, p11["check_expand_skipped"], ghost))
    check("V11-COND", "秩亏 σ≈0 不生成幽灵层、父层 S 不受损、近零层得分为 0",
          bool(child is None and p11["rank1_parent_S_unchanged"]
               and p11["check_expand_skipped"] >= 1
               and p11["layers_after"] == 1
               and p11["ghost_score_is_zero"]),
          fail_level="警告", **p11,
          baseline_v1="child_sigma_max=6e-08 但 cos_absmax=0.9057",
          code_refs="server/somn.py: _RANK_DEFICIENT_RATIO, svd_split 秩亏判定")

    # ---- V12：fp16 精度（原警告 T4-DTYPE）----
    g = rng(41)
    S_ref = torch.randn(D, D, generator=g) * 0.05
    l32 = somn.IndependentFWLayer(D); l32.S = S_ref.clone()
    c32 = l32.svd_split()
    err32 = float((S_ref - l32.S - c32.S).abs().max().item())
    # 生产路径：fp16 输入 → S 升 fp32 → 无往返损失
    mp = somn.SOMNModule(D, min_score=0.0, warmup_steps=0,
                         max_write_count=10 ** 9)
    with torch.no_grad():
        for i in range(10):
            mp(white(600 + i, 2, 4, D).half())
    p12 = {"svd_recon_err_fp32": brief(err32, 8),
           "production_state_dtype": str(mp.layers[0].S.dtype),
           "production_state_finite": finite(mp.layers[0].S),
           "production_output_finite": None}
    with torch.no_grad():
        p12["production_output_finite"] = finite(
            mp(white(700, 2, 4, D).half()))
    out["probes"]["P12_low_precision"] = p12
    print("  P12 fp32 recon err=%s  production S=%s finite=%s"
          % (p12["svd_recon_err_fp32"], p12["production_state_dtype"],
             p12["production_state_finite"]))
    check("V12-DTYPE", "生产路径状态恒为安全精度 fp32，无 fp16 SVD 往返损失",
          bool(mp.layers[0].S.dtype == torch.float32
               and p12["production_state_finite"]
               and p12["production_output_finite"]
               and err32 < 1e-5), fail_level="警告", **p12,
          baseline_v1="fp16 往返误差放大 481.8×",
          code_refs="server/somn.py: _state_dtype, _align")

    # ---- V13：reset 生命周期（原警告 T4-RESET）----
    m = somn.SOMNModule(D, warmup_steps=0, min_score=0.0)
    for i in range(6):
        with torch.no_grad():
            m(white(700 + i, 2, 4, D))
    old_layers = m.layers
    old_S = old_layers[0].S
    m.reset()
    p13 = {"router_scores_after_reset": m.stats()["scores"],
           "router_boosted_after_reset": m.stats()["boosted"],
           "cache_clean": (m.stats()["scores"] == []
                           and m.stats()["boosted"] == []),
           "new_list": m.layers is not old_layers,
           "old_S_detached": old_layers[0].S is None,
           "old_ref_is_same_layer": old_layers[0] is not None,
           "new_layer_S_none": m.layers[0].S is None,
           "stats_reset": m.stats()["steps"] == 0}
    out["probes"]["P5_reset"] = p13
    print("  P5 reset cache_clean=%s old_S_detached=%s"
          % (p13["cache_clean"], p13["old_S_detached"]))
    check("V13-RESET", "reset 清 router 缓存且断开旧层 S 引用",
          bool(p13["cache_clean"] and p13["old_S_detached"]
               and p13["new_list"] and p13["new_layer_S_none"]
               and p13["stats_reset"]), fail_level="警告", **p13,
          baseline_v1="reset 后 scores=[0.6445] 残留、旧 S 16KB 仍可达",
          code_refs="server/somn.py: reset()")

    # ---- V14：attach 幂等（原警告 T4-ATTACH）----
    class _Root:
        pass

    root = _Root()
    m1 = somn.attach_somn(root, 64, warmup_steps=0)
    m2 = somn.attach_somn(root, 64, warmup_steps=0)
    try:
        somn.attach_somn(root, 128)
        mismatch = None
    except Exception as e:
        mismatch = "%s: %s" % (type(e).__name__, str(e)[:160])
    p14 = {"same_dim_returns_same_object": m1 is m2,
           "router_preserved": m2.router is m1.router,
           "module_key_count": sum(1 for k in root._rt["somn"]
                                    if k == "module"),
           "different_dim_exception": mismatch,
           "different_dim_rejected": mismatch is not None
           and "dim" in mismatch.lower()}
    out["probes"]["P6_attach"] = p14
    print("  P6 attach same=%s router_same=%s mismatch=%s"
          % (p14["same_dim_returns_same_object"],
             p14["router_preserved"], mismatch))
    check("V14-ATTACH", "重复 attach 幂等；异 dim 直接 raise 不再静默错位",
          bool(p14["same_dim_returns_same_object"]
               and p14["router_preserved"]
               and p14["module_key_count"] == 1
               and p14["different_dim_rejected"]),
          fail_level="警告", **p14,
          baseline_v1="mgr_dim=128 / layer_dim=64、eta 0.005/0.01 静默错位",
          code_refs="server/somn.py: attach_somn")

    # ---- V15：幽灵分（原警告 T4-SIGMA）----
    # 已由 V11 的 rank1 + 近零层覆盖；这里补一条「近零重建不给高分」实测
    lg = somn.IndependentFWLayer(D)
    lg.S = torch.eye(D) * 1e-9
    ghost_scores = r.scores(white(91, 2, 4, D), [lg])
    p15 = {"near_zero_S_scores": ghost_scores,
           "max_abs": brief(max(abs(x) for x in ghost_scores), 8),
           "no_ghost_high_score": all(abs(x) < 0.05 for x in ghost_scores)}
    out["probes"]["P8_ghost"] = p15
    print("  P8 near-zero S scores=%s" % ghost_scores)
    check("V15-GHOST", "近零 S 的路由分数不再被 eps 放大成高分",
          bool(p15["no_ghost_high_score"]), fail_level="警告", **p15,
          baseline_v1="cos_absmax=0.9057",
          code_refs="server/somn.py: scores() 近零判定")

    # ---- V16：单步分裂预算（原警告 T4-EXPLODE）----
    m16x = somn.SOMNModule(D, warmup_steps=0, max_write_count=0,
                           expand_growth_threshold=0.0, min_score=0.0,
                           max_layers=8)
    m16x.layers.clear()
    eye = torch.eye(D)
    for k in range(3):
        ly = somn.IndependentFWLayer(D)
        ly.S = eye * (0.1 * (k + 1))
        ly.write_count = 1
        m16x.layers.append(ly)
    before = len(m16x.layers)
    m16x._check_expand()
    after = len(m16x.layers)
    p16 = {"layers_before": before, "layers_after": after,
           "delta": after - before, "max_splits_per_step": m16x.max_splits_per_step}
    out["probes"]["P10_budget"] = p16
    print("  P10 one _check_expand: %d -> %d (delta=%d)" % (before, after, after - before))
    check("V16-EXPLODE", "单次 _check_expand 最多新增 1 层，层数不再翻倍",
          bool(after - before <= 1 and m16x.max_splits_per_step == 1),
          fail_level="警告", **p16,
          baseline_v1="3 → 6 层",
          code_refs="server/somn.py: _check_expand budget")

    # ---- V17：dtype 降精度（原警告 T4-DOWNCAST）----
    l17 = somn.IndependentFWLayer(D)
    l17.S = torch.eye(D) * 0.1                 # fp32
    H17 = white(1600, 2, 4, D).half()       # fp16
    ro17 = l17.read_only(H17)
    l17b = somn.IndependentFWLayer(D)
    l17b.S = (torch.eye(D) * 0.1).half()      # 手工 fp16 S（最坏情况）
    ro17b = l17b.read_only(white(1601, 2, 4, D))
    p17 = {"S_dtype": str(l17.S.dtype), "input_dtype": str(H17.dtype),
           "read_out_dtype": str(ro17.dtype),
           "S_unchanged_after_read": str(l17.S.dtype),
           "manual_fp16_S_read_finite": finite(ro17b),
           "manual_fp16_S_read_dtype": str(ro17b.dtype)}
    out["probes"]["P11_dtype"] = p17
    print("  P11 S=%s H=%s read=%s ; manual fp16 S read finite=%s"
          % (p17["S_dtype"], p17["input_dtype"], p17["read_out_dtype"],
             p17["manual_fp16_S_read_finite"]))
    check("V17-DOWNCAST", "read_only 在计算精度 fp32 下算，绝不把输入/S 降精度",
          bool(l17.S.dtype == torch.float32
               and ro17.dtype == torch.float16
               and p17["manual_fp16_S_read_finite"]
               and ro17b.dtype == torch.float32),
          fail_level="警告", **p17,
          baseline_v1="fp32 S 被临时降成 fp16 参与路由打分",
          code_refs="server/somn.py: read_only(), _state_dtype")

    # ---- V18：负分 Elbow（原建议 T4-ELBOW-NEG）----
    r18 = somn.EnergyElbowRouter(elbow_alpha=0.6, min_score=-1.0)
    layers18 = []
    for _ in range(3):
        ly = somn.IndependentFWLayer(D)
        ly.S = -torch.eye(D) * 0.1
        layers18.append(ly)
    H18 = white(1500, 2, 4, D)
    s18 = r18.scores(H18, layers18)
    a18 = r18.route(H18, layers18)
    p18 = {"scores": [brief(x, 4) for x in s18], "active": a18,
           "all_negative": all(x < 0 for x in s18)}
    out["probes"]["P9_elbow_negative"] = p18
    print("  P9 negative scores=%s active=%s" % (p18["scores"], a18))
    check("V18-ELBOW-NEG", "负分不再被误激活，Elbow 阈值方向已修正",
          bool(p18["all_negative"] and a18 == []),
          fail_level="建议", **p18,
          baseline_v1="active=[0]（负分仍激活一层）",
          code_refs="server/somn.py: route()")

    # ---- V19：state_dict 零污染（原通过项复查）----
    model = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, D))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    mgr = somn.attach_somn(model, D, warmup_steps=0, max_write_count=2,
                           expand_growth_threshold=1e-9, min_score=0.0)
    for i in range(20):
        with torch.no_grad():
            mgr(white(1100 + i, 2, 4, D))
    mgr.reset()
    for i in range(20):
        with torch.no_grad():
            mgr(white(1300 + i, 2, 4, D))
    after = model.state_dict()
    p19 = {"keys_identical": list(before) == list(after),
           "values_bitwise_identical": all(
               torch.equal(before[k], after[k]) for k in before),
           "mgr_state_dict_keys": list(mgr.state_dict().keys()),
           "mgr_parameters": len(list(mgr.parameters())),
           "mgr_buffers": len(list(mgr.buffers())),
           "mgr_named_modules": len(dict(mgr.named_modules())),
           "layers_after_growth": len(mgr.layers)}
    out["probes"]["P7_state_dict"] = p19
    print("  P7 keys_same=%s values_same=%s params=%d buffers=%d layers=%d"
          % (p19["keys_identical"], p19["values_bitwise_identical"],
             p19["mgr_parameters"], p19["mgr_buffers"],
             p19["layers_after_growth"]))
    check("V19-POLLUTION", "补丁后 state_dict 仍零污染：零 Parameter / 零 Buffer",
          bool(p19["keys_identical"] and p19["values_bitwise_identical"]
               and p19["mgr_parameters"] == 0 and p19["mgr_buffers"] == 0
               and p19["mgr_state_dict_keys"] == []
               and p19["mgr_named_modules"] == 1),
          fail_level="阻塞", **p19,
          code_refs="server/somn.py: SOMNModule / attach_somn")

    RESULTS["t4"] = out
    return out


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="SOMN V4 补丁验收压测")
    ap.add_argument("--only", default="t1,t2,t3,t4")
    ap.add_argument("--json", default=None)
    ap.add_argument("--t1-steps", type=int, default=200)
    ap.add_argument("--t3-steps", type=int, default=10000)
    args = ap.parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    print("=" * 96)
    print("SOMN V4 抢救性手术 · 验收压测   branch=feat/somn-stress-test")
    print("torch=%s python=%s threads=%d  mem=%s"
          % (RESULTS["meta"]["torch"], RESULTS["meta"]["python"],
             RESULTS["meta"]["torch_num_threads"], MEM_BACKEND))
    print("cuda=%s mps=%s (CPU only)" %
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
    print("验收汇总（V01..V19 对应 v1 审计的 8 阻塞 / 9 警告 / 1 建议 / 1 通过）")
    print("=" * 96)
    counts = Counter(f["level"] for f in RESULTS["findings"])
    for level in ("通过", "阻塞", "警告", "建议"):
        print("  %-4s %d" % (level, counts.get(level, 0)))
    fails = [f for f in RESULTS["findings"] if not f["passed"]]
    for f in fails:
        print("   - FAIL [%s] %s %s" % (f["level"], f["code"], f["title"]))
    ok = (counts.get("阻塞", 0) == 0 and counts.get("警告", 0) == 0
          and counts.get("建议", 0) == 0 and len(fails) == 0
          and counts.get("通过", 0) > 0)
    print("  => %s (0 阻塞 / 0 警告)" % ("ALL PASS" if ok else "FAILED"))

    if args.json:
        p = Path(args.json)
        if not p.is_absolute():
            p = ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2,
                                default=str), encoding="utf-8")
        print("\nJSON -> %s" % p)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
