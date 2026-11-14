#!/usr/bin/env python3
# ===========================================================================
# GARGANTUA — manifold / Muon 落地实测脚本（2026-11-13）
# ---------------------------------------------------------------------------
# 报告 docs/analysis/manifold_muon_mhc_feasibility.md 阶段 0/1/2 的验证动作：
#   1) 微基准：NS5（现有 Muon） vs Stiefel/DGram Manifold Muon 的单步投影
#      wall-clock，实测倍率（报告 3.2.3 只给了解析量级估算，没跑 benchmark）；
#   2) 谱诊断开销：spectrum_stats / ModelDiagnostics.observe 的单步耗时；
#   3) 同种子短程训练对照：adamw vs muon（可选 manifold 档），记录 loss、
#      **裁剪前** grad norm、单步 wall-clock；
#   4) 输出 JSON 到 data/tmp/，供交付报告引用。
#
# 纯内存构造轻量模型（1+1 blocks / d_model 可配 / vocab_size 可配），
# **不调用 build_model / save_checkpoint，不写任何模型目录**，不污染用户权重。
#
# 用法：
#   /usr/local/bin/python3 service/calib/manifold_bench.py --steps 20
#   /usr/local/bin/python3 service/calib/manifold_bench.py --steps 30 \
#       --d-model 128 --block-size 16 --repeats 3 --with-manifold
# ===========================================================================
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys as _sys
import time
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
import torch.nn.functional as F

import common
import manifold_diag
import manifold_muon
import model as model_mod
import muon
import paths
import spec

SEED = 20261113


# ---------------------------------------------------------------------------
# 1) 投影微基准
# ---------------------------------------------------------------------------
def _median_time(fn, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def bench_projection(shapes, repeats: int) -> dict:
    out = {"shapes": {}, "note": "单矩阵单步投影 wall-clock；exact_msign=False（NS5）"}
    for shape in shapes:
        m, n = shape
        g = torch.Generator().manual_seed(SEED + m + n)
        w = torch.randn(m, n, generator=g)
        grad = torch.randn(m, n, generator=g)
        t_ns5 = _median_time(
            lambda: muon.newton_schulz5(grad, steps=spec.MUON_NS_STEPS),
            repeats)
        row = {"shape": [m, n], "ns5_sec": round(t_ns5, 6)}
        for kind, K in (("stiefel", 20), ("dgram", 20), ("stiefel", 1)):
            key = f"{kind}_K{K}_sec"
            try:
                t = _median_time(
                    lambda kk=kind, k=K: manifold_muon.manifold_muon_direction(
                        w, grad, kk, K=k, alpha=0.1,
                        ns_steps=spec.MUON_NS_STEPS, exact_msign=False),
                    repeats)
                row[key] = round(t, 6)
                row[f"{kind}_K{K}_ratio"] = round(t / max(t_ns5, 1e-9), 2)
            except Exception as e:
                row[key] = f"ERROR: {e}"
        out["shapes"][f"{m}x{n}"] = row
    return out


def bench_diagnostics(repeats: int) -> dict:
    w = torch.randn(256, 64, generator=torch.Generator().manual_seed(SEED))
    t = _median_time(lambda: manifold_diag.spectrum_stats(w), repeats)
    return {"spectrum_stats_256x64_sec": round(t, 6),
            "note": "Gram = 64x64，eigvalsh，无 SVD"}


# ---------------------------------------------------------------------------
# 2) 同种子短程训练对照
# ---------------------------------------------------------------------------
def _make_corpus(vocab: int, length: int, seed: int):
    rng = random.Random(seed)
    ids = [rng.randrange(vocab) for _ in range(length)]
    steps = [i // 4 for i in range(length)]     # 每 4 token 一个时间步
    return ids, steps


def _build_model(d_model: int, vocab: int, seed: int, manifold: dict | None):
    torch.manual_seed(seed)
    m = model_mod.Gargantua(n_enc_blocks=1, n_dec_blocks=1,
                            d_model=d_model, vocab_size=vocab)
    if manifold is not None:
        m.manifold_config = manifold
    m.train()
    return m


def run_training(kind: str, steps: int, block_size: int, d_model: int,
                 vocab: int, manifold: dict | None = None,
                 with_diag: bool = False) -> dict:
    m = _build_model(d_model, vocab, SEED, manifold)
    opts = muon.build_optimizers(m, kind, spec.ADAMW_LR, spec.MUON_LR,
                                 spec.MUON_MOMENTUM)
    ids, step_ids = _make_corpus(vocab, 256, SEED)
    rng = random.Random(SEED)
    diag = None
    if with_diag:
        diag = manifold_diag.ModelDiagnostics(
            m, opts=opts, every=max(1, steps // 4),
            log_path=os.path.join(paths.project_tmp_dir(),
                                  f"manifold_bench_diag_{kind}.jsonl"))
        diag.attach()
    losses, preclips, secs = [], [], []
    try:
        for step in range(1, steps + 1):
            t0 = time.perf_counter()
            i = rng.randint(0, len(ids) - block_size - 1)
            x_in, s_in, dec_in, s_dec, y = common.make_batch(
                ids, i, block_size, step_ids)
            for o in opts:
                o.zero_grad(set_to_none=True)
            logits = m.forward_train(x_in, s_in, dec_in, s_dec)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]),
                                   y.view(-1))
            loss.backward()
            pre = manifold_diag.grad_norm_before_clip(m.parameters())
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            for o in opts:
                o.step()
            m.reset_memory()
            losses.append(round(float(loss.item()), 6))
            preclips.append(round(float(pre), 6))
            secs.append(time.perf_counter() - t0)
            if diag is not None:
                diag.observe(step, loss=loss.item(), preclip_grad_norm=pre)
    finally:
        if diag is not None:
            diag.detach()
    return {
        "kind": kind,
        "steps": steps,
        "optimizer_types": [type(o).__name__ for o in opts],
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_delta": round(losses[-1] - losses[0], 6),
        "loss_series": losses,
        "preclip_grad_norm_first": preclips[0],
        "preclip_grad_norm_last": preclips[-1],
        "preclip_grad_norm_series": preclips,
        "mean_step_sec": round(statistics.mean(secs), 4),
        "total_sec": round(sum(secs), 3),
        "finite_all": all(v == v and abs(v) != float("inf") for v in losses),
        "manifold_enabled": bool(manifold and manifold.get("enabled")),
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="manifold / Muon 落地实测")
    ap.add_argument("--steps", type=int, default=20, help="每档训练步数")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--shapes", default="256x64,128x32",
                    help="投影微基准矩阵形状，逗号分隔")
    ap.add_argument("--with-manifold", action="store_true",
                    help="额外跑一档 K=2 的 Manifold Muon 训练（演示用）")
    ap.add_argument("--no-train", action="store_true", help="只跑微基准")
    ap.add_argument("--out", default="", help="JSON 输出路径（缺省写 data/tmp/）")
    args = ap.parse_args()

    shapes = []
    for part in args.shapes.split(","):
        a, b = part.strip().lower().split("x")
        shapes.append((int(a), int(b)))

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__,
        "seed": SEED,
        "block_size": args.block_size,
        "d_model": args.d_model,
        "vocab": args.vocab,
        "repeats": args.repeats,
        "projection": bench_projection(shapes, args.repeats),
        "diagnostics": bench_diagnostics(args.repeats),
        "training": {},
    }
    print("[bench] 投影微基准完成")

    if not args.no_train:
        report["training"]["adamw"] = run_training(
            "adamw", args.steps, args.block_size, args.d_model, args.vocab)
        print(f"[bench] adamw 完成  {report['training']['adamw']['mean_step_sec']}s/it")
        report["training"]["muon"] = run_training(
            "muon", args.steps, args.block_size, args.d_model, args.vocab,
            with_diag=True)
        print(f"[bench] muon 完成   {report['training']['muon']['mean_step_sec']}s/it")
        if args.with_manifold:
            cfg = spec.manifold_default_config()
            cfg.update({"enabled": True, "K": 2, "ns_steps": 3,
                        "max_matrices": 4})
            report["training"]["manifold_K2"] = run_training(
                "muon", args.steps, args.block_size, args.d_model, args.vocab,
                manifold=cfg)
            print("[bench] manifold_K2 完成")

    out = args.out or os.path.join(
        paths.project_tmp_dir(),
        "manifold_bench_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 68)
    print("投影微基准（相对现有 NS5 的倍率）")
    print("=" * 68)
    for name, row in report["projection"]["shapes"].items():
        print(f"  {name:10s} NS5={row.get('ns5_sec')}s  "
              f"Stiefel/K20={row.get('stiefel_K20_ratio')}x  "
              f"DGram/K20={row.get('dgram_K20_ratio')}x  "
              f"Stiefel/K1={row.get('stiefel_K1_ratio')}x")
    print(f"  诊断 spectrum_stats: "
          f"{report['diagnostics']['spectrum_stats_256x64_sec']}s/次")
    if report["training"]:
        print("\n同种子短程训练对照")
        print("=" * 68)
        for k, v in report["training"].items():
            print(f"  {k:14s} loss {v['loss_first']:.4f} -> {v['loss_last']:.4f} "
                  f"(Δ{v['loss_delta']:+.4f})  "
                  f"preclip {v['preclip_grad_norm_first']:.3f} -> "
                  f"{v['preclip_grad_norm_last']:.3f}  "
                  f"{v['mean_step_sec']:.3f}s/it  finite={v['finite_all']}")
    print(f"\n[bench] JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
