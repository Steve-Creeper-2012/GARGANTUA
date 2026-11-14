# ===========================================================================
# GARGANTUA v1 — service/calib/momw_calibrate.py：MoMW 阈值离线标定
# ---------------------------------------------------------------------------
# 目的：在**真实权重**上跑一批覆盖 general/error/code/math 的种子输入，
# 把 Router 的余弦打分分布打成直方图，为「双峰谷底」定 threshold / B0 /
# expand_delta 提供实测依据。不写任何默认值进 spec —— 标定只产出建议。
#
# 为什么用对照池：生产模型通常 initial_layers=1（只有一个 general 锚点），
# 四个类别的分数会挤在同一个锚点上，看不出分离。标定时临时扩成 4 层
# （每类一个锚点），于是每个样本得到 4 个分数：
#   · 同类分数 = 信号分布（该类别文本 vs 该类别锚点）
#   · 异类分数 = 背景分布（该类别文本 vs 其它三类锚点）
# 两者的谷底就是「勉强算同类」的准入边界。这是双峰的来源，不是硬凑的。
#
# H_1 的取法：严格复现 MoMW 的插入点 —— token embedding → 第一个 Block
# → + fast weight 读出。不跑完整 encode（那是最后一层输出，不是 H_1）。
# fast weight 每次传新的空 state，保证样本之间不泄漏。
#
# 运行：
#   /usr/local/bin/python3 service/calib/momw_calibrate.py --per-class 12
# 输出：全体直方图（S=同类 / .=异类）、分类统计、分位数、谷底、建议阈值。
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

import argparse
import glob
import json
import math
import os
import time

import torch

import spec
from common import build_model, model_vocab_dir
from tokens import enctok
from tokens.vocab import Vocab

# 种子输入：纯 ASCII 为主，避免中文/符号被词表 fallback 静默丢掉。
# 每类 12 条，风格刻意贴近 spec.MOMW_ANCHOR_TEXTS 的语义。
SEEDS = {
    "general": [
        "hello world", "the weather is nice today", "this is a short note",
        "please review the document", "we should meet tomorrow morning",
        "the system is running normally", "a cat sat on the mat",
        "thank you for your reply", "this file contains plain text",
        "the quick brown fox jumps", "let us begin the session",
        "everything looks fine so far",
    ],
    "error": [
        "Traceback (most recent call last)", "SyntaxError invalid syntax",
        "failed to parse the input", "runtime error null pointer",
        "connection refused by server", "ValueError list index out of range",
        "assertion failed at line 42", "cannot open file permission denied",
        "timeout while waiting for response", "undefined variable name",
        "TypeError unsupported operand", "the operation was aborted",
    ],
    "code": [
        "def main(): return None", "for i in range(10): print(i)",
        "import torch.nn as nn", "class Foo: pass",
        "if x > 0: return x", "x = lambda a: a + 1",
        "SELECT * FROM users", "const result = await fetch(url)",
        "while True: break", "try: pass except Exception: pass",
        "@decorator def f(): pass", "git commit -m fix",
    ],
    "math": [
        "1 + 2 = 3", "0.5 * 4 = 2", "the sum is 10",
        "x^2 + y^2 = z^2", "3.14159 is pi", "12 / 4 = 3",
        "sum from 1 to n", "probability 0.95", "matrix 2 by 3",
        "sqrt(16) equals 4", "100 minus 37 is 63",
        "average of 2 and 8 is 5",
    ],
}


def _pick_model_dir(root: str) -> str:
    dirs = [d for d in sorted(glob.glob(os.path.join(root, "models", "*")))
            if os.path.isdir(d)
            and os.path.isfile(os.path.join(d, "config.json"))]
    if not dirs:
        raise SystemExit("models/ 下没有可标定的模型")
    return dirs[0]


def _encode(text: str, vocab) -> list:
    ids, _skipped = enctok.encode_text(text, vocab)
    return [int(i) for i in ids]


def _h1(m, ids: list):
    """严格复现 MoMW 插入点的 H_1（第一个 Block 输出 + fast weight 读出）。"""
    x_in = torch.tensor([ids], dtype=torch.long)
    type_ids = torch.full_like(x_in, spec.TYPE_USER)
    emb = m.embed_text(x_in, type_ids)
    sids = torch.arange(len(ids), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        x, blocks = m.enc_blocks[0](
            emb, (None, None, None), (None, None, None),
            x_step_ids=sids, blocks=[emb])
        # 每个样本用全新的空 state，杜绝跨样本状态泄漏
        fw_out = m.fast_weight(x, state={})
        return x + fw_out


def _quantile(values: list, p: float):
    if not values:
        return float("nan")
    return float(torch.tensor(values, dtype=torch.float32)
                 .quantile(p).item())


def _histogram(pairs, bins: int = 20, width: int = 44) -> str:
    """pairs: [(score, is_same_class)]。S=同类信号，.=异类背景。"""
    vals = [v for v, _ in pairs]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        hi = lo + 1e-6
    sig = [0] * bins
    bg = [0] * bins
    for v, same in pairs:
        i = min(bins - 1, max(0, int((v - lo) / (hi - lo) * bins)))
        (sig if same else bg)[i] += 1
    peak = max(max(sig), max(bg)) or 1
    lines = ["  区间                 分布 (S=同类 / .=异类)             S    ."]
    for i in range(bins):
        a = lo + (hi - lo) * i / bins
        b = lo + (hi - lo) * (i + 1) / bins
        # 同一行同时显示信号条与背景条（各自独立归一到峰值）
        s_width = round(sig[i] / peak * width)
        b_width = round(bg[i] / peak * width)
        cells = ("S" * s_width + "." * b_width).ljust(width)
        lines.append(f"  {a:+.3f}..{b:+.3f} |{cells}| {sig[i]:3d} {bg[i]:3d}")
    return "\n".join(lines)


def _valley(pairs, bins: int = 40):
    """双峰谷底：扫描候选阈值，最小化「异类误纳 + 同类漏召」。

    比找直方图局部极小稳健：局部极小对分桶宽度敏感，而这条直接对应
    「准入边界」的代价函数。同时返回直方图上的密度局部极小作参考。
    """
    vals = [v for v, _ in pairs]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return float("nan"), {}
    best_t, best_err = lo, len(pairs) + 1
    best = {}
    for i in range(1, 200):
        t = lo + (hi - lo) * i / 200
        fp = sum(1 for v, s in pairs if (not s) and v >= t)
        fn = sum(1 for v, s in pairs if s and v < t)
        if fp + fn < best_err:
            best_err = fp + fn
            best_t = t
            best = {"errors": fp + fn, "false_accept": fp,
                    "miss": fn, "rate": (fp + fn) / len(pairs)}
    return best_t, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--bins", type=int, default=20)
    args = ap.parse_args()

    root = str(boot.project_root())
    model_dir = args.model_dir or _pick_model_dir(root)
    print(f"[calib] 模型目录: {model_dir}")
    t0 = time.time()
    m = build_model(model_dir)
    if getattr(m, "momw", None) is None:
        raise SystemExit("该模型未启用 MoMW，无法标定（建模时勾选 MoMW）")
    mo = m.momw
    print(f"[calib] d_model={m.d_model}  enc={len(m.enc_blocks)} "
          f"dec={len(m.dec_blocks)}  params={sum(p.numel() for p in m.parameters())/1e6:.1f}M"
          f"  加载 {time.time()-t0:.1f}s")

    vocab = Vocab.load(os.path.join(model_vocab_dir(model_dir), "vocab.vfc"))
    print(f"[calib] 词表 {vocab.name} / {len(vocab)} 条")

    # 标定对照池：4 类各一层（生产池只有 1 层，看不到类间分离）
    types = list(spec.MOMW_ANCHOR_GROUPS)
    while len(mo.pool.kda_layers) < len(types):
        mo.pool._append_layer(len(mo.pool.kda_layers) % len(types))
    print(f"[calib] 对照池 {len(mo.pool.kda_layers)} 层: "
          + ", ".join(l.type_name for l in mo.pool.kda_layers))

    # 先把所有 H_1 算好缓存（短序列，几 MB），两遍复用，避免重复跑模型。
    samples = []        # (类别, H_1)
    skipped = 0
    for cls in types:
        for text in SEEDS.get(cls, [])[:args.per_class]:
            ids = _encode(text, vocab)
            if not ids:
                skipped += 1
                continue
            samples.append((cls, _h1(m, ids)))
    if samples:
        mo._ensure_anchors(samples[0][1])

    # ---- Pass A：冻结状态，只读打分 ----
    # 关键：这一遍绝不写入。若把 forward 混进同一循环，γ 会随样本递增，
    # 后面的样本打分从锚点主导漂移到 E 主导，分布就不可比了。
    pairs = []          # (score, is_same_class)
    per_class = {t: [] for t in types}
    with torch.no_grad():
        for cls, H in samples:
            raw, _scores = mo.router.score_all(
                H, mo.pool.kda_layers, mo._anchors, mo.cfg)
            for i, v in enumerate(float(x) for x in raw.tolist()):
                same = mo.pool.kda_layers[i].type_name == cls
                pairs.append((v, same))
                if same:
                    per_class[cls].append(v)

    # ---- Pass B：真实 forward（写状态），采样 ‖S‖ 单步增长 ----
    # expand_delta 的量纲是范数增长，不是余弦分，必须单独采样。
    growths = []
    with torch.no_grad():
        for _cls, H in samples:
            mo(H, state=None, fw_energy=None)
            for l in mo.pool.kda_layers:
                growths.append(abs(float(l._rt["growth"])))

    if not pairs:
        raise SystemExit("所有种子输入都编码失败")
    sig = [v for v, s in pairs if s]
    bg = [v for v, s in pairs if not s]
    print(f"\n[calib] 样本 {sum(len(v) for v in per_class.values())} 条"
          f"（跳过 {skipped}）· 打分 {len(pairs)} 个"
          f" · 同类 {len(sig)} / 异类 {len(bg)} · 耗时 {time.time()-t0:.1f}s")

    print("\n=== 全体打分直方图（S=同类信号，.=异类背景）===")
    print(_histogram(pairs, bins=args.bins))

    print("\n=== 分位数 ===")
    def _row(name, v):
        if not v:
            return f"  {name:<10} (空)"
        return (f"  {name:<10} n={len(v):3d}  min={min(v):+.4f}"
                f"  p5={_quantile(v,0.05):+.4f}  p25={_quantile(v,0.25):+.4f}"
                f"  p50={_quantile(v,0.50):+.4f}  p75={_quantile(v,0.75):+.4f}"
                f"  p95={_quantile(v,0.95):+.4f}  max={max(v):+.4f}")
    print(_row("同类 S", sig))
    print(_row("异类 .", bg))
    print(_row("全体", [v for v, _ in pairs]))
    print("\n=== 各类「同类分数」===")
    for cls, v in per_class.items():
        print(_row(cls, v))

    valley, detail = _valley(pairs)
    print("\n=== 双峰谷底 / 建议阈值 ===")
    print(f"  最优点 t* = {valley:+.4f}  "
          f"误判 {detail.get('errors')}/{len(pairs)}"
          f"（异类误纳 {detail.get('false_accept')}、同类漏召 {detail.get('miss')}）"
          f"  错误率 {detail.get('rate', 0)*100:.1f}%")
    s_p5 = _quantile(sig, 0.05)
    b_p95 = _quantile(bg, 0.95)
    sep = s_p5 - b_p95
    print(f"  同类 p5 = {s_p5:+.4f}   异类 p95 = {b_p95:+.4f}"
          f"   间隔 = {sep:+.4f}")
    maxv = max(v for v, _ in pairs)
    if sep > 0:
        thr = 0.5 * (b_p95 + s_p5)
        verdict = "两分布分离，双峰谷底可信"
    else:
        thr = valley
        verdict = ("两分布重叠（同类 p5 < 异类 p95）——**没有双峰**，"
                   "谷底只是最小误判点，不能据此定死绝对阈值")
    print(f"  判定: {verdict}")
    print(f"  当前 spec 默认 threshold={spec.MOMW_THRESHOLD} "
          f"vs 实测全体最高分 {maxv:+.4f} → "
          + ("默认阈值下除 bonus 外永不激活"
             if spec.MOMW_THRESHOLD > maxv else "默认阈值可激活"))
    print("  --- 建议（只给建议，不写回 spec） ---")
    print(f"  threshold   = {thr:.4f}"
          + ("   （两分布中点）" if sep > 0 else "   （仅供参考，建议改用 quantile 模式）"))
    print(f"  B0          = {thr + max(0.15, 0.5*thr):.4f}"
          f"   （必须 > threshold；余量取 max(0.15, 0.5·threshold)）")
    if growths:
        print(f"  expand_delta= {_quantile(growths, 0.90):.4f}"
              f"   （‖S‖ 单步增长 p90；量纲与余弦分不同，单独采样）")
        print(f"  growth 分布  min={min(growths):.4f}"
              f"  p50={_quantile(growths,0.50):.4f}"
              f"  p90={_quantile(growths,0.90):.4f}"
              f"  max={max(growths):.4f}")

    # 顺带输出一份 MoMW.stats() 快照（用户要的实时字段契约）
    snap = mo.stats()
    print("\n=== MoMW.stats() 快照（末次前向）===")
    print(json.dumps({k: snap[k] for k in (
        "steps", "anchor_source", "threshold", "threshold_mode",
        "fw_gate", "queue_pending", "layer_count", "last_gamma",
        "last_raw", "last_scores")}, ensure_ascii=False, indent=2))
    print("\n[calib] 完成。注意：分布依赖当前权重；模型继续训练后需重新标定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
