# ===========================================================================
# GARGANTUA v1 — init.py：创建并初始化模型
# 用法：python init.py --name alpha [--out models/] [--force]
#       [--enc-blocks N] [--dec-blocks N]
#       [--ctx-full N] [--ctx-compress N]
#       [--csa-ratio N] [--hca-ratio N] [--csa-window N]
#
# 2026-10-20 修复（审查结论）：
#   1. 落盘改走 common.save_checkpoint（临时文件 + replace 原子写），
#      与 server.py / train.py 的 config.json 格式完全同构。
#   2. 默认拒绝覆盖已存在模型（防止误操作抹掉训练成果），--force 才允许。
# 2026-10-23：新增上下文长度选项（完整/压缩容量 + CSA/HCA 压缩比与
#   局部窗）。语义：KV = 完整区（ctx_full 个原始 token）｜压缩区
#   （ctx_compress 容量，对外有 CSA/HCA 两个同源视图，压缩比分别可配）。
#   缺省即 spec 默认（1024/3072/4/32/512），与旧行为逐位一致。
# ===========================================================================
import argparse
import os
import time

import torch

# ---- GARGANTUA 路径引导（目录树重构：把 server/ 与 service/* 放上 path）----
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "start" / "boot.py").is_file():
        _sys.path.insert(0, str(_anc / "start"))
        break
else:
    raise SystemExit("找不到 GARGANTUA 项目根（start/boot.py）")
import boot as _boot
_boot.bootstrap()
import spec
from common import (MAX_BLOCKS, save_checkpoint, valid_model_name,
                    validate_ctx_config)
from model import Gargantua


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="模型名（目录名）")
    ap.add_argument("--out", default="models", help="模型根目录")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enc-blocks", type=int, default=None,
                    help=f"encoder block 数（默认 {spec.N_ENC_BLOCKS}）")
    ap.add_argument("--dec-blocks", type=int, default=None,
                    help=f"decoder block 数（默认 {spec.N_DEC_BLOCKS}）")
    ap.add_argument("--force", action="store_true",
                    help="允许覆盖已存在的同名模型")
    # ---- 上下文长度（2026-10-23 新功能）----
    ap.add_argument("--ctx-full", type=int, default=None,
                    help=f"完整 KV 区容量，token（默认 {spec.CTX_FULL}）")
    ap.add_argument("--ctx-compress", type=int, default=None,
                    help=f"压缩区容量（原始 token 计，默认 "
                         f"{spec.CTX_COMPRESS}；CSA/HCA 是它的两个视图）")
    ap.add_argument("--csa-ratio", type=int, default=None,
                    help=f"CSA 压缩比 N:1（默认 {spec.CSA_RATIO}，"
                         f"局部滑窗注意力）")
    ap.add_argument("--hca-ratio", type=int, default=None,
                    help=f"HCA 压缩比 N:1（默认 {spec.HCA_RATIO}，"
                         f"全局注意力；须 ≥ csa-ratio）")
    ap.add_argument("--csa-window", type=int, default=None,
                    help=f"CSA 在完整区尾部的局部窗（默认 "
                         f"{spec.CSA_LOCAL_WINDOW}，≤ ctx-full）")
    args = ap.parse_args()

    bad = valid_model_name(args.name)
    if bad:
        raise SystemExit(f"[init] 非法模型名 {args.name!r}: {bad}")
    for label, v in (("enc-blocks", args.enc_blocks),
                     ("dec-blocks", args.dec_blocks)):
        if v is not None and not (1 <= int(v) <= MAX_BLOCKS):
            # 0/负数会造出 0-block 实例，config 写默认值 → 目录再也无法
            # 被加载（state_dict 缺 key），属不可逆的坏目录
            raise SystemExit(f"[init] --{label} 必须在 [1, {MAX_BLOCKS}]: {v}")
    try:
        ctx = validate_ctx_config(ctx_full=args.ctx_full,
                                  ctx_compress=args.ctx_compress,
                                  csa_ratio=args.csa_ratio,
                                  hca_ratio=args.hca_ratio,
                                  csa_local_window=args.csa_window)
    except ValueError as e:
        raise SystemExit(f"[init] {e}")

    torch.manual_seed(args.seed)
    t0 = time.time()
    m = Gargantua(n_enc_blocks=args.enc_blocks, n_dec_blocks=args.dec_blocks,
                  **ctx)
    n = sum(p.numel() for p in m.parameters())

    d = os.path.join(args.out, args.name)
    if os.path.exists(os.path.join(d, "model.safetensors")) and not args.force:
        raise SystemExit(f"[init] {d} 已存在，拒绝覆盖（--force 强制）")
    config = {
        "name": args.name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": n,
        "d_model": spec.D_MODEL, "head_dim": spec.HEAD_DIM,
        # is not None：0/None 都要如实记录（0 已被上面拒绝，防御性保留）
        "n_enc_blocks": (args.enc_blocks if args.enc_blocks is not None
                         else spec.N_ENC_BLOCKS),
        "n_dec_blocks": (args.dec_blocks if args.dec_blocks is not None
                         else spec.N_DEC_BLOCKS),
        "vocab_size_padded": spec.VOCAB_SIZE_PADDED,
        # 上下文长度（建模期可选；CSA/HCA 是同一段压缩区内容的两个视图）
        "ctx_full": ctx["ctx_full"],
        "ctx_compress": ctx["ctx_compress"],
        "csa_ratio": ctx["csa_ratio"],
        "hca_ratio": ctx["hca_ratio"],
        "csa_local_window": ctx["csa_local_window"],
        "optimizer": spec.OPTIMIZER,
    }
    save_checkpoint(d, m, config)
    total_ctx = ctx["ctx_full"] + ctx["ctx_compress"]
    print(f"[init] {d}  params={n/1e6:.1f}M  ({time.time()-t0:.1f}s)")
    print(f"[init] 上下文: full={ctx['ctx_full']} + compress="
          f"{ctx['ctx_compress']} = {total_ctx}；"
          f"CSA {ctx['csa_ratio']}:1(窗 {ctx['csa_local_window']}) · "
          f"HCA {ctx['hca_ratio']}:1")


if __name__ == "__main__":
    main()
