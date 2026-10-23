# ===========================================================================
# GARGANTUA v1 — init.py：创建并初始化模型
# 用法：python init.py --name alpha [--out models/] [--force]
#
# 2026-10-20 修复（审查结论）：
#   1. 落盘改走 common.save_checkpoint（临时文件 + replace 原子写），
#      与 server.py / train.py 的 config.json 格式完全同构。
#   2. 默认拒绝覆盖已存在模型（防止误操作抹掉训练成果），--force 才允许。
# ===========================================================================
import argparse
import os
import time

import torch

import spec
from common import save_checkpoint, valid_model_name
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
    args = ap.parse_args()

    bad = valid_model_name(args.name)
    if bad:
        raise SystemExit(f"[init] 非法模型名 {args.name!r}: {bad}")
    for label, v in (("enc-blocks", args.enc_blocks),
                     ("dec-blocks", args.dec_blocks)):
        if v is not None and not (1 <= int(v) <= 64):
            # 0/负数会造出 0-block 实例，config 写默认值 → 目录再也无法
            # 被加载（state_dict 缺 key），属不可逆的坏目录
            raise SystemExit(f"[init] --{label} 必须在 [1, 64]: {v}")

    torch.manual_seed(args.seed)
    t0 = time.time()
    m = Gargantua(n_enc_blocks=args.enc_blocks, n_dec_blocks=args.dec_blocks)
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
        "ctx_full": spec.CTX_FULL, "ctx_compress": spec.CTX_COMPRESS,
        "csa_ratio": spec.CSA_RATIO, "hca_ratio": spec.HCA_RATIO,
        "optimizer": spec.OPTIMIZER,
    }
    save_checkpoint(d, m, config)
    print(f"[init] {d}  params={n/1e6:.1f}M  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
