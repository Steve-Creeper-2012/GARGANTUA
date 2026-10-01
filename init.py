# ===========================================================================
# GARGANTUA v1 — init.py：创建并初始化模型
# 用法：python init.py --name alpha [--out models/]
# ===========================================================================
import argparse
import json
import os
import time

import torch
from safetensors.torch import save_file

import spec
from model import Gargantua

CKPT_NAME = "model.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="模型名（目录名）")
    ap.add_argument("--out", default="models", help="模型根目录")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enc-blocks", type=int, default=None,
                    help=f"encoder block 数（默认 {spec.N_ENC_BLOCKS}）")
    ap.add_argument("--dec-blocks", type=int, default=None,
                    help=f"decoder block 数（默认 {spec.N_DEC_BLOCKS}）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    t0 = time.time()
    m = Gargantua(n_enc_blocks=args.enc_blocks, n_dec_blocks=args.dec_blocks)
    n = sum(p.numel() for p in m.parameters())

    d = os.path.join(args.out, args.name)
    os.makedirs(d, exist_ok=True)
    save_file(m.state_dict(), os.path.join(d, CKPT_NAME))
    config = {
        "name": args.name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": n,
        "d_model": spec.D_MODEL, "head_dim": spec.HEAD_DIM,
        "n_enc_blocks": args.enc_blocks or spec.N_ENC_BLOCKS,
        "n_dec_blocks": args.dec_blocks or spec.N_DEC_BLOCKS,
        "vocab_size_padded": spec.VOCAB_SIZE_PADDED,
        "ctx_full": spec.CTX_FULL, "ctx_compress": spec.CTX_COMPRESS,
        "csa_ratio": spec.CSA_RATIO, "hca_ratio": spec.HCA_RATIO,
        "optimizer": spec.OPTIMIZER,
    }
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[init] {d}  params={n/1e6:.1f}M  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
