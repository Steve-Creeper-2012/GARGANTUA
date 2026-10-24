# ===========================================================================
# GARGANTUA v1 — train.py：文本预训练（教师强制，B=1，FP32）
# ---------------------------------------------------------------------------
# 数据：data/ 下所有 .txt 文件，text_codec 编码为时间步序列；
# 每个窗口切成 user 段（encoder）+ model 段（decoder 预测目标）。
# 优化器：Muon（隐藏层 2D）+ AdamW（embedding/head/norm），--optimizer 可关。
# 用法：
#   python train.py --model models/alpha --data data --steps 200 --block-size 32
#
# 2026-10-20 修复（审查结论）：
#   1. main() 改用 common.build_model：按模型目录 config.json 的 block 数
#      构造与加载，非默认 block 数的模型不再崩溃。
#   2. block_size 经 common.validate_block_size 校验（偶数 ≥2）——奇数会
#      令 s_dec 与 dec_in 长度不一致，forward_train 必崩（已实测复现）。
#   3. 删除死变量 tgt（与 y 完全相同）。
# ===========================================================================
import argparse
import glob
import os
import random
import time

import torch
import torch.nn.functional as F

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
from common import (CKPT_NAME, build_model, make_batch, read_config,
                    save_checkpoint, validate_block_size)
from muon import build_optimizers
from tokens.text_codec import encode as text_encode


def load_corpus_ids(data_dir: str):
    """ data/ 下全部 .txt → 拼接的 token id 流（每 token 一个时间步）。 """
    ids = []
    for path in sorted(glob.glob(os.path.join(data_dir, "**", "*.txt"),
                                 recursive=True)):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for st in text_encode(text):
            ids.extend(st["token_ids"])
        ids.append(spec.STOP_ID)            # 文档结束 → 停止 token，教模型学会终止
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型目录（含 model.safetensors）")
    ap.add_argument("--data", default="data")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--block-size", type=int, default=32,
                    help="窗口大小（token/时间步，偶数）；user/model 各一半")
    ap.add_argument("--lr", type=float, default=spec.ADAMW_LR)
    ap.add_argument("--muon-lr", type=float, default=spec.MUON_LR)
    ap.add_argument("--optimizer", default=spec.OPTIMIZER,
                    choices=["muon", "adamw"])
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    B = validate_block_size(args.block_size)
    m = build_model(args.model)
    m.train()

    opts = build_optimizers(m, args.optimizer, args.lr, args.muon_lr,
                            spec.MUON_MOMENTUM)
    ids = load_corpus_ids(args.data)
    assert len(ids) > B + 1, f"语料太少：{len(ids)} tokens < {B + 2}"
    print(f"[train] corpus={len(ids)} tokens  steps={args.steps}  "
          f"block={B}  optimizer={args.optimizer}")

    t0 = time.time()
    for step in range(1, args.steps + 1):
        i = random.randint(0, len(ids) - B - 2)
        x_in, s_in, dec_in, s_dec, y = make_batch(ids, i, B)

        logits = m.forward_train(x_in, s_in, dec_in, s_dec)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        for o in opts:
            o.step()
        m.fast_weight.reset()          # 样本间截断持久记忆

        if step % 10 == 0 or step == 1:
            el = time.time() - t0
            print(f"[train] step {step:4d}/{args.steps}  "
                  f"loss {loss.item():.4f}  {el/step:.2f}s/it")
        if step % args.save_every == 0:
            save_checkpoint(args.model, m, _config_dict(args, m))
            print(f"[train] saved {os.path.join(args.model, CKPT_NAME)}")
    save_checkpoint(args.model, m, _config_dict(args, m))
    print(f"[train] done, saved {os.path.join(args.model, CKPT_NAME)}")


def _config_dict(args, m) -> dict:
    """ 与 init.py / server.py 同构的 config.json 内容（block 数以实例为准，
    保留首次创建的 created 时间戳）。

    2026-10-23：上下文长度从模型实例读回（建模时可配）——此前这里硬写
    spec 常量，训练一存盘就会把自定义 ctx_full/CSA/HCA 覆写成默认值，
    磁盘配置与真实结构脱节。 """
    cfg = dict(read_config(args.model))          # 保留既有字段（如 created）
    cfg.update({
        "name": os.path.basename(os.path.normpath(args.model)),
        "params": sum(p.numel() for p in m.parameters()),
        "d_model": spec.D_MODEL, "head_dim": spec.HEAD_DIM,
        "n_enc_blocks": len(m.enc_blocks),
        "n_dec_blocks": len(m.dec_blocks),
        "vocab_size_padded": spec.VOCAB_SIZE_PADDED,
        "ctx_full": getattr(m, "ctx_full", spec.CTX_FULL),
        "ctx_compress": getattr(m, "ctx_compress", spec.CTX_COMPRESS),
        "csa_ratio": getattr(m, "csa_ratio", spec.CSA_RATIO),
        "hca_ratio": getattr(m, "hca_ratio", spec.HCA_RATIO),
        "csa_local_window": getattr(m, "csa_local_window",
                                    spec.CSA_LOCAL_WINDOW),
        "optimizer": args.optimizer,
    })
    cfg.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
    return cfg


if __name__ == "__main__":
    main()
