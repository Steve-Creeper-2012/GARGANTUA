# ===========================================================================
# GARGANTUA v1 — train.py：文本预训练（教师强制，B=1，FP32）
# ---------------------------------------------------------------------------
# 数据：data/ 下所有 .txt 文件，text_codec 编码为时间步序列；
# 每个窗口切成 user 段（encoder）+ model 段（decoder 预测目标）。
# 优化器：Muon（隐藏层 2D）+ AdamW（embedding/head/norm），--optimizer 可关。
# 用法：
#   python train.py --model models/alpha --data data --steps 200 --block-size 32
# ===========================================================================
import argparse
import glob
import os
import random
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

import spec
from model import Gargantua
from muon import build_optimizers
from tokens.text_codec import encode as text_encode

CKPT_NAME = "model.safetensors"


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
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--block-size", type=int, default=32,
                    help="窗口大小（token/时间步）；user/model 各一半")
    ap.add_argument("--lr", type=float, default=spec.ADAMW_LR)
    ap.add_argument("--muon-lr", type=float, default=spec.MUON_LR)
    ap.add_argument("--optimizer", default=spec.OPTIMIZER,
                    choices=["muon", "adamw"])
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    m = Gargantua()
    ckpt = os.path.join(args.model, CKPT_NAME)
    if os.path.exists(ckpt):
        m.load_state_dict(load_file(ckpt))
        print(f"[train] 载入 {ckpt}")
    m.train()

    opts = build_optimizers(m, args.optimizer, args.lr, args.muon_lr,
                            spec.MUON_MOMENTUM)
    ids = load_corpus_ids(args.data)
    B = args.block_size
    half = B // 2
    assert len(ids) > B + 1, f"语料太少：{len(ids)} tokens < {B + 1}"
    print(f"[train] corpus={len(ids)} tokens  steps={args.steps}  "
          f"block={B}  optimizer={args.optimizer}")

    t0 = time.time()
    for step in range(1, args.steps + 1):
        i = random.randint(0, len(ids) - B - 2)
        x_in = torch.tensor([ids[i:i + half]])
        s_in = torch.arange(half).unsqueeze(0)
        tgt = torch.tensor([ids[i + half:i + B]])
        y = torch.tensor([ids[i + half : i + B]])  
        # decoder 输入 = 目标段左移一位（首 token 用 user 段尾 token 播种）
        dec_in = torch.tensor([[ids[i + half - 1]] + ids[i + half:i + B - 1]])
        s_dec = torch.arange(half, half + half).unsqueeze(0)

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
            save_file(m.state_dict(), ckpt)
            print(f"[train] saved {ckpt}")
    save_file(m.state_dict(), ckpt)
    print(f"[train] done, saved {ckpt}")


if __name__ == "__main__":
    main()
