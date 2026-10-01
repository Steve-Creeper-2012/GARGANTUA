# ===========================================================================
# GARGANTUA v1 — infer.py：文本推理（自回归，共享 KV 逐步回写）
# 用法：
#   python infer.py --model models/alpha --prompt "你好" --max-new-tokens 32 \
#       --temperature 0.8 --top-p 0.95
# ===========================================================================
import argparse
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import spec
from model import Gargantua
from tokens.text_codec import encode as text_encode, decode as text_decode

CKPT_NAME = "model.safetensors"


@torch.no_grad()
def generate(m: Gargantua, prompt: str, max_new: int = 32,
             temperature: float = 0.8, top_p: float = 0.95):
    """ 编码 prompt → encoder → KV；decoder 逐步生成并回写共享 KV。
    返回生成的时间步序列（List[Step]，text codec 可直接 decode）。 """
    steps = text_encode(prompt)
    ids = [t for st in steps for t in st["token_ids"]]
    if not ids:
        raise ValueError("prompt 编码为空")
    x_in = torch.tensor([ids])
    s_in = torch.arange(len(ids)).unsqueeze(0)
    emb_in = m.embed_text(x_in, torch.full_like(x_in, spec.TYPE_USER))
    enc_out = m.encode(emb_in, s_in)

    # 共享 KV 初始 = encoder 输出；decoder 每步生成的 embedding 追加回写
    kv_vecs = enc_out.detach()
    kv_steps = s_in
    cur = ids[-1]                            # 用 prompt 尾 token 播种
    cur_step = len(ids)
    out_ids = []
    m.fast_weight.reset()
    for _ in range(max_new):
        dec_in = torch.tensor([[cur]])
        s_dec = torch.tensor([[cur_step]])
        emb = m.embed_text(dec_in, torch.full_like(dec_in, spec.TYPE_MODEL))
        kv_hca, kv_csa = m.build_views(
            torch.cat([kv_vecs, emb], 1),
            torch.cat([kv_steps, s_dec], 1))
        dec_out = m.decode(emb, s_dec, kv_hca, kv_csa)
        logits = m.logits(dec_out)[0, -1].float() / max(temperature, 1e-5)
        # top-p 核采样
        probs = F.softmax(logits, dim=-1)
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=0)
        keep = cum - sorted_p < top_p
        sorted_p = sorted_p * keep
        sorted_p = sorted_p / sorted_p.sum().clamp(min=1e-9)
        nxt = sorted_i[torch.multinomial(sorted_p, 1)].item()
        if nxt == spec.STOP_ID:             # 模型输出停止 token → 立即停止
            break
        out_ids.append(nxt)
        # 回写共享 KV（类型=模型输出）
        emb_n = m.embed_text(torch.tensor([[nxt]]),
                             torch.tensor([[spec.TYPE_MODEL]])).detach()
        kv_vecs = torch.cat([kv_vecs, emb_n], 1)
        kv_steps = torch.cat([kv_steps, torch.tensor([[cur_step]])], 1)
        cur, cur_step = nxt, cur_step + 1
    return [{"token_ids": [i], "type": spec.TYPE_MODEL,
             "modality": "text", "meta": {}} for i in out_ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    m = Gargantua()
    ckpt = os.path.join(args.model, CKPT_NAME)
    m.load_state_dict(load_file(ckpt))
    m.eval()
    steps = generate(m, args.prompt, args.max_new_tokens,
                     args.temperature, args.top_p)
    print(text_decode(steps))


if __name__ == "__main__":
    main()
