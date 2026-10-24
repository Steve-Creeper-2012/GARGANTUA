# ===========================================================================
# GARGANTUA v1 — infer.py：文本推理（自回归，共享 KV 逐步回写）
# 用法：
#   python infer.py --model models/alpha --prompt "你好" --max-new-tokens 32 \
#       --temperature 0.8 --top-p 0.95
#
# 2026-10-20 修复（审查结论）：
#   1. main() 改用 common.build_model：按模型目录 config.json 的 block 数
#      构造，非默认 block 数的模型不再 load_state_dict 崩溃。
#   2. fast_weight.reset() 移到 encode 之前：原实现在 encode 之后 reset，
#      把 encoder 阶段写入的 fast weight 状态清掉，违反 spec「enc/dec
#      共享 bank、状态跨调用推进」，且与 train.py 的 forward_train 不一致。
#   3. KV 回写结构与训练逐位置对齐（review-pipeline 探针实测）：
#      训练 KV 流 = enc(prompt) + [emb(seed)@n, emb(t0)@n+1, …]，即每个
#      decoder 输入的 embedding 放在【它要预测的那个 token 的步】上。
#      本实现 therefore 把「当前输入 emb」拼进视图（s_dec=cur_step），
#      decode 后把该 emb 持久化进 KV；不再单独写回生成 token 的 emb
#      （旧实现因此让种子的 MODEL 嵌入从未进入 KV，且每个生成 token 比
#      训练早一步——train/infer 分布偏差，已回归测试固定）。
# ===========================================================================
import argparse
import os

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
from common import CKPT_NAME, build_model
from tokens.text_codec import encode as text_encode, decode as text_decode

# API/CLI 层硬上限：CPU 上单步是全序列重解码，2048 已是宽松保护值
MAX_NEW_TOKENS = 2048


@torch.no_grad()
def generate(m, prompt: str, max_new: int = 32,
             temperature: float = 0.8, top_p: float = 0.95):
    """ 编码 prompt → encoder → KV；decoder 逐步生成并回写共享 KV。
    返回生成的时间步序列（List[Step]，text codec 可直接 decode）。 """
    max_new = int(max_new)
    if not (1 <= max_new <= MAX_NEW_TOKENS):
        raise ValueError(f"max_new 必须在 [1, {MAX_NEW_TOKENS}]: {max_new}")
    if not (temperature > 0):
        raise ValueError(f"temperature 必须 > 0: {temperature}")
    if not (0 < top_p <= 1):
        raise ValueError(f"top_p 必须在 (0, 1]: {top_p}")

    steps = text_encode(prompt)
    ids = [t for st in steps for t in st["token_ids"]]
    if not ids:
        raise ValueError("prompt 编码为空")

    # 新生成序列：fast weight 状态从零开始；encode 写入的状态由 decode 继续推进
    m.fast_weight.reset()
    x_in = torch.tensor([ids])
    s_in = torch.arange(len(ids)).unsqueeze(0)
    emb_in = m.embed_text(x_in, torch.full_like(x_in, spec.TYPE_USER))
    enc_out = m.encode(emb_in, s_in)

    # 共享 KV 初始 = encoder 输出；decoder 每步把当前输入 emb 拼进视图，
    # decode 后持久化进 KV（与 forward_train 的 KV 流逐位置同构）
    kv_vecs = enc_out.detach()
    kv_steps = s_in
    cur = ids[-1]                            # 用 prompt 尾 token 播种
    cur_step = len(ids)                      # 即将预测的 token 的时间步
    out_ids = []
    for _ in range(max_new):
        # 输入 token 的 embedding 放在其预测步上（s_dec = cur_step），
        # 与训练 make_batch 的 s_dec=arange(half, …) 约定一致
        s_dec = torch.tensor([[cur_step]])
        dec_in = torch.tensor([[cur]])
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
        # 持久化【当前输入】的 emb（训练 KV 流同构）；下一轮输入即 nxt
        kv_vecs = torch.cat([kv_vecs, emb.detach()], 1)
        kv_steps = torch.cat([kv_steps, s_dec], 1)
        cur, cur_step = nxt, cur_step + 1
    return [{"token_ids": [i], "type": spec.TYPE_MODEL,
             "modality": "text", "meta": {}} for i in out_ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型目录（含 model.safetensors）")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    # 缺权重文件必须显式报错：build_model 静默用随机初始化会让手误的
    # 路径产出「看似正常」的乱码输出（审查记录 P2）
    if not os.path.exists(os.path.join(args.model, CKPT_NAME)):
        raise SystemExit(f"[infer] 模型目录无 {CKPT_NAME}: {args.model}")
    m = build_model(args.model)
    m.eval()
    steps = generate(m, args.prompt, args.max_new_tokens,
                     args.temperature, args.top_p)
    print(text_decode(steps))


if __name__ == "__main__":
    main()
