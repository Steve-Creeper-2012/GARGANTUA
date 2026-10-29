# ===========================================================================
# GARGANTUA v1 — train_step.py：训练窗口 → 前向 → loss 的唯一实现
# ---------------------------------------------------------------------------
# train.py（独立入口）与 auto_train.py（队列守护）共用本模块，避免两条
# 训练路径的窗口切法/损失口径悄悄分叉。
#
# 两种文档：
#   纯文本  → make_batch + forward_train（查表路径，最省）
#   多模态  → make_batch_emb + forward_train_embeddings（媒体走线性投影）
# 媒体位置的训练目标固定为 -100（ignore_index）：视觉/音频没有词表 ID，
# lm_head 预测不了，只能作为上下文；整窗无文本目标时调用方跳过。
# ===========================================================================
"""train_step: 文档 + 窗口 → (logits, y, n_valid) → loss。"""

from __future__ import annotations

import torch.nn.functional as F

from common import make_batch, make_batch_emb
from tokens.tokenset import doc_embeddings, pick_window


def window_index(doc, block_size: int, rng, force_last: bool = False):
    """选一个窗口起点；文档太短或无解返回 None。

    纯文本文档沿用原随机策略（0 .. n-B-1）；多模态文档走 pick_window，
    以文本目标为锚点反推起点——否则随机起点常常整窗都是媒体。
    """
    n = len(doc["pos_ids"])
    if n < block_size + 2:
        return None
    if doc["multimodal"]:
        return pick_window(doc, block_size, rng=rng, force_last=force_last)
    if force_last:
        # 固定取尾部窗口：让 y 覆盖文档末尾的 <stop>
        return n - block_size
    return rng.randint(0, n - block_size - 1)


def forward_window(m, doc, i: int, block_size: int):
    """跑一次前向 → (logits, y, n_valid)。

    n_valid = y 里可训练目标的个数（多模态窗可能整段都是媒体）。
    """
    if doc["multimodal"]:
        emb = doc_embeddings(m, doc)
        emb_in, s_in, emb_dec, s_dec, y, n_valid = make_batch_emb(
            emb, doc["pos_ids"], i, block_size, doc["step_ids"])
        logits = m.forward_train_embeddings(emb_in, s_in, emb_dec, s_dec)
        return logits, y, n_valid
    x_in, s_in, dec_in, s_dec, y = make_batch(
        doc["pos_ids"], i, block_size, doc["step_ids"])
    logits = m.forward_train(x_in, s_in, dec_in, s_dec)
    return logits, y, int(y.numel())


def window_loss(logits, y):
    """交叉熵；媒体位置（y = -100）不计入。"""
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1),
                           ignore_index=-100)
