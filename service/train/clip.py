# ===========================================================================
# GARGANTUA v1 — clip.py：文本×媒体对比训练（CLIP 风格对称 InfoNCE）
# ---------------------------------------------------------------------------
# 2026-11-06 新增。配对 .tokens 的解析 / 校验在 clip_data.py；本模块只负责
# 「把一段 Step 编码成 [D] 向量」与「对称 InfoNCE 损失 + 一步训练」。
#
# 关键约束（不要改动）：
#   * ContrastiveHead 是**独立模块**，绝不挂到 Gargantua 上（不要
#     `m.clip_head = head`）。common.save_checkpoint 用 model.state_dict()
#     且加载侧是 strict，附加子模块会改变键集合、破坏所有既有模型/推理。
#   * head 单独存成模型目录下的 contrast_head.pt；save_checkpoint 只回收
#     weights/*.safetensors，不会碰这个文件。
#   * 每个 segment 编码前都 reset 全部持久记忆：状态跨调用持久，不重置会
#     跨样本 / 跨段泄漏（与 forward_train 入口 reset 同一口径）。
# 本模块只依赖 torch / os / math，不引入任何项目内或第三方依赖。
# ===========================================================================
"""clip: CLIP 风格的对称对比损失与一步训练。"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_FILENAME = "contrast_head.pt"
_LOGIT_SCALE_MAX = math.log(100.0)   # CLIP 标准：温度下限 1/100

__all__ = [
    "HEAD_FILENAME", "ContrastiveHead", "head_path", "load_head",
    "save_head", "encode_segment", "clip_loss", "run_clip_step",
    "build_head_optimizer",
]


class ContrastiveHead(nn.Module):
    """文本 / 媒体各一个 D→D 投影 + 可学习 logit 尺度（温度倒数）。"""

    def __init__(self, d_model: int, init_temperature: float = 0.07):
        super().__init__()
        self.d_model = int(d_model)
        self.text_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.media_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        # 存 log(1/t)；scale() 再 exp 回来，初值令 scale ≈ 1/0.07
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / float(init_temperature))))

    def scale(self) -> torch.Tensor:
        """可微的 logit 尺度 = exp(clamp(logit_scale, max=log 100))。"""
        return self.logit_scale.clamp(max=_LOGIT_SCALE_MAX).exp()

    def temperature(self) -> float:
        """当前温度（1/scale），float。"""
        return 1.0 / float(self.scale().item())


def head_path(model_dir) -> str:
    """head 独立文件路径：<model_dir>/contrast_head.pt。"""
    return os.path.join(model_dir, HEAD_FILENAME)


def load_head(model_dir, d_model, init_temperature=0.07) -> ContrastiveHead:
    """文件不存在→新建；存在→校验 d_model 一致后 strict 载入 state_dict。"""
    head = ContrastiveHead(d_model, init_temperature)
    path = head_path(model_dir)
    if not os.path.exists(path):
        return head
    obj = torch.load(path, map_location="cpu")
    saved = int(obj["d_model"])
    if saved != head.d_model:
        raise ValueError(
            f"contrast_head.pt 的 d_model={saved} 与当前模型 {head.d_model} "
            f"不一致：模型宽度变了，请删除旧 head 或重建模型")
    head.load_state_dict(obj["state_dict"], strict=True)
    return head


def save_head(model_dir, head: ContrastiveHead) -> None:
    """原子写：先隐藏临时文件 + fsync，再 os.replace（与 common 同口径）。"""
    os.makedirs(model_dir, exist_ok=True)
    path = head_path(model_dir)
    tmp = os.path.join(model_dir, f".{HEAD_FILENAME}.{os.getpid()}.tmp")
    payload = {"d_model": head.d_model, "state_dict": head.state_dict()}
    try:
        torch.save(payload, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def encode_segment(m, steps) -> torch.Tensor:
    """一段 Step → [D]。每段独立 reset 全部持久记忆（fast weight + 可选
    MoMW），避免跨样本泄漏。

    assemble_embeddings → reset_memory → encode(emb, sids) → 沿位置平均池化。
    """
    emb, sids = m.assemble_embeddings(steps)
    m.reset_memory()
    h = m.encode(emb, sids)                  # [1, N, D]
    return h.mean(dim=1).squeeze(0)         # [D]


def clip_loss(m, head, pairs):
    """对称双向 InfoNCE（CLIP 标准）。pairs 是 clip_data.read_pair() 的 dict
    列表（含 text_steps / media_steps）；长度 <2 抛 ValueError（需要负样本）。"""
    n = len(pairs)
    if n < 2:
        raise ValueError(f"InfoNCE 至少需要 2 个配对作为负样本，收到 {n}")
    t = torch.stack([encode_segment(m, p["text_steps"]) for p in pairs])
    v = torch.stack([encode_segment(m, p["media_steps"]) for p in pairs])
    t = F.normalize(head.text_proj(t), dim=-1)       # [N, D]
    v = F.normalize(head.media_proj(v), dim=-1)      # [N, D]
    logits = head.scale() * (t @ v.T)                # [N, N]
    labels = torch.arange(n, device=logits.device)
    loss = 0.5 * (F.cross_entropy(logits, labels)
                  + F.cross_entropy(logits.T, labels))
    with torch.no_grad():
        t2m = logits.argmax(dim=1)                   # 文本 → 媒体命中率
        m2t = logits.argmax(dim=0)                   # 媒体 → 文本命中率
        diag = logits.diagonal().mean()
        off = ((logits.sum() - logits.diagonal().sum())
               / (n * n - n))
        stats = {
            "n_pairs": n,
            "temperature": head.temperature(),
            "t2m_acc": float((t2m == labels).float().mean().item()),
            "m2t_acc": float((m2t == labels).float().mean().item()),
            "diag_mean": float(diag.item()),
            "offdiag_mean": float(off.item()),
        }
    return loss, stats


def run_clip_step(m, head, pairs, model_opts, head_opt, max_grad_norm=1.0):
    """一步：清梯度 → backward → 全局裁剪（m + head）→ 所有 optimizer.step
    → reset 全部持久记忆。返回 (loss.detach(), stats)。"""
    for o in model_opts:
        o.zero_grad(set_to_none=True)
    head_opt.zero_grad(set_to_none=True)
    loss, stats = clip_loss(m, head, pairs)
    loss.backward()
    params = list(m.parameters()) + list(head.parameters())
    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    for o in model_opts:
        o.step()
    head_opt.step()
    m.reset_memory()
    return loss.detach(), stats


def build_head_optimizer(head, lr) -> torch.optim.AdamW:
    """head 参数独立走 AdamW（不进 model_opts）。"""
    return torch.optim.AdamW(head.parameters(), lr=lr)
