# ===========================================================================
# GARGANTUA v1 — Muon 优化器（MomentUm Orthogonalized by Newton-Schulz）
# ---------------------------------------------------------------------------
# 参考：Keller Jordan, "Muon: An optimizer for hidden layers in neural
# networks" (2024-12)；Moonshot "Muon is Scalable for LLM Training"
# (arXiv 2502.16982) 的 RMS 对齐与解耦 weight decay。
#
# 分工（spec 13 定案）：
#   Muon   —— 隐藏层 2D 权重矩阵（model.param_groups_for_optimizers 分组）
#   AdamW  —— embedding / lm_head / RMSNorm / 标量 / 卷积
# 开关：spec.OPTIMIZER = "muon" | "adamw"，关掉即全部参数走 AdamW。
# ===========================================================================
from __future__ import annotations

from typing import Iterable, List

import torch


@torch.no_grad()
def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7
                   ) -> torch.Tensor:
    """ Newton-Schulz 迭代近似正交化（ quintic 系数，bf16 稳定）。
    返回与 G 同形的最近半正交矩阵（USVᵀ 的 UVᵀ）。 """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.float32)                      # x86 CPU 无 bf16 加速，fp32 即可
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """ 对 2D 参数：SGD-momentum → Newton-Schulz 正交化 → RMS 对齐更新。

    参数：
      lr        学习率（Muon 组，与 AdamW 学习率不同量级）
      momentum  动量系数（默认 spec.MUON_MOMENTUM = 0.95）
      nesterov  Nesterov 动量
      ns_steps  Newton-Schulz 迭代步数（默认 5）
      weight_decay 解耦权重衰减（Moonlight 版，默认 0）
      rms_align 把更新 RMS 对齐到 Adam 量级（Moonlight 版，便于超参迁移）
    """

    def __init__(self, params: Iterable[torch.Tensor], lr: float = 2e-2,
                 momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, weight_decay: float = 0.0,
                 rms_align: bool = True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        rms_align=rms_align)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                u = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

                if p.ndim == 2:                      # 核心：正交化 2D 更新
                    u = newton_schulz5(u, steps=group["ns_steps"]).to(p.dtype)
                    if group["rms_align"]:
                        # Moonlight RMS 对齐：更新幅度与矩阵形状解耦，
                        # 使 Muon 学习率可跨形状迁移（对齐 Adam 的 ~0.2 RMS）
                        u = u * (max(p.shape) ** 0.5)
                # 非 2D 参数理论上不该进 Muon 组（分组契约保证），原样兜底
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u, alpha=-group["lr"])
        return loss


def build_optimizers(model, optimizer: str = "muon",
                     adamw_lr: float = 3e-4, muon_lr: float = 2e-2,
                     muon_momentum: float = 0.95):
    """ 按 spec 13 分组构建优化器。
    返回 List[Optimizer]（训练循环里对每个 step()）。 """
    import model as _m
    groups = _m.param_groups_for_optimizers(model)
    opts: List[torch.optim.Optimizer] = []
    if optimizer == "muon" and groups["muon"]:
        opts.append(Muon(groups["muon"], lr=muon_lr, momentum=muon_momentum))
    else:
        if groups["muon"]:
            opts.append(torch.optim.AdamW(groups["muon"], lr=adamw_lr,
                                          betas=(0.9, 0.95), weight_decay=0.1))
    opts.append(torch.optim.AdamW(groups["adamw"], lr=adamw_lr,
                                  betas=(0.9, 0.95), weight_decay=0.0))
    return opts
