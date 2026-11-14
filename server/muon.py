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
#
# 2026-10-20 修复（审查结论）：
#   1. rms_align 的注释与实现对齐：newton_schulz5 输出为半正交矩阵，元素
#      RMS ≈ 1/sqrt(max(shape))，乘 sqrt(max(shape)) 后更新 RMS ≈ 1.0
#      （Keller Jordan 原版标度），并非注释声称的「Adam 的 ~0.2 RMS」。
#      按 RMS≈1.0 理解后，Muon lr=2e-2 对应每步权重更新 RMS≈0.02，与
#      modded-nanogpt 的常用配置一致。若要 Moonlight 式 0.2 标度，应把
#      系数改为 0.2*sqrt(max(shape))——那需要同步重调 MUON_LR，本次不改
#      数值行为，仅修正文档。
#   2. weight_decay 参数化：Muon 组默认 0（其 lr=2e-2 比 AdamW 大两个
#      数量级，同一 wd 值在两种优化器下的有效衰减速率差 ~100 倍，故不
#      强行统一数值，改为显式传入并由调用方知晓）；adamw 模式下隐藏层
#      矩阵沿用 0.1、embedding/norm 组 0（原行为不变）。
# ===========================================================================
from __future__ import annotations

import time
import warnings
from typing import Iterable, List

import torch

# Manifold Muon + share_enc_dec 高风险告警只发一次（跨多次 build_optimizers）。
_SHARE_WARNED = False


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
                 rms_align: bool = True, profile: bool = False):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        rms_align=rms_align)
        super().__init__(params, defaults)
        # profile 仅作性能计时开关，刻意不放进 defaults：避免污染 param_group
        # 字典、改变 Optimizer 的 state_dict 结构；默认 False 时数值逐位不变。
        self.profile = profile
        self.ns5_seconds = 0.0
        self.step_seconds = 0.0
        self.ns5_calls = 0

    @torch.no_grad()
    def step(self, closure=None):
        _t_step0 = time.perf_counter() if self.profile else None
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
                    _t_ns = time.perf_counter() if self.profile else 0.0
                    u = newton_schulz5(u, steps=group["ns_steps"]).to(p.dtype)
                    if self.profile:
                        self.ns5_seconds += time.perf_counter() - _t_ns
                        self.ns5_calls += 1
                    if group["rms_align"]:
                        # Moonlight RMS 对齐：更新幅度与矩阵形状解耦，
                        # 使 Muon 学习率可跨形状迁移（对齐 Adam 的 ~0.2 RMS）
                        u = u * (max(p.shape) ** 0.5)
                # 非 2D 参数理论上不该进 Muon 组（分组契约保证），原样兜底
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u, alpha=-group["lr"])
        if self.profile:
            self.step_seconds += time.perf_counter() - _t_step0
        return loss


def build_optimizers(model, optimizer: str = "muon",
                     adamw_lr: float = 3e-4, muon_lr: float = 2e-2,
                     muon_momentum: float = 0.95, manifold=None,
                     muon_rms_align=None, muon_weight_decay=None):
    """ 按 spec 13 分组构建优化器。
    返回 List[Optimizer]（训练循环里对每个 step()）。

    optimizer 三档（2026-11-07 训练界面新增）：
      "adamw"    全部参数 AdamW；
      "muon"     参杂：隐藏层 2D 矩阵 Muon，embedding / lm_head / norm /
                 标量 / 卷积等其余参数 AdamW（历史默认分工）；
      "allmuon"  全 Muon：把 **所有满秩 2D 参数**（含 embedding /
                 lm_head）都交给 Muon；1D 参数与 **min(shape) < 2 的
                 退化 2D 矩阵**（如 [1,D] 的 attn_res_proj /
                 mlp_res_proj）仍走 AdamW——Muon 的核心是矩阵正交化，
                 对向量/秩 1 矩阵没有意义，且 2e-2 的 lr 叠加 RMS 对齐
                 会把它们直接推爆。判据与 model._classify_param 一致，
                 是 Muon 的物理边界，不是「没做到全 Muon」。

    manifold（阶段 2，Manifold Muon）：
      可选配置 dict；None 时回退 getattr(model, "manifold_config", None)。
      仅当 enabled 为真且 optimizer ∈ {"muon","allmuon"} 时，把命中的
      attention / MLP 投影从 Muon 组拆出交给 ManifoldMuon，其余参数仍走
      原 Muon。enabled=False / 缺配置 / "adamw" 档时完全不走该分支，
      返回列表、分组、数值行为与改动前逐位一致。

    muon_rms_align / muon_weight_decay（阶段 0 校准对照，2026-11-13）：
      可选覆盖 Muon 组的 rms_align / weight_decay。**默认 None = 完全沿用
      Muon 类自身默认值**（rms_align=True / wd=0），所有既有调用逐位不变；
      仅显式传值时才透传，用于验证「步子太大」假设的受控消融。
    """
    import model as _m
    global _SHARE_WARNED
    groups = _m.param_groups_for_optimizers(model)
    opts: List[torch.optim.Optimizer] = []
    muon_kw = {}
    if muon_rms_align is not None:
        muon_kw["rms_align"] = bool(muon_rms_align)
    if muon_weight_decay is not None:
        muon_kw["weight_decay"] = float(muon_weight_decay)

    # —— Manifold Muon 接入（默认关闭）——
    cfg = manifold if manifold is not None else getattr(model, "manifold_config", None)
    use_manifold = (isinstance(cfg, dict) and bool(cfg.get("enabled"))
                    and optimizer in ("muon", "allmuon"))
    # id(Parameter) -> kind。用 id 而非 name：select_manifold_params 已按 id
    # 去重共享权重，这里才能把 muon 组里同一个 Parameter 精确拆出。
    id_to_kind = {}
    if use_manifold:
        import manifold_muon
        selected = manifold_muon.select_manifold_params(model, cfg)
        name_to_p = dict(model.named_parameters())
        for name, kind in selected.items():
            p = name_to_p.get(name)
            if p is not None and kind in ("stiefel", "dgram"):
                id_to_kind[id(p)] = kind
        if getattr(model, "share_enc_dec", False) and not _SHARE_WARNED:
            warnings.warn(
                "Manifold Muon 在 share_enc_dec=True 下为高风险：enc/dec "
                "共享同一 Parameter，梯度累加后一次 retraction 同时服务 "
                "encoder 双向与 decoder 因果两种语义（报告 3.4.3 标记为"
                "无文献支撑的高风险）。", RuntimeWarning, stacklevel=2)
            _SHARE_WARNED = True

    def _split(param_list):
        """把命中 Manifold 的 Parameter 拆走，返回 (剩余列表, mm_groups|None)。
        id_to_kind 为空（未启用）时原样返回，保证数值/分组逐位一致。"""
        if not id_to_kind:
            return param_list, None
        buckets = {"stiefel": [], "dgram": []}
        remaining = []
        for p in param_list:
            k = id_to_kind.get(id(p))
            if k is None:
                remaining.append(p)
            else:
                buckets[k].append(p)
        mm_groups = [{"params": buckets[k], "kind": k}
                     for k in ("stiefel", "dgram") if buckets[k]]
        return remaining, (mm_groups or None)

    def _make_manifold(mm_groups):
        import manifold_muon
        return manifold_muon.ManifoldMuon(
            mm_groups,
            lr=cfg.get("lr", 2e-2), momentum=cfg.get("momentum", 0.95),
            K=cfg.get("K", 20), alpha=cfg.get("alpha", 0.1),
            ns_steps=cfg.get("ns_steps", 5),
            exact_msign=cfg.get("exact_msign", False))

    if optimizer == "allmuon":
        # 2026-11-14 审计修复（P0）：原实现只按 p.ndim == 2 从两组里捞
        # 矩阵，绕过了 model._classify_param 的形状兜底，把 4 个 [1,D]
        # 秩 1 投影（attn_res_proj / mlp_res_proj）塞进 Newton-Schulz。
        # 这里补上 min(shape) >= 2 判据，并用 id 做一次完整划分，保证
        # 每个可训练参数恰好落进 Muon 或 AdamW 之一（不重不漏）。
        all_params = groups["muon"] + groups["adamw"]
        matrices = [p for p in all_params
                    if p.ndim == 2 and min(p.shape) >= 2]
        matrix_ids = {id(p) for p in matrices}
        vectors = [p for p in all_params if id(p) not in matrix_ids]
        matrices, mm_groups = _split(matrices)
        if mm_groups is not None:
            opts.append(_make_manifold(mm_groups))
        if matrices:
            opts.append(Muon(matrices, lr=muon_lr, momentum=muon_momentum,
                             **muon_kw))
        if vectors:
            opts.append(torch.optim.AdamW(vectors, lr=adamw_lr,
                                          betas=(0.9, 0.95),
                                          weight_decay=0.0))
        return opts
    if optimizer == "muon" and groups["muon"]:
        muon_params, mm_groups = _split(groups["muon"])
        if mm_groups is not None:
            opts.append(_make_manifold(mm_groups))
        if muon_params:
            opts.append(Muon(muon_params, lr=muon_lr, momentum=muon_momentum,
                             **muon_kw))
    else:
        if groups["muon"]:
            opts.append(torch.optim.AdamW(groups["muon"], lr=adamw_lr,
                                          betas=(0.9, 0.95), weight_decay=0.1))
    opts.append(torch.optim.AdamW(groups["adamw"], lr=adamw_lr,
                                  betas=(0.9, 0.95), weight_decay=0.0))
    return opts
