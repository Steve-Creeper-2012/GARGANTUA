# ===========================================================================
# GARGANTUA v1 — server/momw.py：MoMW 三阶段无训练记忆阀门
# ---------------------------------------------------------------------------
# 规格来源：steve 2026-11-08 交付的两版《MoMW 系统架构与实现提示词》
#           （第二版 = 三阶段路由：锚点播种 → 状态特征漂移 → 空层分裂）。
#
# ==== 与规格伪码的差异（都是本机架构的硬约束，不能照抄）====
#   1) 规格假设 kda.S 是 [Dim,Dim] 可调用矩阵、kda(H) 返回 hidden_states。
#      项目里的 KDA 是 KDAAttention：状态 S 是 [B,H,dk,dv] 递推终态，
#      forward 返回 (out, new_S, new_conv_cache)。所以「主特征向量」是从
#      G_h = S_hᵀ S_h 做幂迭代得到 v_h，再经 KDA 自己的 w_o 投影回 D 维
#      得到 E_i —— 只有落回隐空间才能和 H_t 算余弦相似度。
#   2) 「零梯度训练 / 零可学习参数」+ common.build_model 是 strict load +
#      safetensors 禁共享 storage ⇒ 一个 nn.Parameter / persistent buffer
#      都不能新增，否则旧模型目录全部 missing keys。所有池层共享同一份
#      KDAAttention **引用**（放在普通 dict 里绕开 __setattr__ 注册），
#      只各自持有独立状态。锚点是 persistent=False buffer（随 .to() 迁移、
#      不进 state_dict）。启用/关闭、扩展前后 state_dict 完全一致。
#   3) 锚点在**首次 forward 惰性计算**：建模期 wte 仍是随机初始化，那时算
#      「代表性文本 Embedding 均值」没有意义；build_model 是先构造再
#      load_state_dict，必须等权重加载完第一次前向才算。
#   4) H_t 取 H_1 在 (B,T) 上的均值而不是逐 token：池状态按整个输入序列
#      推进，逐 token 路由会让同一序列写进不同层，破坏「未召回则不遗忘」。
#   5) 「全异步 / 静默激活」落地为**会话级 silent 开关**：路由与状态写入
#      照常执行（后台实时写入），但 ΔH 不注入输出。所有读写仍在同一个
#      串行推理进程内完成——规格自己提醒过并发写冲突，不做跨线程写。
#
# ==== 三条不可破的契约 ====
#   · 读写隔离：未被 Router 选中的层绝不调用 forward，S_i 不被污染。
#   · 单一出口：无层激活 → 直接返回 H_1（不分配零张量，退化为标准模型）。
#   · 无训练：整段 torch.no_grad()，ΔH 不回传梯度，主残差梯度保留。
#
# 超参唯一真源 = spec 4b 节；本文件不硬编码任何阈值/维度。
# ===========================================================================
from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import spec


def _unit(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """L2 归一化（eps 内置，全零向量安全返回全零）。"""
    return F.normalize(x, p=2, dim=dim)


def _check_hidden(H: torch.Tensor, dim: int) -> None:
    if H.ndim != 3:
        raise RuntimeError(f"MoMW：H 必须是 [Batch, Seq, Dim]，收到 {tuple(H.shape)}")
    if H.shape[-1] != dim:
        raise RuntimeError(
            f"MoMW：H 末维 {H.shape[-1]} != d_model {dim}，"
            "锚点/E_i/H_t 必须在同一隐空间")


# ===========================================================================
# 规格接口 1：FastWeightLayer —— 黑盒 KDA 记忆层
# ===========================================================================
class FastWeightLayer(nn.Module):
    """一个 MW 层：共享的 KDA 读出函数 + 一份**独立**持久状态 S_i。

    零 nn.Parameter、零 persistent buffer。`_refs` / `_rt` 都是普通 dict，
    故意绕开 nn.Module.__setattr__ 的模块/buffer 注册：kda 是
    Gargantua.fast_weight.layer 的引用，注册进去会让 state_dict 多出一整份
    共享 storage 的权重（strict load 炸、safetensors 也拒）。

    `self._rt` 里全是会话级运行时数据（不落盘）：
      state      (S [B,H,dk,dv], conv_cache dict) | None —— None 即 S 全零
      eig_v      [H,dv] 幂迭代向量（随机初始化，跨步增量更新）
      E          [D] 状态特征向量（层「实际在记什么」），首次幂迭代后填充
      P_override [D] | None 扩展派生锚点（继承父层 E_i）
      writes / bonus_age / prev_norm / growth
    """

    def __init__(self, kda: nn.Module, cfg: dict, type_idx: int = 0,
                 anchor_override: Optional[torch.Tensor] = None,
                 seed: int = 0):
        super().__init__()
        self._refs = {"kda": kda}
        self.cfg = dict(cfg)
        self.type_idx = int(type_idx) % len(spec.MOMW_ANCHOR_GROUPS)

        w_o = kda.w_o.weight                  # [D, H*dv]
        D, val_dim = int(w_o.shape[0]), int(w_o.shape[1])
        H = int(kda.H)
        if val_dim % H:
            raise RuntimeError(
                f"MoMW：w_o 输入宽度 {val_dim} 不能被 head 数 {H} 整除")
        dv = val_dim // H
        self.D, self.H, self.dv = D, H, dv

        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        self._rt: dict = {
            "state": None,
            "eig_v": _unit(torch.randn(H, dv, generator=g), dim=-1),
            "E": None,
            "P_override": (anchor_override.detach().clone()
                           if anchor_override is not None else None),
            "P_fallback": None,
            "writes": 0,
            "bonus_age": 0,
            "prev_norm": 0.0,
            "growth": 0.0,
            "seed": int(seed),
        }

    # ---- 规格要求的状态读取接口 ----
    @property
    def S(self) -> Optional[torch.Tensor]:
        return None if self._rt["state"] is None else self._rt["state"][0]

    @property
    def type_name(self) -> str:
        return spec.MOMW_ANCHOR_GROUPS[self.type_idx]

    @property
    def gamma(self) -> float:
        """混合系数 γ_i = 1 - exp(-writes / k)，随写入次数单调递增 [0,1)。"""
        k = max(1.0, float(self.cfg["gamma_k"]))
        return 1.0 - math.exp(-float(self._rt["writes"]) / k)

    def get_norm(self) -> float:
        """状态矩阵 ‖S‖_F（对 [dk,dv] 求 Frobenius，再对 B/H 平均）。

        torch.norm(..., p='fro') 只收 2D，而 S 是 [B,H,dk,dv]——按矩阵
        范数定义展开，语义与规格的 ‖S‖ 一致。
        """
        S = self.S
        if S is None:
            return 0.0
        return float(S.detach().float().pow(2).sum(dim=(-2, -1))
                     .sqrt().mean().item())

    def saturation(self) -> float:
        """v1 兼容别名。"""
        return self.get_norm()

    def get_eigenvector(self) -> torch.Tensor:
        """状态主特征向量 E_i ∈ R^D（隐空间）。尚未积累写入时返回零向量。"""
        E = self._rt["E"]
        if E is None:
            return torch.zeros(self.D, device=self._eig_device(),
                               dtype=self._eig_dtype())
        return E

    def _eig_device(self):
        return self._rt["eig_v"].device

    def _eig_dtype(self):
        return self._rt["eig_v"].dtype

    def anchor(self, anchors: Optional[torch.Tensor]) -> torch.Tensor:
        """该层当前锚点：扩展派生锚点优先，否则取类别原型；无原型则降级。"""
        if self._rt["P_override"] is not None:
            return self._rt["P_override"]
        if anchors is not None:
            return anchors[self.type_idx % anchors.shape[0]]
        if self._rt["P_fallback"] is None:
            g = torch.Generator(device="cpu")
            g.manual_seed(spec.MOMW_ANCHOR_SEED + 104729 * self.type_idx)
            self._rt["P_fallback"] = _unit(
                torch.randn(self.D, generator=g), dim=-1)
        return self._rt["P_fallback"]

    def bonus(self, B0: float, M: int) -> float:
        """阶段三偏置 B0·e^{-age/M}，只在新建后的前 M 步生效。"""
        age = int(self._rt["bonus_age"])
        m = max(1, int(M))
        return float(B0) * math.exp(-age / m) if age < m else 0.0

    def tick_bonus(self) -> None:
        self._rt["bonus_age"] += 1

    # ---- 规格要求的前向：边读边写 ----
    def forward(self, H_in: torch.Tensor) -> torch.Tensor:
        """被召回才调用：先以旧状态读出，再按门控 delta 规则写入并推进。

        调用方整体在 torch.no_grad() 下（无训练阀门）。状态按
        FastWeightBank 的口径 detach，避免跨步递推图越攒越长。
        """
        _check_hidden(H_in, self.D)
        kda = self._refs["kda"]
        st = self._rt["state"]
        S, cc = st if st is not None else (None, None)
        if S is not None and (S.shape[0] != H_in.shape[0]
                              or S.device != H_in.device):
            S, cc = None, None                  # batch / 设备变化则丢弃旧状态
        if S is not None and S.dtype != H_in.dtype:
            S = S.to(H_in.dtype)                # apply_precision 后 dtype 对齐
            cc = ({k: (v.to(H_in.dtype) if v is not None else None)
                   for k, v in cc.items()} if cc else cc)
        out, new_S, new_cc = kda(H_in, state=S, conv_cache=cc)
        if new_S.ndim != 4:
            raise RuntimeError(
                f"MoMW：KDA 状态应为 [B,H,dk,dv]，收到 {tuple(new_S.shape)}")
        self._rt["state"] = (
            new_S.detach(),
            {k: (v.detach() if v is not None else None)
             for k, v in new_cc.items()},
        )
        self._rt["writes"] += 1
        norm = self.get_norm()
        self._rt["growth"] = norm - self._rt["prev_norm"]
        self._rt["prev_norm"] = norm
        # 增量幂迭代：每 eigen_interval 次写入做【一步】，不做全量 SVD
        interval = max(1, int(self.cfg["eigen_interval"]))
        if self._rt["writes"] % interval == 0:
            self._power_iteration()
        return out

    def _power_iteration(self) -> None:
        """一步幂迭代提取 SᵀS 的主特征向量，再投影回隐空间得到 E_i。

        G_h = S_hᵀ S_h ∈ R^{dv×dv}；v_h ← normalize(G_h v_h)。
        把 [H,dv] 展平成 [H*dv] 后过 KDA 的 w_o（[D,H*dv]）→ E_i ∈ R^D。
        每步只做 H 次 dv×dv 的矩阵-向量乘，代价固定且远小于 SVD。
        """
        S = self.S
        if S is None:
            return
        S0 = S[0].detach().float()              # [H,dk,dv]
        G = torch.einsum('hkv,hkl->hvl', S0, S0)   # [H,dv,dv]
        v = _unit(self._rt["eig_v"].to(S0.device, S0.dtype), dim=-1)
        v = _unit(torch.einsum('hvl,hl->hv', G, v), dim=-1)
        self._rt["eig_v"] = v.detach()
        w_o = self._refs["kda"].w_o.weight    # [D, H*dv]
        flat = v.reshape(-1)                   # [H*dv]
        if w_o.shape[1] != flat.numel():
            raise RuntimeError(
                f"MoMW：w_o 输入宽度 {w_o.shape[1]} != H*dv {flat.numel()}，"
                "KDA 参数化已变，需同步幂迭代投影")
        E = F.linear(flat.to(w_o.dtype), w_o)  # [D]
        self._rt["E"] = _unit(E, dim=-1).detach()

    def describe(self) -> dict:
        return {
            "type": self.type_name,
            "gamma": round(self.gamma, 4),
            "writes": int(self._rt["writes"]),
            "norm": round(self.get_norm(), 4),
            "growth": round(float(self._rt["growth"]), 4),
            "bonus_age": int(self._rt["bonus_age"]),
            "has_state": self.S is not None,
            "derived_anchor": self._rt["P_override"] is not None,
        }


# v1 调用方兼容别名（方法签名按 v2，测试已同步重写）
MoMWLayer = FastWeightLayer


# ===========================================================================
# 规格接口 2：MoMWRouter —— 三阶段混合打分（无任何训练参数）
# ===========================================================================
class MoMWRouter(nn.Module):
    """零 nn.Parameter：全部依赖余弦相似度 / 动态阈值 / 张量运算。

    打分分两层：
      raw_i    = (1-γ_i)·cos(H_t,P_i) + γ_i·cos(H_t,E_i)   ← 语义分
      scores_i = raw_i + balance_i + bonus_i                ← 实际路由分
    raw 用于饥饿/统计（bonus 不该掩盖「确实缺记忆」），scores 用于 top-N。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.last_raw: List[float] = []
        self.last_scores: List[float] = []
        self.last_cos_anchor: List[float] = []
        self.last_cos_state: List[float] = []
        self.last_gamma: List[float] = []

    def score_all(self, H_t: torch.Tensor,
                  layers: Sequence[FastWeightLayer],
                  anchors: Optional[torch.Tensor], cfg: dict
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (raw, scores)，各 [L]。维度检查集中在入口做一次。"""
        _check_hidden(H_t, self.dim)
        if not layers:
            z = H_t.new_zeros(0)
            self.last_raw = self.last_scores = []
            self.last_cos_anchor = self.last_cos_state = self.last_gamma = []
            return z, z
        A = (anchors.to(H_t.device, H_t.dtype)
             if anchors is not None else None)
        if A is not None and (A.ndim != 2 or A.shape[1] != self.dim):
            raise RuntimeError(
                f"MoMW：锚点应为 [K,{self.dim}]，收到 {tuple(A.shape)}")

        # 序列级摘要：路由判断「当前输入整体属于哪类记忆」
        h = _unit(H_t.detach().float().mean(dim=(0, 1)), dim=-1)   # [D]

        # 同类层软均衡：范数低于同类均值的层加分（余量更大）
        norms = [l.get_norm() for l in layers]
        by_type: Dict[int, List[float]] = {}
        for l, n in zip(layers, norms):
            by_type.setdefault(l.type_idx, []).append(n)
        mean_norm = {k: sum(v) / len(v) for k, v in by_type.items()}
        lam = float(cfg["load_lambda"])

        raw, scores = [], []
        cP_all, cE_all, g_all = [], [], []
        for l, n in zip(layers, norms):
            P = _unit(l.anchor(A).to(h.device, h.dtype).float(), dim=-1)
            E = _unit(l.get_eigenvector().to(h.device, h.dtype).float(),
                      dim=-1)
            cP = torch.dot(h, P)
            cE = torch.dot(h, E)
            g = l.gamma
            base = (1.0 - g) * cP + g * cE
            balance = 0.0
            if len(by_type[l.type_idx]) >= 2:
                mu = mean_norm[l.type_idx]
                balance = lam * (mu - n) / (mu + 1e-6)
            bonus = l.bonus(cfg["B0"], cfg["M"])
            raw.append(base)
            scores.append(base + balance + bonus)
            cP_all.append(float(cP.item()))
            cE_all.append(float(cE.item()))
            g_all.append(round(g, 4))
        raw_t, scores_t = torch.stack(raw), torch.stack(scores)
        self.last_raw = [float(v) for v in raw_t.detach().tolist()]
        self.last_scores = [float(v) for v in scores_t.detach().tolist()]
        self.last_cos_anchor = cP_all
        self.last_cos_state = cE_all
        self.last_gamma = g_all
        return raw_t, scores_t

    def route(self, scores: torch.Tensor, N: int, tau: float,
              threshold: float
              ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """阈值过滤 → top-N → softmax(scores/tau)。"""
        if scores.numel() == 0 or float(scores.max().item()) < threshold:
            return None, None
        k = min(int(N), int(scores.numel()))    # 层数可能 < N，topk 会报错
        top_scores, indices = torch.topk(scores, k)
        return indices, torch.softmax(top_scores / tau, dim=-1)

    def forward(self, H_t: torch.Tensor, pool, N: int, tau: float,
                threshold: float
                ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """规格接口。(MoMW.forward 用拆开的 score_all/route 以复用 raw。)"""
        _raw, scores = self.score_all(H_t, pool.kda_layers, pool.anchors,
                                      pool.cfg)
        return self.route(scores, N, tau, threshold)


Router = MoMWRouter   # v1 兼容别名


# ===========================================================================
# 规格接口 3：MoMWPool —— 读写隔离 + 空层分裂 + 负载均衡
# ===========================================================================
class MoMWPool(nn.Module):
    """记忆层池。`kda_layers` 是 nn.ModuleList，但每个层都零参数零
    persistent buffer，所以 state_dict 一个 key 都不增加；运行时 append
    新层也不会改变权重键集合（动态扩展不落盘）。

    `anchors` / `cfg` / `coverage` 都是普通属性，不注册为 buffer。
    池对象本身按会话存放在外部 state["momw"]（与 fast weight 同一口径），
    因此不同会话的池状态天然隔离。
    """

    def __init__(self, kda: nn.Module, cfg: dict,
                 anchors: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = dict(cfg)
        self._refs = {"kda": kda}
        self.anchors = anchors
        self.kda_layers = nn.ModuleList()
        self.coverage = deque(maxlen=max(1, int(cfg["M"])))
        self.last_indices: List[int] = []
        self.expansions = 0
        self._serial = 0
        self._cooldown = 0
        for i in range(int(cfg["initial_layers"])):
            self._append_layer(i % len(spec.MOMW_ANCHOR_GROUPS))

    @property
    def layers(self) -> List[FastWeightLayer]:
        """v1 / Router 兼容：把 ModuleList 当普通序列用。"""
        return list(self.kda_layers)

    # ---- 阶段三：空层分裂 ----
    def _append_layer(self, type_idx: int,
                      anchor_override: Optional[torch.Tensor] = None) -> bool:
        if len(self.kda_layers) >= int(self.cfg["max_layers"]):
            return False
        seed = spec.MOMW_ANCHOR_SEED + 7919 * self._serial + type_idx
        self._serial += 1
        # **绝不 clone 原层状态**：S_new=None（全零）、eig_v 重新随机、
        # writes=0 / γ=0 / bonus_age=0，全部从零开始。
        self.kda_layers.append(FastWeightLayer(
            self._refs["kda"], self.cfg, type_idx, anchor_override, seed))
        self.expansions += 1
        return True

    def _derive_anchor(self, parent: FastWeightLayer
                       ) -> Optional[torch.Tensor]:
        """派生 P_new：继承父层当前 E_i（已积累语义）或退回初始类别锚点。

        只复制【锚点方向】，不复制任何状态 / 权重——规格禁止的是后者
        （叠加会干扰原层技能），锚点继承是规格明确要求的。
        """
        if (int(parent._rt["writes"]) >= int(self.cfg["eigen_min_writes"])
                and parent._rt["E"] is not None):
            return parent._rt["E"].detach().clone()
        return None      # None = 用 A[type_idx]（初始锚点）

    def maybe_expand(self, indices: Optional[torch.Tensor],
                     raw: Optional[torch.Tensor] = None) -> bool:
        """阶段三触发：被召回层 ‖S‖ 增长过快 / 绝对范数超限 → 分裂空层。

        附带 v1 兼容的饥饿窗口（starve_threshold>0 时启用）。两道防呆：
        M 步冷却（bonus 期间不得连续扩展）+ max_layers 硬上限。
        """
        c = self.cfg
        # 冷却计数器每步递减；触发成功后重置为 M
        if self._cooldown > 0:
            self._cooldown -= 1
        overloaded, parent = False, None
        if indices is not None and indices.numel():
            for i in indices:
                l = self.kda_layers[int(i.item())]
                if (float(l._rt["growth"]) > float(c["expand_delta"])
                        or l.get_norm() > float(c["saturation_limit"])):
                    overloaded, parent = True, l
                    break
        starved = False
        if raw is not None and float(c["starve_threshold"]) > 0.0:
            self.coverage.append(float(raw.max().item())
                                 if raw.numel() else 0.0)
            M = max(1, int(c["M"]))
            starved = (len(self.coverage) >= M
                       and sum(self.coverage) / len(self.coverage)
                       < float(c["starve_threshold"]))
        if not (overloaded or starved) or self._cooldown > 0:
            return False
        ok = self._append_layer(
            parent.type_idx if parent is not None else 0,
            self._derive_anchor(parent) if parent is not None else None)
        if ok:
            self._cooldown = max(1, int(c["M"]))
            self.coverage.clear()
        return ok

    # ---- Pool 前向：只计算被召回的层 ----
    def forward(self, H_t: torch.Tensor, indices: Optional[torch.Tensor],
                alpha: Optional[torch.Tensor]):
        """无激活返回 0.0（规格口径）；MoMW.forward 据此直接返回 H_t，
        不分配零张量。未被选中的层**完全不调用**，S_i 不被污染。"""
        if indices is None:
            self.last_indices = []
            return 0.0
        delta_H = torch.zeros_like(H_t)
        for idx_t, weight in zip(indices, alpha):
            out = self.kda_layers[int(idx_t.item())].forward(H_t)
            delta_H = delta_H + weight * out
        self.last_indices = [int(i.item()) for i in indices]
        return delta_H

    def tick_bonus(self) -> None:
        for l in self.kda_layers:
            l.tick_bonus()

    def describe(self) -> dict:
        active = set(self.last_indices)
        return {
            "layers": [dict(l.describe(), active=(i in active))
                       for i, l in enumerate(self.kda_layers)],
            "layer_count": len(self.kda_layers),
            "expansions": self.expansions,
            "last_indices": list(self.last_indices),
            "coverage": [round(v, 6) for v in self.coverage],
            "types": list(spec.MOMW_ANCHOR_GROUPS),
        }


# ===========================================================================
# 规格接口 4：MoMW —— 整体组装
# ===========================================================================
class MoMW(nn.Module):
    """插在第一个 Block 之后、与 fast weight 并列的可选记忆阀门。

        H_out = H_1 + ΔH；无层激活 或 silent → 直接返回 H_1。

    状态归属（与 fast weight 完全同口径）：
      state=None   → 内部 self.pool（训练 / 无状态路径；reset() 复位）
      state=dict   → state["momw"] 的会话池（跨轮持久、会话隔离）
    """

    STATE_KEY = "momw"
    SILENT_KEY = "momw_silent"

    def __init__(self, kda: nn.Module, cfg: Optional[dict] = None,
                 wte: Optional[nn.Module] = None, dim: Optional[int] = None,
                 N: Optional[int] = None, tau: Optional[float] = None,
                 threshold: Optional[float] = None):
        super().__init__()
        merged = spec.momw_default_config()
        if cfg:
            merged.update({k: v for k, v in cfg.items() if k in merged})
        # 兼容规格的构造签名 MoMW(dim, N, tau, threshold)
        if N is not None:
            merged["N"] = int(N)
        if tau is not None:
            merged["tau"] = float(tau)
        if threshold is not None:
            merged["threshold"] = float(threshold)
        self.cfg = merged

        D = int(kda.w_o.weight.shape[0])
        if dim is not None and int(dim) != D:
            raise RuntimeError(
                f"MoMW：dim={dim} 与模型 d_model={D} 不一致")
        self.dim = D
        self._refs = {"kda": kda, "wte": wte}
        # 锚点是外部原型，惰性首次 forward 计算。persistent=False ⇒ 不进
        # state_dict（strict load 契约），但会随 model.to(dtype/device) 迁移。
        K = len(spec.MOMW_ANCHOR_GROUPS)
        self.register_buffer("_anchors", torch.zeros(K, D),
                             persistent=False)
        self._anchors_ready = False
        self._anchor_provider = None
        self.router = MoMWRouter(D)
        self.pool = MoMWPool(kda, merged)   # 注册为子模块（零参数）
        self._local = self.pool             # 内部池别名
        self.silent = False                 # 进程级静默（会话级优先）
        self.steps = 0
        self.last_raw: List[float] = []
        self.last_scores: List[float] = []
        self.last_indices: List[int] = []

    # ---- 阶段一：锚点播种 ----
    def set_anchor_provider(self, fn) -> None:
        """注入真正的「类别代表性文本 Embedding 均值」实现。

        fn(K, D) -> Tensor[K, D]。不注入时默认用 wte.weight 中确定性采样
        的代表性 token 行均值（零硬编码 ID、与模型词表天然一致）；
        连 wte 都没有时降级为确定性随机单位原型。设置后下次 forward 生效。
        """
        self._anchor_provider = fn
        self._anchors_ready = False

    def _ensure_anchors(self, H: torch.Tensor) -> None:
        if self._anchors_ready:
            return
        K, D = self._anchors.shape
        with torch.no_grad():
            A = torch.zeros(K, D, device=H.device, dtype=torch.float32)
            provider, wte = self._anchor_provider, self._refs.get("wte")
            done = False
            if provider is not None:
                try:
                    got = provider(K, D)
                    if isinstance(got, torch.Tensor) and got.shape == (K, D):
                        A = got.detach().to(A.device, A.dtype)
                        done = True
                except Exception:
                    done = False          # 注入实现失败 → 静默降级
            if not done and wte is not None:
                W = wte.weight.detach().float()
                V = int(W.shape[0])
                sample = max(1, min(int(self.cfg["anchor_sample"]), V))
                for k in range(K):
                    g = torch.Generator(device="cpu")
                    g.manual_seed(spec.MOMW_ANCHOR_SEED + 7919 * k)
                    idx = torch.randint(0, V, (sample,), generator=g)
                    A[k] = W[idx].mean(0)
                done = True
            if not done:
                for k in range(K):
                    g = torch.Generator(device="cpu")
                    g.manual_seed(spec.MOMW_ANCHOR_SEED + 7919 * k)
                    A[k] = torch.randn(D, generator=g)
            A = _unit(A, dim=-1)
        self._anchors.copy_(A.to(self._anchors.device, self._anchors.dtype))
        self._anchors_ready = True

    # ---- 状态复位 / 会话取池 ----
    def _make_pool(self) -> MoMWPool:
        return MoMWPool(self._refs["kda"], self.cfg, self._anchors)

    def reset(self) -> None:
        """样本边界复位（只清内部池，外部会话状态由调用方持有）。"""
        self._local = self._make_pool()
        self.pool = self._local
        self.steps = 0
        self.last_raw, self.last_scores, self.last_indices = [], [], []

    def _pool_for(self, state: Optional[dict]) -> MoMWPool:
        if state is None:
            return self._local
        pool = state.get(self.STATE_KEY)
        if not isinstance(pool, MoMWPool):
            pool = self._make_pool()
            state[self.STATE_KEY] = pool
        return pool

    def forward(self, H_1: torch.Tensor,
                state: Optional[dict] = None) -> torch.Tensor:
        self._ensure_anchors(H_1)
        pool = self._pool_for(state)
        pool.anchors = self._anchors          # 会话池同步到最新锚点
        c = self.cfg
        silent = (bool(state.get(self.SILENT_KEY, False))
                  if state is not None else self.silent)
        # 无训练：整段不建图，ΔH 不回传梯度；主残差路径梯度照常保留
        with torch.no_grad():
            raw, scores = self.router.score_all(
                H_1, pool.kda_layers, self._anchors, c)
            indices, alpha = self.router.route(
                scores, c["N"], c["tau"], c["threshold"])
            delta_H = pool.forward(H_1, indices, alpha)
            # 先 tick 既有层、再扩展：新层必须拿到 age=0 的完整 B0，
            # 否则 M=1 时 bonus 诞生即归零 → 零状态层永远不被召回。
            pool.tick_bonus()
            pool.maybe_expand(indices, raw)
        self.steps += 1
        self.last_raw = list(self.router.last_raw)
        self.last_scores = list(self.router.last_scores)
        self.last_indices = pool.last_indices
        if indices is None or silent:
            return H_1                        # 单一出口 / 静默激活
        return H_1 + delta_H

    def stats(self, state: Optional[dict] = None) -> dict:
        """前端面板 / 阈值标定用的纯 Python 数值快照（不落盘）。"""
        pool = self._pool_for(state)
        out = pool.describe()
        r = self.router
        out.update({
            "enabled": True,
            "silent": (bool(state.get(self.SILENT_KEY, False))
                       if state is not None else self.silent),
            "steps": self.steps,
            "dim": self.dim,
            "anchors_ready": self._anchors_ready,
            "last_raw": [round(v, 6) for v in self.last_raw],
            "last_scores": [round(v, 6) for v in self.last_scores],
            "last_cos_anchor": [round(v, 6) for v in r.last_cos_anchor],
            "last_cos_state": [round(v, 6) for v in r.last_cos_state],
            "last_gamma": list(r.last_gamma),
            "config": dict(self.cfg),
        })
        return out
