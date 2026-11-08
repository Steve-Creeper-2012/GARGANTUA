# ===========================================================================
# GARGANTUA v1 — server/momw.py：MoMW 三阶段无训练记忆阀门（v3）
# ---------------------------------------------------------------------------
# 规格来源：steve 2026-11-08 的三轮提示词。v3 在 v2 基础上并入六项改进：
#   ① 阈值改为**运行时分位数**（滑动窗口 + 可配置分位位置），离线校准
#      由 MoMW.calibrate() 提供（跑种子输入、记分布、取低尾分位）；
#   ② 锚点支持**真文本 Embedding**：用模型自己的词表副本编码
#      spec.MOMW_ANCHOR_TEXTS，再取 wte 均值 → 不依赖任何硬编码 token ID
#      （版本升级时 ID 整体位移也不影响）；γ 让锚点随写入量退居次席；
#   ③ 主方向跟踪改为 **Oja 规则**：每步 O(H·dk·dv) 的向量运算，
#      不再「每 N 步走一步幂迭代」；
#   ④ 状态可持久化到 models/NAME/momw_state/（见 momw_state.py），
#      不进权重契约；启动时按快照重建，缺失/损坏从零开始；
#   ⑤ cost-aware 门控：fast weight 读出能量高时按 exp(-c·E_fw) 收缩 ΔH；
#   ⑥ 可选**异步写**：单写者线程串行提交状态，主路径只读快照，flush() 兜底。
#
# 与 v1/v2 的硬约束一致（不能破）：
#   · 零 nn.Parameter / 零 persistent buffer —— 启用与否、扩展前后
#     state_dict 完全一致（build_model 是 strict load）。
#   · 读写隔离：未召回层完全不 forward，S_i 不被污染。
#   · 单一出口：无层激活 → 返回 H_1；silent → 照常写入但不注入。
#   · 无训练：整段 torch.no_grad()，ΔH 不回传梯度，主残差梯度保留。
#
# 超参唯一真源 = spec 4b 节；本文件不硬编码任何阈值/维度。
# ===========================================================================
from __future__ import annotations

import math
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import spec


def _unit(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim)


def _median(v: List[float]) -> float:
    """偶数个取中间两数的平均（`peers[n//2]` 会取上中位数，单层对不齐）。"""
    s = sorted(v)
    n = len(s)
    if not n:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _check_hidden(H: torch.Tensor, dim: int) -> None:
    if H.ndim != 3:
        raise RuntimeError(
            f"MoMW：H 必须是 [Batch, Seq, Dim]，收到 {tuple(H.shape)}")
    if H.shape[-1] != dim:
        raise RuntimeError(
            f"MoMW：H 末维 {H.shape[-1]} != d_model {dim}，"
            "锚点/E_i/H_t 必须在同一隐空间")


# ===========================================================================
# 异步写：单写者（串行提交，主路径只读快照）
# ===========================================================================
class _StateWriter:
    """单 worker 线程池：同一时刻只有一个写者，天然无并发写冲突。

    主线程只把 `(layer, pending_state)` 丢进来就返回；`flush()` 在会话结束 /
    落盘 / 切换模型前强制等待，保证持久化看到的是最终状态。
    `max_workers=1` 是语义核心，不是性能参数。
    """

    def __init__(self, max_pending: int = 256):
        self._ex = ThreadPoolExecutor(max_workers=1,
                                      thread_name_prefix="momw-writer")
        self._futs: deque = deque(maxlen=max_pending)
        self._closed = False

    def submit(self, layer, pending) -> None:
        if self._closed:
            layer.commit_state(pending)
            return
        self._futs.append(self._ex.submit(self._apply, layer, pending))

    @staticmethod
    def _apply(layer, pending) -> None:
        try:
            with torch.no_grad():
                layer.commit_state(pending)
        except Exception:
            pass          # 写者线程绝不能把主推理路径带崩

    def flush(self, timeout: float = 10.0) -> int:
        n = 0
        while self._futs:
            f = self._futs.popleft()
            n += 1
            try:
                f.result(timeout=timeout)
            except Exception:
                pass
        return n

    def pending(self) -> int:
        return len(self._futs)

    def close(self) -> None:
        self.flush()
        self._closed = True
        try:
            self._ex.shutdown(wait=True)
        except Exception:
            pass


# ===========================================================================
# 规格接口 1：FastWeightLayer —— 黑盒 KDA 记忆层
# ===========================================================================
class FastWeightLayer(nn.Module):
    """一个 MW 层：共享的 KDA 读出函数 + 一份**独立**持久状态 S_i。

    零参数零 persistent buffer；`_refs` / `_rt` 都是普通 dict，绕开
    nn.Module.__setattr__ 的模块/buffer 注册（否则 state_dict 会多出一整份
    共享 storage 的权重，strict load 与 safetensors 都不允许）。

    `_rt["eig_v"]` 是 Oja 规则在线跟踪的主方向（每步更新），`_rt["E"]`
    是它经 w_o 投影回 d_model 的结果；两者都是运行时数据。
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
        self.D, self.H, self.dv = D, H, val_dim // H

        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        self._rt: dict = {
            "state": None,
            "eig_v": _unit(torch.randn(H, self.dv, generator=g), dim=-1),
            "E": None,
            "P_override": (anchor_override.detach().clone()
                           if anchor_override is not None else None),
            "P_fallback": None,
            "writes": 0,
            "bonus_age": 0,
            "prev_norm": 0.0,
            "growth": 0.0,
            "oja_steps": 0,
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
        """γ_i = 1 - exp(-writes / k)，单调递增到 1。"""
        k = max(1.0, float(self.cfg["gamma_k"]))
        return 1.0 - math.exp(-float(self._rt["writes"]) / k)

    def get_norm(self) -> float:
        """‖S‖_F（对 [dk,dv] 求 Frobenius，再对 B/H 平均）。"""
        S = self.S
        if S is None:
            return 0.0
        return float(S.detach().float().pow(2).sum(dim=(-2, -1))
                     .sqrt().mean().item())

    def saturation(self) -> float:
        return self.get_norm()

    def get_eigenvector(self) -> torch.Tensor:
        E = self._rt["E"]
        if E is None:
            return torch.zeros(self.D, dtype=self._rt["eig_v"].dtype)
        return E

    def anchor(self, anchors: Optional[torch.Tensor]) -> torch.Tensor:
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
        age = int(self._rt["bonus_age"])
        m = max(1, int(M))
        return float(B0) * math.exp(-age / m) if age < m else 0.0

    def tick_bonus(self) -> None:
        self._rt["bonus_age"] += 1

    # ---- 前向：只算 out / pending，提交时机由调用方决定 ----
    def forward(self, H_in: torch.Tensor, commit: bool = True
                ) -> Tuple[torch.Tensor, tuple]:
        _check_hidden(H_in, self.D)
        kda = self._refs["kda"]
        st = self._rt["state"]
        S, cc = st if st is not None else (None, None)
        if S is not None and (S.shape[0] != H_in.shape[0]
                              or S.device != H_in.device):
            S, cc = None, None
        if S is not None and S.dtype != H_in.dtype:
            S = S.to(H_in.dtype)
            cc = ({k: (v.to(H_in.dtype) if v is not None else None)
                   for k, v in cc.items()} if cc else cc)
        out, new_S, new_cc = kda(H_in, state=S, conv_cache=cc)
        if new_S.ndim != 4:
            raise RuntimeError(
                f"MoMW：KDA 状态应为 [B,H,dk,dv]，收到 {tuple(new_S.shape)}")
        pending = (new_S.detach(),
                   {k: (v.detach() if v is not None else None)
                    for k, v in new_cc.items()})
        if commit:
            self.commit_state(pending)
        return out, pending

    def commit_state(self, pending: tuple) -> None:
        """写入状态 → 更新计数/增长 → 一步 Oja 主方向跟踪。

        调用方保证串行（同一层不会有两个并发 commit）：同步模式由
        forward(commit=True) 直接调，异步模式由单写者线程调。
        """
        self._rt["state"] = pending
        self._rt["writes"] += 1
        norm = self.get_norm()
        self._rt["growth"] = norm - self._rt["prev_norm"]
        self._rt["prev_norm"] = norm
        self._oja_step()

    def _oja_step(self) -> None:
        """Oja 规则：v ← normalize(v + η·(G v − (vᵀG v)·v))，G = S_hᵀS_h。

        等价于把 S_h 的每一行当一个样本做 streaming PCA：每步只做两次
        [H,dk,dv]×[H,dv] 的收缩，代价固定、没有周期性矩阵-向量乘。
        """
        S = self.S
        if S is None:
            return
        eta = float(self.cfg["oja_eta"])
        S0 = S[0].detach().float()                  # [H,dk,dv]
        # 尺度归一化：Oja 的稳定步长受 λ_max(SᵀS) 约束，而各层 ‖S‖ 差几个
        # 数量级。先把每头 S 归一到单位 Frobenius 范数，学习率就与层尺度
        # 解耦（否则同一个 eta 在一层收敛、在另一层过冲发散）。
        scale = (S0.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt() + 1e-8)
        Sn = S0 / scale
        v = _unit(self._rt["eig_v"].to(S0.device, S0.dtype), dim=-1)
        y = torch.einsum('hkv,hv->hk', Sn, v)       # [H,dk] = S v
        gv = torch.einsum('hkv,hk->hv', Sn, y)      # [H,dv] = SᵀS v
        if self._rt["oja_steps"] == 0:
            # 冷启动：先做**一次**幂迭代把 v 拉到主方向附近。从随机方向出发
            # 时 v·e0 ≈ 1/√dv，Oja 的线性项太小，要几百步才爬出来；一次
            # 幂迭代把冷启动从 O(1/η·√dv) 压到一步，之后仍是每步 O(d) 的
            # Oja 跟踪（这才是用户要的「每步更新、代价固定」）。
            v = _unit(gv, dim=-1)
            y = torch.einsum('hkv,hv->hk', Sn, v)
            gv = torch.einsum('hkv,hk->hv', Sn, y)
        vy = (y * y).sum(dim=-1, keepdim=True)      # [H,1]  = vᵀSᵀS v
        v = _unit(v + eta * (gv - vy * v), dim=-1)
        self._rt["eig_v"] = v.detach()
        self._rt["oja_steps"] += 1
        w_o = self._refs["kda"].w_o.weight           # [D, H*dv]
        flat = v.reshape(-1)
        if w_o.shape[1] != flat.numel():
            raise RuntimeError(
                f"MoMW：w_o 输入宽度 {w_o.shape[1]} != H*dv {flat.numel()}")
        E = F.linear(flat.to(w_o.dtype), w_o)
        self._rt["E"] = _unit(E, dim=-1).detach()

    # ---- 快照 / 克隆 ----
    def clone(self) -> "FastWeightLayer":
        new = FastWeightLayer(self._refs["kda"], self.cfg, self.type_idx,
                              seed=self._rt["seed"])
        st = self._rt["state"]
        new._rt.update({
            "state": ((st[0].clone(),
                       {k: (v.clone() if v is not None else None)
                        for k, v in (st[1] or {}).items()})
                      if st is not None else None),
            "eig_v": self._rt["eig_v"].clone(),
            "E": (self._rt["E"].clone()
                  if self._rt["E"] is not None else None),
            "P_override": (self._rt["P_override"].clone()
                           if self._rt["P_override"] is not None else None),
            "writes": int(self._rt["writes"]),
            "bonus_age": int(self._rt["bonus_age"]),
            "prev_norm": float(self._rt["prev_norm"]),
            "growth": float(self._rt["growth"]),
            "oja_steps": int(self._rt["oja_steps"]),
        })
        return new

    def restore(self, desc: dict, tensors: dict, index: int,
                device=None, dtype=None) -> None:
        """从 momw_state 快照恢复一层（张量缺失按零状态处理）。"""
        def _t(key):
            v = tensors.get(key)
            if v is None:
                return None
            return v.to(device=device, dtype=dtype)

        S = _t(f"L{index}.S")
        if S is not None:
            cc = {}
            prefix = f"L{index}.cc."
            for k, v in tensors.items():
                if k.startswith(prefix):
                    cc[k[len(prefix):]] = v.to(device=device, dtype=dtype)
            self._rt["state"] = (S, cc)
        else:
            self._rt["state"] = None
        v = _t(f"L{index}.v")
        if v is not None:
            self._rt["eig_v"] = v
        self._rt["E"] = _t(f"L{index}.E")
        self._rt["P_override"] = _t(f"L{index}.P")
        self._rt["writes"] = int(desc.get("writes", 0))
        self._rt["bonus_age"] = int(desc.get("bonus_age", 0))
        self._rt["prev_norm"] = float(desc.get("prev_norm", 0.0))
        self._rt["growth"] = float(desc.get("growth", 0.0))
        self._rt["oja_steps"] = int(desc.get("oja_steps", 0))

    def describe(self) -> dict:
        return {
            "type": self.type_name,
            "gamma": round(self.gamma, 4),
            "writes": int(self._rt["writes"]),
            "norm": round(self.get_norm(), 4),
            "growth": round(float(self._rt["growth"]), 4),
            "bonus_age": int(self._rt["bonus_age"]),
            "oja_steps": int(self._rt["oja_steps"]),
            "has_state": self.S is not None,
            "derived_anchor": self._rt["P_override"] is not None,
        }


MoMWLayer = FastWeightLayer   # v1 调用方兼容别名


# ===========================================================================
# 规格接口 2：MoMWRouter —— 三阶段混合打分
# ===========================================================================
class MoMWRouter(nn.Module):
    """零 nn.Parameter：余弦相似度 + 动态阈值 + 张量运算。"""

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

        h = _unit(H_t.detach().float().mean(dim=(0, 1)), dim=-1)   # [D]

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
            raw.append(base)
            scores.append(base + balance + l.bonus(cfg["B0"], cfg["M"]))
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
        if scores.numel() == 0 or float(scores.max().item()) < threshold:
            return None, None
        k = min(int(N), int(scores.numel()))
        top_scores, indices = torch.topk(scores, k)
        return indices, torch.softmax(top_scores / tau, dim=-1)

    def forward(self, H_t: torch.Tensor, pool, N: int, tau: float,
                threshold: float
                ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        _raw, scores = self.score_all(H_t, pool.kda_layers, pool.anchors,
                                      pool.cfg)
        return self.route(scores, N, tau, threshold)


Router = MoMWRouter   # v1 兼容别名


# ===========================================================================
# 规格接口 3：MoMWPool —— 读写隔离 + 空层分裂 + 相对量过载 + 异步写
# ===========================================================================
class MoMWPool(nn.Module):
    """记忆层池。零参数零 persistent buffer，动态扩展不改变权重键集合。"""

    def __init__(self, kda: nn.Module, cfg: dict,
                 anchors: Optional[torch.Tensor] = None):
        super().__init__()
        self.cfg = dict(cfg)
        self._refs = {"kda": kda}
        self.anchors = anchors
        self.kda_layers = nn.ModuleList()
        self.coverage = deque(maxlen=max(1, int(cfg["M"])))
        self.score_history = deque(
            maxlen=max(4, int(cfg["quantile_window"])))
        self.last_indices: List[int] = []
        self.last_threshold = float(cfg["threshold"])
        self.last_threshold_mode = "fixed"
        self.expansions = 0
        self._serial = 0
        self._cooldown = 0
        self._writer = _StateWriter() if cfg.get("async_write") else None
        self._persist_path: Optional[str] = None
        for i in range(int(cfg["initial_layers"])):
            self._append_layer(i % len(spec.MOMW_ANCHOR_GROUPS))

    @property
    def layers(self) -> List[FastWeightLayer]:
        return list(self.kda_layers)

    # ---- 阶段三：空层分裂 ----
    def _append_layer(self, type_idx: int,
                      anchor_override: Optional[torch.Tensor] = None) -> bool:
        if len(self.kda_layers) >= int(self.cfg["max_layers"]):
            return False
        seed = spec.MOMW_ANCHOR_SEED + 7919 * self._serial + type_idx
        self._serial += 1
        # **绝不 clone 原层状态**：S_new=None、Oja 方向重新随机、
        # writes/γ/age 全部从零开始；只有锚点方向可以继承。
        self.kda_layers.append(FastWeightLayer(
            self._refs["kda"], self.cfg, type_idx, anchor_override, seed))
        self.expansions += 1
        return True

    def _derive_anchor(self, parent: FastWeightLayer
                       ) -> Optional[torch.Tensor]:
        if (int(parent._rt["writes"]) >= int(self.cfg["eigen_min_writes"])
                and parent._rt["E"] is not None):
            return parent._rt["E"].detach().clone()
        return None

    def _overload_parent(self, indices: Optional[torch.Tensor]
                         ) -> Optional[FastWeightLayer]:
        """相对量过载判定：同类型层范数中位数的倍数，而非绝对阈值。"""
        if indices is None or not indices.numel() or not self.kda_layers:
            return None
        c = self.cfg
        norms = {i: l.get_norm() for i, l in enumerate(self.kda_layers)}
        by_type: Dict[int, List[float]] = {}
        for i, l in enumerate(self.kda_layers):
            by_type.setdefault(l.type_idx, []).append(norms[i])
        min_writes = max(2, int(c["M"]))
        for it in indices:
            i = int(it.item())
            l = self.kda_layers[i]
            n = norms[i]
            med = _median(by_type[l.type_idx])
            if int(l._rt["writes"]) < min_writes:
                continue                      # bonus 保护期内不判过载
            if (float(c["saturation_ratio"]) > 1.0 and med > 0
                    and n > float(c["saturation_ratio"]) * med):
                return l
            if float(c["saturation_limit"]) > 0 \
                    and n > float(c["saturation_limit"]):
                return l
            # 增长率用**该层自身** prev_norm 做基线：同类中位数在单层时
            # 恒等于自己（n > ratio·n 只有 ratio<1 才成立），永远判不出过载。
            prev = float(l._rt["prev_norm"])
            if (float(c["growth_ratio"]) > 0 and prev > 0
                    and float(l._rt["growth"])
                    > float(c["growth_ratio"]) * prev):
                return l
            # v2 的绝对增量阈值保留为兜底（0 = 不启用）
            if float(c["expand_delta"]) > 0 \
                    and float(l._rt["growth"]) > float(c["expand_delta"]):
                return l
        return None

    def maybe_expand(self, indices: Optional[torch.Tensor],
                     raw: Optional[torch.Tensor] = None) -> bool:
        c = self.cfg
        if self._cooldown > 0:
            self._cooldown -= 1
        parent = self._overload_parent(indices)
        starved = False
        if raw is not None and float(c["starve_threshold"]) > 0.0:
            self.coverage.append(float(raw.max().item())
                                 if raw.numel() else 0.0)
            M = max(1, int(c["M"]))
            starved = (len(self.coverage) >= M
                       and sum(self.coverage) / len(self.coverage)
                       < float(c["starve_threshold"]))
        if (parent is None and not starved) or self._cooldown > 0:
            return False
        ok = self._append_layer(
            parent.type_idx if parent is not None else 0,
            self._derive_anchor(parent) if parent is not None else None)
        if ok:
            self._cooldown = max(1, int(c["M"]))
            self.coverage.clear()
        return ok

    # ---- 动态阈值：滑动窗口分位数 ----
    def effective_threshold(self) -> Tuple[float, str]:
        """返回 (阈值, 模式)。分位模式下用历史分数的 p-分位。

        **先算阈值再路由，最后才把本步分数记进历史**：否则当前样本会污染
        自己的门槛（自举）。历史不足时退回固定 threshold。
        """
        c = self.cfg
        if c.get("threshold_mode") != "quantile":
            return float(c["threshold"]), "fixed"
        h = list(self.score_history)
        if len(h) < max(spec.MOMW_QUANTILE_MIN_SAMPLES, 2):
            return float(c["threshold"]), "cold"
        t = torch.tensor(h, dtype=torch.float32).quantile(
            float(c["quantile_p"])).item()
        return float(t), "quantile"

    def record_scores(self, raw: torch.Tensor) -> None:
        if raw.numel():
            self.score_history.append(float(raw.max().item()))

    # ---- Pool 前向 ----
    def forward(self, H_t: torch.Tensor, indices: Optional[torch.Tensor],
                alpha: Optional[torch.Tensor]):
        if indices is None:
            self.last_indices = []
            return 0.0
        delta_H = torch.zeros_like(H_t)
        for idx_t, weight in zip(indices, alpha):
            l = self.kda_layers[int(idx_t.item())]
            out, pending = l.forward(H_t, commit=self._writer is None)
            if self._writer is not None:
                self._writer.submit(l, pending)
            delta_H = delta_H + weight * out
        self.last_indices = [int(i.item()) for i in indices]
        return delta_H

    def tick_bonus(self) -> None:
        for l in self.kda_layers:
            l.tick_bonus()

    # ---- 异步 / 快照 ----
    def flush(self, timeout: float = 10.0) -> int:
        return self._writer.flush(timeout) if self._writer is not None else 0

    def pending_writes(self) -> int:
        return self._writer.pending() if self._writer is not None else 0

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    def clone(self) -> "MoMWPool":
        p = MoMWPool(self._refs["kda"], self.cfg, self.anchors)
        p.close()                                # 丢掉构造期自建的 writer
        p._writer = (_StateWriter() if self.cfg.get("async_write") else None)
        p.kda_layers = nn.ModuleList([l.clone() for l in self.kda_layers])
        p.coverage = deque(self.coverage, maxlen=self.coverage.maxlen)
        p.score_history = deque(self.score_history,
                                maxlen=self.score_history.maxlen)
        p.last_indices = list(self.last_indices)
        p.last_threshold = self.last_threshold
        p.last_threshold_mode = self.last_threshold_mode
        p.expansions = self.expansions
        p._serial = self._serial
        p._cooldown = self._cooldown
        p._persist_path = self._persist_path
        return p

    def restore(self, snap: dict, device=None, dtype=None) -> None:
        """按 momw_state 快照重建整个池（层数 / 状态 / 计数）。"""
        layout, tensors = snap["layout"], snap["tensors"]
        descs = layout.get("layers") or []
        if not descs:
            return
        self.close()
        self._writer = (_StateWriter()
                        if self.cfg.get("async_write") else None)
        self.kda_layers = nn.ModuleList()
        for d in descs:
            self.kda_layers.append(FastWeightLayer(
                self._refs["kda"], self.cfg, int(d.get("type_idx", 0)),
                seed=int(d.get("seed", 0))))
        for i, d in enumerate(descs):
            self.kda_layers[i].restore(d, tensors, i, device, dtype)
        self.expansions = int(layout.get("expansions", len(descs)))
        self._serial = int(layout.get("serial", len(descs)))
        self.last_indices = []
        self.score_history.clear()
        self.coverage.clear()

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
            "threshold": round(self.last_threshold, 6),
            "threshold_mode": self.last_threshold_mode,
            "score_window": len(self.score_history),
            "queue_pending": self.pending_writes(),
            "persisted": self._persist_path,
            "async_write": self._writer is not None,
        }


# ===========================================================================
# 规格接口 4：MoMW —— 整体组装
# ===========================================================================
class MoMW(nn.Module):
    """插在第一个 Block 之后、与 fast weight 并列的可选记忆阀门。

        H_out = H_1 + ΔH；无层激活 / silent / 门控为 0 → 直接返回 H_1。
    """

    STATE_KEY = "momw"
    SILENT_KEY = "momw_silent"

    def __init__(self, kda: nn.Module, cfg: Optional[dict] = None,
                 wte: Optional[nn.Module] = None,
                 vocab_dir: Optional[str] = None,
                 dim: Optional[int] = None, N: Optional[int] = None,
                 tau: Optional[float] = None,
                 threshold: Optional[float] = None):
        super().__init__()
        merged = spec.momw_default_config()
        if cfg:
            merged.update({k: v for k, v in cfg.items() if k in merged})
        if N is not None:
            merged["N"] = int(N)
        if tau is not None:
            merged["tau"] = float(tau)
        if threshold is not None:
            merged["threshold"] = float(threshold)
        self.cfg = merged

        D = int(kda.w_o.weight.shape[0])
        if dim is not None and int(dim) != D:
            raise RuntimeError(f"MoMW：dim={dim} 与模型 d_model={D} 不一致")
        self.dim = D
        self._refs = {"kda": kda, "wte": wte, "vocab_dir": vocab_dir}
        K = len(spec.MOMW_ANCHOR_GROUPS)
        self.register_buffer("_anchors", torch.zeros(K, D),
                             persistent=False)
        self._anchors_ready = False
        self._anchor_provider = None
        self.router = MoMWRouter(D)
        self.pool = MoMWPool(kda, merged)
        self._local = self.pool
        self.silent = False
        self.steps = 0
        self.last_raw: List[float] = []
        self.last_scores: List[float] = []
        self.last_indices: List[int] = []
        self.last_fw_energy = 0.0
        self.last_gate = 1.0

    # ---- 阶段一：锚点播种（真文本优先） ----
    def set_anchor_provider(self, fn) -> None:
        """注入 fn(K, D) -> Tensor[K,D]；下次 forward 生效。"""
        self._anchor_provider = fn
        self._anchors_ready = False

    def _text_provider(self):
        """用**模型自己的词表副本**编码代表文本 → wte 均值。

        用模型 vocab/ 而不是全局词表：版本升级时全局 ID 会整体位移，
        模型目录里的副本永远与它自己的 wte 行号对得上。
        """
        vocab_dir = self._refs.get("vocab_dir")
        wte = self._refs.get("wte")
        if not vocab_dir or wte is None:
            return None

        def provider(K: int, D: int) -> torch.Tensor:
            from tokens import enctok
            from tokens.vocab import Vocab
            v = Vocab.load(os.path.join(vocab_dir, "vocab.vfc"))
            rows = []
            for name in spec.MOMW_ANCHOR_GROUPS:
                ids: List[int] = []
                for text in spec.MOMW_ANCHOR_TEXTS.get(name, ()):
                    try:
                        got, _skipped = enctok.encode_text(text, v)
                    except Exception:
                        continue
                    ids.extend(int(t) for t in got)
                if not ids:
                    raise RuntimeError(f"锚点文本编码为空: {name}")
                idx = torch.tensor(ids, dtype=torch.long,
                                   device=wte.weight.device)
                rows.append(wte(idx).mean(dim=0))
            return torch.stack(rows)
        return provider

    def _ensure_anchors(self, H: torch.Tensor) -> None:
        if self._anchors_ready:
            return
        K, D = self._anchors.shape
        with torch.no_grad():
            A = torch.zeros(K, D, device=H.device, dtype=torch.float32)
            provider = (self._anchor_provider or self._text_provider())
            wte = self._refs.get("wte")
            done = False
            if provider is not None:
                try:
                    got = provider(K, D)
                    if isinstance(got, torch.Tensor) and got.shape == (K, D):
                        A = got.detach().to(A.device, A.dtype)
                        done = True
                except Exception:
                    done = False       # 编码失败 → 静默降级到 token 采样
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

    # ---- 状态归属 ----
    def _make_pool(self) -> MoMWPool:
        return MoMWPool(self._refs["kda"], self.cfg, self._anchors)

    def reset(self) -> None:
        old = self._local
        self._local = self._make_pool()
        self.pool = self._local
        if old is not None:
            old.close()
        self.steps = 0
        self.last_raw, self.last_scores, self.last_indices = [], [], []
        self.last_fw_energy, self.last_gate = 0.0, 1.0

    def _pool_for(self, state: Optional[dict]) -> MoMWPool:
        if state is None:
            return self._local
        pool = state.get(self.STATE_KEY)
        if not isinstance(pool, MoMWPool):
            # 新会话从「启动基线」克隆一份，之后独立演进（互不串味）
            pool = (self._local.clone() if self._local.kda_layers
                    else self._make_pool())
            state[self.STATE_KEY] = pool
        return pool

    def flush(self, state: Optional[dict] = None) -> int:
        return self._pool_for(state).flush()

    # ---- 前向 ----
    def forward(self, H_1: torch.Tensor, state: Optional[dict] = None,
                fw_energy: Optional[float] = None) -> torch.Tensor:
        self._ensure_anchors(H_1)
        pool = self._pool_for(state)
        pool.anchors = self._anchors
        c = self.cfg
        silent = (bool(state.get(self.SILENT_KEY, False))
                  if state is not None else self.silent)
        with torch.no_grad():
            threshold, mode = pool.effective_threshold()
            raw, scores = self.router.score_all(
                H_1, pool.kda_layers, self._anchors, c)
            indices, alpha = self.router.route(
                scores, c["N"], c["tau"], threshold)
            # cost-aware 门控：fast weight 读出越强，MoMW 注入越小。
            # 用 exp(-c·E) 而不是 σ(-c·E)：后者在 E=0 时恒为 0.5，会把
            # 「没有 fast weight 能量」的输入也砍半。
            gate = 1.0
            if fw_energy is not None and float(c["fw_gate_c"]) > 0.0:
                gate = math.exp(-float(c["fw_gate_c"]) * max(0.0, fw_energy))
            if indices is not None and gate < 1.0:
                alpha = alpha * gate
            delta_H = pool.forward(H_1, indices, alpha)
            pool.tick_bonus()
            pool.maybe_expand(indices, raw)
            pool.record_scores(raw)     # 路由之后才入历史，避免自举
        pool.last_threshold, pool.last_threshold_mode = threshold, mode
        self.steps += 1
        self.last_raw = list(self.router.last_raw)
        self.last_scores = list(self.router.last_scores)
        self.last_indices = pool.last_indices
        self.last_fw_energy = float(fw_energy or 0.0)
        self.last_gate = float(gate)
        if indices is None or silent or gate <= 1e-6:
            return H_1
        return H_1 + delta_H

    # ---- 快照 ----
    def save_state(self, model_dir: str, state: Optional[dict] = None,
                   extra: Optional[dict] = None) -> Optional[str]:
        """把当前池写成 momw_state/ 快照（先 flush 异步写，保证一致）。"""
        import momw_state as _ms
        if not self.cfg.get("persist"):
            return None
        pool = self._pool_for(state)
        pool.flush()
        dst = _ms.state_dir(model_dir)
        _ms.save(dst, pool, extra={"steps": self.steps,
                                   "fw_gate_c": self.cfg["fw_gate_c"],
                                   **(extra or {})})
        pool._persist_path = dst
        return dst

    def load_state(self, model_dir: str, state: Optional[dict] = None,
                   device=None, dtype=None) -> bool:
        """按快照重建池；无快照 / 损坏 → False（从零初始化）。"""
        import momw_state as _ms
        snap = _ms.load(_ms.state_dir(model_dir))
        if snap is None:
            return False
        self._pool_for(state).restore(snap, device=device, dtype=dtype)
        return True

    # ---- 标定 / 统计 ----
    def calibrate(self, batches: Sequence[torch.Tensor]) -> dict:
        """离线校准：跑一批种子输入，返回各类打分分布与建议阈值。

        规格说的「类内得分 5% 分位数」落地为：对每个类型收集「最高分落在
        该类型」的样本 raw 分，取 p 分位与均值。返回的是**建议值**，不自动
        写回 config——阈值是运维决策，不该被一次标定静默改掉。
        """
        per_type: Dict[str, List[float]] = {
            n: [] for n in spec.MOMW_ANCHOR_GROUPS}
        all_scores: List[float] = []
        with torch.no_grad():
            for x in batches:
                self._ensure_anchors(x)
                raw, _ = self.router.score_all(
                    x, self._local.kda_layers, self._anchors, self.cfg)
                if not raw.numel():
                    continue
                vals = [float(v) for v in raw.tolist()]
                all_scores.extend(vals)
                best = int(torch.tensor(vals).argmax())
                name = self._local.kda_layers[best].type_name
                per_type[name].append(vals[best])

        def _q(v: List[float], p: float):
            if not v:
                return None
            return float(torch.tensor(v, dtype=torch.float32)
                         .quantile(p).item())
        p = float(self.cfg["quantile_p"])
        return {
            "samples": len(all_scores),
            "all_p5": _q(all_scores, 0.05),
            "all_p50": _q(all_scores, 0.5),
            "all_p95": _q(all_scores, 0.95),
            "suggest_threshold": _q(all_scores, p),
            "per_type": {k: {"n": len(v), "p5": _q(v, 0.05),
                             "mean": (sum(v) / len(v)) if v else None}
                         for k, v in per_type.items()},
        }

    def stats(self, state: Optional[dict] = None) -> dict:
        pool = self._pool_for(state)
        out = pool.describe()
        out.update({
            "enabled": True,
            "silent": (bool(state.get(self.SILENT_KEY, False))
                       if state is not None else self.silent),
            "steps": self.steps,
            "dim": self.dim,
            "anchors_ready": self._anchors_ready,
            "anchor_source": ("provider"
                              if self._anchor_provider is not None
                              else ("text"
                                    if self._refs.get("vocab_dir")
                                    and self._refs.get("wte") else "sample")),
            "last_raw": [round(v, 6) for v in self.last_raw],
            "last_scores": [round(v, 6) for v in self.last_scores],
            "last_cos_anchor": [round(v, 6)
                                for v in self.router.last_cos_anchor],
            "last_cos_state": [round(v, 6)
                               for v in self.router.last_cos_state],
            "last_gamma": list(self.router.last_gamma),
            "fw_energy": round(self.last_fw_energy, 6),
            "fw_gate": round(self.last_gate, 6),
            "config": dict(self.cfg),
        })
        return out
