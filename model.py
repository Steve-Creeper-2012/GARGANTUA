# ===========================================================================
# GARGANTUA v1 — 模型核心
# ---------------------------------------------------------------------------
# 架构（冻结契约见 spec.py）：
#   encoder-decoder，共享统一 KV 序列流，无 cross attention；
#   block = KDA → MLA-HCA → KDA → MLA-CSA → KDA → FFN；
#   fast weight = 全局 KDA 结构 bank（带遗忘门），enc/dec 各调一次；
#   双层 KV：完整区(1K)｜缓冲区｜压缩区(3K)，CSA 4:1 局部 / HCA 32:1 全局；
#   时间步为全局原子单位：压缩/滑窗/因果掩码均不拆分时间步。
#
# 技术参考：
#   [1] DeepSeek-V2 (arXiv 2405.04434)  MLA 低秩联合 KV 压缩（本模型 NoPE，
#       无 RoPE / decoupled 分支）
#   [2] Kimi Linear (arXiv 2510.26692)  KDA：channel-wise 门控 delta 递推
#       S = (I - βkkᵀ)Diag(α)S + βkvᵀ, o = Sᵀq
#   [3] DeepSeek NSA (arXiv 2502.11089) 可学习 MLP 压缩器（CSA/HCA 形态）
#   [4] Schmidhuber 1991 (FKI-147-91)   fast weight（外积加性更新）
# ===========================================================================
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import spec


# ===========================================================================
# 基础组件
# ===========================================================================
class RMSNorm(nn.Module):
    """ Root Mean Square LayerNorm（无均值平移，优化器归 AdamW 组）。 """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))


class FusedRMSNormGated(nn.Module):
    """ RMSNorm（per-head，对 dv 维）+ 数据依赖 sigmoid 门控（KDA 输出门）。 """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(dim, eps)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H, dv]；gate: [B, T, H*dv] → 对齐到 per-head 形状
        g = F.silu(gate.view(*x.shape))
        return self.norm(x) * g


class CausalDepthwiseConv1d(nn.Module):
    """ 因果 depthwise 一维短卷积（KDA 的 q/k/v 局部混合，kernel=4）。
    左侧补零保证严格因果：位置 t 只看 ≤ t 的输入。 """

    def __init__(self, dim: int, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.weight = nn.Parameter(torch.randn(dim, 1, kernel) * 0.02)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        x = F.pad(x, (self.kernel - 1, 0))          # 左侧因果 padding
        x = F.conv1d(x, self.weight, self.bias, groups=x.shape[1])
        return x.transpose(1, 2)


# ===========================================================================
# [2] KDA —— channel-wise 门控 delta-rule 递推（NoPE，天然因果）
# ---------------------------------------------------------------------------
# 递推式（逐 token）：
#   S'  = Diag(α_t) · S_{t-1}                    # channel-wise 遗忘
#   S_t = S' + β_t · k_t ⊗ (v_t - S'ᵀ k_t)      # delta rule（先纠错再写入）
#   o_t = S_tᵀ q_t                              # 读出
# 状态 S: [H, dk, dv]，序列维递推；encoder 段需要双向时由调用方翻转序列。
# ===========================================================================
def kda_recurrence(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   alpha: torch.Tensor, beta: torch.Tensor,
                   state: Optional[torch.Tensor] = None
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """ 逐 token 精确递推（正确性基准实现；UT-transform 分块核为后续优化项）。

    q/k:    [B, T, H, dk]（已 L2 归一化）
    v:      [B, T, H, dv]
    alpha:  [B, T, H, dk] ∈ (0,1]，channel-wise 遗忘率
    beta:   [B, T, H, 1]  ∈ (0,1)，写入门
    state:  [B, H, dk, dv] 或 None（零初始化）
    返回:   out [B, T, H, dv], new_state [B, H, dk, dv]
    """
    B, T, H, dk = q.shape
    dv = v.shape[-1]
    S = q.new_zeros(B, H, dk, dv) if state is None else state
    outs = []
    for t in range(T):
        qt = q[:, t]                       # [B, H, dk]
        kt = k[:, t]
        vt = v[:, t]                       # [B, H, dv]
        at = alpha[:, t].unsqueeze(-1)     # [B, H, dk, 1]
        bt = beta[:, t]                    # [B, H, 1]
        S = S * at                                        # 遗忘
        err = vt - torch.einsum('bhsv,bhs->bhv', S, kt)   # 纠错：v - Sᵀk
        S = S + bt.unsqueeze(-1) * kt.unsqueeze(-1) * err.unsqueeze(-2)
        outs.append(torch.einsum('bhsv,bhs->bhv', S, qt))
    return torch.stack(outs, dim=1), S


class KDAAttention(nn.Module):
    """ KDA 线性递推注意力层。
    负责：位置/顺序信息（NoPE 下唯一的位置算子）、长程递归记忆、
    视频帧间状态传递（帧序列沿时间轴流过 KDA 状态）。 """

    def __init__(self, d_model: int = spec.D_MODEL, n_head: int = spec.N_HEADS,
                 head_dim: int = spec.KDA_HEAD_DIM):
        super().__init__()
        self.H = n_head
        self.dk = head_dim
        self.dv = int(head_dim * spec.KDA_EXPAND_V)
        key_dim = self.H * self.dk
        val_dim = self.H * self.dv

        # q/k/v 投影 + ShortConv + Swish + (q,k) L2Norm（Kimi Linear 参数化）
        self.w_q = nn.Linear(d_model, key_dim, bias=False)
        self.w_k = nn.Linear(d_model, key_dim, bias=False)
        self.w_v = nn.Linear(d_model, val_dim, bias=False)
        self.conv_q = CausalDepthwiseConv1d(key_dim, spec.KDA_CONV_KERNEL)
        self.conv_k = CausalDepthwiseConv1d(key_dim, spec.KDA_CONV_KERNEL)
        self.conv_v = CausalDepthwiseConv1d(val_dim, spec.KDA_CONV_KERNEL)

        # channel-wise 遗忘门 α：低秩投影（rank = head_dim）+ 衰减函数
        self.w_alpha_down = nn.Linear(d_model, head_dim, bias=False)
        self.w_alpha_up = nn.Linear(head_dim, key_dim, bias=False)
        # 写入门 β：sigmoid
        self.w_beta = nn.Linear(d_model, self.H, bias=False)
        # 输出：RMSNorm + 数据依赖门控 + 输出投影
        self.o_norm = FusedRMSNormGated(self.dv)
        self.w_gate = nn.Linear(d_model, val_dim, bias=False)
        self.w_o = nn.Linear(val_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor,
                state: Optional[torch.Tensor] = None,
                reverse: bool = False,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ x: [B, T, D]。reverse=True 时先翻转序列再递推（encoder 用 KDA
        获得反向信息流；正向/反向两个 KDA 子层由 block 组装保证双向性）。 """
        B, T, _ = x.shape
        if reverse:
            x = x.flip(1)
        q = F.silu(self.conv_q(self.w_q(x))).view(B, T, self.H, self.dk)
        k = F.silu(self.conv_k(self.w_k(x))).view(B, T, self.H, self.dk)
        v = F.silu(self.conv_v(self.w_v(x))).view(B, T, self.H, self.dv)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        # α ∈ (0,1]：log 空间衰减参数化，log α = -softplus(w)，α = exp(-softplus(...))
        a = self.w_alpha_up(F.silu(self.w_alpha_down(x))).view(B, T, self.H, self.dk)
        alpha = torch.exp(-F.softplus(a)).clamp(max=1.0)
        beta = torch.sigmoid(self.w_beta(x)).unsqueeze(-1)   # [B, T, H, 1]

        out, state = kda_recurrence(q, k, v, alpha, beta, state)
        gate = self.w_gate(x)
        out = self.o_norm(out, gate)                        # [B, T, H, dv]
        out = self.w_o(out.reshape(B, T, self.H * self.dv))
        if reverse:
            out = out.flip(1)
        return out, state


# ===========================================================================
# [1] MLA —— DeepSeek-V2 低秩联合 KV 压缩注意力（NoPE）
# ---------------------------------------------------------------------------
# 共享 KV 设计：统一 KV 序列流存的是「词向量」（d_model 维语义向量 +
# step_ids + type_ids），不是各层私有的 K/V。每个 MLA 层用自己的投影
# 从流中向量重建 K/V：c = W_DKV·h_store（低秩隐向量），k = W_UK·c，
# v = W_UV·c；q 从当前 hidden 低秩投影。
# ===========================================================================
class MLAAttention(nn.Module):
    """ MLA 注意力。causal=True 时按时间步因果（步内双向、步间因果）。"""

    def __init__(self, d_model: int = spec.D_MODEL, n_head: int = spec.N_HEADS,
                 head_dim: int = spec.HEAD_DIM, causal: bool = True):
        super().__init__()
        self.H = n_head
        self.dh = head_dim
        self.causal = causal
        inner = n_head * head_dim

        self.w_dq = nn.Linear(d_model, spec.MLA_Q_LORA_RANK, bias=False)
        self.w_uq = nn.Linear(spec.MLA_Q_LORA_RANK, inner, bias=False)
        self.w_dkv = nn.Linear(d_model, spec.MLA_KV_LORA_RANK, bias=False)
        self.w_uk = nn.Linear(spec.MLA_KV_LORA_RANK, inner, bias=False)
        self.w_uv = nn.Linear(spec.MLA_V_LORA_RANK, inner, bias=False)
        self.w_o = nn.Linear(inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, kv: torch.Tensor,
                kv_step_ids: Optional[torch.Tensor] = None,
                x_step_ids: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """ x:  [B, Tq, D] 当前 hidden（q 侧）
        kv: [B, Tk, D] 共享 KV 流（或 encoder 段自身 hidden，双向）
        因果规则：x 的每个位置可看 kv 中 step_id 严格小于自身 step 的全部
        位置，以及同 step 内的全部位置（步内双向）。kv 为 encoder 段时
        causal=False 全双向。 """
        B, Tq, _ = x.shape
        Tk = kv.shape[1]
        q = self.w_uq(self.w_dq(x)).view(B, Tq, self.H, self.dh).transpose(1, 2)
        c = self.w_dkv(kv)                                   # [B, Tk, r]
        k = self.w_uk(c).view(B, Tk, self.H, self.dh).transpose(1, 2)
        v = self.w_uv(c).view(B, Tk, self.H, self.dh).transpose(1, 2)

        attn = torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(self.dh)
        if self.causal and kv_step_ids is not None and x_step_ids is not None:
            # 步间因果 + 步内双向：kv_step <= x_step 即可见
            mask = kv_step_ids.unsqueeze(1) <= x_step_ids.unsqueeze(2)  # [B,Tq,Tk]
            attn = attn.masked_fill(~mask.unsqueeze(1), float('-inf'))
        out = torch.einsum('bhqk,bhkd->bhqd', attn.softmax(-1), v)
        out = out.transpose(1, 2).reshape(B, Tq, self.H * self.dh)
        return self.w_o(out)


# ===========================================================================
# [3] 可学习压缩器（NSA 式）：一组 token → 1 个 d_model 向量
# ---------------------------------------------------------------------------
# 以时间步为原子单位聚组；组内 token 数凑满 ratio（不足 padding + mask）。
# 组内带相对位置嵌入；2 层 MLP。压缩全程可微，端到端训练。
# ===========================================================================
class StepCompressor(nn.Module):
    def __init__(self, ratio: int, d_model: int = spec.D_MODEL):
        super().__init__()
        self.ratio = ratio
        self.pos_emb = nn.Parameter(torch.randn(ratio, d_model) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

    def forward(self, vecs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ vecs: [G, ratio, D]（padding 区可为任意值），mask: [G, ratio] bool
        返回 [G, D]：组内加相对位置嵌入 → mask 加权平均 → MLP + 残差。 """
        v = vecs + self.pos_emb.unsqueeze(0)
        m = mask.unsqueeze(-1).float()
        pooled = (v * m).sum(1) / m.sum(1).clamp(min=1.0)      # [G, D]
        return self.mlp(pooled) + pooled                        # 残差


def group_by_step(step_ids: torch.Tensor, ratio: int
                  ) -> List[Tuple[int, int]]:
    """ 把一段位置按时间步聚成「压缩组」，返回 (start, end) 切片列表。
    规则：整步合并，组内 token 数达到 ratio 即封组；
    单步 token 数 ≥ ratio 时步内按 ratio 切多组（步内压缩，允许）；
    绝不把不同步的 token 与超过 ratio 的组合并。 """
    groups = []
    n = step_ids.numel()
    i = 0
    while i < n:
        sid = step_ids[i].item()
        j = i
        while j < n and step_ids[j].item() == sid:
            j += 1
        # [i, j) 是一个完整时间步
        step_len = j - i
        if step_len >= ratio:
            for s in range(i, j, ratio):           # 步内切分（不跨步）
                groups.append((s, min(s + ratio, j)))
        else:
            g_start, count = i, step_len
            k = j
            while k < n and count < ratio:          # 整步累加到 ratio
                nid = step_ids[k].item()
                e = k
                while e < n and step_ids[e].item() == nid:
                    e += 1
                if count + (e - k) > ratio:
                    break                            # 下一步放进来会超 ratio
                count += e - k
                k = e
            groups.append((g_start, k))
            i = k
            continue
        i = j
    return groups


# ===========================================================================
# 统一 KV 序列流（双层：完整区｜缓冲区｜CSA/HCA 压缩区）
# ---------------------------------------------------------------------------
# 存：词向量 [D]、step_id、type_id、来源（enc/dec）。
# 约束：压缩/丢弃全部以时间步为原子；缓冲区吸收图元/音元就地压缩的长度差；
# 原始 token json 全量存档（archive），溢出仅丢 KV 视图不丢存档。
# ===========================================================================
@dataclass
class KVEntry:
    vec: torch.Tensor          # [D] 词向量（共享语义向量）
    step_id: int
    type_id: int
    source: str                # "enc" | "dec"


class SharedKV:
    """ 统一 KV 序列流。负责：追加、双层视图（完整/CSA/HCA）、
    时间步原子滑窗、图元 9:1 就地压缩、json 存档。 """

    def __init__(self, csa_compressor: StepCompressor, hca_compressor: StepCompressor,
                 draw_packer: Optional[StepCompressor] = None):
        self.entries: List[KVEntry] = []
        self.csa = csa_compressor
        self.hca = hca_compressor
        # 图元 9:1 打包器（CSA 打包摘要语义，但 ratio=9，与 CSA 4:1 不同模块）
        self.draw_packer = draw_packer or StepCompressor(spec.DRAW_KV_COMPRESS)
        self.archive: List[dict] = []        # 原始数据全量存档（溢出不丢）
        self._pending_draw: List[int] = []   # dec 侧图元 token 计数（9:1 压缩）

    # ---- 写入 ----
    def append(self, vec: torch.Tensor, step_id: int, type_id: int, source: str):
        for i in range(vec.shape[0]):
            self.entries.append(KVEntry(vec[i].detach(), step_id, type_id, source))
        self._evict()

    def mark_primitive_token(self):
        """ decoder 每输出一个图元 token 调一次；满 9 个触发就地 9:1 压缩
        （CSA 打包摘要语义：9 参数 → 1 向量，长度差由缓冲区吸收）。 """
        self._pending_draw.append(1)
        if len(self._pending_draw) >= spec.DRAW_KV_COMPRESS:
            n = len(self.entries)
            idx = list(range(n - spec.DRAW_KV_COMPRESS, n))
            if len(idx) == spec.DRAW_KV_COMPRESS:
                vs = torch.stack([self.entries[i].vec for i in idx])
                m = torch.ones(len(idx), dtype=torch.bool, device=vs.device)
                packed = self.draw_packer(vs.unsqueeze(0), m.unsqueeze(0))[0]
                e0 = self.entries[idx[0]]
                self.entries[idx[0]] = KVEntry(packed, e0.step_id, e0.type_id, e0.source)
                for i in reversed(idx[1:]):
                    del self.entries[i]
            self._pending_draw.clear()

    # ---- 滑窗（时间步原子）----
    def _evict(self):
        while len(self.entries) > spec.CTX_TOTAL:
            sid = self.entries[0].step_id
            while self.entries and self.entries[0].step_id == sid:
                self.entries.pop(0)          # 整步丢弃

    # ---- 读取视图 ----
    def _stack(self, entries: List[KVEntry]) -> Tuple[torch.Tensor, torch.Tensor]:
        vs = torch.stack([e.vec for e in entries])
        ss = torch.tensor([e.step_id for e in entries], device=vs.device)
        return vs, ss

    def view(self, kind: str, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """ kind ∈ {"full", "csa", "hca"}。
        full：完整区原样。csa：压缩区 4:1 + 完整区尾部局部窗口。
        hca：压缩区 32:1 全局 + 完整区。返回 (vecs [1, T, D], step_ids [1, T])。 """
        n = len(self.entries)
        full_n = min(n, spec.CTX_FULL)
        full, comp = self.entries[n - full_n:], self.entries[:n - full_n]
        if kind == "full":
            vs, ss = self._stack(full) if full else (None, None)
        elif kind == "csa":
            out_e = self._compress_entries(comp, spec.CSA_RATIO)
            keep = out_e + full[-min(len(full), spec.CSA_LOCAL_WINDOW):]
            vs, ss = self._stack(keep) if keep else (None, None)
        else:  # hca
            out_e = self._compress_entries(comp, spec.HCA_RATIO) + full
            vs, ss = self._stack(out_e) if out_e else (None, None)
        if vs is None:
            return None, None
        return vs.unsqueeze(0).to(device), ss.unsqueeze(0).to(device)

    def _compress_entries(self, entries: List[KVEntry], ratio: int) -> List[KVEntry]:
        if not entries:
            return []
        vs, ss = self._stack(entries)
        comp = self.csa if ratio == spec.CSA_RATIO else self.hca
        out = []
        for (s, e) in group_by_step(ss, ratio):
            g = vs[s:e]
            m = torch.ones(e - s, dtype=torch.bool, device=vs.device)
            pad = ratio - (e - s)
            if pad:
                g = torch.cat([g, g.new_zeros(pad, g.shape[-1])])
                m = torch.cat([m, m.new_zeros(pad)])
            vec = comp(g.unsqueeze(0), m.unsqueeze(0))[0]
            out.append(KVEntry(vec, entries[s].step_id, entries[s].type_id,
                               entries[s].source))
        return out


# ===========================================================================
# [4] fast weight —— 全局型，KDA 结构带遗忘门，enc/dec 共享
# ---------------------------------------------------------------------------
# 语义（spec 定案）：enc/dec 各自在第一个 block 之后调用同一个 bank；
# 先用当前状态读出（当权重层用），用后按 KDA 门控 delta 规则更新；
# 状态跨调用推进。仅全局型：所有内容都写入。
# ===========================================================================
class FastWeightBank(nn.Module):
    """ 结构与 KDAAttention 相同，区别在于状态 S 跨调用持久（成员变量），
    且每次调用都执行「读出 → 更新」两步。 """

    def __init__(self, d_model: int = spec.D_MODEL):
        super().__init__()
        self.H = spec.FW_N_HEAD
        self.dk = spec.FW_KEY_DIM
        self.dv = spec.FW_VALUE_DIM
        kd, vd = self.H * self.dk, self.H * self.dv
        self.w_q = nn.Linear(d_model, kd, bias=False)
        self.w_k = nn.Linear(d_model, kd, bias=False)
        self.w_v = nn.Linear(d_model, vd, bias=False)
        self.w_alpha = nn.Linear(d_model, kd, bias=False)
        self.w_beta = nn.Linear(d_model, self.H, bias=False)
        self.o_norm = RMSNorm(self.dv)
        self.w_o = nn.Linear(vd, d_model, bias=False)
        self._state: Optional[torch.Tensor] = None   # [B?, H, dk, dv]，B=1 持久

    def reset(self):
        self._state = None

    def forward(self, x: torch.Tensor, write: bool = True,
                detach_state: bool = True) -> torch.Tensor:
        """ x: [B, T, D]。先读后写（同一序列先以旧状态读出，再逐 token 推进）。
        detach_state=True：样本边界处截断状态梯度（状态ful 训练惯例）。 """
        B, T, _ = x.shape
        q = F.normalize(self.w_q(x).view(B, T, self.H, self.dk), p=2, dim=-1)
        k = F.normalize(self.w_k(x).view(B, T, self.H, self.dk), p=2, dim=-1)
        v = self.w_v(x).view(B, T, self.H, self.dv)
        alpha = torch.exp(-F.softplus(self.w_alpha(x).view(B, T, self.H, self.dk)))
        beta = torch.sigmoid(self.w_beta(x)).unsqueeze(-1)
        state = self._state
        if state is not None and state.shape[0] != B:
            state = None                              # batch 变化则重置
        out, new_state = kda_recurrence(q, k, v, alpha, beta, state)
        if write:
            self._state = new_state.detach() if detach_state else new_state
        out = self.o_norm(out)                            # [B, T, H, dv]
        return self.w_o(out.reshape(B, T, self.H * self.dv))


# ===========================================================================
# Block = KDA → MLA-HCA → KDA → MLA-CSA → KDA → FFN（+ attnRes）
# ---------------------------------------------------------------------------
# encoder block：KDA 天然因果，靠 forward/backward 两个 KDA 子层 + 双向
# MLA 保证双向信息流；decoder block：全部因果（步内双向由 MLA mask 负责）。
# ===========================================================================
class Block(nn.Module):
    def __init__(self, is_decoder: bool):
        super().__init__()
        self.is_decoder = is_decoder
        # 5 个注意力/递推子层 + 1 个 FFN，顺序见 spec.BLOCK_ORDER
        self.kda1 = KDAAttention()
        self.mla_hca = MLAAttention(causal=is_decoder)
        self.kda2 = KDAAttention()
        self.mla_csa = MLAAttention(causal=is_decoder)
        self.kda3 = KDAAttention()
        # GLU-FFN：w1 投影到 2×hidden，chunk 后 silu(a)⊙b，再 w2 降维
        self.ffn_up = nn.Linear(spec.D_MODEL, spec.FFN_HIDDEN * 2, bias=False)
        self.ffn_down = nn.Linear(spec.FFN_HIDDEN, spec.D_MODEL, bias=False)
        self.norms = nn.ModuleList([RMSNorm(spec.D_MODEL) for _ in range(6)])
        # attnRes：块级注意力残差（moonshot）：块输入经零初始化标量门混回输出
        self.attn_res_gate = nn.Parameter(torch.zeros(1))

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.ffn_up(x).chunk(2, dim=-1)       # GLU：SiLU(a) ⊙ b
        return self.ffn_down(F.silu(a) * b)

    def forward(self, x: torch.Tensor, kv_view_hca, kv_view_csa,
                x_step_ids: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """ kv_view_*: (vecs [1, Tk, D], step_ids [1, Tk]) 共享 KV 流视图；
        encoder 段传 (None, None) 时 MLA 退化为对自身 hidden 的双向注意力。 """
        x0 = x
        h, _ = self.kda1(self.norms[0](x))
        x = x + h
        kv_v, kv_s = kv_view_hca if kv_view_hca[0] is not None else (x, x_step_ids)
        x = x + self.mla_hca(self.norms[1](x), kv_v, kv_s, x_step_ids)
        # encoder 的中间 KDA 反向递推：天然因果的 KDA 在 encoder 里靠
        # 正/反两个方向获得双向信息流（decoder 全因果，不反向）。
        h, _ = self.kda2(self.norms[2](x), reverse=not self.is_decoder)
        x = x + h
        kv_v, kv_s = kv_view_csa if kv_view_csa[0] is not None else (x, x_step_ids)
        x = x + self.mla_csa(self.norms[3](x), kv_v, kv_s, x_step_ids)
        h, _ = self.kda3(self.norms[4](x))
        x = x + h
        x = x + self._ffn(self.norms[5](x))
        return x + torch.tanh(self.attn_res_gate) * x0   # attnRes（零初始化门）


# ===========================================================================
# Gargantua 主模型
# ---------------------------------------------------------------------------
# 输入流（spec 8）：分模态 embedding → encoder → 注入统一 KV（类型嵌入）
# → decoder 读 KV（无 cross attention）→ 自回归输出 → 回写 KV。
# ===========================================================================
class Gargantua(nn.Module):
    def __init__(self, n_enc_blocks: int | None = None,
                 n_dec_blocks: int | None = None):
        super().__init__()
        D = spec.D_MODEL
        n_enc = n_enc_blocks if n_enc_blocks is not None else spec.N_ENC_BLOCKS
        n_dec = n_dec_blocks if n_dec_blocks is not None else spec.N_DEC_BLOCKS
        # —— 词表 embedding（不绑定；查表类，可放 CPU）——
        self.wte = nn.Embedding(spec.VOCAB_SIZE_PADDED, D)
        self.type_emb = nn.Embedding(spec.NUM_TYPES, D)
        self.lm_head = nn.Linear(D, spec.VOCAB_SIZE_PADDED, bias=False)
        # —— 模态输入投影（embedding 不统一，各自投影到 d_model）——
        self.visual_proj = nn.Linear(spec.PATCH_VEC_DIM, D, bias=False)
        self.visual_coord_x = nn.Embedding(256, D)   # 2D 位置嵌入（加性，非 embed token）
        self.visual_coord_y = nn.Embedding(256, D)
        self.audio_proj = nn.Linear(spec.AUDIO_SAMPLES_PER_TOKEN, D, bias=False)
        # —— KV 压缩器与共享流 ——
        self.csa_compressor = StepCompressor(spec.CSA_RATIO, D)
        self.hca_compressor = StepCompressor(spec.HCA_RATIO, D)
        self.draw_packer = StepCompressor(spec.DRAW_KV_COMPRESS, D)  # 图元 9:1
        # —— encoder / decoder ——
        self.enc_blocks = nn.ModuleList([Block(is_decoder=False)
                                         for _ in range(n_enc)])
        self.dec_blocks = nn.ModuleList([Block(is_decoder=True)
                                         for _ in range(n_dec)])
        self.enc_final_norm = RMSNorm(D)
        self.dec_final_norm = RMSNorm(D)
        # —— fast weight：全局 bank，enc/dec 共享 ——
        self.fast_weight = FastWeightBank(D)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ---- 分模态 embedding ----
    def embed_text(self, token_ids: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        return self.wte(token_ids) + self.type_emb(type_ids)

    def embed_visual(self, patch_vecs: torch.Tensor, coords: torch.Tensor,
                     type_ids: torch.Tensor) -> torch.Tensor:
        """ patch_vecs: [N, 4096] uint8→float/255；coords: [N, 2] (gx, gy)。 """
        v = self.visual_proj(patch_vecs.float() / 255.0)
        v = v + self.visual_coord_x(coords[:, 0]) + self.visual_coord_y(coords[:, 1])
        return v + self.type_emb(type_ids)

    def embed_audio(self, samples: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        """ samples: [N, 533] float32（[-1,1] 波形）。 """
        return self.audio_proj(samples) + self.type_emb(type_ids)

    # ---- encoder：输入段 → 词向量（双向）----
    def encode(self, x: torch.Tensor, step_ids: torch.Tensor) -> torch.Tensor:
        for i, blk in enumerate(self.enc_blocks):
            x = blk(x, (None, None), (None, None), x_step_ids=step_ids)
            if i == 0:   # fast weight：第一个 block 之后调用一次（先用后更新）
                x = x + self.fast_weight(x)
        return self.enc_final_norm(x)

    # ---- decoder：读共享 KV，自回归（步间因果 + 步内双向）----
    def decode(self, x: torch.Tensor, step_ids: torch.Tensor,
               kv_hca, kv_csa) -> torch.Tensor:
        for i, blk in enumerate(self.dec_blocks):
            x = blk(x, kv_hca, kv_csa, x_step_ids=step_ids)
            if i == 0:   # fast weight：与 encoder 共享同一 bank，状态继续推进
                x = x + self.fast_weight(x)
        return self.dec_final_norm(x)

    def build_views(self, vecs: torch.Tensor, step_ids: torch.Tensor):
        """ 从活张量构建 HCA/CSA 视图（训练路径，梯度不断）。
        vecs: [1, N, D]，step_ids: [1, N]。N ≤ CTX_FULL 时双视图都是全量。 """
        n = vecs.shape[1]
        if n <= spec.CTX_FULL:
            return (vecs, step_ids), (vecs, step_ids)
        full = vecs[:, n - spec.CTX_FULL:]
        full_s = step_ids[:, n - spec.CTX_FULL:]
        comp = vecs[:, :n - spec.CTX_FULL]
        comp_s = step_ids[:, :n - spec.CTX_FULL]
        hca_c = self._compress_live(comp, comp_s, spec.HCA_RATIO)
        csa_c = self._compress_live(comp, comp_s, spec.CSA_RATIO)
        kv_hca = (torch.cat([hca_c[0], full], 1), torch.cat([hca_c[1], full_s], 1))
        csa_tail = full[:, -min(n, spec.CSA_LOCAL_WINDOW):]
        csa_tail_s = full_s[:, -min(n, spec.CSA_LOCAL_WINDOW):]
        kv_csa = (torch.cat([csa_c[0], csa_tail], 1),
                  torch.cat([csa_c[1], csa_tail_s], 1))
        return kv_hca, kv_csa

    def _compress_live(self, vecs: torch.Tensor, step_ids: torch.Tensor,
                       ratio: int):
        comp = self.csa_compressor if ratio == spec.CSA_RATIO else self.hca_compressor
        vs, ss = vecs[0], step_ids[0]
        out_v, out_s = [], []
        for (s, e) in group_by_step(ss, ratio):
            g = vs[s:e]
            m = torch.ones(e - s, dtype=torch.bool, device=vs.device)
            if ratio - (e - s) > 0:
                g = torch.cat([g, g.new_zeros(ratio - (e - s), g.shape[-1])])
                m = torch.cat([m, m.new_zeros(ratio - (e - s))])
            out_v.append(comp(g.unsqueeze(0), m.unsqueeze(0))[0])
            out_s.append(ss[s:s + 1])
        return torch.stack(out_v).unsqueeze(0), torch.cat(out_s).unsqueeze(0)

    def forward_train(self, input_ids: torch.Tensor, input_steps: torch.Tensor,
                      target_ids: torch.Tensor, target_steps: torch.Tensor,
                      ) -> torch.Tensor:
        """ 教师强制训练前向（B=1，纯文本路径；多模态 embed 走 embed_visual/
        embed_audio 拼入后再调本函数的同构逻辑）。
        input_*：用户输入段（TYPE_USER）；target_*：模型输出段（TYPE_MODEL，
        已左移一位作为 decoder 输入）。
        返回 decoder 每个位置的 logits [1, T_dec, V]。 """
        emb_in = self.embed_text(input_ids, torch.full_like(input_ids, spec.TYPE_USER))
        enc_out = self.encode(emb_in, input_steps)
        # 共享 KV = encoder 输出 + decoder 已生成 token 的 embedding（类型=模型输出）
        emb_dec = self.embed_text(target_ids, torch.full_like(target_ids, spec.TYPE_MODEL))
        kv_vecs = torch.cat([enc_out, emb_dec], 1)
        kv_steps = torch.cat([input_steps, target_steps], 1)
        kv_hca, kv_csa = self.build_views(kv_vecs, kv_steps)
        dec_out = self.decode(emb_dec, target_steps, kv_hca, kv_csa)
        return self.logits(dec_out)

    def logits(self, dec_out: torch.Tensor) -> torch.Tensor:
        return self.lm_head(dec_out)

    # ---- 多模态装配：混合 Step 列表 → (emb, step_ids) ----
    def assemble_embeddings(self, steps: List[dict]):
        """ 把 text/visual/audio（可混合）Step 列表装配成统一嵌入序列。
        Step 结构见 spec 8；视觉步 meta.patches=[{x,y,vec}]，音频步
        meta.samples=bytes(533 float32)；视频步可同时含 patches 与
        audio（同 t 刻，patches 在前、音频 token 在后，共享 step_id）。
        返回 (emb [1, N, D], step_ids [1, N])，step 内顺序保持。 """
        embs, sids = [], []
        for sid, st in enumerate(steps):
            tid_val = st.get("type", spec.TYPE_NONE)
            parts = []
            meta = st.get("meta", {})
            if st["token_ids"]:                       # 文本/符号 token
                ids = torch.tensor([st["token_ids"]])
                parts.append(self.embed_text(
                    ids, torch.full_like(ids, tid_val)))
            if "patches" in meta:                     # 视觉 patch
                vecs = torch.stack([torch.frombuffer(bytearray(p["vec"]),
                                  dtype=torch.uint8) for p in meta["patches"]])
                coords = torch.tensor([[p["x"], p["y"]]
                                       for p in meta["patches"]])
                parts.append(self.embed_visual(
                    vecs, coords, torch.full((len(vecs),), tid_val)).unsqueeze(0))
            if "samples" in meta:                     # 单音频 token
                s = torch.frombuffer(bytearray(meta["samples"]),
                                     dtype=torch.float32).unsqueeze(0)
                parts.append(self.embed_audio(
                    s, torch.full((1,), tid_val)).unsqueeze(0))
            if "audio" in meta:                       # 视频步内的音频 token 列表
                for a in meta["audio"]:
                    s = torch.frombuffer(bytearray(a["samples"]),
                                         dtype=torch.float32).unsqueeze(0)
                    parts.append(self.embed_audio(
                        s, torch.full((1,), tid_val)).unsqueeze(0))
            assert parts, f"空 Step: {st}"
            e = torch.cat(parts, 1)                   # [1, n, D]
            embs.append(e)
            sids.extend([sid] * e.shape[1])
        return torch.cat(embs, 1), torch.tensor([sids])

    def forward_train_embeddings(self, emb_in: torch.Tensor,
                                 s_in: torch.Tensor,
                                 emb_dec: torch.Tensor,
                                 s_dec: torch.Tensor) -> torch.Tensor:
        """ 多模态教师强制前向（embedding 已装配）。与 forward_train 同构，
        区别仅在输入段/输出段直接使用给定嵌入（文本路径走 forward_train）。 """
        enc_out = self.encode(emb_in, s_in)
        kv_vecs = torch.cat([enc_out, emb_dec], 1)
        kv_steps = torch.cat([s_in, s_dec], 1)
        kv_hca, kv_csa = self.build_views(kv_vecs, kv_steps)
        dec_out = self.decode(emb_dec, s_dec, kv_hca, kv_csa)
        return self.logits(dec_out)


# ===========================================================================
# 参数分组（优化器契约：Muon 只管隐藏层 2D 矩阵；embedding / lm_head /
# RMSNorm / 标量 / 卷积恒为 AdamW）
# ===========================================================================
def param_groups_for_optimizers(model: Gargantua) -> Dict[str, List[torch.Tensor]]:
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (p.ndim == 2 and "wte" not in name and "lm_head" not in name
                and "coord" not in name and "type_emb" not in name
                and "pos_emb" not in name):
            muon.append(p)
        else:
            adamw.append(p)
    return {"muon": muon, "adamw": adamw}
