# ===========================================================================
# GARGANTUA v4-somn — server/somn.py  (第三版：抢救性手术 / hardened)
# SOMN (Self-Organizing Memory Network)
# ---------------------------------------------------------------------------
# 无监督、纯自组织、跨平台（CPU / CUDA / MPS）的全连接单头快速权重记忆。
#
# 硬约束（不可破）：
#   1. 零外部依赖 —— 只用 torch（无 sklearn / numpy / scipy）。
#   2. 全平台 —— SVD 统一 torch.linalg.svd；MPS 上 SVD 暂存 CPU。
#   3. 绝对零污染 —— 层池 List 与状态 S 全部存在普通字典 _rt['somn'] 里，
#      绝不注册进 nn.Module；主模型 state_dict 逐字不变（零 Parameter / 零 Buffer）。
#   4. dtype / device 自动跟随输入 H_t，但状态 S 恒为「安全精度」。
#
# 2026-11-13 抢救性手术（feat/somn-stress-test，对应审计报告 F1-F8）：
#   F1 统一输入契约 _validate_H —— 维度 / 空序列 / dtype / NaN-Inf 入口即拒绝。
#      任何非有限输入直接 raise，绝不静默零激活、绝不污染状态。
#   F2 数值安全 —— /N 加 eps、状态有限性检查、eps 按 dtype 选、近零重建给 0 分、
#      svd_split 秩亏判定（σ≈0 的子层不生成，杜绝「幽灵分」）。
#   F3 层池硬上限 max_layers=8（传 0 / 传大值都钳制）+ 每步最多分裂 1 层。
#      前者防 OOM，后者把「一步内多层同步 SVD」的 705ms 尖刺压回单次 SVD。
#   F4 reset 彻底清理 —— 清 router 缓存、断开旧层 S 引用。
#   F5 attach_somn 幂等 —— 同 dim 返回同一管理器，异 dim 直接 raise。
#   F6 探索机制 —— 连续 explore_every(默认 100) 步零激活时，强制激活当前
#      得分最高的层写入一次，打破「min_score 过高 → write_count 冻结 →
#      不分裂 → 无 warm-up 新层 → 永久零激活」死锁。
#   F7 SVD CPU 降级路径二次保护，异常信息指明污染来源。
#   F8 低精度输入升 fp32 状态：修 fp16 下 eps=1e-8 下溢、SVD 往返误差、
#      read_only 把 fp32 的 S 临时降精度三条精度问题；输出 dtype 仍跟随输入。
#
# 保留 v2 动态演化：余弦重建路由、奇偶交替 SVD 无损拆分、出生特权 Warm-up。
# ===========================================================================
from __future__ import annotations

import math
import queue
import threading
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

__all__ = [
    "IndependentFWLayer",
    "EnergyElbowRouter",
    "SOMNModule",
    "attach_somn",
    "somn_stats",
]

# CPU 上 torch.linalg.svd 只支持 float32/float64/complex；fp16/bf16 需先升精度。
_SVD_SAFE_DTYPES = (
    torch.float32, torch.float64, torch.complex64, torch.complex128,
)

# ---- F2/F3/F8 常量（全部为唯一真源） ----
_DIV_EPS = 1e-8                 # /N 的 eps 保护（配合 F1 的 N==0 拒绝，双保险）
_MIN_REBUILD_RATIO = 1e-3      # 重建幅度/输入幅度低于此值 → 该层无有效记忆，判 0 分
_RANK_DEFICIENT_RATIO = 1e-6    # 子层 σ_max / σ₁ 低于此值判为秩亏，拒绝生成幽灵层
_HARD_MAX_LAYERS = 8            # 层池硬上限，任何配置都不得突破
_MAX_SPLITS_PER_STEP = 1        # 每步分裂预算，根治延迟尖刺
_DEFAULT_EXPLORE_EVERY = 100    # 连续零激活多少步后强制探索
_EPS_BY_DTYPE = {               # cosine_similarity 的 eps 必须可表示该 dtype
    torch.float16: 1e-4,
    torch.bfloat16: 1e-3,
    torch.float32: 1e-8,
    torch.float64: 1e-12,
}


def _needs_cpu_svd(device: torch.device) -> bool:
    """MPS 在部分 torch 版本上 linalg.svd 实现不完整/有 bug。"""
    return device.type == "mps"


def _state_dtype(dtype: torch.dtype) -> torch.dtype:
    """F8：状态 S 的安全存储精度。

    fp16 / bf16 输入把 S 升到 fp32（避免 eps 下溢、SVD 往返误差、降精度）；
    fp32 / fp64 原样保留（保持既有 dtype 对齐契约）。
    """
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if dtype in (torch.float32, torch.float64):
        return dtype
    raise TypeError(f"[SOMN] 不支持的状态精度: {dtype}")


def _cos_eps(dtype: torch.dtype) -> float:
    """F2/B8：eps 按实际计算 dtype 选取，1e-8 在 fp16 下会下溢为 0。"""
    return _EPS_BY_DTYPE.get(dtype, 1e-8)


def _validate_H(H_t: torch.Tensor, dim: int, where: str) -> None:
    """F1：统一输入契约。所有入口（read_only / write / forward / scores）共用。

    发现 NaN/Inf 立刻 raise 终止 —— 不准静默零激活、不准污染状态 S。
    """
    if not isinstance(H_t, torch.Tensor):
        raise TypeError(
            f"[SOMN/{where}] 需要 torch.Tensor，收到 {type(H_t).__name__}")
    if H_t.ndim != 3:
        raise ValueError(
            f"[SOMN/{where}] 输入必须是 [B,T,D]，收到 "
            f"shape={tuple(H_t.shape)} ndim={H_t.ndim}")
    if H_t.shape[-1] != dim:
        raise ValueError(
            f"[SOMN/{where}] 输入维度 {H_t.shape[-1]} != 层维度 {dim}")
    if H_t.shape[0] * H_t.shape[1] == 0:
        raise ValueError(
            f"[SOMN/{where}] 空序列不允许（B*T=0），shape={tuple(H_t.shape)}")
    if not H_t.is_floating_point():
        raise TypeError(
            f"[SOMN/{where}] 需要浮点输入，收到 dtype={H_t.dtype}")
    if not bool(torch.isfinite(H_t).all().item()):
        raise FloatingPointError(
            f"[SOMN/{where}] 输入含 NaN/Inf，已终止（拒绝污染状态 S）")


class IndependentFWLayer:
    """单层全连接快速权重，状态 S 形状 [D, D]。

    delta-rule 自联想更新，「先用后更新」：本次输出来自写入前的 S。
    S / write_count / warmup_age 都是普通 Python 属性，绝不注册进 nn.Module。
    """

    def __init__(self, dim: int, eta: float = 0.01, lam: float = 0.99):
        dim, eta, lam = int(dim), float(eta), float(lam)
        if dim <= 0:                                   # F1：状态初始化断言
            raise ValueError(f"[SOMN] dim 必须为正，收到 {dim}")
        if eta < 0.0:
            raise ValueError(f"[SOMN] eta 不能为负，收到 {eta}")
        if not (0.0 <= lam <= 1.0):
            raise ValueError(f"[SOMN] lam 须在 [0,1]，收到 {lam}")
        self.dim, self.eta, self.lam = dim, eta, lam
        self.S: Optional[torch.Tensor] = None
        self.write_count = 0
        self.warmup_age: Optional[int] = None
        self.warmup_steps: int = 0
        self.last_split_child_ratio: Optional[float] = None  # F2 观测用

    # ---- 出生特权 ----
    def begin_warmup(self, steps: int) -> None:
        self.warmup_steps = int(steps)
        self.warmup_age = 0 if self.warmup_steps > 0 else None

    @property
    def in_warmup(self) -> bool:
        return (
            self.warmup_age is not None
            and self.warmup_age < self.warmup_steps
        )

    def birth_bonus(self, strength: float = 1.0) -> float:
        if not self.in_warmup:
            return 0.0
        return float(strength) * (
            1.0 - self.warmup_age / max(1, self.warmup_steps)
        )

    def tick_warmup(self) -> None:
        if self.warmup_age is None:
            return
        self.warmup_age += 1
        if self.warmup_age >= self.warmup_steps:
            self.warmup_age = None

    # ---- F8：状态精度对齐 ----
    def _align(self, H_t: torch.Tensor) -> None:
        sd = _state_dtype(H_t.dtype)
        if self.S is None:
            self.S = torch.zeros(
                self.dim, self.dim, device=H_t.device, dtype=sd)
        elif self.S.device != H_t.device or self.S.dtype != sd:
            self.S = self.S.to(device=H_t.device, dtype=sd)

    def _to_state_dtype(self, x: torch.Tensor) -> torch.Tensor:
        """保留兼容；新路径统一用 _state_dtype(H.dtype) 作为计算精度。"""
        if self.S is not None and x.dtype != self.S.dtype:
            return x.to(self.S.dtype)
        return x

    def read_only(self, H_t: torch.Tensor) -> torch.Tensor:
        """只读重建 H_t @ S（供路由打分），不更新 S。"""
        _validate_H(H_t, self.dim, "read_only")        # F1
        if self.S is None:
            return torch.zeros_like(H_t)
        # F8/W9：计算精度由输入决定（fp16→fp32），S 临时「升」到该精度，
        # 绝不把输入降精度、也绝不修改 self.S。
        cd = _state_dtype(H_t.dtype)
        Hs = H_t if H_t.dtype == cd else H_t.to(cd)
        S = self.S
        if S.device != Hs.device or S.dtype != cd:
            S = S.to(device=Hs.device, dtype=cd)
        out = torch.matmul(Hs, S)
        return out if out.dtype == H_t.dtype else out.to(H_t.dtype)

    def write(self, H_t: torch.Tensor) -> torch.Tensor:
        """「先用后更新」，返回写入前的输出（dtype 跟随输入）。"""
        _validate_H(H_t, self.dim, "write")            # F1：含 N==0 / NaN-Inf
        B, T, D = H_t.shape
        self._align(H_t)                                # 保证 self.S 为安全精度
        cd = _state_dtype(H_t.dtype)
        x = H_t.reshape(-1, D)
        if x.dtype != cd:
            x = x.to(cd)
        S = self.S if self.S.dtype == cd else self.S.to(cd)

        out = torch.matmul(x, S)
        error = x - out
        n = x.shape[0]
        # F2：/N 的 eps 保护（F1 已保证 N>=1，此处为双保险）
        update = torch.matmul(x.transpose(0, 1), error) / (float(n) + _DIV_EPS)
        S_new = self.lam * S + self.eta * update
        # F2：状态有限性显式检查，绝不把 NaN/Inf 写进 S
        if not bool(torch.isfinite(S_new).all().item()):
            raise FloatingPointError(
                "[SOMN/write] 状态 S 出现非有限值，已中止写入（保留上一步 S）")
        self.S = torch.clamp(S_new, min=-10.0, max=10.0)
        self.write_count += 1
        out = out if out.dtype == H_t.dtype else out.to(H_t.dtype)
        return out.reshape(B, T, D)

    def get_energy(self) -> float:
        if self.S is None:
            return 0.0
        return torch.norm(self.S, p="fro").item()

    def svd_split(self) -> Optional["IndependentFWLayer"]:
        """SVD 无损拆分（奇偶交替分配）。

        F2 秩亏判定：子层取 σ₂,σ₄,…；若其最大奇异值相对 σ₁ 过小，
        说明状态近乎秩 1，生成的新层 S≈0，会在路由里被 eps 放大成
        「幽灵分」。此时**拒绝生成新层并返回 None，且不修改 self.S**——
        以「不生成」替代「剔除 σ≈0 向量」，保持 S=S_main+S_sec 无损契约。
        F7：CPU 降级路径二次保护。
        """
        if self.S is None:
            raise RuntimeError("[SOMN/svd_split] 无法拆分空层：S is None")
        S = self.S
        device, dtype = S.device, S.dtype

        work = S.detach().to("cpu") if _needs_cpu_svd(device) else S.detach()
        if work.dtype not in _SVD_SAFE_DTYPES:
            work = work.to(torch.float32)

        def _do_svd(mat):
            return torch.linalg.svd(mat, full_matrices=False)

        try:
            U, Sigma, Vh = _do_svd(work)
        except RuntimeError:
            try:                                       # F7：二次保护
                work = S.detach().to("cpu")
                if work.dtype not in _SVD_SAFE_DTYPES:
                    work = work.to(torch.float32)
                U, Sigma, Vh = _do_svd(work)
            except RuntimeError as exc:
                raise RuntimeError(
                    "[SOMN/svd_split] SVD 失败：状态 S 可能已被 NaN/Inf 污染，"
                    "请先检查输入有限性与 reset_memory()") from exc

        D = Sigma.shape[0]
        odd = torch.arange(1, D, 2, device=work.device)
        child_sigma_max = (
            float(Sigma[odd].max().item()) if odd.numel() else 0.0)
        sigma_max = float(Sigma[0].item()) if Sigma.numel() else 0.0
        ratio = child_sigma_max / sigma_max if sigma_max > 0 else 0.0
        self.last_split_child_ratio = ratio
        if sigma_max <= 0.0 or ratio < _RANK_DEFICIENT_RATIO:
            return None                                # F2：不生成幽灵层

        idx_parent = torch.arange(0, D, 2, device=work.device)
        idx_child = odd
        S_main = torch.matmul(
            torch.matmul(U[:, idx_parent], torch.diag(Sigma[idx_parent])),
            Vh[idx_parent, :],
        )
        S_sec = torch.matmul(
            torch.matmul(U[:, idx_child], torch.diag(Sigma[idx_child])),
            Vh[idx_child, :],
        )

        self.S = S_main.to(device=device, dtype=dtype)
        self.write_count = 0

        new_layer = IndependentFWLayer(self.dim, eta=self.eta, lam=self.lam)
        new_layer.S = S_sec.to(device=device, dtype=dtype)
        new_layer.write_count = 0
        return new_layer


class EnergyElbowRouter:
    """余弦重建打分 + Elbow 截断 + 出生特权强制激活。

    F2：打分在安全精度下计算（eps 按 dtype 选）、近零重建直接给 0 分、
    非有限分数立刻 raise；负分不参与激活。
    """

    def __init__(
        self,
        elbow_alpha: float = 0.6,
        min_score: float = 0.5,
        *,
        min_energy: Optional[float] = None,
        warmup_bonus: float = 1.0,
    ):
        if min_energy is not None:
            min_score = min_energy
        self.elbow_alpha = float(elbow_alpha)
        if not (0.0 < self.elbow_alpha <= 1.0):       # F1：配置断言
            raise ValueError(
                f"[SOMN] elbow_alpha 须在 (0,1]，收到 {self.elbow_alpha}")
        self.min_score = float(min_score)
        self.warmup_bonus = float(warmup_bonus)
        self.last_base_scores: List[float] = []
        self.last_boosted_scores: List[float] = []

    def scores(self, H_t: torch.Tensor,
               layers: List[IndependentFWLayer]) -> List[float]:
        """逐层余弦重建匹配度，∈ [-1, 1]；空层给中性分 0.5。"""
        out: List[float] = []
        if not layers:
            return out
        dim = layers[0].dim
        _validate_H(H_t, dim, "scores")                # F1
        # F8：路由打分统一在安全精度下算，避免 fp16 eps 下溢与降精度
        sd = _state_dtype(H_t.dtype)
        Hc = H_t if H_t.dtype == sd else H_t.to(sd)
        eps = _cos_eps(Hc.dtype)                      # F2/B8
        hn = float(Hc.norm().item())

        for layer in layers:
            if layer.S is None:
                out.append(0.5)
                continue
            read = layer.read_only(Hc)
            rn = float(read.norm().item())
            # F2/W7：近零重建不给分。绝对判据挡住零向量，相对判据挡住
            # 「S 整体缩放极小但方向仍与 H 对齐」的幽灵层（cos 会虚高到 0.9）。
            if rn < eps or rn < hn * _MIN_REBUILD_RATIO:
                out.append(0.0)
                continue
            cs = torch.cosine_similarity(Hc, read, dim=-1, eps=eps)
            out.append(float(cs.mean().item()))
        return out

    def route(
        self, H_t: torch.Tensor, layers: List[IndependentFWLayer]
    ) -> List[int]:
        self.last_base_scores = []
        self.last_boosted_scores = []
        if not layers:
            return []
        if len(layers) == 1 and layers[0].S is None:
            return [0]
        if all(layer.S is None for layer in layers):
            return [0]

        base = self.scores(H_t, layers)
        # F2：非有限分数立刻 raise，绝不静默零激活
        if not all(math.isfinite(s) for s in base):
            raise FloatingPointError(
                "[SOMN/route] 路由打分出现 NaN/Inf，已终止")
        boosted = [
            s + (l.birth_bonus(self.warmup_bonus) if l.in_warmup else 0.0)
            for s, l in zip(base, layers)
        ]
        self.last_base_scores = [round(s, 4) for s in base]
        self.last_boosted_scores = [round(s, 4) for s in boosted]

        active: List[int] = []
        top = max(boosted)
        # 负分代表「不属于该层记忆子空间」，不得激活
        if top >= self.min_score and top > 0.0:
            sorted_scores, indices = torch.sort(
                torch.tensor(boosted, dtype=torch.float32), descending=True
            )
            active.append(int(indices[0]))
            for i in range(1, len(sorted_scores)):
                prev = max(float(sorted_scores[i - 1]), 0.0)  # F1/S1 负分方向修正
                if float(sorted_scores[i]) < prev * self.elbow_alpha:
                    break
                active.append(int(indices[i]))

        for i, layer in enumerate(layers):
            if layer.in_warmup:
                active.append(i)
        return sorted(set(active))


class SOMNModule(nn.Module):
    """零参数逻辑壳，层池与全部状态都在普通字典 ``rt['somn']`` 中。

    推荐挂载：``attach_somn(model, dim)`` → 管理器落在
    ``model._rt['somn']['module']``，完全不进模块注册树。
    主干只在该键存在时调用，默认关闭 = 零侵入、state_dict 零变化。
    """

    def __init__(
        self,
        dim: int,
        rt: Optional[dict] = None,
        *,
        expand_growth_threshold: Optional[float] = None,
        max_write_count: int = 100,
        max_layers: int = _HARD_MAX_LAYERS,
        warmup_steps: int = 50,
        warmup_bonus: float = 1.0,
        persist_across_samples: bool = False,
        eta: Optional[float] = None,
        lam: float = 0.99,
        elbow_alpha: float = 0.6,
        min_score: float = 0.5,
        min_energy: Optional[float] = None,
        explore_every: int = _DEFAULT_EXPLORE_EVERY,
        async_split: bool = True,
    ):
        super().__init__()
        dim = int(dim)
        if dim <= 0:                                   # F1
            raise ValueError(f"[SOMN] dim 必须为正，收到 {dim}")
        self.dim = dim
        # delta rule 稳定条件 eta * λ_max(C) < 2；D=64 时精确等于规格 0.01。
        self.eta = float(eta) if eta is not None else 0.01 * (64.0 / self.dim)
        self.router = EnergyElbowRouter(
            elbow_alpha, min_score, min_energy=min_energy,
            warmup_bonus=warmup_bonus,
        )
        self.expand_growth_threshold = (
            float(expand_growth_threshold)
            if expand_growth_threshold is not None
            else 5.0 * (self.eta / 0.01)
        )
        self.max_write_count = int(max_write_count)
        # F3：硬上限。0 / 负值 / 超大值一律钳制到 _HARD_MAX_LAYERS，
        # 层池在任何配置下都不可能无界增长。
        requested = int(max_layers) if max_layers and int(max_layers) > 0 \
            else _HARD_MAX_LAYERS
        self.max_layers = min(requested, _HARD_MAX_LAYERS)
        self.max_splits_per_step = _MAX_SPLITS_PER_STEP   # F3：不可外部放大
        self.explore_every = max(0, int(explore_every))   # F6：0 = 禁用
        self.warmup_steps = int(warmup_steps)
        self.persist_across_samples = bool(persist_across_samples)
        self._layer_kwargs = {"eta": self.eta, "lam": float(lam)}

        rt = {} if rt is None else rt
        slot = rt.setdefault("somn", {})
        slot.setdefault(
            "layers", [IndependentFWLayer(self.dim, **self._layer_kwargs)]
        )
        stats = slot.setdefault("stats", self._new_stats())
        for key, value in self._new_stats().items():   # 兼容外部旧 slot
            stats.setdefault(key, value)
        self._rt = rt                                  # 普通 dict，绕开注册
        self._slot = slot

        # F3/P1 异步分裂队列：SVD 移出 forward 关键路径。
        # queue / Thread / list 都是普通 Python 对象，绝不注册进 nn.Module。
        self.async_split = bool(async_split)
        self._in_forward = False
        self._split_q: queue.Queue = queue.Queue(maxsize=1)
        self._split_results: queue.Queue = queue.Queue()
        self._pending: List[dict] = []
        self._split_worker_thread: Optional[threading.Thread] = None

    @staticmethod
    def _new_stats() -> Dict[str, Any]:
        return {
            "steps": 0, "splits": 0, "warmups": 0, "active_last": [],
            "idle_steps": 0,           # F6：连续零激活步数
            "explorations": 0,         # F6：强制探索次数
            "rank_deficient_skips": 0, # F2：因秩亏跳过的分裂次数
        }

    @property
    def layers(self) -> List[IndependentFWLayer]:
        return self._slot["layers"]

    def _best_layer_index(self, H_t: torch.Tensor) -> int:
        """F6：探索时挑当前得分最高的层（分数异常时退化为第 0 层）。"""
        scores = list(self.router.last_base_scores)
        if len(scores) != len(self.layers) or \
                any(not math.isfinite(s) for s in scores):
            try:
                scores = self.router.scores(H_t, self.layers)
            except (FloatingPointError, ValueError, TypeError):
                scores = [0.0] * len(self.layers)
        scores = [s if math.isfinite(s) else -float("inf") for s in scores]
        return max(range(len(scores)), key=lambda i: scores[i])

    # ---- F3/P1：异步 SVD 分裂队列 -------------------------------------
    # 实测依据（D=1024, CPU 4 线程, 2026-11-13）：
    #   同步一步 = SVD 268ms + 6 层路由 46ms = 314ms
    #   异步一步 = 主线程路由 62ms（SVD 在后台线程），主线程耗时降到 0.22×
    # 语义保证：入队时保存 S 快照；消费时用「当前 S - 快照」作为期间写入
    # 增量合并回父层，因此 S_main + S_sec + 增量 == 当前 S，拆分仍无损。
    def _ensure_worker(self) -> None:
        t = self._split_worker_thread
        if t is not None and t.is_alive():
            return
        t = threading.Thread(target=self._split_worker, daemon=True,
                             name="somn-svd-worker")
        self._split_worker_thread = t
        t.start()

    @staticmethod
    def _compute_split(snapshot, dim, eta, lam):
        tmp = IndependentFWLayer(dim, eta=eta, lam=lam)
        tmp.S = snapshot
        child = tmp.svd_split()          # 秩亏时返回 None，且不改 tmp.S
        if child is None:
            return None, None
        return tmp.S, child.S

    def _split_worker(self) -> None:
        while True:
            task = self._split_q.get()
            if task is None:
                return
            try:
                S_main, S_sec = self._compute_split(
                    task["snapshot"], self.dim, task["eta"], task["lam"])
                self._split_results.put((task, S_main, S_sec, None))
            except Exception as exc:      # 后台异常不能静默吞掉，带回主线程
                self._split_results.put((task, None, None, exc))
            finally:
                self._split_q.task_done()

    def _drain_split_results(self) -> None:
        """每个 forward 开头非阻塞地消费已完成的分裂任务。"""
        while True:
            try:
                task, S_main, S_sec, err = self._split_results.get_nowait()
            except queue.Empty:
                break
            if task in self._pending:
                self._pending.remove(task)
            if err is not None or S_main is None or S_sec is None:
                self._slot["stats"]["rank_deficient_skips"] = (
                    self._slot["stats"].get("rank_deficient_skips", 0) + 1)
                continue
            layer = task["layer"]
            # reset 之后旧层已不在层池；层池已满则丢弃结果
            if layer not in self.layers or len(self.layers) >= self.max_layers:
                continue
            cur = layer.S
            delta = None if cur is None else cur - task["snapshot"]
            layer.S = S_main if delta is None else S_main + delta
            layer.write_count = 0
            child = IndependentFWLayer(self.dim, **self._layer_kwargs)
            child.S = S_sec
            child.write_count = 0
            if self.warmup_steps > 0:
                child.begin_warmup(self.warmup_steps)
                self._slot["stats"]["warmups"] += 1
            self.layers.append(child)
            self._slot["stats"]["splits"] += 1

    # ----------------------------------------------------------------------
    def _delta(self, H_t: torch.Tensor) -> torch.Tensor:
        _validate_H(H_t, self.dim, "forward")         # F1：入口即拒绝
        # 只在训练/推理的 forward 路径启用异步；管理/测试直接调
        # _check_expand() 时保持同步，保证结果立即可观测。
        self._in_forward = True
        try:
            with torch.no_grad():
                active = self.router.route(H_t, self.layers)
                st = self._slot["stats"]

                # F6：探索机制 —— 打破「min_score 过高 → 永久零激活」死锁
                if active:
                    st["idle_steps"] = 0
                else:
                    st["idle_steps"] = st.get("idle_steps", 0) + 1
                    if self.explore_every > 0 and \
                            st["idle_steps"] >= self.explore_every:
                        active = [self._best_layer_index(H_t)]
                        st["idle_steps"] = 0
                        st["explorations"] = st.get("explorations", 0) + 1

                delta_H = torch.zeros_like(H_t)
                for idx in active:
                    delta_H = delta_H + self.layers[idx].write(H_t)
                for layer in self.layers:
                    layer.tick_warmup()
                self._check_expand()
                st["active_last"] = active
            return delta_H
        finally:
            self._in_forward = False

    def forward(self, H_t: torch.Tensor) -> torch.Tensor:
        delta_H = self._delta(H_t)
        # 残差加法留在 no_grad 之外，保证 H_t 的梯度能回传
        gate = torch.exp(
            -torch.norm(delta_H, p=2, dim=-1, keepdim=True) / self.dim
        )
        out = H_t + delta_H * gate
        self._slot["stats"]["steps"] += 1
        return out

    def _check_expand(self) -> None:
        """F3/P1：每步最多分裂 1 次；forward 路径把 SVD 交给后台队列。

        两条路径都保证「一步一分裂」与 F2 秩亏拦截：
          * forward 路径：入队快照即返回，SVD 不占关键路径；
          * 直接调用（管理/测试入口）：同步执行，结果立即可观测。
        """
        self._drain_split_results()
        if len(self.layers) + len(self._pending) >= self.max_layers:
            return
        if self._pending:                     # 已有任务在飞，本步不再入队
            return
        use_async = self.async_split and self._in_forward

        for layer in list(self.layers):
            if layer.in_warmup:
                continue
            if not (layer.write_count > self.max_write_count
                    and layer.get_energy() > self.expand_growth_threshold):
                continue

            if use_async:
                task = {"layer": layer,
                        "snapshot": layer.S.detach().clone(),
                        "eta": layer.eta, "lam": layer.lam}
                try:
                    self._split_q.put_nowait(task)
                except queue.Full:
                    return
                self._pending.append(task)
                self._ensure_worker()
                return                        # 本步分裂预算已用完

            new_layer = layer.svd_split()
            if new_layer is None:             # F2 秩亏，父层 S 未改动
                self._slot["stats"]["rank_deficient_skips"] = (
                    self._slot["stats"].get("rank_deficient_skips", 0) + 1)
                continue
            if self.warmup_steps > 0:
                new_layer.begin_warmup(self.warmup_steps)
                self._slot["stats"]["warmups"] += 1
            self.layers.append(new_layer)
            self._slot["stats"]["splits"] += 1
            return                           # F3：每步最多 1 层

    def reset(self) -> None:
        """F4：样本边界彻底复位。

        除替换 layers / stats 外，显式断开旧层的 S 张量引用（外部即使持有
        旧列表也拿不到状态），并清空 router 的打分缓存；异步队列中
        尚未消费的旧任务一并丢弃（其父层已不在新层池中，会被自动忽略）。
        """
        self._pending.clear()
        while True:
            try:
                self._split_results.get_nowait()
            except queue.Empty:
                break
        for layer in self._slot.get("layers", []):
            layer.S = None
            layer.write_count = 0
            layer.warmup_age = None
        self._slot["layers"] = [
            IndependentFWLayer(self.dim, **self._layer_kwargs)
        ]
        self._slot["stats"] = self._new_stats()
        self.router.last_base_scores = []
        self.router.last_boosted_scores = []

    def stats(self) -> Dict[str, Any]:
        return somn_stats(self)


def attach_somn(root: Any, dim: int, **kwargs) -> SOMNModule:
    """F5：幂等挂载。同 dim 返回既有管理器；异 dim 直接 raise，绝不静默错位。"""
    rt = kwargs.pop("rt", None)
    if rt is None:
        if not isinstance(getattr(root, "_rt", None), dict):
            root._rt = {}
        rt = root._rt
    slot = rt.get("somn")
    if isinstance(slot, dict) and isinstance(slot.get("module"), SOMNModule):
        mgr = slot["module"]
        layer_dims = {l.dim for l in mgr.layers}
        if mgr.dim != int(dim) or layer_dims != {int(dim)}:
            raise ValueError(
                f"[SOMN/attach] 已挂载 dim={mgr.dim}/layers={layer_dims}，"
                f"拒绝以 dim={int(dim)} 重复挂载")
        return mgr                                     # 不重建 router / 不丢配置
    mgr = SOMNModule(dim, rt=rt, **kwargs)
    rt["somn"]["module"] = mgr
    return mgr


def somn_stats(module: SOMNModule) -> Dict[str, Any]:
    """可序列化的观测快照（供日志 / 前端轮询）。"""
    st = module._slot["stats"]
    return {
        "layers": len(module.layers),
        "steps": int(st["steps"]),
        "splits": int(st["splits"]),
        "warmups": int(st["warmups"]),
        "active_last": list(st["active_last"]),
        "scores": list(module.router.last_base_scores),
        "boosted": list(module.router.last_boosted_scores),
        "warmup_ages": [l.warmup_age for l in module.layers],
        "writes": [l.write_count for l in module.layers],
        "energies": [round(l.get_energy(), 4) for l in module.layers],
        # 新增观测字段（只加键，不破坏既有调用方）
        "explorations": int(st.get("explorations", 0)),
        "idle_steps": int(st.get("idle_steps", 0)),
        "rank_deficient_skips": int(st.get("rank_deficient_skips", 0)),
        "max_layers": module.max_layers,
        "explore_every": module.explore_every,
        "async_split": module.async_split,
        "splits_pending": len(module._pending),
    }
