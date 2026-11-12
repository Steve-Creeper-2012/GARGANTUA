# ===========================================================================
# GARGANTUA v4-somn — server/somn.py  (第二版：动态演化)
# SOMN (Self-Organizing Memory Network)
# ---------------------------------------------------------------------------
# 无监督、纯自组织、跨平台（CPU / CUDA / MPS）的全连接单头快速权重记忆。
#
# 硬约束（不可破）：
#   1. 零外部依赖 —— 只用 torch（无 sklearn / numpy / scipy）。
#   2. 全平台 —— SVD 统一 torch.linalg.svd，矩阵乘统一 torch.matmul；
#      不碰任何 N 卡专属算子。MPS 上 SVD 暂存 CPU，算完搬回。
#   3. 绝对零污染 —— 层池 List 与状态 S 全部存在普通字典 _rt['somn'] 里，
#      绝不注册进 nn.Module；主模型 state_dict 逐字不变。
#   4. dtype / device 自动跟随输入 H_t。
#
# v2 动态演化（架构师 2026-11-12 批复）：
#   A. 路由打分废弃 ||H@S|| 范数，改余弦重建匹配
#      score = mean(cosine_similarity(H_t, H_t @ S))。
#      S 是自联想矩阵：输入落在该层记忆子空间 → 重建近似自身 → cos→1；
#      随机白噪声只保留低秩子空间分量 → cos→0。
#   B. SVD 拆分改奇偶交替分配（父拿 σ1,σ3,σ5…，子拿 σ2,σ4,σ6…），
#      仍然 S = S_main + S_sec 无损，但两层初始能量/表达力对等，
#      不再出现「新层出生即被 Elbow 判死刑」。
#   C. 出生特权 Warm-up：新层前 N 步无条件进入 active（绕过 Elbow 与
#      min_score），分数加线性衰减 bonus 1.0→0.0，且豁免分裂判定，
#      直到积累起自己的 S 再参与自由竞争。
#
# v1 已修的规格缺陷（保留）：以 x@S 为准、delta_H 空集保护、
#   路由先于写入的 dtype/device 临时对齐、no_grad 只包记忆更新。
# ===========================================================================
from __future__ import annotations

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


def _needs_cpu_svd(device: torch.device) -> bool:
    """MPS 在部分 torch 版本上 linalg.svd 实现不完整/有 bug。

    这里只做设备判定，不做硬件探测，因此在纯 CPU 机器上也可被测试覆盖。
    """
    return device.type == "mps"


class IndependentFWLayer:
    """单层全连接快速权重，状态 S 形状 [D, D]。

    delta-rule 自联想更新，「先用后更新」：本次输出来自写入前的 S，
    写入只影响下一次调用（与项目 FastWeightBank 语义一致）。

    S / write_count / warmup_age 都是普通 Python 属性，绝不注册进 nn.Module。
    """

    def __init__(self, dim: int, eta: float = 0.01, lam: float = 0.99):
        self.dim = int(dim)
        self.eta = float(eta)
        self.lam = float(lam)
        self.S: Optional[torch.Tensor] = None
        self.write_count = 0
        # 出生特权：None = 不在 warm-up；否则是已存活步数
        self.warmup_age: Optional[int] = None
        self.warmup_steps: int = 0

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
        """加分项从 1.0 线性衰减到 0.0（同 MoMW 的 B0·e^{-age/M} 语义）。"""
        if not self.in_warmup:
            return 0.0
        return float(strength) * (
            1.0 - self.warmup_age / max(1, self.warmup_steps)
        )

    def tick_warmup(self) -> None:
        """每个全局步推进一次；到 N 步毕业，回归自由竞争。"""
        if self.warmup_age is None:
            return
        self.warmup_age += 1
        if self.warmup_age >= self.warmup_steps:
            self.warmup_age = None

    def _align(self, H_t: torch.Tensor) -> None:
        """状态随输入 H_t 的 device / dtype 自动对齐。"""
        if self.S is None:
            self.S = torch.zeros(
                self.dim, self.dim, device=H_t.device, dtype=H_t.dtype
            )
        elif self.S.device != H_t.device or self.S.dtype != H_t.dtype:
            self.S = self.S.to(device=H_t.device, dtype=H_t.dtype)

    def read_only(self, H_t: torch.Tensor) -> torch.Tensor:
        """只读重建 H_t @ S（供路由打分），不更新 S。"""
        if self.S is None:
            return torch.zeros_like(H_t)
        # 路由先于 write 执行，此时 S 可能仍是上一步的 dtype/device。
        # 用临时视图对齐，绝不改 self.S（保持只读语义）。
        S = self.S
        if S.device != H_t.device or S.dtype != H_t.dtype:
            S = S.to(device=H_t.device, dtype=H_t.dtype)
        return torch.matmul(H_t, S)

    def write(self, H_t: torch.Tensor) -> torch.Tensor:
        """「先用后更新」，返回写入前的输出。"""
        B, T, D = H_t.shape
        if D != self.dim:
            raise ValueError(f"输入维度 {D} != 层维度 {self.dim}")
        x = H_t.reshape(-1, D)
        self._align(H_t)

        out = torch.matmul(x, self.S)
        error = x - out
        update = torch.matmul(x.transpose(0, 1), error) / x.shape[0]
        self.S = self.lam * self.S + self.eta * update
        self.S = torch.clamp(self.S, min=-10.0, max=10.0)
        self.write_count += 1
        return out.reshape(B, T, D)

    def get_energy(self) -> float:
        if self.S is None:
            return 0.0
        return torch.norm(self.S, p="fro").item()

    def svd_split(self) -> "IndependentFWLayer":
        """SVD 无损拆分（奇偶交替分配）：

        按奇异值降序，父层取索引 0,2,4…（σ1,σ3,σ5…），
        新层取索引 1,3,5…（σ2,σ4,σ6…）。两者能量近似对等，
        且 S_main + S_sec 精确等于原 S。
        """
        if self.S is None:
            raise RuntimeError("无法拆分空层：S is None")
        S = self.S
        device, dtype = S.device, S.dtype

        work = S.detach().to("cpu") if _needs_cpu_svd(device) else S.detach()
        if work.dtype not in _SVD_SAFE_DTYPES:
            work = work.to(torch.float32)
        try:
            U, Sigma, Vh = torch.linalg.svd(work, full_matrices=False)
        except RuntimeError:
            # 其他后端偶发不支持时统一降级 CPU 重算
            work = S.detach().to("cpu")
            if work.dtype not in _SVD_SAFE_DTYPES:
                work = work.to(torch.float32)
            U, Sigma, Vh = torch.linalg.svd(work, full_matrices=False)

        D = Sigma.shape[0]
        idx_parent = torch.arange(0, D, 2, device=work.device)
        idx_child = torch.arange(1, D, 2, device=work.device)
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

    打分 score = mean(cos(H_t, H_t @ S))，度量「输入是否属于该层记忆
    子空间」，而不是响应幅度。空层给中性分 0.5，冷启动另有强制写入。
    """

    def __init__(
        self,
        elbow_alpha: float = 0.6,
        min_score: float = 0.5,
        *,
        min_energy: Optional[float] = None,
        warmup_bonus: float = 1.0,
    ):
        if min_energy is not None:  # 兼容 v1 的 min_energy 关键字
            min_score = min_energy
        self.elbow_alpha = float(elbow_alpha)
        self.min_score = float(min_score)
        self.warmup_bonus = float(warmup_bonus)
        self.last_base_scores: List[float] = []
        self.last_boosted_scores: List[float] = []

    def scores(self, H_t: torch.Tensor,
               layers: List[IndependentFWLayer]) -> List[float]:
        """逐层余弦重建匹配度，∈ [-1, 1]。"""
        out: List[float] = []
        for layer in layers:
            if layer.S is None:
                out.append(0.5)  # 空层中性分
                continue
            read = layer.read_only(H_t)
            out.append(
                float(torch.cosine_similarity(
                    H_t, read, dim=-1, eps=1e-8
                ).mean().item())
            )
        return out

    def route(
        self, H_t: torch.Tensor, layers: List[IndependentFWLayer]
    ) -> List[int]:
        self.last_base_scores = []
        self.last_boosted_scores = []
        if not layers:
            return []
        # 冷启动：唯一一层尚未成型，强制写入
        if len(layers) == 1 and layers[0].S is None:
            return [0]
        # 防御：多层但全空时打分恒为 0.5，仍强制写第 0 层避免死锁
        if all(layer.S is None for layer in layers):
            return [0]

        base = self.scores(H_t, layers)
        boosted = [
            s + (l.birth_bonus(self.warmup_bonus) if l.in_warmup else 0.0)
            for s, l in zip(base, layers)
        ]
        self.last_base_scores = [round(s, 4) for s in base]
        self.last_boosted_scores = [round(s, 4) for s in boosted]

        active: List[int] = []
        if max(boosted) >= self.min_score:
            sorted_scores, indices = torch.sort(
                torch.tensor(boosted, dtype=torch.float32), descending=True
            )
            active.append(int(indices[0]))
            for i in range(1, len(sorted_scores)):
                if sorted_scores[i] < sorted_scores[i - 1] * self.elbow_alpha:
                    break
                active.append(int(indices[i]))

        # 出生特权：warm-up 期新层无条件入列，绕过 Elbow 与 min_score
        for i, layer in enumerate(layers):
            if layer.in_warmup:
                active.append(i)
        return sorted(set(active))


class SOMNModule(nn.Module):
    """零参数逻辑壳，层池与全部状态都在普通字典 ``rt['somn']`` 中。

    推荐挂载：``attach_somn(model, dim)`` → 管理器落在
    ``model._rt['somn']['module']``，完全不进模块注册树。
    主干只在该键存在时调用，默认关闭 = 零侵入。
    """

    def __init__(
        self,
        dim: int,
        rt: Optional[dict] = None,
        *,
        expand_growth_threshold: Optional[float] = None,
        max_write_count: int = 100,
        max_layers: int = 0,
        warmup_steps: int = 50,
        warmup_bonus: float = 1.0,
        persist_across_samples: bool = False,
        eta: Optional[float] = None,
        lam: float = 0.99,
        elbow_alpha: float = 0.6,
        min_score: float = 0.5,
        min_energy: Optional[float] = None,
    ):
        super().__init__()
        self.dim = int(dim)
        # delta rule 稳定条件 eta * λ_max(C) < 2；结构化（低秩相关）输入下
        # λ_max(C) ~ O(D)。步长按维度反比缩放，D=64 时精确等于规格 0.01，
        # D=1024 时 0.000625。实测 D=256 用 0.01 会振荡发散（cos 1.0→0.07）。
        self.eta = float(eta) if eta is not None else 0.01 * (64.0 / self.dim)
        self.router = EnergyElbowRouter(
            elbow_alpha, min_score, min_energy=min_energy,
            warmup_bonus=warmup_bonus,
        )
        # ‖S‖ 与 eta 同量级，分裂阈值随步长联动，避免高维下永不分裂。
        self.expand_growth_threshold = (
            float(expand_growth_threshold)
            if expand_growth_threshold is not None
            else 5.0 * (self.eta / 0.01)
        )
        self.max_write_count = int(max_write_count)
        self.max_layers = int(max_layers)      # 0 = 不限制；防御无上界增长
        self.warmup_steps = int(warmup_steps)
        self.persist_across_samples = bool(persist_across_samples)
        self._layer_kwargs = {"eta": self.eta, "lam": float(lam)}

        rt = {} if rt is None else rt
        slot = rt.setdefault("somn", {})
        slot.setdefault(
            "layers", [IndependentFWLayer(self.dim, **self._layer_kwargs)]
        )
        slot.setdefault("stats", self._new_stats())
        # 普通 dict 属性，绕开 nn.Module.__setattr__ 的模块注册
        self._rt = rt
        self._slot = slot

    @staticmethod
    def _new_stats() -> Dict[str, Any]:
        return {"steps": 0, "splits": 0, "warmups": 0, "active_last": []}

    @property
    def layers(self) -> List[IndependentFWLayer]:
        return self._slot["layers"]

    def _delta(self, H_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            active = self.router.route(H_t, self.layers)
            delta_H = torch.zeros_like(H_t)  # 不能是 int 0
            for idx in active:
                delta_H = delta_H + self.layers[idx].write(H_t)
            for layer in self.layers:
                layer.tick_warmup()
            self._check_expand()
            self._slot["stats"]["active_last"] = active
        return delta_H

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
        if self.max_layers and len(self.layers) >= self.max_layers:
            return
        # 快照迭代：循环体内会 append 新层
        for layer in list(self.layers):
            if layer.in_warmup:
                continue  # 出生特权期不参与分裂，先把自己的 S 养起来
            if (
                layer.write_count > self.max_write_count
                and layer.get_energy() > self.expand_growth_threshold
            ):
                new_layer = layer.svd_split()
                if self.warmup_steps > 0:
                    new_layer.begin_warmup(self.warmup_steps)
                    self._slot["stats"]["warmups"] += 1
                self.layers.append(new_layer)
                self._slot["stats"]["splits"] += 1
                if self.max_layers and len(self.layers) >= self.max_layers:
                    break

    def reset(self) -> None:
        """样本边界调用：清空层池回到 1 层 S=None，不触碰 state_dict。"""
        self._slot["layers"] = [
            IndependentFWLayer(self.dim, **self._layer_kwargs)
        ]
        self._slot["stats"] = self._new_stats()

    def stats(self) -> Dict[str, Any]:
        return somn_stats(self)


def attach_somn(root: Any, dim: int, **kwargs) -> SOMNModule:
    """把 SOMN 挂到 root 的普通字典 ``_rt`` 上，返回不在模块树里的管理器。

    ``root._rt['somn']['module']`` 是主干取用的可调用入口。
    """
    rt = kwargs.pop("rt", None)
    if rt is None:
        if not isinstance(getattr(root, "_rt", None), dict):
            root._rt = {}
        rt = root._rt
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
    }
