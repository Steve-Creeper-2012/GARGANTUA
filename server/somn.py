# ===========================================================================
# GARGANTUA v4-somn — server/somn.py
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
# 与规格的三处偏差（都是修复，理由见 test_somn.py 的证据用例）：
#   a) read_only 改用 H_t @ S，与 write 的 x @ S 统一。规格给的
#      F.linear(H_t, S) 实际是 H_t @ S.T，会让路由能量与实际写入脱节。
#   b) delta_H 初始化为 torch.zeros_like(H_t)，不是 Python int 0。
#      路由返回 [] 时规格版会在 torch.norm(0) 处抛 TypeError。
#   c) 记忆更新整体包在 torch.no_grad() 内，但残差加法留在外面，
#      保证 H_t 的梯度仍能回传（否则会截断主模型训练图）。
#   另加 max_layers 可选上限、svd_split 继承 eta/lam、低精度 SVD 升 fp32。
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

    S / write_count 都是普通 Python 属性，绝不注册进 nn.Module。
    """

    def __init__(self, dim: int, eta: float = 0.01, lam: float = 0.99):
        self.dim = int(dim)
        self.eta = float(eta)
        self.lam = float(lam)
        self.S: Optional[torch.Tensor] = None
        self.write_count = 0

    def _align(self, H_t: torch.Tensor) -> None:
        """状态随输入 H_t 的 device / dtype 自动对齐。"""
        if self.S is None:
            self.S = torch.zeros(
                self.dim, self.dim, device=H_t.device, dtype=H_t.dtype
            )
        elif self.S.device != H_t.device or self.S.dtype != H_t.dtype:
            self.S = self.S.to(device=H_t.device, dtype=H_t.dtype)

    def read_only(self, H_t: torch.Tensor) -> torch.Tensor:
        """只读计算响应（供路由打分），不更新 S。"""
        if self.S is None:
            return torch.zeros_like(H_t)
        # 路由先于 write 执行，此时 S 可能仍是上一步的 dtype/device。
        # 用临时视图对齐，绝不改 self.S（保持只读语义）。
        S = self.S
        if S.device != H_t.device or S.dtype != H_t.dtype:
            S = S.to(device=H_t.device, dtype=H_t.dtype)
        # 与 write() 使用同一个读矩阵。规格里的 F.linear(H_t, S) == H_t @ S.T。
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
        """SVD 无损拆分：S = S_main + S_sec，父层留 Top-k，新层收剩余成分。"""
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

        k = self.dim // 2
        S_main = torch.matmul(
            torch.matmul(U[:, :k], torch.diag(Sigma[:k])), Vh[:k, :]
        )
        S_sec = torch.matmul(
            torch.matmul(U[:, k:], torch.diag(Sigma[k:])), Vh[k:, :]
        )

        self.S = S_main.to(device=device, dtype=dtype)
        self.write_count = 0

        new_layer = IndependentFWLayer(self.dim, eta=self.eta, lam=self.lam)
        new_layer.S = S_sec.to(device=device, dtype=dtype)
        new_layer.write_count = 0
        return new_layer


class EnergyElbowRouter:
    """按各层只读响应范数排序，遇到能量「肘部」跌破 alpha 倍即截断。"""

    def __init__(self, elbow_alpha: float = 0.6, min_energy: float = 1e-4):
        self.elbow_alpha = float(elbow_alpha)
        self.min_energy = float(min_energy)

    def route(
        self, H_t: torch.Tensor, layers: List[IndependentFWLayer]
    ) -> List[int]:
        if not layers:
            return []
        # 冷启动：唯一一层尚未成型，强制写入
        if len(layers) == 1 and layers[0].S is None:
            return [0]
        # 防御：多层但全空时能量恒为 0，会「永远不写」死锁
        if all(layer.S is None for layer in layers):
            return [0]

        energies = [
            torch.norm(layer.read_only(H_t), p=2, dim=-1).mean().item()
            for layer in layers
        ]
        if max(energies) < self.min_energy:
            return []

        sorted_energies, indices = torch.sort(
            torch.tensor(energies, dtype=torch.float32), descending=True
        )
        active_indices = [int(indices[0])]
        for i in range(1, len(sorted_energies)):
            if sorted_energies[i] < sorted_energies[i - 1] * self.elbow_alpha:
                break
            active_indices.append(int(indices[i]))
        return active_indices


class SOMNModule(nn.Module):
    """零参数逻辑壳，层池与全部状态都在普通字典 ``rt['somn']`` 中。

    推荐挂载：``model._rt['somn']['module'] = attach_somn(model, dim)``
    —— 完全不进模块注册树。即便误挂成 ``model.somn = mgr``，因为内部
    零 Parameter / 零 buffer，state_dict 也不会多出任何键。
    """

    def __init__(
        self,
        dim: int,
        rt: Optional[dict] = None,
        *,
        expand_growth_threshold: float = 5.0,
        max_write_count: int = 100,
        max_layers: int = 0,
        eta: float = 0.01,
        lam: float = 0.99,
        elbow_alpha: float = 0.6,
        min_energy: float = 1e-4,
    ):
        super().__init__()
        self.dim = int(dim)
        self.router = EnergyElbowRouter(elbow_alpha, min_energy)
        self.expand_growth_threshold = float(expand_growth_threshold)
        self.max_write_count = int(max_write_count)
        self.max_layers = int(max_layers)  # 0 = 不限制；防御无上界增长
        self._layer_kwargs = {"eta": float(eta), "lam": float(lam)}

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
        return {"steps": 0, "splits": 0, "active_last": []}

    @property
    def layers(self) -> List[IndependentFWLayer]:
        return self._slot["layers"]

    def _delta(self, H_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            active = self.router.route(H_t, self.layers)
            delta_H = torch.zeros_like(H_t)  # 不能是 int 0
            for idx in active:
                delta_H = delta_H + self.layers[idx].write(H_t)
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
            if (
                layer.write_count > self.max_write_count
                and layer.get_energy() > self.expand_growth_threshold
            ):
                self.layers.append(layer.svd_split())
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
    """把 SOMN 挂到 root 的普通字典 ``_rt`` 上，返回不在模块树里的管理器。"""
    rt = kwargs.pop("rt", None)
    if rt is None:
        if not isinstance(getattr(root, "_rt", None), dict):
            root._rt = {}
        rt = root._rt
    return SOMNModule(dim, rt=rt, **kwargs)


def somn_stats(module: SOMNModule) -> Dict[str, Any]:
    """可序列化的观测快照（供日志 / 前端轮询）。"""
    st = module._slot["stats"]
    return {
        "layers": len(module.layers),
        "steps": int(st["steps"]),
        "splits": int(st["splits"]),
        "active_last": list(st["active_last"]),
        "writes": [l.write_count for l in module.layers],
        "energies": [round(l.get_energy(), 4) for l in module.layers],
    }
