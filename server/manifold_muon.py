# ===========================================================================
# GARGANTUA — Manifold Muon（模块级流形约束优化器，阶段 2）
# ---------------------------------------------------------------------------
# 论文依据：
#   [1] arXiv:2606.13276 §2.1 "Different Layers, Different Manifolds"
#       （Stiefel / DGram 定义、对偶上升、retraction、HETERO 配置）
#   [2] Thinking Machines Lab, "Modular Manifolds" (2025-09-26)
#       （Manifold Muon：A_opt = -η·msign(G + 2W(Λ+Λᵀ))，dual ascent）
#
# 核心思想：在权重的切空间里求一个受流形约束的 Muon 更新方向 A，再做一次
#   retraction 把权重拉回流形（tangent-space-constrained update + retract）。
#   - Stiefel：WᵀW = I_n，σ(W) ≡ 1（谱范数由构造有界）；
#   - DGram  ：Off(WᵀW) = 0 ⇒ WᵀW 对角，但 σ 可任意增长。
#   本模块零新增 parameter、零 buffer、不注册任何 Module，不改 state_dict。
#
# ---------------------------------------------------------------------------
# 1) 默认关闭；enabled=False 时契约零变化
#    本模块只有在 mu on.build_optimizers 收到 enabled=True 的配置时才被调用；
#    enabled=False / 缺配置时优化器列表、分组、数值行为与阶段 1 之前逐位
#    一致，state_dict / config 契约完全不变。
#
# 2) 回滚契约（重要）
#    retraction 会把权重整体投影到流形上（Stiefel 把谱压到 1、DGram 抹掉
#    Gram 非对角元），**改变权重数值分布**。仅仅把开关拨回 enabled=False
#    不能回滚这种数值变化。完整的回滚必须同时恢复启用前的权重快照——见
#    common.save_checkpoint 的版本化快照机制。
#
# 3) 成本警示
#    K=20 的对偶上升每次迭代都要做一次 msign，外加一次 retraction，投影
#    开销约为标准 Muon 的 21×；报告 3.2.3 在 CPU/fp32 + block-size 32 下
#    T·m/B ≈ 160。不建议在生产环境默认启用，仅作研究开关。
#
# 4) 共享权重风险（报告 3.4.3）
#    share_enc_dec=True 时 enc/dec 逐层共享同一 Parameter，梯度会累加；
#    一次 retraction 同时服务 encoder 双向与 decoder 因果两种语义，
#    没有任何文献支撑这种「一流形两语义」的做法，明确标记为高风险。
#
# 5) KDA 不约束（报告 3.2.0 / 3.4.4）
#    论文完全没有 linear attention / delta rule / recurrent state 的概念。
#    KDA 的运行时状态 S = (I-βkkᵀ)Diag(α)S + βkvᵀ 不是 nn.Parameter，
#    对流形约束「无落点」。select_manifold_params 默认且显式跳过 kda。
# ===========================================================================
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# msign：矩阵符号函数 / 极性因子
# ---------------------------------------------------------------------------
@torch.no_grad()
def msign(X: torch.Tensor, steps: int = 5, eps: float = 1e-7,
          exact: bool = False) -> torch.Tensor:
    """返回与 X 同形的半正交矩阵（极性因子 U Vh）。

    exact=False：Newton-Schulz 五次多项式 (a,b,c)=(3.4445,-4.7750,2.0315)。
        **内部 tall→wide 转置**让 A = X@Xᵀ 落在 min 维（与 server/muon.py
        的 newton_schulz5 同一计算约定）。注意：这只是 NS 迭代的计算效率
        约定，并不改变 msign 作为极性因子的数学定义，也**不等于**流形约束
        方向——后者由 to_tall 的 wide→tall 转置决定（报告 3.4.10）。
        因此这里自包含实现一份，不 import muon，避免循环依赖。
    exact=True：U,S,Vh = svd(X) → U@Vh（精确极性因子，供测试核对）。

    fp32 计算；非有限输入直接 raise ValueError。
    """
    if not torch.is_tensor(X):
        X = torch.as_tensor(X)
    X = X.to(torch.float32)
    if X.ndim != 2:
        raise ValueError(f"msign: 需要 2D 矩阵，收到 shape {tuple(X.shape)}")
    if not bool(torch.isfinite(X).all()):
        raise ValueError("msign: 输入含非有限值 (nan/inf)")

    if exact:
        U, _, Vh = torch.linalg.svd(X, full_matrices=False)
        return U @ Vh

    a, b, c = 3.4445, -4.7750, 2.0315
    Xn = X / (X.norm() + eps)
    transposed = Xn.size(0) > Xn.size(1)
    if transposed:
        Xn = Xn.T
    for _ in range(steps):
        A = Xn @ Xn.T
        B = b * A + c * (A @ A)
        Xn = a * Xn + B @ Xn
    if transposed:
        Xn = Xn.T
    return Xn


# ---------------------------------------------------------------------------
# 切空间投影
# ---------------------------------------------------------------------------
@torch.no_grad()
def tangent_project(M: torch.Tensor, kind: str) -> torch.Tensor:
    """把矩阵投影到对应流形的切空间。

    stiefel：P_St 恒等（论文 §2.1 Stiefel 切空间约束由对偶变量自身满足），
             返回 M 本身；
    dgram  ：P_DG(M) = M - Diag(diag(M))，抹掉对角分量，只保留正交于
             「对角扰动」的部分（DGram 只约束 Gram 的对角性）。
    其它 kind raise ValueError。
    """
    if kind == "stiefel":
        return M
    if kind == "dgram":
        return M - torch.diag_embed(torch.diagonal(M, dim1=-2, dim2=-1))
    raise ValueError(f"tangent_project: 未知 kind={kind!r}")


# ---------------------------------------------------------------------------
# wide ↔ tall 转置（流形约束方向）
# ---------------------------------------------------------------------------
@torch.no_grad()
def to_tall(W: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """把 wide 矩阵转置成 tall（m>=n），返回 (W_tall, transposed)。

    **这是与现有 server/muon.py:41-49 Newton-Schulz5 相反的转置方向**
    （报告 3.4.10）：NS5 是 tall→wide，让 A=X@Xᵀ 落小维；而 Stiefel 约束
    WᵀW = I_n 要求 m>=n，wide 矩阵（out<in）若直接施加，WᵀW 是 in×in
    且秩 <= out，恒等矩阵不可能。把 wide 转置成 tall 后，实际约束变成
    W Wᵀ = I_out（约束输出侧 Gram）。调用方在 retract 之后必须用
    from_tall 还原形状。
    """
    if W.ndim != 2:
        raise ValueError(f"to_tall: 需要 2D 矩阵，收到 shape {tuple(W.shape)}")
    if W.shape[0] < W.shape[1]:
        return W.T, True
    return W, False


@torch.no_grad()
def from_tall(Wt: torch.Tensor, transposed: bool) -> torch.Tensor:
    """to_tall 的逆操作：transposed 为真时转置回来。"""
    return Wt.T if transposed else Wt


# ---------------------------------------------------------------------------
# retraction：把切空间更新后的 W̃ 拉回流形
# ---------------------------------------------------------------------------
@torch.no_grad()
def retract(W_tilde: torch.Tensor, kind: str, ns_steps: int = 5,
            exact_msign: bool = False) -> torch.Tensor:
    """收缩回流形；**输入必须是 tall 或方阵**（调用方需先 to_tall）。

    stiefel：R_St(W̃) = msign(W̃)，把全部奇异值压到 1 ⇒ WᵀW = I。
    dgram  ：SVD W̃ = U Σ Vh，返回 (U @ Vh) * Σ（Σ 按列广播乘）。
        数学理由：U Vh 列正交（(UVh)ᵀ(UVh) = VhᵀUᵀU Vh = I），故
            RᵀR = Σᵀ (UVh)ᵀ(UVh) Σ = ΣᵀΣ = diag(S²)
        是对角阵 ⇒ 满足 DGram 的 Off(WᵀW)=0；同时 RᵀR 的特征值为 S²，
        即 R 的奇异值恰为 S，被完整保留。这正对应论文
            R_DG(W̃) = (U Vᵀ) Σ
        与 Stiefel 不同：DGram **只约束 Gram 对角性、不约束谱尺度**
        （原文 "permits arbitrary singular-value growth"）。

    wide 输入直接 raise ValueError。
    """
    W = W_tilde.to(torch.float32)
    if W.ndim != 2 or W.shape[0] < W.shape[1]:
        raise ValueError(
            f"retract: 输入必须是 tall 或方阵 (m>=n)，收到 shape "
            f"{tuple(W.shape)}；调用方需先 to_tall")
    if not bool(torch.isfinite(W).all()):
        raise ValueError("retract: 输入含非有限值 (nan/inf)")

    if kind == "stiefel":
        return msign(W, steps=ns_steps, exact=exact_msign)
    if kind == "dgram":
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return (U @ Vh) * S
    raise ValueError(f"retract: 未知 kind={kind!r}")


# ---------------------------------------------------------------------------
# 切空间约束的 Muon 方向（对偶上升）
# ---------------------------------------------------------------------------
@torch.no_grad()
def manifold_muon_direction(W: torch.Tensor, G: torch.Tensor, kind: str,
                            K: int = 20, alpha: float = 0.1,
                            ns_steps: int = 5,
                            exact_msign: bool = False) -> torch.Tensor:
    """求受流形约束的 Muon 更新方向 A，返回与 W 同形的**正方向**。

    W/G 形状为 nn.Linear.weight 的 [out,in]，两者一起 to_tall（用 W 的
    transposed 标志），返回前 from_tall 还原成 [out,in]。

    与 arXiv:2606.13276 §2.1 对偶上升逐式对应：
        Λ   = -1/4 · P(WtᵀGt + GtᵀWt)
        for k in 0..K-1:
            Sym = P((Λ + Λᵀ)/2)
            A   = msign(Gt + 2·Wt·Sym)
            H   = P(WtᵀA + AᵀWt)
            Λ   = Λ - α(1 - k/K)·H
    K=0 时跳过对偶上升，退化成 A = msign(G)（标准 Muon 方向）。

    符号约定：返回**正方向** A。论文写 A_opt = -η·msign(...)，其中的负号
    由优化器用 `W - lr*A` 实现，故这里不带负号。
    全程 torch.no_grad()、fp32；非有限输入直接 raise ValueError。
    """
    if kind not in ("stiefel", "dgram"):
        raise ValueError(f"manifold_muon_direction: 未知 kind={kind!r}")
    Wt = W.to(torch.float32)
    Gt = G.to(torch.float32)
    if Wt.ndim != 2 or Wt.shape != Gt.shape:
        raise ValueError(
            f"manifold_muon_direction: W/G 必须同形 2D，收到 "
            f"{tuple(Wt.shape)} / {tuple(Gt.shape)}")
    if not bool(torch.isfinite(Wt).all()) or not bool(torch.isfinite(Gt).all()):
        raise ValueError("manifold_muon_direction: 输入含非有限值 (nan/inf)")

    Wt, transposed = to_tall(Wt)
    Gt = Gt.T if transposed else Gt

    if K <= 0:
        A = msign(Gt, steps=ns_steps, exact=exact_msign)
        return from_tall(A, transposed)

    lam = -0.25 * tangent_project(Wt.T @ Gt + Gt.T @ Wt, kind)
    A = None
    for k in range(K):
        sym = tangent_project((lam + lam.T) / 2, kind)
        A = msign(Gt + 2.0 * (Wt @ sym), steps=ns_steps, exact=exact_msign)
        H = tangent_project(Wt.T @ A + A.T @ Wt, kind)
        lam = lam - alpha * (1.0 - k / K) * H
    return from_tall(A, transposed)


# ---------------------------------------------------------------------------
# 优化器
# ---------------------------------------------------------------------------
class ManifoldMuon(torch.optim.Optimizer):
    """对命中的 2D 参数：momentum/Nesterov → 切空间约束方向 → retraction。

    groups 两种形式：
      [{"params": [...], "kind": "stiefel"}, {"params": [...], "kind": "dgram"}]
      或直接 Iterable[Tensor]（此时用 kind 关键字参数指定默认 kind）。

    更新必须用 p.copy_：保持 Parameter 对象身份，共享权重绑定
    （Gargantua.retie_shared_weights）依赖这一点，绝不能替换 Parameter。
    """

    def __init__(self, groups, lr: float = 2e-2, momentum: float = 0.95,
                 nesterov: bool = True, K: int = 20, alpha: float = 0.1,
                 ns_steps: int = 5, exact_msign: bool = False,
                 weight_decay: float = 0.0, kind: str = "stiefel"):
        groups = list(groups)
        if groups and isinstance(groups[0], dict):
            # 复制 group dict：Optimizer.add_param_group 会往里补 defaults 键，
            # 不要就地改动调用方传入的结构。
            norm_groups = [dict(g) for g in groups]
        elif groups:
            norm_groups = [{"params": groups, "kind": kind}]
        else:
            norm_groups = []
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        K=K, alpha=alpha, ns_steps=ns_steps,
                        exact_msign=exact_msign, weight_decay=weight_decay)
        super().__init__(norm_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            kind = group.get("kind", "stiefel")
            K = group["K"]
            alpha = group["alpha"]
            ns_steps = group["ns_steps"]
            exact_msign = group["exact_msign"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:        # 分组契约保证；非 2D 兜底跳过
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                u = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

                # 解耦 weight decay，口径与 Muon.step 一致（先乘衰减再更新）
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                A = manifold_muon_direction(p, u, kind, K=K, alpha=alpha,
                                            ns_steps=ns_steps,
                                            exact_msign=exact_msign)
                W_tilde = p - group["lr"] * A
                Wt, transposed = to_tall(W_tilde)
                Wt_r = retract(Wt, kind, ns_steps=ns_steps,
                               exact_msign=exact_msign)
                W_new = from_tall(Wt_r, transposed).to(p.dtype)
                # 必须 copy_：保持 Parameter 对象身份，共享权重绑定不破。
                p.copy_(W_new)
        return loss


# ---------------------------------------------------------------------------
# 参数选择（纯查询，不注册、不修改模型）
# ---------------------------------------------------------------------------
_ATTN_SUFFIXES = ("w_dq.weight", "w_uq.weight", "w_dkv.weight",
                  "w_uk.weight", "w_uv.weight", "w_o.weight")
_MLP_SUFFIXES = ("ffn_up.weight", "ffn_down.weight")


def select_manifold_params(model, cfg) -> dict[str, str]:
    """按 cfg 选出要施加流形约束的 2D 参数，返回 {name: kind}。

    - attention：含 `mla_csa.` / `mla_hca.` 且以 w_*.weight 结尾 → cfg["attention"]
    - mlp      ：含 `.ffns.` 且以 ffn_up.weight / ffn_down.weight 结尾 → cfg["mlp"]
    - kind=="none" / 非法值不收；cfg.exclude_kda（默认 True）跳过含 kda 的；
      cfg.max_matrices>0 时按 name 排序后确定性截断。
    纯查询：不注册任何东西、不修改模型。

    share_enc_dec=True 时 dec 侧与 enc 侧是同一 Parameter 但名字不同——
    用 id(p) 去重只保留首次出现的名字，避免同一个 Parameter 被两个
    优化器组重复更新（named_parameters 默认已去重，此处是显式保险）。
    """
    if not isinstance(cfg, dict):
        return {}
    attn_kind = cfg.get("attention", "none")
    mlp_kind = cfg.get("mlp", "none")
    exclude_kda = bool(cfg.get("exclude_kda", True))
    max_matrices = int(cfg.get("max_matrices", 0))

    seen_ids = set()
    picked: dict[str, str] = {}
    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue
        if exclude_kda and "kda" in name:
            continue
        if ("mla_csa." in name or "mla_hca." in name) \
                and name.endswith(_ATTN_SUFFIXES):
            kind = attn_kind
        elif ".ffns." in name and name.endswith(_MLP_SUFFIXES):
            kind = mlp_kind
        else:
            continue
        if kind not in ("stiefel", "dgram"):
            continue
        if id(p) in seen_ids:
            continue
        seen_ids.add(id(p))
        picked[name] = kind

    if max_matrices > 0:
        keep = sorted(picked)[:max_matrices]
        picked = {n: picked[n] for n in keep}
    return picked
