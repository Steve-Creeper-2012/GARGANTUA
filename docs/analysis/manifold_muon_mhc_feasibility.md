# mHC / Muon / 流形约束 —— GARGANTUA 工程可行性评估与落地方案

> **文档性质**：只分析、不实施的架构可行性报告。未修改任何代码文件，未运行训练，未做 benchmark。
> **评估日期**：2026-11-12　**基线**：`main @ 15a17ac`（`v4-somn`），工作区含用户未提交改动，本报告不触碰。
> **证据分级**（全文严格遵守）：
> - **[一手]** arXiv HTML/abs、官方仓库、官方博客原文，附链接与逐字引文；
> - **[代码]** `文件路径:行号` 或 `文件路径:函数名`，全部取自当前工作区实际文件；
> - **[推导]** 由上述两类事实做出的量级估算或结构推理，明确标注「需消融」，不与实测数据混淆；
> - **[无证据]** 写「检索未找到直接证据」，不填充推测；
> - **[无落点]** 现有代码库不存在该技术可挂载的对象。
>
> 第三阶段的交付约定原为「不提交、不 push」；会话末尾另有一条「新建一个分支 commit」的最终指令，按后者执行：报告提交在**新建的本地分支**上、**不 push**、保留在本地等待审阅。提交内容只含本文件。

---

## 0. 前置校正：任务书前提与证据不符之处

| # | 任务书前提 | 核验结果 | 依据 |
|---|---|---|---|
| 1 | 「DeepSeek V4 中 `n_hc` 取多少」 | V4 报告**只定义符号** `n_hc`，全文未给数值；`n=4` 是 mHC 原论文的实验扩展率，符号是 `n` 不是 `n_hc` | [一手] [arXiv:2606.19348 §2.2](https://arxiv.org/html/2606.19348v1)："The standard HC expands the width of the residual stream by a factor of n_hc"；[arXiv:2512.24880 §5.1](https://arxiv.org/html/2512.24880v2)："the expansion rate n for both HC and mHC is set to 4" |
| 2 | 「HC 出自 DeepSeek」 | HC 出自**字节跳动 Seed**；mHC 才是 DeepSeek | [一手] [arXiv:2409.19606](https://arxiv.org/abs/2409.19606) *Hyper-Connections*，Defa Zhu 等，2024-09-29；被 mHC 引为 Zhu et al. (2024) |
| 3 | 「ICML 2026 的模块化流形约束工作」 | 真实存在，但是 **WSS @ ICML 2026 workshop**（非主会），单作者，GPT-2 small / OpenWebText | [一手] [arXiv:2606.13276](https://arxiv.org/abs/2606.13276) Comments："Accepted at WSS @ ICML 2026" |
| 4 | 「DGram = 双随机 / Gram 流形」 | DGram 是 **对角 Gram 约束** `Off(WᵀW)=0`，与 mHC 的 Birkhoff 多面体是两种完全不同的流形 | [一手] [arXiv:2606.13276 §2.1](https://arxiv.org/html/2606.13276v1) |
| 5 | 「标准 Muon 替换 AdamW」 | GARGANTUA **已默认集成 Muon**（`spec.OPTIMIZER="muon"`），阶段 1 不是「替换」而是「审计 + 分组修正 + 开销实测」 | **[代码]** `server/spec.py:1394`、`server/muon.py:102`、`server/model.py:1311` |
| 6 | 「Muon 一步含 SVD，可与 SOMN 分裂合并」 | 标准 Muon 用 **5 步 Newton-Schulz 五次多项式**，刻意不做 SVD；一步两次 SVD 的前提不成立 | [一手] [Muon 博客](https://kellerjordan.github.io/posts/muon/)："it suffices to run the NS iteration for only 5 steps"；**[代码]** `server/muon.py:34` |
| 7 | 「SOMN 出生 bonus 为 `B0·e^{-age/M}`」 | SOMN 是**线性** `strength·(1 − age/steps)`；指数式在 MoMW | **[代码]** `server/somn.py:90`、`server/momw.py:210` |
| 8 | 「SOMN 插入点约 1038–1050 / 1064–1074」 | 实际 `encode` 在 `server/model.py:1046`，SOMN 在 `:1070-1073`；`decode` 在 `:1077`，SOMN 在 `:1100-1103`。`:1037` 是 `embed_audio` | **[代码]** `server/model.py:1037,1046,1070,1077,1100` |
| 9 | 「GARGANTUA 当前用 AdamW」 | 默认 `muon`（隐藏层 2D 走 Muon，其余 AdamW）；AdamW 只是三档之一 | **[代码]** `server/spec.py:1394`、`service/train/clip_data.py:39` |
| 10 | 「6.7% = 27B 模型相对基线的开销」 | 一手只写 `n=4` + in-house large-scale training；V4 补明是 **overlapped 1F1B pipeline stage 的 wall-time**。「27B / 相对基线」是二手推断 | [一手] [arXiv:2512.24880 §1/§4.3](https://arxiv.org/html/2512.24880v2)、[arXiv:2606.19348 §3.4.2](https://arxiv.org/html/2606.19348v1) |

---

## 3.1 技术证据摘要

### 3.1.1 mHC（[arXiv:2512.24880](https://arxiv.org/abs/2512.24880)，DeepSeek-AI，v1 2025-12-31 / v2 2026-01-05）

- **[一手]** 被投影的是**残差映射** `ℋ_l^res ∈ ℝ^{n×n}`；Eq.(6)：`P_{ℳ^res}(ℋ) ≔ {ℋ ∈ ℝ^{n×n} | ℋ1ₙ = 1ₙ, 1ₙᵀℋ = 1ₙᵀ, ℋ ≥ 0}`，即 Birkhoff 多面体／双随机矩阵。原句：*"we restrict ℋ_l^res to be a doubly stochastic matrix … also known as the Birkhoff polytope"*。`ℋ^pre`/`ℋ^post` 只过 Sigmoid，不受该约束。
- **[一手]** Sinkhorn-Knopp：Eq.(9) `M⁽⁰⁾ = exp(H̃^res)`、`M⁽ᵗ⁾ = 𝒯_r(𝒯_c(M⁽ᵗ⁻¹⁾))`，`ℋ^res = M⁽ᵗ_max⁾`。原句：*"We choose t_max=20 as a practical value in our experiments."*（§4.2、§5.4、Appendix A Table 5 一致）。**固定 20 次，无收敛判据、无 tolerance、无提前停止——检索未找到直接证据**。官方参考实现 `sinkhorn_normalize_ref(x, repeat=10, eps=1e-6)` 默认 **10** 次，与论文 20 不同（[TileKernels `tile_kernels/torch/mhc.py`](https://github.com/deepseek-ai/TileKernels/blob/main/tile_kernels/torch/mhc.py)）。
- **[一手]** 开销量化：§1 *"introduces only a 6.7% additional time overhead when expansion rate n=4"*；§4.3 *"marginal training overhead of only 6.7%"*；V4 §3.4.2 *"constrain the wall-time overhead of mHC to only 6.7% of the overlapped 1F1B pipeline stage"*。**硬件型号、卡数、步数、baseline 绑定：检索未找到直接证据**。Sinkhorn 迭代单独 FLOPs 数字：**检索未找到直接证据**。
- **[一手]** 稳定性：§5.4 复合增益 *"reaching a maximum value of approximately 1.6 … compared to the maximum gain magnitude of nearly 3000 in HC, mHC significantly reduces it by three orders of magnitude"*。理论性质 `‖ℋ^res‖₂ ≤ 1`、乘法闭包。
- **[一手]** Kernel 融合（§4.3.1，TileLang）：三次 mHC 系数核算子把 bias/线性投影/RMSNorm 权重合并进 `b_l`、`φ_l`；Eq.(14)-(15) 两次 scan 合成单 kernel；Eq.(16)-(18) 小系数 elementwise 合并；Eq.(19) *"We implement the Sinkhorn-Knopp iteration within a single kernel. For the backward pass, we derive a custom backward kernel that recomputes the intermediate results on-chip"*；另两个残差混合 kernel 把读量从 `(3n+1)C` 降到 `(n+1)C`、写量从 `3nC` 降到 `nC`。官方实现在 [deepseek-ai/TileKernels](https://github.com/deepseek-ai/TileKernels)（`tile_kernels/mhc/`），**不存在** `DeepSeek-AI/mHC` 仓库。
- **[一手]** 规模与增量：主实验 3B / 9B / 27B，`n=4`，gating factor init `α=0.01`（Appendix A Table 5）；`φ_l ∈ ℝ^{nC×(n²+2n)}`（Eq.(10)）。mHC 论文**未给参数增量数字**。前驱 HC 论文 Appendix B Table 7：`OLMo-1B-DHC×4  0.0003940B / 1.17715846B / +0.03349%`、`OLMo-7B-DHC×4  0.0013124B / 6.88967027B / +0.02286%`；Table 9（8 GPU 实测）：`+26.1%`、`+28.28%`、`+9.7%` 显存，*"For n=2, this contributes less than 15% to the total memory usage"*（[arXiv:2409.19606v3](https://arxiv.org/html/2409.19606v3)）。mHC §3.2 Table 2 给 I/O：HC 单残差层每 token 读 `(5n+1)C+n²+2n`、写 `(3n+1)C+n²+2n`；§4.3.2 transient memory `(n+2)C×L_r`，`L_r* ≈ √(nL/(n+2))`。
- **[一手]** 配套措施：mHC 采用 **pre-norm**，RMSNorm 在 **HC 路径内部**（Eq.(7) `x⃗'_l = RMSNorm(vec(x_l))`）；V4 §2.4 的优化器拆分：*"We maintain the AdamW optimizer for the embedding module, the prediction head module, the static biases and gating factors of mHC modules, and the weights of all RMSNorm modules. All other modules are updated with Muon."*。**gradient clipping：mHC 全文无——检索未找到直接证据**。**归一化位置 / gating / Sinkhorn 迭代数的消融，以及「去掉配套措施后流形约束单独仍有效」的实验：检索未找到直接证据**（唯一消融是 §3 Table 1 对 `ℋ^res/ℋ^pre/ℋ^post` 的开关，Absolute Loss Gap `0.0 → −0.022 → −0.025 → −0.027`）。
- **[一手]** 有限次 Sinkhorn 的已知缺陷：[arXiv:2601.05732](https://arxiv.org/abs/2601.05732) *mHC-lite*：*"finite SK iterations do not guarantee exact doubly stochasticity, leaving an approximation gap that can accumulate through network depth"*，且 *"requires highly specialized CUDA kernels"*。经典算法 Sinkhorn & Knopp (1967)，[Pacific J. Math. 21(2):343–348](https://msp.org/pjm/1967/21-2/p14.xhtml)；log 域稳定性见 [arXiv:1610.06519](https://arxiv.org/abs/1610.06519)。

### 3.1.2 模块化流形约束（[arXiv:2606.13276](https://arxiv.org/abs/2606.13276) *Different Layers, Different Manifolds*，K. Yoshihara，WSS@ICML 2026）

- **[一手]** Stiefel：`WᵀW = I`，`σ_i(W)=1`，`‖W‖₂=1` 由构造有界。DGram：`Off(WᵀW)=0` ⇒ `WᵀW = D ⪰ 0`（对角），`σ_j(W)=√d_j`，原文 *"‖W‖₂ = max √d_j is unconstrained by the manifold definition"*、**"DGram preserves Gram diagonality while permitting arbitrary singular-value growth"**。
- **[一手]** HETERO 配置（Table 1）：`Uncon. 3.3855` / `All-St. 3.3679` / `All-DG. Unst.` / **`Hetero (Attn=Stiefel, MLP=DGram) 3.3544`** / `Het.-Inv Unst.`。代码：[github.com/kiratoyoshihara/module-wise-manifold-muon](https://github.com/kiratoyoshihara/module-wise-manifold-muon)。
- **[一手]** 更新规则（§2.1）：*"We implement Manifold Muon as a tangent-space-constrained Muon update followed by a retraction."* 切空间 `P_ℳ(WᵀA + AᵀW)=0`；`P_St` 恒等、`P_DG(M)=M−Diag(diag(M))`；内层 LMO 解 `argmax = msign(B)`；对偶上升 `A_k = msign(G + 2W P_ℳ((Λ_k+Λ_kᵀ)/2))`、`Λ_0 = −¼P_ℳ(WᵀG+GᵀW)`、`H_k = P_ℳ(WᵀA_k+A_kᵀW)`、`Λ_{k+1} = Λ_k − α(1−k/K)H_k`，**`K=20`、`α=0.1`**。Stiefel 收缩 `R_St(W̃)=msign(W̃)`；DGram 收缩 `W̃=UΣVᵀ → R_DG(W̃)=(UVᵀ)Σ`（**需要一次 SVD**）。
- **[一手]** 失稳链原文：*"singular value growth → logit inflation → softmax saturation"*，并用 `J_softmax(z)=diag(p)−ppᵀ`、`‖J‖_F` 形式化梯度退化（Prop. 4.1 / 4.2）。**但原文限定**：*"This is a mechanistic explanation, not a complete proof that DGram attention is inherently unstable under all optimizer settings."* 以及 *"We do not claim that DGram attention can never be stabilized; rather, our results show that, without additional scale control or retuned hyperparameters, placing DGram on attention exposes an unstable singular-value growth pathway."*
- **[一手]** 实际记录的谱诊断（§4.1，每 1000 iter）：*"singular values, the Gram diagonal diag(WᵀW), the Frobenius norm of the off-diagonal part of WᵀW, and the Frobenius norm ‖W‖_F"*；聚合取**每组最大奇异值**。**logit max/absmax、attention entropy、gradient norm 曲线：论文未报告**——Limitations 原句 *"future work should directly measure logit scales, attention entropy, and gradient flow throughout training"*。
- **[一手]** 实验条件与混淆变量（Appendix A Table 2–5）：GPT-2 small（12 层 / 12 头 / d=768 / ctx 1024 / vocab 50,304）、OpenWebText、50,000 iter、65,536 tokens/step、bf16、`torch.compile`、全局裁剪 1.0；Manifold Muon **未对受约束权重施加 weight decay**。原句：*"For DGram-managed weights, the diagonal of WᵀW is unconstrained, so no explicit scale control is applied in our experiments. This is a potential confound: weight decay or other scale regularization may stabilize DGram attention."*
- **[一手] / [无证据]** KDA / 线性注意力 / Delta Rule / recurrent state **全文均未出现（absent）**；论文的 Attention 严格是 `Z = XW_QW_KᵀXᵀ/√d`、`p=softmax(z)` 的 GPT-2 softmax 注意力。**结论不可直接迁移到无 softmax 的线性注意力：未覆盖 / 检索未找到直接证据**。

### 3.1.3 Muon / iMuon / Mousse / 非残差先例

- **[一手]** 原始 Muon **不是 arXiv 论文**，是博客 [kellerjordan.github.io/posts/muon/](https://kellerjordan.github.io/posts/muon/)（2024-12-08）+ [github.com/KellerJordan/Muon](https://github.com/KellerJordan/Muon)。`a,b,c = (3.4445, −4.7750, 2.0315)`、`newtonschulz5(G, steps=5, eps=1e-7)`；开销 *"the FLOP overhead of Muon is at most Tm/B"*（T=5），典型训练 *"below 1%"*。原句 *"Muon is an optimizer for 2D parameters of neural network hidden layers"*、*"AdamW should be used for the embedding and final classifier head layers"*。
- **[一手]** iMuon [arXiv:2605.09238](https://arxiv.org/abs/2605.09238) *Intrinsic Muon*（2026-05-10）：*"closed-form updates on the fixed-rank, SPD, Stiefel, and Grassmann manifolds for any unitarily invariant norm"*；Corollary 4.1 给出 `Ḃ* = τ·Ortho((∇_B f)(AAᵀ)^{-1/2})(AAᵀ)^{-1/2}`、`Ȧ* = τ(BᵀB)^{-1/2}Ortho((BᵀB)^{-1/2}∇_A f)`；LoRA 用 `X=BA`（`B∈ℝ^{m×r}, A∈ℝ^{r×n}`）与 `GL(r)` 商对称。**显式 per-step FLOPs / 复杂度、条件数 / epsilon / damping 数值稳定性分析、以及 SVD-split 层或网络生长的讨论：全部 absent → 检索未找到直接证据**（仅有收敛率 `O(1/T)` 与 "rate independent of factor conditioning"）。
- **[一手]** Mousse [arXiv:2603.09697](https://arxiv.org/abs/2603.09697)（v2 2026-04-01）：*"reduces the training steps required to reach a target loss by approximately 12%, while incurring only a 3% wall-clock time overhead compared to standard Muon"*；`ΔW = −L^{−1/4} msign(L^{−1/4} G R^{−1/4}) R^{−1/4}`，`T=10` 步刷新一次预条件子（`Eigh` 两次）。测量条件：GPT-2 式 162M/247M/494M/834M、FineWeb、10,000 steps × 2M tokens = 20B tokens、**8× NVIDIA H200**、DDP、NS5 / β₁=0.95 / wd=0.01。机制上是**在白化坐标内保留 Stiefel 谱约束**（摘要：*"constraining update steps to the Stiefel manifold"*），属于融合而非可独立叠加的插件；该 Stiefel ≠ mHC 的 Birkhoff 多面体。
- **[一手]** mHC 用于非残差场景：严格意义上「mHC/HC 用于 SOMN 式动态层池 / 记忆混合」**无先例 / 检索未找到直接证据**。但存在一篇一手反例：[arXiv:2601.13953](https://arxiv.org/abs/2601.13953) *Differentiable Logic Synthesis* 原句 *"Our approach draws on recent insights from Manifold-Constrained Hyper-Connections (mHC) … We adapt this framework to logic synthesis"*，把 Birkhoff 投影用于 **Sinkhorn-constrained routing**（谱系数选择），是「mHC 思想外挂到非残差路由」的先例，但**不是**层间路由/记忆混合。Sinkhorn MoE 路由（[arXiv:2511.08972](https://arxiv.org/abs/2511.08972)、[arXiv:2606.01666](https://arxiv.org/abs/2606.01666)）属独立 OT 脉络，不是 mHC 应用。
- **[一手]** KDA 本体：[arXiv:2510.26692](https://arxiv.org/abs/2510.26692) *Kimi Linear*（2025-10-30）：*"Kimi Delta Attention (KDA), an expressive linear attention module that extends Gated DeltaNet … enabling more effective use of limited finite-state RNN memory"*，采用 DPLR 转移矩阵、*"remaining more consistent with the classical delta rule"*。

### 3.1.4 汇总：检索未找到直接证据 / 未覆盖的条目

| 问题 | 状态 |
|---|---|
| mHC Sinkhorn 收敛容差、提前停止、单独 FLOPs 数字 | **检索未找到直接证据**（只有固定 `t_max=20`） |
| mHC 依赖特定 norm 位置 / gradient clipping；去掉配套措施后流形约束单独仍有效 | **检索未找到直接证据**（论文无该消融、全文无 clipping） |
| DeepSeek-V4 的 `n_hc` 数值 | **检索未找到直接证据**（一手只有符号；`n_hc=4` 仅二手） |
| mHC 论文自身的参数增量 / 显存增量数字 | **检索未找到直接证据**（只有 HC 前驱论文 Table 7/9 有） |
| 6.7% 的 GPU 型号 / 卡数 / baseline 绑定 | **检索未找到直接证据** |
| Kernel 融合的具体加速倍数 | **检索未找到直接证据** |
| HETERO 的 logit / entropy / grad-norm 实测曲线 | **论文未报告**（Limitations 自承） |
| KDA / 线性注意力 / Delta Rule 在 HETERO 的覆盖 | **未覆盖 / 检索未找到直接证据** |
| iMuon 每步开销、数值稳定性数值分析、SVD-split 层应用 | **检索未找到直接证据** |
| Muon + SVD 层分裂 / 一步两次 SVD 成本的文献 | **检索未找到直接证据**（前提本身不成立） |
| mHC/HC 用于 SOMN 层路由 / memory mixing | **无先例 / 检索未找到直接证据**（仅有 arXiv:2601.13953 的非残差路由旁证） |

---

## 3.2 GARGANTUA 落点映射表（核心）

### 3.2.0 主表

| 技术项 | 论文/来源中的形态 | GARGANTUA 对应位置 | 可行性判断 | 阻塞点 |
|---|---|---|---|---|
| **mHC 残差混合** | 双随机矩阵 + Sinkhorn-Knopp（`ℋ^res∈ℝ^{n×n}`，`t_max=20`，`n=4`），另有 `ℋ^pre/ℋ^post` 与 `φ_l∈ℝ^{nC×(n²+2n)}` 可学习参数 | **单流**，残差累加在 `server/model.py:813`（`partial = partial + fn(h)`）；块末门 `server/model.py:842`；AttnRes 深度聚合 `server/model.py:685`、`:809-810`；`encode` `server/model.py:1046-1074`、`decode` `server/model.py:1077-1104`；SOMN 插入点 `server/model.py:1070-1073` / `:1100-1103` | **有条件，风险最高** | ① 需把 `x` 从 `[B,T,D]` 改成 `[B,T,n,D]`，`Block.forward`、`_sub`、`encode/decode`、`reset_memory`、`build_views` 全部改签名；② 新增 `φ_l/b_l/α` 参数会破坏 `build_model` 的 strict load（`server/common.py:705`）与 `config.json` 契约；③ `6.7%` 依赖 TileLang + DualPipe 1F1B 重叠（`arXiv:2512.24880 §4.3`），项目是 CPU/fp32 无该基础设施；④ 与 SOMN/MoMW 状态生命周期冲突（见 3.4.1） |
| **MLA Stiefel 约束** | `WᵀW=I` 的 Stiefel Manifold Muon（`msign` 收缩），论文对象是 GPT-2 `attn.c_attn/c_proj` | `server/model.py:300-305` 六个投影：`w_dq [256,1024]`、`w_uq [1024,256]`、`w_dkv [512,1024]`、`w_uk [1024,512]`、`w_uv [1024,512]`、`w_o [1024,1024]`；维度常数 `server/spec.py:1023-1025` | **有条件** | ① `w_dq/w_dkv` 是 wide（`out<in`），`WᵀW=I_in` **秩不足不可能**，须按论文转置成 tall → 约束变成 `WWᵀ=I_out`（不同约束）；② 论文「Attention」是单一 softmax 块，GARGANTUA 每 Block 有 3 KDA + 2 MLA 共 **48 个 2D Linear**，分组无一一对应；③ Stiefel 强制所有 σ=1，会抹掉 MLA 低秩压缩的尺度自由度；④ 共享权重下 enc/dec 梯度叠加后单次投影（见 3.4.3） |
| **FFN DGram 约束** | `Off(WᵀW)=0` 的 DGram Manifold Muon，`K=20` 对偶上升 + SVD 收缩 | `server/model.py:708-709` `GLUFFN.ffn_up [8192,1024]`、`ffn_down [1024,4096]`；每 Block 5 个独立 FFN（`server/model.py:739-740`、`server/spec.py:1012`） | **有条件（数值可行、算力不可行）** | ① Gram 为 `1024×1024`（4 MiB fp32/矩阵）；② `ffn_down` 是 wide 需转置；③ `R_DG` 每步一次 SVD + `K=20` 次 `msign`，**约 21× 现有标准 Muon**（见 3.2.3）；④ 论文自承无 scale control 是混淆变量，而本项目 weight decay 默认 0（见 3.4.9） |
| **KDA 约束** | **未覆盖**（论文全篇无 linear attention / delta rule / recurrent state） | 可学习投影 `server/model.py:203-218`；运行时状态 `S [B,H,dk,dv]` 在 `server/model.py:158-185`、`:643`、`:770-776` | **无证据 / 需消融；`S` 本身「无落点」** | ① DGram 只能约束 `w_q/w_k/w_v/w_alpha_*/w_beta/w_gate/w_o`，`S`/`conv_cache` 是运行时 Tensor（FastWeight 写回时 `.detach()`，`:665-668`）、不是 `nn.Parameter`；② 无 softmax → 论文的 logit/softmax 饱和链整体失效；③ 需另建 `ρ(S)`、`‖S‖_F` 诊断 |
| **SOMN 低秩层 + iMuon** | 固定秩流形上 `X=BA` 的闭式更新（Corollary 4.1），论文实验对象是 **LoRA 适配器** | `server/somn.py:147`（`svd_split`）、`:163`（`torch.linalg.svd`）、`:59-76`（`IndependentFWLayer.S`）、`server/somn.py:389`（`attach_somn`） | **无落点（SOMN 分支）**；另有 MLA 低秩对作为**未验证候选** | ① SOMN 的 `S` 是普通 Python 属性、**非 `nn.Parameter`**（`:72`），整段 `_delta` 在 `with torch.no_grad()` 内（`:337`），优化器根本看不到它；② 拆分产物 `S_main/S_sec` 是运行时状态，不是 `X=BA` 的可训练因子；③ iMuon 未讨论 SVD-split 层（**检索未找到直接证据**），也未给每步开销/稳定性数值；④ 生产代码中 `attach_somn` **只在测试里调用**（`service/test/test_somn.py:356,385,466,484`），默认关闭 |

> **补充落点**：GARGANTUA 确实存在 `X=BA` 形态的**可训练**低秩对 —— MLA 的 `q = w_uq(w_dq(x))`（`server/model.py:334`）隐含 `W_Q = w_uq.weight @ w_dq.weight`（rank ≤256），K/V 同理（`w_uk/w_uv` 与 `w_dkv`）。这是 iMuon 唯一可能的挂载点，但 iMuon 论文只在 LoRA 微调上验证，**检索未找到直接证据**支持把它用于预训练中的 MLA 低秩对；且会与 `param_groups_for_optimizers` 把两个 `nn.Linear` 当独立 2D 矩阵的契约冲突。

### 3.2.1 残差路径分析（对应 2.1）

**当前是单条 Residual Stream。** `Block.forward`（`server/model.py:778-844`）里 `x0 = x`、``partial = x``（`:790-791`），闭包 `_sub`（`:799-813`）执行 `partial = partial + fn(h)`（`:813`）——**标量累加的单流**。`blocks` 列表（`:792`、`:843`）是 **AttnRes 的跨层深度历史**（`block_attn_res`，`:685-691`：`V = stack(blocks + [partial])` → `softmax(norm(V)·w)ᵀV`），是「不同深度的 D 维表示」，**不是同层的 n 条并行残差流**。块末另有零初始化门 `partial += tanh(attn_res_gate) * x0`（`:842`）。因此 mHC 的 `n×n` 混合矩阵在当前代码中**无落点**。

**引入 mHC 需改的函数签名与状态变量**（全部为新增，当前代码不存在）：

| 对象 | 当前 | mHC 需要 |
|---|---|---|
| `Block.forward(x, kv_view_hca, kv_view_csa, x_step_ids, blocks, rec_state)` `server/model.py:778` | `x: [B,T,D]` | `x: [B,T,n,D]`；新增 `hc_state`（`ℋ^pre/ℋ^post/ℋ^res` 或 `φ_l/b_l/α_l`）；返回 `(partial, blocks, hc_state)` |
| `_sub(fn, is_mlp=False)` `server/model.py:799` | 单 `partial` | 需按 `ℋ^pre` 选流、`ℋ^post` 写回、`ℋ^res` 混合；10 个子层 × 每层一次 |
| `Gargantua.encode/decode` `server/model.py:1046` / `:1077` | `blocks=[x]`、`state` 字典 | 需在块循环里推进 n 流状态，`blocks` 与 n 流的语义需重新定义 |
| `Gargantua.reset_memory` `server/model.py:994` | 清 fast weight / MoMW / SOMN | 必须新增 mHC n 流复位（否则跨样本泄漏，见 3.4.1） |
| `Gargantua.forward_train` `:1195` / `forward_train_embeddings` `:1291` / `build_views` `:1106` | 假设单流 | KV 视图来自 `enc_out`，需先经 `ℋ^pre` 聚合回单流 |
| `common.build_model` `server/common.py:670`、`:705` | strict `load_state_dict` | 新参数 → 旧权重目录不可读（见 3.4.7） |

**SOMN 插入点与 mHC 不在同一位置**：SOMN 在 `encode`/`decode` 的 **`if i == 0` 分支内**（`:1070-1073` / `:1100-1103`），只在 enc/dec 各第一个 Block **之后**调用一次，与 fast weight（`:1056`/`:1088`）、MoMW（`:1063-1066`/`:1093-1096`）并列残差；mHC 的 `ℋ^res` 混合是**每个子层/每层边界**都发生（`x_{l+1}=ℋ^res x_l + ℋ^postᵀ F(ℋ^pre x_l)`，[arXiv:2512.24880 Eq.(3)](https://arxiv.org/html/2512.24880v2)）。二者粒度差 10 倍（每 Block 10 个子层）且位置不同。

三种顺序的语义差异与数值风险（**[推导]**，无文献直接证据，必须消融）：

| 顺序 | 语义 | 数值风险 | 代码阻塞点 |
|---|---|---|---|
| **SOMN → mHC**（推荐若必须共存） | SOMN 先在单流 `[B,T,D]` 上注入 `ΔH·gate`（`server/somn.py:351-354`），mHC 再把它展开/混入 n 流 | `ℋ^res` 双随机 ⇒ `‖ℋ^res‖₂ ≤ 1`（非扩张），理论上不会放大 SOMN 增量；但 mHC 的 `RMSNorm(vec(x_l))` 作用在 **nC 维**拼接向量上，单流 bonus 会被 `≈1/√n` 稀释；`α=0.01` 初期门控使影响被进一步压小，后期才显现 → **训练前期/后期行为不一致** | SOMN 要求输入最后一维 `D == self.dim`（`server/somn.py:129-130`），mHC 的 `[B,T,n,D]` 不能直接喂入 |
| **mHC → SOMN** | mHC 先产出 n 流，SOMN 再处理 | SOMN 无法消费 `[B,T,n,D]`：要么先 `ℋ^pre` 压回单流（信息已被双随机混合不可逆地混合，bonus 的层归属丢失），要么对 n 流各跑一份 SOMN（层池 ×n，`svd_split` 的 `D×D` SVD 次数 ×n） | `SOMNModule.forward` `server/somn.py:348`、`IndependentFWLayer.write` `:126` 都硬编码 `[B,T,D]` |
| **共享同一个混合矩阵** | 用 `ℋ^res` 直接充当 SOMN 层间路由权重 | **语义错误，不可行**：① `ℋ^res` 是固定 `n×n` 方阵，SOMN 层池长度随 `svd_split` **动态增长**（`:369-373`），Birkhoff 多面体要求固定方阵维度；② SOMN 路由是 `cos(H, H@S)` + Elbow + `min_score`（`:215-268`），与双随机矩阵不是同一量；③ SOMN 层是 `[D,D]` 记忆矩阵，不是残差流 | `EnergyElbowRouter.scores` `server/somn.py:215`、`SOMNModule._check_expand` `:358` |

**`_rt` 字典架构与多流状态先例**：

| 现有容器 | 代码位置 | 是否为「同层并行流」 |
|---|---|---|
| `model._rt['somn']`（`layers` / `stats` / `module`） | `server/somn.py:318-326`、`:389-401`；读取 `server/model.py:982-992` | 否（动态层池，非残差流） |
| `MoMW._rt`（`state` / `E` / `eig_v` / `writes` …） | `server/momw.py:150`；快照 `models/NAME/momw_state/`（`server/momw.py:868`） | 否（单池 + 标量计数） |
| 外部 `state` 字典（`fw` / `momw`） | `server/model.py:997-999`；`FastWeightBank.forward` `:648-673` | 否（跨 enc→dec 会话状态） |
| `rec_state`（键 `kda1/kda2/kda3` → `(S, conv_cache)`） | `server/model.py:770-776`、`:1079-1082` | **近似先例**：多槽、异构、按层名索引的普通 dict，且明确「不进 `state_dict`」 |
| `blocks`（跨层表示列表） | `server/model.py:792`、`:843` | **近似先例**：`List[Tensor]`，但语义是深度历史而非并行流 |
| `SharedKV`（full + compressed 双段） | `server/model.py:467-627` | 否（KV 归档） |

→ **结论**：`_rt` 里**有**「普通字典托管多槽运行时状态、不进 `state_dict`、由 `reset_memory()` 统一复位」的成熟先例（`somn` / `momw` / `rec_state`），所以 mHC 的 n 流状态建议放 `model._rt['mhc']['streams']`（**不注册 `nn.Module`**），并在 `Gargantua.reset_memory` `server/model.py:994` 增加复位分支。但**没有**「同层 n 条并行残差流」的先例，这是新增语义，不是已有能力。**[代码]** 注意：MoMW / SOMN 的零参数设计让 `state_dict` 完全不变；而 mHC 的 `φ_l/b_l/α_l` 是**可学习参数**，必然进入 `state_dict`，无法照搬零侵入模式。

### 3.2.2 注意力与 MLP 的模块划分（对应 2.2）

维度基线 **[代码]**：`server/spec.py:945-947` `D_MODEL=1024 / HEAD_DIM=128 / N_HEADS=8` ⇒ `inner=1024`；`:1016-1017` `KDA_HEAD_DIM=128 / KDA_EXPAND_V=1.0` ⇒ `key_dim=val_dim=1024`；`:1023-1025` `MLA_Q_LORA_RANK=256 / MLA_KV_LORA_RANK=512 / MLA_V_LORA_RANK=512`；`:1028` `FFN_HIDDEN=4096`；`:1010-1013` `BLOCK_ORDER`（10 子层 = 3 KDA + 2 MLA + 5 FFN）。

PyTorch `nn.Linear(in, out)` 的 `weight` 形状为 `[out, in]`。Stiefel `WᵀW=I_n` 要求 `m ≥ n`（高瘦/方阵）；`out < in` 时 `WᵀW` 会是 `in×in`，而秩 ≤ `out` ⇒ **恒等矩阵不可能**。论文的做法是把 wide 矩阵转置成 tall（*"we internally transpose wide matrices so that m ≥ n"*），此时实际约束的是 `WWᵀ=I_out`。

**MLA 投影矩阵清单**

| 矩阵 | 代码位置 | `weight` 形状 | `m≥n`？ | Stiefel 直接施加 | 说明 |
|---|---|---|---|---|---|
| `MLAAttention.w_dq` | `server/model.py:300` | `[256,1024]` | 否（wide） | **不可直接** | 转置后 `[1024,256]`，约束 `WWᵀ=I_256`；Gram `256²`=256 KiB |
| `w_uq` | `server/model.py:301` | `[1024,256]` | 是 | 可 | Gram `256²` |
| `w_dkv` | `server/model.py:302` | `[512,1024]` | 否（wide） | **不可直接** | 转置后 `[1024,512]`；Gram `512²`=1 MiB |
| `w_uk` | `server/model.py:303` | `[1024,512]` | 是 | 可 | K 重建，`inner=1024` |
| `w_uv` | `server/model.py:304` | `[1024,512]` | 是 | 可 | `in=MLA_V_LORA_RANK=512` 与 `c=w_dkv(kv)` 的 512 对齐（`server/model.py:335-337`） |
| `w_o` | `server/model.py:305` | `[1024,1024]` | 是（方） | 可 | 正交矩阵；Gram `1024²`=4 MiB |

**KDA：可学习权重 vs 运行时状态（明确区分）**

| 类别 | 对象 | 代码位置 | 形状 / 分组 |
|---|---|---|---|
| **可学习权重** | `w_q / w_k / w_v` | `server/model.py:203-205` | `[1024,1024]` 方阵，Muon 组 |
| | `w_alpha_down / w_alpha_up` | `:211-212` | `[128,1024]` wide / `[1024,128]` tall |
| | `w_beta` | `:214` | `[8,1024]` wide，转置后秩 8，约束几乎无意义 |
| | `w_gate / w_o` | `:217-218` | `[1024,1024]` 方阵 |
| | `conv_q/k/v.weight`、`bias` | `:129-130` | `[1024,1,4]` **3D**、`[1024]` 1D → `param_groups_for_optimizers` 判 `ndim!=2` ⇒ AdamW 组 |
| | `o_norm.norm.weight` | `:109` / `RMSNorm:98` | `[128]` 1D → AdamW 组 |
| **运行时状态（非参数）** | `S` | `kda_recurrence` `:158-185`；`FastWeightBank._state` `:643` | `[B,H,dk,dv]=[B,8,128,128]`，逐 token `S = S*α + β·k⊗(v − Sᵀk)`（`:181-183`）；FastWeight 路径写回持久状态时 `.detach()`（`:665-668`），**递推本身无 clamp**，不进 `state_dict` |
| | `conv_cache` | `:239-243`、`:770-776` | `{"q","k","v"} → [B,C,K-1]`，普通 dict |
| | `rec_state` | `:770-776`、`:1079-1082` | 按层名索引的 `(S, conv_cache)`，推理跨调用持有 |

→ DGram 只能加在**上表上半部分的可学习投影**上；`S` / `conv_cache` / `rec_state` 是运行时状态，**对它们施加流形约束无落点**。

**FFN（MLP）**

| 矩阵 | 代码位置 | `weight` 形状 | `m≥n`？ | Gram 大小 | 每步投影备注 |
|---|---|---|---|---|---|
| `GLUFFN.ffn_up` | `server/model.py:708` | `[8192,1024]`（=`FFN_HIDDEN*2`） | 是 | `WᵀW` = `1024²` = 1,048,576 元素 ≈ **4 MiB fp32** | 直接可施加；`R_DG` 需 `8192×1024` SVD |
| `GLUFFN.ffn_down` | `server/model.py:709` | `[1024,4096]` | 否（wide） | 转置后 `1024²` ≈ **4 MiB** | `WᵀW=I_4096` 不可能（秩 ≤1024），必须转置 |
| `attn_res_proj` / `mlp_res_proj` | `server/model.py:745,747` | `[1,1024]` | 否 | `1²` | 秩 1，Stiefel 退化；**当前被误分进 Muon 组**（见下） |

每 Block 5 个 FFN（`server/model.py:739-740`），10 个 Block（`N_ENC_BLOCKS=4 + N_DEC_BLOCKS=6`，`server/spec.py:991-992`）⇒ 50 个 `ffn_up` + 50 个 `ffn_down`；若每步物化全部 Gram ≈ **400 MiB/step** 量级 **[推导]**（可流式规避，但说明规模）。

**Muon 集成现状（更正任务书前提）**：GARGANTUA **已集成** `server/muon.py:34`（`newton_schulz5`，系数 `(3.4445,-4.7750,2.0315)`、`steps=5`）、`:53`（`Muon` 类）、`:102`（`build_optimizers`），默认 `spec.OPTIMIZER="muon"`（`server/spec.py:1394`）。因此**不需要「替换 AdamW」**；需要的是修正分组并评估开销：

- **[代码]** `param_groups_for_optimizers`（`server/model.py:1311-1322`）把「所有 `ndim==2` 且名字不含 `wte/lm_head/coord/type_emb/pos_emb`」划给 Muon。这与 Muon 原版「隐藏层 2D 走 Muon、embedding/head/1D 走 AdamW」大体一致，但 **`attn_res_proj.weight` / `mlp_res_proj.weight` `[1,1024]` 是 2D 秩 1 矩阵，会被送进 `newton_schulz5`**（`server/muon.py:89-90`）——对秩 1 矩阵做正交化无意义。这是引入 Manifold Muon 前必须先修的分组缺陷。
- **[代码]** `newton_schulz5` 的转置约定是 `transposed = G.size(0) > G.size(1)` → 把**高瘦转置成矮宽**，使 `A = X@Xᵀ` 落在较小维度（`server/muon.py:41-49`）；而 HETERO 是「把 wide 转置成 tall，约束 `WᵀW`」。两者 Gram 维度同为 `min(in,out)²`，但**约束方向相反**，Manifold Muon 不能盲目复用这个函数的转置逻辑。
- **共享权重**：`spec.SHARE_ENC_DEC=False` 默认分离（`server/spec.py:996`），开启时 `retie_shared_weights`（`server/model.py:967-980`）逐层绑定 `nn.Parameter`，`common.strip_tied_dec_keys` / `expand_tied_dec_keys`（`server/common.py:760-782`）镜像 `state_dict` 键。风险详见 3.4.3。

### 3.2.3 优化器现状与每步开销量级估算（对应 2.3）

**当前默认值（全部为代码实测，非推测）**

| 项 | 值 | 代码位置 |
|---|---|---|
| 默认优化器 | `muon`（混合：隐藏层 2D → Muon，其余 → AdamW） | `server/spec.py:1394`；`service/train/clip_data.py:39` `OPTIMIZERS=("muon","adamw","allmuon")` |
| AdamW lr | `3e-4` | `server/spec.py:1397`；`server/muon.py:103` |
| Muon lr | `2e-2` | `server/spec.py:1398`；`server/muon.py:103` |
| Muon momentum / NS steps | `0.95` / `5` | `server/spec.py:1395-1396`；`server/muon.py:65-68` |
| Muon weight decay | `0.0`（默认） | `server/muon.py:67` |
| AdamW 组 weight decay | `0.0`（`muon` 档）；仅 `--optimizer adamw` 时隐藏层 2D 走 `0.1` | `server/muon.py:137`、`:139` |
| 全局梯度裁剪 | `clip_grad_norm_(..., 1.0)` | `service/train/train.py:114`、`service/train/auto_train.py:487`、`service/train/clip.py:150` |
| 每步 token 数 | `--block-size 32`、B=1（`forward_train` docstring `server/model.py:1198`） | `service/train/train.py:66`、`service/train/auto_train.py:110`、`service/train/train_step.py:40` |
| 精度 / 硬件 | `DTYPE="float32"`；`torch 2.2.2（macOS x86_64 最后官方轮子）` | `server/spec.py:1424`、`:1417` |

**若「引入标准 Muon」——结论：已经是默认，无需替换。** 真正需要改的是 `param_groups_for_optimizers`（`server/model.py:1311`）与 `build_optimizers`（`server/muon.py:102`）两处参数分组，以及把 `train.py:117` / `clip.py:154` 的直连 `fast_weight.reset()` 统一到 `reset_memory()`（见 3.4.1）。

**Manifold Muon 每步额外投影开销（[推导]，量级估算，非 benchmark）**

依据：[Muon 博客](https://kellerjordan.github.io/posts/muon/) 的 `2(2nm²+m³)` FLOPs/NS 步、`T·m/B` 开销比；[arXiv:2606.13276 §2.1](https://arxiv.org/html/2606.13276v1) 的 `K=20` 对偶上升 + 一次 retraction（DGram 需完整 SVD）。

| 场景（单个 `1024×1024` 矩阵） | 每步 FLOPs 量级 | 相对现有 Muon |
|---|---|---|
| 现有 `newton_schulz5`（5 步） | `5 × 6×1024³ ≈ 3.2×10¹⁰` | 1× |
| + Stiefel Manifold Muon（20 次 `msign` + polar 收缩） | `≈ 21 × 3.2×10¹⁰ ≈ 6.8×10¹¹` | **≈21×** |
| + DGram Manifold Muon（20 次 `msign` + SVD 收缩） | 同上量级，另加一次 `UΣVᵀ` SVD 常数 | **≈21×+** |

放大到架构：每 Block **48 个 2D Linear**（3 KDA×8 + 2 MLA×6 + 5 FFN×2 + 2 AttnRes proj），10 个 Block ⇒ **≈480 个矩阵**。单个 `[8192,1024]` `ffn_up` 的 NS5 已约 `1.8×10¹¹` FLOPs/步，按 `K=20` 放大到 `≈3.7×10¹²`；只算 50 个 `ffn_up` 就是 `≈10¹⁴` FLOPs/步 **[推导]**。

**更致命的是 Muon 原版的开销比公式**：`T·m/B`（`T=5`、`m=1024`、`B`=每步 token 数）。GARGANTUA 默认 `B=32` ⇒ `5×1024/32 = 160`，即正交化 FLOPs 约为**单个隐藏层 fwd+bwd 的 160 倍**；即便把窗口放大到 `B=4096` 仍有 `1.25×`。而 HETERO 论文是 `65,536 tokens/step`、d=768、bf16、`torch.compile`、GPU（`Appendix A Table 2/3`），按同一公式为 `5×768/65536 ≈ 0.06`。**[推导]** 结论：**论文里「可忽略」的投影开销，在本项目的小窗口 + CPU/fp32 条件下是主导项**；Manifold Muon 的 20× 放大在这个基线上不可接受。所有数字均为解析量级估算，本轮**未运行 benchmark**。

---

## 3.3 推荐执行顺序（风险递增）

| 阶段 | 动作 | 验证指标 | 不达预期的回滚方案 |
|---|---|---|---|
| **阶段 0 — 零风险诊断**（不改任何训练行为，只加只读观测） | ① 在现有 `--optimizer muon` 训练里每 N 步记录 MLA 投影组（`w_dq/w_uq/w_dkv/w_uk/w_uv/w_o`，`server/model.py:300-305`）的 `σ_max`、`diag(WᵀW)`、`‖Off(WᵀW)‖_F`、`‖W‖_F`（对齐 [arXiv:2606.13276 §4.1](https://arxiv.org/html/2606.13276v1)）；② 额外记录**裁剪前**全局 grad norm（现在 `train.py:114` 裁剪到 1.0 会掩盖退化）、MLA `attn` logits 的 absmax（`server/model.py:339-347`）、softmax 熵；③ 对 KDA 记录 `ρ(S)`、`‖S‖_F`、`η·λ_max(C)`（`server/model.py:158-185`）；④ 记录 `server/muon.py` NS5 的实测 wall-clock 占比 | 连续 ≥N 步 `σ_max(W_attn)` 是否单调膨胀；`‖Off(WᵀW)‖_F` 是否增长；MLA logits absmax / 熵是否进入饱和；裁剪前 grad norm 是否塌缩；KDA `‖S‖_F` 是否有界；Muon 单步耗时占比 | **纯观测，无回滚成本**：删除日志代码即可，不影响 `state_dict`、不改变数值。建议单独分支 + 日志落 `data/tmp/` |
| **阶段 1 — 低风险单项改动**（有明确文献支撑） | **不是「替换 Muon」**（已默认）。改为：① 修 `param_groups_for_optimizers`（`server/model.py:1311`）把 `[1,1024]` 的 `attn_res_proj/mlp_res_proj` 从 Muon 组移到 AdamW 组，并显式列出 embedding / head / RMSNorm / conv 白名单；② 统一所有样本边界到 `Gargantua.reset_memory()`（`service/train/train.py:117`、`service/train/clip.py:154`）；③ 做一次 `--optimizer adamw` vs `muon` 的同种子对照，实测收敛与单步耗时 | 同种子下 `muon` 的 loss 曲线不劣于 `adamw`；单步耗时比 ≤ 预设阈值；`state_dict` 键集合与改动前**逐字一致**（回归对照）；参数分组计数可打印核对 | `git revert` 单条提交即可恢复；优化器档位是 `clip_data.OPTIMIZERS` 运行时选择（`service/train/clip_data.py:39`），切换回 `adamw` 无数据迁移；分组改动不影响已保存权重 |
| **阶段 2 — 需消融的模块化约束** | 先 **All-Stiefel**（只对 MLA 组中 tall 矩阵 `w_uq/w_uk/w_uv/w_o`），再试 **Hetero**（MLA→Stiefel、FFN→DGram）；**KDA 一律不约束**；每次只开一组、固定种子、固定窗口 | 阶段 0 的全部谱指标 + val loss + 是否触发 HETERO Appendix A 的失稳判据（`loss > max(2·loss_best, 5.0)`）；`‖Off(WᵀW)‖_F` 是否下降；`σ_max` 是否被压住；单步耗时 | 关闭约束开关（config 驱动，权重仍按普通 Muon 更新）。**注意**：一旦权重被 retraction 投影过，旧 checkpoint 的数值分布已变，**不能只关开关了事**——回滚必须同时恢复该阶段开始前的权重快照与 config（参照 `common.save_checkpoint` `server/common.py:964` 的版本化快照） |
| **阶段 3 — 高复杂度残差重构** | mHC：n 流状态 + `ℋ^pre/ℋ^post/ℋ^res` + Sinkhorn；仅在阶段 0–2 全部达标、且有 GPU/TileLang 类融合能力后启动 | 复合增益是否如论文控制在 `≈1.6`（vs HC `≈3000`）；`ℋ^res` 行/列和偏离 1 的最大值（有限 20 次 SK 的近似间隙，[arXiv:2601.05732](https://arxiv.org/abs/2601.05732)）；enc/dec 跨样本 `reset_memory` 后 logits 一致性（项目已有「不 reset 差 20.19」的回归先例，`server/model.py:1203-1204`）；显存与单步耗时 | 需要**独立的模型目录**（新 `config.json` + 新权重，旧目录 strict load 必然失败，`server/common.py:705`）；保留阶段 2 末的完整快照作为唯一可回退点；`n=1` 应能在数值上精确退化为改动前单流（作为等价性门禁） |

---

## 3.4 实际工程难点与冲突（重点）

> 以下每条都区分 **[一手] / [代码] / [推导] / [无证据]**。凡属结构性推理且无文献支撑的，一律标注需消融，不作为结论。

### 3.4.1 SOMN 与 mHC 的状态管理冲突：生命周期、重置、序列化边界

- **[代码]** SOMN 层池在 `model._rt['somn']['layers']`（`server/somn.py:318-326`），由 `reset_memory()`（`server/model.py:994-1009`）按 `persist_across_samples` 决定复位；MoMW 快照落 `models/NAME/momw_state/`（`server/momw.py:868`），不进 `weights/config` 契约；fast weight 的 `(S, conv_cache)` 在 `FastWeightBank._state`（`server/model.py:643`）或外部 `state['fw']`（`:672`）。
- **[代码] 实证缺陷**：并非所有入口都走 `reset_memory()`。`service/train/train.py:117` 与 `service/train/clip.py:154` 直接调 `m.fast_weight.reset()`；而 `service/train/auto_train.py:490`、`service/train/clip.py:105`、`service/infer/infer.py:511`、`server/model.py:1206`/`:1298` 走 `m.reset_memory()`。**若 mHC 的 n 流状态只挂在 `reset_memory()` 里，CLI 训练与 CLIP 单步这两条路径会跨样本泄漏**，复现项目已记录的「不 reset 实测 logits 差 20.19」（`server/model.py:1204`）。这是落地前必须统一的先决条件。
- **[代码]** 序列化边界不同：SOMN/MoMW 是**零参数**（`server/somn.py:11-12`、`server/momw.py:124`），`state_dict` 逐字不变；mHC 的 `φ_l/b_l/α_l` 是 `nn.Parameter`，必然新增 key → `build_model` 的 strict `load_state_dict`（`server/common.py:705`）、`split_state_dict`（`:788-807`）、`config.json` manifest（`:964-980`）四处契约全部受影响。此外对偶乘子 `Λ`、动量缓冲若在优化器内，需决定是进 checkpoint 还是按「可重建运行时数据」处理（项目已有 MoMW 快照的先例）。
- **[推导]** 生命周期错配：SOMN 层池由 `svd_split` **动态增长**（`server/somn.py:358-376`），mHC 的 `ℋ^res` 是固定 `n×n`；SOMN 是 `persist_across_samples` 可跨样本，mHC 的 n 流是样本内残差状态。二者不能共用同一复位开关，否则要么 SOMN 被误清、要么 mHC 流被误保留。**需消融，无文献直接证据**。

### 3.4.2 「SOMN 分裂 + Muon 正交化 = 一步两次 SVD」前提证伪，以及真正的重叠在哪

- **[一手]** 标准 Muon **不计算 SVD**：[Muon 博客](https://kellerjordan.github.io/posts/muon/) 明说 NS 迭代「equivalent to replacing the update by UVᵀ」，但实现是 5 步五次多项式；**[代码]** `server/muon.py:34-50` 只有 `X@X.T` 与矩阵加乘。**故「同一步两次 SVD」在当前代码里不成立 → 检索未找到直接证据支持该前提。**
- **[代码]** 真正用 SVD 的是两处：`server/somn.py:163`/`:169` 的 `torch.linalg.svd(S)`（事件触发，`write_count > max_write_count` 且能量超阈值，`:365-374`，**不是每步**），以及 [arXiv:2606.13276 §2.1](https://arxiv.org/html/2606.13276v1) 的 DGram 收缩 `W̃=UΣVᵀ`（**每步、每个受约束矩阵**）。即重叠只可能出现在**阶段 2 之后**，且只在 SOMN 恰好当步分裂时。
- **[推导] 不可合并**：两者作用在**不同张量**（SOMN 的 `S` 是 `[D,D]` 运行时状态、`with torch.no_grad()` 内、`server/somn.py:337`；Manifold Muon 的 `W` 是 `nn.Parameter`，在 `optimizer.step()` 的反向之后）、**不同阶段**（forward 内 vs backward 之后）、**可能不同设备**（SOMN 在 MPS 上显式搬 CPU，`:159-169`）。SOMN 的 SVD 还带 `detach()` 与 dtype 提升（`:159-162`）。因此不存在「复用一次分解」的优化空间；正确的优化方向是**给 SOMN 分裂加频率上限或跳过当步的 DGram retraction**，但后者会破坏流形约束，需消融。**无文献直接证据**。
- **[一手]** 相邻但不同的成本证据：[arXiv:2603.09697](https://arxiv.org/abs/2603.09697) Mousse Algorithm 1 每 `T=10` 步做两次 `Eigh`（非 SVD、非每步），§5.3 提到 single-sided 预条件可「halves the cost of the eigen-decomposition」——这是唯一可类比的摊还思路，但与本项目 SVD 无关。

### 3.4.3 Stiefel 约束与共享权重架构的冲突（**明确标记为高风险**）

- **[代码]** `share_enc_dec=True` 时 `retie_shared_weights`（`server/model.py:967-980`）让 `dec_blocks[i]` 的每个 `nn.Parameter` 指向 `enc_blocks[i]` 的同一对象；注释明确 *"autograd 把两次使用的梯度累加到同一张量"*（`:853-855`）。`SHARE_ENC_DEC=False` 为默认（`server/spec.py:996`），但选项存在。
- **[推导] 竞争机制**：Stiefel 的 retraction `W ← msign(W − ηA)` 作用于**同一个共享 Parameter 一次**，使用的 `G` 是 encoder 梯度与 decoder 梯度的**和**。因此不是「两个优化器抢同一组奇异值」，而是**一次投影同时服务于两种语义**：encoder 的 MLA 是双向无掩码（`MLAAttention(causal=is_decoder)`，`server/model.py:737-738`，`causal=False` 时 `:340` 的 mask 分支不执行），decoder 的 MLA 是步间 + 步内 token 级因果（`:315-347`）。对双向注意力合适的谱界，对因果注意力未必合适；且 `msign` 强制所有 `σ=1` 会抹掉两侧各自学出的尺度。**这是明确风险，必须在报告中单独标记**。
- **[一手] / [无证据]** [arXiv:2606.13276](https://arxiv.org/abs/2606.13276) 的实验是**单一 decoder-only GPT-2**、无 enc/dec 不对称、无共享权重；**检索未找到直接证据**说明 Manifold Muon 在共享 enc/dec 参数上的行为。最近的相关工作 [arXiv:2509.24406](https://arxiv.org/abs/2509.24406)（Muon + MLA/MoE）亦未覆盖共享权重。
- **[代码]** 状态字典连带问题：`spec.TIED_KEY_PAIRS`（`server/spec.py:1003-1004`）只镜像 `enc_blocks.` ↔ `dec_blocks.` 与两侧 final norm。mHC 若新增逐层的 `φ_l`、或 Manifold Muon 若引入逐层的 `Λ_k`，必须同步扩展这张前缀表，否则 `strip_tied_dec_keys` / `expand_tied_dec_keys`（`server/common.py:760-782`）会让 dec 侧 auxiliary 参数落盘/加载不一致。

### 3.4.4 KDA 的 Delta Rule 状态更新 × DGram 约束：没有 softmax 的「流形版 η·λ_max」问题

- **[代码]** KDA 状态 `S` 按 `S = S*α + β·k⊗(v − Sᵀk)` 更新（`server/model.py:181-183`），`α=exp(−softplus(w))`、`β=sigmoid(w)`（`:249-252`）；`q/k` 经 `F.normalize` L2 归一化（`:247-248`）；FastWeight 路径的持久状态写回时 `.detach()`（`server/model.py:665-668`），但 `kda_recurrence`（`:158-185`）与 `FastWeightBank` **均无 clamp**。对比之下 SOMN 的 `S` 有 `torch.clamp(-10, 10)`（`server/somn.py:138`）。
- **[代码]** SOMN 已踩过同类坑：`SOMNModule.__init__` 注释（`:298-301`）*"delta rule 稳定条件 eta * λ_max(C) < 2 … D=256 用 0.01 会振荡发散（cos 1.0→0.07）"*，唯一解 `eta = 0.01·64/D`，分裂阈值 `5.0·eta/0.01` 联动（`:307-310`）。
- **[推导]** 若对 KDA 投影 `w_q/w_k/w_v` 施加 **DGram**（`σ` 不受约束，[arXiv:2606.13276 Appendix C](https://arxiv.org/html/2606.13276v1)），输入协方差 `C` 的 `λ_max` 可随 `σ_max` 增长而增长，`S` 又无 clamp，等于把 SOMN 的 `η·λ_max(C)<2` 问题在**无界状态**上重演；若施加 **Stiefel**（`‖W‖₂=1`，等距），`λ_max(C)` 被压住，但会抹掉 `α/β` 门控的尺度自由度。这与论文讨论的「logit → softmax 饱和」是**不同机制**。
- **[一手] / [无证据]** HETERO 全篇无 linear attention / delta rule / recurrent state（§3.1.2 已核实 absent），**其结论不可迁移到 KDA：未覆盖 / 检索未找到直接证据**。需独立消融，诊断量应是 `ρ(S)`、`‖S‖_F`、`λ_max(C)`、`η·λ_max(C)`，而不是 attention entropy。

### 3.4.5 Sinkhorn × SOMN 出生 bonus：加性、乘性、以及稀释/放大

- **[代码] 先校正事实**：SOMN 的 `birth_bonus` 是**线性** `strength·(1 − warmup_age/warmup_steps)`（`server/somn.py:90-96`），作为**加性项加到路由分数**上（`:245-249`），并且 warm-up 期**无条件入列**、绕过 Elbow 与 `min_score`（`:264-267`）；`B0·e^{-age/M}` 的指数式在 MoMW（`server/momw.py:210-215`）。SOMN 对主干的残差贡献是 `H_t + ΔH·gate`，其中 `gate = exp(−‖ΔH‖/D)`（`server/somn.py:351-354`），是**范数依赖的乘性衰减**。
- **[一手]** mHC 的 `ℋ^res` 是**乘性**双随机混合，`‖ℋ^res‖₂ ≤ 1`（非扩张），且作用在 n 流残差状态上；`ℋ^pre = σ(H̃)`、`ℋ^post = 2σ(H̃)`、gating `α` 初始化 `0.01`（[arXiv:2512.24880 Eq.(7)-(8)](https://arxiv.org/html/2512.24880v2)、Appendix A Table 5）。
- **[推导] 交互分析**：两者**不作用于同一量**——bonus 是路由分数的加性偏置，`ℋ^res` 是残差状态的乘性收缩，因此不存在字面意义上的「相互抵消」。但有三条间接耦合：① mHC 的 `RMSNorm(vec(x_l))` 在 **nC 维**上归一化，单流 bonus 造成的范数抬升被 `≈1/√n` 稀释；② 若 `ℋ^res` 先收缩流，`‖ΔH‖` 相对变小 → `gate = exp(−‖ΔH‖/D)` 变大 → **相对放大** SOMN 的贡献；③ `α=0.01` 使 mHC 早期近似恒等，bonus 的行为随训练推进而改变，导致「warm-up 期（50 步，`server/somn.py:287`）」与「mHC 门控成熟期」不对齐。**以上为结构推导，检索未找到直接证据**（mHC 论文无任何 SOMN/动态层池实验），必须消融。
- **[一手]** 有限 20 次 Sinkhorn 本身不保证精确双随机（[arXiv:2601.05732](https://arxiv.org/abs/2601.05732)），行和偏离会随深度累积；若把该矩阵再用于路由打分，偏差会直接污染 `min_score=0.5` 阈值判断（`server/somn.py:254`）。这也是反对「共享同一混合矩阵」的量化理由。

### 3.4.6 6.7% 开销在本项目不可复现：基础设施缺位

- **[一手]** 6.7% 是 **TileLang 融合 kernel + selective recomputing + 调整后的 DualPipe 1F1B 重叠**三项共同作用的结果（[arXiv:2512.24880 §4.3](https://arxiv.org/html/2512.24880v2)、[arXiv:2606.19348 §3.4.2](https://arxiv.org/html/2606.19348v1)）；官方实现依赖 [TileKernels](https://github.com/deepseek-ai/TileKernels)（要求 CUDA SM90/SM100 或 Ascend 950、`torch>=2.13`）。
- **[代码]** 本项目 `server/spec.py:1417` 锁定 `torch 2.2.2（macOS x86_64 最后官方轮子）`、`:1424` `DTYPE="float32"`；`server/somn.py:45-56` 明确 CPU/MPS 的 SVD 降级路径。**无 CUDA kernel / 无 pipeline 并行 / 无 1F1B 重叠**。故 6.7% 不能作为本项目预算依据 **[推导]**。
- **[一手]** mHC §3.2 Table 2 的 I/O 分析显示 HC 单残差层每 token 读量随 `n` 线性增长（`(5n+1)C+n²+2n`），V4 §3.4.2 也承认 *"increases both activation memory consumption and communication volume between pipeline stages"*。在单进程 CPU 上，这部分是纯内存带宽成本，无重叠可藏。

### 3.4.7 新参数与 strict load / config 契约的硬冲突

- **[代码]** `common.build_model`（`server/common.py:670-738`）按 `config.json` 构造后 `m.load_state_dict(sd)`（`:705`），任何 missing/unexpected key 直接转成 `ValueError`（`:706-714`）；`split_state_dict`（`:788-807`）断言三组前缀不重不漏地覆盖所有 key；`save_checkpoint`（`:964-980`）按版本化快照发布。
- **[代码]** 项目已有先例：`server/common.py:691-693` 与 `server/model.py:948-951` 明确 MoMW/SOMN 之所以能零侵入，是因为**不新增任何 parameter/buffer**；mHC 做不到。
- **[项目史 / 代码外]** 长期记忆记载 1.2.0 / 1.3.0 词表改头部导致 ID 整体位移、旧权重不可读且不做兼容 —— 说明该项目对「新增 key」采取的是**不兼容重建**策略，而非迁移。落地 mHC 必须新建一个模型目录并明确声明旧权重不可读。

### 3.4.8 论文「Attention」与 GARGANTUA 的 3 KDA + 2 MLA 混合结构不对应

- **[一手]** HETERO Table 4 的 attention 组是 `attn.c_attn`、`attn.c_proj`（单个融合 QKV + 输出投影），MLP 组是 `mlp.c_fc`、`mlp.c_proj`；其失稳链依赖 `Z = XW_QW_KᵀXᵀ/√d` 与 `p=softmax(z)`。
- **[代码]** GARGANTUA 每 Block 是 `KDA×3 → FFN×3 → MLA-CSA → FFN → MLA-HCA → FFN → AttnRes`（`server/spec.py:1010-1011`、`server/model.py:816-837`），MLA 的 q/k 来自**共享 KV 流**的低秩重建（`server/model.py:334-337`）、带步内 token 级因果 mask（`:340-347`），KDA 完全没有 softmax。**「Attention → Stiefel」该映射到 6 个 MLA 矩阵、还是 6+24 个矩阵、还是全部 3 KDA 也一并约束，论文没有定义** → 需消融，**检索未找到直接证据**。
- **[一手]** 唯一可部分复现的链路是 MLA 的 `attn.softmax(-1)`（`server/model.py:348`）：可测 logits absmax、`‖J_softmax‖_F`、attention entropy（论文自己未测，Limitations 明说）。KDA 侧必须换指标（见 3.4.4）。

### 3.4.9 梯度裁剪 1.0 与 weight decay 0 会让诊断和结论双双失真

- **[代码]** 三条训练路径都做 `clip_grad_norm_(..., 1.0)`（`service/train/train.py:114`、`service/train/auto_train.py:487`、`service/train/clip.py:150`）；默认 `muon` 档两组 weight decay 均为 0（`server/muon.py:67,139`）。
- **[一手]** HETERO Appendix A 同样用 global norm 1.0 裁剪，并**明确把「无 scale control」列为混淆变量**（§3.1.2 引文）：*"weight decay or other scale regularization may stabilize DGram attention."* 本项目 wd=0 意味着这个混淆变量**完全存在**；若照搬 Hetero 配置却不加 scale control，观察到的发散不能归因于流形分配本身。
- **[推导]** 更实际的问题：梯度裁剪会**压平**「梯度退化」这一诊断信号。阶段 0 必须在 `clip_grad_norm_` **之前**记录 grad norm，否则论文 Prop. 4.2 预言的 `‖J_softmax‖_F → 0` 在本项目里观测不到。

### 3.4.10 其它次级冲突（简表）

| 冲突 | 依据 | 影响 |
|---|---|---|
| mHC `n` 固定 vs SOMN 动态层池 | [arXiv:2512.24880 Eq.(6)](https://arxiv.org/html/2512.24880v2) 要求 `ℋ^res∈ℝ^{n×n}` 固定方阵；`server/somn.py:358-376` 层池可增长 | 「共享混合矩阵」方案在数学上不成立 |
| `newton_schulz5` 转置方向与 Manifold Muon 相反 | `server/muon.py:41-49` tall→wide；[arXiv:2606.13276 §2.1](https://arxiv.org/html/2606.13276v1) wide→tall | 不能直接复用现有 NS 函数，需新实现并单测约束方向 |
| SOMN / MoMW 生产环境未挂载 | `attach_somn` 仅见于 `service/test/test_somn.py:356,385,466,484`；`Gargantua._somn_module` `server/model.py:982` 默认返回 `None` | 「两者都启用」目前只存在于测试构造，需先定义生产挂载入口与 config 开关 |
| KDA 状态无 clamp | `server/model.py:181-184` 递推无 clamp（仅 FastWeightBank 写回时 `:665-668` detach）；`server/somn.py:138` 有 clamp | 无界状态与无界 `σ`（DGram）叠加，风险高于论文设定 |
| 参数分组把秩 1 矩阵送进 Muon | `server/model.py:1311-1322`、`server/model.py:745,747` | 阶段 1 必须先修，否则 Manifold Muon 的分组基础是错的 |

---

## 3.5 结论与建议

在 GARGANTUA 当前代码状态下，**最应该先做的是阶段 0 的零风险谱诊断 + 阶段 1 的参数分组/复位统一**——因为 Muon 其实早已默认集成（`server/spec.py:1394`），而 `T·m/B = 5×1024/32 = 160` 的小窗口开销比、秩 1 投影被误送进 Muon、以及 `train.py:117` / `clip.py:154` 绕过 `reset_memory()` 这三项，都是在谈任何流形约束之前必须先量清和修好的地基。**最应该暂缓的是 mHC 残差重构**：它要求把单流改成 `[B,T,n,D]`、新增可学习参数从而打破 strict load 与 `config.json` 契约、依赖本项目没有的 TileLang/DualPipe 基础设施（6.7% 因此不可复现），还要解决与 SOMN/MoMW 的状态生命周期冲突，属于整体架构变更而非优化器改动。**当前代码明确「无落点」的是 iMuon→SOMN SVD 拆分路径与 KDA 状态约束**：SOMN 的 `S` 是 `no_grad` 下的普通 Python 属性而非 `nn.Parameter`（`server/somn.py:72,337`），KDA 的 `S` 同样不是可学习权重，且 iMuon 与 HETERO 两篇论文分别「未讨论 SVD-split 层」与「完全未覆盖线性注意力/Delta Rule」，均属**检索未找到直接证据**。MLA→Stiefel / FFN→DGram 只能列为「有条件、需消融」：前者受 wide 矩阵秩不足与共享权重梯度竞争限制，后者每步约 21× 现有 Muon 的投影开销在 CPU/fp32 上难以承受，且论文自承「无 scale control」是混淆变量而本项目 weight decay 恰为 0。

---

## 附录 A：一手来源清单

| 编号 | 来源 | 用途 |
|---|---|---|
| [arXiv:2512.24880](https://arxiv.org/abs/2512.24880) / [HTML v2](https://arxiv.org/html/2512.24880v2) | *mHC: Manifold-Constrained Hyper-Connections*, DeepSeek-AI, 2025-12-31 | Eq.(3)/(6)/(7)/(8)/(9)/(10)/(20)、§4.3.1 kernel 融合、§5.4 增益、Table 1/2/3/5、6.7% |
| [arXiv:2409.19606](https://arxiv.org/abs/2409.19606) / [HTML v3](https://arxiv.org/html/2409.19606v3) | *Hyper-Connections*, 字节跳动 Seed, 2024-09-29 | 前驱工作归属、Appendix B Table 7/9 参数/显存增量 |
| [arXiv:2606.19348](https://arxiv.org/abs/2606.19348) / [HTML v1](https://arxiv.org/html/2606.19348v1) | *DeepSeek-V4*, 2026-04-26 | `n_hc` 符号、`t_max=20`、Muon/AdamW 拆分、6.7% wall-time 条件 |
| [arXiv:2606.13276](https://arxiv.org/abs/2606.13276) / [HTML v1](https://arxiv.org/html/2606.13276v1) | *Different Layers, Different Manifolds*, WSS@ICML 2026 | Stiefel/DGram 定义、对偶上升、retraction、Table 1、谱诊断、Appendix A 混淆变量 |
| [Muon 博文](https://kellerjordan.github.io/posts/muon/) / [GitHub](https://github.com/KellerJordan/Muon) | Keller Jordan 等，2024-12-08 | NS 系数/步数、`T·m/B` 开销公式、参数分组契约 |
| [arXiv:2605.09238](https://arxiv.org/abs/2605.09238) | *Intrinsic Muon*, 2026-05-10 | 固定秩闭式解、Corollary 4.1、LoRA `X=BA` |
| [arXiv:2603.09697](https://arxiv.org/abs/2603.09697) / [HTML v2](https://arxiv.org/html/2603.09697v2) | *Mousse*, 2026-03-10 | 12%/3%、白化坐标更新、`T=10`、实验条件 |
| [arXiv:2510.26692](https://arxiv.org/abs/2510.26692) | *Kimi Linear*, 2025-10-30 | KDA = 线性注意力 / Delta Rule / finite-state RNN memory |
| [arXiv:2601.05732](https://arxiv.org/abs/2601.05732) | *mHC-lite*, 2026-01-09 | 有限 SK 迭代的近似间隙与 kernel 工程门槛 |
| [arXiv:2601.13953](https://arxiv.org/abs/2601.13953) | *Differentiable Logic Synthesis*, 2026-01-20 | mHC 思想外挂到非残差 Sinkhorn 路由的唯一一手旁证 |
| [deepseek-ai/TileKernels](https://github.com/deepseek-ai/TileKernels) | 官方 mHC kernel 仓库 | `tile_kernels/mhc/`、`sinkhorn_normalize_ref(repeat=10)`、TileLang/CUDA 依赖 |
| [module-wise-manifold-muon](https://github.com/kiratoyoshihara/module-wise-manifold-muon) | HETERO 论文 Comments 给出的代码仓库 | Hetero 配置实现 |
| [Modular Manifolds](https://thinkingmachines.ai/blog/modular-manifolds/) / [manifolds](https://github.com/thinking-machines-lab/manifolds/) | Jeremy Bernstein, Thinking Machines Lab, 2025-09-26 | Manifold Muon 原始定义：Stiefel 流形 `WᵀW=Iₙ`、`A_opt=−η·msign(G+2W(Λ+Λᵀ))`、`W←msign(W)`，dual ascent 求解 |
| [arXiv:2504.17577](https://arxiv.org/abs/2504.17577) | *TileLang* | mHC 融合 kernel 的底层框架 |
| [arXiv:2409.20325](https://arxiv.org/abs/2409.20325) | *Old Optimizer, New Norm: An Anthology*, Bernstein & Newhouse, 2024-09-30 | 谱范数最速下降与「按张量角色分配不同范数」的理论源头 |
| [arXiv:2502.16982](https://arxiv.org/abs/2502.16982) | *Muon is Scalable for LLM Training* (Moonlight) | GARGANTUA `server/muon.py:5-6` 引用的 RMS 对齐来源 |
| [arXiv:2509.24406](https://arxiv.org/abs/2509.24406) | *Muon: Training and Trade-offs with Latent Attention and MoE*, 2025-09-29 | Muon×MLA 相邻工作（30M–200M decoder-only，未覆盖共享权重 / KDA） |
| [arXiv:1610.06519](https://arxiv.org/abs/1610.06519) | *Stabilized Sparse Scaling Algorithms…*, Schmitzer, 2016 | Sinkhorn log 域稳定化 / epsilon-scaling |
| [arXiv:2511.08972](https://arxiv.org/abs/2511.08972) / [arXiv:2606.01666](https://arxiv.org/abs/2606.01666) | *Selective Sinkhorn Routing* / *DOT-MoE* (ICML 2026) | 独立 OT 脉络的 Sinkhorn 路由，非 mHC 应用 |

## 附录 B：代码引用索引（本报告全部引用均已逐行核对）

| 文件 | 关键行 / 函数 |
|---|---|
| `server/model.py` | `:92` `RMSNorm`、`:104` `FusedRMSNormGated`、`:117-146` `CausalDepthwiseConv1d`、`:158-185` `kda_recurrence`、`:188-260` `KDAAttention`（`:203-218` 投影）、`:271-285` `within_step_pos`、`:288-350` `MLAAttention`（`:300-305` 投影、`:339-348` softmax）、`:467-627` `SharedKV`、`:628-673` `FastWeightBank`（`:643` `_state`、`:665-668` detach）、`:685-691` `block_attn_res`、`:703-713` `GLUFFN`、`:716-844` `Block`（`:726-730` 顺序断言、`:778` `forward`、`:799-813` `_sub`、`:813` 残差累加、`:842` 块末门）、`:859-865` `tie_module_parameters`、`:882-965` `Gargantua.__init__`、`:967-980` `retie_shared_weights`、`:982-992` `_somn_module`、`:994-1009` `reset_memory`、`:1046-1074` `encode`、`:1077-1104` `decode`、`:1106` `build_views`、`:1195-1218` `forward_train`/`logits`、`:1291-1304` `forward_train_embeddings`、`:1311-1322` `param_groups_for_optimizers` |
| `server/spec.py` | `:945-947` 维度、`:963-988` `check_d_model`、`:991-992` block 数、`:996` `SHARE_ENC_DEC`、`:1003-1004` `TIED_KEY_PAIRS`、`:1010-1013` `BLOCK_ORDER`、`:1016-1020` KDA 常数、`:1023-1025` MLA rank、`:1028` `FFN_HIDDEN`、`:1035` `ATTNRES_ENABLED`、`:1394-1398` 优化器常数、`:1417` / `:1424` 运行时精度 |
| `server/muon.py` | `:34-50` `newton_schulz5`、`:53-99` `Muon`、`:102-140` `build_optimizers` |
| `server/somn.py` | `:59-76` `IndependentFWLayer`、`:90-104` 出生特权、`:126-140` `write`（`:138` clamp）、`:147-189` `svd_split`、`:192-268` `EnergyElbowRouter`、`:271-386` `SOMNModule`（`:298-310` eta/阈值、`:318-326` `_rt`、`:337` `no_grad`、`:348-356` `forward`、`:358-376` `_check_expand`、`:378-383` `reset`）、`:389-401` `attach_somn` |
| `server/momw.py` | `:124` 零参数声明、`:150` `_rt`、`:210-215` 指数 bonus、`:868` 快照 |
| `server/common.py` | `:670-738` `build_model`（`:705` strict load）、`:742-782` 共享键镜像、`:788-807` `split_state_dict`、`:964-980` `save_checkpoint`、`:1008` `validate_block_size` |
| `service/train/train.py` | `:66-71` 默认参数、`:84-85` 构建优化器、`:114` 梯度裁剪、`:117` `fast_weight.reset()` |
| `service/train/auto_train.py` | `:110` 默认 `block_size=32`、`:459-460` 构建优化器、`:487` 梯度裁剪、`:490` `reset_memory()` |
| `service/train/clip.py` | `:105` `reset_memory()`、`:110-138` `clip_loss`、`:141-155` `run_clip_step`（`:150` 裁剪、`:154` `fast_weight.reset()`） |
| `service/train/clip_data.py` | `:36` `MODES`、`:39` `OPTIMIZERS` |
| `service/train/train_step.py` | `:40-54` `forward_window` |
| `service/infer/infer.py` | `:505-511` `reset_memory()` 与外部 `state` 契约 |
| `service/test/test_somn.py` | `:356,385,466,484` 唯一 `attach_somn` 调用点 |

---

*报告结束。本轮未修改任何代码、未运行训练或 benchmark、未提交除本文件外的任何改动。*
