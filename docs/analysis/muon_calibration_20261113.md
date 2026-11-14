# Muon 谱校准实测报告 —— 阶段 0 诊断修复与三组对照（2026-11-13）

> **分支**：`analysis/manifold-muon-mhc-feasibility`（未合入主干、未 push）
> **角色**：底层修复工程师（账号 A）。上轮报告 = `docs/analysis/manifold_muon_mhc_feasibility.md`
> **本轮触发**：交叉审计指出 `reset_memory()` 排在 `diag.observe()` 之前，KDA 状态采不到。
> **证据等级**：全部数字来自本机真实训练（`torch 2.2.2` / CPU / fp32），非解析估算、非推测。

---

## 0. 结论先行（给架构师 / 账号 B 的交接）

**降 lr 后谱完全稳定 → 判定「步子太大」，流形约束彻底作废，不需要启动 Stiefel 消融，不向账号 B 移交消融任务。**

三组同种子 24 步实测，最大条件数演化：

| 组 | 配置 | cond 首→末 | 倍数 | 判定 |
|---|---|---|---|---|
| G1 | 默认 Muon `lr=2e-2` + `rms_align` | 6.50 → 83.89 | **×12.91** | 爆炸 |
| G2 | 降 `muon_lr=3e-4` + `rms_align` | 5.92 → 6.08 | **×1.027** | **稳定** |
| G3 | `lr=2e-2` + **关闭** `rms_align` | 5.95 → 5.97 | **×1.004** | **稳定** |

更强的定位：**G3 保持 `lr=2e-2` 不动，仅关掉 `rms_align` 就完全稳定** —— 元凶是
`server/muon.py:110` 的 `u = u * (max(p.shape) ** 0.5)` 把 NS5 输出放大了 `√max(shape)`
倍，而不是学习率数值本身；`lr=2e-2` 只是把放大后的步长再乘一遍。这解释了上轮
「每步更新 RMS≈0.6 ≫ 初始权重 std 0.02」的推断，且与 HETERO 论文自承的
「无 scale control 是混淆变量」完全吻合。

**因此：MLA→Stiefel / FFN→DGram / mHC 全部维持暂缓；`server/manifold_muon.py`
保持 `enabled=False`。** 后续若需要修，方向是 Muon 的尺度控制（RMS 对齐系数 / lr /
weight decay），不是给权重施加流形约束。

---

## 1. 交叉审计前提核实（一处误判，需澄清）

| 审计断言 | 核实结果 | 依据 |
|---|---|---|
| `train.py` 中 `reset_memory()` 排在 `diag.observe()` 之前 | **成立**（旧 `:139` vs `:141`） | `service/train/train.py` |
| 导致 `fast_weight._state` 被清空、`kda_state` 恒 `present=False` | **成立** | 上轮 `data/tmp/step3_diag.jsonl` 13/13 `present=False`、`fro=None` |
| 「谱诊断钩子在真实训练中一次都没执行」 | **不成立** | 上轮日志 13 行 JSONL，每行 44 个矩阵的 `spectra` 全部有值 |
| 「采集到的膨胀数据是假阳性/空数据」 | **不成立** | 见下方逐位对比 |

**逐位对比证据**（修复前 `step3_diag.jsonl` vs 修复后同配置 G1 `calib_g1.jsonl`）：

```
13 采样 × 44 矩阵 = 572 个 σ_max/cond 数值对
逐位相同 = 572    不同 = 0
kda_state.present : 旧 ['False'] → 新 ['True']
kda_state.fro     : 旧 [None, None, None] → 新 [0.007234, 0.054297, 1.11866]
```

原因：`ModelDiagnostics._collect_spectra()`（`server/manifold_diag.py`）读的是
`model.named_modules()` 里 `nn.Linear.weight`，与 `fast_weight._state` 完全无关；
只有 `_collect_kda()` 依赖后者。**所以上轮的谱膨胀是真数据，不是空数据**；审计
真正确指出的是 KDA 状态这一路指标为空。顺序缺陷照修，但结论不能因此推翻。

> 附带收益：修好后 KDA 状态终于可见，G1 的 `spectral_radius` 从 0.0025 涨到
> **1.96×10¹⁰**、`fro` 到 2.98×10¹⁰ —— 说明 Muon 的过大步长同时污染了 fast
> weight 递推状态，不只是静态权重。这是上轮看不到的新证据，进一步支持「步长」根因。

---

## 2. P0 修复清单（本轮落地）

| # | 缺陷 | 修复 | 位置 |
|---|---|---|---|
| 1 | observe 在 reset 之后，KDA 状态恒空 | `diag.observe(...)` 移到 `m.reset_memory()` **之前**；复位仍在同一 step 内、下个 forward 之前完成，数值行为不变 | `service/train/train.py:132-145` |
| 2 | `auto_train.py`（GUI 常驻队列）无诊断钩子 | 新增 env opt-in 钩子 `GARGANTUA_DIAG_EVERY` / `_LOG` / `_MAX_MATRICES`；默认关闭时**连 `manifold_diag` 都不 import**、不挂任何属性；含裁剪前 grad norm、模型热替换时 `_rebind_diag` 重挂、`try/finally` detach；observe 同样在 reset 之前 | `service/train/auto_train.py`（`_diag_settings` / `_open_diag` / `_rebind_diag` / `_train_file`） |
| 3 | `reset_memory()` 漏清 MoMW/SOMN 观测缓存 | `MoMWRouter.reset_cache()` 清 `last_raw/last_scores/last_cos_anchor/last_cos_state/last_gamma`；`EnergyElbowRouter.reset_cache()` 清 `last_base_scores/last_boosted_scores`；分别由 `MoMW.reset()`、`SOMNModule.reset()` 调用 | `server/momw.py`、`server/somn.py`、`server/model.py:reset_memory` |

**为什么缓存是死角**：`MoMW.reset()` 会 `_make_pool()` 重建 pool（pool 内的 `last_*`
随之归零），但 `self.router = MoMWRouter(D)`（`server/momw.py:714`）是 MoMW **长期持有**
的成员，不随 pool 重建；`SOMNModule.reset()` 同理只重置 `layers`/`stats`，而
`somn_stats()` 会直接读 `router.last_*`。清理下沉到各组件 `reset()`，`reset_memory()`
保持「只调度、不复制清理逻辑」的单一真源。

**校准开关（P1 支撑，默认零改动）**：`build_optimizers(..., muon_rms_align=None,
muon_weight_decay=None)`，默认 `None` 完全沿用 `Muon` 类默认值（`rms_align=True` /
`wd=0`），既有调用逐位不变；`train.py` 新增 `--no-muon-rms-align` 与
`--muon-weight-decay`。仅用于本轮对照，未写入任何生产 config。

---

## 3. 验收标准检查（全部通过）

| 验收项 | 结果 |
|---|---|
| 诊断日志中 `kda_state.present=True` | **G1/G2/G3 各 13/13 = 39/39**；`auto_train` 钩子端到端探针 4/4 |
| 有真实的奇异值记录 | 每组 13 采样 × **44 矩阵**，`sigma_top10` 长度恒为 **10**；`kda_state.spectral_radius` / `fro` 均为真实数值 |
| 三组诊断日志 | `data/tmp/calib_g{1,2,3}.jsonl`，各 13 行 |
| `auto_train` 钩子真的生效 | 端到端实测：默认关闭时 `_diag_settings() is None`；设 `GARGANTUA_DIAG_EVERY=2` 后 `Trainer._train_file()` 返回 `ok` 并写出 4 行（steps 1/2/4/6），`kda_state.present` 全 `True`、KDA `ρ(S)` 0.0026→583.6、44 矩阵 / `sigma_top10` 长度 10；`finally: detach()` 后 4 个 MLA 模块的 `diag_probe` 全部复原为 `None`。日志 `data/tmp/auto_train_diag_probe.jsonl` |

---

## 4. 三组对照完整数据

条件：`d_model=128`、1+1 blocks、`vocab_size=80262`、真实 `base1.3.0` 词表、
真实 token-6 语料 `data/tmp/step3_corpus/`（3 文档 / 474 tokens，从 `logs/infer_archive/`
筛出的合法文件）、`--steps 24 --block-size 16 --diag-every 2 --diag-max-matrices 0`、
同 seed=0、每组独立全新模型目录（不复用上轮已训练权重、不碰用户 1.3 GB 真实权重）。

### 4.1 谱与稳定性主表

| 指标（首→末） | G1 默认 | G2 降 lr | G3 关 rms_align |
|---|---|---|---|
| 最大 σ (全部矩阵) | 2.780 → 49.425（×17.78） | 2.046 → 2.102（**×1.027**） | 2.046 → 2.076（**×1.014**） |
| 最大条件数 | 6.499 → 83.888（×12.91） | 5.923 → 6.084（**×1.027**） | 5.949 → 5.974（**×1.004**） |
| `‖Off(WᵀW)‖_F` | 9.67 → 7014.69（×725.7） | 6.57 → 6.63（×1.008） | 6.57 → 6.89（×1.048） |
| MLA σ_max | 1.176 → 16.319（×13.87） | 1.097 → 1.102（×1.00） | 1.097 → 1.125（×1.03） |
| FFN σ_max | 2.780 → 49.425（×17.78） | 2.046 → 2.102（×1.03） | 2.046 → 2.076（×1.01） |
| softmax 熵 | 2.2939 → **0.2274**（塌缩） | 2.2939 → 2.2939 | 2.2939 → 2.2939 |
| logits absmax | 0.024 → **4.07×10¹⁰** | 0.024 → 0.041 | 0.024 → 0.080 |
| `‖J_softmax‖_F` | 0.3025 → 0.0715 | 0.3025 → 0.3025 | 0.3025 → 0.3025 |
| KDA `ρ(S)` | 0.0025 → **1.96×10¹⁰** | 0.0025 → 0.0118 | 0.0025 → 0.0136 |
| KDA `‖S‖_F` | 0.0072 → **2.98×10¹⁰** | 0.0072 → 0.0300 | 0.0072 → 0.0356 |
| loss | 11.4417 → 11.3202 | 11.4417 → **11.1704** | 11.4417 → 11.2253 |
| 裁剪前 grad norm | 17.98 → 4.91 | 17.98 → 9.81 | 17.98 → 9.09 |

### 4.2 真实 top-10 奇异值（每组 step 1 / step 24 中条件数最大的矩阵）

| 组 | step | 矩阵 | top-10 σ（降序） | cond |
|---|---|---|---|---|
| G1 | 1 | `dec_blocks.0.mla_hca.w_dq` | 0.593 0.578 0.559 0.551 0.527 0.521 0.511 0.507 0.497 0.495 | 6.50 |
| G1 | 24 | `enc_blocks.0.mla_hca.w_uv` | **16.18 15.75 14.37 12.46 12.36 12.14 10.66 10.02 9.79 9.54** | 83.89 |
| G2 | 1 | `dec_blocks.0.mla_hca.w_dq` | 0.533 0.522 0.510 0.508 0.503 0.498 0.492 0.486 0.483 0.476 | 5.92 |
| G2 | 24 | `dec_blocks.0.mla_hca.w_dq` | 0.545 0.532 0.527 0.514 0.510 0.506 0.498 0.493 0.489 0.486 | 6.08 |
| G3 | 1 | `dec_blocks.0.mla_hca.w_dq` | 0.533 0.522 0.510 0.508 0.503 0.498 0.491 0.485 0.483 0.476 | 5.95 |
| G3 | 24 | `enc_blocks.0.mla_hca.w_dq` | 0.653 0.639 0.631 0.617 0.612 0.604 0.599 0.588 0.584 0.580 | 5.97 |

条件数逐采样序列（每步取 44 矩阵最大值）：

- G1：`6.50 8.60 13.50 19.41 26.38 33.40 40.46 47.50 54.67 61.90 69.18 76.53 83.89`（近似线性增长，斜率 ≈ +3.43/step）
- G2：`5.92 5.93 5.95 5.97 5.99 5.99 6.00 6.01 6.07 6.09 6.10 6.11 6.08`（平）
- G3：`5.95 5.99 6.02 6.05 5.98 5.98 5.90 5.82 5.81 5.83 5.84 5.86 5.97`（平，甚至微降）

### 4.3 条件数判定的口径

`spectrum_stats()`（`server/manifold_diag.py`）用 `Gram = WᵀW` 的 `eigvalsh` 取
`sqrt(clamp≥0)` 得到 σ，不做 SVD；`cond = σ_max / max(σ_min, 1e-8·σ_max)`，秩亏时
`cond_clamped=True`。fp32 Gram 会平方条件数，小奇异值精度有限，**cond 只用于
「是否随时间膨胀」的趋势判定**，不作为绝对精度值。方法标记 `method=gram_eigvalsh_fp32`
随每行日志落盘。

---

## 5. 根因与决策

- **G1 完整复现 HETERO 的失稳链**：奇异值膨胀 → logits 膨胀 → softmax 饱和 →
  `‖J_softmax‖_F` 退化；并新增观测到 KDA 递推状态同步爆炸。
- **G2 与 G3 同时把该链条整条掐断**：谱、熵、Jacobian、KDA 状态全部持平，
  且 loss 下降更多（G2 11.1704 < G1 11.3202）。
- **G3 是关键判别组**：`lr` 不变、只关 `rms_align` 即稳定 ⇒ 放大因子
  `√max(shape)`（`server/muon.py:110`）是主因，`lr=2e-2` 是共犯。
- 按批复判据「降 lr 后谱稳定 ⇒ 证明步子太大 ⇒ 流形约束彻底作废」：**成立**。
  按「若依然爆炸 ⇒ 交给账号 B 启动 Stiefel 消融」：**不触发**。

**决策**：MLA→Stiefel / FFN→DGram / mHC 维持暂缓；KDA→DGram「未覆盖」不碰；
iMuon→SOMN「无落点」放弃；`manifold_muon.enabled` 保持 `False`。
后续唯一值得做的是给 Muon 加受控尺度（关 rms_align 或降 lr 或加 wd）并在
生产形状模型上复核，而不是引入流形约束。

---

## 6. 局限

- 24 步 / 单 seed / 全新 `d_model=128` 小模型 / 474 tokens 真实语料；是
  **校准级信号**，不是生产规模（4+6 blocks、`d_model=1024`）结论。
- 生产形状复核必须继续用 `data/tmp/` 下的一次性模型目录，避免覆盖用户权重；
  若需要更长训练，建议同时把 `GARGANTUA_DIAG_EVERY` 调大以控制 JSONL 体积。
- `rms_align` 的 `×√max(shape)` 是为大模型设计的 RMS 对齐（见 `server/muon.py`
  文件头注释）；本结论只说明**在这个小模型 + `lr=2e-2` + wd=0 的组合下它是
  失稳主因**，不等于该实现在默认 1024 宽模型上错误。

---

## 7. 复现与产物

```bash
# 三组（每组独立一次性模型目录，真实语料）
PY=/usr/local/bin/python3
$PY service/train/train.py --model data/tmp/calib_g1_default \
  --data data/tmp/step3_corpus --steps 24 --block-size 16 --optimizer muon \
  --diag --diag-every 2 --diag-max-matrices 0 \
  --diag-log data/tmp/calib_g1.jsonl --save-every 1000
# G2 追加 --muon-lr 3e-4 ; G3 追加 --no-muon-rms-align
```

| 产物 | 路径 |
|---|---|
| 三组诊断 JSONL | `data/tmp/calib_g1.jsonl`、`calib_g2.jsonl`、`calib_g3.jsonl` |
| 三组训练日志 | `data/tmp/calib_g{1,2,3}.log` |
| 上轮（修复前）对照 | `data/tmp/step3_diag.jsonl`、`step3_diag_adamw.jsonl` |
| 回归日志 | `data/tmp/auditfix_suite_20261113.log`、`auditfix_final_regression_20261113.log` |
| 一次性模型 | `data/tmp/calib_g{1,2,3}_*/`（`data/` 已被 `.gitignore:35` 忽略） |

---

*本报告所有数字均由本机真实训练产出；未修改任何生产 config，未启用任何流形约束，
未合入主干、未 push。*
