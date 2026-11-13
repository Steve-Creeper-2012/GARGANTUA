# SOMN V4 对抗性压测与代码审计报告

| 项目 | 内容 |
| --- | --- |
| 分支 | `feat/somn-stress-test`（基于 `6bd2bc5` / `analysis/manifold-muon-mhc-feasibility`） |
| 日期 | 2026-11-13 |
| 角色 | 对抗性测试工程师 / 代码审计员 |
| 目标 | **证明它哪里会崩**，不是证明它能跑 |
| 审计对象 | `server/somn.py`（419 行，与 HEAD 一致）、账号 A 的未提交修复代码 |
| 产物 | `test/somn_stress.py`、`docs/audit/somn_v4_audit.md`（本文件）；**未修改任何主干业务代码** |
| 原始数据 | `data/tmp/somn_stress.log`（完整输出）、`data/tmp/somn_stress_result.json`（结构化结果） |

---

## 0. 前提与口径（先说清楚，避免误读）

### 0.1 实测环境（非估算）
| 项 | 值 |
| --- | --- |
| 解释器 | `/usr/local/bin/python3` = Python 3.12.10 |
| torch | 2.2.2，`torch.get_num_threads()=4`，**CPU only**（`cuda=False` / `mps=False`） |
| 平台 | macOS 15.7.7 / x86_64（Intel Mac） |
| 内存采样 | `psutil 7.2.2` → `Process.memory_info().rss`，**每步采样取峰值** |
| 延迟采样 | `time.perf_counter()`，逐步记录 avg / median / p95 / max |
| 数据来源 | 全部来自 `test/somn_stress.py` 的一次完整真实运行（2026-11-13 15:28:16），无一条数字来自推算 |

### 0.2 账号 A 的分支状态 —— 重要前提校正
任务书要求「等 A 提交 `fix/optimizer-and-reset` 后拉下来检查」。**该分支不存在**：`git branch -a` 只有 `main`、`v4-somn`、`analysis/manifold-muon-mhc-feasibility` 与本次新建的 `feat/somn-stress-test`。

账号 A 的修复代码以 **40 项未提交工作区改动** 的形式存在于同一工作区（`server/muon.py`、`server/model.py`、`server/common.py`、`server/momw.py` 的 diff，以及新增的 `server/manifold_diag.py`、`server/manifold_muon.py`、`service/test/test_manifold_20261113.py` 等）。因此：

- 本报告的交叉审计基于**工作区快照**，不是远端分支；
- 全程 **未执行** `git add` / `commit` / `checkout --` / `stash` 触碰这些改动，40 项在切分支前后数量一致；
- 若 A 后续正式提交，第 6 节所有行号需按提交版本复核。

### 0.3 审计脚本自身的假阴性已修正
首轮压测的 NaN 探针只覆盖「S 已成型 + 打分路由」路径，结果 `S` 干净，导致「NaN 污染」结论缺失，且 `T2-INF` 得出「clamp 截断为 ±10」的**错误结论**（实测 `max_abs_S=0.0802` 证明 S 根本没被更新）。已重写为 **A_routed / B_cold_start / C_direct_write 三条真实执行路径**，并收敛「清晰错误」判定口径。本报告采用修正后的最终数据。

### 0.4 结论计数
**阻塞 8 / 警告 9 / 建议 1 / 通过 1**（来自脚本 `findings`），另有第 5 节静态审查补充的 5 条建议项。

---

## 1. 结论摘要

| 任务 | 判定 | 一句话结论 |
| --- | --- | --- |
| 1 物理极限 D=1024 | **高危** | 单次 SVD 未超 500ms（最差 283.71ms），但**分裂步整步延迟达 705.09ms**；无 OOM，RSS 峰值 235.76 MB |
| 2 极端输入 | **崩溃** | NaN 同时造成「路由静默停写」与「冷启动路径 S 100% 污染 + SVD 抛 `_LinAlgError`」；空序列静默除零；层池 18 步 1→274 层 |
| 3 长期演化 | **死锁** | 10000 步仅 23 次激活，最后一次在 step 5727，之后 **4272 步零激活、0 分裂、层数恒 1**；S 条件数 ~1e10 |
| 4 静态与逻辑 | **多处缺口** | `state_dict` 复查干净（唯一通过项），但维度断言、fp16 数值安全、reset 生命周期、attach 幂等、无上界层池均失守 |
| 5 账号 A | **需返工** | 诊断钩子在真实训练里一次都不执行；Muon 秩 1 剔除后 lr 未适配；`allmuon` 档绕过修复 |

**立场**：SOMN V4 当前**不具备进入训练主路径的条件**。零污染契约做得干净（这是真优点），但输入契约与数值安全基本是空白，一次 NaN 或一次空窗口就能把记忆状态永久写坏，且失败是静默的。建议先合入第 7 节的 F1/F2/F3/F5 四个补丁，再谈启用。

---

## 2. 任务 1：物理极限与延迟压测（D=1024 / Seq=512 / Batch=1 / 200 步）

### 2.1 实测数据

| 场景 | 配置 | 平均 | 中位 | p95 | 最大 | 分裂次数 | RSS 峰值 | RSS 增量 | 最终层数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `A_default` | 全默认（warmup=50, max_write_count=100） | **26.22 ms** | 27.94 ms | 43.32 ms | 253.17 ms | 1 | **218.62 MB** | +54.76 MB | 2 |
| `B_forced_split` | threshold=1e-9, max_write_count=20, max_layers=6 | **148.10 ms** | 154.26 ms | 211.41 ms | **705.09 ms** | 5 | **235.76 MB** | +71.90 MB | 6 |

**分裂步真实尖刺（`B_forced_split`）**：

| step | 整步延迟 | 同一步分裂层数 |
| --- | --- | --- |
| 20 | 310.16 ms | 1 |
| 41 | **543.09 ms** | 2 |
| 62 | **705.09 ms** | 2 |

**SVD 微观基准**（`D=1024` 有能量矩阵上真实调用 `svd_split()` 5 次）：
`[219.90, 224.63, 205.07, 213.60, 214.13] ms`，median **214.13 ms**，max **224.63 ms**，`>500ms = []`。
循环内实测的 `svd_split()` 单次调用：A 场景 209.85 ms；B 场景 min 233.22 / median 265.46 / max **283.71 ms**。

### 2.2 高危判定
```
worst_svd_call_ms   = 283.71 ms   （单次 SVD：未超 500ms，余量 216.29ms）
worst_split_step_ms = 705.09 ms   （整步：超阈值 205.09ms）
oom                 = False
=> high_risk = True
```

**根因不是 SVD 本身，是「一步内可同步分裂多层」**：`server/somn.py:358-376` 的 `_check_expand()` 用 `for layer in list(self.layers)` 快照迭代，循环体内 `append`，所有合格层在同一次 `forward` 里各做一次 1024×1024 SVD。step 62 两层叠加 ≈ 2×265ms + 路由/读写 → 705.09ms。层数越多，尖刺越陡，`A_default` 只分裂 1 层时是 253.17ms。

**修复方向**：每步分裂预算上限 = 1（或把分裂移出同步 forward，见 F3/F7）。仅优化 SVD 算法解决不了这个叠加问题。

---

## 3. 任务 2：极端输入与鲁棒性破坏

### 3.1 NaN / Inf 注入 —— 三条路径，全部实测

| 注入类型 | A_routed（S 已成型） | B_cold_start（强制写） | C_direct_write（绕过路由） | 污染后 `svd_split()` |
| --- | --- | --- | --- | --- |
| `single_nan` | `out_finite=False`, `active=[]`, `writes 5→5`, `scores=[nan]` | `S_nan=True`, `nan_frac=1.0` | `S_nan=True`, `nan_frac=1.0` | `_LinAlgError` |
| `all_nan` | 同上 | 同上 | 同上 | `_LinAlgError` |
| `single_posinf` | 同上 | 同上 | `S_nan=True` (frac 0.0156), `max_abs_finite=10.0` | `_LinAlgError` |
| `single_neginf` | 同上 | 同上 | 同上 | `_LinAlgError` |
| `mixed` | 同上 | 同上 | `S_nan=True`, `nan_frac=1.0` | `_LinAlgError` |

**两种独立故障，都是静默的**：

1. **路由停写**（`server/somn.py:245-254`）：`scores()` 返回 `nan` → Python `max(boosted)` 为 NaN → `NaN >= min_score` 恒 `False` → `active=[]`，`write()` 根本没被调用（`writes 5→5` 是硬证据）。`forward:348-354` 把 `delta_H=0` 加回 `H_t`，**NaN 原样透传到下游**，无任何日志/告警/断言。这是「一次脏输入让整个记忆网络静默罢工」。
2. **状态永久污染**（`server/somn.py:126-140`）：冷启动分支 `route:238-243` 强制返回 `[0]`，NaN 进入 `write()`。`error = x - out` 与 `update = x.T @ error / N` 把 NaN 扩散到 `S` 全矩阵（`nan_frac=1.0`）；`torch.clamp(S, -10, 10)` **不能过滤 NaN**（NaN 比较恒 False）。后续 5 步干净输入实测 `out_finite=True` 但 `active=[]` —— 表面「没崩」，实际记忆已死。再调 `svd_split()` 抛 `_LinAlgError: linalg.svd: ... contained non-finite values`，且 `162-169` 的 `except RuntimeError` 降级 CPU 重算会**二次抛异常且无二次 try**，直接穿透 `forward`。

**Inf 的真实行为**（只依据 C 路径实测，不预设）：`clamp` 确实把 ±Inf 截成 ±10（`max_abs_finite=10.0`），但 `inf - inf` 已先产生 NaN，`S` 仍有 1.56% 元素为 NaN。所以「截断 = 安全」是错的：**数值爆炸被掩盖成静默污染**。

### 3.2 维度谋杀 —— 10 例实测

| 输入 | `read_only()` | 完整 `forward()` | 清晰错误？ |
| --- | --- | --- | --- |
| 2D `[2,64]` | **静默接受** → `[2,64]` | `ValueError: not enough values to unpack (expected 3, got 2)` | 否 |
| 1D `[64]` | **静默接受** → `[64]` | `ValueError: not enough values to unpack (expected 3, got 1)` | 否 |
| 4D `[1,4,2,64]` | **静默接受** → `[1,4,2,64]` | `ValueError: too many values to unpack (expected 3)` | 否 |
| D 不匹配 `[1,4,65]` | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x65 and 64x64)` | `ValueError: 输入维度 65 != 层维度 64` | **是（唯一）** |
| **`T=0 [1,0,64]`** | 静默 | **静默接受**，且 `S` 变为 `finite=False` | 否 |
| **`B=0 [0,4,64]`** | 静默 | **静默接受**，且 `S` 变为 `finite=False` | 否 |
| `int64 [1,4,64]` | 静默 | `RuntimeError: linalg.vector_norm: Expected a floating point ... Got Long` | 否 |
| fp16 / 全零 / 非连续 | 静默 | 静默接受（合法输入，但见 W4/W9） | 不适用 |

结论：`server/somn.py:115-124` 的 `read_only()` **没有任何维度/dtype 断言**，输入 1D/2D/4D 都能通过；`write():128` 靠 `B, T, D = H_t.shape` 解包报错，消息只暴露实现细节，不含「期望 `[B,T,D]`」；`129-130` 的 `raise ValueError("输入维度 ... != 层维度 ...")` 是**全文件唯一的显式维度断言**。空窗口最危险：`x.shape[0]==0` 时 `136` 行 `/ x.shape[0]` 变成 `0/0`，`S` 全 NaN，且**不抛异常**（详见 B7）。

### 3.3 高频分裂攻击（阈值 1e-12，1000 条目标）

| 场景 | 实跑步数 | 层数 | 分裂数 | RSS 起点 → 峰值 | 增量 | 耗时 | 中止原因 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D=64, `max_layers=0`（默认） | **18** | **274** | 273 | 238.48 → 244.79 MB | +6.31 MB | 0.27 s | `guard_layers=256` at step 17 |
| D=64, `max_layers=8` | **1000** | 8 | 7 | 244.79 → 244.80 MB | +0.00 MB | 1.57 s | 无 |
| **D=1024, `max_layers=0`** | **14** | **70** | 69 | 244.80 → **501.84 MB** | **+257.04 MB** | 11.08 s | `guard_layers=48` at step 13 |

**结论**：默认配置 `max_layers=0`（`server/somn.py:313`）在极低阈值下是**指数爆炸**，不是线性增长 —— `_check_expand():358-376` 的快照迭代让每个合格层同一步各分裂一次，层数按步翻倍（P10 实测：一次 `_check_expand()` 让 3 层 → 6 层）。D=64 时瓶颈是层数不是内存（+6.31 MB）；**D=1024 实测 14 步就吃掉 257.04 MB RSS**（每层 S 为 4 MiB，实测增量与 70 层吻合）。脚本设置了 256 层 / 2048 MB / 120 s 三道保护才停住，**没有让进程真 OOM，是为了不把机器打挂**；去掉 guard 后 1000 条必然 OOM，这一点不做「理论外推」，只陈述已实测的 14 步 / +257 MB 事实。

设 `max_layers=8` 后 1000 条跑满、RSS 零增长、1.57 s —— **上界是有效且几乎零成本的修复**，这是本轮最明确的一条建议。

---

## 4. 任务 3：长期数学演化（[1,1,64] / 10000 步 / 全默认配置）

| 指标 | 实测值 |
| --- | --- |
| 总耗时 | 1.94 s |
| 最终层数 / 分裂数 / warmup 数 | **1 / 0 / 0** |
| 各层 writes | `[23]` |
| 总激活次数 | **23** |
| 最后一次激活 | **step 5727** |
| 末次激活后连续零激活 | **4272 步**（step 5728–9999） |
| 归一化激活熵 / top1 占比 | `-0.0` / **1.0** |
| 最终 scores / min_score | `[-0.2151]` / `0.5` |
| 末段 1000 步 scores 方差 | 0.0881213 |
| S 条件数（layer0） | 首 1.5284e10 → 末 9.9874e9（峰值 4.7859e11 @ step 3500） |
| σ_min | `0.0` → `1.3e-09` |
| RSS 峰值 | 254.88 MB |

### 4.1 层数与收敛
**没有「收敛到合理层数」，是收敛到 1 层后彻底静止**。全程 `splits=0`，`expand_growth_threshold=5.0`，能量早已到 15.98，但 `_check_expand():365-368` 还要求 `write_count > max_write_count(100)`，而路由把 `write_count` 冻结在 23 → 永远不分裂。

### 4.2 死锁链（架构级，非参数问题）
```
score(-0.2151) < min_score(0.5)  →  active=[]  →  write_count 冻结在 23
→ 不满足 write_count > 100  →  不分裂  →  没有新层，也就没有 warm-up 出生特权
→ 没有任何机制能重新激活  →  永久零激活
```
`route():265-268` 的出生特权**只发给新诞生的层**，而新层诞生又依赖旧层被写入。这是鸡生蛋死锁。10000 步里 SOMN 实际只工作了 23 步（0.23%），其余全部空转，`forward` 仍在每步计算 `gate` 与残差。

### 4.3 条件数是否爆炸
**是，且是绝对值意义上的病态**：σ_min 塌到 `0` / `1.3e-09`，条件数长期维持在 1e10 量级、峰值 4.79e11。低秩结构化输入让 S 秩亏，`svd_split():163` 直接对这种矩阵做 `torch.linalg.svd`，数值结果不可靠（也解释了 W7 的幽灵分数）。虽然本轮 `cond` 首尾比值 0.6534（没有单调增长），但**一开始就病态**，不是"后来才恶化"。

### 4.4 Router 是否永久僵化
**是**。得分本身还在波动（末段方差 0.0881213），但被 `min_score=0.5` 一刀切成 `active=[]`，激活分布完全塌缩（归一化熵 −0.0、top1 占比 1.0）。没有僵尸层是因为**根本没有产生过第二层**。

---

## 5. 任务 4：代码静态与逻辑审查（核心）

### 5.1 问题主清单（按严重程度，全部附行号）

#### 阻塞（8 项）

| 编号 | 问题 | 行号 | 实测证据 |
| --- | --- | --- | --- |
| **B1** | 分裂步整步延迟 705.09ms > 500ms（一步内多层同步 SVD） | `server/somn.py:147-189`, `358-376` | step 62 = 705.09ms / 2 层；单次 SVD 最差 283.71ms |
| **B2** | NaN/Inf 使 `max(boosted)` 为 NaN → 路由静默零激活、输出透传 NaN、记忆停写 | `245-254`, `348-354` | `active=[]`, `writes 5→5`, `scores=[nan]`, `out_finite=False` |
| **B3** | NaN 穿透 `clamp` 永久污染 S；污染后 `svd_split` 抛 `_LinAlgError` 且降级无二次保护 | `126-140`, `238-243`, `162-169`, `163` | `S nan_frac=1.0`；后续 5 步干净输入仍 `active=[]`；SVD 异常穿透 |
| **B4** | 维度异常未立刻抛清晰错误；`read_only` 完全无断言；空窗口静默接受 | `115-124`, `128-130` | 2D/1D/4D 静默过 `read_only`；解包报错不含维度语义；`T=0`/`B=0` 静默且污染 S |
| **B5** | 层池无上界（默认 `max_layers=0`）+ 每步可全员分裂 → 指数爆炸 | `313`, `358-376` | D=64：18 步 1→274 层、273 分裂；P10：一次 `_check_expand` 3→6 |
| **B6** | 长期演化永久零激活死锁（分数阈值与分裂前置条件耦合） | `254`, `265-268`, `365-368` | 10000 步仅 23 次激活，末次 step 5727，之后 4272 步零激活、0 分裂 |
| **B7** | 空序列 `B*T=0` 时 `/ x.shape[0]` 静默除零，S 全 NaN | `131`, `136` | `T=0` / `B=0` 均 `S_is_nan=True`, `nan_frac=1.0`，**不抛异常** |
| **B8** | `eps=1e-8` 在 fp16 下下溢为 0，零向量余弦退化 `0/0=NaN` | `225-227` | fp16 零输入 `scores=[nan]`；fp32 同输入 `[0.0]` |

#### 警告（9 项）

| 编号 | 问题 | 行号 | 实测证据 |
| --- | --- | --- | --- |
| **W1** | `clamp(±10)` 静默截断 Inf，掩盖数值爆炸；`inf-inf` 仍生成 NaN | `138` | C 路径 `max_abs_finite=10.0` 且 `nan_frac=0.0156` |
| **W2** | D=1024 高频分裂 RSS 14 步 +257.04 MB | `313`, `358-376` | 244.80 → 501.84 MB（实测，非外推） |
| **W3** | S 长期演化后极度病态（σ_min→0、cond ~1e10，峰值 4.79e11），SVD 输入无秩保护 | `136-138`, `163` | σ_min `0 → 1.3e-09` |
| **W4** | fp16 下 `svd_split` 往返重建误差放大 **481.8×**；`read_only` 差 5.6e-4 | `106-113`, `122-123`, `159-161`, `183-188` | fp32 err 1.8e-07 → fp16 err 8.677e-05 |
| **W5** | `reset()` 只换 `layers`/`stats`，未清 router 缓存；旧列表引用仍持有 S | `378-383`, `212-213`, `413-414` | reset 后 `scores=[0.6445]` 残留；旧 S 16 384 B 仍可达 |
| **W6** | 重复 `attach_somn` 不挂多模块，但 `setdefault` 保留旧层池 → dim/eta/router 静默错位 | `319-323`, `389-401` | `mgr_dim=128` vs `layer_dim=64`；`mgr_eta=0.005` vs `layer_eta=0.01`；喂新 dim 抛尺寸错 |
| **W7** | `svd_split` 后子层 σ≈0，`read_only` 近零却被 eps 放大出高余弦分（幽灵激活） | `178-188`, `225-227` | child σ_max=6e-08、read_absmax=2e-07，但 **cos_absmax=0.9057** |
| **W8** | 单次 `_check_expand` 可让每个合格层各分裂一次（层数按步翻倍） | `358-376` | 3 → 6 层 |
| **W9** | `read_only` 把 fp32 的 S **临时降**成输入 dtype（fp16），路由打分静默损失精度 | `122-123`（对比 `159-161` 升精度） | S=fp32 + H=fp16 → read=fp16，S 本身未被改 |

#### 建议（脚本 1 项 + 静态补充 5 项）

| 编号 | 问题 | 行号 |
| --- | --- | --- |
| **S1** | Elbow 用「乘 alpha」比较，负分时阈值方向反转；`scores` 无语义约束 | `254-262`；实测 3 层全 `-1.0` 分、`min_score=-1.0` 时只激活 `[0]`（阈值 `-0.6` 使第二个立即被截断） |
| **S2** | 构造器无参数范围断言：`dim<=0`、`eta<=0`、`lam>1`、`min_score` 任意、`expand_growth_threshold<0`、`warmup_steps<0` 全部放行 | `68-76`, `199-213`, `279-326` |
| **S3** | dtype 路径不对称：SVD 主动升 fp32（`159-161`），读写却无条件降到输入 dtype（`112-113`/`122-123`）；`route:256` 又固定用 fp32 构造分数张量 | `106-113`, `122-123`, `159-161`, `256` |
| **S4** | `tick_warmup()` 按**全局步**推进（对所有层、含未激活层），而不是按「该层被写入次数」；注释说「新层前 N 步」但语义是全局 N 步 | `98-104`, `342-344` |
| **S5** | 测试覆盖缺口（见 5.6 节） | `service/test/test_somn.py` |
| **S6** | 压测脚本放在根目录 `test/`，与项目惯例 `service/test/` + `run_tests.py` 的 `SUITES` 注册不符（**按任务书指定路径放置**，后续建议迁移并注册） | `test/somn_stress.py` |

#### 通过（1 项）

| 编号 | 结论 | 行号 / 证据 |
| --- | --- | --- |
| **P1** | `state_dict` 零污染复查通过 | `server/somn.py:296-326`, `389-401`；P7 实测：分裂到 4 层 + reset 后，键集合一致、值**逐位一致**，`mgr.state_dict()={}`、0 Parameter、0 Buffer、named_modules 仅 1 个 |

### 5.2 逐条回答任务书的静态检查项

**① 断言与异常检查（维度、状态初始化）**
全文件显式断言只有 2 处：`write():129-130` 的 `D != self.dim` 和 `svd_split():154-155` 的 `S is None`。其余全部缺失：

| 位置 | 应检查 | 现状 |
| --- | --- | --- |
| `IndependentFWLayer.__init__:68-76` | `dim > 0`、`eta > 0`、`0<=lam<=1` | 无，仅 `int()/float()` 转换 |
| `_align:106-113` | 输入 dtype 是浮点、S 形状 `[D,D]` | 无 |
| `read_only:115-124` | `H_t.ndim==3`、最后一维 == dim | **完全无断言**（B4） |
| `write:126-140` | `B*T>0`、输入 finite | 无（B3/B7） |
| `svd_split:162-169` | S finite、降级路径二次异常 | 无，`except RuntimeError` 只包第一次 SVD |
| `EnergyElbowRouter.__init__:199-213` | `0<elbow_alpha<=1`、`-1<=min_score<=1`、`bonus>=0` | 无（S2） |
| `route:231-268` | `boosted` 全 finite、layers 非空一致性 | 无（B2） |
| `SOMNModule.__init__:279-326` | 全部配置范围、`rt` 类型 | 无（S2） |
| `forward:348-356` | `H_t` 契约、`delta_H` finite | 无 |
| `attach_somn:389-401` | dim 一致性、重复挂载 | 无（W6） |

**② dtype 隐式转换（fp16↔fp32）**
- **降精度**：`_align:112-113` 会把已存在的 `S` **永久** `.to(dtype=H_t.dtype)`；`read_only:122-123` 做临时降精度（W9）。fp32 状态遇 fp16 输入，路由打分在 fp16 下算完才回传 Python float，精度不可逆。
- **升精度**：`svd_split:159-161` 主动把 fp16/bf16 升到 fp32 做 SVD，`183-188` 再降回原 dtype → 往返误差 481.8×（W4）。
- **不一致**：`route:256` 固定 `torch.tensor(boosted, dtype=torch.float32)`，而 `scores()` 在输入 dtype 下算；`forward:351-353` 的 `gate` 又跟随输入 dtype。三条路径各用各的精度（S3）。
- **精度风险结论**：**有实质风险**，且已量化（W4/W9/B8）。`apply_precision()` 后若模型跑 fp16/bf16，SOMN 的路由分数会系统性偏移，甚至产生 NaN。

**③ 状态生命周期 —— 明确回答两个提问**

- **`reset()` 后 `layers` 是否「彻底清空」？——否。** `378-383` 是**替换列表对象**（`self._slot["layers"] = [新层]`），不是 `clear()`。因为 `layers` 是每次读 `_slot` 的 property（`333-334`），新 mgr 自己看不到旧层；**但任何外部持有了 `mgr.layers` 旧列表引用的代码，仍能访问旧层及旧 `S` 张量**（P5 实测 `old_S_still_reachable=True`、16 384 B 未释放）。D=1024 时每个旧 S 是 4 MiB，长会话里反复 reset 会累积。同时 `router.last_base_scores/last_boosted_scores`（`212-213`）**没有清**，P5 实测 reset 后 `somn_stats()` 仍吐出上一轮的 `[0.6445]`。属于观测污染，不是数值泄漏（`route:234-235` 下次会覆盖），但 `/api/infer/stats` 类前端会显示脏数据。

- **`IndependentFWLayer` 的 buffer 会不会无界增长？——不会，但问题问错了地方。** `S` 固定 `[D,D]`、`write_count`/`warmup_age` 是标量、`router.last_*` 每次 `route()` 开头清空（`234-235`）、`stats.active_last` 每步覆盖（`345`）—— 单层与 router 都是**有界**的。真正无界的是 **`SOMNModule._slot["layers"]`**：`max_layers` 默认 `0`（`313`），`_check_expand` 无每步分裂上限（W8/B5）。所以「加 `max_buffer_size`」应该加在**层池**上，不是 `IndependentFWLayer`。清理策略见 F3/F4。

**④ 数值安全**
- `S = lam*S + eta*update`（`137`）：`update = x.T @ error / x.shape[0]`（`136`）。`error→0` 时 `update→0`，数学上无除零；但 **`x.shape[0]==0` 时是 `0/0` → 静默 NaN**（B7，已实测）。
- 输入极端值：`137` 无 `isfinite` 检查，`138` 的 `clamp` 不过滤 NaN、把 Inf 压成 ±10（B3/W1）。
- `svd_split` 后新层 σ 过小：**确实不稳定**，但不是因为 NaN，而是因为 `cosine_similarity` 的 eps 把近零 `read` 的浮点噪声放大成 **0.9057 的高分**（W7）。`eps=1e-8` 只在 fp32 有效，fp16 下直接下溢（B8）。**建议按 dtype 选 eps，并对近零重建直接给 0 分**（F2）。
- `torch.linalg.svd(work)`（`163`）：对含 NaN/Inf 的 S 抛 `_LinAlgError`；`164-169` 的 CPU 降级会二次抛且无 try（B3）。

**⑤ 契约与污染**
- **state_dict：干净。** P7 实测 + 交叉审计静态复查：账号 A 新增项全是普通属性（`diag_probe`、`manifold_config`、Optimizer 上的 `profile` 计时器），**没有新增 `nn.Parameter` / `register_buffer` / 子模块注册**；`common.py:818` 的 strict `load_state_dict` 键集合不变。唯一既有 `register_buffer` 是 `momw.py:710` 的 `_anchors`（`persistent=False`），不在本次 diff 内。
- **`attach_somn` 重复调用：不会挂多个重复模块**（`400` 覆盖 `module` 键，P6 实测 `module_key_count=1`），**但会静默产生更危险的错位**：`319-323` 的 `setdefault` 保留旧 `layers`/`stats`，新 mgr 却重建 `router` 并按新 `dim` 计算 `eta`。P6 实测 `attach(root,64)` 后再 `attach(root,128)` → `mgr.dim=128` / `layers[0].dim=64`、`mgr.eta=0.005` / `layer.eta=0.01`、`router` 是新对象、旧 mgr 与新 mgr **共享同一 `_slot`**。按新 dim 喂数据得到 `RuntimeError: mat1 and mat2 shapes cannot be multiplied (8x128 and 64x64)`，不是清晰的维度错误。**幂等性不合格**（W6）。

**⑥ 现有测试覆盖缺口（S5）**
`service/test/test_somn.py` 30 项测试全部通过，但**以下边界一条都没写断言**：NaN/Inf 注入、2D/1D/4D/0 长度输入、fp16 余弦 eps、空窗口除零、重复 attach、reset 后 router 缓存、σ≈0 子层、负分 Elbow、1000 条高频分裂、D=1024 的 200 步压测（只有 `test_1024_svd_on_cpu` 单次 SVD）、长期 10000 步演化。这解释了为什么上述 8 个阻塞项能一直留在代码里。

---

## 6. 账号 A 修复代码交叉审计

### 6.1 `reset_memory()` 替换后，临时缓存是否清理干净？—— **不干净，且前提要纠正**

`server/model.py:999-1014` 的 `reset_memory()` 本体在 HEAD 已存在，**账号 A 没有重写它一行**；A 的改动是把 `service/train/train.py:139` 与 `service/train/clip.py:154` 的直连 `fast_weight.reset()` 统一到 `reset_memory()`（调用点统一，方向正确）。

清理情况（`reset_memory():1009-1014`）：

| 状态 | 位置 | 是否清理 |
| --- | --- | --- |
| `FastWeightBank._state`（含 `conv_cache`） | `model.py:648, 650-651` | 是 |
| `MoMWPool._rt` 全字段 / 模块级 `last_*` | `momw.py:803-811` | 是 |
| SOMN `_slot["layers"]` / `stats` | `somn.py:380-383` | 是（**但 router 缓存没清**，见 W5） |
| `MoMWRouter.last_raw/last_scores/last_cos_anchor/last_cos_state/last_gamma` | `momw.py:373-377, 424-428` | **否（中危）** |
| `EnergyElbowRouter.last_base_scores/last_boosted_scores` | `somn.py:212-213` | 否（低，下次 route 覆盖；P5 已实测残留 `[0.6445]`） |
| `ModelDiagnostics._attn` 累加器 | `manifold_diag.py:313, 323-326` | 否（仅 `--diag` 开启时存在） |
| `Muon.ns5_seconds/step_seconds/ns5_calls` | `muon.py:81-83` | 否（全程累计量，导致 `manifold_diag.py:459-478` 的「NS5 占比」是累计均值而非区间值） |
| 外部 `state["fw"]/["momw"]` | `model.py:1002-1004` | 设计上保留，不算残留 |

最刺眼的是 `momw.py:955-961`：`stats()` 里 `last_raw`/`last_scores` 取**模块级已清空的字段**，`last_cos_anchor`/`last_cos_state`/`last_gamma` 取**router 级未清空的字段** —— 同一个字典一半归零、一半是上一样本的陈旧值。**这正是任务书点名的「只替换了 layers 却没清 router 缓存」实例。**

### 6.2 Muon 剔除秩 1 投影后，学习率是否需要重新适配？—— **需要，且这是明确立场**

- `model.py:1316-1328` 的注释只论证了「为什么 rank-1 2D 矩阵不能进 Muon」，**没有一句话论证替换后的步长**；`_classify_param:1338-1349` 把它们改判 AdamW。
- 数学尺度变化：改前 Muon `lr=2e-2` + `u * sqrt(max(p.shape))`（`muon.py:107-110`，`[1,1024]` → ×32，RMS≈1）；改后 AdamW `lr=3e-4` + `betas(0.9,0.95)` + `weight_decay=0.0`（`muon.py:221-222`）。逐元素步长落差约 **67×**，且归一化语义完全不同。全批改动里**没有 lr 扫描、没有独立 param group、没有适配验证**。
- 新增的 `ManifoldMuon`（`manifold_muon.py:287-290`）返回 `msign(...)` 半正交方向后直接 `W_tilde = p - lr*A`，**缺少 Muon 的 `*sqrt(max(shape))` RMS 对齐**；同样 `lr=2e-2` 下幅度约为标准 Muon 的 `1/32`。直接复用 `MUON_LR` 不成立。
- **结论**：留在 Muon 组的满秩 2D 矩阵公式与 `lr=2e-2` 逐字未动，**无需重调**；但被移出的残差门投影必须单独标定步长，Manifold 路径必须加回 RMS 对齐或重设 lr。
- **修复遗漏**：`muon.py:197-200` 的 `allmuon` 档从 `groups["muon"] + groups["adamw"]` 重新收集**全部 2D**，把 `_classify_param` 刻意排除的 `[1,D]/[D,1]` 又送回 Muon，秩 1 兜底被绕过。
- 除零 / shape / 1D 误用：**未发现**（`manifold_muon.py:69-71, 205, 331` 均显式 `ndim != 2` 拒绝；`muon.py:45`、`manifold_muon.py:79` 带 eps）。但 `manifold_muon.py:71-72, 164-165` 的非有限值会 `raise ValueError`，而 `ManifoldMuon.step:261-297` **无 try/except**，异常穿透 `optimizer.step()` 崩训练 —— 与诊断模块「绝不崩训练」的口径相反（默认 `enabled=False`，暂不触发）。

### 6.3 谱诊断钩子的 `svdvals` 是否影响训练性能？—— **设计上按间隔；实现上一次都不执行**

- 唯一 `svdvals` 在 `manifold_diag.py:258`（`kda_state_stats()`），调用链是 `_collect_kda():443-457` ← `observe():502-523` ← `train.py:140-144` 的日志回调，**不在 forward、不在 optimizer.step**。门控在 `observe:506`，`_DEFAULT_EVERY = 50`（`manifold_diag.py:54`），step 1 也采 —— 已满足「建议每 100 步才跑一次」的量级，`--diag` 默认关闭时零开销。
- **但真实训练里它一次都不会执行（高危功能性 bug，已实测）**：`train.py:139` 的 `m.reset_memory()` 排在 `train.py:141` 的 `diag.observe()` **之前** → `model.py:1009` 把 `fast_weight._state` 置 `None` → `_collect_kda():445` 拿到 `None` → `kda_state_stats:243-244` 直接 `return {"present": False}` → `:258` 不可达。
  实测证据（本次独立最小复现）：
  ```
  BEFORE reset  _state is tuple = True        kda_state_stats(...).present = True
  call m.reset_memory()
  AFTER  reset  _state = None                 kda_state_stats(None) = {'present': False}
  ```
  JSONL 里 `kda_state` 恒为 `{"present": false}`，设计文档承诺的 ρ(S)/‖S‖_F/λ_max(C) 全部采集不到。`service/test/test_manifold_20261113.py` 只单独测 `kda_state_stats`，**没有把 `_collect_kda` 与真实训练顺序串起来的测试**。
- 真正的每步开销不在 SVD：`--diag` 开启后 `train.py:134-135` 的 `grad_norm_before_clip()` **每步**遍历全模型参数做标量同步，而 `observe()` 只在 1/50 步记录它，**49/50 的计算被丢弃**；`_attn_probe`（`model.py:351-352`）挂在每次 MLA forward 内。建议把这两项也纳入 `every` 门控。
- `no_grad` 覆盖完整；未发现大矩阵整体 `.cpu()` 回拷，但大量 `float(tensor)` 在 GPU 上是隐式设备同步；`observe():520` 的 `_write` 带 `flush()` 落盘，采样步会阻塞训练循环。

### 6.4 其他交叉发现
- `server/common.py`（diff hunk `@@ -637,6 +645,119 @@`）：配置未知键静默保留，把 `enabled` 拼成 `enabeld` 不报错也不生效，只是按关闭处理 —— 低危易用性问题。
- `server/manifold_muon.py:22-26` 自述 `retract` 会改变权重数值分布、关掉开关无法回滚；对已有 checkpoint 启用第一步即整体改写权重谱 —— 启用时需谨慎。
- **state_dict / strict load：未发现污染**（见 5.2 ⑤；`test_manifold_20261113.py:225-240` 有逐位一致性回归）。

---

## 7. 修复建议代码（严重问题，未应用到主干）

> 本分支禁止修改业务代码，以下均为**建议补丁**，供主干修复时参考。行号对应当前 `server/somn.py`。

### F1 统一输入契约（修 B2/B3/B4/B7）
在模块级新增校验，并在 `read_only` / `write` / `forward` 入口调用：

```python
def _validate_H(H_t: torch.Tensor, dim: int) -> None:
    if not isinstance(H_t, torch.Tensor):
        raise TypeError(f"SOMN 需要 torch.Tensor，收到 {type(H_t).__name__}")
    if H_t.ndim != 3:
        raise ValueError(
            f"SOMN 输入必须是 [B, T, D]，收到 shape={tuple(H_t.shape)} ndim={H_t.ndim}")
    if H_t.shape[-1] != dim:
        raise ValueError(f"输入维度 {H_t.shape[-1]} != 层维度 {dim}")
    if H_t.shape[0] * H_t.shape[1] == 0:
        raise ValueError(f"空序列不允许（B*T=0），shape={tuple(H_t.shape)}")
    if not H_t.is_floating_point():
        raise TypeError(f"SOMN 需要浮点输入，收到 dtype={H_t.dtype}")
    if not torch.isfinite(H_t).all():
        raise ValueError("SOMN 输入含 NaN/Inf，拒绝写入以防污染状态 S")

# read_only():115 开头加 _validate_H(H_t, self.dim)
# write():126 开头加 _validate_H(H_t, self.dim)，然后可删掉 128 的隐式解包依赖
# forward():348 开头加一次（路由之前），保证 NaN 不进残差输出
```

### F2 数值安全（修 B3/B8/W1/W7）
```python
import math                      # 文件头新增：server/somn.py 当前未导入 math

_EPS_BY_DTYPE = {torch.float16: 1e-4, torch.bfloat16: 1e-3,
                 torch.float32: 1e-8, torch.float64: 1e-12}

# scores():215-229 —— eps 按 dtype 选 + 近零重建直接判 0 分，杜绝幽灵激活
read = layer.read_only(H_t)
eps = _EPS_BY_DTYPE.get(H_t.dtype, 1e-8)
if float(read.norm().item()) < eps:
    out.append(0.0)
    continue
out.append(float(torch.cosine_similarity(H_t, read, dim=-1, eps=eps).mean().item()))

# write():137-138 —— clamp 前先拒绝非有限值
update = torch.matmul(x.transpose(0, 1), error) / x.shape[0]   # F1 已保证 != 0
S_new = self.lam * self.S + self.eta * update
if not torch.isfinite(S_new).all():
    raise FloatingPointError("SOMN 状态 S 出现非有限值，已中止写入（保留上一步 S）")
self.S = torch.clamp(S_new, min=-10.0, max=10.0)

# route():245 —— 打分有限性显式断言（替代静默 max() 判定）
base = self.scores(H_t, layers)
if not all(math.isfinite(s) for s in base):
    raise FloatingPointError("SOMN 路由打分出现 NaN/Inf")
```

### F3 层池上界 + 每步分裂预算（修 B1/B5/W2/W8）
```python
# __init__:279-326
if self.dim <= 0:
    raise ValueError(f"dim 必须为正，收到 {self.dim}")
self.max_layers = int(max_layers) if max_layers > 0 else 8   # 默认给上界，不再用 0
self.max_splits_per_step = 1                                 # 新增，根治 705ms 尖刺

# _check_expand():358-376
budget = self.max_splits_per_step
for layer in list(self.layers):
    if budget <= 0:
        break
    if self.max_layers and len(self.layers) >= self.max_layers:
        break
    if layer.in_warmup:
        continue
    if layer.write_count > self.max_write_count and \
       layer.get_energy() > self.expand_growth_threshold:
        new_layer = layer.svd_split()
        # 有效秩保护：σ 过小时跳过分裂，避免生成 S≈0 的幽灵层
        ...
        budget -= 1
```
实测依据：设 `max_layers=8` 后 1000 条跑满、RSS 零增长、1.57 s；`max_splits_per_step=1` 可把 705.09ms 的尖刺压回 ~310ms 量级（单次 SVD + 单层读写）。

### F4 reset 彻底清理（修 W5，并与账号 A 的 `MoMW.reset()` 同口径）
```python
def reset(self) -> None:
    for layer in self._slot.get("layers", []):
        layer.S = None            # 主动断开张量：外部持有旧列表也拿不到 S
        layer.write_count = 0
        layer.warmup_age = None
    self._slot["layers"] = [IndependentFWLayer(self.dim, **self._layer_kwargs)]
    self._slot["stats"] = self._new_stats()
    self.router.last_base_scores = []
    self.router.last_boosted_scores = []
```
账号 A 侧对应在 `momw.py:803-811` 的 `reset()` 末尾补：
```python
self.router.last_raw = []; self.router.last_scores = []
self.router.last_cos_anchor = []; self.router.last_cos_state = []
self.router.last_gamma = []
```

### F5 attach 幂等 + dim 一致性（修 W6）
```python
def attach_somn(root: Any, dim: int, **kwargs) -> SOMNModule:
    rt = kwargs.pop("rt", None)
    if rt is None:
        if not isinstance(getattr(root, "_rt", None), dict):
            root._rt = {}
        rt = root._rt
    slot = rt.get("somn")
    if isinstance(slot, dict) and isinstance(slot.get("module"), SOMNModule):
        mgr = slot["module"]
        if mgr.dim != dim or mgr.layers[0].dim != dim:
            raise ValueError(
                f"SOMN 已挂载 dim={mgr.dim}/layer={mgr.layers[0].dim}，"
                f"拒绝以 dim={dim} 重复挂载")
        return mgr                      # 幂等：不重建 router、不丢配置
    mgr = SOMNModule(dim, rt=rt, **kwargs)
    rt["somn"]["module"] = mgr
    return mgr
```

### F6 破除长期零激活死锁（修 B6）
```python
# _new_stats() 增 "idle_steps": 0；_delta():338 后
if active:
    self._slot["stats"]["idle_steps"] = 0
else:
    self._slot["stats"]["idle_steps"] += 1

# _check_expand() 开头：长期零激活时给第 0 层一次 re-birth
if self._slot["stats"]["idle_steps"] >= self.rebirth_after:   # 建议 200
    self.layers[0].begin_warmup(self.warmup_steps or 50)
    self._slot["stats"]["idle_steps"] = 0
```
**这只是止血**。根因是「固定 `min_score=0.5`」与「分裂必须 `write_count>100`」的耦合：建议把分裂前置条件从写入计数改为「能量 + 有效秩」，并把 `min_score` 做成随层数/得分区间自适应的分位数（MoMW v3 已有 `threshold_mode="quantile"` 的现成做法可复用）。

### F7 SVD 降级二次保护（修 B3 的崩溃穿透）
```python
# svd_split():162-169
try:
    U, Sigma, Vh = torch.linalg.svd(work, full_matrices=False)
except RuntimeError:
    try:
        work = S.detach().to("cpu")
        if work.dtype not in _SVD_SAFE_DTYPES:
            work = work.to(torch.float32)
        U, Sigma, Vh = torch.linalg.svd(work, full_matrices=False)
    except RuntimeError as e:
        raise RuntimeError(
            "SOMN SVD 分裂失败：状态 S 可能已被 NaN/Inf 污染；"
            "请先检查 reset_memory() 与输入有限性") from e
```

### F8 账号 A 侧三处（见第 6 节）
1. **诊断顺序**：把 `train.py:141` 的 `diag.observe()` 移到 `train.py:139` 的 `m.reset_memory()` **之前**，或在 reset 前把 `fw._state` 快照传入 `observe()`；并补一条「训练顺序 + `_collect_kda`」的集成测试。
2. **lr 适配**：给被移出 Muon 的 rank-1 投影单独 param group 做步长扫描；`ManifoldMuon` 加回 RMS 对齐或重设 lr；`allmuon` 档（`muon.py:197-200`）复用 `_classify_param` 而非重新收集全部 2D。
3. **限频**：`svdvals` 现已是每 50 步，无需再降到 100；真正要做的是把 `grad_norm_before_clip`（`train.py:134-135`）与 `_attn_probe`（`model.py:351-352`）纳入 `every` 门控，避免每步空转 49/50。

---

## 8. 复现方式

```bash
cd /Users/steve/Desktop/GARGANTUA
git checkout feat/somn-stress-test

# 完整四项（约 1 分钟，CPU only）
/usr/local/bin/python3 test/somn_stress.py \
    --json data/tmp/somn_stress_result.json 2>&1 | tee data/tmp/somn_stress.log

# 单项复现
/usr/local/bin/python3 test/somn_stress.py --only t1     # D=1024 / 200 步
/usr/local/bin/python3 test/somn_stress.py --only t2     # 破坏性输入
/usr/local/bin/python3 test/somn_stress.py --only t3     # 10000 步演化
/usr/local/bin/python3 test/somn_stress.py --only t4     # 静态审查取证探针

# 依赖：torch 2.2.2 + psutil 7.2.2（本机用清华镜像 --trusted-host 安装）
```

脚本自带显式断言（冷启动前置条件、跑满步数、输出有限性、state_dict 逐位一致、负分构造、重复挂载唯一性等），任一断言失败会以非零码退出，不会静默「跑通」。

---

## 9. 局限与未覆盖项（如实列出）

1. **只测 CPU**：本机 torch 2.2.2 无 MPS/CUDA（`cuda=False / mps=False`）。`_needs_cpu_svd`（`somn.py:51-56`）的 MPS 降级路径只做了静态判定，未在真机验证；GPU 上 `float(tensor)` 隐式同步的开销也未实测。
2. **未跑全量回归**：按项目约定「未经 steve 允许不跑 30 套件全量（≈13.5 分钟）」，本轮只做了 `py_compile` + 新增压测脚本 + 改动范围外的单点探针。修复合入主干后需补跑 `test_somn` 与 `test_model` 套件。
3. **高频分裂攻击设置了保护阈值**（256 层 / 48 层 / 2048 MB / 120 s），因此**没有让进程真正 OOM**；D=1024 只实跑到 14 步 / 70 层 / +257.04 MB。这是刻意的安全阀，不是「跑满 1000 条」。
4. **账号 A 代码未提交**：第 6 节行号基于工作区快照（`6bd2bc5` + 40 项未提交改动），分支 `fix/optimizer-and-reset` 不存在，A 提交后需复核。
5. **路径规范**：按任务书要求把压测脚本放在 `test/somn_stress.py`，与项目惯例 `service/test/` + `run_tests.py` 的 `SUITES` 注册不一致（S6）；本次未改 `run_tests.py`（属主干测试基础设施，超出授权范围）。
6. **未修改任何主干业务代码**：`server/`、`service/` 全部只读；本分支只新增 `test/somn_stress.py` 与 `docs/audit/somn_v4_audit.md`。账号 A 的 40 项未提交改动完整保留、未被 add / stash / checkout。
7. **不 push**：按要求等待审阅。
