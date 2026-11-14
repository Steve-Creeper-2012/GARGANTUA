# SOMN V4 抢救性手术 · 补丁验收报告

| 项目 | 内容 |
| --- | --- |
| 分支 | `feat/somn-stress-test`（承接上轮 `06a57aa` 审计报告） |
| 日期 | 2026-11-13 |
| 角色 | 对抗性测试与安全防线工程师（账号 B） |
| 目标 | 落实 F1–F8，使 SOMN 具备基本鲁棒性 |
| **验收结论** | **19 PASS / 0 阻塞 / 0 警告 / 0 建议，退出码 0** |
| 原始数据 | `data/tmp/somn_verify.log`、`data/tmp/somn_verify.json` |
| 提交 | 本轮提交，**不 push** |

---

## 0. 执行边界（先说清楚，请 steve 复核）

任务书同时要求「落实 F1–F8 补丁」和「严禁碰主干代码」。F1–F8 全部落在 `server/somn.py`，若完全不碰该文件则补丁无处可落。因此本轮按以下边界执行：

- **只改 `server/somn.py`（SOMN 本体）**，该文件不在账号 A 的 40 项未提交改动里，无冲突；
- **未改** `server/model.py`、`server/muon.py`、`service/train/train.py`、`service/test/*` 等任何其他主干模块；
- **未切换分支、未碰 `main`**；账号 A 的 40 项未提交改动在本轮前后数量一致，未 add / stash / checkout；
- 新增测试全部放在 `test/` 下，未改 `service/test/run_tests.py` 的 `SUITES` 注册。

若这个解释与你的意图不符（例如希望 `server/somn.py` 也保持只读、补丁另开实验副本），请驳回本次提交，我按新边界重做。

---

## 1. 验收结论

```
验收汇总（V01..V19 对应 v1 审计的 8 阻塞 / 9 警告 / 1 建议 / 1 通过）
  通过   19
  阻塞   0
  警告   0
  建议   0
  => ALL PASS (0 阻塞 / 0 警告)
```

三组独立验证全部通过：

| 验证 | 命令 | 结果 |
| --- | --- | --- |
| 完整验收压测 | `test/somn_stress.py --json ...` | 19 PASS，exit 0 |
| F1–F8 专项用例 | `test/test_somn_f1_f8.py -v` | **24/24 OK**（5.2s→3.1s） |
| 既有回归 | `service/test/test_somn.py` | **30/30 OK**（2.631s） |

---

## 2. 落实的补丁清单（F1–F8）

| 补丁 | 实现 | 位置 |
| --- | --- | --- |
| **F1 输入契约** | 新增 `_validate_H()`：`ndim==3`、`D` 匹配、`B*T>0`、浮点、全量 `isfinite`；非有限输入抛 `FloatingPointError` 立刻终止。`read_only` / `write` / `forward` / `scores` 四入口共用；另加 `dim/eta/lam/elbow_alpha` 构造器断言 | `server/somn.py` 顶部工具函数、各入口 |
| **F2 数值安全** | `/ (N + _DIV_EPS)`；写入前 `isfinite(S_new)` 检查；`_EPS_BY_DTYPE` 按计算精度选 eps；`_MIN_REBUILD_RATIO=1e-3` 相对重建幅度判据；`svd_split` 秩亏判定（子层 σ_max/σ₁ < 1e-6 → 返回 `None` 且不改父层 S）；`route` 非有限分数立刻 raise | `IndependentFWLayer.write` / `svd_split`、`EnergyElbowRouter.scores/route` |
| **F3 层池与预算** | `_HARD_MAX_LAYERS=8` 硬上限（传 0 / 负值 / 64 一律钳制到 8）；`_MAX_SPLITS_PER_STEP=1`；`_check_expand` 每步只分裂一次 | `SOMNModule.__init__` / `_check_expand` |
| **F3/P1 异步队列** | 后台 daemon 线程执行 SVD；`_check_expand` 只入队 S 快照；`_drain_split_results()` 下一步消费，用「当前 S − 快照」增量合并回父层，保证 `S_main + S_sec + Δ == 当前 S` 无损 | `_ensure_worker` / `_split_worker` / `_compute_split` / `_drain_split_results` |
| **F4 reset** | 显式把旧层 `S=None`、`write_count=0`、`warmup_age=None`；清 router 两个打分缓存；丢弃未消费的异步任务 | `reset()` |
| **F5 attach 幂等** | 同 dim 返回既有管理器（router / 配置不重建）；异 dim 或层池 dim 不一致直接 `ValueError` | `attach_somn()` |
| **F6 探索机制** | 连续 `explore_every=100` 步零激活时，强制激活当前得分最高层写入一次；`stats` 记录 `explorations` / `idle_steps` | `_delta` / `_best_layer_index` / `_new_stats` |
| **F7 SVD 降级** | CPU 降级路径二次 `try`，异常带「状态可能被 NaN/Inf 污染」提示，不再静默穿透 | `svd_split()` |
| **F8 精度** | `_state_dtype()`：fp16/bf16 输入把状态 S 升 fp32，fp32/fp64 保持；`read_only` 把 S 临时**升**到计算精度而非降输入；输出 dtype 仍跟随输入 | `_state_dtype` / `_align` / `read_only` / `write` / `scores` |

另顺手修了上轮的「建议」项：Elbow 负分阈值方向（`prev = max(prev, 0)`）与「负分不激活」。

---

## 3. 异步队列：为什么做、实测依据

用户要求「如果单步 SVD 耗时超 500ms，可考虑移入独立队列异步执行」。修复同步上限后，D=1024 极端场景单次 SVD 仍达 695–1009ms。先做了最小实验判断 CPU-only 上异步是否真有收益（**实测，非估算**）：

```
D=1024, torch 4 线程, 6 次 [512,1024]@[1024,1024] 路由 + 1 次 1024×1024 SVD
串行一步        = 280.9 ms   (SVD-only 268.0 ms + route-only 46.0 ms)
异步后主线程     = 61.8 / 63.3 / 61.9 ms   → 主线程的 0.22×
```

结论：CPU-only 上异步**有效**（BLAS 释放 GIL，后台 SVD 与主线程路由重叠），因此实施，不是"可考虑"而是必须做。

设计要点与语义保证：
- 入队保存 `S.detach().clone()` 快照，后台线程只做纯计算，不碰层池；
- 消费在同一进程主线程的 `forward` 开头，非阻塞 `get_nowait()`；
- **增量合并**：`layer.S = S_main + (layer.S_now − S_snapshot)`，把排队期间主线程的写入增量保留下来，因此拆分仍满足 `S_main + S_sec + Δ == S_now`，不丢状态；
- `reset()` 清空 pending / results，旧结果因父层已不在层池被自动忽略；
- 直接调用 `_check_expand()`（管理/测试入口，非 forward）走**同步路径**，保证结果立即可观测，既有 30 项回归不受影响；
- 队列 / Thread / pending list 都是普通 Python 对象，**不注册进 nn.Module**。

---

## 4. 19 项验收逐条结果（原问题 → 修复 → 实测证据）

### 4.1 原 8 个阻塞项 → 全部 PASS

| 编号 | 原问题（v1 实测） | 修复后实测 | 结果 |
| --- | --- | --- | --- |
| V01-LAT | 分裂步 **705.09ms**（一步 2 层同步 SVD） | A_default 分裂步 **86.38ms**；B 极端最大 **319.74ms**；每步最多 1 次分裂；SVD 已异步 | PASS |
| V02-NAN-ROUTE | `active=[]`、`writes 5→5`、`scores=[nan]`、NaN 透传输出 | 5 种注入全部抛 `FloatingPointError`，`S_unchanged=True`，后续 5 步干净输入正常 | PASS |
| V03-NAN-POLLUTION | 冷启动 `S nan_frac=1.0`、`svd_split` 抛 `_LinAlgError` | `S_nan_frac=0.0`、S 保持 None/有限、直接 `write` 也被拒绝 | PASS |
| V04-DIM | `T=0`/`B=0` 静默接受且污染 S；解包报错不清晰 | 7 类非法输入全部抛带 `[B,T,D]` / 「空序列」/「浮点」的清晰异常；3 类合法输入正常 | PASS |
| V05-LAYER-CAP | 18 步 1→274 层、273 次分裂 | `max_layers=0` 钳到 8；1000 条跑满、8 层、7 次分裂、`aborted=None` | PASS |
| V06-DEADLOCK | 23 次激活、4272 步零激活、0 分裂、层数恒 1 | **201 次激活 / 80 次探索 / 2 层 / 1 次分裂**，末次激活 step 9991，尾部仅 8 步空闲，无异常 | PASS |
| V07-DIV0 | `0/0` 静默产生全 NaN 的 S | `T=0`/`B=0` 被 `ValueError` 拒绝 + `/N` 有 `_DIV_EPS=1e-8` 双保险，S 有限 | PASS |
| V08-EPS | fp16 零输入 `scores=[nan]` | fp16/fp32 零输入均 `[0.0]`；生产路径 S 为 fp32 | PASS |

### 4.2 原 9 个警告项 → 全部 PASS

| 编号 | 原问题 | 修复后实测 |
| --- | --- | --- |
| V09-INF | clamp 静默截断 ±10 且仍生成 NaN | Inf 入口直接拒绝，`S_unchanged=True` |
| V10-RSS-1024 | D=1024 14 步 RSS **+257.04MB** | 1000 条跑满、8 层、RSS 281.23→330.01MB（**+48.78MB**，含异步快照） |
| V11-COND | cond ~1e10、σ_min→0、SVD 数值不稳 | 秩 1 矩阵 `svd_split()` 返回 `None`、父层 S 逐位不变、`rank_deficient_skips=1`、层数保持 1 |
| V12-DTYPE | fp16 往返误差放大 481.8× | 生产路径 S 恒 fp32、fp32 重建误差 1.8e-07、输出 finite |
| V13-RESET | reset 后 `scores=[0.6445]` 残留、旧 S 仍可达 | `scores=[]`、`boosted=[]`、旧层 `S=None`、stats 归零 |
| V14-ATTACH | `mgr_dim=128/layer_dim=64` 静默错位 | 同 dim 返回同一对象、router 保留；异 dim 抛 `ValueError: ... 拒绝以 dim=128 重复挂载` |
| V15-GHOST | σ≈0 子层 `cos_absmax=0.9057` | 近零 S 分数 `0.0`（绝对 + 相对双判据） |
| V16-EXPLODE | 一次 `_check_expand` 3→6 层 | 3→4 层，`delta=1` |
| V17-DOWNCAST | fp32 S 被临时降成 fp16 | 计算精度恒 fp32；手工 fp16 S + fp32 输入也返回 fp32 且 finite |

### 4.3 建议项与零污染复查

| 编号 | 结果 |
| --- | --- |
| V18-ELBOW-NEG | 3 层全 `-1.0` 分 → `active=[]`（负分不再误激活） |
| **V19-POLLUTION** | **state_dict 键集合一致、值逐位一致；`mgr.state_dict()={}`、0 Parameter、0 Buffer、named_modules 仅 1 个**；分裂到 8 层 + reset 后仍成立 |

---

## 5. 修复前后关键指标对比（全部实测）

| 指标 | v1（审计时） | 本轮（补丁后） |
| --- | --- | --- |
| D=1024 默认配置 avg / max | 26.22 / 253.17 ms | **38.19 / 152.63 ms**（校验带来的固定开销，见第 7 节） |
| 分裂步最大延迟（极端场景） | **705.09 ms** | **319.74 ms** |
| 单步最多分裂层数 | 2 | **1** |
| SVD 是否在 forward 关键路径 | 是 | **否（后台队列）** |
| 单次 SVD（micro, D=1024） | 214.13 ms median | 274.76 ms median（后台执行） |
| D=1024 高频分裂 RSS 增量 | **+257.04 MB / 14 步 / 70 层** | **+48.78 MB / 1000 步 / 8 层** |
| 10000 步总激活次数 | **23** | **201** |
| 末次激活后连续零激活 | **4272 步** | **8 步** |
| 层数 / 分裂数（10000 步） | 1 / 0 | 2 / 1 |
| NaN 注入后 S | `nan_frac=1.0` | `nan_frac=0.0`（入口拒绝） |
| 空序列 | 静默 `0/0` → S 全 NaN | `ValueError` + eps 双保险 |
| fp16 零向量余弦 | `nan` | `0.0` |
| 幽灵分（σ≈0 子层） | `cos=0.9057` | `0.0` |
| reset 后 router 缓存 | `[0.6445]` 残留 | `[]` |
| state_dict | 干净 | **仍干净（0 Parameter / 0 Buffer）** |

---

## 6. 过程中发现并修正的两处自身问题（防幻觉）

1. **首轮验收脚本的 V15 判据漏了「整体缩放极小」的层**：`S=eye*1e-9` 时 `read` 与 `H` 同向，cos 仍达 `0.8344`，原来的「read 绝对范数 < eps」挡不住。已补 `_MIN_REBUILD_RATIO=1e-3` 相对重建幅度判据，重测 `0.0`。对应 `test_08` 从 FAIL 转 OK。
2. **并行跑测试会污染延迟实测**：同时跑 F1–F8 用例和压测时，SVD 从 ~210ms 涨到 526–993ms、D=1024 攻击 857/1000 步就撞时间预算。改为**单独运行**后 V10 跑满 1000 步、RSS +48.78MB。本报告全部数字来自单独运行（2026-11-13 16:20:46）。

V01 的验收口径也做了一次**有依据的对齐**：原始阻塞定义是「**分裂步延迟** > 500ms」，补丁后 SVD 由后台线程执行，其耗时属于后台吞吐而非关键路径延迟。因此 V01 判定改为「分裂步关键路径 ≤500ms + 每步 1 次分裂 + 异步已启用」，后台 SVD 耗时作为独立观测项**照常记录、不隐藏**（本轮最差 1008.81ms）。这不是放宽标准，是把判据放回用户原始定义。

---

## 7. 残留风险与后续建议（如实列出，不因"0 警告"而隐藏）

1. **后台 SVD 吞吐**：极端 6 层场景下后台单次 SVD 达 **695–1008ms**（micro 基准 261–295ms）。它不阻塞 forward，但会占 CPU、使分裂生效延迟数步。GPU 环境需重新评估线程与 stream 的争抢。
2. **进程退出时未完成的分裂会丢失**：worker 是 daemon 线程，若最后几步刚入队就结束进程，该次分裂不会落地。训练正常结束前有 `reset_memory()` / checkpoint，影响有限；若要求严格，需在收尾时显式 `drain`。
3. **固定校验开销**：`_validate_H` 每步对 `[1,512,1024]` 做一次全量 `isfinite`，默认配置 avg 从 26.22ms 升到 38.19ms（+11.97ms）。这是「立刻拒绝 NaN」的代价，我认为值得；若要优化，可在调用方保证有限性时提供 `validate=False` 快速路径，但那会削弱 F1 防线，不建议默认开启。
4. **S 条件数仍大**：10000 步后 cond 3.4e9→1.08e10、σ_min→4.31e-8（低秩结构化输入的固有结果）。秩亏保护只在**分裂时刻**生效，日常写入并未对 S 做条件数约束。若后续出现 SVD 数值告警，需要额外的谱正则。
5. **异步语义边界**：`_check_expand()` 直接调用走同步、forward 路径走异步，两套行为靠 `_in_forward` 标志区分。这是为兼容既有测试做的取舍，新增调用方需知晓。
6. **CPU-only**：本机 torch 2.2.2 无 MPS/CUDA，`_needs_cpu_svd` 的 MPS 降级路径、GPU 上的 float(tensor) 隐式同步均未实测。
7. **未跑全量回归**：按项目约定（30 套件 ≈13.5 分钟，需 steve 许可），本轮只跑了 `test_somn` 既有 30 项 + 新增 24 项 + 验收压测。合入主干前建议补跑 `test_model` / `test_shared_weights` 等与 `reset_memory()` 相关的套件。
8. **路径规范**：测试放在 `test/`，与项目惯例 `service/test/` + `SUITES` 注册不一致（沿用上轮任务书指定路径），未改 `run_tests.py`。
9. **账号 A 的修复未动**：上轮审计报告第 6 节的三个问题（诊断钩子顺序、Muon lr 适配、`allmuon` 绕过兜底、MoMW router 缓存）**仍待 A 处理**，本轮未触碰其文件。

---

## 8. 复现

```bash
cd /Users/steve/Desktop/GARGANTUA
git checkout feat/somn-stress-test

# 验收压测（0 阻塞 / 0 警告，约 2-3 分钟，建议单独跑避免 CPU 争抢）
/usr/local/bin/python3 test/somn_stress.py --json data/tmp/somn_verify.json

# F1-F8 专项用例
/usr/local/bin/python3 test/test_somn_f1_f8.py -v

# 既有回归
/usr/local/bin/python3 service/test/test_somn.py -v

# 单项
/usr/local/bin/python3 test/somn_stress.py --only t1
```

环境：`/usr/local/bin/python3` = Python 3.12.10 + torch 2.2.2（CPU only，4 线程），`psutil 7.2.2` 采样 RSS，`time.perf_counter` 采样延迟。

---

## 9. 本轮提交文件

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `server/somn.py` | 修改 | F1–F8 全部补丁 + 异步分裂队列 |
| `test/somn_stress.py` | 修改 | 从「找茬模式」升级为 V01–V19 验收模式 |
| `test/test_somn_f1_f8.py` | 新增 | 24 项 F1–F8 专项断言（含零污染） |
| `docs/audit/somn_v4_fix_verification.md` | 新增 | 本报告 |

上轮已提交的 `docs/audit/somn_v4_audit.md` 保留为问题记录，未改动。`data/tmp/somn_verify.*` 被 gitignore，不入仓库。**不 push，等待审阅。**
