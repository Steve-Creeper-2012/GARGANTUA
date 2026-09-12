# Gargantua 架构设计文档 v1.0
# 日期: 2026-09-08
# 状态: 定稿（以此文档为准，之前讨论有冲突处以此为准）

---

## 0. 核心原则

- **一套内核架构，两种执行模式外壳**
  - Edith（端侧/眼镜）: 固定 33.333ms 周期，硬实时，30 tok/s 稳定输出
  - Jarvis（云端/PC）: 自由节奏，硬件上限决定吞吐，不强制对齐
- **放弃 SSM 状态矩阵**，改用 Fast Weight（Schmidhuber 1991 + Schlag et al. 2021 优化）
- **逐像素视频处理**，不采用 Patch 级别（当前业界无人做到）
- **所有记忆更新机制统一**：Fast Weight 双层结构替代传统 KV Cache 膨胀问题

---

## 1. 整体架构：硬耦合 Encoder-Decoder

### 1.1 架构保留理由

CEO 确认：Encoder-Decoder 架构有意义，保留。虽然增加了复杂度，但带来的收益：
- **输入/输出天然分离**：Encoder 只处理用户输入，Decoder 只生成模型输出
- **双向理解能力**：Encoder 内部使用双向注意力，整帧/整段输入同时可见，寻找更多关系
- **硬耦合传递**：Encoder 每层隐藏状态直接注入 Decoder 对应层，不是只有最后一层

### 1.2 精确结构

```
输入层（多轨并行）
├─ 视频轨: 逐像素 RGBA (H×W×4)
├─ 音频轨: 序列 token (1-5 tok/步)
├─ 文本轨: 序列 token (1 tok/步)
└─ 控制轨: 控制信号 token

        ↓
    [ENCODER] (多层 Transformer，双向注意力)
    Enc-L1 → Enc-L2 → ... → Enc-LL
        ↓ (硬耦合: 每层输出直接传递)
    [DECODER] (多层 Transformer，因果自注意力 + 交叉注意力)
    Dec-L1 → Dec-L2 → ... → Dec-LL
        ↓
输出层（多轨并行）
├─ 文本输出
├─ 音频输出
└─ 控制输出
```

**硬耦合公式**（第 l 层）：

Encoder:
```
H_enc^(l) = BiSelfAttn^(l)(H_enc^(l-1))
```

Decoder:
```
H̃_dec^(l) = CausalSelfAttn^(l)(H_dec^(l-1), SharedKV[l].dec)
H_dec^(l) = FFN^(l)(H̃_dec^(l) + CrossAttn^(l)(H̃_dec^(l), SharedKV[l].enc))
```

CrossAttn 的 K/V 来自 **Encoder 同层输出**，不是最后一层。

---

## 2. 输入 vs 输出：如何区分？

**不需要分两套 token。Encoder-Decoder 架构天然区分：**

| | 来源 | 处理位置 | 注意力类型 |
|---|---|---|---|
| 用户输入 | 麦克风/摄像头/键盘/文件 | Encoder | 双向注意力 |
| 模型输出 | 模型自回归生成 | Decoder | 因果自注意力 |

Decoder 通过两种注意力机制自然区分：
- **自注意力**：看到的是自己的输出历史（因果 mask）
- **交叉注意力**：看到的是 Encoder 处理后的用户输入（双向，无 mask）

**模型不需要"知道"这是用户输入还是自己输出——架构已经分开了。**

---

## 3. SharedKV Cache 设计

### 3.1 核心问题

传统 Transformer KV Cache 不适用，因为：
- 输入和输出交错进行（同时输入同时输出）
- 输入事件（视频帧、音频块）和输出 token 索引方式不同
- 不能打断 Decoder 自回归流程来插入新输入

### 3.2 Gargantua SharedKV 结构

每层一个 SharedKV 单元：

```python
SharedKV[l] = {
    # Encoder 侧: 按"输入事件"索引
    'enc_steps': [
        {'step': 0, 'k': tensor, 'v': tensor},  # 零状态
        {'step': 1, 'k': tensor, 'v': tensor},  # 第1次用户输入
        {'step': 2, 'k': tensor, 'v': tensor},  # 第2次用户输入
        ...
    ],

    # Decoder 侧: 按"token 位置"索引
    'dec_pos': [
        {'pos': 0, 'k': tensor, 'v': tensor},   # 生成的第1个token
        {'pos': 1, 'k': tensor, 'v': tensor},   # 生成的第2个token
        ...
    ],
}
```

### 3.3 并发更新机制（不中断 Decoder）

```
Decoder 正在生成 token_t:
  读取 SharedKV[l].enc_steps[-1]   # 最新编码状态
  读取 SharedKV[l].dec_pos[0:t]    # 历史生成 token
  → 计算 token_t → 输出

与此同时，用户输入到达:
  Encoder 异步处理 → 计算新的 enc_k, enc_v
  SharedKV[l].enc_steps.append(new_state)  # 只是列表 append

Decoder 生成 token_{t+1}:
  自动读取 enc_steps[-1]（新追加的状态）
  → token_{t+1} 已包含新输入信息
```

**关键：Decoder 不需要重启、不需要清空、不需要重新计算。**

### 3.4 交叉注意力读取策略

| 策略 | 读取范围 | 适用场景 |
|---|---|---|
| Latest | 只读 enc_steps[-1] | Edith 实时对话 |
| Full | 读全部 enc_steps | Jarvis 长上下文 |
| Window | 最近 W 个 enc_steps | 长视频，滑动窗口 |

---

## 4. Fast Weight 双层设计（替代 SSM）

### 4.1 放弃 SSM 的理由

- SSM 状态矩阵训练困难
- 信息量不如 Fast Weight（Fast Weight 是完整权重矩阵，SSM 只是状态向量）
- Fast Weight 有完整论文支撑（Schmidhuber 1991 → Schlag 2021）

### 4.2 基础公式（Schmidhuber 1991）

```
a(i), b(i) = W_a · x(i),  W_b · x(i)
W(i) = σ(W(i-1) + a(i) ⊗ b(i))
y(i) = W(i) · x(i)
```

- W_a, W_b: Slow Weight（反向传播训练）
- W(i): Fast Weight（每步由 Slow Net 生成更新）
- ⊗: outer product

### 4.3 Delta Rule 优化（Schlag et al. 2021）

```
W(i) = W(i-1) + β(i) · (v(i) - v̄(i)) ⊗ φ(k(i))
```

其中：
- v̄(i) = W(i-1) · φ(k(i))   # 检索到的旧值
- β(i) = σ(W_β · x(i))       # 动态学习率/写强度（模型自学习）

**优势：可以覆盖/修正已有记忆，解决容量上限问题。**

### 4.4 双层 Fast Weight 结构

| 层级 | 名称 | 更新频率 | 特性 | 作用 |
|---|---|---|---|---|
| 第一层 | Fast Weight L1 | 每次输入输出 | 快速写入，窗口大，压缩所有输入 | 实时感知，不保存完整历史 |
| 第二层 | Fast Weight L2 | 选择性更新 | 门控控制，精度更高，选择性记忆 | 长期积累，高质量记忆 |

**第一层（感知层）**：
- 每次 Encoder/Decoder 迭代都更新
- 不管输入质量，全部压缩到记忆矩阵
- 用于即时上下文感知

**第二层（积累层）**：
- 通过门控决定是否更新
- 只保留"重要"信息
- 用于跨 session 长期记忆

### 4.5 与 KV Cache 的关系

**视频序列不进入标准 KV Cache。**

原因：
- 一帧 400 万 token，10 帧就 4000 万，KV Cache 立刻爆炸
- 即使压缩，也会把早期重要内容挤出

**替代方案：**
- 视频经过 Encoder 后，其隐藏状态通过 **第一层 Fast Weight** 压缩到记忆矩阵
- Decoder 交叉注意力读取的是 Fast Weight 记忆矩阵，而不是原始视频 KV
- 这样既保留了视频的序列信息（通过 Fast Weight 的时间衰减），又避免了 KV Cache 爆炸

**文本/音频**：可以进入 KV Cache（token 数可控）。

---

## 5. 视频处理：逐像素

### 5.1 像素格式

- **RGBA 四通道**（不是 RGB 三通道）
- **10bit 色深**（不是 8bit）
- 每像素 4 个参数，每个参数 10bit
- 总颜色空间：1024^4 种

### 5.2 帧内并行，帧间序列

```
帧 t:
  像素(0,0) ──┐
  像素(0,1) ──┼──→ 帧内双向注意力（所有像素同时互相可见）
  像素(1,0) ──┤      ↓
  ...        ──┘  输出: 帧聚合状态

帧 t+1:
  同上，但帧间通过 Fast Weight / KV Cache 建立时序关系
```

### 5.3 超过并行宽度：分 Patches

模型并行宽度上限（如 400 万 token = 720P）：

- 输入 4K 视频（>400 万像素）
- 分为多个 Patches（如 4 个 Patches，每个 100 万像素）
- 每个 Patch 单独过 Encoder
- 所有 Patches 输出合并为同一帧的表示
- 对模型来说，仍然是一个时间步（帧内并行）

**注意：这里的 Patches 是硬件/调度层面的分块，不是 ViT 的语义 Patch。模型仍然逐像素理解，只是硬件一次处理不完，分多次处理。**

### 5.4 Delta 帧策略（降低算力）

灵感来源：H.265 视频编码（I帧/P帧/B帧）

```
关键帧（I-帧）: 全像素输入
  └─ 每 30 帧（1秒）强制一个关键帧

中间帧（P-帧）: 只输入变化 token
  └─ 只输入与上一帧相比发生变化的像素

重置规则:
  └─ 如果某帧全像素变化（场景切换），该帧自动变为关键帧
  └─ 重置计时器，该帧作为新 Block 的开始
```

**示例：**
```
帧1: 全像素输入（关键帧，Block开始）
帧2: 变化像素（P-帧）
帧3: 变化像素（P-帧）
帧4: 全像素变化 → 自动变为关键帧，Block重置
帧5-33: 变化像素（P-帧）
帧34: 1秒到达 → 强制关键帧
```

**收益：**
- 静态场景下，90%+ 像素不变化，算力降低 90%+
- 只有场景切换时才全量输入
- 保持逐像素精度，不损失细节

### 5.5 注意力策略：KDA 线性注意力

```
75% 层使用 KDA 线性注意力（降低 O(N²) 到 O(N)）
25% 层使用标准 Softmax 注意力（保留精度）
```

参考：Moonshot AI FlashKDA / Kimi-Linear

---

## 6. 各模态处理方式

| 模态 | 帧间关系 | 帧内/序列内关系 | 每步 token 数 | 进入 KV Cache？ |
|---|---|---|---|---|
| 视频 | 序列（帧1→帧2→帧3） | 并行（帧内所有像素同时） | 400万（720P） | **否** → Fast Weight L1 |
| 图片 | 单帧（无序列） | 并行（所有像素同时） | 400万 | **否** → Fast Weight L1 |
| 音频 | 序列（采样点按时间） | 聚合（33ms内采样点→1-5token） | 1-5 | **是** |
| 文本 | 序列（token-by-token） | 无 | 1 | **是** |
| 控制 | 序列（指令流） | 无 | 1 | **是** |

---

## 7. 两种执行模式

### 7.1 Edith 模式（固定 33.333ms 周期）

```
周期 n（33ms）:
  Phase 1（0-10ms）: Encoder 运行
    - 处理当前视频帧 + 音频块 + 文本
    - 输出 enc_state → SharedKV.enc_steps.append()

  Phase 2（10-33ms）: Decoder 运行
    - 生成 1 个 token
    - 交叉注意力读取 enc_steps[-1]
    - dec_pos.append()

  周期结束，等待下一 33ms 中断
```

**特性：**
- 严格 30 tok/s 输出
- 说话快慢由内容控制（停顿 token / 语速 token）
- 小模型（1-3B MOE），端侧芯片可承载
- 视频帧每周期必到，Encoder 每周期必运行

### 7.2 Jarvis 模式（自由节奏）

```
while True:
  1. 非阻塞采样各轨道最新数据
  2. 组装输入 Token Block Vector
  3. 如果存在新输入: Encoder 运行 → 更新 SharedKV.enc_steps
  4. Decoder 生成下一个 token → 更新 SharedKV.dec_pos
  5. 分发输出
  6. 立即进入下一次迭代（无固定 sleep）
```

**特性：**
- 硬件多快，循环多快
- 用户输入随时打断，下一次迭代自动看到
- 支持边看边操控（Blender 建模等）
- 大模型（3-5B+），PC/云端运行

---

## 8. 记忆与状态

### 8.1 三层记忆体系

| 层级 | 机制 | 容量 | 更新方式 | 作用 |
|---|---|---|---|---|
| 短期 | SharedKV Cache | 最近 N 个输入事件 + M 个输出 token | 每步追加 | 即时上下文 |
| 中期 | Fast Weight L1 | 完整权重矩阵（每帧更新） | Hebbian + Slow Net | 当前 session 感知 |
| 长期 | Fast Weight L2 | 完整权重矩阵（选择性更新） | 门控 Slow Net | 跨 session 积累 |

### 8.2 S_t 循环状态位

- 不依赖外部时钟，依赖帧序号/迭代序号
- 内部维护计数器：s_t^{time} = s_{t-1}^{time} + 1
- 模型通过 s_t^{time} 感知"这是第几帧/第几步"
- Edith 和 Jarvis 共用同一套计数逻辑

---

## 9. 训练数据格式

```python
{
    "mode": 0,  # 0=Edith, 1=Jarvis
    "step": n,  # 迭代序号（不绑定真实时间）

    "inputs": {
        "video": {
            "frame_type": "I" or "P",  # 关键帧或变化帧
            "pixels": [H, W, 4],         # RGBA，10bit
            "changed_mask": [H, W],      # P-帧时标记变化像素
        },
        "audio": [1-5, d_model],
        "text": [1, d_model],
        "control": [1, d_model],  # 或 NULL
    },

    "metadata": {
        "n": 帧序号/迭代序号,
        "M": 0 or 1,  # 模式标志
        "wall_time": 真实时间戳（可选）,
    },

    "targets": {
        "text_out": [1, d_model],
        "audio_out": [1, d_model],
        "control_out": [1, d_model],
    }
}
```

**训练策略：**
- 70% Edith 样本 + 30% Jarvis 样本
- 同一 batch 可混合
- 损失函数：只计算存在的 target 轨道

---

## 10. 关键公式汇总

### Encoder 双向注意力
```
H_enc^(l) = BiSelfAttn^(l)(H_enc^(l-1))
```

### Decoder 因果自注意力 + 交叉注意力
```
H̃_dec^(l) = CausalSelfAttn^(l)(H_dec^(l-1), SharedKV[l].dec)
H_dec^(l) = FFN^(l)(H̃_dec^(l) + CrossAttn^(l)(H̃_dec^(l), SharedKV[l].enc))
```

### Fast Weight 基础更新（Schmidhuber 1991）
```
W_{t+1} = λ · W_t + η · SlowNet(h_t, ∇_t^{pseudo})
```

### Fast Weight Delta Rule（Schlag 2021）
```
W(i) = W(i-1) + β(i) · (v(i) - v̄(i)) ⊗ φ(k(i))
β(i) = σ(W_β · x(i))
v̄(i) = W(i-1) · φ(k(i))
```

### KDA 线性注意力
```
75% 层: 线性注意力（O(N) 复杂度）
25% 层: 标准 Softmax 注意力（O(N²) 复杂度）
```

---

## 11. 待实现清单

### Phase 1（0.0.1 alpha，100M 参数验证）
- [ ] Fast Weight 基础实现（outer product 版本）
- [ ] Delta Rule 优化版本
- [ ] 双层 Fast Weight 门控机制
- [ ] SharedKV 并发安全实现
- [ ] 硬耦合 Encoder-Decoder（每层传递）
- [ ] 视频逐像素输入（简化版，低分辨率测试）
- [ ] Delta 帧策略（简化版）
- [ ] Edith 固定周期调度器
- [ ] Jarvis 自由运行调度器
- [ ] 纯 PyTorch 实现，无 CUDA kernel

### Phase 2（0.1 beta，3-5B 参数）
- [ ] KDA 线性注意力集成
- [ ] 完整视频分辨率支持（720P+）
- [ ] 多轨并发优化
- [ ] 性能优化（考虑 CUDA kernel）

---

## 12. 参考文献

1. Schmidhuber, J. (1991). Learning to control fast-weight memories. FKI-147-91.
2. Schmidhuber, J. (1992). Learning to control fast-weight memories: An alternative to dynamic recurrent networks. Neural Computation, 4(1), 131-139.
3. Schmidhuber, J. (1993). Reducing the ratio between learning complexity and number of time varying variables in fully recurrent nets. ICANN.
4. Schlag, I., Irie, K., & Schmidhuber, J. (2021). Linear Transformers Are Secretly Fast Weight Programmers. ICML.
5. Schlag, I., Munkhdalai, T., & Schmidhuber, J. (2021). Learning Associative Inference Using Fast Weight Memory. ICLR.
6. Moonshot AI. FlashKDA / Kimi-Linear (开源实现).

---

*本文档为 Gargantua 项目架构定稿。所有实现以此为准。*
