# Gargantua 框架规范 v0.3 — 三层记忆 + 自进化架构

> **代号**: Gargantua（黑洞，星际穿越致敬）  
> **定位**: 承载大型人工智能的底层统一框架，支持自主进化与私人定制  
> **角色**: Edith（前端超轻量聊天模型）与 Jarvis（后端大任务模型）共用同一套框架  
> **协议**: Copyleft（GPL-3.0 / AGPL-3.0 待最终确认）  
> **阶段**: Phase 1（2026.9.5–10.5），一个月冲刺：确定框架 → 写代码 → 训练100M验证模型  
> **竞品参考**: OpenAI Bel（10T参数，双循环自进化）、Schmidhuber FWP（1991）、DeltaNet

---

## 1. 架构总览：三层记忆 + 四层存储

### 1.1 核心架构图

```
输入 (文本/图像/音频/视频)
    ↓
[Encoder] ——→ 隐藏状态 H_enc
    ↓
[Memory Layer] ←—— 全局记忆表示 M_global
    │              (Fast Weight + Slow Weight + SS视频缓存)
    ↓
[Decoder] ——→ 输出 (文本/图像/音频)
    ↑
[Base Weights] ——→ 预训练知识（冻结/微调）
```

**全程参与**：Memory Layer 的输出 M_global 在每一层都参与计算（通过残差连接或门控融合）。

### 1.2 三层记忆系统

| 层级 | 名称 | 生物映射 | 数学对象 | 时间尺度 | 可迁移性 |
|------|------|----------|----------|----------|----------|
| **L0** | **Base Weights** | 基因/本能 | W_base (预训练) | 永久 | ❌ 绑定模型 |
| **L1** | **Fast Weight** | 海马体短期记忆 | S_t ∈ R^{d×d} | 当前session | ❌ 临时缓存 |
| **L2** | **Slow Weight** | 大脑皮层长期记忆 | W_mem ∈ R^{d×d} | 跨session | ✅ **可迁移到新模型** |

### 1.3 四层存储体系

| 层级 | 名称 | 用途 | 容量 | 精度 |
|------|------|------|------|------|
| **T0** | **KV Cache** | 精确寻址，逐字检索 | 0–100万tokens | FP16/BF16 |
| **T1** | **Fast Weight** | 短期上下文，实时更新 | 1–2个session | FP32 |
| **T2** | **Slow Weight** | 长期人格/偏好/知识 | 无上限（存档） | FP32 |
| **T3** | **SS Video Cache** | 视频流快速缓存（低秩） | 1小时视频 | FP16 |

---

## 2. Base 模型：Encoder-Decoder Transformer

### 2.1 标准架构

```python
class BaseModel:
    def __init__(self, config):
        self.encoder = TransformerEncoder(config.num_layers // 2)
        self.decoder = TransformerDecoder(config.num_layers // 2)
        # 总层数：Jarvis=32层, Edith=8层

    def forward(self, x_input, x_output=None):
        # Encoder: 处理输入（文本/图像/音频编码）
        H_enc = self.encoder(x_input)

        # Decoder: 自回归生成输出
        if x_output is not None:
            H_dec = self.decoder(x_output, encoder_hidden=H_enc)
        else:
            H_dec = self.decoder.generate(H_enc, max_length=...)

        return H_dec
```

### 2.2 注意力机制

```python
# 1/3 层: 标准 Softmax Attention（精确检索层）
O_softmax = Softmax(Q K^T / √d) V
# 用于: 代码、事实、引用、精确匹配

# 2/3 层: KDA 线性注意力（长上下文层）
O_kda = Q · (K^T V)  # O(1)状态复杂度
# 用于: 长文档、流式推理、视频理解
# 参考: FlashKDA (Moonshot AI)

# 层类型配置
layer_types: List[str] = ["softmax", "kda", "kda", "softmax", "kda", "kda", ...]
```

### 2.3 Attention Residuals (AttnRes)

```python
# 跨层注意力替代标准残差
# 早期token信息通过AttnRes直达深层，防止1M上下文遗忘
h_l = Σ_{i=0}^{n-1} α_{i→l} · v_i
# α通过每层伪查询w_l与前面Block表示做Softmax得到
# 等效Scaling Law: 匹配基线1.25×算力的Loss
# 参考: MoonshotAI/Attention-Residuals
```

### 2.4 MoE (Mixture of Experts)

| 参数 | Jarvis | Edith |
|------|--------|-------|
| 专家总数 | 64/128 | 8–16 |
| 激活专家 | top-2/top-4 | top-2 |
| 总参数量 | 1B–100B | <2B (激活<500M) |
| 参考 | MoonEP动态冗余专家 | 简化版 |

---

## 3. Fast Weight 系统：海马体短期记忆

### 3.1 核心数学（FWP + Delta Rule）

基于Schmidhuber FWP (1991) 和 DeltaNet 改进：

```python
# Slow Net: 生成记忆指令（可训练）
v_t = x_t · W_v          # 值向量: 记什么
k_t = x_t · W_k          # 键向量: 检索键  
q_t = x_t · W_q          # 查询向量: 怎么读
β_t = sigmoid(x_t · w_b) # 学习率: 记多强

# Fast Weight: 实时更新（不可训练，前向确定性）
# Delta Rule: 只更新"预测错误"的部分
error_t = v_t - S_{t-1} · k_t   # 如果S_{t-1}已能预测v_t，error≈0
S_t = S_{t-1} + β_t · outer(error_t, k_t)

# 读取（直接矩阵乘）
h_fast = S_t · q_t

# 输出融合
h_out = h_base + α · LayerNorm(h_fast)   # α=0.1, Base占90%
```

### 3.2 为什么稳定？

| 机制 | 效果 |
|------|------|
| **Delta Rule** | 只记"惊讶"，预期之内不更新 |
| **β_t 门控** | 学习率由Slow Net动态控制，可训练 |
| **Base占90%** | 即使S_t全错，输出只是轻微偏离 |
| **谱归一化** | 每100token执行，σ_max≤1 |
| **稀疏更新** | top-k 5%神经元参与，95%保持Base |

### 3.3 稀疏更新实现

```python
# 只更新最活跃的5%维度
importance = |error_t| * |k_t|   # 外积重要性
active_dims = top_k(importance, k=int(0.05 * d))
mask = zeros(d); mask[active_dims] = 1

# 掩码外积
S_t = S_{t-1} + β_t * (error_t * mask)[:, None] * (k_t * mask)[None, :]
# 实际更新量: 0.25%的矩阵元素 (5% × 5%)
```

### 3.4 谱归一化

```python
# 每100token执行一次
u = random_normal(d)
for _ in range(3):           # 幂迭代3步
    u = S_t @ u
    u = u / norm(u)
sigma = u.T @ S_t @ u
if sigma > 1.0:
    S_t = S_t / sigma
```

### 3.5 视频流专用：简化 SS 缓存

```python
# 视频数据量太大，Fast Weight完整矩阵不够快
# 额外增加低秩SS缓存，专用于视频patch流

class VideoSSCache:
    def __init__(self, d=512, rank=64):
        self.A = zeros(d, rank)   # 左因子
        self.B = zeros(rank, d)   # 右因子
        self.ptr = 0

    def update(self, patch_token):
        # 每帧196个patch，每秒30帧 = 5880 tokens/秒
        # SS缓存低秩更新，O(d·rank)复杂度
        a = patch_token @ W_a      # d → rank
        b = patch_token @ W_b      # d → rank
        self.A += outer(patch_token, a)
        self.B += outer(b, patch_token)
        self.ptr += 1

    def read(self, query):
        # 低秩读取
        return query @ self.A @ self.B   # O(d·rank)

    def summarize(self):
        # 每小时总结一次，输出给Fast Weight
        summary = RNN(self.A, self.B)
        return summary
```

**为什么保留SS？**
- 视频流：30fps × 196patch × 3600秒 = 2100万patch/小时
- Fast Weight完整矩阵：每patch更新O(d²) = 16M FLOP，太慢
- SS低秩缓存：每patch更新O(d·rank) = 256K FLOP，快65倍
- 每小时总结一次，将SS内容压缩给Fast Weight/Slow Weight

---

## 4. Slow Weight 系统：皮层长期记忆

### 4.1 核心特性：可迁移

```python
# Slow Weight是固化后的独立Transformer层
# 关键特性：可以直接接到新模型上！

# 用户用了3年Jarvis-1.0，积累了100个Slow Weight层
# Jarvis-2.0发布时：
new_model = Jarvis2()          # 更强的Base模型
for layer in user_slow_weights:
    new_model.attach_memory_layer(layer)   # 直接接上！

# 结果：新模型更强 + 保留全部历史记忆
```

### 4.2 固化触发条件

```python
def should_consolidate():
    return (
        token_count >= 128000          # A: 周期性固化
        or user_explicit_mark()         # B: 用户标记"记住"
        or model_judges_important()     # C: 模型自主判断
        or session_ended()              # D: session结束总结
    )
```

### 4.3 固化过程：空闲时标准梯度下降

```python
# 前台（实时推理）不受影响
# 后台（低优先级线程）执行：

def consolidate(S_t, context_128K):
    # 1. 初始化新层
    W_new = initialize_from_S_t(S_t)   # 从Fast Weight提取初始化

    # 2. 标准梯度下降（冻结Base，只训W_new）
    optimizer = AdamW(W_new.parameters(), lr=1e-4)

    for epoch in range(10):
        for batch in context_128K.batches():
            # 前向: Base输出 + 新层
            h_base = Base(batch)
            h_new = SwiGLU(h_base, W_new)
            logits = Head(h_base + 0.1 * h_new)

            # 损失
            loss = cross_entropy(logits, batch.targets)

            # 反向: 只更新W_new
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # 3. 质量验证
    val_ppl = evaluate(W_new, validation_set)
    if val_ppl < baseline_ppl * 0.95:   # 降低5%以上
        # 通过 → 固化
        insert_layer(W_new, position="top")
        archive_old_if_needed()
        S_t *= 0.1                         # Fast Weight保留10%
        return True
    else:
        # 失败 → 丢弃
        return False
```

### 4.4 层管理：固定上限 + LRU替换

```python
MAX_LAYERS = 64   # Jarvis
# MAX_LAYERS = 16 # Edith

class LayerManager:
    def __init__(self):
        self.layers = []           # 活跃Slow Weight层
        self.archive = {}          # 存档层 (SSD)
        self.access_count = {}     # 访问计数

    def insert(self, W_new):
        if len(self.layers) >= MAX_LAYERS:
            # LRU替换
            oldest = min(self.layers, key=lambda l: self.access_count[l.id])
            self.archive[oldest.id] = oldest.to_disk()
            self.layers.remove(oldest)

        self.layers.append(W_new)

    def forward(self, h_base):
        h_mem = 0
        for layer in self.layers:
            h_mem += layer(h_base)
            self.access_count[layer.id] += 1
        return h_mem

    def recall(self, layer_id):
        # 用户提及旧内容时，按需加载
        if layer_id in self.archive:
            W_loaded = self.archive[layer_id].to_gpu()
            self.insert(W_loaded)
            return True
        return False
```

### 4.5 存档元数据

```python
layer_metadata = {
    "id": "layer_42",
    "created": "2026-09-01T10:00:00Z",
    "token_range": [128000, 256000],
    "topics": ["python", "machine_learning", "travel"],
    "emotions": ["curious", "happy"],
    "importance_score": 0.85,
    "user_feedback": [" helpful", "remember this"],
    "quality_score": 0.92,
    "last_accessed": "2026-09-05T14:00:00Z",
    "access_count": 15
}
```

---

## 5. Memory Layer：全局记忆层

### 5.1 架构位置

```python
# Memory Layer位于Encoder和Decoder之间
# 但输出参与所有层的计算

class MemoryLayer:
    def __init__(self, config):
        self.fast_weight = FastWeight(config.d)
        self.slow_weights = LayerManager(config.max_layers)
        self.ss_video = VideoSSCache(config.d, rank=64)

    def forward(self, H_enc, H_dec=None):
        # 读取Fast Weight
        h_fast = self.fast_weight.read(H_enc)

        # 读取Slow Weight
        h_slow = self.slow_weights.forward(H_enc)

        # 读取视频SS（如果有视频输入）
        h_video = self.ss_video.read(H_enc) if video_input else 0

        # 全局记忆表示
        M_global = h_fast + h_slow + h_video
        M_global = LayerNorm(M_global)

        # 参与Encoder各层（通过残差）
        H_enc_enhanced = H_enc + 0.1 * M_global

        # 参与Decoder各层（通过交叉注意力）
        if H_dec is not None:
            H_dec_enhanced = H_dec + CrossAttn(H_dec, M_global)

        return H_enc_enhanced, H_dec_enhanced
```

### 5.2 全程参与机制

```python
# 不是只在Encoder-Decoder之间加一层
# 而是Memory Layer的输出M_global在每一层都参与

for layer_idx in range(num_layers):
    # Base计算
    h_base = transformer_layer[h_base]

    # Memory注入（每层都加）
    h = h_base + gate[layer_idx] * M_global

    # gate可学习: 不同层需要不同程度的记忆参与
```

---

## 6. 自进化系统

### 6.1 用户偏好分析

```python
def analyze_user(session_log):
    # 每session结束后执行

    topics = extract_topics(session_log)           # 常聊话题
    style = analyze_style(session_log)              # 回答风格
    gaps = identify_knowledge_gaps(session_log)     # 知识缺口

    user_profile = {
        "name": extract_name(session_log),
        "topics": topics,
        "style": {
            "formality": style.formality,      # 0-1
            "verbosity": style.verbosity,      # 简洁/详细
            "humor": style.humor,
            "technical_depth": style.depth
        },
        "knowledge_gaps": gaps,
        "preferred_languages": ["zh", "en"],
        "active_hours": [9, 22]   # 用户活跃时段
    }

    # 生成个性化system prompt
    system_prompt = generate_persona_prompt(user_profile)

    # 固化到Slow Weight
    consolidate(system_prompt, context=session_log)

    return user_profile
```

### 6.2 自主信息觅食

```python
def information_foraging(knowledge_gaps):
    # 模型像人类一样"看新闻"、"学新知"
    # 空闲时执行

    for gap in knowledge_gaps:
        # 1. 生成搜索查询
        queries = generate_search_queries(gap)
        # e.g., "quantum computing basics 2026"

        # 2. 检索信息
        documents = web_search(queries) + local_knowledge_base.search(queries)

        # 3. 质量评估
        scored = [(doc, score_relevance(doc, gap), score_credibility(doc)) 
                  for doc in documents]
        scored.sort(key=lambda x: x[1] * x[2], reverse=True)

        # 4. 生成QA对
        for doc in scored[:5]:
            qa_pairs = generate_qa(doc)

            # 5. 自主学习（只更新Slow Weight）
            for q, a in qa_pairs:
                train_step(q, a, only_layers="slow_weight")

        # 6. 更新知识图谱
        update_knowledge_graph(gap, scored[0])

    # 7. 生成学习报告
    report = summarize_learnings(knowledge_gaps, scored)
    log(report)
```

### 6.3 System Prompt 自我进化

```python
def evolve_system_prompt():
    current = load_current_prompt()
    current_score = evaluate_prompt(current, test_cases)

    # 生成变体
    variants = [
        mutate_add_instruction(current),
        mutate_change_tone(current),
        mutate_reorder(current),
        mutate_combine(current, best_historical_prompts)
    ]

    # A/B测试
    for variant in variants:
        score = trial_run(variant, sample_conversations)
        if score > current_score * 1.05:   # 提升5%以上
            current = variant
            current_score = score
            consolidate(variant, context="prompt_evolution")

    save_current_prompt(current)
```

### 6.4 空闲训练调度

```python
priority_queue = [
    (0, "user_inference"),      # P0: 用户推理，最高优先
    (1, "consolidation"),        # P1: 固化（session结束）
    (2, "user_preference"),      # P2: 偏好分析
    (3, "information_foraging"), # P3: 信息觅食
    (4, "prompt_evolution"),     # P4: prompt进化
    (5, "knowledge_graph"),      # P5: 知识图谱维护
]

def scheduler(gpu_utilization):
    if gpu_utilization > 80%:
        return []   # 只处理P0
    elif gpu_utilization > 50%:
        return [P1]  # 固化
    elif gpu_utilization > 20%:
        return [P1, P2]  # 固化+偏好
    else:
        return [P1, P2, P3, P4, P5]  # 全部
```

---

## 7. KV Cache：精确寻址层

### 7.1 分层KV Cache

| 层级 | 范围 | 精度 | 机制 |
|------|------|------|------|
| L0 | 0 – 100万tokens | FP16/BF16 | 标准PagedAttention |
| L1 | 100万 – 200万 | INT8量化 | KDA线性注意力 |
| L2 | 200万+ | 丢弃/摘要 | S_t语义摘要 |

### 7.2 100万Token精确寻址

```python
# 目标: KV Cache精确保存最近100万tokens
# 100万 × 4096dim × 2B(FP16) × 2(K+V) = 16GB

# Jarvis: H100 80GB → 可容纳
# Edith: 端侧8GB → 只存32K

# 分页管理
kv_manager = PagedKVCache(
    page_size=256,           # 每页256 tokens
    max_pages=3906,          # 100万 / 256
    dtype=torch.bfloat16
)
```

### 7.3 时间衰减

```python
# 早期上下文注意力减弱（类似人类遗忘）
# 通过Attention Score乘以时间衰减因子

time_decay = 1 / (1 + age_in_tokens / T_half)
# T_half = 50万tokens (约等于"半天对话")

# 在Attention计算中:
scores = Q @ K.T / sqrt(d)
scores = scores * time_decay[None, None, :]   # 早期token分数降低
attn = softmax(scores)
```

---

## 8. 训练规范

### 8.1 四阶段训练

```
阶段1: Base预训练 (4周)
    标准Transformer+MoE+KDA+AttnRes
    数据: 通用语料 + 代码 + 多模态
    Fast Weight不参与 (S_t=0)

阶段2: Fast Weight预热 (1周)
    冻结Base
    训练Slow Net参数: W_v, W_k, W_q, w_b
    目标: 学会"生成有用的记忆指令"
    数据: 对话数据，评估指标: 下游perplexity

阶段3: 固化机制训练 (1周)
    冻结Base + Slow Net
    训练固化验证器 + HyperNet初始化
    目标: 学会"何时固化"、"如何初始化新层"

阶段4: 端到端微调 (2周)
    解冻Base
    Fast Weight按规则更新 (不BP通过S_t)
    Slow Weight按规则固化
    Base学会"生成适合被记忆"的表示
```

### 8.2 混合精度

| 组件 | 精度 | 原因 |
|------|------|------|
| Base权重 | BF16 | 预训练完成 |
| S_t (Fast) | FP32 | 实时更新，数值敏感 |
| W_mem (Slow) | FP32 | 长期累积 |
| 计算 | BF16 | 推理加速 |
| 梯度 | FP32 | 防止下溢 |

### 8.3 100M参数验证模型 (Phase 1目标)

```python
# 1个月内训练出Jarvis-100M验证概念

config_100M = {
    "d_model": 1024,
    "num_layers": 16,        # 8 encoder + 8 decoder
    "num_heads": 16,
    "d_ff": 4096,
    "num_experts": 8,
    "active_experts": 2,
    "max_seq_len": 32768,
    "vocab_size": 32000,

    # Memory
    "fast_weight_dim": 1024,
    "max_slow_layers": 8,
    "kv_cache_max": 128000,

    # 总参数量: ~100M
}

# 训练数据: 
# - 通用语料 50B tokens
# - 对话数据 10B tokens
# - 代码 20B tokens
# 训练时间: A100×8, ~1周
```

---

## 9. 推理规范

### 9.1 实时推理流程

```python
def inference_step(x_input, user_state):
    # 1. 加载用户状态
    S_t = user_state.fast_weight          # 32MB
    W_mems = user_state.slow_weights      # 活跃层
    kv_cache = user_state.kv_cache        # 16GB

    # 2. Encoder
    H_enc = encoder(x_input, kv_cache)

    # 3. Memory Layer
    M_global = memory_layer(H_enc, S_t, W_mems)

    # 4. Decoder (自回归)
    for i in range(max_new_tokens):
        H_dec = decoder_step(H_enc, M_global, kv_cache)
        token = sample(head(H_dec))

        # 5. 后台更新Fast Weight（异步）
        if i % 8 == 0:   # 每8 tokens批量更新
            enqueue_fast_weight_update(S_t, H_dec, token)

    # 6. 保存用户状态
    user_state.fast_weight = S_t
    user_state.kv_cache = kv_cache

    return tokens
```

### 9.2 视频流推理

```python
def video_stream_inference(video_stream):
    ss_cache = VideoSSCache(d=512, rank=64)

    for frame in video_stream:
        patches = patchify(frame)   # 196 patches
        for patch in patches:
            token = vision_encoder(patch)
            ss_cache.update(token)   # 低秩更新，快

        # 每秒总结一次
        if frame_count % 30 == 0:
            summary = ss_cache.summarize()
            fast_weight.update(summary)   # 转入Fast Weight

    # 每小时固化
    if hour_elapsed:
        consolidate(fast_weight.S_t, context=last_hour)
```

### 9.3 跨Session恢复

```python
def resume_session(user_id):
    # 1. 加载Slow Weight（长期记忆）
    slow_weights = load_from_disk(f"users/{user_id}/slow_weights/")

    # 2. 加载Fast Weight（上次session残留）
    fast_weight = load_from_disk(f"users/{user_id}/fast_weight.pt")
    fast_weight *= 0.5   # 衰减50%，上次session不是本次

    # 3. 加载用户画像
    profile = load_json(f"users/{user_id}/profile.json")
    system_prompt = profile["system_prompt"]

    # 4. 加载KV Cache（若启用持久化）
    kv_cache = load_kv_cache(f"users/{user_id}/kv_cache/")

    return UserState(fast_weight, slow_weights, kv_cache, profile)
```

---

## 10. 分布式与端侧

### 10.1 Jarvis分布式

```python
# 服务器部署
base_model = load_base("jarvis-100b")    # 共享，只读

per_user:
    fast_weight: 64MB (FP32, HBM)
    slow_weights_active: 2-8GB (FP32, HBM)
    slow_weights_archive: SSD集群
    kv_cache: 16GB (BF16, HBM)
    profile: 1MB (JSON)

batch_inference:
    # 不同用户共享Base，各自Fast/Slow Weight
    # vLLM多LoRA Batch模式
```

### 10.2 Edith端侧

| 参数 | 值 |
|------|-----|
| Base | 500M-1B参数 |
| Fast Weight | 512×512 FP32 = 1MB |
| Slow Weight max | 8层 |
| KV Cache | 32K tokens |
| SS Video | rank=32, 256KB |
| 自主训练 | 仅用户偏好（无信息觅食） |
| 常驻模式 | 后台进程，S_t持续演化 |

---

## 11. 时间线：Phase 1 冲刺 (2026.9.5–10.5)

| 周 | 目标 | 产出 |
|----|------|------|
| **W1** | 确定框架 | 本文档 v1.0 + 技术评审通过 |
| **W2** | 写代码 | PyTorch实现: Base + Fast Weight + Slow Weight + Memory Layer |
| **W3** | 训练100M | Jarvis-100M验证模型训练完成 |
| **W4** | 验证 + 迭代 | 评估自进化能力，修复bug，确定Phase 2计划 |

### 11.1 100M验证模型指标

| 指标 | 目标 |
|------|------|
| 基础perplexity | < 15 (on C4) |
| Fast Weight收敛 | 128K上下文后perplexity降低>5% |
| Slow Weight固化 | 固化后perplexity降低>3% |
| 跨session记忆 | 恢复后前10轮对话准确率>80% |
| 视频理解 | 1小时视频问答准确率>60% |

---

## 12. 术语表

| 术语 | 定义 |
|------|------|
| **Gargantua** | 本框架代号 |
| **Edith** | 前端超轻量模型，常驻端侧 |
| **Jarvis** | 后端大模型，对标Bel/K3 |
| **Base** | 预训练Transformer权重 |
| **Fast Weight** | 实时更新的短期记忆权重 (S_t) |
| **Slow Weight** | 固化后的长期记忆权重层 (W_mem) |
| **FWP** | Fast Weight Programmer (Schmidhuber 1991) |
| **Delta Rule** | 误差修正更新规则 |
| **SS** | State Space，视频流低秩缓存 |
| **Memory Layer** | 全局记忆层，Encoder-Decoder之间 |
| **Consolidation** | 固化：Fast Weight → Slow Weight |
| **信息觅食** | 模型自主搜索学习新信息 |
| **KDA** | Kimi Delta Attention |
| **AttnRes** | Attention Residuals |

---

## 13. 待定项

| # | 待定项 | 倾向 |
|---|--------|------|
| 1 | 开源协议 | GPL-3.0 |
| 2 | Fast Weight稀疏度 | 5% |
| 3 | 固化周期 | 128K tokens |
| 4 | 最大Slow层数 | 64 (Jarvis) / 8 (Edith) |
| 5 | SS Video rank | 64 (Jarvis) / 32 (Edith) |
| 6 | 谱归一化频率 | 每100 tokens |
| 7 | 跨session衰减 | Fast Weight ×0.5 |
| 8 | 信息觅食数据源 | 联网搜索 + 本地库 |

---

> **文档状态**: Phase 1 框架规范 v0.3 — 三层记忆 + 自进化架构  
> **下次会议**: 代码实现评审  
> **记录时间**: 2026-09-05
