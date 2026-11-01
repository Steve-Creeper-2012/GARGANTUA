# ===========================================================================
# GARGANTUA v1 — 冻结架构契约（FROZEN SPEC）
# ---------------------------------------------------------------------------
# 本文件是整个项目的唯一事实来源（single source of truth）。
# 所有模块（tokens / model / codecs / train / infer / server / web）
# 必须从这里 import 常数，禁止各自硬编码维度或词表区间。
#
# 定案记录（2026-09-27，steve 拍板）：
#   - 压缩允许在时间步【内】进行；禁止跨时间步边界压缩、禁止拆分时间步。
#   - 规模 ~500M（允许略超）；embedding 与 lm_head 不绑定（查表可放 CPU）。
#   - FFN 每 block 一个（GLU，d_ff=4096）。
#   - 英文词表 30K（非常用词 fallback 到 a-z 字母拼写）。
#   - 图元种类：rectangle / ellipse(含圆) / triangle / background。
#   - fast weight：KDA 结构带遗忘门，仅全局型；enc/dec 各自在第一个
#     block 之后调用一次（不是每个 block 都调）。
#   - KV：完整 1024 token + 压缩 3072 token = 4096 上下文；
#     CSA 4:1 可学习压缩 + NSA 式局部滑窗 attention；
#     HCA 32:1 可学习压缩 + 全局 attention。
#   - 优化器：Muon（隐藏层 2D 矩阵）+ AdamW（embedding / lm_head /
#     RMSNorm），Muon 可开关，关掉即全量 AdamW。
#   - 音频 16kHz 单声道，533 采样/token（~30 tok/s）；
#     A 值 = 透明度，1000 档（0.1% 步进）。
#   - 技术参考：MLA=DeepSeek-V2 低秩联合 KV 压缩（NoPE，无 RoPE 分支）；
#     KDA=Kimi Linear 门控 delta 递推；CSA/HCA 压缩器形态参考
#     DeepSeek NSA 的可学习 MLP 压缩；fast weight=Schmidhuber 1991。
#
# 定案记录（2026-10-24，steve 拍板，tok-32 词表重构）：
#   - ID 为 32-bit；.tokens 二进制 = 大端 uint32 流，4 字节/「字符」，
#     一字符一 token（tok-32 编码，解释器见 tokens/tok32.py）。
#     【已被 2026-10-28 的 token-5 取代，见下条】
#   - 词表分四大类：regular 常规可见 / fallback 可见（a–z）/
#     special 特殊（不可见）/ system 系统（不可见，{ } |）。
#   - 旧 28 个特殊 token 移除，只保留 <pad> <caps> <user> <model>
#     <stop>；其余特殊 token 由用户后续自行添加。
#   - 添加 token 功能支持四类，全局查重（同文本全表唯一）。
#   - 时间步：{ } 包裹、| 分隔；文本 1 token/步，图片 1 步，
#     音频 533 采样 = 1 token = 1 步，视频 1 帧 = 1 步。
#   - 音频/视觉输入token 不入词表，走投影；其向量存档格式为
#     .tokenvector（UTF-8，头部声明维度与精度）。
#   - 图元（draw）token 布局照搬 wonderfulearth/primitive-operation-painter
#     的 geometrize_256_v1（service/prims/ 内 vendored，转换器为 Rust/wgpu）；
#     MIDI 输出 token 暂时删除。
#
# 定案记录（2026-10-28，steve 拍板，token-5 编码；2026-10-31 修订绑定 token）：
#   - .tokens 改为 token-5：27 个单字符（0-9 { } < > | : w t v A-F x y）+ 4 个
#     原子绑定 token（<model> <user> <model/> <user/>，各占一个 5-bit 码位
#     27-30），5 bit/字符（2^5=32 码位，码 31 为终止符），位流落盘，末字符后
#     跟终止符码 31。4 个绑定 token 分别映射到扩展系统 token ID
#     （<model>=3、<user>=1、<model/>=4、<user/>=2，与词表契约头一致）。
#   - {} 内是正常 token（用户 token）：可见、有向量坐标、进 loss。
#     <id> 是扩展系统 token：有嵌入但没有位置坐标（位置嵌入的职责交给
#     它们，KDA 自学位置关系），不进 loss、不显示为内容。
#     { } | 是结构字符，本身不是 token。
#   - 文本 token 写作 {t:<十六进制 ID>}（自界定，长度因词表而异），
#     显示为 {text: C732}。
#   - 音频：533 采样 × 16 bit 顺序拼接 → 2132 个 hex 字符，{w:...}；
#     末尾不足 533 的残段直接丢弃。
#   - 视觉：patch 32×32，逐行、行内从左到右，每像素 RGBA 各 10 bit
#     （40 bit），共 40960 bit → 10240 个 hex 字符，写作
#     {v:<hex>x:<gx>y:<gy>}，同帧多个 patch 用 | 分隔。
#
# 定案记录（2026-10-30，steve 拍板，词表契约重构）：
#   - 新格式 .vfc（vocabulary for converting）：UTF-8、一行一个 token、
#     每行「内容:ID」，可显示用引号包裹、不可显示用 <> 包裹；编码 .tokens
#     时读取。解释器见 tokens/vfc.py。
#   - **词表长度不再由 spec 固定**：删除 VOCAB_SIZE / VOCAB_SIZE_PADDED /
#     REGULAR_END / FALLBACK_BEGIN 以及可见区各固定 SIZE 上限。具体长度
#     每个词表可以 vary，声明在该词表目录的 config.json 里。
#   - **词表必须占满**：ID 严格连续 0..N-1，不允许任何空位（空位也占
#     一个 embedding 行，白吃 RAM 且有副作用）。删除 token 走紧凑化
#     （后续 ID 整体前移），不再是软删占位。
#   - 架构契约头（special/system/modal-draw）的名字与顺序保持冻结，
#     但**去掉历史预留空位**：SPECIAL_SIZE 64→6、SYSTEM_SIZE 32→3、
#     MODAL_SIZE 12288→2961，故 HEAD_SIZE 12384→2970。
#     头对所有词表一致，所以按 ID 定位（STOP_ID / TS_*_ID / DRAW_BEGIN）
#     的语义不变——只是数值变了。
#   - 每个词表是一个文件夹：<name>/{vocab.vfc, config.json}。
#   - 每个模型是一个文件夹：<name>/{config.json, weights/, vocab/}
#     （weights 拆三个 safetensors：主权重 / 离散 embedding / 连续
#     投影；vocab 即使复用也整目录复制，防误删共享词表致模型报废）。
#
# 定案记录（2026-10-25，steve 拍板，内部流程对齐）：
#   - block 内顺序改为 KDA → KDA → KDA → MLA-CSA → MLA-HCA → FFN → AttnRes
#     （三个 KDA 连续，CSA 在前、HCA 在后）。
#   - AttnRes = Kimi/Moonshot《Attention Residuals》(arXiv 2603.15031) 的
#     Block AttnRes：每个子层前对「token embedding + 已完成的 block 表示 +
#     当前 partial sum」做 softmax 注意力聚合；块内仍用标准残差累加。
#     block 末尾再保留一个零初始化标量门（两者互补，steve 定）。
#   - 删除 CSA 局部窗选项：CSA 的保留区域就是完整 KV 区，两个视图都读
#     【压缩段 + 全部完整段】；处理完整未压缩 KV 时就是标准 MLA 逻辑。
#   - 压缩目标值 = 压缩比：每组凑够 ratio 个原始 token → 1 个压缩 token；
#     整步凑不满/会超出时取误差最小的整步组合（允许略超），绝不跨步拆分。
#   - fast weight 结构 = 一个 KDA 层（同参数化），区别只是状态跨调用持久。
#   - 标识 = 类型嵌入（type_emb）：user 段内所有 token 带 user 标识、model 段
#     带 model 标识，不分段首段尾；边界 token 本身不进 KV、不进 loss。
#   - KV 超出上下文上限：丢 KV，保留原始 token id 存档（不保留向量）。
#   - 训练数据 = .tokens（tok-32，必须带 { } | 时间步结构），不再用 JSON。
#   - Token 编辑器导出 .tokens；词表操作全部归「词表」页。
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. 词表布局
# ---------------------------------------------------------------------------
# 【2026-10-30 词表契约重构 —— 与旧版的根本差别】
#
# 旧版：spec 钉死 12 个固定大小的可见子区（合计 88690），再 pad 到 64 倍数
#   得 VOCAB_SIZE_PADDED = 88704。实际基础词表只有约 6.6 万条真实 token，
#   剩下全是 `<reserved:N>` 空位。空位照样占一个 embedding 行（wte 与
#   lm_head 各 1024 维），纯属白吃 RAM，且会让 logits 里出现一堆永远
#   不该被采样到的死行（副作用）。
#
# 新版：
#   1) **词表长度不由 spec 规定**。删掉 VOCAB_SIZE / VOCAB_SIZE_PADDED /
#      REGULAR_END / FALLBACK_BEGIN 与各可见区固定 SIZE。长度由每个词表
#      目录下的 config.json 的 "vocab_size" 声明——每个词表可以 vary。
#   2) **词表必须占满**：ID 严格连续 0..N-1，一个空位都不留。删除 token
#      走紧凑化（其后的 ID 整体前移），不再是软删占位 `<del:N>`。
#   3) spec 只冻结**架构契约头**：special / system / modal(draw) 三类
#      不可见 token 的名字、顺序与 ID。头对所有词表一模一样。
#
# 为什么可以压缩头：头里原先有大量"预留"空位（SPECIAL 58 个、SYSTEM
# 29 个、MODAL 9327 个），它们没有对应任何 token，纯粹是历史占位。
# 压缩只改变**数值**，不改变**语义**——spec.TS_OPEN_ID 依然是"`{` 的
# ID"。由于头对所有词表相同，按 ID 定位的 dectok / gts / draw_codec
# 逻辑全部原样成立，一个函数签名都不用动（这一条是动工前专门核过的）。
#
# 大类划分（enctok 只查 regular/fallback 两类；special/system 不可显示，
# 不参与文本编码）：
#   regular  可显示常规：数字 / 空白 / 符号 / emoji / 中文 / 英文词
#   fallback 可显示兜底：a–z 26 个字母（常规未收录时逐字母兜底）
#   special  不可显示特殊：<caps> <user> <model> <stop> 等
#   system   不可显示系统：{ } | 时间步边界与步内分隔（架构契约头内）
# 同一字符串在全词表内唯一，token_to_id 反查无歧义。
#
# 在 .vfc 文件里，这两类形态就是：可显示 = 引号包裹，不可显示 = <> 包裹
# （见 tokens/vfc.py）。

# ---------------------------------------------------------------------------
# 1a. 图元输出 token 布局（先定义：MODAL_SIZE 依赖 DRAW_VOCAB_SIZE）
# ---------------------------------------------------------------------------
# 照搬 wonderfulearth/primitive-operation-painter geometrize_256_v1。
# 仅输出侧——视觉输入的像素参数不入词表，走线性投影。
#
# 每个图元 = 9 个 token，字段序（cx, cy, theta, w, h, shape, r, g, b）；
# 各字段是独立 bin 区间（token ID = DRAW_BEGIN + 字段 offset + bin）：
#   cx/cy  [0,256) px   ×2/bin   → 512 bins
#   theta  [0,90) 度    ×3/bin   → 270 bins（CSV 里是弧度，≥90° 交换宽高）
#   w/h    [0,128) px   ×4/bin   → 512 bins
#   shape  -1=背景 0=矩形 1=椭圆（bin = shape_type+1，256 bins）
#   r/g/b  0-255 //2（各 128 bins）
#   special 3 bins（仓库布局自带：pad + 2 预留）
DRAW_TOKENS_PER_PRIMITIVE = 9               # cx, cy, theta, w, h, shape, r, g, b
DRAW_FIELD_ORDER = ("cx", "cy", "theta", "w", "h", "shape", "r", "g", "b")
DRAW_CSV_COLUMNS = ("image_name", "cx", "cy", "w", "h", "shape_type",
                    "theta", "r", "g", "b")
DRAW_SHAPES = ("background", "rectangle", "ellipse")   # shape_type -1/0/1
XY_BINS_PER_PIXEL = 2
ANGLE_BINS_PER_DEGREE = 3
SIZE_BINS_PER_PIXEL = 4
DRAW_X_BINS = 512
DRAW_Y_BINS = 512
DRAW_ANGLE_BINS = 270
DRAW_W_BINS = 512
DRAW_H_BINS = 512
DRAW_SHAPE_BINS = 256
DRAW_COLOR_BINS = 128
DRAW_SPECIAL_BINS = 3
# 字段 offset（相对 DRAW_BEGIN）——由上面的 bin 数累加而来，与 base 无关。
# 所以 DRAW_BEGIN 变成运行时值也不影响这段算术（draw_codec 直接用
# `b + offset + bin`，b 一变自动跟随）。
DRAW_X_OFFSET = 0
DRAW_Y_OFFSET = DRAW_X_OFFSET + DRAW_X_BINS            # 512
DRAW_ANGLE_OFFSET = DRAW_Y_OFFSET + DRAW_Y_BINS        # 1024
DRAW_W_OFFSET = DRAW_ANGLE_OFFSET + DRAW_ANGLE_BINS    # 1294
DRAW_H_OFFSET = DRAW_W_OFFSET + DRAW_W_BINS             # 1806
DRAW_SHAPE_OFFSET = DRAW_H_OFFSET + DRAW_H_BINS         # 2318
DRAW_R_OFFSET = DRAW_SHAPE_OFFSET + DRAW_SHAPE_BINS    # 2574
DRAW_G_OFFSET = DRAW_R_OFFSET + DRAW_COLOR_BINS        # 2702
DRAW_B_OFFSET = DRAW_G_OFFSET + DRAW_COLOR_BINS        # 2830
DRAW_SPECIAL_OFFSET = DRAW_B_OFFSET + DRAW_COLOR_BINS  # 2958
DRAW_VOCAB_SIZE = DRAW_SPECIAL_OFFSET + DRAW_SPECIAL_BINS   # 2961

# ---------------------------------------------------------------------------
# 1b. 架构契约头（冻结：名字 + 顺序 + ID；所有词表共享，不得占用/重排）
# ---------------------------------------------------------------------------
# —— special 特殊（不可显示）——
SPECIAL_TOKENS = [
    "<caps>",                 # 大写标记：其后单词/字母首字母大写
    "<user>", "<user/>",      # 用户输入段边界（TYPE_USER）
    "<model>", "<model/>",    # 模型输出段边界（TYPE_MODEL）
    "<stop>",                 # 生成停止标记：模型输出即停止
]
SPECIAL_BEGIN = 0x00000000
# 旧值 64（6 已用 + 58 预留）→ 现在不留预留：按实际收录数排布
SPECIAL_SIZE = len(SPECIAL_TOKENS)           # 6

# 生成停止 token：模型输出该 token 即停止生成
STOP_TOKEN = "<stop>"
STOP_ID = SPECIAL_TOKENS.index(STOP_TOKEN)   # 5（与旧版一致）

# —— system 系统（不可显示，时间步结构）——
# 显示即字面量：{ = 时间步开始，} = 时间步结束，| = 步内 token 分隔。
TS_OPEN = "{"
TS_CLOSE = "}"
TS_SEP = "|"
SYSTEM_CHARS = [TS_OPEN, TS_CLOSE, TS_SEP]
SYSTEM_BEGIN = SPECIAL_BEGIN + SPECIAL_SIZE   # 6（旧：64）
# 旧值 32（3 已用 + 29 预留）→ 按实际收录数
SYSTEM_SIZE = len(SYSTEM_CHARS)               # 3
TS_OPEN_ID = SYSTEM_BEGIN + 0                  # 6（旧：64）
TS_CLOSE_ID = SYSTEM_BEGIN + 1                 # 7（旧：65）
TS_SEP_ID = SYSTEM_BEGIN + 2                   # 8（旧：66）

# —— modal 模态输出（图元 DRAW；token 布局见 1a）——
MODAL_BEGIN = SYSTEM_BEGIN + SYSTEM_SIZE      # 9（旧：96）
# 旧值 12288（DRAW 2961 + 9327 预留）→ 只留实际用到的
MODAL_SIZE = DRAW_VOCAB_SIZE                   # 2961
DRAW_BEGIN = MODAL_BEGIN                       # 9（旧：96）

# 架构契约头长度 = special + system + modal。**所有词表共享、固定**。
# 可见区从这里开始；头内不允许任何空位（空位同样吃 embedding 行）。
HEAD_SIZE = SPECIAL_SIZE + SYSTEM_SIZE + MODAL_SIZE     # 2970（旧：12384）

# 词表长度防呆上界。**这不是 spec 规定的长度**——词表多长由各词表的
# config.json 说了算，token ID 是自界定十六进制（见第 5b 节），格式本身
# 对 ID 宽度没有任何限制。这个数只防 RAM 炸弹：每个 ID 占 wte + lm_head
# 两行 1024 维 fp32 ≈ 8KB，10 万 ID ≈ 800MB，再往上就该严肃对待了。
MAX_VOCAB_SIZE = 100_000

# 目录/文件命名（目录树契约，见第 9 节）
VOCAB_DIR_NAME = "vocab"          # 模型目录内的词表子目录
VOCAB_FILE_NAME = "vocab.vfc"      # 词表文件名（.vfc 格式）
WEIGHTS_DIR_NAME = "weights"       # 模型目录内的权重子目录
VOCAB_CONFIG_NAME = "config.json"   # 词表配置（含 vocab_size）

# ---------------------------------------------------------------------------
# 1c. 可见区：子区顺序与字符来源（**长度不固定**，由词表 config.json 决定）
# ---------------------------------------------------------------------------
# 可见区从 HEAD_SIZE 起，各子区按 VISIBLE_REGION_ORDER 的顺序**连续**
# 排布；每个子区的长度 = 它实际收录的 token 数，不带任何预留。
# 因此**不同词表的分区表是不同的**——这正是前端词表页要渲染的"分区"。
VISIBLE_REGION_ORDER = ("digit", "whitespace", "symbol", "emoji",
                        "cjk", "english", "extra", "fallback")

# 各类字符来源（build_vocab.py 生成基础词表时取用；**不是长度上限**）
WHITESPACE_CHARS = [" ", "\t", "\n", "\r", "\u00a0", "\u3000"]
FALLBACK_LETTERS = "abcdefghijklmnopqrstuvwxyz"
# 子区 → 大类的归属（category_of 用）
REGION_CATEGORY = {
    "digit": "regular", "whitespace": "regular", "symbol": "regular",
    "emoji": "regular", "cjk": "regular", "english": "regular",
    "extra": "regular", "fallback": "fallback",
}

# 2026-10-24：MIDI 输出 token 暂时删除（steve：研究后再定），midi_codec
# 与 /api/codec/midi 一并移除；如恢复从 git 历史取回。


# 2026-10-24：MIDI 输出 token 暂时删除（steve：研究后再定），midi_codec
# 与 /api/codec/midi 一并移除；如恢复从 git 历史取回。

# 边界 token → 类型嵌入 id（「有 token 无 embedding 投影」的边界 token
# 本身不查词表投影，但其界定的区间内容会加类型嵌入）
TYPE_USER = 0          # <user>..<user/> 之间：用户输入
TYPE_MODEL = 1         # <model>..<model/> 之间：模型输出
TYPE_NONE = 2          # 边界之外 / 边界 token 自身
NUM_TYPES = 3

# ---------------------------------------------------------------------------
# 2. 模型维度
# ---------------------------------------------------------------------------
D_MODEL = 1024
HEAD_DIM = 128
N_HEADS = D_MODEL // HEAD_DIM                      # 8

N_ENC_BLOCKS = 4
N_DEC_BLOCKS = 6
# block 内顺序（冻结，2026-10-25）：KDA×3 → MLA-CSA → MLA-HCA → FFN → AttnRes
BLOCK_ORDER = ["kda", "kda", "kda", "mla_csa", "mla_hca", "ffn", "attnres"]

# KDA（Kimi Linear 式：channel-wise 门控 delta 递推）
KDA_HEAD_DIM = 128
KDA_EXPAND_V = 1.0
KDA_CONV_KERNEL = 4
KDA_CHUNK_SIZE = 64
KDA_NORM_EPS = 1e-6

# MLA（DeepSeek-V2 式低秩联合压缩，NoPE：无 RoPE / 无 decoupled 分支）
MLA_Q_LORA_RANK = 256
MLA_KV_LORA_RANK = 512
MLA_V_LORA_RANK = 512

# FFN（每 block 一个，GLU 结构：SiLU 门控）
FFN_HIDDEN = 4096

# Block Attention Residual（Moonshot AttnRes, arXiv 2603.15031）
# 每个子层（KDA / MLA / FFN）之前做一次 block_attn_res：
#   V = [已完成的 block 表示..., 当前 block 的 partial sum]
#   h = softmax(norm(V)·w)ᵀ V
# block 末尾另有一个零初始化标量门把块输入混回输出（两种残差互补）。
ATTNRES_ENABLED = True

# ---------------------------------------------------------------------------
# 3. 统一 KV 序列流（时间步为原子单位）
# ---------------------------------------------------------------------------
CTX_FULL = 1024              # 第一层：完整 KV 容量（token）
CTX_COMPRESS = 3072          # 第二层：被压缩的原始 token 容量
CTX_TOTAL = CTX_FULL + CTX_COMPRESS      # 4096
KV_BUFFER = 128              # 完整区与压缩区之间的缓冲区（吸收图元/音元
                             # 10:1 就地压缩带来的长度差）
CSA_RATIO = 4                # CSA：4:1 可学习压缩（检索视图）
HCA_RATIO = 32               # HCA：32:1 可学习压缩（全局概括视图）
# KV 分两段（2026-10-25 冻结）：
#   第一段 完整 KV：不压缩，原样保留；
#   第二段 压缩 KV：同一段内容的两份不同压法——
#     CSA 检索视图（MLA-CSA 处理）与 HCA 全局概括视图（MLA-HCA 处理）。
#   两个视图都读【压缩段 + 全部完整段】：处理完整未压缩 KV 时就是标准
#   MLA 逻辑，没有额外的局部窗限制（旧 csa_local_window 已删除）。
# 压缩器：可学习 MLP（NSA 式），以时间步为原子单位。
#   目标值 = 压缩比：一个压缩组凑够 ratio 个原始 token → 1 个压缩 token；
#   整步凑不齐时取误差 |组内 token 数 − ratio| 最小的整步组合（可略超），
#   绝不把同一步拆到不同组；单步 token 数 ≥ ratio 时允许步内切分成多个
#   压缩 token（它们仍属同一个时间步）。
# decoder 输出的图元（9 token/个）与音元（n 参数/个）写满一个后
# 在 KV 中就地压成 1 个向量（10:1 / n:1，CSA 打包摘要）。

# 滑动窗口：以整个时间步为原子溢出即弃；被丢弃条目的原始 token id
# 全量存档（SharedKV.archive），只丢 KV 向量不丢 id。

# ---------------------------------------------------------------------------
# 4. fast weight（全局型，与一个 KDA 层完全同构，状态持久）
# ---------------------------------------------------------------------------
# 2026-10-25：不再是简化的 4 头 bank——直接复用 KDAAttention 的参数化
# （q/k/v 投影 + 因果短卷积 + L2 归一化 + channel-wise 遗忘门 + 输出门控
# + RMSNorm），唯一区别是状态 S 作为成员变量跨调用推进（实时写入 + 遗忘）。
FW_N_HEAD = N_HEADS              # 与 KDA 层同构（8 头）
FW_KEY_DIM = KDA_HEAD_DIM        # 128
FW_VALUE_DIM = KDA_HEAD_DIM      # 128
FW_CHUNK = KDA_CHUNK_SIZE
# 语义：enc/dec 各自在第一个 block 之后调用同一个全局 bank：
# 先用当前状态读出（当作权重层用），用后按 KDA 门控 delta 规则更新，
# 状态跨调用推进。enc 与 dec 共享同一个 bank 实例。当前为无差别写入
# （序列里所有 token 都写进状态，不做选择性筛选）。

# ---------------------------------------------------------------------------
# 5. 模态参数
# ---------------------------------------------------------------------------
# 视觉输入（仅输入，像素参数不入词表）
PATCH_SIZE = 32              # 32×32 像素 / patch
PATCH_PIXELS = PATCH_SIZE * PATCH_SIZE        # 1024
PATCH_VEC_DIM = PATCH_PIXELS * 4              # RGBA 平铺 → 4096
VIDEO_FPS = 30
# 帧 delta 规则：首帧全量；2..30 帧只送有变化 patch；每秒首帧全量；
# 某帧全部 patch 都变 → 全量并从该帧重新计数；最后一帧全量。
# 图片 = 单帧视频。一帧内全部 patch + 该帧音频 token 处于同一时间步。
# 输入 embedding：patch 平铺向量过线性投影 → d_model；
# (x, y) patch 坐标不 embed，以加性 2D 位置嵌入注入该 patch 的向量。
VISUAL_R_QUANT = 1024        # R/G/B 各 1024 档（仅 token 存档/输出词表用）
VISUAL_A_QUANT = 1000        # A 透明度 1000 档（0.1% 步进）

# 音频输入（仅输入）
AUDIO_SR = 16000             # 16kHz 单声道
AUDIO_SAMPLES_PER_TOKEN = 533        # 533 采样/token ≈ 30 tok/s
# 输入 embedding：533 个采样点平铺成一条向量 → 线性投影 → d_model

# ---------------------------------------------------------------------------
# 5b. token-5 编码（2026-10-28 定案，取代 tok-32 的 .tokens 二进制）
# ---------------------------------------------------------------------------
# 动机：tok-32 的 .tokens 只能装「离散 ID」，音频/视觉必须另存 .tokenvector，
# 训练与存档是两套文件。token-5 把三种模态压进同一个字符流：
#   {t:<id>}               文本 token（可见、有向量、进 loss）
#   {w:<hex>}              音频 token（533 采样）
#   {v:<hex>x:<gx>y:<gy>}  视觉 patch token（32×32 像素）
#   <id>                   扩展系统 token（有嵌入、不进 loss、不可显示）
#   { } |                  结构字符（不是独立 token，不占位置、不进 loss）
#
# —— 字符集：27 个单字符（码 0-26）+ 4 个原子绑定 token（码 27-30）——
#   5 bit/字符（2^5 = 32 码位）。码 31 为流终止符，不对应任何字符。
#
#   —— 27 个单字符（T5_CHARS，码即下表顺序 0..26）——
#   '0'-'9'  数字（坐标十进制 / ID 十六进制数字位；ID 另用 'A'-'F'）
#   '{' '}'  时间步边界
#   '<' '>'  扩展系统 token 边界
#   '|'      步内 token 分隔 / patch 分隔
#   ':'      字段分隔
#   'w'      音频载荷标识（wave）
#   't'      文本载荷标识（text）
#   'v'      视觉载荷标识（visual）
#   'A'-'F'  十六进制字母（大写；小写不属于字符集）
#   'x' 'y'  视觉坐标标识
#   没有空格，没有其它字母（w/t/v/x/y 是仅有的小写，只作载荷/坐标标识）。
#
#   —— 4 个原子绑定 token（码 27-30，各占一个 5-bit 码位，不是多字符）——
#   编码时把字符流里的整串 <model>/<user>/<model/>/<user/> 当作一个码位；
#   解码时码 27/28/29/30 还原成各自的多字符展示名（见 T5_BOUND_TOKENS）。
#   它们分别绑定到扩展系统 token 的 ID：<model>=3、<user>=1、<model/>=4、
#   <user/>=2（与词表契约头里的名字一致），所以既能被位流当作原子字符存，
#   又能在元素层当作普通 ext 元素（带 ID）参与编解码。
T5_CHARS = "0123456789{}<>|:wtvABCDEFxy"
T5_ALPHABET_SIZE = 27                     # 单字符码 0..26（绑定 token 不计入）
T5_BITS_PER_CHAR = 5
T5_END_CODE = 31                          # 终止符：码 31（2^5-1），解码到此为止
T5_CHAR_CODE = {c: i for i, c in enumerate(T5_CHARS)}
# 原子绑定 token：码位 → 多字符展示名（编码/解码的原子单元）
T5_BOUND_TOKENS = {27: "<model>", 28: "<user>", 29: "<model/>", 30: "<user/>"}
# 绑定 token 名字 → 码位（编码时整串匹配）
T5_BOUND_NAME_TO_CODE = {"<model>": 27, "<user>": 28, "<model/>": 29, "<user/>": 30}
# 绑定 token 名字 → 扩展系统 token ID（解析为元素用）
T5_NAME_TO_BOUND_ID = {"<model>": 3, "<user>": 1, "<model/>": 4, "<user/>": 2}
# 扩展系统 token ID → 绑定 token 名字（元素→字符流时写出名字而非 <id>）
T5_BOUND_ID_TO_NAME = {3: "<model>", 1: "<user>", 4: "<model/>", 2: "<user/>"}
T5_HEX_CHARS = "0123456789ABCDEF"

# —— 载荷尺寸 ——
# 文本/扩展 token 的 ID：**大写十六进制、自界定**——{t:...} 里读到 '}' 或
# '|' 为止，<...> 里读到 '>' 为止。长度因词表而异（小词表 ID 短、大词表
# ID 长），格式本身不给 ID 定宽、定上限；前导零允许（<00005> 就是 5）。
# 十六进制必须大写：5 bit 字符表只有 31 个码位，装不下小写 a-f。
# 坐标（{v:...x:N y:M} 的 N/M）仍是自界定十进制。
T5_COORD_LIMIT = 100_000        # x/y 坐标防呆上界（自界定十进制）

# 音频：每 token 恰好 533 个采样点，16 bit 有符号 PCM（mp3/flac 解码后的
# 通用形态），按时间顺序拼接 → 533*16 = 8528 bit = 2132 个十六进制字符。
AUDIO_SAMPLE_BITS = 16
T5_WAVE_BITS = AUDIO_SAMPLES_PER_TOKEN * AUDIO_SAMPLE_BITS   # 8528
T5_WAVE_HEX = T5_WAVE_BITS // 4                              # 2132

# 视觉：一个 patch 32×32 像素，逐行（第 1 行→第 32 行）、行内从左到右，
# 每个像素 R/G/B/A 各 10 bit 共 40 bit，按 R→G→B→A 拼接。
# 1024 * 40 = 40960 bit = 10240 个十六进制字符。
T5_PIXEL_BITS = 4 * 10                    # RGBA 各 10 bit
T5_PATCH_BITS = PATCH_PIXELS * T5_PIXEL_BITS    # 40960
T5_PATCH_HEX = T5_PATCH_BITS // 4               # 10240
T5_CHANNEL_MAX = (1 << 10) - 1            # 1023

# —— 语义 ——
# {t:} {w:} {v:} 元素：可见 token，有向量坐标（走 embedding / 投影），进 loss。
# <id> 元素：扩展系统 token，有嵌入但【没有】位置坐标——KDA 让模型自己学
#   位置关系，标准 Transformer 位置嵌入的职责就交给这类 token；它们不进
#   loss，也不作为可见内容显示。
# { } | ：结构字符，本身不是 token（不占 token 位置、不进 loss、无嵌入）。
# 时间步仍是原子单位：{} 必须至少含一个元素，空步非法。
# 落盘（.tokens）：5 bit/字符的位流，末字符后跟终止符码 27，再补零到字节。

# 绘图输出（仅输出，token 区间见词表布局；画布照搬仓库 256×256）
CANVAS_W = 256
CANVAS_H = 256
DRAW_KV_COMPRESS = 9            # 输出完一个图元后 KV 就地 9:1 压成 1 向量

# 音频输出：MIDI → 离散 token 暂时删除（2026-10-24，steve：研究后再定）

# ---------------------------------------------------------------------------
# 6. 优化器
# ---------------------------------------------------------------------------
OPTIMIZER = "muon"           # "muon" | "adamw"；muon 只作用隐藏层 2D 矩阵，
                             # embedding / lm_head / RMSNorm / 标量恒为 AdamW
MUON_MOMENTUM = 0.95
MUON_NS_STEPS = 5
ADAMW_LR = 3e-4
MUON_LR = 2e-2

# ---------------------------------------------------------------------------
# 7. 运行环境
# ---------------------------------------------------------------------------
# Python 3.12 系列最新版（standalone，项目内隔离安装；steve 2026-10-24：
# 用 latest 不锁 patch）+ torch 2.2.2（macOS x86_64 最后官方轮子）。
# 隔离运行时全部装在 environment/，不碰系统 Python：
#   environment/python/  python-build-standalone 解包（setup.sh 自动下载）
#   environment/venv/    以上述解释器建的虚拟环境（core/window/dev 依赖）
#   environment/rust/    Rust 工具链（service/prims/setup.sh 隔离安装）
# torch 2.2.2 最高支持 Python 3.12（3.13 无预编译轮子）——所以锁 3.12
# 系列而不是 latest overall。
DTYPE = "float32"            # 训练/推理默认 FP32（x86_64 mac 无 CUDA）

# ---------------------------------------------------------------------------
# 9. 项目目录树与服务架构（2026-10-24 重构，steve 定的结构）
# ---------------------------------------------------------------------------
# 目录树（契约，新增文件必须归位）：
#   start/
#     terminal/    终端脚本：setup.{sh,bat}（装隔离环境）+ start.{sh,bat,command}
#     requirements/ 分类依赖：core（必装）/ window（GUI）/ dev（测试）
#     boot.py      路径引导（把 server/ 与 service/* 放上 sys.path）
#   environment/   全部依赖/工具链（与系统隔离，见第 7 节）：
#     python/       standalone CPython（setup 脚本下载）
#     venv/         以上述解释器建的虚拟环境（core/window/dev 依赖）
#     rust/         Rust 工具链（rustup/cargo，service/prims/setup.sh 装）
#   server/        全部 server 文件：spec / common / model / muon +
#                  server.py（装配+路由+启动编排）+ server_env（环境
#                  预检）+ server_services（服务监督器）+
#                  server_models / media / codecs / train / tokens / util
#   service/       每个服务一个目录：infer / train / init / tokens / prims /
#                  web / window / test（按需启动，见下）
#     prims/       图元管线：image_to_primitives.py（驱动+CPU 兜底）+
#                  fast_shape_render/（逐字照搬的第三方 Rust/wgpu 转换器，
#                  未改动；setup.sh 用 environment/rust 隔离构建）+
#                  upstream/（上游仓库完整留档）
#   models/        models/NAME/{config.json, weights/, vocab/}
#                  （config.json 是 manifest：它的 "weights"/
#                    "weights_discrete_embed"/"weights_continuous_embed" 指向
#                    weights/ 下的三个 safetensors，读者加载哪个由它说了算；
#                    vocab/ 是建模时整目录复制的词表副本 vocab.vfc + config.json）
#   data/          data/queue/（待训练）+ data/done/（已训练归档）
#   logs/          auto_train.jsonl 等运行日志
#
# 服务按需启动（省内存的关键设计）：
#   infer / train 是重活（模型权重单个 safetensors 可达数 GB），以**独立
#   子进程**按需拉起，协议 = JSON-lines over stdin/stdout（请求
#   {"id","op",...} → 响应 {"id","ok",result|error}；服务的 stdout 只准
#   写 JSON 行，人类日志走 stderr）。server 本体只路由 + 记账，不持权重。
#   退出条件：shutdown / stdin EOF（server 死了）/ 空闲超时
#   （--idle-timeout，默认 300s）。init 是罕见一次性重活，走
#   run_oneshot（--json 末行协议），跑完即退。
#   PyTorch CPU 分配器在进程内不向 OS 归还内存——所以「按需」必须是
#   进程级的，进程退出才是真省内存。
#
# 每模型词表（2026-10-30 契约）：
#   models/NAME/vocab/ 是建模时从选定词表（vocabs/<name>/）整目录复制的
#   副本（vocab.vfc + config.json）——即使多个模型复用同一词表也各自持有
#   一份，用户误删共享词表不让模型报废。正文格式见 tokens/vfc.py：
#   UTF-8、一行一个 token、`内容:ID`，ID 是十六进制（大小写不限，写出
#   统一大写），恰好连续 0..N-1；前 HEAD_SIZE 个 ID 是 special/system/draw
#   架构契约（<stop>/<caps>/{ } |/图元输出 token 按 ID 定位，decode 的
#   停止/大小写逻辑靠它们），不可删改。词表长度（N）与分区表住同目录的
#   config.json，每个词表可以不同。
#   切换活动模型时，server / infer / train 进程各自 set_model_vocab()。
#
# 启动链（唯一入口）：
#   launcher/start-server.sh / launcher/start-server.command（由根目录 启动.sh / 启动.command 调用）/ start/terminal/start.bat（Windows）
#     → launcher/setup-environment.sh / setup.bat（幂等：装/复用 environment/python + environment/venv；
#        下载镜像与 pip 源均自动测速选取）
#     → server/server.py
#         → server_env.check_environment（解释器版本 + 主依赖）
#         → HTTP 服务 + 按需拉起 service/window/window.py 与各服务

# ---------------------------------------------------------------------------
# 8. 输入流约定（所有模块遵守）
# ---------------------------------------------------------------------------
# tokenizer/codec 输出 = List[Step]：
#   Step = {
#       "token_ids": List[int],        # 本时间步内的 token（文本通常 1 个；
#                                      #   视觉帧 = 全部 patch + 音频 token）
#       "type": TYPE_USER/TYPE_MODEL/TYPE_NONE,
#       "modality": "text"|"visual"|"audio"|"mixed",
#       "meta": dict,                  # 视觉：patch 坐标与原始 RGBA 平铺向量
#                                      # 音频：原始 533 采样（供投影，不经词表）
#   }
# 模型前向输入 = 扁平 token_ids + 并行 step_ids + type_ids；
# decoder 因果 = 步间因果 + 步内双向（无步内掩码）。
#
# 时间步结构（token-5 体系，2026-10-28 起，tokens/tok5.py）：
#   流中的每个时间步由字符 { } 包裹，步内多个元素由 | 分隔；元素形态
#   见第 5b 节（t: 文本 / w: 音频 / v: 视觉）。文本每 token 独占一步；
#   图片 = 一步（内部多 patch 元素）；音频每 533 采样 = 一个元素 = 一步；
#   视频每帧 = 一步。KDA 循环状态递归 / 逻辑并行以时间步为原子。
#   结构字符 { } | 本身不是 token，不占 token 位置、不进 loss、无嵌入。
#
# 训练数据（2026-10-28 起）：.tokens 文件（token-5 位流，5 bit/字符）。
#   内容必须带 { } | 时间步结构；训练侧读取后按 Step 展开成「扁平
#   token_ids + 并行 step_ids」。旧 tok-32 大端 uint32 流不再生成，
#   tok32 模块仅作为历史数据的读取/转换工具保留。
#   Token 编辑器导出同一格式，训练与编辑器共用一套流。
#
# 标识（2026-10-25）：段类型用类型嵌入表达——user 段内所有 token 都加
#   type_emb(TYPE_USER)，model 段都加 type_emb(TYPE_MODEL)，不分段首
#   段尾。<user>/<user/>/<model>/<model/> 这类边界 token 是扩展系统级
#   标记（token-5 写作 <1> / <2> / <3> / <4>），不进入共享 KV、不作为
#   普通 token 参与 loss。
