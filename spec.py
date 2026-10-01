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
#   - 英文词表 40K（非常用词 fallback 到 a-z 字母拼写）。
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
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. 词表布局（连续区间，ID 不可变，只可在 RESERVE 段内消耗）
# ---------------------------------------------------------------------------
# 特殊 / 控制 token（0–63）
PAD_ID = 0                    # <pad>：填充（不参与 loss）
TS_PAD_NOTE = "无 in-stream 时间步分隔符；步边界由 step_ids 侧张量承载（无损）"
SPECIAL_TOKENS = [
    "<pad>",
    # 格式化（12）
    "<caps>", "<allcaps>",
    "<italic>", "<italic/>", "<bold>", "<bold/>",
    "<under>", "<under/>", "<mid>", "<mid/>", "<high>", "<high/>",
    # 边界（14）+ 停止 token（1）
    "<think>", "<think/>", "<answer>", "<answer/>",
    "<text>", "<text/>", "<visual>", "<visual/>",
    "<audio>", "<audio/>",
    "<model>", "<model/>", "<user>", "<user/>",
    "<stop>",                       # 生成停止标记：模型输出即停止
    # 预留（剩余槽位补 "<reserved:i>"）
]
SPECIAL_BEGIN = 0
SPECIAL_SIZE = 64             # 28 个已用（含 <stop>）+ 36 预留

# 生成停止 token：模型输出该 token 即停止生成（落在 SPECIAL 预留区，
# 不占用任何内容 token 区间；词表长度与模型输出维度均不变）
STOP_TOKEN = "<stop>"
STOP_ID = SPECIAL_TOKENS.index(STOP_TOKEN)   # 27

DIGIT_BEGIN = SPECIAL_BEGIN + SPECIAL_SIZE        # 64–73：字符 '0'..'9'
DIGIT_SIZE = 10

SYMBOL_BEGIN = DIGIT_BEGIN + DIGIT_SIZE           # 74–585
# ASCII 可打印字符 0x20–0x7E（95 个，含 a-z fallback 字母与半角符号）
# + 全角符号区 U+FF01–U+FF5E、U+3000–U+303F 常用全角标点
# 剩余槽位 reserved。注：a–z 既是 ASCII 符号也是英文 fallback token（同一 ID）。
SYMBOL_SIZE = 512

CJK_BEGIN = SYMBOL_BEGIN + SYMBOL_SIZE            # 586–28169
# 中文单字：Unicode 基本区 U+4E00–U+9FFF（20992）+ A 扩展 U+3400–U+4DBF（6592）
CJK_SIZE = 20992 + 6592                           # 27584

ENGLISH_BEGIN = CJK_BEGIN + CJK_SIZE              # 28170–68169
ENGLISH_SIZE = 40000          # 英文常用词 + 常见专有名词，全小写

# —— 绘图输出 token（仅输出侧；视觉输入的像素参数不入词表，走投影）——
DRAW_X_BEGIN = ENGLISH_BEGIN + ENGLISH_SIZE       # 68170–70089：<x-axis:n>
DRAW_X_SIZE = 1920
DRAW_Y_BEGIN = DRAW_X_BEGIN + DRAW_X_SIZE         # 70090–71169：<y-axis:n>
DRAW_Y_SIZE = 1080
DRAW_W_BEGIN = DRAW_Y_BEGIN + DRAW_Y_SIZE         # 71170–73089：<width:n>
DRAW_W_SIZE = 1920
DRAW_L_BEGIN = DRAW_W_BEGIN + DRAW_W_SIZE         # 73090–74169：<length:n>
DRAW_L_SIZE = 1080
DRAW_ROT_BEGIN = DRAW_L_BEGIN + DRAW_L_SIZE       # 74170–74529：<rot:n>（360 档）
DRAW_ROT_SIZE = 360
DRAW_R_BEGIN = DRAW_ROT_BEGIN + DRAW_ROT_SIZE     # 74530–75553：<Rn>（1024 档）
DRAW_R_SIZE = 1024
DRAW_G_BEGIN = DRAW_R_BEGIN + DRAW_R_SIZE         # 75554–76577：<Gn>
DRAW_G_SIZE = 1024
DRAW_B_BEGIN = DRAW_G_BEGIN + DRAW_G_SIZE         # 76578–77601：<Bn>
DRAW_B_SIZE = 1024
DRAW_SHAPE_BEGIN = DRAW_B_BEGIN + DRAW_B_SIZE     # 77602–77609
DRAW_SHAPES = ["background", "rectangle", "ellipse", "triangle"]  # +4 预留
DRAW_SHAPE_SIZE = 8

MIDI_BEGIN = DRAW_SHAPE_BEGIN + DRAW_SHAPE_SIZE   # 77610–78633
MIDI_SIZE = 1024
# MIDI token 子布局（相对 MIDI_BEGIN 的偏移）：
#   0–127     note_on  (pitch 0–127)
#   128–255   note_off (pitch 0–127)
#   256–287   velocity 档（32 档，幅值 0–127 线性量化）
#   288–415   program / 乐器（0–127）
#   416–447   tempo 档（32 档，40–240 BPM）
#   448–547   time_shift（100 档，10ms 步进，最长 1s；超过则重复多个）
#   548–1023  reserved

VOCAB_SIZE = MIDI_BEGIN + MIDI_SIZE               # 78634
# 为矩阵对齐补齐到 64 的倍数
VOCAB_SIZE_PADDED = (VOCAB_SIZE + 63) // 64 * 64  # 78656（尾部 reserved）

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
# block 内顺序（冻结）：KDA, MLA-HCA, KDA, MLA-CSA, KDA, FFN
BLOCK_ORDER = ["kda", "mla_hca", "kda", "mla_csa", "kda", "ffn"]

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

# Block Attention Residual（moonshot attnRes：块级残差混合，零初始化门控）
ATTNRES_ENABLED = True

# ---------------------------------------------------------------------------
# 3. 统一 KV 序列流（时间步为原子单位）
# ---------------------------------------------------------------------------
CTX_FULL = 1024              # 第一层：完整 KV 容量（token）
CTX_COMPRESS = 3072          # 第二层：被压缩的原始 token 容量
CTX_TOTAL = CTX_FULL + CTX_COMPRESS      # 4096
KV_BUFFER = 128              # 完整区与压缩区之间的缓冲区（吸收图元/音元
                             # 10:1 就地压缩带来的长度差）
CSA_RATIO = 4                # CSA：4:1 可学习压缩 + 局部滑窗 attention
CSA_LOCAL_WINDOW = 512       # CSA 局部 attention 窗口（原始 token 计）
HCA_RATIO = 32               # HCA：32:1 可学习压缩 + 全局 attention
# 压缩器：可学习 MLP（NSA 式），以时间步为原子单位；
# 一个时间步内部 token 多时允许步内压缩，绝不跨步合并。
# decoder 输出的图元（9 token/个）与音元（n 参数/个）写满一个后
# 在 KV 中就地压成 1 个向量（10:1 / n:1，CSA 打包摘要）。

# 滑动窗口：以整个时间步为原子溢出即弃；原始 token json 全量存档。

# ---------------------------------------------------------------------------
# 4. fast weight（全局型，KDA 结构，带遗忘门）
# ---------------------------------------------------------------------------
FW_N_HEAD = 4
FW_KEY_DIM = 128
FW_VALUE_DIM = 128
FW_CHUNK = 64
# 语义：enc/dec 各自在第一个 block 之后调用同一个全局 bank：
# 先用当前状态读出（当作权重层用），用后按 KDA 门控 delta 规则更新，
# 状态跨调用推进。enc 与 dec 共享同一个 bank 实例。

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

# 绘图输出（仅输出，token 区间见词表布局）
CANVAS_W = 1920
CANVAS_H = 1080
DRAW_TOKENS_PER_PRIMITIVE = 9   # x, y, shape, width, length, rot, R, G, B
DRAW_KV_COMPRESS = 9            # 输出完一个图元后 KV 就地 9:1 压成 1 向量

# 音频输出（仅输出）：MIDI → 离散 token（布局见 MIDI_* 区间注释）

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
# Python 3.12.14（standalone，x86_64）+ torch 2.2.2（macOS x86_64 最后官方轮子）
# venv：~/.workbuddy/binaries/python/envs/gargantua
DTYPE = "float32"            # 训练/推理默认 FP32（x86_64 mac 无 CUDA）

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
