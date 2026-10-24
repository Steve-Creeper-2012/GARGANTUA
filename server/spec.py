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
#     的 geometrize_256_v1（third_party/ 内 vendored，转换器为 Rust/wgpu）；
#     MIDI 输出 token 暂时删除。
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. 词表布局（32-bit ID，四大类，只可在各类 RESERVE 段内消耗）
# ---------------------------------------------------------------------------
# 大类划分（enctok 只查 regular/fallback 两类；special/system 不可见，
# 不参与文本编码）：
#   regular  常规可见：数字 / 空白 / 符号 / emoji / 中文 / 英文词
#   fallback 可见：a–z 26 个字母（常规未收录时逐字母兜底）
#   special  特殊（不可见）：<caps> <user> <model> <stop> 等
#   （<pad> 已删除：GARGANTUA 无 padding 批次；图元侧的 pad bin 由仓库
#    布局自带的 special bin 承担）
#   system   系统（不可见）：{ } | 时间步边界与步内分隔
# 同一字符串在全词表内唯一（单字母词 a/i 不收录，由 fallback 承担），
# token_to_id 反查无歧义。
SPECIAL_TOKENS = [
    "<caps>",                 # 大写标记：其后单词/字母首字母大写
    "<user>", "<user/>",      # 用户输入段边界（TYPE_USER）
    "<model>", "<model/>",    # 模型输出段边界（TYPE_MODEL）
    "<stop>",                 # 生成停止标记：模型输出即停止
]
SPECIAL_BEGIN = 0x00000000
SPECIAL_SIZE = 64             # 6 个已用 + 58 预留

# 生成停止 token：模型输出该 token 即停止生成（落在 SPECIAL 区，
# 不占用任何内容 token 区间；词表长度与模型输出维度均不变）
STOP_TOKEN = "<stop>"
STOP_ID = SPECIAL_TOKENS.index(STOP_TOKEN)   # 5

# —— 系统 token（不可见，system 类：时间步结构）——
# 显示即字面量：{ = 时间步开始，} = 时间步结束，| = 步内 token 分隔。
TS_OPEN = "{"
TS_CLOSE = "}"
TS_SEP = "|"
SYSTEM_BEGIN = SPECIAL_BEGIN + SPECIAL_SIZE   # 64
SYSTEM_SIZE = 32               # 3 个已用 + 29 预留
TS_OPEN_ID = SYSTEM_BEGIN + 0
TS_CLOSE_ID = SYSTEM_BEGIN + 1
TS_SEP_ID = SYSTEM_BEGIN + 2

# —— 模态输出预留（图元 DRAW；token 布局照搬 primitive-operation-painter
#    geometrize_256_v1）——
# 这个区间的 token 是模型输出侧离散 token，归类为 special（不可见），
# 但不通过「添加 token」功能开放（属预留演进位）。
MODAL_BEGIN = SYSTEM_BEGIN + SYSTEM_SIZE      # 96
MODAL_SIZE = 12288             # DRAW 2961 + 9327 预留

DIGIT_BEGIN = MODAL_BEGIN + MODAL_SIZE        # 12384–12393：字符 '0'..'9'
DIGIT_SIZE = 10

WHITESPACE_BEGIN = DIGIT_BEGIN + DIGIT_SIZE   # 12394–12401
WHITESPACE_CHARS = [" ", "\t", "\n", "\r", "\u00a0", "\u3000"]
WHITESPACE_SIZE = 8

SYMBOL_BEGIN = WHITESPACE_BEGIN + WHITESPACE_SIZE   # 12402
# 全部 Unicode P*/S* 类码点（半角 + 全角 + 各平面符号，排除 emoji 区），
# 另含全角字母数字（U+FF10–FF19 / U+FF21–FF3A / U+FF41–FF5A 共 62 个，
# 保证全角文本无损 roundtrip）。
SYMBOL_SIZE = 7680

EMOJI_BEGIN = SYMBOL_BEGIN + SYMBOL_SIZE      # 20082
# Unicode Emoji=Yes 属性码点（排除 ASCII 键帽基字符 # * 0–9——它们已在
# SYMBOL/DIGIT 区）；ZWJ / VS16 / 肤色修饰等组件字符一并收录，ZWJ
# 序列由多个 emoji token 组成。
EMOJI_SIZE = 2048

CJK_BEGIN = EMOJI_BEGIN + EMOJI_SIZE          # 22130
# 中文单字：基本区 U+4E00–U+9FFF（20992）+ 扩展A U+3400–U+4DBF（6592）。
# 扩展B 及以后暂不收录（未映射字符由 enctok 跳过并记录，可后续添加）。
CJK_SIZE = 28672

ENGLISH_BEGIN = CJK_BEGIN + CJK_SIZE          # 50802
# 常用英文词 30000 个，全小写 [a-z]+（允许一个撇号后缀，如 don't）。
# 单字母词（a / i）不收录——由 fallback 字母区承担，杜绝同文本双 ID。
# 词源：hermitdave/FrequencyWords（经上一版词表按词频序提取）。
ENGLISH_SIZE = 32768

EXTRA_BEGIN = ENGLISH_BEGIN + ENGLISH_SIZE    # 83570
# 用户新增常规可见 token 的分配区（「添加 token」功能消耗此区尾段）。
EXTRA_SIZE = 4096
REGULAR_END = EXTRA_BEGIN + EXTRA_SIZE        # 87666

# —— fallback 可见区（a–z）——
FALLBACK_BEGIN = REGULAR_END                  # 87666
FALLBACK_LETTERS = "abcdefghijklmnopqrstuvwxyz"
FALLBACK_SIZE = 1024            # 26 个已用 + 998 预留

VOCAB_SIZE = FALLBACK_BEGIN + FALLBACK_SIZE   # 88690
# 为矩阵对齐补齐到 64 的倍数
VOCAB_SIZE_PADDED = (VOCAB_SIZE + 63) // 64 * 64  # 88704（尾部 reserved）

# 不可见区总边界（SPECIAL + SYSTEM + MODAL）
INVISIBLE_END = MODAL_BEGIN + MODAL_SIZE      # 12384

# —— 图元输出 token（照搬 wonderfulearth/primitive-operation-painter
#    geometrize_256_v1 布局；仅输出侧，视觉输入像素参数不入词表走投影）——
# 每个图元 = 9 个 token，字段序（cx, cy, theta, w, h, shape, r, g, b）；
# 各字段是独立 bin 区间（token ID = DRAW_BEGIN + 字段 offset + bin）：
#   cx/cy  [0,256) px   ×2/bin   → 512 bins
#   theta  [0,90) 度    ×3/bin   → 270 bins（CSV 里是弧度，≥90° 交换宽高）
#   w/h    [0,128) px   ×4/bin   → 512 bins
#   shape  -1=背景 0=矩形 1=椭圆（bin = shape_type+1，256 bins）
#   r/g/b  0-255 //2（各 128 bins）
#   special 3 bins（仓库布局自带：pad + 2 预留）
DRAW_BEGIN = MODAL_BEGIN                    # 96
DRAW_TOKENS_PER_PRIMITIVE = 9               # cx, cy, theta, w, h, shape, r, g, b
DRAW_FIELD_ORDER = ("cx", "cy", "theta", "w", "h", "shape", "r", "g", "b")
DRAW_CSV_COLUMNS = ("image_name", "cx", "cy", "w", "h", "shape_type",
                    "theta", "r", "g", "b")
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
DRAW_SHAPES = ("background", "rectangle", "ellipse")   # shape_type -1/0/1
# 字段 offset（相对 DRAW_BEGIN）
DRAW_X_OFFSET = 0
DRAW_Y_OFFSET = DRAW_X_OFFSET + DRAW_X_BINS            # 512
DRAW_ANGLE_OFFSET = DRAW_Y_OFFSET + DRAW_Y_BINS        # 1024
DRAW_W_OFFSET = DRAW_ANGLE_OFFSET + DRAW_ANGLE_BINS    # 1294
DRAW_H_OFFSET = DRAW_W_OFFSET + DRAW_W_BINS            # 1806
DRAW_SHAPE_OFFSET = DRAW_H_OFFSET + DRAW_H_BINS        # 2318
DRAW_R_OFFSET = DRAW_SHAPE_OFFSET + DRAW_SHAPE_BINS    # 2574
DRAW_G_OFFSET = DRAW_R_OFFSET + DRAW_COLOR_BINS        # 2702
DRAW_B_OFFSET = DRAW_G_OFFSET + DRAW_COLOR_BINS        # 2830
DRAW_SPECIAL_OFFSET = DRAW_B_OFFSET + DRAW_COLOR_BINS  # 2958
DRAW_VOCAB_SIZE = DRAW_SPECIAL_OFFSET + DRAW_SPECIAL_BINS   # 2961
DRAW_SIZE = 4096            # 2961 已用 + 1135 预留（图元方案演进空间）

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
# 隔离运行时全部装在项目里，不碰系统 Python：
#   start/.python/   python-build-standalone 解包（setup.sh 自动下载）
#   start/.venv/     以上述解释器建的虚拟环境（core/window/dev 依赖）
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
#     window.sha256  window.py 完整性基准（server 启动时校验）
#   server/        全部 server 文件：spec / common / model / muon +
#                  server.py（装配+路由+启动编排）+ server_env（环境检查/
#                  窗口校验/弹窗）+ server_services（服务监督器）+
#                  server_models / media / codecs / train / tokens / util
#   service/       每个服务一个目录：infer / train / init / tokens / web /
#                  window / test（按需启动，见下）
#   models/        models/NAME/{model.safetensors, config.json, vocab.txt?}
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
# 每模型词表（vocab.txt）：
#   models/NAME/vocab.txt 一行一个 token，第 N 行 → token ID =
#   INVISIBLE_END + N（不可见区 [0, INVISIBLE_END) 是 special/system/
#   modal 固定 token 的架构契约——<stop>/<caps>/{ } |/图元输出 token 的
#   ID 由上面的常量按 ID 定位，decode 的停止/大小写逻辑靠它们，自定义
#   词表不得占用）；空行跳过；重复拒绝（反查歧义）；条目数上限
#   VOCAB_SIZE_PADDED - INVISIBLE_END。扁平词表的可见部分全部按
#   regular 处理（special/system 不可见、不参与文本编码查表）。没有
#   vocab.txt 的模型用全局词表 service/tokens/vocab.json。
#   切换活动模型时，server / infer / train 进程各自 set_model_vocab()。
#
# 启动链（唯一入口）：
#   start/terminal/start.{sh,bat,command}
#     → setup.{sh,bat}（幂等：装/复用 start/.python + start/.venv）
#     → server/server.py
#         → server_env.check_environment（解释器版本 + 主依赖）
#         → server_env.verify_window_integrity（window.py sha256 校验，
#            不一致弹窗，可跳过；--no-verify 静默）
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
# 时间步结构（2026-10-24 tok-32 体系，tokens/timestep.py）：
#   离散 token 流中，每个时间步由系统 token { } 包裹，步内多个 token
#   由 | 分隔。文本每 token 独占一步（{ } | 不显式出现，由 step 边界
#   承载）；图片 = 一步（内部多 patch token）；音频每 533 采样 = 一个
#   token = 一步；视频每帧 = 一步。KDA 循环状态递归 / 逻辑并行以
#   时间步为原子：每处理完一个时间步内的 token，KDA 状态递归一次。
