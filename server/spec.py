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
#   - FFN 为 GLU（d_ff=4096）；2026-11-04 起每 block 5 个（见下方定案）。
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
# 定案记录（2026-10-31，steve 拍板，图元画布可变化）：
#   - 图元序列最前面加**画布声明**（DRAW 区第 0 段，`<x>` 数字 `<x/>` …），
#     先声明画多大再画；缺省/旧数据按 256×256。
#   - x/y 轴 bin 扩到 1920 / 1080（1 token = 1 像素），即
#     「xy 轴的 token 扩展到支持 1920×1080」。
#   - 画布上限 1920×1080（目前最大），竖屏不在预设表里。
#   - 步数上限取消：转换器想跑多少步就跑多少步（server 不再卡 1000）。
#
# 定案记录（2026-11-02，steve 拍板，w/h 取消 128px 上限）：
#   - w/h 与 x/y 同构：1 token = 1 像素，上限 = 画布上限（1920 / 1080）。
#     旧布局 w/h 是 4 bin/px × 512 bins → 最大 127.75 px，256 画布上一半宽
#     的形状根本表达不出来，第三方 GPU 转换器产出的 w=256 全被截半。
#   - 「画布变大靠更多更小的图元堆细节」这条随之作废：单个图元现在可以
#     和画布一样大（背景/大色块本来就该是这么大）。
#   - 代价：DRAW 区 7629 → 9605，HEAD_SIZE 7638 → 9614（词表已重建）。
#
# 定案记录（2026-10-25，steve 拍板，内部流程对齐）：
#   - block 内顺序（已被 2026-11-04 定案取代，见文件末）：
#     KDA → KDA → KDA → MLA-CSA → MLA-HCA → FFN → AttnRes。
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
# 定案记录（2026-11-05，steve 拍板，token-6 + base1.0.0 + 工具调用）：
#   - .tokens 升级为 token-6：6 bit/字符（2^6=64 码位），终止符码 63。
#     字符集 29 个单字符（原 27 + '[' ']'），6 个原子边界 token 各占一个
#     码位（29-34），码 35-62 预留。**不做向后兼容**：token-5 位流不再可读，
#     旧 base 词表整份删除重建为 base1.0.0。
#   - 边界 token 的显示方式改变：<user>/<user/>/<model>/<model/> 四个
#     token 语义与 ID 照旧，但改用 [] 包裹 → [user]/[user/]/[model]/[model/]；
#     另加系统段边界 [sys]/[sys/]（ID 6/7），供工具结果、系统提示注入。
#   - 原先 <> 的含义变成**工具调用**：< > 是两个独立字符，词表里的
#     <asktime>/<bash>/<thinking>/<b> 等专用 token 都是模型可预测、
#     有 KV、进 loss 的 agent 区 token，不再是「扩展系统 token」。
#   - 新增 agent 区（架构契约头内，位于 system 与 draw 之间）：工具调用
#     段边界、排版专用 token（全部是 token 而非 markdown 字面量）、
#     工具动作 token，以及专门的绘画结束 token <draw:end>。
#   - 词表带版本号：全局 base → base1.0.0（vocab_version 1.0.0），
#     vocabs/base/ 删除；旧的 ID 布局不兼容，下次再做兼容层。
# 定案记录（2026-11-06，steve 拍板，base1.1.0 + agent 段类型语义化）：
#   - 词表升到 base1.1.0（vocab_version 1.1.0）：可见区**末尾追加** code 区
#     （跨语言代码专用 token，去重后 474 条），只追加不插队——契约头与
#     全部既有可见区 ID 一个不动，旧权重 / .tokens / 插件缓存 ID 全兼容。
#     vocabs/base1.0.0/ 退役为 base1.1.0/。
#   - agent 区 53 个 token 的 type_emb 从 TYPE_NONE / TYPE_MODEL 改为
#     TYPE_SYS（与 [sys] 系统段同型），统一由 segment_type_of() 给定；
#     tok6.to_steps / infer._input_type_ids / 生成循环 _dec_type / out_steps
#     四处口径一致。token ID 不变，只改类型嵌入取值。
#   - 进 loss 的 token 一律注入 KV + fast weight（模型侧无按 token 类别的
#     过滤）；DRAW 图元 prims 不输入模型，故不涉及。
# 定案记录（2026-11-06 下午，steve 拍板，base1.2.0 契约头重排）：
#   - **一个预定义工具都不要**：53 个 agent token（边界/排版/动作）全部
#     退役，改成 **编号槽位** `<s0>`…`<sN-1>`（N 由词表 config 决定），
#     base 零预留；语义由使用者自己定义，
#     编号直接显示；工具执行先全部不接（工具台与测试台，留给推理 app）。
#   - `<caps>` / `<stop>` / `<draw:end>` 这类最早就有的机制单标记**单开
#     core 一类**：` <>` 形态、无闭合、进 loss、有 KV；不是 [] 段边界、
#     不是系统/特殊、也不是工具调用。段类型挂 TYPE_NONE。
#   - 头部五段：core 3 + boundary 6 + system 3 + slot N + draw 9605
#     = HEAD_SIZE 9617+N，DRAW_BEGIN 12+N。**N 由词表 config 的
#     slot_count 决定，spec 不钉死**；base 的 N=0 → 9617 / 12。
#     `gen_vocab.py` 以 base 为模板生成带 N 个槽位的新词表。
#     ID 整体位移，旧 .tokens / 权重 / 存档一律不可读。词表升 base1.2.0。
#
# 定案记录（2026-11-07，steve 拍板，base1.3.0 扩表 + Typst 结构 token）：
#   - 依据两份桌面文档：`gargantua-typst-token-spec.md`（全量 Typst
#     语法构造）与 `词表新增内容.md`（31 个 Unicode 区块、精确 3537 字）。
#   - **Typst 结构 token 并入架构契约头**（用户明确选定「ID 整体位移」
#     而非末尾追加）：114 个语法构造 × 显式开/闭 = 228 个 `<>` tools，
#     插在 slot 与 draw 之间。HEAD_SIZE 9617+N → 9845+N，
#     DRAW_BEGIN 12+N → 240+N。旧 .tokens / 权重 / 存档全部不可读。
#   - **形态约定**：`{}|` = system、`[]` = border、`<>` = tools
#     （core 机制单标记 + Typst 结构）、`<>` = 用户自定义预留（slot）、
#     `内容:HEX` = 可见常规 token。Typst 结构 token 与 core 同走 token-6
#     的 `<HEX>` tool 元素：进 loss、有 KV、段类型 TYPE_NONE。
#   - **3537 个新字符末尾追加** `unicode_ext` 区（code 之后），
#     category=regular，enctok 自动覆盖，无需改编码逻辑。
#   - 本次**不实现映射器**（token 流 → .typ）；改为交付机器可读
#     `typst_map.json` + 人可读 `TYPST_TOKENS.md` 对应表格，由
#     `service/tokens/gen_typst_map.py` 从 spec 唯一真源生成，禁止手抄 ID。
#
# 定案记录（2026-11-04，steve 拍板，enc/dec 权重模式可选）：
#   - 创建模型时可选「分离权重」（默认，v1 现状）或「共享权重」。共享 =
#     enc_blocks[i] 与 dec_blocks[i] 逐层共用同一批参数（两侧 final norm 也
#     一起绑定），因此**强制 n_enc_blocks == n_dec_blocks**——数量不等就
#     没有一一对应的层可绑。
#   - 共享的是参数对象，不是 Block 实例：encoder 双向（MLA 非因果、KDA 有
#     反向子层），decoder 因果（步间 + 步内 token 级因果、增量推理递推
#     状态）。这些行为差异由两个 Block 实例各自的 is_decoder / MLA.causal
#     保留，权重共享不改变任何因果语义。
#   - 权重落盘只存 enc 侧一份：safetensors 不允许两个 key 共享同一块
#     storage，而且存两份等于把省下来的磁盘又吃回去。config.json 的
#     share_enc_dec 是唯一开关，加载时按它把 dec 侧 key 镜像回内存
#     state_dict（再 strict load）。
#   - 旧模型目录没有该字段 → 视为分离权重，行为逐字不变。
#
# 定案记录（2026-11-04，steve 拍板，d_model 建模可选）：
#   - 创建模型时可选模型宽度 d_model（默认 1024）。它只改**语义向量宽度**：
#     wte/lm_head/type_emb/模态投影、共享 KV 流、各 block 的输入输出投影、
#     RMSNorm 全部跟随；注意力头数/头维、MLA 低秩秩、FFN 隐藏宽度仍按
#     本文件冻结值不变（换宽度不换架构，权重形状可控、旧目录全兼容）。
#   - 必须是 HEAD_DIM=128 的整数倍，且 ≥ D_MODEL_MIN（完全无上限，
#     steve 2026-11-05 拍板；宽度只影响 embedding/lm_head 的 V×D 行内存，
#     由用户自行权衡规模）。
#   - config.json 的 d_model 是加载侧的唯一依据；缺省 → 1024。
#
# 定案记录（2026-11-04，steve 拍板，block 子层重排）：
#   - 顺序改为 KDA → FFN → KDA → FFN → KDA → FFN → MLA-CSA → FFN →
#     MLA-HCA → FFN → AttnRes：每次递推/注意力之后立刻接一个 FFN 做逐位置
#     变换，不再把三个 KDA 连着堆完再统一过一次 FFN。
#   - 5 个 FFN 是**各自独立的 GLUFFN 实例**（不是同一层复用 5 次）；
#     10 个计算子层各有一份 RMSNorm，最后一个 "attnres" 是块末聚合/门控。
#   - 参数量随之上浮：默认 4+6 blocks 从 ~527M 变为 ~1.03B，权重文件约
#     4.1GB（fp32）。建模时可用 d_model / block 数 / 共享权重控制规模。
#   - state_dict 的 FFN 键从 enc_blocks.N.ffn_up/down 变成
#     enc_blocks.N.ffns.i.ffn_up/down；旧检查点结构不同，需重新建模。
#
# 定案记录（2026-11-08，steve 交付 MoMW 规格，实现侧按本机架构适配）：
#   - MoMW（Mixture-of-Memory-Weights）是**可选记忆阀门**，插在第一个
#     Block 之后、与既有 fast weight 同址并列（H_out = H_1 + ΔH）。
#     **默认关闭**：旧模型目录零影响——build_model 是 strict load，
#     state_dict 一个 key 都不许变。
#   - Router 无训练：不注册任何 nn.Parameter。Query 直接复用 fast weight
#     里 KDA 的 w_q 投影（已训练、确定性），按 head 分块与各层状态 S
#     做 einsum 打分；项目里 S 是 [B,H,dk,dv] 递推终态，不是 [D,D] 矩阵。
#   - Pool 层共享同一份 KDAAttention 权重（参数零新增），只各自持有独立
#     递推状态 (S, conv_cache)——沿用「共享参数不共享实例」的既有口径；
#     未被召回的层不前向，状态就不推进（未被召回则不遗忘）。
#   - 动态扩展只改运行时状态（层数 / bonus / coverage 不落盘），不改变
#     权重契约；新层零状态必须靠 bonus 路由偏置打破激活死锁。
#   - 与 fast weight 一样，MoMW 状态在样本边界由 reset_memory() 复位，
#     推理会话则放在外部 state 容器里跨轮持久。
#
# 定案记录（2026-11-08 第二版，steve 交付三阶段规格）：
#   - Router 打分从「Q@S 能量范数」改为**余弦混合**：阶段一锚点原型 P_k
#     （wte 代表性 token Embedding 均值，惰性首次 forward 计算、
#     persistent=False 缓存），阶段二幂迭代提取状态主特征 E_i 并用
#     γ_i=1-exp(-writes/k) 混合，阶段三空层分裂（禁止 clone）+ 指数衰减
#     bonus + 同类软负载均衡。详见 4b 节。
#   - 余弦打分量级 [-1,1]，threshold / B0 默认值随之整体重标；v1 的 10 个
#     config 字段全部保留，旧 config 无需迁移即可加载（新增字段取默认）。
#   - 「全异步 / 静默激活」落地为会话级 silent 开关：路由与写入照常执行
#     （后台实时写入），但 ΔH 不注入输出；状态读写仍在同一串行推理进程内，
#     不做跨线程并发写（避免竞争，也符合规格自己的工程提醒）。
#   - 推理界面加 MoMW 面板，统计走既有 /api/infer/stats 1.2s 轮询通道。
# ===========================================================================
import json
import math
from pathlib import Path

# 全局词表 config 决定「预留槽位数量」：spec 自己**不定死**这个数。
# base 词表零预留（slot_count=0），要预留就跑 service/tokens/gen_vocab.py
# 以 base 为模板生成一份新词表，它自己的 config 里写明 slot_count。
# 运行时布局常量从这里读到的值派生；读不到 / 非法一律回退 0（= base）。
_GLOBAL_VOCAB_CONFIG = (Path(__file__).resolve().parent.parent
                        / "service" / "tokens" / "config.json")
BASE_SLOT_COUNT = 0                 # base 永远零预留
MAX_SLOT_COUNT = 4096               # 防呆：一份词表最多预留这么多


def _global_slot_count() -> int:
    """从全局词表 config 读 slot_count；缺省 / 非法 → 0（base 布局）。"""
    try:
        cfg = json.loads(_GLOBAL_VOCAB_CONFIG.read_text(encoding="utf-8"))
        n = cfg.get("slot_count", BASE_SLOT_COUNT)
        if (isinstance(n, int) and not isinstance(n, bool)
                and 0 <= n <= MAX_SLOT_COUNT):
            return n
    except Exception:
        pass
    return BASE_SLOT_COUNT


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
# 照搬 wonderfulearth/primitive-operation-painter geometrize_256_v1，
# 2026-11-01 起重排 x/y 与颜色（steve 定案）。
# 仅输出侧——视觉输入的像素参数不入词表，走线性投影。
#
# 一条图元序列 = 一段画布声明 + N × 9 个图元 token：
#   <x> 数字… <x/> <y> 数字… <y/>   [x y theta w h shape r g b] ...
# 画布声明必须在最前面（「刚开画先说画多大」），它决定 x/y 的取值范围；
# 缺省（旧数据 / 没有画布声明）按 DRAW_DEFAULT_CANVAS = 256×256 处理。
#
# 画布声明用 4 个定界 token + 可见区的文本数字 token（"0".."9"）：
#   <x> 1 9 2 0 <x/> <y> 1 0 8 0 <y/>   → 1920×1080
# 好处是画布**不再是固定档位**：1920×1080 以内的任意尺寸都写得出来，
# 而词表只为 4 个定界符付代价（数字 token 本来就存在）。
#
# 每个图元 = 9 个 token，字段序（x, y, theta, w, h, shape, r, g, b）；
# 各字段是独立 bin 区间（token ID = DRAW_BEGIN + 字段 offset + bin）：
#   x      [0,1920) px  1 token/px → 1920 bins（x0…x1919，从 0 起）
#   y      [0,1080) px  1 token/px → 1080 bins（y0…y1079）
#   theta  [0,90) 度    ×3/bin     → 270 bins（CSV 里是弧度，≥90° 交换宽高）
#   w      [0,1920) px  1 token/px → 1920 bins（w0…w1919）
#   h      [0,1080) px  1 token/px → 1080 bins（h0…h1079）
#   shape  -1=背景 0=矩形 1=椭圆（bin = shape_type+1，256 bins）
#   r/g/b  0-1023（10 bit，1 bin/值，各 1024 bins）
#   special 3 bins（仓库布局自带：pad + 2 预留）
#
# 为什么 x/y/w/h 全是 1 token/像素：原来 x/y 是 2 bin/px、w/h 是 4 bin/px，
# 「token 编号 ≠ 像素」逼着人做心算，而且 w/h 被 512 bins ÷ 4 卡死在
# 127.75 px——256 画布上一半宽的形状根本表达不出来（第三方 GPU 转换器
# 产出的 w 最大到 256，token 化时全被截成 127.75，预览与 token 对不上）。
# 2026-11-02 起四个几何字段统一 1 token = 1 像素、上限 = 画布上限
# （1920 / 1080）：`w1234` 就是 1234 像素宽，任何画布尺寸都表达得下。
#
# 为什么颜色是 10 bit：图像侧本来就是 RGBA 各 10 bit（见第 5 节），
# 图元输出也用 10 bit 才能无损表达同一套颜色，不再 //2 掉一半精度。
DRAW_CANVAS_SIZES = (
    (256, 256),      # 默认：geometrize_256_v1 原画布
    (384, 384),
    (512, 512),
    (640, 480),      # 4:3
    (768, 768),
    (1024, 1024),
    (1024, 576),     # 16:9
    (1280, 720),     # 16:9
    (1920, 1080),    # 上限（steve 2026-11-01：目前最大 1920×1080）
)
DRAW_DEFAULT_CANVAS = DRAW_CANVAS_SIZES[0]             # (256, 256)
# 画布上限：x/y 轴 token 数由它推出。**不是**「只能用这几档」——
# 具体一张图用多大，由序列开头那段画布声明说了算（见上）。
DRAW_CANVAS_MAX_W = 1920
DRAW_CANVAS_MAX_H = 1080

# 画布声明的 4 个定界 token（DRAW 区最前面 4 个），顺序即 bin 下标
DRAW_CANVAS_MARKERS = ("<x>", "<x/>", "<y>", "<y/>")
DRAW_CANVAS_OPEN_X, DRAW_CANVAS_CLOSE_X, \
    DRAW_CANVAS_OPEN_Y, DRAW_CANVAS_CLOSE_Y = DRAW_CANVAS_MARKERS
DRAW_CANVAS_BINS = len(DRAW_CANVAS_MARKERS)            # 4
# 画布数字用可见区的文本数字 token（"0".."9"）。可见区的第一段就是
# digit（见 VISIBLE_REGION_ORDER），所以 ID 恰好在 HEAD_SIZE 之后连续
# 10 个——没有词表实例时按这个位置兜底（有词表时一律查表）。
DRAW_CANVAS_DIGITS = "0123456789"

DRAW_TOKENS_PER_PRIMITIVE = 9               # x, y, theta, w, h, shape, r, g, b
DRAW_FIELD_ORDER = ("x", "y", "theta", "w", "h", "shape", "r", "g", "b")
# DRAW 区的字段顺序（画布在最前，其余 9 个是图元字段）。词表头按这个
# 顺序推导名字，draw_codec 只认 DRAW_FIELD_ORDER 那 9 个。
DRAW_HEAD_FIELD_ORDER = ("canvas",) + DRAW_FIELD_ORDER
DRAW_CSV_COLUMNS = ("image_name", "cx", "cy", "w", "h", "shape_type",
                    "theta", "r", "g", "b")
DRAW_SHAPES = ("background", "rectangle", "ellipse")   # shape_type -1/0/1
XY_BINS_PER_PIXEL = 1                       # 1 token = 1 像素
ANGLE_BINS_PER_DEGREE = 3
# w/h 与 x/y 同构：1 token = 1 像素，上限 = 画布上限（不是 128 那个历史值）
SIZE_BINS_PER_PIXEL = 1
DRAW_X_BINS = DRAW_CANVAS_MAX_W * XY_BINS_PER_PIXEL    # 1920
DRAW_Y_BINS = DRAW_CANVAS_MAX_H * XY_BINS_PER_PIXEL    # 1080
DRAW_ANGLE_BINS = 270
DRAW_W_BINS = DRAW_CANVAS_MAX_W * SIZE_BINS_PER_PIXEL  # 1920
DRAW_H_BINS = DRAW_CANVAS_MAX_H * SIZE_BINS_PER_PIXEL  # 1080
DRAW_SHAPE_BINS = 256
# 颜色：10 bit / 通道，1 bin 一个值（0..1023）
DRAW_COLOR_BITS = 10
DRAW_COLOR_BINS = 1 << DRAW_COLOR_BITS                 # 1024
DRAW_COLOR_MAX = DRAW_COLOR_BINS - 1                   # 1023
DRAW_SPECIAL_BINS = 3
# 字段 offset（相对 DRAW_BEGIN）——由上面的 bin 数累加而来，与 base 无关。
# 所以 DRAW_BEGIN 变成运行时值也不影响这段算术（draw_codec 直接用
# `b + offset + bin`，b 一变自动跟随）。
DRAW_CANVAS_OFFSET = 0
DRAW_X_OFFSET = DRAW_CANVAS_OFFSET + DRAW_CANVAS_BINS  # 4
DRAW_Y_OFFSET = DRAW_X_OFFSET + DRAW_X_BINS            # 1924
DRAW_ANGLE_OFFSET = DRAW_Y_OFFSET + DRAW_Y_BINS        # 3004
DRAW_W_OFFSET = DRAW_ANGLE_OFFSET + DRAW_ANGLE_BINS    # 3274
DRAW_H_OFFSET = DRAW_W_OFFSET + DRAW_W_BINS            # 5194
DRAW_SHAPE_OFFSET = DRAW_H_OFFSET + DRAW_H_BINS        # 6274
DRAW_R_OFFSET = DRAW_SHAPE_OFFSET + DRAW_SHAPE_BINS    # 6530
DRAW_G_OFFSET = DRAW_R_OFFSET + DRAW_COLOR_BINS        # 7554
DRAW_B_OFFSET = DRAW_G_OFFSET + DRAW_COLOR_BINS        # 8578
DRAW_SPECIAL_OFFSET = DRAW_B_OFFSET + DRAW_COLOR_BINS  # 9602
DRAW_VOCAB_SIZE = DRAW_SPECIAL_OFFSET + DRAW_SPECIAL_BINS   # 9605


def draw_canvas_valid(w, h) -> bool:
    """画布尺寸是否在契约范围内（1..1920 × 1..1080）。

    2026-11-01 起画布不再是固定档位——`<x>`/`<y>` 之间的数字可以是任意
    整数，所以这里只守上下限，不查表。
    """
    try:
        wi, hi = int(w), int(h)
    except (TypeError, ValueError):
        return False
    return 1 <= wi <= DRAW_CANVAS_MAX_W and 1 <= hi <= DRAW_CANVAS_MAX_H


def draw_canvas_auto(img_w, img_h):
    """按图片长宽比取**精确比例**的画布，上限 1920×1080。

    2026-11-03（steve）：原来是从 DRAW_CANVAS_SIZES 里挑「比例最接近的一档」，
    竖图会被塞进 1024×1024 —— 比例差一大截，画面要么被拉伸、要么四周留边。
    既然画布声明本来就支持任意尺寸，就没有理由再回到预设表：

        scale = min(1920/iw, 1080/ih)   →   取整(iw*scale, ih*scale)

    也就是「塞得进上限的最大同比例画布」。预设表 DRAW_CANVAS_SIZES 仍然保留，
    给图元编辑器的下拉框用。
    """
    try:
        iw, ih = float(img_w), float(img_h)
    except (TypeError, ValueError):
        return DRAW_DEFAULT_CANVAS
    if iw <= 0 or ih <= 0:
        return DRAW_DEFAULT_CANVAS
    scale = min(DRAW_CANVAS_MAX_W / iw, DRAW_CANVAS_MAX_H / ih)
    w = max(1, min(DRAW_CANVAS_MAX_W, int(round(iw * scale))))
    h = max(1, min(DRAW_CANVAS_MAX_H, int(round(ih * scale))))
    return (w, h)

# ---------------------------------------------------------------------------
# 1b. 架构契约头（冻结：名字 + 顺序 + ID；所有词表共享，不得占用/重排）
# ---------------------------------------------------------------------------
# 2026-11-06（steve 定案）：不再预定义任何工具 / 排版 token。整个 agent
# 区改成 **编号槽位**，数量 N 由每份词表 config 的 slot_count 决定
# （spec 不钉死；base 的 N=0，gen_vocab.py 生成带 N 个槽位的新词表），
# 语义由使用者自己定义；而 <caps>/<stop>/<draw:end> 这种最早就有的
# 机制单标记 **单开 core 一类**——既不是 [] 段边界，也不是工具调用。
# 头部六段顺序冻结（前 12 个 ID 恒定，其余随 N 平移；base1.3.0 起）：
#   core(3) → boundary(6) → system(3) → slot(N) → typst(228) → draw(9605)
#   = 9845+N，DRAW_BEGIN = 240+N。
# 本轮 ID 布局全变，按项目约定不做向后兼容（旧 .tokens / 权重不可读）。
# typst 段在 slot **之后**是刻意的：vfc.infer_slot_count() 靠「固定前缀
# 12 + 直到第一个 Typst 结构 token 之前」自描述槽位段长度（2026-11-07
# 起 slot 名字可自定义、不再只认连续 <sN>），typst 放前面会把它截断。

# —— core 控制标记（<> 形态、无闭合、进 loss、有 KV）——
CORE_TOKENS = [
    "<caps>",        # 0 大写标记：其后单词/字母首字母大写
    "<stop>",        # 1 生成停止标记：模型输出即停止
    "<draw:end>",    # 2 绘画结束：其后输出恢复为文本
]
CORE_BEGIN = 0x00000000
CORE_SIZE = len(CORE_TOKENS)                     # 3
CORE_END = CORE_BEGIN + CORE_SIZE
CAPS_ID = CORE_TOKENS.index("<caps>")            # 0
STOP_TOKEN = "<stop>"
STOP_ID = CORE_TOKENS.index(STOP_TOKEN)          # 1
DRAW_END_TOKEN = "<draw:end>"
DRAW_END_ID = CORE_TOKENS.index(DRAW_END_TOKEN)  # 2

# —— boundary 段边界（[] 形态；界定 TYPE_USER / TYPE_MODEL / TYPE_SYS）——
# 推理存档（.tokens）用它把「用户输入 / 模型输出 / 系统」包成结构。
BOUNDARY_TOKENS = [
    "[user]", "[user/]",      # 3/4 用户输入段
    "[model]", "[model/]",    # 5/6 模型输出段
    "[sys]", "[sys/]",        # 7/8 系统段
]
BOUNDARY_BEGIN = CORE_END                        # 3
BOUNDARY_SIZE = len(BOUNDARY_TOKENS)             # 6
BOUNDARY_END = BOUNDARY_BEGIN + BOUNDARY_SIZE
USER_ID = BOUNDARY_BEGIN + BOUNDARY_TOKENS.index("[user]")          # 3
USER_END_ID = BOUNDARY_BEGIN + BOUNDARY_TOKENS.index("[user/]")      # 4
MODEL_ID = BOUNDARY_BEGIN + BOUNDARY_TOKENS.index("[model]")        # 5
MODEL_END_ID = BOUNDARY_BEGIN + BOUNDARY_TOKENS.index("[model/]")    # 6
SYS_ID = BOUNDARY_BEGIN + BOUNDARY_TOKENS.index("[sys]")            # 7
SYS_END_ID = BOUNDARY_BEGIN + BOUNDARY_TOKENS.index("[sys/]")        # 8

# 兼容别名：头两段合称 special（vfc.head_entries / 加载侧沿用这个名字）。
SPECIAL_TOKENS = CORE_TOKENS + BOUNDARY_TOKENS
SPECIAL_BEGIN = CORE_BEGIN
SPECIAL_SIZE = CORE_SIZE + BOUNDARY_SIZE        # 9

# —— system 系统（不可显示，时间步结构）——
# 显示即字面量：{ = 时间步开始，} = 时间步结束，| = 步内 token 分隔。
TS_OPEN = "{"
TS_CLOSE = "}"
TS_SEP = "|"
SYSTEM_CHARS = [TS_OPEN, TS_CLOSE, TS_SEP]
SYSTEM_BEGIN = SPECIAL_BEGIN + SPECIAL_SIZE   # 9
# 旧值 32（3 已用 + 29 预留）→ 按实际收录数
SYSTEM_SIZE = len(SYSTEM_CHARS)               # 3
TS_OPEN_ID = SYSTEM_BEGIN + 0                  # 9
TS_CLOSE_ID = SYSTEM_BEGIN + 1                 # 10
TS_SEP_ID = SYSTEM_BEGIN + 2                   # 11

# —— slot 编号槽位（**数量由词表 config 决定，spec 不定死**）——
# 名字 <s0>..<sN-1>：s 不是十六进制字符，不会被 tok6 的 <HEX> 解析误吞；
# 编号直接显示，不预设任何工具 / 排版 / 段边界语义。使用者自己在 SLOT_DEFS
# 里给槽位起名、配对、接执行；项目默认**一个都不接**（工具台与测试台，
# 实际执行留给将来的推理 app）。
#
# base 词表零预留（slot_count=0 → 没有 slot 段，DRAW_BEGIN=240）。
# 想要预留槽位：python service/tokens/gen_vocab.py --name NAME --slots N
# 以 base 为模板生成新词表，新词表的 config 自己写明 slot_count=N。
SLOT_COUNT = _global_slot_count()
SLOT_TOKENS = tuple(f"<s{i}>" for i in range(SLOT_COUNT))
SLOT_BEGIN = SYSTEM_BEGIN + SYSTEM_SIZE         # 12（固定）
SLOT_SIZE = SLOT_COUNT
SLOT_END = SLOT_BEGIN + SLOT_SIZE
SLOT_ID = {name: SLOT_BEGIN + i
           for i, name in enumerate(SLOT_TOKENS)}
SLOT_NAME_BY_ID = {i: n for n, i in SLOT_ID.items()}


# ---------------------------------------------------------------------------
# 1b-2. Typst 结构 token（2026-11-07 base1.3.0 新增，并入架构契约头）
# ---------------------------------------------------------------------------
# 唯一真源：gargantua-typst-token-spec.md §3「全量 token 词表」+ §2 属性机制。
# 硬约束（steve）：每个语法构造都有**显式开/闭两个 token**，无 hybrid；
# 结构用特殊 token 承担、散文用普通子词；模型只吐 token ID，映射器
# （确定性代码，后续独立实现）翻译成 .typ 再编译 PDF。
#
# 形态约定（steve 2026-11-07）：`{}|` = system、`[]` = border、
# `<>` = tools（core 机制单标记 + 本节 Typst 结构）、`<>` = 用户自定义
# 预留（slot 编号槽位）、`内容:HEX` = 可见常规 token。
#
# 命名：开 `<NAME>` / 闭 `</NAME>`。闭标签首字符是 '/'，所以 vfc 的标记
# 名字正则必须允许可选前导 '/'（见 tokens/vfc.py）。
# 排布：每个构造占**两个连续 ID，开在前、闭在后**，配对 id = open_id + 1。
# 映射器与生成表格都可直接按此取，不必再查一次表。
#
# 字段：(name, section, typst, kind, empty, attrs, note)
#   typst = 规范 §3「Typst」列原文（保留原文，不臆断精确语法）
#   kind  = 构造大类；empty = 无内容元素（仍保留显式空对，见规范 §2）
#   attrs = 允许的具名属性（由 <ATTR> 承载），多个用 ',' 分隔
TYPST_CONSTRUCT_FIELDS = ("name", "section", "typst", "kind", "empty",
                          "attrs", "note")
TYPST_CONSTRUCTS = (
    # —— §3.1 文档根与页面 ——
    ("DOC", "3.1", "（文档根，可省略）", "root", False, "", "顶层包裹"),
    ("PAGESET", "3.1", "set page(..)", "page", False, "margin,paper,columns", "页边距/纸张/分栏"),
    ("TEXTSET", "3.1", "set text(..)", "page", False, "font,size,lang,cjk", "字体/字号/CJK"),
    ("PARSET", "3.1", "set par(..)", "page", False, "spacing,justify,leading", "段落间距/对齐"),
    ("DOCSET", "3.1", "set document(..)", "page", False, "title,author", "标题/作者"),
    ("HEADER", "3.1", "page(header: ..)", "page", False, "", "页眉"),
    ("FOOTER", "3.1", "page(footer: ..)", "page", False, "", "页脚"),
    ("BIB", "3.1", "bibliography(..)", "empty", True, "path,style", "参考文献 ∅"),
    ("OUTLINE", "3.1", "outline", "empty", True, "title,depth", "目录 ∅"),
    ("INCLUDE", "3.1", 'include ".."', "empty", True, "src", "引入模块 ∅"),
    ("IMPORT", "3.1", 'import ".."', "page", False, "src", "导入"),
    # —— §3.2 标题 / 段落 / 块级容器 ——
    ("H1", "3.2", "= ", "heading", False, "", "一级标题"),
    ("H2", "3.2", "== ", "heading", False, "", "二级标题"),
    ("H3", "3.2", "=== ", "heading", False, "", "三级标题"),
    ("H4", "3.2", "==== ", "heading", False, "", "四级标题"),
    ("H5", "3.2", "===== ", "heading", False, "", "五级标题"),
    ("H6", "3.2", "====== ", "heading", False, "", "六级标题"),
    ("P", "3.2", "段落（空行分隔）", "block", False, "", "显式段落"),
    ("QUOTE", "3.2", "quote[..]", "block", False, "", "引用块"),
    ("BLOCK", "3.2", "block(..)[..]", "block", False, "width,height,fill", "通用块"),
    ("BOX", "3.2", "box(..)[..]", "block", False, "width,height", "行内块"),
    ("PAD", "3.2", "pad(..)[..]", "block", False, "x,y,left,right,top,bottom", "内边距"),
    ("COLUMNS", "3.2", "columns(2)[..]", "block", False, "n", "分栏"),
    ("GRID", "3.2", "grid(..)", "block", False, "columns,rows,gutter", "网格布局"),
    ("STACK", "3.2", "stack(..)", "block", False, "dir,spacing", "堆叠布局"),
    ("ALIGN", "3.2", "align(center)[..]", "block", False, "mode", "对齐"),
    ("PLACE", "3.2", "place(..)[..]", "block", False, "dx,dy", "绝对定位"),
    ("FIGURE", "3.2", "figure(..)", "block", False, "caption", "图/表浮动体"),
    ("CAPTION", "3.2", "figure 的 caption", "block", False, "", "图注"),
    ("PARBREAK", "3.2", "parbreak", "empty", True, "", "段落分隔 ∅"),
    ("LINEBREAK", "3.2", "linebreak", "empty", True, "", "换行 ∅"),
    ("PAGEBREAK", "3.2", "pagebreak", "empty", True, "", "分页 ∅"),
    ("SPACER", "3.2", "v(..) / h(..)", "empty", True, "amount", "间距 ∅"),
    # —— §3.3 行内强调与文本 ——
    ("B", "3.3", "*..*", "inline", False, "", "加粗"),
    ("I", "3.3", "_.._", "inline", False, "", "斜体"),
    ("CODE", "3.3", "行内代码反引号", "inline", False, "", "行内代码"),
    ("LINK", "3.3", 'link("..")[..]', "inline", False, "url", "链接"),
    ("RAW", "3.3", "围栏原始块", "inline", False, "lang", "代码块（保留格式）"),
    ("LABEL", "3.3", "label 名", "inline", False, "", "标签"),
    ("REF", "3.3", "@name", "inline", False, "", "引用"),
    ("SYM", "3.3", "-> 等符号简写", "empty", True, "name", "符号 ∅"),
    ("COMMENT", "3.3", "/* .. */", "inline", False, "", "注释（输出丢弃）"),
    # —— §3.4 列表 ——
    ("UL", "3.4", "list(..)[..]", "list", False, "", "无序列表"),
    ("OL", "3.4", "enum(..)[..]", "list", False, "", "有序列表"),
    ("TERMS", "3.4", "terms(..)[..]", "list", False, "", "术语列表"),
    ("LI", "3.4", "列表项", "list", False, "", "UL/OL 子项"),
    ("TERM", "3.4", "术语", "list", False, "", "TERMS 子项"),
    ("DESC", "3.4", "描述", "list", False, "", "TERMS 子项"),
    # —— §3.5 表格 ——
    ("TABLE", "3.5", "#table(columns:.., ..)", "table", False, "columns", "表格"),
    ("ROW", "3.5", "一行", "table", False, "", "含若干 CELL"),
    ("CELL", "3.5", "单元格内容", "table", False, "colspan,rowspan", "单元格"),
    ("TH", "3.5", "表头单元格", "table", False, "colspan", "加粗表头"),
    # —— §3.6 数学（通道 + 构造，全 token 化，不使用自由数学文本）——
    ("MATH", "3.6", "$..$", "math", False, "", "行内数学"),
    ("MATHB", "3.6", "$ .. $", "math", False, "", "块级数学"),
    ("MSUB", "3.6", "_", "math", False, "", "下标"),
    ("MSUP", "3.6", "^", "math", False, "", "上标"),
    ("MFRAC", "3.6", "frac(..)", "math", False, "", "分数（含 NUM/DEN）"),
    ("NUM", "3.6", "分子", "math", False, "", "MFRAC 分子"),
    ("DEN", "3.6", "分母", "math", False, "", "MFRAC 分母"),
    ("MSQRT", "3.6", "sqrt(..)", "math", False, "", "平方根"),
    ("MROOT", "3.6", "root(..)", "math", False, "", "n 次根"),
    ("MSUM", "3.6", "sum", "math", False, "", "求和（含 MBELOW/MABOVE）"),
    ("MBELOW", "3.6", "下限", "math", False, "", "MSUM 下限"),
    ("MABOVE", "3.6", "上限", "math", False, "", "MSUM 上限"),
    ("MINT", "3.6", "integral", "math", False, "", "积分"),
    ("MLIM", "3.6", "lim", "math", False, "", "极限"),
    ("MMAT", "3.6", "mat(..)", "math", False, "", "矩阵（含 MROW）"),
    ("MROW", "3.6", "矩阵一行", "math", False, "", "MMAT 子项"),
    ("MCASE", "3.6", "cases(..)", "math", False, "", "分段"),
    ("MALIGN", "3.6", "&", "empty", True, "", "对齐点 ∅"),
    ("MFUNC", "3.6", "sin(..)", "math", False, "name", "数学函数"),
    ("MID", "3.6", "变量名", "math", False, "", "标识符"),
    ("MN", "3.6", "数字字面量", "math", False, "", "数字"),
    ("MOP", "3.6", "+ - * / = < > 等", "math", False, "", "运算符（允许集见规范）"),
    ("MTEXT", "3.6", '数学中文本 ".."', "math", False, "", "数学中文本"),
    # —— §3.7 代码 / 脚本 / 样式 ——
    ("CODEBLOCK", "3.7", "#{ .. }", "code", False, "", "代码块"),
    ("CBLOCK", "3.7", "[..]", "code", False, "", "内容块"),
    ("SET", "3.7", "set text(..)", "code", False, "sel", "设值规则"),
    ("SETIF", "3.7", "set text(..) if ..", "code", False, "sel,cond", "条件设值"),
    ("SHOW", "3.7", "show heading: ..", "code", False, "sel", "显示规则"),
    ("SHOWSET", "3.7", "show heading: set ..", "code", False, "sel", "显示-设值"),
    ("SHOWFUNC", "3.7", "show raw: it => ..", "code", False, "sel", "变换规则"),
    ("SHOWALL", "3.7", "show: template", "code", False, "", "全局套用"),
    ("SHOWTEXT", "3.7", 'show "badly": ..', "code", False, "text", "文本替换"),
    ("SHOWREGEX", "3.7", "show regex(..): ..", "code", False, "pat", "正则选择"),
    ("CONTEXT", "3.7", "context ..", "code", False, "", "自省上下文"),
    ("LET", "3.7", "let x = ..", "code", False, "name", "绑定"),
    ("DEF", "3.7", "let f(x) = ..", "code", False, "name", "具名函数"),
    ("IF", "3.7", "if ..", "code", False, "", "条件"),
    ("ELIF", "3.7", "else if ..", "code", False, "cond", "条件分支"),
    ("ELSE", "3.7", "else ..", "code", False, "", "条件分支"),
    ("FOR", "3.7", "for x in ..", "code", False, "var,in", "循环"),
    ("WHILE", "3.7", "while ..", "code", False, "cond", "循环"),
    ("RETURN", "3.7", "return ..", "code", False, "", "返回"),
    ("BREAK", "3.7", "break", "empty", True, "", "循环控制 ∅"),
    ("CONTINUE", "3.7", "continue", "empty", True, "", "循环控制 ∅"),
    ("CALL", "3.7", "#rect(..)[..]", "code", False, "name", "通用函数调用"),
    ("ARG", "3.7", "具名参数 k: v", "code", False, "name", "CALL 参数"),
    # —— §3.8 值 / 字面量（代码通道内）——
    ("STR", "3.8", '"hello"', "value", False, "", "字符串"),
    ("INT", "3.8", "10", "value", False, "", "整数"),
    ("FLOAT", "3.8", "3.14", "value", False, "", "浮点"),
    ("BOOL", "3.8", "true / false", "value", False, "", "布尔"),
    ("NONE", "3.8", "none", "empty", True, "", "占位 ∅"),
    ("AUTO", "3.8", "auto", "empty", True, "", "占位 ∅"),
    ("LEN", "3.8", "2pt", "value", False, "", "长度"),
    ("ANG", "3.8", "90deg", "value", False, "", "角度"),
    ("FRAC", "3.8", "2fr", "value", False, "", "弹性比例"),
    ("RATIO", "3.8", "50%", "value", False, "", "百分比"),
    ("ARR", "3.8", "(..)", "value", False, "", "数组（含 ITEM）"),
    ("ITEM", "3.8", "数组项", "value", False, "", "ARR 子项"),
    ("DICT", "3.8", "(k: v)", "value", False, "", "字典（含 KV）"),
    ("KV", "3.8", "k: v", "value", False, "k", "DICT 子项"),
    ("CITE", "3.8", "cite(..)", "empty", True, "key", "引用 ∅"),
    # —— §2 通用属性机制（规范 §4.1/§4.3 需要；本次按显式开/闭对实现，
    #     属性字面走内容通道，避免引入单 token 的 hybrid 例外）——
    ("ATTR", "2", 'name="k" value="v"', "meta", False, "name,value",
     "通用属性，紧跟开标签，映射器聚合成具名参数"),
)

_TYPST_NAMES = tuple(c[0] for c in TYPST_CONSTRUCTS)
TYPST_OPEN_TOKENS = tuple(f"<{n}>" for n in _TYPST_NAMES)
TYPST_CLOSE_TOKENS = tuple(f"</{n}>" for n in _TYPST_NAMES)
# 交错排布：<NAME>, </NAME>, <NEXT>, </NEXT>, …
TYPST_TOKENS = tuple(t for pair in zip(TYPST_OPEN_TOKENS, TYPST_CLOSE_TOKENS)
                     for t in pair)
TYPST_CONSTRUCT_COUNT = len(TYPST_CONSTRUCTS)      # 114
TYPST_SIZE = len(TYPST_TOKENS)                     # 228
assert TYPST_SIZE == 2 * TYPST_CONSTRUCT_COUNT
assert len(set(_TYPST_NAMES)) == TYPST_CONSTRUCT_COUNT, "Typst 构造重名"
assert len(set(TYPST_TOKENS)) == TYPST_SIZE, "Typst token 重名"

# 契约头内位置：紧跟 slot 段（插在 slot 与 draw 之间）。放在 slot 之后是
# 为了让 vfc.infer_slot_count() 的「固定前缀 12 + 到首个 Typst token 为止」
# 自描述逻辑保持原样（slot 名字自定义也照数不误）；gen_vocab.py 的
# tail = entries[12+base_slots:] 也自动包含本段。
TYPST_BEGIN = SLOT_END
TYPST_END = TYPST_BEGIN + TYPST_SIZE
TYPST_ID = {name: TYPST_BEGIN + 2 * i
            for i, name in enumerate(_TYPST_NAMES)}       # 开 token ID
TYPST_CLOSE_ID = {name: TYPST_BEGIN + 2 * i + 1
                  for i, name in enumerate(_TYPST_NAMES)}  # 闭 token ID
TYPST_TOKEN_ID = {t: TYPST_BEGIN + i
                  for i, t in enumerate(TYPST_TOKENS)}
TYPST_NAME_BY_ID = {i: t for t, i in TYPST_TOKEN_ID.items()}
TYPST_BY_NAME = {c[0]: dict(zip(TYPST_CONSTRUCT_FIELDS, c))
                 for c in TYPST_CONSTRUCTS}


def slot_tokens(n: int) -> tuple:
    """任意预留数 N → N 个编号槽位名字（gen_vocab 用，不依赖运行时布局）。"""
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError(f"slot_count 必须是 ≥0 的整数: {n!r}")
    if n > MAX_SLOT_COUNT:
        raise ValueError(f"slot_count 超过上限 {MAX_SLOT_COUNT}: {n}")
    return tuple(f"<s{i}>" for i in range(n))


def head_layout(n: int = None) -> dict:
    """给定预留数 N → 该词表的完整头部布局（纯函数，不读全局状态）。

    core / boundary / system 三段固定（0..11），slot 段长度 N，
    typst 段固定 TYPST_SIZE（base1.3.0 起并入契约头），draw 段固定 9605；
    DRAW_BEGIN 与 HEAD_SIZE 随 N 平移。gen_vocab 与前端用它算任意词表的
    ID，不必 import 时就把 128 之类的数字钉死在 spec 里。
    """
    if n is None:
        n = SLOT_COUNT
    names = slot_tokens(n)
    slot_begin = SYSTEM_BEGIN + SYSTEM_SIZE
    typst_begin = slot_begin + n
    draw_begin = typst_begin + TYPST_SIZE
    return {
        "slot_count": n,
        "core_begin": CORE_BEGIN, "core_size": CORE_SIZE,
        "boundary_begin": BOUNDARY_BEGIN, "boundary_size": BOUNDARY_SIZE,
        "system_begin": SYSTEM_BEGIN, "system_size": SYSTEM_SIZE,
        "slot_begin": slot_begin, "slot_size": n,
        "slot_tokens": names,
        "typst_begin": typst_begin, "typst_size": TYPST_SIZE,
        "typst_tokens": TYPST_TOKENS,
        "draw_begin": draw_begin, "draw_size": DRAW_VOCAB_SIZE,
        "head_size": draw_begin + DRAW_VOCAB_SIZE,
    }

# 兼容别名：旧代码里的 AGENT_* 现在统一指向 slot 区。
AGENT_TOKENS = SLOT_TOKENS
AGENT_BEGIN = SLOT_BEGIN
AGENT_SIZE = SLOT_SIZE
AGENT_END = SLOT_END
AGENT_ID = SLOT_ID
AGENT_NAME_BY_ID = SLOT_NAME_BY_ID

# 自定义槽位元数据（**默认全空**）。想定义工具/排版时自己填，例如：
#   SLOT_DEFS = {
#       0: {"name": "bash", "kind": "action", "close": 1,
#           "executable": False, "description": "终端命令", "arg_hint": "命令"},
#       1: {"name": "bash/", "kind": "close"},
#   }
# kind ∈ {"action","format","boundary","close"}；close 是闭合槽位编号（可选）。
# executable=True 才会被 /api/agent/tool 放行——默认全部 False。
SLOT_DEFS: dict = {}
TOOL_SPECS: dict = {}          # 退役（2026-11-06）：工具语义改由 SLOT_DEFS 自定义
TOOL_EXECUTABLE = frozenset()  # 先全部不接

# —— modal 模态输出（图元 DRAW；token 布局见 1a）——
# base1.3.0：typst 结构段插在 slot 与 draw 之间，DRAW_BEGIN 随之 +TYPST_SIZE。
MODAL_BEGIN = TYPST_END
MODAL_SIZE = DRAW_VOCAB_SIZE                   # 9605（= DRAW_VOCAB_SIZE）
DRAW_BEGIN = MODAL_BEGIN                      # 12 + SLOT_COUNT + 228（base = 240）

# 架构契约头长度 = core 3 + boundary 6 + system 3 + slot N
#                + typst 228 + draw 9605 = 9845 + N，N 由全局词表
# config 的 slot_count 决定。可见区从这里开始。slot 区虽是预留位，但每个
# 都是真实 token（有 ID、有 embedding、进 loss、有 KV），只是语义未定义；
# typst 段是结构 token，同样不可显示、进 loss、有 KV，段类型挂 TYPE_NONE。
HEAD_SIZE = (CORE_SIZE + BOUNDARY_SIZE + SYSTEM_SIZE + SLOT_SIZE
             + TYPST_SIZE + MODAL_SIZE)       # base N=0 → 9845
assert HEAD_SIZE == (SPECIAL_SIZE + SYSTEM_SIZE + AGENT_SIZE
                     + TYPST_SIZE + MODAL_SIZE)
assert HEAD_SIZE == head_layout(SLOT_COUNT)["head_size"]

# 词表版本（2026-11-05 base1.0.0 → 11-06 base1.1.0 → 11-06 base1.2.0
#           → 11-07 base1.3.0）：
#   1.1.0：可见区末尾追加 code 代码专用 token 区，旧 ID 全兼容。
#   1.2.0：**契约头重排**（core/boundary/system/slot N/draw），
#         agent 区 53 个预定义工具全部退役；N 交给词表 config（base=0，
#         gen_vocab.py 生成带 N 个槽位的派生词表）。ID 布局整体位移。
#   1.3.0：**契约头新增 typst 结构段**（slot 之后、draw 之前，228 个
#         = 114 个构造 × 开/闭）+ 可见区末尾追加 unicode_ext 区
#         （3537 个字符，31 个 Unicode 区块）。draw 与全部可见区 ID
#         整体 +228，旧 .tokens / 旧权重 / 旧存档一律不可读，不做兼容。
#         映射器（token 流 → .typ）留作后续独立实现；本次只交付词表 +
#         机器/人可读的对应表格（service/tokens/typst_map.py 生成）。
BASE_VOCAB_NAME = "base1.3.0"
VOCAB_VERSION = "1.3.0"
TOKEN_ENCODING = "token-6"

# 上传文件的扩展名 → 转换器种类（/api/agent/spec 下发，前端不抄常量）。
# 取值与 server_codecs.py 的 /api/codec/* 一一对应：
#   text → /api/codec/text（离散 token，可直接拼进 tokenIds）
#   image/audio/video → 连续投影，走 /api/infer 的 media 通道
#   draw → /api/codec/draw 的图元 CSV/tokens（离散 token）
AGENT_FILE_KINDS = {
    ".txt": "text", ".md": "text", ".json": "text", ".log": "text",
    ".csv": "text", ".py": "text", ".js": "text", ".html": "text",
    ".css": "text", ".xml": "text", ".yaml": "text", ".yml": "text",
    ".vfc": "text", ".gts": "text", ".tokens": "text",
    ".png": "image", ".jpg": "image", ".jpeg": "image",
    ".bmp": "image", ".webp": "image", ".gif": "image",
    ".wav": "audio", ".mp3": "audio", ".flac": "audio",
    ".ogg": "audio", ".m4a": "audio", ".aac": "audio",
    ".mp4": "video", ".mov": "video", ".webm": "video",
    ".mkv": "video", ".avi": "video",
    ".prim": "draw", ".prim.csv": "draw",
}
AGENT_FILE_LABELS = {
    "text": "文本", "image": "图片", "audio": "音频",
    "video": "视频", "draw": "图元",
}
AGENT_MAX_FILE_BYTES = 64 * 1024 * 1024

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
# 2026-11-06（base1.1.0）：末尾追加 "code" 区（跨语言代码语法/标识符/
# 运算符 token）。**只追加、不插队**——可见区既有子区的 ID 区间保持不动，
# 老 ID 全部兼容；code 区文本与已收录文本撞名的由 build_vocab 去重跳过。
# 2026-11-07（base1.3.0）：再在 code 之后追加 "unicode_ext" 区——31 个
# Unicode 区块里 base1.2.0 未逻辑覆盖的 3537 个已分配字符（拉丁扩展、
# 西里尔、希腊、假名、数学/字母符号等）。同样只追加、不插队：可见区
# 既有子区 ID 一个不动（但契约头 +228 会让整段可见区平移）。
VISIBLE_REGION_ORDER = ("digit", "whitespace", "symbol", "emoji",
                        "cjk", "english", "extra", "fallback", "code",
                        "unicode_ext")

# ---------------------------------------------------------------------------
# 1c-2. 子区添加策略（2026-11-07，唯一真源）
# ---------------------------------------------------------------------------
# 词表页「增删 token」表单不再让用户选 regular / special 两档形态，而是选
# **加到哪个子区**（全部子区都列出）。每个子区的 token 文本有强制格式：
#   locked  架构契约头（core / boundary / system / typst / draw）——
#           名字与顺序由本文件冻结，插一条会让后面所有 ID 位移，旧
#           .tokens / 权重 / 存档全部报废，故只展示、禁用增删。
#   angled  slot——用户自定义预留工具名，必须写成 <名字>；名字任意、
#           只要不与全表已有 token 重名（编号 <sN> 只是其中一种写法）。
#   text    可见文本子区——文本无格式要求，非空且不重名即可。
# 前端一律通过 /api/vocab/regions 拿 regions[].add_policy 与
# add_policy_hints，禁止在 JS 里抄子区名或规则。
HEAD_REGION_ORDER = ("core", "boundary", "system", "slot", "typst", "draw")
REGION_ORDER = HEAD_REGION_ORDER + VISIBLE_REGION_ORDER
REGION_LOCKED = frozenset(r for r in HEAD_REGION_ORDER if r != "slot")
POLICY_LOCKED = "locked"
POLICY_ANGLED = "angled"
POLICY_TEXT = "text"
REGION_ADD_POLICY = {
    **{r: POLICY_LOCKED for r in REGION_LOCKED},
    "slot": POLICY_ANGLED,
    **{r: POLICY_TEXT for r in VISIBLE_REGION_ORDER},
}
ADD_POLICY_HINTS = {
    POLICY_LOCKED: "架构契约头，不可增删（名字与顺序由 spec 冻结）",
    POLICY_ANGLED: "必须用 <名字> 包裹；名字自定义，全表不得重名",
    POLICY_TEXT: "文本无格式要求（非空、不与现有 token 重名即可）",
}


def check_add_region(region: str) -> str:
    """校验「添加到哪个子区」→ 返回策略；锁定 / 未知子区抛 ValueError。"""
    if region not in REGION_ADD_POLICY:
        raise ValueError(
            f"未知子区: {region!r}（合法：{', '.join(REGION_ORDER)}）")
    policy = REGION_ADD_POLICY[region]
    if policy == POLICY_LOCKED:
        raise ValueError(
            f"子区 {region!r} 属架构契约头，不可修改：{ADD_POLICY_HINTS[policy]}")
    return policy


# 各类字符来源（build_vocab.py 生成基础词表时取用；**不是长度上限**）
WHITESPACE_CHARS = [" ", "\t", "\n", "\r", "\u00a0", "\u3000"]
FALLBACK_LETTERS = "abcdefghijklmnopqrstuvwxyz"
# 子区 → 大类的归属（category_of 用）
REGION_CATEGORY = {
    "digit": "regular", "whitespace": "regular", "symbol": "regular",
    "emoji": "regular", "cjk": "regular", "english": "regular",
    "extra": "regular", "fallback": "fallback", "code": "regular",
    "unicode_ext": "regular",
}

# 2026-10-24：MIDI 输出 token 暂时删除（steve：研究后再定），midi_codec
# 与 /api/codec/midi 一并移除；如恢复从 git 历史取回。


# 2026-10-24：MIDI 输出 token 暂时删除（steve：研究后再定），midi_codec
# 与 /api/codec/midi 一并移除；如恢复从 git 历史取回。

# 边界 token → 类型嵌入 id（「有 token 无 embedding 投影」的边界 token
# 本身不查词表投影，但其界定的区间内容会加类型嵌入）
TYPE_USER = 0          # [user]..[user/] 之间：用户输入
TYPE_MODEL = 1         # [model]..[model/] 之间：模型输出
TYPE_NONE = 2          # 边界之外 / 边界 token 自身
TYPE_SYS = 3           # [sys]..[sys/] 之间：系统提示 / 工具结果（2026-11-05）
NUM_TYPES = 4

# 段类型归类（2026-11-06，11-07 补 typst）：给定 token id → type_emb。
#   core 区（caps/stop/draw:end）单开一类 → TYPE_NONE：它们不属于任何段，
#     只是机制单标记；但仍进 loss、有 KV（type_emb 与是否进 loss 无关）。
#   typst 结构 token（base1.3.0）→ TYPE_NONE：与 core 同类的 <> tools，
#     不属于任何对话段，段类型由外层上下文决定；仍进 loss、有 KV。
#   slot 区编号槽位 → TYPE_SYS（与系统段同型），语义由使用者自定义。
#   boundary 六个段边界各归其类；普通文本/媒体按上下文默认值处理。
def segment_type_of(tid: int) -> int:
    if not isinstance(tid, int) or tid < 0:
        return TYPE_NONE
    if CORE_BEGIN <= tid < CORE_END:
        return TYPE_NONE
    if TYPST_BEGIN <= tid < TYPST_END:
        return TYPE_NONE
    if tid in (USER_ID, USER_END_ID):
        return TYPE_USER
    if tid in (MODEL_ID, MODEL_END_ID):
        return TYPE_MODEL
    if tid in (SYS_ID, SYS_END_ID):
        return TYPE_SYS
    if SLOT_BEGIN <= tid < SLOT_END:
        return TYPE_SYS
    return TYPE_NONE

# ---------------------------------------------------------------------------
# 2. 模型维度
# ---------------------------------------------------------------------------
D_MODEL = 1024
HEAD_DIM = 128
N_HEADS = D_MODEL // HEAD_DIM                      # 8
# 建模可选宽度（2026-11-04）：默认档位 + 合法范围，唯一真源在 check_d_model。
# 无上限（steve 2026-11-05 拍板：完全无上限，只约束 ≥ D_MODEL_MIN 且为 128 倍数）。
D_MODEL_MIN = 128
D_MODEL_CHOICES = (256, 512, 768, 1024, 1536, 2048)   # 仅示例档位，无上限

# 精度名 → 每参数比特数（唯一真源；precision.py 与建模参数量预估共用）。
# 低比特档在 CPU 上按「权重量化 → 反量化回 fp32」模拟，落盘仍是 fp32；
# 这里的 bits 只用于「理论占用」展示，不等于 safetensors 文件大小。
DTYPE_BITS = {
    "fp64": 64, "fp32": 32, "fp16": 16, "bf16": 16,
    "fp8": 8, "fp4": 4,
    "int16": 16, "int8": 8, "int4": 4, "int2": 2,
}


def check_d_model(v) -> int:
    """建模时指定的 d_model 严格校验（唯一规则，所有入口共用）。

    合法返回 int；非法抛 ValueError（HTTP 层映射 400）。拒绝 bool、
    非整数 float、非数字字符串、过小值（< D_MODEL_MIN）与非 HEAD_DIM 倍数值
    ——任何一项松了都会造出 config 与权重形状对不上的模型目录。"""
    if isinstance(v, bool):
        raise ValueError(f"d_model 必须是整数，不能是布尔值: {v!r}")
    if isinstance(v, int):
        n = v
    elif isinstance(v, float):
        if not v.is_integer():
            raise ValueError(f"d_model 必须是整数: {v!r}")
        n = int(v)
    elif isinstance(v, str):
        s = v.strip()
        if not s.isdigit():
            raise ValueError(f"d_model 必须是整数: {v!r}")
        n = int(s)
    else:
        raise ValueError(f"d_model 必须是整数: {v!r}")
    if n < D_MODEL_MIN:
        raise ValueError(f"d_model 必须 ≥ {D_MODEL_MIN}: {n}")
    if n % HEAD_DIM:
        raise ValueError(f"d_model 必须是 {HEAD_DIM} 的整数倍: {n}")
    return n


N_ENC_BLOCKS = 4
N_DEC_BLOCKS = 6

# ---- enc/dec 权重模式（2026-11-04，见文件头定案记录）----
# 默认分离：旧模型目录 / 所有未显式指定该选项的入口都保持 v1 行为。
SHARE_ENC_DEC = False
# 共享模式下、两侧 block 数都没给时的共同默认值。不能让 Gargantua 各自
# 取 N_ENC_BLOCKS=4 / N_DEC_BLOCKS=6——那两个数不等，共享模型必须等深。
SHARED_DEFAULT_BLOCKS = N_ENC_BLOCKS
# 共享范围（state_dict 前缀对）：只绑定两个 block 栈与两侧 final norm。
# wte / lm_head 按 spec 不绑定；type_emb / 模态投影 / fast weight 本来就
# 是全局单份，不在此列。
TIED_KEY_PAIRS = (("enc_blocks.", "dec_blocks."),
                  ("enc_final_norm.", "dec_final_norm."))

# block 内顺序（2026-11-04 重排）：每个 KDA/MLA 之后都紧跟一个 FFN。
# "attnres" 不是独立计算子层，而是块末的聚合/零初始化门；其余 10 项各
# 对应一份 RMSNorm（Block.__init__ 从这张表推导 norms 数量，并断言
# forward 的顺序与之严格一致，防止两处脱节）。
BLOCK_ORDER = ["kda", "ffn", "kda", "ffn", "kda", "ffn",
               "mla_csa", "ffn", "mla_hca", "ffn", "attnres"]
N_FFN_PER_BLOCK = sum(1 for s in BLOCK_ORDER if s == "ffn")   # 5
N_SUBL_LAYERS = len(BLOCK_ORDER) - 1                          # 10

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

# FFN（2026-11-04：每 block 5 个独立实例，GLU 结构：SiLU 门控）
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

# 滑动窗口：以整个时间步为原子溢出即弃。主干 build_views 只丢整步 KV
# 向量、**不归档 token id**；「archive 全量存档」是历史 SharedKV 类（主干
# 从不实例化）里的机制，别当成生效契约。

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
#
# 可视化（2026-11-04，/api/infer/stats）：状态 S 是 [H, dk, dv] 的实矩阵，
# 前端不可能画 8×128×128 个点——降采样成 FW_VIZ_BINS² 的平均幅值热力图，
# 写入幅度按步保留最近 FW_WRITE_HISTORY 个点。纯展示，不参与任何计算。
#
# 2026-11-05：24² → 48²（线性分辨率翻倍、像素数 4 倍），开发者选项里那张
# 热力图不再是一块糊掉的马赛克。代价只是流式每帧多带 ~1.7K 个已消毒浮点
# （服务端仍只把最新一份快照存进 buf["fw"]，token 里的 fw 恒为 None），
# 不参与任何计算、不改 checkpoint 布局，也不影响训练/推理数值。
FW_VIZ_BINS = 48
FW_WRITE_HISTORY = 96

# ---------------------------------------------------------------------------
# 4b. MoMW —— Mixture-of-Memory-Weights（无训练记忆阀门，2026-11-08）
# ---------------------------------------------------------------------------
# 位置：第一个 Block 之后、与既有 fast weight 并列的一条可选记忆支路。
#       H_out = H_1 + ΔH；无层被激活 → 直接返回 H_1（退化为标准模型）。
#       2026-11-08 第二版：三阶段路由（锚点播种 → 状态特征漂移 → 空层分裂）。
#
# 阶段一 · 冷启动（锚点播种）
#   每个 MW 层绑定一个类别锚点 P_k ∈ R^D（外部原型）。默认由 wte.weight
#   中确定性采样的「代表性 token Embedding 均值」得到，可经
#   MoMW.set_anchor_provider() 注入真正的文本→embedding 均值实现。
#   **惰性在首次 forward 计算**（建模期 wte 还是随机初始化，此时算锚点
#   没有意义），缓存为 persistent=False buffer：随 .to() 迁移、不进
#   state_dict（strict load 契约）。H_t = normalize(mean_{B,T} H_1)，
#   Score = cos(H_t, P_k)。用序列均值而非逐 token：池状态是按整个输入
#   序列推进的，逐 token 路由会让同一序列写进不同层，破坏「未召回则不遗忘」。
#
# 阶段二 · 运行期（从锚点向状态特征漂移）
#   每 eigen_interval 次写入做**一步幂迭代**（不是全量 SVD）：对
#   G_h = S_hᵀ S_h 迭代主特征向量 v_h，拼成 [H*dv] 后过 KDA 的 w_o
#   投影回 D 维得到 E_i —— 该层「实际在记什么」的隐空间方向。
#   γ_i = 1 - exp(-writes_i / gamma_k) 随写入次数单调递增，
#   Score_i = (1-γ_i)·cos(H_t,P_i) + γ_i·cos(H_t,E_i)。
#
# 阶段三 · 扩展期（空层分裂 + 负载均衡）
#   ‖S_i‖ 单步增长 > expand_delta 或 绝对范数 > saturation_limit → 扩展。
#   **绝不 clone 原层状态**：S_new = None（全零）、E/v/writes/γ 全部从零开始。
#   P_new 继承父层当前 E_i（已积累语义）或初始锚点。
#   bonus = B0·exp(-age/M)（指数衰减，仅前 M 步），强制引导输入写入新层；
#   同类层之间按「当前激活范数低于同类均值」加软均衡偏置 load_lambda，
#   把写入导向余量更大的层，而不是硬分配。
#
# 与 v1 的兼容：v1 的 10 个 config 字段全部保留；新增字段缺省即默认。
# v1 的阈值是「能量范数」量级，v2 改余弦相似度（[-1,1]）后默认值整体
# 重标（threshold 1.0 → 0.2、B0 2.0 → 0.6），否则 v2 会永远不激活。
# B0 > threshold 仍是硬约束（否则零状态层永远跨不过阈值）。
#
# 【阈值仍是保守占位，必须在真实权重上标定】余弦打分的分布取决于训练后
# 的 wte / KDA 权重，未训练模型上推不出合适绝对值。启用前用 MoMW.stats()
# 观察分数分布再调，否则模块会长期退化为恒等映射。
MOMW_CONFIG_KEY = "momw"          # config.json / /api/model/init 的键名
MOMW_ENABLED_DEFAULT = False      # 默认关闭：旧目录零影响、训练路径不变
MOMW_TOP_N = 2                    # N：每次激活的记忆层数
MOMW_TAU = 1.0                    # tau：路由 softmax 温度
MOMW_THRESHOLD = 0.2              # v2：余弦相似度阈值（不是 v1 的能量范数）
MOMW_WINDOW = 8                   # M：bonus 指数衰减时间常数 / 生效步数
MOMW_BONUS = 0.6                  # B_0：新层 bonus 幅值（须 > threshold）
MOMW_STARVE_THRESHOLD = 0.0       # v1 兼容：饥饿窗口（0 = 不启用该触发）
MOMW_SATURATION_LIMIT = 100.0     # ‖S‖ 绝对范数上限，超过即判过载
MOMW_INITIAL_LAYERS = 1           # 初始池层数（每次只扩 1 层）
MOMW_MAX_LAYERS = 8               # 硬上限：防无训练扩展把池撑爆

# v2 三阶段新增
MOMW_EXPAND_DELTA = 8.0           # ‖S‖ 单步增长超过它 → 判定写入密度过载
MOMW_GAMMA_K = 32.0               # γ = 1 - exp(-writes / k) 的收敛尺度
MOMW_EIGEN_INTERVAL = 16          # 每 N 次写入做一步幂迭代（不做全量 SVD）
MOMW_LOAD_LAMBDA = 0.05           # 同类层软均衡强度
MOMW_ANCHOR_SAMPLE = 32           # 每类锚点采样的代表性 token 行数
MOMW_EIGEN_MIN_WRITES = 4         # E_i 至少积累几次写入才可被继承为 P_new

# ---- v3（2026-11-08 第三版：数据驱动阈值 / Oja / 持久化 / 门控 / 异步写）----
# 阈值分位数：规格给的「p=10%」是「低尾定义边界」口径（≈90% 准入）。
# MoMW 的语义是「有相关记忆才介入」，所以取保守侧：p=0.9 ⇒ 只让历史
# 分数最高的约 10% 的输入跨过阈值。p 是**分位位置，不是激活率**。
MOMW_THRESHOLD_MODE = "fixed"     # "fixed" | "quantile"
MOMW_QUANTILE_P = 0.9             # 历史分数分位位置（1-p ≈ 准入比例）
MOMW_QUANTILE_WINDOW = 64         # 打分历史窗口 W（步）
# 相对量替代绝对值：‖S‖ > ratio × 同类层范数中位数 → 判过载。
# 绝对 saturation_limit / expand_delta 降级为兜底（0 = 不启用绝对判定）。
MOMW_SATURATION_RATIO = 1.5
MOMW_GROWTH_RATIO = 0.5
MOMW_OJA_ETA = 0.02               # Oja 在线主方向跟踪学习率
MOMW_FW_GATE_C = 2.0              # cost-aware 门控强度（0 = 关闭）
MOMW_ASYNC_WRITE = False          # 单写者队列异步写（默认同步）
MOMW_PERSIST = True               # 启用时把池快照落 models/NAME/momw_state/
# 快照落盘精度：fp16 让 8 层池的 tensors.safetensors 从 ~4MB 降到 ~2MB，
# 对长期记忆方向几乎无损（锚点/E/状态本来就是 no_grad 的近似跟踪量）。
# 计算时仍按模型自身 dtype 恢复，落盘精度只影响磁盘与重启后的舍入误差。
MOMW_PERSIST_DTYPE = "fp32"
MOMW_PERSIST_DTYPES = ("fp32", "fp16", "bf16")
MOMW_PERSIST_DTYPE_MAP = {"fp32": "float32", "fp16": "float16",
                          "bf16": "bfloat16"}
MOMW_QUANTILE_MIN_SAMPLES = 8     # 历史不足时退回固定 threshold
MOMW_QUANTILE_RANGE = (0.01, 0.99)
MOMW_QUANTILE_WINDOW_RANGE = (4, 4096)
MOMW_RATIO_MIN = 1.0              # saturation_ratio > 1 才有意义
MOMW_OJA_ETA_RANGE = (1e-4, 1.0)
MOMW_FW_GATE_C_RANGE = (0.0, 50.0)

# 锚点类别（MW 层类型）。同类层之间才做软均衡；类型名会下发前端展示。
MOMW_ANCHOR_GROUPS = ("general", "error", "code", "math")
MOMW_ANCHOR_SEED = 0x4D4F4D57     # "MOMW"：确定性采样，禁用全局随机状态

# 类别代表性文本（v3）：由调用方用**模型自己的词表副本**编码成 token id，
# 再取 wte 均值作为锚点。用模型词表而不是全局词表，就绕开了「版本升级时
# ID 整体位移」的问题——锚点不依赖任何硬编码 ID。编码失败时逐级降级：
# 文本 → 代表性 token 采样 → 确定性随机单位向量。
MOMW_ANCHOR_TEXTS = {
    "general": ("hello", "the quick brown fox", "this is a note"),
    "error": ("Traceback (most recent call last)", "SyntaxError",
              "failed to parse the input"),
    "code": ("def main():", "return None", "for i in range(10)"),
    "math": ("1 + 2 = 3", "0.5 * 4", "the sum is"),
}

# 合法范围（HTTP / CLI 校验用；非法值一律拒绝而非静默钳制）
MOMW_TOP_N_RANGE = (1, 16)
MOMW_WINDOW_RANGE = (1, 256)
MOMW_MAX_LAYERS_RANGE = (1, 64)
MOMW_INITIAL_LAYERS_RANGE = (1, 16)
MOMW_TAU_MIN = 1e-3               # tau > 0，太小会让 softmax 退化成 argmax
MOMW_POSITIVE_MIN = 0.0           # threshold / B0 / starve / 范数阈值 ≥ 0
MOMW_GAMMA_K_MIN = 1.0
MOMW_EIGEN_INTERVAL_RANGE = (1, 1024)
MOMW_ANCHOR_SAMPLE_RANGE = (1, 512)
MOMW_EIGEN_MIN_WRITES_RANGE = (0, 1024)
MOMW_LOAD_LAMBDA_MAX = 1.0        # 软均衡不能盖过语义打分

# config 里允许的字段（唯一真源；前端从 /api/status.momw 拿，不许抄常量）
# v1 的 10 个字段全部保留 → 上一版创建的 config.json 不做迁移也能加载。
MOMW_CONFIG_FIELDS = ("enabled", "N", "tau", "threshold", "M", "B0",
                      "starve_threshold", "saturation_limit",
                      "initial_layers", "max_layers", "expand_delta",
                      "gamma_k", "eigen_interval", "load_lambda",
                      "anchor_sample", "eigen_min_writes",
                      "threshold_mode", "quantile_p", "quantile_window",
                      "saturation_ratio", "growth_ratio", "oja_eta",
                      "fw_gate_c", "async_write", "persist",
                      "persist_dtype")
MOMW_THRESHOLD_MODES = ("fixed", "quantile")


def momw_default_config() -> dict:
    """MoMW 默认配置（唯一真源）。enabled=False 时其余字段只是备用值。

    前端 / CLI 都从这份 dict 派生，避免把十六个数字抄到三个地方。
    """
    return {
        "enabled": MOMW_ENABLED_DEFAULT,
        "N": MOMW_TOP_N,
        "tau": MOMW_TAU,
        "threshold": MOMW_THRESHOLD,
        "M": MOMW_WINDOW,
        "B0": MOMW_BONUS,
        "starve_threshold": MOMW_STARVE_THRESHOLD,
        "saturation_limit": MOMW_SATURATION_LIMIT,
        "initial_layers": MOMW_INITIAL_LAYERS,
        "max_layers": MOMW_MAX_LAYERS,
        "expand_delta": MOMW_EXPAND_DELTA,
        "gamma_k": MOMW_GAMMA_K,
        "eigen_interval": MOMW_EIGEN_INTERVAL,
        "load_lambda": MOMW_LOAD_LAMBDA,
        "anchor_sample": MOMW_ANCHOR_SAMPLE,
        "eigen_min_writes": MOMW_EIGEN_MIN_WRITES,
        "threshold_mode": MOMW_THRESHOLD_MODE,
        "quantile_p": MOMW_QUANTILE_P,
        "quantile_window": MOMW_QUANTILE_WINDOW,
        "saturation_ratio": MOMW_SATURATION_RATIO,
        "growth_ratio": MOMW_GROWTH_RATIO,
        "oja_eta": MOMW_OJA_ETA,
        "fw_gate_c": MOMW_FW_GATE_C,
        "async_write": MOMW_ASYNC_WRITE,
        "persist": MOMW_PERSIST,
        "persist_dtype": MOMW_PERSIST_DTYPE,
    }

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
VISUAL_COORD_BINS = 256      # patch 坐标嵌入表大小（x / y 各一张）

# 音频输入（仅输入）
AUDIO_SR = 16000             # 16kHz 单声道
AUDIO_SAMPLES_PER_TOKEN = 533        # 533 采样/token ≈ 30 tok/s
# 输入 embedding：533 个采样点平铺成一条向量 → 线性投影 → d_model

# ---------------------------------------------------------------------------
# 5b. token-6 编码（2026-11-05 定案，取代 token-5 的 .tokens 二进制）
# ---------------------------------------------------------------------------
# 动机：token-5 只有 32 个码位，27 单字符 + 4 边界 + 1 终止全占满，
# 装不下新的一代边界（sys）与工具调用语法。token-6 每字符 6 bit
# （2^6 = 64 码位），载荷语法与 token-5 完全一样，只是位宽和字符集变了：
#   {t:<id>}               文本 token（可见、有向量、进 loss）
#   {w:<hex>}              音频 token（533 采样）
#   {v:<hex>x:<gx>y:<gy>}  视觉 patch token（32×32 像素）
#   [HEX] / [name]         扩展系统 token（有嵌入、不进 loss、不可显示）
#   <HEX> / <name>         core 单标记 / slot 编号槽位（有 KV、进 loss）
#   [HEX] / [name]         扩展 / boundary 段边界 token（有嵌入、进主 AR loss）
#   { } |                  结构字符（不是独立 token，不占位置、不进 loss）
#
# —— 字符集：29 个单字符（码 0-28）+ 6 个原子边界 token（码 29-34）——
#   6 bit/字符。码 35-62 预留，码 63 为流终止符（不对应任何字符）。
#
#   —— 29 个单字符（T6_CHARS，码即下表顺序 0..28）——
#   '0'-'9'  数字（坐标十进制 / ID 十六进制数字位；ID 另用 'A'-'F'）
#   '{' '}'  时间步边界
#   '<' '>'  **core 单标记 / slot 编号槽位**边界（token-5 里是扩展系统 token）
#   '[' ']'  **边界 token** 包裹（新加入的两个字符）
#   '|'      步内 token 分隔 / patch 分隔
#   ':'      字段分隔
#   'w' 't' 'v'   音频 / 文本 / 视觉载荷标识
#   'A'-'F'  十六进制字母（大写；小写不属于字符集）
#   'x' 'y'  视觉坐标标识
#   没有空格，没有其它字母（w/t/v/x/y 是仅有的小写，只作载荷/坐标标识）。
#
#   —— 6 个原子边界 token（码 29-34，各占一个 6-bit 码位）——
#   编码时把整串 [model]/[user]/[model/]/[user/]/[sys]/[sys/] 当作一个
#   码位；解码时码 29-34 还原成展示名。它们绑定到 boundary 区的 ID
#   （1.2.0：ID 3..8，由 T6_NAME_TO_BOUND_ID 从 USER_ID / MODEL_ID /
#   SYS_ID 等常量派生，不硬编码），所以既是位流里的原子字符，又是元素层
#   带 ID 的 ext 元素。
T6_CHARS = "0123456789{}<>|:wtvABCDEFxy[]"
T6_ALPHABET_SIZE = len(T6_CHARS)          # 29（单字符码 0..28）
T6_BITS_PER_CHAR = 6
T6_END_CODE = 63                          # 2^6-1，解码到此为止
T6_CHAR_CODE = {c: i for i, c in enumerate(T6_CHARS)}
T6_BOUND_TOKENS = {
    29: "[model]", 30: "[user]", 31: "[model/]", 32: "[user/]",
    33: "[sys]", 34: "[sys/]",
}
T6_BOUND_NAME_TO_CODE = {v: k for k, v in T6_BOUND_TOKENS.items()}
T6_NAME_TO_BOUND_ID = {
    "[model]": MODEL_ID, "[user]": USER_ID,
    "[model/]": MODEL_END_ID, "[user/]": USER_END_ID,
    "[sys]": SYS_ID, "[sys/]": SYS_END_ID,
}
T6_BOUND_ID_TO_NAME = {v: k for k, v in T6_NAME_TO_BOUND_ID.items()}
T6_HEX_CHARS = "0123456789ABCDEF"
# 码 35-62 预留（28 个），留给后续模态/结构化 token；不得挪作终止符。
T6_RESERVED_CODES = tuple(range(len(T6_BOUND_TOKENS) + T6_ALPHABET_SIZE,
                                T6_END_CODE))

# —— 载荷尺寸 ——
# 文本/扩展 token 的 ID：**大写十六进制、自界定**——{t:...} 里读到 '}' 或
# '|' 为止，<...> 里读到 '>' 为止。长度因词表而异（小词表 ID 短、大词表
# ID 长），格式本身不给 ID 定宽、定上限；前导零允许（<00005> 就是 5）。
# 十六进制必须大写：token-6 的 29 字符集里没有小写 a-f。
# 坐标（{v:...x:N y:M} 的 N/M）仍是自界定十进制。
T6_COORD_LIMIT = 100_000        # x/y 坐标防呆上界（自界定十进制）

# 音频：每 token 恰好 533 个采样点，16 bit 有符号 PCM（mp3/flac 解码后的
# 通用形态），按时间顺序拼接 → 533*16 = 8528 bit = 2132 个十六进制字符。
AUDIO_SAMPLE_BITS = 16
T6_WAVE_BITS = AUDIO_SAMPLES_PER_TOKEN * AUDIO_SAMPLE_BITS   # 8528
T6_WAVE_HEX = T6_WAVE_BITS // 4                              # 2132

# 视觉：一个 patch 32×32 像素，逐行（第 1 行→第 32 行）、行内从左到右，
# 每个像素 R/G/B/A 各 10 bit 共 40 bit，按 R→G→B→A 拼接。
# 1024 * 40 = 40960 bit = 10240 个十六进制字符。
T6_PIXEL_BITS = 4 * 10                    # RGBA 各 10 bit
T6_PATCH_BITS = PATCH_PIXELS * T6_PIXEL_BITS    # 40960
T6_PATCH_HEX = T6_PATCH_BITS // 4               # 10240
T6_CHANNEL_MAX = (1 << 10) - 1            # 1023

# —— 语义 ——
# {t:} {w:} {v:} 元素：可见 token，有向量坐标（走 embedding / 投影），进 loss。
# [id] / [name] 元素：扩展系统 token，有嵌入但【没有】位置坐标，不进 loss。
# <id> / <name> 元素：agent 区工具调用与排版 token，模型可预测、有 KV、
#   进 loss，UI 按名字渲染（粗体/思考段/工具徽标），后端按 ID 拦截执行。
# { } | 以及 [ ] < > 本身：结构字符，不是 token（不占 token 位置、不进 loss）。
# 时间步仍是原子单位：{} 必须至少含一个元素，空步非法。
# 落盘（.tokens）：6 bit/字符的位流，末字符后跟终止符码 63，再补零到字节。

# 旧名别名：位宽升级后仍有一些模块/测试按 T5_ 前缀取**载荷尺寸**（那些
# 常量与 bit 宽无关）。别名一律指向 token-6 的真值，避免两份常量分叉；
# token-5 的 5 bit 位流不再可读（本次不做向后兼容）。
T5_HEX_CHARS = T6_HEX_CHARS
T5_COORD_LIMIT = T6_COORD_LIMIT
T5_WAVE_BITS = T6_WAVE_BITS
T5_WAVE_HEX = T6_WAVE_HEX
T5_PIXEL_BITS = T6_PIXEL_BITS
T5_PATCH_BITS = T6_PATCH_BITS
T5_PATCH_HEX = T6_PATCH_HEX
T5_CHANNEL_MAX = T6_CHANNEL_MAX

# 绘图输出（仅输出，token 区间见词表布局）
# 2026-10-31：画布不再钉死 256×256——序列开头的画布 token 决定尺寸，
# 上限 1920×1080。这两个常量是**默认画布**（旧数据 / 未指定时用它）。
CANVAS_W = DRAW_DEFAULT_CANVAS[0]
CANVAS_H = DRAW_DEFAULT_CANVAS[1]
DRAW_KV_COMPRESS = 9            # 输出完一个图元后 KV 就地 9:1 压成 1 向量

# 音频输出：MIDI → 离散 token 暂时删除（2026-10-24，steve：研究后再定）

# ---------------------------------------------------------------------------
# 6. 优化器
# ---------------------------------------------------------------------------
# 三档（可选值真源 = clip_data.OPTIMIZERS，训练界面下拉与队列文件名共用）：
#   "adamw"   全部 AdamW；
#   "muon"    参杂：隐藏层 2D 矩阵 Muon，其余参数 AdamW（历史默认）；
#   "allmuon" 全 Muon：所有 2D 矩阵 Muon，1D 参数仍 AdamW（矩阵正交化
#             对向量无意义，见 muon.build_optimizers）。
OPTIMIZER = "muon"
MUON_MOMENTUM = 0.95
MUON_NS_STEPS = 5
ADAMW_LR = 3e-4
MUON_LR = 2e-2

# ---------------------------------------------------------------------------
# 6a. 阶段 2 Manifold Muon（可选，默认关闭）——与 MoMW 同款 config 契约
# ---------------------------------------------------------------------------
# config.json 有 "manifold_muon" 字段才启用；缺字段 / 坏值一律按关闭处理。
# 不新增任何 nn.Parameter / buffer / state_dict key——配置只是普通 Python
# 属性，旧模型目录 strict load 照常可读。键名是唯一真源，别在别处抄字符串。
MANIFOLD_CONFIG_KEY = "manifold_muon"
MANIFOLD_ENABLED_DEFAULT = False
MANIFOLD_K = 20
MANIFOLD_ALPHA = 0.1
MANIFOLD_EXACT_MSIGN = False
MANIFOLD_EXCLUDE_KDA = True
MANIFOLD_MAX_MATRICES = 0


def manifold_default_config() -> dict:
    """Manifold Muon 默认配置（唯一真源）。enabled=False 时其余只是备用值。

    前端 / CLI / muon-dev 都从这份 dict 派生，避免把数字抄到三处。
    lr / momentum / ns_steps 复用优化器常量，K / alpha / exclude_kda /
    max_matrices / exact_msign 复用上面的新常量；键名与值由测试逐字依赖。
    """
    return {
        "enabled": MANIFOLD_ENABLED_DEFAULT,
        "attention": "stiefel",        # stiefel | dgram | none
        "mlp": "dgram",                # stiefel | dgram | none
        "K": MANIFOLD_K,
        "alpha": MANIFOLD_ALPHA,
        "lr": MUON_LR,
        "momentum": MUON_MOMENTUM,
        "ns_steps": MUON_NS_STEPS,
        "exclude_kda": MANIFOLD_EXCLUDE_KDA,
        "max_matrices": MANIFOLD_MAX_MATRICES,
        "exact_msign": MANIFOLD_EXACT_MSIGN,
    }


# ---------------------------------------------------------------------------
# 6b. CLIP 数据集文件夹 v1（2026-11-08）
# ---------------------------------------------------------------------------
# 唯一真源：契约见 service/train/clip_folder.py，常量放这里避免三处硬编码。
# 对比训练不再「一次传两个文件」，而是强制上传一个按此格式组织的文件夹；
# 解析出的 N 对真实配对同批做对称 InfoNCE，负样本来自批内其它配对。
CLIP_FOLDER_FORMAT = "gargantua-clip-folder"
CLIP_FOLDER_VERSION = 1
CLIP_FOLDER_MIN_PAIRS = 2          # InfoNCE 至少 2 对才有批内负样本
CLIP_FOLDER_MAX_BATCH = 64         # 与 server_train._validate_clip_batch_size 同界
CLIP_FOLDER_DEFAULT_BATCH = 16
CLIP_FOLDER_TEMPERATURE = 0.07

# ---------------------------------------------------------------------------
# 6c. SOMN（v4-somn，可选，默认关闭）——2026-11-14 主干接线
# ---------------------------------------------------------------------------
# config.json 的 "somn" 是唯一开关：enabled=true 时 common.build_model 把
# 配置传给 Gargantua.__init__，由它显式 attach_somn()。缺字段 / 坏值一律
# 按关闭处理；管理器只落在普通 dict 里，不新增 state_dict key。
SOMN_CONFIG_KEY = "somn"
SOMN_ENABLED_DEFAULT = False
# 允许透传给 somn.attach_somn(**kwargs) 的调优字段（白名单，唯一真源）。
SOMN_CONFIG_FIELDS = (
    "enabled", "max_write_count", "max_layers", "warmup_steps",
    "warmup_bonus", "persist_across_samples", "eta", "lam",
    "elbow_alpha", "min_score", "min_energy", "explore_every",
    "async_split", "expand_growth_threshold",
)
SOMN_INT_FIELDS = ("max_write_count", "max_layers", "warmup_steps",
                   "explore_every")
SOMN_BOOL_FIELDS = ("persist_across_samples", "async_split")
SOMN_FLOAT_FIELDS = ("warmup_bonus", "eta", "lam", "elbow_alpha",
                     "min_score", "min_energy", "expand_growth_threshold")


def somn_default_config() -> dict:
    """SOMN 默认配置：只带开关，其余调优参数由 server/somn.py 自带默认。"""
    return {"enabled": SOMN_ENABLED_DEFAULT}

# ---------------------------------------------------------------------------
# 7. 运行环境
# ---------------------------------------------------------------------------
# Python 3.12 系列最新版（standalone，项目内隔离安装；steve 2026-10-24：
# 用 latest 不锁 patch）+ torch 2.2.2（macOS x86_64 最后官方轮子）。
# 隔离运行时全部装在 environment/，不碰系统 Python：
#   environment/python/  python-build-standalone 解包（setup.sh 自动下载）
#   environment/venv/    以上述解释器建的虚拟环境（must/necessary 依赖）
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
#     requirements/ 分类依赖：must（启动必备，轻量）/ necessary（正常
#                   工作需要，启动不查；2026-11-01 起 core/window/dev 并入）
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
#                    "share_enc_dec" 决定是否 enc/dec 逐层共享权重（共享时
#                    只落 enc 侧一份，加载时镜像出 dec 侧 key）；
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
# 时间步结构（token-6 体系，2026-11-05 起，tokens/tok6.py）：
#   流中的每个时间步由字符 { } 包裹，步内多个元素由 | 分隔；元素形态
#   见第 5b 节（t: 文本 / w: 音频 / v: 视觉 / [ ] 扩展 / < > 工具调用）。
#   文本每 token 独占一步；图片 = 一步（内部多 patch 元素）；音频每 533
#   采样 = 一个元素 = 一步；视频每帧 = 一步。KDA 循环状态递归 / 逻辑
#   并行以时间步为原子。结构字符 { } | [ ] < > 本身不是 token。
#
# 训练数据（2026-11-05 起）：.tokens 文件（token-6 位流，6 bit/字符）。
#   内容必须带 { } | 时间步结构；训练侧读取后按 Step 展开成「扁平
#   token_ids + 并行 step_ids」。旧 tok-32 / token-5 位流不再生成，
#   tok32 模块仅作为历史数据的读取/转换工具保留。
#   Token 编辑器导出同一格式，训练与编辑器共用一套流。
#
# 标识（2026-10-25，2026-11-05 补 sys）：段类型用类型嵌入表达——
#   user 段加 type_emb(TYPE_USER)、model 段加 type_emb(TYPE_MODEL)、
#   sys 段（工具结果/系统提示）加 type_emb(TYPE_SYS)，不分段首尾。
#   [user]/[user/]/[model]/[model/]/[sys]/[sys/] 这些边界 token 是
#   扩展系统级标记（token-6 各占一个原子码位），不进共享 KV、不参与 loss。
