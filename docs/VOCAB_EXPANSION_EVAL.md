# 词表扩展评估（base1.2.0 → 扩表）

> **【历史评估记录 · 请勿据此实施】** 本次升级的 **base1.3.0 已按新清单实施**：
> 契约头插入 `typst` 段（228 个 tools）、可见区末尾追加 `unicode_ext`
> **3537 个字符 / 31 个 Unicode 区块**，`vocab_size = 80262`。
> 这与本文评估的 **+14078 字 / 90575** 口径**不同**；本文「评估稿 v2，决策已定，
> 未实施」的结论仅作**历史评估记录**保留，正文不代表当前实现。

> 状态：**评估稿 v2，决策已定，未实施。** 范围/计数已确认，待拍板"是否配套多语种语料"后落地。
> 配套数据：`data/corpus/coverage_audit.json`（缺口审计）、`data/corpus/vocab_expansion_eval.json`（精确计数）。
>
> 2026-11-12 项目整理：本文件由项目根移入 `docs/`（历史稿归档区，见
> [docs/README.md](./README.md)）。现行口径以根 [`README.md`](../README.md)
> 的「base1.3.0 vocabulary」一节为准，本稿不再随代码更新。

## 1. 背景与动机

- GARGANTUA 是**中英文**多模态 encoder-decoder 小模型。base1.2.0 词表刻意做小（76497），结果大量常用字符被静默丢弃（见第 3 节）。
- 决策（steve 定）：**除「过于生僻」的 CJK 扩展 B+（约 65,256 生僻汉字）外，其余缺失的常用字符全部补上**；**去掉 `MAX_VOCAB_SIZE` 上限**；日/俄/韩/阿/希伯来/泰/梵文等所有常用文字体系**一个不落**。
- **IBM Research 与学界共识均表明：扩表能提能**（详见第 6 节），但有两个前提与代价。

## 2. 当前词表状态（base1.2.0）

| 项 | 值 |
|---|---|
| vocab_size | 76497 |
| HEAD_SIZE / DRAW_BEGIN | 9617 / 12 |
| MAX_VOCAB_SIZE（将移除） | 100000 |
| 已有覆盖 | CJK 常用字（BMP 20992 + 扩展A 6592 = 100%）、英文 30000 词、基础拉丁（小写，大写折叠）、数学/箭头/框线/几何/货币符号 ~100%、emoji 充足 |
| 已确认语料内缺字 | 仅 12 个：〇 + 11 个希腊字母（β ε θ α λ δ σ ρ Θ γ Ω） |

> 大写拉丁字母经 `enctok` 大小写折叠走 fallback 小写，**并非真缺**（已实测零误丢）。

## 3. 缺口审计结论（节选，详见 coverage_audit.json）

| 区块 | 覆盖 | 性质 |
|---|---|---|
| Latin-1 补充（欧洲重音） | 24/96 (25%) | **真缺口** |
| Latin 扩展A/B、IPA、组合符 | 0% | 真缺口 |
| Greek 希腊字母 | 6/135 (4.4%) | 真缺口（仅数学用希腊存了几个） |
| Cyrillic 西里尔 | 1/256 (0.4%) | 真缺口 |
| Hiragana / Katakana | 2/93、2/96 | 真缺口 |
| Bopomofo / Hangul | 0% | 真缺口 |
| Hebrew / Arabic / Devanagari / Thai | 7%/9%/2%/5% | 真缺口 |
| CJK 扩展 B+ | 0%（约 65000） | **生僻，排除** |
| CJK 符号与标点 | 36/64 (56%) | 部分（含〇） |
| Halfwidth/Fullwidth | 115/225 (51%) | 部分 |
| Basic Latin | 69/95 (72.6%) | 非真缺口（大写折叠） |

## 4. 已拍板的扩展范围

- **全补（本次实施）**：Latin-1 重音、Latin 扩展A/B、IPA、组合符、Greek、Cyrillic（+补充）、Armenian、Hebrew、Arabic、Devanagari、Thai、General Punctuation、上下标、Letterlike、Number Forms、Enclosed、Dingbats、CJK 符号与标点（含〇）、Hiragana、Katakana、Bopomofo、Hangul Compatibility Jamo、**Hangul Syllables（谚文音节 11172）**、CJK Compatibility Ideographs、Halfwidth/Fullwidth。
- **排除（生僻）**：CJK 统一汉字扩展 B/C/D/E/F/G（约 65,256 个生僻汉字）。
- **去掉上限**：移除 `spec.MAX_VOCAB_SIZE`（token-6 以 hex 存 ID，无位宽约束；该常量仅用于 `tok6._check_id` 上界防呆）。本次扩完约 90575，远低于旧上限，去掉上限仅为未来（如以后真要加 Ext-B+）留余地。

## 5. 精确计数（选定口径 = 全补，scenario A）

| 区块 | 补 |
|---|---:|
| Latin-1 补充 | 72 |
| Latin 扩展-A / -B | 128 / 208 |
| IPA / Spacing Mod / Combining | 96 / 37 / 112 |
| Greek and Coptic | 129 |
| Cyrillic / Cyrillic Supp | 255 / 48 |
| Armenian / Hebrew / Arabic | 80 / 82 / 233 |
| Devanagari / Thai | 125 / 83 |
| General Punct / SuperSub / Letterlike / Number Forms | 39 / 32 / 45 / 58 |
| Enclosed / Dingbats | 82 / 30 |
| CJK Symbols and Punctuation（含〇） | 28 |
| Hiragana / Katakana / Bopomofo | 91 / 94 / 43 |
| Hangul Compat Jamo / **Hangul Syllables** | 94 / **11172** |
| CJK Compatibility / Halfwidth-Fullwidth | 472 / 110 |

**合计增量 +14078 → 扩展后 vocab_size = 90575**（距旧上限 100000 余 9425；上限移除后无约束）。
**排除项**：CJK Ext-B+ 约 65,256（不补）。

## 6. 扩表是否能让模型能力变强？（IBM Research + 学界共识）

**结论：能，但有两个前提；对本项目是"清楚的正收益 + 一个必须正视的坑"。**

### 6.1 证据

- **IBM Research（INDUS / NASA-IBM 合作）**：定制领域词表 + 定制 tokenizer，相较通用 RoBERTa 在 BLURB 生物医学基准 **+5%**、地球科学 NER **+5.5%**、科学 QA **+2.4% F1**。其论文 *Multi-stage pre-training for low-resource domain adaptation*（IBM Research）明确写道：**"extending the vocabulary of the LM with domain-specific terms leads to further gains"**（扩展 LM 词表带来进一步收益）。IBM Granite 4.x 已采用 128K（Llama-3 风格）词表，Granite 4.2 30B 为 100,352 tokens；Granite Embedding Multilingual R2 用 262K / 180K 词表覆盖 **200+ 语言**，并把 tokenizer 效率（fertility，每词 token 数）作为核心指标。
- **学界共识（Takase et al., Findings of ACL 2025, *Large Vocabulary Size Improves LLMs*）**：在 12 种语言上，500k 词表相较 32k **跨语言平均 +8.7%**；日语 **+9.2**（汉字复合词能成单 token）；机制是"数据优势"——固定 token 预算下大词表每 token 承载更长文本，训练更快、注意力更省。
- **代价（必须知道）**：词表增大会线性放大**嵌入层 + 输出层**（输出层 softmax 在每个生成步都算，是更尖锐的成本）；**词汇膨胀（bloat）** 风险——加入极少出现的 token 又没足够数据训练，会得到劣质 embedding，NeurIPS 2024 工作显示超大词表平均 **loss +2.8%**；~256k 之后收益递减。

### 6.2 对 GARGANTUA 的具体判断

- 本项目词表是**字符/token 级**（非 BPE 子词），所以"扩表"在这里等价于**可表示性**：目前希腊/西里尔/假名/谚文等常用字符被 `enctok` **静默丢弃**——这比任何 BPE 碎片都糟（BPE 至少能拼回来，我们是直接丢）。把这些常用字补进词表 = 让真实文本**第一次能被表示**，是**无条件的正收益**，且它们是常用字，只要有对应数据就会被充分训练，不存在 bloat。
- **生僻汉字（CJK Ext-B+）正是 bloat 的反面教材**——65k 生僻字出现频率极低，无海量数据必得劣质 embedding、拖累 loss。所以"排除生僻"与学界"别加罕见 token"的结论**完全对齐**。
- **唯一的真坑**：本次要加的日/俄/阿/泰等体系，当前训练语料只有 zh+en，**没有对应训练数据**。IBM 的提能来自"定制词表 + 大规模专用语料"二者齐备。只加词表不加数据 → 这些 token embedding 欠训练（仍是 bloat 的一种），模型在这些语言上**能表示但不一定强**。→ 要真正吃到扩表红利，需**配套多语种语料**（哪怕少量）来训练这些新 token。

## 7. 关键剩余决策（待拍板）

1. **是否配套多语种训练语料？** —— 决定本次扩表是"仅恢复可表示性"还是"真正提能"。建议：扩表的同时，至少引入各目标语言的少量干净语料（维基/新闻摘要级），让新 token 有训练信号。
2. **CJK 兼容汉字(472) + 全角(110)走「加 token」还是「编码前 NFKC 归一化」？** —— 用户已定"全都加"，按加 token 实施（scenario A）。NFKC 作为后续可选优化（省 ~582 token，代价是变体字形不再独立）。

## 8. 实施约束（落地时必须遵守）

- **真·末尾追加，零 ID 平移**：新 token 落 IDs 76497..，新建位于 `code` 之后的区；**绝不**走内置 `user_added→extra` 通道（它插在 `english` 与 `fallback` 之间，会把 fallback(26)+code(474) 整体 +N 平移，已证实会错配现有 98,464 个 token）。
- **移除上限**：改 `server/spec.py` 的 `MAX_VOCAB_SIZE`（及任何引用处），或将其放松；`tok6._check_id` 上界随之放宽。
- **派生词表，不动 base1.2.0 本体**：建模 `config.json` 的 `vocab_size` 与词表目录指这份派生词表（与「base 不动、派生对拍」口径一致）。
- **enctok 识别**：新单字进入 `regular_ids`（category=regular），`enctok.encode_text` 的字符→ID 查表自动覆盖，无需改编码逻辑。
- **语料重编码**：全语料用派生词表重编码（约 27s），验证 `skipped` 归零（除 NFKC 已归一化的变体外）。
- **多语种语料**：若第 7 节决策为"加"，需新增多语种语料并纳入训练。

## 9. 风险与验证

- 风险①：误用 `extra` 通道导致 ID 平移（已识别，规避见第 8 节）。
- 风险②：只扩词表不加数据 → 新 token 欠训练（bloat）。缓解：配套多语种语料 + 训练后对比新 token 的 embedding 范数/命中率。
- 验证：扩展后跑 `tokens inspect` + 全量文本往返 + 官方 `queue add --mode lm`；对比扩表前后编码 token 数（碎片化下降 = 扩表有效直接证据）；训练后监控新 token 的 loss 贡献。
- 测试：按项目约定，全量回归需 steve 授权；改动不提交、不进公开仓库。

---
*下一步：拍板第 7 节（是否配套多语种语料），再执行「改 spec 上限 → 建派生末尾追加词表 →（可选）引入多语种语料 → 重编码 → 验证」五步走。*
