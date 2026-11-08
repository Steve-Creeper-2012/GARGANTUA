# 契约头与编号槽位 token 契约（base1.3.0）

> 适用版本：**base1.3.0 / token-6**（契约头 2026-11-06 下午定案；1.3.0 增补
> `typst` 段与 `unicode_ext` 可见区）
> 唯一事实源：`server/spec.py`。前端一律从 `GET /api/agent/spec` 取 ID，禁止抄常量。
>
> **本轮推翻了 1.1.0 的 53 个预定义工具 / 排版 token**：一个都不留，改成
> **编号槽位**，语义由使用者自己定义；`<caps>` / `<stop>` /
> `<draw:end>` 这类最早就有的机制单标记**单开 core 一类**。
>
> **预留槽位数量不在 spec 里钉死**：由每份词表自己的 `config.json` 的
> `slot_count` 决定。**base 词表零预留**（`slot_count=0`）；想要 N 个就跑
> `python service/tokens/gen_vocab.py --name NAME --slots N` 以 base 为模板
> 生成新词表。spec 只提供布局模板（`spec.head_layout(n)`），绝对 ID 由
> 当前全局词表的 config 派生。

本文是 [插件 README](./README.md) 第 8 节的展开，回答四个问题：
1. 契约头分成哪六段、各是什么语义（含 `typst` 段与可见区末尾的 `unicode_ext`）；
2. `core` / `typst` 为什么单开一类、和 `[]` 段边界 / 工具调用有什么区别；
3. 编号槽位怎么自己定义语义、调用结构长什么样、怎么生成带槽位的词表；
4. 这些 token 与 token-6、段类型嵌入、KV / fast weight 的关系。

---

## 1. 契约头六段（N = 该词表的 `config.slot_count`）

| # | 区 | 数量 | ID | 名字 | 形态 | 段类型 |
|---|---|---:|---|---|---|---|
| 1 | `core` | 3 | 0–2 | `<caps>` `<stop>` `<draw:end>` | `<>` 单标记 | `TYPE_NONE` |
| 2 | `boundary` | 6 | 3–8 | `[user]` `[user/]` `[model]` `[model/]` `[sys]` `[sys/]` | `[]` 段边界 | USER / MODEL / SYS |
| 3 | `system` | 3 | 9–11 | `{` `}` `\|` | 结构字符 | — |
| 4 | `slot` | **N** | 12 … 11+N | `<s0>` … `<sN-1>`（默认，可自定义） | `<>` 工具位 | `TYPE_SYS` |
| 5 | `typst` | **228** | (12+N) … (239+N) | `<DOC>`/`</DOC>` … `<ATTR>`/`</ATTR>` | `<>` 开/闭对 | `TYPE_NONE` |
| 6 | `draw` | 9605 | (240+N) … (9844+N) | 画布声明 + 图元字段 | 输出侧 token | — |

`typst` = 114 个 Typst 语法构造 × 显式开/闭对 = 228 个 `<>` tools，插在
`slot` 之后、`draw` 之前；与 `core` 同类：不可显示、进 loss、有 KV、段类型
`TYPE_NONE`（依据桌面 `gargantua-typst-token-spec.md` §3 + §2）。
固定前缀 = 12（core 3 + boundary 6 + system 3），`SLOT_BEGIN = 12` 恒定；
`TYPST_BEGIN = 12 + N`、`TYPST_END = 240 + N`。

> **2026-11-07 放宽：slot 段的名字不再要求编号。** `<mood>` / `<mytool:1>`
> 这类自定义名同样合法，唯一硬约束是「`<名字>` 形态 + 不与全表已有 token
> 重名」；段长仍由正文自描述——`vfc.infer_slot_count()` 从固定前缀 12 起
> 数到**第一个 Typst 结构 token** 为止，所以 typst 必须排在 slot 之后。
> 其余五段（core / boundary / system / typst / draw）依旧逐字冻结。
> 词表页「增删 token」表单里选 `slot` 子区即可增删，裸名会自动补上 `<>`；
> 前端策略由 `/api/vocab/regions` 的 `add_regions` / `add_policy_hints`
> 下发，唯一真源是 `spec.REGION_ADD_POLICY`。

| 量 | 公式 | base (N=0) | 例：N=128 |
|---|---|---:|---:|
| `TYPST_BEGIN` | `12 + N` | **12** | 140 |
| `TYPST_END` / `DRAW_BEGIN` | `240 + N` | **240** | 368 |
| `HEAD_SIZE` | `9845 + N` | **9845** | 9973 |
| `vocab_size` | `80262 + N` | **80262** | 80390 |

可见区从 `HEAD_SIZE` 开始（数字 / 空白 / 符号 / emoji / 中日韩 / 英文 /
兜底字母 / 1.1.0 追加的 474 个代码 token），末尾再追加 1.3.0 的
**`unicode_ext` 区 3537 个字符**（31 个 Unicode 区块：拉丁扩展 / 西里尔 /
希腊 / 假名 / 数学符号等），category=`regular`。

**ID 整体位移**（1.1.0 的 DRAW_BEGIN 64 → 1.2.0 的 12 → 1.3.0 的 240；
draw 与全部可见区 +228），旧 `.tokens` / 权重 / 存档一律不可读，按项目约定
不做向后兼容。
三份 base 词表（全局 `service/tokens/`、`vocabs/base1.3.0/`、
`vocabs/test/`）头部逐字一致，全部 `slot_count=0`。

---

## 2. core 区：单开一类的机制单标记

| ID | token | 作用 |
|---:|---|---|
| 0 | `<caps>` | 其后单词 / 字母首字母大写 |
| 1 | `<stop>` | 模型输出即停止生成 |
| 2 | `<draw:end>` | 绘画结束，其后输出恢复为文本 |

这 3 个是**最早就有的控制标记**，本轮按 steve 定案单开 `core` 一类：

- **不是 `[]`**：不折成扩展系统 token，位流里就写 `<0>` / `<1>` / `<2>`；
- **不是系统 / 特殊段边界**：`[]` 只用于 6 个 boundary；
- **不是工具调用**：没有名字、没有参数、不是任何 agent 技能；
- **无闭合**：单标记，没有 `<caps/>` / `<stop/>` 这种变体；
- **进 loss、有 KV + fast weight**，段类型挂 `TYPE_NONE`（不属于任何段，
  只是机制标记；`type_emb` 与是否进 loss 无关）。

`<stop>` 尤其关键：`train_step.window_index(force_last=True)` 会把训练窗口
尾部对齐到文档末尾的 `<stop>`，让终止监督真正生效（2026-10-27 审计修复）。

---

## 3. 用户自定义槽位（数量由词表 config 决定，名字可自定义）

默认名字是编号：`<s0>`、`<s1>`、…、`<sN-1>`，**编号直接显示**。
前缀 `s` 是必需的——`tok6._HEX_CHARS` 大小写都收，若写成 `<a0>` 会被当成
十六进制 ID 解析掉；`s` 不是 hex 字符，所以 `<sN>` 只会走名字查表。

**base 零预留 → base 里一个槽位都没有**；用 `gen_vocab.py` 生成 N 个：

```bash
python service/tokens/gen_vocab.py --name mymodel --slots 128
# 生成 vocabs/mymodel/{vocab.vfc, config.json}，config 写 slot_count=128
# --activate 可把它设为全局运行时布局（重启 server 后 spec 读到 128）
```

每份派生词表的 `config.json` 自带 `slot_count` / `head_size` / `draw_begin`；
`vfc.infer_slot_count()` 从 `.vfc` 正文数出槽位数（前缀 12 起、到首个
Typst 结构 token 之前），所以加载任意布局的词表都不依赖全局 spec 恰好也是
同一个 N，也不要求槽位名是连续的 `<sN>`。

**自定义名字（2026-11-07）**：编号只是 `gen_vocab.py` 建表时的默认写法。
在词表页把「添加到哪个子区」选成 `slot`，输入 `mood` 就会落成 `<mood>`；
名字任意，只要不跟全表已有 token 撞名。加一条 slot 会让后面所有 ID 平移 1，
`head_size` / `slot_count` / `regions` 由后端同步 +1；删除同样可逆。
`core / boundary / system / typst / draw` 五段在该下拉里标为「不可改」，
前端禁用、后端 `spec.check_add_region()` 也会拒绝，防止绕过 UI 改坏契约头。

| 字段 | 说明 |
|---|---|
| `slot` | `0 .. slot_count-1`（编号） |
| `id` | `12 + slot` |
| `name` | `<sN>`（永远保留，编号显示） |
| `custom_name` / `kind` / `description` / `close` / `arg_hint` | 全空，等使用者自己填 |
| `executable` | **一律 False**（先全部不接） |

**语义完全不预设**：不叫工具、不叫排版、不叫边界，不预设任何闭合关系。
它们是 N 个真实 token（有 ID、有 embedding、进 loss、有 KV），只是含义留空。

`core` 与 `slot` 都是 `<>` 形态的 `tool` 元素，但定型不同：
`core → TYPE_NONE`，`slot → TYPE_SYS`（与系统段同型），由
`spec.segment_type_of()` 统一给出。

---

## 4. 自己定义语义

在 `server/spec.py` 的 `SLOT_DEFS` 填（**默认 `{}`，全空**）：

```python
SLOT_DEFS = {
    0: {"name": "bash", "kind": "action", "close": 1,
        "description": "终端命令", "arg_hint": "命令",
        "executable": False},
    1: {"name": "bash/", "kind": "close"},
    7: {"name": "bold", "kind": "format", "close": 8,
        "description": "粗体开始"},
}
```

- `kind ∈ {"action", "format", "boundary", "close"}`；
- `close` 是闭合槽位**编号**（不是 ID）；不填就是单标记；
- `executable=True` 只会让协议标记可执行，**本项目的 `run_tool` 一律拒绝**
  （工具台与测试台，实际执行留给将来的推理 app）。

`GET /api/agent/spec` 下发 `core`（3 项）、`slots`（当前词表 `slot_count` 项，
base 为空）、`slot_begin`、`slot_count`、`head_size`、`draw_begin`、
`special`、`draw_layout` 等；
`boundaries` / `formats` / `tools` 三个旧字段保留为**空数组**，避免旧前端读
`undefined`。Infer-Plug 把每个槽位渲染成「槽位 N」徽标。

---

## 5. 调用结构（仅在自定义了 action/close 之后）

协议极简，**没有嵌套包裹**：一个调用就是一对槽位，参数直接写在中间。

```
<s0>  ls -la vocabs  <s1>        # open → 参数 token → close
<s7>  粗体内容  <s8>
```

`agent_tools.detect_tool_calls()` 只扫描 `SLOT_DEFS` 里定义了
`kind="action"` 且带 `close` 的槽位对，返回
`[{name, slot, args, start, end, tool_id}]`；**默认 `SLOT_DEFS` 为空 → 恒返回
`[]`**，模型输出不会被误解析成工具调用。

工具结果回灌也去掉了嵌套：

```
[sys]  结果文本 tokens  [sys/]
```

`agent_tools.tool_result_ids()` 就产这段（不再有 `<tool_result>` 对）。

### 相关 HTTP 端点

| 端点 | 用途 |
|---|---|
| `GET /api/agent/spec` | **唯一入口**：契约头 ID、`core`、`slots`（当前词表 N 项）、`slot_count`、`draw_layout`、`file_types` |
| `GET /api/agent/tools` | 槽位表（旧 `tools/formats/boundaries` 字段为空数组） |
| `POST /api/agent/encode` · `/decode` | 文本 ↔ token ID |
| `POST /api/agent/tool` | **一律拒绝**（先全部不接，返回 400） |
| `POST /api/agent/tool_result` | 结果 → `[sys] … [sys/]` token ID 序列 |
| `POST /api/agent/file` | 文件 → `{kind,token_ids,media,display}` |

---

## 6. 与 token-6 / 段类型的关系

`service/tokens/tok6.py` 是 `.tokens` 的现行格式：每字符 6 bit，29 个单字符
+ 6 个原子边界码（29–34）+ 终止码 63，码 35–62 预留。

- `core` 与 `slot` 在元素模型里都是 `{"kind":"tool","id":N}`，字符流写作
  `<名字>` 或 `<HEX>`；`from_steps()` 见这两段 ID 就产 `tool` 元素。
- `[id]` 是扩展 token（`{"kind":"ext"}`），**有嵌入、进主自回归 loss、不显示为
  可见内容**；`[]` 只用于 6 个 boundary。
- `to_steps()` 把 tool 元素单独成一步（`meta.tool = True`），按
  `spec.segment_type_of()` 定型；`infer._input_type_ids()` 与生成侧
  `_gen_type()` 各自叠加上下文默认值，但「core → NONE、slot → SYS」一致。
- `embed_text(ids, type_ids) = wte(ids) + type_emb(type_ids)`：段类型只改
  `type_ids` 这一路，不改 token ID。

---

## 7. 与 KV / fast weight 两套核心机制的关系

用户定的硬规则：**只要是进 loss 的 token，全部注入 KV 和 fast weight**。
精确口径（按代码核实，不按 token 类别想当然）：

- **纯文本窗口**（`common.make_batch`）：`y` 就是 `ids` 的滑动切片，
  **所有词表 ID 原样进 loss**，没有任何按类别的 masking；
- **多模态窗口**（`common.make_batch_emb`）：只有视觉 / 音频位置（无词表 ID）
  填 `-100` 排除；
- **`core` 3 个 + `boundary` 6 个 + `slot` N 个 + `typst` 228 个 +
  全部可见文本**都进 loss、都有 KV + fast weight
  （`embed_text = wte + type_emb`、`SharedKV.append` 无差别写入、
  `FastWeightBank.forward` 对整个 `[B,T,D]` 逐 token 推进）；
- **DRAW 图元 prims 压根不进模型**（只作输出侧结果），属于另一类「不进」。

一句话：**媒体位置不进 loss（目标 -100）；契约头与全部可见文本都进 loss、
都有 KV + fast weight；DRAW prims 不进模型。** 这正是「prims 不输入，所以没事」。

---

## 8. 版本与兼容

| 版本 | 契约头 | 变化 |
|---|---|---|
| `base1.0.0` | special 8 + system 3 + agent 53 + draw 9605 = 9669 | token-6 首发 |
| `base1.1.0` | 同上（9669） | 可见区末尾追加 `code` 474 条，旧 ID 全兼容 |
| `base1.2.0` | core 3 + boundary 6 + system 3 + slot N + draw 9605 = 9617+N；base 的 N=0 → 9617 | 契约头重排，53 个预定义工具退役；预留数交给词表 config；ID 整体位移，不做兼容（历史记录） |
| **`base1.3.0`** | **core 3 + boundary 6 + system 3 + slot N + typst 228 + draw 9605 = 9845+N**；**base 的 N=0 → 9845**；**vocab_size 76497 → 80262** | **六段契约头：slot 后插 `typst` 段（114 个 Typst 构造 × 显式开/闭对 = 228 个 `<>` tools），可见区末尾追加 `unicode_ext` 3537 字 / 31 区块；HEAD_SIZE 9617+N → 9845+N、DRAW_BEGIN 12+N → 240+N；又一次 ID 整体位移（+228），不做兼容** |

`spec.BASE_VOCAB_NAME = "base1.3.0"`、`VOCAB_VERSION = "1.3.0"`、
`TOKEN_ENCODING = "token-6"`、`HEAD_SIZE = 9845 + N`（base N=0 → 9845）、
`DRAW_BEGIN = 240 + N`（base → 240）、`TYPST_BEGIN = 12 + N`、
`TYPST_END = 240 + N`、`vocab_size = 80262`、`MAX_VOCAB_SIZE = 100000`。
`spec.BASE_SLOT_COUNT = 0`（base 零预留），
`spec.MAX_SLOT_COUNT = 4096`（防呆上界）。

---

## 9. 误区速查

| 误区 | 正解 |
|---|---|
| 认为 `<caps>` / `<stop>` 是 `[]` 扩展系统 token | 它们是 `core` 单开一类，`<>` 形态、进 loss、有 KV |
| 认为 core 是工具调用 | 不是，它是机制单标记，无参数无闭合 |
| 认为 `<sN>` 是某个具体工具 | 不是，只是编号槽位，语义在 `SLOT_DEFS` 里自己定义 |
| 认为槽位默认会执行工具 | 一律不接，`run_tool` 恒拒绝 |
| 认为 `<bash>` / `<asktime>` / `<b>` 还在 | 全部退役，1.2.0 里连名字都不存在 |
| 认为 `.tokens` 里 `<a0>` 能用 | 会被当 hex ID；必须用 `<s0>` 这类含非 hex 字母的名字 |
| 认为 1.2.0 兼容旧存档 | 不兼容，ID 整体位移 |
| 在插件里硬编码 `HEAD_SIZE = 9845` | 从 `/api/agent/spec` 读 `head_size`（base 9845，派生词表 +N） |
