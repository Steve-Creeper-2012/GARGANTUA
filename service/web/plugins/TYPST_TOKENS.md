# Typst 结构 token 对应表（base1.3.0 / 1.3.0）

> 适用版本：**base1.3.0 / token-6**（2026-11-07 定案）。唯一真源 = `server/spec.py` 的 `TYPST_CONSTRUCTS`；本表由 `python service/tokens/gen_typst_map.py` 生成，**禁止手改 ID**。
> 依据：`gargantua-typst-token-spec.md` §3 全量 token 词表 + §2 属性机制。本次只交付词表与表格，**映射器（token 流 → .typ）留作后续独立实现**。

## 1. 布局与规则

| 项 | 值 |
|---|---|
| 契约头六段 | core 3 + boundary 6 + system 3 + slot N + **typst 228** + draw 9605 |
| HEAD_SIZE | 9845（base N=0） |
| DRAW_BEGIN | 240（base N=0） |
| typst 区间 | [12, 240) |
| 构造数 / token 数 | 114 / 228 |

形态约定（steve）：`{}|` = system、`[]` = border、`<>` = tools（core 机制单标记 + Typst 结构）、`<>` = 用户自定义预留（slot 编号）、`内容:HEX` = 可见常规 token。

- **pair**：每个构造占两个连续 ID：开在前、闭在后，close_id = open_id + 1
- **wrappers**：`{}|` = system；`[]` = border；`<>` = tools（core + typst / slot 用户自定义预留）；`内容:HEX` = 可见常规 token
- **empty**：empty=true 的构造仍保留显式空对 `<X></X>`（规范 §2）
- **attrs**：属性走 `<ATTR>…</ATTR>`，紧跟开标签；映射器聚合成具名参数
- **math**：MATH/MATHB 通道内不做内容转义，运算符/数字/标识符全 token 化
- **unicode_ext**：可见区末尾（code 之后）另有 `unicode_ext` 3537 个字符 / 31 个 Unicode 区块（base1.3.0 新增，regular）；散文子词用这些普通可见 token 承载，不属于本表，映射器按内容通道原样输出并做 §4.2 转义
- **id_shift**：派生词表带 N 个预留槽位时，本表所有 ID 与 typst_begin 一同 +N；用 /api/agent/spec 的 typst.begin 换算
- **token6**：typst 与 core/slot 同为 `<>` tool 元素：进 loss、有 KV、type_emb=TYPE_NONE

## 2. 构造对应表

ID 列为「十六进制 / 十进制」；配对规则恒为 `闭 ID = 开 ID + 1`。

| 构造 | 开 token (ID) | 闭 token (ID) | § | 类别 | Typst 构造 | 空 | 属性 | 说明 |
|---|---|---|---|---|---|:---:|---|---|
| `DOC` | `<DOC>` (C / 12) | `</DOC>` (D / 13) | 3.1 | root | （文档根，可省略） | — | — | 顶层包裹 |
| `PAGESET` | `<PAGESET>` (E / 14) | `</PAGESET>` (F / 15) | 3.1 | page | set page(..) | — | `margin`, `paper`, `columns` | 页边距/纸张/分栏 |
| `TEXTSET` | `<TEXTSET>` (10 / 16) | `</TEXTSET>` (11 / 17) | 3.1 | page | set text(..) | — | `font`, `size`, `lang`, `cjk` | 字体/字号/CJK |
| `PARSET` | `<PARSET>` (12 / 18) | `</PARSET>` (13 / 19) | 3.1 | page | set par(..) | — | `spacing`, `justify`, `leading` | 段落间距/对齐 |
| `DOCSET` | `<DOCSET>` (14 / 20) | `</DOCSET>` (15 / 21) | 3.1 | page | set document(..) | — | `title`, `author` | 标题/作者 |
| `HEADER` | `<HEADER>` (16 / 22) | `</HEADER>` (17 / 23) | 3.1 | page | page(header: ..) | — | — | 页眉 |
| `FOOTER` | `<FOOTER>` (18 / 24) | `</FOOTER>` (19 / 25) | 3.1 | page | page(footer: ..) | — | — | 页脚 |
| `BIB` | `<BIB>` (1A / 26) | `</BIB>` (1B / 27) | 3.1 | empty | bibliography(..) | ∅ | `path`, `style` | 参考文献 ∅ |
| `OUTLINE` | `<OUTLINE>` (1C / 28) | `</OUTLINE>` (1D / 29) | 3.1 | empty | outline | ∅ | `title`, `depth` | 目录 ∅ |
| `INCLUDE` | `<INCLUDE>` (1E / 30) | `</INCLUDE>` (1F / 31) | 3.1 | empty | include ".." | ∅ | `src` | 引入模块 ∅ |
| `IMPORT` | `<IMPORT>` (20 / 32) | `</IMPORT>` (21 / 33) | 3.1 | page | import ".." | — | `src` | 导入 |
| `H1` | `<H1>` (22 / 34) | `</H1>` (23 / 35) | 3.2 | heading | =  | — | — | 一级标题 |
| `H2` | `<H2>` (24 / 36) | `</H2>` (25 / 37) | 3.2 | heading | ==  | — | — | 二级标题 |
| `H3` | `<H3>` (26 / 38) | `</H3>` (27 / 39) | 3.2 | heading | ===  | — | — | 三级标题 |
| `H4` | `<H4>` (28 / 40) | `</H4>` (29 / 41) | 3.2 | heading | ====  | — | — | 四级标题 |
| `H5` | `<H5>` (2A / 42) | `</H5>` (2B / 43) | 3.2 | heading | =====  | — | — | 五级标题 |
| `H6` | `<H6>` (2C / 44) | `</H6>` (2D / 45) | 3.2 | heading | ======  | — | — | 六级标题 |
| `P` | `<P>` (2E / 46) | `</P>` (2F / 47) | 3.2 | block | 段落（空行分隔） | — | — | 显式段落 |
| `QUOTE` | `<QUOTE>` (30 / 48) | `</QUOTE>` (31 / 49) | 3.2 | block | quote[..] | — | — | 引用块 |
| `BLOCK` | `<BLOCK>` (32 / 50) | `</BLOCK>` (33 / 51) | 3.2 | block | block(..)[..] | — | `width`, `height`, `fill` | 通用块 |
| `BOX` | `<BOX>` (34 / 52) | `</BOX>` (35 / 53) | 3.2 | block | box(..)[..] | — | `width`, `height` | 行内块 |
| `PAD` | `<PAD>` (36 / 54) | `</PAD>` (37 / 55) | 3.2 | block | pad(..)[..] | — | `x`, `y`, `left`, `right`, `top`, `bottom` | 内边距 |
| `COLUMNS` | `<COLUMNS>` (38 / 56) | `</COLUMNS>` (39 / 57) | 3.2 | block | columns(2)[..] | — | `n` | 分栏 |
| `GRID` | `<GRID>` (3A / 58) | `</GRID>` (3B / 59) | 3.2 | block | grid(..) | — | `columns`, `rows`, `gutter` | 网格布局 |
| `STACK` | `<STACK>` (3C / 60) | `</STACK>` (3D / 61) | 3.2 | block | stack(..) | — | `dir`, `spacing` | 堆叠布局 |
| `ALIGN` | `<ALIGN>` (3E / 62) | `</ALIGN>` (3F / 63) | 3.2 | block | align(center)[..] | — | `mode` | 对齐 |
| `PLACE` | `<PLACE>` (40 / 64) | `</PLACE>` (41 / 65) | 3.2 | block | place(..)[..] | — | `dx`, `dy` | 绝对定位 |
| `FIGURE` | `<FIGURE>` (42 / 66) | `</FIGURE>` (43 / 67) | 3.2 | block | figure(..) | — | `caption` | 图/表浮动体 |
| `CAPTION` | `<CAPTION>` (44 / 68) | `</CAPTION>` (45 / 69) | 3.2 | block | figure 的 caption | — | — | 图注 |
| `PARBREAK` | `<PARBREAK>` (46 / 70) | `</PARBREAK>` (47 / 71) | 3.2 | empty | parbreak | ∅ | — | 段落分隔 ∅ |
| `LINEBREAK` | `<LINEBREAK>` (48 / 72) | `</LINEBREAK>` (49 / 73) | 3.2 | empty | linebreak | ∅ | — | 换行 ∅ |
| `PAGEBREAK` | `<PAGEBREAK>` (4A / 74) | `</PAGEBREAK>` (4B / 75) | 3.2 | empty | pagebreak | ∅ | — | 分页 ∅ |
| `SPACER` | `<SPACER>` (4C / 76) | `</SPACER>` (4D / 77) | 3.2 | empty | v(..) / h(..) | ∅ | `amount` | 间距 ∅ |
| `B` | `<B>` (4E / 78) | `</B>` (4F / 79) | 3.3 | inline | *..* | — | — | 加粗 |
| `I` | `<I>` (50 / 80) | `</I>` (51 / 81) | 3.3 | inline | _.._ | — | — | 斜体 |
| `CODE` | `<CODE>` (52 / 82) | `</CODE>` (53 / 83) | 3.3 | inline | 行内代码反引号 | — | — | 行内代码 |
| `LINK` | `<LINK>` (54 / 84) | `</LINK>` (55 / 85) | 3.3 | inline | link("..")[..] | — | `url` | 链接 |
| `RAW` | `<RAW>` (56 / 86) | `</RAW>` (57 / 87) | 3.3 | inline | 围栏原始块 | — | `lang` | 代码块（保留格式） |
| `LABEL` | `<LABEL>` (58 / 88) | `</LABEL>` (59 / 89) | 3.3 | inline | label 名 | — | — | 标签 |
| `REF` | `<REF>` (5A / 90) | `</REF>` (5B / 91) | 3.3 | inline | @name | — | — | 引用 |
| `SYM` | `<SYM>` (5C / 92) | `</SYM>` (5D / 93) | 3.3 | empty | -> 等符号简写 | ∅ | `name` | 符号 ∅ |
| `COMMENT` | `<COMMENT>` (5E / 94) | `</COMMENT>` (5F / 95) | 3.3 | inline | /* .. */ | — | — | 注释（输出丢弃） |
| `UL` | `<UL>` (60 / 96) | `</UL>` (61 / 97) | 3.4 | list | list(..)[..] | — | — | 无序列表 |
| `OL` | `<OL>` (62 / 98) | `</OL>` (63 / 99) | 3.4 | list | enum(..)[..] | — | — | 有序列表 |
| `TERMS` | `<TERMS>` (64 / 100) | `</TERMS>` (65 / 101) | 3.4 | list | terms(..)[..] | — | — | 术语列表 |
| `LI` | `<LI>` (66 / 102) | `</LI>` (67 / 103) | 3.4 | list | 列表项 | — | — | UL/OL 子项 |
| `TERM` | `<TERM>` (68 / 104) | `</TERM>` (69 / 105) | 3.4 | list | 术语 | — | — | TERMS 子项 |
| `DESC` | `<DESC>` (6A / 106) | `</DESC>` (6B / 107) | 3.4 | list | 描述 | — | — | TERMS 子项 |
| `TABLE` | `<TABLE>` (6C / 108) | `</TABLE>` (6D / 109) | 3.5 | table | #table(columns:.., ..) | — | `columns` | 表格 |
| `ROW` | `<ROW>` (6E / 110) | `</ROW>` (6F / 111) | 3.5 | table | 一行 | — | — | 含若干 CELL |
| `CELL` | `<CELL>` (70 / 112) | `</CELL>` (71 / 113) | 3.5 | table | 单元格内容 | — | `colspan`, `rowspan` | 单元格 |
| `TH` | `<TH>` (72 / 114) | `</TH>` (73 / 115) | 3.5 | table | 表头单元格 | — | `colspan` | 加粗表头 |
| `MATH` | `<MATH>` (74 / 116) | `</MATH>` (75 / 117) | 3.6 | math | $..$ | — | — | 行内数学 |
| `MATHB` | `<MATHB>` (76 / 118) | `</MATHB>` (77 / 119) | 3.6 | math | $ .. $ | — | — | 块级数学 |
| `MSUB` | `<MSUB>` (78 / 120) | `</MSUB>` (79 / 121) | 3.6 | math | _ | — | — | 下标 |
| `MSUP` | `<MSUP>` (7A / 122) | `</MSUP>` (7B / 123) | 3.6 | math | ^ | — | — | 上标 |
| `MFRAC` | `<MFRAC>` (7C / 124) | `</MFRAC>` (7D / 125) | 3.6 | math | frac(..) | — | — | 分数（含 NUM/DEN） |
| `NUM` | `<NUM>` (7E / 126) | `</NUM>` (7F / 127) | 3.6 | math | 分子 | — | — | MFRAC 分子 |
| `DEN` | `<DEN>` (80 / 128) | `</DEN>` (81 / 129) | 3.6 | math | 分母 | — | — | MFRAC 分母 |
| `MSQRT` | `<MSQRT>` (82 / 130) | `</MSQRT>` (83 / 131) | 3.6 | math | sqrt(..) | — | — | 平方根 |
| `MROOT` | `<MROOT>` (84 / 132) | `</MROOT>` (85 / 133) | 3.6 | math | root(..) | — | — | n 次根 |
| `MSUM` | `<MSUM>` (86 / 134) | `</MSUM>` (87 / 135) | 3.6 | math | sum | — | — | 求和（含 MBELOW/MABOVE） |
| `MBELOW` | `<MBELOW>` (88 / 136) | `</MBELOW>` (89 / 137) | 3.6 | math | 下限 | — | — | MSUM 下限 |
| `MABOVE` | `<MABOVE>` (8A / 138) | `</MABOVE>` (8B / 139) | 3.6 | math | 上限 | — | — | MSUM 上限 |
| `MINT` | `<MINT>` (8C / 140) | `</MINT>` (8D / 141) | 3.6 | math | integral | — | — | 积分 |
| `MLIM` | `<MLIM>` (8E / 142) | `</MLIM>` (8F / 143) | 3.6 | math | lim | — | — | 极限 |
| `MMAT` | `<MMAT>` (90 / 144) | `</MMAT>` (91 / 145) | 3.6 | math | mat(..) | — | — | 矩阵（含 MROW） |
| `MROW` | `<MROW>` (92 / 146) | `</MROW>` (93 / 147) | 3.6 | math | 矩阵一行 | — | — | MMAT 子项 |
| `MCASE` | `<MCASE>` (94 / 148) | `</MCASE>` (95 / 149) | 3.6 | math | cases(..) | — | — | 分段 |
| `MALIGN` | `<MALIGN>` (96 / 150) | `</MALIGN>` (97 / 151) | 3.6 | empty | & | ∅ | — | 对齐点 ∅ |
| `MFUNC` | `<MFUNC>` (98 / 152) | `</MFUNC>` (99 / 153) | 3.6 | math | sin(..) | — | `name` | 数学函数 |
| `MID` | `<MID>` (9A / 154) | `</MID>` (9B / 155) | 3.6 | math | 变量名 | — | — | 标识符 |
| `MN` | `<MN>` (9C / 156) | `</MN>` (9D / 157) | 3.6 | math | 数字字面量 | — | — | 数字 |
| `MOP` | `<MOP>` (9E / 158) | `</MOP>` (9F / 159) | 3.6 | math | + - * / = < > 等 | — | — | 运算符（允许集见规范） |
| `MTEXT` | `<MTEXT>` (A0 / 160) | `</MTEXT>` (A1 / 161) | 3.6 | math | 数学中文本 ".." | — | — | 数学中文本 |
| `CODEBLOCK` | `<CODEBLOCK>` (A2 / 162) | `</CODEBLOCK>` (A3 / 163) | 3.7 | code | #{ .. } | — | — | 代码块 |
| `CBLOCK` | `<CBLOCK>` (A4 / 164) | `</CBLOCK>` (A5 / 165) | 3.7 | code | [..] | — | — | 内容块 |
| `SET` | `<SET>` (A6 / 166) | `</SET>` (A7 / 167) | 3.7 | code | set text(..) | — | `sel` | 设值规则 |
| `SETIF` | `<SETIF>` (A8 / 168) | `</SETIF>` (A9 / 169) | 3.7 | code | set text(..) if .. | — | `sel`, `cond` | 条件设值 |
| `SHOW` | `<SHOW>` (AA / 170) | `</SHOW>` (AB / 171) | 3.7 | code | show heading: .. | — | `sel` | 显示规则 |
| `SHOWSET` | `<SHOWSET>` (AC / 172) | `</SHOWSET>` (AD / 173) | 3.7 | code | show heading: set .. | — | `sel` | 显示-设值 |
| `SHOWFUNC` | `<SHOWFUNC>` (AE / 174) | `</SHOWFUNC>` (AF / 175) | 3.7 | code | show raw: it => .. | — | `sel` | 变换规则 |
| `SHOWALL` | `<SHOWALL>` (B0 / 176) | `</SHOWALL>` (B1 / 177) | 3.7 | code | show: template | — | — | 全局套用 |
| `SHOWTEXT` | `<SHOWTEXT>` (B2 / 178) | `</SHOWTEXT>` (B3 / 179) | 3.7 | code | show "badly": .. | — | `text` | 文本替换 |
| `SHOWREGEX` | `<SHOWREGEX>` (B4 / 180) | `</SHOWREGEX>` (B5 / 181) | 3.7 | code | show regex(..): .. | — | `pat` | 正则选择 |
| `CONTEXT` | `<CONTEXT>` (B6 / 182) | `</CONTEXT>` (B7 / 183) | 3.7 | code | context .. | — | — | 自省上下文 |
| `LET` | `<LET>` (B8 / 184) | `</LET>` (B9 / 185) | 3.7 | code | let x = .. | — | `name` | 绑定 |
| `DEF` | `<DEF>` (BA / 186) | `</DEF>` (BB / 187) | 3.7 | code | let f(x) = .. | — | `name` | 具名函数 |
| `IF` | `<IF>` (BC / 188) | `</IF>` (BD / 189) | 3.7 | code | if .. | — | — | 条件 |
| `ELIF` | `<ELIF>` (BE / 190) | `</ELIF>` (BF / 191) | 3.7 | code | else if .. | — | `cond` | 条件分支 |
| `ELSE` | `<ELSE>` (C0 / 192) | `</ELSE>` (C1 / 193) | 3.7 | code | else .. | — | — | 条件分支 |
| `FOR` | `<FOR>` (C2 / 194) | `</FOR>` (C3 / 195) | 3.7 | code | for x in .. | — | `var`, `in` | 循环 |
| `WHILE` | `<WHILE>` (C4 / 196) | `</WHILE>` (C5 / 197) | 3.7 | code | while .. | — | `cond` | 循环 |
| `RETURN` | `<RETURN>` (C6 / 198) | `</RETURN>` (C7 / 199) | 3.7 | code | return .. | — | — | 返回 |
| `BREAK` | `<BREAK>` (C8 / 200) | `</BREAK>` (C9 / 201) | 3.7 | empty | break | ∅ | — | 循环控制 ∅ |
| `CONTINUE` | `<CONTINUE>` (CA / 202) | `</CONTINUE>` (CB / 203) | 3.7 | empty | continue | ∅ | — | 循环控制 ∅ |
| `CALL` | `<CALL>` (CC / 204) | `</CALL>` (CD / 205) | 3.7 | code | #rect(..)[..] | — | `name` | 通用函数调用 |
| `ARG` | `<ARG>` (CE / 206) | `</ARG>` (CF / 207) | 3.7 | code | 具名参数 k: v | — | `name` | CALL 参数 |
| `STR` | `<STR>` (D0 / 208) | `</STR>` (D1 / 209) | 3.8 | value | "hello" | — | — | 字符串 |
| `INT` | `<INT>` (D2 / 210) | `</INT>` (D3 / 211) | 3.8 | value | 10 | — | — | 整数 |
| `FLOAT` | `<FLOAT>` (D4 / 212) | `</FLOAT>` (D5 / 213) | 3.8 | value | 3.14 | — | — | 浮点 |
| `BOOL` | `<BOOL>` (D6 / 214) | `</BOOL>` (D7 / 215) | 3.8 | value | true / false | — | — | 布尔 |
| `NONE` | `<NONE>` (D8 / 216) | `</NONE>` (D9 / 217) | 3.8 | empty | none | ∅ | — | 占位 ∅ |
| `AUTO` | `<AUTO>` (DA / 218) | `</AUTO>` (DB / 219) | 3.8 | empty | auto | ∅ | — | 占位 ∅ |
| `LEN` | `<LEN>` (DC / 220) | `</LEN>` (DD / 221) | 3.8 | value | 2pt | — | — | 长度 |
| `ANG` | `<ANG>` (DE / 222) | `</ANG>` (DF / 223) | 3.8 | value | 90deg | — | — | 角度 |
| `FRAC` | `<FRAC>` (E0 / 224) | `</FRAC>` (E1 / 225) | 3.8 | value | 2fr | — | — | 弹性比例 |
| `RATIO` | `<RATIO>` (E2 / 226) | `</RATIO>` (E3 / 227) | 3.8 | value | 50% | — | — | 百分比 |
| `ARR` | `<ARR>` (E4 / 228) | `</ARR>` (E5 / 229) | 3.8 | value | (..) | — | — | 数组（含 ITEM） |
| `ITEM` | `<ITEM>` (E6 / 230) | `</ITEM>` (E7 / 231) | 3.8 | value | 数组项 | — | — | ARR 子项 |
| `DICT` | `<DICT>` (E8 / 232) | `</DICT>` (E9 / 233) | 3.8 | value | (k: v) | — | — | 字典（含 KV） |
| `KV` | `<KV>` (EA / 234) | `</KV>` (EB / 235) | 3.8 | value | k: v | — | `k` | DICT 子项 |
| `CITE` | `<CITE>` (EC / 236) | `</CITE>` (ED / 237) | 3.8 | empty | cite(..) | ∅ | `key` | 引用 ∅ |
| `ATTR` | `<ATTR>` (EE / 238) | `</ATTR>` (EF / 239) | 2 | meta | name="k" value="v" | — | `name`, `value` | 通用属性，紧跟开标签，映射器聚合成具名参数 |

## 3. 后续映射器怎么接

1. 从 `/api/agent/spec` 读 `typst.begin / size`（或从 `typst_map.json` 的 `layout`）确定本份词表的区间；派生词表带 N 个槽位时整体 +N。
2. 逐个 ID 查 `typst_map.json` 的 `tokens`，拿到 `kind`（open/close）、`construct` 与 `pair_id`；用栈校验配对与嵌套，非法报位置。
3. 开/闭 token 之间的普通子词是内容通道，按规范 §4.2 转义触发字符；`MATH` / `MATHB` 之内不转义。
4. `ATTR` 聚合为具名参数、`CALL` 展开成 `#name(args)[body]`；`typst` 列给出每个构造对应的 Typst 写法（规范 §3 原文）。
5. 其余结构按 `typst` 列 + `empty` 标记生成 `.typ`，再交 `typst compile`。

---
*由 `service/tokens/gen_typst_map.py` 生成；2026-11-07T20:51:04Z*
