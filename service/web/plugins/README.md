# 推理显示插件（Infer Display Plugins）

推理页中间那一整块——**左右边栏之间、顶部磨砂条之下，包含输入框**——不归主
程序，归插件。宿主只提供一块容器和一个受控 API，剩下全部由插件自己决定。

这么分的原因很直接：不同词表的 token 长得完全不一样。把「怎么显示」写死在
`app.js` 里，每加一个词表就要动一次核心；做成插件后，新词表 = 一个新文件。

---

## 1. 目录与注册

```
service/web/plugins/
  README.md        本文件（协议 + 开发文档）
  infer-plug.js    内置默认插件：Infer-Plug v1.0.0（正常文字交互 + 文件 chip + 画布）
  infer-plug.css   它的样式
  tokens.js        内置插件：token 对话（多轮、微信式、转换器格式）
  tokens.css       它的样式
  _imported.json   导入插件的注册表（由服务端维护，不要手改）
  <id>.js          开发者导入的插件（运行时生成）
  <id>.css         可选的同名样式
```

- **内置插件**写死在 `app.js` 的 `BUILTIN_PLUGINS` 里：`infer-plug`（默认）与
  `tokens`。localStorage 没存过选择时 `selectedPluginId()` 返回 `infer-plug`，
  已存过的选择一律保留。
- **正常文字交互已由 Infer-Plug v1.0.0 实现并注册**（见第 8 节），不再处于
  「开发中 / 不显示」状态。目录里没有 `base.js`，也不需要。
- **导入插件**：「开发者选项 → 显示插件 → 导入插件…」，选一个本地 `.js`
  （可同时选一份 `.css`）。服务端通过 `POST /api/plugin/import` 把文件落到本
  目录并写进 `_imported.json`；`GET /api/plugin/list` 下发清单，前端合并进
  同一个自绘下拉。`POST /api/plugin/remove` 移除。
  - `id` 规则：`^[a-z0-9][a-z0-9_-]{0,31}$`，不能以 `_` 开头，不能占用
    `tokens` / `base` / `_imported` 等保留名。
  - `code` ≤ 512 KiB 且必须含 `export default`；`css` ≤ 256 KiB。
  - 文件头可选写元信息（服务端解析前 40 行）：
    ```js
    // @name 回声显示
    // @desc 逐 token 回显，附一个计数器
    ```
- 服务端静态服务只放行 `plugins/` 下的 `.js` / `.css` / `.md`
  （`server/server.py::_serve_static`），其它路径与路径穿越一律 403。
  `.js` 的 `Content-Type: application/javascript`，可被 `import()` 加载。

---

## 2. 生命周期

```js
export default {
  id: "tokens",
  name: "token 对话",

  /**
   * @param {HTMLElement} root  宿主给的满高容器（已清空、已挂到页面上）
   * @param {object}      ctx   宿主能力（见第 3 节）
   * @returns {object}          可选：{ reset(), destroy() }
   */
  mount(root, ctx) {
    // 建自己的 DOM、绑事件、按需拉数据
    return {
      reset() {},     // 用户点顶栏「清空」时调用（清会话，不负责后端）
      destroy() {},   // 切插件 / 卸载时调用：解绑、停回调、别碰已卸载的 DOM
    };
  },
};
```

时序：

1. 用户第一次进推理页（或切换插件）→ 宿主注入 `css`（`<link data-plugin>`）；
2. `import(entry)` → 取 `default.mount`；
3. 清空 `root` → 调 `mount(root, ctx)`；
4. 切插件时：先 `destroy()` → `AbortController.abort()` → 移除 `<link>` →
   清空 `root` → 挂新插件。

`destroy()` 之后**不要再动 DOM**。异步回调（流式 token、定时器）用 `dead`
标志或 `ctx.signal.aborted` 短路——`tokens.js` 就是这么做的。宿主用「挂载世代」
机制处理并发切换：旧 `import()` 回来时若已有更新的切换请求，结果会被丢弃。

**顶部状态条由宿主绘制，插件只负责对话流 + 输入区**：推理页顶部那条
（状态点 + 阶段文案 + TTFT / tok·s / tokens / 上下文 / model 五个指标）是宿主
`#infer-statusbar` 画的。插件 **不再自建顶部指标条**（默认插件原有的
`.tk-head` / `.tk-metrics` 已删除）、不显示状态、不处理 `onPhase`、不上报指标；
`root` 里现在只有对话流（吃满高度）与底部输入区。

**会话切换会重新 mount 插件并注入历史消息**：宿主按会话维护 `ctx.session`，
切换会话时走和「切插件」同样的卸载流程，再重新 `mount()`；新插件从
`ctx.session.messages` 恢复多轮气泡（见第 3 节）。所以插件不必（也不应）自己
长期持有会话状态，历史由宿主持久化。

---

## 3. ctx：宿主给什么

```ts
ctx = {
  root: HTMLElement,          // 同 mount 的第一个参数
  plugin: {id, name},         // 当前插件元信息
  signal: AbortSignal,        // 卸载时 abort；长任务请自己检查

  api: {
    // 流式推理：起任务 + 按游标拉增量，返回 Promise（跑完/失败才 resolve）
    infer(req, handlers): Promise<void>,
    reset(): Promise,         // POST /api/infer/reset
    stats(): Promise,         // GET /api/infer/stats（上下文 / KV / fast weight）
    status(): Promise,        // GET /api/status（当前模型 + max_new_tokens 等）
    models(): Promise,        // GET /api/models
    loadModel(name): Promise, // POST /api/model/load
    clearFw(): Promise,       // 可选：清 fast weight
    toast(msg, isErr?): void,
    confirm(title, body): Promise,
    fmtBytes(n): string,
  },

  // 按插件隔离的 localStorage（键前缀 gargantua.plugin.<id>.）
  config: {get(key, dflt), set(key, value)},

  // 会话状态：宿主按会话维护，切换会话会重新 mount 插件并注入历史
  session: {
    id: string,
    // 历史消息（mount 时用来恢复多轮气泡）
    messages: [{role: "me"|"bot", token_ids: number[], ts: number}],
    // per-session 开发者选项（maxNew / kvEnabled …），由宿主写入；
    // 与上面的 ctx.config（按插件隔离的 localStorage）不是一回事
    config: {get(key, dflt), set(key, value)},
    // 落盘一条消息：发送后 append("me", ids)，onDone 后 append("bot", genIds)
    append(role: "me"|"bot", tokenIds: number[]): void,
  },

  // 注册清理函数（等价于 destroy 里手写，多个时更省事）
  onCleanup(fn): void,
}
```

> 采样参数 `temperature` / `top_p` 由右侧开发者选项统一控制，宿主在挂载时
> 与每次改动时写进 `ctx.config`；插件 `submit` 前读
> `ctx.config.get("temperature", 0.8)` / `("topP", 0.95)` 即可。

`maxNew` 与 `kvEnabled` 属于 **per-session 开发者选项**，读
`ctx.session.config.get("maxNew", 32)` / `("kvEnabled", true)`；`max_new` 不再
由插件自己画输入框（开发者选项里的 `#infer-maxnew` 接管，旧版 `.tk-num` 已移除）。

### 3.1 `api.infer(req, handlers)`

```js
await ctx.api.infer({
  // kvEnabled=true 时只发本轮增量 ids（服务端按 session 复用 KV/fast weight）；
  // kvEnabled=false 时发完整上下文（累计序列 + 本轮 ids），服务端重算整段。
  // 宿主会自动注入 session_id 与 kv_enabled，插件不要自己传。
  tokenIds: [0xC732, 0xC733, 0x1F],  // 纯 token 输入（与 prompt 二选一）
  prompt: "你好",                     // 文本输入，走词表编码器
  maxNew: 32,                        // 1..2048（来自 ctx.session.config）
  temperature: 0.8,
  topP: 0.95,
}, {
  onStart({taskId}) {},
  onToken(token, n) {},              // n = 第几个（从 1 开始）
  onDone(result, summary) {},
  onError(err) {},
});
```

> `onPhase(phase)`：**宿主消费，插件不需要**。phase ∈
> `idle|loading|encoding|decoding|done|error`（变化时去重通知一次，一轮结束
> 必有一次 `done`），由宿主用来绘制推理页顶部状态条 `#infer-statusbar`
> （状态点 + 阶段文案）。插件 **不自建顶部指标条、不显示状态、不处理 onPhase、
> 不上报指标**；TTFT / tok·s / tokens / 上下文 / model 这 5 个指标都由宿主
> 从流式回调里统计显示。

### 3.1.1 KV 复用开关与上下文契约（硬约定）

- `kvEnabled = ctx.session.config.get("kvEnabled", true)`；
- **开启**（默认）：后端按 `session_id` 持久化 KV cache 与 fast weight，插件每轮
  **只发本轮用户输入**的 `ids`（增量），不要把历史 `ctxIds` 拼进请求，否则会被
  服务端重复计算；
- **关闭**：发**完整上下文**（累计序列 + 本轮 `ids`），服务端重算整段；
- 无论哪种，UI 始终显示完整多轮对话（恢复历史 + 本轮）。`ctxIds` 是插件本地
  累计的滚动窗口序列，仅供 `kvEnabled=false` 时作为完整上下文发送，与 UI 气泡
  渲染解耦；别把「本地累计」和「发给服务端的长度」混用——两者在 `kvEnabled`
  开启时差别很大。

`onToken` 收到的 `token`：

| 字段 | 含义 |
| --- | --- |
| `id` | 生成的 token ID（十进制整数；显示时转成大写十六进制） |
| `index` | 本轮第几个（从 0 开始） |
| `t_ms` | 从**服务端收到请求**到产出这个 token 的毫秒数（含模型加载与排队）；首 token 即真实 TTFT |
| `kv_steps` | 这一步共享 KV 的步数 |
| `write` | 这一步 fast weight 的写入幅度 `‖S_t − S_{t−1}‖`（首步为 `null`） |
| `fw` | **恒为 `null`**：完整快照走 `/api/infer/stats`，宿主不把 24×24 矩阵塞进每个 token，避免长生成把缓冲吃穿 |

`onDone(result, summary)`：

```js
result = {
  text: "……",              // 用该模型词表解码的结果（纯 ID 输入可能为空/乱码）
  token_ids: [ ... ],      // 全部生成 ID
  archive: "logs/infer_archive/xxx.tokens",
  stats: {n_prompt, n_generated, ttft_ms, elapsed_ms, tps,
          ctx_full, ctx_compress, kv_steps, kv_bytes, d_model, vocab_size,
          fw, fw_writes},
}
summary = {status, tokens, next, ttft_ms, elapsed_ms, result}
```

宿主给流式轮询加了两道保护：**墙钟上限 10 分钟**（服务端一直回 `running` 也
不会卡死），以及**单次 GET 失败重试 3 次**。插件卸载时 `signal` 会 abort，
`infer()` 抛 `Error("已取消")`，`finally` 里记得解锁按钮。

### 3.2 底层 HTTP（插件一般不用直接碰）

```
POST /api/infer
  {stream: true, token_ids: [..] | prompt: "…",
   max_new_tokens, temperature, top_p}
  → 202 {task_id, stream: true}

GET  /api/infer/stream?task_id=<id>&from=<已消费数>
  → 200 {status: "running"|"done"|"error",
         tokens: [ {id, index, t_ms, kv_steps, write, fw}, … ],
         next: <下一次的 from>,
         ttft_ms, elapsed_ms, error, result}
```

`from` 是**已消费的 token 数**，不是时间戳，慢轮询不会漏 token 也不会重复
渲染。宿主每 90 ms 拉一次。**注意**：`task_id` 只存在内存里，刷新页面不能续
拉；网络中断超过 3 次连续失败整轮会结束（缓冲在服务端保留到流淘汰为止）。

### 3.3 输入约定（tokens.js：转换器格式插件）

`tokens.js` 把推理界面做成**微信式多轮对话**：你的输入落在右侧气泡、模型输出
落在左侧气泡，历史消息由 `ctx.session.messages` 在 mount 时恢复，本轮通过
`ctx.session.append` 落盘。发给服务端的上下文长度由 KV 复用开关决定（开启时
只发本轮增量、关闭时发完整序列，见 3.1.1），但 UI 始终显示完整多轮对话。
每个 token 直接以**转换器格式**呈现——文本 token 写成 `{text: B46A}`
（大写十六进制、自界定），特殊 token（`<caps>` / `[user]` / `[model]` /
`<stop>` …）写成名字。顶部状态 / 指标条由宿主绘制，插件容器内只有对话流与
输入区（不再自建 `.tk-head`）。

**Enter 不发送（防误触）**：输入框里 Enter 即换行，只有点「发送」按钮才会发起
推理。

插件只接受 **token 格式**，不接受自然语言。实际可粘的形态：

```
C732 C733 1F            裸十六进制 ID（可带 0x）
{t:C732}                紧凑格式
{text: C732}            可读显示格式（文本 / token 编辑器的输出）
C732|C733               步内分隔符
<5> / <C>               扩展系统 token（尖括号内是十六进制 ID）
[user] [model/] <stop>   [] 边界 / <stop>：固定名字的特殊 token（按 spec 映射到 0..7）
```

`wave:` / `visual:` / `draw:` 媒体载荷**明确拒绝**——它们是采样 / 像素向量，
不是离散 ID，硬按十六进制解析会静默产出一串错误 ID。

---

## 4. 样式约定

- 插件自带 CSS，由宿主注入/移除；**不要**改 `app.css`，也不要用主程序的
  类名（`.act` / `.card` / `.field` …）——那是宿主的东西，会变。
- 颜色、圆角、字体一律用 `app.css` 里的 CSS 变量：
  `--fg / --fg2 / --fg3 / --panel / --panel-2 / --fill / --fill-2 / --border /
   --accent / --accent-2 / --accent-soft / --on-accent / --track /
   --red / --amber / --green / --r / --r-pill / --ease`。
  这样亮/暗主题、圆角滑杆都能自动跟随。
- **不要用系统默认控件**（`<select>`、`<input type=file>`、`alert/confirm`）。
  需要下拉就自己画，需要确认就用 `ctx.api.confirm`。全站所有 `<select>`
  （含运行时动态插入的）都会被宿主换成自绘菜单，这条是项目铁律：
  界面上每一个像素都由前端代码绘制。
- 插件内文本要能选中复制（气泡 / token chip / 耗时小注 / 输入框）：`app.css`
  全局 `user-select:none` 的白名单不含这些元素，需在插件根显式放开
  `user-select:text`。

---

## 5. 最小可运行示例

```js
// my-echo.js —— 保存后从「开发者选项 → 显示插件 → 导入插件…」导入
// @name 回声（示例）
// @desc 逐 token 打印十六进制 ID
export default {
  id: "my-echo",
  name: "回声（示例）",
  mount(root, ctx) {
    const out = document.createElement("div");
    const btn = document.createElement("button");
    btn.textContent = "生成 8 个 token";
    out.style.cssText = "font:12px ui-monospace;white-space:pre-wrap";
    root.append(out, btn);

    let dead = false;
    btn.onclick = async () => {
      out.textContent = "";
      try {
        await ctx.api.infer({tokenIds: [0xC732], maxNew: 8}, {
          onToken: t => { if (!dead) out.textContent += t.id.toString(16) + " "; },
          onError: e => { if (!dead) out.textContent = "错误: " + e.message; },
        });
      } catch (e) {
        if (!dead && e.message !== "已取消") out.textContent = "错误: " + e.message;
      }
    };
    return {destroy() { dead = true; }};
  },
};
```

导入后它会出现在「推理界面渲染器」下拉里，选择结果存 `localStorage`，下次进
推理页直接用。

---

## 6. fast weight 可视化的数据契约

右侧开发者选项里的两张图由**宿主**画（不属于插件管辖范围），数据源是
`GET /api/infer/stats`：

| 字段 | 含义 |
| --- | --- |
| `fw.matrix` | `24×24`，`\|S\|` 对注意力头取均值后 adaptive-avg-pool 的结果，越亮写得越重 |
| `fw.head_norms` | 每个头的 Frobenius 范数（谁在记东西） |
| `fw.norm` | 整体 `‖S‖` |
| `fw_writes` | 最近 `FW_WRITE_HISTORY` 步的 `‖S_t − S_{t−1}‖`，画成折线 |
| `streaming` | 是否正在生成 |

生成**过程中**这些值由服务端从流缓冲里实时合成（不再穿透到被 `h.lock` 占用
的 infer 子进程），所以热力图和写入曲线会跟着 token 一起动。所有浮点值在
序列化前都做了非有限值（NaN/Inf）消毒，浏览器 `JSON.parse` 不会炸。

---

## 7. 已知边界

- **一次只能跑一个推理**：infer 服务按句柄串行，第二条请求会排队。
- **没有「停止生成」按钮**：中断 = 顶栏「清空」（`api.reset()`），它会杀掉
  infer 服务进程；或切插件 / 卸载，宿主的 `AbortController` 会中止前端轮询。
- **跨轮状态由后端按 session 持久化**：`infer.py::generate()` 现在按
  `session_id` 复用 KV cache 与 fast weight，跨 HTTP 请求保留，不再是每次
  从零开始。宿主为每个会话维护 `ctx.session`（`id` / `messages` / `config` /
  `append`），会话切换会重新 mount 插件并把历史消息注入进来。开发者选项里的
  「KV 复用」开关（`ctx.session.config.get("kvEnabled", true)`）决定本轮请求
  带多少上下文：开启时插件只发本轮增量 token（服务端复用已有 KV / fast
  weight）；关闭时发完整上下文让服务端重算整段。无论哪种，UI 都显示完整多轮
  对话；顶部状态条与 5 个指标由宿主绘制、从流式回调统计，插件不再自建。
- **纯 token 输入的解码文本不可靠**：`result.text` 只在 `prompt` 输入时有意义。
- **插件跑在主页面上下文**：它能拿到 `window` 和主程序的全局变量。这是
  本地单人工具的取舍——不做沙箱，靠约定。导入的插件代码请自行审阅。

---

## 8. Infer-Plug v1.0.0（正常文字交互，内置默认插件）

`infer-plug.js` / `infer-plug.css` 是新一代插件，与默认插件 `tokens.js` 的
「转换器格式 token 对话」是两套完全不同的显示口径：

| 维度 | tokens.js（旧） | Infer-Plug v1.0.0（默认） |
| --- | --- | --- |
| 输入控件 | 纯 `textarea`，只收十六进制 token | `contenteditable` div，收**正常文字** |
| 文件 | 无 | 左下角 `+` / 拖放 → 内联文件 chip，与文字**同一行无缝拼接** |
| 我气泡 | `{text: HEX}` 小 chip | 正常文字 + 高亮文件名 chip（只显示文件名，无预览） |
| 输出 | 逐 token 十六进制 chip | 流式正常字符 + agent / 特殊 / 工具**小徽标** |
| 图元 | 无 | 识别 DRAW 区 → SVG 画布**实时渲染** + 可拖拽时间轴回放 |
| 顶部状态条 | 宿主绘制（插件不碰） | 宿主绘制（插件不碰，不处理 `onPhase`） |

交互细节：Enter 换行不发送、只能点右下角 36×36 上箭头「发送」；paste 拦截成
纯文本；chip 用 `<span class="ip-file" contenteditable="false">` 承载，文件名
一律 `textContent`（绝不 `innerHTML` 拼用户数据），chip 的真实 `token_ids` /
`media` 存在 JS `Map` 里，`dataset` 只放索引；上传失败移除 chip 并提示。
图元画布按 `draw_layout.tokens_per`(9) 实时追加图元，slider 或播放只画前 k 个，
绘图结束后仍可拖回看每一步。所有异步回调都检查 `dead` 与 `ctx.signal.aborted`。

### 8.1 依赖的 `/api/agent/*` 契约（全部经 `ctx.api.*` 或相对路径）

| 端点 | 方法 | `ctx.api` | 用途 |
| --- | --- | --- | --- |
| `/api/agent/spec` | GET | `agentSpec()` | 下发词表 / 契约头 / 图元布局 / 文件类型表（不抄常量） |
| `/api/agent/encode` | POST `{text}` | `encodeText(text)` | 文字 → `token_ids`（按 DOM 顺序逐段编码） |
| `/api/agent/decode` | POST `{ids}` | `decodeIds(ids)` | mount 时恢复历史气泡（文件 chip 无法还原时只显示文字） |
| `/api/agent/file` | POST `{filename,b64}` | `uploadFile(file)` | 文件 → `{kind,token_ids,media,display}`；`FileReader` 转 base64 |
| `/api/agent/tools` | GET | `tools()` | 槽位表（旧 `tools/formats/boundaries` 为空数组） |
| `/api/agent/tool` | POST `{name,args}` | `runTool(name,args)` | **一律拒绝**（1.3.0 先全部不接，返回 400） |
| `/api/agent/tool_result` | POST `{name,result}` | `toolResult(name,result)` | 结果 → `[sys] 结果文本 [sys/]` token ID 序列 |
| `/api/infer` | POST | `infer(req)` | req 增加 `media:[{kind,b64,filename}]`（`inferStream` 透传） |

`/api/agent/spec` 关键字段：`vocab_name=base1.3.0`、`token_encoding=token-6`、
`head_size`（base 9845，派生词表 +N）、`stop_id`、`draw_end_id`、
`draw_begin`（base 240 + N）、`slot_begin=12`、`slot_count`（base 0）、
`special:{caps,user,user_end,model,model_end,stop,sys,sys_end}`、
`core:[{id,name}]`（3 项）、
`slots:[{slot,id,name,custom_name,kind,description,close,arg_hint,executable}]`（N 项，base 为空）、
`boundaries/formats/tools:[]`（保留空数组兼容）、
`file_types:[{ext,kind,label}]`、`max_file_bytes`、
`draw_layout:{begin,tokens_per,canvas:{offset,bins,markers},digits,
special_offset,color_max,fields:[{name,offset,bins,per}]}`。
`GET /api/infer/stream` 每个 token 新增 `char`（可显示字符或 token 名）与
`tool`（对象或 null）字段。

### 8.2 token-6 / base1.3.0 的新约定

> 契约头六段（含 `typst` 段）、core 区、编号槽位、自定义语义、KV 与
> fast weight 的完整说明见专文 [`AGENT_TOKENS.md`](./AGENT_TOKENS.md)。
> 本节只留插件侧要点。

- 词表叫 **base1.3.0**（`vocab_version=1.3.0`），编码口径 **token-6**；
  契约头 = **六段**：`core 3`（ID 0..2：`<caps>` / `<stop>` / `<draw:end>`）
  + `boundary 6`（ID 3..8，`[]` 段边界）
  + `system 3`（ID 9..11）
  + `slot N`（`<s0>`…`<sN-1>`，**N = 该词表 config 的 `slot_count`**）
  + `typst 228`（114 个 Typst 语法构造 × 显式开/闭对，`<DOC>`/`</DOC>` …
    `<ATTR>`/`</ATTR>`；`<>` tools，与 core 同类、挂 `TYPE_NONE`）
  + `draw 9605`。`HEAD_SIZE = 9845 + N`、`DRAW_BEGIN = 240 + N`、
  `TYPST_BEGIN = 12 + N`、`TYPST_END = 240 + N`、
  `STOP_ID=1`、`DRAW_END_ID=2`。
  **base 零预留（N=0 → HEAD_SIZE 9845 / DRAW_BEGIN 240）**；要 N 个槽位跑
  `python service/tokens/gen_vocab.py --name NAME --slots N`（`--activate`
  可设为全局运行时布局，需重启 server）。spec 不钉死 N。
- **`core` 单开一类**：三个最早就有的机制单标记，`<>` 形态、无闭合、
  不是 `[]` 边界、不是系统/特殊、不是工具调用；挂 `TYPE_NONE`。
  `<caps>` / `<stop>` 不再被 `parse_display` 折成 `[]`。
- **自定义槽位取代全部预定义工具 / 排版 token**（base 里一个都没有）：建表
  默认名字是编号 `<s0>…<sN-1>`，**编号直接显示**（Infer-Plug 渲染成「槽位 N」
  徽标）；前缀 `s` 是为了避开 `tok6` 的十六进制 ID 解析。
  2026-11-07 起名字也可自定义成任意不重名的 `<名字>`（如 `<mood>`）：词表页
  「增删 token」选 `slot` 子区即可，段长由「到首个 Typst token 为止」自描述。
  语义由使用者在 `spec.SLOT_DEFS` 自己定义；默认全空、不预设闭合关系、
  不接任何执行。
- **`<draw:end>` 语义**：DRAW 区由图元 token 组成，`<draw:end>`（ID = 2 =
  `spec.draw_end_id`）是绘画结束标记——收到它即结束绘图模式，其后的输出
  **恢复为普通文本**。画布声明的 `<x>` 标记（`draw_layout.begin +
  canvas.offset + 0`）出现即进入绘图模式；图元按 `tokens_per` 个 token 一组
  解析，各字段 `bin = id - begin - field.offset`，10 bit 颜色转 8 bit 用
  `Math.round(v*255/color_max)`。
- **1.1.0 的 `code` 区仍在**：可见区末尾的 474 个跨语言代码专用 token
  （`==` / `->` / `async` / `useState` / `SELECT` / `rgba` / `HashMap` …）。
  1.3.0 在其后再追加 **`unicode_ext` 区 3537 个字符**（31 个 Unicode 区块），
  category=`regular`。1.3.0 base 词表长 80262（派生 N 个槽位则 80262+N）。
  **1.2.0 契约头整体位移，1.3.0 又是一次 ID 整体位移**（draw 与全部可见区
  +228），所有 ID 都变了，旧 `.tokens` / 权重 / 存档不可读；插件一律从
  `/api/agent/spec` 取 ID，不得抄常量。
- **段类型**：slot 区挂 `TYPE_SYS`（值 3，与 `[sys]` 系统段同型），core 区
  与 `typst` 228 个都挂 `TYPE_NONE`；这些都「进 loss 就进 KV +
  fast weight」，机制上无特例。

### 8.3 自动 agent 闭环（1.3.0 默认不触发）

`done` 帧的 `result.tool_calls` 来自后端 `detect_tool_calls()`；1.3.0 里
`spec.SLOT_DEFS` 默认为空，**该函数恒返回 `[]`，闭环不会启动**，工具执行端点
也一律拒绝。这是工具台与测试台，实际执行留给将来的推理 app。

等使用者在 `SLOT_DEFS` 定义了 `kind="action"` + `close` 的槽位对后，
`runInfer()` 才会自动：`runTool()` → `sys` 气泡 → `toolResult()` 编成
`[sys]` 段 → 再发一轮推理，最多 `MAX_AGENT_ROUNDS = 3` 轮；
`ctx.config.get("autoAgent", true)` 为 false 时关闭。
