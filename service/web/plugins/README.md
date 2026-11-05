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
  tokens.js        唯一内置插件：纯 token 流
  tokens.css       它的样式
  _imported.json   导入插件的注册表（由服务端维护，不要手改）
  <id>.js          开发者导入的插件（运行时生成）
  <id>.css         可选的同名样式
```

- **内置插件**写死在 `app.js` 的 `BUILTIN_PLUGINS` 里，当前只有 `tokens`。
- **base 基础词表插件正在开发中，按约定不注册、不显示**——所以目录里没有
  `base.js`。它的接口就是本文档描述的这一套，开发完直接注册即可。
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
  name: "纯 token 显示",

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
    toast(msg, isErr?): void,
    confirm(title, body): Promise,
    fmtBytes(n): string,
  },

  // 按插件隔离的 localStorage（键前缀 gargantua.plugin.<id>.）
  config: {get(key, dflt), set(key, value)},

  // 注册清理函数（等价于 destroy 里手写，多个时更省事）
  onCleanup(fn): void,
}
```

> 采样参数 `temperature` / `top_p` 由右侧开发者选项统一控制，宿主在挂载时
> 与每次改动时写进 `ctx.config`；插件 `submit` 前读
> `ctx.config.get("temperature", 0.8)` / `("topP", 0.95)` 即可。

### 3.1 `api.infer(req, handlers)`

```js
await ctx.api.infer({
  tokenIds: [0xC732, 0xC733, 0x1F],  // 纯 token 输入（与 prompt 二选一）
  prompt: "你好",                     // 文本输入，走词表编码器
  maxNew: 32,                        // 1..2048（上限由 /api/status 下发）
  temperature: 0.8,
  topP: 0.95,
}, {
  onStart({taskId}) {},
  onToken(token, n) {},              // n = 第几个（从 1 开始）
  onDone(result, summary) {},
  onError(err) {},
});
```

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

### 3.3 输入约定（默认插件）

`tokens.js` 只接受 **token 格式**，不接受自然语言。实际可粘的形态：

```
C732 C733 1F            裸十六进制 ID（可带 0x）
{t:C732}                紧凑格式
{text: C732}            可读显示格式（文本 / token 编辑器的输出）
C732|C733               步内分隔符
<5> / <C>               扩展系统 token（尖括号内是十六进制 ID）
<user> <model/> <stop>   固定名字的特殊 token（按 spec 映射到 0..5）
```

`wave:` / `visual:` / `draw:` 媒体载荷**明确拒绝**——它们是采样 / 像素向量，
不是离散 ID，硬按十六进制解析会静默产出一串错误 ID。

---

## 4. 样式约定

- 插件自带 CSS，由宿主注入/移除；**不要**改 `app.css`，也不要用主程序的
  类名（`.act` / `.card` / `.field` …）——那是宿主的东西，会变。
- 颜色、圆角、字体一律用 `app.css` 里的 CSS 变量：
  `--fg / --fg2 / --fg3 / --panel / --panel-2 / --fill / --border /
   --accent / --accent-soft / --on-accent / --track / --r / --ease`。
  这样亮/暗主题、圆角滑杆都能自动跟随。
- **不要用系统默认控件**（`<select>`、`<input type=file>`、`alert/confirm`）。
  需要下拉就自己画，需要确认就用 `ctx.api.confirm`。全站所有 `<select>`
  （含运行时动态插入的）都会被宿主换成自绘菜单，这条是项目铁律：
  界面上每一个像素都由前端代码绘制。

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
- **跨轮持久状态尚未接线**：当前 `infer.py::generate()` 每次从零开始，
  fast weight 与 KV 都不跨 HTTP 请求持久。`reset` 的实际效果是「下次重新
  加载模型」，不是「清空长期记忆」。这是架构级待办，不是插件问题。
- **纯 token 输入的解码文本不可靠**：`result.text` 只在 `prompt` 输入时有意义。
- **插件跑在主页面上下文**：它能拿到 `window` 和主程序的全局变量。这是
  本地单人工具的取舍——不做沙箱，靠约定。导入的插件代码请自行审阅。
