// ===========================================================================
// GARGANTUA — 默认显示插件：token 对话（多轮、转换器格式）
// ---------------------------------------------------------------------------
// 这是「插件协议」的参考实现，也是当前唯一的可用插件。
//
// 交互模型（steve 2026-11-05 定案）：
//   - 像微信一样：这边（右）是我的输入，那边（左）是模型输出，多轮上下文保留。
//   - 每个 token 直接以**转换器格式**呈现：文本 token 写成 {text: B46A}（大写
//     十六进制、自界定），特殊 token 写成名字（<caps> / [user] / [model] /
//     <stop> …，对应契约头 core 3 + boundary 6 的 ID 0..8）。
//   - Enter 不发送（防误触），只能点「发送」按钮。
//   - 顶部状态 / 指标条整条归宿主（#infer-statusbar）：插件不再自建顶部指标条、
//     不再显示状态、不再处理 onPhase、不再上报指标。root 里只有对话流 + 输入区。
//
// 会话 / 上下文（2026-11-05 起，后端按 session 持久化 KV 与 fast weight）：
//   - 历史消息由宿主放在 ctx.session.messages，mount 时恢复成气泡；本轮通过
//     ctx.session.append("me"|"bot", ids) 落盘，插件不再自己长期持有会话状态。
//   - kvEnabled（ctx.session.config.get("kvEnabled", true)）决定本轮发多少：
//       开启  → 只发本轮用户输入 ids（增量），服务端按 session 复用 KV；
//       关闭  → 发完整上下文（累计序列 + 本轮 ids），服务端重算整段。
//     UI 始终显示完整多轮对话；ctxIds 是本地累计序列，仅供 kvEnabled=false
//     时作为完整上下文发送，与「发给服务端的长度」解耦。宿主自动给 req 注入
//     session_id / kv_enabled。
//   - maxNew 来自开发者选项 ctx.session.config.get("maxNew", 32)，插件里
//     不再有 max_new 可编辑控件。
//
// 它只连 ctx.api.infer()，DOM 全在 root 里；按插件隔离的配置走 ctx.config，
// 卸载时 destroy() 会被调用。协议细节见 README.md。
// ===========================================================================

const HEX = /^[0-9a-f]+$/i;
/* 架构契约头里的固定 ID（server/spec.py 的 CORE_TOKENS + BOUNDARY_TOKENS）。
   前 12 个 ID 由 spec 冻结，不随词表 / slot_count 变化，所以插件不用查词表
   也能认这些名字；同时用于「转换器格式」的渲染。
   2026-11-07 修正（base1.3.0）：原表停留在 1.0 时代（<stop>=5、[sys/]=7），
   1.2.0 契约头重排后已全部错位，会把 <stop> 显示成 [user]、[user] 显示成
   <stop>。现行布局：core 0..2 = <caps>/<stop>/<draw:end>，
   boundary 3..8 = [user]/[user/]/[model]/[model/]/[sys]/[sys/]。
   9..11 的 { } | 是结构字符，不作为 token 名字渲染。
   注意：这里只覆盖 core/boundary；typst 结构 token（<H1> 等）不在本表，
   需用裸十六进制 ID 输入，或从 /api/agent/spec + typst_map.json 查。 */
const NAMED_IDS = {
  "<caps>": 0, "<stop>": 1, "<draw:end>": 2,
  "[user]": 3, "[user/]": 4,
  "[model]": 5, "[model/]": 6,
  "[sys]": 7, "[sys/]": 8,
};
// 反向表：ID → 名字（0..8 的头部 token 渲染成名字，其余渲染成 {text: HEX}）。
const ID_TO_NAME = {};
for (const [name, id] of Object.entries(NAMED_IDS)) ID_TO_NAME[id] = name;

/** 造元素：不拼 innerHTML，用户数据一律 textContent。 */
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

/**
 * 把一个 token ID 渲染成转换器格式的 DOM 片段（安全：textContent）。
 *   id 0  → <caps>      （大写标记，名字来自 SPECIAL_TOKENS）
 *   其它  → {text: B46A}（大写十六进制、自界定，与 tok5 可读格式一致）
 */
function tokenSpan(id) {
  const name = ID_TO_NAME[id];
  const s = el("span", "tk-tok" + (name ? " sp" : ""));
  s.textContent = name ? name : "{text: " + id.toString(16).toUpperCase() + "}";
  s.title = name
    ? `特殊 token ${id}（${name}）`
    : `token id ${id}（十进制）`;
  return s;
}

/**
 * 解析 token 输入——按项目自己的 token-6 文本口径，而不是只认裸 ID。
 *
 * 接受（都是 service/tokens/tok5.py 与文本 / token 编辑器实际产出的形态）：
 *   C732 C733 1F        裸十六进制 ID（可带 0x 前缀）
 *   {text: C732}        可读显示格式；{t:C732} 是它的紧凑格式
 *   C732|C733           步内分隔符
 *   <5> / <C>           扩展系统 token（尖括号里是十六进制 ID）
 *   [user] [model/] …   固定名字的特殊 token（[] 边界），按 spec 映射到 3..8
 *
 * 明确拒绝 wave / visual / draw 媒体载荷：它们是采样 / 像素向量，不是
 * 离散 ID，硬按十六进制解析会静默产出一串完全错误的 ID。宁可报错。
 */
function parseTokenInput(text) {
  const ids = [];
  const bad = [];
  let s = String(text || "");
  if (/\b(?:wave|visual|draw)\s*:/i.test(s) ||
      /(?:^|[{|,\s])[wv]\s*:/i.test(s)) {
    return {ids, bad: ["<媒体载荷>"], media: true};
  }
  // 扩展 token / 特殊名字：就地换成 ID 文本，交给下面同一次顺序扫描解析。
  // 不能先 push 到一个数组里——那样 "[user] {t:C732} [model/]" 会被重排成
  // [1,4,C732]，prompt 顺序就毁了。
  // 正则同时认 <...>（工具 / 扩展 token、<caps>/<stop>）与 [...]（[] 边界），
  // 按匹配到的包裹形态还原名字，避免把 [] 名字误当成 <> 名字。
  s = s.replace(/<([^>]*)>|\[([^\]]*)\]/g, (m, angleInner, bracketInner) => {
    const isBracket = bracketInner !== undefined;
    const inner = isBracket ? bracketInner : angleInner;
    const name = isBracket
      ? "[" + inner.trim().toLowerCase() + "]"
      : "<" + inner.trim().toLowerCase() + ">";
    let id = null;
    if (Object.prototype.hasOwnProperty.call(NAMED_IDS, name)) {
      id = NAMED_IDS[name];
    } else if (!isBracket) {
      // 只有尖括号形态允许写十六进制 ID（<C> / <5>）；[] 是纯名字。
      const hex = inner.trim().replace(/^0x/i, "");
      if (HEX.test(hex)) {
        const n = parseInt(hex, 16);
        if (Number.isSafeInteger(n)) id = n;
      }
    }
    if (id === null) { bad.push(m); return " "; }
    return " " + id.toString(16).toUpperCase() + " ";
  });
  // text: / t: 前缀（显示格式与紧凑格式），随后剥掉结构字符 { } | ,
  s = s.replace(/\b(?:text|t)\s*:/gi, " ").replace(/[{}|,]/g, " ");
  for (const raw of s.split(/\s+/)) {
    if (!raw) continue;
    const hex = raw.replace(/^0x/i, "");
    if (!HEX.test(hex)) { bad.push(raw); continue; }
    const id = parseInt(hex, 16);
    if (!Number.isSafeInteger(id)) { bad.push(raw); continue; }
    ids.push(id);
  }
  return {ids, bad, media: false};
}

function fmtMs(ms) {
  if (ms == null || !Number.isFinite(ms)) return "—";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export default {
  id: "tokens",
  name: "token 对话",

  mount(root, ctx) {
    let dead = false;
    let busy = false;
    let vocabSize = 0;
    let maxNewLimit = 2048;   // /api/status.max_new_tokens：clamp 开发者选项的 maxNew + hint 上限
    let ctxCap = 2048;        // /api/infer/stats.ctx_full：滚动窗口，只约束发给服务端的完整上下文
    let noModel = false;      // 仅供 hint 提示「先加载模型」
    // 本地累计的完整对话 token 序列（滚动窗口，截尾到 ctxCap）。
    // 用途：kvEnabled=false 时作为「完整上下文」发给服务端；UI 不靠它渲染（气泡独立渲染）。
    let ctxIds = [];
    let genIds = null;        // 当前轮已生成的 token（done 后并入 ctxIds / 落盘）

    root.classList.add("tk-root");

    // ---------------- 中间：对话列表 ----------------
    const chat = el("div", "tk-chat");
    const empty = el("div", "tk-empty");
    empty.append(
      el("b", null, "输入 token，开始多轮对话"),
      el("span", null,
         "这边是你的输入，那边是模型输出，上下文逐轮保留。" +
         "token 以转换器格式呈现：{text: B46A} 与 <caps> / [user] 等。" +
         "自然语言请先在「文本」页编码成 token。"));
    chat.appendChild(empty);

    // ---------------- 底部：输入区 ----------------
    const dock = el("div", "tk-dock");
    const box = el("div", "tk-input-box");
    const ta = el("textarea", "tk-input");
    // (c) 定位结论：原来这里用 rows=2 把 textarea 钉死成两行固定高度，输入多行
    // 后内容溢出、textarea 内部冒出一条竖直滚动条；而 .tk-input-box 的圆角边框
    // + 10px/12px 内边距把这条（app.css:1463 的 9px 全局）滚动条压在窄输入区里，
    // 看起来就像「输入框的外围遮挡了输入的竖条」。
    // 修法：textarea 改 auto-grow（随内容增高，正常不出现内部滚动条），CSS 给
    // max-height 兜底，超出才用 6px 细滚动条（thumb 用 var(--fill-2)）；
    // .tk-input-box 保持 overflow:visible 不裁切。详见 tokens.css。
    ta.spellcheck = false;
    ta.placeholder =
      "粘贴 token：{text: B46A} 或 C732 …（支持 <caps> [user] [model] <stop>）";
    ta.addEventListener("input", autoGrow);
    const tools = el("div", "tk-tools");
    // (d) max_new 已移到开发者选项（#infer-maxnew / ctx.session.config），插件内
    // 不再有可编辑控件；hint 只保留词表范围与服务端上限等只读提示，不显示 model 名。
    const hint = el("span", "tk-hint", "max_new ≤ 2048");
    const send = el("button", "tk-send", "发送");
    tools.append(hint, send);
    box.append(ta, tools);
    dock.appendChild(box);

    root.append(chat, dock);      // 状态 / 指标条由宿主绘制，插件只有这两块

    // ---------------- 提示 / 布局 ----------------
    function updateHint() {
      if (noModel) {
        hint.textContent = "还没有活动模型：先去「模型」页加载一个";
        return;
      }
      hint.textContent = vocabSize
        ? `ID 0…${(vocabSize - 1).toString(16).toUpperCase()} · 词表 ${vocabSize} · max_new ≤ ${maxNewLimit}`
        : `max_new ≤ ${maxNewLimit}`;
    }

    /** (c) textarea 随内容增高；超过 CSS 的 max-height 才由细滚动条兜底。 */
    function autoGrow() {
      if (dead) return;
      ta.style.height = "auto";
      ta.style.height = ta.scrollHeight + "px";
    }

    function scrollFollow() {
      const nearBottom =
        chat.scrollHeight - chat.scrollTop - chat.clientHeight < 80;
      if (nearBottom) chat.scrollTop = chat.scrollHeight;
    }

    /** 新建一条消息（气泡），返回 {bubble, toks}。who: "me" | "bot"。 */
    function newMsg(who, label) {
      empty.remove();
      const msg = el("div", "tk-msg " + who);
      const whoEl = el("div", "tk-who", label);
      const bubble = el("div", "tk-bubble");
      const toks = el("div", "tk-toks");
      bubble.appendChild(toks);
      msg.append(whoEl, bubble);
      chat.appendChild(msg);
      chat.scrollTop = chat.scrollHeight;
      return {bubble, toks};
    }

    // 生成中的光标（尾巴那个 ▍），done / error 时摘掉
    function caret() {
      const c = el("span", "tk-caret", "▍");
      return c;
    }

    function setBusy(on) {
      busy = on;
      send.disabled = on;
      send.textContent = on ? "生成中…" : "发送";
      ta.disabled = on;
      root.classList.toggle("tk-busy", on);
    }

    // ---------------- 发送（唯一入口：按钮） ----------------
    async function submit() {
      if (busy) return;
      const parsed = parseTokenInput(ta.value);
      if (parsed.media) {
        ctx.api.toast("只收离散 token ID；wave / visual / draw 媒体载荷请先编码成 ID", true);
        return;
      }
      if (parsed.bad.length) {
        ctx.api.toast(`只收 token ID，这些看不懂：${parsed.bad.slice(0, 4).join(" ")}`,
                      true);
        return;
      }
      const ids = parsed.ids;
      if (!ids.length) { ctx.api.toast("先输入至少一个 token ID", true); return; }
      if (vocabSize && ids.some(i => i >= vocabSize)) {
        ctx.api.toast(`token ID 超出词表范围（< ${vocabSize}）`, true);
        return;
      }

      // (d) max_new 从开发者选项读（per-session），插件不再持有可编辑控件；
      // maxNewLimit 只用于把它 clamp 到服务端上限。
      const maxNew = Math.max(1, Math.min(maxNewLimit,
        Number(ctx.session.config.get("maxNew", 32)) || 32));

      // (KV 契约) 本轮发给服务端的 token 长度取决于 KV 复用开关：
      //  - kvEnabled（默认）：后端按 session 持久化 KV / fast weight，只发本轮
      //    用户输入 ids（增量），历史不重复发送；
      //  - kvEnabled=false：发完整上下文（累计序列 + 本轮 ids），服务端重算整段。
      // 宿主会自动给 req 注入 session_id 与 kv_enabled，插件不要自己传。
      const kvEnabled = ctx.session.config.get("kvEnabled", true) !== false;

      // 本地累计：把本轮输入并入完整对话序列（滚动窗口）。
      // ctxIds 只用于 kvEnabled=false 的完整上下文，UI 不靠它渲染。
      ctxIds = ctxIds.concat(ids);
      if (ctxIds.length > ctxCap) ctxIds = ctxIds.slice(-ctxCap);
      const reqIds = kvEnabled ? ids.slice() : ctxIds.slice();

      // (e) 会话落盘：本轮用户输入
      ctx.session.append("me", ids);

      // UI：只渲染本轮输入（历史已在 mount 时从 ctx.session.messages 恢复）
      const myMsg = newMsg("me", "你");
      ids.forEach(id => myMsg.toks.appendChild(tokenSpan(id)));

      // 模型这一侧的气泡，流式填充
      const botMsg = newMsg("bot", "模型");
      const c = caret();
      botMsg.toks.appendChild(c);

      genIds = [];
      setBusy(true);
      try {
        await ctx.api.infer({
          tokenIds: reqIds,
          maxNew,
          temperature: Number(ctx.config.get("temperature", 0.8)),
          topP: Number(ctx.config.get("topP", 0.95)),
        }, {
          // 状态 / 指标由宿主统一处理：插件不传 onPhase、不上报指标
          onToken: (tk) => {
            if (dead) return;
            genIds.push(tk.id);
            botMsg.toks.insertBefore(tokenSpan(tk.id), c);
            scrollFollow();
          },
          onDone: (result, s) => {
            if (dead) return;
            c.remove();
            // 生成的 token 并入本地累计（滚动窗口）
            const out = genIds || [];
            ctxIds = ctxIds.concat(out);
            if (ctxIds.length > ctxCap) ctxIds = ctxIds.slice(-ctxCap);
            // (e) 会话落盘：本轮模型输出
            ctx.session.append("bot", out);
            if (s && s.elapsed_ms != null) {
              botMsg.bubble.appendChild(el("div", "tk-note",
                `总耗时 ${fmtMs(s.elapsed_ms)} · TTFT ${fmtMs(s.ttft_ms)}`));
            }
            if (result && result.archive) {
              botMsg.bubble.appendChild(el("div", "tk-note",
                `存档 ${result.archive}`));
            }
            scrollFollow();
          },
          onError: (e) => {
            if (dead) return;
            c.remove();
            botMsg.bubble.appendChild(el("div", "tk-err", "⚠ " + e.message));
          },
        });
      } catch (e) {
        if (!dead && e.message !== "已取消") {
          c.remove();
          botMsg.bubble.appendChild(el("div", "tk-err", "⚠ " + e.message));
        }
      } finally {
        // 任何路径都要解锁：done / error / 超时 / 卸载取消，不能留下
        // 一个永远灰着的「生成中…」按钮。
        genIds = null;
        if (!dead) setBusy(false);
      }
      // (c) 清空后把输入框收回初始高度；destroy 之后不再碰 DOM。
      if (!dead) {
        ta.value = "";
        autoGrow();
        ta.focus();
      }
    }

    // 注意：Enter 不发送（防误触），只能点「发送」。textarea 里 Enter 即换行。
    send.onclick = submit;

    // ---------------- 元信息（词表大小 / 上下文上限 / 生成上限 / 活动模型） ----------------
    (async () => {
      try {
        const s = await ctx.api.stats();
        if (dead) return;
        vocabSize = Number(s.vocab_size) || 0;
        ctxCap = Math.max(64, Number(s.ctx_full) || 2048);
        updateHint();
      } catch (e) {}
      try {
        const st = await ctx.api.status();
        if (dead) return;
        noModel = !st.model;
        maxNewLimit = Math.max(1, Math.min(2048, Number(st.max_new_tokens) || 2048));
        updateHint();
      } catch (e) {}
    })();

    // (e) 会话历史：会话切换会重新 mount 插件，宿主把历史消息放进
    // ctx.session.messages；这里按角色恢复成气泡（me→右 / bot→左），
    // token 仍用 tokenSpan 渲染；同时把历史并入本地累计序列。
    const history = (ctx.session && ctx.session.messages) || [];
    for (const m of history) {
      const role = m.role === "me" ? "me" : "bot";
      const ids = Array.isArray(m.token_ids) ? m.token_ids : [];
      const msg = newMsg(role, role === "me" ? "你" : "模型");
      ids.forEach(id => msg.toks.appendChild(tokenSpan(id)));
      ctxIds = ctxIds.concat(ids);
    }
    if (ctxIds.length > ctxCap) ctxIds = ctxIds.slice(-ctxCap);

    autoGrow();                      // (c) 初始高度
    ta.focus();

    return {
      /** 宿主点「清空」时调用：清掉对话、上下文与输入（后端 reset 由宿主负责）。 */
      reset() {
        chat.innerHTML = "";
        chat.appendChild(empty);
        ctxIds = [];
        genIds = null;
        setBusy(false);
        ta.value = "";
        autoGrow();
      },
      /** 切插件 / 离开页面：停掉回调，别让迟到的 token 碰已卸载的 DOM。 */
      destroy() {
        dead = true;
        root.classList.remove("tk-root");
      },
    };
  },
};
