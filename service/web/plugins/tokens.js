// ===========================================================================
// GARGANTUA — 默认显示插件：token 对话（多轮、转换器格式）
// ---------------------------------------------------------------------------
// 这是「插件协议」的参考实现，也是当前唯一的可用插件。
//
// 交互模型（steve 2026-11-05 定案）：
//   - 像微信一样：这边（右）是我的输入，那边（左）是模型输出，多轮上下文保留。
//   - 每个 token 直接以**转换器格式**呈现：文本 token 写成 {text: B46A}（大写
//     十六进制、自界定），特殊 token 写成名字（<caps> / <user> / <model> /
//     <stop> …，对应 server/spec.py::SPECIAL_TOKENS 的 0..5）。
//   - Enter 不发送（防误触），只能点「发送」按钮。
//   - 上下文：每一轮把用户输入 + 模型输出接在 token 序列尾部，下一轮整段重发；
//     长度超 ctx_full 时截尾（滚动窗口），模型侧也会自行限长。
//
// 它只连 ctx.api.infer()，不碰任何全局状态：DOM 全在 root 里，配置走
// ctx.config（按插件隔离的 localStorage），卸载时 destroy() 会被调用。
// 协议细节见 README.md。
// ===========================================================================

const HEX = /^[0-9a-f]+$/i;
/* 架构契约头里的固定 ID（server/spec.py::SPECIAL_TOKENS），不随词表变化，
   所以插件不用查词表也能认这些名字；同时用于「转换器格式」的渲染。 */
const NAMED_IDS = {
  "<caps>": 0, "<user>": 1, "<user/>": 2,
  "<model>": 3, "<model/>": 4, "<stop>": 5,
};
// 反向表：ID → 名字（0..5 的特殊 token 渲染成名字，其余渲染成 {text: HEX}）。
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
 * 解析 token 输入——按项目自己的 token-5 文本口径，而不是只认裸 ID。
 *
 * 接受（都是 service/tokens/tok5.py 与文本 / token 编辑器实际产出的形态）：
 *   C732 C733 1F        裸十六进制 ID（可带 0x 前缀）
 *   {text: C732}        可读显示格式；{t:C732} 是它的紧凑格式
 *   C732|C733           步内分隔符
 *   <5> / <C>           扩展系统 token（尖括号里是十六进制 ID）
 *   <user> <model/> …   固定名字的特殊 token，按 spec 映射到 0..5
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
  // 不能先 push 到一个数组里——那样 "<user> {t:C732} <model/>" 会被重排成
  // [1,4,C732]，prompt 顺序就毁了。
  s = s.replace(/<([^>]*)>/g, (_, inner) => {
    const name = "<" + inner.trim().toLowerCase() + ">";
    let id = null;
    if (Object.prototype.hasOwnProperty.call(NAMED_IDS, name)) {
      id = NAMED_IDS[name];
    } else {
      const hex = inner.trim().replace(/^0x/i, "");
      if (HEX.test(hex)) {
        const n = parseInt(hex, 16);
        if (Number.isSafeInteger(n)) id = n;
      }
    }
    if (id === null) { bad.push("<" + inner + ">"); return " "; }
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
    let count = 0;            // 当前这一轮已生成的 token 数
    let tFirst = null;
    let tLast = null;
    let vocabSize = 0;
    let maxNewLimit = 2048;   // 由 /api/status.max_new_tokens 校准
    let ctxCap = 2048;        // 由 /api/infer/stats.ctx_full 校准（滚动窗口）
    let noModel = false;
    let ctxIds = [];          // 跨轮累积的 token 序列（唯一上下文真源）
    let genIds = null;        // 当前轮已生成的 token（done 后并入 ctxIds）

    root.classList.add("tk-root");

    // ---------------- 顶部：指标条 ----------------
    const head = el("div", "tk-head");
    const title = el("div", "tk-title", "token 对话");
    const metrics = el("div", "tk-metrics");
    const mTTFT = metric("TTFT");
    const mTPS = metric("tok/s");
    const mCount = metric("tokens");
    const mCtx = metric("上下文");
    const mModel = metric("model");
    metrics.append(mTTFT.box, mTPS.box, mCount.box, mCtx.box, mModel.box);
    head.append(title, metrics);
    function metric(label) {
      const box = el("span", "tk-m");
      const k = el("i", null, label);
      const v = el("b", null, "—");
      box.append(k, v);
      return {box, v};
    }

    // ---------------- 中间：对话列表 ----------------
    const chat = el("div", "tk-chat");
    const empty = el("div", "tk-empty");
    empty.append(
      el("b", null, "输入 token，开始多轮对话"),
      el("span", null,
         "这边是你的输入，那边是模型输出，上下文逐轮保留。" +
         "token 以转换器格式呈现：{text: B46A} 与 <caps> 等。" +
         "自然语言请先在「文本」页编码成 token。"));
    chat.appendChild(empty);

    // ---------------- 底部：输入区 ----------------
    const dock = el("div", "tk-dock");
    const box = el("div", "tk-input-box");
    const ta = el("textarea", "tk-input");
    ta.rows = 2;
    ta.spellcheck = false;
    ta.placeholder =
      "粘贴 token：{text: B46A} 或 C732 …（支持 <caps> <user> <model> <stop>）";
    const tools = el("div", "tk-tools");
    const hint = el("span", "tk-hint", "max_new ≤ 2048");
    const numWrap = el("label", "tk-num");
    numWrap.append(el("span", null, "max_new"));
    const num = el("input");
    num.type = "number";
    num.min = "1";
    num.max = "2048";
    num.value = String(ctx.config.get("maxNew", 32));
    numWrap.appendChild(num);
    const send = el("button", "tk-send", "发送");
    tools.append(hint, numWrap, send);
    box.append(ta, tools);
    dock.appendChild(box);

    root.append(head, chat, dock);

    // ---------------- 指标 / 会话 ----------------
    function updateHint() {
      if (noModel) {
        hint.textContent = "还没有活动模型：先去「模型」页加载一个";
        return;
      }
      hint.textContent = vocabSize
        ? `ID 0…${(vocabSize - 1).toString(16).toUpperCase()} · 词表 ${vocabSize} · max_new ≤ ${maxNewLimit}`
        : `max_new ≤ ${maxNewLimit}`;
    }
    function setMetrics() {
      mTTFT.v.textContent = tFirst == null ? "—" : fmtMs(tFirst);
      mCount.v.textContent = String(count);
      mCtx.v.textContent = String(ctxIds.length);
      mTPS.v.textContent = "—";
      if (tFirst != null && tLast != null && count > 1 && tLast > tFirst) {
        mTPS.v.textContent = ((count - 1) / ((tLast - tFirst) / 1000)).toFixed(2);
      }
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
      num.disabled = on;
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

      const maxNew = Math.max(1, Math.min(maxNewLimit, Number(num.value) || 32));
      num.value = String(maxNew);
      ctx.config.set("maxNew", maxNew);

      // 把用户输入接进上下文，并显示在「我」这一侧
      ctxIds = ctxIds.concat(ids);
      if (ctxIds.length > ctxCap) ctxIds = ctxIds.slice(-ctxCap);
      const myMsg = newMsg("me", "你");
      ids.forEach(id => myMsg.toks.appendChild(tokenSpan(id)));

      // 模型这一侧的气泡，流式填充
      const botMsg = newMsg("bot", "模型");
      const c = caret();
      botMsg.toks.appendChild(c);

      count = 0;
      tFirst = tLast = null;
      genIds = [];
      setMetrics();
      setBusy(true);
      const t0 = performance.now();
      try {
        await ctx.api.infer({
          tokenIds: ctxIds.slice(),
          maxNew,
          temperature: Number(ctx.config.get("temperature", 0.8)),
          topP: Number(ctx.config.get("topP", 0.95)),
        }, {
          onToken: (tk) => {
            if (dead) return;
            count += 1;
            genIds.push(tk.id);
            const t = tk.t_ms != null ? tk.t_ms : performance.now() - t0;
            if (tFirst == null) tFirst = t;
            tLast = t;
            botMsg.toks.insertBefore(tokenSpan(tk.id), c);
            mCount.v.textContent = String(count);
            // 先无条件清掉上一轮的 tok/s，再按需要重写
            mTPS.v.textContent = "—";
            if (tFirst != null && tLast != null && count > 1 && tLast > tFirst) {
              mTPS.v.textContent = ((count - 1) / ((tLast - tFirst) / 1000)).toFixed(2);
            }
            mTTFT.v.textContent = fmtMs(tFirst);
            scrollFollow();
          },
          onDone: (result, s) => {
            if (dead) return;
            c.remove();
            // 生成的 token 并入上下文，下一轮整段重发（滚动窗口）
            ctxIds = ctxIds.concat(genIds);
            if (ctxIds.length > ctxCap) ctxIds = ctxIds.slice(-ctxCap);
            mCtx.v.textContent = String(ctxIds.length);
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
      ta.value = "";
      ta.focus();
    }

    // 注意：Enter 不发送（防误触），只能点「发送」。textarea 里 Enter 即换行。
    send.onclick = submit;

    // ---------------- 元信息（词表大小 / 上下文上限 / 当前模型 / 生成上限） ----------------
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
        mModel.v.textContent = st.model || "未加载";
        maxNewLimit = Math.max(1, Math.min(2048, Number(st.max_new_tokens) || 2048));
        num.max = String(maxNewLimit);
        if (Number(num.value) > maxNewLimit) num.value = String(maxNewLimit);
        updateHint();
      } catch (e) {}
    })();

    ta.focus();

    return {
      /** 宿主点「清空」时调用：清掉对话、上下文与指标（后端 reset 由宿主负责）。 */
      reset() {
        chat.innerHTML = "";
        chat.appendChild(empty);
        ctxIds = [];
        genIds = null;
        count = 0;
        tFirst = tLast = null;
        setMetrics();
        setBusy(false);
        ta.value = "";
      },
      /** 切插件 / 离开页面：停掉回调，别让迟到的 token 碰已卸载的 DOM。 */
      destroy() {
        dead = true;
        root.classList.remove("tk-root");
      },
    };
  },
};
