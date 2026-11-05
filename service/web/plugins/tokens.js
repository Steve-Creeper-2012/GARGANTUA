// ===========================================================================
// GARGANTUA — 默认显示插件：纯 token 流
// ---------------------------------------------------------------------------
// 这是「插件协议」的参考实现，也是当前唯一的可用插件。
//
// 它只做三件事：
//   1. 把用户输入解析成 token ID（十六进制；只收 ID，不解析自然语言）；
//   2. 调 ctx.api.infer() 发起流式推理，来一个 token 追加一个 chip；
//   3. 实时显示 TTFT / tok/s / token 数。
//
// 它**不碰**任何全局状态：DOM 全在 root 里，配置走 ctx.config（按插件
// 隔离的 localStorage），卸载时 destroy() 会被调用。协议细节见 README.md。
// ===========================================================================

const HEX = /^[0-9a-f]+$/i;
/* 架构契约头里的固定 ID（server/spec.py::SPECIAL_TOKENS），不随词表变化，
   所以插件不用查词表也能认这些名字。 */
const NAMED_IDS = {
  "<caps>": 0, "<user>": 1, "<user/>": 2,
  "<model>": 3, "<model/>": 4, "<stop>": 5,
};

/** 造元素：不拼 innerHTML，用户数据一律 textContent。 */
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
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
  name: "纯 token 显示",

  mount(root, ctx) {
    let dead = false;
    let busy = false;
    let turn = null;          // 当前这一轮的 DOM 引用
    let count = 0;
    let tFirst = null;
    let tLast = null;
    let vocabSize = 0;
    let maxNewLimit = 2048;     // 由 /api/status.max_new_tokens 校准
    let noModel = false;

    root.classList.add("tk-root");

    // ---------------- 顶部：指标条 ----------------
    const head = el("div", "tk-head");
    const title = el("div", "tk-title", "token 流");
    const metrics = el("div", "tk-metrics");
    const mTTFT = metric("TTFT");
    const mTPS = metric("tok/s");
    const mCount = metric("tokens");
    const mModel = metric("model");
    metrics.append(mTTFT.box, mTPS.box, mCount.box, mModel.box);
    head.append(title, metrics);
    function metric(label) {
      const box = el("span", "tk-m");
      const k = el("i", null, label);
      const v = el("b", null, "—");
      box.append(k, v);
      return {box, v};
    }

    // ---------------- 中间：会话记录 ----------------
    const log = el("div", "tk-log");
    const empty = el("div", "tk-empty");
    empty.append(
      el("b", null, "输入 token ID 开始"),
      el("span", null,
         "这个插件只认 token：十六进制 ID，空格或换行分隔。" +
         "自然语言请先在「文本」页编码成 token。"));
    log.appendChild(empty);

    // ---------------- 底部：输入区 ----------------
    const dock = el("div", "tk-dock");
    const box = el("div", "tk-input-box");
    const ta = el("textarea", "tk-input");
    ta.rows = 2;
    ta.spellcheck = false;
    ta.placeholder = "C732 C733 1F …（十六进制 token ID）";
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

    root.append(head, log, dock);

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
      // 先无条件清掉上一轮的 tok/s：旧实现在 tFirst==null（新轮 / reset）
      // 和「多个 token 同一毫秒」两种情况下都不写它，指标会一直残留。
      mTPS.v.textContent = "—";
      if (tFirst != null && tLast != null && count > 1 && tLast > tFirst) {
        mTPS.v.textContent = ((count - 1) / ((tLast - tFirst) / 1000)).toFixed(2);
      }
    }

    function newTurn(promptIds) {
      empty.remove();
      turn = el("div", "tk-turn");
      const inRole = el("div", "tk-role", `prompt · ${promptIds.length} tok`);
      const inChips = el("div", "tk-chips");
      promptIds.forEach(id => inChips.appendChild(chip(id, false)));
      const outRole = el("div", "tk-role out", "output · 0 tok");
      const outChips = el("div", "tk-chips out");
      turn.append(inRole, inChips, outRole, outChips);
      log.appendChild(turn);
      log.scrollTop = log.scrollHeight;
      return {outRole, outChips};
    }

    function chip(id, isOut) {
      const c = el("span", "tk-chip" + (isOut ? " out" : ""),
                   id.toString(16).toUpperCase());
      c.title = `id ${id}（十进制）`;
      return c;
    }

    function scrollFollow() {
      const nearBottom =
        log.scrollHeight - log.scrollTop - log.clientHeight < 80;
      if (nearBottom) log.scrollTop = log.scrollHeight;
    }

    function setBusy(on) {
      busy = on;
      send.disabled = on;
      send.textContent = on ? "生成中…" : "发送";
      ta.disabled = on;
      num.disabled = on;
      root.classList.toggle("tk-busy", on);
    }

    // ---------------- 发送 ----------------
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

      const view = newTurn(ids);
      count = 0;
      tFirst = tLast = null;
      setMetrics();
      setBusy(true);
      const t0 = performance.now();
      try {
        await ctx.api.infer({
          tokenIds: ids,
          maxNew,
          temperature: Number(ctx.config.get("temperature", 0.8)),
          topP: Number(ctx.config.get("topP", 0.95)),
        }, {
          onToken: (tk) => {
            if (dead) return;
            count += 1;
            const t = tk.t_ms != null ? tk.t_ms : performance.now() - t0;
            if (tFirst == null) tFirst = t;
            tLast = t;
            view.outChips.appendChild(chip(tk.id, true));
            view.outRole.textContent = `output · ${count} tok`;
            setMetrics();
            scrollFollow();
          },
          onDone: (result, s) => {
            if (dead) return;
            view.outRole.textContent =
              `output · ${count} tok` +
              (result && result.archive ? ` · 存档 ${result.archive}` : "");
            if (s && s.elapsed_ms != null) {
              const el2 = el("div", "tk-note",
                `总耗时 ${fmtMs(s.elapsed_ms)} · TTFT ${fmtMs(s.ttft_ms)}`);
              turn.appendChild(el2);
            }
          },
          onError: (e) => {
            if (dead) return;
            turn.appendChild(el("div", "tk-err", "⚠ " + e.message));
          },
        });
      } catch (e) {
        if (!dead && e.message !== "已取消") {
          turn.appendChild(el("div", "tk-err", "⚠ " + e.message));
        }
      } finally {
        // 任何路径都要解锁：done / error / 超时 / 卸载取消，不能留下
        // 一个永远灰着的「生成中…」按钮。
        if (!dead) setBusy(false);
      }
    }

    send.onclick = submit;
    ta.addEventListener("keydown", ev => {
      if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); submit(); }
    });

    // ---------------- 元信息（词表大小 / 当前模型 / 生成上限） ----------------
    (async () => {
      try {
        const s = await ctx.api.stats();
        if (dead) return;
        vocabSize = Number(s.vocab_size) || 0;
        updateHint();
      } catch (e) {}
      try {
        const st = await ctx.api.status();
        if (dead) return;
        noModel = !st.model;
        mModel.v.textContent = st.model || "未加载";
        // 上限以后端为准（common.MAX_NEW_TOKENS），不再前端写死 512
        maxNewLimit = Math.max(1, Math.min(2048,
          Number(st.max_new_tokens) || 2048));
        num.max = String(maxNewLimit);
        if (Number(num.value) > maxNewLimit) num.value = String(maxNewLimit);
        updateHint();
      } catch (e) {}
    })();

    ta.focus();

    return {
      /** 宿主点「清空」时调用：清掉会话与指标（后端 reset 由宿主负责）。 */
      reset() {
        log.innerHTML = "";
        log.appendChild(empty);
        turn = null;
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
