// @name Infer-Plug v1.0.0
// @desc 新一代正常文字交互插件：文字与文件胶囊在同一行无缝内联、+ 上传 / 拖放、
//       流式文字与 agent 工具徽标、DRAW 图元 SVG 画布实时预览 + 可拖拽时间轴。
//       依赖 /api/agent/spec 下发的 token-6（base1.3.0）契约头与图元布局，
//       不抄任何数值常量。128 个编号槽位渲染成「槽位 N」徽标（语义自己定义、
//       执行先全不接）。协议见 service/web/plugins/README.md，契约头全表见
//       service/web/plugins/AGENT_TOKENS.md。
// ===========================================================================
// 与默认插件（tokens.js）的区别：
//   - 输入用 contenteditable div（不是 textarea），文字与文件 chip 内联拼接；
//   - 用户看到的是「正常文字」，不是 {text: HEX} 转换器格式；
//   - 文件 chip 由 /api/agent/file 返回 token_ids / media，chip.dataset 只放索引，
//     真实数据存在 JS Map 里（绝不把数组塞进 DOM，也绝不 innerHTML 拼文件名）；
//   - 输出侧识别 DRAW 区 token → 实时画 SVG 图元 + 时间轴回放；
//   - agent / 特殊 token 渲染成小徽标，工具调用渲染成工具徽标；
//   - 顶部状态条归宿主：本插件不建指标条、不处理 onPhase。
// ===========================================================================

const SVG_NS = "http://www.w3.org/2000/svg";
const ZW = "\u200B";                              // 零宽字符：chip 之后承载光标
const ZW_RE = /[\u200B\uFEFF\u200C\u200D]/g;     // 提交时从文本里剔除
const MEDIA_KINDS = new Set(["image", "audio", "video"]);
const BLOCK_TAGS = new Set(["DIV", "P", "LI"]);

/** 造普通元素：用户数据一律 textContent，绝不 innerHTML 拼接。 */
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}
/** 造 SVG 元素（图标 / 画布都用 DOM API，不拼字符串）。 */
function svgEl(tag, attrs) {
  const n = document.createElementNS(SVG_NS, tag);
  if (attrs) for (const k in attrs) {
    if (attrs[k] != null) n.setAttribute(k, String(attrs[k]));
  }
  return n;
}
function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
function stripZW(s) { return String(s == null ? "" : s).replace(ZW_RE, ""); }
function cleanName(n) { return String(n == null ? "" : n).replace(/[<>\[\]\/]/g, ""); }
function isTokenName(s) { return /^<[^>]*>$/.test(s) || /^\[[^\]]*\]$/.test(s); }
function fmtMs(ms) {
  if (ms == null || !Number.isFinite(ms)) return "—";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
}

/** 蓝色文件小图标（圆角文档 + 折角）。currentColor 由 .ip-file-icon 染成 accent。 */
function fileIcon() {
  const s = svgEl("svg", {viewBox: "0 0 16 16", width: "13", height: "13",
                          "aria-hidden": "true"});
  s.appendChild(svgEl("path", {
    d: "M3.6 1.3h5.1l3.9 3.9v9.4a.9.9 0 0 1-.9.9H3.6a.9.9 0 0 1-.9-.9V2.2a.9.9 0 0 1 .9-.9z",
    fill: "currentColor",
  }));
  s.appendChild(svgEl("path", {
    d: "M8.7 1.3v3.9h3.9", fill: "none",
    stroke: "rgba(255,255,255,.55)", "stroke-width": "1", "stroke-linejoin": "round",
  }));
  return s;
}
/** 发送按钮的上箭头（内联 SVG，stroke=currentColor）。 */
function sendArrow() {
  const s = svgEl("svg", {viewBox: "0 0 24 24", width: "18", height: "18",
                          fill: "none", stroke: "currentColor", "stroke-width": "2.2",
                          "stroke-linecap": "round", "stroke-linejoin": "round",
                          "aria-hidden": "true"});
  s.appendChild(svgEl("path", {d: "M12 19V5"}));
  s.appendChild(svgEl("path", {d: "M6 11l6-6 6 6"}));
  return s;
}
function iconPlay() {
  const s = svgEl("svg", {viewBox: "0 0 16 16", width: "12", height: "12",
                          "aria-hidden": "true"});
  s.appendChild(svgEl("path", {d: "M4 2.5 L13 8 L4 13.5 Z", fill: "currentColor"}));
  return s;
}
function iconPause() {
  const s = svgEl("svg", {viewBox: "0 0 16 16", width: "12", height: "12",
                          "aria-hidden": "true"});
  s.appendChild(svgEl("rect", {x: 4, y: 3, width: 3, height: 10, fill: "currentColor"}));
  s.appendChild(svgEl("rect", {x: 9, y: 3, width: 3, height: 10, fill: "currentColor"}));
  return s;
}

/** core 区三个机制单标记的中文短标签（base1.3.0 起不再有预定义工具）。 */
const FRIENDLY = {
  "<caps>": "大写", "<stop>": "停止", "<draw:end>": "绘图结束",
};

function badgeLabel(info) {
  if (!info) return "token";
  // slot 区 128 个编号槽位：直接显示编号（语义由使用者自己定义）。
  if (info.category === "slot") return "槽位 " + info.slot;
  if (FRIENDLY[info.name]) return FRIENDLY[info.name];
  if (info.category === "core" || info.category === "special") return info.name;
  return info.description || cleanName(info.name);
}
/** 自动 agent 最多续跑几轮工具结果，防止模型反复调用工具形成死循环。 */
const MAX_AGENT_ROUNDS = 3;

function toolBadgeText(t) {
  if (t == null) return "⚙ 工具";
  if (typeof t === "string") return "⚙ " + t;
  const name = t.name || t.tool || t.tool_id || "工具";
  let s = "⚙ " + name;
  if (t.result != null) {
    const r = typeof t.result === "object" ? JSON.stringify(t.result) : String(t.result);
    s += " · " + r.slice(0, 48);
  }
  return s;
}

export default {
  id: "infer-plug",
  name: "Infer-Plug v1.0.0",

  mount(root, ctx) {
    // ---------------------------------------------------------------- 状态
    let dead = false;
    let busy = false;
    let spec = null;                 // /api/agent/spec 的完整载荷
    let idInfo = new Map();          // token id → {name, category, description}
    let specPromise = null;
    let ctxCap = 2048;
    let maxNewLimit = 2048;
    let ctxIds = [];                 // kvEnabled=false 时用的完整上下文
    let genIds = null;               // 本轮已生成 id
    let roundSeq = 0;                // 轮次：忽略迟到的异步回调
    let chipSeq = 0;                 // chip 索引自增
    const chipMap = new Map();       // idx → {name, nameEl, chipEl, token_ids, media, kind, pending}
    const listeners = [];            // destroy 时统一解绑
    let curDraw = null;              // 当前活动 DRAW 解析状态
    // 本轮 bot 气泡的渲染目标（submit 时重置）
    let botBody = null, caret = null, curTextNode = null;

    const alive = () => !dead && !(ctx.signal && ctx.signal.aborted);
    function on(target, ev, fn, opts) {
      target.addEventListener(ev, fn, opts);
      listeners.push([target, ev, fn, opts]);
    }

    root.classList.add("ip-root");

    // ---------------------------------------------------------------- DOM
    const chat = el("div", "ip-chat");
    const empty = el("div", "ip-empty");
    empty.append(
      el("b", null, "输入文字，开始对话"),
      el("span", null,
         "文字与文件胶囊在同一行无缝内联；点左下角 + 或把文件拖进来添加附件，" +
         "Enter 换行、点右下角发送。模型输出图元时会出现可拖拽的画布时间轴。"));
    chat.appendChild(empty);

    const dock = el("div", "ip-dock");
    const box = el("div", "ip-input-box");
    const add = el("button", "ip-add");
    add.type = "button";
    add.title = "添加文件";
    add.setAttribute("aria-label", "添加文件");
    const input = el("div", "ip-input");
    input.contentEditable = "true";
    input.spellcheck = false;
    input.dataset.ph = "输入文字，点 + 添加文件…";
    const send = el("button", "ip-send");
    send.type = "button";
    send.title = "发送";            // title="发送"
    send.setAttribute("aria-label", "发送");
    send.appendChild(sendArrow());
    box.append(add, input, send);
    dock.appendChild(box);

    // 隐藏的文件输入：动态创建（官方内置插件允许），不出现在静态 HTML 骨架里。
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.multiple = true;
    fileInput.className = "ip-file-input";
    fileInput.accept = "*/*";          // 等 /api/agent/spec 的 file_types 下来再收紧
    dock.appendChild(fileInput);

    root.append(chat, dock);        // 顶部状态条由宿主绘制，插件只有这两块

    // ---------------------------------------------------------------- spec
    function buildIdInfo(s) {
      const m = new Map();
      const put = (id, name, category, description, slot) => {
        const n = Number(id);
        if (Number.isInteger(n) && name) {
          m.set(n, {name: String(name), category: category || "",
                    description: description || "", slot: slot ?? null});
        }
      };
      const SPECIAL_NAME = {
        caps: "<caps>", user: "[user]", user_end: "[user/]",
        model: "[model]", model_end: "[model/]", stop: "<stop>",
        sys: "[sys]", sys_end: "[sys/]",
      };
      if (s.special) {
        for (const [k, v] of Object.entries(s.special)) {
          put(v, SPECIAL_NAME[k] || k, "special", "");
        }
      }
      // core 三个机制单标记 + 128 个编号槽位（编号直接显示）。
      if (Array.isArray(s.core)) {
        for (const c of s.core) put(c && c.id, c && c.name, "core", "");
      }
      if (Array.isArray(s.slots)) {
        for (const t of s.slots) {
          put(t && t.id, t && t.name, "slot",
              t && (t.custom_name || t.description), t && t.slot);
        }
      }
      // 旧字段保留兜底（后端已下发空数组）。
      for (const arr of [s.boundaries, s.formats, s.tools]) {
        if (Array.isArray(arr)) {
          for (const t of arr) put(t && t.id, t && t.name, t && t.category, t && t.description);
        }
      }
      return m;
    }
    function applySpec(s) {
      if (!s) return;
      spec = s;
      idInfo = buildIdInfo(s);
      if (Array.isArray(s.file_types)) {
        fileInput.accept = s.file_types.map(ft => ft.ext).filter(Boolean).join(",");
      }
    }
    function loadSpec() {
      if (specPromise) return specPromise;
      specPromise = (async () => {
        try {
          const s = await ctx.api.agentSpec();
          if (!alive()) return null;
          applySpec(s);
          return s;
        } catch (e) {
          // 后端未就绪时静默降级：文字仍可用，徽标 / 画布暂不可用。
          if (alive() && typeof console !== "undefined") {
            console.warn("Infer-Plug: /api/agent/spec 不可用", e);
          }
          return null;
        }
      })();
      return specPromise;
    }
    loadSpec();
    (async () => {
      try { const s = await ctx.api.stats(); if (!alive()) return;
            ctxCap = Math.max(64, Number(s.ctx_full) || 2048); } catch (e) {}
      try { const st = await ctx.api.status(); if (!alive()) return;
            maxNewLimit = Math.max(1, Math.min(2048, Number(st.max_new_tokens) || 2048)); } catch (e) {}
    })();

    // ---------------------------------------------------------------- 输入辅助
    function ensureCaret() {
      input.focus();
      const sel = window.getSelection();
      if (sel && sel.rangeCount) {
        const r = sel.getRangeAt(0);
        if (input.contains(r.commonAncestorContainer)) return r;
      }
      const r = document.createRange();
      r.selectNodeContents(input);
      r.collapse(false);
      if (sel) { sel.removeAllRanges(); sel.addRange(r); }
      return r;
    }
    /** 在光标处插入纯文本（paste / Enter 用），避免浏览器塞入富文本节点。 */
    function insertPlainText(text) {
      if (!text) return;
      const sel = window.getSelection();
      const range = ensureCaret();
      range.deleteContents();
      const tn = document.createTextNode(text);
      range.insertNode(tn);
      const r = document.createRange();
      r.setStart(tn, tn.data.length);
      r.collapse(true);
      if (sel) { sel.removeAllRanges(); sel.addRange(r); }
    }
    /** 让 :empty 占位符重新生效：清掉残留的 <br> 与零宽文本节点。 */
    function cleanupPlaceholder() {
      if (input.querySelector(".ip-file")) return;     // 还有 chip，不算空
      const txt = stripZW(input.textContent || "");
      if (txt.trim() !== "") return;                   // 有真实文字，不算空
      for (const node of Array.from(input.childNodes)) {
        if (node.nodeType === 1 && node.tagName === "BR") { node.remove(); continue; }
        if (node.nodeType === 3 && !stripZW(node.data || "").trim()) node.remove();
      }
    }

    /** 造一枚可删除的 chip，返回 {chip, nameEl}。文件名一律 textContent。 */
    function makeChip(name, idx) {
      const chip = el("span", "ip-file ip-file-pending");
      chip.contentEditable = "false";
      chip.dataset.chip = String(idx);
      chip.title = name;
      const icon = el("span", "ip-file-icon");
      icon.appendChild(fileIcon());
      const nameEl = el("span", "ip-file-name", name);
      const x = el("button", "ip-file-x");
      x.type = "button";
      x.contentEditable = "false";
      x.title = "移除";
      x.setAttribute("aria-label", "移除 " + name);
      x.textContent = "×";
      on(x, "click", (ev) => {
        if (dead) return;
        ev.preventDefault(); ev.stopPropagation();
        chipMap.delete(idx);
        chip.remove();
        cleanupPlaceholder();
        input.focus();
      });
      chip.append(icon, nameEl, x);
      return {chip, nameEl};
    }
    /** 在光标处插入 chip + 零宽文本节点（保证继续输入），返回索引。 */
    function insertChipAtCaret(name) {
      const idx = ++chipSeq;
      const {chip, nameEl} = makeChip(name, idx);
      chipMap.set(idx, {name, nameEl, chipEl: chip,
                        token_ids: [], media: null, kind: "text", pending: true});
      const sel = window.getSelection();
      const range = ensureCaret();
      range.deleteContents();
      const frag = document.createDocumentFragment();
      frag.appendChild(chip);
      frag.appendChild(document.createTextNode(ZW));
      range.insertNode(frag);
      const zw = chip.nextSibling;
      if (zw && zw.nodeType === 3) {
        const r = document.createRange();
        r.setStart(zw, zw.data.length);
        r.collapse(true);
        if (sel) { sel.removeAllRanges(); sel.addRange(r); }
      }
      cleanupPlaceholder();
      return idx;
    }
    /** 「我」气泡里的高亮文件名 chip（无预览、无删除按钮）。 */
    function bubbleChip(name) {
      const c = el("span", "ip-file ip-file-static");
      c.title = name;
      const icon = el("span", "ip-file-icon");
      icon.appendChild(fileIcon());
      c.append(icon, el("span", "ip-file-name", name));
      return c;
    }
    function nameOfChip(chipEl) {
      const n = chipEl && chipEl.querySelector(".ip-file-name");
      return n ? n.textContent : "";
    }

    /** 按 DOM 顺序收集输入：文本段 / chip 段。处理 <br> 与 <div> 换行。 */
    function collectSegments() {
      const segs = [];
      let buf = "";
      const flush = () => { if (buf.length) { segs.push({type: "text", text: buf}); buf = ""; } };
      const walk = (node) => {
        if (node.nodeType === 3) { buf += stripZW(node.data || ""); return; }
        if (node.nodeType !== 1) return;
        if (node.classList && node.classList.contains("ip-file")) {
          flush();
          segs.push({type: "chip",
                     data: chipMap.get(Number(node.dataset.chip)) || null,
                     chipEl: node});
          return;
        }
        const tag = node.tagName;
        if (tag === "BR") { buf += "\n"; return; }
        if (BLOCK_TAGS.has(tag) && buf && !buf.endsWith("\n")) buf += "\n";
        for (const c of Array.from(node.childNodes)) walk(c);
      };
      for (const c of Array.from(input.childNodes)) walk(c);
      flush();
      // 浏览器常在末尾补一个换行，去掉它以免影响 prompt。
      if (segs.length && segs[segs.length - 1].type === "text") {
        const last = segs[segs.length - 1];
        last.text = last.text.replace(/\n$/, "");
        if (!last.text) segs.pop();
      }
      return segs;
    }

    function clearInput() {
      while (input.firstChild) input.removeChild(input.firstChild);
      chipMap.clear();
      cleanupPlaceholder();
    }

    // ---------------------------------------------------------------- 上传
    async function uploadOne(idx, f) {
      try {
        const r = await ctx.api.uploadFile(f);
        if (!alive()) return;
        const d = chipMap.get(idx);
        if (!d) return;                       // 用户已把 chip 删了
        d.token_ids = Array.isArray(r.token_ids) ? r.token_ids : [];
        d.media = r.media || null;
        d.kind = r.kind || "text";
        d.name = r.name || d.name;
        d.pending = false;
        if (d.nameEl) d.nameEl.textContent = d.name;
        if (d.chipEl) { d.chipEl.title = d.name; d.chipEl.classList.remove("ip-file-pending"); }
      } catch (e) {
        if (!alive()) return;
        const d = chipMap.get(idx);
        if (d && d.chipEl) d.chipEl.remove();
        chipMap.delete(idx);
        cleanupPlaceholder();
        ctx.api.toast("上传失败：" + (e && e.message ? e.message : f.name), true);
      }
    }
    function handleFiles(fileList) {
      const files = Array.from(fileList || []);
      for (const f of files) {
        if (!f) continue;
        if (spec && spec.max_file_bytes && f.size > Number(spec.max_file_bytes)) {
          const lim = ctx.api.fmtBytes ? ctx.api.fmtBytes(spec.max_file_bytes)
                                       : String(spec.max_file_bytes);
          ctx.api.toast(`文件过大：${f.name}（上限 ${lim}）`, true);
          continue;
        }
        const idx = insertChipAtCaret(f.name);
        uploadOne(idx, f);
      }
    }

    // ---------------------------------------------------------------- 消息气泡
    function scrollFollow() {
      const nearBottom = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 90;
      if (nearBottom) chat.scrollTop = chat.scrollHeight;
    }
    function newMsg(who) {
      if (empty && empty.parentNode) empty.remove();
      const msg = el("div", "ip-msg " + who);
      const whoEl = el("div", "ip-who",
        who === "me" ? "我" : (who === "sys" ? "系统 / 工具结果" : "模型"));
      const bubble = el("div", "ip-bubble");
      const body = el("div", "ip-body");
      bubble.appendChild(body);
      msg.append(whoEl, bubble);
      chat.appendChild(msg);
      scrollFollow();
      return {msg, bubble, body};
    }
    /** 往当前 bot 气泡追加节点（始终插在光标之前）。 */
    function addNode(node) {
      if (!botBody) return;
      if (caret && caret.parentNode === botBody) botBody.insertBefore(node, caret);
      else botBody.appendChild(node);
    }
    function appendText(s) {
      if (!s || !botBody) return;
      if (curTextNode && curTextNode.parentNode === botBody) curTextNode.data += s;
      else { curTextNode = document.createTextNode(s); addNode(curTextNode); }
    }
    function addBadge(label, kind) {
      if (!botBody) return;
      curTextNode = null;
      addNode(el("span", "ip-badge" + (kind ? " ip-badge-" + kind : ""), label));
    }

    function setBusy(on_) {
      busy = on_;
      send.disabled = on_;
      add.disabled = on_;
      send.title = on_ ? "生成中…" : "发送";
      root.classList.toggle("ip-busy", on_);
    }

    // ---------------------------------------------------------------- DRAW 画布
    function digitOf(L, id) {
      const ds = L && L.digits;
      return Array.isArray(ds) ? ds.indexOf(Number(id)) : -1;
    }
    function canvasSize(n) {
      return clamp(Number.isFinite(n) && n > 0 ? Math.round(n) : 256, 1, 4096);
    }
    function to8(v, max) {
      return Math.round(clamp(v, 0, max) * 255 / max);
    }
    function setPlayIcon(d, playing) {
      while (d.playBtn.firstChild) d.playBtn.removeChild(d.playBtn.firstChild);
      d.playBtn.appendChild(playing ? iconPause() : iconPlay());
    }
    function updateStep(d) {
      d.stepLabel.textContent = `第 ${d.step} / ${d.prims.length} 步`;
    }
    function rotate(node, p) {
      if (p.thetaDeg) {
        node.setAttribute("transform", `rotate(${p.thetaDeg} ${p.cx} ${p.cy})`);
      }
    }
    function primNode(p, W, H) {
      const col = `rgb(${p.r},${p.g},${p.b})`;
      let node;
      if (p.shape === -1) {                       // 背景：整幅 rect
        node = svgEl("rect", {x: 0, y: 0, width: W, height: H, fill: col});
      } else if (p.shape === 0) {                 // 矩形
        node = svgEl("rect", {x: p.cx - p.w / 2, y: p.cy - p.h / 2,
                              width: Math.max(0, p.w), height: Math.max(0, p.h), fill: col});
        rotate(node, p);
      } else if (p.shape === 1) {                 // 椭圆
        node = svgEl("ellipse", {cx: p.cx, cy: p.cy,
                                 rx: Math.max(0, p.w / 2), ry: Math.max(0, p.h / 2), fill: col});
        rotate(node, p);
      } else {                                    // 三角形（polygon）
        const hw = p.w / 2, hh = p.h / 2;
        const pts = `${p.cx},${p.cy - hh} ${p.cx - hw},${p.cy + hh} ${p.cx + hw},${p.cy + hh}`;
        node = svgEl("polygon", {points: pts, fill: col});
        rotate(node, p);
      }
      return node;
    }
    function renderStep(d) {
      if (!d.g) return;
      while (d.g.firstChild) d.g.removeChild(d.g.firstChild);
      const n = Math.min(d.step, d.prims.length);
      for (let i = 0; i < n; i++) d.g.appendChild(primNode(d.prims[i], d.W, d.H));
    }
    /** 9 token → 一个图元：各字段 bin = id - begin - field.offset。 */
    function onPrimitive(d, t) {
      const L = d.L, b = L.begin, f = L.fields;
      const bins = t.map((tok, i) => {
        const fld = f[i];
        if (!fld) return 0;
        return clamp(Number(tok) - b - fld.offset, 0, fld.bins - 1);
      });
      const per = i => (f[i] && f[i].per) || 1;
      d.prims.push({
        cx: bins[0] / per(0), cy: bins[1] / per(1),
        thetaDeg: bins[2] / per(2),
        w: bins[3] / per(3), h: bins[4] / per(4),
        shape: bins[5] - 1,
        r: to8(bins[6], L.color_max),
        g: to8(bins[7], L.color_max),
        b: to8(bins[8], L.color_max),
      });
      d.range.max = String(d.prims.length);
      if (d.follow) { d.step = d.prims.length; d.range.value = String(d.step); }
      updateStep(d);
      renderStep(d);
      scrollFollow();
    }
    function initCanvas(d) {
      const dispW = Math.min(d.W, 420);           // 大小适中：上限 420，按比例缩放高度
      const dispH = Math.max(1, Math.round(dispW * d.H / d.W));
      const svg = svgEl("svg", {width: dispW, height: dispH,
                                viewBox: `0 0 ${d.W} ${d.H}`,
                                preserveAspectRatio: "xMidYMid meet",
                                class: "ip-svg"});
      const g = svgEl("g", {class: "ip-prims"});
      svg.appendChild(g);
      d.canvasWrap.appendChild(svg);
      d.svg = svg; d.g = g;
      renderStep(d);
      scrollFollow();
    }
    function feedDraw(d, id, L) {
      const off = L.canvas.offset, b = L.begin;
      const m1 = b + off + 1, m2 = b + off + 2, m3 = b + off + 3;
      const dig = digitOf(L, id);
      switch (d.phase) {
        case "w":
          if (dig >= 0) d.w += String(dig);
          else if (id === m1) { d.W = canvasSize(parseInt(d.w || "0", 10)); d.phase = "await_y"; }
          break;
        case "await_y":
          if (id === m2) { d.phase = "h"; d.h = ""; }
          break;
        case "h":
          if (dig >= 0) d.h += String(dig);
          else if (id === m3) {
            d.H = canvasSize(parseInt(d.h || "0", 10));
            d.phase = "prims";
            initCanvas(d);
          }
          break;
        case "prims":
          d.primBuf.push(id);
          if (d.primBuf.length === L.tokens_per) {
            onPrimitive(d, d.primBuf);
            d.primBuf = [];
          }
          break;
      }
    }
    function stopPlay(d) {
      d.playing = false;
      if (d.playTimer) { clearTimeout(d.playTimer); d.playTimer = null; }
      setPlayIcon(d, false);
    }
    function togglePlay(d) {
      if (d.playing) { stopPlay(d); return; }
      if (!d.prims.length) return;
      d.playing = true;
      d.follow = false;
      setPlayIcon(d, true);
      if (d.step >= d.prims.length) d.step = 0;
      d.range.value = String(d.step);
      updateStep(d); renderStep(d);
      const tick = () => {
        if (!alive() || !d.playing) return;
        if (d.step >= d.prims.length) { stopPlay(d); return; }
        d.step += 1;
        d.range.value = String(d.step);
        updateStep(d); renderStep(d);
        d.playTimer = setTimeout(tick, 260);
      };
      d.playTimer = setTimeout(tick, 260);
    }
    /** 进入绘图模式：建画布容器 + 时间轴，返回状态机（phase=w，等待宽的数字）。 */
    function startDraw(L) {
      const d = {
        L, phase: "w", w: "", h: "", W: 0, H: 0,
        primBuf: [], prims: [], step: 0, follow: true,
        playing: false, playTimer: null,
      };
      const wrap = el("div", "ip-draw");
      const canvasWrap = el("div", "ip-canvas");
      const timeline = el("div", "ip-timeline");
      const playBtn = el("button", "ip-play");
      playBtn.type = "button";
      playBtn.title = "播放 / 暂停";
      playBtn.setAttribute("aria-label", "播放绘图过程");
      const range = document.createElement("input");
      range.type = "range";
      range.className = "ip-range";
      range.min = "0"; range.max = "0"; range.value = "0"; range.step = "1";
      range.setAttribute("aria-label", "绘图步数");
      const stepLabel = el("span", "ip-step", "第 0 / 0 步");
      timeline.append(playBtn, range, stepLabel);
      wrap.append(canvasWrap, timeline);
      addNode(wrap);
      Object.assign(d, {wrap, canvasWrap, timeline, playBtn, range, stepLabel});
      setPlayIcon(d, false);
      on(range, "input", () => {
        if (dead) return;
        d.follow = false;
        d.step = clamp(Number(range.value) || 0, 0, d.prims.length);
        updateStep(d); renderStep(d);
      });
      on(playBtn, "click", () => { if (!dead) togglePlay(d); });
      scrollFollow();
      return d;
    }
    function finishDraw(d) {
      if (!d) return;
      stopPlay(d);
      d.range.max = String(d.prims.length);
      d.range.value = String(clamp(d.step, 0, d.prims.length));
      updateStep(d);
      renderStep(d);
    }

    // ---------------------------------------------------------------- 流式输出
    function handleToken(tk) {
      if (!alive() || !botBody) return;
      const id = Number(tk.id);
      const L = spec && spec.draw_layout;

      // <draw:end>：结束绘图，之后恢复普通文本
      if (spec && id === Number(spec.draw_end_id)) {
        if (curDraw) { finishDraw(curDraw); curDraw = null; }
        curTextNode = null;
        return;
      }
      // canvas.markers[0]（<x>）：进入 draw 模式，这个 token 被消费
      if (L && !curDraw && id === L.begin + L.canvas.offset + 0) {
        curDraw = startDraw(L);
        curTextNode = null;
        return;
      }
      if (curDraw) { feedDraw(curDraw, id, L); return; }

      // 工具调用徽标
      if (tk.tool) {
        addBadge(toolBadgeText(tk.tool), "tool");
        curTextNode = null;
      }
      // 正常字符 / agent·特殊徽标
      const ch = typeof tk.char === "string" ? tk.char : "";
      const info = idInfo.get(id);
      if (info && !tk.tool) {
        addBadge(badgeLabel(info), info.category === "action" ? "tool" : "agent");
        curTextNode = null;
        return;
      }
      if (ch) {
        if (isTokenName(ch)) { addBadge(ch, "agent"); curTextNode = null; }
        else appendText(ch);
      }
    }

    // ---------------------------------------------------------------- 自动 agent 闭环
    /** 开一个新的模型气泡（每轮推理一个），后续 onToken 往这里追加。 */
    function beginBot() {
      if (curDraw) { finishDraw(curDraw); curDraw = null; }
      const bot = newMsg("bot");
      botBody = bot.body;
      curTextNode = null;
      caret = el("span", "ip-caret", "▍");
      bot.body.appendChild(caret);
      scrollFollow();
      return bot;
    }

    /**
     * 执行本轮解析出的工具调用：逐个调 /api/agent/tool，用一条 sys 气泡
     * 显示结果，再经 /api/agent/tool_result 编成
     * [sys] 结果文本 [sys/] 的 token ID 序列（1.3.0 起无嵌套包裹）。
     */
    async function runToolCalls(calls) {
      const ids = [];
      for (const c of calls) {
        if (!alive()) return [];
        const name = String(c.name || c.tool || "");
        let text = "";
        try {
          const r = await ctx.api.runTool(name, String(c.args || ""));
          text = String(r && r.result != null ? r.result : "");
        } catch (e) {
          text = "工具执行失败：" + (e && e.message ? e.message : e);
        }
        if (!alive()) return [];
        const sys = newMsg("sys");
        sys.body.appendChild(el("span", "ip-badge ip-badge-tool", "⚙ " + name));
        sys.body.appendChild(el("div", "ip-text", text.slice(0, 2000)));
        scrollFollow();
        try {
          const r = await ctx.api.toolResult(name, text);
          if (r && Array.isArray(r.token_ids)) ids.push(...r.token_ids);
        } catch (e) { /* 结果编码失败就停在这一轮，不静默续跑 */ }
      }
      return ids;
    }

    /**
     * 一轮或多轮推理：普通对话跑 1 轮；若本轮 result.tool_calls 非空，
     * 自动执行工具并把 [sys] 段结果喂回模型再跑一轮，直到没有工具调用
     * 或达到 MAX_AGENT_ROUNDS。这样 10 亿参数小模型也能「出生即 agent」。
     */
    async function runInfer(reqIds0, media0, maxNew, depth0) {
      let reqIds = reqIds0;
      let media = media0 || [];
      let depth = depth0;
      setBusy(true);
      try {
        for (;;) {
          const myRound = ++roundSeq;
          genIds = [];
          const bot = beginBot();
          let lastResult = null;
          let failed = null;
          try {
            await ctx.api.infer({
              tokenIds: reqIds, media, maxNew,
              temperature: Number(ctx.config.get("temperature", 0.8)),
              topP: Number(ctx.config.get("topP", 0.95)),
            }, {
              onStart() {},
              onToken(tk) {
                if (!alive() || roundSeq !== myRound) return;
                genIds.push(Number(tk.id));
                handleToken(tk);
                scrollFollow();
              },
              onDone(result, summary) {
                if (!alive() || roundSeq !== myRound) return;
                if (curDraw) { finishDraw(curDraw); curDraw = null; }
                if (caret) { caret.remove(); caret = null; }
                curTextNode = null;
                const ids = genIds.slice();
                ctxIds = ctxIds.concat(ids);
                if (ctxIds.length > ctxCap) ctxIds = ctxIds.slice(-ctxCap);
                ctx.session.append("bot", ids);
                if (summary && summary.elapsed_ms != null) {
                  bot.body.appendChild(
                    el("div", "ip-note", `耗时 ${fmtMs(summary.elapsed_ms)}`));
                }
                lastResult = result || {};
                scrollFollow();
              },
              onError(e) {
                if (!alive() || roundSeq !== myRound) return;
                failed = e;
                if (curDraw) { finishDraw(curDraw); curDraw = null; }
                if (caret) { caret.remove(); caret = null; }
                curTextNode = null;
                bot.body.appendChild(el("div", "ip-err",
                  "⚠ " + (e && e.message ? e.message : e)));
                scrollFollow();
              },
              // 不处理 onPhase：顶部状态条由宿主绘制
            });
          } catch (e) {
            failed = e;
            if (alive() && roundSeq === myRound && e && e.message !== "已取消") {
              if (caret) { caret.remove(); caret = null; }
              bot.body.appendChild(el("div", "ip-err", "⚠ " + e.message));
            }
          }
          if (!alive() || failed) break;
          if (ctx.config.get("autoAgent", true) === false) break;
          if (depth >= MAX_AGENT_ROUNDS) break;
          const calls = lastResult && Array.isArray(lastResult.tool_calls)
            ? lastResult.tool_calls : [];
          if (!calls.length) break;
          const ids = await runToolCalls(calls);
          if (!alive() || !ids.length) break;
          reqIds = ids;
          media = [];
          depth += 1;
        }
      } finally {
        genIds = null;
        if (alive()) setBusy(false);
      }
    }

    // ---------------------------------------------------------------- 提交（唯一入口）
    async function submit() {
      if (busy || dead) return;
      await loadSpec();
      if (!alive()) return;

      const segments = collectSegments();
      const hasContent = segments.some(
        s => s.type === "chip" || (s.type === "text" && s.text.length));
      if (!hasContent) { ctx.api.toast("先输入文字或添加文件", true); return; }

      // 文本段依次 encodeText，chip 段拼 token_ids + 按出现顺序收 media[]
      let tokenIds = [];
      const media = [];
      try {
        for (const seg of segments) {
          if (seg.type === "text") {
            const r = await ctx.api.encodeText(seg.text);
            if (!alive()) return;
            if (r && Array.isArray(r.token_ids)) tokenIds = tokenIds.concat(r.token_ids);
          } else {
            const d = seg.data;
            if (!d) continue;
            if (MEDIA_KINDS.has(d.kind)) {
              if (Array.isArray(d.media)) media.push(...d.media);
              else if (d.media) media.push(d.media);
            }
            // 图元 / 文本类（以及媒体自带的离散标记）一律按序追加 token_ids
            if (Array.isArray(d.token_ids) && d.token_ids.length) {
              tokenIds = tokenIds.concat(d.token_ids);
            }
          }
        }
      } catch (e) {
        if (alive()) ctx.api.toast("编码失败：" + (e && e.message ? e.message : e), true);
        return;
      }
      if (!alive()) return;
      // 纯媒体消息也要给 /api/infer 一个非空 token 序列
      if (!tokenIds.length && media.length) {
        try {
          const r = await ctx.api.encodeText(" ");
          if (r && Array.isArray(r.token_ids)) tokenIds = r.token_ids.slice();
        } catch (e) {}
      }
      if (!tokenIds.length && !media.length) { ctx.api.toast("没有可发送的内容", true); return; }

      const kvEnabled = ctx.session.config.get("kvEnabled", true) !== false;
      ctxIds = ctxIds.concat(tokenIds);
      if (ctxIds.length > ctxCap) ctxIds = ctxIds.slice(-ctxCap);
      const reqIds = kvEnabled ? tokenIds.slice() : ctxIds.slice();
      const maxNew = Math.max(1, Math.min(maxNewLimit,
        Number(ctx.session.config.get("maxNew", 32)) || 32));

      ctx.session.append("me", tokenIds);

      // 我气泡：正常文字 + 高亮文件名 chip（无预览）
      const me = newMsg("me");
      for (const seg of segments) {
        if (seg.type === "text") me.body.appendChild(el("span", "ip-text", seg.text));
        else {
          const nm = (seg.data && seg.data.name) || nameOfChip(seg.chipEl) || "文件";
          me.body.appendChild(bubbleChip(nm));
        }
      }
      clearInput();
      // 普通对话一轮；1.3.0 里 SLOT_DEFS 默认为空、tool_calls 恒为 []，
      // 闭环不会启动。等使用者自定义了 action/close 槽位对后才会自动
      // 把 [sys] 段结果喂回模型（见 runInfer）。
      await runInfer(reqIds, media, maxNew, 0);
    }

    // ---------------------------------------------------------------- 事件绑定
    on(send, "click", submit);
    on(add, "click", () => { if (!busy) fileInput.click(); });
    on(fileInput, "change", () => {
      const files = Array.from(fileInput.files || []);
      fileInput.value = "";                 // 允许再次选同一文件
      if (files.length) handleFiles(files);
    });
    on(input, "paste", (e) => {
      e.preventDefault();
      const cd = e.clipboardData || window.clipboardData;
      insertPlainText(cd ? cd.getData("text/plain") : "");
      cleanupPlaceholder();
    });
    on(input, "keydown", (e) => {
      // Enter 只换行、不发送；IME 组词期间的 Enter 不拦截。
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        insertPlainText("\n");
      }
    });
    on(input, "input", cleanupPlaceholder);
    on(dock, "dragover", (e) => { e.preventDefault(); dock.classList.add("ip-drag"); });
    on(dock, "dragleave", (e) => {
      if (!dock.contains(e.relatedTarget)) dock.classList.remove("ip-drag");
    });
    on(dock, "drop", (e) => {
      e.preventDefault();
      dock.classList.remove("ip-drag");
      if (!busy) handleFiles(e.dataTransfer && e.dataTransfer.files);
    });

    // ---------------------------------------------------------------- 历史恢复
    (async () => {
      const history = (ctx.session && ctx.session.messages) || [];
      for (const m of history) {
        if (!alive()) return;
        const role = m.role === "me" ? "me" : "bot";
        const ids = Array.isArray(m.token_ids) ? m.token_ids : [];
        const view = newMsg(role);
        let text = "";
        try {
          const d = await ctx.api.decodeIds(ids);
          if (!alive()) return;
          if (d && Array.isArray(d.parts) && d.parts.length) {
            text = d.parts.map(p => (p && typeof p.text === "string" ? p.text : "")).join("");
          } else if (d && typeof d.text === "string") {
            text = d.text;
          }
        } catch (e) { /* 历史里的文件 chip 无法还原：只显示文字，不报错 */ }
        if (text) view.body.appendChild(el("span", "ip-text", text));
        ctxIds = ctxIds.concat(ids);
      }
      if (ctxIds.length > ctxCap) ctxIds = ctxIds.slice(-ctxCap);
      scrollFollow();
    })();

    input.focus();

    return {
      /** 宿主点「清空」：清对话、上下文、输入与画布，后端 reset 由宿主负责。 */
      reset() {
        if (dead) return;
        roundSeq += 1;                     // 作废所有在途回调
        if (curDraw) { stopPlay(curDraw); curDraw = null; }
        while (chat.firstChild) chat.removeChild(chat.firstChild);
        chat.appendChild(empty);
        ctxIds = [];
        genIds = null;
        botBody = null; caret = null; curTextNode = null;
        clearInput();
        setBusy(false);
      },
      /** 切插件 / 卸载：停回调、解绑、忽略一切异步结果。 */
      destroy() {
        dead = true;
        if (curDraw && curDraw.playTimer) clearTimeout(curDraw.playTimer);
        for (const [t, ev, fn, opts] of listeners) {
          try { t.removeEventListener(ev, fn, opts); } catch (e) {}
        }
        listeners.length = 0;
        fileInput.remove();
        root.classList.remove("ip-root");
      },
    };
  },
};
