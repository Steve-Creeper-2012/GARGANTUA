
const $ = id => document.getElementById(id);
const $$ = (sel, el=document) => el.querySelectorAll(sel);

// 数值入参钳位：空输入 / 手敲越界值在前端就地修正，不让 400 打到用户脸上。
// NaN（空输入框）回落到 default，而不是被 `|| default` 把合法的 0 也吃掉。
function clampNum(v, lo, hi, dflt) {
  const n = Number(v);
  if (!Number.isFinite(n)) return dflt;
  return Math.min(hi, Math.max(lo, n));
}

// 所有请求带超时 + 可取消：一个悬挂请求（后端卡住/网络半断）不能让轮询
// 循环永远停在 await 上——超时即 AbortController.abort()，调用方拿到明确错误。
const API_TIMEOUT_MS = 20000;
const API_UPLOAD_TIMEOUT_MS = 180000;   // 上传 .tokens / 视频等大 body
async function api(path, body, opts) {
  const o = opts || {};
  const ms = o.timeout || API_TIMEOUT_MS;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ms);
  // 外部取消令牌（切插件 / 卸载时中止流式轮询）：宿主的请求超时和插件
  // 生命周期取消要能同时生效，否则卸掉的插件后台还在一直 GET /stream。
  const ext = o.signal;
  let onAbort = null;
  if (ext) {
    if (ext.aborted) ctrl.abort();
    else {
      onAbort = () => ctrl.abort();
      ext.addEventListener("abort", onAbort, {once: true});
    }
  }
  const opt = body === undefined
    ? {signal: ctrl.signal}
    : {method: "POST", headers: {"Content-Type": "application/json"},
       body: JSON.stringify(body), signal: ctrl.signal};
  try {
    const r = await fetch(path, opt);
    const txt = await r.text();
    let j = {};
    if (txt) {
      try { j = JSON.parse(txt); }
      catch (e) { throw new Error("响应不是 JSON: " + txt.slice(0, 120)); }
    }
    if (!r.ok) {
      // 整体响应挂在 e.data 上：像 /api/vocab/source 这类会带 errors[{line,message}]
      // 数组的失败，调用方要能取到逐条原因，只靠 e.message 会全丢。
      const err = new Error(j.error || ("HTTP " + r.status));
      err.data = j; err.status = r.status;
      throw err;
    }
    return j;
  } catch (e) {
    if (e && e.name === "AbortError") {
      throw new Error(`请求超时（${ms}ms）: ${path}`);
    }
    throw e;
  } finally {
    clearTimeout(timer);
    if (ext && onAbort) ext.removeEventListener("abort", onAbort);
  }
}
// 统一任务轮询：每轮【先】检查 deadline，超时抛错，并复用 api() 的单请求超时。
async function pollTask(taskId, opts) {
  const o = opts || {};
  const interval = o.interval || 1500;
  const timeout = o.timeout || 120000;
  const t0 = Date.now();
  for (;;) {
    if (Date.now() - t0 > timeout) {
      throw new Error(`轮询超时（${Math.round(timeout / 1000)}s）`);
    }
    await new Promise(r => setTimeout(r, interval));
    const t = await api("/api/task/" + taskId);
    if (o.onTick) o.onTick(t);
    if (t.status === "done") return t;
    if (t.status === "error") throw new Error(t.error);
  }
}
function esc(s) {
  // 补引号转义：textContent→innerHTML 只转义 & < >，拼进属性时可注入
  const d = document.createElement("div"); d.textContent = s == null ? "" : String(s);
  return d.innerHTML.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
function fileToB64(f) {
  return new Promise((res, rej) => {
    const rd = new FileReader();
    rd.onload = () => res(rd.result.split(",")[1]);
    rd.onerror = rej;
    rd.readAsDataURL(f);
  });
}
function fileToUrl(f) { return URL.createObjectURL(f); }

/* ---------- 自绘对话框 / 轻提示 ----------
   禁令：系统原生 alert / confirm 一律不出现。uiAlert / uiConfirm 返回 Promise，
   调用处用 await 即可，语义与原生版本一致。 */
let MODAL_RESOLVE = null;
function modalBodySet(body) {
  const el = $("modal-body");
  el.textContent = "";
  // body 可以是字符串，也可以是一个 DOM 节点（供 uiPrompt 塞输入框）
  if (body && body.nodeType) el.appendChild(body);
  else el.textContent = body == null ? "" : String(body);
}
/* 需要富文本的对话框（关于页、任务选择器）走这个：body 是 HTML 字符串。
   内容全部由本文件自己拼，不含用户输入。 */
function modalOpenHTML(title, html, okText, withCancel, wide) {
  return new Promise(res => {
    MODAL_RESOLVE = res;
    $("modal-title").textContent = title;
    $("modal-body").innerHTML = html;
    $("modal-ok").textContent = okText || "确定";
    $("modal-cancel").style.display = withCancel ? "" : "none";
    const box = document.querySelector(".modal");
    if (box) box.classList.toggle("wide", !!wide);
    $("modal-mask").hidden = false;
    $("modal-ok").focus();
  });
}
function modalOpen(title, body, okText, withCancel) {
  return new Promise(res => {
    MODAL_RESOLVE = res;
    $("modal-title").textContent = title;
    modalBodySet(body);
    $("modal-ok").textContent = okText || "确定";
    $("modal-cancel").style.display = withCancel ? "" : "none";
    const box = document.querySelector(".modal");
    if (box) box.classList.remove("wide");
    $("modal-mask").hidden = false;
    $("modal-ok").focus();
  });
}
function modalClose(v) {
  $("modal-mask").hidden = true;
  const r = MODAL_RESOLVE; MODAL_RESOLVE = null;
  if (r) r(v);
}
$("modal-ok").onclick = () => modalClose(true);
$("modal-cancel").onclick = () => modalClose(false);
$("modal-mask").onclick = e => {
  if (e.target === $("modal-mask")) modalClose(false);
};
document.addEventListener("keydown", e => {
  if ($("modal-mask").hidden) return;
  if (e.key === "Escape") { e.preventDefault(); modalClose(false); }
  else if (e.key === "Enter") { e.preventDefault(); modalClose(true); }
});
const uiAlert = msg => modalOpen("提示", msg, "知道了", false);
const uiConfirm = msg => modalOpen("确认", msg, "确定", true);
/* 带一个输入框的对话框（Fork / 复制时问新词表名）。取消 → null。 */
async function uiPrompt(title, label, dflt) {
  const wrap = document.createElement("div");
  const lb = document.createElement("div");
  lb.className = "hint"; lb.textContent = label || "";
  const inp = document.createElement("input");
  inp.type = "text"; inp.value = dflt || ""; inp.autocomplete = "off";
  wrap.appendChild(lb); wrap.appendChild(inp);
  const p = modalOpen(title, wrap, "确定", true);
  inp.focus(); inp.select();
  const ok = await p;
  return ok ? inp.value.trim() : null;
}

function toast(msg, isErr) {
  const t = document.createElement("div");
  t.className = "toast" + (isErr ? " err" : "");
  t.textContent = msg;
  $("toast-stack").appendChild(t);
  setTimeout(() => {
    t.style.transition = "opacity .2s";
    t.style.opacity = "0";
    setTimeout(() => t.remove(), 220);
  }, isErr ? 5200 : 3200);
}

/* ---------- 主题 / 对比度 ----------
   三档模式：dark / light / auto（auto 跟 matchMedia 走）。
   只改 <html> 上的 data-* 属性，配色全部由 CSS 变量接管。
   持久化在 localStorage，首帧由 <head> 里的内联脚本先落定，不会闪。 */
const THEME_KEY = "gargantua.theme", CONTRAST_KEY = "gargantua.contrast";
const mqDark = window.matchMedia("(prefers-color-scheme: dark)");

function themeMode() {
  try { return localStorage.getItem(THEME_KEY) || "auto"; } catch (e) { return "auto"; }
}
function contrastMode() {
  try { return localStorage.getItem(CONTRAST_KEY) || "normal"; } catch (e) { return "normal"; }
}
function applyTheme() {
  const mode = themeMode();
  const dark = mode === "dark" || (mode === "auto" && mqDark.matches);
  const el = document.documentElement;
  el.setAttribute("data-theme", dark ? "dark" : "light");
  el.setAttribute("data-theme-mode", mode);
  el.setAttribute("data-contrast", contrastMode() === "high" ? "high" : "normal");
  // 分段控件的选中态
  const seg = $("set-theme");
  if (seg) seg.querySelectorAll("button").forEach(b =>
    b.classList.toggle("on", b.dataset.v === mode));
  const segc = $("set-contrast");
  if (segc) segc.querySelectorAll("button").forEach(b =>
    b.classList.toggle("on", b.dataset.v === contrastMode()));
}
function setTheme(mode) {
  try { localStorage.setItem(THEME_KEY, mode); } catch (e) {}
  applyTheme();
}
function setContrast(v) {
  try { localStorage.setItem(CONTRAST_KEY, v); } catch (e) {}
  applyTheme();
}
// 跟随系统：系统外观变了要立刻重算（auto 之外的模式不受影响）
mqDark.addEventListener("change", () => { if (themeMode() === "auto") applyTheme(); });
$("set-theme").querySelectorAll("button").forEach(b => {
  b.onclick = () => setTheme(b.dataset.v);
});
$("set-contrast").querySelectorAll("button").forEach(b => {
  b.onclick = () => setContrast(b.dataset.v);
});

/* ---------- 圆角半径（全站唯一的 --r） ----------
   直接改 <html> 上的 --r 变量，卡片/按钮/表格/弹窗/边栏同时跟着变。 */
const RADIUS_KEY = "gargantua.radius";
const RADIUS_MIN = 0, RADIUS_MAX = 24, RADIUS_DEF = 16;
function applyRadius(px) {
  const n = Number(px);
  const v = Math.min(RADIUS_MAX, Math.max(RADIUS_MIN,
    Number.isFinite(n) ? Math.round(n) : RADIUS_DEF));
  document.documentElement.style.setProperty("--r", v + "px");
  const slider = $("set-radius"), label = $("set-radius-val");
  if (slider) slider.value = v;
  if (label) label.textContent = v + "px";
  return v;
}
function savedRadius() {
  try {
    const s = localStorage.getItem(RADIUS_KEY);
    if (s == null || s === "") return RADIUS_DEF;
    const v = Number(s);
    return Number.isFinite(v) && v >= 0 ? v : RADIUS_DEF;
  } catch (e) { return RADIUS_DEF; }
}
$("set-radius").addEventListener("input", e => {
  const v = applyRadius(e.target.value);
  try { localStorage.setItem(RADIUS_KEY, String(v)); } catch (err) {}
});
applyRadius(savedRadius());

/* ---------- 边栏折叠 ----------
   按钮是 fixed 常驻窗口左上的唯一一枚（12,13，与边栏原按钮位逐像素一致），
   不从属任何裁切层——收起全程原地可见、可点。收起/展开是 CSS 的四条过渡
   同一帧起步、两方向同速匀速（面板 ~1.49px/ms、内容 ~1.52px/ms）：
     · 侧栏 width 244→50（194px）、标题 translateX、.side-inner clip-path
       四边内缩到按钮矩形 —— 都是 var(--dur-side)=130ms
     · 内容区 margin-left 单独走 33px（var(--dur-tail)=33ms），
       收起时排在面板停稳之后、展开时排在面板出发之前 ——
       于是面板在动的 130ms 里，内容左沿与面板右沿逐帧重合，不重叠
   JS 只负责给 <html> 挂 .side-anim 让这套过渡仅在用户点击时生效
   （首帧恢复折叠状态直接落位、不播）。
   状态存 localStorage，首帧由 <head> 内联脚本恢复，不会先闪一下再收。 */
function applySidebar(collapsed) {
  $("sidebar").classList.toggle("collapsed", collapsed);
  document.documentElement.setAttribute("data-side",
                                        collapsed ? "collapsed" : "open");
  $("side-toggle").title = collapsed ? "展开边栏" : "收起边栏";
}
/* 时长只有一个真源（CSS 的 --dur-side / --dur-tail），别在这里另写一份毫秒数：
   388/488 那次的漂移就是两处数字各写各的。 */
function cssMs(name) {
  const v = getComputedStyle(document.documentElement)
              .getPropertyValue(name).trim();
  const n = parseFloat(v) || 0;
  return /ms$/.test(v) ? n : n * 1000;
}
function moveDurMs() {
  return cssMs("--dur-side") + cssMs("--dur-tail");
}
function toggleSidebar() {
  const collapsed = !$("sidebar").classList.contains("collapsed");
  try { localStorage.setItem("gargantua.side", collapsed ? "0" : "1"); } catch (e) {}
  document.documentElement.classList.add("side-anim");
  applySidebar(collapsed);
  clearTimeout(toggleSidebar._t);
  /* 只比过渡长一点就摘掉 .side-anim：连点时计时器会重排，过渡期间
     这个类一直挂着，所以中途反向不会被掐断。 */
  toggleSidebar._t = setTimeout(() => {
    document.documentElement.classList.remove("side-anim");
  }, moveDurMs() + 60);
}
$("side-toggle").onclick = toggleSidebar;
applySidebar(document.documentElement.getAttribute("data-side") === "collapsed");
/* 首帧状态：主页 + 右侧边栏收起。写在 HTML 属性上，CSS 直接按收起态
   落位，不会先展开再收（与 data-side 同一套做法）。 */
document.documentElement.setAttribute("data-page", "home");
document.documentElement.setAttribute("data-dev", "closed");

/* ---------- 导航（2026-10-29 重构：按类型分组，灰色加粗组名分隔，
   底部固定「设置」；所有文本 user-select 已由全局 CSS 禁止） ---------- */
const NAV_GROUPS = [
  {label: "概览", items: [
    {id:"home", label:"主页", icon:"M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"},
  ]},
  {label: "模型", items: [
    {id:"models", label:"模型", icon:"M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"},
    {id:"vocab", label:"词表", icon:"M4 19.5A2.5 2.5 0 0 1 6.5 17H20M4 19.5A2.5 2.5 0 0 0 6.5 22H20V2H6.5A2.5 2.5 0 0 0 4 4.5v15z"},
  ]},
  {label: "训练与推理", items: [
    {id:"train", label:"训练", icon:"M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"},
    {id:"infer", label:"推理", icon:"M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"},
  ]},
  {label: "转换工具", items: [
    {id:"text", label:"文本", icon:"M4 6h16M4 12h16M4 18h7"},
    {id:"audio", label:"音频", icon:"M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2z"},
    {id:"image", label:"图片", icon:"M4 16l4.586-5.657a2 2 0 013 0L15 14m-2-2l2.586-3.657a2 2 0 013 0L20 12M4 6h16M4 12h16M4 18h16"},
    {id:"video", label:"视频", icon:"M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"},
    {id:"prims", label:"图元绘画", icon:"M12 3l1.9 5.8H20l-4.9 3.6 1.9 5.8-5-3.6-5 3.6 1.9-5.8L4 8.8h6.1z"},
  ]},
  {label: "编辑器", items: [
    {id:"json", label:"Token 编辑器", icon:"M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"},
    {id:"pdraw", label:"图元编辑器", icon:"M15.232 5.232l3.536 3.536M9 13l6.5-6.5a2.5 2.5 0 113.536 3.536L12.5 16.5 8 18l1.5-4.5zM3 21h18"},
  ]},
];
const NAV_BOTTOM = [
  {id:"term", label:"终端", icon:"M4 17l6-5-6-5M12 19h8"},
  {id:"settings", label:"设置", icon:"M12 15a3 3 0 100-6 3 3 0 000 6zm7.5-3a7.4 7.4 0 00-.1-1.2l2-1.6-2-3.4-2.4 1a7.6 7.6 0 00-2-1.2L14.6 3H9.4l-.4 2.6a7.6 7.6 0 00-2 1.2l-2.4-1-2 3.4 2 1.6a7.4 7.4 0 000 2.4l-2 1.6 2 3.4 2.4-1a7.6 7.6 0 002 1.2l.4 2.6h5.2l.4-2.6a7.6 7.6 0 002-1.2l2.4 1 2-3.4-2-1.6c.06-.4.1-.8.1-1.2z"},
  {id:"about", label:"关于", icon:"M12 22a10 10 0 100-20 10 10 0 000 20zM12 16v-4M12 8h.01"},
];
// 页面标题只此一处：顶行那个 h1。切页时改文字，不再每页各写一个 h1。
const PAGE_TITLES = {
  home: "主页", models: "模型", vocab: "词表", train: "训练", infer: "推理",
  text: "文本", audio: "音频", image: "图片", video: "视频",
  prims: "图元绘画", json: "Token 编辑器", pdraw: "图元编辑器",
  about: "关于", settings: "设置", term: "终端",
};
function navItemHTML(n) {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="${n.icon}"/></svg><span class="label">${esc(n.label)}</span>`;
}
const NAV_CLOSED_KEY = "gargantua.navclosed";
function navClosedSet() {
  try {
    return new Set((localStorage.getItem(NAV_CLOSED_KEY) || "")
      .split(",").filter(Boolean));
  } catch (e) { return new Set(); }
}
function saveNavClosed(set) {
  try { localStorage.setItem(NAV_CLOSED_KEY, Array.from(set).join(",")); }
  catch (e) {}
}
/* 高亮滑块：唯一一块 accent-soft 底，切页时从旧项滑到新项。
   坐标系取 .side-inner（它本身不是滚动容器），#nav 滚动时由 JS 重算坐标，
   所以跨越 #nav / #nav-bottom 也能一路滑到底。 */
const NAV_INDICATOR = document.createElement("div");
NAV_INDICATOR.className = "nav-indicator";
document.querySelector(".side-inner").prepend(NAV_INDICATOR);
function positionIndicator(animate) {
  const act = document.querySelector(".nav-item.active");
  const host = document.querySelector(".side-inner");
  if (!act || !host) return;
  // 当前高亮项所在的组被折叠了：别把滑块留在半空，先收起来
  const box = act.closest(".nav-items");
  if (box && box.classList.contains("collapsed")) {
    NAV_INDICATOR.classList.remove("show");
    return;
  }
  const r = act.getBoundingClientRect(), hr = host.getBoundingClientRect();
  if (!animate) NAV_INDICATOR.style.transition = "none";
  NAV_INDICATOR.classList.add("show");
  NAV_INDICATOR.style.width = r.width + "px";
  NAV_INDICATOR.style.height = r.height + "px";
  // .nav-indicator 是 .side-inner 的 abs 子元素（position:relative + 1px border）。
  // getBoundingClientRect() 返回 border-box，而 abs 定位 top/left:0 落在 padding-box
  // 顶点 = border-box 顶点 + border 宽，故需减去 host 的 border 宽抵消 1px 偏移，
  // 否则高亮块比 hover 背景低/右 1px。
  const dx = r.left - hr.left - host.clientLeft;
  const dy = r.top - hr.top - host.clientTop;
  NAV_INDICATOR.style.transform =
    `translate(${dx.toFixed(2)}px, ${dy.toFixed(2)}px)`;
  if (!animate) {
    void NAV_INDICATOR.offsetHeight;          // 强制 reflow 让上面的 none 生效
    NAV_INDICATOR.style.transition = "";
  }
}
// 导航区滚动会改变条目视口坐标，滚动时重算滑块（不改 .side-inner 的框）
$("nav").addEventListener("scroll", () => positionIndicator(false), {passive: true});

function setGroupClosed(gid, closed) {
  const box = document.querySelector(`.nav-items[data-g="${gid}"]`);
  const btn = document.querySelector(`.nav-group[data-g="${gid}"]`);
  if (box) box.classList.toggle("collapsed", closed);
  if (btn) btn.classList.toggle("collapsed", closed);
  const set = navClosedSet();
  if (closed) set.add(gid); else set.delete(gid);
  saveNavClosed(set);
  // 收起/展开过程中行高在变，滑块跟着重算（含动画结束后的最终位置）
  positionIndicator(false);
  setTimeout(() => positionIndicator(false), 340);
}
function toggleGroup(gid) {
  const box = document.querySelector(`.nav-items[data-g="${gid}"]`);
  setGroupClosed(gid, !(box && box.classList.contains("collapsed")));
}

/* 分组行 = 一整行按钮（比条目更靠左，靠缩进体现层级），右侧带展开/收起箭头 */
function buildNav(container, groups, prefix) {
  groups.forEach((g, gi) => {
    const gid = prefix + gi;
    const btn = document.createElement("button");
    btn.className = "nav-group" +
      (gi === 0 && container.id === "nav" ? " first" : "");
    btn.dataset.g = gid;
    btn.innerHTML = `<span>${esc(g.label)}</span>` +
      `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"` +
      ` stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>`;
    btn.onclick = () => toggleGroup(gid);
    container.appendChild(btn);

    const box = document.createElement("div");
    box.className = "nav-items";
    box.dataset.g = gid;
    // 内层单独承载条目：grid 1fr→0fr 才能平滑收拢，直接对按钮组
    // display:none 是瞬间消失，没有过渡。
    const inner = document.createElement("div");
    inner.className = "nav-items-inner";
    g.items.forEach(n => {
      const b = document.createElement("button");
      b.className = "nav-item";
      b.dataset.p = n.id;
      b.innerHTML = navItemHTML(n);
      b.onclick = () => switchPage(n.id);
      inner.appendChild(b);
    });
    box.appendChild(inner);
    container.appendChild(box);
  });
}
buildNav($("nav"), NAV_GROUPS, "g");
buildNav($("nav-bottom"), [{label: "系统", items: NAV_BOTTOM}], "b");
// 主页默认高亮（先定 active，再恢复折叠，滑块才能算到正确位置）
$$(".nav-item").forEach(b => b.classList.toggle("active", b.dataset.p === "home"));
navClosedSet().forEach(gid => setGroupClosed(gid, true));
positionIndicator(false);   // 首帧无动画就位

function switchPage(id) {
  $$(".nav-item").forEach(x => x.classList.toggle("active", x.dataset.p === id));
  $$(".page").forEach(x => x.classList.toggle("active", x.id === "p-" + id));
  // 顶栏右侧的工具条跟着页面切换：它们是 #topbar 的直接子元素，
  // 只切 hidden，不参与 .page 的淡入/位移动画（旧实现会先被磨砂条盖住）。
  document.documentElement.setAttribute("data-page", id);
  $("vtab-bar").hidden = (id !== "vocab");
  $("infer-toolbar").hidden = (id !== "infer");
  $("prims-bar").hidden = (id !== "prims");
  // 「关于」是**每页独立**的：同一颗按钮，切页换目标；没有文档的页就不出现。
  const doc = TOOL_DOCS.find(x => x.id === id);
  const aboutBtn = $("about-btn");
  if (aboutBtn) {
    aboutBtn.hidden = !doc;
    aboutBtn.title = doc ? `关于「${doc.name}」` : "";
  }
  positionIndicator(true);
  const t = PAGE_TITLES[id];
  if (t) {
    const el = $("page-title");
    el.textContent = t;
    // 标题轻微淡入；内容区淡入 + 上移 8px，220ms 收住，不拖沓
    if (el.animate) el.animate([{opacity: .35}, {opacity: 1}],
                               {duration: 200, easing: "ease-out"});
  }
  const page = $("p-" + id);
  if (page && page.animate) {
    if (page._anim) page._anim.cancel();
    page._anim = page.animate(
      [{opacity: 0, transform: "translateY(12px)"},
       {opacity: 1, transform: "none"}],
      {duration: 320, easing: "cubic-bezier(.22, .61, .36, 1)"});
  }
  if (id === "models") { refreshModels(); refreshModelEstimate(); }
  if (id === "train") refreshQueue();
  if (id === "vocab") refreshVocab();
  if (id === "pdraw") pdrawInit();
  if (id === "prims") primsInit();
  if (id === "term") termInit();
  if (id !== "infer") {
    setDevOpen(false);                     // 离开推理页就收起右侧边栏
    $("model-pop").hidden = true;
  }
  if (id === "infer") {
    // 插件按需挂载：第一次进推理页才加载它的 JS/CSS；离开不卸载，
    // 保留会话（回来还在），只有换插件时才重建。
    if (!PLUGIN_CUR) mountInferPlugin();
    startInferStats();
  } else {
    stopInferStats();
  }
  if (id === "home") refreshMonitors();
}

/* ---------- 文件选择器（自定义外观） ---------- */
function wireFileInput(id) {
  const input = $(id);
  const label = input.closest(".file-pick");
  const name = $(id + "-name");
  if (!input || !label || !name) return;
  const sync = () => {
    const f = input.files && input.files[0];
    name.textContent = f ? f.name : "未选择";
    name.title = f ? `${f.name}（${(f.size / 1024).toFixed(1)} KB）` : "";
    label.classList.toggle("has-file", !!f);
  };
  input.addEventListener("change", sync);
  sync();
}
["train-file", "audio-file", "image-file", "video-file",
 "prims-file"].forEach(wireFileInput);

/* ---------- 活动模型（后台静默跟踪，供建模后自动激活判断） ---------- */
let ACTIVE_MODEL = "";
async function refreshStatus() {
  try {
    const s = await api("/api/status");
    ACTIVE_MODEL = s.model || "";
    if (s.draw_layout) PDRAW.layout = s.draw_layout;
    fillCanvasOptions(s.draw_canvas);
  } catch (e) {}
}
/* 图元画布的档位表来自 /api/status（唯一真源是 server/spec.py，
   前端不复制一份常量，spec 改档位这里自动跟随）。
   「图元绘画」页要 auto 选项，「图元编辑器」页只认具体尺寸。 */
function fillCanvasOptions(dc) {
  if (!dc || !Array.isArray(dc.sizes) || !dc.sizes.length) return;
  PDRAW.canvasList = dc.sizes.map(s => [s[0], s[1]]);
  const opts = dc.sizes.map(s => {
    const [w, h] = s;
    return `<option value="${w}x${h}">${w}×${h}</option>`;
  }).join("");
  const ed = $("pdraw-canvas");
  if (ed) {
    const prev = ed.value;
    ed.innerHTML = opts;
    if (prev) ed.value = prev;
    if (!ed.value) ed.value = `${dc.default[0]}x${dc.default[1]}`;
    if (!PDRAW.prims.length) pdrawSetCanvas(ed.value);
  }
}

/* =========================================================================
   主页：环境与依赖
   -------------------------------------------------------------------------
   三段，各自一张网格，卡片样式统一：
     · 环境包   —— environment/ 下每个文件夹一张卡（后端扫目录给出，
                   两边永远对得上：多一个文件夹就多一张卡）
     · Python 依赖 —— 全部 pip 包，一项不漏（含可选/窗口/测试依赖）
     · 系统依赖 —— 装不进 environment/ 的外部命令（ffmpeg / 编译器）
   ========================================================================= */
const DEP_META = {
  python:      {display: "Python 运行时",   desc: "跑 server 与训练 / 推理的解释器"},
  pywebview:   {display: "pywebview",       desc: "开独立桌面窗口（无 GUI 环境可跳过）"},
  torch:       {display: "PyTorch",        desc: "深度学习框架（训练与推理）"},
  numpy:       {display: "NumPy",          desc: "数值计算库"},
  pillow:      {display: "Pillow",         desc: "图像处理库"},
  safetensors: {display: "Safetensors",    desc: "模型权重存储格式"},
  pytest:      {display: "pytest",         desc: "测试套件"},
  ffmpeg:      {display: "FFmpeg",         desc: "音视频解码与抽帧"},
  rust:        {display: "Rust 工具链",    desc: "构建 GPU 图元转换器"},
  wgpu:        {display: "wgpu",           desc: "GPU 图元渲染后端（随 Rust 转换器构建）"},
  cv2:         {display: "OpenCV",         desc: "图像/视频解码加速（可选，缺失时用 Pillow + ffmpeg）"},
  soundfile:   {display: "SoundFile",      desc: "音频解码加速（可选，缺失时用 ffmpeg）"},
};
/* 依赖分组的中文名（后端给 group 字符串，前端只负责翻译） */
const DEP_GROUP = {
  core: "核心", window: "窗口", dev: "测试", optional: "可选",
};
let SYSTEM_CACHE = null, DEVICE_CACHE = null;

/* 字节 → 人话：<1GB 用 MB、<1MB 用 KB，一律两位小数（如 2.57GB） */
function fmtBytes(b) {
  if (b === null || b === undefined || isNaN(b)) return "—";
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let i = 0, v = b;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return v.toFixed(2) + units[i];
}
const fmtPct = p => (p === null || p === undefined || isNaN(p))
  ? "—" : Number(p).toFixed(2) + "%";

async function refreshEnv(force) {
  const q = force ? "?force=1" : "";
  const msg = $("env-msg");
  if (force) msg.textContent = "读取中…";
  try {
    const [s, d] = await Promise.all([
      api("/api/system"),
      // 首次调用要跑 system_profiler（探测 GPU 型号/显存，最长 25s）+
      // 一次 torch GPU 探测，默认 20s 会被掐断 → 主页第一屏报「获取失败」。
      api("/api/system/devices" + q, undefined, {timeout: 45000})]);
    SYSTEM_CACHE = s;
    DEVICE_CACHE = d;
    renderDevices(d);
    renderDeps();
    renderMonitors(d);
    if (force) msg.textContent = "";
    // 2026-11-04：GUI 不再自动检查依赖更新（也不提供安装按钮）。
    // 安装 / 更新统一走终端脚本 start/setup_env.py（install / update）。
  } catch (e) {
    msg.textContent = "环境检查失败: " + e.message;
    $("device-card").innerHTML =
      `<div class="dev-item"><div class="dev-k">设备</div>
       <div class="dev-v">获取失败</div>
       <div class="dev-s">${esc(e.message)}</div></div>`;
  }
}

/* 设备大框：CPU / GPU / 内存 / 显存（统一内存机型合并为一项「统一内存」） */
function renderDevices(d) {
  const mem = d.memory || {};
  const unified = !!d.unified_memory;
  const items = [];

  // 4C8T 这类超线程 CPU 不能把逻辑线程数当核心数报（旧版把 8 线程
  // 显示成「8 核心」）。后端给出 physical_cores / logical_cores 后分开写。
  const cpu = d.cpu || {};
  const pc = (cpu.physical_cores != null) ? cpu.physical_cores : cpu.cores;
  const lc = (cpu.logical_cores != null) ? cpu.logical_cores : pc;
  let coreTxt = "";
  if (pc != null && lc != null && lc > pc)
    coreTxt = `${pc} 核 ${lc} 线程（超线程）`;
  else if (lc != null)
    coreTxt = `${lc} 核心${cpu.hyperthreaded ? "（逻辑）" : ""}`;
  else if (pc != null)
    coreTxt = `${pc} 核心`;
  items.push(`<div class="dev-item"><div class="dev-k">CPU</div>
    <div class="dev-v" title="${esc(cpu.model || "")}">${esc(cpu.model || "")}</div>
    <div class="dev-s">${esc(coreTxt)}</div></div>`);

  const gpus = d.gpus || [];
  if (gpus.length) {
    for (const g of gpus) {
      // 有 GPU 但 torch 用不了：小字点明是这张卡的型号不支持，而不是含糊报错
      const sub = g.torch_ok
        ? "支持 PyTorch"
        : `<span class="dev-s warn">${esc(g.note || "不支持 torch")}</span>`;
      items.push(`<div class="dev-item"><div class="dev-k">GPU</div>
        <div class="dev-v" title="${esc(g.model)}">${esc(g.model)}</div>
        <div class="dev-s">${sub}</div></div>`);
    }
  } else {
    items.push(`<div class="dev-item"><div class="dev-k">GPU</div>
      <div class="dev-v">—</div><div class="dev-s">未发现</div></div>`);
  }

  if (unified) {
    // 统一内存：内存与显存是同一块，只报一项，写清多大
    items.push(`<div class="dev-item"><div class="dev-k">统一内存</div>
      <div class="dev-v">${fmtBytes(mem.total)}</div>
      <div class="dev-s">CPU 与 GPU 共用</div></div>`);
  } else {
    items.push(`<div class="dev-item"><div class="dev-k">内存</div>
      <div class="dev-v">${fmtBytes(mem.total)}</div>
      <div class="dev-s">${mem.used != null ? "已用 " + fmtBytes(mem.used) : ""}</div></div>`);
    const vram = d.vram || {};
    items.push(`<div class="dev-item"><div class="dev-k">显存</div>
      <div class="dev-v">${vram.total == null ? "—" : fmtBytes(vram.total)}</div>
      <div class="dev-s">${vram.used != null ? "已用 " + fmtBytes(vram.used) : ""}</div></div>`);
  }
  $("device-card").innerHTML = items.join("");
}

function renderDeps() {
  const s = SYSTEM_CACHE;
  if (!s) return;
  const grid = $("dep-grid");
  // 只读展示：已装 / 未装 / 启动核心，三种状态都没有可点的安装按钮。
  // 安装与更新统一在终端跑 start/setup_env.py（install / update）。
  grid.innerHTML = (s.dependencies || []).map(d => {
    const meta = DEP_META[d.name] ||
      {display: d.title || d.name, desc: d.why || d.install || ""};
    const line2 = d.ok
      ? (meta.desc || d.why || "")
      : (d.group === "must"
           ? "启动核心（用终端脚本重装）"
           : (d.install ? "未安装 · 终端脚本可装" : "未安装"));
    const right = d.group === "must"
      ? `<span class="dep-state">启动核心</span>`
      : d.ok
        ? `<span class="dep-state ok">已安装</span>`
        : `<span class="dep-state">未安装</span>`;
    const ver = (d.ok && d.version)
      ? `<span class="dep-ver">${esc(d.version)}</span>` : "";
    return `<div class="dep-card"><div class="dep-info">
      <div class="dep-title"><span class="dep-name" title="${esc(meta.display)}">${esc(meta.display)}</span>${ver}</div>
      <div class="dep-desc" title="${esc(line2)}">${esc(line2)}</div>
    </div>${right}</div>`;
  }).join("");
}
/* 刷新按钮：点击后图标持续旋转，请求完成（或至少转满 650ms）后停下，
   期间禁用按钮防重复点击。三个刷新按钮共用这一套。 */
function wireRefreshSpin(id, fn) {
  const btn = $(id), svg = btn.querySelector("svg");
  btn.addEventListener("click", async () => {
    if (btn.disabled) return;
    if (svg) svg.classList.add("spinning");
    btn.disabled = true;
    const t0 = Date.now();
    try { await fn(); }
    finally {
      setTimeout(() => {
        if (svg) svg.classList.remove("spinning");
        btn.disabled = false;
      }, Math.max(0, 650 - (Date.now() - t0)));
    }
  });
}
wireRefreshSpin("env-refresh", () => {
  // 只重读设备 / 依赖状态；不联网、不检查更新、不安装
  $("env-msg").textContent = "";
  return Promise.all([refreshEnv(true), refreshMonitors()]);
});

/* =========================================================================
   主页：资源监视器
   -------------------------------------------------------------------------
   每栏一个被拉长的框：左（标题 + 型号副标题）/ 中（灰底槽 + 蓝色占用条）/
   右（百分比 + 实际占用量，两位小数）。只有内存与显存报占用量。
   ========================================================================= */
/* 资源监视器：DOM 只建一次，之后每轮轮询只改文本和 width。
   之前每轮 innerHTML 整块重建 → .mon-fill 的 width transition 永远不触发
   （新元素直接以最终宽度出生），用户看到的就是「bar 突然跳到新值」。
   复用 DOM 后 0.6s 的平滑滑动才真正生效；行数变化（GPU 增减）时才重建。 */
function monSkeleton() {
  return `<div class="mon-card">
    <div class="mon-left"><div class="mon-title"></div><div class="mon-sub"></div></div>
    <div class="mon-bar"><div class="mon-fill"></div></div>
    <div class="mon-right"><div class="mon-pct"></div><div class="mon-amt"></div></div>
  </div>`;
}
function renderMonitors(d) {
  const rows = [];
  const cpu = d.cpu || {};
  rows.push({title: "CPU", sub: cpu.model || "", pct: cpu.usage,
             used: null, total: null, warn: false});

  const gpus = d.gpus || [];
  for (const g of gpus) {
    const sub = g.torch_ok ? g.model : (g.note || (g.model + " 不支持 torch"));
    rows.push({title: "GPU", sub, pct: g.usage, used: null, total: null,
               warn: !g.torch_ok});
  }
  if (!gpus.length)
    rows.push({title: "GPU", sub: "未发现", pct: null, used: null,
               total: null, warn: false});

  const mem = d.memory || {};
  const memPct = (mem.total && mem.used != null) ? mem.used / mem.total * 100 : null;
  rows.push({title: d.unified_memory ? "统一内存" : "内存",
             sub: d.unified_memory ? "CPU 与 GPU 共用" : "系统物理内存",
             pct: memPct, used: mem.used, total: mem.total, warn: false});

  if (!d.unified_memory) {
    const vram = d.vram || {};
    const vPct = (vram.total && vram.used != null) ? vram.used / vram.total * 100 : null;
    rows.push({title: "显存",
               sub: gpus.map(g => g.model).join(" / ") || "未发现",
               pct: vPct, used: vram.used, total: vram.total, warn: false});
  }

  const host = $("monitors");
  if (host.children.length !== rows.length)
    host.innerHTML = rows.map(() => monSkeleton()).join("");

  rows.forEach((r, i) => {
    const card = host.children[i];
    if (!card) return;
    const p = (r.pct === null || r.pct === undefined || isNaN(r.pct))
      ? null : Math.max(0, Math.min(100, r.pct));
    const amt = (r.used === null || r.used === undefined)
      ? "" : fmtBytes(r.used) + (r.total ? " / " + fmtBytes(r.total) : "");
    card.querySelector(".mon-title").textContent = r.title;
    const sub = card.querySelector(".mon-sub");
    sub.textContent = r.sub || "";
    sub.title = r.sub || "";
    sub.classList.toggle("warn", !!r.warn);
    card.querySelector(".mon-fill").style.width =
      (p === null ? 0 : p.toFixed(2)) + "%";
    card.querySelector(".mon-pct").textContent = fmtPct(p);
    const amtEl = card.querySelector(".mon-amt");
    amtEl.textContent = amt;
    amtEl.style.display = amt ? "" : "none";
  });
}

/* 进程监视器：内存 / 显存两列各自独立切换「百分比 ↔ 实际占用」 */
let MEM_ACTUAL = false, VRAM_ACTUAL = false;
let PROC_CACHE = [], PROC_ERROR = "";
function procMemCell(p) {
  if (MEM_ACTUAL) return p.mem_bytes == null ? "—" : fmtBytes(p.mem_bytes);
  return p.mem_pct == null ? "—" : fmtPct(p.mem_pct);
}
function procVramCell(p) {
  if (VRAM_ACTUAL) return p.vram_bytes == null ? "—" : fmtBytes(p.vram_bytes);
  return p.gpu_pct == null ? "—" : fmtPct(p.gpu_pct);
}
function syncProcHeaders() {
  $("proc-mem-th").textContent = MEM_ACTUAL ? "内存（实际）" : "内存";
  $("proc-vram-th").textContent = VRAM_ACTUAL ? "显存（实际）" : "显存";
}
function renderProcesses(procs, err) {
  PROC_CACHE = procs || [];
  PROC_ERROR = err || "";
  syncProcHeaders();
  const tb = $("proc-tbody");
  $("proc-note").textContent = PROC_ERROR
    ? "进程列表读取失败"
    : PROC_CACHE.length
      ? `${PROC_CACHE.length} 个进程　·　内存 / 显存列点击可切换「百分比 ↔ 实际占用」`
      : "内存 / 显存列点击可在「百分比 ↔ 实际占用」间切换";
  if (!PROC_CACHE.length) {
    // 读不到 ≠ 没有：后端给了 error 就照实显示，别让用户以为进程都退了
    tb.innerHTML = `<tr><td colspan="7" class="hint">${esc(
      PROC_ERROR || "没有正在运行的 GARGANTUA 进程")}</td></tr>`;
    return;
  }
  tb.innerHTML = PROC_CACHE.map(p => `<tr>
    <td>${esc(p.name)}</td>
    <td class="mono">${p.pid}</td>
    <td class="mono">${(p.ports && p.ports.length) ? p.ports.join(", ") : "—"}</td>
    <td class="mono">${p.cpu == null ? "—" : fmtPct(p.cpu)}</td>
    <td class="mono clickable" data-col="mem" title="点击切换 百分比 / 实际占用">${procMemCell(p)}</td>
    <td class="mono clickable" data-col="vram" title="点击切换 百分比 / 实际占用">${procVramCell(p)}</td>
    <td class="right">${p.external
      ? `<button class="act sec xs" disabled title="该进程不属于当前项目目录，不能在此结束">外部</button>`
      : `<button class="act danger xs" data-kill="${p.pid}"${p.self
          ? ` data-self="1" title="这是当前 GARGANTUA 自身进程，结束会导致应用退出"`
          : ""}>结束</button>`}</td>
  </tr>`).join("");
  tb.querySelectorAll("[data-kill]").forEach(b => {
    b.onclick = () => killProc(Number(b.dataset.kill), b.dataset.self === "1");
  });
  tb.querySelectorAll("[data-col]").forEach(td => {
    td.onclick = () => toggleProcCol(td.dataset.col);
  });
}
function toggleProcCol(col) {
  if (col === "mem") MEM_ACTUAL = !MEM_ACTUAL; else VRAM_ACTUAL = !VRAM_ACTUAL;
  renderProcesses(PROC_CACHE);
}
$("proc-mem-th").onclick = () => toggleProcCol("mem");
$("proc-vram-th").onclick = () => toggleProcCol("vram");

/* 顶部磨砂条常驻在 #topbar（跨所有页面），页内不再需要滚动状态类 */

async function killProc(pid, isSelf) {
  if (isSelf && !await uiConfirm(
      "这是 GARGANTUA 自身的 server 进程。结束它会导致当前应用退出，确定要继续吗？")) {
    return;
  }
  const msg = $("mon-msg");
  try {
    await api("/api/system/kill", {pid});
    msg.textContent = `已向 ${pid} 发送终止信号`;
    toast(`已结束进程 ${pid}`);
  } catch (e) {
    msg.textContent = "结束失败: " + e.message;
    toast("结束失败: " + e.message, true);
  }
  refreshMonitors();
}

// 首次调用要跑 system_profiler（最长 25s），期间 2s 轮询会把请求堆起来；
// 用一个 in-flight 标记挡住重入，避免慢请求堆积、渲染乱序。
let MON_BUSY = false;
async function refreshMonitors() {
  if (MON_BUSY) return;
  MON_BUSY = true;
  try {
    const [d, p] = await Promise.all([
      api("/api/system/devices", undefined, {timeout: 45000}),
      api("/api/system/processes")]);
    DEVICE_CACHE = d;
    renderDevices(d);
    renderMonitors(d);
    renderProcesses(p.processes, p.error);
    $("mon-msg").textContent = "";
  } catch (e) {
    $("mon-msg").textContent = "监视数据获取失败: " + e.message;
  } finally {
    MON_BUSY = false;
  }
}
// 资源监视 2s 轮询（主页激活时才发请求）
setInterval(() => {
  if ($("p-home").classList.contains("active")) refreshMonitors();
}, 2000);
applyTheme();
refreshEnv(false);
refreshMonitors();
refreshStatus();

/* ---------- 定制 token 时间步格式 ----------
   格式（GARGANTUA token stream .gts，以 {} 为时间步原子）：
     {} = 一个时间步；同一 {} 内多 token 以 | 分隔，无空格；
     <...> = 特殊 token（<user>/<model>/<stop>/<caps>）；裸字符 = 可见 token。
     特殊 token（<user>、<model> 等）各自独占一个时间步。
     视觉段：每一帧一个 {}，该 {} 内平铺该帧全部 patch token。
     独立音频：每 533 采样一个 {<wave:hex>}。
     绘图时间步：每个图元 = 一个 {}，内部 9 个 <id:...> token 平铺。
*/
const _B64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
function b64ToBytes(b64) {
  const out = new Uint8Array(b64.length * 3 / 4);
  let p = 0, buf = 0, bits = 0;
  for (const c of b64) {
    if (c === "=") break;
    buf = (buf << 6) | _B64_CHARS.indexOf(c); bits += 6;
    if (bits >= 8) { bits -= 8; out[p++] = (buf >> bits) & 0xFF; }
  }
  return out.subarray(0, p);
}
/* token-5 文本（spec 第 5b 节）：{text: C732} / {wave: HEX} / {visual: HEX x: 1 y: 2}
   音频/视觉的 hex 载荷一律由后端生成（r.token5），前端只在没有后端文本时兜底。
   ID 是大写十六进制、不定宽（2026-10-31 契约）。 */
const _IDHEX = n => Number(n).toString(16).toUpperCase();

function stepsToToken5(steps, bound) {
  const openTag = bound === "user" ? "<user>" : "<model>";   // <user> / <model>
  const closeTag = bound === "user" ? "<user/>" : "<model/>";  // <user/> / <model/>
  const parts = [openTag];
  for (const st of steps) {
    const ids = st.token_ids || [];
    if (!ids.length) continue;                 // 模态步：由后端 token5 给出
    if (st.modality === "draw") {
      // 绘图：序列最前面是画布 token（单独一步），其后一个图元 = 9 token = 一步
      let j = 0;
      if (ids.length % 9 === 1) {
        parts.push("{" + "text: " + _IDHEX(ids[0]) + "}");
        j = 1;
      }
      for (; j + 9 <= ids.length; j += 9)
        parts.push("{" + ids.slice(j, j + 9).map(t => "text: " + _IDHEX(t)).join(" | ") + "}");
      continue;
    }
    for (const id of ids) parts.push("{text: " + _IDHEX(id) + "}");
  }
  parts.push(closeTag);
  return parts.join("");
}

/* ---------- 模型 ---------- */
async function refreshModels() {
  try {
    const r = await api("/api/models");
    $("model-tbody").innerHTML = r.models.map(m =>
      `<tr><td>${esc(m.name)} ${m.name === r.active ? '<span class="badge on">当前</span>' : ""}${m.share_enc_dec ? '<span class="badge">共享权重</span>' : ""}</td>
       <td>${m.params ? (m.params/1e6).toFixed(1)+"M" : "—"}</td>
       <td class="right">
         <button class="act sec xs" data-load="${esc(m.name)}">加载</button>
         <button class="act sec xs del" data-del="${esc(m.name)}">删除</button>
       </td></tr>`).join("");
    $("model-tbody").querySelectorAll("[data-load]").forEach(b => {
      b.onclick = () => loadModel(b.dataset.load);
    });
    $("model-tbody").querySelectorAll("[data-del]").forEach(b => {
      b.onclick = () => deleteModel(b.dataset.del);
    });
    const target = $("train-target");
    const cur = target.value;
    target.innerHTML = '<option value="">当前活动模型</option>' +
      r.models.map(m => `<option value="${esc(m.name)}">${esc(m.name)}</option>`).join("");
    target.value = cur;
  } catch (e) {}
}
async function loadModel(name) {
  try { await api("/api/model/load", {name});
        $("model-create-msg").textContent = "已加载 " + name;
        refreshModels(); refreshStatus(); }
  catch (e) { $("model-create-msg").textContent = "错误: " + e.message; }
}
async function deleteModel(name) {
  if (!confirm(`确定删除模型「${name}」？此操作不可恢复。`)) return;
  try {
    await api("/api/model/delete", {name});
    $("model-create-msg").textContent = "已删除 " + name;
    refreshModels(); refreshStatus();
  } catch (e) { $("model-create-msg").textContent = "错误: " + e.message; }
}
/* 权重模式联动（2026-11-04）：共享权重要求 enc/dec 等深。
   共享时 dec 输入框锁定并镜像 enc，提交也不读它的值——「强制一样」
   不能只靠 UI 提示，后端另有同一份归一化规则兜底。 */
function modelShareMode() { return $("model-share").value === "shared"; }
let savedDecBlocks = null;   // 切回「分离」时还原用户原来填的 dec 值
function syncModelShareMode() {
  const shared = modelShareMode();
  const enc = $("model-enc"), dec = $("model-dec");
  if (shared) {
    if (savedDecBlocks === null) savedDecBlocks = dec.value;
    dec.value = enc.value;
  } else if (savedDecBlocks !== null) {
    dec.value = savedDecBlocks;
    savedDecBlocks = null;
  }
  dec.disabled = shared;
  dec.title = shared ? "共享权重下 dec blocks 强制等于 enc blocks" : "";
}
$("model-share").addEventListener("change", () => {
  syncModelShareMode();
  refreshModelEstimate();
});
$("model-enc").addEventListener("input", () => {
  if (modelShareMode()) $("model-dec").value = $("model-enc").value;
  refreshModelEstimate();
});
syncModelShareMode();

/* ---------- 建模参数量预测 ----------
   与 /api/model/init 用同一套入参（modelFormValues），后端按 spec 解析式
   估算、不建模型。表单任一字段变化 → 180ms 防抖 → 重算。 */
function modelFormValues() {
  // 空值/非数字不下发（服务端取 spec 默认）。0 对压缩上下文是合法值，
  // 不能用 `parseInt(v) || undefined` 把 0 吞掉。
  const intOrUndef = (v) => {
    const n = parseInt(v, 10);
    return Number.isFinite(n) ? n : undefined;
  };
  const shared = modelShareMode();
  const encBlocks = intOrUndef($("model-enc").value);
  return {
    share_enc_dec: shared,
    d_model: intOrUndef($("model-dmodel").value),
    n_enc_blocks: encBlocks,
    // 共享时强制相等：直接复用 enc 的值，不依赖被锁定的 dec 输入框
    n_dec_blocks: shared ? encBlocks : intOrUndef($("model-dec").value),
    ctx_full: intOrUndef($("model-ctx-full").value),
    ctx_compress: intOrUndef($("model-ctx-compress").value),
    csa_ratio: intOrUndef($("model-csa-ratio").value),
    hca_ratio: intOrUndef($("model-hca-ratio").value),
    vocab: $("model-vocab").value || "base",
    dtype: $("model-dtype").value || "fp32",
  };
}

function fmtParams(n) {
  if (!Number.isFinite(n)) return "—";
  if (n >= 1e9) return (n / 1e9).toFixed(3) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}

let EST_TIMER = null, EST_SEQ = 0;
function refreshModelEstimate() {
  if (!$("model-est-body")) return;
  clearTimeout(EST_TIMER);
  EST_TIMER = setTimeout(runModelEstimate, 180);
}
async function runModelEstimate() {
  const body = $("model-est-body");
  if (!body) return;
  const seq = ++EST_SEQ;
  try {
    const r = await api("/api/model/estimate", modelFormValues());
    if (seq !== EST_SEQ) return;             // 只认最后一次请求
    const rows = (r.parts || []).map(p => `<tr>
      <td>${esc(p.label)}</td>
      <td class="right est-num">${fmtParams(p.params)}</td>
      <td class="right est-num">${fmtBytes(p.bytes)}</td>
      <td class="right est-num">${fmtBytes(p.file_bytes)}</td>
    </tr>`).join("");
    const t = r.total || {};
    body.innerHTML = rows + `<tr class="total">
      <td>合计</td>
      <td class="right est-num">${fmtParams(t.params)}</td>
      <td class="right est-num">${fmtBytes(t.bytes)}</td>
      <td class="right est-num">${fmtBytes(t.file_bytes)}</td>
    </tr>`;
    const c = r.config || {};
    $("model-est-note").textContent =
      `${c.dtype} · ${c.bits} bit/参数 · 词表 ${Number(c.vocab_size).toLocaleString()}`;
    $("model-est-foot").textContent =
      `d_model=${c.d_model} · enc ${c.n_enc_blocks} / dec ${c.n_dec_blocks}` +
      `${c.share_enc_dec ? "（共享权重）" : ""} · ` +
      `理论存储按所选精度折算；init 一律写 fp32 权重文件，` +
      `低比特在加载后量化，不改变磁盘占用。`;
  } catch (e) {
    if (seq !== EST_SEQ) return;
    body.innerHTML =
      `<tr><td colspan="4" class="hint">估算失败: ${esc(e.message)}</td></tr>`;
    $("model-est-note").textContent = "参数不合法";
    $("model-est-foot").textContent = "";
  }
}

// d_model 完全无上限（steve 2026-11-05 拍板）：只约束 ≥128 且为 128 的倍数，
// 用 <input type="number" min="128" step="128"> 直接输入，不再下拉枚举。

["model-dmodel", "model-enc", "model-dec", "model-ctx-full",
 "model-ctx-compress", "model-csa-ratio", "model-hca-ratio",
 "model-vocab", "model-dtype"].forEach(id => {
  const el = $(id);
  if (!el) return;
  el.addEventListener("input", refreshModelEstimate);
  el.addEventListener("change", refreshModelEstimate);
});

$("model-create").onclick = async () => {
  const name = $("model-name").value.trim();
  if (!name) return;
  $("model-create-msg").textContent = "初始化中…";
  try {
    await createModel(name, false);
  } catch (e) {
    // 已存在 → 确认后覆盖（服务端 init_new 默认拒绝覆盖）
    if (e.message.includes("已存在") &&
        await uiConfirm("模型 " + name + " 已存在，覆盖其检查点？")) {
      try {
        await createModel(name, true);
        return;
      } catch (e2) {
        $("model-create-msg").textContent = "错误: " + e2.message;
      }
    } else {
      $("model-create-msg").textContent = "错误: " + e.message;
    }
  }
};
async function createModel(name, overwrite) {
  const {task_id} = await api("/api/model/init",
                              {name, overwrite, ...modelFormValues()});
  await pollTask(task_id, {interval: 1500, timeout: 120000});
  $("model-create-msg").textContent = "创建成功";
  refreshModels(); await refreshStatus();
  // 首个模型创建后自动激活（此前不加载，训练/推理仍报「无活动模型」）。
  if (ACTIVE_MODEL !== name) {
    try { await api("/api/model/load", {name});
          refreshModels(); await refreshStatus(); }
    catch (e) { $("model-create-msg").textContent =
      "已创建（自动激活失败: " + e.message + "）"; }
  }
}
wireRefreshSpin("model-refresh", refreshModels);

/* ---------- 训练 ---------- */
$("train-enqueue").onclick = async () => {
  const file = $("train-file").files[0];
  const msg = $("train-enqueue-msg");
  if (!file) { msg.textContent = "请先选择 .tokens 训练数据文件"; return; }
  try {
    const r = await api("/api/train/enqueue", {
      tokens_b64: await fileToB64(file),
      optimizer: $("train-mode").value,
    }, {timeout: API_UPLOAD_TIMEOUT_MS});
    msg.textContent = `已入队 ${r.queued[0]}（${r.n_tokens} tokens / ` +
      `${r.n_steps} 时间步）`;
    $("train-file").value = "";
    refreshQueue(); refreshStatus();
  } catch (e) { msg.textContent = "错误: " + e.message; }
};
async function refreshQueue() {
  try {
    const r = await api("/api/train/queue");
    const el = $("train-queue");
    if (!r.items.length) { el.innerHTML = '<span class="hint">队列为空</span>'; return; }
    el.innerHTML = r.items.map(it =>
      `<div class="queue-item"><span class="name" title="${esc(it.name)}">${esc(it.name)}
        <span class="hint">（${esc(it.optimizer || "?")}）</span></span>
       <button class="act danger xs" data-del="${esc(it.name)}">删除</button></div>`
    ).join("");
    el.querySelectorAll("[data-del]").forEach(b => {
      b.onclick = () => deleteQueue(b.dataset.del);
    });
  } catch (e) { $("train-queue").textContent = "错误: " + e.message; }
}
async function deleteQueue(name) {
  try { await api("/api/train/delete", {name}); refreshQueue(); refreshStatus(); }
  catch (e) { toast(e.message, true); }
}
wireRefreshSpin("queue-refresh", refreshQueue);
$("train-start").onclick = async () => {
  try {
    const r = await api("/api/train/start", {
      steps: clampNum($("train-steps").value, 1, 1e6, 20),
      // & ~1：强制为偶数（block_size 会二分给 enc/dec，奇数令前向崩溃）
      block_size: clampNum($("train-block").value, 4, 1024, 32) & ~1,
      optimizer: $("train-mode").value,
      target: $("train-target").value});
    $("train-state").textContent = r.started ? "已启动" : "已在运行中";
  } catch (e) { $("train-state").textContent = "错误: " + e.message; }
};
$("train-stop").onclick = async () => {
  try { await api("/api/train/stop", {});
        $("train-state").textContent = "停止请求已发送"; }
  catch (e) { $("train-state").textContent = "错误: " + e.message; }
};

/* Loss 曲线：每步 = 单个样本的独立 loss，逐点折线，悬停显示数值 */
function drawLoss(tail) {
  const cv = $("loss-canvas"), ctx = cv.getContext("2d");
  const cs = getComputedStyle(document.documentElement);
  const cFg2 = cs.getPropertyValue("--fg2").trim() || "#98a2b3";
  const cTrack = cs.getPropertyValue("--track").trim() || "#242b35";
  const cAcc = cs.getPropertyValue("--accent").trim() || "#4c8dff";
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = cFg2; ctx.font = "12px ui-monospace, monospace";
  if (!tail.length) { ctx.fillText("暂无 loss 数据", 20, 30); cv._pts = null; return; }
  const losses = tail.map(r => r.loss);
  const lo = Math.min(...losses), hi = Math.max(...losses);
  const pad = 40, W = cv.width - pad * 2, H = cv.height - pad * 2;
  ctx.strokeStyle = cTrack; ctx.strokeRect(pad, pad, W, H);
  ctx.fillText(hi.toFixed(3), 4, pad + 4); ctx.fillText(lo.toFixed(3), 4, pad + H);
  ctx.fillText("step " + tail[0].step + " → " + tail[tail.length - 1].step,
               pad, cv.height - 10);
  const pts = tail.map((r, i) => {
    const x = pad + (tail.length === 1 ? W / 2 : i / (tail.length - 1) * W);
    const y = pad + H - (hi === lo ? H / 2 : (r.loss - lo) / (hi - lo) * H);
    return {x, y, step: r.step, loss: r.loss, file: r.file || ""};
  });
  cv._pts = pts;
  ctx.strokeStyle = cAcc; ctx.lineWidth = 1.2; ctx.beginPath();
  pts.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
  ctx.stroke();
  ctx.fillStyle = cAcc;
  pts.forEach(p => { ctx.beginPath(); ctx.arc(p.x, p.y, 2, 0, 7); ctx.fill(); });
}
$("loss-canvas").onmousemove = e => {
  const cv = e.currentTarget, pts = cv._pts;
  if (!pts || !pts.length) return;
  const rect = cv.getBoundingClientRect();
  const sx = cv.width / rect.width, sy = cv.height / rect.height;
  const mx = (e.clientX - rect.left) * sx, my = (e.clientY - rect.top) * sy;
  let best = null, bd = 1e9;
  for (const p of pts) {
    const d = (p.x - mx) ** 2 + (p.y - my) ** 2;
    if (d < bd) { bd = d; best = p; }
  }
  if (best && bd < 4000) {
    cv.title = `step ${best.step}  loss=${best.loss}` +
               (best.file ? `\n样本: ${best.file}` : "");
  }
};
async function refreshTrain() {
  try {
    const s = await api("/api/train/status");
    drawLoss(s.loss_tail || []);
    $("train-log").textContent = (s.log_tail || []).slice(-200).join("\n") || "—";
    $("train-state").textContent =
      (s.running ? "运行中 step=" + s.step : "空闲 step=" + s.step)
      + (s.optimizer ? " · " + s.optimizer : "");
  } catch (e) {}
}
setInterval(refreshTrain, 2000);

/* ===========================================================================
   推理：显示插件宿主（2026-11-04）
   ---------------------------------------------------------------------------
   推理界面（左右边栏之间、顶栏之下，**含输入框**）整块归插件。宿主只做四件
   事，不碰任何聊天/输入的视觉：
     1. 给一块满高容器（#plugin-host）；
     2. 提供受控 API：流式推理 / 重置 / 实时统计 / 模型列表 / toast；
     3. 管生命周期：挂载、卸载、切插件（含插件自己的 CSS）；
     4. 在开发者选项里提供选择器。
   插件协议见 service/web/plugins/README.md——它只连 /api/infer 与
   /api/infer/stream 两个端点，其余（DOM、输入解析、展示方式）完全自理。
   =========================================================================== */
/* 内置插件只注册「token 对话」一个。base 词表插件仍在开发中，按需求
   「不显示」——不注册、不出现在选择器里；开发计划写在 plugins/README.md。
   其它显示器由开发者用「开发者选项 → 显示插件 → 导入插件…」从本地 .js
   导入，服务端落到 service/web/plugins/ 并在 /api/plugin/list 登记。 */
const BUILTIN_PLUGINS = [
  {id: "tokens", name: "token 对话", entry: "plugins/tokens.js",
   css: "plugins/tokens.css",
   desc: "微信式多轮对话：右边是你的输入、左边是模型输出，上下文逐轮保留；" +
         "每个 token 以转换器格式 {text: B46A} / <caps> 呈现；Enter 不发送，" +
         "只能点「发送」。"},
];
const PLUGINS = BUILTIN_PLUGINS.slice();   // 运行期 append 导入的插件
const PLUGIN_KEY = "gargantua.infer.plugin";
let PLUGIN_CUR = null;          // {id, instance, cleanup, cssEl, ctx, controller}
let PLUGIN_GEN = 0;            // 挂载世代：并发切插件时丢弃过期 import 结果

function pluginById(id) {
  return PLUGINS.find(p => p.id === id && !p.status) ||
         PLUGINS.find(p => !p.status) || PLUGINS[0];
}
function selectedPluginId() {
  let id = null;
  try { id = localStorage.getItem(PLUGIN_KEY); } catch (e) {}
  return pluginById(id).id;
}
function setSelectedPluginId(id) {
  try { localStorage.setItem(PLUGIN_KEY, id); } catch (e) {}
}

/* 流式推理客户端：POST 起任务 → 按 from 游标增量拉 token。
   为什么不用长连接：server 是纯标准库 http.server，一条连接一个线程，
   长连接会把「停止/切页」都变成连接生命周期问题；轮询 90ms 一帧在这个
   数据量下完全够（一个 token 几秒才出来）。 */
/* 后端上限 common.MAX_NEW_TOKENS = 2048；前端不再写死 512，
   否则 513..2048 这一档永远用不上（/api/status 也会下发同一个值）。 */
const INFER_MAX_NEW = 2048;
const INFER_WALL_MS = 10 * 60 * 1000;   // 整轮流式的墙钟上限

/* 流式推理客户端：POST 起任务 → 按 from 游标增量拉 token。
   为什么不用长连接：server 是纯标准库 http.server，一条连接一个线程，
   长连接会把「停止/切页」都变成连接生命周期问题；轮询 90ms 一帧在这个
   数据量下完全够（一个 token 几秒才出来）。

   2026-11-04 审查修复：
     · signal —— 切插件/卸载时中止轮询，不让死实例继续占用后端串行队列；
     · 墙钟上限 —— 服务端若一直回 running，不能永远卡在「生成中…」；
     · 单次 GET 失败重试 3 次 —— 本地服务偶发断连不该整轮报废。 */
async function inferStream(req, h, signal) {
  h = h || {};
  const aborted = () => !!(signal && signal.aborted);
  const body = {
    stream: true,
    max_new_tokens: clampNum(req.maxNew, 1, INFER_MAX_NEW, 32),
    temperature: clampNum(req.temperature, 0.05, 5, 0.8),
    top_p: clampNum(req.topP, 0.01, 1, 0.95),
  };
  if (Array.isArray(req.tokenIds) && req.tokenIds.length) {
    body.token_ids = req.tokenIds;
  } else if (req.prompt) {
    body.prompt = req.prompt;
  } else {
    throw new Error("需要 tokenIds 或 prompt");
  }
  const r = await api("/api/infer", body, {signal});
  const sid = r.task_id;
  if (h.onStart) h.onStart({taskId: sid});
  const t0 = Date.now();
  let from = 0;
  let fails = 0;
  for (;;) {
    if (aborted()) throw new Error("已取消");
    if (Date.now() - t0 > INFER_WALL_MS) {
      throw new Error(`推理超时（${Math.round(INFER_WALL_MS / 1000)}s）`);
    }
    let s;
    try {
      s = await api(
        `/api/infer/stream?task_id=${encodeURIComponent(sid)}&from=${from}`,
        undefined, {timeout: 120000, signal});
      fails = 0;
    } catch (e) {
      if (aborted()) throw new Error("已取消");
      if (++fails <= 3) {                 // 瞬时失败：退避重试，不整轮报废
        await new Promise(res => setTimeout(res, 250 * fails));
        continue;
      }
      throw e;
    }
    const got = s.tokens || [];
    for (const tk of got) {
      from += 1;
      if (h.onToken) h.onToken(tk, from);
    }
    // 服务端的 next 是权威游标（慢轮询/丢帧时靠它对齐，不靠本地累加）
    if (typeof s.next === "number") from = s.next;
    if (s.status === "done") { if (h.onDone) h.onDone(s.result || {}, s); break; }
    if (s.status === "error") {
      if (h.onError) h.onError(new Error(s.error || "推理失败"));
      break;
    }
    await new Promise(res => setTimeout(res, 90));
  }
}

function makePluginCtx(meta, host, signal) {
  const cleanups = [];
  const ctx = {
    root: host,
    plugin: {id: meta.id, name: meta.name},
    signal,                       // 插件可自选监听：异步回调别碰已卸载的 DOM
    api: {
      infer: (req, h) => inferStream(req, h, signal),
      reset: () => api("/api/infer/reset", {}),
      stats: () => api("/api/infer/stats"),
      status: () => api("/api/status"),
      models: () => api("/api/models"),
      loadModel: (name) => api("/api/model/load", {name}),
      toast: (msg, bad) => toast(msg, bad),
      confirm: (title, body) => modalOpen(title, body, "确定", true),
      fmtBytes,
    },
    /* 每个插件一份命名空间，互不污染；值走 JSON，读不到就用默认 */
    config: {
      get(k, d) {
        try {
          const v = localStorage.getItem(`gargantua.plugin.${meta.id}.${k}`);
          return v === null ? d : JSON.parse(v);
        } catch (e) { return d; }
      },
      set(k, v) {
        try {
          localStorage.setItem(`gargantua.plugin.${meta.id}.${k}`,
                               JSON.stringify(v));
        } catch (e) {}
      },
    },
    onCleanup(fn) { if (typeof fn === "function") cleanups.push(fn); },
  };
  ctx._cleanup = () => {
    cleanups.splice(0).forEach(fn => { try { fn(); } catch (e) {} });
  };
  return ctx;
}

/* 把侧栏的采样参数（temperature / top_p）推到当前插件的配置命名空间。
   插件在 submit 时读 ctx.config，所以这里只要保持 config 与滑块同步即可，
   不碰插件内部状态。两个滑块在 init 时各挂一次监听，切换/重挂插件后也
   靠挂载末尾的调用补同步。 */
function syncSamplingToPlugin() {
  if (PLUGIN_CUR && PLUGIN_CUR.ctx) {
    const t = parseFloat($("infer-temp").value);
    const p = parseFloat($("infer-topp").value);
    if (Number.isFinite(t)) PLUGIN_CUR.ctx.config.set("temperature", t);
    if (Number.isFinite(p)) PLUGIN_CUR.ctx.config.set("topP", p);
  }
}

/* 挂载世代机制（替换原来的 PLUGIN_MOUNTING 布尔丢弃）：
   旧实现在 await import() 期间若再来一次切换会直接 return，localStorage
   已写新值而 UI 仍是旧插件，且 PLUGIN_CUR 非空导致重进页面也不重挂。
   现在每次请求先 ++gen 作废旧挂载；import 回来后校验 gen 与当前选中 id，
   不一致就丢弃结果，最后一次请求一定能生效。 */
async function mountInferPlugin(force) {
  const host = $("plugin-host");
  if (!host) return;
  const gen = ++PLUGIN_GEN;
  const meta = pluginById(selectedPluginId());
  if (PLUGIN_CUR && PLUGIN_CUR.id === meta.id && !force) return;
  unmountInferPlugin();
  host.innerHTML = `<div class="plugin-boot">正在加载「${esc(meta.name)}」…</div>`;
  let cssEl = null;
  try {
    if (meta.css) {
      cssEl = document.createElement("link");
      cssEl.rel = "stylesheet";
      cssEl.href = meta.css;
      cssEl.dataset.plugin = meta.id;
      document.head.appendChild(cssEl);
    }
    const mod = await import("./" + meta.entry);
    // 期间又切了一次插件 / 改了选择：这次结果已过期，别覆盖新插件
    if (gen !== PLUGIN_GEN || pluginById(selectedPluginId()).id !== meta.id) {
      if (cssEl) cssEl.remove();
      return;
    }
    const def = mod.default || mod;
    if (typeof def.mount !== "function") {
      throw new Error("插件没有导出 mount(root, ctx)");
    }
    host.innerHTML = "";
    const controller = new AbortController();
    const ctx = makePluginCtx(meta, host, controller.signal);
    const instance = await def.mount(host, ctx);
    if (gen !== PLUGIN_GEN) {                 // mount 自身也可能是异步的
      try { if (instance && instance.destroy) instance.destroy(); } catch (e) {}
      controller.abort();
      if (cssEl) cssEl.remove();
      return;
    }
    PLUGIN_CUR = {id: meta.id, instance: instance || {}, cssEl,
                  cleanup: ctx._cleanup, ctx, controller};
    // 侧栏的 temperature / top_p 是全局采样参数：把它们写进当前插件的
    // 配置命名空间，插件 submit 时从 ctx.config 读——这样滑块真的能控制生成。
    syncSamplingToPlugin();
  } catch (e) {
    if (gen !== PLUGIN_GEN) return;         // 过期失败不该盖掉新插件的错误态
    if (cssEl) cssEl.remove();
    host.innerHTML = `<div class="plugin-boot err">插件「${esc(meta.name)}」
      加载失败：${esc(e.message)}</div>`;
  }
}

function unmountInferPlugin() {
  if (!PLUGIN_CUR) return;
  try {
    if (PLUGIN_CUR.instance && PLUGIN_CUR.instance.destroy) {
      PLUGIN_CUR.instance.destroy();
    }
  } catch (e) {}
  try { if (PLUGIN_CUR.controller) PLUGIN_CUR.controller.abort(); } catch (e) {}
  try { if (PLUGIN_CUR.cleanup) PLUGIN_CUR.cleanup(); } catch (e) {}
  if (PLUGIN_CUR.cssEl) PLUGIN_CUR.cssEl.remove();
  PLUGIN_CUR = null;
  const host = $("plugin-host");
  if (host) host.innerHTML = "";
}

/* ---------------- 开发者选项：上下文占用 + fast weight ---------------- */
let STATS_TIMER = null;
const FW_WRITES = [];          // 客户端也留一份（服务没起来时也有曲线）

function meterSet(barId, valId, used, total) {
  const bar = $(barId), val = $(valId);
  if (!bar || !val) return;
  const r = total > 0 ? Math.min(1, used / total) : 0;
  bar.style.width = (r * 100).toFixed(2) + "%";
  bar.classList.toggle("warn", r >= 0.75 && r < 0.98);
  bar.classList.toggle("full", r >= 0.98);
  val.textContent = `${used} / ${total}`;
}

function renderCtxMeters(s) {
  const full = s.ctx_full || 0, comp = s.ctx_compress || 0;
  const n = s.kv_steps || 0;
  const fu = Math.min(n, full);
  const cu = Math.max(0, Math.min(n - full, comp));
  meterSet("ctx-full-bar", "ctx-full-val", fu, full);
  meterSet("ctx-comp-bar", "ctx-comp-val", cu, comp);
  const kv = $("ctx-kv-val");
  if (kv) {
    kv.textContent = n
      ? (s.kv_bytes ? `${fmtBytes(s.kv_bytes)} · ` : "") + `${n} tok`
      : "—";
    kv.title = n
      ? `共享 KV 序列：${n} 步 × d_model ${s.d_model} × ${
          s.kv_bytes && n ? Math.round(s.kv_bytes / n / Math.max(1, s.d_model)) : 0
        } 字节`
      : "尚未推理（KV 为空）";
  }
}

function cssVar(name, dflt) {
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue(name).trim();
  return v || dflt;
}

/* fast weight 可视化：
   左图 = 状态 |S| 的头均值（bins×bins 热力图，颜色越暖=写得越重）；
   中行 = 每个注意力头的 ‖S‖（哪颗头在记东西）；
   右图 = 每步写入幅度（‖S_t − S_{t−1}‖ 的折线，最近 N 步）。
   三样都是纯 canvas 自绘，没有系统控件。

   颜色ramp（fwColorRamp）：旧版用固定蓝色 + 只调透明度，所以「永远蓝、
   看不出加深」。改成 暗→蓝→青→黄→红 的感知梯度，轻重写在色相+
   亮度上同时编码，热点一眼可见。 */
function fwColorRamp(t) {
  // t ∈ [0,1] → [r,g,b]。低值贴近背景(暗)，高值偏暖(亮)。
  t = Math.max(0, Math.min(1, t));
  const stops = [
    [0.00, [36, 43, 53]],    // 暗灰（≈背景，空区域）
    [0.25, [40, 95, 165]],   // 蓝
    [0.50, [40, 175, 180]],  // 青
    [0.75, [232, 190, 60]],  // 黄
    [1.00, [220, 70, 50]],   // 红（最重）
  ];
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const a = stops[i - 1], b = stops[i];
      const k = (t - a[0]) / (b[0] - a[0]);
      return [
        Math.round(a[1][0] + (b[1][0] - a[1][0]) * k),
        Math.round(a[1][1] + (b[1][1] - a[1][1]) * k),
        Math.round(a[1][2] + (b[1][2] - a[1][2]) * k),
      ];
    }
  }
  return stops[stops.length - 1][1];
}

function renderFastWeight(s) {
  const cv = $("fw-canvas"), wr = $("fw-write"), headsEl = $("fw-heads");
  const normEl = $("fw-norm"), writeEl = $("fw-write-val");
  if (!cv || !wr) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const fw = s.fw;
  const track = cssVar("--track", "#242b35");
  const accent = cssVar("--accent", "#4c8dff");
  const fg3 = cssVar("--fg3", "#7a8699");

  // —— 状态热力图（颜色 ramp，不再固定蓝）——
  const w = cv.clientWidth || 200;
  const side = Math.round(w * dpr);
  if (cv.width !== side) { cv.width = side; cv.height = side; }
  const c1 = cv.getContext("2d");
  c1.fillStyle = track;
  c1.fillRect(0, 0, cv.width, cv.height);
  if (fw && fw.matrix && fw.matrix.length) {
    const bins = fw.matrix.length;
    let max = 1e-9;
    // 后端已把 NaN/Inf 消毒成 0，这里再守一道：一个脏值会让整张热力图
    // 的 max 变成 NaN，随后 Math.pow(NaN) 把全图静默画成空白。
    for (const row of fw.matrix) for (const v of row)
      if (Number.isFinite(v) && v > max) max = v;
    const cell = cv.width / bins;
    for (let y = 0; y < bins; y++) {
      for (let x = 0; x < bins; x++) {
        const v = Number(fw.matrix[y][x]);
        const t = Math.pow(Math.min(1, (Number.isFinite(v) ? v : 0) / max), 0.65);
        const [r, g, b] = fwColorRamp(t);
        c1.fillStyle = `rgb(${r},${g},${b})`;
        c1.fillRect(x * cell, y * cell, Math.ceil(cell), Math.ceil(cell));
      }
    }
    c1.globalAlpha = 1;
    if (normEl) normEl.textContent = `‖S‖ ${fw.norm}`;
  } else {
    c1.fillStyle = fg3;
    c1.font = `${11 * dpr}px ui-monospace, monospace`;
    c1.fillText("尚未推理", 10 * dpr, 20 * dpr);
    if (normEl) normEl.textContent = "—";
  }

  // —— 每头 ‖S‖（后端已算好 head_norms，这里画成小柱；空则清掉）——
  if (headsEl) {
    const hs = (fw && fw.head_norms) || [];
    if (hs.length) {
      const hmax = Math.max(...hs, 1e-9);
      let html = "";
      hs.forEach((hn, i) => {
        const pct = (hn / hmax * 100).toFixed(1);
        html += `<i class="fw-head" style="height:${pct}%" `
              + `title="head ${i}: ‖S‖=${hn}"></i>`;
      });
      headsEl.innerHTML = html;
      headsEl.style.display = "flex";
    } else {
      headsEl.innerHTML = "";
      headsEl.style.display = "none";
    }
  }

  // —— 写入幅度 ——
  const vals = ((s.fw_writes && s.fw_writes.length) ? s.fw_writes : FW_WRITES)
    .filter(v => Number.isFinite(Number(v))).map(Number);
  const h = Math.round(52 * dpr);
  if (wr.width !== side) { wr.width = side; wr.height = h; }
  const c2 = wr.getContext("2d");
  c2.clearRect(0, 0, wr.width, wr.height);
  c2.fillStyle = track;
  c2.fillRect(0, 0, wr.width, wr.height);
  if (vals.length) {
    const max = Math.max(...vals, 1e-9);
    const n = vals.length;
    c2.strokeStyle = accent;
    c2.lineWidth = Math.max(1, dpr);
    c2.beginPath();
    vals.forEach((v, i) => {
      const x = n === 1 ? wr.width / 2 : i / (n - 1) * (wr.width - 2) + 1;
      const y = wr.height - 2 - (v / max) * (wr.height - 4);
      i ? c2.lineTo(x, y) : c2.moveTo(x, y);
    });
    c2.stroke();
    if (writeEl) writeEl.textContent = `Δ ${vals[vals.length - 1]}`;
  } else if (writeEl) {
    writeEl.textContent = "—";
  }
}

async function refreshInferStats() {
  try {
    const s = await api("/api/infer/stats");
    renderCtxMeters(s);
    renderFastWeight(s);
    if (s.fw_writes && s.fw_writes.length) {
      FW_WRITES.length = 0;
      FW_WRITES.push(...s.fw_writes.slice(-96));
    }
  } catch (e) {}
}
function startInferStats() {
  if (STATS_TIMER) return;
  refreshInferStats();
  STATS_TIMER = setInterval(refreshInferStats, 1200);
}
function stopInferStats() {
  if (!STATS_TIMER) return;
  clearInterval(STATS_TIMER);
  STATS_TIMER = null;
}

/* 插件选择器（自绘下拉，值仍然是 <select>）。
   只有已可用的插件进下拉；开发中的 base 不注册，自然也不会出现。 */
function fillPluginSelect() {
  const sel = $("plugin-select");
  if (!sel) return;
  sel.innerHTML = PLUGINS.filter(p => !p.status).map(p =>
    `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join("");
  sel.value = selectedPluginId();
  const sync = () => {
    const p = PLUGINS.find(x => x.id === sel.value) || PLUGINS[0];
    if ($("plugin-hint")) $("plugin-hint").textContent = p.desc || "";
  };
  // 导入/移除会重绘 options，但 change 监听只绑一次
  if (!sel.dataset.pluginBound) {
    sel.dataset.pluginBound = "1";
    sel.addEventListener("change", () => {
      setSelectedPluginId(sel.value);
      sync();
      mountInferPlugin(true);
    });
  }
  sync();
}

/* ---------------- 导入插件（开发者选项 → 显示插件） ----------------
   服务端把 .js/.css 落到 service/web/plugins/ 并登记在 _imported.json，
   前端只从 /api/plugin/list 读清单；注册表放在服务端而不是 localStorage，
   换一个 WebView 配置也不会「插件还在、文件没了」。 */
function renderImportedPlugins() {
  const box = $("plugin-imported");
  if (!box) return;
  box.innerHTML = "";
  for (const p of PLUGINS.filter(x => x.imported)) {
    const row = document.createElement("div");
    row.className = "plugin-row";
    const nm = document.createElement("span");
    nm.className = "nm"; nm.textContent = p.name;
    nm.title = p.desc || "";
    const tag = document.createElement("span");
    tag.className = "tag"; tag.textContent = p.id;
    const del = document.createElement("button");
    del.className = "del"; del.type = "button";
    del.textContent = "✕"; del.title = `移除「${p.name}」`;
    del.onclick = async () => {
      const ok = await uiConfirm(
        `移除显示插件「${p.name}」？\n本地文件 ${p.entry} 会被删除；` +
        `若正在使用它，界面会自动切回「token 对话」。`);
      if (!ok) return;
      try {
        await api("/api/plugin/remove", {id: p.id});
        toast(`已移除「${p.name}」`);
        await refreshPluginRegistry();
      } catch (e) { toast("移除失败: " + e.message, true); }
    };
    row.append(nm, tag, del);
    box.appendChild(row);
  }
}

async function refreshPluginRegistry() {
  let list = [];
  let outdated = false;
  try { list = (await api("/api/plugin/list")).plugins || []; }
  catch (e) {
    // 旧 server 进程没有 /api/plugin/*：别静默装作「没有插件」，明确告诉
    // 用户重启后端，否则点导入只会看到一个莫名的 404。
    outdated = !!(e && e.status === 404);
  }
  const hintEl = $("plugin-import-hint");
  if (hintEl) hintEl.textContent = outdated
    ? "后端还是旧进程：请重启 GARGANTUA 后再导入插件"
    : "自带 ES 模块，协议见 plugins/README.md";
  PLUGINS.length = 0;
  PLUGINS.push(...BUILTIN_PLUGINS);
  for (const p of list) {
    if (!p || !p.id || typeof p.entry !== "string") continue;
    const id = String(p.id);
    if (PLUGINS.some(q => q.id === id)) continue;   // 内置同名 / 重复项
    PLUGINS.push({id, name: String(p.name || id),
                  entry: String(p.entry), css: p.css ? String(p.css) : "",
                  desc: String(p.desc || "导入的显示插件"), imported: true});
  }
  renderImportedPlugins();
  fillPluginSelect();
  // 选中的导入插件被移除 → 回落到内置并立刻重挂，别留着死实例
  if (PLUGIN_CUR && !PLUGINS.some(p => p.id === PLUGIN_CUR.id)) {
    setSelectedPluginId(BUILTIN_PLUGINS[0].id);
    $("plugin-select").value = BUILTIN_PLUGINS[0].id;
    mountInferPlugin(true);
  }
}

const readTextFile = f => new Promise((res, rej) => {
  const rd = new FileReader();
  rd.onload = () => res(String(rd.result || ""));
  rd.onerror = () => rej(new Error("读取失败: " + f.name));
  rd.readAsText(f);
});

function bindPluginImport() {
  const btn = $("plugin-import"), file = $("plugin-file");
  if (!btn || !file) return;
  btn.onclick = () => file.click();          // 隐藏的原生 input，外观全自绘
  file.onchange = async () => {
    const files = Array.from(file.files || []);
    file.value = "";
    if (!files.length) return;
    const js = files.find(f => /\.js$/i.test(f.name));
    if (!js) { toast("需要一个 .js 插件文件（可选附一份 .css）", true); return; }
    const stem = js.name.replace(/\.js$/i, "").toLowerCase()
      .replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 32);
    const id = await uiPrompt("导入显示插件",
      "插件 ID（小写字母 / 数字 / - / _，最长 32 字符）", stem || "my-plugin");
    if (!id) return;
    if (!/^[a-z0-9][a-z0-9_-]{0,31}$/.test(id)) {
      toast("插件 ID 不合法：需以小写字母或数字开头，只含 a-z 0-9 - _", true);
      return;
    }
    const css = files.find(f => /\.css$/i.test(f.name));
    try {
      const body = {id, code: await readTextFile(js),
                    css: css ? await readTextFile(css) : ""};
      let r;
      try {
        r = await api("/api/plugin/import", body);
      } catch (e) {
        if (e.status !== 409) throw e;
        const overwrite = await uiConfirm(
          `插件「${id}」已存在，覆盖它？（旧 .js/.css 会被替换）`);
        if (!overwrite) return;
        r = await api("/api/plugin/import", {...body, force: true});
      }
      toast(`已导入「${r.plugin.name}」，在上面切换即可使用`);
      await refreshPluginRegistry();
    } catch (e) {
      toast(e && e.status === 404
        ? "后端还是旧进程：请重启 GARGANTUA 后再导入插件"
        : "导入失败: " + e.message, true);
    }
  };
}

$("infer-clear").onclick = async () => {
  if (PLUGIN_CUR && PLUGIN_CUR.instance && PLUGIN_CUR.instance.reset) {
    try { PLUGIN_CUR.instance.reset(); } catch (e) {}
  }
  try {
    // 后端：杀掉 infer 服务进程 → 释放持久 fast weight + KV 上下文
    await api("/api/infer/reset", {});
    toast("已清空会话；infer 服务已终止，下次推理重新加载模型");
    refreshInferStats();
  } catch (e) {
    toast("后端重置失败: " + e.message, true);
  }
};
/* 右侧边栏：左侧边栏的镜像（状态挂在 <html data-dev> 上）。
   开合动画与左栏共用 --dur-side / --dur-tail / --move-ease：width 与
   clip-path 同相，只有用户点击时挂 .right-anim（首帧恢复直接落位）。
   固定的 #right-toggle 不属于这一层，全程原地不动。 */
function setDevOpen(on) {
  const el = document.documentElement;
  const next = on ? "open" : "closed";
  if (el.getAttribute("data-dev") === next) return;
  el.classList.add("right-anim");
  el.setAttribute("data-dev", next);
  const d = $("right-sidebar");
  if (d) d.setAttribute("aria-hidden", on ? "false" : "true");
  const t = $("right-toggle");
  if (t) t.title = on ? "收起开发者选项" : "开发者选项";
  clearTimeout(setDevOpen._t);
  setDevOpen._t = setTimeout(() => {
    el.classList.remove("right-anim");
  }, moveDurMs() + 60);
}
function devOpen() {
  return document.documentElement.getAttribute("data-dev") === "open";
}
/* 右上角固定按钮：开 / 合右侧边栏（位置永不移动）。
   侧栏底部不再有「收起」按钮——同一个按钮就是开合开关。 */
$("right-toggle").onclick = () => {
  setDevOpen(!devOpen());
  if (devOpen()) refreshInferStats();
};
$("infer-model-btn").onclick = async () => {
  const pop = $("model-pop");
  pop.hidden = !pop.hidden;
  if (!pop.hidden) await refreshModelPop();
};
$("model-pop-refresh").onclick = refreshModelPop;
async function refreshModelPop() {
  const list = $("model-pop-list");
  if (!list) return;
  try {
    const r = await api("/api/models");
    list.innerHTML = r.models.map(m =>
      `<div class="model-pop-row${m.name === r.active ? " on" : ""}" data-load="${esc(m.name)}">
         <span class="nm">${esc(m.name)}<span class="sub">${m.name === r.active ? "当前活动" : "点此加载"} · `
      + `${m.params ? (m.params / 1e6).toFixed(1) + "M" : "—"}</span></span>
         <button class="act sec xs del" data-del="${esc(m.name)}" title="删除模型">删除</button>
       </div>`).join("")
      || `<div class="hint">还没有模型，去「模型」页创建一个</div>`;
    list.querySelectorAll("[data-load]").forEach(row => {
      row.onclick = async e => {
        if (e.target.closest("[data-del]")) return;   // 点删除不触发加载
        await loadModel(row.dataset.load);
        $("model-pop").hidden = true;
        await refreshModelPop();
      };
    });
    list.querySelectorAll("[data-del]").forEach(b => {
      b.onclick = e => {
        e.stopPropagation();
        deleteModel(b.dataset.del).then(() => refreshModelPop());
      };
    });
  } catch (e) {
    list.innerHTML = `<div class="hint">获取模型失败: ${esc(e.message)}</div>`;
  }
}
/* 点击工具栏之外关闭模型弹窗 */
document.addEventListener("click", e => {
  const pop = $("model-pop"), btn = $("infer-model-btn");
  if (pop && !pop.hidden && btn && !pop.contains(e.target) && !btn.contains(e.target))
    pop.hidden = true;
});

/* ---------- 编解码器 ---------- */
/* 清空与输出保护
   ---------------------------------------------------------------------
   转换器的输出量随输入线性膨胀（4K 图 = 两千多个 patch × 10240 个 hex
   字符；视频按帧再乘；图元可以跑上万步），一个 <pre> 堆到几十 MB 时
   浏览器排版会直接卡死。两道闸：
     · setOut 只在 DOM 里放前 OUT_MAX_CHARS 个字符，超出部分明说被截断；
     · 每页一个「清空」把该页 token 文本、预览、文件选择一起复位，
       并释放 blob URL（否则每转一张图就漏一份原图在内存里）。 */
const OUT_MAX_CHARS = 200000;

function setOut(el, text) {
  if (!el) return;
  const s = text == null ? "" : String(text);
  if (!s) { el.textContent = "—"; return; }
  if (s.length > OUT_MAX_CHARS) {
    el.textContent = s.slice(0, OUT_MAX_CHARS) +
      `\n\n… 已截断：全文 ${s.length} 字符，此处只显示前 ${OUT_MAX_CHARS}。` +
      `\n点「清空」可释放这一页占用的内存。`;
    return;
  }
  el.textContent = s;
}
/* 复位一个自绘文件选择器（触发 change 让 wireFileInput 的 sync 生效） */
function resetFileInput(id) {
  const inp = $(id);
  if (!inp) return;
  inp.value = "";
  inp.dispatchEvent(new Event("change"));
}
/* 卸掉 <img>/<video>/<audio> 的 blob URL 并解引用 */
function releaseBlob(el) {
  if (!el) return;
  const src = el.getAttribute("src") || "";
  if (src.startsWith("blob:")) URL.revokeObjectURL(src);
  el.removeAttribute("src");
}

$("text-clear").onclick = () => { setOut($("text-out"), ""); };
$("audio-clear").onclick = () => {
  setOut($("audio-out"), "");
  const box = $("audio-preview");
  box.querySelectorAll("audio").forEach(releaseBlob);
  box.innerHTML = "";
  resetFileInput("audio-file");
};
$("image-clear").onclick = () => {
  setOut($("image-out"), "");
  releaseBlob($("image-preview"));
  resetFileInput("image-file");
};
$("video-clear").onclick = () => {
  setOut($("video-out"), "");
  $("video-stats").textContent = "";
  const vp = $("video-preview");
  try { vp.pause(); } catch (e) {}
  releaseBlob(vp);
  vp.classList.remove("on");
  resetFileInput("video-file");
};
$("json-clear").onclick = () => {
  $("json-editor").textContent = "";
  $("json-msg").textContent = "";
};

$("text-encode").onclick = async () => {
  try {
    // vocab：用哪个词表来转换（后端尚未认这个字段时忽略即可，不影响编码）
    const r = await api("/api/codec/text",
                        {text: $("text-in").value, vocab: $("text-vocab").value});
    setOut($("text-out"), r.token5);
  } catch (e) { setOut($("text-out"), "错误: " + e.message); }
};
$("audio-encode").onclick = async () => {
  const f = $("audio-file").files[0]; if (!f) return;
  const box = $("audio-preview");
  box.querySelectorAll("audio").forEach(releaseBlob);
  box.innerHTML = `<audio src="${fileToUrl(f)}" controls></audio>`;
  setOut($("audio-out"), "编码中…");
  try {
    const r = await api("/api/codec/audio", {audio_b64: await fileToB64(f),
                             filename: f.name},
                        {timeout: API_UPLOAD_TIMEOUT_MS});
    setOut($("audio-out"),
           `# 音元数 ${r.n_tokens}，时长 ${r.duration_s}s\n` + r.token5);
  } catch (e) { setOut($("audio-out"), "错误: " + e.message); }
};
$("image-encode").onclick = async () => {
  const f = $("image-file").files[0]; if (!f) return;
  releaseBlob($("image-preview"));
  $("image-preview").src = fileToUrl(f);
  setOut($("image-out"), "编码中…");
  try {
    const r = await api("/api/codec/image",
                        {image_b64: await fileToB64(f)},
                          {timeout: API_UPLOAD_TIMEOUT_MS});
    const st = r.steps[0];
    setOut($("image-out"),
           `# grid ${st.meta.grid[0]}×${st.meta.grid[1]}，patch 数 ${st.meta.n_patches}\n` +
           r.token5);
  } catch (e) { setOut($("image-out"), "错误: " + e.message); }
};
$("video-encode").onclick = async () => {
  const f = $("video-file").files[0]; if (!f) return;
  const vp = $("video-preview"); releaseBlob(vp);
  vp.src = fileToUrl(f); vp.classList.add("on");
  setOut($("video-out"), "编码中（ffmpeg 抽帧 + 提取音轨）…");
  try {
    const r = await api("/api/codec/video", {video_b64: await fileToB64(f),
                             filename: f.name},
                        {timeout: API_UPLOAD_TIMEOUT_MS});
    $("video-stats").textContent =
      `帧数: ${r.n_frames}　总 patches: ${r.total_patches}　音元: ${r.n_audio_tokens}`;
    setOut($("video-out"), r.token5);
  } catch (e) { setOut($("video-out"), "错误: " + e.message);
                $("video-stats").textContent = ""; }
};
/* ---------- Token 编辑器 ----------
   调色板：扩展系统 token（<0>..<5>，也可直接写名字）+ 结构字符。 */
const TOKEN_PALETTE = [
  {label:"<user>",    insert:"<user>"},
  {label:"<user/>",   insert:"<user/>"},
  {label:"<model>",   insert:"<model>"},
  {label:"<model/>",  insert:"<model/>"},
  {label:"<stop>",    insert:"<stop>"},
  {label:"<caps>",    insert:"<caps>"},
  {label:"时间步 {",  insert:"{"},
  {label:"时间步 }",  insert:"}"},
  {label:"步内 |",    insert:"|"},
  {label:"完整 {}",   insert:"{}", back: 1},
];
const paletteEl = $("json-palette");
TOKEN_PALETTE.forEach(t => {
  const b = document.createElement("button");
  b.textContent = t.label;
  b.onclick = () => insertAtCursor(t.insert, t.back || 0);
  paletteEl.appendChild(b);
});
function insertAtCursor(text, back) {
  const ed = $("json-editor"); ed.focus();
  document.execCommand("insertText", false, text);
  if (back > 0) {
    const sel = window.getSelection();
    for (let i = 0; i < back && sel.modify; i++)
      sel.modify("move", "backward", "character");
  }
}
$("json-format").onclick = () => {
  // 每个 {} 时间步独占一行；只清排版空白（\n\r\t），空格是合法 token 要保留
  const t = $("json-editor").innerText.replace(/[\n\r\t]+/g, "");
  $("json-editor").textContent = t.replace(/\}\{/g, "}\n{");
  $("json-msg").textContent = "";
};
$("json-minify").onclick = () => {
  $("json-editor").textContent =
    $("json-editor").innerText.replace(/[\n\r\t]+/g, "");
  $("json-msg").textContent = "";
};
$("json-copy").onclick = () => {
  navigator.clipboard.writeText($("json-editor").innerText)
    .then(() => $("json-msg").textContent = "已复制");
};
$("json-export").onclick = async () => {
  const msg = $("json-msg");
  msg.textContent = "导出中…";
  try {
    const body = {text: $("json-editor").innerText};
    // 2026-10-31 契约：Token 编辑器与四个转换器一致，按所选词表解析
    // <stop> 这类名字与校验 ID；未选（列表没加载出来）→ 后端用进程词表
    const v = $("json-vocab").value;
    if (v) body.vocab = v;
    const r = await api("/api/tokens/export", body);
    const bytes = b64ToBytes(r.tokens_b64);
    const blob = new Blob([bytes], {type: "application/octet-stream"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `stream_${r.n_tokens}tok.tokens`;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 0);
    msg.textContent = `已导出 ${r.n_tokens} tokens / ${r.n_steps} 时间步`;
  } catch (e) { msg.textContent = "错误: " + e.message; }
};

/* =========================================================================
   图元编辑器：手动摆图元 → 实时 SVG + 实时 token 流
   -------------------------------------------------------------------------
   · 画布档位与字段 offset 全部来自 /api/status.draw_layout（spec 是唯一
     真源），前端只做「值 → bin」这一步算术，公式与 draw_codec 逐行对应：
       x/y  = trunc(px)          （1 token = 1 像素）
       theta = trunc(deg × 3)     （≥90° 交换宽高）
       w/h  = trunc(px × 4)      shape = shape_type + 1
       r/g/b = 值本身             （10 bit，0..1023）
   · 画布声明 = `<x>` 数字… `<x/>` `<y>` 数字… `<y/>`（数字用可见区的文本
     数字 token，ID 也在 draw_layout.digits 里）。
   · 画布可缩放（滚轮）与平移（拖动）；单击落一个图元。
   · 右侧列表可拖拽排序 = 调图层顺序（越靠后越在上面）。
   · 底图可开关、可调透明度，用来对比「画出来的」和「原图」差多少；
     取色笔从合成结果（底图 + 图元）上取色。
   ========================================================================= */
const PDRAW = {
  w: 256, h: 256,
  prims: [],          // {cx, cy, w, h, shape_type, theta(rad), r, g, b}
  sel: -1,
  layout: null,       // /api/status.draw_layout
  canvasList: [],     // [[w,h], …]（/api/status.draw_canvas.sizes）
  brush: {r: 256, g: 512, b: 1023},
  brushShape: 0,      // 画布点击 / ＋ 按钮默认落下的形状（0 矩形 / 1 椭圆 / -1 背景）
  view: {zoom: 1, ox: 0, oy: 0},   // 视图缩放 / 平移
  picking: false,     // 取色笔待命
};

const _PD_SHAPE = { "-1": "背景", "0": "矩形", "1": "椭圆" };
const _r2 = v => Math.round(Number(v) * 100) / 100;
/* 值 → bin：与 draw_codec._bin 同为截断（不是四舍五入） */
function _pdBin(v, fd) {
  const n = Math.trunc(Number(v) * (fd.per || 1));
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(fd.bins - 1, n));
}
/* 10 bit ↔ 8 bit（显示用） */
function _pdTo8(v) { return Math.round(Number(v) * 255 / 1023); }
function _pdTo10(v) { return Math.round(Number(v) * 1023 / 255); }
/* 10 bit 色的十六进制：每通道 3 位（0..3FF），如 #3FF000200 */
function _pdHex(r, g, b) {
  const h3 = v => Math.max(0, Math.min(1023, Math.round(v)))
    .toString(16).toUpperCase().padStart(3, "0");
  return "#" + h3(r) + h3(g) + h3(b);
}

/* 一条绘图序列 → token-5 文本：
   画布声明独占一步（`<x> 1 9 2 0 <x/> <y> 1 0 8 0 <y/>`），
   其后每个图元一步（9 个 token）。 */
function pdrawTokenText() {
  const L = PDRAW.layout;
  if (!L) return "（正在读取 spec 常量…）";
  const F = {};
  L.fields.forEach(f => { F[f.name] = f; });
  const B = L.begin;
  const steps = [];
  // —— 画布声明 ——
  const mk = L.canvas.markers || ["<x>", "<x/>", "<y>", "<y/>"];
  const dig = L.digits || [];
  const digitsOf = n => String(n).split("").map(ch => dig[Number(ch)]);
  const cw = B + L.canvas.offset;
  steps.push("{" + [cw + 0].concat(digitsOf(PDRAW.w), [cw + 1],
                                 [cw + 2], digitsOf(PDRAW.h), [cw + 3])
    .map(t => "text: " + _IDHEX(t)).join(" | ") + "}");
  // —— 每个图元 9 个 token ——
  for (const p of PDRAW.prims) {
    let deg = (p.theta * 180 / Math.PI) % 180;
    if (deg < 0) deg += 180;
    let pw = p.w, ph = p.h;
    if (deg >= 90) { deg -= 90; const t = pw; pw = ph; ph = t; }
    const shapeBin = Math.max(0, Math.min(F.shape.bins - 1, p.shape_type + 1));
    const ids = [
      B + F.x.offset + _pdBin(p.cx, F.x),
      B + F.y.offset + _pdBin(p.cy, F.y),
      B + F.theta.offset + _pdBin(deg, F.theta),
      B + F.w.offset + _pdBin(pw, F.w),
      B + F.h.offset + _pdBin(ph, F.h),
      B + F.shape.offset + shapeBin,
      B + F.r.offset + _pdBin(p.r, F.r),
      B + F.g.offset + _pdBin(p.g, F.g),
      B + F.b.offset + _pdBin(p.b, F.b),
    ];
    steps.push("{" + ids.map(t => "text: " + _IDHEX(t)).join(" | ") + "}");
  }
  return steps.join("");
}

/* 图元 → SVG（与 draw_codec.primitives_to_svg 同语义） */
function pdrawRenderSVG() {
  const svg = $("pdraw-svg");
  const { w, h } = PDRAW;
  svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  svg.setAttribute("width", w);
  svg.setAttribute("height", h);
  const out = [];
  PDRAW.prims.forEach((p, i) => {
    const fill = `rgb(${_pdTo8(p.r)},${_pdTo8(p.g)},${_pdTo8(p.b)})`;
    const deg = p.theta * 180 / Math.PI;
    const rot = Math.abs(deg) > 1e-9
      ? ` transform="rotate(${deg} ${p.cx} ${p.cy})"` : "";
    const hl = i === PDRAW.sel
      ? ' style="stroke:var(--accent);stroke-width:1.5"' : "";
    if (p.shape_type === -1) {
      out.push(`<rect x="0" y="0" width="${w}" height="${h}" fill="${fill}"${hl}/>`);
    } else if (p.shape_type === 1) {
      out.push(`<ellipse cx="${p.cx}" cy="${p.cy}" rx="${p.w / 2}" `
        + `ry="${p.h / 2}" fill="${fill}"${rot}${hl}/>`);
    } else {
      out.push(`<rect x="${p.cx - p.w / 2}" y="${p.cy - p.h / 2}" `
        + `width="${p.w}" height="${p.h}" fill="${fill}"${rot}${hl}/>`);
    }
  });
  svg.innerHTML = out.join("");
  $("pdraw-count").textContent = PDRAW.prims.length
    ? `${PDRAW.prims.length} 个图元 · ${PDRAW.w}×${PDRAW.h}`
    : "点空白处加一个矩形";
  pdrawFit();
  pdrawApplyView();
}

/* 完整重绘（画布/表单变更用）：SVG + 把当前序列写回 token 文本框 */
function pdrawRender() {
  pdrawRenderSVG();
  setOut($("pdraw-out"), pdrawTokenText());
}

/* 文本框 → 图元序列：与 pdrawTokenText 互逆。
   用布局（L）里的 offset / bins / per 把每个 token ID 还原成字段值。
   画布声明块结构固定为 [cw, w位, cw+1, cw+2, h位, cw+3]（cw = begin + canvas.offset），
   据此抽出宽高；其后每个 9-token 块还原成一个图元。解析失败（半截输入、
   非法字符）返回 false，调用方静默忽略——绝不拿脏数据清掉画布。 */
function pdrawParse(text) {
  if (!PDRAW.layout) return false;
  const L = PDRAW.layout;
  const B = L.begin;
  const cw = B + L.canvas.offset;
  const dig = L.digits || [];
  const blocks = (text.match(/\{[^}]*\}/g) || []).map(s => s.slice(1, -1));
  if (!blocks.length) return false;
  const parseBlock = raw => raw.split("|").map(s => s.trim()).filter(Boolean)
    .map(tok => {
      const m = /text:\s*([0-9A-Fa-f]+)/.exec(tok) || /([0-9A-Fa-f]+)/.exec(tok);
      return m ? parseInt(m[1], 16) : NaN;
    });
  const cb = parseBlock(blocks[0]);
  const iCW = cb.indexOf(cw), iCW1 = cb.indexOf(cw + 1),
        iCW2 = cb.indexOf(cw + 2), iCW3 = cb.indexOf(cw + 3);
  if (iCW >= 0 && iCW1 > iCW && iCW2 > iCW1 && iCW3 > iCW2) {
    const digVal = t => { const i = dig.indexOf(t); return i >= 0 ? i : NaN; };
    const wd = cb.slice(iCW + 1, iCW1).map(digVal);
    const hd = cb.slice(iCW2 + 1, iCW3).map(digVal);
    if (wd.length && hd.length && wd.every(Number.isFinite) && hd.every(Number.isFinite)) {
      const w = parseInt(wd.map(String).join(""), 10);
      const h = parseInt(hd.map(String).join(""), 10);
      if (Number.isFinite(w) && Number.isFinite(h) && w >= 1 && h >= 1) {
        PDRAW.w = w; PDRAW.h = h;
      }
    }
  }
  const F = {};
  L.fields.forEach(f => { F[f.name] = f; });
  const unbin = (id, name) => {
    const f = F[name]; const bin = id - (B + f.offset);
    return bin / (f.per || 1);
  };
  const prims = [];
  for (let bi = 1; bi < blocks.length; bi++) {
    const ids = parseBlock(blocks[bi]);
    if (ids.length < 9) continue;
    if (ids.some(n => !Number.isFinite(n))) continue;
    let deg = unbin(ids[2], "theta");
    let pw = unbin(ids[3], "w"), ph = unbin(ids[4], "h");
    if (deg >= 90) { deg -= 90; const t = pw; pw = ph; ph = t; }   // 与编码器一致
    prims.push({
      cx: unbin(ids[0], "x"), cy: unbin(ids[1], "y"),
      w: pw, h: ph,
      shape_type: (ids[5] - (B + F.shape.offset)) - 1,
      theta: Math.max(0, deg * Math.PI / 180),
      r: Math.round(unbin(ids[6], "r")),
      g: Math.round(unbin(ids[7], "g")),
      b: Math.round(unbin(ids[8], "b")),
    });
  }
  PDRAW.prims = prims;
  PDRAW.sel = prims.length ? prims.length - 1 : -1;
  return true;
}

/* 文本框变更 → 重绘（不回写文本框，保留用户光标与输入）。
   空文本 = 清空全部图元；解析失败则保持原样。 */
function pdrawFromText() {
  const ta = $("pdraw-out");
  if (!ta) return;
  if (!ta.value.replace(/\s+/g, "")) {
    PDRAW.prims = []; PDRAW.sel = -1; pdrawEnsureCanvasOption(PDRAW.w, PDRAW.h); pdrawSync(false);
    return;
  }
  if (!pdrawParse(ta.value)) return;
  pdrawEnsureCanvasOption(PDRAW.w, PDRAW.h);
  pdrawSync(false);
}

/* 当前画布尺寸若不在下拉框里，动态补一个选项并选中（解析出非预设尺寸时） */
function pdrawEnsureCanvasOption(w, h) {
  const sel = $("pdraw-canvas");
  if (!sel) return;
  const val = `${w}x${h}`;
  const exists = Array.from(sel.options).some(o => o.value === val);
  if (!exists) {
    const opt = document.createElement("option");
    opt.value = val; opt.textContent = `${w}×${h}`;
    sel.appendChild(opt);
    if (!PDRAW.canvasList.some(s => s[0] === w && s[1] === h))
      PDRAW.canvasList.push([w, h]);
  }
  sel.value = val;
}

/* 视图：缩放 + 平移（作用在 frame 上，SVG 内部坐标不变） */
function pdrawApplyView() {
  const f = $("pdraw-frame");
  if (!f) return;
  const v = PDRAW.view;
  f.style.transform = `translate(${v.ox}px, ${v.oy}px) scale(${v.zoom})`;
  const z = $("pdraw-zoom");
  if (z) z.textContent = Math.round(v.zoom * 100) + "%";
}

/* 画布铺满可用空间：frame 的**布局**尺寸 = 画布等比缩放到 stage 内。
   缩放/平移仍由 pdrawApplyView 的 transform 叠加在上面，两者互不干扰。
   以前 frame 由 SVG 的 width/height 属性撑成 256px，画布永远只有一小块。 */
function pdrawFit() {
  const stage = $("pdraw-stage"), frame = $("pdraw-frame");
  if (!stage || !frame) return;
  const aw = stage.clientWidth, ah = stage.clientHeight;
  if (!aw || !ah) return;
  const s = Math.min(aw / PDRAW.w, ah / PDRAW.h);
  frame.style.width = Math.max(1, Math.floor(PDRAW.w * s)) + "px";
  frame.style.height = Math.max(1, Math.floor(PDRAW.h * s)) + "px";
}

/* 列表 + 选中项表单 + SVG 一起刷新。
   writeText=false 时不回写 token 文本框（用户在文本框里输入、由 pdrawFromText
   触发重绘时用，避免光标跳动与输入被覆盖）。 */
function pdrawSync(writeText) {
  if (writeText === undefined) writeText = true;
  const list = $("pdraw-list");
  list.innerHTML = PDRAW.prims.map((p, i) => `
    <div class="pdraw-item${i === PDRAW.sel ? " on" : ""}" data-i="${i}"
         draggable="true" title="拖动可调整图层顺序（越靠后越靠上）">
      <span class="pdraw-grip" aria-hidden="true">⋮⋮</span>
      <span class="sw" style="background:rgb(${_pdTo8(p.r)},${_pdTo8(p.g)},${_pdTo8(p.b)})"></span>
      <span class="nm">#${i} ${_PD_SHAPE[p.shape_type]} `
    + `${_r2(p.cx)},${_r2(p.cy)} · ${_r2(p.w)}×${_r2(p.h)}</span>`
    + `<button class="pdraw-del" data-del="${i}" title="删除该图元" aria-label="删除">✕</button>`
    + `</div>`).join("")
    || `<div class="hint">还没有图元。点画布空白处，或右上角的 ＋ 按钮。</div>`;
  list.querySelectorAll("[data-i]").forEach(el => {
    el.onclick = e => {
      if (e.target.closest(".pdraw-del")) return;   // 点删除按钮不触发选中
      PDRAW.sel = Number(el.dataset.i); pdrawSync();
    };
    el.ondragstart = e => {
      e.dataTransfer.setData("text/plain", el.dataset.i);
      e.dataTransfer.effectAllowed = "move";
      el.classList.add("dragging");
    };
    el.ondragend = () => el.classList.remove("dragging");
    el.ondragover = e => { e.preventDefault(); el.classList.add("over"); };
    el.ondragleave = () => el.classList.remove("over");
    el.ondrop = e => {
      e.preventDefault();
      el.classList.remove("over");
      const from = Number(e.dataTransfer.getData("text/plain"));
      const to = Number(el.dataset.i);
      if (Number.isNaN(from) || from === to) return;
      const [it] = PDRAW.prims.splice(from, 1);
      PDRAW.prims.splice(to, 0, it);
      PDRAW.sel = to;
      pdrawSync();
    };
  });
  list.querySelectorAll("[data-del]").forEach(b => {
    b.onclick = e => {
      e.stopPropagation(); e.preventDefault();
      const i = Number(b.dataset.del);
      PDRAW.prims.splice(i, 1);
      if (PDRAW.sel === i) PDRAW.sel = Math.max(-1, i - 1);
      else if (PDRAW.sel > i) PDRAW.sel -= 1;
      pdrawSync();
    };
  });
  pdrawForm();
  pdrawRenderSVG();
  if (writeText) setOut($("pdraw-out"), pdrawTokenText());
}

function pdrawForm() {
  const box = $("pdraw-form");
  const p = PDRAW.prims[PDRAW.sel];
  if (!p) { box.hidden = true; box.innerHTML = ""; return; }
  box.hidden = false;
  const { w, h } = PDRAW;
  const cmax = (PDRAW.layout && PDRAW.layout.color_max) || 1023;
  const deg = _r2(p.theta * 180 / Math.PI);
  box.innerHTML = `
    <div class="pdraw-grid">
      <label class="form-field"><span>x（0–${w - 1}）</span>
        <input type="number" data-f="cx" value="${_r2(p.cx)}" min="0" max="${w - 1}" step="1"></label>
      <label class="form-field"><span>y（0–${h - 1}）</span>
        <input type="number" data-f="cy" value="${_r2(p.cy)}" min="0" max="${h - 1}" step="1"></label>
      <label class="form-field"><span>宽 w（≤${w}）</span>
        <input type="number" data-f="w" value="${_r2(p.w)}" min="0" max="${w}" step="0.5"></label>
      <label class="form-field"><span>高 h（≤${h}）</span>
        <input type="number" data-f="h" value="${_r2(p.h)}" min="0" max="${h}" step="0.5"></label>
      <label class="form-field"><span>角度 theta（度）</span>
        <input type="number" data-f="theta" value="${deg}" min="0" max="179" step="1"></label>
      <label class="form-field"><span>形状</span>
        <select data-f="shape">
          <option value="-1"${p.shape_type === -1 ? " selected" : ""}>背景</option>
          <option value="0"${p.shape_type === 0 ? " selected" : ""}>矩形</option>
          <option value="1"${p.shape_type === 1 ? " selected" : ""}>椭圆</option>
        </select></label>
    </div>
    <div class="pdraw-color">
      <button class="pdraw-sw" data-f="sw" title="当前颜色（10 bit，每通道 0–1023）"
        style="background:rgb(${_pdTo8(p.r)},${_pdTo8(p.g)},${_pdTo8(p.b)})"></button>
      <div class="pdraw-cs">
        <label>R<input type="range" class="slider" data-f="r" min="0" max="${cmax}" value="${p.r}"></label>
        <label>G<input type="range" class="slider" data-f="g" min="0" max="${cmax}" value="${p.g}"></label>
        <label>B<input type="range" class="slider" data-f="b" min="0" max="${cmax}" value="${p.b}"></label>
      </div>
    </div>
    <div class="pdraw-color-row">
      <button class="act sec xs" data-f="pick" title="取色笔：到画布上点一下取那里的颜色">取色笔</button>
      <code class="pdraw-hex" data-f="hex">${_pdHex(p.r, p.g, p.b)}</code>
      <span class="hint">10 bit（每通道 0–3FF）</span>
    </div>`;
  const pick = f => box.querySelector(`[data-f="${f}"]`);
  const bind = (f, key, cast) => {
    const el = pick(f);
    if (!el) return;
    el.oninput = () => {
      const q = PDRAW.prims[PDRAW.sel];
      if (!q) return;
      q[key] = cast(el.value);
      PDRAW.brush = { r: q.r, g: q.g, b: q.b };
      // 只重画不重建表单：否则正在拖的滑杆会被 innerHTML 换掉，手感断掉
      pdrawRender();
      if (f === "r" || f === "g" || f === "b") {
        const sw = pick("sw");
        if (sw) sw.style.background = `rgb(${_pdTo8(q.r)},${_pdTo8(q.g)},${_pdTo8(q.b)})`;
        const hx = pick("hex");
        if (hx) hx.textContent = _pdHex(q.r, q.g, q.b);
      }
      const item = $("pdraw-list").querySelector(`[data-i="${PDRAW.sel}"] .nm`);
      if (item) item.textContent = `#${PDRAW.sel} ${_PD_SHAPE[q.shape_type]} `
        + `${_r2(q.cx)},${_r2(q.cy)} · ${_r2(q.w)}×${_r2(q.h)}`;
    };
  };
  bind("cx", "cx", Number);
  bind("cy", "cy", Number);
  bind("w", "w", Number);
  bind("h", "h", Number);
  bind("theta", "theta", v => Number(v) * Math.PI / 180);
  bind("r", "r", v => Math.round(Number(v)));
  bind("g", "g", v => Math.round(Number(v)));
  bind("b", "b", v => Math.round(Number(v)));
  pick("shape").onchange = e => {
    const q = PDRAW.prims[PDRAW.sel];
    if (!q) return;
    q.shape_type = Number(e.target.value);
    pdrawSync();
  };
  pick("pick").onclick = () => {
    PDRAW.picking = !PDRAW.picking;
    pick("pick").classList.toggle("on", PDRAW.picking);
    $("pdraw-msg").textContent = PDRAW.picking
      ? "取色笔已就绪：在画布上点一下取那里的颜色" : "";
  };
}

function pdrawAdd(shapeType, cx, cy) {
  const { w, h } = PDRAW;
  const size = Math.max(8, Math.round(Math.min(w, h) * 0.22));
  const b = PDRAW.brush;
  PDRAW.prims.push({
    cx: cx == null ? w / 2 : Math.max(0, Math.min(w - 1, cx)),
    cy: cy == null ? h / 2 : Math.max(0, Math.min(h - 1, cy)),
    w: shapeType === -1 ? 0 : size,
    h: shapeType === -1 ? 0 : size,
    shape_type: shapeType,
    theta: 0,
    r: b.r, g: b.g, b: b.b,
  });
  PDRAW.sel = PDRAW.prims.length - 1;
  pdrawSync();
}

/* 画布档位：两处下拉（图元绘画页 + 本页）共用同一份 spec 列表 */
function pdrawSetCanvas(value) {
  const m = /^(\d+)x(\d+)$/.exec(value || "");
  if (!m) return;
  PDRAW.w = Number(m[1]);
  PDRAW.h = Number(m[2]);
  PDRAW.prims.forEach(p => {
    p.cx = Math.min(p.cx, PDRAW.w - 1);
    p.cy = Math.min(p.cy, PDRAW.h - 1);
  });
  pdrawSync();
}

/* 取色：把「底图 + 图元」合成到离屏 canvas 上再读像素，
   与屏幕上的呈现一致（SVG 层是透明的，底图在下层）。 */
function pdrawSample(x, y) {
  const cv = document.createElement("canvas");
  cv.width = PDRAW.w; cv.height = PDRAW.h;
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  const img = $("pdraw-bg-img");
  if (img.classList.contains("on") && img.getAttribute("src")) {
    try { ctx.drawImage(img, 0, 0, PDRAW.w, PDRAW.h); } catch (e) {}
  } else {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, PDRAW.w, PDRAW.h);
  }
  PDRAW.prims.forEach(p => {
    ctx.save();
    ctx.translate(p.cx, p.cy);
    ctx.rotate(p.theta);
    ctx.fillStyle = `rgb(${_pdTo8(p.r)},${_pdTo8(p.g)},${_pdTo8(p.b)})`;
    if (p.shape_type === 1) {
      ctx.beginPath();
      ctx.ellipse(0, 0, p.w / 2, p.h / 2, 0, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
    }
    ctx.restore();
  });
  const px = Math.max(0, Math.min(PDRAW.w - 1, Math.floor(x)));
  const py = Math.max(0, Math.min(PDRAW.h - 1, Math.floor(y)));
  const d = ctx.getImageData(px, py, 1, 1).data;
  return { r: _pdTo10(d[0]), g: _pdTo10(d[1]), b: _pdTo10(d[2]) };
}

/* 画布交互：单击落图元 / 拖动平移 / 滚轮缩放（以光标为锚点） */
function pdrawCanvasPoint(e) {
  const svg = $("pdraw-svg");
  const r = svg.getBoundingClientRect();
  if (!r.width || !r.height) return null;
  return { x: (e.clientX - r.left) / r.width * PDRAW.w,
           y: (e.clientY - r.top) / r.height * PDRAW.h };
}
(function wireCanvas() {
  const stage = $("pdraw-stage");
  if (!stage) return;
  let drag = null;
  stage.addEventListener("pointerdown", e => {
    if (e.button !== 0) return;
    drag = { x: e.clientX, y: e.clientY, ox: PDRAW.view.ox, oy: PDRAW.view.oy,
             moved: false };
    stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener("pointermove", e => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
    if (!drag.moved) return;
    PDRAW.view.ox = drag.ox + dx;
    PDRAW.view.oy = drag.oy + dy;
    pdrawApplyView();
  });
  stage.addEventListener("pointerup", e => {
    const d = drag;
    drag = null;
    if (!d || d.moved) return;              // 拖过 = 平移，不加图元
    const pt = pdrawCanvasPoint(e);
    if (!pt) return;
    if (PDRAW.picking) {
      const c = pdrawSample(pt.x, pt.y);
      PDRAW.picking = false;
      PDRAW.brush = c;
      const q = PDRAW.prims[PDRAW.sel];
      if (q) { q.r = c.r; q.g = c.g; q.b = c.b; }
      $("pdraw-msg").textContent =
        `已取色 ${_pdHex(c.r, c.g, c.b)}`;
      pdrawSync();
      return;
    }
    pdrawAdd(PDRAW.brushShape, pt.x, pt.y);
  });
  stage.addEventListener("wheel", e => {
    e.preventDefault();
    const v = PDRAW.view;
    const before = pdrawCanvasPoint(e);
    const k = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    v.zoom = Math.max(0.15, Math.min(8, v.zoom * k));
    pdrawApplyView();
    // 以光标为锚点缩放：先量出缩放后光标下的画布坐标偏了多少
    // （after - before，单位 = 画布 px），再换算回屏幕 px 补进平移——
    // 1 画布 px = zoom 屏幕 px，所以必须乘当前 zoom。
    const after = pdrawCanvasPoint(e);
    if (before && after) {
      v.ox += (after.x - before.x) * v.zoom;
      v.oy += (after.y - before.y) * v.zoom;
      pdrawApplyView();
    }
  }, { passive: false });
})();

$("pdraw-canvas").onchange = e => pdrawSetCanvas(e.target.value);
/* 形状选择器：三个图形选项挑「当前画笔形状」，＋ 号按当前形状落一个。
   画布单击也按当前形状落子（不再永远矩形）。 */
(function wirePdrawTools() {
  const tools = $("pdraw-tools");
  if (!tools) return;
  tools.querySelectorAll(".shape-opt").forEach(b => {
    b.onclick = () => {
      PDRAW.brushShape = Number(b.dataset.shape);
      tools.querySelectorAll(".shape-opt").forEach(x => x.classList.remove("on"));
      b.classList.add("on");
    };
  });
  const addBtn = $("pdraw-add");
  if (addBtn) addBtn.onclick = () => pdrawAdd(PDRAW.brushShape);
})();
$("pdraw-clear").onclick = () => { PDRAW.prims = []; PDRAW.sel = -1; pdrawSync(); };
$("pdraw-undo").onclick = () => {
  if (!PDRAW.prims.length) return;
  PDRAW.prims.pop();
  PDRAW.sel = Math.min(PDRAW.sel, PDRAW.prims.length - 1);
  pdrawSync();
};
$("pdraw-view-reset").onclick = () => {
  PDRAW.view = { zoom: 1, ox: 0, oy: 0 };
  pdrawApplyView();
};
$("pdraw-bg").onchange = () => {
  const f = $("pdraw-bg").files[0];
  const img = $("pdraw-bg-img");
  if (!f) { releaseBlob(img); img.classList.remove("on"); return; }
  releaseBlob(img);
  img.src = fileToUrl(f);
  img.classList.add("on");
  vocabSegSync("pdraw-bgseg", "on");
  $("pdraw-msg").textContent = "底图已载入，调透明度对比";
};
vocabSegWire("pdraw-bgseg", v => {
  const img = $("pdraw-bg-img");
  img.classList.toggle("on", v === "on" && !!img.getAttribute("src"));
  $("pdraw-msg").textContent = v === "on" ? "" : "已隐藏底图";
});
vocabSegSync("pdraw-bgseg", "off");
$("pdraw-bgalpha").oninput = e => {
  $("pdraw-bg-img").style.opacity = (Number(e.target.value) / 100).toFixed(2);
};
$("pdraw-bg-img").style.opacity = "0.6";
/* 文本框直接编辑 token 序列 → 自动重绘 SVG / 列表（不回写文本框本身） */
const _pdrawOut = $("pdraw-out");
if (_pdrawOut) _pdrawOut.addEventListener("input", pdrawFromText);

/* 进页面时按当前档位铺一次画布（尺寸来自 spec 列表，取默认那档） */
function pdrawInit() {
  if (!PDRAW.prims.length) {
    const sel = $("pdraw-canvas");
    if (sel && sel.value) pdrawSetCanvas(sel.value);
    else pdrawSync();
  } else {
    pdrawSync();
  }
  // 窗口/侧栏变化时重新贴合（ResizeObserver 只建一次）
  const stage = $("pdraw-stage");
  if (stage && !PDRAW._ro && typeof ResizeObserver === "function") {
    PDRAW._ro = new ResizeObserver(() => pdrawFit());
    PDRAW._ro.observe(stage);
  }
  requestAnimationFrame(pdrawFit);
}

/* =========================================================================
   词表页：标签页 + 分区 + 增删 token + 词表级操作
   -------------------------------------------------------------------------
   · 一个标签页 = 一个「视图」。关掉标签只是关视图，不删词表文件。
   · 分区表完全按接口返回渲染：不同词表的区块划分可以完全不同，前端不写死一套。
   · main 词表不允许直接编辑：发起编辑操作会被拦下弹窗，出口是 Fork。
   · 文本式保存失败时编辑区内容必须原样留着——先把文本塞进 VOCAB.src 再发请求。
   ========================================================================= */
const VOCAB = {
  list: [],       // /api/vocab/list 全量词表
  max: 0,
  tabs: [],       // 已打开的标签（词表名）
  active: "",     // 当前标签
  info: null,     // 当前标签的 /api/vocab/regions 结果
  src: {},        // name -> .vfc 源文本缓存（保存失败不丢）
  mode: "form",   // form | text
  write: "this",  // this | fork
};
/* 转换器四页 + 建模页「词表预设」+ Token 编辑器导出共用同一份词表列表 */
const VOCAB_SELECTS = ["text-vocab",
                       "model-vocab", "json-vocab"];
const MAIN_WHY =
  "main 词表是全站基准：模型维度、不可见区契约、系统 token 都锚在它身上，"
  + "直接改动会牵连所有已建模型。请 Fork 出一份新词表，再在新词表上编辑。";

function vocabMeta(name) {
  return VOCAB.list.find(v => v.name === name) || null;
}
function vocabIsMain(name) {
  const m = vocabMeta(name);
  return !!(m && m.is_main);
}
/* 新词表名：base+suffix，撞名就往后加序号 */
function vocabSuggestName(base, suffix) {
  const s = (base || "vocab") + (suffix || "_copy");
  if (!VOCAB.list.some(v => v.name === s)) return s;
  for (let i = 2; i < 999; i++) {
    const c = s + i;
    if (!VOCAB.list.some(v => v.name === c)) return c;
  }
  return s + "_" + Date.now();
}
function vocabMenuClose() { $("vtab-menu").hidden = true; }
function vocabMenuToggle() {
  const m = $("vtab-menu");
  if (m.hidden) { renderVocabMenu(); m.hidden = false; }
  else vocabMenuClose();
}
$("vtab-add").onclick = e => { e.stopPropagation(); vocabMenuToggle(); };
// 点空白处收起下拉（下拉内部的点击已在各自 handler 里收起）
document.addEventListener("click", e => {
  if ($("vtab-menu").hidden) return;
  if (!e.target.closest(".vtab-add-wrap")) vocabMenuClose();
});

/* ---------- 词表列表：全量 + 全站下拉 ---------- */
async function refreshVocabList() {
  const r = await api("/api/vocab/list");
  VOCAB.list = r.vocabs || [];
  VOCAB.max = r.max_vocab_size || 0;
  const opts = VOCAB.list.map(v =>
    `<option value="${esc(v.name)}">${esc(v.name)}</option>`).join("");
  for (const id of VOCAB_SELECTS) {
    const el = $(id);
    if (!el) continue;
    const prev = el.value;
    el.innerHTML = opts;
    if (prev && VOCAB.list.some(v => v.name === prev)) el.value = prev;
    else if (!el.value) {
      const main = VOCAB.list.find(v => v.is_main);
      if (main) el.value = main.name;
    }
  }
  // 词表列表变了 → 建模页的参数量预测跟着重算（V 变了）
  refreshModelEstimate();
  // 已被删除的词表：从打开的标签里摘掉
  VOCAB.tabs = VOCAB.tabs.filter(n => VOCAB.list.some(v => v.name === n));
  if (VOCAB.active && !VOCAB.tabs.includes(VOCAB.active)) VOCAB.active = "";
  if (!VOCAB.tabs.length && VOCAB.list.length) {
    const main = VOCAB.list.find(v => v.is_main);
    VOCAB.active = (main || VOCAB.list[0]).name;
    VOCAB.tabs = [VOCAB.active];
  } else if (VOCAB.active && !VOCAB.tabs.includes(VOCAB.active)) {
    // 自动打开的 base 等：确保 active 一定落在标签页条上
    VOCAB.tabs.push(VOCAB.active);
  }
  renderVocabTabs();
  return VOCAB.list;
}

/* ---------- 标签页条 ---------- */
function renderVocabTabs() {
  const strip = $("vtab-strip");
  strip.innerHTML = VOCAB.tabs.map(n => {
    const m = vocabMeta(n);
    const badge = m && m.is_main ? ' <span class="badge">main</span>' : "";
    return `<div class="vtab${n === VOCAB.active ? " on" : ""}" data-tab="${esc(n)}"
                 title="${esc(n)}"><span class="nm">${esc(n)}</span>${badge}` +
      `<button class="x" data-close="${esc(n)}" aria-label="关闭标签"
               title="关闭这个视图（不删词表文件）">×</button></div>`;
  }).join("");
  strip.querySelectorAll("[data-tab]").forEach(el => {
    el.onclick = e => {
      if (e.target.closest("[data-close]")) return;
      vocabActivate(el.dataset.tab);
    };
  });
  strip.querySelectorAll("[data-close]").forEach(b => {
    b.onclick = e => { e.stopPropagation(); vocabCloseTab(b.dataset.close); };
  });
}
function renderVocabMenu() {
  const menu = $("vtab-menu");
  // ＋ 入口永远提供「新建空词表」：即便列表为空也能从零建一张
  const head = `<button class="it create" data-new-empty="1">` +
    `<span>＋ 新建空词表</span><span class="sub">仅契约头·从零开始</span></button>`;
  if (!VOCAB.list.length) {
    menu.innerHTML = head + `<div class="empty">还没有任何词表</div>`;
  } else {
    menu.innerHTML = head + VOCAB.list.map(v => {
      const opened = VOCAB.tabs.includes(v.name);
      return `<button class="it${opened ? " opened" : ""}" data-open="${esc(v.name)}">` +
        `<span>${esc(v.name)}${v.is_main ? ' <span class="badge">main</span>' : ""}</span>` +
        `<span class="sub">${opened ? "已打开" : (v.vocab_size || "")}</span></button>`;
    }).join("");
  }
  const nb = menu.querySelector("[data-new-empty]");
  if (nb) nb.onclick = () => { vocabMenuClose(); vocabCreateEmpty(); };
  menu.querySelectorAll("[data-open]").forEach(b => {
    b.onclick = () => { vocabMenuClose(); vocabActivate(b.dataset.open); };
  });
}

/* ＋ 菜单的「新建空词表」：命名 → 后端建空词表 → 打开成标签页 */
async function vocabCreateEmpty() {
  const nn = await uiPrompt("新建空词表", "新词表名字：",
                            vocabSuggestName("vocab", "_new"));
  if (!nn) return;
  try {
    const r = await api("/api/vocab/create", {name: nn});
    toast("已创建空词表 " + nn, false);
    await refreshVocabList();
    await vocabActivate(nn);
  } catch (e) {
    toast("创建失败: " + e.message, true);
  }
}
async function vocabActivate(name) {
  if (!VOCAB.tabs.includes(name)) VOCAB.tabs.push(name);
  VOCAB.active = name;
  renderVocabTabs();
  await vocabLoadActive();
}
function vocabCloseTab(name) {
  VOCAB.tabs = VOCAB.tabs.filter(n => n !== name);
  delete VOCAB.src[name];
  if (VOCAB.active === name)
    VOCAB.active = VOCAB.tabs[VOCAB.tabs.length - 1] || "";
  renderVocabTabs();
  if (VOCAB.active) vocabLoadActive();
  else vocabClearPanes();
}
function vocabClearPanes() {
  VOCAB.info = null;
  renderVocabRegions([]);
  drawVocabBar(null);
  $("vocab-bar-note").textContent = "";
  $("vsrc-text").value = "";
  $("vsrc-msg").textContent = "";
  $("vsrc-err").hidden = true;
  vocabWriteNote();
}

/* ---------- 当前标签的数据 ---------- */
async function vocabLoadActive() {
  const name = VOCAB.active;
  renderVocabTabs();
  vocabWriteNote();
  if (!name) { vocabClearPanes(); return; }
  try {
    const r = await api("/api/vocab/regions?name=" + encodeURIComponent(name));
    VOCAB.info = r;
    drawVocabBar(r);
    renderVocabRegions(r.regions || []);
    const ct = r.category_totals || {}, m = vocabMeta(name);
    $("vocab-bar-note").textContent =
      `${name}　容量 ${r.vocab_size}` +
      (m ? `　可见 ${m.visible_size}　不可见 ${m.invisible_size}` : "") +
      (m && m.source ? `　派生自 ${m.source}` : "") +
      `　已用 regular=${ct.regular || 0} fallback=${ct.fallback || 0} ` +
      `special=${ct.special || 0} system=${ct.system || 0}`;
  } catch (e) {
    VOCAB.info = null;
    drawVocabBar(null);
    renderVocabRegions([]);
    $("vocab-bar-note").textContent = "分区信息获取失败: " + e.message;
  }
  if (VOCAB.mode === "text") await vocabLoadSource(name, false);
}

/* ---------- 第 1 块：分区展示 ---------- */
function catColor(cat) {
  return {regular: "#4c8dff", fallback: "#a78bfa",
          special: "#f0b429", system: "#3ddc97"}[cat] || "#6b7280";
}
function drawVocabBar(info) {
  const cv = $("vocab-bar"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = getComputedStyle(document.documentElement)
    .getPropertyValue("--track").trim() || "#242b35";
  ctx.fillRect(0, 0, W, H);
  const total = (info && info.vocab_size) || 0;
  if (!total) return;
  // 浅色块 = 该子区的容量，实心块 = 已用部分（按 begin 定位，不假设区是连续的）
  for (const rg of info.regions || []) {
    const x = rg.begin / total * W, w = rg.size / total * W;
    ctx.fillStyle = catColor(rg.category);
    ctx.globalAlpha = .32;
    ctx.fillRect(x, 0, Math.max(w, .5), H);
    ctx.globalAlpha = 1;
    const uw = rg.size ? rg.used / rg.size * w : 0;
    if (uw > 0) ctx.fillRect(x, 0, Math.max(uw, .5), H);
  }
}
function renderVocabRegions(regions) {
  const tb = $("vocab-regions").querySelector("tbody");
  if (!regions.length) {
    tb.innerHTML = `<tr><td colspan="7" class="hint">暂无分区数据</td></tr>`;
    return;
  }
  tb.innerHTML = regions.map(rg => {
    const pct = rg.size ? (rg.used / rg.size * 100).toFixed(2) : "0.00";
    const samples = (rg.samples || []).map(t =>
      `<code>${esc(t)}</code>`).join(" ");
    return `<tr><td>${esc(rg.region)}</td><td>${esc(rg.category)}</td>` +
      `<td class="mono">${rg.begin}–${rg.begin + rg.size - 1}</td>` +
      `<td class="mono">${rg.size}</td><td class="mono">${rg.used}</td>` +
      `<td class="mono">${pct}%</td>` +
      `<td class="samples">${samples || '<span class="hint">—</span>'}</td></tr>`;
  }).join("");
}
/* 大类下拉写死两档：.vfc 的文件格式只有「引号包裹=可显示 / 尖括号包裹=不可显示」
   两种形态，正好对应 regular / special。fallback（可显示的单字母）与 system
   （{ } | 架构契约头）是派生的、不是用户可选的，所以不按分区推导。
   注意：分区表里的 regions[].category 仍是 four-category 语义，照实渲染即可。 */

/* ---------- 第 2 块：两个模式 + 写入目标 ---------- */
async function vocabSetMode(m) {
  VOCAB.mode = m;
  const isText = m === "text";
  $("vocab-pane-form").hidden = isText;
  $("vocab-pane-text").hidden = !isText;
  if (isText && VOCAB.active) await vocabLoadSource(VOCAB.active, false);
}
function vocabWriteNote() {
  const el = $("vocab-write-note");
  if (!VOCAB.active) { el.textContent = ""; return; }
  el.textContent = VOCAB.write === "fork"
    ? "改动会先 Fork 出一份新词表，再写进新词表"
    : vocabIsMain(VOCAB.active)
      ? "当前是 main 词表，添加 / 保存时会先要求 Fork"
      : `改动直接写入 ${VOCAB.active}`;
}
function vocabSegSync(id, v) {
  $(id).querySelectorAll("button").forEach(b =>
    b.classList.toggle("on", b.dataset.v === v));
}
function vocabSegWire(id, setter) {
  $(id).querySelectorAll("button").forEach(b => {
    b.onclick = () => { setter(b.dataset.v); vocabSegSync(id, b.dataset.v); };
  });
}
vocabSegWire("vocab-mode-seg", v => vocabSetMode(v));
vocabSegWire("vocab-write-seg", v => { VOCAB.write = v; vocabWriteNote(); });
vocabSegSync("vocab-mode-seg", VOCAB.mode);
vocabSegSync("vocab-write-seg", VOCAB.write);

/* 编辑前的统一闸门：main 词表不让直接改 → 弹窗解释 + 给 Fork 出口。
   返回 {name, mode, new_name}；null = 用户放弃。 */
async function vocabResolveWrite() {
  const name = VOCAB.active;
  if (!name) { toast("请先打开一个词表标签页", true); return null; }
  if (VOCAB.write === "fork") {
    const nn = await vocabAskNewName(name, "_fork", "Fork 出新词表");
    return nn ? {name, mode: "fork", new_name: nn} : null;
  }
  if (vocabIsMain(name)) {
    const ok = await modalOpen("main 词表不能直接修改",
      `「${name}」是 main 词表。${MAIN_WHY}`, "Fork 一份再改", true);
    if (!ok) return null;
    const nn = await vocabAskNewName(name, "_fork", "Fork 出新词表");
    return nn ? {name, mode: "fork", new_name: nn} : null;
  }
  return {name, mode: "this", new_name: null};
}
async function vocabAskNewName(base, suffix, title) {
  const nn = await uiPrompt(title, `从「${base}」派生一份新词表，名字：`,
                            vocabSuggestName(base, suffix));
  if (!nn) return null;
  if (VOCAB.list.some(v => v.name === nn)) {
    toast("词表 " + nn + " 已存在", true);
    return null;
  }
  return nn;
}
async function vocabDoFork(name, newName) {
  const r = await api("/api/vocab/fork", {name, new_name: newName});
  const nn = r.name || newName;
  await refreshVocabList();
  await vocabActivate(nn);
  toast("已 fork 出 " + nn);
  return nn;
}

/* 模式一：表单式（添加 / 删除 / 查询） */
async function vocabTokenAction(action) {
  const token = $("vtok-token").value;
  const msg = $("vtok-msg");
  if (!token) { msg.textContent = "请输入 token 文本"; return; }
  if (!VOCAB.active) { toast("请先打开一个词表标签页", true); return; }
  // 查询是只读的：不写词表，也就不必过 Fork 闸门
  const tgt = action === "query"
    ? {name: VOCAB.active, mode: "this", new_name: null}
    : await vocabResolveWrite();
  if (!tgt) return;
  msg.textContent = "处理中…";
  try {
    let name = tgt.name;
    if (tgt.mode === "fork") name = await vocabDoFork(tgt.name, tgt.new_name);
    const r = await api("/api/vocab/token",
      {name, action, token, category: $("vtok-cat").value});
    if (action === "add")
      msg.textContent = `已添加 ${r.token} → #${r.id}（${r.category || "—"}）；`
        + `词表容量 ${r.vocab_size}`;
    else if (action === "remove")
      msg.textContent = `已删除 ${r.token}（原 #${r.id}）；词表容量 ${r.vocab_size}`;
    else
      msg.textContent = r.found
        ? `命中 #${r.id}　${r.token}（${r.category || "—"}）`
        : `未命中：${token}`;
    if (action !== "query") {
      $("vtok-token").value = "";
      await refreshVocabList();
      await vocabLoadActive();
    }
  } catch (e) {
    msg.textContent = "错误: " + e.message;
    toast(e.message, true);
  }
}
$("vtok-add").onclick = () => vocabTokenAction("add");
$("vtok-remove").onclick = () => vocabTokenAction("remove");
$("vtok-query").onclick = () => vocabTokenAction("query");
$("vtok-token").addEventListener("keydown", e => {
  if (e.key === "Enter") $("vtok-add").click();
});

/* 模式二：文本式（裸文本编辑区，保存时语法校验） */
async function vocabLoadSource(name, force) {
  const ta = $("vsrc-text");
  if (!name) { ta.value = ""; return; }
  if (!force && VOCAB.src[name] != null) {
    ta.value = VOCAB.src[name];
    $("vsrc-msg").textContent = `${name}　${VOCAB.src[name].length} 字符`;
    return;
  }
  ta.value = "载入中…";
  try {
    const r = await api("/api/vocab/source?name=" + encodeURIComponent(name));
    VOCAB.src[name] = r.text || "";
    ta.value = VOCAB.src[name];
    $("vsrc-msg").textContent = `${r.name}　${r.vocab_size} 槽位`;
    $("vsrc-err").hidden = true;
  } catch (e) {
    delete VOCAB.src[name];
    ta.value = "";
    $("vsrc-msg").textContent = "源文本获取失败: " + e.message;
  }
}
function renderVocabSourceErrors(e) {
  const box = $("vsrc-err");
  // 语法错误逐条带行号；e.data 由 api() 挂上的完整响应体
  const errs = (e && e.data && Array.isArray(e.data.errors)) ? e.data.errors : null;
  $("vsrc-msg").textContent = "保存失败: " + e.message + "（内容已保留在编辑区）";
  if (!errs || !errs.length) { box.hidden = true; toast("保存失败: " + e.message, true); return; }
  box.innerHTML = `<div class="eh">共 ${errs.length} 处语法错误，未写入词表：</div><ul>` +
    errs.map(x => `<li>${x.line == null ? "" : `<span class="ln">第 ${x.line} 行</span>`}` +
      `${esc(x.message || "")}</li>`).join("") + `</ul>`;
  box.hidden = false;
}
$("vsrc-save").onclick = async () => {
  const text = $("vsrc-text").value;
  // 先缓存再发请求：保存失败也不能把用户刚写的文本弄丢
  if (VOCAB.active) VOCAB.src[VOCAB.active] = text;
  const tgt = await vocabResolveWrite();
  if (!tgt) return;
  const msg = $("vsrc-msg");
  msg.textContent = "校验并保存中…";
  $("vsrc-err").hidden = true;
  try {
    const body = {name: tgt.name, text, mode: tgt.mode};
    if (tgt.new_name) body.new_name = tgt.new_name;   // mode=this 时不带
    const r = await api("/api/vocab/source", body);
    const nn = r.name || (tgt.mode === "fork" ? tgt.new_name : tgt.name);
    VOCAB.src[nn] = text;
    if (nn !== tgt.name) delete VOCAB.src[tgt.name];
    await refreshVocabList();
    await vocabActivate(nn);
    msg.textContent = `已保存 ${nn}（${r.vocab_size} 槽位）`;
  } catch (e) {
    renderVocabSourceErrors(e);
  }
};
$("vsrc-reload").onclick = () => vocabLoadSource(VOCAB.active, true);

/* ---------- 第 3 块：删除 / 复制该词表 ---------- */
$("vocab-del").onclick = async () => {
  const name = VOCAB.active, msg = $("vocab-foot-msg");
  if (!name) { msg.textContent = "请先打开一个词表标签页"; return; }
  if (vocabIsMain(name)) {
    await uiAlert("main 词表不能删除。" + MAIN_WHY);
    return;
  }
  if (!await uiConfirm(`删除词表 ${name}？其文件将被移除，且不可恢复。`)) return;
  msg.textContent = "删除中…";
  try {
    await api("/api/vocab/delete", {name});
    delete VOCAB.src[name];
    VOCAB.tabs = VOCAB.tabs.filter(n => n !== name);
    VOCAB.active = VOCAB.tabs[0] || "";
    msg.textContent = "已删除 " + name;
    await refreshVocabList();
    if (VOCAB.active) await vocabLoadActive(); else vocabClearPanes();
    toast("已删除 " + name);
  } catch (e) {
    msg.textContent = "错误: " + e.message;
    toast(e.message, true);
  }
};
$("vocab-dup").onclick = async () => {
  const name = VOCAB.active, msg = $("vocab-foot-msg");
  if (!name) { msg.textContent = "请先打开一个词表标签页"; return; }
  const nn = await vocabAskNewName(name, "_copy", "复制词表");
  if (!nn) return;
  msg.textContent = "复制中…";
  try {
    await vocabDoFork(name, nn);
    msg.textContent = "已复制为 " + nn;
  } catch (e) {
    msg.textContent = "错误: " + e.message;
    toast(e.message, true);
  }
};

/* ---------- 页面入口 ---------- */
async function refreshVocab() {
  const note = $("vtab-note");
  try {
    await refreshVocabList();
  } catch (e) {
    VOCAB.tabs = []; VOCAB.active = "";
    note.textContent = "词表列表获取失败: " + e.message;
    vocabClearPanes();
    renderVocabTabs();
    return;
  }
  if (!VOCAB.list.length) {
    note.textContent = "还没有任何词表";
    vocabClearPanes();
    renderVocabTabs();
    return;
  }
  note.textContent = `${VOCAB.list.length} 个词表　·　` +
    `点最右侧 ＋ 可打开其它词表（上限 ${VOCAB.max || "—"} 槽位）`;
  await vocabLoadActive();
}
// 首屏就把转换器四页 / 建模页的词表下拉填上（失败静默，切到词表页会再报一次）
refreshVocabList().catch(() => {});

/* =========================================================================
   自定义快捷键
   -------------------------------------------------------------------------
   可绑定动作：隐藏/展开边栏、切换工具（上一个 / 下一个）。
   - 绑定存 localStorage: gargantua.hotkeys（JSON：动作名 → 组合串）。
   - 组合串规范：Mod（macOS=⌘ / 其它=Ctrl）+ Alt + Shift + 主键，
     例如 "Mod+B" / "Alt+["。Mod 同时匹配 metaKey 与 ctrlKey。
   - 设置页「录制」：点录制后下一次按键直接落定，Esc 取消；新组合与
     其它动作冲突时自动解绑旧的那一项并提示。
   - 输入框聚焦时只放行带 Mod/Alt 的组合，避免从打字里抢走普通字母键。
   - 模态框打开期间全部让路（Escape / Enter 归对话框自己处理）。
   ========================================================================= */
const HK_KEY = "gargantua.hotkeys";
const HK_DEFAULTS = {
  toggleSidebar: "Mod+B",
  prevTool: "Alt+[",
  nextTool: "Alt+]",
};
const TOOL_PAGES = ["text", "audio", "image", "video", "prims",
                    "json", "pdraw"];
function cycleTool(dir) {
  const cur = document.querySelector(".page.active");
  const id = cur ? cur.id.slice(2) : "";
  const i = TOOL_PAGES.indexOf(id);
  if (i < 0) {
    switchPage(dir > 0 ? TOOL_PAGES[0] : TOOL_PAGES[TOOL_PAGES.length - 1]);
    return;
  }
  switchPage(TOOL_PAGES[(i + dir + TOOL_PAGES.length) % TOOL_PAGES.length]);
}
const HK_ACTIONS = {
  toggleSidebar: () => toggleSidebar(),
  prevTool: () => cycleTool(-1),
  nextTool: () => cycleTool(1),
};
function loadHotkeys() {
  try {
    const raw = JSON.parse(localStorage.getItem(HK_KEY) || "{}");
    const out = Object.assign({}, HK_DEFAULTS);
    // 只接受已知动作 + 字符串值，脏数据（旧版本/手改）不污染运行时
    Object.keys(out).forEach(k => {
      if (raw && typeof raw[k] === "string") out[k] = raw[k];
    });
    return out;
  } catch (e) { return Object.assign({}, HK_DEFAULTS); }
}
let HOTKEYS = loadHotkeys();
let HK_RECORDING = null;
function saveHotkeys() {
  try { localStorage.setItem(HK_KEY, JSON.stringify(HOTKEYS)); } catch (e) {}
}
/* 主键优先用物理 code：macOS 上 Option(Alt)+[ 的 event.key 会变成 “ 之类
   的排版字符，用 key 永远匹配不上 "Alt+["；code 是 BracketLeft，稳定。 */
function keyFromEvent(e) {
  const code = e.code || "";
  let m = /^Key([A-Z])$/.exec(code); if (m) return m[1];
  m = /^Digit([0-9])$/.exec(code); if (m) return m[1];
  const cm = {BracketLeft: "[", BracketRight: "]", Semicolon: ";", Quote: "'",
              Comma: ",", Period: ".", Slash: "/", Backslash: "\\",
              Minus: "-", Equal: "=", Backquote: "`"};
  if (cm[code]) return cm[code];
  const k = e.key;
  if (!k) return "";
  const map = {" ": "Space", "ArrowUp": "Up", "ArrowDown": "Down",
               "ArrowLeft": "Left", "ArrowRight": "Right", "Escape": "Esc",
               "Enter": "Return", "Backspace": "Delete"};
  return map[k] || (k.length === 1 ? k.toUpperCase() : k);
}
function eventToCombo(e) {
  const k = e.key;
  if (!k || ["Control", "Shift", "Alt", "Meta", "CapsLock", "Tab",
             "Unidentified"].indexOf(k) >= 0) return null;
  const parts = [];
  if (e.metaKey || e.ctrlKey) parts.push("Mod");
  if (e.altKey) parts.push("Alt");
  if (e.shiftKey) parts.push("Shift");
  const key = keyFromEvent(e);
  if (!key) return null;
  parts.push(key);
  return parts.join("+");
}
function comboLabel(combo) {
  if (!combo) return "未绑定";
  const isMac = /Mac|iPhone|iPad/.test(
    navigator.platform || navigator.userAgent || "");
  const parts = combo.split("+");
  const key = parts.pop();
  const names = {Mod: isMac ? "⌘" : "Ctrl", Alt: isMac ? "⌥" : "Alt",
                 Shift: isMac ? "⇧" : "Shift"};
  return parts.map(m => names[m] || m).concat(key).join(" ");
}
function renderHotkeys() {
  Object.keys(HK_DEFAULTS).forEach(a => {
    const label = $("hk-label-" + a);
    if (!label) return;
    label.textContent = comboLabel(HOTKEYS[a]);
    label.classList.toggle("recording", HK_RECORDING === a);
  });
  $$("[data-hk]").forEach(b => {
    b.textContent = HK_RECORDING === b.dataset.hk ? "按任意键…" : "录制";
  });
}
$$("[data-hk]").forEach(b => {
  b.onclick = () => {
    HK_RECORDING = b.dataset.hk;
    $("hotkey-msg").textContent = "请按下想要的组合键（Esc 取消）…";
    renderHotkeys();
  };
});
$("hotkey-reset").onclick = () => {
  HOTKEYS = Object.assign({}, HK_DEFAULTS);
  HK_RECORDING = null;
  saveHotkeys(); renderHotkeys();
  $("hotkey-msg").textContent = "已恢复默认快捷键";
};
renderHotkeys();

// 捕获阶段监听：抢在输入框 / 其它 keydown 处理之前判定；匹配即
// preventDefault，避免浏览器默认行为（如 ⌘B）同时触发。
document.addEventListener("keydown", e => {
  if (!$("modal-mask").hidden) return;          // 对话框期间全部让路
  if (HK_RECORDING) {
    e.preventDefault();
    if (e.key === "Escape") {
      HK_RECORDING = null; renderHotkeys();
      $("hotkey-msg").textContent = "已取消录制";
      return;
    }
    const combo = eventToCombo(e);
    if (!combo) return;                          // 只按了修饰键，继续等
    let stolen = null;
    Object.keys(HOTKEYS).forEach(a => {
      if (a !== HK_RECORDING && HOTKEYS[a] === combo) {
        HOTKEYS[a] = ""; stolen = a;
      }
    });
    HOTKEYS[HK_RECORDING] = combo;
    HK_RECORDING = null;
    saveHotkeys(); renderHotkeys();
    $("hotkey-msg").textContent = stolen
      ? `已绑定；原「${stolen}」的相同组合已自动解绑`
      : "已保存，立即生效";
    return;
  }
  const combo = eventToCombo(e);
  if (!combo) return;
  const ae = document.activeElement;
  const typing = !!(ae && (ae.tagName === "INPUT" || ae.tagName === "TEXTAREA" ||
                           ae.tagName === "SELECT" || ae.isContentEditable));
  for (const a of Object.keys(HOTKEYS)) {
    const c = HOTKEYS[a];
    if (!c || c !== combo) continue;
    // 正在打字时只有带 Mod/Alt 的组合才抢，纯字母/箭头留给输入
    if (typing && !/(^|\+)(Mod|Alt)(\+|$)/.test(c)) continue;
    e.preventDefault();
    HK_ACTIONS[a]();
    return;
  }
}, true);

/* =========================================================================
   终端：项目目录内的交互式 shell（后端 server_term.py）
   -------------------------------------------------------------------------
   行模式：回车把整行发给后端，PTY 的输出由后端渲染成纯文本，这里每
   250ms 轮询一次整体替换 <pre>。不做增量拼接——少一个包就永久错位，
   而本地 localhost 全量传 100KB 的代价可以忽略。
   ========================================================================= */
const TERM = {
  opened: false,
  timer: null,
  hist: [],          // 命令历史（↑↓ 翻）
  hi: -1,
  stick: true,       // 是否吸附在底部（用户往上翻就不抢滚动位置）
  secret: false,     // 后端最后一行是密码提示 → 输入框切 password
};

function termScroll() {
  const out = $("term-out");
  if (!out) return;
  if (TERM.stick) out.scrollTop = out.scrollHeight;
}

async function termOpen() {
  const r = await api("/api/term/open", {});
  TERM.opened = true;
  $("term-cwd").textContent = r.cwd;
  $("term-msg").textContent = "";
  TERM.stick = true;
  await termPoll();
}

async function termPoll() {
  try {
    const r = await api("/api/term/poll");
    const out = $("term-out");
    if (!out) return;
    if (out.textContent !== r.text) {
      out.textContent = r.text;
      termScroll();
    }
    $("term-cwd").textContent = r.cwd || "—";
    // 密码提示（sudo / ssh / passwd）：输入框切成 password、不回显、
    // 不进命令历史；提示结束后自动切回 text。
    const inp = $("term-line");
    if (inp) {
      const secret = !!r.awaiting_secret;
      if (secret !== TERM.secret) {
        TERM.secret = secret;
        inp.type = secret ? "password" : "text";
        inp.placeholder = secret
          ? "输入密码后回车（不回显，也不会记进历史）"
          : "输入命令，回车执行（↑↓ 翻历史，Ctrl-C 中断）";
        if (secret) inp.value = "";
      }
    }
    if (!r.alive) $("term-msg").textContent = "会话已结束，点「重开」";
    else if (TERM.secret) $("term-msg").textContent = "正在等待密码输入…";
    else if ($("term-msg").textContent === "正在等待密码输入…")
      $("term-msg").textContent = "";
  } catch (e) {
    $("term-msg").textContent = "轮询失败: " + e.message;
  }
}

async function termInit() {
  if (!TERM.timer) {
    TERM.timer = setInterval(() => {
      // 只在终端页可见时轮询，切走就停，别白耗
      if ($("p-term") && $("p-term").classList.contains("active")) termPoll();
    }, 250);
  }
  try {
    if (!TERM.opened) await termOpen();
    else await termPoll();
    $("term-line").focus();
  } catch (e) {
    $("term-msg").textContent = "打开终端失败: " + e.message;
  }
}

async function termSubmit() {
  const inp = $("term-line");
  const line = inp.value;
  const secret = TERM.secret;         // 密码行：不 trim 判空、不进历史
  inp.value = "";
  if (!secret && !line.trim()) return;
  if (secret) TERM.secret = false;    // 发出去就退出密码模式，等下一轮 poll 判定
  else {
    TERM.hist.push(line);
    TERM.hi = TERM.hist.length;
  }
  TERM.stick = true;
  try {
    await api("/api/term/input", {line, secret});
  } catch (e) {
    $("term-msg").textContent = e.message;
  }
  await termPoll();
}

/* 中断：写 Ctrl-C，并直接给前台进程组补一发 SIGINT（force）。
   sudo / ssh 这类把终端设成原始模式的程序，单靠 \x03 可能没反应。 */
async function termInterrupt(force) {
  try {
    await api("/api/term/interrupt", {force: !!force});
  } catch (e) {
    $("term-msg").textContent = e.message;
  }
  await termPoll();
}

(function wireTerm() {
  const inp = $("term-line");
  if (!inp) return;
  inp.addEventListener("keydown", e => {
    if (e.key === "Enter") { e.preventDefault(); termSubmit(); return; }
    // Ctrl-C = 中断前台任务（与真实终端一致）；有选中文本时让浏览器复制
    if (e.ctrlKey && !e.metaKey && !e.altKey &&
        (e.key === "c" || e.key === "C")) {
      const sel = String(window.getSelection() || "");
      if (!sel) { e.preventDefault(); termInterrupt(true); }
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      if (!TERM.hist.length) return;
      TERM.hi = Math.max(0, (TERM.hi < 0 ? TERM.hist.length : TERM.hi) - 1);
      inp.value = TERM.hist[TERM.hi] || "";
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (!TERM.hist.length) return;
      TERM.hi = Math.min(TERM.hist.length, TERM.hi + 1);
      inp.value = TERM.hi >= TERM.hist.length ? "" : TERM.hist[TERM.hi];
    }
  });
  const out = $("term-out");
  if (out) out.addEventListener("scroll", () => {
    TERM.stick = out.scrollTop + out.clientHeight >= out.scrollHeight - 24;
  });
  $("term-clear").onclick = async () => {
    try { await api("/api/term/clear", {}); } catch (e) {}
    await termPoll();
  };
  $("term-interrupt").onclick = () => termInterrupt(true);
  $("term-kill").onclick = async () => {
    try {
      await api("/api/term/kill", {});
      $("term-msg").textContent = "已强制结束前台进程";
    } catch (e) {
      $("term-msg").textContent = e.message;
    }
    await termPoll();
  };
  $("term-restart").onclick = async () => {
    try { await api("/api/term/close", {}); } catch (e) {}
    TERM.opened = false;
    await termInit();
  };
})();

/* =========================================================================
   图元绘画：任务列队 / 渲染 / token 输出
   -------------------------------------------------------------------------
   任务制：建任务 → 入队 → 后端顺序跑 → 结果落盘（service/prims/output/jobs）。
   前端只做三件事：轮询任务表、按 `since` 增量拉实时图元、把图元画到 canvas。

   为什么用 canvas 而不是 SVG：一次转换能出几千个图元，SVG 每加一个节点都要
   重新布局；canvas 是「画上去就不管了」，实时渲染和 4000 步的时间线都扛得住。
   ========================================================================= */
const PRIMS = {
  view: "queue",        // queue | render | tokens
  jobs: [],             // 全部任务 meta（新→旧）
  selected: null,       // 队列里选中的任务 id
  editing: null,        // 正在编辑的排队任务 id（null = 新建）
  listTimer: null,
  logMode: "job",       // job | all
  canvasAuto: true,     // 画布跟随图片比例
  // 渲染页
  tabs: [],             // [{id, name}]
  active: null,
  mode: "compare",      // compare | timeline
  prims: [],            // 当前标签页已拿到的图元
  canvas: null,         // [w,h]
  status: "idle",
  drawn: 0,             // 已经画到第几个（增量绘制）
  tlStep: 0,
  playing: false,
  playTimer: null,
  // token 页
  tokTabs: [],
  tokActive: null,
};

const _PRIM_DOT = { queued: "", running: "running", done: "done", error: "error" };

/* ---------------- 通用：图元 → canvas ---------------- */
function primFill(p) {
  return `rgb(${_pdTo8(p.r)},${_pdTo8(p.g)},${_pdTo8(p.b)})`;
}
function drawOne(ctx, p, w, h) {
  if (p.shape_type === -1) {
    ctx.fillStyle = primFill(p);
    ctx.fillRect(0, 0, w, h);
    return;
  }
  ctx.save();
  ctx.translate(p.cx, p.cy);
  if (p.theta) ctx.rotate(p.theta);
  ctx.fillStyle = primFill(p);
  if (p.shape_type === 1) {
    ctx.beginPath();
    ctx.ellipse(0, 0, p.w / 2, p.h / 2, 0, 0, Math.PI * 2);
    ctx.fill();
  } else {
    ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
  }
  ctx.restore();
}
/* 把 canvas 等比塞进 box（不改内部分辨率，只改 CSS 尺寸） */
function fitCanvas(box, cv, cw, ch) {
  if (!box || !cv || !cw || !ch) return;
  // box 自己有 12px 内边距（框内留一圈），这里再扣掉一点呼吸量
  const bw = Math.max(40, box.clientWidth - 24);
  const bh = Math.max(40, box.clientHeight - 24);
  const s = Math.min(bw / cw, bh / ch);
  cv.style.width = Math.round(cw * s) + "px";
  cv.style.height = Math.round(ch * s) + "px";
}
/* 缩放 + 平移：作用在元素 transform 上 */
function wireZoomPan(box, target) {
  const st = { z: 1, x: 0, y: 0 };
  const apply = () => {
    target.style.transform =
      `translate(${st.x}px, ${st.y}px) scale(${st.z})`;
  };
  box.addEventListener("wheel", e => {
    e.preventDefault();
    st.z = Math.max(0.15, Math.min(16, st.z * Math.exp(-e.deltaY * 0.0015)));
    apply();
  }, { passive: false });
  let drag = null;
  box.addEventListener("pointerdown", e => {
    if (e.button !== 0) return;
    drag = { x: e.clientX, y: e.clientY, ox: st.x, oy: st.y };
    box.setPointerCapture(e.pointerId);
    box.classList.add("dragging");
  });
  box.addEventListener("pointermove", e => {
    if (!drag) return;
    st.x = drag.ox + (e.clientX - drag.x);
    st.y = drag.oy + (e.clientY - drag.y);
    apply();
  });
  const end = () => { drag = null; box.classList.remove("dragging"); };
  box.addEventListener("pointerup", end);
  box.addEventListener("pointercancel", end);
  return { st, apply, reset() { st.z = 1; st.x = 0; st.y = 0; apply(); } };
}

/* ---------------- 视图切换 ---------------- */
function primsSetView(v) {
  PRIMS.view = v;
  ["queue", "render", "tokens"].forEach(k => {
    const el = $("prims-view-" + k);
    if (el) el.hidden = k !== v;
    const b = document.querySelector(`#prims-viewseg [data-v="${k}"]`);
    if (b) b.classList.toggle("on", k === v);
  });
  if (v === "render") primsRenderRefresh();
  if (v === "tokens") primsTokensRefresh();
}

/* ---------------- 任务表 ---------------- */
async function primsRefreshJobs() {
  try {
    const r = await api("/api/prims/jobs");
    PRIMS.jobs = r.jobs || [];
    primsRenderQueue();
    primsRenderDone();
  } catch (e) { /* 服务还没起来时静默 */ }
}

function jobSub(j) {
  const p = j.params || {};
  const bits = [];
  bits.push(String(p.engine || "auto").toUpperCase());
  bits.push(`${p.steps || 0} 步`);
  if (j.canvas) bits.push(`${j.canvas[0]}×${j.canvas[1]}`);
  if (j.status === "running") {
    const pr = j.progress || {};
    bits.push(`${pr.step || 0}/${pr.total || 0}`);
  }
  if (j.status === "error" && j.error) bits.push(j.error.slice(0, 40));
  return bits.join(" · ");
}

function jobItemHTML(j) {
  const dot = _PRIM_DOT[j.status] || "";
  const t = j.finished
    ? new Date(j.finished * 1000).toLocaleString("zh-CN", { hour12: false })
    : "";
  const sub = j.status === "done"
    ? `${j.n_primitives || 0} 图元 · ${(j.elapsed || 0).toFixed(1)}s · ${t}`
    : jobSub(j);
  return `<div class="q-item${PRIMS.selected === j.id ? " on" : ""}" data-id="${j.id}">
    <span class="q-dot ${dot}"></span>
    <span class="q-main"><span class="q-name">${esc(j.name)}</span>
      <span class="q-sub">${esc(sub)}</span></span>
    <button class="q-del" data-del="${j.id}" title="删除">✕</button>
  </div>`;
}

function primsRenderQueue() {
  const el = $("prims-queue-list");
  if (!el) return;
  const act = PRIMS.jobs.filter(j => j.status === "queued" || j.status === "running");
  el.innerHTML = act.length ? act.map(jobItemHTML).join("")
    : `<div class="hint" style="padding:8px">队列是空的。点右上角 ＋ 建一个任务。</div>`;
  wireJobItems(el);
}
function primsRenderDone() {
  const el = $("prims-done-list");
  if (!el) return;
  const done = PRIMS.jobs.filter(j => j.status === "done" || j.status === "error");
  el.innerHTML = done.length ? done.map(jobItemHTML).join("")
    : `<div class="hint" style="padding:8px">还没有完成的任务。</div>`;
  wireJobItems(el);
  const n = $("prims-done-note");
  if (n) n.textContent = done.length ? `${done.length} 个` : "";
}

function wireJobItems(root) {
  root.querySelectorAll(".q-item").forEach(el => {
    el.onclick = e => {
      if (e.target.closest(".q-del")) return;
      primsSelect(el.dataset.id);
    };
  });
  root.querySelectorAll(".q-del").forEach(b => {
    b.onclick = async e => {
      e.stopPropagation();
      try {
        await api("/api/prims/job/delete", { id: b.dataset.del });
        if (PRIMS.selected === b.dataset.del) primsSelect(null);
        await primsRefreshJobs();
      } catch (err) { toast("删除失败: " + err.message, true); }
    };
  });
}

async function primsSelect(id) {
  PRIMS.selected = id;
  primsRenderQueue(); primsRenderDone();
  const job = PRIMS.jobs.find(j => j.id === id);
  const title = $("prims-edit-title");
  if (!job) {
    PRIMS.editing = null;
    if (title) title.textContent = "新建任务";
    const log = $("prims-log");
    if (log) log.textContent = "";
    return;
  }
  if (title) title.textContent = job.status === "queued" ? "编辑任务" : "任务详情";
  // 表单回填（只有排队中的任务可改）
  PRIMS.editing = job.status === "queued" ? job.id : null;
  const p = job.params || {};
  if ($("prims-engine")) $("prims-engine").value = p.engine || "auto";
  if ($("prims-quality")) $("prims-quality").value = p.quality || "fast";
  if ($("prims-steps")) $("prims-steps").value = p.steps || 100;
  if ($("prims-early")) $("prims-early").value = p.early_stop === false ? "0" : "1";
  const cv = p.canvas;
  primsSetCanvasAuto(cv === "auto" || !cv);
  if (Array.isArray(cv)) {
    if ($("prims-cw")) $("prims-cw").value = cv[0];
    if ($("prims-ch")) $("prims-ch").value = cv[1];
  }
  if ($("prims-name")) $("prims-name").value = job.name || "";
  await primsLoadLog(id);
}

async function primsLoadLog(id) {
  const log = $("prims-log");
  if (!log) return;
  try {
    let lines;
    if (PRIMS.logMode === "all") {
      const r = await api("/api/prims/log?limit=400");
      lines = r.lines || [];
    } else {
      if (!id) { log.textContent = "（没选任务）"; return; }
      const r = await api(`/api/prims/job/live?id=${encodeURIComponent(id)}&since=999999`);
      lines = r.log || [];
    }
    log.textContent = lines.join("\n");
    log.scrollTop = log.scrollHeight;
  } catch (e) { /* ignore */ }
}

/* ---------------- 新建 / 保存 ---------------- */
async function primsCreate() {
  const f = $("prims-file") && $("prims-file").files[0];
  const msg = $("prims-form-msg");
  const steps = Math.floor(Number($("prims-steps").value) || 0);
  const body = {
    engine: $("prims-engine").value,
    quality: $("prims-quality").value,
    steps: steps >= 1 ? steps : 100,
    early_stop: $("prims-early").value === "1",
    name: ($("prims-name").value || "").trim() ||
          (f ? f.name.replace(/\.[^.]+$/, "") : "未命名"),
  };
  if (PRIMS.canvasAuto) {
    body.canvas = "auto";
  } else {
    body.canvas = [Number($("prims-cw").value), Number($("prims-ch").value)];
  }
  try {
    if (PRIMS.editing) {
      await api("/api/prims/job/update", { id: PRIMS.editing, ...body });
      if (msg) msg.textContent = "已保存";
    } else {
      if (!f) { if (msg) msg.textContent = "先选一张图片"; return; }
      body.image_b64 = await fileToB64(f);
      const r = await api("/api/prims/job/create", body);
      if (msg) msg.textContent = "已加入队列";
      PRIMS.selected = r.id;
      if ($("prims-name")) $("prims-name").value = "";
      resetFileInput("prims-file");
      if ($("prims-file-name")) $("prims-file-name").textContent = "未选择";
    }
    await primsRefreshJobs();
    primsSelect(PRIMS.selected);
  } catch (e) {
    if (msg) msg.textContent = "失败: " + e.message;
  }
}

/* 画布 auto / 手动：一个小按钮，点一下换一次（steve 2026-11-03 指定） */
function primsSetCanvasAuto(on) {
  PRIMS.canvasAuto = !!on;
  const b = $("prims-canvas-mode");
  if (b) {
    b.textContent = PRIMS.canvasAuto ? "auto" : "手动";
    b.classList.toggle("on", PRIMS.canvasAuto);
  }
  if ($("prims-cw")) $("prims-cw").disabled = PRIMS.canvasAuto;
  if ($("prims-ch")) $("prims-ch").disabled = PRIMS.canvasAuto;
}

function primsResetForm() {
  PRIMS.editing = null;
  PRIMS.selected = null;
  ["prims-name"].forEach(id => { if ($(id)) $(id).value = ""; });
  if ($("prims-engine")) $("prims-engine").value = "auto";
  if ($("prims-quality")) $("prims-quality").value = "fast";
  if ($("prims-steps")) $("prims-steps").value = 275;
  if ($("prims-early")) $("prims-early").value = "1";
  primsSetCanvasAuto(true);
  const t = $("prims-edit-title"); if (t) t.textContent = "新建任务";
  const l = $("prims-log"); if (l) l.textContent = "";
  primsRenderQueue(); primsRenderDone();
}

/* ---------------- 渲染页：标签页 ---------------- */
function primsTabHTML(j, active, kind) {
  return `<button class="prims-tab${active ? " on" : ""}" data-tab="${j.id}" data-kind="${kind}">
    ${esc(j.name)}<span class="x" data-close="${j.id}" data-kind="${kind}">✕</span></button>`;
}
function primsRenderTabs() {
  const strip = $("prims-render-tabs");
  if (!strip) return;
  strip.innerHTML = PRIMS.tabs.map(id => {
    const j = PRIMS.jobs.find(x => x.id === id) || { id, name: id };
    return primsTabHTML(j, PRIMS.active === id, "render");
  }).join("") || `<span class="hint">点 ＋ 打开一个任务</span>`;
  strip.querySelectorAll("[data-tab]").forEach(b => {
    b.onclick = e => {
      if (e.target.closest("[data-close]")) return;
      primsOpenTab(b.dataset.tab, false);
    };
  });
  strip.querySelectorAll("[data-close]").forEach(b => {
    b.onclick = e => {
      e.stopPropagation();
      const id = b.dataset.close;
      PRIMS.tabs = PRIMS.tabs.filter(x => x !== id);
      if (PRIMS.active === id) {
        PRIMS.active = PRIMS.tabs[0] || null;
        primsRenderRefresh();
      }
      primsRenderTabs();
    };
  });
}

async function primsOpenTab(id, add = true) {
  if (add && !PRIMS.tabs.includes(id)) PRIMS.tabs.push(id);
  PRIMS.active = id;
  PRIMS.drawn = 0; PRIMS.prims = []; PRIMS.canvas = null;
  primsRenderTabs();
  await primsRenderRefresh();
}

/* 打开任务选择器（＋） */
function primsPickJob(kind) {
  // 运行中的也能开：渲染页对「还没跑完」的任务走实时增量，边跑边看。
  const cands = PRIMS.jobs.filter(j => j.status !== "error");
  if (!cands.length) { toast("还没有任务", true); return; }
  const body = `<div class="pop-list">${cands.map(j => {
    const st = j.status === "done" ? `${j.n_primitives || 0} 图元 · ${(j.elapsed || 0).toFixed(1)}s`
      : (j.status === "running" ? "运行中…" : "排队中");
    return `<button data-pick="${j.id}">${esc(j.name)}
      <span class="sub">${st}</span></button>`;
  }).join("")}</div>`;
  modalOpenHTML("打开任务", body, "关闭", false);
  $("modal-body").querySelectorAll("[data-pick]").forEach(b => {
    b.onclick = () => {
      const id = b.dataset.pick;
      modalClose(true);
      if (kind === "tokens") primsOpenTokenTab(id);
      else primsOpenTab(id);
    };
  });
}

/* ---------------- 渲染页：取数据 + 画 ---------------- */
async function primsRenderRefresh() {
  const id = PRIMS.active;
  const note = $("cmp-note");
  if (!id) {
    if (note) note.textContent = "没有打开的任务";
    primsPaintAll();
    return;
  }
  const job = PRIMS.jobs.find(j => j.id === id);
  if (!job) return;
  const img = $("cmp-src-img");
  if (img) img.src = `/api/prims/job/source?id=${encodeURIComponent(id)}&t=${job.created}`;
  if (job.status === "done") {
    try {
      const r = await api(`/api/prims/job/result?id=${encodeURIComponent(id)}`);
      PRIMS.canvas = r.canvas;
      PRIMS.prims = r.prims || [];
      PRIMS.status = "done";
      PRIMS.drawn = 0;
      primsPaintAll();
      if (note) note.textContent = `${r.prims.length} 个图元 · 画布 ${r.canvas[0]}×${r.canvas[1]}`;
      const pct = $("cmp-pct"); if (pct) pct.textContent = "完成";
      const fill = $("cmp-fill"); if (fill) fill.style.width = "100%";
      const prog = $("cmp-progress"); if (prog) prog.hidden = false;
      primsSyncTimeline();
    } catch (e) {
      if (note) note.textContent = "读取失败: " + e.message;
    }
    return;
  }
  // 排队/运行中：拉实时增量
  const prog = $("cmp-progress"); if (prog) prog.hidden = false;
  try {
    const r = await api(`/api/prims/job/live?id=${encodeURIComponent(id)}&since=${PRIMS.prims.length}`);
    if (r.canvas) PRIMS.canvas = r.canvas;
    PRIMS.status = r.status;
    if (r.prims && r.prims.length) {
      PRIMS.prims.push(...r.prims);
      primsPaintIncremental();
    }
    const pr = r.progress || {};
    const total = pr.total || 1;
    const pct = Math.min(100, Math.round((pr.step || 0) / total * 100));
    const fill = $("cmp-fill"); if (fill) fill.style.width = pct + "%";
    const pctEl = $("cmp-pct");
    if (pctEl) pctEl.textContent = `${pr.step || 0} / ${total} 步 · ${PRIMS.prims.length} 图元`;
    if (note) note.textContent = r.status === "running"
      ? `运行中 · ${PRIMS.prims.length} 个图元` : (r.note || r.status);
    if (r.status === "done" || r.status === "error") {
      if (r.status === "done") { PRIMS.status = "done"; await primsRenderRefresh(); }
    }
  } catch (e) {
    if (note) note.textContent = "读取失败: " + e.message;
  }
}

function primsEnsureCanvas() {
  const cv = $("cmp-canvas");
  const box = $("cmp-out");
  if (!cv || !box || !PRIMS.canvas) return null;
  const [w, h] = PRIMS.canvas;
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  fitCanvas(box, cv, w, h);
  return cv;
}

function primsPaintAll() {
  const cv = primsEnsureCanvas();
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const [w, h] = PRIMS.canvas;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  PRIMS.prims.forEach(p => drawOne(ctx, p, w, h));
  PRIMS.drawn = PRIMS.prims.length;
  primsLayout();
}

function primsPaintIncremental() {
  const cv = primsEnsureCanvas();
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const [w, h] = PRIMS.canvas;
  if (PRIMS.drawn === 0) { ctx.clearRect(0, 0, w, h); ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, w, h); }
  for (let i = PRIMS.drawn; i < PRIMS.prims.length; i++) drawOne(ctx, PRIMS.prims[i], w, h);
  PRIMS.drawn = PRIMS.prims.length;
  primsLayout();
}

/* 左右 / 上下自动布局：哪个方向能给出更大的显示面积就用哪个 */
function primsLayout() {
  const stage = $("prims-stage-compare");
  const box = $("cmp-out");
  if (!stage || !box || !PRIMS.canvas) return;
  const [cw, ch] = PRIMS.canvas;
  const W = stage.clientWidth, H = stage.clientHeight;
  if (!W || !H) return;
  const A = cw / ch;
  // 并排时每格 W/2 × H；上下时每格 W × H/2。取面积大者。
  const sideArea = Math.min(W / 2, H * A) * Math.min(H, (W / 2) / A);
  const stackArea = Math.min(W, (H / 2) * A) * Math.min(H / 2, W / A);
  const vert = stackArea > sideArea;
  stage.classList.toggle("vert", vert);
  const cv = $("cmp-canvas");
  if (cv) fitCanvas(box, cv, cw, ch);
  const tl = $("tl-canvas");
  if (tl) fitCanvas($("tl-wrap"), tl, cw, ch);
}

/* ---------------- 时间线 ---------------- */
function primsSyncTimeline() {
  const range = $("tl-range");
  if (!range) return;
  range.max = Math.max(0, PRIMS.prims.length - 1);
  PRIMS.tlStep = Math.min(PRIMS.tlStep, Number(range.max));
  range.value = PRIMS.tlStep;
  primsPaintTimeline();
}
function primsPaintTimeline() {
  const cv = $("tl-canvas");
  const box = $("tl-wrap");
  if (!cv || !box || !PRIMS.canvas) return;
  const [w, h] = PRIMS.canvas;
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  fitCanvas(box, cv, w, h);
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, w, h);
  const n = Math.min(PRIMS.prims.length, PRIMS.tlStep + 1);
  for (let i = 0; i < n; i++) drawOne(ctx, PRIMS.prims[i], w, h);
  const lab = $("tl-label");
  if (lab) lab.textContent = `${n} / ${PRIMS.prims.length}`;
}
function primsPlay() {
  PRIMS.playing = !PRIMS.playing;
  const b = $("tl-play");
  if (b) b.textContent = PRIMS.playing ? "⏸ 暂停" : "▶ 播放";
  if (PRIMS.playTimer) { clearInterval(PRIMS.playTimer); PRIMS.playTimer = null; }
  if (!PRIMS.playing) return;
  PRIMS.playTimer = setInterval(() => {
    const range = $("tl-range");
    if (!range) return;
    let v = Number(range.value) + Math.max(1, Math.round(PRIMS.prims.length / 300));
    if (v >= Number(range.max)) { v = Number(range.max); primsPlay(); }
    range.value = v;
    PRIMS.tlStep = v;
    primsPaintTimeline();
  }, 40);
}

/* ---------------- token 页 ---------------- */
function primsRenderTokenTabs() {
  const strip = $("prims-token-tabs");
  if (!strip) return;
  strip.innerHTML = PRIMS.tokTabs.map(id => {
    const j = PRIMS.jobs.find(x => x.id === id) || { id, name: id };
    return primsTabHTML(j, PRIMS.tokActive === id, "tok");
  }).join("") || `<span class="hint">点 ＋ 打开一个已完成任务</span>`;
  strip.querySelectorAll("[data-tab]").forEach(b => {
    b.onclick = e => {
      if (e.target.closest("[data-close]")) return;
      primsOpenTokenTab(b.dataset.tab, false);
    };
  });
  strip.querySelectorAll("[data-close]").forEach(b => {
    b.onclick = e => {
      e.stopPropagation();
      const id = b.dataset.close;
      PRIMS.tokTabs = PRIMS.tokTabs.filter(x => x !== id);
      if (PRIMS.tokActive === id) PRIMS.tokActive = PRIMS.tokTabs[0] || null;
      primsRenderTokenTabs(); primsTokensRefresh();
    };
  });
}
async function primsOpenTokenTab(id, add = true) {
  if (add && !PRIMS.tokTabs.includes(id)) PRIMS.tokTabs.push(id);
  PRIMS.tokActive = id;
  primsRenderTokenTabs();
  await primsTokensRefresh();
}
async function primsTokensRefresh() {
  const pre = $("prims-tokens");
  if (!pre) return;
  primsRenderTokenTabs();
  const id = PRIMS.tokActive;
  if (!id) { pre.textContent = "—"; return; }
  try {
    const r = await api(`/api/prims/job/result?id=${encodeURIComponent(id)}`);
    const lines = [];
    lines.push(`# 画布 ${r.canvas[0]}×${r.canvas[1]} · ${r.prims.length} 个图元 · ${r.tokens.length} 个 token`);
    lines.push("# " + (r.note || ""));
    lines.push(stepsToToken5([{ modality: "draw", token_ids: r.tokens, meta: {} }], "model"));
    pre.textContent = lines.join("\n");
  } catch (e) {
    pre.textContent = "读取失败: " + e.message;
  }
}

/* ---------------- 工具说明（右上角「关于」） ----------------
   所有工具页原先那行「小字介绍」统一搬到这里，并且写得更细。
   工具页只留操作控件——右上角这一个入口，按当前页自动定位到对应条目。 */
const TOOL_DOCS = [
  {
    id: "infer", name: "推理", title: "推理界面 = 可插拔的显示插件",
    body: `
      <p>推理页中间那一整块（左右边栏之间、顶栏之下，<b>含输入框</b>）不属于
      主程序，而是由<b>显示插件</b>渲染。原因很直接：不同词表的 token 长得
      完全不一样，把「怎么显示」写死在主程序里，每加一个词表就要改一次核心。</p>
      <ul>
        <li><b>推理只连两个端点</b>：插件把用户输入转成 infer 接受的格式
        （含 <code>max_new_tokens</code> / temperature / top_p），调
        <code>POST /api/infer</code> 起任务，再用
        <code>GET /api/infer/stream</code> 按 <code>from</code> 游标拉增量
        token；读出来怎么解释、怎么排版全归它，HTML 自己写。宿主另外提供
        <code>reset / stats / status / models / toast</code> 等辅助能力，
        但生成链路只有这两个端点。</li>
        <li><b>宿主给什么</b>：一块满高容器 <code>root</code>，加
        <code>ctx.api</code>、<code>ctx.config</code>（按插件隔离的
        localStorage）、<code>ctx.signal</code>（卸载时中止异步回调）。</li>
        <li><b>当前唯一可用插件</b>：<b>token 对话</b>——微信式多轮对话，右边是
        你的输入、左边是模型输出，上下文逐轮保留；每个 token 以转换器格式
        <code>{text: B46A}</code>（大写十六进制、自界定）或特殊 token
        名字（<code>&lt;caps&gt;</code> / <code>&lt;user&gt;</code> /
        <code>&lt;model&gt;</code> / <code>&lt;stop&gt;</code>…）呈现，顶部实时
        给出 <b>TTFT</b> 与 <b>tok/s</b>。<b>Enter 不发送（防误触）</b>，只能点
        「发送」。输入只接受 token 格式（十六进制 ID、<code>{t:…}</code> /
        <code>{text: …}</code>、固定名字的特殊 token），不解析自然语言。</li>
        <li><b>base 词表插件</b>：开发中，按约定<b>不注册、不显示</b>。
        协议与它未来要用的接口已经写完，见 <code>plugins/README.md</code>。</li>
        <li><b>切换 / 导入</b>：右侧开发者选项 → 显示插件。选择会记住；
        「导入插件…」可选一个本地 <code>.js</code>（可附同名 <code>.css</code>），
        服务端落到 <code>service/web/plugins/</code> 并登记，导入后可直接在
        下拉里切换、也可移除。</li>
      </ul>
      <div class="doc-tip">写自己的插件：默认导出
      <code>{ id, name, mount(root, ctx) }</code> 的 ES 模块，然后从
      「开发者选项 → 显示插件 → 导入插件…」导入即可（不需要改主程序）。
      完整协议、生命周期、ctx 字段与流式时序见
      <code>service/web/plugins/README.md</code>。</div>`
  },
  {
    id: "text", name: "文本", title: "文本 → token 时间步",
    body: `
      <p>把一段自然语言按所选<b>词表</b>编码成 token-5 时间步序列。每个文本
      token 独占一个时间步，时间步是模型的原子单位——压缩、滑窗、丢弃都不
      允许把它切开。</p>
      <ul>
        <li><b>输入</b>：任意文本，UTF-8。英文词未命中词表时逐字母回退；
        完全未收录的字符会被跳过并记入该步的 <code>skipped</code>，
        不会让整次编码报错。</li>
        <li><b>输出</b>：<code>{text: HEX}</code> 序列，<code>{ }</code> 是一个
        时间步，步内多个 token 用 <code>|</code> 分隔。HEX 是词表 ID（十六进制，
        统一大写）。</li>
        <li><b>词表</b>：右上角下拉框选一个词表；不同词表的可见区可以完全不同，
        但契约头（特殊/系统/图元区）一致。</li>
        <li><b>清空</b>只清输出，输入文本保留，方便改一个词再编一次。</li>
      </ul>
      <div class="doc-tip">文本 token 有离散 ID、有 embedding、参与 loss；
      与音频/视觉的「投影向量 token」不是一回事。</div>`
  },
  {
    id: "audio", name: "音频", title: "音频 → 音元 token",
    body: `
      <p>音频被切成固定长度的<b>音元</b>：每 533 个采样点（16 kHz 单声道）
      平铺成一条 533 维向量，再经 <code>audio_proj</code> 投影到 d_model。
      533 采样 ≈ 30 个音元/秒。</p>
      <ul>
        <li><b>格式</b>：wav / mp3 / flac / ogg 都走 ffmpeg 解码，自动重采样到
        16 kHz 单声道，不要求源文件格式统一。</li>
        <li><b>时间步</b>：一个音元 = 一个时间步；时长决定步数，不补零、不跨步。</li>
        <li><b>没有离散 ID</b>：音频 token 是连续投影向量，词表里查不到；
        存档用 <code>.tokenvector</code>，而不是 <code>.tokens</code>。</li>
        <li><b>预览</b>：编码后可直接播放，确认采样率/声道没被弄错。</li>
      </ul>
      <div class="doc-tip">训练与推理时音频向量与文本 ID 走同一条序列，
      但只有文本 token 进 loss。</div>`
  },
  {
    id: "image", name: "图片", title: "图片 → patch token",
    body: `
      <p>图片先拆成 <b>32×32 像素</b>的 patch，每个 patch 内部按 RGBA 平铺成
      4096 维向量，再经 <code>visual_proj</code> 投影到 d_model。一张图的
      <b>全部 patch 放在同一个时间步</b>里——图片是一个原子单位，不会被拆到
      两步。</p>
      <ul>
        <li><b>坐标</b>：patch 的 (x, y) 不单独成 token，而是用可学习的 2D
        位置嵌入加性注入该 patch 的向量。</li>
        <li><b>颜色</b>：内部按 RGBA 各 10 bit 处理；PNG/JPEG 的 8 bit 会在
        边界换算，不丢档位。</li>
        <li><b>上限</b>：patch 总数有硬上限（防止一张超大图把内存打穿），
        超限会直接报错而不是静默缩放。</li>
        <li><b>存档</b>：视觉 token 同样是投影向量，用 <code>.tokenvector</code>。</li>
      </ul>
      <div class="doc-tip">视频就是「每帧一张图」，帧内规则与这里完全一致。</div>`
  },
  {
    id: "video", name: "视频", title: "视频 → 每帧一个时间步",
    body: `
      <p>视频用 ffmpeg 抽帧（默认 30 fps）并单独提取音轨。每一帧 = 一个时间步，
      该步内平铺这一帧的全部 patch token，以及这一帧对应的音频 token。</p>
      <ul>
        <li><b>格式</b>：mp4 / mov / webm / mkv / gif 等，取决于 ffmpeg 的解码能力。</li>
        <li><b>帧 delta</b>：首帧全量；中间帧只送有变化的 patch；每秒首帧、
        最后一帧、以及「整帧都变」的帧会回到全量。这样长视频的 token 量不会
        随帧数线性爆炸。</li>
        <li><b>音轨</b>：按帧时间轴切片，每帧最多携带固定数量的音元；音轨不会
        被截断成「前半段有声音、后半段没有」。</li>
        <li><b>统计</b>：编码后显示帧数、patch 数、音元数与总步数，方便判断
        是不是抽帧太密。</li>
      </ul>
      <div class="doc-tip">视频是重活：几十秒的 1080p 就可能上万个 patch。
      先用短片段试，确认参数再跑长的。</div>`
  },
  {
    id: "prims", name: "图元绘画", title: "位图 → 矢量图元序列",
    body: `
      <p><b>图元绘画</b>把一张位图「重画」成一串矢量图元（矩形 / 椭圆），
      每画一个都记录 9 个参数：位置 <code>x,y</code>、旋转 <code>theta</code>、
      尺寸 <code>w,h</code>、形状 <code>shape</code>、颜色 <code>r,g,b</code>。
      序列最前面还有一段<b>画布声明</b>，说明这张画有多大。</p>
      <ul>
        <li><b>GPU 引擎</b>：项目内 vendored 的 Rust/wgpu 转换器
        （<code>service/prims/fast_shape_render</code>）。按目标画布长宽比挑一个
        小画布拟合（长边 256），结果等比放大；耗时与步数成正比。</li>
        <li><b>CPU 引擎</b>：本地 numpy 贪心拟合，单线程，几秒出结果，适合快速试。</li>
        <li><b>步数</b>：GPU 下是搜索轮数（每轮最多采纳 1 个图元）；CPU 下是
        迭代次数。<b>早停</b>：连续 1000 步没有明显改善就收工——填了 20000
        只出几千个图元通常就是它；关掉才会跑满。</li>
        <li><b>画布</b>：1..1920 × 1..1080 任意尺寸。<code>auto</code> 按原图
        长宽比算出「塞得进上限的最大同比例画布」；<code>手动</code> 才用你填的宽高。</li>
        <li><b>任务列队</b>：一次只跑一个（GPU 独占），其余排队；结果落在
        <code>service/prims/output/jobs/&lt;id&gt;/</code>，可回看、对比、取 token。</li>
        <li><b>渲染</b>：左原图 / 右图元，都能滚轮缩放、拖动平移；边算边画，
        底部有进度条；时间线可按步回放、拖动、自动播放。</li>
      </ul>
      <div class="doc-tip">图元序列的 token 是离散 ID（在词表的 DRAW 区），
      所以它能直接进 <code>.tokens</code>，也能被模型学着画出来。</div>`
  },
  {
    id: "json", name: "Token 编辑器", title: "手写 / 检查 token-5 序列",
    body: `
      <p>直接编辑 token-5 文本流：<code>{ }</code> 是时间步原子，步内元素用
      <code>|</code> 分隔。<code>{ }</code> 和 <code>|</code> 本身不是 token，
      不占位置、不进 loss。</p>
      <ul>
        <li><code>{text: C732}</code>：文本 token，有向量、进 loss，ID 为十六进制。</li>
        <li><code>{wave: HEX}</code>：音频 token（533 采样），连续向量。</li>
        <li><code>{visual: HEX x: 1 y: 2}</code>：视觉 patch（32×32，RGBA 各 10 bit）。</li>
        <li><code>&lt;5&gt;</code>：扩展系统 token（有嵌入、不进 loss），也可以
        直接写名字，如 <code>&lt;stop&gt;</code>。</li>
        <li><b>格式化 / 压缩</b>只改排版，不改 token；<b>复制</b> 复制全文；
        <b>导出 .tokens</b> 按所选词表解析 token 名并校验 ID，生成训练用的位流。</li>
        <li><b>词表</b>：查询与增删统一在「词表」页，这里只做序列编辑。</li>
      </ul>
      <div class="doc-tip">导出前先确认词表：同一个 ID 在不同词表里可能是完全
      不同的 token。</div>`
  },
  {
    id: "pdraw", name: "图元编辑器", title: "手动摆图元，实时看 token",
    body: `
      <p>手动摆放图元：左边实时 SVG 画布，右边字段编辑，底部实时 token 流。
      适合调单个图元的参数、检查画布声明、或手工拼一张小图。</p>
      <ul>
        <li><b>单击</b>空白处落一个图元；<b>拖动</b>平移画布；<b>滚轮</b>缩放。</li>
        <li><b>图层</b>：右侧列表可拖拽排序，越靠后越在上层。</li>
        <li><b>底图</b>：可开关、可调透明度，用来对比「画出来的」和「原图」差多少。</li>
        <li><b>画布</b>：档位与字段 offset 全部来自 <code>/api/status</code>
        （唯一真源是 <code>server/spec.py</code>），前端不抄常量。</li>
        <li><b>token 序列</b>：画布声明打头，其后每个图元 9 个 token；改 token
        会自动重绘，改图元会反写 token。</li>
      </ul>
      <div class="doc-tip">这里的 token 可以直接复制到「Token 编辑器」继续改，
      或导出成训练数据。</div>`
  },
  {
    id: "term", name: "终端", title: "项目目录内的交互式 shell",
    body: `
      <p>装依赖、跑测试、看日志的自留地。工作目录固定在项目根，<code>cd</code>
      到项目外会被拒绝——这是防手滑，不是安全沙箱。</p>
      <ul>
        <li><b>sudo / ssh / passwd</b>：出现 <code>Password:</code> 时输入框会
        自动切成密码模式，输入不回显、也不会记进命令历史；回车即提交。
        需要的是本机当前用户的密码。</li>
        <li><b>中断</b>：输入框里按 <b>Ctrl-C</b>，或点「中断」。它会写 Ctrl-C
        并直接给前台进程组补一发 SIGINT，sudo 卡住时也能打断。</li>
        <li><b>强制结束</b>：SIGKILL 只杀当前前台作业；空闲时等同于结束整个
        shell，需要点「重开」。</li>
        <li><b>重开</b>：关掉当前 shell 重开，环境变量与当前目录都会复位。</li>
      </ul>
      <div class="doc-tip">依赖与环境的安装 / 更新统一在终端里跑：
      <code>python3 start/setup_env.py install</code> 或
      <code>python3 start/setup_env.py update</code>。</div>`
  },
];
let DOC_SELECTED = null;

/* 「关于」是**每页独立**的：打开只显示当前工具那一条，没有目录、没有
   跨页跳转。它描述的是「你现在在用的这个东西」，不是一个全局文档浏览器。 */
function showToolDoc(id) {
  const d = TOOL_DOCS.find(x => x.id === id);
  if (!d) return Promise.resolve();
  DOC_SELECTED = d.id;
  return modalOpenHTML(`关于 · ${d.name}`, `
    <div class="doc-body">
      <h4>${esc(d.title)}</h4>
      ${d.body}
    </div>`, "知道了", false, true);
}
$("about-btn").onclick = () =>
  showToolDoc(document.documentElement.getAttribute("data-page"));

/* ---------------- 初始化 ---------------- */
function primsInit() {
  if (!PRIMS.listTimer) {
    PRIMS.listTimer = setInterval(() => {
      const page = $("p-prims");
      if (!page || !page.classList.contains("active")) return;
      primsRefreshJobs();
      if (PRIMS.view === "render" && PRIMS.active) primsRenderRefresh();
      if (PRIMS.selected) primsLoadLog(PRIMS.selected);
    }, 1200);
  }
  primsRefreshJobs().then(() => {
    if (PRIMS.view === "render") primsRenderRefresh();
    if (PRIMS.view === "tokens") primsTokensRefresh();
  });
}

(function wirePrims() {
  const seg = $("prims-viewseg");
  if (seg) {
    seg.querySelectorAll("[data-v]").forEach(b => {
      b.onclick = () => primsSetView(b.dataset.v);
    });
  }
  const mode = $("prims-mode-seg");
  if (mode) {
    mode.querySelectorAll("[data-v]").forEach(b => {
      b.onclick = () => {
        PRIMS.mode = b.dataset.v;
        mode.querySelectorAll("[data-v]").forEach(x => x.classList.toggle("on", x === b));
        const cmp = $("prims-stage-compare");
        const tl = $("prims-stage-timeline");
        if (cmp) cmp.hidden = PRIMS.mode !== "compare";
        if (tl) tl.hidden = PRIMS.mode !== "timeline";
        if (PRIMS.mode === "timeline") { primsSyncTimeline(); primsPaintTimeline(); }
        else primsLayout();
      };
    });
  }
  const logSeg = $("prims-log-seg");
  if (logSeg) {
    logSeg.querySelectorAll("[data-v]").forEach(b => {
      b.onclick = () => {
        PRIMS.logMode = b.dataset.v;
        logSeg.querySelectorAll("[data-v]").forEach(x => x.classList.toggle("on", x === b));
        primsLoadLog(PRIMS.selected);
      };
    });
  }
  if ($("prims-add")) $("prims-add").onclick = primsResetForm;
  if ($("prims-create")) $("prims-create").onclick = primsCreate;
  if ($("prims-canvas-mode")) {
    $("prims-canvas-mode").onclick = () => primsSetCanvasAuto(!PRIMS.canvasAuto);
  }
  if ($("prims-render-add")) $("prims-render-add").onclick = () => primsPickJob("render");
  if ($("prims-token-add")) $("prims-token-add").onclick = () => primsPickJob("tokens");
  if ($("tl-range")) {
    $("tl-range").oninput = e => {
      PRIMS.tlStep = Number(e.target.value);
      primsPaintTimeline();
    };
  }
  if ($("tl-play")) $("tl-play").onclick = primsPlay;

  // 缩放 / 平移（三个画布各自一套）
  const srcImg = $("cmp-src-img");
  if ($("cmp-src") && srcImg) wireZoomPan($("cmp-src"), srcImg);
  if ($("cmp-out") && $("cmp-canvas")) wireZoomPan($("cmp-out"), $("cmp-canvas"));
  if ($("tl-wrap") && $("tl-canvas")) wireZoomPan($("tl-wrap"), $("tl-canvas"));

  // 尺寸变化时重新贴合
  if (typeof ResizeObserver === "function") {
    const ro = new ResizeObserver(() => {
      if (PRIMS.view === "render") {
        primsLayout();
        const cv = $("cmp-canvas");
        if (cv && PRIMS.canvas) fitCanvas($("cmp-out"), cv, PRIMS.canvas[0], PRIMS.canvas[1]);
        const tl = $("tl-canvas");
        if (tl && PRIMS.canvas) fitCanvas($("tl-wrap"), tl, PRIMS.canvas[0], PRIMS.canvas[1]);
      }
    });
    ["prims-stage-compare", "prims-stage-timeline"].forEach(id => {
      const el = $(id); if (el) ro.observe(el);
    });
  }
})();

/* ===========================================================================
   自绘下拉：把每个 <select> 换成「按钮 + 自绘菜单」
   ---------------------------------------------------------------------------
   原生 <select> 的弹层由系统绘制，配色/圆角/字号跟不了主题（界面铁律：
   每一个像素都由本文件画）。这里保留 select 本体当值容器——JS 里所有
   `.value` 读写、`change` 事件、`selectedOptions` 全部照旧——只把外观换掉：
     · 实例上重定义 value 的 setter，程序赋值后标签立即同步；
     · MutationObserver 盯 option 增删（词表列表是异步填的）；
     · 菜单往上/往下翻由可用空间决定，不贴边溢出。
   =========================================================================== */
function enhanceSelect(sel) {
  if (!sel || sel.dataset.csel) return;
  sel.dataset.csel = "1";

  const wrap = document.createElement("span");
  wrap.className = "csel";
  // 宽度语义要跟着原 select 走：全局 input/select 是 width:100%（吃满容器），
  // 写了 style="width:auto" 的才是「按内容收缩」。包装层必须复刻这个判断，
  // 否则没写宽度的下拉会从「占满一行」变成「缩成一小截」。
  if (sel.style.width) wrap.style.width = sel.style.width;
  else wrap.classList.add("csel-fill");
  if (sel.style.minWidth) wrap.style.minWidth = sel.style.minWidth;
  sel.parentNode.insertBefore(wrap, sel);
  wrap.appendChild(sel);

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "csel-btn";
  btn.innerHTML = `<span class="csel-label"></span>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
         stroke-linecap="round" stroke-linejoin="round">
      <path d="M6 9l6 6 6-6"/></svg>`;
  const menu = document.createElement("div");
  menu.className = "csel-menu";
  menu.hidden = true;
  wrap.appendChild(btn);
  // 菜单挂在 body 上（fixed 定位）：侧栏 / 表格 / 弹窗里的滚动容器都有
  // overflow，菜单如果留在原地会被裁掉一半。
  document.body.appendChild(menu);

  const label = btn.querySelector(".csel-label");
  const TICK = `<svg class="tick" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" stroke-width="2.4" stroke-linecap="round"
      stroke-linejoin="round"><path d="M20 6L9 17l-5-5"/></svg>`;

  const renderMenu = () => {
    menu.innerHTML = Array.from(sel.options).map((o, i) =>
      `<button type="button" class="csel-opt${o.selected ? " on" : ""}"
               data-i="${i}"${o.disabled ? " disabled" : ""}>${
        esc(o.textContent)}${TICK}</button>`).join("");
    menu.querySelectorAll("[data-i]").forEach(b => {
      b.onclick = ev => {
        ev.stopPropagation();
        const o = sel.options[+b.dataset.i];
        if (!o || o.disabled) return;
        sel.selectedIndex = +b.dataset.i;
        sel.dispatchEvent(new Event("change", {bubbles: true}));
        closeMenu();
        btn.focus();
      };
    });
  };

  const sync = () => {
    const o = sel.selectedOptions && sel.selectedOptions[0];
    label.textContent = o ? o.textContent : "";
    btn.disabled = !!sel.disabled;
    wrap.title = o ? o.textContent : "";
    if (!menu.hidden) renderMenu();
  };

  const closeMenu = () => {
    menu.hidden = true;
    wrap.classList.remove("open");
  };

  const openMenu = () => {
    if (sel.disabled) return;
    document.querySelectorAll(".csel.open").forEach(w => {
      if (w === wrap) return;
      w.classList.remove("open");
      const m = w._cselMenu;
      if (m) m.hidden = true;
    });
    renderMenu();
    menu.hidden = false;
    wrap.classList.add("open");
    // fixed 定位：以按钮矩形为准，下方放不下就往上翻
    const r = btn.getBoundingClientRect();
    menu.style.width = r.width + "px";
    // 水平钳制：右侧边栏里的下拉贴着窗口右沿，菜单不能溢出屏幕
    menu.style.left = Math.round(Math.max(8, Math.min(
      r.left, window.innerWidth - r.width - 8))) + "px";
    const h = Math.min(260, menu.scrollHeight);
    const below = window.innerHeight - r.bottom;
    const above = r.top;
    const top = (below < h + 12 && above > below)
      ? Math.max(8, r.top - h - 6)
      : Math.min(window.innerHeight - h - 8, r.bottom + 6);
    menu.style.top = Math.round(Math.max(8, top)) + "px";
    const on = menu.querySelector(".csel-opt.on");
    if (on) on.scrollIntoView({block: "nearest"});
  };
  wrap._cselMenu = menu;

  btn.onclick = ev => {
    ev.stopPropagation();
    menu.hidden ? openMenu() : closeMenu();
  };
  wrap.addEventListener("keydown", ev => {
    if (ev.key === "Escape") { closeMenu(); btn.focus(); return; }
    if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
      if (menu.hidden) { openMenu(); ev.preventDefault(); return; }
      const opts = Array.from(menu.querySelectorAll(".csel-opt:not([disabled])"));
      if (!opts.length) return;
      const cur = opts.findIndex(b => b.classList.contains("on"));
      const step = ev.key === "ArrowDown" ? 1 : -1;
      const next = opts[Math.max(0, Math.min(opts.length - 1,
                                             (cur < 0 ? 0 : cur + step)))];
      opts.forEach(b => b.classList.remove("on"));
      next.classList.add("on");
      next.scrollIntoView({block: "nearest"});
      ev.preventDefault();
      return;
    }
    if (ev.key === "Enter" && !menu.hidden) {
      const on = menu.querySelector(".csel-opt.on");
      if (on) on.click();
      ev.preventDefault();
    }
  });

  // 程序赋值（sel.value = ...）后标签要立刻跟上：重定义实例属性
  const desc = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,
                                                "value");
  if (desc && desc.get && desc.set) {
    Object.defineProperty(sel, "value", {
      configurable: true, enumerable: true,
      get() { return desc.get.call(sel); },
      set(v) { desc.set.call(sel, v); sync(); },
    });
  }
  new MutationObserver(sync).observe(sel, {childList: true, subtree: true});
  sync();
}

// 点空白处 / Esc / 滚动 / 改窗口大小 → 关掉所有自绘下拉（全局监听，
// 不是每个 select 一个；菜单挂在 body 上，滚动后 fixed 位置会错，必须关）
function closeAllCsel() {
  document.querySelectorAll(".csel.open").forEach(w => {
    w.classList.remove("open");
    const m = w._cselMenu;
    if (m) m.hidden = true;
  });
}
document.addEventListener("click", closeAllCsel);
document.addEventListener("keydown", ev => {
  if (ev.key === "Escape") closeAllCsel();
});
window.addEventListener("scroll", ev => {
  // 菜单自己滚动（选项多时）不该把它关掉；其它任何容器滚动都会让
  // fixed 坐标失效，必须关。
  const t = ev.target;
  if (t && t.closest && t.closest(".csel-menu")) return;
  closeAllCsel();
}, true);
window.addEventListener("resize", closeAllCsel);

$$("select").forEach(enhanceSelect);
/* 运行时动态插入的 <select>（例如图元编辑器 pdrawForm 用 innerHTML 造的
   `data-f="shape"`）也必须自绘。只增强一次的话，那个下拉会漏回系统原生
   UI —— 这里用 body 级观察器兜住所有后来插入的节点。 */
new MutationObserver(muts => {
  for (const m of muts) for (const node of m.addedNodes) {
    if (node.nodeType !== 1) continue;
    if (node.matches && node.matches("select")) enhanceSelect(node);
    if (node.querySelectorAll) node.querySelectorAll("select").forEach(enhanceSelect);
  }
}).observe(document.body, {childList: true, subtree: true});
fillPluginSelect();
bindPluginImport();
refreshPluginRegistry();            // 异步拉导入插件清单，回来后重绘选择器

// 采样参数滑块：任何改动都实时推到当前插件配置（切换插件后挂载末尾也会补一次）
["infer-temp", "infer-topp"].forEach(id => {
  const elx = $(id);
  if (elx) elx.addEventListener("input", syncSamplingToPlugin);
});
