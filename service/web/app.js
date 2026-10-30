
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
    if (!r.ok) throw new Error(j.error || ("HTTP " + r.status));
    return j;
  } catch (e) {
    if (e && e.name === "AbortError") {
      throw new Error(`请求超时（${ms}ms）: ${path}`);
    }
    throw e;
  } finally {
    clearTimeout(timer);
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
function modalOpen(title, body, okText, withCancel) {
  return new Promise(res => {
    MODAL_RESOLVE = res;
    $("modal-title").textContent = title;
    $("modal-body").textContent = body;
    $("modal-ok").textContent = okText || "确定";
    $("modal-cancel").style.display = withCancel ? "" : "none";
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
const RADIUS_MIN = 4, RADIUS_MAX = 24, RADIUS_DEF = 16;
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
    const v = Number(localStorage.getItem(RADIUS_KEY));
    return Number.isFinite(v) && v > 0 ? v : RADIUS_DEF;
  } catch (e) { return RADIUS_DEF; }
}
$("set-radius").addEventListener("input", e => {
  const v = applyRadius(e.target.value);
  try { localStorage.setItem(RADIUS_KEY, String(v)); } catch (err) {}
});
applyRadius(savedRadius());

/* ---------- 边栏折叠 ----------
   收起时整块（含 GARGANTUA 标题）向左缩回；展开时向外扩。
   按钮在 GARGANTUA 左边；收起后边栏连同按钮一起滑走，顶栏最左侧补一个
   同样大小的浮动按钮，标题落在它右边。
   状态存 localStorage，首帧由 <head> 内联脚本恢复，不会先闪一下再收。 */
function applySidebar(collapsed) {
  $("sidebar").classList.toggle("collapsed", collapsed);
  document.documentElement.setAttribute("data-side",
                                        collapsed ? "collapsed" : "open");
  $("float-toggle").classList.toggle("show", collapsed);
  $("side-toggle").title = collapsed ? "展开边栏" : "收起边栏";
}
function toggleSidebar() {
  const collapsed = !$("sidebar").classList.contains("collapsed");
  try { localStorage.setItem("gargantua.side", collapsed ? "0" : "1"); } catch (e) {}
  applySidebar(collapsed);
}
$("side-toggle").onclick = toggleSidebar;
$("float-toggle").onclick = toggleSidebar;
applySidebar(document.documentElement.getAttribute("data-side") === "collapsed");

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
    {id:"json", label:"Token 编辑器", icon:"M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"},
  ]},
];
const NAV_BOTTOM = [
  {id:"settings", label:"设置", icon:"M12 15a3 3 0 100-6 3 3 0 000 6zm7.5-3a7.4 7.4 0 00-.1-1.2l2-1.6-2-3.4-2.4 1a7.6 7.6 0 00-2-1.2L14.6 3H9.4l-.4 2.6a7.6 7.6 0 00-2 1.2l-2.4-1-2 3.4 2 1.6a7.4 7.4 0 000 2.4l-2 1.6 2 3.4 2.4-1a7.6 7.6 0 002 1.2l.4 2.6h5.2l.4-2.6a7.6 7.6 0 002-1.2l2.4 1 2-3.4-2-1.6c.06-.4.1-.8.1-1.2z"},
];
// 页面标题只此一处：顶行那个 h1。切页时改文字，不再每页各写一个 h1。
const PAGE_TITLES = {
  home: "主页", models: "模型", vocab: "词表", train: "训练", infer: "推理",
  text: "文本", audio: "音频", image: "图片", video: "视频",
  prims: "图元绘画", json: "Token 编辑器", settings: "设置",
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
  NAV_INDICATOR.style.transform =
    `translate(${(r.left - hr.left).toFixed(2)}px, ${(r.top - hr.top).toFixed(2)}px)`;
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
      [{opacity: 0, transform: "translateY(8px)"},
       {opacity: 1, transform: "none"}],
      {duration: 220, easing: "cubic-bezier(.2, .7, .3, 1)"});
  }
  if (id === "models") refreshModels();
  if (id === "train") refreshQueue();
  if (id === "vocab") refreshVocab();
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
  } catch (e) {}
}

/* =========================================================================
   主页：环境与依赖
   -------------------------------------------------------------------------
   一张卡 = 一行两列（等宽）。左侧两行字：标题（加粗）+ 版本标注 / 小字解释
   （未安装写「未发现」）；右侧一个按钮（安装 / 更新）或一句状态文字。
   ========================================================================= */
const DEP_META = {
  torch:       {display: "PyTorch",     desc: "深度学习框架（训练与推理）"},
  numpy:       {display: "NumPy",       desc: "数值计算库"},
  PIL:         {display: "Pillow",      desc: "图像处理库"},
  safetensors: {display: "Safetensors", desc: "模型权重存储格式"},
  cv2:         {display: "OpenCV",      desc: "图像/视频解码加速（可选，缺失时用 Pillow + ffmpeg）"},
  soundfile:   {display: "SoundFile",   desc: "音频解码加速（可选，缺失时用 ffmpeg）"},
};
let SYSTEM_CACHE = null, UPDATES_CACHE = null, DEVICE_CACHE = null;

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
  if (force) msg.textContent = "检查中…";
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
    // 更新检查是网络活（镜像源），异步补上后再重渲染按钮
    api("/api/system/updates" + q, undefined, {timeout: 60000})
      .then(u => { UPDATES_CACHE = u.updates || []; renderDeps(); })
      .catch(() => { UPDATES_CACHE = null; renderDeps(); });
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
        : `<span class="dev-s warn">${esc(g.note || (g.model + " 不支持 torch"))}</span>`;
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
  const um = {};
  (UPDATES_CACHE || []).forEach(u => { um[u.name] = u; });
  grid.innerHTML = (s.dependencies || []).map(d => {
    const meta = DEP_META[d.name] || {display: d.name, desc: d.why || ""};
    const line2 = d.ok ? meta.desc : "未发现";
    let right;
    if (!d.ok) {
      right = `<button class="act" data-install="${esc(d.name)}" data-upgrade="0">安装</button>`;
    } else {
      const u = um[d.name];
      if (!u) right = `<span class="dep-state">检查更新中…</span>`;
      else if (u.update_available)
        right = `<button class="act" data-install="${esc(d.name)}" data-upgrade="1"
                         title="更新到 ${esc(u.latest || "")}">更新</button>`;
      else if (!u.checked) right = `<span class="dep-state">未能检查更新</span>`;
      else right = `<span class="dep-state ok">当前已是最新版本</span>`;
    }
    return `<div class="dep-card"><div class="dep-info">
      <div class="dep-title"><span class="dep-name" title="${esc(meta.display)}">${esc(meta.display)}</span>${
        d.ok && d.version ? `<span class="dep-ver">${esc(d.version)}</span>` : ""}</div>
      <div class="dep-desc" title="${esc(line2)}">${esc(line2)}</div>
    </div>${right}</div>`;
  }).join("");
  // 事件委托：依赖名可能带引号，不拼进 onclick 字符串
  grid.querySelectorAll("button[data-install]").forEach(b => {
    b.onclick = () => installDep(b.dataset.install, b.dataset.upgrade === "1");
  });
}

async function installDep(name, upgrade) {
  const msg = $("env-msg");
  msg.textContent = (upgrade ? "正在更新 " : "正在安装 ") + name + " …";
  try {
    const {task_id} = await api("/api/system/install",
                                {names: [name], upgrade: !!upgrade});
    const t = await pollTask(task_id, {interval: 2000, timeout: 1800000});
    msg.textContent = (t.result && t.result.note) || "完成";
    toast((upgrade ? "已更新 " : "已安装 ") + name);
    await refreshEnv(true);
  } catch (e) {
    msg.textContent = "错误: " + e.message;
    toast("安装失败: " + e.message, true);
  }
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
  $("env-msg").textContent = "";
  UPDATES_CACHE = null;
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
let PROC_CACHE = [];
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
function renderProcesses(procs) {
  PROC_CACHE = procs || [];
  syncProcHeaders();
  const tb = $("proc-tbody");
  $("proc-note").textContent = PROC_CACHE.length
    ? `${PROC_CACHE.length} 个进程　·　内存 / 显存列点击可切换「百分比 ↔ 实际占用」`
    : "内存 / 显存列点击可在「百分比 ↔ 实际占用」间切换";
  if (!PROC_CACHE.length) {
    tb.innerHTML = `<tr><td colspan="6" class="hint">没有正在运行的 GARGANTUA 进程</td></tr>`;
    return;
  }
  tb.innerHTML = PROC_CACHE.map(p => `<tr>
    <td>${esc(p.name)}</td>
    <td class="mono">${p.pid}</td>
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
    renderProcesses(p.processes);
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
/* token-5 文本（spec 第 5b 节）：{text: NNNNN} / {wave: HEX} / {visual: HEX x: 1 y: 2}
   音频/视觉的 hex 载荷一律由后端生成（r.token5），前端只在没有后端文本时兜底。 */
const _ID5 = n => String(n).padStart(5, "0");

function stepsToToken5(steps, bound) {
  const openTag = bound === "user" ? "<00001>" : "<00003>";   // <user> / <model>
  const closeTag = bound === "user" ? "<00002>" : "<00004>";  // <user/> / <model/>
  const parts = [openTag];
  for (const st of steps) {
    const ids = st.token_ids || [];
    if (!ids.length) continue;                 // 模态步：由后端 token5 给出
    if (st.modality === "draw") {
      // 绘图：一个图元 = 9 个 token = 一个时间步
      for (let j = 0; j + 9 <= ids.length; j += 9)
        parts.push("{" + ids.slice(j, j + 9).map(t => "text: " + _ID5(t)).join(" | ") + "}");
      continue;
    }
    for (const id of ids) parts.push("{text: " + _ID5(id) + "}");
  }
  parts.push(closeTag);
  return parts.join("");
}

/* ---------- 模型 ---------- */
async function refreshModels() {
  try {
    const r = await api("/api/models");
    $("model-tbody").innerHTML = r.models.map(m =>
      `<tr><td>${esc(m.name)} ${m.name === r.active ? '<span class="badge on">当前</span>' : ""}</td>
       <td>${m.params ? (m.params/1e6).toFixed(1)+"M" : "—"}</td>
       <td class="right"><button class="act sec xs" data-load="${esc(m.name)}">加载</button></td></tr>`).join("");
    $("model-tbody").querySelectorAll("[data-load]").forEach(b => {
      b.onclick = () => loadModel(b.dataset.load);
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
  // 空值/非数字不下发（服务端取 spec 默认）。
  // 注意不能用 `parseInt(v) || undefined`——0 对压缩上下文是合法值
  // （表示不压缩），会被静默吞成默认 3072
  const intOrUndef = (v) => {
    const n = parseInt(v, 10);
    return Number.isFinite(n) ? n : undefined;
  };
  const {task_id} = await api("/api/model/init", {
    name, overwrite,
    n_enc_blocks: intOrUndef($("model-enc").value),
    n_dec_blocks: intOrUndef($("model-dec").value),
    ctx_full: intOrUndef($("model-ctx-full").value),
    ctx_compress: intOrUndef($("model-ctx-compress").value),
    csa_ratio: intOrUndef($("model-csa-ratio").value),
    hca_ratio: intOrUndef($("model-hca-ratio").value),
    vocab: $("model-vocab").value || "base",
    dtype: $("model-dtype").value || "fp32",
  });
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

/* ---------- 推理（类 Kimi 聊天） ---------- */
const chatHistory = $("chat-history");
let lastMedia = null, mediaObjUrl = null;
function hideEmpty() { const e = $("chat-empty"); if (e) e.style.display = "none"; }
function addMsg(role, html) {
  hideEmpty();
  const row = document.createElement("div"); row.className = "msg-row";
  const d = document.createElement("div"); d.className = "msg " + role;
  d.innerHTML = html; row.appendChild(d);
  chatHistory.appendChild(row); chatHistory.scrollTop = chatHistory.scrollHeight;
}
$("infer-media").onchange = () => {
  const f = $("infer-media").files[0]; if (!f) return;
  lastMedia = f;
  if (mediaObjUrl) URL.revokeObjectURL(mediaObjUrl);
  mediaObjUrl = fileToUrl(f);
  let tag = "";
  if (f.type.startsWith("image/")) tag = `<img src="${mediaObjUrl}">`;
  else if (f.type.startsWith("video/")) tag = `<video src="${mediaObjUrl}" controls></video>`;
  else tag = `<div class="hint">已选择音频/文件：${esc(f.name)}</div><audio src="${mediaObjUrl}" controls></audio>`;
  const ap = $("attach-preview"); ap.innerHTML = tag; ap.style.display = "block";
};
$("send-btn").onclick = async () => {
  const st = $("infer-status");
  const prompt = $("infer-prompt").value;
  if (!prompt.trim() && !lastMedia) return;
  let userHtml = esc(prompt);
  if (lastMedia) {
    let tag = "";
    if (lastMedia.type.startsWith("image/")) tag = `<img src="${mediaObjUrl}">`;
    else if (lastMedia.type.startsWith("video/")) tag = `<video src="${mediaObjUrl}" controls></video>`;
    else tag = `<audio src="${mediaObjUrl}" controls></audio>`;
    userHtml += `<div class="media">${tag}</div>`;
  }
  addMsg("user", userHtml);
  $("infer-prompt").value = "";
  st.textContent = "思考中…";
  $("send-btn").disabled = true;
  try {
    // 媒体已接入生成路径——作为 prompt 前缀参与编码
    // （视觉/音频各走自己的线性投影到 d_model），生成仍是文本。
    const body = {prompt,
                  max_new_tokens: clampNum($("infer-maxnew").value, 1, 512, 32),
                  temperature: clampNum($("infer-temp").value, 0.05, 5, 0.8),
                  top_p: clampNum($("infer-topp").value, 0.01, 1, 0.95)};
    if (lastMedia) {
      const MAX_MEDIA_BYTES = 64 * 1024 * 1024;
      if (lastMedia.size > MAX_MEDIA_BYTES) {
        throw new Error(`附件 ${(lastMedia.size / 1048576).toFixed(1)}MB `
                      + `超过上限 ${MAX_MEDIA_BYTES / 1048576}MB，未发送`);
      }
      const kind = lastMedia.type.startsWith("image/") ? "image"
                 : lastMedia.type.startsWith("video/") ? "video" : "audio";
      body.media = [{kind, b64: await fileToB64(lastMedia),
                     filename: lastMedia.name}];
    }
    const {task_id} = await api("/api/infer", body);
    const t = await pollTask(task_id, {
      interval: 1500, timeout: 600000,
      onTick: x => { st.textContent = "思考中… " + x.status; }});
    addMsg("model", esc(t.result.text || "（无文本输出）"));
    st.textContent = t.result.archive ? "已存档 id" : "";
    st.title = t.result.archive || "";
    $("attach-preview").style.display = "none";
    $("attach-preview").innerHTML = "";
    $("infer-media").value = "";
    lastMedia = null;
    return;
  } catch (e) { st.textContent = "错误: " + e.message; addMsg("model", "⚠ " + esc(e.message)); }
  finally { $("send-btn").disabled = false; }
};
$("infer-prompt").addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("send-btn").click(); }
});
$("debug-panel").querySelector(".toggle").onclick =
  () => $("debug-panel").classList.toggle("collapsed");

/* ---------- 编解码器 ---------- */
$("text-encode").onclick = async () => {
  try {
    const r = await api("/api/codec/text", {text: $("text-in").value});
    $("text-out").textContent = r.token5;
  } catch (e) { $("text-out").textContent = "错误: " + e.message; }
};
$("audio-encode").onclick = async () => {
  const f = $("audio-file").files[0]; if (!f) return;
  $("audio-preview").innerHTML = `<audio src="${fileToUrl(f)}" controls></audio>`;
  $("audio-out").textContent = "编码中…";
  try {
    const r = await api("/api/codec/audio", {audio_b64: await fileToB64(f),
                             filename: f.name},
                        {timeout: API_UPLOAD_TIMEOUT_MS});
    $("audio-out").textContent =
      `# 音元数 ${r.n_tokens}，时长 ${r.duration_s}s\n` + r.token5;
  } catch (e) { $("audio-out").textContent = "错误: " + e.message; }
};
$("image-encode").onclick = async () => {
  const f = $("image-file").files[0]; if (!f) return;
  $("image-preview").src = fileToUrl(f);
  $("image-out").textContent = "编码中…";
  try {
    const r = await api("/api/codec/image", {image_b64: await fileToB64(f)},
                          {timeout: API_UPLOAD_TIMEOUT_MS});
    const st = r.steps[0];
    $("image-out").textContent =
      `# grid ${st.meta.grid[0]}×${st.meta.grid[1]}，patch 数 ${st.meta.n_patches}\n` +
      r.token5;
  } catch (e) { $("image-out").textContent = "错误: " + e.message; }
};
$("video-encode").onclick = async () => {
  const f = $("video-file").files[0]; if (!f) return;
  const vp = $("video-preview"); vp.src = fileToUrl(f); vp.style.display = "block";
  $("video-out").textContent = "编码中（ffmpeg 抽帧 + 提取音轨）…";
  try {
    const r = await api("/api/codec/video", {video_b64: await fileToB64(f),
                             filename: f.name},
                        {timeout: API_UPLOAD_TIMEOUT_MS});
    $("video-stats").textContent =
      `帧数: ${r.n_frames}　总 patches: ${r.total_patches}　音元: ${r.n_audio_tokens}`;
    $("video-out").textContent = r.token5;
  } catch (e) { $("video-out").textContent = "错误: " + e.message;
                $("video-stats").textContent = ""; }
};
$("prims-convert").onclick = async () => {
  const f = $("prims-file").files[0]; if (!f) return;
  $("prims-src").src = fileToUrl(f);
  $("prims-msg").textContent = "转换中（贪心图元拟合，后台任务）…";
  $("prims-render").innerHTML = ""; $("prims-out").textContent = "—";
  try {
    const {task_id} = await api("/api/codec/image2prims",
                                {image_b64: await fileToB64(f),
                                 max_steps: clampNum($("prims-steps").value, 1, 1000, 100),
                                 quality: $("prims-quality").value || "fast"},
                                {timeout: API_UPLOAD_TIMEOUT_MS});
    const t = await pollTask(task_id, {
      interval: 1200, timeout: 120000,
      onTick: x => { $("prims-msg").textContent = "转换中… " + x.status; }});
    const r = t.result;
    $("prims-render").innerHTML = r.svg;
    $("prims-out").textContent =
      `# ${r.note}\n# 图元 ${r.n_primitives} 个，token ${r.tokens.length} 个（9/图元）\n` +
      stepsToToken5([{modality: "draw", token_ids: r.tokens, meta: {}}], "model");
    $("prims-msg").textContent = r.n_primitives + " 个图元";
  } catch (e) { $("prims-msg").textContent = "错误: " + e.message; }
};

/* ---------- Token 编辑器 ----------
   调色板：扩展系统 token（<00000>..<00005>，也可直接写名字）+ 结构字符。 */
const TOKEN_PALETTE = [
  {label:"<user>",    insert:"<00001>"},
  {label:"<user/>",   insert:"<00002>"},
  {label:"<model>",   insert:"<00003>"},
  {label:"<model/>",  insert:"<00004>"},
  {label:"<stop>",    insert:"<00005>"},
  {label:"<caps>",    insert:"<00000>"},
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
    const r = await api("/api/tokens/export",
                        {text: $("json-editor").innerText});
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

/* ---------- 词表搜索（/api/vocab，GET ?q=） ---------- */
async function vocabSearch() {
  const q = $("vocab-q").value.trim();
  const msg = $("vocab-msg"), hits = $("vocab-hits");
  if (!q) { msg.textContent = ""; hits.textContent = ""; return; }
  msg.textContent = "搜索中…";
  try {
    const r = await api("/api/vocab?q=" + encodeURIComponent(q));
    msg.textContent = `${r.hits.length} 项命中（词表共 ${r.vocab_size}）`;
    hits.innerHTML = r.hits.length
      ? r.hits.map(h => `<code>${esc(h.token)}</code><span class="hint">#${h.id} ${esc(h.category || "")}</span>`).join("　")
      : '<span class="hint">无命中</span>';
  } catch (e) { msg.textContent = "错误: " + e.message; hits.textContent = ""; }
}
$("vocab-search").onclick = vocabSearch;
$("vocab-q").addEventListener("keydown", e => {
  if (e.key === "Enter") vocabSearch();
});

/* ---------- 添加 token（/api/vocab/add，POST） ---------- */
$("vocab-add").onclick = async () => {
  const token = $("vocab-add-token").value;
  const category = $("vocab-add-cat").value;
  const msg = $("vocab-add-msg");
  if (!token) { msg.textContent = "token 文本不能为空"; return; }
  msg.textContent = "添加中…";
  try {
    const r = await api("/api/vocab/add", {token, category});
    const fs = Object.entries(r.free_slots || {})
      .map(([k, v]) => `${k}=${v}`).join(" ");
    msg.textContent = `已添加 ${r.token} → #${r.id}（${r.category}）；` +
      (fs ? `剩余槽位 ${fs}` : "已写入词表");
    $("vocab-add-token").value = "";
    vocabSearch();
  } catch (e) { msg.textContent = "错误: " + e.message; }
};
$("vocab-add-token").addEventListener("keydown", e => {
  if (e.key === "Enter") $("vocab-add").click();
});

/* ---------- 词表页面（区间占用 + 预设管理） ---------- */
function catColor(cat) {
  return {regular:"#4c8dff", fallback:"#a78bfa",
          special:"#f0b429", system:"#3ddc97"}[cat] || "#6b7280";
}
async function refreshVocab() {
  try {
    const r = await api("/api/vocab/ranges");
    drawVocabBar(r);
    renderVocabRegions(r.regions);
    renderVocabPresets(r.presets);
    populateVocabSelects(r.presets);
    const ct = r.category_totals || {};
    $("vocab-bar-note").textContent =
      `总槽位 ${r.vocab_size_padded}　已用 ` +
      `regular=${ct.regular||0} fallback=${ct.fallback||0} ` +
      `special=${ct.special||0} system=${ct.system||0}`;
  } catch (e) {}
}
function drawVocabBar(r) {
  const cv = $("vocab-bar"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const total = r.vocab_size_padded;
  let x = 0;
  // 不可见头 [0, invisible_end)：special/system 固定契约，画一段灰底
  const headW = r.invisible_end / total * W;
  ctx.fillStyle = getComputedStyle(document.documentElement)
    .getPropertyValue("--track").trim() || "#242b35";
  ctx.fillRect(0, 0, headW, H);
  x = headW;
  for (const reg of r.regions) {
    const w = reg.size / total * W;
    ctx.fillStyle = catColor(reg.category);
    ctx.fillRect(x, 0, Math.max(w, 0.5), H);
    x += w;
  }
}
function renderVocabRegions(regions) {
  const tb = $("vocab-regions").querySelector("tbody");
  tb.innerHTML = regions.map(rg => {
    const pct = rg.padded ? (rg.used / rg.padded * 100).toFixed(2) : "0.00";
    return `<tr><td>${esc(rg.region)}</td><td>${esc(rg.category)}</td>` +
      `<td>${rg.begin}–${rg.begin + rg.size - 1}</td>` +
      `<td>${rg.size}</td><td>${rg.used}</td><td>${pct}%</td></tr>`;
  }).join("");
}
function renderVocabPresets(presets) {
  const tb = $("vocab-presets").querySelector("tbody");
  tb.innerHTML = presets.map(p => {
    const pct = p.capacity ? (p.used / p.capacity * 100).toFixed(2) : "0.00";
    const del = p.is_base ? "" :
      `<button class="act danger xs" data-delpreset="${esc(p.name)}">删除</button>`;
    return `<tr><td>${esc(p.name)} ` +
      `${p.is_base ? '<span class="badge on">基础</span>' : ""}</td>` +
      `<td>${p.used}/${p.capacity}（${pct}%）</td><td class="right">${del}</td></tr>`;
  }).join("");
  tb.querySelectorAll("[data-delpreset]").forEach(b => {
    b.onclick = () => deletePreset(b.dataset.delpreset);
  });
}
function populateVocabSelects(presets) {
  // 重建下拉前记住当前选择并回填，避免每次 refreshVocab 都弹回第一项
  const opts = presets.map(p =>
    `<option value="${esc(p.name)}">${esc(p.name)}</option>`).join("");
  for (const id of ["vocab-new-source", "vocab-token-preset", "model-vocab"]) {
    const el = $(id);
    if (!el) continue;
    const prev = el.value;
    el.innerHTML = opts;
    if (prev && presets.some(p => p.name === prev)) el.value = prev;
  }
}
$("vocab-create-preset").onclick = async () => {
  const name = $("vocab-new-name").value.trim();
  if (!name) { $("vocab-create-msg").textContent = "请输入预设名"; return; }
  $("vocab-create-msg").textContent = "创建中…";
  try {
    await api("/api/vocab/presets", {action:"create", name,
                                    source: $("vocab-new-source").value});
    $("vocab-create-msg").textContent = "已创建 " + name;
    $("vocab-new-name").value = "";
    refreshVocab();
  } catch (e) { $("vocab-create-msg").textContent = "错误: " + e.message; }
};
async function deletePreset(name) {
  if (!await uiConfirm("删除词表预设 " + name + "？（其 vocab.txt 将被移除）")) return;
  try { await api("/api/vocab/presets", {action:"delete", name});
        refreshVocab(); }
  catch (e) { toast(e.message, true); }
}
$("preset-add").onclick = async () => {
  const name = $("vocab-token-preset").value;
  const token = $("preset-add-token").value;
  if (!token) { $("preset-token-msg").textContent = "请输入 token"; return; }
  $("preset-token-msg").textContent = "添加中…";
  try {
    const r = await api("/api/vocab/preset/token",
      {name, action:"add", token, category: $("preset-add-cat").value});
    $("preset-token-msg").textContent =
      `已添加 ${r.token} → #${r.id}（${r.category}）；剩余可见槽 ${r.used}`;
    $("preset-add-token").value = "";
    refreshVocab();
  } catch (e) { $("preset-token-msg").textContent = "错误: " + e.message; }
};
$("preset-remove").onclick = async () => {
  const name = $("vocab-token-preset").value;
  const token = $("preset-remove-token").value;
  if (!token) { $("preset-token-msg").textContent = "请输入 token"; return; }
  $("preset-token-msg").textContent = "删除中…";
  try {
    const r = await api("/api/vocab/preset/token",
      {name, action:"remove", token});
    $("preset-token-msg").textContent = `已软删 ${r.token}（原 #${r.id}）`;
    $("preset-remove-token").value = "";
    refreshVocab();
  } catch (e) { $("preset-token-msg").textContent = "错误: " + e.message; }
};

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
const TOOL_PAGES = ["text", "audio", "image", "video", "prims", "json"];
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
