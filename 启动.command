#!/usr/bin/env bash
# ===========================================================================
# GARGANTUA v1 — 启动.command（项目根目录，中文名）：macOS 双击启动
# ---------------------------------------------------------------------------
# 薄包装：中文名启动入口，直接调用 launcher/start-server.sh
# （环境预检 → 把 server 交接给后台独立进程 → 自己退出）。
#
# 2026-10-29 改：双击启动时，交接成功后**连终端窗口一起关掉**——不再留一个
# 被 server 占着的窗口。关窗前记住本窗口 id，只关这一个（不碰用户别的窗口）；
# 只在本窗口确实是 Terminal.app、且交接成功（rc=0）时才关，失败时保留窗口
# 让用户看到错误。
# ===========================================================================
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

# ---------------------------------------------------------------------------
# 先「关干净」再启动（2026-11-03 起内置；原来的 重启.command 已删除）
# ---------------------------------------------------------------------------
# 窗口进程（service/window/window.py）和 server 是两个独立进程，界面里 ⌘Q
# 只关窗口，server 继续在后台跑（设计如此：关窗口不该打断训练/推理）。于是
# 「重开界面」拿到的还是旧代码的 server——新接口 404、改过的后端不生效。
# 这里在启动前把旧的收掉，并清掉 __pycache__（被移动/改过路径的模块偶尔会
# 拿到旧 pyc）。不删 models/ data/ logs/ environment/ 与 web/index.html。
say() { printf '[启动] %s\n' "$*"; }

if pgrep -f "$DIR/server/server.py" >/dev/null 2>&1 \
   || pgrep -f "$DIR/service/window/window.py" >/dev/null 2>&1; then
  say "发现旧进程，正在结束……"
  for pat in "$DIR/server/server.py" "$DIR/service/window/window.py" \
             "$DIR/service/.*--serve"; do
    pkill -f "$pat" 2>/dev/null || true
  done
  for _ in 1 2 3 4 5 6; do
    sleep 0.2
    pgrep -f "$DIR/server/server.py" >/dev/null 2>&1 || break
  done
  for pat in "$DIR/server/server.py" "$DIR/service/window/window.py" \
             "$DIR/service/.*--serve"; do
    pkill -9 -f "$pat" 2>/dev/null || true
  done
  sleep 0.3
  say "✓ 旧进程已结束"
fi

find "$DIR" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null
find "$DIR" -name "*.pyc" -type f -delete 2>/dev/null
say "✓ 缓存已清"

WIN_ID=""
if [ "${TERM_PROGRAM:-}" = "Apple_Terminal" ]; then
  WIN_ID="$(osascript -e 'tell application "Terminal" to id of front window' 2>/dev/null || true)"
fi

"./launcher/start-server.sh" "$@"
RC=$?

if [ "$RC" -eq 0 ] && [ -n "$WIN_ID" ]; then
  # 等上面那几行输出刷到屏幕上再关（1.5s，够看清地址），关窗动作放后台，
  # 这样本脚本先退出，Terminal 不会弹「要终止正在运行的进程吗」。
  ( sleep 1.5
    osascript -e "tell application \"Terminal\" to close (every window whose id is $WIN_ID)" \
      >/dev/null 2>&1 ) &
fi

exit "$RC"
