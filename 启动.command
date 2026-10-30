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
