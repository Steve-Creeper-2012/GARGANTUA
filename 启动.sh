#!/usr/bin/env bash
# ===========================================================================
# GARGANTUA v1 — 启动.sh（项目根目录，中文名）：POSIX 一键启动（macOS / Linux）
# ---------------------------------------------------------------------------
# 薄包装：中文名启动入口，直接调用 launcher/start-server.sh
# （环境预检 → 把 server 交接给后台独立进程 → 自己退出）。
#
# 用法同 start-server.sh：
#   ./启动.sh                交接给后台 server（独立窗口），终端随即释放
#   ./启动.sh --no-window    仅 HTTP server
#   ./启动.sh --port 9000    换端口（其余参数透传给 server.py）
#   ./启动.sh --foreground   前台运行（占住终端，调试用）
#   ./启动.sh --stop         停止后台 server
# ===========================================================================
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$DIR/launcher/start-server.sh" "$@"
