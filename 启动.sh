#!/usr/bin/env bash
# ===========================================================================
# GARGANTUA v1 — 启动.sh（项目根目录，中文名）：POSIX 一键启动（macOS / Linux）
# ---------------------------------------------------------------------------
# 薄包装：中文名启动入口，直接调用 launcher/start-server.sh
# （含启动前 GitHub Release 检查、环境安装、拉起 server）。
#   用法同 start-server.sh：
#     ./启动.sh               正常启动（独立窗口）
#     ./启动.sh --no-window   仅 HTTP server
#     ./启动.sh --port 9000   换端口（其余参数透传给 server.py）
# ===========================================================================
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/launcher/start-server.sh" "$@"
