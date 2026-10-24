#!/usr/bin/env bash
# ===========================================================================
# GARGANTUA v1 — start/terminal/start.sh：POSIX 一键启动（macOS / Linux）
# ---------------------------------------------------------------------------
# 1. 调 setup.sh 确保隔离 Python 3.12.x + venv + 依赖（幂等，秒回）；
# 2. 用隔离 venv 的解释器 exec server/server.py。
#    环境检查、window.py 完整性校验（sha256 + 弹窗）、窗口拉起、
#    服务按需调度全部由 server 负责（见 server/server_env.py、
#    server/server_services.py）。
#
# 用法：
#   ./start.sh                      正常启动（独立窗口 + 完整性弹窗）
#   ./start.sh --no-window          仅 HTTP server，不起窗口
#   ./start.sh --no-verify          跳过窗口完整性弹窗（CI / 无 GUI）
#   ./start.sh --port 9000          换端口（其余参数透传给 server.py）
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 第一棒：装环境（失败则退出，不继续拉起）
"$SCRIPT_DIR/setup.sh"

exec "$ROOT/start/.venv/bin/python3" "$ROOT/server/server.py" "$@"
