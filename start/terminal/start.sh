#!/usr/bin/env bash
# ===========================================================================
# GARGANTUA v1 — start/terminal/start.sh：POSIX 一键启动（macOS / Linux）
# ---------------------------------------------------------------------------
# 1. 调 setup.sh 确保隔离 Python + 依赖（幂等）；
# 2. 用隔离 venv 的解释器拉起 start/launcher.py（环境预检 + 窗口完整性
#    校验 + 拉起 server/window）。
#
# 用法：
#   ./start.sh                      正常启动（带独立窗口 + 完整性弹窗）
#   ./start.sh --no-window          仅 HTTP server，不起窗口
#   ./start.sh --no-verify          跳过窗口完整性弹窗（CI / 无 GUI）
# 其余参数透传给 launcher.py（--port / --host / --model-dir 等）。
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 第一棒：装环境（失败则退出，不继续拉起）
"$SCRIPT_DIR/setup.sh"

exec "$ROOT/start/.venv/bin/python3" "$ROOT/start/launcher.py" "$@"
