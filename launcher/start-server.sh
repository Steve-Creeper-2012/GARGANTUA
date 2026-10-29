#!/usr/bin/env bash
# ===========================================================================
# GARGANTUA v1 — launcher/start-server.sh：POSIX 一键启动（macOS / Linux，由根目录 启动.sh / 启动.command 调用）
# ---------------------------------------------------------------------------
# 纯本地启动，绝不联网做「更新检查」：
#   1. 确认隔离 Python 解释器（environment/venv/bin/python3）在不在；
#      - 在   → 直接启动（无网络、无阻塞）；
#      - 不在 → 调 setup-environment.sh 搭建一次（首次需联网下载 Python 与依赖）。
#   2. 轻量预检：仅窗口模式软查 pywebview（缺失仅警告，HTTP server 仍可用）；
#      torch/rust 等重依赖不在此阻塞（由 setup 装好、server 按需子进程自理）。
#   3. 用隔离 venv 的解释器 exec server/server.py（并设置 GARGANTUA_NO_DEPCHECK=1
#      让 server 跳过自身的主依赖阻塞检查）。窗口拉起、服务调度由 server 负责。
#
# 用法：
#   ./start-server.sh                      正常启动（独立窗口）
#   ./start-server.sh --no-window          仅 HTTP server，不起窗口
#   ./start-server.sh --port 9000          换端口（其余参数透传给 server.py）
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PY="$ROOT/environment/venv/bin/python3"

# ---------------------------------------------------------------------------
# 确保隔离 Python（>= 3.12.14，默认 3.12.15）就绪：
#   - 已存在且达标 → setup-environment.sh 幂等秒回（不联网、不重装）；
#   - 否则用系统包管理器（brew/apt/dnf/apk）或下载兜底装好。
#   全程可见输出，绝不静默卡死。
# ---------------------------------------------------------------------------
echo "[launcher] 确保隔离 Python（>= 3.12.14）就绪……"
"$SCRIPT_DIR/setup-environment.sh"
[ -x "$PY" ] || { echo "[launcher] ✗ setup 未能准备好隔离 Python，请查看上方报错。"; exit 1; }
echo "[launcher] 隔离 Python ✓（$PY）"

# ---------------------------------------------------------------------------
# 轻量预检（不联网、不阻塞）
#   - pywebview：仅「有窗口」模式需要——软检查，缺失仅警告，HTTP server 仍可用。
#   - torch/rust 等重依赖不在此阻塞：由 setup 装好、server 按需子进程自理。
# ---------------------------------------------------------------------------
NEED_WIN=1
for a in "$@"; do
  if [ "$a" = "--no-window" ]; then NEED_WIN=0; fi
done
if [ "$NEED_WIN" = 1 ]; then
  if "$PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('webview') is not None else 1)" 2>/dev/null; then
    echo "[launcher] pywebview ✓"
  else
    echo "[launcher] ⚠ pywebview 未安装：桌面窗口可能起不来，但 HTTP server 仍可用。"
    echo "[launcher]   重装窗口依赖：GARGANTUA_NO_WINDOW=0 launcher/setup-environment.sh"
  fi
fi

# 告知 server：环境就绪，跳过 torch/rust 等重依赖阻塞检查
export GARGANTUA_NO_DEPCHECK=1

exec "$PY" "$ROOT/server/server.py" "$@"
