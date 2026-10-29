#!/usr/bin/env bash
# ===========================================================================
# GARGANTUA v1 — launcher/start-server.sh：POSIX 一键启动（macOS / Linux，由根目录 启动.sh / 启动.command 调用）
# ---------------------------------------------------------------------------
# 只做两件事：检查启动核心组件 → 立刻启动 server。
# 不联网做更新检查、不碰 git/brew（除非真的要装一次环境）。
#
#   1. Python：隔离环境 environment/venv/bin/python3（3.12 系列）。
#      没有 → 调 setup-environment.sh 一次性搭好（优先复用本机现成的 3.12，
#      不下载、不触发 brew 的 git auto-update）。
#   2. pywebview：仅窗口模式需要，缺失只警告（HTTP server 照常可用）。
#      torch/rust 等重依赖不在启动链阻塞。
#   3. exec server/server.py（设 GARGANTUA_NO_DEPCHECK=1，让 server 跳过
#      自身的主依赖阻塞检查）。窗口拉起与服务调度由 server 负责。
#
# 用法：
#   ./start-server.sh                      正常启动（独立窗口）
#   ./start-server.sh --no-window          仅 HTTP server，不起窗口
#   ./start-server.sh --port 9000          换端口（其余参数透传给 server.py）
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_PY="$ROOT/environment/venv/bin/python3"

# ---------------------------------------------------------------------------
# 核心组件 1：Python（隔离环境解释器）
# ---------------------------------------------------------------------------
if [ ! -x "$VENV_PY" ]; then
  echo "[launcher] 未找到隔离 Python → 一次性搭建（复用本机现成的 3.12，不下载）……"
  "$SCRIPT_DIR/setup-environment.sh"
fi
[ -x "$VENV_PY" ] || { echo "[launcher] ✗ 隔离 Python 不可用：$VENV_PY"; exit 1; }
echo "[launcher] Python ✓（$("$VENV_PY" -V 2>&1)）"

# ---------------------------------------------------------------------------
# 核心组件 2：pywebview（仅窗口模式需要；缺失不阻塞，HTTP server 仍可用）
# ---------------------------------------------------------------------------
NEED_WIN=1
for a in "$@"; do
  if [ "$a" = "--no-window" ]; then NEED_WIN=0; fi
done
if [ "$NEED_WIN" = 1 ]; then
  if "$VENV_PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('webview') is not None else 1)" 2>/dev/null; then
    echo "[launcher] pywebview ✓"
  else
    echo "[launcher] ⚠ pywebview 未安装：桌面窗口可能起不来，HTTP server 仍可用。"
    echo "[launcher]   补装：$VENV_PY -m pip install pywebview"
  fi
fi

# 告知 server：环境就绪，跳过 torch/rust 等重依赖阻塞检查
export GARGANTUA_NO_DEPCHECK=1

exec "$VENV_PY" "$ROOT/server/server.py" "$@"
