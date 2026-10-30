#!/usr/bin/env bash
# ===========================================================================
# GARGANTUA v1 — launcher/start-server.sh：POSIX 一键启动（macOS / Linux，
# 由根目录 启动.sh / 启动.command 调用）
# ---------------------------------------------------------------------------
# 职责只有两件：检查启动核心组件 → 把控制权交给 server，然后自己退出。
# 不联网做更新检查、不碰 git/brew（除非真的要装一次环境）。
#
#   1. Python：隔离环境 environment/venv/bin/python3（3.12 系列）。
#      没有 → 调 setup-environment.sh 一次性搭好（优先复用本机现成的 3.12，
#      不下载、不触发 brew 的 git auto-update）。
#   2. pywebview：仅窗口模式需要，缺失只警告（HTTP server 照常可用）。
#      torch/rust 等重依赖不在启动链阻塞。
#   3. 交接（2026-10-29 改）：默认把 server **放到后台独立进程**里跑
#      （nohup + disown，stdout 进 logs/server.log，PID 写 logs/server.pid），
#      等到 HTTP 就绪后打印地址就退出——终端窗口不再被 server 占着，
#      关掉终端也不会带走 server。
#      `--foreground` 保留老行为（exec 顶替当前进程），调试 / systemd / 容器用。
#   4. `--stop`：读 logs/server.pid，SIGTERM 收尾。
#
# 用法：
#   ./start-server.sh                      交接给后台 server（独立窗口），随后退出
#   ./start-server.sh --no-window          仅 HTTP server，不起窗口
#   ./start-server.sh --port 9000          换端口（其余参数透传给 server.py）
#   ./start-server.sh --foreground         前台运行（exec，不交接）
#   ./start-server.sh --stop               停止后台 server
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_PY="$ROOT/environment/venv/bin/python3"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/server.log"
PID_FILE="$LOG_DIR/server.pid"

# ---------------------------------------------------------------------------
# 参数解析：--stop / --foreground 是 launcher 自己的，其余原样透传 server.py
# （macOS 自带 bash 3.2：空数组在 set -u 下会炸，统一用 ${A[@]+"${A[@]}"} 展开）
# ---------------------------------------------------------------------------
STOP=0
FOREGROUND=0
NEED_WIN=1
PORT=8712
HOST=127.0.0.1
ARGS=()

add_args() { for a in "$@"; do ARGS[${#ARGS[@]}]="$a"; done; }

while [ $# -gt 0 ]; do
  case "$1" in
    --stop)          STOP=1; shift ;;
    --foreground|-F) FOREGROUND=1; shift ;;
    --no-window)     NEED_WIN=0; add_args "$1"; shift ;;
    --port)          PORT="${2:-8712}";       add_args "--port" "${2:-}"; shift 2 ;;
    --port=*)        PORT="${1#*=}";         add_args "$1"; shift ;;
    --host)          HOST="${2:-127.0.0.1}"; add_args "--host" "${2:-}"; shift 2 ;;
    --host=*)        HOST="${1#*=}";         add_args "$1"; shift ;;
    *)               add_args "$1"; shift ;;
  esac
done

# 审计 M6：PORT/HOST 会直接拼进 lsof -iTCP:$PORT、pgrep -f 模式与 URL，
# 非数字端口或含 shell 元字符的 host 会让模式匹配错乱（甚至误杀无关进程）。
case "$PORT" in
  ''|*[!0-9]*)
    echo "[launcher] ✗ --port 必须是 1-65535 的整数：${PORT}"; exit 1 ;;
esac
if [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
  echo "[launcher] ✗ --port 超出范围（1-65535）：${PORT}"; exit 1
fi
HOST_RE='^[A-Za-z0-9.:-]+$|^\[[A-Za-z0-9:]+\]$'
if [[ ! "$HOST" =~ $HOST_RE ]]; then
  echo "[launcher] ✗ --host 含非法字符：${HOST}"; exit 1
fi

URL="http://$HOST:$PORT"

# ---------------------------------------------------------------------------
# 探活：优先 curl，没有就用隔离解释器打一次
# ---------------------------------------------------------------------------
probe() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsS -m 1 "$URL/api/status" >/dev/null 2>&1
  elif [ -x "$VENV_PY" ]; then
    "$VENV_PY" - "$URL" <<'PY' >/dev/null 2>&1
import sys, urllib.request
urllib.request.urlopen(sys.argv[1] + "/api/status", timeout=1).read()
PY
  else
    return 1
  fi
}

# 进程还在吗（-0 只做存在性/权限检查，不发信号）
alive() { [ -n "${1:-}" ] && kill -0 "$1" 2>/dev/null; }

# ---------------------------------------------------------------------------
# 把进程彻底从「启动它的那个终端」上摘下来。
#   nohup 只让 SIG_IGN 覆盖 SIGHUP；终端窗口一关，会话销毁仍会波及同一会话
#   里的进程（2026-10-29 真实事故：窗口进程随之消失）。macOS 没有 setsid
#   命令，借隔离解释器做一次 os.setsid() + execv，pid 不变。
# 结果写进全局 DETACHED_PID（不用 $(...) 取，避免后台作业挂在子 shell 上）。
# ---------------------------------------------------------------------------
DETACHED_PID=""
detach_run() {
  nohup "$VENV_PY" -c 'import os,sys
try: os.setsid()
except OSError: pass
os.execv(sys.argv[1], sys.argv[1:])' \
    "$@" </dev/null >>"$LOG_FILE" 2>&1 &
  DETACHED_PID=$!
  disown "$DETACHED_PID" 2>/dev/null || true
}

# 这个端口上的监听者 pid（pidfile 丢了也能停：pidfile 在 logs/ 里，
# logs/ 被清理工具扔进废纸篓时 pidfile 就跟着没了）
port_pid() {
  command -v lsof >/dev/null 2>&1 || return 1
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -n 1
}

# 指定端口上是否已经有窗口进程
window_running() {
  command -v pgrep >/dev/null 2>&1 || return 1
  pgrep -f "service/window/window.py --port $PORT --host" >/dev/null 2>&1
}

# 收掉这个端口的窗口（server 收尸只认自己的子进程；launcher 补开的它不认）
kill_window() {
  if window_running; then
    pkill -f "service/window/window.py --port $PORT --host" 2>/dev/null || true
    echo "[launcher]   已关闭窗口"
  fi
}

# 给已在运行的 server 补一个窗口（server 自己只在启动时拉窗口一次）
open_window() {
  [ -x "$VENV_PY" ] || return 1
  [ -f "$ROOT/service/window/window.py" ] || return 1
  mkdir -p "$LOG_DIR"
  detach_run "$VENV_PY" "$ROOT/service/window/window.py" \
    --port "$PORT" --host "$HOST"
  return 0
}

# ---------------------------------------------------------------------------
# --stop：停掉后台 server
# ---------------------------------------------------------------------------
if [ "$STOP" = 1 ]; then
  PID=""
  if [ -f "$PID_FILE" ]; then
    PID="$(cat "$PID_FILE" 2>/dev/null || true)"
    if ! alive "$PID"; then
      echo "[launcher] pidfile 里的 PID ${PID:-?} 已不在，改按端口 $PORT 找。"
      PID=""
    fi
  fi
  if [ -z "$PID" ]; then
    PID="$(port_pid || true)"
    if [ -z "$PID" ]; then
      echo "[launcher] 端口 $PORT 上没有 server 在跑。"
      kill_window
      rm -f "$PID_FILE"
      exit 0
    fi
    echo "[launcher] 没有可用 pidfile，按端口 ${PORT} 找到 PID ${PID}。"
  fi
  # 尽力确认这个 pid 确实是本项目的 server（防止 pid 被复用后误杀别人）
  CMD="$(ps -p "$PID" -o command= 2>/dev/null || true)"
  case "$CMD" in
    *server/server.py*|"") : ;;
    *) echo "[launcher] PID $PID 不是 GARGANTUA server（${CMD}），拒绝终止。"
       rm -f "$PID_FILE"; exit 1 ;;
  esac
  echo "[launcher] 停止 server（PID ${PID}）……"
  kill -TERM "$PID" 2>/dev/null || true
  i=0
  while alive "$PID" && [ "$i" -lt 40 ]; do sleep 0.25; i=$((i + 1)); done
  if alive "$PID"; then
    echo "[launcher] SIGTERM 未生效，改 SIGKILL。"
    kill -KILL "$PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  kill_window
  echo "[launcher] ✓ 已停止"
  exit 0
fi

# ---------------------------------------------------------------------------
# 已经在跑就别再起一个（双击两次不会变成两个 server）
# 但「server 在跑」不等于「窗口开着」——窗口是子进程，关掉就没了，
# server 不会自己再拉一次。这里补上：没有窗口就开一个，否则双击等于没反应。
# ---------------------------------------------------------------------------
if probe; then
  echo "[launcher] server 已在运行：${URL}"
  if [ "$NEED_WIN" = 1 ]; then
    if window_running; then
      echo "[launcher]   窗口已在运行"
    elif open_window; then
      # 等它真的站住再报成功：起来就退（缺依赖/渲染进程崩）要当场看见
      i=0
      while [ "$i" -lt 12 ] && ! window_running; do sleep 0.25; i=$((i + 1)); done
      if window_running; then
        echo "[launcher]   窗口未打开 → 已重新拉起"
      else
        echo "[launcher]   ⚠ 窗口启动后立即退出，日志末尾："
        tail -n 8 "$LOG_FILE" 2>/dev/null | sed 's/^/    /'
      fi
    else
      echo "[launcher]   窗口未打开，且无法自动拉起。手动执行："
      echo "[launcher]     $VENV_PY $ROOT/service/window/window.py --port $PORT --host $HOST"
    fi
  fi
  exit 0
fi

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
# 后台运行时 stdout 是文件，Python 默认块缓冲——不关掉的话日志要攒满 8KB
# 才落盘，启动期崩了 logs/server.log 会是空的，等于没日志。
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# 前台模式：exec 顶替当前进程（老行为，调试 / systemd / 容器用）
# ---------------------------------------------------------------------------
if [ "$FOREGROUND" = 1 ]; then
  exec "$VENV_PY" "$ROOT/server/server.py" ${ARGS[@]+"${ARGS[@]}"}
fi

# ---------------------------------------------------------------------------
# 交接：后台独立进程
#   detach_run → setsid + nohup：连「会话」一起脱离，关终端/断 SSH 都碰不到
#   </dev/null  → 不持有 tty，终端窗口可以立刻释放
#   disown     → 从当前 shell 的作业表里摘掉
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
{
  echo ""
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') 启动 ====="
} >>"$LOG_FILE"

detach_run "$VENV_PY" "$ROOT/server/server.py" ${ARGS[@]+"${ARGS[@]}"}
SRV_PID="$DETACHED_PID"
echo "$SRV_PID" >"$PID_FILE"

# 等 HTTP 就绪（最多 30s；首次加载模型可能慢，超时只提示不算失败）
READY=0
i=0
while [ "$i" -lt 60 ]; do
  if probe; then READY=1; break; fi
  if ! alive "$SRV_PID"; then break; fi
  sleep 0.5
  i=$((i + 1))
done

if ! alive "$SRV_PID"; then
  echo "[launcher] ✗ server 启动后立即退出，日志末尾："
  tail -n 20 "$LOG_FILE" | sed 's/^/    /'
  rm -f "$PID_FILE"
  exit 1
fi

echo "[launcher] ✓ 已交接给 server（PID ${SRV_PID}）"
if [ "$READY" = 1 ]; then
  echo "[launcher]   地址 ${URL}"
else
  echo "[launcher]   地址 ${URL}（仍在启动，稍后可用）"
fi

# 窗口是 server 的子进程，起来要几百毫秒；等它站住，起不来就当场把日志摊开，
# 别让用户对着一个没有窗口的桌面干等（2026-10-29 的真实故障就是这么静默的）。
if [ "$NEED_WIN" = 1 ]; then
  i=0
  while [ "$i" -lt 16 ] && ! window_running; do sleep 0.25; i=$((i + 1)); done
  if window_running; then
    echo "[launcher]   窗口 ✓"
  else
    echo "[launcher]   ⚠ 窗口没有起来，日志末尾："
    tail -n 10 "$LOG_FILE" 2>/dev/null | sed 's/^/    /'
    echo "[launcher]   可手动打开：$VENV_PY $ROOT/service/window/window.py --port $PORT --host $HOST"
  fi
fi

echo "[launcher]   日志 ${LOG_FILE}"
echo "[launcher]   停止 ./启动.sh --stop"
exit 0
