#!/usr/bin/env bash
# ===========================================================================
# GARGANTUA v1 — launcher/start-server.sh：POSIX 一键启动（macOS / Linux，由根目录 启动.sh / 启动.command 调用）
# ---------------------------------------------------------------------------
# 1. 启动前自动检查 GitHub 上是否有比本地（VERSION）更新的 Release；
#    只用 curl --max-time 8 取（gh api 无内置超时，网络不通会永久挂死终端，
#    故不再用）；离线 / 超时自动跳过，绝不阻塞启动。
#    pre-release 默认不建议更新，正式 latest 默认建议更新；y 更新 / n 跳过。
# 2. 调 setup-environment.sh 确保隔离 Python 3.12.x + venv + 依赖（幂等，秒回；
#    镜像自动测速选取）；
# 3. 轻量依赖预检：只确认「隔离 Python 解释器」与（窗口模式下）「pywebview」
#    就绪；torch/rust 等重依赖不在启动链阻塞（由 setup 装好、按需子进程自理）。
# 4. 用隔离 venv 的解释器 exec server/server.py（并设置 GARGANTUA_NO_DEPCHECK=1
#    让 server 跳过自身的主依赖阻塞检查）。窗口拉起、服务调度由 server 负责。
#
# 用法：
#   ./start-server.sh                      正常启动（独立窗口）
#   ./start-server.sh --no-window          仅 HTTP server，不起窗口
#   ./start-server.sh --port 9000          换端口（其余参数透传给 server.py）
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# 启动前：检查 GitHub Release 是否有比本地（VERSION 文件）更新的版本
# ---------------------------------------------------------------------------
REPO="Steve-Creeper-2012/GARGANTUA"

# 取最新 release 的 tag 与是否 pre-release；失败（离线 / 无 curl）返回空。
# 只用 curl（自带 --max-time 8 硬超时，离线 8 秒必退）——不再走 gh api：
# gh api 无内置超时，网络不通时会永久挂死（终端「半天不响应」的根源）。
_fetch_latest_release() {
  command -v curl >/dev/null 2>&1 || return 1
  local json
  json="$(curl -fsSL --max-time 8 "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null)" || return 1
  [ -n "$json" ] || return 1
  if command -v python3 >/dev/null 2>&1; then
    echo "$json" | python3 -c 'import sys,json
d=json.load(sys.stdin)
print(d.get("tag_name",""), d.get("prerelease",False))' 2>/dev/null && return 0
  fi
  return 1
}

# 语义化版本比较：_ver_gt A B 为真当且仅当 A > B（忽略前导 v）
_ver_gt() {
  local a="${1#v}" b="${2#v}"
  [[ "$a" =~ ^[0-9]+(\.[0-9]+)*$ ]] || return 1
  [[ "$b" =~ ^[0-9]+(\.[0-9]+)*$ ]] || return 1
  local ax ay az bx by bz
  IFS=. read -r ax ay az <<<"$a"
  IFS=. read -r bx by bz <<<"$b"
  ax=${ax:-0}; ay=${ay:-0}; az=${az:-0}
  bx=${bx:-0}; by=${by:-0}; bz=${bz:-0}
  [ "$ax" -gt "$bx" ] && return 0
  [ "$ax" -lt "$bx" ] && return 1
  [ "$ay" -gt "$by" ] && return 0
  [ "$ay" -lt "$by" ] && return 1
  [ "$az" -gt "$bz" ]
}

_check_release_update() {
  local vf="$ROOT/VERSION"
  [ -f "$vf" ] || return 0
  local cur; cur="$(tr -d '[:space:]' < "$vf")"
  [ -n "$cur" ] || return 0
  local info; info="$(_fetch_latest_release)" || {
    echo "[release] 无法获取最新版本（可能离线），跳过更新检查。"; return 0; }
  local latest pre
  read -r latest pre <<<"$info"
  [ -n "$latest" ] || return 0
  local skipf="$ROOT/.release_skip"
  if [ -f "$skipf" ] && [ "$(cat "$skipf" 2>/dev/null)" = "$latest" ]; then
    return 0
  fi
  if _ver_gt "$latest" "$cur"; then
    local is_pre=0
    case "$pre" in True|true|1) is_pre=1 ;; esac
    if [ "$is_pre" = 1 ]; then
      echo "[release] 发现新版本 $latest（pre-release）。建议不更新。"
    else
      echo "[release] 发现新版本 $latest（正式 latest）。建议更新。"
    fi
    local tty=/dev/tty; [ -r "$tty" ] || tty=/dev/stdin
    local ans=N
    if [ "$is_pre" = 1 ]; then
      read -r -p "[release] 是否更新到 $latest? [y/N] " -n 1 ans < "$tty" || ans=N
    else
      read -r -p "[release] 是否更新到 $latest? [Y/n] " -n 1 ans < "$tty" || ans=N
    fi
    echo
    case "$ans" in
      y|Y)
        echo "[release] 正在更新到 $latest……"
        if ! git -C "$ROOT" pull --ff-only 2>&1 | tail -5; then
          echo "[release] ⚠️ git pull 失败，请手动更新。"
        else
          echo "[release] ✓ 已更新到 $latest，正在用新版本重新启动……"
          exec "$0" "$@"
        fi
        ;;
      *)
        echo "$latest" > "$skipf"
        echo "[release] 已跳过 $latest（下次发布新版本时会再次提醒）。"
        ;;
    esac
  fi
  return 0
}

# 启动前先检查更新（离线 / 失败自动跳过，不影响正常启动）
set +e
_check_release_update
set -e

# 第一棒：装环境（幂等；已就绪则秒回。失败则退出，不继续拉起）
"$SCRIPT_DIR/setup-environment.sh"

# ---------------------------------------------------------------------------
# 轻量依赖预检（launcher 只查「必要依赖」）
#   - 隔离 Python 解释器：必须有（没有就什么也跑不了）——硬检查，缺失即退出；
#   - pywebview：仅「有窗口」模式需要——软检查，缺失仅警告，HTTP server 仍可用。
#   torch/rust 等重依赖不在此阻塞：由 setup-environment.sh 装好、server 按需
#   拉起的子进程内部自行处理，避免启动长时间卡死。
# ---------------------------------------------------------------------------
PY="$ROOT/environment/venv/bin/python3"
if [ ! -x "$PY" ]; then
  echo "[precheck] ✗ 未找到隔离 Python 解释器：$PY"
  echo "[precheck]   请先运行 launcher/setup-environment.sh 搭建隔离环境。"
  exit 1
fi
echo "[precheck] 隔离 Python ✓（$PY）"

NEED_WIN=1
for a in "$@"; do
  if [ "$a" = "--no-window" ]; then NEED_WIN=0; fi
done
if [ "$NEED_WIN" = 1 ]; then
  if "$PY" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('webview') is not None else 1)" 2>/dev/null; then
    echo "[precheck] pywebview ✓"
  else
    echo "[precheck] ⚠ pywebview 未安装：桌面窗口可能起不来，但 HTTP server 仍可用。"
    echo "[precheck]   重新安装窗口依赖：GARGANTUA_NO_WINDOW=0 launcher/setup-environment.sh"
  fi
fi

# 告知 server：环境就绪，跳过自身的主依赖（torch/rust…）阻塞检查
export GARGANTUA_NO_DEPCHECK=1

exec "$PY" "$ROOT/server/server.py" "$@"
