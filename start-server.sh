#!/usr/bin/env bash
# ===========================================================================
# GARGANTUA v1 — start-server.sh（项目根目录）：POSIX 一键启动（macOS / Linux）
# ---------------------------------------------------------------------------
# 1. 启动前自动检查 GitHub 上是否有比本地（VERSION）更新的 Release；
#    pre-release 默认不建议更新，正式 latest 默认建议更新；y 更新 / n 跳过。
# 2. 调 setup-environment.sh 确保隔离 Python 3.12.x + venv + 依赖（幂等，秒回；
#    镜像自动测速选取）；
# 3. 用隔离 venv 的解释器 exec server/server.py。
#    环境预检、窗口拉起、服务按需调度全部由 server 负责
#    （见 server/server_env.py、server/server_services.py）。
#
# 用法：
#   ./start-server.sh                      正常启动（独立窗口）
#   ./start-server.sh --no-window          仅 HTTP server，不起窗口
#   ./start-server.sh --port 9000          换端口（其余参数透传给 server.py）
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR" && pwd)"

# ---------------------------------------------------------------------------
# 启动前：检查 GitHub Release 是否有比本地（VERSION 文件）更新的版本
# ---------------------------------------------------------------------------
REPO="Steve-Creeper-2012/GARGANTUA"

# 取最新 release 的 tag 与是否 pre-release；失败（离线 / 无 gh / 无 curl）返回空
_fetch_latest_release() {
  if command -v gh >/dev/null 2>&1; then
    gh api "repos/$REPO/releases/latest" --jq '"\(.tag_name) \(.prerelease)"' 2>/dev/null && return 0
  fi
  if command -v curl >/dev/null 2>&1; then
    local json
    json="$(curl -fsSL --max-time 8 "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null)" || return 1
    if command -v python3 >/dev/null 2>&1; then
      echo "$json" | python3 -c 'import sys,json
d=json.load(sys.stdin)
print(d.get("tag_name",""), d.get("prerelease",False))' 2>/dev/null && return 0
    fi
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

# 第一棒：装环境（失败则退出，不继续拉起）
"$SCRIPT_DIR/setup-environment.sh"

exec "$ROOT/environment/venv/bin/python3" "$ROOT/server/server.py" "$@"
