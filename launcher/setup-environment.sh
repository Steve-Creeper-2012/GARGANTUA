#!/usr/bin/env bash
# ===========================================================================
# GARGANTUA v1 — launcher/setup-environment.sh：构建隔离 Python 运行时
# ---------------------------------------------------------------------------
# 目标：保证项目内有一个 >= 3.12.14 的 Python（默认目标 3.12.15），并建好 venv + 依赖。
#
# 策略（按速度 / 可用性依次尝试，慢的下载只作兜底）：
#   1. 已存在达标的 venv（environment/venv/bin/python3 >= 3.12.14）→ 幂等秒回，不联网。
#   2. 用系统包管理器装 python3.12（最快，steve 指定）：
#        macOS      → brew install python@3.12
#        Linux      → apt-get / dnf / apk install python3.12（+venv）
#        （非 root 自动加 sudo）
#   3. 以上都拿不到（无包管理器 / 受限环境）→ 回退 python-build-standalone 下载。
# 之后用该解释器建 venv 并安装 core.txt（+ window.txt，除非 GARGANTUA_NO_WINDOW=1）。
#
# 注意：第 2 步在 steve 的 macOS 上走 brew，比 curl 下载快得多、也稳；
#       且全程可见输出（不静默吞掉），避免「以为卡死」。
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="$ROOT/environment/venv"
REQ_DIR="$ROOT/start/requirements"
STAMP="$VENV/.gargantua_setup_stamp"

# 默认目标版本：3.12.15；最低可接受：3.12.14（steve 的开发环境约定）
PY_TARGET="3.12.15"
MIN_MAJ=3; MIN_MIN=12; MIN_PAT=14

log() { echo "[setup] $*" >&2; }
die() { echo "[setup] ✗ $*" >&2; exit 1; }

# 跨平台 sha256（macOS 无 sha256sum，用 shasum 兜底）
req_hash() {
  if command -v sha256sum >/dev/null 2>&1; then
    cat "$REQ_DIR"/*.txt 2>/dev/null | sha256sum | awk '{print $1}'
  else
    cat "$REQ_DIR"/*.txt 2>/dev/null | shasum -a 256 | awk '{print $1}'
  fi
}

# 版本比较：ver_ge A B → A>=B（A/B 为 major.minor.patch）
ver_ge() {
  local a="$1" b="$2" ax ay az bx by bz
  IFS=. read -r ax ay az <<<"$a"
  IFS=. read -r bx by bz <<<"$b"
  ax=${ax:-0}; ay=${ay:-0}; az=${az:-0}
  bx=${bx:-0}; by=${by:-0}; bz=${bz:-0}
  [ "$ax" -gt "$bx" ] && return 0
  [ "$ax" -lt "$bx" ] && return 1
  [ "$ay" -gt "$by" ] && return 0
  [ "$ay" -lt "$by" ] && return 1
  [ "$az" -ge "$bz" ]
}

py_version() { "$1" -c "import sys;print('%d.%d.%d'%sys.version_info[:3])" 2>/dev/null; }

# ---------------------------------------------------------------------------
# 幂等早退：venv 解释器存在且达标 + 依赖清单未变 → 直接返回（不联网、不重装）
# ---------------------------------------------------------------------------
if [ -x "$VENV/bin/python3" ]; then
  V="$(py_version "$VENV/bin/python3" || true)"
  if [ -n "$V" ] && ver_ge "$V" "$MIN_MAJ.$MIN_MIN.$MIN_PAT"; then
    if [ -f "$STAMP" ] && [ "$(grep -m1 'req_hash=' "$STAMP" 2>/dev/null | cut -d= -f2)" = "$(req_hash)" ]; then
      log "隔离环境已就绪（Python $V >= $MIN_MAJ.$MIN_MIN.$MIN_PAT），跳过安装。"
      exit 0
    fi
    log "依赖清单变化，仅重装依赖（Python $V 仍达标）……"
  else
    log "已存在 venv 的 Python $V 低于要求（>= $MIN_MAJ.$MIN_MIN.$MIN_PAT），重建……"
    rm -rf "$VENV"
  fi
fi

# ---------------------------------------------------------------------------
# 1) 取得 python3.12 解释器（>= 3.12.14）
# ---------------------------------------------------------------------------
PY312=""

_acquire_brew() {
  command -v brew >/dev/null 2>&1 || return 1
  log "使用 brew 安装/更新 python@3.12（快速路径）……"
  brew install python@3.12 2>&1 || brew upgrade python@3.12 2>&1 || true
  local bp; bp="$(brew --prefix python@3.12 2>/dev/null)"
  [ -x "$bp/bin/python3.12" ] && { PY312="$bp/bin/python3.12"; return 0; }
  return 1
}

_acquire_apt() {
  command -v apt-get >/dev/null 2>&1 || return 1
  local s=""; [ "$(id -u)" = "0" ] || s="sudo"
  log "使用 apt 安装 python3.12……"
  $s apt-get update -y 2>&1 || true
  $s apt-get install -y python3.12 python3.12-venv python3.12-dev 2>&1 || return 1
  [ -x /usr/bin/python3.12 ] && { PY312=/usr/bin/python3.12; return 0; }
  return 1
}

_acquire_dnf() {
  command -v dnf >/dev/null 2>&1 || return 1
  local s=""; [ "$(id -u)" = "0" ] || s="sudo"
  log "使用 dnf 安装 python3.12……"
  $s dnf install -y python3.12 python3.12-pip 2>&1 || return 1
  command -v python3.12 >/dev/null 2>&1 && { PY312="$(command -v python3.12)"; return 0; }
  return 1
}

_acquire_apk() {
  command -v apk >/dev/null 2>&1 || return 1
  local s=""; [ "$(id -u)" = "0" ] || s="sudo"
  log "使用 apk 安装 python3.12……"
  $s apk add python3.12 2>&1 || return 1
  command -v python3.12 >/dev/null 2>&1 && { PY312="$(command -v python3.12)"; return 0; }
  return 1
}

# 回退：下载 python-build-standalone（优先滚动 3.12 资产，避免逐个 minor 探测卡死）
_download_standalone() {
  local os arch triple
  os="$(uname -s)"; arch="$(uname -m)"
  case "$os" in
    Darwin) [ "$arch" = "arm64" ] && triple="aarch64-apple-darwin" || triple="x86_64-apple-darwin" ;;
    Linux)  [ "$arch" = "aarch64" ] && triple="aarch64-unknown-linux-gnu" || triple="x86_64-unknown-linux-gnu" ;;
    *) die "不支持的操作系统: $os（仅 macOS / Linux；Windows 请用 setup.bat）" ;;
  esac
  local repo="astral-sh/python-build-standalone"
  local tag
  tag="$(curl -fsSL --max-time 20 --retry 1 "https://github.com/$repo/releases.atom" 2>/dev/null \
        | grep -o "releases/tag/[^<\"]*" | head -1 | cut -d/ -f3)" || true
  [ -n "$tag" ] || tag="20261026"
  log "python-build-standalone tag=$tag（回退下载路径）"

  # 优先滚动资产 cpython-3.12+<tag>；缺失再探测具体 minor（上限收窄到 3.12.8..3.12.30，短超时）
  local asset="" url code
  url="https://github.com/$repo/releases/download/$tag/cpython-3.12+${tag}-${triple}-install_only.tar.gz"
  code="$(curl -s -o /dev/null -w '%{http_code}' -I --connect-timeout 10 --max-time 25 "$url" 2>/dev/null)" || true
  if [ "$code" = "200" ] || [ "$code" = "302" ]; then
    asset="cpython-3.12+${tag}-${triple}-install_only.tar.gz"
  else
    local minor
    for ((minor=8; minor<=30; minor++)); do
      u="https://github.com/$repo/releases/download/$tag/cpython-3.12.${minor}+${tag}-${triple}-install_only.tar.gz"
      c="$(curl -s -o /dev/null -w '%{http_code}' -I --connect-timeout 6 --max-time 12 "$u" 2>/dev/null)" || true
      if [ "$c" = "200" ] || [ "$c" = "302" ]; then asset="cpython-3.12.${minor}+${tag}-${triple}-install_only.tar.gz"; break; fi
    done
  fi
  [ -n "$asset" ] || die "未找到 cpython-3.12 资产（tag=$tag），请检查网络或手动安装 python3.12。"
  url="https://github.com/$repo/releases/download/$tag/$asset"
  log "下载 $asset ……（这步会比 brew 慢，属一次性兜底）"
  local tmp; tmp="$(mktemp -d)"
  curl -fSL --retry 2 --retry-delay 2 -C - --max-time 600 --progress-bar "$url" -o "$tmp/py.tar.gz" 2>/dev/null \
    || curl -fSL --retry 2 -C - --max-time 600 --progress-bar "https://gh-proxy.com/$url" -o "$tmp/py.tar.gz" 2>/dev/null \
    || curl -fSL --retry 2 -C - --max-time 600 --progress-bar "https://mirror.sjtu.edu.cn/github-release/$repo/releases/download/$tag/${asset//+/%2B}" -o "$tmp/py.tar.gz" 2>/dev/null
  [ -s "$tmp/py.tar.gz" ] || die "下载 python-build-standalone 失败，请检查网络。"
  mkdir -p "$ROOT/environment/python"
  tar -xzf "$tmp/py.tar.gz" -C "$ROOT/environment/python"
  PY312="$ROOT/environment/python/python/bin/python3"
}

if ! _acquire_brew && ! _acquire_apt && ! _acquire_dnf && ! _acquire_apk; then
  log "未找到系统包管理器，回退到 python-build-standalone 下载……"
  _download_standalone
fi

[ -x "$PY312" ] || die "无法取得 Python 3.12（请手动安装 python3.12 后重试）。"
V="$(py_version "$PY312")"; [ -n "$V" ] || die "取得的解释器无法读取版本。"
ver_ge "$V" "$MIN_MAJ.$MIN_MIN.$MIN_PAT" || die "取得的 Python $V 低于要求（>= $MIN_MAJ.$MIN_MIN.$MIN_PAT）。"
log "Python $V（$PY312）✓"

# ---------------------------------------------------------------------------
# 2) 建 venv + 装依赖
# ---------------------------------------------------------------------------
if [ ! -x "$VENV/bin/python3" ]; then
  log "创建隔离虚拟环境：$VENV"
  "$PY312" -m venv "$VENV"
fi
PY="$VENV/bin/python3"

# pip 源：国内优先 tuna（6s 探活），不可达回退官方 PyPI
if [ -n "${PIP_INDEX_URL:-}" ]; then
  INDEX="$PIP_INDEX_URL"; INDEX_ARGS=(-i "$INDEX")
else
  if curl -fsS --max-time 6 -o /dev/null "https://pypi.tuna.tsinghua.edu.cn/simple/six/" 2>/dev/null; then
    INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"; INDEX_ARGS=(-i "$INDEX")
  else
    INDEX="https://pypi.org/simple"; INDEX_ARGS=()
  fi
  log "pip 源：$INDEX"
fi

log "升级 pip ……"
"$PY" -m pip install --quiet --upgrade pip "${INDEX_ARGS[@]}"
log "安装核心依赖（core.txt）……"
"$PY" -m pip install --quiet "${INDEX_ARGS[@]}" -r "$REQ_DIR/core.txt"
if [ "${GARGANTUA_NO_WINDOW:-0}" != "1" ]; then
  log "安装窗口依赖（window.txt，失败不致命）……"
  "$PY" -m pip install --quiet "${INDEX_ARGS[@]}" -r "$REQ_DIR/window.txt" \
    || log "⚠️ 窗口依赖安装失败（可稍后：pip install -r start/requirements/window.txt）"
fi

printf 'ts=%s\nreq_hash=%s\npython=%s\npip_index=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(req_hash)" "$("$PY" -V 2>&1)" "${INDEX:-}" > "$STAMP"
log "✓ 隔离环境就绪：$VENV"
log "  启动：./启动.sh（或双击 启动.command；内部启动器在 launcher/）"
