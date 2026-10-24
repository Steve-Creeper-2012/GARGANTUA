#!/usr/bin/env bash
# ===========================================================================
# GARGANTUA v1 — start/terminal/setup.sh：构建隔离 Python 运行时
# ---------------------------------------------------------------------------
# 终端启动链第一棒。职责（全部与系统 Python 隔离，装在项目内）：
#   1. 发现「3.12 系列最新版」python-build-standalone 资产：
#        a. GARGANTUA_PY_RELEASE 指定 release tag（跳过发现）；
#        b. 否则读 GitHub releases atom feed（~15KB，快）拿最新 tag；
#        c. atom 不可达 → 回退内置固定 tag（PBS_FALLBACK_TAG）；
#      tag 内具体 patch 版本（3.12.x）用 HEAD 探测补全——未命中 404 很快，
#      命中一次即可，不拉 2MB 的 releases API JSON（国内直连 GitHub 资产
#      实测 ~1KB/s，API JSON 90s 都拉不完，这条路走不通）。
#   2. 下载走镜像链（国内优先）：gh-proxy.com → GitHub 直连 → SJTU 镜像；
#      解包到 start/.python/（顶层即 python/，绝不影响系统 Python）；
#   3. 用下载的 Python 建虚拟环境 start/.venv/；
#   4. 按分类安装依赖：core.txt 必装；window.txt 桌面环境才装；
#      dev.txt 仅 GARGANTUA_INSTALL_DEV=1 时装；
#   5. 用「依赖文件内容的 sha256」做幂等 stamp——依赖清单没变就跳过，
#      变了才重装；删 start/.venv 可强制重建。
#
# 为什么锁 3.12 系列（steve 2026-10-24 确认：用 3.12 的 latest，不锁
# 具体 patch）：torch 2.2.2 是最后一个带 macOS x86_64 官方轮子的版本，
# 最高支持 Python 3.12（3.13 起无预编译轮子，只能源码编译）。
#
# 环境变量开关：
#   GARGANTUA_NO_WINDOW=1        跳过 pywebview 窗口依赖
#   GARGANTUA_INSTALL_DEV=1      额外装测试依赖（pytest）
#   GARGANTUA_PY_RELEASE=TAG     指定 python-build-standalone release tag
#   GARGANTUA_PY_VERSION=3.12.N  指定 patch 版本（跳过探测）
#   GARGANTUA_PY_BASE_URL=URL    覆盖下载基址（默认走镜像链）
#   PIP_INDEX_URL=URL            自定义 pip 源（默认清华，见下）
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_DIR="$ROOT/start/.python"
VENV="$ROOT/start/.venv"
REQ_DIR="$ROOT/start/requirements"
STAMP="$VENV/.gargantua_setup_stamp"
PY_SERIES="3.12"          # 锁 3.12 系列（见文件头注释）
PY_MINOR_MIN=8            # 版本探测下界（3.12.8 起，足够覆盖历史 tag）
PY_MINOR_MAX=30           # 版本探测上界
PBS_REPO="astral-sh/python-build-standalone"
# atom feed 不可达时的回退 tag（内置时验证过含 cpython-3.12.15 全平台资产）
PBS_FALLBACK_TAG="20261021"

# 注意：log/die 一律写 stderr——discover_tag/probe_version 的 stdout 是
# 被命令替换捕获的返回值通道，混入日志会污染变量（实测踩过：TAG 变成
# 一整行日志文本，URL 全错，探测全 404）。
log() { echo "[setup] $*" >&2; }
die() { echo "[setup] ✗ $*" >&2; exit 1; }

# ---- 幂等：venv 存在且依赖未变 → 直接返回 ----
if [ -x "$VENV/bin/python3" ] && [ -f "$STAMP" ]; then
  EXISTING="$(grep -m1 'req_hash=' "$STAMP" 2>/dev/null | cut -d= -f2 || true)"
  CUR="$(cat "$REQ_DIR"/*.txt 2>/dev/null | sha256sum | awk '{print $1}')"
  if [ "$EXISTING" = "$CUR" ]; then
    log "隔离环境已就绪（${VENV}），依赖未变，跳过安装。"
    exit 0
  fi
  log "依赖清单已变化，重新安装……"
fi

# ---------------------------------------------------------------------------
# 1) 发现 release tag（atom feed；失败回退内置 tag）
# ---------------------------------------------------------------------------
discover_tag() {
  if [ -n "${GARGANTUA_PY_RELEASE:-}" ]; then
    echo "$GARGANTUA_PY_RELEASE"; return 0
  fi
  local tag=""
  tag="$(curl -fsSL --max-time 25 --retry 2 --retry-delay 2 \
         "https://github.com/$PBS_REPO/releases.atom" 2>/dev/null \
         | grep -o "releases/tag/[^<\"]*" | head -1 | cut -d/ -f3 || true)"
  if [ -z "$tag" ]; then
    log "atom feed 不可达，使用内置回退 tag=$PBS_FALLBACK_TAG"
    tag="$PBS_FALLBACK_TAG"
  else
    log "最新 release tag=${tag}（atom feed）"
  fi
  echo "$tag"
}

# ---------------------------------------------------------------------------
# 2) 探测该 tag 内的 3.12.x patch 版本（HEAD 未命中很快，命中一次即停）
# ---------------------------------------------------------------------------
probe_version() {
  local tag="$1"
  if [ -n "${GARGANTUA_PY_VERSION:-}" ]; then
    echo "$GARGANTUA_PY_VERSION"; return 0
  fi
  local minor url code
  for ((minor=PY_MINOR_MIN; minor<=PY_MINOR_MAX; minor++)); do
    url="https://github.com/$PBS_REPO/releases/download/$tag/cpython-${PY_SERIES}.${minor}+${tag}-${TRIPLE}-install_only.tar.gz"
    # 未命中（404）很快（~1s）；命中要等 GitHub 重定向到签名 URL（实测
    # ~16s），所以超时放到 30s——用 15s 会把真资产误判为未命中。
    code="$(curl -s -o /dev/null -w "%{http_code}" -I --connect-timeout 10 --max-time 30 "$url" || true)"
    if [ "$code" = "200" ] || [ "$code" = "302" ]; then
      echo "${PY_SERIES}.${minor}"; return 0
    fi
  done
  return 1
}

# ---------------------------------------------------------------------------
# 3) 下载（镜像链：gh-proxy → 直连 → SJTU；支持断点续传）
# ---------------------------------------------------------------------------
download_asset() {
  local tag="$1" asset="$2" out="$3"
  local direct="https://github.com/$PBS_REPO/releases/download/$tag/$asset"
  local mirrors=()
  if [ -n "${GARGANTUA_PY_BASE_URL:-}" ]; then
    mirrors=("$GARGANTUA_PY_BASE_URL")
  else
    # gh-proxy.com：国内可达（实测 ~400KB/s；GitHub 直连资产 ~1KB/s 不可用）
    mirrors=("https://gh-proxy.com/$direct" "$direct"
             "https://mirror.sjtu.edu.cn/github-release/$PBS_REPO/releases/download/$tag/${asset//+/%2B}")
  fi
  local m
  for m in "${mirrors[@]}"; do
    log "下载 $m"
    if curl -fSL --retry 3 --retry-delay 2 -C - --max-time 900 "$m" -o "$out" \
         && [ -s "$out" ] && tar -tzf "$out" >/dev/null 2>&1; then
      return 0
    fi
    log "⚠️ 该源失败，尝试下一个……"
    rm -f "$out"
  done
  return 1
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
if [ -x "$PYTHON_DIR/python/bin/python3" ]; then
  log "已存在独立 Python：$PYTHON_DIR/python/bin/python3"
else
  OS="$(uname -s)"; ARCH="$(uname -m)"
  case "$OS" in
    Darwin)
      if [ "$ARCH" = "arm64" ]; then TRIPLE="aarch64-apple-darwin"
      else TRIPLE="x86_64-apple-darwin"; fi ;;
    Linux)
      if [ "$ARCH" = "aarch64" ]; then TRIPLE="aarch64-unknown-linux-gnu"
      else TRIPLE="x86_64-unknown-linux-gnu"; fi ;;
    *) die "不支持的操作系统: ${OS}（仅 macOS / Linux；Windows 用 setup.bat）" ;;
  esac

  TAG="$(discover_tag)"
  VERSION="$(probe_version "$TAG")" \
    || die "tag=$TAG 内未找到 cpython-${PY_SERIES}.${PY_MINOR_MIN}..${PY_MINOR_MAX} 资产。" \
          "可设 GARGANTUA_PY_VERSION=3.12.N 或 GARGANTUA_PY_RELEASE=<tag> 手动指定。"
  ASSET="cpython-${VERSION}+${TAG}-${TRIPLE}-install_only.tar.gz"
  log "Python ${VERSION}（${TRIPLE}）asset=$ASSET"

  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  download_asset "$TAG" "$ASSET" "$TMP/py.tar.gz" \
    || die "下载失败：${ASSET}。请检查网络/代理，或设 GARGANTUA_PY_BASE_URL 指向可达镜像。"
  mkdir -p "$PYTHON_DIR"
  tar -xzf "$TMP/py.tar.gz" -C "$PYTHON_DIR"   # 解包出顶层 python/ 目录
  if [ ! -x "$PYTHON_DIR/python/bin/python3" ]; then
    die "解包后未找到 python3，安装失败。"
  fi
fi

# 解释器自检：必须是 3.12 系列（torch 2.2.2 不支持 3.13 预编译轮子）
"$PYTHON_DIR/python/bin/python3" -c \
  "import sys; assert sys.version_info[:2]==(3,12), sys.version" \
  || die "独立 Python 不是 3.12 系列，拒绝继续（torch 2.2.2 需要 ≤3.12）。"
log "独立 Python：$("$PYTHON_DIR/python/bin/python3" -V 2>&1)"

# ---- 建 venv ----
if [ ! -x "$VENV/bin/python3" ]; then
  log "创建隔离虚拟环境：$VENV"
  "$PYTHON_DIR/python/bin/python3" -m venv "$VENV"
fi

# ---- 装依赖 ----
PY="$VENV/bin/python3"
# pip 源：尊重 PIP_INDEX_URL；否则清华镜像（实测国内 PyPI 官方 ~5KB/s、
# 清华 ~18MB/s，torch 2.2.2 的 macOS 轮子 200MB+，源选错首次安装差 3000 倍）
if [ -n "${PIP_INDEX_URL:-}" ]; then INDEX_ARGS=(); else
  INDEX_ARGS=(-i https://pypi.tuna.tsinghua.edu.cn/simple); fi
"$PY" -m pip install --quiet --upgrade pip "${INDEX_ARGS[@]}"
log "安装核心依赖（core.txt）……"
"$PY" -m pip install --quiet "${INDEX_ARGS[@]}" -r "$REQ_DIR/core.txt"

if [ "${GARGANTUA_NO_WINDOW:-0}" != "1" ]; then
  log "安装窗口依赖（window.txt，失败不致命）……"
  "$PY" -m pip install --quiet "${INDEX_ARGS[@]}" -r "$REQ_DIR/window.txt" \
    || log "⚠️ 窗口依赖安装失败（可稍后手动：pip install -r start/requirements/window.txt）"
fi
if [ "${GARGANTUA_INSTALL_DEV:-0}" = "1" ]; then
  log "安装开发依赖（dev.txt）……"
  "$PY" -m pip install --quiet "${INDEX_ARGS[@]}" -r "$REQ_DIR/dev.txt"
fi

# ---- 写 stamp（含依赖哈希，保证清单变化时重装）----
CUR="$(cat "$REQ_DIR"/*.txt | sha256sum | awk '{print $1}')"
printf 'ts=%s\nreq_hash=%s\npython=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CUR" "$("$PY" -V 2>&1)" > "$STAMP"
log "✓ 隔离环境就绪：$VENV"
log "  启动：start/terminal/start.sh（或双击 start.command）"
