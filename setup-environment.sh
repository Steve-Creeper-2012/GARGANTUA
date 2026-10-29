#!/usr/bin/env bash
# ===========================================================================
# GARGANTUA v1 — setup-environment.sh（项目根目录）：构建隔离 Python 运行时
# ---------------------------------------------------------------------------
# 终端启动链第一棒。职责（全部与系统 Python 隔离，装在项目内）：
#   1. 发现「3.12 系列最新版」python-build-standalone 资产：
#        a. GARGANTUA_PY_RELEASE 指定 release tag（跳过发现）；
#        b. 否则读 GitHub releases atom feed（~15KB，快）拿最新 tag；
#        c. atom 不可达 → 回退内置固定 tag（PBS_FALLBACK_TAG）；
#      tag 内具体 patch 版本（3.12.x）用 HEAD 探测补全——未命中 404 很快，
#      命中一次即可，不拉 2MB 的 releases API JSON（国内直连 GitHub 资产
#      实测 ~1KB/s，API JSON 90s 都拉不完，这条路走不通）。
#   2. 下载镜像【自动测速】：对候选镜像并行发 Range 请求（前 2MB）算速率，
#      选最快的那个下载；失败的按速率排序依次回退。pip 源同理（测
#      simple 索引页响应）。候选全部可被环境变量覆盖/指定。
#   3. 解包到 environment/python/（顶层即 python/，绝不影响系统 Python）；
#      用下载的 Python 建虚拟环境 environment/venv/（含自愈：符号链接
#      因项目目录移动而悬空时自动重建）；
#   4. 按分类安装依赖：core.txt 必装；window.txt 桌面环境才装；
#      dev.txt 仅 GARGANTUA_INSTALL_DEV=1 时装；
#   5. 用「依赖文件内容的 sha256」做幂等 stamp——依赖清单没变就跳过，
#      变了才重装；删 environment/venv 可强制重建。
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
#   GARGANTUA_PY_BASE_URL=URL    直接指定下载镜像（跳过测速）
#   PIP_INDEX_URL=URL            自定义 pip 源（跳过测速）
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR" && pwd)"
PYTHON_DIR="$ROOT/environment/python"
VENV="$ROOT/environment/venv"
REQ_DIR="$ROOT/start/requirements"
STAMP="$VENV/.gargantua_setup_stamp"
PY_SERIES="3.12"          # 锁 3.12 系列（见文件头注释）
PY_MINOR_MIN=8            # 版本探测下界（3.12.8 起，足够覆盖历史 tag）
PY_MINOR_MAX=30           # 版本探测上界
PBS_REPO="astral-sh/python-build-standalone"
# atom feed 不可达时的回退 tag（内置时验证过含 cpython-3.12.15 全平台资产）
PBS_FALLBACK_TAG="20261021"
PROBE_BYTES=2097152       # 测速下载量（2MB，Range 请求）
PROBE_TIMEOUT=8           # 单个镜像测速超时（秒）

# 注意：log/die 一律写 stderr——discover_tag/probe_version 的 stdout 是
# 被命令替换捕获的返回值通道，混入日志会污染变量（实测踩过：TAG 变成
# 一整行日志文本，URL 全错，探测全 404）。
log() { echo "[setup] $*" >&2; }
die() { echo "[setup] ✗ $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 镜像自动测速：对候选 URL 发 Range 请求（前 PROBE_BYTES 字节），按 curl
# 自报的 speed_download（bytes/s，整数）排序返回最快者。全部失败返回空
# （调用方按原顺序回退）。
# ---------------------------------------------------------------------------
_probe_one() {
  # $1=URL $2=结果文件（写速率整数；失败写 0）
  local speed
  speed="$(curl -sL --max-time "$PROBE_TIMEOUT" -r "0-$((PROBE_BYTES - 1))" \
           -o /dev/null -w "%{speed_download}" "$1" 2>/dev/null || echo 0)"
  case "$speed" in (*[!0-9]*|"") speed=0 ;; esac
  echo "$speed" >"$2"
}

_pick_fastest() {
  # $1=探测目标名前缀（日志用） $2..=候选 URL；回显最快 URL
  local label="$1"; shift
  local tmp; tmp="$(mktemp -d)"
  local i=0 url
  for url in "$@"; do
    _probe_one "$url" "$tmp/$i" &
    i=$((i + 1))
  done
  wait
  local best="" best_score=-1 idx=0 score
  for url in "$@"; do
    score="$(cat "$tmp/$idx" 2>/dev/null || echo 0)"
    case "$score" in (*[!0-9]*|"") score=0 ;; esac
    log "  测速 ${score} B/s  ← ${url}"
    if [ "$score" -gt "$best_score" ]; then
      best_score="$score"; best="$url"
    fi
    idx=$((idx + 1))
  done
  rm -rf "$tmp"
  if [ "$best_score" -gt 0 ]; then echo "$best"; fi
  return 0
}

fastest_url() { _pick_fastest "下载镜像" "$@"; }

# pip 源测速：simple 索引页响应快慢（元数据解析是每次安装都要付的过路费；
# 文件吞吐差异会体现在页响应上——官方 PyPI 在国内 ~5KB/s，索引页都拉不动）
fastest_pip_index() {
  local tmp; tmp="$(mktemp -d)"
  local i=0 url
  for url in "$@"; do
    _probe_one "${url%/}/simple/six/" "$tmp/$i" &
    i=$((i + 1))
  done
  wait
  local best="" best_score=-1 idx=0 score
  for url in "$@"; do
    score="$(cat "$tmp/$idx" 2>/dev/null || echo 0)"
    case "$score" in (*[!0-9]*|"") score=0 ;; esac
    log "  测速 ${score} B/s  ← ${url}"
    if [ "$score" -gt "$best_score" ]; then
      best_score="$score"; best="$url"
    fi
    idx=$((idx + 1))
  done
  rm -rf "$tmp"
  if [ "$best_score" -gt 0 ]; then echo "$best"; fi
  return 0
}

# ---------------------------------------------------------------------------
# 幂等：venv 存在、stamp 在、依赖清单未变、解释器能跑 → 直接返回
# （解释器探测是自愈：项目目录被移动后 venv 里的绝对符号链接会悬空）
# ---------------------------------------------------------------------------
req_hash() { cat "$REQ_DIR"/*.txt 2>/dev/null | sha256sum | awk '{print $1}'; }

if [ -x "$VENV/bin/python3" ] && [ -f "$STAMP" ] \
   && "$VENV/bin/python3" -c "" 2>/dev/null; then
  EXISTING="$(grep -m1 'req_hash=' "$STAMP" 2>/dev/null | cut -d= -f2 || true)"
  if [ "$EXISTING" = "$(req_hash)" ]; then
    log "隔离环境已就绪（${VENV}），依赖未变，跳过安装。"
    exit 0
  fi
  log "依赖清单已变化，重新安装……"
elif [ -e "$VENV" ]; then
  # venv 在但解释器跑不了（目录移动/链接悬空）→ 整个重建
  log "检测到 environment/venv 不可用（项目目录被移动过？），重建……"
  rm -rf "$VENV"
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
# 主流程
# ---------------------------------------------------------------------------
if [ -x "$PYTHON_DIR/python/bin/python3" ]; then
  log "已存在独立 Python：${PYTHON_DIR}/python/bin/python3"
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
  log "Python ${VERSION}（${TRIPLE}）asset=${ASSET}"

  # ---- 下载：自动测速选镜像（同资产并行 Range 探测）----
  DIRECT="https://github.com/$PBS_REPO/releases/download/$TAG/$ASSET"
  if [ -n "${GARGANTUA_PY_BASE_URL:-}" ]; then
    MIRROR="$GARGANTUA_PY_BASE_URL"
    log "使用指定镜像：${MIRROR}"
  else
    log "镜像测速（前 2MB，并行）……"
    FAST="$(fastest_url \
      "https://gh-proxy.com/$DIRECT" \
      "$DIRECT" \
      "https://mirror.sjtu.edu.cn/github-release/$PBS_REPO/releases/download/$TAG/${ASSET//+/%2B}")"
    # 测速全失败 → 按默认偏好顺序回退
    if [ -n "$FAST" ]; then MIRROR="$FAST"; else MIRROR="https://gh-proxy.com/$DIRECT"; fi
    log "选中镜像：${MIRROR}"
  fi

  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  log "下载 ${MIRROR}"
  if ! curl -fSL --retry 3 --retry-delay 2 -C - --max-time 900 "$MIRROR" \
         -o "$TMP/py.tar.gz" || [ ! -s "$TMP/py.tar.gz" ] \
         || ! tar -tzf "$TMP/py.tar.gz" >/dev/null 2>&1; then
    # 选中镜像失败 → 按默认顺序再试其余两个（不重复测速）
    OK=0
    for m in "https://gh-proxy.com/$DIRECT" "$DIRECT" \
             "https://mirror.sjtu.edu.cn/github-release/$PBS_REPO/releases/download/$TAG/${ASSET//+/%2B}"; do
      [ "$m" = "$MIRROR" ] && continue
      log "⚠️ 主镜像失败，尝试 ${m}"
      rm -f "$TMP/py.tar.gz"
      if curl -fSL --retry 2 -C - --max-time 900 "$m" -o "$TMP/py.tar.gz" \
           && [ -s "$TMP/py.tar.gz" ] && tar -tzf "$TMP/py.tar.gz" >/dev/null 2>&1; then
        MIRROR="$m"; OK=1; break
      fi
    done
    [ "$OK" = "1" ] || die "所有镜像下载失败：${ASSET}。请检查网络/代理，" \
      "或设 GARGANTUA_PY_BASE_URL 指向可达镜像。"
  fi
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
  log "创建隔离虚拟环境：${VENV}"
  "$PYTHON_DIR/python/bin/python3" -m venv "$VENV"
fi

# ---- 装依赖 ----
PY="$VENV/bin/python3"
if [ -n "${PIP_INDEX_URL:-}" ]; then
  INDEX_ARGS=()
  log "使用指定 pip 源：${PIP_INDEX_URL}"
else
  log "pip 源测速……"
  FAST_INDEX="$(fastest_pip_index \
    "https://pypi.tuna.tsinghua.edu.cn/simple" \
    "https://pypi.org/simple")"
  if [ -n "$FAST_INDEX" ]; then CHOSEN_INDEX="$FAST_INDEX"; else
    CHOSEN_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
  fi
  INDEX_ARGS=(-i "$CHOSEN_INDEX")
  log "选中 pip 源：${CHOSEN_INDEX}"
fi
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

# ---- 写 stamp（依赖哈希 + 选中的镜像，保证清单/网络变化时可重装）----
CUR="$(req_hash)"
printf 'ts=%s\nreq_hash=%s\npython=%s\npy_mirror=%s\npip_index=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CUR" "$("$PY" -V 2>&1)" \
  "${MIRROR:-}" "${CHOSEN_INDEX:-}" > "$STAMP"
log "✓ 隔离环境就绪：${VENV}"
log "  启动：./start-server.sh（或双击 start-server.command）"
