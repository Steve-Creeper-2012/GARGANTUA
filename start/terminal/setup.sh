#!/usr/bin/env bash
# ===========================================================================
# GARGANTUA v1 — start/terminal/setup.sh：构建隔离 Python 运行时
# ---------------------------------------------------------------------------
# 终端启动链第一棒。职责（全部与系统 Python 隔离，装在项目内）：
#   1. 探测 OS/arch，从 python-build-standalone 下载最新的 3.12.x
#      （3.12.15 是截至定稿的最新 3.12 版；torch 2.2.2 不支持 3.13 预编译
#      轮子，所以锁 3.12.x）；装到 start/.python/；
#   2. 用下载的 Python 建虚拟环境 start/.venv/（绝不碰系统 Python）；
#   3. 按分类安装依赖：core.txt 必装；window.txt 桌面环境才装；
#      dev.txt 仅 GARGANTUA_INSTALL_DEV=1 时装；
#   4. 用「依赖文件内容的 sha256」做幂等 stamp——依赖清单没变就跳过，
#      变了才重装；删 start/.venv 可强制重建。
#
# 环境变量开关：
#   GARGANTUA_NO_WINDOW=1   跳过 pywebview 窗口依赖
#   GARGANTUA_INSTALL_DEV=1 额外装测试依赖（pytest）
#   GARGANTUA_PY_RELEASE=TAG 指定 python-build-standalone release tag（覆盖自动探测）
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_DIR="$ROOT/start/.python"
VENV="$ROOT/start/.venv"
REQ_DIR="$ROOT/start/requirements"
STAMP="$VENV/.gargantua_setup_stamp"
PY_SERIES="3.12"          # 锁 3.12.x（见文件头注释）

# ---- 幂等：venv 存在且依赖未变 → 直接返回 ----
if [ -x "$VENV/bin/python3" ] && [ -f "$STAMP" ]; then
  EXISTING="$(grep -m1 'req_hash=' "$STAMP" 2>/dev/null | cut -d= -f2 || true)"
  CUR="$(cat "$REQ_DIR"/*.txt 2>/dev/null | sha256sum | awk '{print $1}')"
  if [ "$EXISTING" = "$CUR" ]; then
    echo "[setup] 隔离环境已就绪（$VENV），依赖未变，跳过安装。"
    exit 0
  fi
  echo "[setup] 依赖清单已变化，重新安装……"
fi

# ---- 确保下载的独立 Python ----
ensure_python() {
  if [ -x "$PYTHON_DIR/python/bin/python3" ]; then
    echo "[setup] 已存在独立 Python：$PYTHON_DIR/python/bin/python3"
    return 0
  fi
  echo "[setup] 下载 python-build-standalone $PY_SERIES.x ……"
  OS="$(uname -s)"; ARCH="$(uname -m)"
  case "$OS" in
    Darwin)
      if [ "$ARCH" = "arm64" ]; then TRIPLE="aarch64-apple-darwin"
      else TRIPLE="x86_64-apple-darwin"; fi ;;
    Linux)
      if [ "$ARCH" = "aarch64" ]; then TRIPLE="aarch64-unknown-linux-gnu"
      else TRIPLE="x86_64-unknown-linux-gnu"; fi ;;
    *) echo "[setup] ✗ 不支持的操作系统: $OS（仅 macOS / Linux；Windows 用 setup.bat）"; exit 1 ;;
  esac

  API="https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest"
  if [ -n "${GARGANTUA_PY_RELEASE:-}" ]; then
    TAG="$GARGANTUA_PY_RELEASE"
    REL="$(curl -fsSL "https://api.github.com/repos/astral-sh/python-build-standalone/releases/tags/$TAG")"
  else
    REL="$(curl -fsSL "$API")"
    TAG="$(printf '%s' "$REL" | grep -o '"tag_name": *"[^"]*"' | head -1 | sed 's/.*"\([^"]*\)"$/\1/')"
  fi
  ASSET="$(printf '%s' "$REL" | grep -o "cpython-${PY_SERIES}\.[0-9]*-${TRIPLE}-install_only.tar.gz" | head -1)"
  if [ -z "${TAG:-}" ] || [ -z "$ASSET" ]; then
    echo "[setup] ✗ 找不到匹配的 Python 安装包（series=$PY_SERIES, triple=$TRIPLE）。"
    echo "           可设 GARGANTUA_PY_RELEASE=<tag> 手动指定，或检查网络/代理。"
    exit 1
  fi
  URL="https://github.com/astral-sh/python-build-standalone/releases/download/${TAG}/${ASSET}"
  echo "[setup]   tag=$TAG  asset=$ASSET"

  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  echo "[setup]   下载 $URL"
  curl -fSL "$URL" -o "$TMP/py.tar.gz"
  mkdir -p "$PYTHON_DIR"
  tar -xzf "$TMP/py.tar.gz" -C "$PYTHON_DIR"   # 解包出顶层 python/ 目录
  rm -rf "$TMP"
  if [ ! -x "$PYTHON_DIR/python/bin/python3" ]; then
    echo "[setup] ✗ 解包后未找到 python3，安装失败。"
    exit 1
  fi
}

ensure_python

# ---- 建 venv ----
if [ ! -x "$VENV/bin/python3" ]; then
  echo "[setup] 创建隔离虚拟环境：$VENV"
  "$PYTHON_DIR/python/bin/python3" -m venv "$VENV"
fi

# ---- 装依赖 ----
PY="$VENV/bin/python3"
"$PY" -m pip install --quiet --upgrade pip
echo "[setup] 安装核心依赖（core.txt）……"
"$PY" -m pip install --quiet -r "$REQ_DIR/core.txt"

if [ "${GARGANTUA_NO_WINDOW:-0}" != "1" ]; then
  echo "[setup] 安装窗口依赖（window.txt，失败不致命）……"
  "$PY" -m pip install --quiet -r "$REQ_DIR/window.txt" \
    || echo "[setup] ⚠️ 窗口依赖安装失败（可稍后手动：pip install -r start/requirements/window.txt）"
fi
if [ "${GARGANTUA_INSTALL_DEV:-0}" = "1" ]; then
  echo "[setup] 安装开发依赖（dev.txt）……"
  "$PY" -m pip install --quiet -r "$REQ_DIR/dev.txt"
fi

# ---- 写 stamp（含依赖哈希，保证清单变化时重装）----
CUR="$(cat "$REQ_DIR"/*.txt | sha256sum | awk '{print $1}')"
printf 'ts=%s\nreq_hash=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CUR" > "$STAMP"
echo "[setup] ✓ 隔离环境就绪：$VENV"
echo "[setup]   启动：start/terminal/start.sh（或双击 start.command）"
