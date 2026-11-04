#!/usr/bin/env bash
# ===========================================================================
# GARGANTUA v1 — launcher/setup-environment.sh：隔离环境引导（macOS / Linux）
# ---------------------------------------------------------------------------
# 硬约束（2026-11-04，steve）：**绝不污染系统、绝不复用系统 Python**。
#   * 唯一允许的 Python 底座 = environment/python/ 里下载的 standalone
#     CPython；venv 的 pyvenv.cfg 必须指向它（setup_env.py 会强制校验）；
#   * 不用 brew / apt / dnf / apk，不探测系统 python3.12，不改 PATH 文件、
#     不写 ~/.cache、不写系统临时目录；
#   * 所有下载、缓存、解包、pip、工具链 HOME 都在项目文件夹内；
#   * 空 environment/ 也能跑：先用系统自带的 curl/tar 做**一次性引导**
#     下载 standalone Python（只读取系统二进制，不安装、不写系统目录），
#     之后全部由项目内 Python 执行 start/setup_env.py 完成；
#   * 把压缩包放进 environment/_cache/downloads/ 即可完全离线安装。
#
# 镜像自动测速：对 GitHub 直连 / gh-proxy / SJTU 发 Range 请求取最快者。
# 用法：./setup-environment.sh [--stage runtime|tools|all] [--check]
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="$ROOT/environment/python/python/bin/python3"
CACHE="$ROOT/environment/_cache/downloads"
CERTS="$ROOT/environment/certs"
SETUP_PY="$ROOT/start/setup_env.py"
mkdir -p "$CACHE" "$CERTS" "$ROOT/environment/python"

log() { echo "[setup] $*" >&2; }
die() { echo "[setup] ✗ $*" >&2; exit 1; }

# 启动只需 runtime（Python + venv + pywebview）；ffmpeg/node 之类的可选
# 二进制走 --stage tools 或窗口内按需安装，绝不阻塞启动。
STAGE="runtime"
CHECK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE="${2:-all}"; shift 2 ;;
    --stage=*) STAGE="${1#*=}"; shift ;;
    --check) CHECK=1; shift ;;
    *) shift ;;
  esac
done

# ---------------------------------------------------------------------------
# 1) CA 证书随文件夹走：复制一份系统 CA 到项目内（只读系统文件，零写入）
#    CPython 自带 CA 不含本机透明代理根证书，项目内 Python 的 urllib/pip
#    用这份副本才能正常下载；复制后不再依赖系统证书状态。
# ---------------------------------------------------------------------------
if [ ! -s "$CERTS/cacert.pem" ]; then
  for ca in /etc/ssl/cert.pem /etc/ssl/certs/ca-certificates.crt \
            /etc/pki/tls/certs/ca-bundle.crt; do
    if [ -r "$ca" ]; then
      cp "$ca" "$CERTS/cacert.pem"
      log "已把 CA 证书复制进项目内：$CERTS/cacert.pem"
      break
    fi
  done
fi

# ---------------------------------------------------------------------------
# 2) 引导：项目内没有 standalone Python 就下载（镜像测速）
# ---------------------------------------------------------------------------
TRIPLE=""
case "$(uname -s)" in
  Darwin)
    [ "$(uname -m)" = "arm64" ] && TRIPLE="aarch64-apple-darwin" \
                                || TRIPLE="x86_64-apple-darwin" ;;
  Linux)
    [ "$(uname -m)" = "aarch64" ] && TRIPLE="aarch64-unknown-linux-gnu" \
                                  || TRIPLE="x86_64-unknown-linux-gnu" ;;
  *) die "不支持的操作系统（Windows 请双击 启动.bat）: $(uname -s)" ;;
esac
TAG="${GARGANTUA_PY_RELEASE:-20261026}"
ASSET="${GARGANTUA_PY_ASSET:-cpython-3.12.16+${TAG}-${TRIPLE}-install_only.tar.gz}"
MANIFEST="$CACHE/pbs_release.json"

if [ ! -x "$PY" ]; then
  log "空环境：准备 standalone Python（全部下载到项目内）……"
  # release 清单只用于解析确切资产名；拿不到就用内置资产名
  if [ ! -s "$MANIFEST" ]; then
    curl -fsS -m 90 \
      "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest" \
      -o "$MANIFEST" 2>/dev/null || rm -f "$MANIFEST"
  fi
  if [ -s "$MANIFEST" ] && [ -z "${GARGANTUA_PY_ASSET:-}" ]; then
    FOUND="$(grep -o "cpython-3\.12\.[0-9]*+[0-9]*-${TRIPLE}-install_only\.tar\.gz" \
             "$MANIFEST" | sort -u | tail -1 || true)"
    [ -n "$FOUND" ] && ASSET="$FOUND"
  fi
  log "资产：$ASSET"

  M1="https://github.com/astral-sh/python-build-standalone/releases/download/$TAG"
  M2="https://gh-proxy.com/$M1"
  M3="https://mirror.sjtu.edu.cn/github-release/astral-sh/python-build-standalone/releases/download/$TAG"
  BEST=""; BEST_SPEED=0
  for m in "$M1" "$M2" "$M3"; do
    s="$(curl -sL -m 8 -r 0-1048575 -o /dev/null -w '%{speed_download}' \
         "$m/$ASSET" 2>/dev/null || echo 0)"
    case "$s" in (*[!0-9.]*|"") s=0 ;; esac
    s="${s%%.*}"; [ -n "$s" ] || s=0
    log "  测速 ${s} B/s  <- $m"
    if [ "$s" -gt "$BEST_SPEED" ]; then BEST_SPEED="$s"; BEST="$m"; fi
  done
  # 缓存命中 = 离线安装，不再需要任何网络/系统二进制（除 tar 解包）
  if [ -s "$CACHE/$ASSET" ]; then
    log "复用项目内缓存（离线）：$CACHE/$ASSET"
  else
    [ -n "$BEST" ] || die "所有镜像不可用，且缓存里没有 $ASSET"
    log "选中镜像：$BEST"
    # -sS：不要进度条（几万行刷屏），只在失败时出错误
    curl -fsSL --retry 3 --retry-delay 2 -m 900 \
      "$BEST/$ASSET" -o "$CACHE/$ASSET.part" \
      || die "standalone Python 下载失败：$BEST/$ASSET"
    mv "$CACHE/$ASSET.part" "$CACHE/$ASSET"
  fi
  tar -xzf "$CACHE/$ASSET" -C "$ROOT/environment/python"
fi

[ -x "$PY" ] || die "项目内 standalone Python 不可用：$PY"
log "项目内 Python ✓ $("$PY" -V 2>&1)"

# ---------------------------------------------------------------------------
# 3) 其余全部交给项目内 Python（跨平台同构，见 start/setup_env.py）
# ---------------------------------------------------------------------------
if [ "$CHECK" = 1 ]; then
  exec "$PY" "$SETUP_PY" --check
fi
exec "$PY" "$SETUP_PY" --stage "$STAGE"
