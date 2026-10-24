#!/usr/bin/env bash
# ===========================================================================
# GARGANTUA v1 — service/prims/setup.sh：构建 GPU 图元转换器
# ---------------------------------------------------------------------------
# 职责：把 Rust 工具链装进 environment/rust/（与系统完全隔离），然后构建
# 同目录的 fast_shape_render/（逐字照搬的第三方 Rust/wgpu 转换器，
# 未改动一行）。构建产物：
#   service/prims/fast_shape_render/target/release/gpu_anime_vectorizer
#
# 隔离手段（不碰系统的任何状态）：
#   RUSTUP_HOME / CARGO_HOME 全指向 environment/rust/ 下；
#   rustup-init 用 --no-modify-path（不改 shell 配置、不改 PATH 文件）；
#   crates.io 走国内 sparse 镜像（写进隔离 CARGO_HOME 的 config.toml）。
#
# 镜像【自动测速】：对候选镜像并行/顺序探测，取最快的；全部可用环境变量
# 覆盖：RUSTUP_DIST_SERVER / RUSTUP_UPDATE_ROOT / RUSTUP_INIT_URL /
# CARGO_CRATES_MIRROR。
#
# 前置：macOS 需要 Xcode Command Line Tools（链接器 + Metal SDK），
# Windows 需要 MSVC 生成工具——这两项属于构建链最底层，无法 vendored。
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUSTUP_HOME="$ROOT/environment/rust/rustup"
CARGO_HOME="$ROOT/environment/rust/cargo"
export RUSTUP_HOME CARGO_HOME

PROBE_TIMEOUT=8
log() { echo "[prims] $*" >&2; }
die() { echo "[prims] ✗ $*" >&2; exit 1; }

_probe_speed() {
  # $1=URL → stdout 输出速率整数（bytes/s；失败 0）
  local speed
  speed="$(curl -sL --max-time "$PROBE_TIMEOUT" -r 0-1048575 \
           -o /dev/null -w "%{speed_download}" "$1" 2>/dev/null || echo 0)"
  case "$speed" in (*[!0-9]*|"") speed=0 ;; esac
  echo "$speed"
}

pick_fastest() {
  # $@=候选 URL（同资源不同镜像）；回显最快者（全失败回显空）
  local best="" best_score=-1 url score
  for url in "$@"; do
    score="$(_probe_speed "$url")"
    log "  测速 ${score} B/s  ← ${url}"
    if [ "$score" -gt "$best_score" ]; then
      best_score="$score"; best="$url"
    fi
  done
  [ "$best_score" -gt 0 ] && echo "$best" || true
}

# ---- 平台三元组（rustup-init 分发名）----
OS="$(uname -s)"; ARCH="$(uname -m)"
case "$OS" in
  Darwin)
    [ "$ARCH" = "arm64" ] && RTRIPLE="aarch64-apple-darwin" \
                          || RTRIPLE="x86_64-apple-darwin" ;;
  Linux)
    [ "$ARCH" = "aarch64" ] && RTRIPLE="aarch64-unknown-linux-gnu" \
                            || RTRIPLE="x86_64-unknown-linux-gnu" ;;
  MINGW*|MSYS*|CYGWIN*) RTRIPLE="x86_64-pc-windows-msvc" ;;
  *) die "不支持的操作系统: ${OS}" ;;
esac

# ---- 1) rustup-init 下载（测速选镜像）----
if [ ! -x "$CARGO_HOME/bin/cargo" ]; then
  INIT_URL="${RUSTUP_INIT_URL:-}"
  if [ -z "$INIT_URL" ]; then
    log "rustup-init 镜像测速……"
    INIT_URL="$(pick_fastest \
      "https://mirrors.ustc.edu.cn/rust-static/rustup/dist/$RTRIPLE/rustup-init" \
      "https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup/dist/$RTRIPLE/rustup-init" \
      "https://static.rust-lang.org/rustup/dist/$RTRIPLE/rustup-init")"
    [ -n "$INIT_URL" ] || INIT_URL="https://static.rust-lang.org/rustup/dist/$RTRIPLE/rustup-init"
    log "选中 rustup-init 镜像：${INIT_URL}"
  fi
  TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
  log "下载 rustup-init……"
  curl -fSL --retry 3 --max-time 300 "$INIT_URL" -o "$TMP/rustup-init" \
    || die "rustup-init 下载失败：${INIT_URL}"
  chmod +x "$TMP/rustup-init"

  # ---- 2) 发行版服务器测速（toolchain 本体从这里拉）----
  DIST="${RUSTUP_DIST_SERVER:-}"
  if [ -z "$DIST" ]; then
    log "Rust 发行版镜像测速……"
    DIST="$(pick_fastest \
      "https://mirrors.ustc.edu.cn/rust-static" \
      "https://mirrors.tuna.tsinghua.edu.cn/rustup" \
      "https://static.rust-lang.org")"
    [ -n "$DIST" ] || DIST="https://static.rust-lang.org"
    log "选中发行版镜像：${DIST}"
  fi
  export RUSTUP_DIST_SERVER="$DIST"
  export RUSTUP_UPDATE_ROOT="${RUSTUP_UPDATE_ROOT:-$DIST/rustup}"

  log "安装 Rust 工具链（stable, minimal）到 environment/rust/ ……"
  "$TMP/rustup-init" -y --no-modify-path --profile minimal \
    --default-toolchain stable --default-host "$RTRIPLE" \
    || die "rustup 安装失败"
fi

export PATH="$CARGO_HOME/bin:$PATH"
log "cargo：$(command -v cargo)（$("$CARGO_HOME/bin/cargo" --version 2>/dev/null)）"

# ---- 3) crates.io sparse 镜像（测速选；写进隔离 CARGO_HOME）----
CRATES_MIRROR="${CARGO_CRATES_MIRROR:-}"
if [ -z "$CRATES_MIRROR" ]; then
  log "crates.io 索引镜像测速……"
  CRATES_MIRROR="$(pick_fastest \
    "https://mirrors.ustc.edu.cn/crates.io-index/" \
    "https://mirrors.tuna.tsinghua.edu.cn/crates.io-index/" \
    "https://index.crates.io/")"
  [ -n "$CRATES_MIRROR" ] || CRATES_MIRROR="https://index.crates.io/"
  log "选中 crates.io 镜像：${CRATES_MIRROR}"
fi
mkdir -p "$CARGO_HOME"
cat > "$CARGO_HOME/config.toml" <<EOF
[source.crates-io]
replace-with = "gargantua-mirror"

[source.gargantua-mirror]
registry = "sparse+${CRATES_MIRROR}"

[net]
git-fetch-with-cli = true
EOF
log "已写入隔离 cargo 配置：${CARGO_HOME}/config.toml"

# ---- 4) 构建（第三方代码逐字照搬，未改动）----
log "构建 GPU 转换器（首次约 5-15 分钟）……"
cargo build --release \
  --manifest-path "$SCRIPT_DIR/fast_shape_render/Cargo.toml" \
  || die "cargo build 失败（详见上方输出）"

BIN="$SCRIPT_DIR/fast_shape_render/target/release/gpu_anime_vectorizer"
[ -x "$BIN" ] || die "构建完成但未找到二进制：${BIN}"
log "✓ 构建完成：${BIN}"
log "  image_to_primitives 会自动发现它；也可用 GARGANTUA_SHAPE_RENDERER 覆盖。"
