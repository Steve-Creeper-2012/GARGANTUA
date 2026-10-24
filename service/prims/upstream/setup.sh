#!/bin/sh
# ===========================================================================
# GARGANTUA — primitive-operation-painter GPU 转换器启动依赖一键配置
# ---------------------------------------------------------------------------
# 本目录是 wonderfulearth/primitive-operation-painter 的 vendored 副本
# （代码未改动，照搬第三方）：
#   - fast_shape_render/   Rust + wgpu 的 GPU image-to-sequence converter
#                          （读图片 → 拟合矩形/椭圆序列 → 写 CSV）
#   - 其余 *.py             Python 侧：模型 / 训练 / 生成 / 可视化
#                          （token 布局 token_layout.py 已被 GARGANTUA
#                           spec.py 照搬为 DRAW 区）
#
# 本脚本做三件事：
#   1. 检查/安装 Rust 工具链（rustup，https://rustup.rs ）
#   2. cargo build --release 构建 GPU 转换器（macOS 走 Metal，
#      Windows 走 DirectX 12，Linux 走 Vulkan）
#   3. pip 安装 Python 侧依赖（pandas / matplotlib 等，torch 用现有的）
#
# 用法：  sh setup.sh
# ===========================================================================
set -e
cd "$(dirname "$0")"

echo "==> [1/3] Rust 工具链"
if ! command -v cargo >/dev/null 2>&1; then
    echo "    未找到 cargo，安装 rustup（minimal profile）…"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
fi
cargo --version
rustc --version

echo "==> [2/3] 构建 GPU 转换器（首次约 5–15 分钟，之后增量）"
(cd fast_shape_render && cargo build --release)
BIN=fast_shape_render/target/release/gpu_anime_vectorizer
if [ ! -x "$BIN" ]; then
    echo "!! 构建产物不存在: $BIN" >&2
    exit 1
fi
echo "    OK: $BIN"

echo "==> [3/3] Python 依赖（torch/numpy 用 GARGANTUA 现有环境，不重复装）"
PY="${PYTHON:-python3}"
"$PY" -m pip install -r requirements.txt

echo
echo "全部就绪。GARGANTUA 会自动发现该二进制（third_party/primitive-"
echo "operation-painter/fast_shape_render/target/release/gpu_anime_vectorizer），"
echo "也可用环境变量 GARGANTUA_SHAPE_RENDERER 显式指定路径。"
