# ===========================================================================
# GARGANTUA v1 — service/prims/：primitive（图元）管线服务
# ---------------------------------------------------------------------------
# 图像 → 图元序列（Primitive）转换服务。两部分：
#
#   image_to_primitives.py   我们的驱动：GPU 转换器子进程驱动 + CPU 兜底
#                             拟合器（纯 numpy，算法与第三方对齐）。
#                             被 server_codecs 的 /api/codec/image2prims
#                             调用；也可独立运行。
#   fast_shape_render/       逐字照搬的第三方 Rust/wgpu GPU 转换器
#                             （wonderfulearth/primitive-operation-painter，
#                             未改动一行；LICENSE 见同目录 LICENSE）。
#                             用 ./setup.sh 构建（Rust 工具链装在
#                             environment/rust/，与系统隔离）。
#   upstream/                上游仓库完整留档（其模型/训练代码，本项目
#                             不用；保留是为了「不丢原始材料」）。注意
#                             其中 upstream/setup.sh 是上游原版（会把
#                             rustup 装进系统）——请用本目录的 setup.sh。
#
# 词表侧：图元 token 的布局（geometrize_256_v1）在 server/spec.py，
# 编解码在 service/tokens/draw_codec.py。
# ===========================================================================
"""prims 服务：图像 → 图元序列（GPU 转换器 + CPU 兜底）。"""
