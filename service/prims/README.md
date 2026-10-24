# service/prims/ — 图元（primitive）管线

图像 → 图元序列转换服务：输入图片，输出 `draw_codec.Primitive` 序列
（矩形/椭圆拟合，仓库 CSV schema）。

```
prims/
├── image_to_primitives.py   驱动（GPU 子进程 + CPU 兜底拟合器）
├── setup.sh                 构建脚本（Rust 工具链装进 environment/rust/）
├── fast_shape_render/       ★ 逐字照搬的第三方 Rust/wgpu GPU 转换器
│   └── src/ engine.rs main.rs shader.wgsl types.rs（未改动一行）
├── upstream/                上游仓库完整留档（模型/训练代码，本项目不用）
├── LICENSE                  上游 MIT 许可证（wonderfulearth）
└── README.md                本文件
```

## 两条路径

| 路径 | 实现 | 速度 | 依赖 |
| --- | --- | --- | --- |
| GPU | `fast_shape_render`（Rust/wgpu，Metal/DX12/Vulkan） | 快 | 需构建（见下） |
| CPU 兜底 | 本目录 `image_to_primitives.py` 的 numpy 拟合器 | fast ~2–5s/100 步 | 无额外依赖 |

`renderer_binary()` 优先环境变量 `GARGANTUA_SHAPE_RENDERER`，否则找
`fast_shape_render/target/release/gpu_anime_vectorizer`；都没有就走 CPU
兜底（`/api/codec/image2prims` 的 `quality=fast|slow` 控制 CPU 档位）。

## 构建 GPU 转换器

```bash
sh service/prims/setup.sh
```

- Rust 工具链（rustup + stable toolchain）全部装进 `environment/rust/`，
  用 `RUSTUP_HOME`/`CARGO_HOME` 隔离，不碰 `~/.cargo`、不改系统 PATH；
- 下载镜像自动测速选取（USTC → TUNA → 官方），可用
  `RUSTUP_DIST_SERVER` / `RUSTUP_INIT_URL` / `CARGO_CRATES_MIRROR` 覆盖；
- 首次构建约 5–15 分钟；macOS 走 Metal（需要 Xcode Command Line Tools）。

## 上游

- 仓库：<https://github.com/wonderfulearth/primitive-operation-painter>
- `fast_shape_render/` 与 `upstream/` 均为逐字复制（已验证字节一致），
  本项目只使用其中的 GPU 转换器；图元 token 布局（geometrize_256_v1）
  按 `server/spec.py` 的 DRAW_* 常量实现，编解码在
  `service/tokens/draw_codec.py`。
- 转换器环境变量（`SHAPE_RENDERER_INPUT_DIR` 等）见
  `fast_shape_render/README.md`。
