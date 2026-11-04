# service/prims/ — 图元（primitive）管线

图像 → 图元序列转换服务：输入图片，输出 `draw_codec.Primitive` 序列
（矩形/椭圆拟合，仓库 CSV schema）。

```
prims/
├── image_to_primitives.py   驱动（GPU 子进程 + CPU 兜底拟合器）
├── setup.sh                 构建脚本（Rust 工具链装进 environment/rust/）
├── fast_shape_render/       ★ 逐字照搬的第三方 Rust/wgpu GPU 转换器
│   └── src/ engine.rs main.rs shader.wgsl types.rs（未改动一行）
├── LICENSE                  上游 MIT 许可证（wonderfulearth）
└── README.md                本文件
```

## 两条路径

| 路径 | 实现 | 画布 | 速度 | 依赖 |
| --- | --- | --- | --- | --- |
| GPU | `fast_shape_render`（Rust/wgpu，Metal/DX12/Vulkan） | 按目标长宽比的同比例小画布（长边 256），结果等比放大 | 与步数成正比：60 轮 ~6s、275 轮 ~20s | 需构建（见下） |
| CPU 兜底 | 本目录 `image_to_primitives.py` 的 numpy 拟合器 | 任意尺寸（≤1920×1080） | ~2–6s/100 步 | 无额外依赖 |

`renderer_binary()` 优先环境变量 `GARGANTUA_SHAPE_RENDERER`，否则找
`fast_shape_render/target/release/gpu_anime_vectorizer`；都没有就走 CPU
兜底（`/api/codec/image2prims` 的 `quality=fast|slow` 控制 CPU 档位，
`engine=auto|gpu|cpu` 控制走哪条路径）。

**步数**：`max_steps` 在 GPU 路径上映射到 `SHAPE_RENDERER_MAX_ITERATIONS`
（搜索轮数，每轮最多采纳 1 个图元），所以「步数上限」在两条路径上是同一个
含义，且 GPU 耗时与步数成正比。**不再**使用 `SHAPE_RENDERER_MAX_OUTPUT_STEPS`
—— 那个变量是「必须产出 N 行，不够就报错」，图片一旦产不出就整个失败。

**画布**：转换器的 `W`/`H` 原本是编译期常量（256×256）。2026-11-02 起改成
运行时参数（`SHAPE_RENDERER_CANVAS_WIDTH/HEIGHT`，默认仍是 256×256，
`SHAPE_RENDERER_MAX_HALF_SIZE` 默认 `max(宽,高)/2`），shader 的
`CANVAS_WIDTH/HEIGHT/MAX_HALF_SIZE` 由 `build_shader_source()` 按实际尺寸
生成。于是调用方可以按目标画布的**长宽比**给一个小画布（16:9 → 256×144），
拟合全程比例正确，输出只要等比放大——不再有「先拉成正方形再拉回去」的变形。

**速度提醒**：GPU 转换器是为**批量**图片设计的，单张固定跑满 275 轮
（≈20s），比 CPU 拟合慢 5–10 倍。前端「图元绘画」页的「引擎」下拉可
显式选 CPU。

## 构建 GPU 转换器

```bash
sh service/prims/setup.sh
```

- Rust 工具链（rustup + stable toolchain）全部装进 `environment/rust/`
  （`environment/rust/pack.json` 里有这个包的清单），用
  `RUSTUP_HOME`/`CARGO_HOME` 隔离，不碰 `~/.cargo`、不改系统 PATH；
- 下载镜像自动测速选取（USTC → TUNA → 官方），可用
  `RUSTUP_DIST_SERVER` / `RUSTUP_INIT_URL` / `CARGO_CRATES_MIRROR` 覆盖；
- 首次构建约 5–15 分钟；macOS 走 Metal（需要 Xcode Command Line Tools）。

## 上游

- 仓库：<https://github.com/wonderfulearth/primitive-operation-painter>
- `fast_shape_render/` 为逐字复制（未改动一行），本项目只使用其中的 GPU
  转换器；图元 token 布局（geometrize_256_v1 + 画布 token）按
  `server/spec.py` 的 DRAW_* 常量实现，编解码在
  `service/tokens/draw_codec.py`。
- 上游仓库的其余部分（模型 / 训练脚本 / 示例素材，约 11MB）**已删除**：
  本项目一行都没用到，留着只会让目录变大、让接手的人误以为要维护它。
  需要回看时从 git 历史取（`git log -- service/prims/upstream`）。
- 转换器环境变量（`SHAPE_RENDERER_INPUT_DIR` 等）见
  `fast_shape_render/README.md`。
