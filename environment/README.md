# environment/ — 全部依赖与工具链（与系统隔离）

本目录是 GARGANTUA 的**唯一依赖来源**：Python 解释器、虚拟环境、Rust
工具链都在这里，和系统里装的东西完全隔离。删掉整个目录不会丢任何
不可再生的东西——下面的脚本会把它们逐个装回来。

```
environment/
├── python/     # standalone CPython 3.12.x（python-build-standalone）
│               #   setup-environment.sh 自动下载解包；不碰系统 Python
├── venv/       # 以上述解释器建的虚拟环境
│               #   start/requirements/{core,window,dev}.txt 分类依赖
│               #   内含 .gargantua_setup_stamp（依赖清单哈希，幂等）
└── rust/       # Rust 工具链（rustup + cargo + toolchain）
                #   service/prims/setup.sh 用 RUSTUP_HOME/CARGO_HOME
                #   指到这里安装；不碰 ~/.cargo、不改系统 PATH
```

## 契约

1. **全部本地产物，不进 git**（`.gitignore` 只放行本 README）。
2. **一切从脚本重建**：
   - `launcher/setup-environment.sh`（macOS/Linux）或 `start/terminal/setup.bat`（Windows）
     装 python + venv + 依赖；镜像自动测速选取（也可用
     `GARGANTUA_PY_BASE_URL` / `PIP_INDEX_URL` 手动指定）。
   - `service/prims/setup.sh` 装 Rust 工具链并构建 GPU 图元转换器。
3. **版本策略**：Python 锁 3.12 系列最新版（torch 2.2.2 最高支持
   3.12，3.13 无预编译轮子）；Rust 用 stable 最新版。
4. **自愈**：venv 里的解释器是指向 `environment/python` 的符号链接，
   项目目录被整体移动后会悬空——setup 脚本检测到跑不动会自动重建
   venv（不是每次启动都重建，只在探测失败时）。
5. **系统里仍然需要的**：macOS 的 Xcode Command Line Tools（链接器/
   Metal SDK，Rust 构建要用）和 Windows 的 MSVC 运行库。这两样没有
   合理的办法vendored，属于构建链最底层。

## 手动指定版本/镜像

| 环境变量 | 作用 |
| --- | --- |
| `GARGANTUA_PY_RELEASE` | 指定 python-build-standalone release tag |
| `GARGANTUA_PY_VERSION` | 指定 patch 版本（如 `3.12.15`） |
| `GARGANTUA_PY_BASE_URL` | 直接指定 Python 下载镜像（跳过测速） |
| `PIP_INDEX_URL` | 自定义 pip 源（跳过测速） |
| `RUSTUP_DIST_SERVER` / `RUSTUP_UPDATE_ROOT` | 自定义 Rust 发行版镜像 |
