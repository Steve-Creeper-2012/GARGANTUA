# environment/ — 全部依赖与工具链（与系统隔离）

本目录是 GARGANTUA 的**唯一依赖来源**：Python 解释器、虚拟环境、Rust
工具链都在这里，和系统里装的东西完全隔离。删掉整个目录不会丢任何
不可再生的东西——下面的脚本会把它们逐个装回来。

## 目录契约：一个依赖 = 一个文件夹

```
environment/
├── README.md   本文件（唯一进 git 的说明）
├── python/     standalone CPython 3.12.x（python-build-standalone）
│   └── pack.json
├── venv/       虚拟环境：start/requirements/*.txt 的全部 pip 包装在这里
│   └── pack.json
└── rust/       Rust 工具链（rustup + cargo + toolchain）
    └── pack.json
```

**硬规则**（2026-10-31 steve 定）：

1. 顶层每个子文件夹就是一个**独立的依赖包**，只装它自己那一件事。
   不许把两个东西塞进同一个文件夹，也不许让一个包的产物落到另一个包里
   （互相污染会让容器化迁移时没法按包裁剪）。
2. 每个包目录里必须有一个 `pack.json` 清单，字段：

   | 字段 | 含义 |
   | --- | --- |
   | `title` | 人话名字（主页卡片标题） |
   | `kind` | `runtime` / `venv` / `toolchain` / `binary` |
   | `why` | 一句话说明它管什么 |
   | `install` | 装它的命令（未安装时主页直接显示这句） |
   | `probe` | 相对本包的可执行文件路径；跑得起来 = 已安装，顺便取版本 |
   | `probe_files` | 没有单一可执行文件时（如 venv）用标志文件判断 |
   | `env` | 这个包要求的环境变量（键 → 相对本包的路径） |
   | `contents` | 目录内各子文件夹的用途（接手的人据此知道哪块能删） |

3. **主页「环境包」区就是这个目录的镜像**：`server_env.list_env_packs()`
   扫 `environment/*/`，多一个文件夹就多一张卡，少一个就少一张卡。
   装了东西却没人知道的情况不会再出现（rust 之前就是这样）。

## 不属于本目录的依赖

`ffmpeg`、C 编译器（Xcode Command Line Tools / MSVC）属于**系统依赖**：
它们要么是操作系统自带的，要么必须由构建镜像提供，vendored 进来既不合
法也不可靠。主页把它们单独列在「系统依赖」区，并标明容器化时要写进
基础镜像（`apt install ffmpeg` / `xcode-select --install`）。

## 重建

| 包 | 命令 |
| --- | --- |
| `python` + `venv` | `sh launcher/setup-environment.sh`（Windows：`start\terminal\setup.bat`） |
| `rust` | `sh service/prims/setup.sh` |

镜像自动测速选取，也可用环境变量手动指定（见下表）。

## 契约

1. **全部本地产物，不进 git**（`.gitignore` 只放行 README 与各包的
   `pack.json`——清单是契约，产物不是）。
2. **一切从脚本重建**：python + venv 由 `launcher/setup-environment.sh`
   （Windows：`start/terminal/setup.bat`）装，rust 由 `service/prims/setup.sh` 装。
3. **版本策略**：Python 锁 3.12 系列最新版（torch 2.2.2 最高支持
   3.12，3.13 无预编译轮子）；Rust 用 stable 最新版。
4. **零系统依赖**（2026-11-04 硬规则）：唯一允许的 Python 底座是
   `environment/python/` 里下载的 standalone CPython；启动脚本检测它、隔离
   venv 与 pywebview，缺失就自动调 setup —— 测速选镜像下载 binaries，空
   `environment/` 也能从零跑起来。`venv/pyvenv.cfg` 的 home/command 必须
   落在项目内，否则 `start/setup_env.py` 会用 standalone Python 重建 venv，
   并把旧 venv 的 site-packages 在项目内离线迁移过去（不重新下载轮子）。
   绝不探测系统 Python、绝不用 brew/apt/winget、绝不写系统目录。
5. **缓存不外溢**：下载 / pip / 编译缓存统一写 `environment/_cache/`，运行时
   临时文件写 `data/tmp/`；子进程统一继承 `paths.isolated_env()`（TMPDIR、
   PIP_CACHE_DIR、CARGO_HOME、RUSTUP_HOME、GOPATH、GOCACHE、XDG_*、
   PYTHONNOUSERSITE 全部指向项目内）。
6. **窗口内手动安装/更新**与启动引导共用 `start/setup_env.py`（入口
   `server/env_install.py`）：同样的项目内解释器、同样的镜像测速、同样的
   项目内缓存，不存在第二条用系统解释器的旁路。

## 手动指定版本/镜像

| 环境变量 | 作用 |
| --- | --- |
| `GARGANTUA_PY_RELEASE` | 指定 python-build-standalone release tag |
| `GARGANTUA_PY_VERSION` | 指定 patch 版本（如 `3.12.15`） |
| `GARGANTUA_PY_BASE_URL` | 直接指定 Python 下载镜像（跳过测速） |
| `PIP_INDEX_URL` | 自定义 pip 源（跳过测速） |
| `RUSTUP_DIST_SERVER` / `RUSTUP_UPDATE_ROOT` | 自定义 Rust 发行版镜像 |
