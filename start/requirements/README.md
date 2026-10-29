# GARGANTUA v1 — 依赖分类（start/requirements）

启动器（`setup-environment.sh` / `setup.bat`）按分类安装，避免把
可选/开发依赖污染到训练运行时。

| 文件 | 用途 | 何时安装 |
| --- | --- | --- |
| `core.txt` | torch / numpy / pillow / safetensors | **必装**（每次启动都装，幂等） |
| `window.txt` | pywebview（独立桌面窗口） | 桌面 GUI 环境才装；无 GUI 可跳过 |
| `dev.txt` | pytest（测试套件） | 仅运行测试时需要 |

## 版本契约
- `torch==2.2.2`：最后一个带 macOS x86_64 官方轮子的版本，最高支持 Python 3.12。
- `numpy<2`：torch 2.2.2 不支持 numpy 2.x，装可选依赖必须用 `core.txt` 作
  `-c` 约束（见 `server/server_util.py:optional_install_cmd`），否则会被顶到 2.x。

## 安装顺序（由 setup 脚本自动完成）
1. 下载 python-build-standalone 3.12 系列最新版 → `environment/python/`
   （镜像自动测速选取：gh-proxy / GitHub 直连 / SJTU）
2. 建虚拟环境 → `environment/venv/`
3. `pip install -r core.txt`（必要；pip 源测速选取），按需补
   `window.txt` / `dev.txt`
4. 写 stamp 文件，下次启动跳过重复安装
