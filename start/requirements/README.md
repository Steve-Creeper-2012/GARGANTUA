# GARGANTUA v1 — 依赖分类（start/requirements）

2026-11-01 起（steve 定案）只有两份清单，是「这个项目需要什么」的唯一
声明处（`server/reqs.py` 读这里，主页「环境与依赖」按它**只读渲染**）：

| 文件 | 内容 | 何时安装 |
| --- | --- | --- |
| `must.txt` | python（解释器）+ pywebview（独立窗口） | **启动必装**（轻量，setup 脚本默认装；干净环境首次进入也只装它） |
| `necessary.txt` | torch / numpy / pillow / safetensors / pytest / ffmpeg / rust | 正常工作需要，**不影响启动**——由终端脚本按需安装 |

## 版本写法：下限 + 向后兼容
- 每个依赖写**下限**（`>=`）：默认装下限版本。
- 编译器/运行时向后兼容：用户可以装**更高**版本换更好的支持与优化，不强求。
- 硬上限写成 `>=下限,<上限`（如 `numpy>=1.24,<2`：torch 2.2.2 不支持
  numpy 2.x）——上限是硬约束，可选版本列表不会把违规版本列进去。
- 装可选依赖（cv2 / soundfile 等）必须带 `-c necessary.txt` 约束
  （见 `server/server_util.py:optional_install_cmd`），否则 numpy 会被
  顶到 2.x，悄悄破坏 torch 2.2.2 的主依赖契约。

## 安装入口（唯一路径）
所有安装 / 更新都走 `start/setup_env.py`，**不复用系统 Python、不写系统目录**：

```bash
python3 start/setup_env.py install    # 装齐 must + necessary
python3 start/setup_env.py update     # 升到允许的最新版
GARGANTUA_FULL_SETUP=1 ./启动.sh      # 让启动引导一次装齐（CI / 容器）
```

1. 下载 standalone CPython 3.12 到 `environment/python/`（镜像自动测速，
   缓存命中可完全离线；**绝不探测或复用系统 Python、绝不用系统包管理器**）；
2. 建虚拟环境 `environment/venv/`（`pyvenv.cfg` 必须指向项目内）；
3. `pip install -r must.txt`；`GARGANTUA_FULL_SETUP=1` 时再装 `necessary.txt`；
4. 写依赖清单哈希 + OS/架构指纹到 `environment/venv/.gargantua_setup_stamp`，
   下次启动秒过，换平台自动重建。

**GUI 只读**（2026-11-12 核对 `service/web/index_source.html`）：主页「环境与依赖」
只展示状态，不提供安装 / 更新按钮，也不自动检查更新。窗口侧的手动安装入口
（`server/env_install.py`）同样转发到上面这一个安装器，不存在第二套逻辑。

缺 `necessary.txt` 里的依赖时：server 照常启动，对应功能（训练 / 推理 /
图元转换 / 测试）不可用，主页对应卡片显示「未安装」。
