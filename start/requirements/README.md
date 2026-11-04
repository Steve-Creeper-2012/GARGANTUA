# GARGANTUA v1 — 依赖分类（start/requirements）

2026-11-01 起（steve 定案）只有两份清单，是「这个项目需要什么」的唯一
声明处（`server/reqs.py` 读这里，主页「环境与依赖」按它渲染）：

| 文件 | 内容 | 何时安装 |
| --- | --- | --- |
| `must.txt` | python（解释器）+ pywebview（独立窗口） | **启动必装**（轻量，setup 脚本默认装；干净环境首次进入也只装它） |
| `necessary.txt` | torch / numpy / pillow / safetensors / pytest / ffmpeg / rust | 正常工作需要，**不影响启动**——主页「环境与依赖」里按需安装 |

## 版本写法：下限 + 向后兼容
- 每个依赖写**下限**（`>=`）：默认装下限版本。
- 编译器/运行时向后兼容：用户可以装**更高**版本换更好的支持与优化，
  主页会把可用版本列出来供选，**不强求**。
- 硬上限写成 `>=下限,<上限`（如 `numpy>=1.24,<2`：torch 2.2.2 不支持
  numpy 2.x）——上限是硬约束，主页不会把违规版本列进可选项。
- 装可选依赖（cv2 / soundfile 等）必须带 `-c necessary.txt` 约束
  （见 `server/server_util.py:optional_install_cmd`），否则 numpy 会被
  顶到 2.x，悄悄破坏 torch 2.2.2 的主依赖契约。

## 安装顺序（由 setup 脚本自动完成）
1. 取 Python 3.12（本机现成 → 包管理器 → python-build-standalone 下载，
   镜像自动测速）→ `environment/python/`
2. 建虚拟环境 → `environment/venv/`
3. `pip install -r must.txt`（必装）；设 `GARGANTUA_FULL_SETUP=1` 再装
   `necessary.txt`（CI / 容器一步到位用）
4. 写 stamp 文件，下次启动跳过重复安装

缺 necessary 里的依赖时：server 照常启动，对应功能（训练 / 推理 /
图元转换 / 测试）不可用，主页对应卡片显示「未安装」和安装按钮。
