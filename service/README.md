# service/ — 服务层约定

`server/` 是常驻的轻量 HTTP 服务；`service/` 下的每个目录是一个**按需拉起的
子进程服务**（或一个独立功能层）。重活（模型权重、训练、图元拟合）都在这里，
server 本体不加载权重。子进程协议统一为 JSON-lines over stdin/stdout：
stdout 只准写 JSON 行，人肉日志走 stderr，退出条件为 `shutdown` / stdin EOF /
空闲超时。

| 目录 | 职责 | 主要入口 |
| --- | --- | --- |
| `infer/` | 自回归生成、token-id 存档、多模态 prompt 解码 | `infer.py` |
| `init/` | 建模（CLI / `--json` 一次性重活） | `init.py` |
| `train/` | 训练循环、队列守护、CLIP 对比训练与数据集文件夹 v1 | `train.py` / `auto_train.py` / `train_step.py` / `clip*.py` |
| `tokens/` | 各模态编解码、token-6 位流、`.vfc` 词表 | `tok6.py` / `vfc.py` / `vocab.py` / `enctok.py` … |
| `prims/` | 图像 → 图元序列（GPU 子进程 + numpy CPU 兜底） | `image_to_primitives.py` |
| `web/` | 单页前端、`minify.py`、可插拔显示插件 | `index_source.html` → `index.html` |
| `cli/` | `./gargantua` 终端接口：命名命令 / 离线命令 / 端点映射表 | `gargantua.py` / `local.py` / `api_map.py` |
| `window/` | pywebview 独立桌面窗口 | `window.py` |
| `calib/` | 本地标定脚本（工作区产物，非服务） | `momw_calibrate.py` |
| `test/` | 回归套件 + 静态门 | `run_tests.py` |

## `output/` 产物目录（2026-11-03 约定）

**每个服务一个 `output/`**：凡是「跑一次会生成、用户可能想回头找」的东西都放
这里，不散落到项目根、临时目录或服务自己的源码目录。

- 目录内的一切都是本地产物，由 `.gitignore` 的 `service/*/output/` 规则忽略；
- 可以整个删掉，服务下次运行会用 `os.makedirs(..., exist_ok=True)` 自行重建；
- 需要长期保留的产物由用户自己拷走——这里等同于「工作区」，不是归档；
  真正要留作对比的数据放根目录 `experiments/`。

当前唯一实际落盘的用法：`prims/output/jobs/<id>/` 存图元转换任务
（原图 / `job.json` / `result.json` / `log.txt`）。其余服务的 `output/` 先把
位置占住，避免以后各写各的路径。

> 2026-11-12 整理前，六个 `service/*/output/README.md` 是逐字相同的副本；
> 现已合并成本文件，消除重复，目录本身仍由运行时按需创建。
