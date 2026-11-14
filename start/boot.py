# ===========================================================================
# GARGANTUA — start/boot.py：路径引导（唯一事实来源）
# ---------------------------------------------------------------------------
# 项目目录树（2026-11-12 整理）：
#   start/       启动链（terminal 脚本 + 分类 requirements + 本引导模块）
#   launcher/    POSIX 一键启动器（由根目录 启动.sh / 启动.command 调用）
#   server/      冻结契约（spec.py）+ 模型基础设施 + 全部 server 文件
#   service/     各服务（infer / train / init / tokens / prims / web / test /
#                window / cli ...），每个服务一个目录，按需启动
#   docs/        文档索引与历史评估归档（唯一文档入口见 docs/README.md）
#   experiments/ 后续测试对比数据的预留目录（暂不放数据）
#   vocabs/      版本化基础词表 base<ver>/{vocab.vfc, config.json, ...}
#   models/      models/NAME/{config.json, weights/, vocab/, momw_state/}
#   data/ logs/  运行时数据 / 日志 / 临时文件（均被 .gitignore 忽略）
#
# 旧版所有代码同根平铺，`import spec` 靠 cwd 命中；分目录后由本模块把
# 各代码根放上 sys.path——**业务文件的 import 语句零改动**：
#   import spec                  → server/spec.py
#   from common import ...       → server/common.py
#   from tokens.x import ...     → service/tokens/
#   from infer import generate   → service/infer/
#   import server                → server/server.py
#
# 两种启用方式（都要）：
#   1. 启动器脚本 export PYTHONPATH（覆盖子进程，server 派生窗口用）；
#   2. 每个入口 .py 顶部的引导头（直接 `python server/server.py` 时兜底）。
# ===========================================================================
"""GARGANTUA 路径引导：把 server/ 与各 service/ 目录放上 sys.path。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 需要放上 sys.path 的目录（相对项目根；顺序即 import 优先级）
PATH_DIRS = (
    "server",
    "service",
    "service/infer",
    "service/train",
    "service/init",
    "service/test",
)

# 入口文件顶部的标准引导头（复制到各入口 .py，放在任何项目内 import 之前；
# 注意：若文件有模块 docstring 或 `from __future__`，头要放在它们之后）：
#     import sys as _sys
#     from pathlib import Path as _Path
#     for _anc in _Path(__file__).resolve().parents:
#         if (_anc / "start" / "boot.py").is_file():
#             _sys.path.insert(0, str(_anc / "start"))
#             break
#     else:
#         raise SystemExit("找不到 GARGANTUA 项目根（start/boot.py）")
#     import boot
#     boot.bootstrap()
BOOT_HEADER = '''import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "start" / "boot.py").is_file():
        _sys.path.insert(0, str(_anc / "start"))
        break
else:
    raise SystemExit("找不到 GARGANTUA 项目根（start/boot.py）")
import boot
boot.bootstrap()
'''


def project_root() -> Path:
    """向上查找含 start/boot.py 的目录 = 项目根。"""
    here = Path(__file__).resolve()
    for anc in here.parents:
        if (anc / "start" / "boot.py").is_file():
            return anc
    raise RuntimeError("找不到 GARGANTUA 项目根（start/boot.py）")


def pythonpath() -> str:
    """启动器 export 用的 PYTHONPATH（os.pathsep 分隔）。"""
    root = project_root()
    return os.pathsep.join(
        [str(root / "start")] + [str(root / d) for d in PATH_DIRS])


def bootstrap() -> None:
    """把 PATH_DIRS 放上 sys.path（幂等）。"""
    root = project_root()
    for rel in PATH_DIRS:
        p = str(root / rel)
        if p not in sys.path:
            sys.path.insert(0, p)
