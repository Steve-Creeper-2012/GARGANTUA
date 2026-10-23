#!/bin/sh
# ===========================================================================
# GARGANTUA v1 — Linux / macOS 启动入口（薄委托）
# ---------------------------------------------------------------------------
# 本脚本只做一件事：找到系统 Python，然后把全部逻辑交给 run.py
# （venv 创建 / 依赖校验 / 版本守卫 / server + 独立窗口生命周期）。
#
# 用法：
#   ./run.sh                  启动 server + 独立桌面窗口
#   ./run.sh --port 9000      指定端口
#   ./run.sh --no-window      只起 server
#   ./run.sh --test           跑全量测试（逐套件分进程）
#   ./run.sh --init NAME      创建模型 models/NAME
#   ./run.sh --train NAME     训练
#   ./run.sh --infer NAME ["提示词"]
#
# 2026-10-20：本脚本收敛为薄委托。此前它自带一套 venv 创建逻辑，会绕过
# run.py 的 Python≤3.12 守卫，且引用一个并不存在的 requirements.txt
# （依赖从未装上）。
# ===========================================================================

set -e
cd "$(dirname "$0")"

# 候选解释器顺序：3.12 优先（torch 2.2.2 最高支持 3.12，spec 第 7 节
# 约定 3.12.14），其次 python3，最后 python。
SYS_PY=""
for c in python3.12 python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
        if "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; then
            SYS_PY="$c"
            break
        fi
        # 记住第一个可用的：万一整机只有非 3.12，交给 run.py 去提示
        if [ -z "$SYS_PY" ]; then
            SYS_PY="$c"
        fi
    fi
done

if [ -z "$SYS_PY" ]; then
    echo "[GARGANTUA] 找不到系统 Python"
    echo "请先安装 Python 3.12（spec 第 7 节约定 3.12.14）"
    exit 1
fi

# 版本提示（不阻塞：真正的守卫和回退由 run.py 决定）
if ! "$SYS_PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
    echo "[GARGANTUA] 提示: $SYS_PY 版本过新，torch==2.2.2 没有预编译轮子"
fi

exec "$SYS_PY" run.py "$@"
