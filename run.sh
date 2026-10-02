#!/bin/sh
# ===========================================================================
# GARGANTUA v1 — 一键启动（macOS / Linux）
# 用法：
#   ./run.sh                  启动网页工作台 http://127.0.0.1:8712
#   ./run.sh --port 9000      指定端口
#   ./run.sh --no-open        不自动打开浏览器
#   ./run.sh --test           跑全量测试（model + tokens + codecs + server）
#   ./run.sh --init NAME      创建模型 models/NAME
#   ./run.sh --train NAME     用 data/ 训练 models/NAME
#   ./run.sh --infer NAME "提示词"
# ===========================================================================

set -e
cd "$(dirname "$0")"

VENV=".venv"
PY="$VENV/bin/python"

# 找系统 Python
if [ ! -x "$PY" ]; then
    if command -v python3 >/dev/null 2>&1; then
        SYS_PY="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        SYS_PY="$(command -v python)"
    else
        echo "[GARGANTUA] 找不到系统 Python"
        echo "请先安装 Python 3.10+"
        exit 1
    fi

    echo "[GARGANTUA] 使用系统 Python: $SYS_PY"
    "$SYS_PY" -m venv "$VENV"

    if [ ! -x "$PY" ]; then
        echo "[GARGANTUA] 创建 venv 失败"
        exit 1
    fi

    # 安装依赖
    if [ -f requirements.txt ]; then
        echo "[GARGANTUA] 安装依赖..."
        "$PY" -m pip install -U pip
        "$PY" -m pip install -r requirements.txt
    fi
fi

PORT=8712
OPEN=1
CMD="serve"
ARGS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --port)
            PORT="$2"
            shift 2
            ;;
        --no-open)
            OPEN=0
            shift
            ;;
        --test)
            CMD="test"
            shift
            ;;
        --init)
            CMD="init"
            ARGS="$2"
            shift 2
            ;;
        --train)
            CMD="train"
            ARGS="$2"
            shift 2
            ;;
        --infer)
            CMD="infer"
            ARGS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

case "$CMD" in
    test)
        exec "$PY" -m unittest discover -s tests -p "test_*.py" -v
        ;;
    init)
        exec "$PY" init.py --name "$ARGS"
        ;;
    train)
        exec "$PY" train.py --model "models/$ARGS" --data data
        ;;
    infer)
        exec "$PY" infer.py --model "models/$ARGS" --prompt "${PROMPT:-你好}"
        ;;
    serve)
        if [ "$OPEN" = "1" ]; then
            (
                sleep 2
                if command -v open >/dev/null 2>&1; then
                    open "http://127.0.0.1:$PORT"
                elif command -v xdg-open >/dev/null 2>&1; then
                    xdg-open "http://127.0.0.1:$PORT"
                fi
            ) &
        fi

        exec "$PY" server.py --port "$PORT"
        ;;
esac
