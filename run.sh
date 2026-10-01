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

PY="$HOME/.workbuddy/binaries/python/envs/gargantua312/bin/python"
if [ ! -x "$PY" ]; then
  echo "找不到 GARGANTUA 运行时：$PY"
  echo "安装：python-build-standalone 3.12 + venv + pip install torch==2.2.2 \"numpy<2\" pillow safetensors"
  exit 1
fi

PORT=8712
OPEN=1
CMD="serve"
ARGS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --no-open) OPEN=0; shift ;;
    --test) CMD="test"; shift ;;
    --init) CMD="init"; ARGS="$2"; shift 2 ;;
    --train) CMD="train"; ARGS="$2"; shift 2 ;;
    --infer) CMD="infer"; ARGS="$2"; shift 2 ;;
    *) shift ;;
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
      (sleep 2; open "http://127.0.0.1:$PORT" 2>/dev/null || xdg-open "http://127.0.0.1:$PORT" 2>/dev/null) &
    fi
    exec "$PY" server.py --port "$PORT"
    ;;
esac
