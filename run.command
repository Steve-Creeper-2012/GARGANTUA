#!/bin/sh
# ===========================================================================
# GARGANTUA v1 — macOS 双击启动入口（.command）
# ---------------------------------------------------------------------------
# Finder 里双击本文件即可：macOS 会自动用 Terminal.app 执行它，然后交给
# run.py 起 server + 独立桌面窗口（pywebview，不打开浏览器）。
# 逻辑与 run.sh 完全一致（同一个薄委托），差别只是这个后缀能让 Finder
# 识别为「可双击运行」。
#
# 想开机自启：系统设置 → 通用 → 登录项 → 添加本文件。
# ===========================================================================

# 无论用户从哪个目录双击，都回到脚本所在目录
cd "$(dirname "$0")" || exit 1

SELF_DIR="$(pwd)"

# 候选解释器：3.12 优先（torch 2.2.2 最高支持 3.12，spec 第 7 节约定 3.12.14）
SYS_PY=""
for c in python3.12 python3 python; do
    if command -v "$c" >/dev/null 2>&1; then
        if "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; then
            SYS_PY="$c"
            break
        fi
        if [ -z "$SYS_PY" ]; then
            SYS_PY="$c"
        fi
    fi
done

if [ -z "$SYS_PY" ]; then
    echo "[GARGANTUA] 找不到系统 Python。"
    echo "请先安装 Python 3.12（https://www.python.org/downloads/macos/），"
    echo "或执行:  xcode-select --install && brew install python@3.12"
    echo ""
    echo "按回车键关闭窗口…"
    read _
    exit 1
fi

if ! "$SYS_PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
    echo "[GARGANTUA] 提示: $SYS_PY 版本过新，torch==2.2.2 没有预编译轮子。"
fi

echo "[GARGANTUA] 使用 $SYS_PY ($SELF_DIR)"
echo "[GARGANTUA] 首次启动会自动创建 .venv 并安装依赖，请耐心等待。"
echo ""

"$SYS_PY" run.py "$@"
CODE=$?

echo ""
echo "[GARGANTUA] 已退出（code=$CODE）。按回车键关闭窗口…"
read _
exit $CODE
