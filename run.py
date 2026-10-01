#!/usr/bin/env python3
# ===========================================================================
# GARGANTUA v1 — 跨平台一键启动（Windows / Linux / macOS）
# 用法：
#   python run.py                  启动网页工作台 http://127.0.0.1:8712
#   python run.py --port 9000      指定端口
#   python run.py --no-open        不自动打开浏览器
#   python run.py --test           跑全量测试
#   python run.py --init NAME      创建模型 models/NAME
#   python run.py --train NAME     用 data/ 训练 models/NAME
#   python run.py --infer NAME     用 models/NAME 推理（提示词用环境变量 PROMPT）
# ===========================================================================
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))

VENV_DIR = os.path.join(ROOT, ".venv")
WIN_PY = os.path.join(VENV_DIR, "Scripts", "python.exe")
NIX_PY = os.path.join(VENV_DIR, "bin", "python")

REQUIREMENTS = ["torch==2.2.2", "numpy<2", "pillow", "safetensors"]


def find_python() -> str | None:
    """优先用项目 venv，否则用当前解释器。"""
    for cand in (WIN_PY, NIX_PY):
        if os.path.isfile(cand):
            return cand
    return sys.executable


def ensure_venv() -> str:
    """没有 venv 就建一个并装依赖。"""
    py = find_python()
    if py != sys.executable:
        return py
    print(f"[run] 未检测到 venv，正在创建 {VENV_DIR} …")
    subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    py = find_python()
    pip = [py, "-m", "pip", "install", "--upgrade", "pip"]
    subprocess.check_call(pip)
    subprocess.check_call([py, "-m", "pip", "install"] + REQUIREMENTS)
    return py


def check_deps(py: str) -> bool:
    code = ("import importlib,sys;"
            "mods=['torch','numpy','PIL','safetensors'];"
            "sys.exit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)")
    return subprocess.call([py, "-c", code],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) == 0


def cmd_serve(py: str, port: int, no_open: bool):
    if not no_open:
        url = f"http://127.0.0.1:{port}"
        # 延迟打开，等 server 起来
        def _open():
            time.sleep(2)
            try:
                webbrowser.open(url)
            except Exception:
                print(f"[run] 请手动打开 {url}")
        import threading
        threading.Thread(target=_open, daemon=True).start()
    os.execv(py, [py, os.path.join(ROOT, "server.py"), "--port", str(port)])


def main():
    ap = argparse.ArgumentParser(description="GARGANTUA v1 启动器")
    ap.add_argument("--port", type=int, default=8712)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--init", metavar="NAME")
    ap.add_argument("--train", metavar="NAME")
    ap.add_argument("--infer", metavar="NAME")
    args = ap.parse_args()

    py = find_python()
    if not check_deps(py):
        print("[run] 当前 Python 缺少依赖，尝试创建/使用项目 venv …")
        py = ensure_venv()
        if not check_deps(py):
            print("[run] 依赖安装失败，请手动执行：")
            print(f"  {py} -m pip install {' '.join(REQUIREMENTS)}")
            sys.exit(1)

    if args.test:
        cmd = [py, "-m", "unittest", "discover", "-s", "tests",
               "-p", "test_*.py", "-v"]
    elif args.init:
        cmd = [py, "init.py", "--name", args.init]
    elif args.train:
        cmd = [py, "train.py", "--model", f"models/{args.train}",
               "--data", "data"]
    elif args.infer:
        prompt = os.environ.get("PROMPT", "你好")
        cmd = [py, "infer.py", "--model", f"models/{args.infer}",
               "--prompt", prompt]
    else:
        cmd_serve(py, args.port, args.no_open)
        return

    os.chdir(ROOT)
    os.execv(py, cmd)


if __name__ == "__main__":
    main()
