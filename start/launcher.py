# ===========================================================================
# GARGANTUA v1 — start/launcher.py：启动编排（唯一终端入口之后的一道）
# ---------------------------------------------------------------------------
# 终端脚本（start/terminal/start.sh|.bat|.command）setup 完隔离环境后，
# 由本模块把 GARGANTUA「跑起来」。职责：
#   1. 环境预检：当前解释器 ≥3.12、torch/numpy/PIL/safetensors 可导入；
#   2. 窗口完整性校验：对 service/window/window.py 算 sha256，与
#      start/terminal/window.sha256 比对；首次运行写基准，之后不一致则
#      弹窗告知用户（可跳过：--no-verify 或弹窗里选择继续）；
#   3. 拉起 server.py（子进程，HTTP 服务 + 训练老板）；
#   4. 等 server 就绪后拉起 window.py（子进程，独立桌面窗口）；
#   5. 监听退出信号，按 window → server 顺序收尸。
#
# 进程模型（沿用原 run.py 语义，server 是老板，window 是它生的 UI 子进程；
# 这里 launcher 是更外层的编排者，server/window 都是它的子进程）：
#   launcher ── server.py（训练/推理在此）
#            └─ window.py（GUI，崩了不影响训练）
#
# 纯标准库实现：本模块不 import torch/spec/model，确保即使依赖没装全，
# 预检与提示也能跑出来（而不是一上来就 ImportError 黑屏）。
# ===========================================================================
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
import urllib.request


def project_root() -> str:
    """向上查找含 start/boot.py 的目录 = 项目根。"""
    here = os.path.abspath(__file__)
    cur = os.path.dirname(here)
    while True:
        if os.path.isfile(os.path.join(cur, "start", "boot.py")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError("找不到 GARGANTUA 项目根（start/boot.py）")
        cur = parent


ROOT = project_root()
SHA256_PATH = os.path.join(ROOT, "start", "terminal", "window.sha256")
WINDOW_PY = os.path.join(ROOT, "service", "window", "window.py")
SERVER_PY = os.path.join(ROOT, "server", "server.py")


# ---------------------------------------------------------------------------
# 启动辅助（被测试复用）
# ---------------------------------------------------------------------------
def wait_port(port: int, host: str = "127.0.0.1", timeout: float = 30.0) -> bool:
    """ 端口在 timeout 内可连即返回 True（server 起的信号）。 """
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def wait_status(port: int, host: str = "127.0.0.1", timeout: float = 60.0) -> bool:
    """ 轮询 /api/status，200 即就绪。 """
    url = f"http://{host}:{port}/api/status"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def _kill(proc, grace: float = 5.0) -> None:
    """ 优雅终止子进程：先 SIGTERM，超时再 SIGKILL。 """
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        return
    try:
        proc.wait(timeout=grace)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def check_deps(python_exe: str) -> bool:
    """ 探针：指定解释器能否 import 主依赖（torch/numpy/PIL/safetensors）。

    用子进程跑 `import importlib.util` + find_spec，避免探针自身不绑定
    importlib.util 而恒判缺依赖（2026-10-20 真实踩过的坑，导致每次启动
    都重装 torch）。 """
    probe = (
        "import importlib.util,sys;"
        "mods=['torch','numpy','PIL','safetensors'];"
        "ok=all(importlib.util.find_spec(m) is not None for m in mods);"
        "sys.exit(0 if ok else 1)"
    )
    try:
        r = subprocess.run([python_exe, "-c", probe],
                           capture_output=True, text=True, timeout=120)
    except Exception:
        return False
    return r.returncode == 0


# ---------------------------------------------------------------------------
# 环境预检
# ---------------------------------------------------------------------------
def precheck(python_exe: str) -> bool:
    """ 检查解释器版本与主依赖；缺关键依赖返回 False（致命）。 """
    # 版本
    try:
        out = subprocess.run([python_exe, "-c",
                              "import sys;print('%d.%d'%sys.version_info[:2])"],
                             capture_output=True, text=True, timeout=30)
        ver = out.stdout.strip()
        major, minor = (int(x) for x in ver.split("."))
        py_ok = (major, minor) >= (3, 12)
    except Exception:
        print("[launcher] 无法读取 Python 版本，请检查解释器。")
        return False
    if not py_ok:
        print(f"[launcher] ✗ Python {ver} < 3.12，GARGANTUA 要求 ≥3.12"
              f"（torch 2.2.2 不支持 3.13 预编译轮子）。")
        return False
    print(f"[launcher] Python {ver} ✓")

    # 主依赖
    if not check_deps(python_exe):
        print("[launcher] ✗ 主依赖缺失（torch/numpy/PIL/safetensors）。")
        print(f"           请先安装：{python_exe} -m pip install "
              f"-r start/requirements/core.txt")
        return False
    print("[launcher] 主依赖（torch/numpy/PIL/safetensors）✓")
    return True


# ---------------------------------------------------------------------------
# 窗口完整性校验（sha256 + 弹窗）
# ---------------------------------------------------------------------------
def _sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _popup(title: str, message: str) -> bool:
    """ 跨平台弹窗；返回 True 表示「继续」（False = 跳过窗口启动）。

    优先级：tkinter → macOS osascript → Linux zenity → 纯文本 stdin。
    无图形会话时降级为命令行提示（用户回车继续，输入 s/S 跳过窗口）。 """
    # 1) tkinter（有图形会话时最稳）
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        # 仅提示 + 继续；这里不提供「跳过窗口」的复杂按钮，保持单 OK。
        messagebox.showwarning(title, message)
        root.destroy()
        return True
    except Exception:
        pass

    # 2) macOS 原生对话框
    if sys.platform == "darwin":
        try:
            script = (
                f'display dialog "{message}" with title "{title}" '
                f'buttons {{"继续", "跳过窗口"}} default button "继续"'
            )
            r = subprocess.run(["osascript", "-e", script],
                               capture_output=True, text=True, timeout=30)
            return "跳过" not in (r.stdout or "")
        except Exception:
            pass

    # 3) Linux zenity
    if sys.platform.startswith("linux"):
        try:
            r = subprocess.run(
                ["zenity", "--question", "--title", title,
                 "--text", message, "--ok-label", "继续",
                 "--cancel-label", "跳过窗口"],
                capture_output=True, text=True, timeout=30)
            return r.returncode == 0
        except Exception:
            pass

    # 4) 纯文本降级
    print(f"\n⚠️  {title}\n{message}")
    try:
        ans = input("回车继续启动窗口；输入 s 跳过窗口启动：[ ] ").strip().lower()
    except EOFError:
        return True
    return ans != "s"


def verify_window_integrity(allow_create: bool = True) -> bool:
    """ 校验 window.py 完整性；返回是否「应继续启动窗口」。

    - 基准文件不存在 → 写入当前哈希（首次运行）。
    - 哈希一致 → 静默通过。
    - 哈希不一致 → 弹窗告知（--no-verify 时只打印），返回继续标志。 """
    if not os.path.isfile(WINDOW_PY):
        print("[launcher] 未找到 window.py，跳过窗口校验。")
        return False
    cur = _sha256_of(WINDOW_PY)
    if not os.path.isfile(SHA256_PATH):
        with open(SHA256_PATH, "w", encoding="utf-8") as f:
            f.write(cur + "\n")
        print("[launcher] 已写入 window.py 完整性基准（首次运行）。")
        return True
    with open(SHA256_PATH, "r", encoding="utf-8") as f:
        base = f.read().strip()
    if cur == base:
        return True
    msg = ("检测到 service/window/window.py 与上次启动时的内容不一致"
           "（sha256 不匹配）。\n可能是文件被修改或损坏。"
           "是否仍启动窗口？")
    return _popup("GARGANTUA 窗口完整性警告", msg)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GARGANTUA 启动编排")
    ap.add_argument("--port", type=int, default=8712)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--log-file", default="logs/auto_train.jsonl")
    ap.add_argument("--no-window", action="store_true",
                    help="不起独立窗口（仅 HTTP server）")
    ap.add_argument("--no-verify", action="store_true",
                    help="跳过 window.py 完整性弹窗（CI/无 GUI 用）")
    args = ap.parse_args(argv)

    py = sys.executable

    # 1) 环境预检
    if not precheck(py):
        return 1

    # 2) 计算/导出 PYTHONPATH（server/window 子进程继承）
    boot_mod = _load_boot()
    env = dict(os.environ)
    if boot_mod is not None:
        env["PYTHONPATH"] = boot_mod.pythonpath()

    # 3) 拉起 server.py（子进程）
    server_cmd = [py, SERVER_PY,
                  "--port", str(args.port), "--host", args.host,
                  "--model-dir", args.model_dir,
                  "--data-dir", args.data_dir,
                  "--log-file", args.log_file]
    print(f"[launcher] 启动 server：{' '.join(server_cmd)}")
    server_proc = subprocess.Popen(server_cmd, env=env)
    try:
        if not wait_status(args.port, args.host):
            print(f"[launcher] ✗ server 未在 {args.port} 就绪，退出。")
            _kill(server_proc)
            return 1
        print(f"[launcher] server 就绪：http://{args.host}:{args.port}")

        # 4) 窗口完整性校验 + 拉起 window.py
        launch_window = (not args.no_window)
        if launch_window and not args.no_verify:
            launch_window = verify_window_integrity()
        if args.no_verify and launch_window and \
                not args.no_window and os.path.isfile(SHA256_PATH):
            # 仍做一次静默校验并打印（不弹窗）
            cur = _sha256_of(WINDOW_PY)
            with open(SHA256_PATH, "r", encoding="utf-8") as f:
                base = f.read().strip()
            if cur != base:
                print("[launcher] ⚠️ window.py 完整性不一致"
                      "（--no-verify，已跳过弹窗）。")

        window_proc = None
        if launch_window:
            window_cmd = [py, WINDOW_PY, "--port", str(args.port),
                         "--host", args.host]
            print(f"[launcher] 启动窗口：{' '.join(window_cmd)}")
            window_proc = subprocess.Popen(window_cmd, env=env)

        # 5) 主循环：server 退出则收尾；Ctrl-C 则双向收尸
        try:
            while True:
                rc = server_proc.poll()
                if rc is not None:
                    print(f"[launcher] server 已退出（code={rc}）。")
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[launcher] 收到中断，正在关闭……")
        finally:
            if window_proc is not None:
                _kill(window_proc)
            _kill(server_proc)
    except Exception:
        _kill(server_proc)
        raise
    return 0


def _load_boot():
    """ 把 start/ 放上 sys.path 后导入 boot（拿 pythonpath 辅助）。"""
    start_dir = os.path.join(ROOT, "start")
    if start_dir not in sys.path:
        sys.path.insert(0, start_dir)
    try:
        import boot
        return boot
    except Exception:
        return None


if __name__ == "__main__":
    sys.exit(main())
