# ===========================================================================
# GARGANTUA v1 — server_env.py：server 侧的环境检查与窗口完整性校验
# ---------------------------------------------------------------------------
# 2026-10-24 重构：steve 的要求是「Server 检查环境，启动 windows.py 并进行
# 哈希值校验（不匹配就弹窗告知，可跳过）」。原 start/launcher.py 的这套
# 逻辑并入 server：终端脚本（start/terminal/*）只负责搭好隔离环境并
# exec server.py，其余全部由 server 自己做。
#
# 本模块（纯标准库，server 启动早期就能跑，缺依赖时也能给出可读提示）：
#   check_deps            探针：解释器能否 import 主依赖
#   check_environment     解释器版本 + 主依赖预检（server main 第一步）
#   verify_window_integrity   window.py sha256 校验（首次写基准，不一致弹窗）
#   popup                 跨平台弹窗（tkinter → osascript → zenity → stdin）
#   wait_port / kill_proc 子进程管理小工具（测试复用）
# ===========================================================================
from __future__ import annotations

import hashlib
import os
import socket
import subprocess
import sys
import time
import urllib.request

# 主依赖：缺任何一个，server 的核心功能（模型/训练/推理）都跑不了
REQUIRED_MODULES = ("torch", "numpy", "PIL", "safetensors")

# 窗口完整性基准文件（相对项目根）：记录 window.py 上一次启动时的 sha256
WINDOW_SHA_RELPATH = os.path.join("start", "window.sha256")
WINDOW_PY_RELPATH = os.path.join("service", "window", "window.py")


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


# ---------------------------------------------------------------------------
# 环境预检
# ---------------------------------------------------------------------------
def check_deps(python_exe: str) -> bool:
    """探针：指定解释器能否 import 主依赖（torch/numpy/PIL/safetensors）。

    用子进程跑 `import importlib.util` + find_spec。注意探针里必须写
    `import importlib.util`——只 `import importlib` 不绑定子模块，
    find_spec 必抛 AttributeError → 恒判「缺依赖」→ 每次启动都重装
    torch（2026-10-20 真实踩过的坑，--test 被拖成 16 分钟假死）。
    """
    probe = (
        "import importlib.util,sys;"
        "mods=['torch','numpy','PIL','safetensors'];"
        "ok=all(importlib.util.find_spec(m) is not None for m in mods);"
        "sys.exit(0 if ok else 1)"
    )
    try:
        r = subprocess.run([python_exe, "-c", probe],
                           capture_output=True, text=True, timeout=180)
    except Exception:
        return False
    return r.returncode == 0


def python_version(python_exe: str):
    """返回 (major, minor)；读不到返回 None。"""
    try:
        out = subprocess.run([python_exe, "-c",
                              "import sys;print('%d.%d'%sys.version_info[:2])"],
                             capture_output=True, text=True, timeout=30)
        ver = out.stdout.strip()
        major, minor = (int(x) for x in ver.split("."))
        return (major, minor)
    except Exception:
        return None


def check_environment(python_exe: str = None) -> bool:
    """server 启动第一步：解释器版本 + 主依赖预检。

    返回 True 才继续启动。失败时打印可执行的修复命令——用户最常见的
    情况是「用系统 Python 直接跑了 server/server.py」（那边没有 torch），
    提示必须指向 start/terminal/start.sh 这条正规链路。
    """
    py = python_exe or sys.executable
    ver = python_version(py)
    if ver is None:
        print("[server] ✗ 无法读取 Python 版本，请检查解释器。")
        return False
    if ver < (3, 12):
        print(f"[server] ✗ Python {ver[0]}.{ver[1]} < 3.12，GARGANTUA 要求 "
              f"≥3.12（torch 2.2.2 不支持 3.13 预编译轮子）。")
        return False
    if ver >= (3, 13):
        print(f"[server] ✗ Python {ver[0]}.{ver[1]} ≥ 3.13：torch 2.2.2 没有 "
              f"3.13 预编译轮子，装不上。请用 3.12 系列解释器"
              f"（start/terminal/setup.sh 会自动装）。")
        return False
    print(f"[server] Python {ver[0]}.{ver[1]} ✓（{py}）")

    if not check_deps(py):
        print("[server] ✗ 主依赖缺失（torch/numpy/PIL/safetensors）。")
        print("[server]   正规启动方式（自动装好隔离环境后拉起 server）：")
        print("[server]     ./start/terminal/start.sh")
        print("[server]   或手动安装到当前解释器：")
        print(f"[server]     {py} -m pip install -r start/requirements/core.txt")
        return False
    print("[server] 主依赖（torch/numpy/PIL/safetensors）✓")
    return True


# ---------------------------------------------------------------------------
# 窗口完整性校验（sha256 + 弹窗）
# ---------------------------------------------------------------------------
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def popup(title: str, message: str) -> bool:
    """跨平台弹窗；返回 True 表示「继续」（False = 跳过窗口启动）。

    优先级：tkinter → macOS osascript → Linux zenity → 纯文本 stdin。
    无图形会话时降级为命令行提示（回车继续，输入 s 跳过窗口）。
    """
    # 1) tkinter（有图形会话时最稳；仅提示 + 单 OK 按钮）
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
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


def verify_window_integrity(root: str = None, no_verify: bool = False) -> bool:
    """校验 window.py 完整性；返回是否「应继续启动窗口」。

    - 基准文件不存在 → 写入当前哈希（首次运行）；
    - 哈希一致 → 静默通过；
    - 哈希不一致 → 弹窗告知（可跳过；--no-verify 时只打印不弹窗）。
    """
    root = root or ROOT
    window_py = os.path.join(root, WINDOW_PY_RELPATH)
    sha_path = os.path.join(root, WINDOW_SHA_RELPATH)
    if not os.path.isfile(window_py):
        print("[server] 未找到 service/window/window.py，跳过窗口校验。")
        return False
    cur = sha256_of(window_py)
    if not os.path.isfile(sha_path):
        os.makedirs(os.path.dirname(sha_path), exist_ok=True)
        with open(sha_path, "w", encoding="utf-8") as f:
            f.write(cur + "\n")
        print("[server] 已写入 window.py 完整性基准（首次运行）。")
        return True
    with open(sha_path, "r", encoding="utf-8") as f:
        base = f.read().strip()
    if cur == base:
        return True
    msg = ("检测到 service/window/window.py 与上次启动时的内容不一致"
           "（sha256 不匹配）。\n可能是文件被修改或损坏。"
           "是否仍启动窗口？")
    if no_verify:
        print("[server] ⚠️ window.py 完整性不一致（--no-verify，已跳过弹窗）。")
        return True
    return popup("GARGANTUA 窗口完整性警告", msg)


# ---------------------------------------------------------------------------
# 子进程管理小工具（窗口/服务监督器共用，测试复用）
# ---------------------------------------------------------------------------
def wait_port(port: int, host: str = "127.0.0.1", timeout: float = 30.0) -> bool:
    """端口在 timeout 内可连即返回 True（server 就绪的信号）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def wait_status(port: int, host: str = "127.0.0.1", timeout: float = 60.0) -> bool:
    """轮询 /api/status，200 即就绪。"""
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


def kill_proc(proc, grace: float = 5.0) -> None:
    """优雅终止子进程：先 SIGTERM，超时再 SIGKILL。"""
    if proc is None or proc.poll() is not None:
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
