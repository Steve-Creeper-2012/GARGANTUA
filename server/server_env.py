# ===========================================================================
# GARGANTUA v1 — server_env.py：server 侧的环境预检
# ---------------------------------------------------------------------------
# 2026-10-24 重构：steve 的要求是「Server 检查环境，启动 windows.py」。
# 终端脚本（launcher/ 下的 start-server.* / setup-environment.sh，由根目录 启动.* 调用）只负责搭好隔离环境并 exec server.py，
# 环境预检由 server 自己做（原 start/launcher.py 的职责并入 server）。
#
# 本模块（纯标准库，server 启动早期就能跑，缺依赖时也能给出可读提示）：
#   check_deps            探针：解释器能否 import 主依赖
#   check_environment     解释器版本 + 主依赖预检（server main 第一步）
#   wait_port / wait_status / kill_proc   子进程管理小工具（测试复用）
#
# 注：window.py 的 sha256 完整性校验曾在这里，2026-10-24 steve 判定
# 「过于复杂」已整体移除（基准文件 + 弹窗 + 测试面一并删除）。
# ===========================================================================
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request

# 主依赖：缺任何一个，server 的核心功能（模型/训练/推理）都跑不了
REQUIRED_MODULES = ("torch", "numpy", "PIL", "safetensors")


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
    提示必须指向 launcher/start-server.sh（由根目录 启动.sh 调用）这条正规链路。
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
              f"（setup-environment.sh 会自动装）。")
        return False
    print(f"[server] Python {ver[0]}.{ver[1]} ✓（{py}）")

    if not check_deps(py):
        print("[server] ✗ 主依赖缺失（torch/numpy/PIL/safetensors）。")
        print("[server]   正规启动方式（自动装好隔离环境后拉起 server）：")
        print("[server]     ./启动.sh（或 ./启动.command）")
        print("[server]   或手动安装到当前解释器：")
        print(f"[server]     {py} -m pip install -r start/requirements/core.txt")
        return False
    print("[server] 主依赖（torch/numpy/PIL/safetensors）✓")
    return True


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
