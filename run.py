#!/usr/bin/env python3
# ===========================================================================
# GARGANTUA v1 — 跨平台一键启动（Windows / Linux / macOS）
# ---------------------------------------------------------------------------
# 这是唯一的逻辑载体：三个平台的 bootstrap 脚本（run.bat / run.command /
# run.sh）只负责「找到系统 Python 然后执行本文件」，venv 创建、依赖校验、
# Python 版本守卫、子命令分发全部在这里。
#
# 用法：
#   python run.py                  启动 server + 独立桌面窗口（pywebview）
#   python run.py --port 9000      指定端口
#   python run.py --no-window      只起 server（无 GUI / SSH / 已有浏览器）
#   python run.py --browser        用系统浏览器代替独立窗口（更占内存，退路）
#   python run.py --test           跑全量测试（逐套件分进程，见 cmd_test）
#   python run.py --init NAME      创建模型 models/NAME
#   python run.py --train NAME     用 data/ 训练 models/NAME
#   python run.py --infer NAME [--prompt "提示词"]
#
# 进程模型：
#   run.py ──┬── server.py   老大：HTTP + 训练 + 推理（关掉窗口它照跑）
#            └── window.py   附庸：pywebview 独立窗口（崩了不影响训练）
#   两个都是子进程；Ctrl-C / server 退出时统一收尸，不留孤儿。
#
# 2026-10-20/23 修复记录：
#   1. 依赖单一来源 requirements.txt（此前 run.sh 引用一个不存在的文件：
#      venv 建了但依赖从未安装，起服务即 ModuleNotFound）。
#   2. 创建 venv 前校验 Python ≤ 3.12：torch 2.2.2 无 3.13 预编译轮子。
#   3. venv 已存在但依赖缺失时补装，不自引用创建 venv。
#   4. check_deps 探针必须 `import importlib.util`（只 import importlib
#      不绑定 util，find_spec 必抛 AttributeError → 恒判缺依赖 → 每次
#      都重装 torch，--test 被拖成 16 分钟假死）。
#   5. 当前解释器依赖齐全就直接用它，不再无条件优先 .venv。
#   6. serve 改为「server 子进程 + 窗口子进程」，不再 os.execv 替换自身
#      （否则没有父进程可以管窗口生命周期）。
# ===========================================================================
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

VENV_DIR = os.path.join(ROOT, ".venv")
WIN_PY = os.path.join(VENV_DIR, "Scripts", "python.exe")
NIX_PY = os.path.join(VENV_DIR, "bin", "python")
REQ_FILE = os.path.join(ROOT, "requirements.txt")

# torch 2.2.2 预编译轮子的最高 Python 版本（spec 第 7 节：3.12.14）
MAX_PY_FOR_TORCH = (3, 12)

# requirements.txt 缺失时的兜底（与仓库内 requirements.txt 保持一致）
REQUIREMENTS = ["torch==2.2.2", "numpy<2", "pillow", "safetensors"]

DEFAULT_PORT = 8712


def requirements() -> list:
    """依赖清单：优先读 requirements.txt（单一来源）。"""
    if os.path.exists(REQ_FILE):
        with open(REQ_FILE, encoding="utf-8") as f:
            pkgs = [ln.strip() for ln in f
                    if ln.strip() and not ln.startswith("#")]
        if pkgs:
            return pkgs
    return REQUIREMENTS


def find_python() -> str | None:
    """优先用项目 venv，否则用当前解释器。"""
    for cand in (WIN_PY, NIX_PY):
        if os.path.isfile(cand):
            return cand
    return sys.executable


def ensure_venv() -> str:
    """没有 venv 就建一个并装依赖；venv 在但缺依赖则补装。"""
    py = find_python()
    # 正从项目 venv 内运行：缺依赖直接装进当前环境（绝不能对正在运行的
    # venv 再跑一次 python -m venv——自引用创建会损坏环境）
    if os.path.realpath(sys.prefix) == os.path.realpath(VENV_DIR):
        _pip_install(py)
        return py
    if py != sys.executable and check_deps(py):
        return py                                   # venv 完好，直接用
    if sys.version_info[:2] > MAX_PY_FOR_TORCH:
        print(f"[run] 当前 Python {sys.version_info[0]}.{sys.version_info[1]} "
              f"过新：torch==2.2.2 没有预编译轮子（最高支持 "
              f"{MAX_PY_FOR_TORCH[0]}.{MAX_PY_FOR_TORCH[1]}，"
              f"spec 第 7 节约定 3.12.14）。")
        print("[run] 请使用 Python 3.12 重新运行，例如：")
        print("  python3.12 run.py")
        sys.exit(1)
    if py == sys.executable:
        print(f"[run] 未检测到 venv，正在创建 {VENV_DIR} …")
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
        py = find_python()
    _pip_install(py)
    return py


def _pip_install(py: str) -> None:
    """ 装主依赖。pip 源逻辑与 server 装可选依赖共用（server_util.pip_index_args），
    避免两处漂移。 """
    from server_util import pip_index_args
    pkgs = requirements()
    print(f"[run] 安装依赖: {' '.join(pkgs)}")
    idx = pip_index_args()
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip"]
                          + idx)
    subprocess.check_call([py, "-m", "pip", "install"] + idx + pkgs)


def check_deps(py: str) -> bool:
    """ 目标解释器是否已装齐依赖。

    2026-10-20 修：探针里写的是 `import importlib` —— 它**不会**绑定
    `importlib.util`（CPython 里 util 是子模块，需显式 import），于是
    find_spec 那行必抛 AttributeError、探针恒为「缺依赖」。后果：即使
    当前解释器 torch 齐全，run.py 每次都会去建/补 .venv 并 pip install
    （无网时直接崩），且 --test 会被拖成 16 分钟的假死。 """
    code = ("import importlib.util,sys;"
            "mods=['torch','numpy','PIL','safetensors'];"
            "ok=all(importlib.util.find_spec(m) is not None for m in mods);"
            "sys.exit(0 if ok else 1)")
    try:
        return subprocess.call([py, "-c", code],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL) == 0
    except OSError:
        return False            # 解释器不存在/不可执行 → 视为缺依赖


def has_window_deps(py: str) -> bool:
    """ 该解释器能否创建独立窗口（pywebview 可导入）。 """
    code = "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('webview') else 1)"
    try:
        return subprocess.call([py, "-c", code],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL) == 0
    except OSError:
        return False


def ensure_window_deps(py: str) -> bool:
    """ 缺 pywebview 就装一次（requirements-window.txt，约 18MB）。

    2026-10-23：run.command 走的是系统 python3.12，那里只有主依赖，
    于是窗口起不来、用户被弹回浏览器。窗口是默认体验，不能让人手装。
    装失败不阻塞：返回 False，run.py 照常起 server 并告知手动命令。 """
    if has_window_deps(py):
        return True
    req = os.path.join(ROOT, "requirements-window.txt")
    if not os.path.exists(req):
        print("[run] 缺少 requirements-window.txt，无法自动安装窗口依赖")
        return False
    print("[run] 未检测到窗口依赖，正在安装（pywebview，约 18MB，"
          "使用系统 WebView，不带浏览器内核）…", flush=True)
    from server_util import pip_index_args
    try:
        subprocess.check_call([py, "-m", "pip", "install",
                               "-r", req] + pip_index_args())
    except subprocess.CalledProcessError as e:
        print(f"[run] 窗口依赖安装失败（code={e.returncode}）。"
              f"server 仍会正常启动，可手动执行：")
        print(f"  {py} -m pip install -r requirements-window.txt")
        return False
    if not has_window_deps(py):
        print("[run] 安装后仍无法导入 webview，请手动执行：")
        print(f"  {py} -m pip install -r requirements-window.txt")
        return False
    return True


def wait_port(port: int, host: str = "127.0.0.1",
              timeout: float = 60.0) -> bool:
    """ 等 server 端口可连（模型自动加载可能占几十秒）。 """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.4)
    return False


# 回归套件执行顺序（分进程跑，原因见 cmd_test）
TEST_SUITES = ["test_tokens", "test_codecs", "test_image2prim",
               "test_model", "test_server", "test_fixes"]


def cmd_test(py: str) -> int:
    """ 全量回归：每个套件单独起一个进程。

    不能用一个 `unittest discover` 跑完：全尺寸 Gargantua() 单次构造
    ≈2.1GB（527.9M 参数 fp32），PyTorch 的 CPU 分配器把释放的块留在
    自己的池子里不还给 OS，于是 RSS 只增不减——实测先跑 test_fixes +
    test_server 再进 test_model 会被 SIGKILL（退出码 137），看起来像
    「模型测试挂了」，实际是内存。分进程后每个套件的峰值独立。 """
    os.chdir(ROOT)
    failed = []
    for s in TEST_SUITES:
        print(f"\n===== tests.{s} =====", flush=True)
        rc = subprocess.call([py, "-m", "unittest", f"tests.{s}"])
        if rc != 0:
            failed.append(s)
    print("\n" + "=" * 46)
    if failed:
        print(f"[run] 失败套件: {', '.join(failed)}")
        return 1
    print(f"[run] 全部 {len(TEST_SUITES)} 个套件通过")
    return 0


def _spawn(py: str, *args: str) -> subprocess.Popen:
    """ 子进程统一继承父进程输出 + 项目根 cwd。

    PYTHONUNBUFFERED=1：Python 的 stdout 在非 tty（重定向到文件、被
    .command 双击时）是 8KB 块缓冲，server 的启动日志会憋到进程退出才
    出现，看起来像「卡住没反应」。强制行缓冲，日志即时可见。 """
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen([py, *args], cwd=ROOT, env=env)


def _kill(p: subprocess.Popen, grace: float = 3.0):
    if p is None or p.poll() is not None:
        return
    try:
        p.terminate()
        p.wait(timeout=grace)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def cmd_serve(py: str, port: int, no_window: bool) -> int:
    """ 起 server（子进程）+ 独立窗口（子进程），并统一收尸。

    不用 os.execv 替换自身：父进程必须活着才能管窗口生命周期、转发
    Ctrl-C、在 server 退出时把窗口一起带走（否则窗口指向死端口）。 """
    os.chdir(ROOT)
    server = _spawn(py, os.path.join(ROOT, "server.py"), "--port", str(port))
    win = None
    if not no_window:
        # 窗口是默认体验：缺依赖就地装一次，装不上也不阻塞 server
        if not ensure_window_deps(py):
            print(f"[run] 独立窗口不可用，server 已就绪："
                  f"http://127.0.0.1:{port}", flush=True)
        else:
            if not wait_port(port):
                print("[run] server 端口未就绪，窗口仍会尝试打开", flush=True)
            win = _spawn(py, os.path.join(ROOT, "window.py"),
                         "--port", str(port))
            print(f"[run] 独立窗口已启动（崩了不影响训练）；"
                  f"server 地址 http://127.0.0.1:{port}", flush=True)

    stopping = {"v": False}

    def _shutdown(*_a):
        stopping["v"] = True
        _kill(win)
        _kill(server)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    try:
        while not stopping["v"]:
            time.sleep(0.5)
            if server.poll() is not None:
                print(f"[run] server 已退出（code={server.returncode}）",
                      flush=True)
                _kill(win)               # server 没了，窗口留着也是死的
                return server.returncode or 0
            if win is not None and win.poll() is not None:
                # 窗口自己关了/崩了：训练继续，只告知，不去动 server
                print(f"[run] 窗口已退出（code={win.returncode}）；"
                      f"server 仍在 {port} 端口运行，训练不受影响。"
                      f"重开界面：python window.py --port {port}", flush=True)
                win = None
    except KeyboardInterrupt:
        _shutdown()
    _kill(server)
    _kill(win)
    return 0


def main():
    ap = argparse.ArgumentParser(description="GARGANTUA v1 启动器")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-open", action="store_true",
                    help="--no-window 的旧名（兼容保留）")
    ap.add_argument("--no-window", action="store_true",
                    help="只起 server，不开独立窗口")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--init", metavar="NAME")
    ap.add_argument("--train", metavar="NAME")
    ap.add_argument("--infer", metavar="NAME")
    ap.add_argument("--prompt", default=None,
                    help="配合 --infer 使用的提示词（默认取 PROMPT 环境变量或「你好」）")
    args = ap.parse_args()

    # 当前解释器已装好依赖就直接用它：此前无条件优先 .venv，于是用
    # 「已装好 torch 的」解释器启动也会因为项目里有个空 .venv 而触发
    # 一轮 pip install torch（实测 16 分钟仍未装完，命令看起来像卡死）
    if check_deps(sys.executable):
        py = sys.executable
    else:
        py = find_python()
    if not check_deps(py):
        print("[run] 当前 Python 缺少依赖，尝试创建/使用项目 venv …")
        py = ensure_venv()
        if not check_deps(py):
            print("[run] 依赖安装失败，请手动执行：")
            print(f"  {py} -m pip install -r requirements.txt")
            sys.exit(1)

    if args.test:
        sys.exit(cmd_test(py))
    if args.init:
        cmd = [py, "init.py", "--name", args.init]
    elif args.train:
        cmd = [py, "train.py", "--model", f"models/{args.train}",
               "--data", "data"]
    elif args.infer:
        prompt = args.prompt or os.environ.get("PROMPT", "你好")
        cmd = [py, "infer.py", "--model", f"models/{args.infer}",
               "--prompt", prompt]
    else:
        sys.exit(cmd_serve(py, args.port, args.no_window or args.no_open))

    os.chdir(ROOT)
    os.execv(py, cmd)


if __name__ == "__main__":
    main()
