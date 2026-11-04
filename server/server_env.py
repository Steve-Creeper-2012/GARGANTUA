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
#   list_env_packs        environment/ 下每个依赖包（一个文件夹一个包）
#   detect_tools          系统级外部命令（ffmpeg 等）
#   wait_port / wait_status / kill_proc   子进程管理小工具（测试复用）
#
# 2026-10-31：environment/ 的目录契约改成「一个依赖一个文件夹，各带
# pack.json 清单」。主页的「环境包」区就是扫这个目录出来的——凡是显示在
# 主页上的环境，必定在 environment/ 下有自己的文件夹；反过来 environment/
# 下多出一个文件夹，主页立刻多一张卡。两边永远对得上，不会有「装了但
# 没人知道」的依赖（之前 rust 就是这种状态）。
# ===========================================================================
from __future__ import annotations

import json
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
ENV_ROOT = os.path.join(ROOT, "environment")

# ---------------------------------------------------------------------------
# environment/ 依赖包：一个文件夹 = 一个包
# ---------------------------------------------------------------------------
# 每个包目录里放一个 pack.json（清单），字段：
#   title       人话名字（主页卡片标题）
#   kind        runtime / venv / toolchain / binary
#   why         一句话说明它管什么
#   install     装它的命令（未安装时主页提示用）
#   probe       可执行文件相对本包的路径（用于判断装没装 + 取版本）
#   version_arg 取版本号的参数（默认 --version）
#   env         这个包要求的环境变量（键 → 相对路径，或 {abs} 占位）
#   contents    目录内各个子文件夹的用途说明（用于「分干净」自检）
# pack.json 缺失时也能显示（推断：有 probe 就算装好），但描述会是空的。
PACK_MANIFEST = "pack.json"
_SIZE_TTL = 60.0            # 目录体积缓存（du 走一次几百毫秒，别每次请求都跑）
_SIZE_CACHE: dict = {}


def _du_bytes(path: str) -> int:
    """目录占用（字节）；用 du -sk，比 os.walk 数万个文件快一个量级。"""
    now = time.time()
    hit = _SIZE_CACHE.get(path)
    if hit and now - hit[0] < _SIZE_TTL:
        return hit[1]
    size = 0
    try:
        r = subprocess.run(["du", "-sk", path], capture_output=True,
                           text=True, timeout=20)
        if r.returncode == 0 and r.stdout.strip():
            size = int(r.stdout.split()[0]) * 1024
    except Exception:
        size = 0
    _SIZE_CACHE[path] = (now, size)
    return size


def _probe_exe(path: str, version_arg: str = "--version"):
    """跑一下可执行文件取版本号；跑不起来返回 None。"""
    if not (os.path.isfile(path) and os.access(path, os.X_OK)):
        return None
    try:
        r = subprocess.run([path, version_arg], capture_output=True,
                           text=True, timeout=15)
        out = (r.stdout or r.stderr or "").strip().splitlines()
        if out:
            # 只取第一行里像版本号的那一段，别把整段 banner 塞进卡片
            line = out[0].strip()
            for tok in line.replace("(", " ").replace(")", " ").split():
                if tok[:1].isdigit() and any(c.isdigit() for c in tok):
                    return tok.rstrip(",;")
            return line[:40]
    except Exception:
        return None
    return None


def read_pack_manifest(dir_path: str) -> dict:
    """读某个包目录的 pack.json；缺失/损坏返回 {}。"""
    p = os.path.join(dir_path, PACK_MANIFEST)
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        return m if isinstance(m, dict) else {}
    except Exception:
        return {}


def list_env_packs() -> list:
    """environment/ 下每个子目录 = 一个依赖包。

    返回 [{name, title, kind, why, install, present, version, size, path,
           env, contents, extras}]，按 pack.json 的 order 再按名字排序。
    没有 pack.json 的目录也会列出来（present 按目录非空判断），这样
    「多装了一个东西」不会在主页上隐身。
    """
    out = []
    if not os.path.isdir(ENV_ROOT):
        return out
    for name in sorted(os.listdir(ENV_ROOT)):
        d = os.path.join(ENV_ROOT, name)
        if not os.path.isdir(d) or name.startswith("."):
            continue
        m = read_pack_manifest(d)
        probe_rel = m.get("probe") or ""
        version = None
        present = bool(m.get("present_when_missing"))
        if probe_rel:
            version = _probe_exe(os.path.join(d, probe_rel),
                                 m.get("version_arg") or "--version")
            present = version is not None
        else:
            # 没有可执行探针的包（venv 这类）：看几个标志文件在不在。
            # 直接「目录非空就算装好」会把只剩 pack.json 的空壳当成已安装。
            marks = m.get("probe_files") or []
            if marks:
                present = any(os.path.exists(os.path.join(d, x))
                              for x in marks)
            else:
                try:
                    present = any(not x.startswith((".", PACK_MANIFEST))
                                  for x in os.listdir(d))
                except OSError:
                    present = False
        out.append({
            "name": name,
            "title": m.get("title") or name,
            "kind": m.get("kind") or "pack",
            "why": m.get("why") or "",
            "install": m.get("install") or "",
            "present": bool(present),
            "version": version or m.get("version") or "",
            "size": _du_bytes(d),
            "path": os.path.relpath(d, ROOT),
            "env": m.get("env") or {},
            "contents": m.get("contents") or {},
            "order": int(m.get("order", 100)),
        })
    out.sort(key=lambda x: (x["order"], x["name"]))
    return out


# 2026-11-04（steve）：所有外部二进制必须是下载到 environment/ 内的本地
# 副本。不再列系统命令、不再提示 brew/apt/xcode-select，也绝不查系统 PATH。
BUNDLED_TOOLS = (
    {"name": "ffmpeg", "title": "FFmpeg（项目内）", "version_arg": "-version",
     "why": "音频/视频解码与抽帧（缺了只能处理 wav 与帧序列）",
     "install": "python start/setup_env.py --stage tools"},
    {"name": "node", "title": "Node.js（项目内）", "version_arg": "--version",
     "why": "构建期 app.js 语法检查",
     "install": "python start/setup_env.py --stage tools"},
    {"name": "cargo", "title": "Rust 工具链（项目内）",
     "version_arg": "--version",
     "why": "构建 GPU 图元转换器（environment/rust/ 下）",
     "install": "sh service/prims/setup.sh"},
)


def detect_tools() -> list:
    """只探测 environment/ 里的本地二进制；缺了给项目内安装命令。"""
    from paths import bundled_tool
    out = []
    for t in BUNDLED_TOOLS:
        exe = bundled_tool(t["name"])
        out.append({
            "name": t["name"],
            "title": t["title"],
            "why": t["why"],
            "install": t["install"],
            "present": bool(exe),
            "path": exe or "",
            "isolated": True,
            "version": _probe_exe(exe, t.get("version_arg") or "--version")
                       if exe else "",
        })
    return out


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
    """server 启动第一步：**只检查 must 清单**（轻量必备）。

    2026-11-01（steve 定案）：启动只认 start/requirements/must.txt ——
    Python 运行时（跑 server）+ pywebview（开独立窗口）。torch、ffmpeg、
    rust 这些**不影响启动**，缺了只是训练/推理/图元转换用不了，用户可以
    之后在主页「环境与依赖」里装。所以这里绝不能再因为缺 torch 而拒绝
    启动——那会让「轻装进页面 → 逐个装依赖」这条路走不通。

    返回 True 才继续启动。失败时打印可执行的修复命令。
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

    # 2026-11-01：启动只查 must 清单（python + pywebview）。
    # torch / numpy / ffmpeg / rust 这些在 necessary.txt —— 缺了不影响
    # 启动，只是对应功能用不了，用户在主页里按需安装。
    try:
        from reqs import all_requirements, dep_state
        must = [d for d in all_requirements() if d["group"] == "must"]
    except Exception:
        must = []
    missing = []
    soft = []
    for d in must:
        st = dep_state(d)
        if st["ok"]:
            continue
        (soft if d.get("soft") else missing).append(d)
    if soft:
        # soft = 缺了只影响某项功能（比如 pywebview 只影响独立窗口），
        # 不该拦住 server 起来（--no-window / SSH 场景照常用）。
        print("[server] ⚠ 缺少可选的启动依赖："
              + "、".join(d["name"] for d in soft)
              + "（对应功能不可用，不影响启动）")
    if missing:
        print("[server] ✗ 启动必备依赖缺失："
              + "、".join(d["name"] for d in missing))
        for d in missing:
            print(f"[server]   {d['name']}：{d.get('install', '')}")
        print("[server]   正规启动方式（自动装好隔离环境后拉起 server）：")
        print("[server]     ./启动.sh（或 ./启动.command）")
        return False
    print("[server] 启动必备依赖 ✓（"
          + "、".join(d["name"] for d in must) + "）")
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
