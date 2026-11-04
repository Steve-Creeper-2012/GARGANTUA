# ===========================================================================
# GARGANTUA v1 — paths.py：项目内路径工具（纯标准库，零第三方依赖）
# ---------------------------------------------------------------------------
# 存在意义（2026-11-04 可移植性收敛）：
#   项目文件夹整体搬到任意路径/另一台机器后，双击启动脚本必须照样能跑，
#   且**任何写盘**都不得落到项目目录之外。所以：
#     - ROOT 一律由本文件的 __file__ 推导，绝不依赖 cwd、HOME、TEMP、
#       环境变量或平台专属路径；
#     - 运行时临时文件（mkdtemp / NamedTemporaryFile / TemporaryDirectory）
#       统一 dir=project_tmp_dir()，即 ROOT/data/tmp；
#     - 下载 / 编译 / pip 缓存统一 ROOT/environment/_cache；
#     - 日志、模型、词表、数据本来就在 ROOT 内（logs/ models/ data/ vocabs/）。
#   本模块不 import torch / numpy / safetensors：启动只装 must 依赖时，
#   图元与编解码等纯 Python 路径也必须能安全 import 它。
# ===========================================================================
from __future__ import annotations

import os

# server/paths.py → 上一级即项目根
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def project_dir(*parts: str) -> str:
    """ROOT 下的绝对路径（parts 是相对 ROOT 的分量）。"""
    return os.path.join(PROJECT_ROOT, *parts)


def project_tmp_dir() -> str:
    """运行时临时目录 ROOT/data/tmp（必要时创建）。

    所有 tempfile.* 调用都必须传 dir=本函数返回值——不写系统临时目录
    或用户目录，搬走文件夹时不留任何外部残留。"""
    p = project_dir("data", "tmp")
    os.makedirs(p, exist_ok=True)
    return p


def project_cache_dir(*parts: str) -> str:
    """下载 / 编译 / pip 缓存 ROOT/environment/_cache/...（必要时创建）。"""
    p = project_dir("environment", "_cache", *parts)
    os.makedirs(p, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 强制隔离（2026-11-04，steve：绝不污染系统、绝不复用系统运行时）
# ---------------------------------------------------------------------------
# 规则：
#   1. 唯一允许的 Python 底座 = environment/python/ 里下载的 standalone
#      CPython；venv 必须由它创建，pyvenv.cfg 的 home/command 必须落在
#      项目内（venv_is_isolated 会查）。
#   2. 所有外部二进制（ffmpeg / node / cargo）只在 environment/tools/ 或
#      environment/rust/ 里找，找不到就是没装，**绝不查系统 PATH**。
#   3. 所有子进程继承 isolated_env()：缓存、临时目录、工具链 HOME、
#      用户站点包全部重定向到项目内；系统目录零写入。
# ---------------------------------------------------------------------------
ENV_ROOT = project_dir("environment")
PYTHON_HOME = project_dir("environment", "python")
VENV_DIR = project_dir("environment", "venv")
TOOLS_ROOT = project_dir("environment", "tools")
DOWNLOADS_DIR = project_dir("environment", "_cache", "downloads")


def _exe(name: str) -> str:
    return name + ".exe" if os.name == "nt" else name


def bundled_python() -> str:
    """项目内 standalone Python（唯一合法的 Python 底座，可能尚未下载）。"""
    if os.name == "nt":
        return project_dir("environment", "python", "python", "python.exe")
    return project_dir("environment", "python", "python", "bin", "python3")


def venv_python() -> str:
    if os.name == "nt":
        return project_dir("environment", "venv", "Scripts", "python.exe")
    return project_dir("environment", "venv", "bin", "python3")


def bundled_tool(name: str) -> str:
    """项目内工具二进制的绝对路径；找不到返回 ""（不回退系统 PATH）。"""
    exe = _exe(name)
    candidates = [
        project_dir("environment", "tools", name, "bin", exe),
        project_dir("environment", "tools", name, exe),
        project_dir("environment", "bin", exe),
    ]
    if name == "cargo":
        candidates.insert(0, project_dir("environment", "rust", "cargo",
                                         "bin", exe))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return ""


def _ensure_dirs(*dirs) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def isolated_env(base=None) -> dict:
    """子进程环境：所有会写盘的位置都指向项目内，系统目录零写入。

    PATH 保留系统原值但把项目内 bin 放最前——保留只为确保 Rust 构建期
    能找到平台 linker/SDK（那是 ABI 提供的基础设施，不安装、不写系统）；
    运行时查找一律走 bundled_tool()，不依赖 PATH。"""
    env = dict(base if base is not None else os.environ)
    tmp = project_dir("data", "tmp")
    cache = project_dir("environment", "_cache")
    bin_dirs = [
        project_dir("environment", "bin"),
        project_dir("environment", "tools", "ffmpeg", "bin"),
        project_dir("environment", "tools", "ffmpeg"),
        project_dir("environment", "tools", "node", "bin"),
        project_dir("environment", "tools", "node"),
        project_dir("environment", "rust", "cargo", "bin"),
    ]
    _ensure_dirs(tmp, cache, project_dir("environment", "bin"),
                 project_dir("environment", "config"),
                 project_dir("environment", "data"))
    env.update({
        # 临时文件 / 缓存
        "TMPDIR": tmp, "TMP": tmp, "TEMP": tmp,
        "PIP_CACHE_DIR": project_dir("environment", "_cache", "pip"),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONNOUSERSITE": "1",       # 禁止 ~/.local 用户站点包混进隔离环境
        "PYTHONUSERBASE": project_dir("environment", "userbase"),
        # 工具链 HOME（rust / go / node / XDG）
        "CARGO_HOME": project_dir("environment", "rust", "cargo"),
        "RUSTUP_HOME": project_dir("environment", "rust", "rustup"),
        "GOPATH": project_dir("environment", "go"),
        "GOCACHE": project_dir("environment", "_cache", "go", "build"),
        "GOMODCACHE": project_dir("environment", "_cache", "go", "mod"),
        "NPM_CONFIG_CACHE": project_dir("environment", "_cache", "npm"),
        "npm_config_cache": project_dir("environment", "_cache", "npm"),
        "XDG_CACHE_HOME": project_dir("environment", "_cache"),
        "XDG_CONFIG_HOME": project_dir("environment", "config"),
        "XDG_DATA_HOME": project_dir("environment", "data"),
        # 标记：下游代码可据此拒绝系统回退
        "GARGANTUA_ROOT": PROJECT_ROOT,
        "GARGANTUA_ISOLATED": "1",
    })
    old_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(bin_dirs + ([old_path] if old_path else []))
    return env


def shell_binary() -> str:
    """内置终端用的 shell：优先项目内副本，最后才退回系统 shell。

    终端是交互式功能，用户输入的命令本来就由用户自己决定；这里把系统
    shell 的查找收在一处，并把项目内副本放在最前，方便完全离线打包。"""
    import shutil
    for name in ("zsh", "bash", "sh"):
        local = bundled_tool(name)
        if local:
            return local
    return (shutil.which("zsh") or shutil.which("bash")
            or ("/bin/sh" if os.name != "nt" else "cmd.exe"))


def venv_is_isolated() -> tuple[bool, str]:
    """venv 的 pyvenv.cfg 是否把底座指向项目内（而不是系统 Python）。

    返回 (是否隔离, 原因)。旧 venv 的 home=/Library/Frameworks/... 或
    command=/usr/local/bin/python3.12 都会被判为不隔离 → 启动链必须重建。
    """
    cfg_path = os.path.join(VENV_DIR, "pyvenv.cfg")
    if not os.path.isfile(cfg_path):
        return False, "缺少 pyvenv.cfg"
    vals = {}
    try:
        with open(cfg_path, encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    k, v = line.split("=", 1)
                    vals[k.strip().lower()] = v.strip()
    except OSError as e:
        return False, f"pyvenv.cfg 读取失败: {e}"
    home = vals.get("home", "")
    cmd = vals.get("command", "")
    root = os.path.realpath(PROJECT_ROOT)
    for key, val in (("home", home), ("command", cmd)):
        if not val:
            return False, f"pyvenv.cfg 缺少 {key}"
        # command 是整条命令，只看它引用的第一个绝对路径
        token = val.split()[0] if key == "command" else val
        if not os.path.isabs(token):
            return False, f"pyvenv.cfg {key} 不是绝对路径: {token}"
        if os.path.commonpath([os.path.realpath(token), root]) != root:
            return False, f"pyvenv.cfg {key} 指向系统目录: {token}"
    return True, "项目内 standalone Python"
