#!/usr/bin/env python3
# ===========================================================================
# GARGANTUA v1 — start/setup_env.py：唯一的环境/二进制安装器（跨平台）
# ---------------------------------------------------------------------------
# 硬约束（2026-11-04，steve）：
#   * 绝不复用系统 Python、系统包管理器、系统 PATH 里的二进制；
#   * 所有下载、缓存、解包、venv、pip、工具链 HOME 全部写在项目文件夹内；
#   * environment/ 空也能从零跑起来：引导层用系统 curl/tar 只下载一次
#     standalone CPython，之后全部由这个脚本（项目内 Python）完成；
#   * 镜像自动测速选最快；下载物留在 environment/_cache/downloads/，
#     离线时直接复用缓存，零网络也能装。
#
# 用法（必须由项目内 standalone Python 执行）：
#   python start/setup_env.py --check                 环境是否就绪（0/1）
#   python start/setup_env.py --stage runtime        建隔离 venv + 装 must + stamp
#   python start/setup_env.py --stage tools          测速下载 ffmpeg / node
#   python start/setup_env.py --stage all
#   python start/setup_env.py --install PKG...       窗口内手动安装/更新
#   python start/setup_env.py --install --upgrade PKG...
#   python start/setup_env.py --python-assets        输出引导层要的资产名/URL
#
# 只依赖标准库 + server/paths.py；不 import torch / numpy。
# ===========================================================================
from __future__ import annotations

import argparse
import json
import os
import shutil
import ssl
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))
import paths  # noqa: E402

CACHE = paths.DOWNLOADS_DIR
os.makedirs(CACHE, exist_ok=True)

PY_TAG = os.environ.get("GARGANTUA_PY_RELEASE", "20261026")
PY_PATCH = os.environ.get("GARGANTUA_PY_PATCH", "3.12.16")
PY_MIRRORS = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    + PY_TAG,
    "https://gh-proxy.com/https://github.com/astral-sh/"
    "python-build-standalone/releases/download/" + PY_TAG,
    "https://mirror.sjtu.edu.cn/github-release/astral-sh/"
    "python-build-standalone/releases/download/" + PY_TAG,
)
PIP_MIRRORS = (
    ("https://pypi.tuna.tsinghua.edu.cn/simple", "pypi.tuna.tsinghua.edu.cn"),
    ("https://mirrors.aliyun.com/pypi/simple", "mirrors.aliyun.com"),
    ("https://pypi.org/simple", "pypi.org"),
)
# 固定版本的可选二进制（下载到 environment/tools/，运行时只认项目内路径）
NODE_VERSION = "v22.14.0"
NODE_MIRRORS = (
    "https://nodejs.org/dist",
    "https://npmmirror.com/mirrors/node",
    "https://mirrors.tuna.tsinghua.edu.cn/nodejs-release",
)
FFMPEG_MIRRORS = {
    "Darwin": ("https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip",),
    "Linux": ("https://johnvansickle.com/ffmpeg/releases/"
              "ffmpeg-release-amd64-static.tar.xz",),
    "Windows": ("https://www.gyan.dev/ffmpeg/builds/"
                "ffmpeg-release-essentials.zip",),
}
FFMPEG_ASSET = {"Darwin": "ffmpeg-macos.zip",
                "Linux": "ffmpeg-linux.tar.xz",
                "Windows": "ffmpeg-win.zip"}


def log(msg: str) -> None:
    print(f"[setup] {msg}", file=sys.stderr, flush=True)


def warn(msg: str) -> None:
    print(f"[setup] ⚠ {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 网络：TLS / 测速 / 下载
# ---------------------------------------------------------------------------
def ssl_context():
    """CA 只从项目内取（引导层会把系统 CA 复制进 environment/certs/）。

    本机的透明代理根证书不在 CPython 自带 CA 里；把系统 CA 复制一份进项目
    文件夹后，证书也随文件夹走，不依赖系统状态、也不降低 TLS 校验强度。
    """
    ca = paths.project_dir("environment", "certs", "cacert.pem")
    if os.path.isfile(ca):
        return ssl.create_default_context(cafile=ca)
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    if os.environ.get("GARGANTUA_INSECURE_TLS") == "1":
        warn("按环境变量要求跳过 TLS 校验（仅用于受控离线/代理环境）")
        return ssl._create_unverified_context()
    return ssl.create_default_context()


def urlopen(url: str, timeout=30, range_header=None, ctx=None):
    headers = {"User-Agent": "GARGANTUA/1.0"}
    if range_header:
        headers["Range"] = range_header
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=timeout,
                                  context=ctx or ssl_context())


def pick_fastest(urls, label: str, probe_bytes=1048576) -> str:
    """对候选镜像发 Range 请求测速，返回最快的 URL 前缀；全失败取第一个。"""
    best, best_score = urls[0], 0.0
    ctx = ssl_context()
    for u in urls:
        t0 = time.time()
        try:
            with urlopen(u, timeout=8,
                         range_header=f"bytes=0-{probe_bytes - 1}",
                         ctx=ctx) as r:
                n = len(r.read(probe_bytes))
        except Exception as e:
            log(f"  测速失败 {u.split('/')[2]}: {type(e).__name__}")
            n = 0
        dt = max(time.time() - t0, 0.001)
        score = n / dt
        log(f"  {label} 测速 {int(score):>10} B/s  <- {u.split('/')[2]}")
        if score > best_score:
            best, best_score = u, score
    if best_score <= 0:
        warn(f"{label} 全部镜像测速失败，使用默认源")
        return urls[0]
    return best


def download(asset: str, url: str, dest: str, timeout: int = 600) -> str:
    """下载到项目内缓存（命中即离线复用），再复制到 dest。"""
    cache_path = os.path.join(CACHE, asset)
    if not os.path.isfile(cache_path) or os.path.getsize(cache_path) < 1024:
        part = cache_path + ".part"
        log(f"下载 {asset} ……")
        try:
            with urlopen(url, timeout=timeout) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f, length=1024 * 1024)
            os.replace(part, cache_path)
        finally:
            if os.path.exists(part):
                os.unlink(part)
    else:
        log(f"复用项目内缓存（离线）: {asset}")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    shutil.copyfile(cache_path, dest)
    return dest


def unpack(archive: str, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    if archive.endswith((".tar.gz", ".tgz", ".tar.xz")):
        with tarfile.open(archive) as t:
            try:
                t.extractall(dest, filter="data")
            except TypeError:               # Python < 3.12
                t.extractall(dest)
    elif archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        raise ValueError(f"不认识的压缩包: {archive}")


def flatten(dest: str) -> None:
    """解包后只有一个顶层目录时，把内容上移一层。"""
    entries = [e for e in os.listdir(dest)]
    if len(entries) == 1:
        inner = os.path.join(dest, entries[0])
        if os.path.isdir(inner):
            for name in os.listdir(inner):
                target = os.path.join(dest, name)
                if os.path.exists(target):
                    continue
                shutil.move(os.path.join(inner, name), target)
            shutil.rmtree(inner, ignore_errors=True)


# ---------------------------------------------------------------------------
# 平台 / 资产
# ---------------------------------------------------------------------------
def triple() -> str:
    import platform
    os_, arch = platform.system(), platform.machine().lower()
    arm = arch in ("arm64", "aarch64")
    if os_ == "Darwin":
        return "aarch64-apple-darwin" if arm else "x86_64-apple-darwin"
    if os_ == "Linux":
        return ("aarch64-unknown-linux-gnu" if arm
                else "x86_64-unknown-linux-gnu")
    if os_ == "Windows":
        return "x86_64-pc-windows-msvc"
    raise RuntimeError(f"不支持的操作系统: {os_}")


def python_asset() -> str:
    """优先用缓存里的 release 清单解析资产名，否则用内置 tag/patch。"""
    custom = os.environ.get("GARGANTUA_PY_ASSET")
    if custom:
        return custom
    manifest = os.path.join(CACHE, "pbs_release.json")
    tr = triple()
    if os.path.isfile(manifest):
        try:
            with open(manifest, encoding="utf-8") as f:
                data = json.load(f)
            names = sorted(a["name"] for a in data.get("assets", [])
                           if a["name"].startswith("cpython-3.12.")
                           and tr in a["name"]
                           and a["name"].endswith("-install_only.tar.gz"))
            if names:
                return names[-1]
        except Exception as e:
            warn(f"release 清单解析失败，用内置资产名: {e}")
    return f"cpython-{PY_PATCH}+{PY_TAG}-{tr}-install_only.tar.gz"


def python_urls() -> list:
    asset = python_asset()
    return [f"{base}/{asset}" for base in PY_MIRRORS]


# ---------------------------------------------------------------------------
# venv：必须由项目内 standalone Python 构建
# ---------------------------------------------------------------------------
def site_packages(venv_dir: str, py: str) -> str:
    r = subprocess.run(
        [py, "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
        capture_output=True, text=True, timeout=60,
        env=paths.isolated_env())
    if r.returncode != 0:
        raise RuntimeError(f"无法定位 site-packages: {r.stderr}")
    return r.stdout.strip()


def ensure_venv() -> str:
    """重建/复用 venv；旧 venv 若指向系统 Python，离线迁移其 site-packages。"""
    vp = paths.venv_python()
    ok, why = paths.venv_is_isolated()
    if ok and os.path.isfile(vp):
        log(f"隔离 venv 已就绪（{why}）")
        return vp

    log(f"venv 不合规（{why}）→ 用项目内 standalone Python 重建")
    base_py = paths.bundled_python()
    if not os.path.isfile(base_py):
        raise RuntimeError(
            "项目内 standalone Python 不存在，请先运行 launcher/"
            "setup-environment.sh（Windows: start\\terminal\\setup.bat）")

    # 2026-11-04 修复：原实现把几百 MB 的 site-packages **字节复制**两遍
    # （备份 + 回填），慢到十几分钟。改为同一文件系统内的 os.replace：
    # 旧目录整体改名到项目内暂存（O(1)），重建 venv 后逐项改名回填，
    # 全程不搬一个字节，也仍然完全离线复用已装轮子。
    staging = paths.project_dir("environment", "_cache", "venv_migrate")
    shutil.rmtree(staging, ignore_errors=True)   # 清掉中断留下的半成品
    old_sp = ""
    try:
        old_sp = site_packages(paths.VENV_DIR, vp if os.path.isfile(vp)
                               else sys.executable)
    except Exception:
        pass
    moved = False
    if old_sp and os.path.isdir(old_sp):
        try:
            os.replace(old_sp, staging)
            moved = True
            log("旧轮子就地移到项目内暂存（rename，秒级，不复制字节）")
        except OSError as e:
            warn(f"目录 rename 失败，逐项移动: {e}")
            os.makedirs(staging, exist_ok=True)
            for name in os.listdir(old_sp):
                os.replace(os.path.join(old_sp, name),
                           os.path.join(staging, name))
            moved = True

    shutil.rmtree(paths.VENV_DIR, ignore_errors=True)
    subprocess.run([base_py, "-m", "venv", "--copies", paths.VENV_DIR],
                   check=True, env=paths.isolated_env(), cwd=ROOT)
    if not os.path.isfile(vp):
        raise RuntimeError(f"venv 创建后找不到解释器: {vp}")
    ok2, why2 = paths.venv_is_isolated()
    if not ok2:
        raise RuntimeError(f"venv 仍不隔离（{why2}）——拒绝使用系统 Python")
    if moved:
        new_sp = site_packages(paths.VENV_DIR, vp)
        log("回填已装轮子到新 venv（rename，保留新 venv 自带 pip）……")
        for name in os.listdir(staging):
            src, dst = os.path.join(staging, name), os.path.join(new_sp, name)
            if os.path.exists(dst):
                continue
            os.replace(src, dst)
        shutil.rmtree(staging, ignore_errors=True)
    log("✓ venv 完全由项目内 Python 构建")
    return vp


def pip_index() -> tuple[str, str]:
    urls = [u for u, _h in PIP_MIRRORS]
    best = pick_fastest([u + "/six/" for u in urls], "pip 源")
    base = best.rsplit("/six/", 1)[0]
    host = next((h for u, h in PIP_MIRRORS if u == base), "")
    return base, host


def pip_install(vp: str, args: list) -> None:
    index, host = pip_index()
    trusted = [host, "files.pythonhosted.org", "pypi.org"]
    cmd = [vp, "-m", "pip", "install", "--disable-pip-version-check",
           "--cache-dir", paths.project_cache_dir("pip"),
           "-i", index]
    for h in trusted:
        if h:
            cmd += ["--trusted-host", h]
    cmd += args
    log("pip 安装（缓存/临时目录均在项目内）……")
    r = subprocess.run(cmd, env=paths.isolated_env(), cwd=ROOT,
                       capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise RuntimeError("pip 失败: " + (r.stderr or r.stdout)[-800:])


MUST_IMPORT = ("webview",)          # 启动必需：python 由底座提供 + pywebview


def must_ready(vp: str) -> bool:
    if not os.path.isfile(vp):
        return False
    mods = "import importlib.util,sys;sys.exit(0 if all("
    mods += "importlib.util.find_spec(m) is not None for m in "
    mods += repr(list(MUST_IMPORT)) + ") else 1)"
    try:
        return subprocess.run([vp, "-c", mods], timeout=60,
                              env=paths.isolated_env()).returncode == 0
    except Exception:
        return False


def check() -> bool:
    ok, why = paths.venv_is_isolated()
    vp = paths.venv_python()
    stamp = os.path.join(paths.VENV_DIR, ".gargantua_setup_stamp")
    fresh = os.path.isfile(stamp) and (
        f"platform={_platform_stamp()}" in open(stamp, encoding="utf-8").read())
    return bool(ok and os.path.isfile(vp) and must_ready(vp) and fresh)


def _platform_stamp() -> str:
    import platform
    return f"{platform.system()}-{platform.machine()}"


def write_stamp() -> None:
    stamp = os.path.join(paths.VENV_DIR, ".gargantua_setup_stamp")
    with open(stamp, "w", encoding="utf-8") as f:
        f.write(f"isolated=1\nplatform={_platform_stamp()}\n"
                f"base={os.path.relpath(paths.bundled_python(), ROOT)}\n"
                f"python_asset={python_asset()}\n")


def stage_runtime() -> None:
    vp = ensure_venv()
    if not must_ready(vp):
        req = paths.project_dir("start", "requirements", "must.txt")
        pip_install(vp, ["-r", req])
    if not must_ready(vp):
        raise RuntimeError("pywebview 安装后仍不可导入")
    write_stamp()
    log("✓ 启动必需环境就绪（standalone Python + 隔离 venv + pywebview）")


# ---------------------------------------------------------------------------
# 可选二进制：ffmpeg / node（只落项目内）
# ---------------------------------------------------------------------------
def place_binary(dest: str, name: str) -> bool:
    """在解包目录里找到可执行文件，规整到 dest/ 或 dest/bin/。"""
    exe = name + (".exe" if os.name == "nt" else "")
    target_dir = os.path.join(dest, "bin")
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, exe)
    for base, _dirs, files in os.walk(dest):
        if exe in files:
            src = os.path.join(base, exe)
            if os.path.abspath(src) == os.path.abspath(target):
                return True
            shutil.copy2(src, target)
            os.chmod(target, 0o755)
            return True
    return False


def stage_tools() -> None:
    import platform
    os_ = platform.system()
    # Node（构建期 JS 语法检查用）
    node_dir = paths.project_dir("environment", "tools", "node")
    if not paths.bundled_tool("node"):
        ext = "zip" if os_ == "Windows" else (
            "tar.xz" if os_ == "Linux" else "tar.gz")
        suffix = ("win-x64" if os_ == "Windows" else
                  "linux-x64" if os_ == "Linux" else
                  "darwin-arm64" if platform.machine().lower() in
                  ("arm64", "aarch64") else "darwin-x64")
        asset = f"node-{NODE_VERSION}-{suffix}.{ext}"
        urls = [f"{m}/{NODE_VERSION}/{asset}" for m in NODE_MIRRORS]
        url = pick_fastest(urls, "node")
        try:
            tmp = paths.project_cache_dir("unpack", "node")
            archive = os.path.join(tmp, asset)
            download(asset, url, archive, timeout=90)
            unpack(archive, node_dir)
            flatten(node_dir)
            if not paths.bundled_tool("node"):
                warn("node 解包后未找到可执行文件（不影响启动）")
            else:
                log(f"✓ node → {paths.bundled_tool('node')}")
        except Exception as e:
            warn(f"node 下载失败（不影响启动）: {e}")
    # ffmpeg（音视频解码）
    ff_dir = paths.project_dir("environment", "tools", "ffmpeg")
    if not paths.bundled_tool("ffmpeg") and os_ in FFMPEG_MIRRORS:
        asset = FFMPEG_ASSET[os_]
        url = pick_fastest(list(FFMPEG_MIRRORS[os_]), "ffmpeg")
        try:
            tmp = paths.project_cache_dir("unpack", "ffmpeg")
            archive = os.path.join(tmp, asset)
            download(asset, url, archive, timeout=90)
            unpack(archive, ff_dir)
            flatten(ff_dir)
            if not place_binary(ff_dir, "ffmpeg"):
                warn("ffmpeg 解包后未找到可执行文件（不影响启动）")
            else:
                log(f"✓ ffmpeg → {paths.bundled_tool('ffmpeg')}")
        except Exception as e:
            warn(f"ffmpeg 下载失败（不影响启动）: {e}")


# ---------------------------------------------------------------------------
# 手动安装 / 更新（窗口内「环境与依赖」按钮走这里）
# ---------------------------------------------------------------------------
def install(pkgs: list, upgrade: bool, constraints: str = "") -> None:
    vp = ensure_venv()
    args = []
    if upgrade:
        args.append("--upgrade")
    if constraints:
        args += ["-c", constraints]
    args += pkgs
    pip_install(vp, args)
    log("✓ 安装完成（所有下载/缓存均在项目内）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stage", choices=("runtime", "tools", "all"))
    ap.add_argument("--install", nargs="*", default=None)
    ap.add_argument("--upgrade", action="store_true")
    ap.add_argument("--constraints", default="")
    ap.add_argument("--python-assets", action="store_true")
    a = ap.parse_args()

    if a.python_assets:
        print(python_asset())
        for u in python_urls():
            print(u)
        return 0
    if a.check:
        return 0 if check() else 1
    if a.install is not None:
        if not a.install:
            raise SystemExit("[setup] --install 至少需要一个包名")
        install(a.install, a.upgrade, a.constraints)
        return 0
    if a.stage in ("runtime", "all"):
        stage_runtime()
    if a.stage in ("tools", "all"):
        stage_tools()
    if not a.stage:
        ap.print_help()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[setup] ✗ {e}", file=sys.stderr, flush=True)
        sys.exit(1)
