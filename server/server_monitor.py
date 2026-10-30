# ===========================================================================
# GARGANTUA — server_monitor.py：主页「设备 / 资源监视器 / 依赖更新」后端
# ---------------------------------------------------------------------------
# 2026-10-29 UI 重构配套：主页要显示设备大框（CPU/GPU/内存/显存）、
# 资源监视器（占用 bar + GARGANTUA 进程表）、依赖卡片（安装/更新按钮）。
#
# 实现约束：
#   * 纯标准库——隔离环境里没有 psutil，也不为监视器引入新依赖
#     （macOS 读数走 sysctl / vm_stat / top / ps / system_profiler）。
#   * 读不到的项一律返回 None，前端渲染为「—」——宁可少报，不可误报。
#   * 更新检查走清华 PyPI 镜像（实测官方源 ~5KB/s，镜像 ~18MB/s），
#     并尊重 start/requirements/core.txt 的版本约束：torch==2.2.2 是
#     最后一个带 macOS x86_64 轮子的版本，钉住的包不报「可更新」，
#     否则一个更新按钮就把主依赖契约（torch 2.2.2 / numpy<2）破坏掉。
# ===========================================================================
from __future__ import annotations

import json
import os
import platform
import re
import ssl
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

from server_env import ROOT
from server_util import pip_index_args

PYPI_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"
UPDATES_CACHE_TTL = 600.0      # 更新检查结果缓存 10 分钟（?force=1 可绕过）
GPU_CACHE_TTL = 300.0          # GPU 型号/显存基本不变，缓存 5 分钟


def _run(cmd, timeout=10):
    """子进程小工具：成功返回 stdout，任何失败返回空串。"""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return r.stdout
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# 设备静态信息（型号 / 总量）：变化极少，全部带 TTL 缓存
# ---------------------------------------------------------------------------
_GPU_CACHE = {"ts": 0.0, "gpus": None, "torch_ok": None}
_GPU_LOCK = threading.Lock()


def _parse_vram(s: str):
    """'1536 MB' → 字节数；读不出返回 None。"""
    m = re.search(r"([\d.]+)\s*(KB|MB|GB|TB)", s or "", re.I)
    if not m:
        return None
    v = float(m.group(1))
    return int(v * {"kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3,
                    "tb": 1024 ** 4}[m.group(2).lower()])


def _torch_gpu_ok():
    """GPU 是否被 torch 支持（MPS/CUDA）。torch 未装返回 None。

    torch 在 server 主进程是懒加载（import 一次后驻留 sys.modules），
    首次调用可能慢几秒，但结果随 _GPU_CACHE 一起缓存，不拖累轮询。"""
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            return True
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True
    except Exception:
        pass
    return False


def _gpu_info():
    """GPU 列表：[{model, vram_total}]；读不出返回空列表。"""
    now = time.time()
    with _GPU_LOCK:
        if _GPU_CACHE["gpus"] is not None and now - _GPU_CACHE["ts"] < GPU_CACHE_TTL:
            return _GPU_CACHE["gpus"], _GPU_CACHE["torch_ok"]
    gpus = []
    out = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=25)
    if out:
        try:
            for it in (json.loads(out).get("SPDisplaysDataType") or []):
                name = it.get("sppci_model") or it.get("_name") or "GPU"
                vram = _parse_vram(it.get("spdisplays_vram_shared")
                                   or it.get("spdisplays_vram") or "")
                gpus.append({"model": name, "vram_total": vram})
        except Exception:
            pass
    torch_ok = _torch_gpu_ok()
    with _GPU_LOCK:
        _GPU_CACHE.update({"ts": now, "gpus": gpus, "torch_ok": torch_ok})
    return gpus, torch_ok


def _cpu_model():
    out = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
    if out:
        return out
    return platform.processor() or "CPU"


def _cpu_cores():
    try:
        return int(_run(["sysctl", "-n", "hw.ncpu"]).strip() or 0) or None
    except Exception:
        return None


def _mem_stat():
    """(total_bytes, used_bytes)。used = total - (free + inactive + speculative)。"""
    total = 0
    try:
        total = int(_run(["sysctl", "-n", "hw.memsize"]).strip() or 0)
    except Exception:
        pass
    used = None
    out = _run(["vm_stat"], timeout=5)
    if out and total:
        m = re.search(r"page size of (\d+) bytes", out)
        page = int(m.group(1)) if m else 4096

        def pages(label):
            # 注意：vm_stat 输出形如 "Pages free: 12345."，键名不带引号，
            # 行尾还有句号——之前误带引号匹配，全部落 0，used 恒等于 total
            mm = re.search(re.escape(label) + r":\s+(\d+)", out)
            return int(mm.group(1)) if mm else 0

        freeish = (pages("Pages free") + pages("Pages speculative")
                   + pages("Pages inactive") + pages("Pages purgeable"))
        used = max(0, total - freeish * page)
    return total, used


def _cpu_usage():
    """CPU 占用 %（user+sys）。top 读不到时回退 ps 求和 / 核心数。"""
    out = _run(["top", "-l", "1", "-n", "0"], timeout=15)
    m = re.search(r"CPU usage:\s*([\d.]+)% user,\s*([\d.]+)% sys", out)
    if m:
        return round(min(100.0, float(m.group(1)) + float(m.group(2))), 2)
    out = _run(["ps", "-axo", "%cpu="], timeout=10)
    if out:
        try:
            cores = _cpu_cores() or 1
            s = sum(float(x) for x in out.split())
            return round(min(100.0, s / cores), 2)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# GARGANTUA 进程列表（ps 一次读全量，按 ROOT 路径过滤）
# ---------------------------------------------------------------------------
def _gargantua_processes():
    """本项目相关进程：[{pid, name, cpu, mem_bytes, mem_pct, gpu_pct}]。

    判定标准（两个条件取其一，缺一不可地收紧）：
      * 可执行文件在项目隔离环境 venv 里（environment/venv/bin/*）——
        server / window / infer / train / init 全部由它拉起；
      * 或命令行里有位于项目根下的 .py 脚本。
    只看「args 含 ROOT」不够：从项目目录里起的 zsh / Electron 等无关
    进程的命令行也会带这个路径，之前把它们全收进来了。
    GPU/显存的按进程占用 macOS 无标准库途径可读 → 恒为 None（前端显示 —）。"""
    venv_bin = os.path.join(ROOT, "environment", "venv", "bin")
    out = _run(["ps", "-axo", "pid=,%cpu=,rss=,args="], timeout=10)
    total_mem, _ = _mem_stat()
    procs = []
    for line in out.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid, cpu, rss = int(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        args = parts[3]
        exe = args.split()[0]
        script = re.search(r"(\S+\.py)", args)
        mine = exe.startswith(venv_bin) or (
            script and script.group(1).startswith(ROOT))
        if not mine:
            continue
        name = (os.path.basename(script.group(1)) if script
                else os.path.basename(exe))
        mem_bytes = rss * 1024
        procs.append({
            "pid": pid,
            "name": name,
            "cpu": round(cpu, 2),
            "mem_bytes": int(mem_bytes),
            "mem_pct": (round(mem_bytes / total_mem * 100, 2)
                        if total_mem else None),
            "gpu_pct": None,
            "vram_bytes": None,
        })
    procs.sort(key=lambda p: (-p["cpu"], -p["mem_bytes"]))
    return procs


# ---------------------------------------------------------------------------
# 依赖更新检查（清华镜像 + core.txt 版本约束）
# ---------------------------------------------------------------------------
_DIST_NAME = {"PIL": "pillow", "cv2": "opencv-python"}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _ver_tuple(v: str):
    """"1.26.4" → (1,26,4)；"4.10.0.84" → (4,10,0,84)。非数字段截断。"""
    out = []
    for part in re.split(r"[.+-]", v.split("/")[0]):
        m = re.match(r"\d+", part)
        if not m:
            break
        out.append(int(m.group(0)))
    return tuple(out) or (0,)


def _core_constraints() -> dict:
    """core.txt → {规范包名: [(op, version), ...]}。torch==2.2.2 / numpy<2。"""
    path = os.path.join(ROOT, "start", "requirements", "core.txt")
    spec = {}
    try:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^([A-Za-z0-9_.-]+)\s*([<>=!~].*)?$", line)
            if not m:
                continue
            conds = []
            for tok in re.findall(r"(==|!=|<=|>=|<|>)\s*([\d.]+)", m.group(2) or ""):
                conds.append((tok[0], _ver_tuple(tok[1])))
            spec[_norm(m.group(1))] = conds
    except Exception:
        pass
    return spec


def _spec_ok(ver: tuple, conds) -> bool:
    for op, ref in conds:
        if op == "==" and ver != ref:
            return False
        if op == "!=" and ver == ref:
            return False
        if op == "<" and not ver < ref:
            return False
        if op == "<=" and not ver <= ref:
            return False
        if op == ">" and not ver > ref:
            return False
        if op == ">=" and not ver >= ref:
            return False
    return True


_VERSIONS_CACHE = {}
_VERSIONS_LOCK = threading.Lock()


def _ssl_context():
    """HTTPS 证书链：certifi → macOS 系统 CA（/etc/ssl/cert.pem）→ 兜底不校验。

    python.org 官方解释器默认不带 CA bundle（CERTIFICATE_VERIFY_FAILED），
    pip 能装包是因为它自带证书；这里只做只读的版本查询，逐级回退保证
    功能可用，最后一级不校验证书的风险是版本号可能被中间人篡改——
    可接受（安装动作仍走 pip，pip 用自己的证书链校验）。"""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    if os.path.exists("/etc/ssl/cert.pem"):
        return ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _pypi_versions(pkg: str):
    """镜像 simple 页里出现的所有版本号（未按约束过滤）。失败返回 None。"""
    with _VERSIONS_LOCK:
        hit = _VERSIONS_CACHE.get(pkg)
    if hit is not None:
        return hit if hit != "ERR" else None
    try:
        url = f"{PYPI_MIRROR}/{pkg}/"
        with urlopen(url, timeout=12, context=_ssl_context()) as r:
            html = r.read().decode("utf-8", "replace")
        norm = _norm(pkg)
        # 包名里的分隔符在文件名里 -/_/. 混用（opencv-python / opencv_python），
        # 要先把原始名字按 - 拆段再逐段 re.escape——先 escape 再 replace 会把
        # re.escape 产生的 \- 里的 - 一起替换掉，模式直接报废（torch 等无
        # 连字符的包侥幸不受影响，才不容易发现）
        frag = "[-_.]+".join(re.escape(p) for p in norm.split("-"))
        pat = re.compile(r"/" + frag
                         + r"-([0-9][0-9A-Za-z.]*)(?:-[^/#]+)?"
                           r"\.(?:whl|tar\.gz|zip)", re.I)
        vers = {m.group(1).rstrip(".") for m in pat.finditer(html)}
        if not vers:
            raise RuntimeError("no versions parsed")
        result = sorted(vers, key=_ver_tuple)
        with _VERSIONS_LOCK:
            _VERSIONS_CACHE[pkg] = result
        return result
    except Exception:
        with _VERSIONS_LOCK:
            _VERSIONS_CACHE[pkg] = "ERR"
        return None


def _check_one_update(mod: str, installed: str, constraints: dict):
    """单依赖：{pkg, latest, update_available, checked}。"""
    pkg = _DIST_NAME.get(mod, mod)
    conds = constraints.get(_norm(pkg), [])
    versions = _pypi_versions(pkg)
    if versions is None:
        return {"pkg": pkg, "latest": None, "checked": False,
                "update_available": False}
    pinned = any(op == "==" for op, _ in conds)
    # 非 == 钉住时不考虑预发布版（0.9.0rc0 这类），避免把 rc 当「最新」
    if not pinned:
        versions = [v for v in versions if not re.search(r"[A-Za-z]", v)]
    valid = [v for v in versions if _spec_ok(_ver_tuple(v), conds)]
    latest = max(valid, key=_ver_tuple) if valid else None
    upd = bool(installed and latest
               and _ver_tuple(installed) < _ver_tuple(latest))
    return {"pkg": pkg, "latest": latest, "checked": True,
            "update_available": upd}


_UPDATES_CACHE = {"ts": 0.0, "data": None}
_UPDATES_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# API mixin
# ---------------------------------------------------------------------------
class MonitorApi:
    """主页监视相关端点：devices / processes / kill / updates / install。"""

    def api_system_devices(self):
        gpus, torch_ok = _gpu_info()
        unified = platform.machine() == "arm64"   # Apple Silicon = 统一内存
        mem_total, mem_used = _mem_stat()
        vram_total = sum(g["vram_total"] or 0 for g in gpus) or None
        gpu_models = []
        for g in gpus:
            note = None
            if torch_ok is False:
                note = f"{g['model']} 不支持 torch"
            elif torch_ok is None:
                note = "torch 未安装，GPU 不可用"
            gpu_models.append({"model": g["model"], "vram_total": g["vram_total"],
                               "torch_ok": torch_ok, "note": note})
        return 200, {
            "cpu": {"model": _cpu_model(), "cores": _cpu_cores(),
                    "usage": _cpu_usage()},
            "memory": {"total": mem_total, "used": mem_used,
                       "unified": unified},
            "vram": {"total": None if unified else vram_total,
                     "used": None},       # macOS 无标准库途径读显存占用
            "gpus": gpu_models,
            "unified_memory": unified,
        }

    def api_system_processes(self):
        return 200, {"processes": _gargantua_processes()}

    def api_system_kill(self, body):
        pid = body.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int):
            raise ValueError(f"pid 必须是整数: {pid!r}")
        known = {p["pid"] for p in _gargantua_processes()}
        if pid not in known:
            raise ValueError(f"pid {pid} 不属于 GARGANTUA，拒绝终止")
        import signal
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            raise ValueError(f"进程不存在: {pid}")
        except PermissionError:
            raise RuntimeError(f"无权限终止进程 {pid}")
        return 200, {"killed": pid, "note": "已发送终止信号"}

    def api_system_updates(self, query=None):
        force = bool(query and query.get("force", ["0"])[0] not in ("0", "", "no"))
        now = time.time()
        with _UPDATES_LOCK:
            if (not force and _UPDATES_CACHE["data"] is not None
                    and now - _UPDATES_CACHE["ts"] < UPDATES_CACHE_TTL):
                return 200, _UPDATES_CACHE["data"]
        deps = self.api_system()[1]["dependencies"]
        constraints = _core_constraints()
        items = [{"name": d["name"], "version": d.get("version"),
                  "installed": bool(d["ok"])} for d in deps]
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(_check_one_update, it["name"], it["version"],
                              constraints): it for it in items if it["installed"]}
            for fut, it in futs.items():
                it.update(fut.result())
        data = {"updates": items}
        with _UPDATES_LOCK:
            _UPDATES_CACHE.update({"ts": now, "data": data})
        return 200, data

    def api_system_install(self, body):
        """安装 / 更新依赖（异步任务）。body: {names: [...], upgrade?: bool}

        装进当前跑 server 的解释器；必须带 core.txt 约束（-c），否则装
        opencv/soundfile 时 pip 会顺手把 numpy 顶到 2.x，破坏 torch 2.2.2
        的主依赖契约；升级 torch 也会被 ==2.2.2 的钉住挡回来。"""
        names = body.get("names") or []
        if isinstance(names, str):
            names = [names]
        if not names:
            raise ValueError("缺少 names")
        optional = getattr(self, "_OPTIONAL_DEPS", {})
        dist = dict(_DIST_NAME)
        dist.update({k: v["pkg"] for k, v in optional.items()})
        pkgs = []
        for n in names:
            if n not in {d["name"] for d in self.api_system()[1]["dependencies"]}:
                raise ValueError(f"未知依赖: {n!r}")
            pkgs.append(dist.get(n, n))
        upgrade = bool(body.get("upgrade"))
        req = os.path.join(boot_project_root(), "start", "requirements",
                           "core.txt")

        def work():
            cmd = [sys.executable, "-m", "pip", "install"]
            if os.path.exists(req):
                cmd += ["-c", req]
            if upgrade:
                cmd += ["-U"]
            cmd += pip_index_args() + pkgs
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=1800)
            if r.returncode != 0:
                raise RuntimeError("pip 安装失败: "
                                   + (r.stderr or r.stdout)[-400:])
            return {"installed": pkgs, "upgrade": upgrade,
                    "note": "已安装；Python 不会热更新已 import 的模块，"
                            "重启 server 后生效"}

        tid = self.pool.submit(work)
        return 202, {"task_id": tid}


def boot_project_root() -> str:
    return ROOT
