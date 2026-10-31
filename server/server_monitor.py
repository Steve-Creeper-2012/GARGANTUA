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
import shlex
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


def _read_int(s):
    """把命令输出读成正整数；空/非数字/0 一律返回 None（宁可少报）。"""
    try:
        return int((s or "").strip()) or None
    except (ValueError, TypeError, AttributeError):
        return None


def _cpu_topology_linux():
    """Linux 物理/逻辑核心数：(physical, logical)，读不到为 None。

    优先 /proc/cpuinfo 的 `cpu cores`（单封装物理核）与 `siblings`
    （逻辑线程）；缺则回退 lscpu：Core(s) per socket × Socket(s) = 物理，
    CPU(s) = 逻辑。"""
    cores = siblings = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                if k == "cpu cores" and cores is None:
                    cores = _read_int(v)
                elif k == "siblings" and siblings is None:
                    siblings = _read_int(v)
                if cores is not None and siblings is not None:
                    break
    except OSError:
        pass
    if cores is not None and siblings is not None:
        return cores, siblings
    # 回退 lscpu
    out = _run(["lscpu"])
    physical, logical = cores, siblings
    if out:
        d = {}
        for line in out.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                d[k.strip()] = v.strip()
        cps = _read_int(d.get("Core(s) per socket"))
        socks = _read_int(d.get("Socket(s)"))
        if cps is not None and socks is not None:
            physical = cps * socks
        logical = _read_int(d.get("CPU(s)")) or logical
    return physical, logical


def _cpu_topology_windows():
    """Windows 物理/逻辑核心数：(physical, logical)，读不到为 None。

    PowerShell Get-CimInstance Win32_Processor 直接给 NumberOfCores /
    NumberOfLogicalProcessors；失败回退 wmic ... /format:csv。多 CPU 求和。"""
    out = _run(["powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Processor | "
                "Select-Object NumberOfCores,NumberOfLogicalProcessors | "
                "ConvertTo-Json -Compress"], timeout=15)
    physical = logical = None
    if out:
        try:
            data = json.loads(out)
            items = data if isinstance(data, list) else [data]
            p = sum(int(x["NumberOfCores"]) for x in items
                    if isinstance(x, dict) and x.get("NumberOfCores") is not None)
            l = sum(int(x["NumberOfLogicalProcessors"]) for x in items
                    if isinstance(x, dict)
                    and x.get("NumberOfLogicalProcessors") is not None)
            physical = p or None
            logical = l or None
        except (ValueError, TypeError, KeyError):
            pass
    if physical is None or logical is None:
        out = _run(["wmic", "cpu", "get",
                    "NumberOfCores,NumberOfLogicalProcessors", "/format:csv"],
                   timeout=15)
        if out:
            p = l = 0
            for line in out.splitlines():
                cols = [c.strip() for c in line.split(",")]
                if len(cols) < 3:
                    continue
                try:
                    p += int(cols[-2])
                    l += int(cols[-1])
                except ValueError:
                    continue          # 表头行（Node,NumberOfCores,...）
            physical = physical or (p or None)
            logical = logical or (l or None)
    return physical, logical


def _cpu_topology():
    """CPU 拓扑 {"cores": 物理核|None, "logical_cores": 逻辑|None,
    "hyperthreaded": bool}。

    macOS 用 sysctl hw.physicalcpu / hw.logicalcpu（回退 hw.ncpu）；
    Linux / Windows 见各自函数。任一读不到返回 None，绝不硬编码；
    hyperthreaded 仅在两者都读到且逻辑 > 物理时为真（4C8T → True）。"""
    system = platform.system()
    physical = logical = None
    if system == "Darwin":
        physical = _read_int(_run(["sysctl", "-n", "hw.physicalcpu"]))
        logical = _read_int(_run(["sysctl", "-n", "hw.logicalcpu"]))
        if logical is None:
            logical = _read_int(_run(["sysctl", "-n", "hw.ncpu"]))
    elif system == "Linux":
        physical, logical = _cpu_topology_linux()
    elif system == "Windows":
        physical, logical = _cpu_topology_windows()
    hyperthreaded = bool(physical is not None and logical is not None
                         and logical > physical)
    return {"cores": physical, "logical_cores": logical,
            "hyperthreaded": hyperthreaded}


def _cpu_cores():
    """物理核心数；读不到物理核则回退逻辑线程数（宁可回退，不可硬编码）。

    旧的 hw.ncpu 是逻辑线程数，4C8T 会被误标成 8 核；主页设备框要的是
    物理核，这里改为读 _cpu_topology()['cores']。"""
    topo = _cpu_topology()
    return topo["cores"] or topo["logical_cores"]


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


def _cpu_ticks():
    """Mach host_statistics(HOST_CPU_LOAD_INFO) → (user, system, idle, nice)。

    主页资源监视器 2s 轮询一次，`top -l 1` 单次就要 ~1s 且是自开机以来的
    累计值（不是瞬时值）；这里直接读内核 tick 计数，~1ms 且用两次采样的
    差分得到真正的瞬时占用。非 macOS / ctypes 不可用时返回 None，调用方
    回退到 ps 估算。"""
    try:
        import ctypes
        import ctypes.util
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        libc.mach_host_self.restype = ctypes.c_uint32
        info = (ctypes.c_uint32 * 4)()
        count = ctypes.c_uint32(4)
        # HOST_CPU_LOAD_INFO = 3, HOST_CPU_LOAD_INFO_COUNT = 4
        rc = libc.host_statistics(ctypes.c_uint32(libc.mach_host_self()),
                                  ctypes.c_int(3), ctypes.byref(info),
                                  ctypes.byref(count))
        if rc != 0:
            return None
        return tuple(int(x) for x in info)
    except Exception:
        return None


_CPU_LOCK = threading.Lock()
_CPU_STATE = {"ticks": None, "usage": None, "ts": 0.0}
CPU_SAMPLE_MIN_INTERVAL = 1.0   # 同一秒内重复调用直接复用（轮询去抖）


def _cpu_usage():
    """CPU 占用 %（user+system+nice）。Mach tick 差分优先，ps 求和兜底。"""
    now = time.time()
    with _CPU_LOCK:
        if (_CPU_STATE["usage"] is not None
                and now - _CPU_STATE["ts"] < CPU_SAMPLE_MIN_INTERVAL):
            return _CPU_STATE["usage"]

    usage = None
    ticks = _cpu_ticks()
    if ticks is not None:
        with _CPU_LOCK:
            prev = _CPU_STATE["ticks"]
            _CPU_STATE["ticks"] = ticks
        if prev is not None:
            d = [max(0, a - b) for a, b in zip(ticks, prev)]
            total = sum(d)
            if total > 0:
                usage = round(min(100.0, (d[0] + d[1] + d[3]) / total * 100), 2)
    if usage is None:                    # 首次采样 / 非 macOS
        out = _run(["ps", "-axo", "%cpu="], timeout=10)
        if out:
            try:
                # 归一化分母沿用【逻辑线程数】（旧 hw.ncpu 语义）：ps 单进程
                # %cpu 可超过 100%（多核），只有除以逻辑线程数才能得到
                # 0-100 的整体占用；_cpu_cores() 现返回物理核，不能直接用
                cores = (_cpu_topology()["logical_cores"]
                         or _cpu_cores() or 1)
                s = sum(float(x) for x in out.split())
                usage = round(min(100.0, s / cores), 2)
            except Exception:
                pass
    with _CPU_LOCK:
        _CPU_STATE.update({"usage": usage, "ts": now})
    return usage


# ---------------------------------------------------------------------------
# GPU 运行时读数（利用率 / 占用显存）
# ---------------------------------------------------------------------------
_GPU_RT = {"ts": 0.0, "data": None}
_GPU_RT_LOCK = threading.Lock()
GPU_RT_TTL = 1.0


def _gpu_runtime():
    """[{'usage': %, 'used': bytes}]，每个 IOAccelerator 一条。

    macOS 没有标准库途径读 GPU 利用率；`ioreg -c IOAccelerator` 的
    PerformanceStatistics 是唯一免 sudo 的来源（Apple Silicon 与
    Intel/AMD 独立显卡都有）。统一内存机型上 GPU 吃的是系统内存，
    对应 inUseSysMemoryBytes；独立显存机型对应 inUseVidMemoryBytes。
    读不到返回 []，前端渲染为「—」。"""
    now = time.time()
    with _GPU_RT_LOCK:
        if _GPU_RT["data"] is not None and now - _GPU_RT["ts"] < GPU_RT_TTL:
            return _GPU_RT["data"]
    out = _run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
               timeout=8)
    data = []
    if out:
        # 每张卡的 PerformanceStatistics 是一个 {...} 块；按出现位置切段，
        # 段内取利用率与在用内存。'Device Utilization % at cur p-state'
        # 因带尾随文字不会被 'Device Utilization %"=' 命中。
        for block in out.split('"PerformanceStatistics"')[1:]:
            u = re.search(r'"Device Utilization %"\s*=\s*(\d+)', block)
            if not u:
                continue
            used = 0
            for key in ("inUseSysMemoryBytes", "inUseVidMemoryBytes"):
                m = re.search(r'"' + key + r'"\s*=\s*(\d+)', block)
                if m:
                    used += int(m.group(1))
            data.append({"usage": float(u.group(1)),
                         "used": used if used else None})
    with _GPU_RT_LOCK:
        _GPU_RT.update({"ts": now, "data": data})
    return data


# ---------------------------------------------------------------------------
# GARGANTUA 进程列表（ps / PowerShell 一次读全量，逐条判定是否本项目进程）
# ---------------------------------------------------------------------------
# 项目入口脚本白名单：只有这些 basename 才能作为「其它 GARGANTUA 副本」
# 判定的凭证。单凭 basename 不够（grep 'server.py' 也会命中），必须再叠加
# 「路径含 GARGANTUA 目录段」这一约束，双重收紧后才收。
_PROJECT_SCRIPT_NAMES = {
    "server.py", "window.py", "init.py", "infer.py", "train.py",
    "auto_train.py", "tokenset.py",
}


def _is_project_script(basename):
    """basename 是否属于项目脚本白名单（codec_*.py 用前缀通配）。"""
    b = (basename or "").lower()
    return b in _PROJECT_SCRIPT_NAMES or b.startswith("codec_")


def _extract_scripts(args):
    """从命令行里抽出真正作为「脚本参数」的 .py 路径。

    用 shlex 按参数切分后，只有【独立成参且整体以 .py 结尾】的才算脚本：
      * `-c` / `-m` 的下一个参数是内联代码 / 模块名，不是脚本路径，
        即便整段命令恰好以 .py 结尾（如 `zsh -c "cd ... && python server.py"`
        经 shlex 会合成一个以 server.py 结尾的 token）也必须跳过；
      * 含 shell 元字符（& ; | < > $ `）的 token 是整段 shell 命令而非
        路径参数，跳过——真实路径参数不带这些字符（空格允许）；
      * `grep 'server.py' log` 里 'server.py' 是独立 token，但分支 A 要求
        路径含子目录、分支 B 要求 GARGANTUA 段，均拦得住；
      * 引号包裹的路径（含空格）由 shlex 剥壳；posix=False 保留的引号
        再手动 strip 一遍。
    Windows 用 posix=False：posix=True 会把 `C:\\a\\b.py` 里的反斜杠当
    转义吃掉，破坏 Windows 路径。畸形引号回退简单 split，不抛异常。"""
    posix = platform.system() != "Windows"
    try:
        parts = shlex.split(args or "", posix=posix)
    except ValueError:
        parts = (args or "").split()
    scripts = []
    skip_next = False
    for p in parts:
        if skip_next:
            skip_next = False
            continue
        # -c / -m 的值是内联代码 / 模块名，不是脚本路径；独立标志跳过
        # 下一个参数，附着形式（-cCODE / -mMODULE）直接跳过。
        if p in ("-c", "-m") or p.startswith("-c") or p.startswith("-m"):
            skip_next = (p in ("-c", "-m"))
            continue
        p = p.strip("\"'")
        # 含 shell 元字符的整段命令（sh -c "..." 的值）不是单个路径参数
        if any(ch in p for ch in "&;|<>$`"):
            continue
        if p.endswith(".py") and "=" not in p:
            scripts.append(p)
    return scripts


def _within(root, path):
    """path 是否位于 root 之下（含 root 自身）。跨平台盘符不同返回 False。"""
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:
        return False


def _is_gargantua_args(args, exe, root, allow_other=True):
    """判定一条进程命令行是否属于 GARGANTUA，返回 (mine: bool, name)。

    allow_other=False 时关闭「其它 GARGANTUA 副本」分支，只认当前 ROOT
    下的进程——api_system_kill 用它做严格白名单，防止跨目录误杀。

    name = 命中的 .py basename（server.py / window.py / init.py ...），
    让用户看到真实进程名；venv 可执行文件直接拉起且无脚本参数时用 exe 名。

    命中分支（任一即收）：
      A) 脚本路径落在当前项目根 ROOT 之下——覆盖正常启动的 server /
         window / init 等，无论 exe 是系统 python 还是隔离环境 python；
         相对路径分别基于 root 与 cwd 解析，修掉旧版只看
         「startswith(ROOT)」把相对脚本路径（service/init/init.py）漏掉的问题。
      B) 脚本 basename 在项目白名单内，且路径含 GARGANTUA 目录段——
         覆盖用户从【别的 GARGANTUA 副本目录】启动的进程（ps 里不含当前
         ROOT），同时用双重约束挡住 grep/编辑器里偶然提到的 .py。
      C) 可执行文件位于 root/environment/venv/bin 或 .../Scripts——
         隔离环境拉起的所有子进程（-m 内联、无 .py 参数）也算本项目。
    排除：只看「args 含 root」会误收从项目目录起的 zsh / Electron /
    编辑器；python -c / -m 内联无 .py 脚本不收；上述 A/B 的路径与
    白名单约束把命令行里恰好提及 .py 的无关进程挡在外面。"""
    if not args:
        return False, None
    root_abs = os.path.abspath(root)
    scripts = _extract_scripts(args)
    name = (os.path.basename(scripts[0]) if scripts
            else (os.path.basename(exe) if exe else None))

    # 分支 C：隔离环境 venv。exe 可能是相对路径（launcher 用的就是
    # `environment/venv/bin/python3`），既按原样判，也基于 root 解析后判。
    if exe:
        exe_cands = ([exe] if os.path.isabs(exe)
                     else [exe, os.path.join(root_abs, exe)])
        for ec in exe_cands:
            en = os.path.normcase(os.path.normpath(ec)).replace("\\", "/")
            for sub in ("bin", "Scripts"):
                vd = os.path.join(root_abs, "environment", "venv", sub)
                vn = os.path.normcase(os.path.normpath(vd)).replace("\\", "/") + "/"
                if en.startswith(vn):
                    return True, name

    for tok in scripts:
        base = os.path.basename(tok)
        # 分支 A：落在当前 ROOT 下。真实项目脚本都带子目录（service/init/
        # init.py 或 ROOT/server/server.py），据此与 bare 'server.py' 区分，
        # 避免把 grep 搜索串里的文件名当成被执行的脚本。
        if "/" in tok or "\\" in tok:
            for d in (root_abs, os.getcwd()):
                abs_p = os.path.abspath(tok if os.path.isabs(tok)
                                        else os.path.join(d, tok))
                if _within(root_abs, abs_p):
                    return True, base
        # 分支 B：其它副本。basename 白名单 + 路径含 GARGANTUA 段，缺一不可。
        segs = [s.lower() for s in re.split(r"[\\/]+", tok)]
        if (allow_other and _is_project_script(base)
                and any("gargantua" in s for s in segs)):
            return True, base
    return False, None


def _ps_process_rows():
    """非 Windows：ps -axo pid,%cpu,rss,args → [(pid, cpu, rss_kb, args)]。"""
    out = _run(["ps", "-axo", "pid=,%cpu=,rss=,args="], timeout=10)
    rows = []
    for line in (out or "").splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        try:
            pid, cpu, rss = int(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        rows.append((pid, cpu, rss, parts[3]))
    return rows


def _windows_process_rows():
    """Windows：PowerShell Get-CimInstance → JSON → [(pid, None, rss_kb, args)]。

    Windows 无 ps；按进程瞬时 CPU 占用 CIM 不提供，宁可少报 → cpu=None。
    WorkingSetSize 是字节，转成 KB 与 ps rss 单位对齐（后续 rss*1024 还原）。
    PowerShell 失败回退 wmic（/format:csv）。"""
    out = _run(["powershell", "-NoProfile", "-Command",
                "Get-CimInstance Win32_Process | "
                "Select-Object ProcessId,CommandLine,WorkingSetSize | "
                "ConvertTo-Json -Compress"], timeout=15)
    rows = []
    if out:
        try:
            data = json.loads(out)
            items = data if isinstance(data, list) else [data]
            for it in items:
                if not isinstance(it, dict):
                    continue
                pid = int(it.get("ProcessId"))
                ws = it.get("WorkingSetSize")
                rss_kb = (float(ws) / 1024.0) if ws else 0.0
                rows.append((pid, None, rss_kb, it.get("CommandLine") or ""))
        except (ValueError, TypeError, KeyError):
            rows = []
    if not rows:
        rows = _wmic_process_rows()
    return rows


def _wmic_process_rows():
    """wmic process get ProcessId,CommandLine,WorkingSetSize /format:csv 回退。

    列序：Node,ProcessId,CommandLine,WorkingSetSize；CommandLine 可能含逗号，
    故用右侧倒数定位数值列、中间拼回命令行。"""
    out = _run(["wmic", "process", "get",
                "ProcessId,CommandLine,WorkingSetSize", "/format:csv"],
               timeout=15)
    rows = []
    for line in (out or "").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("node,"):
            continue
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 4:
            continue
        try:
            pid = int(cols[1])
            rss_kb = float(cols[-1]) / 1024.0
        except ValueError:
            continue
        rows.append((pid, None, rss_kb, ",".join(cols[2:-1])))
    return rows


def _process_ports(pids):
    """{pid: [本地端口, ...] 升序}。一次 lsof / netstat 查全部目标 pid。

    macOS/Linux：lsof -nP -i -a -p pid1,pid2,...（单次调用，无逐 pid 开销）。
    NAME 列形如 127.0.0.1:8765 / *:8765 / [::]:8765，取末个冒号后的数字；
    macOS 的 TCP 行尾会缀 "(LISTEN)" 等状态括号，NAME 实际在倒数第二列。
    Windows：netstat -ano（TCP+UDP），末列 PID，本地地址在第 2 列。
    查不到（超时 / 无 lsof）返回 {}，前端渲染「—」——宁可少报，不可误报。"""
    if not pids:
        return {}
    pid_set = set(pids)
    ports = {}
    if platform.system() == "Windows":
        out = _run(["netstat", "-ano"], timeout=12)
        for line in (out or "").splitlines():
            cols = line.split()
            if len(cols) < 3 or cols[0].lower() not in ("tcp", "udp"):
                continue
            try:
                pid = int(cols[-1])
            except ValueError:
                continue          # 表头行 / 分隔线
            if pid not in pid_set:
                continue
            m = re.search(r":(\d+)$", cols[1])
            if m:
                ports.setdefault(pid, set()).add(int(m.group(1)))
    else:
        out = _run(["lsof", "-nP", "-i", "-a",
                    "-p", ",".join(str(p) for p in sorted(pid_set))], timeout=8)
        for line in (out or "").splitlines()[1:]:
            cols = line.split()
            if len(cols) < 9:
                continue
            try:
                pid = int(cols[1])
            except ValueError:
                continue
            if pid not in pid_set:
                continue            # lsof 正常只回目标 pid，此处兜底过滤
            name = cols[-1]
            if name.startswith("(") and len(cols) >= 10:
                name = cols[-2]   # "127.0.0.1:8765 (LISTEN)" → NAME 在倒数第二列
            m = re.search(r":(\d+)$", name)
            if m:
                ports.setdefault(pid, set()).add(int(m.group(1)))
    return {p: sorted(v) for p, v in ports.items()}


def _gargantua_processes(include_other=True):
    """本项目相关进程：[{pid, name, ports, cpu, mem_bytes, mem_pct, gpu_pct,
    vram_bytes, external}]。

    include_other=True 连带列出「其它 GARGANTUA 副本」的进程（信息完整，
    标 external=True）；False 只认当前 ROOT 下的进程。api_system_kill 必须用
    False 的严格集合——否则本工作台的「结束」按钮能杀掉别的副本的 server。
    平台分支：非 Windows 走 ps；Windows 走 PowerShell/wmic（cpu=None）。
    GPU/显存按进程占用无标准库途径可读 → 恒为 None（前端显示 —）。
    端口来自一次 lsof / netstat（_process_ports），读不到为空列表。
    按 CPU/内存降序（cpu=None 视为 0）排序。"""
    is_win = platform.system() == "Windows"
    rows = _windows_process_rows() if is_win else _ps_process_rows()
    total_mem, _ = _mem_stat()
    procs = []
    for pid, cpu, rss_kb, args in rows:
        exe = args.split()[0] if args else ""
        mine, name = _is_gargantua_args(args, exe, ROOT, include_other)
        if not mine:
            continue
        # 是否来自「其它副本」：宽松命中但严格（只看当前 ROOT）不命中
        external = False
        if include_other:
            strict, _ = _is_gargantua_args(args, exe, ROOT, False)
            external = not strict
        mem_bytes = int(rss_kb * 1024)
        procs.append({
            "pid": pid,
            "name": name,
            "cpu": round(cpu, 2) if cpu is not None else None,
            "mem_bytes": mem_bytes,
            "mem_pct": (round(mem_bytes / total_mem * 100, 2)
                        if total_mem else None),
            "gpu_pct": None,
            "vram_bytes": None,
            "external": external,
            "self": (pid == os.getpid()),
        })
    if procs:
        port_map = _process_ports([p["pid"] for p in procs])
        for p in procs:
            p["ports"] = port_map.get(p["pid"]) or []
    procs.sort(key=lambda p: (-(p["cpu"] or 0.0), -p["mem_bytes"]))
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
# 版本检查结果带 TTL：成功缓存 5 分钟，失败只缓存 1 分钟（便于重试）。
# 之前 ERR 永久缓存 + force 不清 → 刷新按钮（?force=1）形同虚设。
VERSIONS_CACHE_TTL = 300.0
VERSIONS_ERR_TTL = 60.0
MAX_SIMPLE_BYTES = 8 * 1024 * 1024   # simple 索引页读取上限，防超大响应

def clear_versions_cache():
    """刷新按钮（force=1）必须真的重新查镜像，清掉版本缓存。"""
    with _VERSIONS_LOCK:
        _VERSIONS_CACHE.clear()


def _ssl_context():
    """HTTPS 证书链：certifi → macOS 系统 CA（/etc/ssl/cert.pem）→ 兜底不校验。

    python.org 官方解释器默认不带 CA bundle（CERTIFICATE_VERIFY_FAILED），
    pip 能装包是因为它自带证书；这里只做只读的版本查询。
    2026-10-29 审计收紧：certifi / 系统 CA 都不可用时不再静默降级到
    不校验证书（版本号可能被中间人篡改）。确需该回退必须显式设置
    GARGANTUA_INSECURE_PYPI=1，否则版本检查返回不可用（宁可少报）。"""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    if os.path.exists("/etc/ssl/cert.pem"):
        return ssl.create_default_context(cafile="/etc/ssl/cert.pem")
    if os.environ.get("GARGANTUA_INSECURE_PYPI") == "1":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    raise RuntimeError(
        "无可用 CA 证书链，已禁用不校验回退"
        "（可设 GARGANTUA_INSECURE_PYPI=1 显式允许）")


def _pypi_versions(pkg: str):
    """镜像 simple 页里出现的所有版本号（未按约束过滤）。失败返回 None。"""
    now = time.time()
    with _VERSIONS_LOCK:
        hit = _VERSIONS_CACHE.get(pkg)
    if hit is not None:
        ttl = VERSIONS_ERR_TTL if hit["data"] == "ERR" else VERSIONS_CACHE_TTL
        if now - hit["ts"] < ttl:
            return None if hit["data"] == "ERR" else hit["data"]
    try:
        url = f"{PYPI_MIRROR}/{pkg}/"
        with urlopen(url, timeout=12, context=_ssl_context()) as r:
            # 限长：镜像异常返回超大页时不把整页读进内存
            html = r.read(MAX_SIMPLE_BYTES).decode("utf-8", "replace")
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
            _VERSIONS_CACHE[pkg] = {"ts": time.time(), "data": result}
        return result
    except Exception:
        with _VERSIONS_LOCK:
            _VERSIONS_CACHE[pkg] = {"ts": time.time(), "data": "ERR"}
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
        rt = _gpu_runtime()
        unified = platform.machine() == "arm64"   # Apple Silicon = 统一内存
        mem_total, mem_used = _mem_stat()
        vram_total = sum(g["vram_total"] or 0 for g in gpus) or None
        gpu_models = []
        for i, g in enumerate(gpus):
            note = None
            if torch_ok is False:
                note = f"{g['model']} 不支持 torch"
            elif torch_ok is None:
                note = "torch 未安装，GPU 不可用"
            r = rt[i] if i < len(rt) else {}
            gpu_models.append({"model": g["model"], "vram_total": g["vram_total"],
                               "torch_ok": torch_ok, "note": note,
                               "usage": r.get("usage"), "used": r.get("used")})
        vram_used = sum(g["used"] or 0 for g in gpu_models) or None
        topo = _cpu_topology()
        return 200, {
            "cpu": {"model": _cpu_model(),
                    # cores 保留做向后兼容：物理核优先，读不到回退逻辑数
                    "cores": topo["cores"] or topo["logical_cores"],
                    "logical_cores": topo["logical_cores"],
                    "physical_cores": topo["cores"],
                    "hyperthreaded": topo["hyperthreaded"],
                    "usage": _cpu_usage()},
            "memory": {"total": mem_total, "used": mem_used,
                       "unified": unified},
            # 统一内存机型没有独立显存：显存与内存是同一块（前端合并显示）
            "vram": {"total": None if unified else vram_total,
                     "used": vram_used},
            "gpus": gpu_models,
            "unified_memory": unified,
        }

    def api_system_processes(self):
        return 200, {"processes": _gargantua_processes()}

    def api_system_kill(self, body):
        pid = body.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int):
            raise ValueError(f"pid 必须是整数: {pid!r}")
        # 诚实：进程监视器会把 server 自身（server.py）也列出来，它确实是一
        # 个 GARGANTUA 进程，用户有权结束它（前端对其会弹二次确认）。后端不
        # 单独拦截 self，只做目录边界校验——只允许结束当前项目目录内的进程，
        # 越界（其它副本 / 外部进程，标 external）一律拒绝，防止误杀别的
        # GARGANTUA 副本。
        known = {p["pid"] for p in _gargantua_processes(include_other=False)}
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
        if force:
            clear_versions_cache()   # 刷新按钮：强制重新查镜像，不走永久 ERR
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
