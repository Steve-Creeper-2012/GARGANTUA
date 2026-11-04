# ===========================================================================
# GARGANTUA v1 — server/reqs.py：依赖清单（requirement）的解析与状态
# ---------------------------------------------------------------------------
# start/requirements/ 下两份清单，是「这个项目需要什么」的唯一声明处：
#   must.txt       启动必备（轻量）：Python 跑 server + pywebview 开窗口。
#                  启动时**只检查这一份**（server_env.check_environment）。
#   necessary.txt  正常工作需要：torch / numpy / pillow / safetensors /
#                  ffmpeg / rust。启动不检查，用户按需在主页安装。
#
# 每行一个依赖：`名字>=下限版本`（可多个约束，用逗号分隔）。语义：
#   · 下限（>=）不是上限——编译器/运行时向后兼容，用户可以装更新的版本
#     （主页里能选具体版本），默认装下限那版；
#   · 上限（< / <=）与 == 是硬约束（numpy 2.x 与 torch 2.2.2 不兼容），
#     主页的可选版本列表会先把违规版本滤掉。
#
# 「怎么查 / 怎么装」写在 META 里：pip 包走 importlib.metadata，系统命令
# 走 which，运行时与工具链走 environment/ 下的探针。
# ===========================================================================
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from server_env import ENV_ROOT, ROOT

REQ_DIR = Path(ROOT) / "start" / "requirements"
MUST_FILE = "must.txt"
NECESSARY_FILE = "necessary.txt"

# 名字 → 展示名 / 种类 / 怎么查 / 怎么装
#   kind: pip（pip 包）| runtime（解释器）| system（系统命令）| toolchain（工具链）
#   module: import 名（pip 包的探测与安装用）
#   dist:   pip 包名（与 module 可能不同：PIL → pillow）
#   pack:   对应 environment/ 下的文件夹（pip 包都在 venv 里）
#   cmd:    项目内二进制名（environment/tools/ 下，绝不查系统 PATH）
META = {
    "python": {"title": "Python 运行时", "kind": "runtime",
               "pack": "python", "probe": "python/bin/python3",
               "version_arg": "-V",
               "install": "sh launcher/setup-environment.sh"},
    "pywebview": {"title": "pywebview（桌面窗口）", "kind": "pip",
                  "module": "webview", "dist": "pywebview", "pack": "venv",
                  "soft": True,
                  "install": "窗口内点「安装」，或 python start/setup_env.py --install pywebview"},
    "torch": {"title": "PyTorch（训练 / 推理）", "kind": "pip",
              "module": "torch", "dist": "torch", "pack": "venv",
              "install": "窗口内点「安装」，或 python start/setup_env.py --install <包名>"},
    "numpy": {"title": "NumPy（数值计算）", "kind": "pip",
              "module": "numpy", "dist": "numpy", "pack": "venv",
              "install": "窗口内点「安装」，或 python start/setup_env.py --install <包名>"},
    "pillow": {"title": "Pillow（图像读写）", "kind": "pip",
               "module": "PIL", "dist": "pillow", "pack": "venv",
               "install": "窗口内点「安装」，或 python start/setup_env.py --install <包名>"},
    "safetensors": {"title": "Safetensors（权重格式）", "kind": "pip",
                    "module": "safetensors", "dist": "safetensors",
                    "pack": "venv",
                    "install": "窗口内点「安装」，或 python start/setup_env.py --install <包名>"},
    "ffmpeg": {"title": "FFmpeg（项目内二进制）", "kind": "bundled",
               "cmd": "ffmpeg",
               "install": "python start/setup_env.py --stage tools"},
    "rust": {"title": "Rust 工具链（GPU 图元转换器）", "kind": "toolchain",
             "pack": "rust", "probe": "cargo/bin/cargo",
             "version_arg": "--version",
             "install": "sh service/prims/setup.sh"},
    # wgpu 不是 pip 包、也不在 requirements/*.txt 里——它是 fast_shape_render
    # 的 Cargo 依赖，构建 Rust GPU 图元转换器时静态链进二进制。但它确实是
    # 「这个项目需要什么」的一部分，单独列出来主页才不会漏掉它（之前 rust
    # 就因为这个被隐身过）。检测到转换器 release 二进制即视为就位。
    "wgpu": {"title": "wgpu（GPU 图元渲染后端）", "kind": "wgpu",
             "group": "necessary",
             "install": "sh service/prims/setup.sh（构建 Rust 转换器时一并链入）",
             "why": "Rust/wgpu GPU 图元转换器依赖（Metal / DX12 / Vulkan 抽象层）"},
}

# requirements/*.txt 之外、仍属于「这个项目需要什么」的依赖（目前只有 wgpu：
# 它是 Cargo 依赖，写不进 pip 清单）。它们随 all_requirements() 一起返回给前端，
# 但不会进 must 分组（不影响启动检查）。
EXTRA_DEPS = ["wgpu"]

_LINE_RE = re.compile(r"^([A-Za-z0-9_.-]+)\s*([<>=!~].*)?$")


def _vt(v: str):
    """"1.26.4" → (1,26,4)；"4.10.0.84" → (4,10,0,84)。非数字段截断。"""
    out = []
    for part in re.split(r"[.+-]", str(v).split("/")[0]):
        m = re.match(r"\d+", part)
        if not m:
            break
        out.append(int(m.group(0)))
    return tuple(out) or (0,)


def ver_ge(a: str, b: str) -> bool:
    """a >= b（按数字段比）。"""
    ta, tb = _vt(a), _vt(b)
    n = max(len(ta), len(tb))
    ta = ta + (0,) * (n - len(ta))
    tb = tb + (0,) * (n - len(tb))
    return ta >= tb


def _parse_constraints(raw: str):
    """'>=1.24,<2' → [(('>=', '1.24'), ('<', '2'))]（去空格）。"""
    out = []
    for tok in re.findall(r"(==|!=|<=|>=|<|>)\s*([\d.]+)", raw or ""):
        out.append((tok[0], tok[1]))
    return out


def parse_file(path: Path) -> list:
    """读一份清单 → [{name, min, constraints, raw}]。"""
    out = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        name = m.group(1).strip().lower()
        cons = _parse_constraints(m.group(2) or "")
        mins = [v for op, v in cons if op == ">="]
        out.append({
            "name": name,
            "min": mins[0] if mins else "",
            "constraints": cons,
            "raw": line,
        })
    return out


def load_requirements() -> dict:
    """{"must": [...], "necessary": [...]}（每次调用都读文件，改了立刻生效）。"""
    return {
        "must": parse_file(REQ_DIR / MUST_FILE),
        "necessary": parse_file(REQ_DIR / NECESSARY_FILE),
    }


def all_requirements() -> list:
    """全部依赖（must 在前），每条补上展示元数据。

    requirements/*.txt 是主声明；EXTRA_DEPS 里那些（wgpu 等）写在 Cargo /
    工具链层、进不了 pip 清单的依赖，也一并列出，前端才不会漏掉。
    """
    reqs = load_requirements()
    out = []
    seen = set()
    for group in ("must", "necessary"):
        for r in reqs[group]:
            name = r["name"]
            if name in seen:
                continue
            seen.add(name)
            meta = META.get(name, {})
            out.append({**r, "group": group, **meta})
    for name in EXTRA_DEPS:
        if name in seen:
            continue
        meta = META.get(name)
        if not meta:
            continue
        seen.add(name)
        out.append({"name": name, "min": "", "constraints": [],
                    "raw": name, "group": meta.get("group", "necessary"), **meta})
    return out


def _installed_pip_version(meta: dict):
    """pip 包：当前解释器里装了没、什么版本。"""
    module = meta.get("module")
    if not module:
        return False, ""
    try:
        import importlib.metadata as md
        ver = md.version(meta.get("dist") or module)
        return True, ver
    except Exception:
        pass
    try:
        __import__(module)
        return True, ""
    except Exception:
        return False, ""


def _installed_runtime_version(meta: dict):
    """解释器：environment/python 里的那个（没有就退到当前进程的解释器）。"""
    probe = meta.get("probe")
    if probe:
        p = Path(ENV_ROOT) / probe
        if p.is_file() and os.access(str(p), os.X_OK):
            return _probe_version(str(p), meta.get("version_arg") or "-V")
    return _probe_version(sys.executable, "-V")


def _installed_toolchain_version(meta: dict):
    probe = meta.get("probe")
    if not probe:
        return False, ""
    # 2026-11-04：只认 environment/ 下的本地二进制，绝不去系统 PATH 找
    from paths import bundled_tool
    exe = bundled_tool(Path(probe).name)
    if exe:
        return _probe_version(exe, meta.get("version_arg") or "--version")
    return False, ""


def _installed_system_version(meta: dict):
    # 2026-11-04：不再查系统 PATH；只看 environment/ 里的本地二进制
    from paths import bundled_tool
    exe = bundled_tool(meta.get("cmd") or "")
    if not exe:
        return False, ""
    return _probe_version(exe, "-version")


def _probe_version(exe: str, arg: str):
    """跑一下可执行文件取版本号；取不到就只报「在」。"""
    try:
        r = subprocess.run([exe, arg], capture_output=True, text=True, timeout=15)
        out = (r.stdout or r.stderr or "").strip().splitlines()
        if out:
            line = out[0].strip()
            for tok in line.replace("(", " ").replace(")", " ").split():
                if tok[:1].isdigit() and any(c.isdigit() for c in tok):
                    return True, tok.rstrip(",;")
            return True, line[:40]
    except Exception:
        pass
    return True, ""


def dep_state(dep: dict) -> dict:
    """单个依赖的当前状态：{ok, version, satisfied, note}。"""
    meta = dep
    kind = meta.get("kind")
    if kind == "pip":
        ok, ver = _installed_pip_version(meta)
    elif kind == "runtime":
        ok, ver = _installed_runtime_version(meta)
    elif kind == "toolchain":
        ok, ver = _installed_toolchain_version(meta)
    elif kind == "wgpu":
        ok, ver = _installed_wgpu()
    else:
        ok, ver = _installed_system_version(meta)
    minv = meta.get("min") or ""
    satisfied = bool(ok) and (not minv or ver_ge(ver, minv) if ver else True)
    return {"ok": bool(ok), "version": ver, "satisfied": satisfied}


def _installed_wgpu():
    """wgpu 随 fast_shape_render 的 release 二进制静态链入；检测到该二进制
    即视为就位（否则需要跑 sh service/prims/setup.sh 构建 GPU 转换器）。"""
    override = os.environ.get("GARGANTUA_SHAPE_RENDERER")
    if override and os.path.isfile(override):
        return True, "（外部二进制）"
    import glob
    base = Path(ROOT) / "service" / "prims" / "fast_shape_render" / "target" / "release"
    if base.is_dir():
        bins = [p for p in glob.glob(str(base / "*"))
                if os.path.isfile(p) and os.access(p, os.X_OK)]
        if bins:
            return True, "（随转换器构建）"
    return False, ""


def version_ok(version: str, constraints) -> bool:
    """版本是否满足约束（用于筛选可安装的版本列表）。

    **只有上限类约束（< <= ==）会筛掉版本**；下限（>=）不筛——用户可以
    装更新的版本，这正是「向后兼容、可以选更快的编译产物」的意图。
    """
    for op, ref in constraints:
        if not version:
            continue
        if op == "==" and _vt(version) != _vt(ref):
            return False
        if op == "!=" and _vt(version) == _vt(ref):
            return False
        if op == "<" and not _vt(version) < _vt(ref):
            return False
        if op == "<=" and not _vt(version) <= _vt(ref):
            return False
        if op == ">" and not _vt(version) > _vt(ref):
            return False
        if op == ">=" and not _vt(version) >= _vt(ref):
            return False
    return True
