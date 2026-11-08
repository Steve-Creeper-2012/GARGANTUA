# ===========================================================================
# GARGANTUA v1 — server/env_manager.py：环境与依赖的「增删查」统一门面
# ---------------------------------------------------------------------------
# 2026-11-05（成员 C）：主页「环境与依赖」后端需要的 5 个能力：
#   list_deps / scan_binaries / available_versions / install / remove
#
# 铁律（与项目其余部分一致）：
#   * 只依赖项目内模块（reqs / server_env / paths / env_install /
#     server_monitor），且全部延迟到函数内 import —— import 本模块零副作用，
#     server 启动早期也能安全加载；
#   * 绝不 shutil.which、绝不读系统 PATH、绝不用系统解释器；
#   * 安装走 env_install.install_packages（唯一安装入口），删除走
#     start/setup_env.py --uninstall（同一隔离链），不另起旁路；
#   * 一切写项目内，不落 HOME，不新增第三方依赖。
# ===========================================================================
from __future__ import annotations

import threading

# 运行时底座（删了整个环境起不来）：list_deps 的 removable 与 remove 共用
_BASE_PACKS = {"python", "venv"}
# must 组 + 底座一律不可删（pywebview 是 must 组里的 soft 依赖）
_PROTECTED = _BASE_PACKS | {"pywebview"}

# install / remove 由 server 的 env 路由放到 TaskPool 后台线程异步调用。
# 两者都会写同一个项目内 venv，且 install_packages 生成的项目内
# constraints 文件名是固定的（optional_constraints.txt）——并发会互相
# 踩踏，pip 也不宜并发改同一 venv。用一把标准库锁把它们串行化，保证
# 线程安全；底层 pip install / uninstall / rmtree 对「已装 / 已删」本身
# 幂等，重复调用安全。
_ENV_LOCK = threading.Lock()


def _removable(name: str, group: str, kind: str):
    """返回 (removable, reason)：must 组 / 运行时底座一律 False。

    底座按 **name** 判断，绝不能按 pack 判断 —— 所有 pip 包的 pack 都是
    "venv"（装在 venv 里），若按 pack 判会把 torch/numpy 这些可选依赖
    一并误判成不可删。
    """
    if group == "must":
        return False, "must（启动必备）依赖，删除会导致 server 无法启动"
    if name in _BASE_PACKS:
        return False, "项目内运行时底座（python/venv），删除会导致整个环境不可用"
    if kind == "runtime":
        return False, "运行时解释器，属于环境底座，不可删除"
    return True, "可选依赖，可安全删除，需要时再单独安装"


def list_deps() -> list[dict]:
    """合并 requirements 清单状态 + environment/*/pack.json 分组。

    每项：{name, kind, required, installed, version, satisfied, pack,
           removable, group, reason}
      · required：是否启动必备（must 组为 True）；
      · installed / version / satisfied：来自 reqs.dep_state()；
      · pack：该依赖落在 environment/ 下的哪一处（remove 时定位用）；
      · group：must / necessary / pack（pack = 只在 environment 里有清单、
        没写进 requirements 的包）；
      · removable：must 组 / 运行时底座一律 False。
    """
    import os
    import reqs
    import server_env

    # environment/*/pack.json：name -> 包清单（分组 + pack 定位）。
    # 只认「有 pack.json」的目录；_cache / bin / certs / config / data /
    # tools / venv 这些运行目录不是可增删的依赖包，不能混进依赖列表。
    packs = {}
    for p in server_env.list_env_packs():
        d = os.path.join(server_env.ENV_ROOT, p["name"])
        if server_env.read_pack_manifest(d):
            packs[p["name"]] = p
    req_names = set()
    out = []

    for d in reqs.all_requirements():
        name = d["name"]
        req_names.add(name)
        st = reqs.dep_state(d)
        group = d.get("group") or "necessary"
        if group not in ("must", "necessary"):
            group = "necessary"
        kind = d.get("kind") or ""
        # pack 定位：META.pack 优先；其次 environment 下同名 pack；
        # ffmpeg/node 落在 environment/tools/<name>/；pip 包都在 venv 里。
        pack = d.get("pack") or None
        if not pack:
            if name in packs:
                pack = name
            elif name in ("ffmpeg", "node"):
                pack = "tools/" + name
            elif kind == "pip":
                pack = "venv"
        removable, reason = _removable(name, group, kind)
        out.append({
            "name": name,
            "kind": kind,
            "required": (group == "must"),
            "installed": bool(st["ok"]),
            "version": (st["version"] or None),
            "satisfied": bool(st["satisfied"]),
            "pack": pack,
            "removable": removable,
            "group": group,
            "reason": reason,
        })

    # 只在 environment/*/pack.json 里、不在 requirements 清单里的包 → group="pack"
    for pname, p in packs.items():
        if pname in req_names:
            continue
        kind = p.get("kind") or "pack"
        removable, reason = _removable(pname, "pack", kind)
        out.append({
            "name": pname,
            "kind": kind,
            "required": False,
            "installed": bool(p.get("present")),
            "version": (p.get("version") or None),
            "satisfied": bool(p.get("present")),
            "pack": pname,
            "removable": removable,
            "group": "pack",
            "reason": reason,
        })
    return out


def scan_binaries() -> list[dict]:
    """扫描 environment/ 内的本地二进制副本及其版本。

    来源：server_env.BUNDLED_TOOLS（ffmpeg/node/cargo）+ 各包 pack.json 的
    probe 字段（python/rust ...）。路径一律由 paths.bundled_tool / 项目内
    相对路径得到 —— 绝不 shutil.which、绝不读系统 PATH。
    每项：{name, path, version, required, satisfied}；取不到版本一律 None，
    绝不编造；satisfied = 找到本地副本且版本 >= 最低要求（无要求则找到即满足）。
    """
    import os
    import reqs
    import server_env
    import paths

    mins = {d["name"]: (d.get("min") or None)
            for d in reqs.all_requirements()}

    def required_for(name):
        # cargo 与 rust 属同一工具链，沿用 rust 的最低版本
        return mins.get(name) or (mins.get("rust") if name == "cargo" else None)

    # 候选 (name, path, version_arg)
    candidates = []

    # 1) BUNDLED_TOOLS：ffmpeg / node / cargo（bundled_tool 不回退 PATH）
    for t in server_env.BUNDLED_TOOLS:
        name = t["name"]
        exe = paths.bundled_tool(name) or ""
        candidates.append((name, exe, t.get("version_arg") or "--version"))

    # 2) environment/*/pack.json 的 probe 字段（python / rust ...）
    for p in server_env.list_env_packs():
        d = os.path.join(server_env.ENV_ROOT, p["name"])
        m = server_env.read_pack_manifest(d)
        probe_rel = m.get("probe") or ""
        if not probe_rel:
            continue
        candidates.append((p["name"], os.path.join(d, probe_rel),
                           m.get("version_arg") or "--version"))

    out = []
    seen = set()          # 同一二进制只报一次（cargo 会在两处出现）
    for name, path, varg in candidates:
        real = os.path.realpath(path) if path else ""
        if real and real in seen:
            continue
        if real:
            seen.add(real)
        found = bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)
        version = (server_env._probe_exe(path, varg) or None) if found else None
        req = required_for(name)
        if not found:
            satisfied = False
        elif not req:
            satisfied = True
        else:
            satisfied = bool(version) and reqs.ver_ge(version, req)
        out.append({
            "name": name,
            "path": path,
            "version": version,     # 取不到为 None，绝不编造
            "required": req,        # 最低要求版本或 None
            "satisfied": satisfied,
        })
    return out


def available_versions(name) -> dict:
    """某依赖在 PyPI 镜像上可选的版本（>= 最低要求、不违反硬约束）。

    复用 server_monitor._pypi_versions + reqs.version_ok。ffmpeg/rust/node
    这类本地工具/运行时不通过 PyPI 分发 → versions:[] 并带 reason。
    返回：{name, pkg, versions, default, min, constraints}（非 pip 时多一个 reason）。
    """
    import re
    import reqs
    import server_env
    import server_monitor

    key = (name or "").strip().lower()
    table = {d["name"]: d for d in reqs.all_requirements()}
    d = table.get(key)

    # node/cargo 这类没写进 requirements 的本地工具，也走「无 PyPI 版本」
    # 分支，而不是抛未知依赖
    local = {t["name"] for t in server_env.BUNDLED_TOOLS}
    local |= {p["name"] for p in server_env.list_env_packs()}
    if d is None:
        if key in local:
            return {"name": key, "pkg": key, "versions": [],
                    "default": "", "min": "", "constraints": [],
                    "reason": f"{key} 是项目内本地工具/环境包，"
                              f"不通过 PyPI 分发，无可在线选择的版本"}
        raise ValueError(f"未知依赖: {key!r}")

    pkg = d.get("dist") or key
    constraints = d.get("constraints") or []
    minv = d.get("min") or ""
    # 只有明确是「本地工具/运行时」的 kind 才没有 PyPI 版本；清单里未
    # 声明 kind 的（如 pytest）仍是 pip 包，应正常查可选版本。
    if (d.get("kind") or "") in ("runtime", "toolchain", "bundled",
                                 "wgpu", "system"):
        return {"name": key, "pkg": pkg, "versions": [],
                "default": minv, "min": minv, "constraints": constraints,
                "reason": f"{key} 是项目内本地工具/运行时"
                          f"（kind={d.get('kind')}），不通过 PyPI 分发，无可选版本"}

    raw = server_monitor._pypi_versions(pkg) or []
    raw = [v for v in raw if not re.search(r"[A-Za-z]", v)]   # 去预发布版
    # version_ok 已含 >= 约束判断；ver_ge 再按 min 兜底「>= 最低要求版本」
    ok = [v for v in raw
          if reqs.version_ok(v, constraints)
          and (not minv or reqs.ver_ge(v, minv))]
    ok.sort(key=server_monitor._ver_tuple, reverse=True)
    default = minv or (ok[-1] if ok else "")
    return {"name": key, "pkg": pkg, "versions": ok,
            "default": default, "min": minv, "constraints": constraints}


def install(name, version=None) -> dict:
    """安装/重装依赖：唯一入口是 env_install.install_packages（项目内 Python）。

    version 非 None 时把该依赖在清单里的硬约束（numpy<2 等）通过 py_deps
    一并传下去生成项目内 constraints 文件，避免可选安装破坏主依赖契约。
    绝不起第二条用系统解释器的路径。
    """
    import reqs
    import env_install

    key = (name or "").strip().lower()
    table = {d["name"]: d for d in reqs.all_requirements()}
    d = table.get(key)
    if d is None:
        raise ValueError(f"未知依赖: {key!r}")
    pkg = d.get("dist") or key
    spec = f"{pkg}=={version}" if version else pkg
    # version 非空才按 constraints 传下去（走 install_packages 的项目内
    # constraints 文件通道），与启动引导共用同一条安装链
    py_deps = [d] if (version and d.get("kind") == "pip") else None
    # 与 remove 串行化：见 _ENV_LOCK 说明（固定名 constraints 文件 + 共享 venv）
    with _ENV_LOCK:
        r = env_install.install_packages([spec], py_deps=py_deps)
    ok = (r.returncode == 0)
    return {
        "ok": ok,
        "name": key,
        "pkg": pkg,
        "version": version,
        "returncode": r.returncode,
        "stdout": r.stdout,
        "stderr": r.stderr,
        "note": ("已安装（下载/缓存均在项目内，重启 server 后生效）"
                 if ok else "安装失败：" + (r.stderr or r.stdout)[-400:]),
    }


def remove(name) -> dict:
    """删除可选依赖：统一调 start/setup_env.py --uninstall <name>。

    与安装共用项目内 Python（env_install.installer_python）+ isolated_env，
    must（启动必备）组与运行时底座直接抛 ValueError。
    """
    import subprocess
    import reqs
    import env_install
    import paths

    key = (name or "").strip().lower()
    table = {d["name"]: d for d in reqs.all_requirements()}
    d = table.get(key)
    if (d is not None and d.get("group") == "must") or key in _PROTECTED:
        raise ValueError(f"{key!r} 属于启动必备 / 运行时底座，"
                         f"删除会导致环境无法启动")

    cmd = [env_install.installer_python(),
           paths.project_dir("start", "setup_env.py"),
           "--uninstall", key]
    with _ENV_LOCK:
        r = subprocess.run(cmd, cwd=paths.PROJECT_ROOT,
                           env=paths.isolated_env(),
                           capture_output=True, text=True, timeout=3600)
    ok = (r.returncode == 0)
    return {
        "ok": ok,
        "name": key,
        "returncode": r.returncode,
        "stdout": r.stdout,
        "stderr": r.stderr,
        "note": ("已删除（项目内）" if ok else
                 "删除失败：" + (r.stderr or r.stdout)[-400:]),
    }
