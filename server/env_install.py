# ===========================================================================
# GARGANTUA v1 — env_install.py：窗口内手动「安装 / 更新」的统一入口
# ---------------------------------------------------------------------------
# 2026-11-04（steve）：用户在打开窗口后手动点「环境包」安装或更新，必须和
# 启动引导走**同一套**隔离 + 镜像测速逻辑，不能另起一条用系统解释器、
# 系统 PATH、用户缓存的旁路。
#
# 这一层很薄：所有实际工作都交给 start/setup_env.py（由项目内 Python 执行），
# 它负责 venv 归属校验、pip 镜像测速、项目内缓存与临时目录、隔离环境变量。
# ===========================================================================
from __future__ import annotations

import os
import subprocess

import paths

ROOT = paths.PROJECT_ROOT
SETUP_PY = paths.project_dir("start", "setup_env.py")


def installer_python() -> str:
    """只用项目内解释器：优先隔离 venv，其次 standalone 底座。"""
    vp = paths.venv_python()
    if os.path.isfile(vp):
        return vp
    bp = paths.bundled_python()
    if os.path.isfile(bp):
        return bp
    raise RuntimeError(
        "项目内 Python 不存在，请先运行 launcher/setup-environment.sh"
        "（Windows: start\\terminal\\setup.bat）")


def install_packages(pkgs, upgrade: bool = False, py_deps=None,
                     timeout: int = 3600) -> subprocess.CompletedProcess:
    """安装/更新 pip 包；返回 CompletedProcess 交给调用方判断。"""
    if not pkgs:
        raise ValueError("缺少要安装的包")
    cmd = [installer_python(), SETUP_PY, "--install", *list(pkgs)]
    if upgrade:
        cmd.append("--upgrade")
    if py_deps:
        # 约束文件也写在项目内（data/tmp），不落到系统临时目录
        from server_util import constraints_file_for
        cf = os.path.join(paths.project_tmp_dir(),
                          "optional_constraints.txt")
        constraints_file_for([d for d in py_deps
                              if d.get("kind") == "pip"], cf)
        cmd += ["--constraints", cf]
    return subprocess.run(cmd, cwd=ROOT, env=paths.isolated_env(),
                          capture_output=True, text=True, timeout=timeout)
