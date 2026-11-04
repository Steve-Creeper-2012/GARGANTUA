# ===========================================================================
# GARGANTUA v1 — service/test/test_portability.py：可移植性 / 隔离静态门
# ---------------------------------------------------------------------------
# 2026-11-04：项目文件夹要能整体搬到任意路径 / 另一台机器后双击运行，
# 所有依赖、缓存、临时文件、写盘目标都必须在项目目录内。本套件不建模型、
# 不联网、不写盘（只读源码），纯静态扫描即可把「又有人写死本机绝对路径」
# 或「tempfile 忘了 dir=」这类回归挡在门外。
#
# 覆盖：
#   1. paths.py 的 ROOT 推导与两个项目内目录；
#   2. 运行时源码里禁止出现特定机器的绝对路径 / 系统临时目录；
#   3. 运行时所有 tempfile.* 调用必须显式 dir= 到项目内；
#   4. 启动脚本的下载缓存不得落在系统临时目录。
# 运行：cd ROOT && python service/test/test_portability.py -v
# ===========================================================================
from __future__ import annotations

# ---- GARGANTUA 路径引导 ----
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "start" / "boot.py").is_file():
        _sys.path.insert(0, str(_anc / "start"))
        break
else:
    raise SystemExit("找不到 GARGANTUA 项目根（start/boot.py）")
import boot
boot.bootstrap()

import os
import re
import unittest
from pathlib import Path

import paths

ROOT = Path(paths.PROJECT_ROOT)
# 只扫运行时 / 构建期源码；测试、vendored 第三方、本地依赖产物不扫
SCAN_DIRS = ("server", "service/init", "service/infer", "service/train",
             "service/tokens", "service/prims", "service/web",
             "service/window", "start", "launcher")
SKIP_PARTS = ("__pycache__", "service/test", "output", "fast_shape_render",
              "upstream", "environment", ".git")
EXTS = (".py", ".sh", ".bat", ".command")

# 这些字符串一旦出现在运行时源码里，就意味着换一台机器会成为死引用，
# 或者把依赖 / 写盘目标指到项目文件夹之外
FORBIDDEN = (
    re.compile(r"/Users/"),
    re.compile(r"C:\\Users\\", re.I),
    re.compile(r"%TEMP%", re.I),
    re.compile(r"~/Library/Application Support", re.I),
    re.compile(r"~/\.workbuddy", re.I),
    re.compile(r"TRAE SOLO", re.I),
    re.compile(r"/usr/local/bin/node"),
    re.compile(r"/opt/homebrew/bin/node"),
)
TEMPFILE_CALL = re.compile(
    r"tempfile\.(mkdtemp|mkstemp|NamedTemporaryFile|TemporaryDirectory)\s*\(")


def _runtime_files():
    seen = set()
    for rel in SCAN_DIRS:
        base = ROOT / rel
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix not in EXTS or not p.is_file():
                continue
            s = p.as_posix()
            if any(part in s for part in SKIP_PARTS):
                continue
            if p.resolve() not in seen:
                seen.add(p.resolve())
                yield p
    for p in sorted(ROOT.glob("*")):
        if p.is_file() and p.suffix in EXTS and p.resolve() not in seen:
            yield p


class TestProjectPaths(unittest.TestCase):
    def test_root_is_project_dir(self):
        self.assertTrue((ROOT / "start" / "boot.py").is_file())
        self.assertEqual(paths.PROJECT_ROOT,
                         os.path.dirname(os.path.dirname(
                             os.path.abspath(paths.__file__))))

    def test_tmp_and_cache_stay_inside_project(self):
        tmp = Path(paths.project_tmp_dir())
        cache = Path(paths.project_cache_dir("pip"))
        for p in (tmp, cache):
            self.assertTrue(p.is_dir())
            self.assertEqual(p.resolve().relative_to(ROOT).parts[0],
                             ("data" if p == tmp else "environment"))


class TestNoExternalReferences(unittest.TestCase):
    def test_no_hardcoded_machine_paths(self):
        bad = []
        for p in _runtime_files():
            text = p.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                for pat in FORBIDDEN:
                    if pat.search(line):
                        bad.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()[:100]}")
        self.assertEqual(bad, [], "发现指向项目外的绝对路径 / 用户目录引用:\n"
                                + "\n".join(bad))

    def test_tempfile_calls_pin_dir(self):
        bad = []
        for p in _runtime_files():
            if p.suffix != ".py":
                continue
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            for i, line in enumerate(lines):
                if not TEMPFILE_CALL.search(line):
                    continue
                window = "\n".join(lines[i:i + 4])
                if "dir=" not in window:
                    bad.append(f"{p.relative_to(ROOT)}:{i + 1}: {line.strip()[:100]}")
        self.assertEqual(bad, [], "tempfile 必须用 dir=project_tmp_dir() "
                                "落到项目内:\n" + "\n".join(bad))

    def test_launcher_is_bootstrap_only(self):
        """启动链必须只引导下载项目内 Python，其余交给 setup_env.py。"""
        sh = (ROOT / "launcher" / "setup-environment.sh").read_text(
            encoding="utf-8")
        self.assertIn('CACHE="$ROOT/environment/_cache/downloads"', sh)
        self.assertIn('exec "$PY" "$SETUP_PY" --stage', sh)
        self.assertIn('cp "$ca" "$CERTS/cacert.pem"', sh)
        bat = (ROOT / "start" / "terminal" / "setup.bat").read_text(
            encoding="utf-8", errors="replace")
        self.assertIn('"%PY%" "%SETUP_PY%" --stage', bat)

    def test_no_system_package_managers_or_system_python(self):
        """setup 脚本的可执行行里不许出现系统包管理器 / 系统解释器。"""
        bad = []
        pats = re.compile(
            r"brew\s+install|apt-get|dnf\s+install|apk\s+add|winget|"
            r"py\s+-3\.12|python3\.12|/usr/bin/python|/usr/local/bin/python",
            re.I)
        for rel in ("launcher/setup-environment.sh",
                    "start/terminal/setup.bat"):
            lines = (ROOT / rel).read_text(
                encoding="utf-8", errors="replace").splitlines()
            for i, line in enumerate(lines, 1):
                s = line.strip()
                if s.startswith(("#", "REM", "::", "*")) or not s:
                    continue          # 注释里说明「不用这些」不算违规
                if pats.search(s):
                    bad.append(f"{rel}:{i}: {s[:100]}")
        self.assertEqual(bad, [], "setup 仍在回退到系统运行时:\n"
                                + "\n".join(bad))

    def test_runtime_never_looks_into_system_path(self):
        """运行时代码不得自己 shutil.which / 读 SHELL。"""
        bad = []
        for p in _runtime_files():
            if p.suffix != ".py" or p.name == "paths.py":
                continue
            for i, line in enumerate(
                    p.read_text(encoding="utf-8",
                                errors="replace").splitlines(), 1):
                if "shutil.which(" in line or 'os.environ.get("SHELL")' in line:
                    bad.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()[:90]}")
        self.assertEqual(bad, [], "运行时仍会查系统 PATH:\n"
                                + "\n".join(bad))


class TestStrictIsolation(unittest.TestCase):
    def test_isolated_env_paths_all_inside_project(self):
        env = paths.isolated_env()
        root = os.path.realpath(str(ROOT))
        for key in ("TMPDIR", "TMP", "TEMP", "PIP_CACHE_DIR", "CARGO_HOME",
                    "RUSTUP_HOME", "GOPATH", "GOCACHE", "GOMODCACHE",
                    "NPM_CONFIG_CACHE", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
                    "XDG_DATA_HOME", "PYTHONUSERBASE", "GARGANTUA_ROOT"):
            val = os.path.realpath(env[key])
            self.assertEqual(os.path.commonpath([val, root]), root,
                             f"{key} 指到了项目外: {val}")
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertEqual(env["GARGANTUA_ISOLATED"], "1")
        self.assertTrue(env["PATH"].split(os.pathsep)[0].startswith(root))

    def test_venv_belongs_to_bundled_python(self):
        """装好的 venv 必须由项目内 standalone Python 构建。"""
        if not os.path.isfile(paths.venv_python()):
            self.skipTest("环境尚未安装（双击启动会自动搭建）")
        ok, why = paths.venv_is_isolated()
        self.assertTrue(ok, f"venv 仍绑系统 Python: {why}")
        self.assertTrue(os.path.isfile(paths.bundled_python()),
                        "缺少项目内 standalone Python")

    def test_installer_is_shared_by_cli_and_window(self):
        """窗口内手动安装/更新与启动引导必须共用同一个安装器。"""
        mon = (ROOT / "server" / "server_monitor.py").read_text(
            encoding="utf-8")
        self.assertIn("from env_install import install_packages", mon)
        self.assertNotIn("sys.executable, \"-m\", \"pip\", \"install\"", mon)
        inst = (ROOT / "server" / "env_install.py").read_text(
            encoding="utf-8")
        self.assertIn("setup_env.py", inst)
        self.assertIn("paths.isolated_env()", inst)


if __name__ == "__main__":
    unittest.main()
