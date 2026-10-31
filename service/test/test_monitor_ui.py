# ===========================================================================
# GARGANTUA — service/test/test_monitor_ui.py：主页「进程监视器 / 设备」后端回归
# ---------------------------------------------------------------------------
# BUG1：_is_gargantua_args 进程识别
#   * 相对路径（service/init/init.py）、绝对路径、系统 python、其它 GARGANTUA
#     副本、隔离环境 venv exe（bin / Scripts）都要命中；
#   * 无关进程（zsh -c / Electron / python -c / grep 'server.py' / python -m）
#     必须排除；
#   * Windows 无 ps：PowerShell/wmic 分支，cpu 列宁可少报返回 None。
# BUG2：_cpu_topology 物理核 vs 逻辑线程（4C8T），api_system_devices 的
#   cpu 字段契约（保留 cores 键向后兼容）。
# 运行：cd <ROOT> && environment/venv/bin/python3 service/test/test_monitor_ui.py -v
# ===========================================================================
# ---- GARGANTUA 路径引导（2026-10-24 目录树重构：server/ service/ start/）----
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

import json
import os
import signal
import sys
import time
import unittest
from unittest import mock

import server_monitor as mon


ROOT = mon.ROOT


class TestIsGargantuaArgs(unittest.TestCase):
    """BUG1 纯函数：命中 / 排除，name 为命中的 .py basename。"""

    # ---- 命中 ----
    def test_relative_script_hits(self):
        # 服务子进程由 server.py 用相对路径 os.path.join("service","init",
        # "init.py") 拉起：旧版 startswith(ROOT) 判不到，必须命中。
        mine, name = mon._is_gargantua_args(
            "python3 service/init/init.py --port 9800", "python3", ROOT)
        self.assertTrue(mine)
        self.assertEqual(name, "init.py")

    def test_absolute_script_hits(self):
        args = "/usr/bin/python3 %s" % os.path.join(ROOT, "server", "server.py")
        mine, name = mon._is_gargantua_args(args, "/usr/bin/python3", ROOT)
        self.assertTrue(mine)
        self.assertEqual(name, "server.py")

    def test_system_python_relative_server_hits(self):
        # 系统 python 直接 `python server/server.py`：exe 不在 venv 也要收
        mine, name = mon._is_gargantua_args(
            "python3 server/server.py", "python3", ROOT)
        self.assertTrue(mine)
        self.assertEqual(name, "server.py")

    def test_other_gargantua_copy_hits(self):
        # 从别的 GARGANTUA 副本目录启动：ps 路径不含当前 ROOT
        args = "/usr/bin/python3 /Users/steve/backup/GARGANTUA/server/window.py"
        mine, name = mon._is_gargantua_args(args, "/usr/bin/python3", ROOT)
        self.assertTrue(mine)
        self.assertEqual(name, "window.py")

    def test_other_gargantua_copy_codec_prefix_hits(self):
        # codec_*.py 用前缀通配，也要能覆盖副本
        args = ("/usr/bin/python3 "
                "/opt/GARGANTUA-old/service/codec/codec_image.py")
        mine, name = mon._is_gargantua_args(args, "/usr/bin/python3", ROOT)
        self.assertTrue(mine)
        self.assertEqual(name, "codec_image.py")

    def test_venv_exe_absolute_hits(self):
        # -m 内联、无 .py 脚本：靠 venv exe 命中（bin）
        exe = os.path.join(ROOT, "environment", "venv", "bin", "python3")
        mine, _ = mon._is_gargantua_args(exe + " -m http.server 8000",
                                         exe, ROOT)
        self.assertTrue(mine)

    def test_venv_exe_relative_hits(self):
        # launcher 用的相对 exe `environment/venv/bin/python3`
        args = "environment/venv/bin/python3 -m foo"
        mine, _ = mon._is_gargantua_args(
            args, "environment/venv/bin/python3", ROOT)
        self.assertTrue(mine)

    def test_venv_scripts_exe_windows_hits(self):
        # Windows 隔离环境在 venv/Scripts 下
        exe = os.path.join(ROOT, "environment", "venv", "Scripts", "python.exe")
        mine, _ = mon._is_gargantua_args(exe + " -m foo", exe, ROOT)
        self.assertTrue(mine)

    # ---- 排除 ----
    def test_zsh_shell_command_not_hits(self):
        # zsh -c 整段命令是一个参数，不以 .py 结尾 → 无脚本 → 不命中
        args = ('/bin/zsh -c "cd /Users/steve/Desktop/GARGANTUA '
                '&& python3 server/server.py && echo done"')
        mine, name = mon._is_gargantua_args(args, "/bin/zsh", ROOT)
        self.assertFalse(mine)
        self.assertIsNone(name)

    def test_electron_not_hits(self):
        args = ("/Applications/Electron.app/Contents/MacOS/Electron "
                "/Users/steve/Desktop/GARGANTUA/project")
        mine, _ = mon._is_gargantua_args(args, args.split()[0], ROOT)
        self.assertFalse(mine)

    def test_python_c_inline_not_hits(self):
        mine, _ = mon._is_gargantua_args(
            'python3 -c "print(1)"', "python3", ROOT)
        self.assertFalse(mine)
        # -c 内联里即使提到某个 .py，也不是独立脚本参数 → 不命中
        mine, _ = mon._is_gargantua_args(
            'python3 -c "open(\'server.py\').read()"', "python3", ROOT)
        self.assertFalse(mine)

    def test_grep_mention_not_hits(self):
        # grep / pgrep 搜索串里的 'server.py' 是 bare basename：无子目录
        # （分支 A 挡）、无 GARGANTUA 段（分支 B 挡）→ 不命中
        mine, _ = mon._is_gargantua_args(
            "grep 'server.py' /var/log/app.log", "grep", ROOT)
        self.assertFalse(mine)
        mine, _ = mon._is_gargantua_args("pgrep -f server.py", "pgrep", ROOT)
        self.assertFalse(mine)

    def test_python_m_no_script_not_hits(self):
        # -m 内联、没有 .py 脚本、exe 不在 venv
        mine, _ = mon._is_gargantua_args(
            "python3 -m http.server", "/usr/bin/python3", ROOT)
        self.assertFalse(mine)

    def test_empty_args_not_hits(self):
        mine, name = mon._is_gargantua_args("", "python3", ROOT)
        self.assertFalse(mine)
        self.assertIsNone(name)

    def test_windows_paths_not_mangled(self):
        # Windows 用 posix=False：反斜杠必须保留，不能把
        # C:\...\server.py 吃成 C:...server.py
        with mock.patch.object(mon.platform, "system", return_value="Windows"):
            scripts = mon._extract_scripts(
                r'"C:\Python\python.exe" C:\App\GARGANTUA\server\server.py')
        self.assertIn(r"C:\App\GARGANTUA\server\server.py", scripts)


class TestCpuTopology(unittest.TestCase):
    """BUG2：物理核 / 逻辑线程 / 超线程；任一读不到返回 None，绝不硬编码。"""

    def _mock_darwin(self, physical, logical):
        # macOS 分支：_run(["sysctl","-n",KEY]) 按 KEY 返回伪造输出
        table = {"hw.physicalcpu": physical, "hw.logicalcpu": logical,
                 "hw.ncpu": logical}

        def fake(cmd, timeout=10):
            v = table.get(cmd[-1] if cmd else "")
            return (v + "\n") if v else ""

        return mock.patch.object(mon, "_run", side_effect=fake)

    def test_4c8t_hyperthreaded(self):
        with self._mock_darwin("4", "8"):
            topo = mon._cpu_topology()
        self.assertEqual(topo["cores"], 4)
        self.assertEqual(topo["logical_cores"], 8)
        self.assertTrue(topo["hyperthreaded"])

    def test_physical_equals_logical_no_ht(self):
        with self._mock_darwin("4", "4"):
            topo = mon._cpu_topology()
        self.assertEqual(topo["cores"], 4)
        self.assertEqual(topo["logical_cores"], 4)
        self.assertFalse(topo["hyperthreaded"])

    def test_unreadable_returns_none(self):
        with self._mock_darwin(None, None):
            topo = mon._cpu_topology()
        self.assertIsNone(topo["cores"])
        self.assertIsNone(topo["logical_cores"])
        self.assertFalse(topo["hyperthreaded"])

    def test_logical_falls_back_to_ncpu(self):
        # hw.logicalcpu 读不到 → 回退 hw.ncpu
        table = {"hw.physicalcpu": "4", "hw.logicalcpu": None, "hw.ncpu": "8"}

        def fake(cmd, timeout=10):
            v = table.get(cmd[-1] if cmd else "")
            return (v + "\n") if v else ""

        with mock.patch.object(mon, "_run", side_effect=fake):
            topo = mon._cpu_topology()
        self.assertEqual(topo["cores"], 4)
        self.assertEqual(topo["logical_cores"], 8)
        self.assertTrue(topo["hyperthreaded"])

    def test_cpu_cores_returns_physical(self):
        with self._mock_darwin("4", "8"):
            self.assertEqual(mon._cpu_cores(), 4)

    def test_cpu_cores_falls_back_to_logical(self):
        # 物理核读不到 → 回退逻辑线程数（宁可回退，不可硬编码）
        table = {"hw.physicalcpu": None, "hw.logicalcpu": "8", "hw.ncpu": "8"}

        def fake(cmd, timeout=10):
            v = table.get(cmd[-1] if cmd else "")
            return (v + "\n") if v else ""

        with mock.patch.object(mon, "_run", side_effect=fake):
            self.assertEqual(mon._cpu_cores(), 8)

    def test_windows_cim_json(self):
        out = '[{"NumberOfCores":4,"NumberOfLogicalProcessors":8}]'
        with mock.patch.object(mon.platform, "system", return_value="Windows"), \
             mock.patch.object(mon, "_run", return_value=out):
            topo = mon._cpu_topology()
        self.assertEqual(topo["cores"], 4)
        self.assertEqual(topo["logical_cores"], 8)
        self.assertTrue(topo["hyperthreaded"])

    def test_windows_cim_single_object(self):
        out = '{"NumberOfCores":8,"NumberOfLogicalProcessors":8}'
        with mock.patch.object(mon.platform, "system", return_value="Windows"), \
             mock.patch.object(mon, "_run", return_value=out):
            topo = mon._cpu_topology()
        self.assertEqual(topo["cores"], 8)
        self.assertFalse(topo["hyperthreaded"])

    def test_linux_proc_cpuinfo(self):
        proc = "cpu cores : 4\nsiblings : 8\n"
        with mock.patch.object(mon.platform, "system", return_value="Linux"), \
             mock.patch("builtins.open", mock.mock_open(read_data=proc)):
            topo = mon._cpu_topology()
        self.assertEqual(topo["cores"], 4)
        self.assertEqual(topo["logical_cores"], 8)
        self.assertTrue(topo["hyperthreaded"])

    def test_linux_lscpu_fallback(self):
        # /proc/cpuinfo 读不到 → 回退 lscpu：Core(s) per socket × Socket(s)
        lscpu = ("Architecture: x86_64\n"
                 "CPU(s): 8\n"
                 "Core(s) per socket: 4\n"
                 "Socket(s): 1\n")
        with mock.patch.object(mon.platform, "system", return_value="Linux"), \
             mock.patch("builtins.open", mock.mock_open(read_data="")), \
             mock.patch.object(mon, "_run", return_value=lscpu):
            topo = mon._cpu_topology()
        self.assertEqual(topo["cores"], 4)
        self.assertEqual(topo["logical_cores"], 8)
        self.assertTrue(topo["hyperthreaded"])


class TestProcPorts(unittest.TestCase):
    """_process_ports：lsof / netstat 输出解析——端口去重升序、状态括号剥离。"""

    LSOF = (
        "COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME\n"
        "python3  1234 steve   12u  IPv4  0xabc      0t0  TCP 127.0.0.1:8765 (LISTEN)\n"
        "python3  1234 steve   15u  IPv6  0xabc      0t0  TCP *:9000 (LISTEN)\n"
        "python3  1234 steve   16u  IPv4  0xabc      0t0  TCP 127.0.0.1:8765 (ESTABLISHED)\n"
        "python3  1234 steve   18u  IPv4  0xabc      0t0  UDP *:5353\n"
    )

    def test_lsof_parse_dedup_sorted(self):
        # 同一端口 LISTEN+ESTABLISHED 去重；结果升序；UDP 也收
        with mock.patch.object(mon, "_run", return_value=self.LSOF):
            r = mon._process_ports([1234])
        self.assertEqual(r, {1234: [5353, 8765, 9000]})

    def test_lsof_pid_filter(self):
        # 不在请求列表里的 pid 不收
        out = self.LSOF + "other     999 steve   20u  IPv4  0xabc      0t0  TCP 1.2.3.4:80\n"
        with mock.patch.object(mon, "_run", return_value=out):
            r = mon._process_ports([1234])
        self.assertEqual(r, {1234: [5353, 8765, 9000]})

    def test_no_output_empty(self):
        with mock.patch.object(mon, "_run", return_value=""):
            self.assertEqual(mon._process_ports([1]), {})
        self.assertEqual(mon._process_ports([]), {})

    def test_windows_netstat_parse(self):
        out = ("\n  TCP    127.0.0.1:8765     0.0.0.0:0    LISTENING    777\n"
               "  UDP    0.0.0.0:5353        *:*                       999\n")
        with mock.patch.object(mon.platform, "system", return_value="Windows"), \
             mock.patch.object(mon, "_run", return_value=out):
            r = mon._process_ports([777])
        self.assertEqual(r, {777: [8765]})


class TestGpuRuntime(unittest.TestCase):
    """_gpu_runtime：ioreg 无符号负值回绕丢弃 + gartUsedBytes 兜底。

    实测 Intel 核显驱动会把负的在用内存按无符号打印（-152322560 印成
    18446744073594331136 ≈ 16384PB），不处理就会在前端显示 16384.00PB。"""

    def setUp(self):
        mon._GPU_RT.update({"ts": 0.0, "data": None})

    def _run(self, out):
        return mock.patch.object(mon, "_run", return_value=out)

    # 真机样本：Accum 块无利用率键会被跳过；真实块 inUse 为无符号负值
    IOREG_INTEL = (
        '  "PerformanceStatisticsAccum" = {"inUseSysMemoryBytes"'
        '=18446744073594331136,"inUseVidMemoryBytes"=0,'
        '"gartUsedBytes"=103919616}\n'
        '  "PerformanceStatistics" = {"Device Utilization %"=7,'
        '"inUseSysMemoryBytes"=18446744073594331136,'
        '"inUseVidMemoryBytes"=0,"gartUsedBytes"=103919616}\n'
    )

    def test_negative_inuse_falls_back_to_gart(self):
        # 16384PB 事故样本：inUse 无效 → 回退 gartUsedBytes=103919616（≈99MB）
        with self._run(self.IOREG_INTEL):
            r = mon._gpu_runtime()
        self.assertEqual(r, [{"usage": 7.0, "used": 103919616}])

    def test_valid_inuse_used_directly(self):
        # Apple Silicon / 正常驱动：inUse 有效值直接采用，gart 不参与
        out = ('  "PerformanceStatistics" = {"Device Utilization %"=12,'
               '"inUseSysMemoryBytes"=268435456,"inUseVidMemoryBytes"=1048576,'
               '"gartUsedBytes"=1}\n')
        with self._run(out):
            r = mon._gpu_runtime()
        self.assertEqual(r, [{"usage": 12.0, "used": 269484032}])

    def test_all_invalid_returns_none(self):
        # inUse 无效且无 gart：宁可少报，used=None（前端显示 —）
        out = ('  "PerformanceStatistics" = {"Device Utilization %"=0,'
               '"inUseSysMemoryBytes"=18446744073709551615,'
               '"inUseVidMemoryBytes"=0}\n')
        with self._run(out):
            r = mon._gpu_runtime()
        self.assertEqual(r, [{"usage": 0.0, "used": None}])


class TestGargantuaProcesses(unittest.TestCase):
    """BUG1 端到端：伪造 ps 输出 → 只剩项目进程，name / 排序 / 类型正确。"""

    def test_end_to_end_ps(self):
        ps = (
            "  101  12.5  102400 /usr/bin/python3 "
            + os.path.join(ROOT, "server", "server.py") + " --port 9800\n"
            "  102   3.0   51200 /usr/bin/python3 "
            + os.path.join(ROOT, "service", "window", "window.py") + "\n"
            "  103   0.0    4096 /bin/zsh -c "
            "\"cd /Users/x/GARGANTUA && python3 server/server.py\"\n"
        )
        with mock.patch.object(mon, "_run", return_value=ps), \
             mock.patch.object(mon, "_mem_stat",
                               return_value=(16 * 1024 ** 3, 8 * 1024 ** 3)):
            procs = mon._gargantua_processes()
        # 只剩 server.py / window.py，zsh 被排除
        self.assertEqual(len(procs), 2)
        # 按 CPU 降序：12.5 的 server.py 在前
        self.assertEqual(procs[0]["name"], "server.py")
        self.assertEqual(procs[1]["name"], "window.py")
        self.assertEqual(procs[0]["pid"], 101)
        self.assertEqual(procs[0]["cpu"], 12.5)
        self.assertEqual(procs[0]["mem_bytes"], 102400 * 1024)
        # 字段类型契约
        for p in procs:
            self.assertIsInstance(p["pid"], int)
            self.assertIsInstance(p["name"], str)
            self.assertIsInstance(p["cpu"], float)
            self.assertIsInstance(p["mem_bytes"], int)
            self.assertTrue(p["mem_pct"] is None
                            or isinstance(p["mem_pct"], float))
            self.assertIsNone(p["gpu_pct"])
            self.assertIsNone(p["vram_bytes"])
        self.assertIsInstance(procs[0]["mem_pct"], float)
        # _run 被 mock 成 ps 文本，lsof 解析不出端口 → 空列表契约
        self.assertEqual(procs[0]["ports"], [])

    def test_windows_branch_no_ps(self):
        # Windows 无 ps：改用 PowerShell JSON；cpu 列必须为 None（宁可少报）
        cmdline = (os.path.join(ROOT, "environment", "venv", "Scripts",
                                "python.exe")
                   + " " + os.path.join(ROOT, "server", "server.py"))
        data = json.dumps([
            {"ProcessId": 201, "CommandLine": cmdline,
             "WorkingSetSize": 64 * 1024 * 1024},
        ])
        with mock.patch.object(mon.platform, "system", return_value="Windows"), \
             mock.patch.object(mon, "_run", return_value=data), \
             mock.patch.object(mon, "_mem_stat", return_value=(0, None)):
            procs = mon._gargantua_processes()
        self.assertEqual(len(procs), 1)
        self.assertEqual(procs[0]["name"], "server.py")
        self.assertIsNone(procs[0]["cpu"])
        self.assertEqual(procs[0]["mem_bytes"], 64 * 1024 * 1024)
        self.assertIsNone(procs[0]["mem_pct"])

    def test_other_copy_marked_external(self):
        # 其它副本进程：宽松列表显示并标 external=True；严格列表不含它
        other = "/Users/x/GARGANTUA-v1.1.2/server/server.py"
        ps = ("  301  1.0  4096 /usr/bin/python3 " + other + "\n"
              "  302  1.0  4096 /usr/bin/python3 "
              + os.path.join(ROOT, "server", "server.py") + "\n")
        with mock.patch.object(mon, "_run", return_value=ps), \
             mock.patch.object(mon, "_mem_stat", return_value=(0, None)):
            loose = mon._gargantua_processes(include_other=True)
            strict = mon._gargantua_processes(include_other=False)
        self.assertEqual({p["pid"] for p in loose}, {301, 302})
        ext = {p["pid"]: p["external"] for p in loose}
        self.assertTrue(ext[301])
        self.assertFalse(ext[302])
        self.assertEqual({p["pid"] for p in strict}, {302})

    def test_kill_rejects_external_pid(self):
        # 越权防线：外部副本 pid 虽在宽松列表里，kill 白名单必须拒绝
        other = "/Users/x/GARGANTUA-v1.1.2/server/server.py"
        ps = "  301  1.0  4096 /usr/bin/python3 " + other + "\n"
        with mock.patch.object(mon, "_run", return_value=ps), \
             mock.patch.object(mon, "_mem_stat", return_value=(0, None)):
            with self.assertRaises(ValueError):
                mon.MonitorApi().api_system_kill({"pid": 301})


class TestApiSystemDevicesContract(unittest.TestCase):
    """BUG2：api_system_devices 的 cpu 字段契约（保留 cores 键向后兼容）。"""

    def _api(self, topo, usage, mem_total):
        with mock.patch.object(mon, "_gpu_info", return_value=([], None)), \
             mock.patch.object(mon, "_gpu_runtime", return_value=[]), \
             mock.patch.object(mon, "_mem_stat",
                               return_value=(mem_total, mem_total // 2)), \
             mock.patch.object(mon, "_cpu_model", return_value="Test CPU"), \
             mock.patch.object(mon, "_cpu_topology", return_value=topo), \
             mock.patch.object(mon, "_cpu_usage", return_value=usage), \
             mock.patch.object(mon.platform, "machine", return_value="x86_64"):
            return mon.MonitorApi().api_system_devices()

    def test_cpu_field_contract(self):
        topo = {"cores": 4, "logical_cores": 8, "hyperthreaded": True}
        code, obj = self._api(topo, 25.0, 8 * 1024 ** 3)
        self.assertEqual(code, 200)
        cpu = obj["cpu"]
        self.assertEqual(cpu["model"], "Test CPU")
        self.assertEqual(cpu["cores"], 4)            # 向后兼容键 = 物理核
        self.assertEqual(cpu["logical_cores"], 8)
        self.assertEqual(cpu["physical_cores"], 4)
        self.assertTrue(cpu["hyperthreaded"])
        self.assertEqual(cpu["usage"], 25.0)
        # 契约其它字段不回退
        self.assertIn("memory", obj)
        self.assertIn("vram", obj)
        self.assertIn("gpus", obj)

    def test_cpu_field_falls_back_when_physical_unreadable(self):
        # 物理核读不到：cores 回退逻辑数，physical_cores 为 None
        topo = {"cores": None, "logical_cores": 8, "hyperthreaded": False}
        code, obj = self._api(topo, None, 0)
        self.assertEqual(code, 200)
        cpu = obj["cpu"]
        self.assertEqual(cpu["cores"], 8)
        self.assertIsNone(cpu["physical_cores"])
        self.assertEqual(cpu["logical_cores"], 8)
        self.assertFalse(cpu["hyperthreaded"])


class _FakeResponse:
    """_pypi_versions 的 urlopen 替身（支持 with 与 read(n) 限长）。"""
    def __init__(self, body):
        self._body = body.encode("utf-8")
    def read(self, n=-1):
        return self._body[:n] if n >= 0 else self._body
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


class TestAuditHardening(unittest.TestCase):
    """2026-10-29 审计加固：M1 自身进程 / M4 版本缓存 / M5 TLS 兜底。"""

    def test_kill_allows_server_itself(self):
        # 诚实：server 自身也是合法的 GARGANTUA 进程，允许用户结束
        # （前端会弹二次确认）。后端只做目录边界校验，不单独拦截 self。
        # 用 fake_kill 替掉真正的 os.kill，避免测试进程自杀（SIGTERM）。
        me = os.getpid()
        ps = ("  %d  1.0  4096 /usr/bin/python3 %s\n"
              % (me, os.path.join(ROOT, "server", "server.py")))
        killed = []
        def fake_kill(pid, sig):
            killed.append((pid, sig))
        with mock.patch.object(mon, "_run", return_value=ps), \
             mock.patch.object(mon, "_mem_stat", return_value=(0, None)), \
             mock.patch.object(mon.os, "kill", side_effect=fake_kill):
            procs = mon._gargantua_processes(include_other=False)
            self.assertTrue(any(p["pid"] == me and p["self"] for p in procs))
            code, obj = mon.MonitorApi().api_system_kill({"pid": me})
            self.assertEqual(code, 200)
            self.assertEqual(obj["killed"], me)
            self.assertEqual(killed, [(me, signal.SIGTERM)])

    def test_force_clears_versions_cache(self):
        mon._VERSIONS_CACHE["torch"] = {"ts": time.time(), "data": "ERR"}
        mon._VERSIONS_CACHE["numpy"] = {"ts": time.time(),
                                        "data": ["1.26.4"]}
        mon.clear_versions_cache()
        self.assertEqual(mon._VERSIONS_CACHE, {})

    def test_versions_cache_err_expires(self):
        # 过期的 ERR 不再被当成永久失败；TTL 到期后允许重新请求
        mon._VERSIONS_CACHE["torch"] = {
            "ts": time.time() - mon.VERSIONS_ERR_TTL - 1, "data": "ERR"}
        html = '<a href="/torch-2.2.2-cp312.whl">torch-2.2.2</a>'
        with mock.patch.object(mon, "urlopen",
                               return_value=_FakeResponse(html)), \
             mock.patch.object(mon, "_ssl_context", return_value=object()):
            got = mon._pypi_versions("torch")
        self.assertIsNotNone(got)
        self.assertIn("2.2.2", got)

    def test_ssl_fallback_requires_explicit_opt_in(self):
        # certifi 不可用 + 无系统 CA + 未显式允许 → 不再静默不校验证书
        env = {k: v for k, v in os.environ.items()
               if k != "GARGANTUA_INSECURE_PYPI"}
        with mock.patch.dict(sys.modules, {"certifi": None}), \
             mock.patch.object(mon.os.path, "exists", return_value=False), \
             mock.patch.dict(mon.os.environ, env, clear=True):
            with self.assertRaises(RuntimeError):
                mon._ssl_context()


if __name__ == "__main__":
    unittest.main()
