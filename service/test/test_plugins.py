# ===========================================================================
# GARGANTUA v1 — service/test/test_plugins.py：推理显示插件 + 实时统计回归
# ---------------------------------------------------------------------------
# 2026-11-04 界面整改的代码审查修复，固化成测试：
#   1. 插件导入 / 列表 / 移除：id 白名单、路径穿越、内置文件保护、
#      同名冲突、元信息解析、损坏注册表容错、原子写；
#   2. _finite：NaN/Inf 递归消毒（否则 _send_json 会输出非法 JSON）；
#   3. api_infer_stats：生成期从流缓冲合成 fw / fw_writes / kv_steps，
#      不再穿透到持锁的 infer 服务；running / streaming 字段。
# 只起临时目录，不起 server、不加载 torch 权重，秒级可跑。
# 运行：cd ROOT && python service/test/test_plugins.py -v
# ===========================================================================
from __future__ import annotations

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
import shutil
import tempfile
import threading
import types
import unittest

import server as srv
import spec

CODE = "// @name 测试插件\n// @desc 一行说明\nexport default { id: 'x', mount(){} };\n"


class _AppStub:
    """只提供被测 API 方法需要的 self 属性，避免构造真 App 起线程池。"""
    _bool_arg = staticmethod(srv.App._bool_arg)


class _SupStub:
    def __init__(self, running=True, query_result=None):
        self._running = running
        self._query = query_result
        self.queries = 0

    def running(self, name):
        return self._running

    def query(self, name, op, timeout=15.0, **args):
        self.queries += 1
        return self._query


class _MgrStub:
    name = None
    model_dir = ""
    ctx = None


class TestPluginApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="garg_plugin_")
        self.old = (srv._PLUGIN_DIR, srv._PLUGIN_REGISTRY)
        srv._PLUGIN_DIR = self.tmp
        srv._PLUGIN_REGISTRY = os.path.join(self.tmp, "_imported.json")
        self.app = _AppStub()

    def tearDown(self):
        srv._PLUGIN_DIR, srv._PLUGIN_REGISTRY = self.old
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_import_list_remove_roundtrip(self):
        r = srv.App.api_plugin_import(
            self.app, {"id": "echo", "code": CODE, "css": ".x{color:red}"})
        p = r["plugin"]
        self.assertEqual(p["id"], "echo")
        self.assertEqual(p["name"], "测试插件")
        self.assertEqual(p["desc"], "一行说明")
        self.assertEqual(p["entry"], "plugins/echo.js")
        self.assertEqual(p["css"], "plugins/echo.css")
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "echo.js")))
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "echo.css")))
        self.assertEqual(srv.App.api_plugin_list(self.app)["plugins"], [p])

        # 去掉 CSS 的 force 覆盖不能留下幽灵样式
        srv.App.api_plugin_import(
            self.app, {"id": "echo", "code": CODE, "force": True})
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "echo.css")))
        self.assertEqual(
            srv.App.api_plugin_list(self.app)["plugins"][0]["css"], "")

        srv.App.api_plugin_remove(self.app, {"id": "echo"})
        self.assertEqual(srv.App.api_plugin_list(self.app)["plugins"], [])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "echo.js")))

    def test_same_id_requires_force(self):
        srv.App.api_plugin_import(self.app, {"id": "dup", "code": CODE})
        with self.assertRaises(FileExistsError):
            srv.App.api_plugin_import(self.app, {"id": "dup", "code": CODE})

    def test_bad_ids_are_rejected_and_builtins_untouched(self):
        sentinel = os.path.join(self.tmp, "tokens.js")
        with open(sentinel, "w", encoding="utf-8") as f:
            f.write("BUILTIN")
        for pid in ("../evil", "Tokens", "_x", "tokens", "base", "a/b",
                    "a\\b", "x" * 33, "", 123):
            with self.subTest(pid=pid):
                with self.assertRaises(ValueError):
                    srv.App.api_plugin_import(
                        self.app, {"id": pid, "code": CODE})
        with open(sentinel, encoding="utf-8") as f:
            self.assertEqual(f.read(), "BUILTIN")

    def test_bad_code_is_rejected(self):
        for code in ("", "no default export", None, 42, "x" * (513 * 1024)):
            with self.subTest(code=repr(code)[:30]):
                with self.assertRaises(ValueError):
                    srv.App.api_plugin_import(
                        self.app, {"id": "ok", "code": code})

    def test_remove_rejects_builtin_and_unknown(self):
        for pid in ("tokens", "base", "never-imported"):
            with self.assertRaises(ValueError):
                srv.App.api_plugin_remove(self.app, {"id": pid})

    def test_corrupt_registry_is_tolerated(self):
        with open(srv._PLUGIN_REGISTRY, "w", encoding="utf-8") as f:
            f.write("{ broken")
        self.assertEqual(srv.App.api_plugin_list(self.app)["plugins"], [])

    def test_finite_sanitizer_makes_valid_json(self):
        dirty = {"a": float("nan"), "b": [float("inf"), {"c": -float("inf")}],
                 "ok": 1.25}
        with self.assertRaises(ValueError):
            json.dumps(dirty, allow_nan=False)
        out = json.dumps(srv._finite(dirty), allow_nan=False)
        self.assertEqual(json.loads(out),
                         {"a": 0, "b": [0, {"c": 0}], "ok": 1.25})

    def test_meta_parser(self):
        name, desc = srv._plugin_meta_from_code(CODE, "fallback")
        self.assertEqual((name, desc), ("测试插件", "一行说明"))
        self.assertEqual(srv._plugin_meta_from_code("export default {};", "zz"),
                         ("zz", ""))


class TestInferStatsRealtime(unittest.TestCase):
    """生成期 stats 必须读流缓冲，而不是持锁的 infer 服务。"""

    def _app_with_buffer(self, *, done=False, sup=None):
        app = _AppStub()
        app._streams_lock = threading.Lock()
        app._streams = {
            "s1": {"t0": 1.0, "done": done, "fw": {"norm": 2.0,
                                                   "matrix": [[1.0]]},
                   "writes": [0.25, 0.5], "kv_steps": 12,
                   "tokens": [{"id": 1}, {"id": 2}], "ttft_ms": 500},
        }
        app.mgr = _MgrStub()
        app.supervisor = sup or _SupStub()
        return app

    def test_streaming_buffer_is_authoritative(self):
        sup = _SupStub(query_result={"fw": {"norm": 999}, "fw_writes": [9]})
        app = self._app_with_buffer(sup=sup)
        st = srv.App.api_infer_stats(app)
        self.assertTrue(st["streaming"])
        self.assertTrue(st["running"])
        self.assertEqual(st["fw"]["norm"], 2.0)      # 不是服务里的旧 999
        self.assertEqual(st["fw_writes"], [0.25, 0.5])
        self.assertEqual(st["kv_steps"], 12)
        self.assertEqual(st["n_generated"], 2)
        self.assertEqual(st["ttft_ms"], 500)
        self.assertEqual(sup.queries, 0)             # 绝不穿透到持锁服务

    def test_done_stream_reports_not_streaming(self):
        st = srv.App.api_infer_stats(self._app_with_buffer(done=True))
        self.assertFalse(st["streaming"])

    def test_no_buffer_falls_back_to_service_query(self):
        sup = _SupStub(query_result={"kv_steps": 7, "fw": {"norm": 3}})
        app = self._app_with_buffer(sup=sup)
        app._streams = {}
        st = srv.App.api_infer_stats(app)
        self.assertEqual(sup.queries, 1)
        self.assertEqual(st["kv_steps"], 7)
        self.assertEqual(st["fw"]["norm"], 3)
        self.assertIn("fw_bins", st)
        self.assertEqual(st["fw_bins"], spec.FW_VIZ_BINS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
