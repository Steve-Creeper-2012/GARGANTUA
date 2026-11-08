# ===========================================================================
# GARGANTUA v1 — test_ui_sessions_20261105.py：推理界面大改回归
# ---------------------------------------------------------------------------
# 2026-11-05 这次改造固化成测试，覆盖三块：
#   1. 静态契约：前后端 DOM id 逐字对齐、状态条归宿主（插件不得再画一条）、
#      新端点已注册、快捷键动作齐全、id 不重复；
#   2. 会话持久化：SessionStore 的 CRUD / 原子写 / id 白名单 / 限额；
#   3. 后端能力：fw 热力图分辨率、模型加载与选中拆分的状态字段、
#      环境管理读接口、infer 服务新增 op。
# 秒级：不起 HTTP server、不加载 torch 权重、不建真模型。
# 运行：python -m unittest service.test.test_ui_sessions_20261105 -v
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

import inspect
import json
import os
import re
import shutil
import tempfile
import unittest

import spec

ROOT = boot.project_root()
WEB = os.path.join(ROOT, "service", "web")


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 1. 静态契约
# ---------------------------------------------------------------------------
# D 造 DOM、E 取 DOM、F 不再造 DOM——三方的唯一真源。少一个都会让界面
# 启动时静默失效（$("...") 返回 null 后 onclick 抛错），所以逐字钉死。
REQUIRED_HTML_IDS = [
    # 会话标签条 / 模型弹窗
    "conv-tabs", "conv-add", "conv-history", "infer-model-btn",
    "model-pop", "model-pop-refresh", "model-pop-list",
    # 宿主状态条（原属插件，2026-11-05 收归系统）
    "infer-statusbar", "infer-status", "infer-status-text",
    "im-ttft", "im-tps", "im-count", "im-ctx", "im-model",
    "plugin-host",
    # 开发者选项：KV 独立分类 + fast weight + max new
    "dev-kv-enable", "ctx-full-bar", "ctx-full-val", "ctx-comp-bar",
    "ctx-comp-val", "kv-mem-val", "kv-pct-bar", "kv-pct-val",
    "fw-canvas", "fw-norm", "fw-heads", "fw-write", "fw-write-val",
    "fw-clear", "infer-maxnew", "infer-temp", "infer-topp",
    "plugin-select", "plugin-import", "plugin-file", "plugin-imported",
    # 设置页四级分类
    "set-nav", "set-pane-appearance", "set-pane-hotkeys",
    "set-pane-history", "set-pane-env", "set-theme", "set-contrast",
    "set-radius", "set-radius-val", "hotkey-reset", "hotkey-msg",
    "hist-list", "hist-detail", "set-hist-refresh", "hist-open",
    "hist-delete", "hist-plugin", "hist-content",
    "set-env-refresh", "set-env-msg", "env-dep-list", "env-bin-list",
    # Token 编辑器导入 .tokens
    "json-import", "json-import-file", "json-editor", "json-export",
]
GOTO_ACTIONS = [
    "gotoHome", "gotoInfer", "gotoModels", "gotoTrain", "gotoText",
    "gotoAudio", "gotoImage", "gotoVideo", "gotoPrims", "gotoJson",
    "gotoPdraw", "gotoVocab", "gotoTerm", "gotoSettings",
]
NEW_GET_ROUTES = [
    "/api/sessions", "/api/session/get", "/api/env/deps",
    "/api/env/binaries", "/api/env/versions",
]
NEW_POST_ROUTES = [
    "/api/session/create", "/api/session/save", "/api/session/append",
    "/api/session/delete", "/api/infer/clear_fw", "/api/model/preload",
    "/api/model/unload", "/api/model/select", "/api/env/install",
    "/api/env/remove",
]


class TestStaticContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = _read("service/web/index_source.html")
        cls.css = _read("service/web/app.css")
        cls.appjs = _read("service/web/app.js")
        cls.plugin = _read("service/web/plugins/tokens.js")
        cls.serverpy = _read("server/server.py")

    def test_all_required_ids_present(self):
        present = set(re.findall(r'\bid="([^"]+)"', self.html))
        missing = [i for i in REQUIRED_HTML_IDS if i not in present]
        self.assertFalse(missing, f"index_source.html 缺 id: {missing}")

    def test_no_duplicate_ids(self):
        ids = re.findall(r'\bid="([^"]+)"', self.html)
        dup = sorted({i for i in ids if ids.count(i) > 1})
        self.assertFalse(dup, f"HTML 里有重复 id: {dup}")

    def test_clear_button_removed(self):
        self.assertNotIn('id="infer-clear"', self.html,
                         "顶栏清空按钮应已换成会话标签页")
        self.assertNotIn('$("infer-clear").onclick', self.appjs,
                         "app.js 里残留 infer-clear 绑定会在启动时抛错")

    def test_statusbar_owned_by_host(self):
        # 用户 2026-11-05 明确：状态/指标条归系统；默认插件同步删掉自己那条
        self.assertIn('id="infer-statusbar"', self.html)
        for stale in ("tk-head", "tk-metrics", "tk-title", "mTTFT", "mTPS",
                      "mCount", "mCtx", "mModel", "setMetrics"):
            self.assertNotIn(stale, self.plugin,
                             f"插件仍残留旧指标条代码: {stale}")
        for stale in (".tk-head", ".tk-metrics", ".tk-title"):
            self.assertNotIn(stale, _read("service/web/plugins/tokens.css"),
                             f"插件 CSS 仍残留旧指标条样式: {stale}")

    def test_statusbar_and_metrics_css(self):
        for cls in (".infer-statusbar", ".infer-status", ".infer-metrics",
                    ".dev-toggle", ".set-layout", ".hist-layout",
                    ".env-dep-list", ".conv-tab"):
            self.assertIn(cls, self.css, f"app.css 缺样式 {cls}")

    def test_hotkey_actions_present(self):
        for a in GOTO_ACTIONS:
            self.assertIn(f'data-hk="{a}"', self.html, f"缺快捷键动作 {a}")
            self.assertIn(f'id="hk-label-{a}"', self.html,
                          f"缺快捷键标签 {a}")

    def test_server_routes_registered(self):
        for r in NEW_GET_ROUTES + NEW_POST_ROUTES:
            self.assertIn(f'"{r}"', self.serverpy, f"server.py 未注册 {r}")

    def test_app_js_session_and_status_wiring(self):
        for needle in ("SESSIONS", "renderInferStatus", "updateInferMetrics",
                       "switchSetCat", "/api/model/preload",
                       "/api/model/select", "/api/tokens/decode"):
            self.assertIn(needle, self.appjs, f"app.js 缺 {needle}")

    def test_legacy_routes_untouched(self):
        # /api/model/load 仍供建模页使用，不能被拆分动作误删
        self.assertIn('"/api/model/load"', self.serverpy)
        self.assertIn('"/api/model/delete"', self.serverpy)


# ---------------------------------------------------------------------------
# 2. fast weight 可视化分辨率
# ---------------------------------------------------------------------------
class TestFastWeightResolution(unittest.TestCase):
    def test_bins_doubled(self):
        self.assertEqual(spec.FW_VIZ_BINS, 48,
                         "24²→48² 是本次「分辨率提升」的唯一真源")

    def test_snapshot_uses_bins_dynamically(self):
        src = _read("service/infer/infer.py")
        self.assertIn("adaptive_avg_pool2d", src)
        self.assertIn("fw_bins", src,
                      "_fw_snapshot 必须按参数取 bins，不能写死 24")


# ---------------------------------------------------------------------------
# 3. 会话持久化
# ---------------------------------------------------------------------------
class TestSessionStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import server_sessions as mod
            cls.mod = mod
        except ImportError as e:
            raise unittest.SkipTest(f"server_sessions 尚未就绪: {e}")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="garg_sessions_")
        # 支持构造参数 / 模块级目录常量两种实现，都不碰真实 logs/
        self.store = None
        try:
            self.store = self.mod.SessionStore(self.tmp)
        except TypeError:
            self.store = self.mod.SessionStore()
        for attr in ("DIR", "ROOT", "BASE_DIR", "_DIR"):
            if hasattr(self.mod, attr):
                self._patched = (attr, getattr(self.mod, attr))
                setattr(self.mod, attr, self.tmp)
                break
        for attr in ("dir", "root", "base_dir", "_dir"):
            if hasattr(self.store, attr):
                self._inst_patched = (attr, getattr(self.store, attr))
                setattr(self.store, attr, self.tmp)
                break

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_create_get_roundtrip(self):
        s = self.store.create({"title": "测试会话", "kv_enabled": False})
        self.assertTrue(s["id"].startswith("s_"))
        got = self.store.get(s["id"])
        self.assertEqual(got["title"], "测试会话")
        self.assertFalse(got["kv_enabled"])
        self.assertEqual(got["messages"], [])
        # 默认值必须齐全，前端按这些字段回填开发者选项
        for k in ("plugin", "model", "max_new", "temperature", "top_p"):
            self.assertIn(k, got)

    def test_save_append_delete_and_list(self):
        s = self.store.create({})
        self.store.save(s["id"], {"title": "改名", "max_new": 64})
        self.assertEqual(self.store.get(s["id"])["max_new"], 64)
        self.store.append(s["id"], "me", [0xC732, 0x1F])
        self.store.append(s["id"], "bot", [0x20])
        got = self.store.get(s["id"])
        self.assertEqual([m["role"] for m in got["messages"]], ["me", "bot"])
        self.assertEqual(got["messages"][0]["token_ids"], [0xC732, 0x1F])
        # 摘要不含 messages，避免 /api/sessions 把整段历史全量下发
        summary = self.store.list()
        self.assertTrue(summary)
        self.assertNotIn("messages", summary[0])
        self.assertTrue(self.store.delete(s["id"]))
        self.assertIsNone(self.store.get(s["id"]))

    def test_path_traversal_rejected(self):
        for bad in ("../evil", "s_../../x", "/etc/passwd", "abc"):
            with self.assertRaises((ValueError, KeyError, FileNotFoundError)):
                self.store.get(bad)

    def test_id_format(self):
        ids = {self.store.create({})["id"] for _ in range(5)}
        self.assertEqual(len(ids), 5)
        for i in ids:
            self.assertRegex(i, r"^s_[0-9a-f]{12}$")


# ---------------------------------------------------------------------------
# 4. 后端能力：模型拆分 / 环境管理 / infer 新 op
# ---------------------------------------------------------------------------
class TestBackendCapabilities(unittest.TestCase):
    def test_model_manager_has_loaded_flag(self):
        src = _read("server/server_models.py")
        self.assertIn("loaded", src,
                      "ModelManager 需要 loaded 状态才能区分「加载」与「选中」")
        self.assertIn('"loaded"', _read("server/server.py"))

    def test_infer_session_ops(self):
        src = _read("service/infer/infer.py")
        for op in ("clear_fw", "drop_session", "unload"):
            self.assertIn(f'"{op}"', src, f"infer.py 缺 op {op}")
        for key in ("session_id", "kv_enabled", "sessions"):
            self.assertIn(key, src, f"infer.py 缺 {key}")
        sig = str(inspect.signature(self._generate()))
        for p in ("session", "kv_enabled", "on_phase"):
            self.assertIn(p, sig, f"generate() 缺参数 {p}")

    @staticmethod
    def _generate():
        import importlib.util
        p = os.path.join(ROOT, "service", "infer", "infer.py")
        s = importlib.util.spec_from_file_location("_garg_infer_probe", p)
        m = importlib.util.module_from_spec(s)
        # 只取签名：不执行模块顶层可能触发的重导入
        src = open(p, encoding="utf-8").read()
        m.__dict__["__name__"] = "_garg_infer_probe"
        s.loader.exec_module(m)
        return m.generate

    def test_phase_frames_emitted(self):
        src = _read("service/infer/infer.py")
        for phase in ("loading", "encoding", "decoding"):
            self.assertIn(f'"{phase}"', src, f"缺阶段帧 {phase}")
        srv = _read("server/server.py")
        self.assertIn('"phase"', srv, "流缓冲/响应未透传 phase")

    def test_env_manager_api(self):
        p = os.path.join(ROOT, "server", "env_manager.py")
        if not os.path.isfile(p):
            raise unittest.SkipTest("env_manager.py 尚未就绪")
        src = open(p, encoding="utf-8").read()
        for fn in ("list_deps", "scan_binaries", "available_versions",
                   "install", "remove"):
            self.assertIn(f"def {fn}(", src, f"env_manager 缺 {fn}")
        # 铁律：绝不查系统 PATH。用 AST 找【真实调用节点】，不能做字符串
        # 匹配——注释和 docstring 里「绝不 shutil.which」这类说明文字会误报。
        import ast
        tree = ast.parse(src)
        which_calls = [n for n in ast.walk(tree)
                       if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute)
                       and n.func.attr == "which"]
        self.assertFalse(which_calls,
                         "env_manager 里出现了 shutil.which 调用（违反零系统依赖）")
        path_reads = [n for n in ast.walk(tree)
                      if isinstance(n, ast.Subscript)
                      and isinstance(n.value, ast.Attribute)
                      and n.value.attr == "environ"]
        self.assertFalse(path_reads,
                         "env_manager 里出现了读 os.environ 的下标访问")

    def test_setup_env_uninstall(self):
        src = _read("start/setup_env.py")
        self.assertIn("--uninstall", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
