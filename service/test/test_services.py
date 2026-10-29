# ===========================================================================
# GARGANTUA v1 — service/test/test_services.py：2026-10-24 服务化重构回归
# ---------------------------------------------------------------------------
# 覆盖 steve 本次重构的三件事：
#   1. 服务按需启动（server_services.ServiceSupervisor + JSON-lines 协议）
#      ——用 stub 服务脚本测，不起 torch、不建模型，秒级完成；
#        真实 infer/train serve 模式由 test_server.py 的 HTTP 集成测试覆盖。
#   2. 每模型词表 vocab.txt（common.read_model_vocab + Vocab.from_tokens）
#      ——ID 布局（不可见区保真）、守卫、编解码往返、贪心最长匹配、
#        add_token 持久化、进程级覆盖不污染全局词表。
# 运行：cd <ROOT> && python -m unittest service.test.test_services -v
# ===========================================================================
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

# ---- GARGANTUA 路径引导（目录树重构：server/ service/ start/）----
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

import spec
import server_services
import common
from common import read_model_vocab
from tokens.vocab import (Vocab, clear_model_vocab, get_vocab,
                          reset_vocab_cache, set_model_vocab)

# ---------------------------------------------------------------------------
# stub 服务脚本（测试 supervisor 协议用；stdout 只写 JSON 行）
# ---------------------------------------------------------------------------
_STUB_SERVICE = '''\
import json, os, sys
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    op = req.get("op")
    if op == "echo":
        resp = {"id": req["id"], "ok": True, "result": {"got": req.get("x")}}
    elif op == "boom":
        resp = {"id": req["id"], "ok": False, "error": "故意失败"}
    elif op == "die":
        os._exit(3)                      # 模拟崩溃：不应答直接死
    elif op == "shutdown":
        resp = {"id": req["id"], "ok": True, "result": {"bye": True}}
        print(json.dumps(resp), flush=True)
        sys.exit(0)
    else:
        resp = {"id": req["id"], "ok": False, "error": f"unknown op {op!r}"}
    print(json.dumps(resp), flush=True)
'''

_STUB_ONESHOT = '''\
import json, sys
print("human summary line（应被忽略，只解析末行）")
print(json.dumps({"ok": True, "result": {"n": 42, "argv_tail": sys.argv[-1]}}))
'''


class _SupervisorFixture(unittest.TestCase):
    """把 SERVICE_REGISTRY 临时指向 stub 脚本的公共夹具。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="garg_svc_")
        self.stub = os.path.join(self.tmp, "stub_service.py")
        with open(self.stub, "w", encoding="utf-8") as f:
            f.write(_STUB_SERVICE)
        self._orig_registry = dict(server_services.SERVICE_REGISTRY)
        server_services.SERVICE_REGISTRY["stub"] = self.stub

    def tearDown(self):
        server_services.SERVICE_REGISTRY.clear()
        server_services.SERVICE_REGISTRY.update(self._orig_registry)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _sup(self, idle_timeout=60):
        return server_services.ServiceSupervisor(boot.project_root(),
                                                 idle_timeout=idle_timeout)


class TestServiceSupervisor(_SupervisorFixture):
    """ JSON-lines 协议：配对 / 错误回传 / 未知 op / 崩溃重启 / 收尸。 """

    def test_call_roundtrip(self):
        sup = self._sup()
        try:
            res = sup.call("stub", "echo", x="hello", timeout=30)
            self.assertEqual(res, {"got": "hello"})
            self.assertTrue(sup.running("stub"))
        finally:
            sup.shutdown_all()
        self.assertFalse(sup.running("stub"))

    def test_error_response_raises_service_error(self):
        sup = self._sup()
        try:
            with self.assertRaises(server_services.ServiceError) as cm:
                sup.call("stub", "boom", timeout=30)
            self.assertIn("故意失败", str(cm.exception))
        finally:
            sup.shutdown_all()

    def test_unknown_op_rejected(self):
        sup = self._sup()
        try:
            with self.assertRaises(server_services.ServiceError):
                sup.call("stub", "no_such_op", timeout=30)
        finally:
            sup.shutdown_all()

    def test_crash_respawns_on_next_call(self):
        sup = self._sup()
        try:
            sup.call("stub", "echo", x=1, timeout=30)
            with self.assertRaises(server_services.ServiceError):
                sup.call("stub", "die", timeout=30)   # 服务当场死亡
            self.assertFalse(sup.running("stub"))
            # 下一次调用应自动重启并成功（请求不应永久失败）
            res = sup.call("stub", "echo", x=2, timeout=60)
            self.assertEqual(res, {"got": 2})
        finally:
            sup.shutdown_all()

    def test_query_does_not_spawn(self):
        # 状态查询不得为「问一句」而起进程——那是按需启动的底线
        sup = self._sup()
        self.assertIsNone(sup.query("stub", "status"))
        self.assertFalse(sup.running("stub"))

    def test_shutdown_all_kills_processes(self):
        sup = self._sup()
        sup.call("stub", "echo", x=1, timeout=30)
        handle = sup._handles["stub"]
        proc = handle.proc
        sup.shutdown_all()
        self.assertIsNotNone(proc.poll(), "shutdown_all 后进程必须退出")

    def test_run_oneshot_parses_last_json_line(self):
        sup = self._sup()
        script = os.path.join(self.tmp, "stub_oneshot.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(_STUB_ONESHOT)
        res = sup.run_oneshot(script, ["--whatever"], timeout=60)
        self.assertEqual(res["n"], 42)
        self.assertEqual(res["argv_tail"], "--json")   # 协议追加的参数

    def test_run_oneshot_missing_script(self):
        sup = self._sup()
        with self.assertRaises(server_services.ServiceError):
            sup.run_oneshot(os.path.join(self.tmp, "nope.py"), [], timeout=30)

    def test_unknown_service_name(self):
        sup = self._sup()
        with self.assertRaises(server_services.ServiceError):
            sup.call("no_such_service", "echo", timeout=5)

    def test_registry_has_real_services(self):
        # 注册表必须覆盖 infer/train（init 是一次性，不走常驻协议）
        self.assertIn("infer", server_services.SERVICE_REGISTRY)
        self.assertIn("train", server_services.SERVICE_REGISTRY)
        for name, rel in server_services.SERVICE_REGISTRY.items():
            self.assertTrue(
                os.path.isfile(os.path.join(boot.project_root(), rel)),
                f"服务 {name} 的脚本不存在: {rel}")


class TestModelVocabTxt(unittest.TestCase):
    """ 每模型词表 vocab.txt：布局 / 守卫 / 编解码 / 持久化 / 覆盖。 """

    def setUp(self):
        reset_vocab_cache()

    def tearDown(self):
        clear_model_vocab()
        reset_vocab_cache()

    # ---------------- 布局与不可见区 ----------------
    def test_flat_layout_preserves_invisible_region(self):
        v = Vocab.from_tokens(["hello", "world"])
        self.assertEqual(len(v), spec.VOCAB_SIZE_PADDED)
        base = spec.INVISIBLE_END
        self.assertEqual(v.token_to_id["hello"], base)
        self.assertEqual(v.token_to_id["world"], base + 1)
        # 不可见区固定 token 原样保真（<stop>/{ } | 的 ID 是架构契约）
        self.assertEqual(v.id_to_token[spec.STOP_ID], "<stop>")
        self.assertEqual(v.id_to_token[spec.TS_OPEN_ID], "{")
        self.assertEqual(v.id_to_token[spec.TS_CLOSE_ID], "}")
        self.assertEqual(v.id_to_token[spec.TS_SEP_ID], "|")
        # 分类：不可见区沿用全局布局，可见部分一律 regular
        self.assertEqual(v.category_of(spec.STOP_ID), "special")
        self.assertEqual(v.category_of(base), "regular")
        # 不可见 token 不参与文本编码查表
        self.assertIsNone(v.lookup("<user>"))
        self.assertEqual(v.lookup("hello"), base)

    def test_flat_vocab_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            Vocab.from_tokens(["a", "b", "a"])

    def test_flat_vocab_rejects_over_limit(self):
        limit = spec.VOCAB_SIZE_PADDED - spec.INVISIBLE_END
        with self.assertRaises(ValueError):
            Vocab.from_tokens(["x"] * (limit + 1))

    # ---------------- 编解码 ----------------
    def test_encode_decode_roundtrip_and_greedy_match(self):
        v = Vocab.from_tokens(["hello", " ", "新token", "你好"])
        from tokens import enctok
        from tokens.text_codec import decode
        import tokens.vocab as vmod
        orig = vmod._MODEL_VOCAB
        vmod._MODEL_VOCAB = v
        try:
            ids, skipped = enctok.encode_text("新token hello")
            self.assertEqual(skipped, [])
            self.assertEqual(ids, [v.token_to_id["新token"],
                                   v.token_to_id[" "],
                                   v.token_to_id["hello"]])
            # OOV 英文字母：扁平词表无 fallback 字母 → 跳过而不是崩
            ids2, skipped2 = enctok.encode_text("qqq")
            self.assertEqual(ids2, [])
            self.assertEqual(skipped2, ["q", "q", "q"])
            # 往返
            steps = [{"token_ids": [i], "type": 0, "modality": "text",
                      "meta": {}} for i in ids]
            self.assertEqual(decode(steps), "新token hello")
        finally:
            vmod._MODEL_VOCAB = orig

    def test_global_vocab_unaffected_by_flat(self):
        # 全局词表路径必须零变化（回归防线）
        from tokens import enctok
        ids, skipped = enctok.encode_text("Hello 世界")
        self.assertEqual(skipped, [])
        self.assertGreater(len(ids), 2)
        # 英文词段 + <caps> 标记（id 0）+ 中文单字
        self.assertEqual(ids[0], 0)
        self.assertIn(ids[1], get_vocab().regular_ids.values())

    # ---------------- 持久化与覆盖 ----------------
    def test_add_token_persists_to_vocab_txt(self):
        tmp = tempfile.mkdtemp(prefix="garg_vocab_")
        try:
            p = os.path.join(tmp, "vocab.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("alpha\nbeta\n")
            v = Vocab.from_tokens(["alpha", "beta"], p)
            slot = v.add_token("gamma", "regular")
            # 扁平词表优先吃 EXTRA 区（用户新增 token 的专用区）
            self.assertEqual(slot, spec.EXTRA_BEGIN)
            # 落盘是 v2 显式 ID 格式，且只含真实 token
            entries, legacy = common.read_flat_vocab_file(p)
            self.assertFalse(legacy)
            self.assertEqual(entries, [(spec.INVISIBLE_END, "alpha"),
                                       (spec.INVISIBLE_END + 1, "beta"),
                                       (slot, "gamma")])
            # 重新加载应看到新 token，且原 ID 不漂移
            v2 = Vocab.from_entries(common.read_model_vocab_entries(tmp)[0], p)
            self.assertEqual(v2.token_to_id["gamma"], slot)
            self.assertEqual(v2.token_to_id["alpha"], spec.INVISIBLE_END)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_read_model_vocab_contract(self):
        tmp = tempfile.mkdtemp(prefix="garg_vocab_")
        try:
            self.assertIsNone(read_model_vocab(tmp))      # 没有 → None
            p = os.path.join(tmp, "vocab.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("a\n\nb\r\nc\n")                    # 空行/CRLF
            self.assertEqual(read_model_vocab(tmp), ["a", "b", "c"])
            with open(p, "w", encoding="utf-8") as f:
                f.write("a\nb\na\n")
            with self.assertRaises(ValueError):
                read_model_vocab(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_set_model_vocab_switches_and_clears(self):
        tmp = tempfile.mkdtemp(prefix="garg_vocab_")
        try:
            p = os.path.join(tmp, "vocab.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write("onlytok\n")
            set_model_vocab(tmp)
            self.assertTrue(get_vocab().flat)
            self.assertEqual(get_vocab().lookup("onlytok"),
                             spec.INVISIBLE_END)
            # 没有 vocab.txt 的目录 → 清退全局
            tmp2 = tempfile.mkdtemp(prefix="garg_vocab2_")
            try:
                set_model_vocab(tmp2)
                self.assertFalse(getattr(get_vocab(), "flat", False))
            finally:
                shutil.rmtree(tmp2, ignore_errors=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
