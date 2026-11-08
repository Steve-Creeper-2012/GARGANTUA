# ===========================================================================
# GARGANTUA v1 — agent 槽位协议后端回归（1.3.0，不起 HTTP server / 不跑模型）
# ---------------------------------------------------------------------------
# 1.3.0 把 53 个预定义工具全部退役：agent 区改成「数量由词表 config 决定」
# 的编号槽位（base 零预留），工具执行先全部不接。本套件测：契约头布局、
# /api/agent/spec 与 tools 载荷、token 元数据、槽位调用解析、结果回灌、
# 执行一律拒绝、HTTP 端点与流式浅拷贝。
# 词表用内存 stub 注入，避免依赖磁盘上具体词表长度。
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

import base64
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import agent_tools
import spec
import tokens.vocab as vocab_mod
from server_agent import AgentApi

_VISIBLE = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789 .,!?;:+-*/=_\n\t")


class _StubVocab:
    """最小可用词表：架构头占位 + 可见区单字符。"""

    def __init__(self):
        head = spec.HEAD_SIZE
        self.id_to_token = ["<h%d>" % i for i in range(head)] + list(_VISIBLE)
        self.regular_ids = {c: head + i for i, c in enumerate(_VISIBLE)}
        self.fallback_ids = {c: self.regular_ids[c]
                             for c in "abcdefghijklmnopqrstuvwxyz"}
        self.special_ids = {}
        self.system_ids = {}
        self.token_to_id = dict(self.regular_ids)
        self._max_match = 1
        self._invisible_above_head = set()
        self.name = "stub1.3.0"
        self.head_size = head
        self.slot_count = spec.SLOT_COUNT
        self.draw_begin = spec.DRAW_BEGIN

    def __len__(self):
        return len(self.id_to_token)

    def is_invisible(self, idx):
        return idx < self.head_size

    def name_of(self, idx):
        t = self.id_to_token[idx]
        return t[1:-1] if t.startswith("<") and t.endswith(">") else t


def _install_stub(tc) -> _StubVocab:
    v = _StubVocab()
    tc.addCleanup(setattr, vocab_mod, "_VOCAB", vocab_mod._VOCAB)
    vocab_mod._VOCAB = v
    return v


class TestSpecContract(unittest.TestCase):
    """契约头五段，前 12 个 ID 固定，其余随 config 的 slot_count 平移。"""

    def test_fixed_prefix(self):
        self.assertEqual(spec.CORE_TOKENS,
                         ["<caps>", "<stop>", "<draw:end>"])
        self.assertEqual(spec.BOUNDARY_SIZE, 6)
        self.assertEqual(spec.SYSTEM_SIZE, 3)
        self.assertEqual(spec.SLOT_BEGIN, 12)
        # 1.3.0：slot 与 draw 之间插入 228 个 typst 结构 token
        self.assertEqual(spec.DRAW_BEGIN,
                         spec.SLOT_BEGIN + spec.SLOT_COUNT + spec.TYPST_SIZE)
        self.assertEqual(spec.HEAD_SIZE,
                         9845 + spec.SLOT_COUNT)

    def test_slot_count_comes_from_config(self):
        cfg = json.loads(
            (Path(boot.project_root()) / "service/tokens/config.json")
            .read_text("utf-8"))
        self.assertEqual(cfg["slot_count"], spec.SLOT_COUNT)
        self.assertEqual(spec.BASE_SLOT_COUNT, 0)
        self.assertEqual(spec.SLOT_COUNT, 0)   # base 零预留
        # 1.3.0：draw_begin = 240+N、head_size = 9845+N
        self.assertEqual(spec.head_layout(128)["draw_begin"], 368)
        self.assertEqual(spec.head_layout(128)["head_size"], 9973)

    def test_slot_ids_and_names(self):
        self.assertEqual(spec.AGENT_END, spec.AGENT_BEGIN + spec.AGENT_SIZE)
        self.assertEqual(spec.AGENT_SIZE, spec.SLOT_COUNT)
        for name, tid in spec.AGENT_ID.items():
            self.assertTrue(name.startswith("<s"))
            self.assertEqual(spec.segment_type_of(tid), spec.TYPE_SYS)
        self.assertEqual(len(spec.AGENT_NAME_BY_ID), spec.SLOT_COUNT)


class TestAgentSpecPayload(unittest.TestCase):
    def test_fields(self):
        p = agent_tools.agent_spec_payload()
        self.assertEqual(p["vocab_name"], spec.BASE_VOCAB_NAME)
        self.assertEqual(p["vocab_version"], "1.3.0")
        self.assertEqual(p["head_size"], spec.HEAD_SIZE)
        self.assertEqual(p["draw_begin"], spec.DRAW_BEGIN)
        self.assertEqual(p["slot_begin"], spec.SLOT_BEGIN)
        self.assertEqual(p["slot_count"], spec.SLOT_COUNT)
        self.assertEqual(p["stop_id"], spec.STOP_ID)
        self.assertEqual(p["draw_end_id"], spec.DRAW_END_ID)
        self.assertEqual(p["special"]["caps"], spec.CAPS_ID)
        self.assertEqual([c["name"] for c in p["core"]], spec.CORE_TOKENS)
        self.assertEqual(len(p["slots"]), spec.SLOT_COUNT)
        self.assertEqual(p["tools"], [])
        self.assertEqual(p["formats"], [])
        self.assertEqual(p["boundaries"], [])
        for s in p["slots"]:
            self.assertFalse(s["executable"])
            self.assertEqual(s["id"], spec.SLOT_BEGIN + s["slot"])
            self.assertEqual(s["name"], f"<s{s['slot']}>")

    def test_no_vocab_does_not_raise(self):
        with mock.patch.object(agent_tools, "_safe_vocab", return_value=None):
            p = agent_tools.agent_spec_payload(draw_layout={})
            self.assertEqual(p["head_size"], spec.HEAD_SIZE)
            self.assertEqual(p["vocab_size"], 0)

    def test_tools_payload(self):
        t = agent_tools.tools_payload()
        self.assertEqual(t["tools"], [])
        self.assertEqual(t["executable"], [])
        self.assertEqual(t["slot_count"], spec.SLOT_COUNT)
        self.assertEqual(len(t["slots"]), spec.SLOT_COUNT)
        self.assertEqual(len(t["core"]), 3)


class TestTokenMeta(unittest.TestCase):
    def test_classification(self):
        v = _install_stub(self)
        m = agent_tools.token_meta(spec.CAPS_ID, v)
        self.assertEqual(m["char"], "<caps>")
        self.assertEqual(m["core"]["name"], "<caps>")
        self.assertEqual(agent_tools.token_meta(spec.USER_ID, v)["char"],
                         "[user]")
        self.assertEqual(agent_tools.token_meta(spec.STOP_ID, v)["char"], "")
        self.assertEqual(agent_tools.token_meta(spec.HEAD_SIZE, v)["char"],
                         "a")
        # 1.3.0：base 零预留时 SLOT_BEGIN(=12) 落在 typst 段，返回结构名
        self.assertEqual(agent_tools.token_meta(spec.SLOT_BEGIN, v)["char"],
                         "<DOC>")            # 该 ID 属 typst 区（<DOC>）

    def test_out_of_range_and_no_vocab(self):
        self.assertEqual(agent_tools.token_meta(10 ** 9, None),
                         {"char": "", "tool": None, "slot": None, "core": None})
        self.assertEqual(agent_tools.token_meta(-1, None)["char"], "")
        self.assertEqual(agent_tools.token_meta(1.5, None)["char"], "")
        self.assertEqual(agent_tools.token_meta(spec.CAPS_ID, None)["char"],
                         "<caps>")

    def test_slot_meta_when_configured(self):
        """有槽位的布局下，槽位走兼容的 tool 徽标字段。"""
        tid = spec.SLOT_BEGIN + 1
        with mock.patch.object(spec, "SLOT_TOKENS", ("<s0>", "<s1>")), \
             mock.patch.object(spec, "SLOT_COUNT", 2), \
             mock.patch.object(spec, "SLOT_END", spec.SLOT_BEGIN + 2), \
             mock.patch.object(spec, "SLOT_ID",
                               {"<s0>": spec.SLOT_BEGIN, "<s1>": tid}), \
             mock.patch.object(spec, "SLOT_DEFS",
                               {1: {"name": "bash", "kind": "action",
                                    "close": 0, "executable": False}}):
            m = agent_tools.token_meta(tid, None)
            self.assertEqual(m["char"], "bash")
            self.assertEqual(m["slot"]["slot"], 1)
            self.assertEqual(m["tool"]["category"], "action")


class TestDetectToolCalls(unittest.TestCase):
    def test_default_empty(self):
        self.assertEqual(agent_tools.detect_tool_calls(
            [spec.SLOT_BEGIN, spec.HEAD_SIZE]), [])
        self.assertEqual(agent_tools.detect_tool_calls("not-a-list"), [])

    def test_custom_slot_pair(self):
        """使用者在 SLOT_DEFS 定义 action/close 后才会解析出调用。"""
        o, c = spec.SLOT_BEGIN + 5, spec.SLOT_BEGIN + 6
        args = [spec.HEAD_SIZE + 1, spec.HEAD_SIZE + 2]
        defs = {5: {"kind": "action", "close": 6, "name": "bash"},
                6: {"kind": "close"}}
        with mock.patch.object(spec, "SLOT_DEFS", defs), \
             mock.patch.object(spec, "SLOT_COUNT", 8), \
             mock.patch.object(spec, "SLOT_TOKENS",
                               spec.slot_tokens(8)):
            calls = agent_tools.detect_tool_calls([o] + args + [c])
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["slot"], 5)
            self.assertEqual(calls[0]["tool_id"], o)
            self.assertEqual(calls[0]["start"], 0)
            self.assertEqual(calls[0]["end"], 3)
            self.assertIsInstance(calls[0]["args"], str)
            # 未闭合不返回
            self.assertEqual(agent_tools.detect_tool_calls([o] + args), [])


class TestRunToolAndResult(unittest.TestCase):
    def test_run_tool_always_refused(self):
        for name in ("<s0>", "<bash>", "", None):
            with self.assertRaises(ValueError):
                agent_tools.run_tool(name, "echo hi")

    def test_result_ids_structure(self):
        v = _install_stub(self)
        ids = agent_tools.tool_result_ids("<s0>", "ok")
        self.assertEqual(ids[0], spec.SYS_ID)
        self.assertEqual(ids[-1], spec.SYS_END_ID)
        self.assertEqual(len(ids), 2 + 2)   # "ok" = 2 个可见 token
        # 不再有 <tool_result> 嵌套包裹
        self.assertNotIn(spec.DRAW_END_ID, ids[1:-1])

    def test_result_illegal(self):
        with self.assertRaises(ValueError):
            agent_tools.tool_result_ids("<s0>", 123)


class TestAgentApi(unittest.TestCase):
    """HTTP handler 层：spec/tools/encode/decode/file 端点。"""

    def setUp(self):
        self.api = AgentApi()
        _install_stub(self)

    def test_spec_and_tools_endpoints(self):
        s = self.api.api_agent_spec({})
        self.assertEqual(s["vocab_name"], spec.BASE_VOCAB_NAME)
        self.assertEqual(s["slot_count"], spec.SLOT_COUNT)
        t = self.api.api_agent_tools({})
        self.assertEqual(t["tools"], [])

    def test_encode_decode_roundtrip(self):
        e = self.api.api_agent_encode({"text": "hello"})
        ids = e["token_ids"]
        self.assertIsInstance(ids, list) and self.assertTrue(ids)
        d = self.api.api_agent_decode({"ids": ids})
        self.assertEqual(d["text"], "hello")

    def test_tool_endpoint_refused(self):
        with self.assertRaises(ValueError):
            self.api.api_agent_tool({"name": "<s0>", "args": "x"})

    def test_tool_result_endpoint(self):
        r = self.api.api_agent_tool_result({"name": "<s0>", "result": "ok"})
        self.assertEqual(r["token_ids"][0], spec.SYS_ID)
        self.assertEqual(r["token_ids"][-1], spec.SYS_END_ID)

    def test_file_endpoint(self):
        raw = b"hello agent"
        r = self.api.api_agent_file({"filename": "a.txt",
                                     "b64": base64.b64encode(raw).decode()})
        self.assertEqual(r["kind"], "text")
        self.assertIn("token_ids", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
