# ===========================================================================
# GARGANTUA v1 — 2026-11-06 专用测试：对比训练（CLIP）+ 全项目终端接口
# ---------------------------------------------------------------------------
# 本次需求的三条硬约束在这里逐条验收：
#   1. 训练界面有对比训练模式，且只吃 .tokens 配对文件；
#   2. 所有 GUI 操作都有终端命令（静态解析 server.py 路由表做覆盖对拍）；
#      终端可离线完成 .tokens 转换与训练任务队列管理；
#   3. 既有 LM 训练路径零回归（结构对拍 + 本套件内的端到端跑通）。
#
# 运行：python service/test/test_contrastive_cli_20261106.py -v
#   或  python service/test/run_tests.py --suite test_contrastive_cli_20261106
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

import base64
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import spec
from tokens import tok6
import clip_data

ROOT = Path(boot.project_root())
CLI = ROOT / "service" / "cli" / "gargantua.py"


# ---------------------------------------------------------------------------
# 工具：构造最小的 .tokens 素材
# ---------------------------------------------------------------------------
def _patch_hex(v: int = 128) -> str:
    return tok6.rgba8_to_hex([v] * spec.PATCH_VEC_DIM)


def _wave_hex(v: int = 100) -> str:
    return tok6.samples_to_hex([v] * spec.AUDIO_SAMPLES_PER_TOKEN)


def _text_items(n=2, start=100):
    return [tok6.step(tok6.text(start + i)) for i in range(n)]


def _media_items(kind="image", n=1, v=128):
    if kind == "audio":
        return [tok6.step(tok6.wave(_wave_hex(v))) for _ in range(n)]
    if kind == "image":
        return [tok6.step(tok6.visual(_patch_hex(v), 0, 0))
                for _ in range(n)]
    # video：多个视觉帧，其中一帧带音元（同一步 patches + wave）
    out = [tok6.step(tok6.visual(_patch_hex(v), 0, 0))]
    out.append(tok6.step(tok6.visual(_patch_hex(v), 1, 0),
                         tok6.wave(_wave_hex(v))))
    return out


def _pair_bytes(kind="image", n_text=2, v=128) -> bytes:
    return tok6.dumps(_text_items(n_text) + _media_items(kind, v=v))


def _write(path, data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    Path(path).write_bytes(data)
    return str(path)


def _run_cli(*args, timeout=180):
    return subprocess.run([sys.executable, str(CLI), *map(str, args)],
                          capture_output=True, text=True, cwd=str(ROOT),
                          timeout=timeout)


def _json_of(r):
    out = (r.stdout or "").strip()
    try:
        return json.loads(out)
    except Exception:
        pass
    i, j = out.find("{"), out.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(out[i:j + 1])
        except Exception:
            pass
    return None


class _Tmp(unittest.TestCase):
    """每个用例一个项目内临时目录（可移植性：临时文件不落项目外）。"""

    def setUp(self):
        # 2026-11-08：**不注册 cleanup，也不调 TemporaryDirectory**。
        # 宿主的批量删除守卫（SAFE_DELETE_BULK_CONFIRM_REQUIRED）在 rmtree
        # 时抛的不是 Exception，`ignore_cleanup_errors=True` 也挡不住；
        # 全量回归跑到后段（累计删除过阈值）会把「清理」污染成测试失败。
        # data/tmp 是项目约定的运行时目录、已被 .gitignore 排除，残留交给
        # OS 回收——测试本身不做删除，就与守卫解耦。
        self.tmp = Path(tempfile.mkdtemp(dir=str(ROOT / "data" / "tmp")))


# ---------------------------------------------------------------------------
# 1. 配对 .tokens 契约
# ---------------------------------------------------------------------------
class TestPairData(_Tmp):
    def test_01_three_media_kinds(self):
        for kind, want in (("image", "image"), ("audio", "audio"),
                           ("video", "video")):
            p = clip_data.parse_bytes(_pair_bytes(kind))
            self.assertEqual(p["media_kind"], want)
            self.assertEqual(p["n_text_tokens"], 2)
            self.assertGreaterEqual(p["n_media_positions"], 1)
            self.assertTrue(p["text_steps"] and p["media_steps"])

    def test_02_text_or_media_first_both_ok(self):
        a = _pair_bytes("image")
        rev = tok6.dumps(_media_items("image") + _text_items())
        self.assertEqual(clip_data.parse_bytes(a)["media_kind"], "image")
        self.assertEqual(clip_data.parse_bytes(rev)["media_kind"], "image")

    def test_03_reject_everything_that_is_not_a_pair(self):
        bad_cases = {
            "text only": tok6.dumps(_text_items()),
            "media only": tok6.dumps(_media_items()),
            "interleaved": tok6.dumps(
                [tok6.step(tok6.text(100))]
                + _media_items("image")
                + [tok6.step(tok6.text(101))]),
            "empty": tok6.dumps([]),
            "not tokens": b"\x00\x01\x02\x03",
        }
        for name, data in bad_cases.items():
            with self.subTest(name=name):
                with self.assertRaises(clip_data.PairError):
                    clip_data.parse_bytes(data)

    def test_04_pair_bytes_roundtrip(self):
        # 走 CLI 也要用的构建入口，而不只是 parse 路径
        p = clip_data.parse_bytes(_pair_bytes("image"))
        rebuilt = clip_data.pair_bytes(p["text_steps"], p["media_steps"])
        q = clip_data.parse_bytes(rebuilt)
        self.assertEqual(q["media_kind"], p["media_kind"])
        self.assertEqual(q["n_text_tokens"], p["n_text_tokens"])
        self.assertEqual(q["n_media_positions"], p["n_media_positions"])

    def test_05_cap_media_keeps_step_grouping(self):
        p = clip_data.parse_bytes(_pair_bytes("image"), 512)
        many = []
        for i in range(4):
            many.append({"token_ids": [], "type": spec.TYPE_USER,
                         "modality": "visual",
                         "meta": {"patches": [
                             {"x": j, "y": 0,
                              "vec": bytes([128] * spec.PATCH_VEC_DIM)}
                             for j in range(3)]}})
        capped = clip_data.cap_media_steps(many, 5)
        self.assertEqual(sum(clip_data.n_positions(s) for s in capped), 5)
        self.assertLessEqual(len(capped), len(many))
        # 重建后仍能被 assemble_embeddings 消费（步内 patches 分组保留）
        self.assertEqual(clip_data.media_kind_of(many), "video")
        self.assertGreater(clip_data.n_positions(many[0]), 0)

    def test_06_queue_name_metadata(self):
        new = clip_data.make_queue_name("muon", "clip",
                                       "20261106_121800", "abcd1234")
        self.assertEqual(new,
                         "20261106_121800_muon_clip_abcd1234.tokens")
        self.assertEqual(clip_data.parse_queue_name(new),
                         {"optimizer": "muon", "mode": "clip"})
        legacy = "20261106_121800_muon_abcd1234.tokens"
        self.assertEqual(clip_data.parse_queue_name(legacy),
                         {"optimizer": "muon", "mode": "lm"})
        for bogus in ("../../evil.tokens", "x_clip_y.tokens"):
            m = clip_data.parse_queue_name(bogus)
            self.assertIn(m["mode"], clip_data.MODES)


# ---------------------------------------------------------------------------
# 2. CLIP 对比头 / 损失 / 端到端训练步
# ---------------------------------------------------------------------------
class TestClipCore(_Tmp):
    @classmethod
    def setUpClass(cls):
        import torch
        from model import Gargantua
        cls.torch = torch
        # 极小模型：1+1 blocks / d_model=128 / 短词表，CPU 上秒级
        cls.m = Gargantua(n_enc_blocks=1, n_dec_blocks=1,
                          d_model=128, vocab_size=256)
        cls.m.eval()

    def test_01_head_is_separate_from_model_state_dict(self):
        import clip
        before = set(self.m.state_dict().keys())
        head = clip.ContrastiveHead(128, init_temperature=0.07)
        self.assertAlmostEqual(head.temperature(), 0.07, places=5)
        self.assertTrue(head.scale().requires_grad)
        after = set(self.m.state_dict().keys())
        self.assertEqual(before, after,
                         "对比头绝不能挂到模型上：会破坏 strict load 契约")
        self.assertEqual(set(head.state_dict().keys()),
                         {"text_proj.weight", "media_proj.weight",
                          "logit_scale"})

    def test_02_head_save_load_roundtrip(self):
        import clip
        head = clip.ContrastiveHead(128, 0.05)
        head.logit_scale.data.fill_(0.3)
        clip.save_head(str(self.tmp), head)
        self.assertTrue((self.tmp / clip.HEAD_FILENAME).is_file())
        got = clip.load_head(str(self.tmp), 128)
        self.assertTrue(self.torch.allclose(head.logit_scale,
                                            got.logit_scale))
        with self.assertRaises(ValueError):
            clip.load_head(str(self.tmp), 256)

    def test_03_clip_loss_symmetric_and_finite(self):
        import clip
        pairs = [clip_data.parse_bytes(_pair_bytes("image", v=100 + 30 * i))
                 for i in range(2)]
        head = clip.ContrastiveHead(128)
        loss, stats = clip.clip_loss(self.m, head, pairs)
        self.assertTrue(bool(self.torch.isfinite(loss)))
        self.assertGreater(float(loss), 0.0)
        for k in ("n_pairs", "temperature", "t2m_acc", "m2t_acc",
                  "diag_mean", "offdiag_mean"):
            self.assertIn(k, stats)
        self.assertEqual(stats["n_pairs"], 2)
        with self.assertRaises(ValueError):
            clip.clip_loss(self.m, head, pairs[:1])

    def test_04_run_clip_step_updates_both_model_and_head(self):
        import clip
        from muon import build_optimizers
        pairs = [clip_data.parse_bytes(_pair_bytes("image", v=90 + 40 * i))
                 for i in range(2)]
        head = clip.ContrastiveHead(128)
        before_head = head.text_proj.weight.detach().clone()
        before_model = self.m.wte.weight.detach().clone()
        opts = build_optimizers(self.m, "adamw", spec.ADAMW_LR,
                                spec.MUON_LR, spec.MUON_MOMENTUM)
        hopt = clip.build_head_optimizer(head, spec.ADAMW_LR)
        loss, stats = clip.run_clip_step(self.m, head, pairs, opts, hopt)
        self.assertTrue(bool(self.torch.isfinite(loss)))
        self.assertFalse(self.torch.allclose(before_head,
                                             head.text_proj.weight))
        self.assertFalse(self.torch.allclose(before_model, self.m.wte.weight))

    def test_05_trainer_dispatches_clip_batch_end_to_end(self):
        """两个 clip 队列文件 → 同批对比训练 → 归档 done + loss 记录 + 头落盘。"""
        from auto_train import Trainer
        import clip
        qdir, ddir = self.tmp / "queue", self.tmp / "done"
        qdir.mkdir(); ddir.mkdir()
        log = self.tmp / "auto_train.jsonl"
        for i in range(2):
            _write(qdir / clip_data.make_queue_name(
                "adamw", "clip", "20261106_12180%d" % i, "pair%d" % i),
                _pair_bytes("image", v=100 + 40 * i))
        mdir = self.tmp / "model"; mdir.mkdir()
        saved = []
        trainer = Trainer(
            get_model=lambda: self.m, save_fn=lambda mod: saved.append(True),
            lock=threading.RLock(), queue_dir=str(qdir), done_dir=str(ddir),
            log_path=str(log), block_size=4, steps_per_file=1,
            clip_batch_size=2, clip_temperature=0.07, model_dir=str(mdir))
        trainer.optimizer = "adamw"
        trainer.start()
        try:
            deadline = time.time() + 120
            while time.time() < deadline and len(list(ddir.glob("*.tokens"))) < 2:
                time.sleep(0.3)
            self.assertEqual(len(list(ddir.glob("*.tokens"))), 2,
                             "两个 clip 任务应同批训练后一起归档；"
                             "日志: " + "\n".join(trainer.status()["log_tail"]))
            self.assertTrue(saved, "CLIP 批次完成后必须落模型检查点")
            self.assertTrue((mdir / clip.HEAD_FILENAME).is_file(),
                            "CLIP 批次完成后必须落对比头")
            recs = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
            self.assertTrue(recs)
            self.assertEqual(recs[0].get("mode"), "clip")
            self.assertEqual(recs[0].get("n_pairs"), 2)
        finally:
            trainer.stop()
            trainer.join(timeout=30)

    def test_06_lone_clip_does_not_starve_lm(self):
        """队首只有 1 个 clip 配对时，后面的 lm 任务仍必须被训练。"""
        from auto_train import Trainer
        qdir, ddir = self.tmp / "queue2", self.tmp / "done2"
        qdir.mkdir(); ddir.mkdir()
        clip_name = clip_data.make_queue_name(
            "adamw", "clip", "20261106_121800", "only1")
        lm_name = clip_data.make_queue_name(
            "adamw", "lm", "20261106_121801", "normal")
        _write(qdir / clip_name, _pair_bytes("image"))
        _write(qdir / lm_name, tok6.dumps(_text_items(20)))
        mdir = self.tmp / "model2"; mdir.mkdir()
        trainer = Trainer(
            get_model=lambda: self.m, save_fn=lambda mod: None,
            lock=threading.RLock(), queue_dir=str(qdir), done_dir=str(ddir),
            log_path=str(self.tmp / "starve.jsonl"), block_size=4,
            steps_per_file=1, clip_batch_size=2, model_dir=str(mdir))
        trainer.optimizer = "adamw"
        trainer.start()
        try:
            deadline = time.time() + 120
            while time.time() < deadline and not (
                    ddir / lm_name).exists():
                time.sleep(0.3)
            self.assertTrue((ddir / lm_name).exists(),
                            "孤立 clip 把后续 lm 任务饿死了；日志: "
                            + "\n".join(trainer.status()["log_tail"]))
            self.assertTrue((qdir / clip_name).exists(),
                            "不足 2 个配对时 clip 文件必须留在队列等待")
        finally:
            trainer.stop()
            trainer.join(timeout=30)


# ---------------------------------------------------------------------------
# 3. 服务端入队/编排（不启 HTTP server，直接测 TrainApi）
# ---------------------------------------------------------------------------
class _FakeSupervisor:
    def __init__(self):
        self.calls = []

    def call(self, svc, op, **kw):
        self.calls.append((svc, op, kw))
        return {"started": True, "running": True}


from server_train import TrainApi


class _FakeTrainApp(TrainApi):
    """只补齐 TrainApi 需要的属性，避免起整个 App。

    2026-11-07：训练 API 增加了任务元数据 sidecar / 事件流 / 分组队列等
    内部方法，逐个「借用」会漏，改为直接继承 TrainApi——仍然只补属性，
    测的依旧是线上代码。
    """

    def __init__(self, root):
        self.data_dir = str(root)
        self.queue_dir = str(root / "queue")
        os.makedirs(self.queue_dir, exist_ok=True)
        self.log_path = str(root / "auto_train.jsonl")
        self.mgr = SimpleNamespace(name="tiny", model_dir=str(root / "models"),
                                   load=lambda n: None)
        self._state_lock = threading.Lock()
        self.trainer = SimpleNamespace(running=False)
        self.supervisor = _FakeSupervisor()


class TestTrainApi(_Tmp):
    def setUp(self):
        super().setUp()
        self.app = _FakeTrainApp(self.tmp)

    def _enqueue(self, data, **kw):
        body = {"tokens_b64": base64.b64encode(data).decode()}
        body.update(kw)
        return self.app.api_train_enqueue(body)

    def test_01_lm_default_and_queue_mode(self):
        code, out = self._enqueue(_pair_bytes("image"))
        self.assertEqual(code, 200)
        self.assertEqual(out["mode"], "lm")
        self.assertIn("_lm_", out["queued"][0])
        code, q = self.app.api_train_queue()
        self.assertEqual(q["items"][0]["mode"], "lm")

    def test_02_clip_pair_enqueued_with_metadata(self):
        code, out = self._enqueue(_pair_bytes("video", v=77), mode="clip")
        self.assertEqual(code, 200)
        self.assertEqual(out["mode"], "clip")
        self.assertEqual(out["media_kind"], "video")
        self.assertIn("_clip_", out["queued"][0])
        code, q = self.app.api_train_queue()
        self.assertEqual(q["items"][0]["mode"], "clip")
        # 落盘内容必须能被训练侧原样读回
        p = self.app.queue_dir + "/" + out["queued"][0]
        self.assertEqual(clip_data.read_pair(p)["media_kind"], "video")

    def test_03_clip_rejects_non_pair_and_bad_mode(self):
        for data, kw, label in (
                (tok6.dumps(_text_items()), {"mode": "clip"}, "text only"),
                (tok6.dumps(_media_items()), {"mode": "clip"}, "media only"),
                (b"\x00\x01\x02", {"mode": "clip"}, "not tokens"),
                (_pair_bytes(), {"mode": "bogus"}, "bad mode")):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    self._enqueue(data, **kw)
        # 非法数据绝不落队列
        self.assertEqual(os.listdir(self.app.queue_dir), [])

    def test_04_start_forwards_clip_params_and_validates(self):
        code, _ = self.app.api_train_start(
            {"steps": 1, "block_size": 4,
             "clip_batch_size": 3, "clip_temperature": 0.05})
        self.assertEqual(code, 200)
        kw = self.app.supervisor.calls[-1][2]
        self.assertEqual(kw["clip_batch_size"], 3)
        self.assertAlmostEqual(kw["clip_temperature"], 0.05)
        for bad in ({"clip_batch_size": 0}, {"clip_batch_size": 65},
                    {"clip_temperature": 0}, {"clip_temperature": "x"}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.app.api_train_start(bad)

    def test_05_queue_delete(self):
        _, out = self._enqueue(_pair_bytes(), mode="clip")
        code, res = self.app.api_train_delete({"name": out["queued"][0]})
        self.assertEqual(code, 200)
        self.assertEqual(res["deleted"], out["queued"][0])


# ---------------------------------------------------------------------------
# 4. 终端接口：全 GUI 端点覆盖 + 离线转换 / 队列管理
# ---------------------------------------------------------------------------
class TestCliCoverage(unittest.TestCase):
    def test_01_every_gui_endpoint_has_a_cli_command(self):
        from cli import api_map
        src = (ROOT / "server" / "server.py").read_text(encoding="utf-8")
        get = set(re.findall(r'elif path == "(/api/[^"]+)"', src))
        block = src.split("routes = {", 1)[1].split("\n        }", 1)[0]
        post = set(re.findall(r'"(/api/[^"]+)"\s*:', block))
        task = {"/api/task/<id>"} if 'startswith("/api/task/")' in src else set()
        expected = get | post | task
        self.assertGreater(len(expected), 70, "路由解析失败")
        missing = expected - set(api_map.endpoint_set())
        self.assertEqual(missing, set(),
                         "以下 GUI 端点没有对应终端命令: " + ", ".join(sorted(missing)))

    def test_02_help_lists_offline_commands(self):
        r = _run_cli("--help")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for word in ("tokens", "queue", "model", "train", "api", "server"):
            self.assertIn(word, r.stdout)


class TestCliOffline(_Tmp):
    def test_01_tokens_roundtrip_and_inspect(self):
        a = self.tmp / "text.tokens"
        r = _run_cli("tokens", "encode", "--text", "hello world", "--out", a)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(a.is_file() and a.stat().st_size > 0)
        self.assertTrue(tok6.loads(a.read_bytes()))
        r = _run_cli("tokens", "inspect", a)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("n_steps", r.stdout)
        r = _run_cli("tokens", "decode", a)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_02_pair_and_queue_management(self):
        text = _write(self.tmp / "t.tokens", tok6.dumps(_text_items(2)))
        media = _write(self.tmp / "m.tokens",
                       tok6.dumps(_media_items("image", v=177)))
        pair = self.tmp / "pair.tokens"
        r = _run_cli("tokens", "pair", "--text-tokens", text,
                     "--media-tokens", media, "--out", pair)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(pair.is_file())
        p = clip_data.parse_bytes(pair.read_bytes())
        self.assertEqual(p["media_kind"], "image")
        self.assertEqual(p["n_text_tokens"], 2)

        qdir = self.tmp / "qdata"
        r = _run_cli("--data-dir", qdir, "queue", "add", pair,
                     "--mode", "clip", "--optimizer", "adamw")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        added = list((qdir / "queue").glob("*.tokens"))
        self.assertEqual(len(added), 1)
        self.assertEqual(clip_data.parse_queue_name(added[0].name)["mode"],
                         "clip")
        r = _run_cli("--data-dir", qdir, "queue", "list")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("clip", r.stdout)
        r = _run_cli("--data-dir", qdir, "queue", "remove", added[0].name)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(list((qdir / "queue").glob("*.tokens")), [])

    def test_03_queue_add_rejects_non_tokens_and_bad_pair(self):
        qdir = self.tmp / "qdata2"
        bad_txt = _write(self.tmp / "bad.txt", b"not a tokens file")
        r = _run_cli("--data-dir", qdir, "queue", "add", bad_txt)
        self.assertNotEqual(r.returncode, 0)
        text_only = _write(self.tmp / "textonly.tokens",
                           tok6.dumps(_text_items()))
        r = _run_cli("--data-dir", qdir, "queue", "add", text_only,
                     "--mode", "clip")
        self.assertNotEqual(r.returncode, 0)

    def test_04_convert_single_source(self):
        src = _write(self.tmp / "note.txt", "some words")
        out = self.tmp / "note.tokens"
        r = _run_cli("tokens", "convert", "--source", src,
                     "--kind", "text", "--out", out)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue(tok6.loads(out.read_bytes()))

    def test_05_queue_add_uses_shared_size_cap(self):
        """离线入队与 HTTP 入队必须共用同一套上限（不许悄悄漂移）。"""
        from unittest import mock
        from cli import local
        from server_train import MAX_ENQUEUE_BYTES
        self.assertEqual(MAX_ENQUEUE_BYTES, clip_data.MAX_ENQUEUE_BYTES)
        pair = _write(self.tmp / "cap.tokens", _pair_bytes("image"))
        ctx = {"data_dir": str(self.tmp / "capdata")}
        with mock.patch.object(clip_data, "MAX_ENQUEUE_BYTES", 8):
            with self.assertRaises(ValueError) as cm:
                local._queue_add([pair], ctx)
        self.assertIn("超限", str(cm.exception))
        self.assertEqual(list((self.tmp / "capdata" / "queue").glob(
            "*.tokens")), [])


# ---------------------------------------------------------------------------
# 5. 真实 server 上的终端 HTTP 端到端（证明 CLI 不只是静态映射表）
# ---------------------------------------------------------------------------
class TestCliHttpIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 同上：不注册清理，避免被批量删除守卫污染成测试失败
        cls.tmp = Path(tempfile.mkdtemp(dir=str(ROOT / "data" / "tmp")))
        cls.models = cls.tmp / "models"
        cls.data = cls.tmp / "data"
        cls.logs = cls.tmp / "logs"
        cls.models.mkdir(); cls.data.mkdir(); cls.logs.mkdir()
        d = cls.models / "tiny"
        d.mkdir()
        from model import Gargantua
        from common import save_checkpoint
        m = Gargantua(n_enc_blocks=1, n_dec_blocks=1,
                      d_model=128, vocab_size=256)
        save_checkpoint(str(d), m,
                        {"name": "tiny", "params": sum(p.numel()
                                                       for p in m.parameters()),
                         "n_enc_blocks": 1, "n_dec_blocks": 1})
        del m
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        cls.port = s.getsockname()[1]
        s.close()
        cls.proc = subprocess.Popen(
            [sys.executable, str(ROOT / "server" / "server.py"),
             "--port", str(cls.port), "--no-window",
             "--model-dir", str(cls.models),
             "--data-dir", str(cls.data),
             "--log-file", str(cls.logs / "srv.log")],
            cwd=str(ROOT), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{cls.port}/api/status",
                        timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(0.3)
        else:
            cls._stop_server()
            raise RuntimeError("测试 server 启动超时")

    @classmethod
    def _stop_server(cls):
        p = getattr(cls, "proc", None)
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()

    @classmethod
    def tearDownClass(cls):
        cls._stop_server()
        # 目录不删（见 setUpClass）：data/tmp 是运行时目录，残留由 OS 回收

    def _cli(self, *args):
        return _run_cli("--server", f"http://127.0.0.1:{self.port}",
                        "--json", *args)

    def test_01_status_and_generic_api(self):
        r = self._cli("status")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        obj = _json_of(r)
        self.assertIsInstance(obj, dict)
        r = self._cli("api", "GET", "/api/status")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIsInstance(_json_of(r), dict)

    def test_02_clip_enqueue_via_cli(self):
        pair = _write(self.tmp / "http_pair.tokens", _pair_bytes("image"))
        r = self._cli("train", "enqueue",
                      "--file", "tokens_b64=" + pair,
                      "--param", "mode=clip",
                      "--param", "optimizer=adamw")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        obj = _json_of(r)
        self.assertEqual(obj["mode"], "clip")
        self.assertEqual(obj["media_kind"], "image")
        self.assertIn("_clip_", obj["queued"][0])
        r = self._cli("train", "queue")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        items = _json_of(r)["items"]
        self.assertTrue(any(it["mode"] == "clip" for it in items))
        # 队列状态也能从终端读（离线组的 start/stop/status 走 HTTP）
        r = self._cli("queue", "status")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("running", _json_of(r))


# ---------------------------------------------------------------------------
# 6. 前端与守护进程的接线静态对拍
# ---------------------------------------------------------------------------
class TestWiring(unittest.TestCase):
    def test_01_training_ui_has_clip_controls(self):
        html = (ROOT / "service" / "web" / "index_source.html").read_text(
            encoding="utf-8")
        js = (ROOT / "service" / "web" / "app.js").read_text(encoding="utf-8")
        built = (ROOT / "service" / "web" / "index.html").read_text(
            encoding="utf-8")
        for token in ('id="train-type"', 'value="clip"',
                      'id="train-clip-batch"', 'id="train-mode-hint"'):
            self.assertIn(token, html)
            self.assertIn(token, built, "index.html 未重新生成: " + token)
        for token in ("syncTrainType", 'mode: $("train-type").value',
                      "clip_batch_size", "it.mode"):
            self.assertIn(token, js)

    def test_02_minify_check_passes(self):
        r = subprocess.run([sys.executable,
                            str(ROOT / "service" / "web" / "minify.py"),
                            "--check"], capture_output=True, text=True,
                           cwd=str(ROOT), timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("ALL PASS", r.stdout)

    def test_03_auto_train_keeps_lm_path_and_adds_clip_dispatch(self):
        src = (ROOT / "service" / "train" / "auto_train.py").read_text(
            encoding="utf-8")
        for token in ("parse_queue_name", "_train_clip_batch",
                      "clip_batch_size", "forward_window", "window_loss",
                      "force_last=(step == 0)"):
            self.assertIn(token, src, "auto_train.py 缺少: " + token)

    def test_04_root_launchers_exist_and_use_isolated_python(self):
        sh = (ROOT / "gargantua")
        bat = (ROOT / "gargantua.bat")
        self.assertTrue(sh.is_file() and bat.is_file())
        txt = sh.read_text(encoding="utf-8")
        self.assertIn("environment/venv/bin/python", txt)
        self.assertIn("service/cli/gargantua.py", txt)
        self.assertNotIn("/Users/", txt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
