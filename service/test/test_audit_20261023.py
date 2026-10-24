# ===========================================================================
# GARGANTUA v1 — tests/test_audit_20261023.py：2026-10-23 严格审查 +
# 上下文长度可选功能的回归测试
# ---------------------------------------------------------------------------
# 审查发现并已修复（本套件逐条钉住）：
#   P0-1  server._read_body：负 Content-Length 绕过 512MB 上限 →
#         rfile.read(-1) 无界读取（内存耗尽 DoS）
#   P0-2  /api/codec/draw 同步渲染 + 无图元数上限（CPU DoS）
#   P1-3  前端 refreshHome 引用不存在的 DOM id（home 环境卡片死亡）
#   P1-4  /api/infer 静默丢弃 media_b64/media_name（契约断裂）
#   P1-5  train-start 不下发 optimizer；status 不返回该字段
#   P1-6  视频帧解码不过尺寸守卫；音轨不随抽帧截断（音画对齐违约）
#   P2-7  Trainer 归档 os.replace 在 try 外 → 守护线程静默死亡
#   P2-8  畸形输入（text 非字符串/图元非 dict/MIDI 截断）→ 500 而非 400
#   P2-9  训练中 init 覆盖活动模型无 409 保护（load 有，契约不对称）
#   P2-10 /api/train/queue getsize 竞态 → 500
#   P3    num_param 类型混淆；Trainer.start() 无锁；enqueue 无样本上限；
#         loss 记录缺 file 字段；patch 总量上限过松；config 与权重不匹配
#         抛 RuntimeError（→500）而非 ValueError
# 新功能：建模时可选上下文长度（完整/压缩容量 + CSA/HCA 压缩比与局部窗），
#         config.json 驱动，全链路（init.py CLI / /api/model/init /
#         前端表单 / build_model）一致。
# 运行：cd <ROOT> && <torch解释器> -m unittest tests.test_audit_20261023 -v
# 注意：2026-10-24 本文件曾因误操作被截断，后按上述清单依据代码内审计
# 注释重建覆盖面（原文件从未提交 git，无副本可恢复）；MIDI 相关用例随
# midi_codec 暂时删除一并移除。
# ===========================================================================
import base64
import gc
import io
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import spec
import common
import model as model_mod
import server as srv
import server_codecs
import server_media
import server_models
import server_train
import auto_train
from common import (build_model, read_ctx_config, save_checkpoint,
                    validate_ctx_config)
from model import Gargantua, SharedKV, StepCompressor
from server_codecs import CodecApi, MAX_PRIMITIVES
from server_media import (MAX_AUDIO_TOKENS, MAX_IMAGE_DIM, MAX_PATCHES_TOTAL,
                          extract_video_frames, find_ffmpeg, size_guard_pil)
from server_models import ConflictError, ModelManager, TaskPool
from server_train import MAX_ENQUEUE_SAMPLES, TrainApi
from server_util import num_param
from auto_train import Trainer

HAVE_FFMPEG = find_ffmpeg() is not None


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------
def _probe_body(headers: dict, payload: bytes):
    """用 socketpair 驱动 Handler._read_body，验证 Content-Length 边界。"""
    a, b = socket.socketpair()
    peer = a.makefile("rb")

    class _FakeSelf:
        pass

    fs = _FakeSelf()
    fs.headers = headers
    fs.rfile = peer

    def _sender():
        try:
            b.sendall(payload)
            b.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    t = threading.Thread(target=_sender)
    t.start()
    try:
        return srv.Handler._read_body(fs)
    finally:
        t.join()
        peer.close()
        a.close()
        b.close()


class _ApiHost(CodecApi, TrainApi):
    """给 mixin 提供最小依赖的宿主（不起真 server）。"""

    def __init__(self):
        self.pool = TaskPool()
        self.queue_dir = tempfile.mkdtemp(prefix="garg_audit_q_")
        os.makedirs(self.queue_dir, exist_ok=True)


def _wait_task(pool, tid, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        t = pool.get(tid)
        if t["status"] != "running":
            return t
        time.sleep(0.1)
    raise AssertionError("任务轮询超时")


class _StubNet(torch.nn.Module):
    """替身模型：够 build_model / save_checkpoint / 训练一步用的最小实现。"""

    def __init__(self, n_enc_blocks=None, n_dec_blocks=None, **ctx):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        for k, v in ctx.items():
            setattr(self, k, v)
        self.fast_weight = types.SimpleNamespace(reset=lambda: None)

    def forward_train(self, x_in, s_in, dec_in, s_dec):
        # logits 宽度必须是真实词表（cross_entropy 的 target 是 token id）
        return (self.w + 1.0).view(1, 1, 1).expand(
            1, dec_in.shape[1], spec.VOCAB_SIZE_PADDED)


def _make_png_b64(w, h, rgba=(200, 100, 50, 255)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGBA", (w, h), rgba).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_wav_b64(n_samples, sr=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(
            (np.arange(n_samples) % 2000 - 1000).astype("<i2").tobytes())
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# P0-1 / P3：HTTP 请求体与数值入参契约
# ---------------------------------------------------------------------------
class TestHTTPContract(unittest.TestCase):
    """ P0-1 负 Content-Length 无界读取；P3 num_param 类型混淆；
    P1-4 推理媒体字段；P2-9 训练中覆盖活动模型。 """

    def test_negative_content_length_rejected(self):
        # 修复前：int("-1") 通过 `n > MAX` 判断，read(-1) 读到 EOF 为止
        for cl in ("-1", " -1 ", "-999999"):
            with self.assertRaises(ValueError, msg=cl):
                _probe_body({"Transfer-Encoding": "",
                             "Content-Length": cl}, b"A" * 4096)

    def test_non_numeric_content_length_rejected(self):
        for cl in ("abc", "1_0", "0x10", ""):
            if cl == "":
                continue                      # 空 → 0 → 空 body，合法
            with self.assertRaises(ValueError, msg=cl):
                _probe_body({"Transfer-Encoding": "",
                             "Content-Length": cl}, b"{}")

    def test_normal_content_length_still_works(self):
        body = _probe_body({"Transfer-Encoding": "", "Content-Length": "7"},
                           b'{"a":1}')
        self.assertEqual(body, {"a": 1})

    def test_chunked_still_rejected(self):
        with self.assertRaises(srv.BodyTooLarge):
            _probe_body({"Transfer-Encoding": "chunked",
                         "Content-Length": "5"}, b"{}")

    def test_num_param_rejects_bool_and_junk(self):
        # 修复前：int(True)==1、float("0.8") 被静默接受
        with self.assertRaises(ValueError):
            num_param({"max_new_tokens": True}, "max_new_tokens", 32, int)
        with self.assertRaises(ValueError):
            num_param({"temperature": True}, "temperature", 0.8, float)
        with self.assertRaises(ValueError):
            num_param({"max_new_tokens": "abc"}, "max_new_tokens", 32, int)
        with self.assertRaises(ValueError):
            num_param({"max_new_tokens": 1.5}, "max_new_tokens", 32, int)
        with self.assertRaises(ValueError):
            num_param({"temperature": []}, "temperature", 0.8, float)
        with self.assertRaises(ValueError):
            num_param({"temperature": "abc"}, "temperature", 0.8, float)

    def test_num_param_accepts_numeric_strings_and_integral_floats(self):
        self.assertEqual(
            num_param({"max_new_tokens": "16"}, "max_new_tokens", 32, int), 16)
        self.assertEqual(
            num_param({"max_new_tokens": 16.0}, "max_new_tokens", 32, int), 16)
        self.assertEqual(
            num_param({"temperature": "0.8"}, "temperature", 0.8, float), 0.8)
        self.assertEqual(
            num_param({"temperature": 1}, "temperature", 0.8, float), 1.0)

    def test_num_param_explicit_zero_kept(self):
        # 2026-10-20 修复的回归：显式 0 不被 or 默认值吞掉
        self.assertEqual(
            num_param({"max_new_tokens": 0}, "max_new_tokens", 32, int), 0)
        self.assertEqual(
            num_param({"temperature": 0}, "temperature", 0.8, float), 0.0)

    def test_infer_rejects_media_fields(self):
        # 修复前：media_b64/media_name 被静默丢弃，用户以为模型看到了附件
        tmp = tempfile.mkdtemp(prefix="garg_audit_app_")
        try:
            app = srv.App(os.path.join(tmp, "models"),
                          os.path.join(tmp, "data"),
                          os.path.join(tmp, "logs", "auto_train.jsonl"))
            app.mgr.model = object()
            app.mgr.name = "stub"
            for body in ({"prompt": "hi", "media_b64": "AAAA"},
                         {"prompt": "hi", "media_name": "cat.png"}):
                with self.assertRaises(ValueError, msg=str(body)):
                    app.api_infer(body)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_model_init_conflict_during_training(self):
        # 修复前：load 有 409 保护、init 没有——训练中覆盖活动模型会抹掉
        # 磁盘上的训练成果
        tmp = tempfile.mkdtemp(prefix="garg_audit_app_")
        try:
            app = srv.App(os.path.join(tmp, "models"),
                          os.path.join(tmp, "data"),
                          os.path.join(tmp, "logs", "auto_train.jsonl"))
            app.mgr.name = "tiny"
            app.mgr.model = object()
            app.trainer = types.SimpleNamespace(running=True)
            with self.assertRaises(ConflictError):
                app.api_model_init({"name": "tiny", "overwrite": True})
            # 不同名的模型不冲突（不碰活动模型）
            code, obj = app.api_model_init({"name": "other"})
            self.assertEqual(code, 202)
            t = _wait_task(app.pool, obj["task_id"])
            self.assertEqual(t["status"], "done", t.get("error"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_clamp_int_rejects_bool(self):
        with self.assertRaises(ValueError):
            srv.App._clamp_int(True, 1, 64, "n_enc_blocks")
        with self.assertRaises(ValueError):
            srv.App._clamp_int("abc", 1, 64, "n_enc_blocks")
        self.assertEqual(srv.App._clamp_int("8", 1, 64, "n_enc_blocks"), 8)
        with self.assertRaises(ValueError):
            srv.App._clamp_int(65, 1, 64, "n_enc_blocks")


# ---------------------------------------------------------------------------
# P0-2 / P2-8 / P1-6：编解码端点契约与资源守卫
# ---------------------------------------------------------------------------
class TestCodecContracts(unittest.TestCase):
    """ draw 异步+上限；入参类型 400；视频尺寸/音画对齐；patch/音元上限。 """

    @classmethod
    def setUpClass(cls):
        cls.host = _ApiHost()

    def test_draw_is_async_and_capped(self):
        # 修复前：同步渲染 129ms/图元且无上限，占死 HTTP handler 线程
        prims = [{"cx": 10, "cy": 10, "w": 20, "h": 20, "shape_type": 0,
                  "theta": 0.0, "r": 100, "g": 100, "b": 100}]
        code, obj = self.host.api_codec_draw({"primitives": prims})
        self.assertEqual(code, 202)
        self.assertIn("task_id", obj)
        t = _wait_task(self.host.pool, obj["task_id"])
        self.assertEqual(t["status"], "done", t.get("error"))
        self.assertTrue(
            base64.b64decode(t["result"]["png_b64"]).
            startswith(b"\x89PNG\r\n\x1a\n"))
        # tokens → 渲染同样异步
        code, obj = self.host.api_codec_draw({"tokens": t["result"]["tokens"]})
        self.assertEqual(code, 202)
        t2 = _wait_task(self.host.pool, obj["task_id"])
        self.assertEqual(t2["status"], "done", t2.get("error"))
        self.assertEqual(t2["result"]["n_primitives"], 1)

    def test_draw_rejects_bad_input(self):
        with self.assertRaises(ValueError):        # 非数组
            self.host.api_codec_draw({"primitives": "x"})
        with self.assertRaises(ValueError):        # 非 dict 成员（修复前 500）
            self.host.api_codec_draw({"primitives": ["notadict"]})
        with self.assertRaises(ValueError):        # tokens 非整数数组
            self.host.api_codec_draw({"tokens": "123"})
        with self.assertRaises(ValueError):
            self.host.api_codec_draw({"tokens": [True]})
        with self.assertRaises(ValueError):        # 两者都缺
            self.host.api_codec_draw({})
        # 图元数上限（用 patch 缩小常量，避免真渲染一千个图元）
        with mock.patch.object(server_codecs, "MAX_PRIMITIVES", 3):
            with self.assertRaises(ValueError) as cm:
                self.host.api_codec_draw(
                    {"primitives": [{"cx": 1, "cy": 1}] * 5})
            self.assertIn("超限", str(cm.exception))
        self.assertGreaterEqual(MAX_PRIMITIVES, 1000)   # 真实上限足够大

    def test_text_rejects_non_string(self):
        with self.assertRaises(ValueError):
            self.host.api_codec_text({"text": 123})
        with self.assertRaises(ValueError):
            self.host.api_codec_text({"text": None})
        code, obj = self.host.api_codec_text({"text": "你好"})
        self.assertEqual(code, 200)
        self.assertTrue(obj["steps"])

    def test_audio_token_cap(self):
        # 修复前：超长音频无上限，533 采样/音元可把内存打爆
        with mock.patch.object(server_codecs, "MAX_AUDIO_TOKENS", 5):
            with self.assertRaises(ValueError) as cm:
                self.host.api_codec_audio(
                    {"audio_b64": _make_wav_b64(16000 * 5),
                     "filename": "long.wav"})
            self.assertIn("超限", str(cm.exception))
        code, obj = self.host.api_codec_audio(
            {"audio_b64": _make_wav_b64(1600), "filename": "t.wav"})
        self.assertEqual(code, 200)
        self.assertEqual(obj["n_tokens"], 4)

    def test_image2prims_input_validation(self):
        # 修复前：非法 quality / max_steps 被静默改成默认值
        with self.assertRaises(ValueError):
            self.host.api_codec_image2prims(
                {"image_b64": _make_png_b64(32, 32), "quality": "medium"})
        for bad in (0, 1001, -1):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.host.api_codec_image2prims(
                    {"image_b64": _make_png_b64(32, 32), "max_steps": bad})
        code, obj = self.host.api_codec_image2prims(
            {"image_b64": _make_png_b64(32, 32), "max_steps": 3,
             "quality": "fast"})
        self.assertEqual(code, 202)
        t = _wait_task(self.host.pool, obj["task_id"])
        self.assertEqual(t["status"], "done", t.get("error"))

    def test_enqueue_caps(self):
        # 修复前：入队无数量/体积上限
        with mock.patch.object(server_train, "MAX_ENQUEUE_SAMPLES", 2):
            body = {"json_b64": base64.b64encode(
                json.dumps([{"text": "a"}, {"text": "b"},
                            {"text": "c"}]).encode()).decode(),
                "filename": "many.json"}
            with self.assertRaises(ValueError) as cm:
                self.host.api_train_enqueue(body)
            self.assertIn("超限", str(cm.exception))
        raw = base64.b64encode(b'[{"text": "hi"}]').decode()
        code, obj = self.host.api_train_enqueue(
            {"json_b64": raw, "filename": "ok.json"})
        self.assertEqual(code, 200)
        self.assertEqual(len(obj["queued"]), 1)

    def test_queue_getsize_race_tolerated(self):
        # 修复前：glob 与 getsize 之间文件被归档/删除 → FileNotFoundError
        # 逃逸 → 500（训练期间前端轮询概率性触发）
        open(os.path.join(self.host.queue_dir, "20261024_120000_muon_ab.txt"),
             "w").close()
        with mock.patch("os.path.getsize", side_effect=FileNotFoundError):
            code, obj = self.host.api_train_queue()      # 不得抛
        self.assertEqual(code, 200)
        self.assertEqual(obj["items"], [])

    def test_queue_reports_optimizer(self):
        # 修复前：队列项不含 optimizer，UI 无从判断 muon/adamw 是否生效
        name = "20261024_120000_adamw_ab12cd34.txt"
        open(os.path.join(self.host.queue_dir, name), "w").close()
        code, obj = self.host.api_train_queue()
        self.assertEqual(code, 200)
        item = [i for i in obj["items"] if i["name"] == name][0]
        self.assertEqual(item["optimizer"], "adamw")


# ---------------------------------------------------------------------------
# P1-3 / P1-5 / P1-6 / P3：前端 DOM、训练状态、媒体守卫、构建一致性
# ---------------------------------------------------------------------------
class TestAuditMisc(unittest.TestCase):
    """ 审计清单里不适合放进上面两类的条目。 """

    def test_frontend_dom_ids_exist(self):
        # 修复前：refreshHome 引用不存在的 DOM id（home 环境卡片死亡）
        html = (Path(ROOT) / "web" / "index_source.html").read_text(
            encoding="utf-8")
        # 负向回顾排除 $$（querySelectorAll，参数是 CSS 选择器不是 id）
        js_ids = set(re.findall(r'(?<!\$)\$\("([^"]+)"\)', html))
        dom_ids = set(re.findall(r'id="([^"]+)"', html))
        self.assertEqual(js_ids - dom_ids, set(),
                         "JS 引用了不存在的 DOM id")

    def test_train_status_has_optimizer_field(self):
        tmp = tempfile.mkdtemp(prefix="garg_audit_app_")
        try:
            app = srv.App(os.path.join(tmp, "models"),
                          os.path.join(tmp, "data"),
                          os.path.join(tmp, "logs", "auto_train.jsonl"))
            st = app.api_train_status()
            self.assertIn("optimizer", st)
            self.assertIn("running", st)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_size_guard_rejects_huge_images(self):
        # 修复前：解出的帧不过尺寸守卫，8192² 单图就是 256MB 向量
        img = Image.new("RGBA", (64, 64), (1, 2, 3, 255))
        size_guard_pil(img)                       # 正常尺寸不拦
        with mock.patch.object(server_media, "MAX_IMAGE_DIM", 32):
            with self.assertRaises(ValueError):
                size_guard_pil(img)

    def test_patch_total_cap_sane(self):
        # 修复前：patch 总量上限 100_000 过松（单请求可上 GB 向量）
        self.assertGreaterEqual(MAX_PATCHES_TOTAL, 16384)
        self.assertLessEqual(MAX_PATCHES_TOTAL, 65536)
        with mock.patch.object(server_codecs, "MAX_PATCHES_TOTAL", 1):
            host = _ApiHost()
            with self.assertRaises(ValueError) as cm:
                host.api_codec_image(
                    {"image_b64": _make_png_b64(64, 64)})
            self.assertIn("超限", str(cm.exception))

    def test_build_model_config_mismatch_is_value_error(self):
        # 修复前：config 与权重不一致抛 RuntimeError → HTTP 500 + 大段
        # missing keys；转成 ValueError（→400）并给结论
        tmp = tempfile.mkdtemp(prefix="garg_audit_mdl_")
        try:
            m = Gargantua(n_enc_blocks=1, n_dec_blocks=1)
            save_checkpoint(tmp, m, {"name": "x", "n_enc_blocks": 1,
                                     "n_dec_blocks": 1})
            del m
            gc.collect()
            # 手改 config 的 block 数 → 与权重结构不符
            cfg_path = os.path.join(tmp, "config.json")
            with open(cfg_path) as f:
                cfg = json.load(f)
            cfg["n_enc_blocks"] = 3
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)
            with self.assertRaises(ValueError) as cm:
                build_model(tmp)
            self.assertIn("不匹配", str(cm.exception))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2026-10-23 新功能回归：建模时可选上下文长度（config.json 驱动）
# ---------------------------------------------------------------------------
class TestContextLength(unittest.TestCase):
    """ validate_ctx_config 强校；read_ctx_config 容错回退；
    kv_note 反映真实配置；init 的 ctx 非法值 → 400。 """

    def test_validate_defaults_and_custom(self):
        cfg = validate_ctx_config()
        self.assertEqual(cfg, {"ctx_full": spec.CTX_FULL,
                               "ctx_compress": spec.CTX_COMPRESS,
                               "csa_ratio": spec.CSA_RATIO,
                               "hca_ratio": spec.HCA_RATIO,
                               "csa_local_window": spec.CSA_LOCAL_WINDOW})
        custom = validate_ctx_config(ctx_full=512, ctx_compress=1024,
                                     csa_ratio=8, hca_ratio=16,
                                     csa_local_window=128)
        self.assertEqual(custom, {"ctx_full": 512, "ctx_compress": 1024,
                                  "csa_ratio": 8, "hca_ratio": 16,
                                  "csa_local_window": 128})

    def test_validate_cross_field_rules(self):
        # HCA 是重压缩视图：比例不得低于 CSA
        with self.assertRaises(ValueError):
            validate_ctx_config(csa_ratio=8, hca_ratio=4)
        # 局部窗不得越过完整区
        with self.assertRaises(ValueError):
            validate_ctx_config(ctx_full=64, csa_local_window=128)
        for kw in ({"ctx_full": 0}, {"ctx_full": common.CTX_FULL_MAX + 1},
                   {"csa_ratio": 1},
                   {"hca_ratio": common.CTX_RATIO_MAX + 1},
                   {"ctx_full": True}, {"csa_ratio": "abc"},
                   {"csa_ratio": 2.5}):
            with self.assertRaises(ValueError, msg=str(kw)):
                validate_ctx_config(**kw)

    def test_read_ctx_config_roundtrip_and_fallback(self):
        tmp = tempfile.mkdtemp(prefix="garg_audit_ctx_")
        try:
            cfg = {"ctx_full": 2048, "ctx_compress": 0, "csa_ratio": 4,
                   "hca_ratio": 32, "csa_local_window": 256}
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump(cfg, f)
            self.assertEqual(read_ctx_config(tmp), cfg)
            # 坏值 → 告警回退默认（不让模型加载崩）
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump({"ctx_full": "bad", "csa_local_window": 99999}, f)
            got = read_ctx_config(tmp)
            self.assertEqual(got["ctx_full"], spec.CTX_FULL)
            self.assertEqual(got["csa_local_window"], spec.CSA_LOCAL_WINDOW)
            # 空目录 → 全默认
            self.assertEqual(read_ctx_config(tmp + "_empty"),
                             validate_ctx_config())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_kv_note_reflects_config(self):
        tmp = tempfile.mkdtemp(prefix="garg_audit_app_")
        try:
            app = srv.App(os.path.join(tmp, "models"),
                          os.path.join(tmp, "data"),
                          os.path.join(tmp, "logs", "auto_train.jsonl"))
            app.mgr.ctx = validate_ctx_config(
                ctx_full=4096, ctx_compress=8192, csa_ratio=8,
                hca_ratio=64, csa_local_window=1024)
            note = app.mgr.ctx_note()
            for frag in ("4096", "8192", "CSA 8:1", "HCA 64:1"):
                self.assertIn(frag, note)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_model_init_rejects_bad_ctx(self):
        # ctx 非法在 submit 之前就 400（不起任务、不建模型）
        tmp = tempfile.mkdtemp(prefix="garg_audit_app_")
        try:
            app = srv.App(os.path.join(tmp, "models"),
                          os.path.join(tmp, "data"),
                          os.path.join(tmp, "logs", "auto_train.jsonl"))
            app.mgr.name = None
            with self.assertRaises(ValueError):
                app.api_model_init({"name": "m", "csa_ratio": 8,
                                    "hca_ratio": 4})
            with self.assertRaises(ValueError):
                app.api_model_init({"name": "m", "ctx_full": 64,
                                    "csa_local_window": 128})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


