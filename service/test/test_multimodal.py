# ===========================================================================
# GARGANTUA v1 — test_multimodal.py：多模态训练 / 推理链路
# ---------------------------------------------------------------------------
# 2026-10-28 接线后的回归：
#   - tokenset.read_doc：媒体位置 = None，纯文本文档 multimodal=False
#   - make_batch_emb：切法与 make_batch 同构，媒体目标 = -100
#   - window_index：多模态文档也能挑出「至少有一个文本目标」的窗口
#   - forward_window + window_loss + backward：梯度真的流到
#     visual_proj / audio_proj（投影层参与训练，不是摆设）
#   - generate(media_steps=...)：媒体作为 prompt 前缀，存档顺序 = 模型
#     实际看到的顺序（媒体在前、文本在后）
# 运行：cd <ROOT> && python -m unittest service.test.test_multimodal -v
# ===========================================================================
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
import random
import shutil
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

import spec
from common import make_batch, make_batch_emb
from tokens import audio_codec, image_codec, tok5, tokenset
from tokens.vocab import Vocab
from train_step import forward_window, window_index, window_loss

# 新契约：spec 不再钉死词表长度。generate() 会对真实文本走全局词表编码
# （ID 可达 6.9 万），所以这里的模型必须按全局词表长度开 embedding。
_VS = len(Vocab.global_vocab())


def _multimodal_items():
    """文本 + 一帧图 + 文本 + 一个音元 + 文本。"""
    items = [{"kind": "ext", "id": 1}, tok5.step(tok5.text(50994))]
    items += tok5.from_steps(
        image_codec.frames_to_steps([Image.new("RGBA", (64, 64),
                                               (10, 20, 30, 255))]))
    items.append(tok5.step(tok5.text(51046)))
    items += tok5.from_steps(
        audio_codec.waveform_to_steps(np.arange(533, dtype=np.int16)))
    items.append(tok5.step(tok5.text(12384)))
    return items


class TestReadDoc(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="garg_mm_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, items, name="doc.tokens"):
        p = os.path.join(self.tmp, name)
        tok5.write_file(p, items)
        return p

    def test_media_positions_are_none(self):
        doc = tokenset.read_doc(self._write(_multimodal_items()))
        self.assertTrue(doc["multimodal"])
        self.assertEqual(len(doc["pos_ids"]), len(doc["step_ids"]))
        self.assertIn(None, doc["pos_ids"])
        # 末尾追加了 <stop>，且是文本位置
        self.assertEqual(doc["pos_ids"][-1], spec.STOP_ID)

    def test_text_only_doc(self):
        items = [tok5.step(tok5.text(i)) for i in (50994, 51046, 12384)]
        doc = tokenset.read_doc(self._write(items))
        self.assertFalse(doc["multimodal"])
        self.assertNotIn(None, doc["pos_ids"])

    def test_read_stream_rejects_multimodal(self):
        p = self._write(_multimodal_items())
        with self.assertRaises(tokenset.TokensetError):
            tokenset.read_stream(p)

    def test_position_cap(self):
        items = [tok5.step(tok5.text(50994))]
        doc = tokenset.read_doc(self._write(items))
        self.assertLess(len(doc["pos_ids"]), tokenset.MAX_DOC_POSITIONS)


class TestBatchEmb(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="garg_mm2_")
        p = os.path.join(self.tmp, "doc.tokens")
        tok5.write_file(p, _multimodal_items())
        self.doc = tokenset.read_doc(p)
        self.emb = torch.zeros(1, len(self.doc["pos_ids"]), spec.D_MODEL)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_shapes_match_make_batch(self):
        B = 8
        n = len(self.doc["pos_ids"])
        i = 0
        e_in, s_in, e_dec, s_dec, y, _ = make_batch_emb(
            self.emb, self.doc["pos_ids"], i, B, self.doc["step_ids"])
        ids = [0 if v is None else v for v in self.doc["pos_ids"]]
        x_in, si, dec_in, sd, y_ref = make_batch(ids, i, B,
                                                 self.doc["step_ids"])
        self.assertEqual(e_in.shape, (1, x_in.shape[1], spec.D_MODEL))
        self.assertEqual(e_dec.shape, (1, dec_in.shape[1], spec.D_MODEL))
        self.assertEqual(y.shape, y_ref.shape)
        self.assertTrue(torch.equal(s_in, si))
        self.assertTrue(torch.equal(s_dec, sd))
        self.assertLessEqual(n, self.emb.shape[1])

    def test_media_targets_are_ignored(self):
        B = 8
        _, _, _, _, y, n_valid = make_batch_emb(
            self.emb, self.doc["pos_ids"], 0, B, self.doc["step_ids"])
        half = B // 2
        for k in range(half):
            tgt = self.doc["pos_ids"][0 + half + k]
            if tgt is None:
                self.assertEqual(int(y[0, k]), -100)
            else:
                self.assertEqual(int(y[0, k]), tgt)
        self.assertEqual(n_valid, int((y != -100).sum()))
        self.assertGreater(n_valid, 0)


class TestWindowSelection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="garg_mm3_")
        p = os.path.join(self.tmp, "doc.tokens")
        tok5.write_file(p, _multimodal_items())
        self.doc = tokenset.read_doc(p)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_picked_window_is_trainable(self):
        rng = random.Random(0)
        B = 6
        for force in (True, False):
            i = window_index(self.doc, B, rng, force_last=force)
            self.assertIsNotNone(i)
            half = B // 2
            targets = self.doc["pos_ids"][i + half:i + B]
            self.assertTrue(any(t is not None for t in targets),
                            f"窗口 {i} 没有可训练目标")


class TestMultimodalTrainStep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import model as model_mod
        torch.manual_seed(0)
        cls.m = model_mod.Gargantua(n_enc_blocks=1, n_dec_blocks=1, vocab_size=_VS)
        cls.m.train()

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="garg_mm4_")
        p = os.path.join(self.tmp, "doc.tokens")
        tok5.write_file(p, _multimodal_items())
        self.doc = tokenset.read_doc(p)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_loss_backprops_into_both_projections(self):
        m = self.m
        m.zero_grad(set_to_none=True)
        i = window_index(self.doc, 8, random.Random(1), force_last=True)
        logits, y, n_valid = forward_window(m, self.doc, i, 8)
        self.assertGreater(n_valid, 0)
        loss = window_loss(logits, y)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(m.visual_proj.weight.grad)
        self.assertIsNotNone(m.audio_proj.weight.grad)
        self.assertGreater(float(m.visual_proj.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(m.audio_proj.weight.grad.abs().sum()), 0.0)

    def test_embedding_cache_reused(self):
        m = self.m
        i = window_index(self.doc, 8, random.Random(2), force_last=True)
        forward_window(m, self.doc, i, 8)
        emb1 = self.doc["_emb"][1]
        forward_window(m, self.doc, i, 8)
        self.assertIs(self.doc["_emb"][1], emb1)      # 第二次命中缓存


class TestMultimodalGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import model as model_mod
        torch.manual_seed(0)
        cls.m = model_mod.Gargantua(n_enc_blocks=1, n_dec_blocks=1, vocab_size=_VS)
        cls.m.eval()

    def test_media_prefix_and_archive_order(self):
        from infer import generate
        tmp = tempfile.mkdtemp(prefix="garg_mm5_")
        try:
            img = image_codec.image_to_step(
                Image.new("RGBA", (64, 64), (10, 20, 30, 255)))
            steps, archive = generate(self.m, "hello", max_new=2,
                                      temperature=0.9, archive_dir=tmp,
                                      media_steps=[img])
            self.assertGreaterEqual(len(steps), 0)
            self.assertIsNotNone(archive)
            items = tok5.read_file(archive)
            kinds = [e["kind"] for it in items if it["kind"] == "step"
                     for e in it["elements"]]
            # 媒体在前（模型实际看到的顺序），文本在后
            self.assertEqual(kinds[0], "visual")
            self.assertIn("text", kinds)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_media_only_prompt_rejected(self):
        from infer import generate
        img = image_codec.image_to_step(
            Image.new("RGBA", (64, 64), (0, 0, 0, 255)))
        with self.assertRaises(ValueError):
            generate(self.m, "", max_new=1, media_steps=[img])


if __name__ == "__main__":
    unittest.main()
