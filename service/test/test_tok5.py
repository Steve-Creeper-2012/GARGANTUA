# ===========================================================================
# GARGANTUA v1 — test_tok5.py：token-5 编码契约回归
# ---------------------------------------------------------------------------
# 覆盖 spec 第 5b 节的全部硬约束：
#   - 字符集恰好 27 个符号，5 bit/字符，终止符码 27
#   - {t:NNNNN} 文本 / {w:HEX} 音频（2132）/ {v:HEXx:..y:..} 视觉（10240）
#   - <NNNNN> 扩展系统 token（不在 {} 内）
#   - 结构：{} 非空、| 位置合法、括号配对、游离字符拒绝
#   - 载荷往返：音频 int16 精确、视觉 8bit→10bit→8bit 精确
#   - 文件：dumps/loads 与 BitWriter 增量写一致
#   - 与旧 Step 结构的互转（模型管线兼容层）
# 运行：cd <ROOT> && python -m unittest service.test.test_tok5 -v
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

import io
import random
import unittest

import spec
from tokens import tok5
from tokens.vocab import Vocab

# 新契约：spec 不再钉死词表长度。这些结构测试沿用全局词表长度
# （与重构前的固定默认一致），合成 ID（如 50994）也落在其内。
_VS = len(Vocab.global_vocab())


class TestAlphabet(unittest.TestCase):
    def test_alphabet_is_27_unique_symbols(self):
        self.assertEqual(len(spec.T5_CHARS), 27)
        self.assertEqual(len(set(spec.T5_CHARS)), 27)
        self.assertEqual(spec.T5_ALPHABET_SIZE, 27)
        # 没有空格、没有小写字母（除 w/t/v/x/y）
        for ch in spec.T5_CHARS:
            self.assertNotEqual(ch, " ")
        self.assertEqual({c for c in spec.T5_CHARS if c.islower()},
                         {"w", "t", "v", "x", "y"})

    def test_required_symbols_present(self):
        for ch in "0123456789{}<>|:wtvABCDEFxy":
            self.assertIn(ch, spec.T5_CHAR_CODE)


class TestBitstream(unittest.TestCase):
    def test_roundtrip_lengths(self):
        for n in (0, 1, 5, 7, 8, 9, 16, 100):
            text = ("{t:00001}" * 20)[:n]
            self.assertEqual(tok5.decode_chars(tok5.encode_chars(text)), text)

    def test_terminator_marks_end(self):
        # 同样的字符流 + 额外垃圾字节，解码必须停在终止符
        data = tok5.encode_chars("{t:00001}")
        self.assertEqual(tok5.decode_chars(data + b"\xff\xff"), "{t:00001}")

    def test_truncated_stream_rejected(self):
        data = tok5.encode_chars("{t:00001}")
        with self.assertRaises(tok5.Tok5Error):
            tok5.decode_chars(data[:-1])     # 砍掉终止符

    def test_non_alphabet_char_rejected(self):
        # 位流层只校验字符集：空格 / 小写字母 / 大写 T 都不在 27 个符号里
        for bad in ("{t:00001 }", "abc", "{T:00001}", "{}é"):
            with self.assertRaises(tok5.Tok5Error):
                tok5.encode_chars(bad)
        # 语法层：ID 是自界定十六进制——非 hex 字符（G）终止并报错；
        # hex 字母（A-F，大小写都收）是合法 ID 数字
        with self.assertRaises(tok5.Tok5Error):
            tok5.parse_items("{t:0000G}")
        self.assertEqual(tok5.parse_items("{t:0000A}")[0]["elements"][0],
                         tok5.text(0xA))

    def test_bitwidth_is_five(self):
        # 8 个字符 = 40 bit = 5 字节（+ 终止符 5 bit → 6 字节）
        self.assertEqual(len(tok5.encode_chars("0" * 8)), 6)
        # 3 个字符 = 15 bit + 5 bit 终止符 = 20 bit → 3 字节
        self.assertEqual(len(tok5.encode_chars("000")), 3)


class TestElements(unittest.TestCase):
    def test_text_id_is_self_delimiting_hex(self):
        # 2026-10-31：ID 是自界定十六进制（读到 } / | 为止），不定宽——
        # 长度因词表而异，前导零允许，大小写都收、写出统一大写
        self.assertEqual(tok5.element_to_chars(tok5.text(1)), "t:1")
        self.assertEqual(tok5.element_to_chars(tok5.text(0xC732)),
                         "t:C732")
        self.assertEqual(
            tok5.parse_items("{t:0001}")[0]["elements"][0], tok5.text(1))
        self.assertEqual(
            tok5.parse_items("{t:000001}")[0]["elements"][0], tok5.text(1))
        self.assertEqual(
            tok5.parse_items("{t:C732}")[0]["elements"][0], tok5.text(50994))
        self.assertEqual(
            tok5.parse_items("{t:c732}")[0]["elements"][0],
            tok5.text(50994))

    def test_text_id_bounds(self):
        # ID 上界 = spec.MAX_VOCAB_SIZE（RAM 防呆，非格式约束——自界定
        # 十六进制对宽度没有限制）；词表长度每个词表各自声明
        self.assertEqual(tok5.text(tok5.ID_LIMIT - 1)["id"],
                         tok5.ID_LIMIT - 1)
        with self.assertRaises(tok5.Tok5Error):
            tok5.text(tok5.ID_LIMIT)
        with self.assertRaises(tok5.Tok5Error):
            tok5.text(-1)

    def test_ext_is_stream_level(self):
        items = tok5.parse_items("<00005>")
        self.assertEqual(items, [{"kind": "ext", "id": 5}])
        # < > 不能出现在时间步内
        with self.assertRaises(tok5.Tok5Error):
            tok5.parse_items("{<00005>}")

    def test_multiple_elements_in_one_step(self):
        items = tok5.parse_items("{t:00001|t:00002}")
        self.assertEqual(len(items), 1)
        self.assertEqual([e["id"] for e in items[0]["elements"]], [1, 2])

    def test_structure_errors(self):
        bad = [
            "{}",                 # 空时间步
            "{t:00001",           # 未闭合
            "t:00001}",           # 游离 }
            "{|t:00001}",         # | 在步首
            "{t:00001|}",         # | 在步尾
            "{t:00001||t:00002}", # 连续 |
            "t:00001",            # 无 {} 包裹
            "{t:00001}t:00002",   # 第二个 token 游离
        ]
        for s in bad:
            with self.assertRaises(tok5.Tok5Error, msg=s):
                tok5.parse_items(s)


class TestWavePayload(unittest.TestCase):
    def test_hex_length_is_2132(self):
        self.assertEqual(spec.T5_WAVE_HEX, 2132)
        samples = list(range(-266, 267))            # 533 个
        h = tok5.samples_to_hex(samples)
        self.assertEqual(len(h), 2132)

    def test_roundtrip_exact(self):
        rnd = random.Random(7)
        samples = [rnd.randint(-32768, 32767) for _ in range(533)]
        self.assertEqual(tok5.hex_to_samples(tok5.samples_to_hex(samples)),
                         samples)

    def test_wrong_length_rejected(self):
        with self.assertRaises(tok5.Tok5Error):
            tok5.samples_to_hex([0] * 532)
        with self.assertRaises(tok5.Tok5Error):
            tok5.wave("AB" * 100)

    def test_float32_bytes_accepted(self):
        # audio_codec 的 meta["samples"] 是 533 个 float32 小端
        import struct
        raw = struct.pack("<533f", *([0.5] * 533))
        el = tok5.wave(raw)
        self.assertEqual(len(el["hex"]), 2132)
        got = tok5.hex_to_samples(el["hex"])
        self.assertTrue(all(abs(v - 16384) <= 1 for v in got))


class TestVisualPayload(unittest.TestCase):
    def test_hex_length_is_10240(self):
        self.assertEqual(spec.T5_PATCH_HEX, 10240)
        h = tok5.rgba8_to_hex([0] * spec.PATCH_VEC_DIM)
        self.assertEqual(len(h), 10240)
        self.assertEqual(h, "0" * 10240)

    def test_roundtrip_exact_8bit(self):
        rnd = random.Random(11)
        vals = [rnd.randint(0, 255) for _ in range(spec.PATCH_VEC_DIM)]
        self.assertEqual(tok5.hex_to_rgba8(tok5.rgba8_to_hex(vals)), vals)

    def test_extremes_exact(self):
        vals = [0, 255] * (spec.PATCH_VEC_DIM // 2)
        self.assertEqual(tok5.hex_to_rgba8(tok5.rgba8_to_hex(vals)), vals)

    def test_pixel_packing_is_row_major(self):
        """第一个像素的 RGBA 必须落在最高 40 bit。"""
        vals = [0] * spec.PATCH_VEC_DIM
        vals[0:4] = [255, 0, 0, 0]                  # 首像素 R=255 → 10bit 1023
        # 1023 = 0b1111111111 → 首 10 bit 全 1 → 首字节 0xFF
        self.assertEqual(tok5.rgba8_to_hex(vals)[:2], "FF")
        # 末像素的 A 落在最低 10 bit：0 之外的值必须改变末尾
        vals2 = [0] * spec.PATCH_VEC_DIM
        vals2[-1] = 255
        self.assertNotEqual(tok5.rgba8_to_hex(vals2)[-2:], "00")

    def test_visual_element_chars(self):
        h = "0" * spec.T5_PATCH_HEX
        el = tok5.visual(h, 3, 7)
        self.assertEqual(tok5.element_to_chars(el), "v:" + h + "x:3y:7")
        back = tok5.parse_items("{" + tok5.element_to_chars(el) + "}")
        self.assertEqual(back[0]["elements"][0], el)


class TestDisplay(unittest.TestCase):
    def test_display_uses_long_markers(self):
        items = [tok5.step(tok5.text(1)), {"kind": "ext", "id": 5}]
        self.assertEqual(tok5.to_display(items), "{text: 1}<5>")

    def test_display_roundtrip(self):
        items = [{"kind": "ext", "id": 1},
                 tok5.step(tok5.text(51046), tok5.text(12419)),
                 {"kind": "ext", "id": 2}]
        txt = tok5.to_display(items)
        self.assertEqual(tok5.parse_display(txt), items)

    def test_display_accepts_compact_form(self):
        self.assertEqual(tok5.parse_display("{t:00001}"),
                         [tok5.step(tok5.text(1))])

    def test_display_resolves_names(self):
        names = {1: "<user>", 2: "<user/>"}
        items = [{"kind": "ext", "id": 1}, {"kind": "ext", "id": 2}]
        txt = tok5.to_display(items, name_of=lambda i: names.get(i))
        self.assertEqual(txt, "<user><user/>")
        ids = {"<user>": 1, "<user/>": 2}
        self.assertEqual(tok5.parse_display(txt, id_of=lambda n: ids.get(n)),
                         items)


class TestFile(unittest.TestCase):
    def test_dumps_loads_roundtrip(self):
        items = [{"kind": "ext", "id": 5},
                 tok5.step(tok5.text(1)),
                 tok5.step(tok5.text(2), tok5.text(3))]
        self.assertEqual(tok5.loads(tok5.dumps(items)), items)

    def test_bitwriter_matches_dumps(self):
        items = [tok5.step(tok5.text(i)) for i in range(1, 40)]
        buf = io.BytesIO()
        w = tok5.BitWriter(buf)
        for it in items:
            w.write_chars(tok5.items_to_chars([it]))
        w.close()
        self.assertEqual(buf.getvalue(), tok5.dumps(items))

    def test_write_read_file(self):
        import tempfile
        import os
        items = [tok5.step(tok5.text(42))]
        d = tempfile.mkdtemp(prefix="garg_tok5_")
        try:
            p = os.path.join(d, "a.tokens")
            tok5.write_file(p, items)
            self.assertEqual(tok5.read_file(p), items)
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


class TestStepBridge(unittest.TestCase):
    def test_text_steps_roundtrip(self):
        steps = [{"token_ids": [51046], "type": spec.TYPE_USER,
                  "modality": "text", "meta": {}}]
        items = tok5.from_steps(steps)
        self.assertEqual(items, [tok5.step(tok5.text(51046))])
        back = tok5.to_steps(items)
        self.assertEqual(back[0]["token_ids"], [51046])
        self.assertEqual(back[0]["modality"], "text")

    def test_invisible_id_becomes_ext(self):
        steps = [{"token_ids": [spec.STOP_ID], "type": spec.TYPE_NONE,
                  "modality": "text", "meta": {}}]
        self.assertEqual(tok5.from_steps(steps),
                         [{"kind": "ext", "id": spec.STOP_ID}])

    def test_visual_step_roundtrip(self):
        vec = bytes([7] * spec.PATCH_VEC_DIM)
        steps = [{"token_ids": [], "type": spec.TYPE_USER,
                  "modality": "visual",
                  "meta": {"patches": [{"x": 0, "y": 0, "vec": vec}]}}]
        items = tok5.from_steps(steps)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["elements"][0]["kind"], "visual")
        back = tok5.to_steps(items)
        self.assertEqual(back[0]["modality"], "visual")
        self.assertEqual(back[0]["meta"]["patches"][0]["vec"], vec)

    def test_short_audio_tail_dropped(self):
        """末尾不足 533 采样的残段整步丢弃（每 token 必须恰好 533）。"""
        steps = [
            {"token_ids": [], "type": spec.TYPE_USER, "modality": "audio",
             "meta": {"samples": b"\x00\x00\x00\x00" * 533, "n": 533}},
            {"token_ids": [], "type": spec.TYPE_USER, "modality": "audio",
             "meta": {"samples": b"\x00\x00\x00\x00" * 533, "n": 17}},
        ]
        items = tok5.from_steps(steps)
        self.assertEqual(len(items), 1)          # 残段被丢

    def test_mixed_visual_audio_step(self):
        vec = bytes([1] * spec.PATCH_VEC_DIM)
        steps = [{"token_ids": [], "type": spec.TYPE_USER,
                  "modality": "mixed",
                  "meta": {"patches": [{"x": 0, "y": 0, "vec": vec}],
                           "samples": b"\x00\x00\x00\x00" * 533,
                           "n": 533}}]
        items = tok5.from_steps(steps)
        kinds = [e["kind"] for e in items[0]["elements"]]
        self.assertEqual(kinds, ["visual", "wave"])

    def test_text_mixed_with_modal_rejected(self):
        items = [tok5.step(tok5.text(1), tok5.visual("0" * 10240, 0, 0))]
        with self.assertRaises(tok5.Tok5Error):
            tok5.to_steps(items)


class TestProjectionBridge(unittest.TestCase):
    """token-5 载荷 → 模态投影 → 统一 d_model 的边界契约。

    音频/视觉 token 没有词表 ID，靠两个纯线性投影进 d_model：
        visual_proj: Linear(PATCH_VEC_DIM=4096, d_model)
        audio_proj : Linear(533, d_model)
    这条测试锁住「都是同一个 d_model、都是单层 Linear、载荷能接上」。
    """

    def test_both_projections_are_single_linear_to_d_model(self):
        import torch.nn as nn
        import model as model_mod
        m = model_mod.Gargantua(n_enc_blocks=1, n_dec_blocks=1, vocab_size=_VS)
        for name, in_dim in (("visual_proj", spec.PATCH_VEC_DIM),
                             ("audio_proj", spec.AUDIO_SAMPLES_PER_TOKEN)):
            lin = getattr(m, name)
            self.assertIsInstance(lin, nn.Linear, f"{name} 必须是单层 Linear")
            self.assertEqual(lin.in_features, in_dim)
            self.assertEqual(lin.out_features, spec.D_MODEL,
                             f"{name} 必须投影到统一的 d_model")
            self.assertIsNone(lin.bias)

    def test_visual_and_audio_share_the_same_d_model(self):
        import model as model_mod
        m = model_mod.Gargantua(n_enc_blocks=1, n_dec_blocks=1, vocab_size=_VS)
        self.assertEqual(m.visual_proj.out_features,
                         m.audio_proj.out_features)
        self.assertEqual(m.visual_proj.out_features, m.wte.embedding_dim)

    def test_payload_reaches_projection_with_expected_shape(self):
        import torch
        import model as model_mod
        m = model_mod.Gargantua(n_enc_blocks=1, n_dec_blocks=1, vocab_size=_VS)
        items = [
            tok5.step(tok5.visual([7] * spec.PATCH_VEC_DIM, 0, 0),
                      tok5.visual([9] * spec.PATCH_VEC_DIM, 1, 0)),
            tok5.step(tok5.wave([100] * spec.AUDIO_SAMPLES_PER_TOKEN)),
            tok5.step(tok5.text(50994)),
        ]
        steps = tok5.to_steps(tok5.loads(tok5.dumps(items)))
        emb, sids = m.assemble_embeddings(steps)
        # 2 patch + 1 audio + 1 text = 4 个位置，全部 d_model 维
        self.assertEqual(tuple(emb.shape), (1, 4, spec.D_MODEL))
        self.assertEqual(sids.tolist(), [[0, 0, 1, 2]])   # 同帧 patch 共享步号
        self.assertTrue(torch.isfinite(emb).all())

    def test_mixed_frame_keeps_patch_then_audio_order(self):
        import model as model_mod
        m = model_mod.Gargantua(n_enc_blocks=1, n_dec_blocks=1, vocab_size=_VS)
        items = [tok5.step(
            tok5.visual([1] * spec.PATCH_VEC_DIM, 0, 0),
            tok5.wave([2] * spec.AUDIO_SAMPLES_PER_TOKEN))]
        steps = tok5.to_steps(items)
        self.assertEqual(steps[0]["modality"], "mixed")
        emb, sids = m.assemble_embeddings(steps)
        self.assertEqual(emb.shape[1], 2)
        self.assertEqual(sids.tolist(), [[0, 0]])

    def test_multimodal_forward_runs(self):
        """多模态教师强制前向可跑通（logits 落在词表维度）。"""
        import model as model_mod
        m = model_mod.Gargantua(n_enc_blocks=1, n_dec_blocks=1, vocab_size=_VS)
        items = [tok5.step(tok5.visual([3] * spec.PATCH_VEC_DIM, 0, 0)),
                 tok5.step(tok5.text(50994))]
        steps = tok5.to_steps(items)
        emb, sids = m.assemble_embeddings(steps)
        logits = m.forward_train_embeddings(emb, sids, emb, sids)
        self.assertEqual(tuple(logits.shape),
                         (1, 2, m.n_vocab))


if __name__ == "__main__":
    unittest.main()
