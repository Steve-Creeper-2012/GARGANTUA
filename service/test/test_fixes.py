# ===========================================================================
# GARGANTUA v1 — tests/test_fixes.py：2026-10-20 框架审查修复的回归测试
# ---------------------------------------------------------------------------
# 覆盖当日修复的全部 P0/P1（及可固化的 P2）：
#   P0-1  train/infer/auto_train 硬编码 Gargantua() → 非默认 block 数
#         模型加载崩溃（common.build_model 统一入口）
#   P0-2  infer.py fast_weight.reset() 在 encode 之后 → encoder 写入的
#         状态被清（违反 spec「状态跨调用推进」）
#   P1-3  奇数 block_size → s_dec 与 dec_in 长度不一致、前向必崩
#   P1-1  forward_train 不重置 fast weight → 跨样本状态泄漏
#   P1-2  压缩组 step_id 取首步 → 组内后续步 token 泄漏给更早查询
#   P1-4  build_views 的 CSA 视图与推理路径不一致（512<n≤1024）
#   P1-6  build_views 压缩区无界增长
#   P1-5  causal MLA 缺 step_ids 静默跳掩码 → 未来泄漏
#   P1-m  assemble_embeddings step_id 撞名（多段拼接）
#   P2-5  mark_primitive_token 条目不足 9 时负索引错打包
#   P2-8  \t\n\r 控制字符入词表（text_codec roundtrip）
#   P2-10 waveform_to_steps float 输入截断
#   P2-m  midi running status / 孤儿 velocity
#   P2-r  render_rgba 向量化（与参考实现逐字节一致）
#   P2-t  Trainer 参数校验
# 运行：cd <ROOT> && <torch解释器> -m unittest tests.test_fixes -v
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
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest


def _has_pywebview() -> bool:
    """本机是否装了可选窗口依赖 pywebview（缺它只跳过窗口 smoke，不算失败）。"""
    try:
        import webview  # noqa: F401
        return True
    except Exception:
        return False

import numpy as np
import torch

import spec
import model
from tokens.vocab import Vocab

# 新契约：spec 不再钉死词表长度，显式取全局词表长度
_VS = len(Vocab.global_vocab())
from common import (build_model, make_batch, save_checkpoint,
                    valid_model_name, validate_block_size, validate_steps)

# 进程级小模型单例（1+1 blocks，CPU ~30s，多个测试类共享）
_SMALL = None


def small_model():
    global _SMALL
    if _SMALL is None:
        torch.manual_seed(0)
        _SMALL = model.Gargantua(n_enc_blocks=1, n_dec_blocks=1,
                                 vocab_size=_VS)
        _SMALL.eval()
    return _SMALL


def tearDownModule():
    # 释放 1GB 小模型：否则其后的 test_model（多个 2.1GB 全尺寸模型）
    # 会把内存压爆（2026-10-20 实测 OOM 杀进程）
    global _SMALL
    _SMALL = None


class TestCheckpointContract(unittest.TestCase):
    """ P0-1：模型目录契约（config.json 驱动 block 数）。 """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="garg_fix_")
        cls.md = os.path.join(cls.tmp, "models", "small")
        m = small_model()
        save_checkpoint(cls.md, m, {"n_enc_blocks": 1, "n_dec_blocks": 1})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_build_model_reads_config_blocks(self):
        m2 = build_model(self.md)
        self.assertEqual(len(m2.enc_blocks), 1)
        self.assertEqual(len(m2.dec_blocks), 1)
        for (n, p), (n2, p2) in zip(small_model().state_dict().items(),
                                    m2.state_dict().items()):
            self.assertTrue(torch.equal(p, p2), f"参数不一致: {n}")

    def test_build_model_bad_config_falls_back(self):
        # 只有 config.json、无权重：坏 block 数应容错为默认值且不抛异常
        # （新契约下词表长度没有全局默认，目录必须自己声明 vocab_size）
        d = os.path.join(self.tmp, "models", "badcfg")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"n_enc_blocks": "abc", "n_dec_blocks": 99,
                       "vocab_size": _VS}, f)
        m3 = build_model(d)          # 不抛异常，回退默认值
        self.assertEqual(len(m3.enc_blocks), spec.N_ENC_BLOCKS)
        self.assertEqual(len(m3.dec_blocks), spec.N_DEC_BLOCKS)

    def test_valid_model_name(self):
        self.assertIsNone(valid_model_name("alpha-1"))
        for bad in ("", "..", ".", "a/b", "a\\b", 'a"b', "a<b", "a:b",
                    "a|b", "a?b", "a*b", "NUL", "con", " x "):
            self.assertIsNotNone(valid_model_name(bad), f"应拒绝: {bad!r}")

    def test_save_checkpoint_atomic_and_complete(self):
        # 2026-10-27 M1：manifest 化发布——config.json 指向的权重文件必须
        # 存在且无残留 tmp（权重文件名带 uuid，不再固定 model.safetensors）
        from common import resolve_weights_path
        wp = resolve_weights_path(self.md)
        self.assertTrue(os.path.exists(wp), f"manifest 指向的权重缺失: {wp}")
        self.assertTrue(wp.endswith(".safetensors"))
        self.assertTrue(os.path.exists(os.path.join(self.md, "config.json")))
        self.assertFalse(
            os.path.exists(os.path.join(self.md, "model.safetensors.tmp")))
        import glob as _glob
        self.assertFalse(_glob.glob(os.path.join(self.md, "*.tmp")),
                         "save_checkpoint 不得留下 .tmp 半成品")


class TestBlockSizeContract(unittest.TestCase):
    """ P1-3：block_size 偶数约束 + make_batch 形状一致性。 """

    def test_odd_block_size_rejected(self):
        for b in (3, 15, 31):
            with self.assertRaises(ValueError):
                validate_block_size(b)
        with self.assertRaises(ValueError):
            validate_block_size(0)

    def test_make_batch_shapes(self):
        ids = list(range(200, 240))
        for B in (2, 16, 32):
            half = B // 2
            x, si, di, sd, y = make_batch(ids, 3, B)
            self.assertEqual(x.shape[1], half)
            self.assertEqual(di.shape[1], half)
            self.assertEqual(sd.shape[1], half)     # 旧实现奇数 B 时这里不等
            self.assertEqual(y.shape[1], half)
            self.assertEqual(si[0].tolist(), list(range(half)))
            self.assertEqual(sd[0].tolist(), list(range(half, half + half)))
            # decoder 输入 = 目标左移一位，首 token 用 user 段尾播种
            self.assertEqual(di[0, 0].item(), ids[3 + half - 1])

    def test_steps_validation(self):
        with self.assertRaises(ValueError):
            validate_steps(0)
        self.assertEqual(validate_steps(50), 50)


class TestFastWeightContract(unittest.TestCase):
    """ P0-2 + P1-1：fast weight 状态语义。 """

    def setUp(self):
        self.m = small_model()

    def test_forward_train_resets_state(self):
        torch.manual_seed(3)
        ids = torch.randint(64, 70000, (1, 6))
        si = torch.arange(6).unsqueeze(0)
        dt = torch.randint(64, 70000, (1, 4))
        sd = torch.arange(6, 10).unsqueeze(0)
        l1 = self.m.forward_train(ids, si, dt, sd)
        l2 = self.m.forward_train(ids, si, dt, sd)
        # 连续两次前向必须逐位一致（状态不跨样本泄漏）
        self.assertTrue(torch.allclose(l1, l2, atol=1e-6),
                        f"跨样本状态泄漏: max diff "
                        f"{(l1 - l2).abs().max().item():.4f}")

    def test_generate_resets_before_encode(self):
        from infer import generate
        order, states = [], []
        orig_reset = model.FastWeightBank.reset
        orig_fw = model.FastWeightBank.forward

        def reset_spy(self):
            order.append("reset")
            return orig_reset(self)

        def fw_spy(self, x, **kw):
            order.append("fw")
            # 2026-10-27 S2：状态可存内部 self._state（训练路径）或外部
            # state dict（增量推理路径）；两者任一非空即「入口可见状态」
            ext = kw.get("state")
            visible = (self._state is not None
                       or (ext is not None and ext.get("fw") is not None))
            states.append(visible)
            return orig_fw(self, x, **kw)

        model.FastWeightBank.reset = reset_spy
        model.FastWeightBank.forward = fw_spy
        try:
            torch.manual_seed(11)
            with tempfile.TemporaryDirectory() as ad:
                generate(self.m, "你好", max_new=2, archive_dir=ad)
        finally:
            model.FastWeightBank.reset = orig_reset
            model.FastWeightBank.forward = orig_fw
        # 旧 bug 的顺序是 [fw(encode), reset, fw(decode)...]；
        # 正确顺序：reset 必须先于第一次 forward（encode）
        self.assertEqual(order[0], "reset", f"reset 未在 encode 前执行: {order}")
        self.assertEqual(order[1], "fw")
        if len(states) >= 2:
            self.assertFalse(states[0], "encode 时的入口状态应为空")
            self.assertTrue(states[1],
                            "decode 必须看到 encoder 写入的 fast weight 状态")

    def test_generate_kv_matches_training_layout(self):
        """ P1-1（review-pipeline 实锤）：KV 回写必须与 forward_train 逐位置
        同构——训练 KV 流 = enc(prompt) + [emb(seed)@n, emb(t0)@n+1, …]，
        即每个 decoder 输入的 embedding 放在其预测步上。 """
        from infer import generate
        from tokens.text_codec import encode as text_encode
        views = []
        orig = model.Gargantua.build_views

        def spy(self, vecs, steps):
            views.append((vecs.clone(), steps[0].tolist()))
            return orig(self, vecs, steps)

        model.Gargantua.build_views = spy
        try:
            torch.manual_seed(12)
            with tempfile.TemporaryDirectory() as ad:
                generate(self.m, "你好世界", max_new=2, archive_dir=ad)
        finally:
            model.Gargantua.build_views = orig
        n = 4                                    # prompt token 数
        # 第 j 次构视图（j 从 0）：KV = prompt@0..n-1 + 前 j+1 个输入 emb@n..
        for j, (vecs, ss) in enumerate(views):
            self.assertEqual(ss, list(range(n + 1 + j)),
                             f"第 {j} 次视图的 step 布局不符训练同构")
        # 第一次视图的最后一项必须是 seed（prompt 尾 token）的 MODEL 嵌入
        seed_vec = self.m.embed_text(
            torch.tensor([[text_encode("你好世界")[-1]["token_ids"][0]]]),
            torch.tensor([[spec.TYPE_MODEL]]))
        self.assertTrue(torch.allclose(views[0][0][0, -1], seed_vec[0, 0],
                                       atol=1e-6),
                        "首次视图缺少 seed 的 MODEL 嵌入（train/infer 偏差）")
        # step 序列严格递增、无重复（旧实现每步重复拼接当前输入）
        for _, ss in views:
            self.assertEqual(ss, sorted(set(ss)))


class TestCompressionCausality(unittest.TestCase):
    """ P1-2 / P1-4 / P1-6：压缩视图的因果与容量。 """

    def setUp(self):
        self.m = small_model()

    def test_compressed_entry_uses_max_step(self):
        kv = model.SharedKV(self.m.csa_compressor, self.m.hca_compressor,
                            self.m.draw_packer)
        # 4 个 token、两个步：step0×2 + step1×2，ratio=4 → 1 个跨步组
        for sid in (0, 0, 1, 1):
            kv.append(torch.randn(1, spec.D_MODEL), step_id=sid,
                      type_id=0, source="enc")
        # 2026-10-23：_compress_entries 改收压缩器模块（CSA/HCA 比例
        # 可配后不能再用整数 ratio 反猜模块）；整数形式仍兼容
        out = kv._compress_entries(kv.entries[:], kv.csa)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].step_id, 1,
                         "压缩组 step_id 必须取组内最大步"
                         "（否则组内后续步会泄漏给更早查询）")

    def test_csa_view_matches_inference_semantics(self):
        torch.manual_seed(4)
        n = 600                       # ≤ ctx_full：无压缩区
        vecs = torch.randn(1, n, spec.D_MODEL)
        steps = torch.arange(n).unsqueeze(0)
        kv_hca, kv_csa = self.m.build_views(vecs, steps)
        # 2026-10-25：CSA 的保留区就是完整 KV 区（无局部窗），
        # 未超完整区时两个视图都是全量
        self.assertEqual(kv_csa[0].shape[1], n)
        self.assertEqual(kv_hca[0].shape[1], n)

    def test_build_views_equiv_sharedkv_long(self):
        """ 与推理路径 SharedKV 在 n=1100 时逐字节一致（审查探针 [8]）。
        注意：两侧必须使用【相同 step 结构】（本测试 1 token/step）。 """
        torch.manual_seed(5)
        n = 1100
        vecs = torch.randn(1, n, spec.D_MODEL)
        steps = torch.arange(n).unsqueeze(0)
        kv_hca, kv_csa = self.m.build_views(vecs, steps)

        kv = model.SharedKV(self.m.csa_compressor, self.m.hca_compressor,
                            self.m.draw_packer)
        for i in range(n):
            kv.append(vecs[0, i:i + 1].clone(), step_id=i,
                      type_id=0, source="enc")
        vh = kv.view("hca")
        vc = kv.view("csa")
        self.assertEqual(kv_hca[0].shape, vh[0].shape)
        self.assertEqual(kv_csa[0].shape, vc[0].shape)
        self.assertTrue(torch.allclose(kv_hca[0], vh[0], atol=1e-5))
        self.assertTrue(torch.allclose(kv_csa[0], vc[0], atol=1e-5))
        # 2026-10-27（S1）：视图同时携带步内序号 pos，两侧必须一致
        self.assertTrue(torch.equal(kv_hca[2], vh[2]))
        self.assertTrue(torch.equal(kv_csa[2], vc[2]))

    def test_comp_region_capped(self):
        """ P1-6：压缩区截断到 CTX_COMPRESS，注意力规模不再无界。 """
        torch.manual_seed(6)
        n = spec.CTX_FULL + spec.CTX_COMPRESS + 600
        vecs = torch.randn(1, n, spec.D_MODEL)
        steps = torch.arange(n).unsqueeze(0)
        kv_hca, _ = self.m.build_views(vecs, steps)
        bound = (spec.CTX_FULL
                 + math.ceil(spec.CTX_COMPRESS / spec.HCA_RATIO) + 200)
        self.assertLessEqual(kv_hca[0].shape[1], bound)


class TestCausalMaskContract(unittest.TestCase):
    """ P1-5：causal MLA 缺 step_ids 必须显式报错。 """

    def test_causal_requires_step_ids(self):
        mla = model.MLAAttention(causal=True)
        x = torch.randn(1, 2, spec.D_MODEL)
        kv = torch.randn(1, 4, spec.D_MODEL)
        with self.assertRaises(ValueError):
            mla(x, kv)                          # 两个都缺
        with self.assertRaises(ValueError):
            mla(x, kv, kv_step_ids=torch.arange(4).unsqueeze(0))  # 缺 x 侧
        # encoder 双向路径不受影响
        mla_b = model.MLAAttention(causal=False)
        out = mla_b(x, kv)
        self.assertEqual(out.shape, (1, 2, spec.D_MODEL))


class TestAssembleContract(unittest.TestCase):
    """ P1-m：assemble_embeddings 的 step_id 撞名与输入校验。 """

    def setUp(self):
        self.m = small_model()

    def _frame_step(self, color):
        from tokens.image_codec import image_to_step
        from PIL import Image
        return image_to_step(Image.new("RGBA", (32, 32), color))

    def test_base_step_prevents_collision(self):
        s1 = [self._frame_step((10, 20, 30, 255))]
        s2 = [self._frame_step((200, 100, 50, 255))]
        e1, i1 = self.m.assemble_embeddings(s1, base_step=0)
        e2, i2 = self.m.assemble_embeddings(s2, base_step=1)
        self.assertEqual(i1[0].tolist(), [0])
        self.assertEqual(i2[0].tolist(), [1])     # 不从 0 重开
        all_ids = i1[0].tolist() + i2[0].tolist()
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_no_cross_frame_leak_with_base_step(self):
        """ 两帧拼接后，第一帧的查询不得看到第二帧内容（causal 侧）。 """
        torch.manual_seed(8)
        s1 = [self._frame_step((10, 20, 30, 255))]
        s2 = [self._frame_step((200, 100, 50, 255))]
        e1, i1 = self.m.assemble_embeddings(s1, base_step=0)
        e2, i2 = self.m.assemble_embeddings(s2, base_step=1)
        emb = torch.cat([e1, e2], 1)
        sids = torch.cat([i1, i2], 1)
        kv_hca, _ = self.m.build_views(emb, sids)
        mla = model.MLAAttention(causal=True)      # causal 侧才有掩码
        q = emb[:, :1]                            # 第一帧的查询（step 0）
        qs = torch.tensor([[0]])
        # 2026-10-27（S1）：causal 掩码已细化到 token 粒度，必须同时给
        # 步内序号；查询是帧内第一个 token → pos 0
        qpos = torch.tensor([[0]])
        out1 = mla(q, kv_hca[0], kv_hca[1], qs,
                   kv_pos=kv_hca[2], x_pos=qpos)
        emb2 = emb.clone()
        emb2[0, 1:] = torch.randn_like(emb2[0, 1:])   # 篡改第二帧
        kv_hca2, _ = self.m.build_views(emb2, sids)
        out2 = mla(q, kv_hca2[0], kv_hca2[1], qs,
                   kv_pos=kv_hca2[2], x_pos=qpos)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5),
                        "第一帧查询看到了第二帧内容（step_id 撞名泄漏）")

    def test_bad_patch_vec_rejected(self):
        st = {"token_ids": [], "type": 0, "modality": "visual",
              "meta": {"patches": [{"x": 0, "y": 0, "vec": b"\x00" * 16}]}}
        with self.assertRaises(ValueError):
            self.m.assemble_embeddings([st])

    def test_bad_audio_len_rejected(self):
        st = {"token_ids": [], "type": 0, "modality": "audio",
              "meta": {"samples": b"\x00" * 8}}
        with self.assertRaises(ValueError):
            self.m.assemble_embeddings([st])


class TestPrimitivePackingContract(unittest.TestCase):
    """ P2-5：mark_primitive_token 条目不足 9 的守卫。 """

    def setUp(self):
        self.m = small_model()

    def _mk(self):
        return model.SharedKV(self.m.csa_compressor, self.m.hca_compressor,
                              self.m.draw_packer)

    def test_fewer_than_nine_entries_no_crash(self):
        kv = self._mk()
        for i in range(3):
            kv.append(torch.randn(1, spec.D_MODEL), step_id=i,
                      type_id=1, source="dec")
        # 只有 3 条却标记 9 次：旧实现用负索引把首尾条目错打包成 1 条
        # （entries[-6] 与 entries[2] 重复）→ 现在必须原样保留
        for _ in range(9):
            kv.mark_primitive_token()
        self.assertEqual(len(kv.entries), 3)
        self.assertEqual(kv._pending_draw, [])

    def test_nine_entries_packed(self):
        kv = self._mk()
        for i in range(9):
            kv.append(torch.randn(1, spec.D_MODEL), step_id=i,
                      type_id=1, source="dec")
            kv.mark_primitive_token()
        self.assertEqual(len(kv.entries), 1)


class TestTextCodecControlChars(unittest.TestCase):
    """ P2-8：\t\n\r 入词表（多行文本不再丢换行）。 """

    def test_control_chars_roundtrip(self):
        from tokens.text_codec import encode, decode
        for text in ("a\nb", "第一行\n第二行", "a\tb\tc", "x\r\ny",
                     "col1\tcol2\n1\t2\n"):
            steps = encode(text)
            ids = [t for st in steps for t in st["token_ids"]]
            out = decode([{"token_ids": [i], "type": 0, "modality": "text",
                           "meta": {}} for i in ids])
            self.assertEqual(out, text, f"roundtrip 丢失: {text!r}")

    def test_no_skipped_for_common_text(self):
        from tokens.text_codec import encode
        steps = encode("hello 世界\n第二行\ttab")
        skipped = [c for st in steps
                   for c in st.get("meta", {}).get("skipped", [])]
        self.assertEqual(skipped, [])


class TestAudioCodecFloatInput(unittest.TestCase):
    """ P2-10：waveform_to_steps 浮点输入不得截断成 0。 """

    def test_float_input_scaled(self):
        from tokens.audio_codec import waveform_to_steps, steps_to_waveform
        x = np.array([0.5, -0.5, 0.25, -0.25] * 20, dtype=np.float32)
        steps = waveform_to_steps(x)
        back = steps_to_waveform(steps)
        # 0.5 → 约 ±0.5（旧实现 asarray(int16) 会得到全 0）
        self.assertAlmostEqual(float(np.abs(back).max()), 0.5, places=1)


class TestRenderRgbaVectorized(unittest.TestCase):
    """ P2-r：render_rgba 向量化与参考实现逐字节一致（仓库 schema：
    shape_type -1/0/1、theta 弧度、w/h 全尺寸）。 """

    def _reference(self, prims, w, h):
        from tokens.draw_codec import clamp_primitive, _rgba8
        buf = bytearray(w * h * 4)
        for prim in prims:
            p = clamp_primitive(prim)
            px = _rgba8(p)
            if p.shape_type == -1:
                buf[:] = px * (w * h)
            elif p.shape_type in (0, 1):
                hw, hl = p.w / 2.0, p.h / 2.0
                if hw <= 0 or hl <= 0:
                    continue
                ct, st_ = math.cos(p.theta), math.sin(p.theta)
                ext = math.hypot(hw, hl)
                for y in range(max(0, int(p.cy - ext)),
                               min(h - 1, int(p.cy + ext)) + 1):
                    for x in range(max(0, int(p.cx - ext)),
                                   min(w - 1, int(p.cx + ext)) + 1):
                        u = (x - p.cx) * ct + (y - p.cy) * st_
                        v = -(x - p.cx) * st_ + (y - p.cy) * ct
                        if p.shape_type == 0:
                            ok = abs(u) <= hw and abs(v) <= hl
                        else:
                            ok = (u * u / (hw * hw)
                                  + v * v / (hl * hl) <= 1)
                        if ok:
                            buf[(y * w + x) * 4:(y * w + x) * 4 + 4] = px
        return bytes(buf)

    def test_byte_identical(self):
        from tokens.draw_codec import Primitive, render_rgba
        W, H = 160, 100
        prims = [
            Primitive(0, 0, 0, 0, -1, 0.0, 64, 64, 96),
            Primitive(80, 50, 60, 40, 0, math.radians(17), 230, 76, 51),
            Primitive(40, 30, 30, 20, 1, 0.0, 128, 128, 230),
            Primitive(-5, -5, 30, 30, 0, math.radians(83), 1, 2, 3),
        ]
        self.assertEqual(bytes(render_rgba(prims, W, H)),
                         self._reference(prims, W, H))

    def test_fast_enough(self):
        import time
        from tokens.draw_codec import Primitive, render_rgba
        prims = [Primitive(100 + i, 60, 30, 20, 1, math.radians(i),
                           128, 128, 128)
                 for i in range(20)]
        t0 = time.time()
        render_rgba(prims, 320, 200)
        self.assertLess(time.time() - t0, 2.0)


class TestTrainerValidation(unittest.TestCase):
    """ P2-t：Trainer 参数校验。 """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="garg_tr_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _trainer(self, **kw):
        from auto_train import Trainer
        return Trainer(get_model=lambda: None, save_fn=lambda m: None,
                       lock=threading.RLock(),
                       queue_dir=os.path.join(self.tmp, "queue"),
                       done_dir=os.path.join(self.tmp, "done"),
                       log_path=os.path.join(self.tmp, "log.jsonl"),
                       **kw)

    def test_rejects_bad_block_size(self):
        with self.assertRaises(ValueError):
            self._trainer(block_size=15)

    def test_rejects_zero_steps(self):
        with self.assertRaises(ValueError):
            self._trainer(steps_per_file=0)

    def test_setters_validate(self):
        t = self._trainer()
        with self.assertRaises(ValueError):
            t.block_size = 7
        with self.assertRaises(ValueError):
            t.steps_per_file = -1
        with self.assertRaises(ValueError):
            t.optimizer = "sgd"
        t.block_size = 32
        self.assertEqual(t.block_size, 32)


class TestMuonNewtonSchulz(unittest.TestCase):
    """ P2-1（review-pipeline 验证）：newton_schulz5 正交化正确性。 """

    def test_approximately_semi_orthogonal(self):
        from muon import newton_schulz5
        torch.manual_seed(0)
        for shape in [(64, 64), (32, 128), (128, 32)]:
            G = torch.randn(*shape)
            X = newton_schulz5(G)
            self.assertEqual(X.shape, G.shape)
            # 5 次 NS 迭代对随机高斯输入误差 ~0.4（病态矩阵收敛有限，属
            # 正常行为）；断言「比归一化输入更接近半正交」这一不变性
            m = min(shape)
            prod = (X @ X.T) if shape[0] <= shape[1] else (X.T @ X)
            err = (prod - torch.eye(m)).abs().max().item()
            Gn = G / (G.norm() + 1e-7)
            prod_g = (Gn @ Gn.T) if shape[0] <= shape[1] else (Gn.T @ Gn)
            err_g = (prod_g - torch.eye(m)).abs().max().item()
            self.assertLess(err, err_g, f"{shape}: NS5 未改善正交性")
            # 奇异值均值应接近 1（半正交矩阵的 sv 全为 1）
            sv = torch.linalg.svdvals(X)
            self.assertGreater(sv.mean().item(), 0.6)
            self.assertLess(sv.max().item(), 1.6)

    def test_zero_grad_safe(self):
        from muon import newton_schulz5
        X = newton_schulz5(torch.zeros(8, 8))
        self.assertTrue(torch.isfinite(X).all())


class TestInferCliCustomBlocks(unittest.TestCase):
    """ P0-1 端到端：CLI 推理可加载非默认 block 数的模型。 """

    def test_cli_infer_with_custom_blocks(self):
        import subprocess
        import sys
        tmp = tempfile.mkdtemp(prefix="garg_cli_")
        try:
            md = os.path.join(tmp, "models", "small")
            arch = os.path.join(tmp, "arch")
            m = small_model()
            save_checkpoint(md, m, {"n_enc_blocks": 1, "n_dec_blocks": 1})
            r = subprocess.run(
                [sys.executable,
                 os.path.join(boot.project_root(), "service", "infer",
                              "infer.py"),
                 "--model", md,
                 "--prompt", "你好", "--max-new-tokens", "1",
                 "--archive-dir", arch],
                cwd=boot.project_root(), capture_output=True, text=True,
                timeout=600)
            self.assertEqual(r.returncode, 0,
                             f"CLI 推理失败:\n{r.stderr[-800:]}")
            # 2026-10-25：推理侧 id 存档（.tokens + .json 元数据）
            names = os.listdir(arch)
            self.assertTrue(any(n.endswith(".tokens") for n in names), names)
            self.assertTrue(any(n.endswith(".json") for n in names), names)
            self.assertIn("id 存档", r.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestInferIdArchive(unittest.TestCase):
    """ 推理侧 id 存档：KV 可丢，原始 token id 必须落盘且可读回。 """

    def test_archive_roundtrip_and_metadata(self):
        import json as _json
        from infer import IdArchive
        from tokens import tok5
        tmp = tempfile.mkdtemp(prefix="garg_arch_")
        try:
            a = IdArchive("models/demo", tmp)
            # 真实推理的 ID 都落在可见区（>= HEAD_SIZE）
            a.add([51046, 51047])
            a.add([51048])
            self.assertTrue(a.path is not None)
            a.close(n_prompt=2, n_generated=1, ctx_total=2)
            items = tok5.read_file(a.path)        # 必须是合法 token-5 流
            steps = [it for it in items if it["kind"] == "step"]
            self.assertEqual([e["id"] for st in steps
                              for e in st["elements"]],
                             [51046, 51047, 51048])
            meta = _json.loads(
                a.path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(meta["n_tokens"], 3)
            self.assertEqual(meta["n_prompt_tokens"], 2)
            self.assertEqual(meta["n_generated_tokens"], 1)
            self.assertEqual(meta["ctx_total"], 2)
            self.assertEqual(meta["n_evicted"], 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_archive_marks_system_tokens(self):
        """不可见区的 ID 必须写成扩展系统 token <NNNNN>，不是 {t:}。"""
        from infer import IdArchive
        from tokens import tok5
        tmp = tempfile.mkdtemp(prefix="garg_arch2_")
        try:
            a = IdArchive("models/demo", tmp)
            a.add([spec.STOP_ID])
            a.close(n_prompt=0, n_generated=1)
            self.assertEqual(tok5.read_file(a.path),
                             [{"kind": "ext", "id": spec.STOP_ID}])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_archive_failure_does_not_raise(self):
        from infer import IdArchive
        # 目录不可写 → 静默降级（推理不能因为存档挂掉）
        a = IdArchive("m", "/dev/null/not-a-dir")
        a.add([1, 2, 3])
        a.close(n_prompt=3, n_generated=0, ctx_total=8)
        self.assertIsNone(a.rel_path())


class TestGroupByStepContract(unittest.TestCase):
    """ group_by_step：向量化重写后曾出现「产出空组 + 指针不前进」的
    无限循环（内存暴涨直到被 kill）。这里做全覆盖不变量检查 + 与参考
    实现随机对拍。2026-10-25 起语义 = 「组内 token 数向 ratio 靠拢，
    误差最小（允许略超）」，不再要求 ≤ ratio。 """

    def _invariants(self, ss, ratio):
        g = model.group_by_step(torch.tensor(ss), ratio)
        self.assertEqual(g[0][0], 0)
        self.assertEqual(g[-1][1], len(ss))
        for (s, e) in g:
            self.assertGreater(e, s, f"空组: {g} (ratio={ratio}, ss={ss})")
        for (s1, e1), (s2, e2) in zip(g, g[1:]):
            self.assertEqual(e1, s2, f"组不连续: {g}")
        # 长度 < ratio 的步不得被拆到两个组（超长步允许步内切块）
        runs = []
        pos = 0
        for v in ss:
            if runs and runs[-1][0] == v:
                runs[-1][2] += 1
            else:
                runs.append([v, pos, 1])
            pos += 1
        for _v, start, ln in runs:
            if ln >= ratio:
                continue
            owner = [k for k, (s, e) in enumerate(g) if s <= start < e]
            self.assertEqual(len(owner), 1, f"步被拆分: {g} (ss={ss})")
            s, e = g[owner[0]]
            # 整步必须完整落在该组内（组可以包含更早的步）
            self.assertLessEqual(s, start)
            self.assertGreaterEqual(e, start + ln)
        return g

    def test_no_empty_group_and_full_coverage(self):
        # 该输入曾触发死循环：步 3 有 5 个 token ≥ ratio，前一组消费了
        # 两个整步后 si 未再前进 → 重复处理已消费步 → (6, 6) 空组
        self.assertEqual(
            self._invariants([0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 3], 4),
            [(0, 3), (3, 6), (6, 10), (10, 11)])

    def test_random_matches_reference(self):
        import random

        def ref(ss, ratio):
            """ 独立参考实现：整步向 ratio 靠拢，误差最小、平局取欠。 """
            runs = []                       # [step, 长度]
            for v in ss:
                if runs and runs[-1][0] == v:
                    runs[-1][1] += 1
                else:
                    runs.append([v, 1])
            groups, pos, k = [], 0, 0
            while k < len(runs):
                ln = runs[k][1]
                if ln >= ratio:
                    for cs in range(pos, pos + ln, ratio):
                        groups.append((cs, min(cs + ratio, pos + ln)))
                    pos += ln
                    k += 1
                    continue
                total, j = ln, k
                while j + 1 < len(runs):
                    nxt = runs[j + 1][1]
                    if total + nxt <= ratio:
                        total += nxt
                        j += 1
                        continue
                    if (ratio - total) <= (total + nxt - ratio):
                        break
                    total += nxt
                    j += 1
                    break
                groups.append((pos, pos + total))
                pos += total
                k = j + 1
            return groups

        rnd = random.Random(0)
        for _ in range(500):
            n = rnd.randint(1, 40)
            ss = sorted(rnd.randint(0, 8) for _ in range(n))   # 非降序
            ratio = rnd.choice([1, 2, 3, 4, 5, 8, 32])
            self.assertEqual(model.group_by_step(torch.tensor(ss), ratio),
                             ref(ss, ratio))
            self._invariants(ss, ratio)


class TestServerEnvLauncher(unittest.TestCase):
    """ server 侧启动链（2026-10-24 起环境检查/窗口校验归 server 管）：
    依赖探针语义 + 参数默认值语义。 """

    def test_check_deps_detects_current_interpreter(self):
        # 回归：探针曾写成 `import importlib`（不绑定 importlib.util），
        # 恒抛 AttributeError → 恒判「缺依赖」→ 每次启动都重装一遍
        # torch（--test 被拖成 16 分钟假死，无网直接崩）
        import server_env
        self.assertTrue(server_env.check_deps(sys.executable),
                        "当前解释器已装齐依赖，check_deps 必须返回 True")

    def test_check_deps_rejects_interpreter_without_torch(self):
        # 反证：换一个只有标准库的解释器（系统 python3 未必有 torch，
        # 用 -S -I 隔离站点目录必然缺 torch）→ 必须 False
        import server_env
        import shutil
        bare = shutil.which("python3") or shutil.which("python")
        if not bare:
            self.skipTest("找不到系统 python3")
        probe = ("import importlib.util,sys;"
                 "sys.exit(0 if importlib.util.find_spec('torch') else 1)")
        r = subprocess.run([bare, "-c", probe], capture_output=True)
        expect = (r.returncode == 0)
        self.assertEqual(server_env.check_deps(bare), expect,
                         "check_deps 必须与 find_spec('torch') 事实一致")

    def test_num_param_keeps_explicit_zero(self):
        # server.num_param：显式 0 不得被默认值吞掉
        from server import num_param
        self.assertEqual(num_param({"temperature": 0}, "temperature", 0.8, float), 0.0)
        self.assertEqual(num_param({}, "temperature", 0.8, float), 0.8)
        self.assertEqual(num_param({"n": None}, "n", 5, int), 5)
        self.assertEqual(num_param({"n": "7"}, "n", 5, int), 7)
        with self.assertRaises(ValueError):
            num_param({"n": "abc"}, "n", 5, int)


class TestStartupChain(unittest.TestCase):
    """ 2026-10-24 启动链（server 拥有环境检查/窗口校验/服务调度）的回归。

    端到端（setup-environment.sh 装隔离环境 → server.py 起窗口+服务子进程、窗口崩溃
    不影响 server、server 退出带走服务）已手工验证；这里固化不依赖 GUI
    的部分：模块边界、re-export 完整性、窗口冒烟、收尸逻辑、环境预检。 """

    def test_server_reexports_intact(self):
        # 拆分后所有旧 import 路径必须仍然可用（test_server.py 走子进程，
        # 只靠它抓不到「re-export 被删」这种回归）
        import server
        for name in ("ModelManager", "TaskPool", "step_to_json",
                     "ConflictError", "b64_to_pil", "size_guard_pil",
                     "extract_video_frames", "decode_audio_any",
                     "MAX_PATCHES_TOTAL", "MAX_IMAGE_DIM", "App",
                     "Handler", "BodyTooLarge", "num_param", "main",
                     # 2026-10-24 新增模块的符号也要在 server 命名空间可见
                     "ServiceSupervisor", "ServiceError", "ROOT"):
            self.assertTrue(hasattr(server, name), f"server.{name} 丢失")

    def test_app_mixin_assembly(self):
        # App 由 TrainApi + CodecApi + TokensApi 组合而成；任一路由表里的
        # 方法缺失都会在运行时 500，这里提前拦
        import server
        expected = {
            "api_status", "api_models", "api_model_init", "api_model_load",
            "api_infer", "api_system", "api_vocab",
            "api_train_enqueue", "api_train_start", "api_train_stop",
            "api_train_status", "api_train_queue", "api_train_delete",
            "api_codec_image", "api_codec_frames", "api_codec_audio",
            "api_codec_draw", "api_codec_text",
            "api_codec_video", "api_codec_image2prims",
            # tok-32 离散 token API（2026-10-24）
            "api_tokens_encode", "api_tokens_decode",
            # 词表文件夹 API（2026-10-30）
            "api_vocab_list", "api_vocab_regions", "api_vocab_query",
            "api_vocab_source_get", "api_vocab_token", "api_vocab_source",
            "api_vocab_fork", "api_vocab_delete",
        }
        missing = [m for m in expected
                   if not callable(getattr(server.App, m, None))]
        self.assertEqual(missing, [], f"App 缺少路由方法: {missing}")

    def test_server_does_not_import_torch_at_module_level(self):
        # 2026-10-24 架构回归：server 本体不加载 torch（重活全在服务子进程），
        # 否则「按需启动省内存」就是空话——server 常驻即常驻一份 torch+模型
        import subprocess
        import sys
        root = boot.project_root()
        r = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, %r); import boot; boot.bootstrap();"
             "import server; print('torch' in dir(server))" % os.path.join(root, "start")],
            capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "False",
                         "server 模块级不得出现 torch")

    @unittest.skipUnless(_has_pywebview(), "本机未安装 pywebview（可选窗口依赖）")
    def test_window_smoke(self):
        # pywebview 可导入、窗口对象可创建（无显示环境也能验证）。
        # 装不了/没装时给出可执行提示而不是默默失败。
        import subprocess
        import sys
        root = boot.project_root()
        r = subprocess.run([sys.executable, os.path.join(root, "service", "window", "window.py"),
                            "--smoke"], capture_output=True, text=True,
                           timeout=180)
        self.assertEqual(r.returncode, 0,
                         f"window.py --smoke 失败:\n{r.stdout}\n{r.stderr}")
        self.assertIn("smoke OK", r.stdout)

    def test_environment_precheck_rejects_bogus_interpreter(self):
        # server_env.check_environment：读不到版本的解释器必须拒（False），
        # 并且不能抛异常（启动早期要能给出可读提示而不是 traceback）
        import server_env
        self.assertFalse(server_env.check_environment("/no/such/python-3.12"))
        # 版本门槛：torch 2.2.2 只支持 ≤3.12——当前解释器必然在门内
        self.assertGreaterEqual(server_env.python_version(sys.executable),
                                (3, 12))

    def test_launcher_helpers(self):
        import server_env
        # wait_port：关着的端口 False，开着的 True
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        try:
            self.assertTrue(server_env.wait_port(port, timeout=2))
        finally:
            s.close()
        self.assertFalse(server_env.wait_port(port, timeout=1))
        # kill_proc 真的能收尸
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        server_env.kill_proc(p, grace=3)
        self.assertIsNotNone(p.poll(), "kill_proc 后进程必须退出")

    def test_launcher_uses_current_interpreter_when_complete(self):
        # 2026-10-20 坑：探针 `import importlib` 不绑定 importlib.util，
        # 恒判缺依赖 → 每次启动都重装 torch
        import server_env
        self.assertTrue(server_env.check_deps(sys.executable),
                        "当前解释器依赖齐全时 check_deps 必须为 True")
        probe = ("import importlib.util,sys;"
                 "mods=['torch','numpy','PIL','safetensors'];"
                 "ok=all(importlib.util.find_spec(m) is not None "
                 "for m in mods);sys.exit(0 if ok else 1)")
        r = subprocess.run([sys.executable, "-c", probe])
        self.assertEqual(r.returncode, 0)

    def test_system_reports_dependency_versions(self):
        # 回归：api_system 曾用裸 `_DIST_NAME` 访问类属性（NameError），
        # 又被 `except Exception: pass` 吞掉 → 前端永远拿不到版本号，
        # 「环境与依赖」卡片退化成只有名字。
        import server
        code, obj = server.App("models", "data",
                               "logs/auto_train.jsonl").api_system()
        self.assertEqual(code, 200)
        names = {d["name"]: d for d in obj["dependencies"]}
        for mod in ("torch", "numpy", "PIL", "safetensors"):
            self.assertTrue(names[mod]["ok"], f"{mod} 应已安装")
            self.assertTrue(names[mod].get("version"),
                           f"{mod} 缺少版本号（类属性访问/静默 except？）")
        # PIL 的发行包名是 pillow，不是 PIL
        self.assertIn(".", names["PIL"]["version"])
        # 可选依赖缺失时必须给「用途 + pip 包名」，而不是 str(e) 原文
        for mod in ("cv2", "soundfile"):
            if names[mod]["ok"]:
                continue                        # 环境里装过就跳过
            self.assertTrue(names[mod].get("optional"), f"{mod} 应标记为可选")
            self.assertTrue(names[mod].get("why"), f"{mod} 缺用途说明")
            self.assertTrue(names[mod].get("pkg"), f"{mod} 缺 pip 包名")
            self.assertNotIn("No module named", names[mod].get("note", ""),
                             "不许把 traceback 文案倒给前端")
        self.assertEqual(sorted(obj["optional_missing"]),
                         sorted(n for n, d in names.items()
                                if d.get("optional") and not d["ok"]))

    def test_optional_install_carries_constraints(self):
        # 回归：装 cv2/soundfile 时若不带 `-c start/requirements/core.txt`，
        # pip 会把 numpy 升到 2.x，而 torch==2.2.2 只支持 numpy<2——主依赖
        # 契约被一个「可选」安装悄悄破坏（实测发生过：numpy 1.26.4 → 2.5.4）。
        from server_util import optional_install_cmd
        req = os.path.join(boot.project_root(), "start",
                           "requirements", "core.txt")
        cmd = optional_install_cmd("/py", ["opencv-python"], req)
        self.assertIn("-c", cmd)
        self.assertIn(req, cmd)
        self.assertIn("opencv-python", cmd)
        # 源参数也必须带上（国内默认清华）
        self.assertTrue(any(a.startswith("https://") for a in cmd))


if __name__ == "__main__":
    unittest.main()
