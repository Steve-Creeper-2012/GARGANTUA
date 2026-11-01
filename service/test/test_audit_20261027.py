# ===========================================================================
# GARGANTUA v1 — tests/test_audit_20261027.py：2026-10-27 审计修复回归测试
# ---------------------------------------------------------------------------
# 覆盖本轮审计（静态审查 + 最小复现）的修复项，每条对应审计报告条目：
#   S3  fp4 量化 scale 归一化（见 test_precision.py，此处测模型级行为）
#   S4  <stop> 进入训练目标（make_batch 起点上界 + 尾部窗口含 STOP）
#   H3  fp16/bf16/fp64 原生精度不断链（压缩视图 + 视觉/音频投影）
#   H9  train.py 不再引用已删除的 spec.CSA_LOCAL_WINDOW
#   M4  视频网格尺寸变化（同 patch 数）不再 KeyError
#   M5  图元尺寸上限与绘图 token 可表示范围一致
#   M6  绘图 token 解码逐字段校验
#   L1  dectok uint32 校验
# 运行：cd <ROOT> && python -m unittest service.test.test_audit_20261027 -v
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
import unittest

import torch

import model as model_mod
import spec
from tokens import vfc
from common import make_batch


from tokens.vocab import Vocab as _Vocab
_VS = len(_Vocab.global_vocab())   # 新契约：词表长度不再由 spec 钉死


def small_model(**kw):
    torch.manual_seed(0)
    kw.setdefault("vocab_size", _VS)
    return model_mod.Gargantua(n_enc_blocks=1, n_dec_blocks=1, **kw)


class TestMultiTokenStepNoLeak(unittest.TestCase):
    """S1：多 token 时间步内，未来 token 对当前位置不可见。

    审计复现：dec_in=[102,200,201]、s_dec=[3,4,4]、y=[200,201,202]，
    位置 1 预测 201 时旧掩码（kv_step <= x_step）能看到输入 201（标签
    泄漏）。修复后同 step 内按步内序号因果：pos 1 只能看 pos <= 1。"""

    def test_future_token_in_same_step_is_masked(self):
        torch.manual_seed(7)
        mla = model_mod.MLAAttention(causal=True)
        B, T = 1, 3
        x = torch.randn(B, T, spec.D_MODEL)
        steps = torch.tensor([[3, 4, 4]])
        pos = torch.tensor([[0, 0, 1]])          # 步 4 内：token1 pos0, token2 pos1
        kw = dict(kv_step_ids=steps, x_step_ids=steps, kv_pos=pos, x_pos=pos)
        out1 = mla(x, x, **kw)
        # 篡改「未来」token（步 4 pos 1 = 位置 2 的输入）
        x2 = x.clone()
        x2[:, 2] = x[:, 2] + 5.0
        out2 = mla(x2, x2, **kw)
        # 位置 1（步 4 pos 0）的输出不得因未来 token 变化而变化
        self.assertTrue(torch.allclose(out1[:, 1], out2[:, 1], atol=1e-5))
        # 位置 2（步 4 pos 1）自己能看到自己 → 必然变化（掩码确实在工作）
        self.assertFalse(torch.allclose(out1[:, 2], out2[:, 2], atol=1e-5))

    def test_past_token_in_same_step_still_visible(self):
        # 同 step 内 pos 0 对 pos 1 可见（因果方向不能反）
        torch.manual_seed(8)
        mla = model_mod.MLAAttention(causal=True)
        x = torch.randn(1, 2, spec.D_MODEL)
        steps = torch.tensor([[4, 4]])
        pos = torch.tensor([[0, 1]])
        kw = dict(kv_step_ids=steps, x_step_ids=steps, kv_pos=pos, x_pos=pos)
        out1 = mla(x, x, **kw)
        x2 = x.clone()
        x2[:, 0] += 5.0                          # 过去的 token
        out2 = mla(x2, x2, **kw)
        self.assertFalse(torch.allclose(out1[:, 1], out2[:, 1], atol=1e-5))

    def test_compressed_aggregate_strict_past(self):
        # 聚合条目（pos=-1）对同 step 查询隐身
        torch.manual_seed(9)
        mla = model_mod.MLAAttention(causal=True)
        x = torch.randn(1, 2, spec.D_MODEL)
        steps = torch.tensor([[4, 4]])
        kv_pos = torch.tensor([[-1, 0]])         # 位置 0 是聚合条目
        x_pos = torch.tensor([[0, 0]])
        out = mla(x, x, kv_step_ids=steps, x_step_ids=steps,
                  kv_pos=kv_pos, x_pos=x_pos)
        self.assertTrue(torch.isfinite(out).all())

    def test_causal_requires_pos(self):
        mla = model_mod.MLAAttention(causal=True)
        x = torch.randn(1, 2, spec.D_MODEL)
        steps = torch.tensor([[0, 1]])
        with self.assertRaises(ValueError):
            mla(x, x, kv_step_ids=steps, x_step_ids=steps)


class TestIncrementalInferenceState(unittest.TestCase):
    """S2：逐 token 增量前向与整段前向逐位一致（KDA S + 短卷积历史）。"""

    def test_conv_cache_matches_full_sequence(self):
        torch.manual_seed(11)
        conv = model_mod.CausalDepthwiseConv1d(16, spec.KDA_CONV_KERNEL)
        x = torch.randn(1, 6, 16)
        full, _ = conv(x)
        cache = None
        outs = []
        for t in range(6):
            o, cache = conv(x[:, t:t + 1], cache=cache)
            outs.append(o)
        inc = torch.cat(outs, 1)
        self.assertTrue(torch.allclose(full, inc, atol=1e-5))

    def test_kda_incremental_matches_full_sequence(self):
        torch.manual_seed(12)
        kda = model_mod.KDAAttention(d_model=spec.D_MODEL,
                                     n_head=spec.N_HEADS,
                                     head_dim=spec.KDA_HEAD_DIM)
        x = torch.randn(1, 5, spec.D_MODEL)
        full, S_full, _ = kda(x)
        S, cc = None, None
        outs = []
        for t in range(5):
            o, S, cc = kda(x[:, t:t + 1], state=S, conv_cache=cc)
            outs.append(o)
        inc = torch.cat(outs, 1)
        self.assertTrue(torch.allclose(full, inc, atol=1e-4))
        self.assertTrue(torch.allclose(S_full, S, atol=1e-4))

    def test_fast_weight_carries_conv_cache(self):
        torch.manual_seed(13)
        fw = model_mod.FastWeightBank(spec.D_MODEL)
        x = torch.randn(1, 4, spec.D_MODEL)
        full = fw(x)                     # 整段（内部状态 = (S, conv)）
        ref_state = fw._state
        self.assertIsInstance(ref_state, tuple)
        self.assertEqual(len(ref_state), 2)      # (S, conv_cache)
        self.assertIn("q", ref_state[1])
        # 外部 state dict 路径同样写入
        fw.reset()
        ext: dict = {}
        fw(x, state=ext)
        self.assertIn("fw", ext)


class TestStopInTrainingTarget(unittest.TestCase):
    """S4：y 必须能覆盖序列最后一个 token（STOP）。"""

    def test_last_window_includes_final_token(self):
        ids = list(range(40)) + [spec.STOP_ID]
        steps = list(range(len(ids)))
        B = 8
        # 新上界：i = len(ids) - B 合法，y 最后一个元素是 STOP
        x_in, s_in, dec_in, s_dec, y = make_batch(ids, len(ids) - B, B, steps)
        self.assertEqual(y[0, -1].item(), spec.STOP_ID)
        self.assertEqual(x_in.shape[1], B // 2)
        self.assertEqual(dec_in.shape[1], B // 2)

    def test_window_bounds_consistency(self):
        ids = list(range(20))
        steps = list(range(20))
        B = 4
        for i in range(0, len(ids) - B + 1):
            x_in, s_in, dec_in, s_dec, y = make_batch(ids, i, B, steps)
            self.assertEqual(y[0, -1].item(), ids[i + B - 1])


class TestDtypeChain(unittest.TestCase):
    """H3：非 FP32 精度下前向/投影/压缩视图不再 dtype 断链。"""

    def _forward_train(self, m, n=40):
        ids = torch.randint(0, m.n_vocab, (1, n))
        steps = torch.arange(n).unsqueeze(0)
        half = n // 2
        dec_ids = torch.cat([ids[:, half - 1:half], ids[:, half:n - 1]], 1)
        dec_steps = torch.arange(half, n).unsqueeze(0)
        return m.forward_train(ids[:, :half], steps[:, :half],
                               dec_ids, dec_steps)

    def test_native_dtypes_forward(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            m = small_model().to(dtype)
            logits = self._forward_train(m)
            self.assertEqual(logits.dtype, dtype)

    def test_step_compressor_dtype(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            m = small_model().to(dtype)
            comp = m.csa_compressor
            vecs = torch.randn(2, 4, spec.D_MODEL, dtype=dtype)
            mask = torch.ones(2, 4, dtype=torch.bool)
            out = comp(vecs, mask)
            self.assertEqual(out.dtype, dtype)

    def test_multimodal_embed_dtype(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            m = small_model().to(dtype)
            pv = torch.randint(0, 255, (3, spec.PATCH_VEC_DIM),
                               dtype=torch.uint8)
            coords = torch.tensor([[0, 0], [1, 0], [0, 1]])
            tids = torch.full((3,), spec.TYPE_USER)
            v = m.embed_visual(pv, coords, tids)
            self.assertEqual(v.dtype, dtype)
            samples = torch.zeros(2, spec.AUDIO_SAMPLES_PER_TOKEN)
            a = m.embed_audio(samples, torch.full((2,), spec.TYPE_USER))
            self.assertEqual(a.dtype, dtype)


class TestTrainPyCompat(unittest.TestCase):
    """H9：train.py 可导入且不再引用 CSA_LOCAL_WINDOW。"""

    def test_no_csa_local_window_constant(self):
        self.assertFalse(hasattr(spec, "CSA_LOCAL_WINDOW"))

    def test_config_dict_no_local_window(self):
        path = os.path.join(boot.project_root(), "service", "train",
                            "train.py")
        with open(path, encoding="utf-8") as f:
            src = f.read()
        # 不得再出现对该字段的读取/写入（注释提及不算）
        self.assertNotIn('"csa_local_window"', src)
        self.assertNotIn("getattr(m, \"csa_local_window\"", src)
        self.assertNotIn("spec.CSA_LOCAL_WINDOW", src)


class TestVideoGridChange(unittest.TestCase):
    """M4：patch 数相同但网格变化（64×32 → 32×64）不崩溃、按全量处理。"""

    def _frame(self, w, h, color):
        from PIL import Image
        return Image.new("RGBA", (w, h), color)

    def test_grid_shape_change_is_full(self):
        from tokens import image_codec
        f1 = self._frame(64, 32, (255, 0, 0, 255))    # 2×1 = 2 patches
        f2 = self._frame(32, 64, (255, 0, 0, 255))    # 1×2 = 2 patches
        steps = image_codec.frames_to_steps([f1, f2])
        self.assertTrue(steps[0]["meta"]["full"])
        self.assertTrue(steps[1]["meta"]["full"])     # 网格变化 → 全量重置


class TestPrimSizeBounds(unittest.TestCase):
    """M5：拟合源头不允许超出绘图 token 可表示的尺寸。"""

    def test_max_half_size_within_token_range(self):
        import service.prims.image_to_primitives as i2p
        max_full = (spec.DRAW_W_BINS - 1) / spec.SIZE_BINS_PER_PIXEL
        self.assertLessEqual(i2p.MAX_HALF_SIZE * 2.0, max_full)


class TestDrawTokenDecodeValidation(unittest.TestCase):
    """M6：tokens_to_primitives 逐字段校验 bin 区间。"""

    def test_invalid_bins_rejected(self):
        from tokens import draw_codec
        b = spec.DRAW_BEGIN
        good = draw_codec.primitive_to_tokens(
            draw_codec.Primitive(cx=10, cy=10, w=8, h=8, shape_type=0,
                                 theta=0.0, r=128, g=128, b=128))
        bad = list(good)
        bad[0] = b + spec.DRAW_X_OFFSET + spec.DRAW_X_BINS + 5  # cx bin 越界
        with self.assertRaises(ValueError):
            draw_codec.tokens_to_primitives(bad)

    def test_cross_field_bins_rejected(self):
        # cx 字段里塞 shape 区的 offset 也应被拒（bin 区间逐字段检查）
        from tokens import draw_codec
        b = spec.DRAW_BEGIN
        ids = [b + spec.DRAW_SHAPE_OFFSET] * 9
        with self.assertRaises(ValueError):
            draw_codec.tokens_to_primitives(ids)


class TestDectokUint32(unittest.TestCase):
    """L1：decode_ids 拒绝越界 uint32。"""

    def test_negative_rejected(self):
        from tokens import dectok
        with self.assertRaises(ValueError):
            dectok.decode_ids([-1])

    def test_overflow_rejected(self):
        from tokens import dectok
        with self.assertRaises(ValueError):
            dectok.decode_ids([2 ** 32])


# ---------------------------------------------------------------------------
# 第二轮：H1 / H2 / M3 / L3 / M11 / L4
# ---------------------------------------------------------------------------
_VP_TMP = None
_VP_ORIG = None


def setUpModule():
    """建一个临时 vocabs/（base 只生成一次，后续用例 copyfile 派生）。"""
    global _VP_TMP, _VP_ORIG
    import tempfile
    from pathlib import Path
    import vocab_presets as vp
    _VP_ORIG = vp.VOCABS_DIR
    _VP_TMP = Path(tempfile.mkdtemp(prefix="garg_audit_vp_"))
    vp.VOCABS_DIR = _VP_TMP / "vocabs"
    vp.ensure_base()


def tearDownModule():
    global _VP_TMP
    import shutil
    import vocab_presets as vp
    if _VP_ORIG is not None:
        vp.VOCABS_DIR = _VP_ORIG
    if _VP_TMP is not None:
        shutil.rmtree(_VP_TMP, ignore_errors=True)
        _VP_TMP = None


class TestVocabFileFormat(unittest.TestCase):
    """H1：词表文件格式——正文是 .vfc（一行一个 token），词表必须占满。"""

    def test_base_has_no_reserved_and_is_full(self):
        import vocab_presets as vp
        from tokens.vocab import Vocab
        entries = vfc.read_vfc(str(vp.vocab_vfc_path(vp.BASE)))
        self.assertGreater(len(entries), 60000)
        self.assertEqual([t for _, t, _v in entries
                          if t.startswith("<reserved:")], [])
        v = Vocab.load(vp.vocab_vfc_path(vp.BASE))
        # 新契约：词表必须占满——长度即真实长度，没有「空闲槽位」概念
        self.assertEqual(len(v), len(entries))
        occ = vp.vocab_occupancy(vp.BASE)
        self.assertEqual(occ["vocab_size"], len(v))
        self.assertEqual(occ["vocab_size"],
                         occ["visible_size"] + occ["invisible_size"])

    def test_newline_and_cr_tokens_survive(self):
        """token 自身的 \\n / \\r 必须无损往返（旧格式会变成空行、ID 前移）。"""
        import vocab_presets as vp
        from tokens.vocab import Vocab
        v = Vocab.load(vp.vocab_vfc_path(vp.BASE))
        reg = {r["region"]: r for r in v.regions}
        wb = reg["whitespace"]["begin"]
        self.assertEqual(v.token_to_id.get("\n"), wb + 2)
        self.assertEqual(v.token_to_id.get("\r"), wb + 3)
        # 空行吞掉后 ID 会整体前移 2 位——用「中」这类位置敏感的 token 兜底
        self.assertEqual(v.token_to_id.get("中"),
                         reg["cjk"]["begin"] + (ord("中") - 0x4E00))

    def test_preset_roundtrip_with_control_chars(self):
        import vocab_presets as vp
        vp.create_vocab("ctrlp")
        tok = "tab\there\nnewline"
        res = vp.add_token_to("ctrlp", tok)
        got = {t: i for i, t, _v in
               vfc.read_vfc(str(vp.vocab_vfc_path("ctrlp")))}
        self.assertEqual(got[tok], res["id"])
        self.assertEqual(len(vp.load_vocab("ctrlp")),
                         len(vp.load_vocab(vp.BASE)) + 1)


class TestFlatVocabSpecials(unittest.TestCase):
    """H2：词表（含每模型独立副本）的 special / system / fallback 齐全。"""

    def test_special_and_fallback_populated(self):
        import vocab_presets as vp
        from tokens.vocab import Vocab
        v = Vocab.load(vp.vocab_vfc_path(vp.BASE))
        self.assertEqual(v.special_ids.get("<stop>"), spec.STOP_ID)
        self.assertEqual(v.special_ids.get("<caps>"), 0)
        self.assertEqual(v.system_ids.get("{"), spec.TS_OPEN_ID)
        self.assertEqual(v.system_ids.get("|"), spec.TS_SEP_ID)
        # 单字母常规 token 兼作 fallback（拼写兜底）
        for ch in "abcdefghijklmnopqrstuvwxyz":
            self.assertIn(ch, v.fallback_ids)

    def test_enctok_spells_unknown_word_via_flat_fallback(self):
        import vocab_presets as vp
        from tokens.vocab import Vocab
        from tokens import enctok
        v = Vocab.load(vp.vocab_vfc_path(vp.BASE))
        ids, skipped = enctok.encode_text("zzzzq", v)
        self.assertEqual(skipped, [])
        self.assertEqual(ids, [v.fallback_ids["z"]] * 4 + [v.fallback_ids["q"]])

    def test_gts_sees_stop_in_flat_vocab(self):
        """gts 必须能导出 <stop>，否则模型永远学不会终止。"""
        import vocab_presets as vp
        from tokens.vocab import Vocab
        from tokens import gts
        v = Vocab.load(vp.vocab_vfc_path(vp.BASE))
        out = gts.parse_gts("{<user>hi<stop>}", vocab=v)
        self.assertEqual(out[0], spec.TS_OPEN_ID)
        self.assertIn(spec.STOP_ID, out)
        self.assertEqual(out[-1], spec.TS_CLOSE_ID)

    def test_gts_uses_same_vocab_for_text(self):
        """自定义 token 必须用传入的词表编码（旧实现退回全局词表 → 报未收录）。"""
        from tokens.vocab import Vocab
        from tokens import gts
        v = Vocab.from_tokens(["自定义词", "hello"])
        out = gts.parse_gts("{自定义词}", vocab=v)
        self.assertEqual(out, [spec.TS_OPEN_ID, spec.HEAD_SIZE,
                               spec.TS_CLOSE_ID])


class TestStaticVideoExport(unittest.TestCase):
    """M3：静态视频（无变化 delta 帧）也能导出 .tokenvector。"""

    def test_identical_frames_export(self):
        from PIL import Image
        from tokens import image_codec, tokenvector
        f = Image.new("RGBA", (64, 64), (10, 20, 30, 255))
        # 4 帧：0 全量、1/2 静止 delta、3 末帧强制全量
        steps = image_codec.frames_to_steps([f, f, f, f])
        # 每步都必须非空（tokenvector.dumps 拒绝空时间步）
        self.assertTrue(all(s["meta"]["patches"] for s in steps))
        self.assertTrue(steps[1]["meta"]["static"])
        self.assertTrue(steps[2]["meta"]["static"])
        self.assertFalse(steps[3]["meta"]["static"])   # 末帧强制全量
        tv = image_codec.steps_to_tokenvector(steps)
        dim, prec, tls = tokenvector.loads(tv)
        self.assertEqual(len(tls), 4)
        self.assertEqual(dim, spec.PATCH_VEC_DIM)


class TestPrimitivePackingArchive(unittest.TestCase):
    """L3：图元 9:1 打包用组内最大 step，且九个原始 id 全部归档。"""

    def test_pack_uses_max_step_and_archives_all_ids(self):
        import model as model_mod
        comp = model_mod.StepCompressor(ratio=4)
        packer = model_mod.StepCompressor(ratio=spec.DRAW_KV_COMPRESS)
        kv = model_mod.SharedKV(comp, comp, draw_packer=packer)
        for i in range(spec.DRAW_KV_COMPRESS):
            sid = 7 if i < 4 else 8          # 跨两个时间步
            kv.append(torch.randn(1, spec.D_MODEL), sid, spec.TYPE_MODEL,
                      "dec", token_ids=[100 + i])
            kv.mark_primitive_token()
        self.assertEqual(len(kv.entries), 1)
        self.assertEqual(kv.entries[0].step_id, 8)      # 组内最大，不是首条
        self.assertEqual([a.token_id for a in kv.archive],
                         [100 + i for i in range(spec.DRAW_KV_COMPRESS)])


def _load_minify():
    """service/web/minify.py 不在 sys.path 上，按路径加载。"""
    import importlib.util
    path = os.path.join(boot.project_root(), "service", "web", "minify.py")
    spec_ = importlib.util.spec_from_file_location("garg_minify", path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return mod


class TestMinifyFreshness(unittest.TestCase):
    """L4：--check 必须能证明产物是最新源文件生成的。"""

    def test_verify_detects_stale_product(self):
        minify = _load_minify()
        web = os.path.join(boot.project_root(), "service", "web")
        with open(os.path.join(web, "index_source.html"),
                  encoding="utf-8") as f:
            src = f.read()
        css = open(os.path.join(web, "app.css"), encoding="utf-8").read()
        js = open(os.path.join(web, "app.js"), encoding="utf-8").read()
        good = minify.minify_html(src)
        self.assertTrue(minify.verify(src, good, css, js))
        # 旧产物（HTML 漂移：标题被改）必须被判定过期
        stale_html = good.replace("<title>GARGANTUA</title>",
                                  "<title>GARGANTUA2</title>", 1)
        self.assertNotEqual(stale_html, good)
        self.assertFalse(minify.verify(src, stale_html, css, js))
        # app.css / app.js 是直接编辑、直接服务的源文件（无独立构建步），
        # 不会「产物过期」；但语法损坏的 app.js 必须被 node --check 拦下
        self.assertFalse(
            minify.verify(src, good, css, "function ( { // 语法错误"))

    def test_shipped_index_html_is_current(self):
        """仓库里的 index.html 必须与 app.css / app.js 一起构成当前
        index_source.html 的最新产物（三件缺一不可漂移）。"""
        web = os.path.join(boot.project_root(), "service", "web")
        with open(os.path.join(web, "index_source.html"),
                  encoding="utf-8") as f:
            src = f.read()
        with open(os.path.join(web, "index.html"), encoding="utf-8") as f:
            dst = f.read()
        css = open(os.path.join(web, "app.css"), encoding="utf-8").read()
        js = open(os.path.join(web, "app.js"), encoding="utf-8").read()
        self.assertTrue(minify_html_of_verify(src, dst, css, js))


class TestFrontendTaskState(unittest.TestCase):
    """M11：前端请求可超时/取消，活动模型精确判断。"""

    @staticmethod
    def _src():
        # 前端 JS 已拆到 app.js：源码级断言需同时扫描 index_source.html 与 app.js
        web = os.path.join(boot.project_root(), "service", "web")
        with open(os.path.join(web, "index_source.html"),
                  encoding="utf-8") as f:
            html = f.read()
        with open(os.path.join(web, "app.js"), encoding="utf-8") as f:
            js = f.read()
        return html + "\n" + js

    def test_api_has_timeout_and_cancel(self):
        src = self._src()
        self.assertIn("AbortController", src)
        self.assertIn("pollTask", src)

    def test_no_substring_model_match(self):
        # 只检查可执行代码（注释里保留旧写法说明不算）
        code = [ln.split("//")[0] for ln in self._src().splitlines()]
        for ln in code:
            self.assertNotIn('sb-model").textContent.includes(', ln,
                             "活动模型不得用 textContent 子串判断")
        self.assertIn("ACTIVE_MODEL", "\n".join(code))

    def test_vocab_select_keeps_selection(self):
        # 2026-10-30 重构后词表列表来自 /api/vocab/list（VOCAB.list），
        # 断言跟随新实现：下拉刷新后保留用户已选中的词表
        src = self._src()
        self.assertIn("VOCAB.list.some(v => v.name === prev)", src)


def minify_html_of(src: str) -> str:
    return _load_minify().minify_html(src)


def minify_html_of_verify(src: str, dst: str, css: str, js: str) -> bool:
    return _load_minify().verify(src, dst, css, js)


if __name__ == "__main__":
    unittest.main()
