# ===========================================================================
# GARGANTUA v1 — 模型核心回归测试
# 运行：cd <ROOT> && <torch解释器> -m unittest tests.test_model -v
# 解释器：~/.workbuddy/binaries/python/envs/gargantua312/bin/python
# ===========================================================================
import unittest

import torch
import torch.nn.functional as F

import spec
import model


class TestShapes(unittest.TestCase):
    """ 形状与参数量。 """

    def test_total_params_in_budget(self):
        m = model.Gargantua()
        n = sum(p.numel() for p in m.parameters())
        # 预算 ~500M，允许上浮（embedding 不绑定 + 527.9M 实测基线）
        self.assertGreater(n, 450e6)
        self.assertLess(n, 600e6)

    def test_kda_shapes(self):
        kda = model.KDAAttention()
        x = torch.randn(1, 16, spec.D_MODEL)
        out, s = kda(x)
        self.assertEqual(out.shape, (1, 16, spec.D_MODEL))
        self.assertEqual(s.shape, (1, spec.N_HEADS, spec.KDA_HEAD_DIM,
                                   int(spec.KDA_HEAD_DIM * spec.KDA_EXPAND_V)))

    def test_forward_train_and_backward(self):
        torch.manual_seed(0)
        m = model.Gargantua()
        ids_in = torch.randint(64, 70000, (1, 6))
        steps_in = torch.arange(6).unsqueeze(0)
        ids_t = torch.randint(64, 70000, (1, 4))
        steps_t = torch.arange(6, 10).unsqueeze(0)
        logits = m.forward_train(ids_in, steps_in, ids_t, steps_t)
        self.assertEqual(logits.shape, (1, 4, spec.VOCAB_SIZE_PADDED))
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]),
                               torch.randint(64, 70000, (4,)))
        loss.backward()
        # 随机初始化 loss 应接近 ln(V)
        self.assertAlmostEqual(loss.item(), 11.27, delta=0.6)
        grads = [p.grad.norm().item() for p in m.parameters()
                 if p.grad is not None]
        self.assertTrue(all(g == g for g in grads))          # 无 NaN
        self.assertGreater(sum(grads), 0.0)                  # 梯度非零


class TestStepCausalMask(unittest.TestCase):
    """ 时间步因果掩码：步间因果 + 步内双向，绝不泄露未来步。 """

    def test_mask_rule(self):
        x_s = torch.tensor([[2, 2, 3, 3]])
        kv_s = torch.tensor([[0, 1, 2, 2, 3, 4]])
        mask = kv_s.unsqueeze(1) <= x_s.unsqueeze(2)
        expect = torch.tensor([[[1, 1, 1, 1, 0, 0],
                                [1, 1, 1, 1, 0, 0],
                                [1, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 0]]], dtype=torch.bool)
        self.assertTrue(torch.equal(mask, expect))

    def test_no_future_leak_in_attention(self):
        torch.manual_seed(1)
        mla = model.MLAAttention(causal=True)
        x = torch.randn(1, 2, spec.D_MODEL)
        kv = torch.randn(1, 4, spec.D_MODEL)
        x_s = torch.tensor([[0, 1]])
        kv_s = torch.tensor([[0, 1, 2, 3]])
        out1 = mla(x, kv, kv_s, x_s)
        # 篡改未来位置的 kv，输出不应变化（未来不可见）
        kv2 = kv.clone()
        kv2[0, 2:] = torch.randn(2, spec.D_MODEL)
        out2 = mla(x, kv2, kv_s, x_s)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))


class TestGroupByStep(unittest.TestCase):
    """ 压缩分组：以时间步为原子，允许步内切分，绝不跨步超 ratio。"""

    def test_grouping(self):
        ss = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 3])
        g = model.group_by_step(ss, 4)
        self.assertEqual(g, [(0, 3), (3, 6), (6, 10), (10, 11)])

    def test_never_split_across(self):
        ss = torch.tensor([0, 0, 1, 1, 1, 2])
        for ratio in (2, 3, 4, 32):
            for (s, e) in model.group_by_step(ss, ratio):
                self.assertLessEqual(e - s, ratio)


class TestFastWeight(unittest.TestCase):
    """ fast weight：状态跨调用推进（先用后更新，全局写入）。"""

    def test_state_advances(self):
        torch.manual_seed(2)
        bank = model.FastWeightBank()
        x = torch.randn(1, 8, spec.D_MODEL)
        bank(x)
        s1 = bank._state.clone()
        bank(x)
        self.assertFalse(torch.allclose(s1, bank._state))

    def test_reset(self):
        bank = model.FastWeightBank()
        bank(torch.randn(1, 4, spec.D_MODEL))
        bank.reset()
        self.assertIsNone(bank._state)


class TestSharedKV(unittest.TestCase):
    """ 统一 KV 流：滑窗整步丢弃、双层视图、图元 9:1 就地压缩。"""

    def _mk(self):
        m = model.Gargantua()
        return model.SharedKV(m.csa_compressor, m.hca_compressor, m.draw_packer)

    def test_eviction_step_atomic(self):
        kv = self._mk()
        big = torch.randn(spec.CTX_TOTAL + 50, spec.D_MODEL)
        for i in range(0, big.shape[0], 10):
            kv.append(big[i:i + 10], step_id=i // 10, type_id=0, source="enc")
        self.assertLessEqual(len(kv.entries), spec.CTX_TOTAL)
        # 剩余条目必须是完整时间步：除最后一次写入（4146 % 10 = 6 条）外
        # 每个 step id 都应有 10 条，不存在被拦腰截断的步
        from collections import Counter
        cnt = Counter(e.step_id for e in kv.entries)
        self.assertTrue(all(v in (6, 10) for v in cnt.values()))

    def test_views(self):
        kv = self._mk()
        big = torch.randn(spec.CTX_TOTAL, spec.D_MODEL)
        for i in range(0, big.shape[0], 10):
            kv.append(big[i:i + 10], step_id=i // 10, type_id=0, source="enc")
        vh, _ = kv.view("hca")
        vc, _ = kv.view("csa")
        # HCA：1024 完整 + 3072/32≈96 压缩（步边界效应有少量余量）
        self.assertGreater(vh.shape[1], spec.CTX_FULL)
        self.assertLess(vh.shape[1], spec.CTX_FULL + 200)
        # CSA：局部窗口 512 + 4:1 压缩（步内切分有 padding 余量）
        self.assertGreater(vc.shape[1], spec.CSA_LOCAL_WINDOW)
        self.assertLess(vc.shape[1], spec.CSA_LOCAL_WINDOW + 1200)

    def test_primitive_packing(self):
        kv = self._mk()
        for i in range(18):     # 两个图元 = 18 个 token
            kv.append(torch.randn(1, spec.D_MODEL), step_id=i,
                      type_id=1, source="dec")
            kv.mark_primitive_token()
        # 18 token → 2 个压缩向量（9:1 就地打包）
        self.assertEqual(len(kv.entries), 2)


class TestMultimodalAssemble(unittest.TestCase):
    """ 多模态装配：混合 Step → 统一嵌入序列，step_id 共享规则正确。"""

    def test_mixed_steps(self):
        import numpy as np
        from PIL import Image
        from tokens.image_codec import image_to_step
        from tokens.audio_codec import waveform_to_steps
        from tokens.text_codec import encode as text_encode

        m = model.Gargantua()
        steps = list(text_encode("你好"))
        steps.append(image_to_step(Image.new("RGBA", (64, 64), (200, 30, 30, 255))))
        steps.append(waveform_to_steps(np.zeros(600, dtype=np.int16))[0])
        emb, sids = m.assemble_embeddings(steps)
        # 2 文本 token + 4 patch（同一步共享 id）+ 1 音频 token
        self.assertEqual(emb.shape, (1, 7, spec.D_MODEL))
        self.assertEqual(sids[0].tolist(), [0, 1, 2, 2, 2, 2, 3])

    def test_multimodal_forward_backward(self):
        import numpy as np
        from PIL import Image
        from tokens.image_codec import image_to_step
        torch.manual_seed(0)
        m = model.Gargantua()
        steps = [image_to_step(Image.new("RGBA", (64, 64), (1, 2, 3, 255)))]
        emb, sids = m.assemble_embeddings(steps)
        logits = m.forward_train_embeddings(emb, sids, emb, sids + 1)
        self.assertEqual(logits.shape, (1, 4, spec.VOCAB_SIZE_PADDED))
        logits.float().mean().backward()          # 梯度能回穿视觉投影
        self.assertIsNotNone(m.visual_proj.weight.grad)


if __name__ == "__main__":
    unittest.main()
