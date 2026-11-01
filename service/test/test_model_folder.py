# ===========================================================================
# GARGANTUA v1 — service/test/test_model_folder.py：模型文件夹契约回归
# ---------------------------------------------------------------------------
# 背景（2026-10-30 词表契约重构）：
#   模型目录从「一个 model.safetensors 装全部」升级为
#     models/NAME/{config.json,
#                  weights/{model,embed_discrete,embed_continuous}
#                          .<uuid>.safetensors,
#                  vocab/{vocab.vfc, config.json}}
#   同时词表长度不再由 spec 钉死，改成每个模型自己的属性。
#   本文件覆盖这次重构的五条硬约束：
#     1. split_state_dict 三分组不重不漏（含真实 Gargantua 的全部 key）；
#     2. Gargantua 的可变词表长度（维度正确 / 非法值被拒 / 无参回退）；
#     3. save_checkpoint 落出的目录结构符合契约（三个文件都在 weights/ 下）；
#     4. 重复保存回收旧权重，不无限堆积；
#     5. model_vocab_size 的多层优先级回退行为。
#
# 为什么真实 Gargantua 只在必要的用例里建：全配置实例约 280MB（FP32
# 1024 维），反复建/存会吃满磁盘（本机可用空间本就紧张）。与规模无关的
# 契约断言（分组、目录结构、manifest、回收）一律用与真实 state_dict 同名
# 的极小张量替身完成，既快又不往磁盘写几百 MB 垃圾。
#
# 运行：cd ROOT && python -m unittest service.test.test_model_folder -v
# 前置：torch + safetensors 可用（项目 environment/venv）。
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

import json
import os
import shutil
import tempfile
import unittest

import torch
import torch.nn as nn

import common
import model as model_mod
import spec
from common import (CONFIG_NAME, CONTINUOUS_EMBED_PREFIX,
                    DISCRETE_EMBED_PREFIX, MAIN_WEIGHTS_PREFIX,
                    VOCAB_DIRNAME, VOCAB_VFC_NAME, WEIGHTS_DIRNAME,
                    WEIGHTS_KEYS, build_model, install_model_vocab,
                    load_model_state_dict, model_vocab_size,
                    resolve_weights_path, save_checkpoint, split_state_dict)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


class StubGargantua(nn.Module):
    """ 与真实 Gargantua 的 state_dict key 同名的极小替身。

    key 名必须对得上：common 的分组规则是按 key 前缀匹配的，替身用真实
    的那几个子模块名才能验证到真契约。 """

    def __init__(self, n_vocab: int = 16, d_model: int = 8):
        super().__init__()
        self.n_vocab = n_vocab
        # —— 离散 embedding（前缀 wte.）——
        self.wte = nn.Embedding(n_vocab, d_model)
        self.type_emb = nn.Embedding(spec.NUM_TYPES, d_model)
        self.lm_head = nn.Linear(d_model, n_vocab, bias=False)
        # —— 连续 embedding（visual_* / audio_*）——
        self.visual_proj = nn.Linear(4, d_model, bias=False)
        self.visual_coord_x = nn.Embedding(256, d_model)
        self.visual_coord_y = nn.Embedding(256, d_model)
        self.audio_proj = nn.Linear(5, d_model, bias=False)
        # —— 主权重（其余全部）——
        self.enc_blocks = nn.ModuleList([nn.Linear(d_model, d_model)])
        self.dec_blocks = nn.ModuleList([nn.Linear(d_model, d_model)])


def stub_factory(n_enc_blocks=None, n_dec_blocks=None, vocab_size=None,
                 **_kw):
    """替换 model.Gargantua 的构造替身：接受 build_model 传的全部 kwargs。"""
    return StubGargantua(n_vocab=int(vocab_size))


class TestSplitStateDict(unittest.TestCase):
    """ 三分组必须不重不漏地覆盖 state_dict 的每个 key。 """

    def test_fake_state_dict_groups(self):
        sd = {
            "wte.weight": 1,
            "visual_proj.weight": 2,
            "visual_coord_x.weight": 3,
            "visual_coord_y.weight": 4,
            "audio_proj.weight": 5,
            "lm_head.weight": 6,
            "type_emb.weight": 7,
            "enc_blocks.0.kda1.w_q.weight": 8,
        }
        main, disc, cont = split_state_dict(sd)
        self.assertEqual(sorted(disc), ["wte.weight"],
                         "离散 embedding 应只有 wte")
        self.assertEqual(sorted(cont),
                         ["audio_proj.weight", "visual_coord_x.weight",
                          "visual_coord_y.weight", "visual_proj.weight"],
                         "连续 embedding 应是视觉/音频投影四件套")
        self.assertEqual(sorted(main),
                         ["enc_blocks.0.kda1.w_q.weight", "lm_head.weight",
                          "type_emb.weight"],
                         "lm_head 与 type_emb 属于主权重")
        # 不重不漏：并集 == 原集合，且三份长度之和 == 原长度
        self.assertEqual(set(main) | set(disc) | set(cont), set(sd))
        self.assertEqual(len(main) + len(disc) + len(cont), len(sd))
        self.assertFalse(set(main) & set(disc))
        self.assertFalse(set(main) & set(cont))
        self.assertFalse(set(disc) & set(cont))

    def test_covers_real_gargantua_keys(self):
        """真实 Gargantua 的全部 key 恰好落在三分组里（漏一个 = 保存丢参数）。"""
        from model import Gargantua
        m = Gargantua(n_enc_blocks=1, n_dec_blocks=1, vocab_size=64)
        try:
            sd = m.state_dict()
            main, disc, cont = split_state_dict(sd)
            self.assertEqual(len(main) + len(disc) + len(cont), len(sd))
            self.assertEqual(set(main) | set(disc) | set(cont), set(sd))
            self.assertEqual(sorted(disc), ["wte.weight"],
                             f"真实的离散 embedding 分组变了: {sorted(disc)}")
            self.assertEqual(sorted(cont),
                             ["audio_proj.weight", "visual_coord_x.weight",
                              "visual_coord_y.weight", "visual_proj.weight"],
                             f"真实的连续 embedding 分组变了: {sorted(cont)}")
            self.assertIn("lm_head.weight", main)
            self.assertTrue(len(main) > 100, "主权重应占绝大多数参数")
        finally:
            del m


class TestVocabSizeArgument(unittest.TestCase):
    """ Gargantua 的词表长度：每个模型可以不同。 """

    def test_vocab_size_drives_wte_and_lm_head(self):
        from model import Gargantua
        for n in (1, 64, 4096):
            m = Gargantua(n_enc_blocks=1, n_dec_blocks=1, vocab_size=n)
            try:
                self.assertEqual(m.n_vocab, n)
                self.assertEqual(tuple(m.wte.weight.shape),
                                 (n, spec.D_MODEL))
                self.assertEqual(tuple(m.lm_head.weight.shape),
                                 (n, spec.D_MODEL))
            finally:
                del m

    def test_invalid_vocab_size_rejected(self):
        from model import Gargantua
        for bad in (0, -1, -999, True, False, 3.5, "64", [], {}, object()):
            with self.assertRaises(ValueError, msg=f"应拒绝 vocab_size={bad!r}"):
                Gargantua(n_enc_blocks=1, n_dec_blocks=1, vocab_size=bad)

    def test_integral_float_accepted(self):
        from model import Gargantua
        m = Gargantua(n_enc_blocks=1, n_dec_blocks=1, vocab_size=1024.0)
        try:
            self.assertEqual(m.n_vocab, 1024)
        finally:
            del m

    def test_no_arg_falls_back_to_spec_default(self):
        """无参构造：spec 仍规定默认长度就用它；不规定了必须明确报错
        ——不能悄悄造出一个形状随便、几步之后才 RuntimeError 的模型。"""
        from model import Gargantua, default_vocab_size
        try:
            expected = default_vocab_size()
        except ValueError:
            with self.assertRaises(ValueError):
                Gargantua(n_enc_blocks=1, n_dec_blocks=1)
            return
        m = Gargantua(n_enc_blocks=1, n_dec_blocks=1)
        try:
            self.assertEqual(m.n_vocab, expected)
            self.assertEqual(m.wte.weight.shape[0], expected)
        finally:
            del m


class TestModelFolderContract(unittest.TestCase):
    """ save_checkpoint 落出的目录结构 + 加载 / 回收 / 旧布局兼容。 """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="garg_folder_")
        self.md = os.path.join(self.tmp, "models", "alpha")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- 目录结构 ----
    def test_new_layout_structure(self):
        torch.manual_seed(0)
        m = StubGargantua(n_vocab=16)
        save_checkpoint(self.md, m, {"n_enc_blocks": 1, "n_dec_blocks": 1,
                                     "vocab_size": 16})
        wdir = os.path.join(self.md, WEIGHTS_DIRNAME)
        self.assertTrue(os.path.isdir(wdir), "缺少 weights/ 目录")
        with open(os.path.join(self.md, CONFIG_NAME), encoding="utf-8") as f:
            cfg = json.load(f)
        seen = []
        for key, prefix in ((WEIGHTS_KEYS[0], MAIN_WEIGHTS_PREFIX),
                            (WEIGHTS_KEYS[1], DISCRETE_EMBED_PREFIX),
                            (WEIGHTS_KEYS[2], CONTINUOUS_EMBED_PREFIX)):
            rel = cfg.get(key)
            self.assertIsInstance(rel, str, f"manifest 缺 {key}")
            self.assertTrue(rel.startswith(WEIGHTS_DIRNAME + "/"),
                            f"{key} 应指向 weights/ 下: {rel}")
            name = os.path.basename(rel)
            self.assertTrue(name.startswith(prefix + "."),
                            f"{key} 的文件名前缀应为 {prefix}.: {name}")
            self.assertTrue(name.endswith(".safetensors"), name)
            self.assertTrue(os.path.isfile(os.path.join(self.md, rel)),
                            f"manifest 指向的文件不存在: {rel}")
            seen.append(os.path.join(wdir, name))
        # 三个文件互不重名，且 weights/ 下没有第 4 个权重文件
        self.assertEqual(len(set(seen)), 3)
        self.assertEqual(sorted(os.listdir(wdir)),
                         sorted(os.path.basename(p) for p in seen),
                         f"weights/ 下出现了预期外文件: {os.listdir(wdir)}")
        # 不得留下任何 .tmp 半成品
        leftovers = [p for p in os.listdir(wdir) if p.endswith(".tmp")]
        self.assertEqual(leftovers, [], f"残留 tmp: {leftovers}")

    def test_manifest_points_at_the_three_files_and_roundtrip(self):
        torch.manual_seed(1)
        m = StubGargantua(n_vocab=32)
        save_checkpoint(self.md, m, {"vocab_size": 32})
        sd = load_model_state_dict(self.md)
        ref = m.state_dict()
        self.assertEqual(set(sd), set(ref), "加载回来的 key 集合应完整")
        for k, v in ref.items():
            self.assertTrue(torch.equal(sd[k], v), f"参数回读不一致: {k}")

    def test_repeated_save_recycles_old_weights(self):
        m = StubGargantua(n_vocab=16)
        wdir = os.path.join(self.md, WEIGHTS_DIRNAME)
        counts = []
        for i in range(4):
            with torch.no_grad():
                m.wte.weight.fill_(float(i))     # 每次内容都不同，防假通过
            save_checkpoint(self.md, m, {"vocab_size": 16})
            counts.append(len(os.listdir(wdir)))
        self.assertEqual(counts, [3, 3, 3, 3],
                         f"重复保存后 weights/ 应保持 3 个文件: {counts}")
        # 且最新一次的内容确实生效
        sd = load_model_state_dict(self.md)
        self.assertAlmostEqual(float(sd["wte.weight"].flatten()[0]), 3.0,
                               places=5)

    def test_manifest_rejects_path_traversal(self):
        """config.json 里的权重路径必须是干净的向下相对路径。"""
        write_json(os.path.join(self.md, CONFIG_NAME),
                   {"weights": "../../../etc/passwd"})
        self.assertIsNone(resolve_weights_path(self.md),
                          "含 .. 的 manifest 条目必须被忽略")
        self.assertEqual(load_model_state_dict(self.md), {},
                         "非法 manifest 不应把目录外的文件读进来")

    # ---- build_model 全链路（用替身替换 Gargantua）----
    def test_build_model_reads_vocab_size_from_vocab_config(self):
        write_json(os.path.join(self.md, VOCAB_DIRNAME, CONFIG_NAME),
                   {"vocab_size": 33})
        write_json(os.path.join(self.md, CONFIG_NAME), {"vocab_size": 16})
        orig = model_mod.Gargantua
        model_mod.Gargantua = stub_factory
        try:
            got = build_model(self.md)
        finally:
            model_mod.Gargantua = orig
        self.assertEqual(got.n_vocab, 33, "vocab/config.json 优先级最高")

    def test_build_model_loads_three_files(self):
        torch.manual_seed(3)
        m = StubGargantua(n_vocab=24)
        save_checkpoint(self.md, m, {"vocab_size": 24})
        orig = model_mod.Gargantua
        model_mod.Gargantua = stub_factory
        try:
            got = build_model(self.md)
        finally:
            model_mod.Gargantua = orig
        self.assertEqual(got.n_vocab, 24)
        for k, v in m.state_dict().items():
            self.assertTrue(torch.equal(got.state_dict()[k], v),
                            f"build_model 装配出的参数不一致: {k}")

    def test_build_model_vocab_mismatch_is_value_error(self):
        """换了 vocab/config.json 但没重训 → 明确的 ValueError（→400）。"""
        torch.manual_seed(4)
        m = StubGargantua(n_vocab=24)
        save_checkpoint(self.md, m, {"vocab_size": 24})
        write_json(os.path.join(self.md, VOCAB_DIRNAME, CONFIG_NAME),
                   {"vocab_size": 40})
        orig = model_mod.Gargantua
        model_mod.Gargantua = stub_factory
        try:
            with self.assertRaises(ValueError) as cm:
                build_model(self.md)
        finally:
            model_mod.Gargantua = orig
        self.assertIn("不匹配", str(cm.exception))


class TestModelVocabSize(unittest.TestCase):
    """ model_vocab_size 的多层优先级。 """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="garg_vsize_")
        self.md = os.path.join(self.tmp, "models", "beta")
        os.makedirs(self.md, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_vocab_config_wins_over_model_config(self):
        write_json(os.path.join(self.md, VOCAB_DIRNAME, CONFIG_NAME),
                   {"vocab_size": 111})
        write_json(os.path.join(self.md, CONFIG_NAME), {"vocab_size": 222})
        self.assertEqual(model_vocab_size(self.md), 111)

    def test_falls_back_to_model_config(self):
        write_json(os.path.join(self.md, CONFIG_NAME), {"vocab_size": 222})
        self.assertEqual(model_vocab_size(self.md), 222)

    def test_bad_vocab_config_falls_through(self):
        write_json(os.path.join(self.md, VOCAB_DIRNAME, CONFIG_NAME),
                   {"vocab_size": "abc"})
        write_json(os.path.join(self.md, CONFIG_NAME), {"vocab_size": 222})
        self.assertEqual(model_vocab_size(self.md), 222,
                         "vocab/config.json 写坏了应回退到下一层")

    def test_broken_json_files_do_not_crash(self):
        os.makedirs(os.path.join(self.md, VOCAB_DIRNAME), exist_ok=True)
        with open(os.path.join(self.md, VOCAB_DIRNAME, CONFIG_NAME), "w",
                  encoding="utf-8") as f:
            f.write("{ this is not json")
        write_json(os.path.join(self.md, CONFIG_NAME), {"vocab_size": 333})
        self.assertEqual(model_vocab_size(self.md), 333)

    def test_weights_rows_are_the_last_resort(self):
        """config 两层都没有 vocab_size → 用 wte.weight 的行数。"""
        torch.manual_seed(5)
        m = StubGargantua(n_vocab=77)
        save_checkpoint(self.md, m, {})
        with open(os.path.join(self.md, CONFIG_NAME), encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertNotIn("vocab_size", cfg,
                         "本用例要求 config 不带 vocab_size")
        self.assertEqual(model_vocab_size(self.md), 77,
                         "config 没写长度时应从已有权重推断")


class TestModelVocabFolder(unittest.TestCase):
    """ 词表目录：每模型一份独立副本（删了共享源也不能让模型报废）。 """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="garg_vdir_")
        self.src = os.path.join(self.tmp, "vocabs", "base")
        os.makedirs(self.src, exist_ok=True)
        self.md = os.path.join(self.tmp, "models", "gamma")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_install_copies_both_files_and_returns_vocab_size(self):
        with open(os.path.join(self.src, VOCAB_VFC_NAME), "w",
                  encoding="utf-8") as f:
            f.write("hello\t2970\n")
        write_json(os.path.join(self.src, CONFIG_NAME), {"vocab_size": 4096})
        got = install_model_vocab(self.md, self.src)
        self.assertEqual(got, 4096)
        vdir = os.path.join(self.md, VOCAB_DIRNAME)
        self.assertTrue(os.path.isfile(os.path.join(vdir, VOCAB_VFC_NAME)))
        self.assertTrue(os.path.isfile(os.path.join(vdir, CONFIG_NAME)))
        # 复制是「副本」不是「链接」：源目录删掉后模型里的词表还在
        shutil.rmtree(self.src)
        self.assertTrue(os.path.isfile(os.path.join(vdir, VOCAB_VFC_NAME)),
                        "删掉共享词表后，模型目录里的副本必须还在")
        self.assertEqual(model_vocab_size(self.md), 4096)

    def test_install_tolerates_missing_sources(self):
        """词表源还没产出 vocab.vfc（例如 .vfc 生成器未落地）时不能崩。"""
        got = install_model_vocab(self.md, self.src)
        self.assertIsNone(got)
        self.assertTrue(os.path.isdir(os.path.join(self.md, VOCAB_DIRNAME)))


if __name__ == "__main__":
    unittest.main()
