# ===========================================================================
# GARGANTUA v1 — test_infer_archive.py：推理存档的 user/model 段边界包裹
# ---------------------------------------------------------------------------
# 验证 generate() 把「用户输入」包成 <user>…<user/>、「模型输出」包成
# <model>…<model/> 写进 .tokens 存档（与训练 .tokens 同一套边界约定）。
# 同时覆盖媒体步（视觉 patch）经 from_steps → add_items 落盘的路径。
#
# 这是 steve 2026-11-05 拍板的需求：存档里必须能看出谁是谁，而不是扁平
# ID 流。边界 token（id 1/2/3/4）不可显示、不进 loss、不进 KV，只在存档
# 层做结构标记；读回后 tok5 解析为 ext 元素，前端按 spec 映射回名字。
#
# 运行：cd <ROOT> && python -m unittest service.test.test_infer_archive -v
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
import tempfile
import unittest

import spec
from tokens import tok5
from tokens.vocab import Vocab
from infer import IdArchive

# 新契约：词表长度不固定；用全局词表长度兜底选可见文本 token（>HEAD_SIZE）。
_VS = len(Vocab.global_vocab())


def _visible_ids(n, base=100):
    """取 n 个确定可见、合法的文本 token id（HEAD_SIZE 之后、词表之内）。"""
    ids = [spec.HEAD_SIZE + base + i for i in range(n)]
    assert max(ids) < _VS, "测试选的 token id 超出词表"
    return ids


class TestInferArchiveWrapping(unittest.TestCase):
    """存档边界包裹的往返一致性。"""

    def _tmp_root(self):
        return tempfile.mkdtemp(prefix="garg_archive_")

    def test_user_model_boundary_wrapping(self):
        prompt_ids = _visible_ids(2)
        out_ids = _visible_ids(3, base=200)
        root = self._tmp_root()
        arc = IdArchive("", root)
        # —— 复刻 generate() 的真实存档顺序 ——
        arc.add([spec.USER_ID])
        arc.add(prompt_ids)
        arc.add([spec.USER_END_ID])
        arc.add([spec.MODEL_ID])
        arc.add(out_ids)
        arc.add([spec.MODEL_END_ID])
        arc.close(n_prompt=len(prompt_ids), n_generated=len(out_ids))

        files = [f for f in os.listdir(root) if f.endswith(".tokens")]
        self.assertEqual(len(files), 1, files)
        items = tok5.read_file(os.path.join(root, files[0]))

        ext_ids = [it["id"] for it in items if it.get("kind") == "ext"]
        text_ids = []
        for it in items:
            if it.get("kind") == "step":
                for e in it["elements"]:
                    if e.get("kind") == "text":
                        text_ids.append(e["id"])

        # 四个边界按 <user> <user/> <model> <model/> 顺序出现
        self.assertEqual(
            ext_ids,
            [spec.USER_ID, spec.USER_END_ID, spec.MODEL_ID, spec.MODEL_END_ID],
            "边界 token 顺序或数量不对")
        # 内容 token 全部落盘、顺序正确（prompt 在前、生成为后）
        self.assertEqual(text_ids, prompt_ids + out_ids)
        # 边界 id 必须能被前端映射成名（<user>/<user/>/<model>/<model/>）
        for i in ext_ids:
            self.assertIn(i, spec.T5_BOUND_ID_TO_NAME,
                          "边界 id 没有对应名字，前端渲染会丢结构")

    def test_media_step_also_archived(self):
        """媒体（视觉 patch）经 from_steps → add_items 也应落盘。"""
        prompt_ids = _visible_ids(2)
        media_steps = [{
            "token_ids": [],
            "type": spec.TYPE_USER,
            "modality": "visual",
            "meta": {"patches": [{"vec": bytes(4096), "x": 0, "y": 0}]},
        }]
        root = self._tmp_root()
        arc = IdArchive("", root)
        arc.add([spec.USER_ID])
        arc.add_items(tok5.from_steps(media_steps))
        arc.add(prompt_ids)
        arc.add([spec.USER_END_ID])
        arc.add([spec.MODEL_ID])
        arc.add(_visible_ids(1, base=300))
        arc.add([spec.MODEL_END_ID])
        arc.close(n_prompt=len(prompt_ids), n_generated=1)

        files = [f for f in os.listdir(root) if f.endswith(".tokens")]
        self.assertEqual(len(files), 1)
        items = tok5.read_file(os.path.join(root, files[0]))

        # 媒体步应呈现为一个含 visual 元素的 step
        n_visual = sum(
            1 for it in items if it.get("kind") == "step"
            for e in it["elements"] if e.get("kind") == "visual")
        self.assertEqual(n_visual, 1, "媒体 patch 未落盘")
        # 边界结构依然完整
        ext_ids = [it["id"] for it in items if it.get("kind") == "ext"]
        self.assertEqual(
            ext_ids,
            [spec.USER_ID, spec.USER_END_ID, spec.MODEL_ID, spec.MODEL_END_ID])

    def test_empty_generation_still_closes_boundaries(self):
        """无生成时也要写一对空 <model>…<model/>，保持存档结构完整。"""
        prompt_ids = _visible_ids(2)
        root = self._tmp_root()
        arc = IdArchive("", root)
        arc.add([spec.USER_ID])
        arc.add(prompt_ids)
        arc.add([spec.USER_END_ID])
        arc.add([spec.MODEL_ID])        # 没有生成 -> 直接收尾
        arc.add([spec.MODEL_END_ID])
        arc.close(n_prompt=len(prompt_ids), n_generated=0)

        files = [f for f in os.listdir(root) if f.endswith(".tokens")]
        items = tok5.read_file(os.path.join(root, files[0]))
        ext_ids = [it["id"] for it in items if it.get("kind") == "ext"]
        # 即使无生成，<model> 与 <model/> 也必须成对出现
        self.assertIn(spec.MODEL_ID, ext_ids)
        self.assertIn(spec.MODEL_END_ID, ext_ids)
        self.assertEqual(
            ext_ids.count(spec.MODEL_ID), 1)
        self.assertEqual(
            ext_ids.count(spec.MODEL_END_ID), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
