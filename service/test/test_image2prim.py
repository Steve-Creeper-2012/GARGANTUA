# ===========================================================================
# GARGANTUA v1 — 图像→图元转换器回归测试
# 算法来源：wonderfulearth/primitive-operation-painter fast_shape_render
# （GPU 路径照搬第三方 Rust 转换器；此处测 CPU 兜底拟合器，输出 schema
#   与仓库 CSV 一致）
# 运行：cd <ROOT> && python -m unittest tests.test_image2prim -v
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
import unittest

import numpy as np
from PIL import Image

import spec
from tokens.draw_codec import (Primitive, primitives_to_tokens,
                               tokens_to_primitives)
from prims.image_to_primitives import image_to_primitives


def _red_circle_target(path="/tmp/garg_fit_target.png") -> Image.Image:
    """合成靶图：灰底 + 居中红圆（用 draw_codec 渲染，保证可拟合）。"""
    from tokens.draw_codec import primitives_to_png
    truth = [Primitive(0, 0, 0, 0, -1, 0.0, 128, 128, 128),
             Primitive(128, 128, 120, 120, 1, 0.0, 230, 50, 50)]
    primitives_to_png(truth, path)
    return Image.open(path)


class TestImageToPrimitives(unittest.TestCase):
    def _solid(self, rgb, size=(128, 128)):
        return Image.new("RGB", size, rgb)

    def test_background_is_mean_color(self):
        # 纯色图：背景 = 整除平均色（8bit 直接入 token，无量化损失）
        prims, note = image_to_primitives(self._solid((40, 80, 120)),
                                          max_steps=0, backend="cpu")
        bg = prims[0]
        self.assertEqual(bg.shape_type, -1)
        self.assertEqual((bg.r, bg.g, bg.b), (40, 80, 120))
        self.assertEqual(len(prims), 1)          # max_steps=0 → 只有背景

    def test_fit_improves_and_deterministic(self):
        img = _red_circle_target()
        r1, note1 = image_to_primitives(img, max_steps=12, seed=7,
                                        backend="cpu")
        r2, _ = image_to_primitives(img, max_steps=12, seed=7,
                                    backend="cpu")
        # 灰底红圆是易拟合目标：12 步内必须有采纳（背景之外 ≥1 个图元）
        self.assertGreater(len(r1), 1)
        self.assertLessEqual(len(r1), 13)        # 采纳数 ≤ 步数 + 背景
        # 同种子完全可复现
        self.assertEqual(r1, r2)
        self.assertIn("残差 MSE", note1)

    def test_tokens_roundtrip_and_ranges(self):
        prims, _ = image_to_primitives(self._solid((10, 20, 30)),
                                       max_steps=4, seed=1, backend="cpu")
        ids = primitives_to_tokens(prims)
        back = tokens_to_primitives(ids)
        self.assertEqual(len(ids),
                         len(prims) * spec.DRAW_TOKENS_PER_PRIMITIVE)
        self.assertEqual(len(back), len(prims))
        for p in back:
            self.assertTrue(0 <= p.cx < spec.CANVAS_W)
            self.assertTrue(0 <= p.cy < spec.CANVAS_H)
            self.assertIn(p.shape_type, (-1, 0, 1))
            self.assertTrue(0 <= p.r <= 255)
            self.assertTrue(0 <= p.g <= 255)
            self.assertTrue(0 <= p.b <= 255)
        # 背景行 schema：几何清零、shape_type=-1
        self.assertEqual(back[0].shape_type, -1)
        self.assertEqual((back[0].cx, back[0].cy, back[0].w, back[0].h),
                         (0.0, 0.0, 0.0, 0.0))

    def test_canvas_is_256(self):
        # 画布照搬仓库 256×256：图元参数就在该量级，无需缩放
        prims, _ = image_to_primitives(self._solid((0, 0, 0)),
                                       max_steps=3, seed=3, backend="cpu")
        self.assertEqual((spec.CANVAS_W, spec.CANVAS_H), (256, 256))
        for p in prims[1:]:
            self.assertLessEqual(p.cx, spec.CANVAS_W)
            self.assertLessEqual(p.cy, spec.CANVAS_H)
            self.assertLessEqual(p.w, 128.0)     # 仓库 w/h ∈ [0,128)
            self.assertLessEqual(p.h, 128.0)

    def test_step_cap_respected(self):
        # 采纳图元数永远不会超过 max_steps（无改善的步记 dummy）
        prims, _ = image_to_primitives(_red_circle_target(),
                                       max_steps=5, seed=11, backend="cpu")
        self.assertLessEqual(len(prims), 5 + 1)

    def test_backend_validation(self):
        with self.assertRaises(ValueError):
            image_to_primitives(self._solid((1, 2, 3)), backend="bogus")


if __name__ == "__main__":
    unittest.main()
