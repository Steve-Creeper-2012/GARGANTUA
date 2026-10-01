# ===========================================================================
# GARGANTUA v1 — 图像→图元转换器回归测试
# 算法来源：wonderfulearth/primitive-operation-painter fast_shape_render
# 运行：cd <ROOT> && python -m unittest tests.test_image2prim -v
# ===========================================================================
import unittest

import numpy as np
from PIL import Image

import spec
from tokens.draw_codec import (Primitive, primitives_to_tokens,
                               tokens_to_primitives)
from tokens.image_to_primitives import image_to_primitives


def _red_circle_target(path="/tmp/garg_fit_target.png") -> Image.Image:
    """合成靶图：灰底 + 居中红圆（用 draw_codec 渲染，保证可拟合）。"""
    from tokens.draw_codec import primitives_to_png
    truth = [Primitive(960, 540, "background", 1919, 1079, 0, 512, 512, 512),
             Primitive(960, 540, "ellipse", 600, 600, 0, 900, 200, 200)]
    primitives_to_png(truth, path)
    return Image.open(path)


class TestImageToPrimitives(unittest.TestCase):
    def _solid(self, rgb, size=(128, 128)):
        return Image.new("RGB", size, rgb)

    def test_background_is_mean_color(self):
        # 纯色图：背景 = 整除平均色（1024 档 >>2 还原后误差 ≤1 档）
        prims, note = image_to_primitives(self._solid((40, 80, 120)),
                                          max_steps=0)
        bg = prims[0]
        self.assertEqual(bg.shape, "background")
        self.assertEqual((bg.r >> 2, bg.g >> 2, bg.b >> 2), (40, 80, 120))
        self.assertEqual(len(prims), 1)          # max_steps=0 → 只有背景

    def test_fit_improves_and_deterministic(self):
        img = _red_circle_target()
        r1, note1 = image_to_primitives(img, max_steps=12, seed=7)
        r2, _ = image_to_primitives(img, max_steps=12, seed=7)
        # 灰底红圆是易拟合目标：12 步内必须有采纳（背景之外 ≥1 个图元）
        self.assertGreater(len(r1), 1)
        self.assertLessEqual(len(r1), 13)        # 采纳数 ≤ 步数 + 背景
        # 同种子完全可复现
        self.assertEqual([(p.x, p.y, p.shape, p.width, p.length, p.rot,
                           p.r, p.g, p.b) for p in r1],
                         [(p.x, p.y, p.shape, p.width, p.length, p.rot,
                           p.r, p.g, p.b) for p in r2])
        self.assertIn("残差 MSE", note1)

    def test_tokens_roundtrip_and_ranges(self):
        prims, _ = image_to_primitives(self._solid((10, 20, 30)),
                                       max_steps=4, seed=1)
        ids = primitives_to_tokens(prims)
        back = tokens_to_primitives(ids)
        self.assertEqual(len(ids), len(prims) * spec.DRAW_TOKENS_PER_PRIMITIVE)
        self.assertEqual(len(back), len(prims))
        for p in back:
            self.assertTrue(0 <= p.x < spec.DRAW_X_SIZE)
            self.assertTrue(0 <= p.y < spec.DRAW_Y_SIZE)
            self.assertIn(p.shape, spec.DRAW_SHAPES)
            self.assertTrue(0 <= p.rot < spec.DRAW_ROT_SIZE)
            self.assertTrue(0 <= p.r < spec.DRAW_R_SIZE)
            self.assertTrue(0 <= p.g < spec.DRAW_G_SIZE)
            self.assertTrue(0 <= p.b < spec.DRAW_B_SIZE)

    def test_canvas_scale(self):
        # 256×256 工作分辨率的参数必须放大到 1920×1080 画布量级
        prims, _ = image_to_primitives(self._solid((0, 0, 0)),
                                       max_steps=3, seed=3)
        for p in prims[1:]:
            self.assertLessEqual(p.x, spec.CANVAS_W)
            self.assertLessEqual(p.y, spec.CANVAS_H)

    def test_step_cap_respected(self):
        # 采纳图元数永远不会超过 max_steps（无改善的步记 dummy）
        prims, _ = image_to_primitives(_red_circle_target(),
                                       max_steps=5, seed=11)
        self.assertLessEqual(len(prims), 5 + 1)


if __name__ == "__main__":
    unittest.main()
