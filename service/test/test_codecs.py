# ===========================================================================
# GARGANTUA v1 — 模态 codec 测试（image / audio / draw）
# 运行：cd ROOT && python -m unittest tests.test_codecs -v
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
import io
import math
import os
import struct
import sys
import tempfile
import unittest
import wave

import numpy as np
from PIL import Image


import spec
from tokens import audio_codec, draw_codec, image_codec


def make_frame(color, size=(64, 64)):
    img = Image.new("RGBA", size, color)
    return img


def frame_with_block(base_color, block_color, block_rect, size=(64, 64)):
    """base 色画布，在 block_rect=(x0,y0,x1,y1) 画 block_color 方块。"""
    arr = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    arr[:, :] = base_color
    x0, y0, x1, y1 = block_rect
    arr[y0:y1, x0:x1] = block_color
    return Image.fromarray(arr, "RGBA")


def make_wav_bytes(samples, sr=16000, nch=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        data = np.asarray(samples, dtype=np.int16)
        if nch > 1:
            data = np.tile(data[:, None], (1, nch))
        wf.writeframes(data.astype("<i2").tobytes())
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# image_codec
# ---------------------------------------------------------------------------
class TestImageCodec(unittest.TestCase):
    PATCHES_PER_FRAME = 4                       # 64x64 → 2x2

    def test_single_image_patches(self):
        step = image_codec.image_to_step(make_frame((255, 0, 0, 255)))
        self.assertTrue(step["meta"]["full"])
        self.assertEqual(step["token_ids"], [])
        self.assertEqual(step["type"], spec.TYPE_USER)
        self.assertEqual(step["modality"], "visual")
        self.assertEqual(step["meta"]["grid"], [2, 2])
        patches = step["meta"]["patches"]
        self.assertEqual(len(patches), 4)
        # 行优先坐标顺序
        self.assertEqual([(p["x"], p["y"]) for p in patches],
                         [(0, 0), (1, 0), (0, 1), (1, 1)])
        # vec 长度 4096 且内容正确（全红 patch）
        for p in patches:
            self.assertEqual(len(p["vec"]), 4096)
            self.assertEqual(p["vec"][:4], bytes((255, 0, 0, 255)))

    def test_odd_size_padding(self):
        # 33x33 → pad 到 64x64，grid 2x2；pad 区域透明
        img = Image.new("RGBA", (33, 33), (10, 20, 30, 255))
        step = image_codec.image_to_step(img)
        self.assertEqual(step["meta"]["grid"], [2, 2])
        arr = image_codec.patches_to_rgba_array(step)
        self.assertEqual(arr.shape, (64, 64, 4))
        self.assertEqual(tuple(arr[0, 0]), (10, 20, 30, 255))
        self.assertEqual(tuple(arr[33, 33]), (0, 0, 0, 0))   # pad 透明

    def test_image_path_input(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            make_frame((1, 2, 3, 255)).save(path)
            step = image_codec.image_to_step(path)
            self.assertEqual(len(step["meta"]["patches"]), 4)
            self.assertEqual(step["meta"]["patches"][0]["vec"][:4],
                             bytes((1, 2, 3, 255)))
        finally:
            os.unlink(path)

    def test_delta_second_frame_changed_only(self):
        f0 = make_frame((0, 0, 0, 255))
        # 只改 patch (1, 0)（像素 32..63, 0..31）
        f1 = frame_with_block((0, 0, 0, 255), (255, 255, 255, 255),
                              (32, 0, 64, 32))
        # 第三帧与 f1 相同，仅用于避免末帧强制全量干扰中间帧断言
        steps = image_codec.frames_to_steps([f0, f1, f1])
        self.assertTrue(steps[0]["meta"]["full"])
        self.assertEqual(len(steps[0]["meta"]["patches"]), 4)
        self.assertFalse(steps[1]["meta"]["full"])
        got = [(p["x"], p["y"]) for p in steps[1]["meta"]["patches"]]
        self.assertEqual(got, [(1, 0)])
        self.assertTrue(steps[2]["meta"]["full"])       # 末帧强制全量

    def test_delta_31st_frame_full(self):
        # 35 帧：帧 0 全量，帧 1..29 无变化(delta)，帧 30(第31帧)全量，
        # 帧 31..33 delta，帧 34 末帧强制全量
        frames = [make_frame((7, 7, 7, 255)) for _ in range(35)]
        steps = image_codec.frames_to_steps(frames)
        full_flags = [s["meta"]["full"] for s in steps]
        self.assertTrue(full_flags[0])
        self.assertTrue(all(not f for f in full_flags[1:30]))
        self.assertTrue(full_flags[30])                 # 第 31 帧全量
        self.assertTrue(all(not f for f in full_flags[31:34]))
        self.assertTrue(full_flags[34])                 # 末帧强制全量
        # 无变化的 delta 帧：时间步必须非空（.tokenvector 不接受空步），
        # 送 1 个同值占位 patch 并标记 static
        self.assertTrue(steps[1]["meta"]["static"])
        self.assertEqual(len(steps[1]["meta"]["patches"]), 1)
        self.assertEqual(steps[1]["meta"]["patches"][0]["x"], 0)
        # 全量帧带全部 patch
        self.assertEqual(len(steps[30]["meta"]["patches"]), 4)

    def test_all_changed_resets_counter(self):
        f0 = make_frame((0, 0, 0, 255))
        f1 = make_frame((255, 255, 255, 255))           # 全变 → 重置为第 1 帧
        f2 = frame_with_block((255, 255, 255, 255), (0, 0, 0, 255),
                              (0, 0, 32, 32))           # 只变 patch (0,0)
        f3 = frame_with_block((255, 255, 255, 255), (0, 0, 0, 255),
                              (0, 0, 32, 32))           # 与 f2 相同（末帧）
        steps = image_codec.frames_to_steps([f0, f1, f2, f3])
        self.assertTrue(steps[1]["meta"]["full"])       # 全变全量
        self.assertFalse(steps[2]["meta"]["full"])      # 重置后第 2 帧 delta
        self.assertEqual([(p["x"], p["y"])
                          for p in steps[2]["meta"]["patches"]], [(0, 0)])
        self.assertTrue(steps[3]["meta"]["full"])       # 末帧强制全量

    def test_last_frame_forced_full(self):
        f = make_frame((9, 9, 9, 255))
        steps = image_codec.frames_to_steps([f, f])     # 两帧完全相同
        self.assertTrue(steps[1]["meta"]["full"])
        self.assertEqual(len(steps[1]["meta"]["patches"]), 4)

    def test_archive_quantization(self):
        # 全白 patch：均值 255 → R/G/B 1023，A 999
        step = image_codec.image_to_step(make_frame((255, 255, 255, 255)))
        arch = image_codec.step_to_archive(step)
        self.assertIn("visual", arch)
        for rec in arch["visual"]:
            self.assertEqual((rec["R"], rec["G"], rec["B"]),
                             (spec.VISUAL_R_QUANT - 1,) * 3)
            self.assertEqual(rec["A"], spec.VISUAL_A_QUANT - 1)
        # 中间值：128 → round(128/255*1023) = 514
        step = image_codec.image_to_step(make_frame((128, 0, 0, 255)))
        rec = image_codec.step_to_archive(step)["visual"][0]
        self.assertEqual(rec["R"], round(128 / 255 * 1023))
        self.assertTrue(0 <= rec["R"] < spec.VISUAL_R_QUANT)
        self.assertTrue(0 <= rec["A"] < spec.VISUAL_A_QUANT)

    def test_patches_to_rgba_array_delta_rebuild(self):
        f0 = make_frame((5, 5, 5, 255))
        f1 = frame_with_block((5, 5, 5, 255), (200, 100, 50, 255),
                              (32, 32, 64, 64))         # 改 patch (1,1)
        steps = image_codec.frames_to_steps([f0, f1])
        base = image_codec.patches_to_rgba_array(steps[0])
        rebuilt = image_codec.patches_to_rgba_array(steps[1], base)
        truth = np.asarray(f1, dtype=np.uint8)
        np.testing.assert_array_equal(rebuilt, truth)


# ---------------------------------------------------------------------------
# audio_codec
# ---------------------------------------------------------------------------
class TestAudioCodec(unittest.TestCase):
    def test_wav_roundtrip(self):
        rng = np.random.default_rng(42)
        samples = rng.integers(-32768, 32767, size=1200, dtype=np.int16)
        buf = make_wav_bytes(samples, sr=16000)
        steps = audio_codec.audio_to_steps(buf)
        self.assertEqual(len(steps), -(-1200 // spec.AUDIO_SAMPLES_PER_TOKEN))
        self.assertEqual(steps[-1]["meta"]["n"], 1200 % 533)   # 真实长度
        self.assertEqual(len(steps[-1]["meta"]["samples"]), 533 * 4)
        wave_out = audio_codec.steps_to_waveform(steps)
        self.assertEqual(len(wave_out), 1200)
        back = np.rint(wave_out * 32768.0).astype(np.int32)
        np.testing.assert_array_equal(back, samples.astype(np.int32))

    def test_resample_8k_to_16k(self):
        n = 800
        samples = (np.sin(np.arange(n) * 0.1) * 10000).astype(np.int16)
        buf = make_wav_bytes(samples, sr=8000)
        out = audio_codec.load_audio(buf)
        self.assertAlmostEqual(len(out), 1600, delta=2)

    def test_int16_array_input(self):
        samples = np.arange(533, dtype=np.int16) * 10
        steps = audio_codec.waveform_to_steps(samples)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["meta"]["n"], 533)
        self.assertEqual(steps[0]["token_ids"], [])

    def test_stereo_mixed_to_mono(self):
        samples = np.full(100, 1000, dtype=np.int16)
        buf = make_wav_bytes(samples, sr=16000, nch=2)
        out = audio_codec.load_audio(buf)
        self.assertEqual(len(out), 100)
        np.testing.assert_array_equal(out, samples)

    def test_align_audio_to_frames(self):
        steps = audio_codec.waveform_to_steps(np.zeros(533 * 10, np.int16))
        self.assertEqual(len(steps), 10)
        frames = audio_codec.align_audio_to_frames(steps, 3)
        self.assertEqual(len(frames), 3)
        self.assertEqual([len(f) for f in frames], [4, 3, 3])
        self.assertEqual(sum(len(f) for f in frames), 10)   # 总数守恒
        # 少帧多空列表
        frames = audio_codec.align_audio_to_frames(steps[:2], 4)
        self.assertEqual([len(f) for f in frames], [1, 1, 0, 0])


# ---------------------------------------------------------------------------
# draw_codec
# ---------------------------------------------------------------------------
class TestDrawCodec(unittest.TestCase):
    """图元 codec：token 布局照搬 primitive-operation-painter
    geometrize_256_v1（cx,cy,theta,w,h,shape,r,g,b + special 3 bins）。"""

    B = spec.DRAW_BEGIN

    def test_primitive_token_roundtrip(self):
        # bin 精确值（cx 偶数、w/h 为 0.25 倍数、角度为 1/3 度倍数、
        # 颜色偶数）→ roundtrip 严格相等
        prims = [
            draw_codec.Primitive(0, 0, 0, 0, -1, 0.0, 100, 200, 250),
            draw_codec.Primitive(128, 96, 40, 20, 0, math.radians(20),
                                 254, 0, 128),
            draw_codec.Primitive(60, 180, 30, 50, 1, math.radians(12),
                                 10, 20, 30),
        ]
        ids = draw_codec.primitives_to_tokens(prims)
        self.assertEqual(len(ids), 3 * 9)
        back = draw_codec.tokens_to_primitives(ids)
        self.assertEqual(back, prims)

    def test_token_ranges_and_order(self):
        p = draw_codec.Primitive(cx=10, cy=20, w=8, h=12, shape_type=0,
                                 theta=math.radians(20), r=60, g=70, b=80)
        ids = draw_codec.primitives_to_tokens([p])
        b = self.B
        self.assertEqual(ids[0], b + spec.DRAW_X_OFFSET + 20)      # 10*2
        self.assertEqual(ids[1], b + spec.DRAW_Y_OFFSET + 40)      # 20*2
        self.assertEqual(ids[2], b + spec.DRAW_ANGLE_OFFSET + 60)  # 20*3
        self.assertEqual(ids[3], b + spec.DRAW_W_OFFSET + 32)      # 8*4
        self.assertEqual(ids[4], b + spec.DRAW_H_OFFSET + 48)      # 12*4
        self.assertEqual(ids[5], b + spec.DRAW_SHAPE_OFFSET + 1)   # 0+1
        self.assertEqual(ids[6], b + spec.DRAW_R_OFFSET + 30)      # 60//2
        self.assertEqual(ids[7], b + spec.DRAW_G_OFFSET + 35)      # 70//2
        self.assertEqual(ids[8], b + spec.DRAW_B_OFFSET + 40)      # 80//2
        for t in ids:
            self.assertTrue(0 <= t < spec.VOCAB_SIZE_PADDED)

    def test_theta_over_90_swaps_size(self):
        # theta ≥ 90°：归一到 [0,90) 并交换宽高（与仓库 encode 一致）
        p = draw_codec.Primitive(cx=100, cy=100, w=8, h=40, shape_type=0,
                                 theta=math.radians(100), r=0, g=0, b=0)
        ids = draw_codec.primitives_to_tokens([p])
        b = self.B
        self.assertEqual(ids[2] - b - spec.DRAW_ANGLE_OFFSET, 30)   # 100-90=10° ×3
        self.assertEqual(ids[3] - b - spec.DRAW_W_OFFSET, 160)      # 40*4（交换后）
        self.assertEqual(ids[4] - b - spec.DRAW_H_OFFSET, 32)       # 8*4

    def test_special_bin_stops_decode(self):
        # 步首 token 落入 special bin 区 → 序列结束（仓库 decode 语义）
        p = draw_codec.Primitive(cx=64, cy=64, w=16, h=16, shape_type=1,
                                 theta=0.0, r=8, g=8, b=8)
        ids = draw_codec.primitives_to_tokens([p])
        ids += [self.B + spec.DRAW_SPECIAL_OFFSET] + [0] * 8
        back = draw_codec.tokens_to_primitives(ids)
        self.assertEqual(back, [p])

    def test_clamp_out_of_range(self):
        p = draw_codec.Primitive(5000, -3, 9999, 2000, 7, 99.0,
                                 2000, -1, 300)
        with self.assertLogs("tokens.draw_codec", level="WARNING"):
            ids = draw_codec.primitives_to_tokens([p])
        back = draw_codec.tokens_to_primitives(ids)[0]
        b = self.B
        self.assertEqual(back.cx, 255.0)                    # clamp 后 bin 510
        self.assertEqual(back.cy, 0.0)
        self.assertEqual(back.w, (spec.DRAW_W_BINS - 1)
                         / spec.SIZE_BINS_PER_PIXEL)        # 127.75
        self.assertEqual(back.h, (spec.DRAW_H_BINS - 1)
                         / spec.SIZE_BINS_PER_PIXEL)
        self.assertEqual(back.shape_type, 0)                 # 非法 → 矩形
        self.assertEqual(back.r, 254)                       # 255//2=127 → 254
        self.assertEqual(back.g, 0)
        self.assertEqual(back.b, 254)

    def test_csv_roundtrip(self):
        prims = [
            draw_codec.Primitive(0, 0, 0, 0, -1, 0.0, 12, 34, 56),
            draw_codec.Primitive(33.5, 64.0, 40, 20, 0, math.radians(20),
                                 200, 100, 50),
        ]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "seq.csv")
            draw_codec.primitives_to_csv(prims, path, image_name="t")
            with open(path, encoding="utf-8") as f:
                header = f.readline().strip()
            self.assertEqual(header,
                             "image_name,cx,cy,w,h,shape_type,theta,r,g,b")
            back = draw_codec.csv_to_primitives(path)
        self.assertEqual(len(back), 2)
        self.assertEqual(back[0].shape_type, -1)
        self.assertAlmostEqual(back[1].cx, 33.5, places=6)
        self.assertAlmostEqual(back[1].theta, math.radians(20), places=6)
        # CSV → token → 解码 → 再编码：ID 流幂等（bin 量化吸收精度）
        ids = draw_codec.primitives_to_tokens(back)
        back2 = draw_codec.tokens_to_primitives(ids)
        self.assertEqual(back2, back)
        self.assertEqual(draw_codec.primitives_to_tokens(back2), ids)

    def test_png_render_header_and_size(self):
        prims = [
            draw_codec.Primitive(0, 0, 0, 0, -1, 0.0, 0, 0, 0),
            draw_codec.Primitive(32, 24, 20, 10, 0, 0.0, 255, 0, 0),
            draw_codec.Primitive(32, 24, 16, 16, 1, math.radians(30),
                                 0, 0, 255),
        ]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.png")
            draw_codec.primitives_to_png(prims, path, w=64, h=48)
            with open(path, "rb") as f:
                data = f.read()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")    # 合法 PNG 头
        self.assertEqual(data[12:16], b"IHDR")
        w, h, bitdepth, ctype = struct.unpack(">IIBB", data[16:26])
        self.assertEqual((w, h, bitdepth, ctype), (64, 48, 8, 6))
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        self.assertEqual(img.getpixel((23, 20))[:3], (255, 0, 0))

    def test_ellipse_rotation_hit(self):
        # 旋转椭圆：渲染缓冲内应有椭圆颜色的像素
        prims = [draw_codec.Primitive(32, 24, 20, 20, 1, math.radians(45),
                                      0, 255, 0)]
        buf = draw_codec.render_rgba(prims, w=64, h=48)
        arr = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(48, 64, 4)
        green = (arr[:, :, 1] == 255) & (arr[:, :, 0] == 0)
        self.assertTrue(green.sum() > 0)

    def test_svg_elements(self):
        prims = [
            draw_codec.Primitive(0, 0, 0, 0, -1, 0.0, 0, 0, 0),
            draw_codec.Primitive(10, 10, 8, 6, 0, 0.0, 4, 4, 4),
            draw_codec.Primitive(20, 20, 10, 10, 1, math.radians(90),
                                 8, 8, 8),
        ]
        svg = draw_codec.primitives_to_svg(prims, w=64, h=48)
        self.assertTrue(svg.startswith("<svg"))
        self.assertEqual(svg.count("<rect"), 2)             # background+rect
        self.assertIn("<ellipse", svg)
        self.assertIn("rotate(", svg)                        # 旋转 transform


if __name__ == "__main__":
    unittest.main()
