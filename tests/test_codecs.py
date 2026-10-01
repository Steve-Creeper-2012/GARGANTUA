# ===========================================================================
# GARGANTUA v1 — 模态 codec 测试（image / audio / midi / draw）
# 运行：cd ROOT && python -m unittest tests.test_codecs -v
# ===========================================================================

import io
import os
import struct
import sys
import tempfile
import unittest
import wave

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spec
from tokens import audio_codec, draw_codec, image_codec, midi_codec


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
        # delta 帧无变化 → patches 为空
        self.assertEqual(steps[1]["meta"]["patches"], [])
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
# midi_codec
# ---------------------------------------------------------------------------
class TestMidiCodec(unittest.TestCase):
    B = spec.MIDI_BEGIN

    def test_quantizers(self):
        self.assertEqual(midi_codec.vel_to_q(127), 31)
        self.assertEqual(midi_codec.vel_to_q(0), 0)
        for q in range(32):
            self.assertEqual(midi_codec.vel_to_q(midi_codec.q_to_vel(q)), q)
        for q in range(32):
            bpm = midi_codec.q_to_bpm(q)
            self.assertEqual(midi_codec.bpm_to_q(bpm), q)
        self.assertEqual(midi_codec.bpm_to_q(40), 0)
        self.assertEqual(midi_codec.bpm_to_q(240), 31)
        # time_shift：1s 以上拆多个
        toks = midi_codec.ms_to_time_tokens(2500)
        self.assertEqual([t - self.B - 448 for t in toks], [99, 99, 49])
        self.assertEqual(sum(midi_codec.time_token_to_ms(t - self.B - 448)
                             for t in toks), 2500)

    def test_token_offsets(self):
        events = [(0.0, "prog", 5, 0),
                  (0.0, "on", 60, 100),
                  (500.0, "off", 60, 0)]
        ids = midi_codec.events_to_tokens(events)
        self.assertEqual(ids[0], self.B + 288 + 5)          # program
        self.assertEqual(ids[1], self.B + 0 + 60)           # note_on
        self.assertEqual(ids[2], self.B + 256 + midi_codec.vel_to_q(100))
        self.assertEqual(ids[3], self.B + 448 + 49)         # 500ms
        self.assertEqual(ids[4], self.B + 128 + 60)         # note_off

    def test_file_roundtrip_tokens_identical(self):
        ids = [
            self.B + 288 + 5,                               # program 5
            self.B + 416 + midi_codec.bpm_to_q(120),        # tempo
            self.B + 0 + 60, self.B + 256 + 20,             # on 60 vel
            self.B + 448 + 49,                              # 500ms
            self.B + 0 + 64, self.B + 256 + 10,             # on 64
            self.B + 448 + 24,                              # 250ms
            self.B + 128 + 60,                              # off 60
            self.B + 448 + 0,                               # 10ms
            self.B + 128 + 64,                              # off 64
        ]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "t.mid")
            midi_codec.tokens_to_midi(ids, path)
            # 合法 SMF 头
            with open(path, "rb") as f:
                head = f.read(14)
            self.assertEqual(head[:4], b"MThd")
            fmt, ntrk, div = struct.unpack(">HHH", head[8:14])
            self.assertEqual((fmt, ntrk, div), (0, 1, 480))
            back = midi_codec.midi_to_tokens(path)
        self.assertEqual(back, ids)

    def test_parse_ignores_unknown_events(self):
        # 手写一条含控制变化(0xB0)与未知 meta(0x01) 的轨
        track = bytearray()
        track += b"\x00\xB0\x07\x64"                      # CC vol（应忽略）
        track += b"\x00\xFF\x01\x03abc"                   # text meta（忽略）
        track += b"\x00\x90\x3C\x64"                      # on 60 vel 100
        track += b"\x83\x60\x80\x3C\x40"                  # +480 ticks off
        track += b"\x00\xFF\x2F\x00"
        data = (b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480)
                + b"MTrk" + struct.pack(">I", len(track)) + bytes(track))
        with tempfile.NamedTemporaryFile(suffix=".mid",
                                         delete=False) as f:
            f.write(data)
            path = f.name
        try:
            ids = midi_codec.midi_to_tokens(path)
        finally:
            os.unlink(path)
        # 480 ticks @120bpm/480tpq = 500ms
        self.assertEqual(ids, [self.B + 60, self.B + 256 + 25,
                               self.B + 448 + 49, self.B + 128 + 60])


# ---------------------------------------------------------------------------
# draw_codec
# ---------------------------------------------------------------------------
class TestDrawCodec(unittest.TestCase):
    def test_primitive_token_roundtrip(self):
        prims = [
            draw_codec.Primitive(0, 0, "background", 0, 0, 0, 100, 200, 300),
            draw_codec.Primitive(960, 540, "rectangle", 400, 200, 0,
                                 1023, 0, 512),
            draw_codec.Primitive(100, 100, "ellipse", 50, 80, 45, 1, 2, 3),
            draw_codec.Primitive(1919, 1079, "triangle", 60, 60, 359,
                                 7, 8, 9),
        ]
        ids = draw_codec.primitives_to_tokens(prims)
        self.assertEqual(len(ids), 4 * 9)
        back = draw_codec.tokens_to_primitives(ids)
        self.assertEqual(back, prims)

    def test_token_ranges_and_order(self):
        p = draw_codec.Primitive(10, 20, "triangle", 30, 40, 50, 60, 70, 80)
        ids = draw_codec.primitives_to_tokens([p])
        self.assertEqual(ids[0], spec.DRAW_X_BEGIN + 10)
        self.assertEqual(ids[1], spec.DRAW_Y_BEGIN + 20)
        self.assertEqual(ids[2], spec.DRAW_SHAPE_BEGIN
                         + spec.DRAW_SHAPES.index("triangle"))
        self.assertEqual(ids[3], spec.DRAW_W_BEGIN + 30)
        self.assertEqual(ids[4], spec.DRAW_L_BEGIN + 40)
        self.assertEqual(ids[5], spec.DRAW_ROT_BEGIN + 50)
        self.assertEqual(ids[6], spec.DRAW_R_BEGIN + 60)
        self.assertEqual(ids[7], spec.DRAW_G_BEGIN + 70)
        self.assertEqual(ids[8], spec.DRAW_B_BEGIN + 80)
        for t in ids:
            self.assertTrue(0 <= t < spec.VOCAB_SIZE_PADDED)

    def test_clamp_out_of_range(self):
        p = draw_codec.Primitive(5000, -3, "rectangle", 9999, 2000,
                                 400, 2000, -1, 1024)
        with self.assertLogs("tokens.draw_codec", level="WARNING"):
            ids = draw_codec.primitives_to_tokens([p])
        back = draw_codec.tokens_to_primitives(ids)[0]
        self.assertEqual(back.x, spec.DRAW_X_SIZE - 1)      # 1919
        self.assertEqual(back.y, 0)
        self.assertEqual(back.width, spec.DRAW_W_SIZE - 1)
        self.assertEqual(back.length, spec.DRAW_L_SIZE - 1)
        self.assertEqual(back.rot, 359)                     # 非法 rot clamp
        self.assertEqual(back.r, 1023)
        self.assertEqual(back.g, 0)
        self.assertEqual(back.b, 1023)

    def test_png_render_header_and_size(self):
        prims = [
            draw_codec.Primitive(0, 0, "background", 0, 0, 0, 512, 512, 512),
            draw_codec.Primitive(32, 24, "rectangle", 20, 10, 0,
                                 1023, 0, 0),
            draw_codec.Primitive(32, 24, "triangle", 16, 16, 30, 0, 0, 1023),
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
        # 矩形内、三角形外的像素为红色（1023>>2=255），Pillow 可解码验证
        img = Image.open(io.BytesIO(data)).convert("RGBA")
        self.assertEqual(img.getpixel((24, 24))[:3], (255, 0, 0))

    def test_triangle_rotation_hit(self):
        # rot=90 的三角形：渲染缓冲内应有三角形颜色的像素
        prims = [draw_codec.Primitive(32, 24, "triangle", 20, 20, 90,
                                      0, 1023, 0)]
        buf = draw_codec.render_rgba(prims, w=64, h=48)
        arr = np.frombuffer(bytes(buf), dtype=np.uint8).reshape(48, 64, 4)
        green = (arr[:, :, 1] == 255) & (arr[:, :, 0] == 0)
        self.assertTrue(green.sum() > 0)

    def test_svg_elements(self):
        prims = [
            draw_codec.Primitive(0, 0, "background", 0, 0, 0, 0, 0, 0),
            draw_codec.Primitive(10, 10, "rectangle", 8, 6, 0, 4, 4, 4),
            draw_codec.Primitive(20, 20, "ellipse", 10, 10, 0, 8, 8, 8),
            draw_codec.Primitive(30, 30, "triangle", 10, 12, 90, 12, 12, 12),
        ]
        svg = draw_codec.primitives_to_svg(prims, w=64, h=48)
        self.assertTrue(svg.startswith("<svg"))
        self.assertEqual(svg.count("<rect"), 2)             # background+rect
        self.assertIn("<ellipse", svg)
        self.assertIn("<polygon", svg)                      # triangle


if __name__ == "__main__":
    unittest.main()
