# ===========================================================================
# GARGANTUA v1 — server_codecs.py：编解码 API（多模态输入 → token 序列）
# ---------------------------------------------------------------------------
# 从 server.py 拆出。这里只回答「这份 base64 是什么模态、切成哪些时间步」，
# 不碰模型、不碰训练、不碰 HTTP 框架。App 以 mixin 方式继承本类：
#
#     class App(TrainApi, CodecApi): ...
#
# 因此所有方法里的 self.mgr / self.pool / self.queue_dir 都由 App.__init__
# 提供，本模块不持有任何状态。
# 尺寸/规模守卫与媒体解码见 server_media.py。
# ===========================================================================
import base64
import os
import tempfile

import spec
from tokens import image_codec, audio_codec, midi_codec, draw_codec
from tokens.image_to_primitives import image_to_primitives
from server_media import (MAX_PATCHES_TOTAL, b64_to_pil, decode_audio_any,
                          extract_video_audio, extract_video_frames)
from server_models import step_to_json
from server_util import num_param


class CodecApi:
    """ /api/codec/* 全部端点：图片/帧序列/音频/MIDI/图元/文本/视频/图元拟合。 """

    def api_codec_image(self, body):
        b64 = body.get("image_b64")
        if not b64:
            raise ValueError("缺少 image_b64")
        step = image_codec.image_to_step(b64_to_pil(b64))
        if len(step["meta"]["patches"]) > MAX_PATCHES_TOTAL:
            raise ValueError(
                f"patch 数超限: {len(step['meta']['patches'])} > "
                f"{MAX_PATCHES_TOTAL}")
        return 200, {"steps": [step_to_json(step)],
                     "archive": image_codec.step_to_archive(step)}

    def api_codec_frames(self, body):
        frames_b64 = body.get("frames_b64")
        if not frames_b64:
            raise ValueError("缺少 frames_b64")
        frames = [b64_to_pil(b) for b in frames_b64]
        if len(frames) > 300:
            raise ValueError(f"帧数超限: {len(frames)} > 300")
        steps = image_codec.frames_to_steps(frames)
        total = sum(len(s["meta"]["patches"]) for s in steps)
        if total > MAX_PATCHES_TOTAL:
            raise ValueError(f"patch 总数超限: {total} > {MAX_PATCHES_TOTAL}")
        stats = [{"frame_idx": s["meta"]["frame_idx"],
                  "full": s["meta"]["full"],
                  "grid": s["meta"]["grid"],
                  "n_patches": len(s["meta"]["patches"])} for s in steps]
        return 200, {"frames": stats,
                     "delta_rule": ("第 1 帧全量；2..30 帧只送有变化 patch；"
                                    "第 31 帧（每秒首帧）全量；全变帧全量并重新"
                                    "计数；最后一帧强制全量"),
                     "steps": [step_to_json(s) for s in steps]}

    def api_codec_audio(self, body):
        """音频 → 音元 token 序列。支持 wav/mp3/flac/ogg（ffmpeg 强兼容）。"""
        b64 = body.get("audio_b64")
        if not b64:
            raise ValueError("缺少 audio_b64")
        data = base64.b64decode(b64)
        filename = body.get("filename", "audio.wav")
        suffix = os.path.splitext(filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            apath = f.name
        try:
            samples = decode_audio_any(apath, filename, spec.AUDIO_SR)
        finally:
            os.unlink(apath)
        steps = audio_codec.waveform_to_steps(samples)
        n = len(steps)
        return 200, {
            "n_tokens": n,
            "duration_s": round(n * spec.AUDIO_SAMPLES_PER_TOKEN
                                / spec.AUDIO_SR, 3),
            "filename": filename,
            "steps": [step_to_json(s) for s in steps],
        }

    def api_codec_midi(self, body):
        if "midi_b64" in body:
            data = base64.b64decode(body["midi_b64"])
            with tempfile.NamedTemporaryFile(suffix=".mid",
                                             delete=False) as f:
                f.write(data)
                tmp = f.name
            try:
                tokens = midi_codec.midi_to_tokens(tmp)
            finally:
                os.unlink(tmp)
            return 200, {"tokens": tokens, "n": len(tokens)}
        if "tokens" in body:
            tokens = [int(t) for t in body["tokens"]]
            with tempfile.NamedTemporaryFile(suffix=".mid",
                                             delete=False) as f:
                tmp = f.name
            try:
                midi_codec.tokens_to_midi(tokens, tmp)
                with open(tmp, "rb") as f:
                    data = f.read()
            finally:
                os.unlink(tmp)
            return 200, {"midi_b64": base64.b64encode(data).decode(),
                         "n": len(tokens)}
        raise ValueError("缺少 midi_b64 或 tokens")

    def api_codec_draw(self, body):
        if "primitives" in body:
            prims = [draw_codec.Primitive(
                x=p.get("x", 0), y=p.get("y", 0),
                shape=p.get("shape", "rectangle"),
                width=p.get("width", 0), length=p.get("length", 0),
                rot=p.get("rot", 0), r=p.get("r", 0),
                g=p.get("g", 0), b=p.get("b", 0))
                for p in body["primitives"]]
        elif "tokens" in body:
            prims = draw_codec.tokens_to_primitives(
                [int(t) for t in body["tokens"]])
        else:
            raise ValueError("缺少 primitives 或 tokens")
        tokens = draw_codec.primitives_to_tokens(prims)
        buf = draw_codec.render_rgba(prims)
        png = draw_codec.encode_png(bytes(buf), spec.CANVAS_W, spec.CANVAS_H)
        return 200, {
            "png_b64": base64.b64encode(png).decode(),
            "svg": draw_codec.primitives_to_svg(prims),
            "tokens": tokens,
            "n_primitives": len(prims),
        }

    # ---------------- text codec ----------------
    def api_codec_text(self, body):
        from tokens.text_codec import encode as text_encode
        text = body.get("text")
        if text is None:
            raise ValueError("缺少 text")
        steps = text_encode(text)
        return 200, {"steps": [step_to_json(s) for s in steps]}

    # ---------------- video codec ----------------
    def api_codec_video(self, body):
        b64 = body.get("video_b64")
        if not b64:
            raise ValueError("缺少 video_b64")
        data = base64.b64decode(b64)
        suffix = os.path.splitext(body.get("filename", ".mp4"))[1] or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            vpath = f.name
        try:
            frames = extract_video_frames(vpath)
            audio_samples = extract_video_audio(vpath)
            visual_steps = image_codec.frames_to_steps(frames)
            if sum(len(s["meta"]["patches"]) for s in visual_steps) \
                    > MAX_PATCHES_TOTAL:
                raise ValueError(f"视频 patch 总数超限: > {MAX_PATCHES_TOTAL}")
            audio_steps = audio_codec.waveform_to_steps(audio_samples)
            aligned = audio_codec.align_audio_to_frames(audio_steps, len(frames))
            # 合并：每一帧的 visual step + 对应音频 steps 作为同一时间步列表
            combined = []
            total_patches = 0
            for vstep, asteps in zip(visual_steps, aligned):
                total_patches += len(vstep["meta"].get("patches", []))
                combined.append({
                    "time_step": [step_to_json(vstep)] +
                                 [step_to_json(a) for a in asteps]
                })
            return 200, {
                "n_frames": len(frames),
                "n_audio_tokens": len(audio_steps),
                "total_patches": total_patches,
                "steps": combined,
            }
        finally:
            os.unlink(vpath)

    # ---------------- image -> primitives ----------------
    def api_codec_image2prims(self, body):
        b64 = body.get("image_b64")
        if not b64:
            raise ValueError("缺少 image_b64")
        max_steps = num_param(body, "max_steps", 100, int)
        quality = body.get("quality") or "fast"
        # 此前非法 quality 被静默改成 fast，用户以为自己选了 slow
        if quality not in ("fast", "slow"):
            raise ValueError(f"quality 必须是 fast/slow: {quality!r}")
        if not (1 <= max_steps <= 1000):
            raise ValueError(f"max_steps 必须在 [1, 1000]: {max_steps}")
        tid = self.pool.submit(
            lambda: self._image2prims(b64, max_steps, quality))
        return 202, {"task_id": tid}

    def _image2prims(self, b64, max_steps, quality):
        prims, note = image_to_primitives(
            b64_to_pil(b64), max_steps=max_steps, quality=quality)
        tokens = draw_codec.primitives_to_tokens(prims)
        return {
            "primitives": [{"x": p.x, "y": p.y, "shape": p.shape,
                            "width": p.width, "length": p.length,
                            "rot": p.rot, "r": p.r, "g": p.g, "b": p.b}
                           for p in prims],
            "n_primitives": len(prims),
            "tokens": tokens,
            "svg": draw_codec.primitives_to_svg(prims),
            "note": note,
        }
