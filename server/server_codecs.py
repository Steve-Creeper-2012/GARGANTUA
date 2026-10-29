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
from tokens import audio_codec, draw_codec, image_codec, tok5
from prims.image_to_primitives import image_to_primitives
from server_media import (MAX_AUDIO_TOKENS, MAX_PATCHES_TOTAL, b64_to_pil,
                          decode_audio_any, extract_video_audio,
                          extract_video_frames)
from server_models import step_to_json
from server_util import num_param

# 2026-10-23 审计修复（P0）：/api/codec/draw 原是同步端点，render_rgba
# 逐图元向量化绘制 ≈129ms/个（实测 100 个全画布图元 12.9s，全程占用
# HTTP handler 线程），且无数量上限——几 KB 的请求体即可拖死线程池。
# 现在与 image2prims 一致走任务池（202 + task_id），并设图元数上限。
MAX_PRIMITIVES = 1000


def token5_text(steps) -> str:
    """Steps → token-5 可读文本（{text: 00001} / {wave: HEX} / {visual: …}）。

    前端不再自己拼格式：hex 载荷（2132 / 10240 字符）在 JS 里拼既慢又容易
    和 Python 侧的位序不一致，统一由后端给出唯一真相。
    """
    from tokens.vocab import get_vocab
    items = tok5.from_steps(steps)
    return tok5.to_display(items, name_of=tok5.vocab_name_of(get_vocab()))


class CodecApi:
    """ /api/codec/* 全部端点：图片/帧序列/音频/图元/文本/视频/图元拟合。 """

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
                     "token5": token5_text([step]),
                     "archive": image_codec.step_to_archive(step)}

    def api_codec_frames(self, body):
        frames_b64 = body.get("frames_b64")
        if not frames_b64:
            raise ValueError("缺少 frames_b64")
        # 2026-10-27 审计修复：帧数与像素预算在解码【之前】检查——旧实现
        # 先把全部帧完整解码成 PIL（300 帧 × 8192² 可达数十 GB）才数帧数
        if len(frames_b64) > 300:
            raise ValueError(f"帧数超限: {len(frames_b64)} > 300")
        from server_media import MAX_TOTAL_PIXELS
        frames = []
        total_px = 0
        for b in frames_b64:
            im = b64_to_pil(b)          # 单帧尺寸守卫（解码前先查头部）
            total_px += im.size[0] * im.size[1]
            if total_px > MAX_TOTAL_PIXELS:
                raise ValueError(
                    f"帧序列总像素超限: > {MAX_TOTAL_PIXELS}"
                    f"（约 {MAX_TOTAL_PIXELS // 1048576}MP）")
            frames.append(im)
        steps = image_codec.frames_to_steps(frames)
        total = sum(len(s["meta"]["patches"]) for s in steps)
        if total > MAX_PATCHES_TOTAL:
            raise ValueError(f"patch 总数超限: {total} > {MAX_PATCHES_TOTAL}")
        stats = [{"frame_idx": s["meta"]["frame_idx"],
                  "full": s["meta"]["full"],
                  "grid": s["meta"]["grid"],
                  # 2026-10-27（M3）：与上一帧逐像素相同 → delta 无变化，
                  # 用首 patch 占位保证时间步非空，此标记供前端/存档识别
                  "static": bool(s["meta"].get("static")),
                  "n_patches": len(s["meta"]["patches"])} for s in steps]
        return 200, {"frames": stats,
                     "delta_rule": ("第 1 帧全量；2..30 帧只送有变化 patch；"
                                    "第 31 帧（每秒首帧）全量；全变帧全量并重新"
                                    "计数；最后一帧强制全量；静止帧（逐像素相同）"
                                    "送 1 个占位 patch 并标记 static"),
                     "steps": [step_to_json(s) for s in steps],
                     "token5": token5_text(steps)}

    def api_codec_audio(self, body):
        """音频 → 音元 token 序列。支持 wav/mp3/flac/ogg（ffmpeg 强兼容）。"""
        b64 = body.get("audio_b64")
        if not b64:
            raise ValueError("缺少 audio_b64")
        data = base64.b64decode(b64)
        filename = body.get("filename", "audio.wav")
        suffix = os.path.splitext(filename)[1] or ".wav"
        # 2026-10-27 审计修复：try/finally 从「创建临时文件」那一刻开始——
        # 旧实现写在 write() 之后，磁盘满/写入异常时临时文件永久泄漏
        apath = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(data)
                apath = f.name
            samples = decode_audio_any(apath, filename, spec.AUDIO_SR)
        finally:
            if apath:
                try:
                    os.unlink(apath)
                except OSError:
                    pass
        steps = audio_codec.waveform_to_steps(samples)
        n = len(steps)
        if n > MAX_AUDIO_TOKENS:
            raise ValueError(
                f"音元数超限: {n} > {MAX_AUDIO_TOKENS}"
                f"（约 {MAX_AUDIO_TOKENS / 30:.0f} 秒），请裁剪音频")
        return 200, {
            "n_tokens": n,
            "duration_s": round(n * spec.AUDIO_SAMPLES_PER_TOKEN
                                / spec.AUDIO_SR, 3),
            "filename": filename,
            "steps": [step_to_json(s) for s in steps],
            "token5": token5_text(steps),
        }

    def api_codec_draw(self, body):
        if "primitives" in body:
            raw = body["primitives"]
            if not isinstance(raw, list):
                raise ValueError("primitives 必须是数组")
            if len(raw) > MAX_PRIMITIVES:
                raise ValueError(f"图元数超限: {len(raw)} > {MAX_PRIMITIVES}")
            for p in raw:
                # 2026-10-23 审计修复（P2）：非 dict 项此前抛
                # AttributeError → HTTP 500，违反 400 入参契约
                if not isinstance(p, dict):
                    raise ValueError(f"每个图元必须是对象: {p!r}")
            prims = [draw_codec.Primitive(
                cx=p.get("cx", 0), cy=p.get("cy", 0),
                w=p.get("w", 0), h=p.get("h", 0),
                shape_type=p.get("shape_type", 0),
                theta=p.get("theta", 0), r=p.get("r", 0),
                g=p.get("g", 0), b=p.get("b", 0))
                for p in raw]
        elif "tokens" in body:
            toks = body["tokens"]
            if not isinstance(toks, list) or not all(
                    isinstance(t, int) and not isinstance(t, bool)
                    for t in toks):
                raise ValueError("tokens 必须是整数数组")
            # 2026-10-27 审计修复：tokens 路径此前绕过 MAX_PRIMITIVES 上限，
            # 一个巨大数组即可拖死渲染任务
            if len(toks) > spec.DRAW_TOKENS_PER_PRIMITIVE * MAX_PRIMITIVES:
                raise ValueError(
                    f"绘图 token 数超限: {len(toks)} > "
                    f"{spec.DRAW_TOKENS_PER_PRIMITIVE * MAX_PRIMITIVES}"
                    f"（{MAX_PRIMITIVES} 个图元）")
            prims = draw_codec.tokens_to_primitives(toks)
        else:
            raise ValueError("缺少 primitives 或 tokens")

        def work():
            tokens = draw_codec.primitives_to_tokens(prims)
            buf = draw_codec.render_rgba(prims)
            png = draw_codec.encode_png(bytes(buf), spec.CANVAS_W,
                                        spec.CANVAS_H)
            return {
                "png_b64": base64.b64encode(png).decode(),
                "svg": draw_codec.primitives_to_svg(prims),
                "tokens": tokens,
                "n_primitives": len(prims),
            }

        tid = self.pool.submit(work)
        return 202, {"task_id": tid}

    # ---------------- text codec ----------------
    def api_codec_text(self, body):
        from tokens.text_codec import encode as text_encode
        text = body.get("text")
        if text is None:
            raise ValueError("缺少 text")
        # 2026-10-23 审计修复（P2）：text=123 此前让 text_encode 抛
        # TypeError → HTTP 500，违反 400 入参契约
        if not isinstance(text, str):
            raise ValueError(f"text 必须是字符串: {type(text).__name__}")
        steps = text_encode(text)
        return 200, {"steps": [step_to_json(s) for s in steps],
                     "token5": token5_text(steps)}

    # ---------------- video codec ----------------
    def api_codec_video(self, body):
        b64 = body.get("video_b64")
        if not b64:
            raise ValueError("缺少 video_b64")
        data = base64.b64decode(b64)
        suffix = os.path.splitext(body.get("filename", ".mp4"))[1] or ".mp4"
        # 2026-10-27 审计修复：临时文件的清理从创建起就受 try/finally
        # 保护（旧实现 finally 只包住解码，写入失败即泄漏）
        vpath = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(data)
                vpath = f.name
            frames = extract_video_frames(vpath)
            audio_samples = extract_video_audio(vpath)
            # 2026-10-23 审计修复（P1）：抽帧封顶 max_frames 但音轨不
            # 截断——60s 视频每帧被塞 6 个音元（3198 采样），违反 spec
            # 「每帧 533 采样 / 30fps」契约。按帧数截断音元（保留前 N
            # 个 token），响应里显式标记，避免静默丢数据。
            audio_truncated = False
            if len(audio_samples) > len(frames) * spec.AUDIO_SAMPLES_PER_TOKEN:
                audio_samples = audio_samples[
                    :len(frames) * spec.AUDIO_SAMPLES_PER_TOKEN]
                audio_truncated = True
            visual_steps = image_codec.frames_to_steps(frames)
            if sum(len(s["meta"]["patches"]) for s in visual_steps) \
                    > MAX_PATCHES_TOTAL:
                raise ValueError(f"视频 patch 总数超限: > {MAX_PATCHES_TOTAL}")
            audio_steps = audio_codec.waveform_to_steps(audio_samples)
            if len(audio_steps) > MAX_AUDIO_TOKENS:
                raise ValueError(
                    f"音元数超限: {len(audio_steps)} > {MAX_AUDIO_TOKENS}"
                    f"（约 {MAX_AUDIO_TOKENS / 30:.0f} 秒），请裁剪音频")
            aligned = audio_codec.align_audio_to_frames(audio_steps,
                                                       len(frames))
            # 合并：每一帧的 visual step + 对应音频 steps 作为同一时间步列表
            combined = []
            total_patches = 0
            frames_token5 = []
            for vstep, asteps in zip(visual_steps, aligned):
                total_patches += len(vstep["meta"].get("patches", []))
                combined.append({
                    "time_step": [step_to_json(vstep)] +
                                 [step_to_json(a) for a in asteps]
                })
                # token-5：一帧 = 一个 {}，里面先铺 patch 再铺音元
                els = [tok5.visual(p["vec"], p["x"], p["y"])
                       for p in vstep["meta"].get("patches") or []]
                for a in asteps:
                    if (a["meta"].get("n") or 0) >= spec.AUDIO_SAMPLES_PER_TOKEN:
                        els.append(tok5.wave(a["meta"]["samples"]))
                if els:
                    frames_token5.append(tok5.step(els))
            from tokens.vocab import get_vocab
            return 200, {
                "n_frames": len(frames),
                "n_audio_tokens": len(audio_steps),
                "total_patches": total_patches,
                "audio_truncated": audio_truncated,
                "steps": combined,
                "token5": tok5.to_display(
                    frames_token5,
                    name_of=tok5.vocab_name_of(get_vocab())),
            }
        finally:
            if vpath:
                try:
                    os.unlink(vpath)
                except OSError:
                    pass

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
            "primitives": [{"cx": p.cx, "cy": p.cy, "w": p.w, "h": p.h,
                            "shape_type": p.shape_type, "theta": p.theta,
                            "r": p.r, "g": p.g, "b": p.b}
                           for p in prims],
            "n_primitives": len(prims),
            "tokens": tokens,
            "svg": draw_codec.primitives_to_svg(prims),
            "note": note,
        }
