# ===========================================================================
# GARGANTUA v1 — infer.py：文本推理（自回归，共享 KV 逐步回写）
# ---------------------------------------------------------------------------
# 两种用法：
#   1) CLI（人肉调试）：
#        python infer.py --model models/alpha --prompt "你好" \
#            --max-new-tokens 32 --temperature 0.8 --top-p 0.95
#   2) --serve（server 按需拉起的常驻模式，2026-10-24）：
#        python infer.py --serve [--idle-timeout 300]
#      stdin/stdout 走 JSON-lines 协议（见 server_services.py 文件头）：
#        请求 {"id","op","model_dir","prompt","max_new","temperature","top_p"}
#        响应 {"id","ok",result|error}
#      op：load / infer / status / shutdown。模型按请求里的 model_dir 惰性
#      加载（换了模型才重载），空闲超时自行退出——推理是重活，模型不能
#      常驻 server 进程（省内存，见 README「服务按需启动」）。
#
# 2026-10-20 修复（审查结论）：
#   1. main() 改用 common.build_model：按模型目录 config.json 的 block 数
#      构造，非默认 block 数的模型不再 load_state_dict 崩溃。
#   2. fast_weight.reset() 移到 encode 之前：原实现在 encode 之后 reset，
#      把 encoder 阶段写入的 fast weight 状态清掉，违反 spec「enc/dec
#      共享 bank、状态跨调用推进」，且与 train.py 的 forward_train 不一致。
#   3. KV 回写结构与训练逐位置对齐（review-pipeline 探针实测）：
#      训练 KV 流 = enc(prompt) + [emb(seed)@n, emb(t0)@n+1, …]，即每个
#      decoder 输入的 embedding 放在【它要预测的那个 token 的步】上。
#      本实现 therefore 把「当前输入 emb」拼进视图（s_dec=cur_step），
#      decode 后把该 emb 持久化进 KV；不再单独写回生成 token 的 emb
#      （旧实现因此让种子的 MODEL 嵌入从未进入 KV，且每个生成 token 比
#      训练早一步——train/infer 分布偏差，已回归测试固定）。
#
# 2026-11-04 现状说明（纠正此前「跨调用持久 / 杀进程等于重置记忆」的不实文档）：
#   杀掉 infer 服务进程 → 下次推理重新加载模型；当前实现每次 generate 从零
#   开始（state={}、kv_vecs/kv_steps 都是局部变量，m.fast_weight.reset() 清的
#   是没被使用的 self._state），跨轮持久状态尚未接线。下面注释里的「持久/共享
#   KV」「状态跨调用推进」只指单次 generate 内 encode→decode 跨生成步递推，
#   不表示跨 HTTP 请求保留。
# ===========================================================================
import argparse
import json
import math
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F

# ---- GARGANTUA 路径引导（目录树重构：把 server/ 与 service/* 放上 path）----
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
import spec
from common import (MAX_NEW_TOKENS, MAX_PROMPT_TOKENS,
                    build_model, model_exists)
from tokens import tok5
from tokens.text_codec import encode as text_encode, decode as text_decode
from tokens.vocab import set_model_vocab


# ===========================================================================
# 多模态 prompt：server 传来的媒体 → Step 列表
# ---------------------------------------------------------------------------
# 协议里 media = [{"kind": "image"|"audio"|"video", "b64": ..., "filename": ...}]。
# 解码放在推理服务进程里做（而不是 server）：图片要切成 patch、视频要
# ffmpeg 抽帧，都是重活，放服务进程才不会拖住常驻的 HTTP 线程。
# 规模闸门与 codec 端点同源（server_media 的 MAX_*），超限直接报错。
# ===========================================================================
def _decode_media(items):
    """[{kind,b64,filename}] → List[Step]（视觉/音频，顺序即给定顺序）。"""
    if not items:
        return []
    import base64
    import os
    import tempfile

    from paths import project_tmp_dir
    from server_media import (MAX_AUDIO_TOKENS, MAX_PATCHES_TOTAL,
                              b64_to_pil, decode_audio_any,
                              extract_video_audio, extract_video_frames)
    from tokens import audio_codec, image_codec

    steps: list = []
    total_patches = 0
    total_audio = 0
    for it in items:
        kind = it.get("kind")
        try:
            data = base64.b64decode(it["b64"], validate=True)
        except Exception as e:
            raise ValueError(f"media.b64 不是合法 base64: {e}")
        if kind == "image":
            step = image_codec.image_to_step(b64_to_pil(it["b64"]))
            total_patches += len(step["meta"]["patches"])
            steps.append(step)
        elif kind == "audio":
            filename = it.get("filename") or "audio.wav"
            suffix = os.path.splitext(filename)[1] or ".wav"
            apath = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix,
                                                 delete=False,
                                                 dir=project_tmp_dir()) as f:
                    f.write(data)
                    apath = f.name
                samples = decode_audio_any(apath, filename, spec.AUDIO_SR)
            finally:
                if apath:
                    try:
                        os.unlink(apath)
                    except OSError:
                        pass
            asteps = audio_codec.waveform_to_steps(samples)
            total_audio += len(asteps)
            steps.extend(asteps)
        elif kind == "video":
            filename = it.get("filename") or "video.mp4"
            suffix = os.path.splitext(filename)[1] or ".mp4"
            vpath = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix,
                                                 delete=False,
                                                 dir=project_tmp_dir()) as f:
                    f.write(data)
                    vpath = f.name
                frames = extract_video_frames(vpath)
                audio_samples = extract_video_audio(vpath)
                if len(audio_samples) > len(frames) * spec.AUDIO_SAMPLES_PER_TOKEN:
                    audio_samples = audio_samples[
                        :len(frames) * spec.AUDIO_SAMPLES_PER_TOKEN]
            finally:
                if vpath:
                    try:
                        os.unlink(vpath)
                    except OSError:
                        pass
            vsteps = image_codec.frames_to_steps(frames)
            asteps = audio_codec.waveform_to_steps(audio_samples)
            aligned = audio_codec.align_audio_to_frames(asteps, len(frames))
            for vstep, al in zip(vsteps, aligned):
                total_patches += len(vstep["meta"]["patches"])
                total_audio += len(al)
                merged = {"token_ids": [], "type": spec.TYPE_USER,
                          "modality": "mixed",
                          "meta": {"patches": vstep["meta"]["patches"],
                                   "grid": vstep["meta"]["grid"],
                                   "audio": [{"samples": a["meta"]["samples"],
                                              "n": a["meta"]["n"]} for a in al]}}
                steps.append(merged)
        else:
            raise ValueError(f"未知媒体类型: {kind!r}")
        if total_patches > MAX_PATCHES_TOTAL:
            raise ValueError(
                f"媒体 patch 总数超限: {total_patches} > {MAX_PATCHES_TOTAL}")
        if total_audio > MAX_AUDIO_TOKENS:
            raise ValueError(
                f"媒体音元总数超限: {total_audio} > {MAX_AUDIO_TOKENS}")
    return steps


# ===========================================================================
# 推理侧 token id 存档（2026-10-25，steve：KV 可丢，id 必须落盘）
# ---------------------------------------------------------------------------
# 每次推理一个 .tokens 文件（token-5，每个 token 写成一个 {t:NNNNN}
# 时间步），prompt 与生成 token 边产边写并 flush——KV 超出上下文上限只丢
# KV 向量，原始 id 已经在磁盘上；进程崩溃 / 空闲退出也不丢。
# 目录：server 传入 logs/infer_archive（随 log_path 走，测试用临时目录）；
# CLI 缺省用 <项目根>/logs/infer_archive。
# 存档失败绝不阻断推理：打印到 stderr 并自动降级为「不存档」。
# ===========================================================================
class IdArchive:
    """一次推理的 token id 存档（.tokens + .json 元数据）。 """

    def __init__(self, model_dir: str = "", root: str | None = None,
                 vocab=None):
        self.model_dir = model_dir or ""
        self.root = Path(root) if root else (
            Path(boot.project_root()) / "logs" / "infer_archive")
        self.path: Path | None = None
        self._fh = None
        self._w: tok5.BitWriter | None = None
        self._disabled = False
        self.n_tokens = 0
        self._vocab = vocab

    def _is_invisible(self, i: int) -> bool:
        """该 ID 是否不可显示。

        2026-10-30：不再用「i < HEAD_SIZE」这个固定边界——用户可以把
        新的不可显示 token 追加到任意 ID。以词表自身记录的形态为准。
        vocab 延迟到首次调用时才解析：IdArchive 可能在 set_model_vocab()
        之前就被构造出来（见 generate() 与 load_model() 的调用顺序）。"""
        if self._vocab is None:
            from tokens.vocab import get_vocab
            self._vocab = get_vocab()
        return self._vocab.is_invisible(i)

    def _ensure(self) -> bool:
        if self._fh is not None:
            return True
        if self._disabled:
            return False
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            name = (f"{time.strftime('%Y%m%d_%H%M%S')}_"
                    f"{uuid.uuid4().hex[:8]}.tokens")
            self.path = self.root / name
            self._fh = open(self.path, "wb")
            self._w = tok5.BitWriter(self._fh)
            return True
        except OSError as e:
            print(f"[infer] id 存档不可用（不影响推理）: {e}", file=sys.stderr)
            self._disabled = True
            return False

    def add(self, ids) -> None:
        """把一批 token id 以「一 token 一步」写入存档并 flush。

        不可显示 token 的 id 按扩展系统 token 写成 <NNNNN>，其余写成
        {t:NNNNN}——与 token-5 语义一致（后者才是有向量的内容 token）。
        """
        ids = [int(i) for i in ids]
        if not ids or not self._ensure():
            return
        try:
            chars = []
            for i in ids:
                if self._is_invisible(i):
                    chars.append("<%X>" % i)       # ID 大写十六进制（spec 5b）
                else:
                    chars.append("{t:%X}" % i)
            self._w.write_chars("".join(chars))
            self._fh.flush()
            self.n_tokens += len(ids)
        except OSError as e:
            print(f"[infer] id 存档写入失败（不影响推理）: {e}", file=sys.stderr)
            self._disabled = True

    def add_items(self, items) -> None:
        """把 token-5 流项直接写进存档（媒体载荷 {v:}/{w:} 用）。

        媒体没有词表 ID，但存档要能回放完整输入，所以按 token-5 原样
        落盘；n_tokens 按元素个数累计。
        """
        if not items or not self._ensure():
            return
        try:
            self._w.write_chars(tok5.items_to_chars(items))
            self._fh.flush()
            self.n_tokens += sum(len(it.get("elements") or [])
                                 for it in items if it.get("kind") == "step")
        except OSError as e:
            print(f"[infer] 媒体存档写入失败（不影响推理）: {e}", file=sys.stderr)
            self._disabled = True

    def close(self, n_prompt: int = 0, n_generated: int = 0,
              ctx_total: int | None = None) -> None:
        if self._fh is None:
            return
        try:
            self._w.close()             # 写终止符 + 补齐到字节边界
            self._fh.close()
        except OSError:
            pass
        self._fh = None
        if self.path is None:
            return
        # 元数据：说明这次推理有多少 token、KV 窗口多大、多少条已不在 KV
        meta = {
            "model": os.path.basename(os.path.normpath(self.model_dir))
            if self.model_dir else "",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_prompt_tokens": int(n_prompt),
            "n_generated_tokens": int(n_generated),
            "n_tokens": int(self.n_tokens),
            "ctx_total": None if ctx_total is None else int(ctx_total),
            "n_evicted": (max(0, self.n_tokens - int(ctx_total))
                          if ctx_total is not None else 0),
        }
        try:
            side = self.path.with_suffix(".json")
            side.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        except OSError as e:
            print(f"[infer] 存档元数据写入失败: {e}", file=sys.stderr)

    def rel_path(self) -> str | None:
        """存档路径（相对项目根；给 API / 日志用）。 """
        if self.path is None:
            return None
        try:
            return os.path.relpath(self.path, boot.project_root())
        except ValueError:              # 跨盘（Windows）时退回绝对路径
            return str(self.path)


def _trim_tail_steps(kv_vecs: torch.Tensor, kv_steps: torch.Tensor,
                     ctx_total: int):
    """持久 KV 按整步原子淘汰到 ctx_total（保留最新的若干完整步）。

    2026-10-27 审计修复（M2）：旧实现永不淘汰持久 KV，build_views 只裁剪
    注意力视图——长 prompt/长生成时原始向量无限驻留、每步拼接成本线性
    增长。原始 token id 已在 archive 落盘，丢 KV 不丢 id。"""
    n = kv_steps.shape[1]
    if n <= ctx_total:
        return kv_vecs, kv_steps
    ss = kv_steps[0].tolist()
    total, start, i = 0, n, n - 1
    while i >= 0:                                # 从尾部向前累积整步
        sid = ss[i]
        j = i
        while j >= 0 and ss[j] == sid:
            j -= 1
        step_len = i - j
        if total + step_len > ctx_total:
            break
        total += step_len
        start = j + 1
        i = j
    return kv_vecs[:, start:], kv_steps[:, start:]


def _safe(v, nd):
    """NaN/Inf 消毒：非有限值一律返回 0.0，避免写出非法 JSON 炸掉前端。"""
    f = float(v)
    return round(f, nd) if math.isfinite(f) else 0.0


@torch.no_grad()
def _fw_snapshot(state: dict, bins: int = spec.FW_VIZ_BINS):
    """fast weight 状态的可视化快照（纯展示，不影响计算）。

    状态 S 是 [B, H, dk, dv]；这里给三样东西：
      · head_norms —— 每个头的 Frobenius 范数（谁在记东西）；
      · matrix     —— |S| 对头取平均后 adaptive-avg-pool 到 bins×bins
                      （状态本身的形状：哪些 (k,v) 方向被写得最重）；
      · norm       —— 整体范数。
    取绝对值是刻意的：fast weight 是带符号的联想矩阵，正负抵消后
    直接画会接近全零，幅值才看得出「写到哪」。
    """
    cur = state.get("fw")
    if not cur:
        return None
    S = cur[0]
    if S is None:
        return None
    with torch.no_grad():
        a = S.detach().float().abs().mean(0)          # [H, dk, dv]
        head = a.flatten(1).norm(dim=1).tolist()
        m = a.mean(0)[None, None]                     # [1, 1, dk, dv]
        m = F.adaptive_avg_pool2d(m, (bins, bins))[0, 0]
        # 每个数值都过 _safe：NaN/Inf 不能进 JSON（浏览器 JSON.parse 会炸）
        mat = [[_safe(v, 5) for v in row] for row in m]
        return {"head_norms": [_safe(x, 4) for x in head],
                "matrix": mat,
                "norm": _safe(S.detach().float().norm(), 4),
                "bins": bins}


def generate(m, prompt: str | None = None, max_new: int = 32,
             temperature: float = 0.8, top_p: float = 0.95,
             archive_dir: str | None = None, model_dir: str | None = None,
             media_steps=None, on_token=None, token_ids=None,
             fw_bins: int = spec.FW_VIZ_BINS, t0_wall=None):
    """ 编码 prompt → encoder → KV；decoder 逐步生成并回写共享 KV。

    media_steps：可选的图片/音频 Step 列表（image_codec / audio_codec 的
    产物）。给定时它们排在文本 prompt 之前，与文本一起装配成统一嵌入
    （视觉走 visual_proj、音频走 audio_proj，都投影到 d_model），再整段
    编码进 KV——媒体只做上下文，生成仍然是文本。

    token_ids：直接给 token ID 列表（纯 token 输入，不经文本编码）。
    与 prompt 互斥；给定时 prompt 被忽略。

    on_token：每产出一个 token 回调一次（流式显示 / TTFT / 速度 / fast
    weight 写入曲线都靠它）。回调里只放「这一步发生了什么」，不含整段结果。

    返回 (生成的时间步序列, id 存档路径, 本次统计)。
    """
    max_new = int(max_new)
    if not (1 <= max_new <= MAX_NEW_TOKENS):
        raise ValueError(f"max_new 必须在 [1, {MAX_NEW_TOKENS}]: {max_new}")
    # 补上界 100：与 server.api_infer 同一口径，且 logits 里 max(temp,1e-5)
    # 已把下界钳到 1e-5，错误信息写清有效区间 [1e-5, 100]
    if not (1e-5 <= temperature <= 100):
        raise ValueError(f"temperature 必须在 [1e-5, 100]: {temperature}")
    if not (0 < top_p <= 1):
        raise ValueError(f"top_p 必须在 (0, 1]: {top_p}")

    if token_ids is not None:
        # 纯 token 输入：不经过文本编码器，ID 直接进 embedding。
        # 越界立刻拒绝——embedding 索引越界会抛底层错误，用户看不懂。
        try:
            ids = [int(i) for i in token_ids]
        except (TypeError, ValueError):
            raise ValueError("token_ids 必须是整数数组")
        bad = [i for i in ids if i < 0 or i >= int(m.n_vocab)]
        if bad:
            raise ValueError(
                f"token id 越界: {bad[:5]}（词表 {int(m.n_vocab)}）")
        text_steps = []
        prompt = ""
    else:
        text_steps = text_encode(prompt)
        ids = [t for st in text_steps for t in st["token_ids"]]
    if not ids:
        raise ValueError("prompt 编码为空（媒体输入也需要至少一个文本 token "
                         "作为解码种子）")
    media_steps = list(media_steps or [])
    n_media = 0
    for st in media_steps:
        meta = st.get("meta") or {}
        n_media += len(meta.get("patches") or [])
        if meta.get("samples") is not None:
            n_media += 1
        n_media += len(meta.get("audio") or [])
    if len(ids) + n_media > MAX_PROMPT_TOKENS:
        # 2026-10-27 审计修复（H10）：编码后立刻限长，超长 prompt 不进
        # encoder（避免全序列编码的内存/CPU 尖峰）。媒体位置同样计入。
        raise ValueError(
            f"prompt 过长: {len(ids)} 文本 + {n_media} 媒体 = "
            f"{len(ids) + n_media} > {MAX_PROMPT_TOKENS}")

    # 2026-10-25：prompt 与生成 token 边产边落盘（.tokens），KV 溢出只丢
    # KV 向量、不丢 id；存档失败自动降级，不影响推理本身。
    archive = IdArchive(model_dir or "", archive_dir)
    out_ids: list = []
    t0 = time.perf_counter()
    t_first = None
    t_first_wall = None
    prev_S = None
    writes: list = []                # fast weight 每步写入幅度（最近 N 步）
    try:
        # 存档顺序必须与模型实际看到的顺序一致：媒体在前、文本在后。
        # 用段边界 token 把「用户输入」包成 <user>…<user/>（含媒体与文本），
        # 让 .tokens 存档本身可区分谁是谁，而不是扁平 ID 流。
        archive.add([spec.USER_ID])
        if media_steps:
            archive.add_items(tok5.from_steps(media_steps))
        archive.add(ids)
        archive.add([spec.USER_END_ID])

        # 当前实现每次 generate 从零开始：state/kv 都是本次局部量，跨轮（跨 HTTP
        # 请求）持久状态尚未接线；m.fast_weight.reset() 清的是没被使用的
        # self._state。下面的「状态推进」只在本次调用内 encode→decode 有效。
        m.fast_weight.reset()
        # 2026-10-27 审计修复（S2）：decoder 逐层递推状态（KDA S + 短卷积
        # 历史 + fast weight）跨生成步持有——旧实现每层每步从零状态开始、
        # 短卷积无历史缓存，与训练的整段前向不等价（长程记忆/位置递推
        # 全丢）。encode 的 fast weight 写入也存进同一容器。
        state: dict = {}
        if media_steps:
            # 媒体在前、文本在后；统一装配成 [1, N, D]
            for st in text_steps:
                st["type"] = spec.TYPE_USER
            steps = media_steps + text_steps
            emb_in, s_in = m.assemble_embeddings(steps, base_step=0)
            cur_step = int(s_in.max()) + 1
        else:
            x_in = torch.tensor([ids])
            s_in = torch.arange(len(ids)).unsqueeze(0)
            emb_in = m.embed_text(x_in, torch.full_like(x_in, spec.TYPE_USER))
            cur_step = len(ids)
        enc_out = m.encode(emb_in, s_in, state=state)

        # 共享 KV 初始 = encoder 输出；decoder 每步把当前输入 emb 拼进视图，
        # decode 后持久化进 KV（与 forward_train 的 KV 流逐位置同构）。
        # 超出上下文上限时按整步原子淘汰持久 KV（M2：旧实现只裁剪注意力
        # 视图，原始向量无限拼接，长生成内存随历史线性增长）；原始 id
        # 已在 archive 落盘（丢 KV 不丢 id）。
        ctx_total = getattr(m, "ctx_full", spec.CTX_FULL) + \
            getattr(m, "ctx_compress", spec.CTX_COMPRESS)
        kv_vecs = enc_out.detach()
        kv_steps = s_in
        cur = ids[-1]                            # 用 prompt 尾 token 播种
        # cur_step 已在上面的分支里按「最后一个输入步 + 1」算好
        # 模型输出段：<model>…<model/> 包住本轮生成的所有 token
        archive.add([spec.MODEL_ID])
        for _ in range(max_new):
            # 输入 token 的 embedding 放在其预测步上（s_dec = cur_step），
            # 与训练 make_batch 的 s_dec=arange(half, …) 约定一致
            s_dec = torch.tensor([[cur_step]])
            dec_in = torch.tensor([[cur]])
            emb = m.embed_text(dec_in, torch.full_like(dec_in, spec.TYPE_MODEL))
            kv_hca, kv_csa = m.build_views(
                torch.cat([kv_vecs, emb], 1),
                torch.cat([kv_steps, s_dec], 1))
            dec_out = m.decode(emb, s_dec, kv_hca, kv_csa, state=state)
            logits = m.logits(dec_out)[0, -1].float() / max(temperature, 1e-5)
            # top-p 核采样
            probs = F.softmax(logits, dim=-1)
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_p, dim=0)
            keep = cum - sorted_p < top_p
            sorted_p = sorted_p * keep
            sorted_p = sorted_p / sorted_p.sum().clamp(min=1e-9)
            nxt = sorted_i[torch.multinomial(sorted_p, 1)].item()
            if nxt == spec.STOP_ID:             # 模型输出停止 token → 立即停止
                break
            out_ids.append(nxt)
            archive.add([nxt])
            # —— 流式回调（2026-11-04）：每产一个 token 就报一次 ——
            # t0_wall 是服务端收到请求那一刻的墙钟时间（跨进程用 time.time
            # 对齐）：有它时 t_ms 才是用户「点发送 → 看见首 token」的真实
            # 等待，含模型加载与排队；没有就退回本次 generate 内计时。
            now = time.perf_counter()
            now_wall = time.time()
            if t_first is None:
                t_first, t_first_wall = now, now_wall
            t_ms = (int(max(0.0, (now_wall - t0_wall) * 1000))
                    if t0_wall is not None else int((now - t0) * 1000))
            # 写入幅度每步都要（结尾 meta.fw_writes 用）；整张热力图只在
            # 真的有人订阅流式帧时算，非流式推理不必每步做一次全矩阵 pool。
            cur_fw = state.get("fw")
            write = None
            if cur_fw is not None and cur_fw[0] is not None:
                cur_S = cur_fw[0].detach().float()
                if prev_S is not None:
                    with torch.no_grad():
                        write = _safe((cur_S - prev_S).norm(), 4)
                prev_S = cur_S
            if write is not None:
                writes.append(write)
                if len(writes) > spec.FW_WRITE_HISTORY:
                    del writes[:-spec.FW_WRITE_HISTORY]
            if on_token is not None:
                snap = _fw_snapshot(state, fw_bins) if fw_bins else None
                on_token({"id": nxt, "index": len(out_ids) - 1,
                          "t_ms": t_ms,
                          "kv_steps": min(int(kv_steps.shape[1]) + 1,
                                          ctx_total),
                          "fw": snap, "write": write})
            # 持久化【当前输入】的 emb（训练 KV 流同构）；下一轮输入即 nxt
            kv_vecs = torch.cat([kv_vecs, emb.detach()], 1)
            kv_steps = torch.cat([kv_steps, s_dec], 1)
            if kv_steps.shape[1] > ctx_total:
                kv_vecs, kv_steps = _trim_tail_steps(kv_vecs, kv_steps,
                                                     ctx_total)
            cur, cur_step = nxt, cur_step + 1
        # 模型输出段收尾：无生成也写一对空边界，保持存档结构完整
        archive.add([spec.MODEL_END_ID])
        out_steps = [{"token_ids": [i], "type": spec.TYPE_MODEL,
                      "modality": "text", "meta": {}} for i in out_ids]
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        meta = {
            "n_prompt": len(ids), "n_generated": len(out_ids),
            "ttft_ms": (int(max(0.0, (t_first_wall - t0_wall) * 1000))
                        if (t_first_wall is not None and t0_wall is not None)
                        else (int((t_first - t0) * 1000) if t_first else None)),
            "elapsed_ms": elapsed_ms,
            # 非有限 tps 消毒（除法/计时异常时），避免非法 JSON
            "tps": (_safe(len(out_ids) / max((time.perf_counter() - t_first),
                                             1e-6), 2) if t_first else 0.0),
            "ctx_full": int(getattr(m, "ctx_full", spec.CTX_FULL)),
            "ctx_compress": int(getattr(m, "ctx_compress",
                                        spec.CTX_COMPRESS)),
            "kv_steps": int(kv_steps.shape[1]),
            "kv_bytes": int(kv_vecs.numel() * kv_vecs.element_size()),
            "d_model": int(getattr(m, "d_model", spec.D_MODEL)),
            "vocab_size": int(getattr(m, "n_vocab", 0)),
            "fw": _fw_snapshot(state, fw_bins) if fw_bins else None,
            "fw_writes": list(writes),
        }
        return out_steps, archive.rel_path(), meta
    finally:
        archive.close(
            n_prompt=len(ids), n_generated=len(out_ids),
            ctx_total=(getattr(m, "ctx_full", spec.CTX_FULL)
                       + getattr(m, "ctx_compress", spec.CTX_COMPRESS)))


def _load_model(model_dir: str):
    """加载模型 + 应用该模型的词表（vocab/vocab.vfc 覆盖，2026-10-30 契约）。

    缺权重文件必须显式报错：build_model 静默用随机初始化会让手误的
    路径产出「看似正常」的乱码输出（审查记录 P2）。"""
    if not model_exists(model_dir):
        raise ValueError(f"模型目录无检查点: {model_dir}")
    m = build_model(model_dir)
    m.eval()
    set_model_vocab(model_dir)
    return m


# ---------------------------------------------------------------------------
# --serve：JSON-lines 常驻模式（server_services 协议）
# ---------------------------------------------------------------------------
def _emit(obj: dict) -> None:
    """stdout 只准写 JSON 行——人类可读日志一律 stderr（协议硬约束）。"""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _serve_handle(req: dict, state: dict, emit=None) -> dict:
    rid = req.get("id")
    op = req.get("op")
    try:
        if op == "load":
            model_dir = req["model_dir"]
            state["model"] = None            # 先放掉旧模型再加载新的
            state["dir"] = None
            state["stats"] = None
            m = _load_model(model_dir)
            state["model"], state["dir"] = m, model_dir
            return {"id": rid, "ok": True,
                    "result": {"loaded": model_dir,
                               "params": sum(p.numel()
                                             for p in m.parameters())}}
        if op == "infer":
            model_dir = req["model_dir"]
            if state["model"] is None or state["dir"] != model_dir:
                m = _load_model(model_dir)
                state["model"], state["dir"] = m, model_dir
                state["stats"] = None
            media_steps = _decode_media(req.get("media"))
            # 流式：每产一个 token 就发一帧 partial（同一个 id，前端按序追加）
            streaming = bool(req.get("stream")) and emit is not None

            def on_token(info):
                if streaming:
                    emit({"id": rid, "ok": True, "partial": info})

            steps, archive, meta = generate(
                state["model"], req.get("prompt"),
                req.get("max_new", 32),
                req.get("temperature", 0.8),
                req.get("top_p", 0.95),
                archive_dir=req.get("archive_dir"),
                model_dir=model_dir,
                media_steps=media_steps,
                on_token=on_token if streaming else None,
                token_ids=req.get("token_ids"),
                fw_bins=req.get("fw_bins", spec.FW_VIZ_BINS),
                t0_wall=req.get("t0_wall"))
            state["stats"] = meta
            # 纯 token 输入解出来可能是乱码/不可显示字符，解码失败不该让
            # 整次推理报错——ID 序列才是默认 plugin 的展示对象。
            try:
                text = text_decode(steps)
            except Exception:
                text = ""
            return {"id": rid, "ok": True, "final": True,
                    "result": {"text": text,
                               "token_ids": [s["token_ids"][0]
                                             for s in steps],
                               "archive": archive,
                               "stats": meta}}
        if op == "stats":
            # 只读快照：上下文占用 / KV / fast weight 可视化数据。
            # 服务没跑过推理时 stats 为空，但模型配置照样要能报出来。
            m = state.get("model")
            st = dict(state.get("stats") or {})
            if m is not None:
                st.setdefault("ctx_full",
                              int(getattr(m, "ctx_full", spec.CTX_FULL)))
                st.setdefault("ctx_compress",
                              int(getattr(m, "ctx_compress",
                                          spec.CTX_COMPRESS)))
                st.setdefault("d_model",
                              int(getattr(m, "d_model", spec.D_MODEL)))
                st.setdefault("vocab_size", int(getattr(m, "n_vocab", 0)))
            return {"id": rid, "ok": True,
                    "result": {"loaded": state.get("dir"), **st}}
        if op == "status":
            return {"id": rid, "ok": True,
                    "result": {"loaded": state["dir"]}}
        if op == "shutdown":
            return {"id": rid, "ok": True, "result": {"bye": True}}
        return {"id": rid, "ok": False, "error": f"未知 op: {op!r}"}
    except Exception as e:
        return {"id": rid, "ok": False, "error": str(e)}


def serve(idle_timeout: int = 300) -> int:
    """常驻模式：读线程收请求 → 主线程串行处理 → 空闲超时自行退出。"""
    state = {"model": None, "dir": None, "stats": None}
    q: "queue.Queue" = queue.Queue()

    def _reader():
        try:
            for line in sys.stdin:
                line = line.strip()
                if line:
                    q.put(("line", line))
        except Exception:
            pass
        finally:
            q.put(("eof", None))              # stdin 关 = server 没了

    threading.Thread(target=_reader, daemon=True,
                     name="infer-reader").start()
    print(f"[infer] serve 模式就绪（idle-timeout={idle_timeout}s）",
          file=sys.stderr)

    last = time.time()
    while True:
        remaining = idle_timeout - (time.time() - last)
        if remaining <= 0:
            print("[infer] 空闲超时，退出（模型内存已释放）", file=sys.stderr)
            return 0
        try:
            kind, payload = q.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue
        if kind == "eof":
            print("[infer] stdin EOF，退出", file=sys.stderr)
            return 0
        last = time.time()
        try:
            req = json.loads(payload)
            if not isinstance(req, dict):
                raise ValueError("请求必须是 JSON 对象")
        except Exception as e:
            _emit({"id": None, "ok": False, "error": f"请求解析失败: {e}"})
            continue
        resp = _serve_handle(req, state, _emit)
        _emit(resp)
        if req.get("op") == "shutdown":
            print("[infer] 收到 shutdown，退出", file=sys.stderr)
            return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="模型目录（含 model.safetensors）")
    ap.add_argument("--prompt")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--serve", action="store_true",
                    help="JSON-lines 常驻模式（server 按需拉起）")
    ap.add_argument("--idle-timeout", type=int, default=300,
                    help="serve 模式空闲多少秒后自行退出（默认 300）")
    ap.add_argument("--archive-dir", default=None,
                    help="id 存档目录（默认 <项目根>/logs/infer_archive）")
    args = ap.parse_args()

    if args.serve:
        return serve(idle_timeout=args.idle_timeout)

    if not args.model or not args.prompt:
        ap.error("CLI 模式需要 --model 与 --prompt（或用 --serve）")
    m = _load_model(args.model)
    steps, archive, meta = generate(m, args.prompt, args.max_new_tokens,
                                    args.temperature, args.top_p,
                                    archive_dir=args.archive_dir,
                                    model_dir=args.model)
    print(text_decode(steps))
    if archive:
        print(f"[infer] id 存档: {archive}", file=sys.stderr)
    if meta.get("ttft_ms") is not None:
        print(f"[infer] TTFT {meta['ttft_ms']}ms · "
              f"{meta['n_generated']} tok · {meta['tps']} tok/s",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
