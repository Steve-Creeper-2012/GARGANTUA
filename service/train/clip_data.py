# ===========================================================================
# GARGANTUA v1 — clip_data.py：CLIP 对比训练的配对样本契约
# ---------------------------------------------------------------------------
# 本模块是「配对 .tokens 文件」格式与队列文件名元数据的唯一事实来源，
# 被四处共用：server/server_train.py（入队校验）、service/train/clip_folder.py
# （数据集文件夹 v1 解析）、service/train/clip.py + auto_train.py（训练）、
# service/cli/（终端转换）。
#
# 契约：
#   * 训练输入一律是 token-6 `.tokens` 位流（tok6.loads 可解析），不收
#     JSON / JSONL / 原始图片音频文件。
#   * 现行配对（clip_folder v1）= 通用 A/B 两段，模态任意组合
#     （图×图、视频×视频、文本×图都行），用 PAIR_SPLIT_ID 的 tool 步分界；
#     N 个真实配对同批做对称 InfoNCE，负样本来自批内其它配对。
#   * 旧「文本段 × 媒体段」配对仍能被 parse_items 读出来（旧队列文件兼容），
#     但新 UI 不再产生；两段各自连续、只允许一次切换，媒体形态由内容推断
#     （全 audio=audio / 单个纯 visual 步=image / 其余=video）。
#
# 与既有代码的关系：只 import tok6/spec，不改动任何既有模块；LM 训练路径
# 完全不经过本模块（除非文件名被标记为 clip）。
# ===========================================================================
"""clip_data: 配对 .tokens 的解析 / 校验 / 构建 + 队列文件名元数据。"""

from __future__ import annotations

import os
import time
import uuid

import spec
from tokens import tok6

# 训练方式：lm = 教师强制语言模型训练；clip = 对比训练（现行 = 数据集
# 文件夹 v1 的 N 个真实配对）；pair = 旧「两文件对比」队列文件的兼容标记，
# 新 UI 不再产生，仅保证历史队列文件仍能被读完。
MODES = ("lm", "clip", "pair")
# optimizer：muon = 参杂（隐藏层 2D 矩阵走 Muon、其余走 AdamW，历史默认）；
# adamw = 全部 AdamW；allmuon = 全部 Muon（非 2D 参数由 Muon 内部兜底）。
OPTIMIZERS = ("muon", "adamw", "allmuon")

# 通用配对 A/B 分界标记：词表上界的最后一个 ID，落在可见区之外，
# 正常训练数据里不会出现。它被写成 tok6 的 tool 元素（<HEX>，单独一步），
# 解析时只当边界、两段内容都不受它影响。
PAIR_SPLIT_ID = spec.MAX_VOCAB_SIZE - 1

# 单个媒体段的池化位置上限。视觉一帧 32×32 图最多上千个 patch，
# encoder 的 MLA 是 O(N²)，不封顶会把训练进程吃爆；超出时均匀抽样。
DEFAULT_MAX_PAIR_POSITIONS = 512

# 入队体积 / 元素数上限：server 与终端离线入队共用，避免两条路径漂移
# （2026-10-23 审计修复 P3，原先入队无任何上限）。
MAX_ENQUEUE_BYTES = 64 * 1024 * 1024
MAX_ENQUEUE_ELEMENTS = 4_000_000

# 队列文件名：{日期}_{时间}_{optimizer}_{mode}_{hex}.tokens
# 旧文件 {日期}_{时间}_{optimizer}_{hex}.tokens 没有 mode 段 → 一律按 lm。
QUEUE_SUFFIX = ".tokens"


class PairError(ValueError):
    """配对 .tokens 非法（结构不对 / 缺段 / 媒体超限 / 不是 token-6）。"""


# ---------------------------------------------------------------------------
# 队列文件名（server / auto_train / CLI 共用，避免三处各写一套解析）
# ---------------------------------------------------------------------------
def parse_queue_name(name: str) -> dict:
    """→ {"optimizer": "muon"|"adamw"|"", "mode": "lm"|"clip"}。"""
    parts = os.path.basename(str(name)).split("_")
    opt = parts[2] if len(parts) >= 3 and parts[2] in OPTIMIZERS else ""
    mode = parts[3] if len(parts) >= 5 and parts[3] in MODES else "lm"
    return {"optimizer": opt, "mode": mode}


def make_queue_name(optimizer: str = "muon", mode: str = "lm",
                    ts: str | None = None, uid: str | None = None) -> str:
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"未知优化器: {optimizer}")
    if mode not in MODES:
        raise ValueError(f"未知训练方式: {mode}")
    return (f"{ts or time.strftime('%Y%m%d_%H%M%S')}_{optimizer}_{mode}_"
            f"{uid or uuid.uuid4().hex[:8]}{QUEUE_SUFFIX}")


# ---------------------------------------------------------------------------
# Step 分类与位置统计
# ---------------------------------------------------------------------------
def _is_marker(st: dict) -> bool:
    """ext / tool 步只有标记语义，不是文本也不是媒体内容。"""
    meta = st.get("meta") or {}
    return bool(meta.get("ext") or meta.get("tool"))


def _is_text(st: dict) -> bool:
    return (st.get("modality") == "text" and not _is_marker(st)
            and bool(st.get("token_ids")))


def _is_media(st: dict) -> bool:
    return st.get("modality") in ("visual", "audio", "mixed")


def n_positions(st: dict) -> int:
    """一个媒体 Step 展开成多少个模型位置（与 tokenset.read_doc 同口径）。"""
    meta = st.get("meta") or {}
    n = len(meta.get("patches") or [])
    if meta.get("samples") is not None:
        n += 1
    n += len(meta.get("audio") or [])
    return n


def media_kind_of(steps) -> str:
    """audio / image / video。"""
    steps = list(steps)
    if not steps:
        raise PairError("媒体段为空")
    mods = {s.get("modality") for s in steps}
    has_wave = any((s.get("meta") or {}).get("samples") is not None
                   or (s.get("meta") or {}).get("audio") for s in steps)
    if mods == {"audio"} and not any(
            (s.get("meta") or {}).get("patches") for s in steps):
        return "audio"
    if len(steps) == 1 and mods == {"visual"} and not has_wave:
        return "image"
    return "video"


def cap_media_steps(steps, cap: int = DEFAULT_MAX_PAIR_POSITIONS):
    """均匀抽样把媒体段压到 <= cap 个位置，保留原 Step 分组（步内双向）。

    视频帧内「patches + audio 共享 step_id」的语义不能丢，所以不是简单
    截断前 cap 个，而是按原始顺序均匀取位置后重建 Step。
    """
    steps = list(steps)
    total = sum(n_positions(s) for s in steps)
    if total <= cap:
        return steps
    if cap < 1:
        raise PairError(f"媒体位置上限必须 >= 1，收到 {cap}")

    # 展平成 (step_idx, unit)，unit = ("patch", dict) / ("wave", bytes)
    units = []
    for si, st in enumerate(steps):
        meta = st.get("meta") or {}
        for p in meta.get("patches") or []:
            units.append((si, "patch", p))
        if meta.get("samples") is not None:
            units.append((si, "wave", meta["samples"]))
        for a in meta.get("audio") or []:
            units.append((si, "wave", a["samples"]))
    if not units:
        raise PairError("媒体段没有任何位置")

    picked = sorted({round(i * (len(units) - 1) / (cap - 1))
                     for i in range(cap)} if cap > 1 else {0})
    by_step: dict = {}
    for idx in picked:
        si, kind, val = units[idx]
        by_step.setdefault(si, {"type": steps[si].get("type", spec.TYPE_USER),
                                "patches": [], "waves": []})
        (by_step[si]["patches"] if kind == "patch"
         else by_step[si]["waves"]).append(val)

    out = []
    for si in sorted(by_step):
        b = by_step[si]
        meta = {}
        if b["patches"]:
            meta["patches"] = b["patches"]
            meta["grid"] = [max(p["x"] for p in b["patches"]) + 1,
                            max(p["y"] for p in b["patches"]) + 1]
        if len(b["waves"]) == 1:
            meta["samples"] = b["waves"][0]
            meta["n"] = spec.AUDIO_SAMPLES_PER_TOKEN
        elif b["waves"]:
            meta["audio"] = [{"samples": w,
                              "n": spec.AUDIO_SAMPLES_PER_TOKEN}
                             for w in b["waves"]]
        modality = ("mixed" if b["patches"] and b["waves"]
                    else "visual" if b["patches"] else "audio")
        out.append({"token_ids": [], "type": b["type"],
                    "modality": modality, "meta": meta})
    return out


# ---------------------------------------------------------------------------
# 通用配对（2026-11-07）：A 段 × B 段，两段各自可以是任意模态
# ---------------------------------------------------------------------------
def build_pair_items_generic(a_items, b_items,
                             max_media_positions: int = DEFAULT_MAX_PAIR_POSITIONS):
    """两段 tok6 items → 一个通用配对 .tokens 的 items。

    与旧 build_pair_items 的区别：不要求「一段文本 + 一段媒体」，A/B
    都允许 text / visual / audio / mixed 任意组合（图片×图片也可以）。
    中间插一个 tool 元素作分界，解析时按它把两段切开。
    """
    a_items, b_items = list(a_items), list(b_items)
    if not a_items or not b_items:
        raise PairError("通用配对需要非空的 A 段与 B 段")
    for it in a_items + b_items:
        if (it.get("kind") == "tool"
                and it.get("id") == PAIR_SPLIT_ID):
            raise PairError("训练数据里出现了保留的配对分界标记")
    out = a_items + [tok6.tool(PAIR_SPLIT_ID)] + b_items
    parse_items(out, max_media_positions)     # 自校验：产出的必须是合法配对
    return out


def pair_bytes_generic(a_items, b_items,
                       max_media_positions: int = DEFAULT_MAX_PAIR_POSITIONS) -> bytes:
    """→ 可直接落盘的通用配对 .tokens 二进制。"""
    return tok6.dumps(build_pair_items_generic(
        a_items, b_items, max_media_positions))


def _split_pair_items(items):
    """→ (A steps, B steps)；没有分界标记返回 None（走旧严格解析）。"""
    pos = next((i for i, it in enumerate(items)
                if it.get("kind") == "tool"
                and it.get("id") == PAIR_SPLIT_ID), None)
    if pos is None:
        return None
    try:
        a = tok6.to_steps(list(items[:pos]))
        b = tok6.to_steps(list(items[pos + 1:]))
    except tok6.Tok6Error as e:
        raise PairError(f"通用配对不是合法的 token-6 时间步流: {e}") from e
    return a, b


def _cap_pair_segment(steps, cap):
    """通用配对的一段：媒体步按位置上限均匀抽样，文本/标记步原样保留，
    且保持原来的先后次序（连续媒体 run 各自抽样，不跨段重排）。"""
    out, run = [], []
    for st in steps:
        if _is_media(st):
            run.append(st)
            continue
        if run:
            out.extend(cap_media_steps(run, cap))
            run = []
        out.append(st)
    if run:
        out.extend(cap_media_steps(run, cap))
    return out


def _parse_generic_pair(split, max_media_positions):
    """通用 A/B 配对 → clip.py 可直接消费的 pair dict。

    键名沿用 text_steps / media_steps（clip.encode_segment 只看 steps，
    不关心它叫文本还是媒体），另附 a_steps / b_steps 供 UI 展示。
    """
    a, b = split
    a, b = (_cap_pair_segment(a, max_media_positions),
            _cap_pair_segment(b, max_media_positions))
    if not a or not b:
        raise PairError("通用配对的 A 段或 B 段为空")
    n_a_text = sum(len(s.get("token_ids") or [])
                   for s in a if s.get("modality") == "text")
    n_b_pos = sum(n_positions(s) for s in b)
    kinds_a = sorted({s.get("modality", "text") for s in a})
    kinds_b = sorted({s.get("modality", "text") for s in b})
    return {"text_steps": a, "media_steps": b,
            "a_steps": a, "b_steps": b, "generic": True,
            "media_kind": "generic",
            "a_kind": "text" if kinds_a == ["text"] else "+".join(kinds_a),
            "b_kind": "text" if kinds_b == ["text"] else "+".join(kinds_b),
            "n_text_tokens": n_a_text,
            "n_media_positions": n_b_pos,
            "n_media_steps": sum(1 for s in b if _is_media(s))}


# ---------------------------------------------------------------------------
# 解析 / 校验
# ---------------------------------------------------------------------------
def parse_items(items, max_media_positions: int = DEFAULT_MAX_PAIR_POSITIONS) -> dict:
    """tok6 items → 配对描述。非法一律抛 PairError（server 侧映射 400）。

    返回 {
      "text_steps": List[Step], "media_steps": List[Step],
      "media_kind": "audio"|"image"|"video",
      "n_text_tokens": int, "n_media_positions": int, "n_media_steps": int,
    }

    含 A/B 分界标记时走「通用配对」分支（任意模态组合）；旧素材没有
    标记，逐字沿用下面的 text×media 严格校验，两条路径互不干扰。
    """
    if max_media_positions < 1:
        raise PairError(f"max_media_positions 必须 >= 1，收到 {max_media_positions}")
    if not items:
        raise PairError("配对 .tokens 为空（没有任何时间步）")
    split = _split_pair_items(items)
    if split is not None:
        return _parse_generic_pair(split, max_media_positions)
    try:
        steps = tok6.to_steps(items)
    except tok6.Tok6Error as e:
        raise PairError(f"不是合法的 token-6 时间步流: {e}") from e

    text_steps, media_steps, kinds = [], [], []
    for st in steps:
        if _is_marker(st):
            continue
        if _is_text(st):
            text_steps.append(st)
            kinds.append("T")
        elif _is_media(st):
            if n_positions(st) == 0:
                raise PairError("媒体时间步没有载荷（无 patch / 音元）")
            media_steps.append(st)
            kinds.append("M")
        else:
            raise PairError(
                f"配对文件出现无法归类的时间步: modality={st.get('modality')!r}；"
                f"只允许 text / visual / audio / mixed")
    if not text_steps:
        raise PairError("CLIP 配对文件缺少文本段（至少一个 {t:ID} 时间步）")
    if not media_steps:
        raise PairError("CLIP 配对文件缺少媒体段（音频 / 图像 / 视频元素）")
    seq = "".join(kinds)
    if seq not in ("T" * len(text_steps) + "M" * len(media_steps),
                   "M" * len(media_steps) + "T" * len(text_steps)):
        raise PairError(
            "CLIP 配对文件结构非法：文本段与媒体段必须各自连续且只切换一次"
            f"（实际序列 {seq[:80]}…）")

    # 形态判定用未抽样的原始段：视频被 cap 成 1 步时仍应报 video，
    # 不能因为池化预算改变样本的语义类别。
    media_kind = media_kind_of(media_steps)
    media_steps = cap_media_steps(media_steps, max_media_positions)
    n_text = sum(len(s["token_ids"]) for s in text_steps)
    n_media = sum(n_positions(s) for s in media_steps)
    if n_text < 1:
        raise PairError("文本段没有可训练 token")
    if n_media < 1:
        raise PairError("媒体段没有可池化的位置")
    return {"text_steps": text_steps, "media_steps": media_steps,
            "media_kind": media_kind,
            "n_text_tokens": n_text, "n_media_positions": n_media,
            "n_media_steps": len(media_steps)}


def parse_bytes(data: bytes,
                max_media_positions: int = DEFAULT_MAX_PAIR_POSITIONS) -> dict:
    """入队校验入口：.tokens 二进制 → 配对描述。"""
    if not isinstance(data, (bytes, bytearray)):
        raise PairError("训练数据必须是 .tokens 二进制")
    if not data:
        raise PairError("训练数据为空")
    try:
        items = tok6.loads(bytes(data))
    except tok6.Tok6Error as e:
        raise PairError(f"不是合法的 token-6 .tokens: {e}") from e
    return parse_items(items, max_media_positions)


def read_pair(path,
              max_media_positions: int = DEFAULT_MAX_PAIR_POSITIONS) -> dict:
    """训练侧入口：读一个 .tokens 配对文件。"""
    try:
        items = tok6.read_file(path)
    except tok6.Tok6Error as e:
        raise PairError(f"{path}: 不是合法的 token-6 .tokens: {e}") from e
    out = parse_items(items, max_media_positions)
    out["path"] = str(path)
    return out


# ---------------------------------------------------------------------------
# 构建（终端 `tokens pair` 用）
# ---------------------------------------------------------------------------
def build_pair_items(text_steps, media_steps,
                     max_media_positions: int = DEFAULT_MAX_PAIR_POSITIONS):
    """文本 Step 列表 + 媒体 Step 列表 → 校验过的 tok6 items。

    不用 `tok6.from_steps`：它会按进程词表把「不可显示 ID」改写成 ext/tool
    标记步，从既有 .tokens 读回的文本 ID（可能是特殊区 / 别的词表）会被
    静默吞掉，配对文件因此丢段。这里直接按元素重建，与词表完全解耦。
    """
    if not text_steps or not media_steps:
        raise PairError("构建配对文件需要非空文本段与媒体段")
    items = []
    for st in text_steps:
        meta = st.get("meta") or {}
        ids = [int(t) for t in (st.get("token_ids") or [])]
        if meta.get("ext") or meta.get("tool"):
            # 标记步原样保留（parse_items 会忽略，不影响两段判定）
            for tid in ids:
                items.append(tok6.tool(tid) if meta.get("tool")
                             else tok6.ext(tid))
            continue
        if not ids:
            raise PairError("文本段出现空时间步")
        items.append(tok6.step(*[tok6.text(t) for t in ids]))
    for st in media_steps:
        meta = st.get("meta") or {}
        els = []
        for p in meta.get("patches") or []:
            els.append(tok6.visual(bytes(p["vec"]), int(p["x"]), int(p["y"])))
        if meta.get("samples") is not None:
            els.append(tok6.wave(bytes(meta["samples"])))
        for a in meta.get("audio") or []:
            els.append(tok6.wave(bytes(a["samples"])))
        if not els:
            raise PairError("媒体段出现空时间步")
        items.append(tok6.step(*els))
    parse_items(items, max_media_positions)      # 自校验：产出的必须是合法配对
    return items


def pair_bytes(text_steps, media_steps,
               max_media_positions: int = DEFAULT_MAX_PAIR_POSITIONS) -> bytes:
    """→ 可直接落盘的 .tokens 二进制。"""
    return tok6.dumps(build_pair_items(text_steps, media_steps,
                                       max_media_positions))
