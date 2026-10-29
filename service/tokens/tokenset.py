# ===========================================================================
# GARGANTUA v1 — tokenset：.tokens 训练流的读取与展开
# ---------------------------------------------------------------------------
# 训练数据格式（2026-10-28 起，spec 第 5b/8 节）：
#   .tokens = token-5 位流（tokens/tok5.py，5 bit/字符）。
#   内容必须带 { } 时间步结构，元素形态 {t:}/{w:}/{v:}/<id>。
#   Token 编辑器「导出 .tokens」产出的就是这种文件；训练侧直接读它，
#   不再接受 JSON/JSONL。
#
# 两种文档形态，训练侧分别处理：
#   纯文本 → (ids, step_ids)，走 make_batch + forward_train（查表路径，
#            不预装配嵌入，便宜）；
#   多模态 → 额外给出装配好的整篇嵌入，走 make_batch_emb +
#            forward_train_embeddings。视觉/音频位置没有词表 ID，
#            目标一律 -100（ignore_index）——媒体只做上下文。
#
# 展开结果：pos_ids（文本 = 词表 ID / 媒体 = None）+ 一一对应的 step_ids。
# ===========================================================================
"""tokenset: .tokens → 训练窗口素材（纯文本走 ID，多模态走嵌入）。"""

from __future__ import annotations

import random

import spec
from tokens import tok5

# 单文档位置上限：多模态要预装配整篇嵌入（N × d_model），长视频会吃光
# 内存。超限直接拒绝，让用户切分文件，而不是把训练进程 OOM 掉。
MAX_DOC_POSITIONS = 32768


class TokensetError(ValueError):
    """训练流非法（空文件 / 结构错 / ID 越界 / 文档过长）。"""


def read_doc(path) -> dict:
    """读一个 .tokens 文件 → 训练文档。

    返回 {
      "path": str,
      "steps": List[Step],        # 装配嵌入用（末尾已追加 <stop>）
      "pos_ids": List[int|None],  # 每个位置：文本 ID 或 None（媒体）
      "step_ids": List[int],      # 每个位置的时间步编号
      "multimodal": bool,
    }
    """
    try:
        items = tok5.read_file(path)
    except tok5.Tok5Error as e:
        raise TokensetError(f"{path}: 不是合法的 token-5 .tokens: {e}") from e
    if not items:
        raise TokensetError(f"{path}: 空 .tokens 文件")
    try:
        steps = tok5.to_steps(items)
    except tok5.Tok5Error as e:
        raise TokensetError(f"{path}: {e}") from e
    if not steps:
        raise TokensetError(f"{path}: 没有任何时间步")

    # 文档结束 → <stop> 时间步（教模型学会终止）
    steps = list(steps) + [{"token_ids": [spec.STOP_ID], "type": spec.TYPE_NONE,
                            "modality": "text", "meta": {}}]

    pos_ids: list = []
    step_ids: list = []
    multimodal = False
    for k, st in enumerate(steps):
        if st["modality"] == "text":
            ids = st["token_ids"]
            if not ids:
                raise TokensetError(f"{path}: 第 {k} 步没有 token（空时间步）")
            for t in ids:
                if not (0 <= t < spec.VOCAB_SIZE_PADDED):
                    raise TokensetError(
                        f"{path}: token ID {t} 超出词表"
                        f"（< {spec.VOCAB_SIZE_PADDED}）")
            pos_ids.extend(ids)
            step_ids.extend([k] * len(ids))
            continue
        multimodal = True
        meta = st.get("meta") or {}
        n = len(meta.get("patches") or [])
        if meta.get("samples") is not None:
            n += 1
        n += len(meta.get("audio") or [])      # 视频帧内的多个音元
        if n == 0:
            raise TokensetError(
                f"{path}: 第 {k} 步是 {st['modality']} 但没有载荷")
        pos_ids.extend([None] * n)
        step_ids.extend([k] * n)

    if len(pos_ids) > MAX_DOC_POSITIONS:
        raise TokensetError(
            f"{path}: 文档 {len(pos_ids)} 个位置 > 上限 {MAX_DOC_POSITIONS}，"
            f"请切分后训练（多模态要预装配整篇嵌入）")
    return {"path": str(path), "steps": steps, "pos_ids": pos_ids,
            "step_ids": step_ids, "multimodal": multimodal}


def read_stream(path) -> tuple[list, list]:
    """纯文本便捷入口：读 .tokens → (ids, step_ids)。

    含 {w:}/{v:} 元素的文件直接拒绝——需要多模态训练请用 read_doc。
    """
    doc = read_doc(path)
    if doc["multimodal"]:
        raise TokensetError(
            f"{path}: 含音频/视觉元素，请用 read_doc() 走多模态训练路径")
    return list(doc["pos_ids"]), list(doc["step_ids"])


def load_corpus(paths) -> list:
    """多个 .tokens 文件 → 按文档隔离的 [doc, ...]。

    每个文件 = 一个独立文档：read_doc 已在末尾追加 <stop> 时间步；
    各文件的步号都基于 0（make_batch 窗口内会重映射，互不冲突）。

    2026-10-27 审计修复：返回结构从「全部拼接成一条流」改为「按文档分组
    的列表」——旧实现把多个文件直接拼接，随机采样的训练窗口可跨过
    <stop> 进入下一文档，模型会学习「停止后继续生成下一文档」。
    调用方必须在单个文档内采样窗口。"""

    return [read_doc(p) for p in paths]


# ---------------------------------------------------------------------------
# 多模态窗口
# ---------------------------------------------------------------------------
def doc_embeddings(m, doc):
    """装配并缓存整篇嵌入（同一模型实例只装配一次）。

    训练循环每步都会取窗口，图像投影不便宜——缓存按 id(m) 失效，
    模型热替换后自动重算。
    """
    cache = doc.get("_emb")
    if cache is not None and cache[0] is m:
        return cache[1]
    emb, _ = m.assemble_embeddings(doc["steps"])
    doc["_emb"] = (m, emb)
    return emb


def _has_target(doc, i: int, block_size: int) -> bool:
    """窗口 i 的目标段里是否至少有一个文本 token（媒体位置不可训练）。"""
    half = block_size // 2
    return any(t is not None for t in doc["pos_ids"][i + half:i + block_size])


def pick_window(doc, block_size: int, rng=random, force_last: bool = False):
    """为多模态文档挑一个可训练的窗口起点；无解返回 None。

    多模态文档里媒体块可能很长，随机起点经常整窗都是媒体（目标全
    -100 → loss 为 NaN）。这里以「文本目标位置」为锚点反推起点，
    保证窗口一定可训。
    """
    n = len(doc["pos_ids"])
    if n < block_size:
        return None
    half = block_size // 2
    max_i = n - block_size
    if force_last:
        i = max_i
        if _has_target(doc, i, block_size):
            return i
    text_pos = [k for k, v in enumerate(doc["pos_ids"]) if v is not None]
    if not text_pos:
        return None
    for _ in range(12):
        t = rng.choice(text_pos)
        i = min(max(0, t - half), max_i)
        if _has_target(doc, i, block_size):
            return i
    for i in range(max_i, -1, -1):        # 兜底：从尾部线性找
        if _has_target(doc, i, block_size):
            return i
    return None
