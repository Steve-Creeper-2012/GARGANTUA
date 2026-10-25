# ===========================================================================
# GARGANTUA v1 — tokenset：.tokens 训练流的读取与展开
# ---------------------------------------------------------------------------
# 训练数据格式（2026-10-25 起，spec 第 8 节）：
#   .tokens = tok-32 大端 uint32 流（tokens/tok32.py），内容必须带
#   { } | 时间步结构（tokens/timestep.py 的 unwrap 可解析）。
#   Token 编辑器「导出 .tokens」产出的就是这种文件；训练侧直接读它，
#   不再接受 JSON/JSONL。
#
# 展开结果：扁平 token_ids + 与之一一对应的 step_ids（每 token 一个
# 时间步编号）。训练窗口切分用 common.make_batch(ids, i, B, step_ids)。
# ===========================================================================
"""tokenset: .tokens → (token_ids, step_ids)，强制时间步结构。"""

from __future__ import annotations

import spec
from tokens import timestep, tok32


class TokensetError(ValueError):
    """训练流非法（空文件 / 缺 { } 结构 / ID 越界）。"""


def read_stream(path) -> tuple[list, list]:
    """读一个 .tokens 文件 → (ids, step_ids)。

    - 必须能被 timestep.unwrap 解析（每个 token 都在 { } 内、| 合法）；
    - 所有 ID 必须 < VOCAB_SIZE_PADDED（模型输出维度）。
    """
    ids = tok32.read_tokens_file(path)
    if not ids:
        raise TokensetError(f"{path}: 空 .tokens 文件")
    try:
        steps = timestep.unwrap(ids)
    except timestep.TimestepError as e:
        raise TokensetError(
            f"{path}: 时间步结构非法（.tokens 必须带 {{ }} | 结构）: {e}"
        ) from e
    flat: list = []
    step_ids: list = []
    for k, step in enumerate(steps):
        for t in step:
            if not (0 <= t < spec.VOCAB_SIZE_PADDED):
                raise TokensetError(
                    f"{path}: token ID {t} 超出词表（< {spec.VOCAB_SIZE_PADDED}）")
        flat.extend(step)
        step_ids.extend([k] * len(step))
    return flat, step_ids


def load_corpus(paths) -> tuple[list, list]:
    """多个 .tokens 文件 → 拼接的 (ids, step_ids)。

    每个文件末尾追加一个 <stop> 时间步（教模型学会终止）；各文件的
    时间步编号依次偏移，跨文件不会撞号。
    """
    ids: list = []
    step_ids: list = []
    next_step = 0
    for p in paths:
        f_ids, f_steps = read_stream(p)
        ids.extend(f_ids)
        step_ids.extend([s + next_step for s in f_steps])
        next_step += (max(f_steps) + 1) if f_steps else 0
        ids.append(spec.STOP_ID)          # 文档结束 → 停止 token
        step_ids.append(next_step)
        next_step += 1
    return ids, step_ids
