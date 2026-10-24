# ===========================================================================
# GARGANTUA v1 — 时间步结构（{ } | 系统 token 的语义与编解码）
# ---------------------------------------------------------------------------
# 语义（spec.py 第 1/3/8 节，2026-10-24 冻结）：
#   时间步是 KDA 循环状态递归与逻辑并行的信号单位（计算可以是逐 token
#   串行的）。每处理完一个时间步内的 token，KDA 循环状态递归一次；
#   压缩 / 滑窗驱逐 / 因果掩码均以时间步为原子，绝不跨步。
#
#   正常形态下各模态的时间步粒度：
#     - 文本：每个 token 独占一个时间步（{ } | 不显式出现在流中，
#       步边界由 Step 列表承载——text_codec 的 List[Step] 即一步一 Step）。
#     - 图片：一张图片 = 一个时间步（内部有多个 patch token）。
#     - 音频：每 533 个采样点（= 一个音元 token）是一个时间步。
#     - 视频：每一帧 = 一个时间步（帧内多个 patch token + 该帧音频 token）。
#
#   显式结构（系统 token，不可见）：
#     {  时间步开始（TS_OPEN_ID）
#     }  时间步结束（TS_CLOSE_ID）
#     |  步内 token 分隔（TS_SEP_ID）
#   一个时间步内 N 个 token 的流为： { t1 | t2 | ... | tN }
#   约束（2026-10-24，steve）：{} 必须含有 token，不能为空集——
#   wrap([]) 与 unwrap 中的空步均抛 TimestepError。
#
# 本模块提供：
#   wrap(ids)        一个时间步的 token 列表 → 带 { } | 的扁平流
#   wrap_steps(steps) List[List[id]]（一步一列表）→ 扁平流
#   unwrap(ids)      扁平流 → List[List[id]]（结构校验，非法抛错）
#   render(ids, ...) 扁平流 → 人类可读文本（默认 hex，可传表面文本函数）
# ===========================================================================
"""时间步结构：{ } 包裹、| 分隔；KDA 递归以时间步为原子。

模块级 API：wrap / wrap_steps / unwrap / render / count_steps。
"""

from __future__ import annotations

import spec

OPEN = spec.TS_OPEN_ID        # {
CLOSE = spec.TS_CLOSE_ID      # }
SEP = spec.TS_SEP_ID          # |


class TimestepError(ValueError):
    """时间步结构错误（括号不配对 / 分隔符越界 / 嵌套等）。"""


def wrap(ids) -> list:
    """一个时间步内的 token 序列 → [ {, t1, |, t2, ..., } ]。

    空时间步非法（steve 设计约束：{} 必须含有 token，不能为空集）。
    """
    ids = list(ids)
    if not ids:
        raise TimestepError("空时间步非法：{} 必须至少包含一个 token")
    out = [OPEN]
    for i, t in enumerate(ids):
        if i:
            out.append(SEP)
        out.append(t)
    out.append(CLOSE)
    return out


def wrap_steps(steps) -> list:
    """List[List[id]]（一步一列表）→ 带 { } | 的扁平 token 流。"""
    out: list = []
    for ids in steps:
        out.extend(wrap(ids))
    return out


def unwrap(ids) -> list:
    """带 { } | 的扁平流 → List[List[id]]（每项一个时间步的 token）。

    结构规则（违规抛 TimestepError）：
      - { 与 } 必须严格配对、不得嵌套；
      - | 必须出现在 { } 之间且不在首尾（空时间步无 |）；
      - 流外的 token（无括号包裹）→ 错误。
    """
    steps: list = []
    cur: list | None = None
    expect_sep = False       # 上一个元素是 token，下一个应是 | 或 }
    for t in ids:
        if t == OPEN:
            if cur is not None:
                raise TimestepError("时间步嵌套：{ 出现在未闭合的时间步内")
            cur = []
            expect_sep = False
        elif t == CLOSE:
            if cur is None:
                raise TimestepError("时间步括号不配对：} 出现在 { 之外")
            if not cur:
                raise TimestepError(
                    "空时间步非法：{} 必须至少包含一个 token")
            steps.append(cur)
            cur = None
            expect_sep = False
        elif t == SEP:
            if cur is None:
                raise TimestepError("分隔符 | 出现在时间步之外")
            if not expect_sep:
                raise TimestepError("分隔符 | 位置非法（连续 | 或位于步首）")
            expect_sep = False
        else:
            if cur is None:
                raise TimestepError(
                    f"token {t} 出现在时间步之外（缺少 {{ }} 包裹）")
            if expect_sep:
                raise TimestepError(
                    f"token {t} 前缺少分隔符 |（同时间步多 token 必须用 | 分隔）")
            cur.append(t)
            expect_sep = True
    if cur is not None:
        raise TimestepError("时间步括号不配对：流结束时 { 未闭合")
    return steps


def render(ids, surface=None) -> str:
    """扁平时间步流 → 人类可读文本。

    surface: 可选 callable(id) -> str，把 token ID 映射为表面文本
    （如词表 token 文本）。缺省用 8 位十六进制 ID。
    系统 token { } | 恒以字面量显示。
    """
    show = surface if surface is not None else (lambda i: "%08x" % i)
    out: list = []
    for t in ids:
        if t == OPEN:
            out.append("{")
        elif t == CLOSE:
            out.append("}")
        elif t == SEP:
            out.append("|")
        else:
            out.append(show(t))
    return "".join(out)


def count_steps(ids) -> int:
    """扁平流中的时间步数量（不校验结构，仅计数 { 出现次数）。"""
    return sum(1 for t in ids if t == OPEN)
