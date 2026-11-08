# ===========================================================================
# GARGANTUA v1 — agent_tools.py：编号槽位协议 / 元数据（base1.3.0）
# ---------------------------------------------------------------------------
# 2026-11-06（steve 定案）：**删掉全部预定义工具与执行实现**。agent 区改成
# 128 个编号槽位 `<s0>`…`<s127>`，语义由使用者自己在 spec.SLOT_DEFS 定义；
# caps/stop/draw:end 单开 core 一类。本模块只做三件事：
#   1) 下发槽位表 / 契约头元数据（词表不可用时优雅降级，绝不 500）；
#   2) 按 SLOT_DEFS 里自定义的 action/close 配对解析调用（默认全空 → []）；
#   3) 工具结果回灌成 `[sys] 结果文本 [sys/]`。
#
# **执行先全部不接**：这是工具台与测试台，不是最终用户应用；实际执行留给
# 将来的推理 app。run_tool 一律拒绝（HTTP 层把 ValueError 映射成 400）。
# 纯标准库 + spec，tokens.* 一律函数内惰性 import。
# ===========================================================================
from __future__ import annotations

import paths
import spec

ROOT = getattr(paths, "ROOT", None) or paths.PROJECT_ROOT
MAX_ARGS = 4096


def _safe_vocab():
    try:
        from tokens.vocab import get_vocab
        return get_vocab()
    except Exception:
        return None


def _safe_vocab_size() -> int:
    v = _safe_vocab()
    try:
        return len(v) if v is not None else 0
    except Exception:
        return 0


def _slot_def(slot: int) -> dict:
    """槽位自定义元数据；未定义时返回空 dict（默认全部不接）。"""
    d = spec.SLOT_DEFS.get(slot)
    return d if isinstance(d, dict) else {}


def slot_item(slot: int) -> dict:
    """一个槽位的完整元数据；name 一律是编号名 `<sN>`（直接编号显示）。"""
    d = _slot_def(slot)
    return {
        "slot": slot,
        "id": spec.SLOT_BEGIN + slot,
        "name": spec.SLOT_TOKENS[slot],
        "custom_name": d.get("name", ""),
        "kind": d.get("kind", ""),
        "description": d.get("description", ""),
        "close": d.get("close"),
        "arg_hint": d.get("arg_hint", ""),
        "executable": bool(d.get("executable", False)),
    }


def slots_payload() -> list:
    return [slot_item(i) for i in range(spec.SLOT_COUNT)]


def core_payload() -> list:
    """core 区三个机制单标记（单开一类，<> 形态、无闭合、进 loss、有 KV）。"""
    return [{"id": spec.CORE_BEGIN + i, "name": n}
            for i, n in enumerate(spec.CORE_TOKENS)]


def tools_payload() -> dict:
    """GET /api/agent/tools：保留旧字段名（空数组）以兼容前端，另带 slots。"""
    return {
        "tools": [], "formats": [], "boundaries": [], "executable": [],
        "core": core_payload(),
        "slots": slots_payload(),
        "slot_begin": spec.SLOT_BEGIN,
        "slot_count": spec.SLOT_COUNT,
    }


def agent_spec_payload(draw_layout=None) -> dict:
    """GET /api/agent/spec 的唯一载荷；前端一律从这里取 ID，不抄常量。"""
    if draw_layout is None:
        try:
            from server import draw_layout_payload
            draw_layout = draw_layout_payload()
        except Exception:
            draw_layout = {}
    return {
        "vocab_name": spec.BASE_VOCAB_NAME,
        "vocab_version": spec.VOCAB_VERSION,
        "token_encoding": spec.TOKEN_ENCODING,
        "bits_per_char": spec.T6_BITS_PER_CHAR,
        "head_size": spec.HEAD_SIZE,
        "stop_id": spec.STOP_ID,
        "draw_end_id": spec.DRAW_END_ID,
        "draw_begin": spec.DRAW_BEGIN,
        "agent_begin": spec.AGENT_BEGIN,
        "slot_begin": spec.SLOT_BEGIN,
        "slot_count": spec.SLOT_COUNT,
        # base1.3.0：下发 typst 区间与配对规则，后续映射器 / 前端一律
        # 从这里读，禁止抄常量。构造详情见 vocabs/<base>/typst_map.json。
        "typst": {
            "begin": spec.TYPST_BEGIN,
            "end": spec.TYPST_END,
            "size": spec.TYPST_SIZE,
            "construct_count": spec.TYPST_CONSTRUCT_COUNT,
            "pair_rule": "close_id = open_id + 1",
        },
        "special": {
            "caps": spec.CAPS_ID,
            "user": spec.USER_ID,
            "user_end": spec.USER_END_ID,
            "model": spec.MODEL_ID,
            "model_end": spec.MODEL_END_ID,
            "stop": spec.STOP_ID,
            "sys": spec.SYS_ID,
            "sys_end": spec.SYS_END_ID,
        },
        "core": core_payload(),
        "slots": slots_payload(),
        # 旧字段名保留为空数组，避免未同步的前端读 undefined。
        "boundaries": [], "formats": [], "tools": [],
        "file_types": [
            {"ext": ext, "kind": kind,
             "label": spec.AGENT_FILE_LABELS.get(kind, kind)}
            for ext, kind in spec.AGENT_FILE_KINDS.items()
        ],
        "max_file_bytes": spec.AGENT_MAX_FILE_BYTES,
        "draw_layout": draw_layout,
        "vocab_size": _safe_vocab_size(),
    }


def token_meta(tid, vocab=None) -> dict:
    """一个 token 的展示元数据：{"char","tool","slot","core"}。

    core 区用 core 名字、slot 区用编号名（或自定义名），boundary 用 [] 名字，
    <stop> 显示为空串；越界 / 取不到词表一律空串，绝不抛异常。
    `slot` 与兼容字段 `tool` 同时给出，便于前端把槽位渲染成徽标。
    """
    out = {"char": "", "tool": None, "slot": None, "core": None}
    if not (isinstance(tid, int) and not isinstance(tid, bool) and tid >= 0):
        return out
    if spec.CORE_BEGIN <= tid < spec.CORE_END:
        nm = spec.CORE_TOKENS[tid - spec.CORE_BEGIN]
        out["char"] = "" if tid == spec.STOP_ID else nm
        out["core"] = {"id": tid, "name": nm}
        return out
    if tid < spec.SPECIAL_SIZE:
        out["char"] = "" if tid == spec.STOP_ID else spec.SPECIAL_TOKENS[tid]
        return out
    if spec.SLOT_BEGIN <= tid < spec.SLOT_END:
        item = slot_item(tid - spec.SLOT_BEGIN)
        out["char"] = item["custom_name"] or item["name"]
        out["slot"] = item
        # 兼容旧前端：槽位也走「工具徽标」渲染路径。
        out["tool"] = {"id": item["id"], "name": item["name"],
                       "category": item["kind"] or "slot",
                       "description": item["description"],
                       "executable": item["executable"]}
        return out
    # base1.3.0：typst 结构 token 与 core/slot 同为 <> tools；不接执行，
    # 只把名字与类别给前端做徽标展示，避免落到下面的「普通文本」分支。
    if spec.TYPST_BEGIN <= tid < spec.TYPST_END:
        nm = spec.TYPST_NAME_BY_ID.get(tid, "")
        out["char"] = nm
        out["tool"] = {"id": tid, "name": nm, "category": "typst",
                       "description": "Typst 结构 token（开/闭）",
                       "executable": False}
        return out
    v = vocab if vocab is not None else _safe_vocab()
    if v is not None:
        try:
            if tid < len(v):
                out["char"] = v.id_to_token[tid] or ""
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# 槽位调用解析 / 结果回灌
# ---------------------------------------------------------------------------
def _slot_pairs() -> dict:
    """SLOT_DEFS 中定义了 kind=action + close 的槽位 → 闭合槽位 ID。"""
    pairs = {}
    for s, d in spec.SLOT_DEFS.items():
        if not isinstance(d, dict) or d.get("kind") != "action":
            continue
        close = d.get("close")
        if (isinstance(s, int) and 0 <= s < spec.SLOT_COUNT
                and isinstance(close, int) and 0 <= close < spec.SLOT_COUNT):
            pairs[spec.SLOT_BEGIN + s] = spec.SLOT_BEGIN + close
    return pairs


def detect_tool_calls(ids) -> list:
    """按自定义槽位配对解析调用；默认 SLOT_DEFS 为空 → 返回 []。

    返回 [{"name","slot","args","start","end","tool_id"}]；未闭合不返回。
    open 与 close 之间的可见区 token 解码成参数文本。
    """
    if not isinstance(ids, (list, tuple)):
        return []
    pairs = _slot_pairs()
    if not pairs:
        return []
    out = []
    i, n = 0, len(ids)
    while i < n:
        close_id = pairs.get(ids[i])
        if close_id is None:
            i += 1
            continue
        j = i + 1
        while j < n and ids[j] != close_id:
            j += 1
        if j >= n:
            break
        seg = ids[i + 1:j]
        plain = [t for t in seg
                 if isinstance(t, int) and not isinstance(t, bool)
                 and t >= spec.HEAD_SIZE]
        args_text = ""
        if plain:
            try:
                from tokens.text_codec import decode_ids
                args_text = decode_ids(plain)
            except Exception:
                args_text = ""
        slot = ids[i] - spec.SLOT_BEGIN
        out.append({"name": spec.SLOT_TOKENS[slot], "slot": slot,
                    "args": args_text, "start": i, "end": j,
                    "tool_id": ids[i]})
        i = j + 1
    return out


def run_tool(name, args) -> str:
    """执行一律未接入（工具台与测试台）；将来推理 app 自行接。"""
    raise ValueError(
        "工具执行未接入：128 个槽位的语义请自行在 spec.SLOT_DEFS 定义，"
        "本项目只下发协议与元数据（先全部不接）")


def tool_result_ids(name, result) -> list:
    """工具结果 → `[sys] 结果文本 [sys/]`（不再有 tool_result 嵌套包裹）。"""
    if not isinstance(result, str):
        raise ValueError("result 必须是字符串")
    try:
        from tokens.text_codec import encode as text_encode
        steps = text_encode(result)
        body = [t for st in steps for t in st["token_ids"]]
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"工具结果编码失败: {e}") from None
    return [spec.SYS_ID] + body + [spec.SYS_END_ID]
