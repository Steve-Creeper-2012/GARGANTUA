# ===========================================================================
# GARGANTUA v1 — server_agent.py：Infer-Plug v1.0.0 插件 API（AgentApi mixin）
# ---------------------------------------------------------------------------
# App 以 mixin 方式继承本类（见 server.py）：
#     class App(..., AgentApi): ...
# 本模块只 import spec / agent_tools 与纯标准库；draw_layout_payload 从
# server.py 惰性 import（放在函数内部，避免 server ↔ server_agent 循环依赖）。
#
# 所有 api_agent_* 方法：
#   - 统一入参校验（body 必须 dict、缺字段/类型错抛 ValueError → HTTP 400）；
#   - 返回 dict（do_GET 直接 _send_json，do_POST 自动包成 (200, obj)）；
#   - 不写系统临时目录：文件内容全部在内存里校验/编码，不落盘。
# ===========================================================================
from __future__ import annotations

import base64
import binascii
import os
import time

import agent_tools
import spec

MAX_TEXT_CHARS = 100000
MAX_IDS = 100000
MAX_ARGS = 4096
MAX_RESULT_CHARS = 100000


def _require_body(body) -> dict:
    if not isinstance(body, dict):
        raise ValueError("请求体必须是 JSON 对象")
    return body


def _require_str(body, key, *, maxlen=None, allow_empty=False) -> str:
    if key not in body:
        raise ValueError(f"缺少 {key}")
    val = body[key]
    if not isinstance(val, str):
        raise ValueError(f"{key} 必须是字符串: {type(val).__name__}")
    if not allow_empty and not val:
        raise ValueError(f"{key} 不能为空")
    if maxlen is not None and len(val) > maxlen:
        raise ValueError(f"{key} 过长: {len(val)} > {maxlen}")
    return val


def _flat_encode(text: str) -> list:
    """text_codec.encode 的 List[Step] → 扁平 token ID 列表。"""
    from tokens.text_codec import encode as text_encode
    steps = text_encode(text)
    return [t for st in steps for t in st["token_ids"]]


class AgentApi:
    """ /api/agent/* 全部端点（Infer-Plug v1.0.0）。 """

    # ---------------- GET ----------------
    def api_agent_spec(self, body=None):
        # 惰性 import：server.py 在模块级 import 了本文件，这里不能反向 import
        from server import draw_layout_payload
        return agent_tools.agent_spec_payload(draw_layout_payload())

    def api_agent_tools(self, body=None):
        return agent_tools.tools_payload()

    # ---------------- 文本编解码 ----------------
    def api_agent_encode(self, body):
        body = _require_body(body)
        text = _require_str(body, "text", maxlen=MAX_TEXT_CHARS,
                            allow_empty=True)
        try:
            from tokens import enctok
            ids, skipped = enctok.encode_text(text)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"文本编码失败: {e}") from None
        return {
            "token_ids": list(ids),
            "n_tokens": len(ids),
            "skipped": list(dict.fromkeys(skipped)),
        }

    def api_agent_decode(self, body):
        body = _require_body(body)
        ids = body.get("ids")
        if not isinstance(ids, list):
            raise ValueError("ids 必须是整数数组")
        if len(ids) > MAX_IDS:
            raise ValueError(f"ids 过长: {len(ids)} > {MAX_IDS}")
        for t in ids:
            if not isinstance(t, int) or isinstance(t, bool):
                raise ValueError(f"ids 只能包含整数: {t!r}")
        try:
            from tokens.vocab import get_vocab
            v = get_vocab()
            n = len(v)
        except Exception as e:
            raise ValueError(f"词表不可用，无法解码: {e}") from None
        for t in ids:
            if t < 0 or t >= n:
                raise ValueError(f"token ID 越界: {t}")

        try:
            from tokens.text_codec import decode_ids
            text = decode_ids(ids)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"文本解码失败: {e}") from None

        parts = []
        for t in ids:
            if t < spec.SPECIAL_SIZE:
                kind = "special"
                try:
                    txt = v.name_of(t) or spec.SPECIAL_TOKENS[t]
                except Exception:
                    txt = spec.SPECIAL_TOKENS[t] if t < len(
                        spec.SPECIAL_TOKENS) else ""
            elif spec.AGENT_BEGIN <= t < spec.AGENT_END:
                kind = "agent"
                txt = spec.AGENT_NAME_BY_ID.get(t, "")
            elif spec.TYPST_BEGIN <= t < spec.TYPST_END:
                # base1.3.0：typst 结构段在 slot 与 draw 之间，不能落到
                # 末尾的 "text" 分支当成普通可见文本展示。
                kind = "typst"
                txt = spec.TYPST_NAME_BY_ID.get(t, "")
            elif spec.DRAW_BEGIN <= t < spec.HEAD_SIZE:
                kind = "draw"
                txt = v.id_to_token[t] if t < n else ""
            else:
                kind = "text"
                txt = v.id_to_token[t] if t < n else ""
            parts.append({"kind": kind, "id": t, "text": txt})
        return {"text": text, "parts": parts, "n_tokens": len(ids)}

    # ---------------- 文件 ----------------
    def api_agent_file(self, body):
        body = _require_body(body)
        filename = _require_str(body, "filename")
        b64 = _require_str(body, "b64", allow_empty=True)

        # ---- filename 严格校验：basename，禁止 .. / 绝对路径 / 控制字符 ----
        if (os.path.isabs(filename) or "/" in filename or "\\" in filename
                or ".." in filename
                or os.path.basename(filename) != filename):
            raise ValueError(
                f"filename 必须是纯文件名（禁止路径/..）: {filename!r}")
        if any(ord(c) < 32 or ord(c) == 127 for c in filename):
            raise ValueError("filename 含控制字符")
        low = filename.lower()
        ext = ".prim.csv" if low.endswith(".prim.csv") \
            else os.path.splitext(low)[1]
        kind = spec.AGENT_FILE_KINDS.get(ext)
        if kind is None:
            raise ValueError(
                f"不支持的文件扩展名: {ext or '(无)'}（filename={filename!r}）")

        # ---- b64 合法 base64 + 解码后大小上限 ----
        try:
            data = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"b64 不是合法 base64: {e}") from None
        if len(data) > spec.AGENT_MAX_FILE_BYTES:
            raise ValueError(
                f"文件过大: {len(data)} > {spec.AGENT_MAX_FILE_BYTES}")

        if kind in ("text", "draw"):
            text = data.decode("utf-8", errors="replace")
            try:
                token_ids = _flat_encode(text)
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(f"文件内容编码失败: {e}") from None
            media = None
            if kind == "draw":
                # 图元 CSV 的 .prim 不伪造 draw token：按文本编码并在 display
                # 里注明（词表版本化重建期间 draw_codec 也不可用）。
                display = (f"图元 {filename}（{len(text)} 字符 → "
                           f"{len(token_ids)} token，按文本编码；"
                           f"未做图元 token 化，不伪造 draw token）")
            else:
                display = (f"文本 {filename}（{len(text)} 字符 → "
                           f"{len(token_ids)} token）")
        else:
            # image/audio/video：连续投影，token_ids 为空，走 media 通道。
            # 深度 codec 端点校验为可选项，这里只做 base64/大小/扩展名基础
            # 校验，避免词表重建或 ffmpeg 缺失时把合法上传误判成 400。
            token_ids = []
            media = {"kind": kind, "b64": b64, "filename": filename}
            display = (f"{spec.AGENT_FILE_LABELS.get(kind, kind)} {filename}"
                       f"（{len(data)} 字节，走 /api/infer 的 media 通道；"
                       f"已做 base64/大小/扩展名基础校验）")

        return {
            "name": filename,
            "kind": kind,
            "token_ids": token_ids,
            "media": media,
            "n_tokens": len(token_ids),
            "display": display,
            "ext": ext,
        }

    # ---------------- 工具执行 / 结果回灌 ----------------
    def api_agent_tool(self, body):
        body = _require_body(body)
        name = _require_str(body, "name")
        if "args" not in body:
            raise ValueError("缺少 args")
        args = body["args"]
        if not isinstance(args, str):
            raise ValueError(f"args 必须是字符串: {type(args).__name__}")
        if len(args) > MAX_ARGS:
            raise ValueError(f"args 过长: {len(args)} > {MAX_ARGS}")
        result = agent_tools.run_tool(name, args)   # 非法 name / 未启用 → 400
        return {
            "ok": True,
            "name": name,
            "result": result,
            "tool_id": spec.AGENT_ID[name],
            "ts_ms": int(time.time() * 1000),
        }

    def api_agent_tool_result(self, body):
        body = _require_body(body)
        name = _require_str(body, "name")
        result = _require_str(body, "result", maxlen=MAX_RESULT_CHARS,
                              allow_empty=True)
        return {"token_ids": agent_tools.tool_result_ids(name, result)}
