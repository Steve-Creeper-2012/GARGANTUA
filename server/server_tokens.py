# ===========================================================================
# GARGANTUA v1 — server_tokens.py：token-5 离散 token API
# ---------------------------------------------------------------------------
# 从 server.py 拆出。这里只回答「文本 ↔ token 流」与「添加 token」，
# 不碰模型、不碰训练、不碰 HTTP 框架。App 以 mixin 方式继承本类：
#
#     class App(TrainApi, CodecApi, TokensApi): ...
#
# 端点：
#   POST /api/tokens/encode  文本 → token-5 流（.tokens 二进制 b64 / 可读文本）
#   POST /api/tokens/decode  token-5 流 → 可读文本（{text: 00001} …）
#   POST /api/tokens/export  Token 编辑器文本（{t:}/{text:} …）→ .tokens b64
#   POST /api/vocab/add      添加 token（regular/fallback/special/system
#                            四类任选，全局查重，无冲突则入词表）
#
# 格式（spec 第 5b 节）：27 个符号、5 bit/字符；{ } 包裹时间步，| 分隔
# 步内元素；{t:<id>} 文本、{w:<hex>} 音频、{v:<hex>x:..y:..} 视觉、
# <id> 扩展系统 token。结构字符本身不是 token。
#
# HTTP 错误契约：非法入参一律抛 ValueError（→ 400）。
# ===========================================================================
import base64

from tokens import enctok, tok5
from tokens.vocab import CAT_REGULAR, CATEGORIES, get_vocab

# 入参守卫（与 codec 端点的尺寸守卫同思路）
MAX_ENCODE_CHARS = 100_000     # 单次编码文本字符上限
MAX_DECODE_TOKENS = 100_000    # 单次解码元素数上限


def _name_of(v):
    """扩展系统 token 的 ID → 名字（<stop> 等）；可见区返回 None。"""
    return tok5.vocab_name_of(v)


def _id_of(v):
    """名字（"<stop>" 等）→ 扩展系统 token 的 ID；查不到返回 None。"""
    return tok5.vocab_id_of(v)


class TokensApi:
    """ /api/tokens/* 与 /api/vocab/add 全部端点。 """

    # ---------------- 文本 → token 流 ----------------
    def api_tokens_encode(self, body):
        text = body.get("text")
        if text is None:
            raise ValueError("缺少 text")
        # 2026-10-23 审计修复同款：text=123 此前会让 encode 抛
        # TypeError → HTTP 500，违反 400 入参契约
        if not isinstance(text, str):
            raise ValueError(f"text 必须是字符串: {type(text).__name__}")
        if len(text) > MAX_ENCODE_CHARS:
            raise ValueError(
                f"text 过长: {len(text)} > {MAX_ENCODE_CHARS} 字符")
        fmt = body.get("format", "tokens_b64")
        if fmt not in ("tokens_b64", "display"):
            raise ValueError(f"format 必须是 tokens_b64/display: {fmt!r}")
        ids, skipped = enctok.encode_text(text)
        items = [tok5.step(tok5.text(i)) for i in ids]
        out = {"n_tokens": len(ids), "n_steps": len(items),
               "skipped": skipped}
        if fmt == "display":
            out["display"] = tok5.to_display(items)
        else:
            out["tokens_b64"] = base64.b64encode(
                tok5.dumps(items)).decode()
        return 200, out

    # ---------------- token 流 → 可读形式 ----------------
    def api_tokens_decode(self, body):
        v = get_vocab()
        if "tokens_b64" in body:
            b64 = body["tokens_b64"]
            if not isinstance(b64, str):
                raise ValueError("tokens_b64 必须是字符串")
            # 2026-10-27 审计修复：严格 base64 + 预检长度
            try:
                data = base64.b64decode(b64, validate=True)
            except Exception as e:
                raise ValueError(f"tokens_b64 不是合法 base64: {e}")
            if len(data) > MAX_DECODE_TOKENS * 8:
                raise ValueError(
                    f"数据过大: {len(data)} 字节（元素数上限 "
                    f"{MAX_DECODE_TOKENS}）")
            try:
                items = tok5.loads(data)
            except tok5.Tok5Error as e:
                raise ValueError(f"不是合法的 token-5 .tokens: {e}")
        elif "ids" in body:
            ids = body["ids"]
            if not isinstance(ids, list) or any(
                    not isinstance(t, int) or isinstance(t, bool)
                    for t in ids):
                raise ValueError("ids 必须是整数数组")
            if len(ids) > MAX_DECODE_TOKENS:
                raise ValueError(
                    f"token 数超限: {len(ids)} > {MAX_DECODE_TOKENS}")
            try:
                items = [tok5.step(tok5.text(i)) for i in ids]
            except tok5.Tok5Error as e:
                raise ValueError(str(e))
        else:
            raise ValueError("缺少 tokens_b64 / ids")
        n_el = sum(len(it["elements"]) for it in items if it["kind"] == "step")
        if n_el > MAX_DECODE_TOKENS:
            raise ValueError(
                f"元素数超限: {n_el} > {MAX_DECODE_TOKENS}")
        return 200, {"display": tok5.to_display(items, name_of=_name_of(v)),
                     "n_tokens": n_el,
                     "n_steps": sum(1 for it in items
                                    if it["kind"] == "step")}

    # ---------------- Token 编辑器 → .tokens 导出 ----------------
    def api_tokens_export(self, body):
        """把编辑器里的 token-5 文本解析成流，返回 .tokens 二进制。

        结构非法（括号不配对 / 载荷长度不符 / 未收录字符）→ 400，
        不产出坏文件。同时接受紧凑形式 {t:00001} 与显示形式 {text: 00001}。
        """
        text = body.get("text")
        if text is None:
            raise ValueError("缺少 text")
        if not isinstance(text, str):
            raise ValueError(f"text 必须是字符串: {type(text).__name__}")
        if len(text) > MAX_ENCODE_CHARS:
            raise ValueError(
                f"text 过长: {len(text)} > {MAX_ENCODE_CHARS} 字符")
        v = get_vocab()
        try:
            items = tok5.parse_display(text, id_of=_id_of(v))
        except tok5.Tok5Error as e:
            raise ValueError(str(e))
        if not items:
            raise ValueError(
                "token 流为空：至少需要一个时间步（{...}）或扩展系统 token（<...>）")
        n_el = sum(len(it["elements"]) for it in items if it["kind"] == "step")
        if n_el > MAX_DECODE_TOKENS:
            raise ValueError(f"元素数超限: {n_el} > {MAX_DECODE_TOKENS}")
        return 200, {
            "tokens_b64": base64.b64encode(tok5.dumps(items)).decode(),
            "n_tokens": n_el,
            "n_steps": sum(1 for it in items if it["kind"] == "step"),
        }

    # ---------------- 添加 token ----------------
    def api_vocab_add(self, body):
        token = body.get("token")
        category = body.get("category")
        if not isinstance(token, str) or not token:
            raise ValueError("token 必须是非空字符串")
        if category not in CATEGORIES:
            raise ValueError(
                f"category 必须是 {list(CATEGORIES)} 之一: {category!r}")
        v = get_vocab()
        tid = v.add_token(token, category)
        if getattr(v, "flat", False):
            # 每模型词表（models/NAME/vocab.txt）：直接落盘到该文件，
            # 没有 build_vocab 重放机制
            src = getattr(v, "_path", None)
            note = (f"已写入 {src}；该模型的 token 即用即生效"
                    if src else "已写入模型词表")
            free = {"regular": v.free_slots(CAT_REGULAR)}
        else:
            note = "已写入 vocab.json；重新运行 build_vocab.py 会自动重放"
            free = {c: v.free_slots(c) for c in CATEGORIES}
        return 200, {
            "token": token,
            "category": category,
            "id": tid,
            "vocab_size": len(v),
            "free_slots": free,
            "note": note,
        }

    # ---------------- 词表区间占用（前端「词表」页面） ----------------
    def api_vocab_ranges(self, query):
        from vocab_presets import (global_ranges, preset_occupancy,
                                   list_presets)
        preset = (query.get("preset", [""])[0] or None)
        data = global_ranges()
        data["presets"] = list_presets()
        if preset:
            data["selected_preset"] = preset_occupancy(preset)
        return 200, data

    # ---------------- 词表预设 CRUD ----------------
    def api_vocab_presets(self, body):
        from vocab_presets import (create_preset, delete_preset,
                                   list_presets)
        action = body.get("action")
        if action == "create":
            name = body.get("name")
            source = body.get("source") or "base"
            if not name:
                raise ValueError("缺少 name")
            return 200, create_preset(name, source)
        if action == "delete":
            delete_preset(body.get("name"))
            return 200, {"presets": list_presets()}
        # 默认：列出全部预设
        return 200, {"presets": list_presets()}

    # ---------------- 预设内增删 token ----------------
    def api_vocab_preset_token(self, body):
        from vocab_presets import add_token, remove_token
        action = body.get("action")
        name = body.get("name")
        token = body.get("token")
        if not name:
            raise ValueError("缺少 name")
        if action == "add":
            if not token:
                raise ValueError("缺少 token")
            category = body.get("category", "regular")
            return 200, add_token(name, token, category)
        if action == "remove":
            if not token:
                raise ValueError("缺少 token")
            return 200, remove_token(name, token)
        raise ValueError(f"action 必须是 add/remove: {action!r}")
