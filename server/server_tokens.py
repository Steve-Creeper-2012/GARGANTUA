# ===========================================================================
# GARGANTUA v1 — server_tokens.py：tok-32 离散 token API
# ---------------------------------------------------------------------------
# 从 server.py 拆出。这里只回答「文本 ↔ token ID 流」与「添加 token」，
# 不碰模型、不碰训练、不碰 HTTP 框架。App 以 mixin 方式继承本类：
#
#     class App(TrainApi, CodecApi, TokensApi): ...
#
# 端点：
#   POST /api/tokens/encode  文本 → token ID 流（hex / .tokens 二进制 b64）
#   POST /api/tokens/decode  token ID 流 → 可读形式（{ } | + hex / 表面文本）
#   POST /api/vocab/add      添加 token（regular/fallback/special/system
#                            四类任选，全局查重，无冲突则入词表）
#
# HTTP 错误契约：非法入参一律抛 ValueError（→ 400）。
# ===========================================================================
import base64

from tokens import dectok, enctok, tok32
from tokens.vocab import CATEGORIES, get_vocab

# 入参守卫（与 codec 端点的尺寸守卫同思路）
MAX_ENCODE_CHARS = 100_000     # 单次编码文本字符上限
MAX_DECODE_TOKENS = 100_000    # 单次解码 token 数上限


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
        fmt = body.get("format", "hex")
        if fmt not in ("hex", "tokens_b64"):
            raise ValueError(f"format 必须是 hex/tokens_b64: {fmt!r}")
        ids, skipped = enctok.encode_text(text)
        if fmt == "tokens_b64":
            return 200, {
                "tokens_b64": base64.b64encode(
                    tok32.encode_ids(ids)).decode(),
                "n_tokens": len(ids),
                "skipped": skipped,
            }
        return 200, {"hex": tok32.ids_to_hex(ids),
                     "n_tokens": len(ids),
                     "skipped": skipped}

    # ---------------- token 流 → 可读形式 ----------------
    def api_tokens_decode(self, body):
        mode = body.get("mode", "ids")
        if mode not in ("ids", "text"):
            raise ValueError(f"mode 必须是 ids/text: {mode!r}")
        if "tokens_b64" in body:
            b64 = body["tokens_b64"]
            if not isinstance(b64, str):
                raise ValueError("tokens_b64 必须是字符串")
            ids = tok32.decode_ids(base64.b64decode(b64))
        elif "hex" in body:
            h = body["hex"]
            if not isinstance(h, str):
                raise ValueError("hex 必须是字符串")
            ids = tok32.hex_to_ids(h)
        elif "ids" in body:
            ids = body["ids"]
            if not isinstance(ids, list) or any(
                    not isinstance(t, int) or isinstance(t, bool)
                    for t in ids):
                raise ValueError("ids 必须是整数数组")
        else:
            raise ValueError("缺少 tokens_b64 / hex / ids")
        if len(ids) > MAX_DECODE_TOKENS:
            raise ValueError(
                f"token 数超限: {len(ids)} > {MAX_DECODE_TOKENS}")
        return 200, {"text": dectok.decode_ids(ids, mode),
                     "n_tokens": len(ids)}

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
