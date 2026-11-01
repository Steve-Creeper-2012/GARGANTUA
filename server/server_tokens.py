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
#   GET  /api/vocab/*        词表文件夹 API（list/regions/query/source，
#                            见下方 TokensApi 的方法注释）
#
# 格式（spec 第 5b 节）：27 个符号、5 bit/字符；{ } 包裹时间步，| 分隔
# 步内元素；{t:<id>} 文本、{w:<hex>} 音频、{v:<hex>x:..y:..} 视觉、
# <id> 扩展系统 token。结构字符本身不是 token。
#
# HTTP 错误契约：非法入参一律抛 ValueError（→ 400）。
# ===========================================================================
import base64

import spec
from tokens import enctok, tok5
from tokens.vocab import CAT_REGULAR, CATEGORIES, get_vocab


def _q(query, key: str, default: str = "") -> str:
    """parse_qs 的值取第一项（GET 查询参数用）。"""
    v = (query or {}).get(key, [""])[0]
    return v if v else default


def resolve_vocab_query(query):
    """GET 查询参数里的 name → 对应词表；没有 name 就用进程当前词表。

    词表名非法 / 词表不存在 → ValueError（→400）。"""
    name = _q(query, "name")
    if not name:
        return get_vocab()
    from vocab_presets import load_vocab
    return load_vocab(name)


def _norm_cat(cat) -> str:
    """归一化类别：.vfc 只有「引号 / 尖括号」两种形态 → regular / special。

    fallback 不是用户可选的——可显示且是单个 a–z 字母时自动兼作 fallback；
    system 属架构契约头，不接受用户添加。"""
    c = cat or CAT_REGULAR
    if c == "fallback":
        return CAT_REGULAR
    if c not in (CAT_REGULAR, "special"):
        raise ValueError(f"category 必须是 regular/special: {cat!r}")
    return c


def _body_vocab(body):
    """POST 体里的 vocab 名 → 词表；缺省用进程当前词表。

    转换器（文本/音频/图片/视频）与 Token 编辑器都带这个字段：选哪个
    词表就按哪个词表编码/解码，而不是永远用全局词表。
    """
    name = body.get("vocab")
    if not name:
        return get_vocab()
    from vocab_presets import load_vocab
    return load_vocab(name)

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
        v = _body_vocab(body)
        ids, skipped = enctok.encode_text(text, vocab=v)
        items = [tok5.step(tok5.text(i)) for i in ids]
        out = {"n_tokens": len(ids), "n_steps": len(items),
               "skipped": skipped, "vocab": v.name}
        if fmt == "display":
            out["display"] = tok5.to_display(items, name_of=_name_of(v))
        else:
            out["tokens_b64"] = base64.b64encode(
                tok5.dumps(items)).decode()
        return 200, out

    # ---------------- token 流 → 可读形式 ----------------
    def api_tokens_decode(self, body):
        v = _body_vocab(body)
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
        v = _body_vocab(body)
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

    # ==================================================================
    # 2026-10-30 词表契约重构：多词表文件夹 API
    # ==================================================================
    # 词表 = vocabs/<name>/{vocab.vfc, config.json}；main（base）受保护，
    # 编辑前必须 Fork。所有失败抛 ValueError → 400。
    # ------------------------------------------------------------------
    def api_vocab_list(self, query=None):
        """列出全部词表（前端标签页条「+」的下拉数据源）。"""
        from vocab_presets import list_vocabs
        return 200, {"vocabs": list_vocabs(),
                     "max_vocab_size": spec.MAX_VOCAB_SIZE}

    def api_vocab_regions(self, query):
        """某个词表的分区占用（前端「分区」表格数据源）。"""
        from vocab_presets import vocab_occupancy
        return 200, vocab_occupancy(_q(query, "name"))

    def api_vocab_query(self, query):
        """按文本查 token（GET 形态；与 POST action=query 等价）。"""
        from vocab_presets import query_token
        name = _q(query, "name")
        token = _q(query, "token") or _q(query, "q")
        if not token:
            raise ValueError("缺少 token")
        return 200, query_token(name, token)

    def api_vocab_source_get(self, query):
        """读 .vfc 原文（编辑器模式二的初始内容）。"""
        from vocab_presets import read_vocab_text, load_vocab
        name = _q(query, "name")
        return 200, {"name": name,
                     "text": read_vocab_text(name),
                     "vocab_size": len(load_vocab(name))}

    def api_vocab_token(self, body):
        """模式一：单个 token 的 添加 / 删除 / 查询。"""
        from vocab_presets import (add_token_to, query_token,
                                   remove_token_from)
        name = body.get("name")
        action = body.get("action")
        token = body.get("token")
        if not name:
            raise ValueError("缺少 name")
        if action not in ("add", "remove", "query"):
            raise ValueError(f"action 必须是 add/remove/query: {action!r}")
        if not token:
            raise ValueError("缺少 token")
        if action == "query":
            return 200, query_token(name, token)
        if action == "add":
            return 200, add_token_to(name, token,
                                     _norm_cat(body.get("category")))
        return 200, remove_token_from(name, token)

    def api_vocab_source(self, body):
        """模式二：把编辑器里的 .vfc 全文校验后写回。

        mode="this" 直接改（name 是 main → 拒绝，前端应先弹窗引导 Fork）；
        mode="fork" 先 fork 成 new_name 再写入。
        """
        from vocab_presets import load_vocab, write_vocab_text
        name = body.get("name")
        text = body.get("text")
        mode = body.get("mode") or "this"
        if not name:
            raise ValueError("缺少 name")
        if text is None:
            raise ValueError("缺少 text")
        target = write_vocab_text(name, text, mode=mode,
                                  new_name=body.get("new_name"))
        return 200, {"ok": True, "name": target,
                     "vocab_size": len(load_vocab(target))}

    def api_vocab_fork(self, body):
        """Fork / 复制词表（整目录复制）。"""
        from vocab_presets import create_vocab
        name = body.get("name")
        new_name = body.get("new_name")
        if not name:
            raise ValueError("缺少 name")
        if not new_name:
            raise ValueError("缺少 new_name")
        return 200, create_vocab(new_name, source=name)

    def api_vocab_delete(self, body):
        """删除词表（main 不可删）。"""
        from vocab_presets import delete_vocab, list_vocabs
        name = body.get("name")
        if not name:
            raise ValueError("缺少 name")
        delete_vocab(name)
        return 200, {"ok": True, "vocabs": list_vocabs()}
