#!/usr/bin/env python3
# ===========================================================================
# GARGANTUA v1 — gen_typst_map.py：生成 Typst 结构 token 的「对应表格」
# ---------------------------------------------------------------------------
# 背景（2026-11-07 steve 定案）：base1.3.0 把 gargantua-typst-token-spec.md
# §3 的全量语法构造收进架构契约头，但**本次不实现映射器**（token 流 → .typ）。
# 为了让后续映射器能「直接查表转换」，这里从 spec.TYPST_CONSTRUCTS 这一
# 唯一真源生成两份产物：
#   vocabs/<BASE_VOCAB_NAME>/typst_map.json   机器可读（含每个 token 的 ID）
#   service/web/plugins/TYPST_TOKENS.md       人可读（构造 / 配对 / Typst 片段）
#
# ID 一律由 spec 派生，**禁止手抄**：换 slot_count 时重跑本脚本即可。
# 用法：cd ROOT && python service/tokens/gen_typst_map.py
# ===========================================================================
"""gen_typst_map: spec.TYPST_CONSTRUCTS → typst_map.json + TYPST_TOKENS.md。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _anc in Path(__file__).resolve().parents:
    if (_anc / "start" / "boot.py").is_file():
        sys.path.insert(0, str(_anc / "start"))
        break
else:
    raise SystemExit("找不到 GARGANTUA 项目根（start/boot.py）")
import boot  # noqa: E402
boot.bootstrap()
import spec  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "vocabs" / spec.BASE_VOCAB_NAME / "typst_map.json"
OUT_MD = ROOT / "service" / "web" / "plugins" / "TYPST_TOKENS.md"


def build_map() -> dict:
    constructs = []
    tokens = []
    for i, row in enumerate(spec.TYPST_CONSTRUCTS):
        c = dict(zip(spec.TYPST_CONSTRUCT_FIELDS, row))
        open_id = spec.TYPST_BEGIN + 2 * i
        close_id = open_id + 1
        open_tok, close_tok = spec.TYPST_OPEN_TOKENS[i], spec.TYPST_CLOSE_TOKENS[i]
        assert spec.TYPST_ID[c["name"]] == open_id
        assert spec.TYPST_CLOSE_ID[c["name"]] == close_id
        attrs = [a for a in c["attrs"].split(",") if a]
        constructs.append({
            "name": c["name"],
            "section": c["section"],
            "kind": c["kind"],
            "typst": c["typst"],
            "empty": c["empty"],
            "attrs": attrs,
            "note": c["note"],
            "open": {"id": open_id, "hex": f"{open_id:X}", "token": open_tok},
            "close": {"id": close_id, "hex": f"{close_id:X}",
                      "token": close_tok},
        })
        tokens.append({"id": open_id, "hex": f"{open_id:X}",
                       "token": open_tok, "kind": "open",
                       "construct": c["name"], "pair_id": close_id})
        tokens.append({"id": close_id, "hex": f"{close_id:X}",
                       "token": close_tok, "kind": "close",
                       "construct": c["name"], "pair_id": open_id})
    assert len(constructs) == spec.TYPST_CONSTRUCT_COUNT
    assert len(tokens) == spec.TYPST_SIZE
    assert [t["id"] for t in tokens] == list(
        range(spec.TYPST_BEGIN, spec.TYPST_END))
    cfg_path = ROOT / "service" / "tokens" / "config.json"
    try:
        vocab_size = json.loads(
            cfg_path.read_text(encoding="utf-8")).get("vocab_size")
    except (OSError, ValueError):
        vocab_size = None
    return {
        "format": "gargantua-typst-map-v1",
        "vocab_name": spec.BASE_VOCAB_NAME,
        "vocab_version": spec.VOCAB_VERSION,
        "source": "gargantua-typst-token-spec.md §3 + §2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "layout": {
            "slot_count": spec.SLOT_COUNT,
            "typst_begin": spec.TYPST_BEGIN,
            "typst_end": spec.TYPST_END,
            "typst_size": spec.TYPST_SIZE,
            "head_size": spec.HEAD_SIZE,
            "draw_begin": spec.DRAW_BEGIN,
            "vocab_size": vocab_size,
            "max_vocab_size": spec.MAX_VOCAB_SIZE,
        },
        "rules": {
            "pair": "每个构造占两个连续 ID：开在前、闭在后，close_id = open_id + 1",
            "wrappers": "`{}|` = system；`[]` = border；`<>` = tools"
                        "（core + typst / slot 用户自定义预留）；"
                        "`内容:HEX` = 可见常规 token",
            "empty": "empty=true 的构造仍保留显式空对 `<X></X>`（规范 §2）",
            "attrs": "属性走 `<ATTR>…</ATTR>`，紧跟开标签；映射器聚合成具名参数",
            "math": "MATH/MATHB 通道内不做内容转义，运算符/数字/标识符全 token 化",
            "unicode_ext": "可见区末尾（code 之后）另有 `unicode_ext` 3537 个字符"
                           " / 31 个 Unicode 区块（base1.3.0 新增，regular）；"
                           "散文子词用这些普通可见 token 承载，不属于本表，"
                           "映射器按内容通道原样输出并做 §4.2 转义",
            "id_shift": "派生词表带 N 个预留槽位时，本表所有 ID 与 typst_begin "
                        "一同 +N；用 /api/agent/spec 的 typst.begin 换算",
            "token6": "typst 与 core/slot 同为 `<>` tool 元素：进 loss、有 KV、"
                      "type_emb=TYPE_NONE",
        },
        "construct_count": len(constructs),
        "token_count": len(tokens),
        "constructs": constructs,
        "tokens": tokens,
    }


def render_md(m: dict) -> str:
    L = m["layout"]
    lines = [
        "# Typst 结构 token 对应表（%s / %s）" % (
            m["vocab_name"], m["vocab_version"]),
        "",
        "> 适用版本：**%s / token-6**（2026-11-07 定案）。"
        "唯一真源 = `server/spec.py` 的 `TYPST_CONSTRUCTS`；"
        "本表由 `python service/tokens/gen_typst_map.py` 生成，**禁止手改 ID**。"
        % m["vocab_name"],
        "> 依据：`gargantua-typst-token-spec.md` §3 全量 token 词表 + §2 属性机制。"
        "本次只交付词表与表格，**映射器（token 流 → .typ）留作后续独立实现**。",
        "",
        "## 1. 布局与规则",
        "",
        "| 项 | 值 |",
        "|---|---|",
        "| 契约头六段 | core 3 + boundary 6 + system 3 + slot N + "
        "**typst %d** + draw 9605 |" % L["typst_size"],
        "| HEAD_SIZE | %d（base N=0） |" % L["head_size"],
        "| DRAW_BEGIN | %d（base N=0） |" % L["draw_begin"],
        "| typst 区间 | [%d, %d) |" % (L["typst_begin"], L["typst_end"]),
        "| 构造数 / token 数 | %d / %d |" % (
            m["construct_count"], m["token_count"]),
        "",
        "形态约定（steve）：`{}|` = system、`[]` = border、`<>` = tools"
        "（core 机制单标记 + Typst 结构）、`<>` = 用户自定义预留（slot 编号）、"
        "`内容:HEX` = 可见常规 token。",
        "",
    ]
    for k, v in m["rules"].items():
        lines.append(f"- **{k}**：{v}")
    lines += [
        "",
        "## 2. 构造对应表",
        "",
        "ID 列为「十六进制 / 十进制」；配对规则恒为 `闭 ID = 开 ID + 1`。",
        "",
        "| 构造 | 开 token (ID) | 闭 token (ID) | § | 类别 | Typst 构造 | 空 | 属性 | 说明 |",
        "|---|---|---|---|---|---|:---:|---|---|",
    ]
    for c in m["constructs"]:
        attrs = ", ".join(f"`{a}`" for a in c["attrs"]) or "—"
        lines.append(
            "| `{name}` | `{ot}` ({oh} / {oid}) | `{ct}` ({ch} / {cid}) "
            "| {sec} | {kind} | {typst} | {empty} | {attrs} | {note} |".format(
                name=c["name"], ot=c["open"]["token"], oh=c["open"]["hex"],
                oid=c["open"]["id"], ct=c["close"]["token"],
                ch=c["close"]["hex"], cid=c["close"]["id"],
                sec=c["section"], kind=c["kind"], typst=c["typst"],
                empty="∅" if c["empty"] else "—", attrs=attrs, note=c["note"]))
    lines += [
        "",
        "## 3. 后续映射器怎么接",
        "",
        "1. 从 `/api/agent/spec` 读 `typst.begin / size`（或从 "
        "`typst_map.json` 的 `layout`）确定本份词表的区间；派生词表带 N 个"
        "槽位时整体 +N。",
        "2. 逐个 ID 查 `typst_map.json` 的 `tokens`，拿到 `kind`（open/close）"
        "、`construct` 与 `pair_id`；用栈校验配对与嵌套，非法报位置。",
        "3. 开/闭 token 之间的普通子词是内容通道，按规范 §4.2 转义触发字符；"
        "`MATH` / `MATHB` 之内不转义。",
        "4. `ATTR` 聚合为具名参数、`CALL` 展开成 `#name(args)[body]`；"
        "`typst` 列给出每个构造对应的 Typst 写法（规范 §3 原文）。",
        "5. 其余结构按 `typst` 列 + `empty` 标记生成 `.typ`，再交 `typst compile`。",
        "",
        "---",
        f"*由 `service/tokens/gen_typst_map.py` 生成；{m['generated_at']}*",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    m = build_map()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(m, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_md(m), encoding="utf-8")
    print(f"已写出 {OUT_JSON}")
    print(f"已写出 {OUT_MD}")
    print(f"  构造 {m['construct_count']} / token {m['token_count']} / "
          f"区间 [{m['layout']['typst_begin']}, {m['layout']['typst_end']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
