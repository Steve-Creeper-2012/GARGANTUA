# ===========================================================================
# GARGANTUA v1 — dectok：离散 token ID 流 → 可读形式（tok-32 解码器）
# ---------------------------------------------------------------------------
# 规则（2026-10-24 冻结）：
#   输入 = token ID 流（8 位十六进制/token 的文本，或 .tokens 二进制）。
#   默认模式（ids）：按 ID 查表——时间步 token（{ }）与分隔 token（|）
#   显示为字面符号；其余 token 一律输出 8 位十六进制 ID。
#     { 00000006 | 0000abcd }  →  " {00000006|0000abcd} "
#   --text 模式：输出表面文本（中文/单词/emoji/符号原样；<caps> 等
#   特殊 token 显示为其字面字符串；reserved 占位或越界 ID 输出十六进制）。
#
# 用法：
#   python tokens/dectok.py 0000004000000065...
#   python tokens/dectok.py stream.tokens
#   python tokens/dectok.py stream.tokens --text
# ===========================================================================
"""dectok: token ID 流 → { } | 结构 + 十六进制 / 表面文本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- GARGANTUA 路径引导（目录树重构：把 server/ 与 service/* 放上 path）----
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "start" / "boot.py").is_file():
        _sys.path.insert(0, str(_anc / "start"))
        break
else:
    raise SystemExit("找不到 GARGANTUA 项目根（start/boot.py）")
import boot
boot.bootstrap()
import spec  # noqa: E402
from tokens import tok5  # noqa: E402
from tokens.tok32 import hex_to_ids  # noqa: E402
from tokens.vocab import get_vocab, set_vocab_source  # noqa: E402


def decode_ids(ids, mode: str = "ids") -> str:
    """token ID 流 → 可读文本。

    mode="ids"（默认）：{ } | 显示为字面符号，其余输出 8 位十六进制 ID。
    mode="text"：输出表面文本（<caps> 作用于其后英文词/字母、<stop>
    静默跳过、特殊 token 显示字面字符串、reserved/越界 ID 输出十六进制）
    ——与 text_codec.decode_ids 同一套规则。
    """
    if mode not in ("ids", "text"):
        raise ValueError(f"未知模式: {mode!r}（可选 ids / text）")
    if mode == "text":
        from tokens.text_codec import decode_ids as _text_decode
        return _text_decode(ids)
    out: list = []
    for i in ids:
        # 2026-10-27 审计修复：token ID 是 uint32，越界值此前会被
        # "%08x" 静默格式化错（-1 → "-0000001"，2**32 → 九位十六进制）
        if (not isinstance(i, int)) or isinstance(i, bool) \
                or not (0 <= i < 2 ** 32):
            raise ValueError(f"token ID 非法（需 uint32 整数）: {i!r}")
        if i == spec.TS_OPEN_ID:
            out.append("{")
        elif i == spec.TS_CLOSE_ID:
            out.append("}")
        elif i == spec.TS_SEP_ID:
            out.append("|")
        else:
            out.append("%08x" % i)
    return "".join(out)


def decode_hex(text: str, mode: str = "ids") -> str:
    """8 位十六进制/token 的文本 → 可读形式。"""
    return decode_ids(hex_to_ids(text), mode)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="dectok：token-5 .tokens / 历史 hex 流 → 可读形式")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("hexish", nargs="?", default=None,
                     help="8 位十六进制/token 的文本，或 .tokens 文件路径")
    src.add_argument("--file", default=None, help=".tokens 二进制文件")
    ap.add_argument("--text", action="store_true",
                    help="输出表面文本（默认只展开 { } |，其余为十六进制 ID）")
    ap.add_argument("--vocab", default=None,
                    help="用指定的 vocab.txt（或含它的模型目录）解码；"
                         "缺省用全局词表")
    args = ap.parse_args(argv)

    if args.vocab:
        set_vocab_source(args.vocab)

    mode = "text" if args.text else "ids"
    # 文件输入 = token-5 位流（现行格式）
    path = args.file
    if path is None and args.hexish is not None and Path(args.hexish).is_file():
        path = args.hexish
    if path is not None:
        items = tok5.read_file(path)
        if mode == "text":
            ids = [e["id"] for it in items if it["kind"] == "step"
                   for e in it["elements"] if e["kind"] == "text"]
            print(decode_ids(ids, "text"))
        else:
            print(tok5.to_display(
                items, name_of=tok5.vocab_name_of(get_vocab())))
        return
    # 其余：历史 hex 文本（8 位/token）
    if args.hexish is not None:
        ids = hex_to_ids(args.hexish)
    else:
        ids = hex_to_ids(sys.stdin.read())
    print(decode_ids(ids, mode))


if __name__ == "__main__":
    main()
