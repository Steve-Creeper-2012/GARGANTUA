# ===========================================================================
# GARGANTUA v1 — enctok：文本 → 离散 token ID 流（tok-32 编码器）
# ---------------------------------------------------------------------------
# 规则（2026-10-24 冻结，spec.py 第 1 节）：
#   1. 查表优先级：先查常规可见 token（regular），常规未收录再查
#      fallback 可见 token（a–z 字母）；不可见 token（special/system）
#      不参与文本编码。
#   2. 英文单词段（[A-Za-z]+ 连续段）：
#      - 全小写且命中英文词表 → 词 token；
#      - 首字母大写（Hello）→ <caps> + 词 token；
#      - 全大写 / 混合大小写（HELLO / McDonald）→ 逐字母 fallback，
#        大写字母前加 <caps>（无损还原，不做有损归一）；
#      - 未命中词表 → 逐字母 fallback，大写字母前加 <caps>。
#   3. 其余字符逐字符查常规可见 token（数字 / 空白 / 符号 / emoji /
#      中文单字 / 全角字符）；未收录字符跳过并记入 skipped 列表
#      （如 CJK 扩展B 及以后）。
#   4. 字面 "<caps>" 等特殊 token 字符串不再被特殊识别——按普通字符
#      编码（< caps u s > ...），可无损还原。
#
# 输出：十六进制文本（8 位/token 拼接）或 .tokens 二进制文件
#       （大端 uint32 流，4 字节/token，解释器见 tokens/tok32.py）。
#
# 用法：
#   python tokens/enctok.py "Hello World"
#   python tokens/enctok.py --file in.txt --out out.tokens
#   echo "你好" | python tokens/enctok.py
# ===========================================================================
"""enctok: 文本 → token ID 流（常规优先、fallback 兜底）。"""

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
from tokens.tok32 import ids_to_hex  # noqa: E402
from tokens.vocab import get_vocab, set_vocab_source  # noqa: E402

CAPS_ID = spec.SPECIAL_BEGIN + spec.SPECIAL_TOKENS.index("<caps>")


def _is_upper(ch: str) -> bool:
    return "A" <= ch <= "Z"


def _emit_word(word: str, v):
    """[A-Za-z]+ 连续段 → (token ID 列表, 未收录字符列表)。

    整词命中 → 直接出（首字母大写加 <caps> 标记）；未命中逐字母拼写
    （fallback 区）。扁平词表（models/NAME/vocab.txt）没有 fallback
    字母——拼不出的字母跳过并报告，而不是 KeyError 崩掉整次编码。
    <caps> 标记只在「至少还有一个字母能映射」时保留，避免给纯垃圾
    输入打光头标记。"""
    lower = word.lower()
    wid = v.regular_ids.get(lower)
    if wid is not None:
        if word == lower:                       # 全小写命中
            return [wid], []
        if word == lower[0].upper() + lower[1:]:  # 首字母大写
            return [CAPS_ID, wid], []
        # 全大写 / 混合大小写：逐字母 fallback（无损）
    out: list = []
    skipped: list = []
    for ch in word:
        if _is_upper(ch):
            out.append(CAPS_ID)
        fid = v.fallback_ids.get(ch.lower())
        if fid is None:
            skipped.append(ch)
            if out and out[-1] == CAPS_ID:      # 字母没映射，标记也无意义
                out.pop()
            continue
        out.append(fid)
    return out, skipped


def encode_text(text: str, vocab=None):
    """文本 → (token ID 列表, 未收录字符列表)。

    vocab 缺省用进程词表（get_vocab()）；显式传入可保证调用方与查表
    使用同一实例（gts 导出时必须如此，否则每模型词表的 <stop>/字母
    fallback 会落到全局词表上——2026-10-27 审计修复 H2）。
    """
    if not isinstance(text, str):
        raise ValueError(f"输入必须是字符串，得到 {type(text).__name__}")
    v = vocab if vocab is not None else get_vocab()
    flat = getattr(v, "flat", False)
    max_match = getattr(v, "_max_match", 1) if flat else 1
    ids: list = []
    skipped: list = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            j = i + 1
            while j < n and (("a" <= text[j] <= "z")
                             or ("A" <= text[j] <= "Z")):
                j += 1
            wids, wskip = _emit_word(text[i:j], v)
            ids.extend(wids)
            skipped.extend(wskip)
            i = j
            continue
        if flat and max_match > 1:
            # 扁平词表：贪心最长匹配——多字符 token（中文词/短语）才能
            # 被文本编码产出；全局词表无此分支（单字符 + 英文词段策略）
            hit = None
            for L in range(min(max_match, n - i), 1, -1):
                tid = v.regular_ids.get(text[i:i + L])
                if tid is not None:
                    hit = (tid, L)
                    break
            if hit is not None:
                ids.append(hit[0])
                i += hit[1]
                continue
        tid = v.regular_ids.get(ch)
        if tid is not None:
            ids.append(tid)
            i += 1
            continue
        skipped.append(ch)
        i += 1
    return ids, skipped


def encode_text_ids(text: str) -> list:
    """文本 → token ID 列表（丢弃 skipped 信息的便捷入口）。"""
    return encode_text(text)[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="enctok：文本 → tok-32 token 流")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("text", nargs="?", default=None, help="输入文本")
    src.add_argument("--file", default=None, help="输入文本文件（UTF-8）")
    ap.add_argument("--out", default=None,
                    help="输出文件：.tokens 为二进制，否则为十六进制文本；"
                         "缺省打印十六进制到 stdout")
    ap.add_argument("--vocab", default=None,
                    help="用指定的 vocab.txt（或含它的模型目录）编码；"
                         "缺省用全局词表")
    args = ap.parse_args(argv)

    if args.vocab:
        set_vocab_source(args.vocab)

    if args.file is not None:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text is not None:
        text = args.text
    else:
        text = sys.stdin.read()

    ids, skipped = encode_text(text)
    if args.out:
        out = Path(args.out)
        if out.suffix.lower() == ".tokens":
            items = [tok5.step(tok5.text(i)) for i in ids]
            tok5.write_file(out, items)
            print(f"已写出 {out}（{len(ids)} token，"
                  f"{out.stat().st_size} 字节 token-5 位流）", file=sys.stderr)
        else:
            out.write_text(ids_to_hex(ids), encoding="utf-8")
            print(f"已写出 {out}（{len(ids)} token）", file=sys.stderr)
    else:
        print(ids_to_hex(ids))
    if skipped:
        print(f"警告：{len(skipped)} 个未收录字符被跳过: "
              + " ".join(sorted(set(skipped))), file=sys.stderr)


if __name__ == "__main__":
    main()
