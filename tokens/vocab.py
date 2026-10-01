"""词表数据结构的加载与查询封装。

加载 tokens/vocab.json（由 build_vocab.py 生成），提供：
    - id_to_token / token_to_id 原始映射
    - 区间专用查询映射（special / symbol / cjk / english / char）
      —— 规避 DIGIT↔SYMBOL、单字母词↔字母的文本撞名歧义
所有区间常数一律取自 spec.py（冻结契约）。
"""
from __future__ import annotations

import json
from pathlib import Path

import spec

VOCAB_PATH = Path(__file__).resolve().parent / "vocab.json"


class Vocab:
    """词表只读视图。区间专用映射均为 {token 文本: id}。"""

    def __init__(self, path: str | Path | None = None):
        data = json.loads(
            Path(path or VOCAB_PATH).read_text(encoding="utf-8")
        )
        self.id_to_token: list[str] = data["id_to_token"]
        self.token_to_id: dict[str, int] = data["token_to_id"]
        self.meta: dict = data["meta"]
        if len(self.id_to_token) != spec.VOCAB_SIZE_PADDED:
            raise ValueError(
                f"词表长度 {len(self.id_to_token)} != spec.VOCAB_SIZE_PADDED "
                f"{spec.VOCAB_SIZE_PADDED}"
            )

        # 特殊 token（<pad> + 12 格式化 + 14 边界，顺序同 spec.SPECIAL_TOKENS）
        self.special_ids: dict[str, int] = {
            t: spec.SPECIAL_BEGIN + i for i, t in enumerate(spec.SPECIAL_TOKENS)
        }

        # SYMBOL 区字符映射（单字符 token，含 ASCII 可打印与全角符号；
        # 仅排除 "<reserved:i>" 占位，"<" 字符本身是合法 token）
        self.symbol_ids: dict[str, int] = {}
        for i in range(spec.SYMBOL_BEGIN, spec.SYMBOL_BEGIN + spec.SYMBOL_SIZE):
            t = self.id_to_token[i]
            if not t.startswith("<reserved:"):
                self.symbol_ids.setdefault(t, i)

        # CJK 区单字映射
        self.cjk_ids: dict[str, int] = {
            self.id_to_token[i]: i
            for i in range(spec.CJK_BEGIN, spec.CJK_BEGIN + spec.CJK_SIZE)
        }

        # ENGLISH 区单词映射（跳过 "<reserved:i>" 占位）
        self.english_ids: dict[str, int] = {}
        for i in range(
            spec.ENGLISH_BEGIN, spec.ENGLISH_BEGIN + spec.ENGLISH_SIZE
        ):
            t = self.id_to_token[i]
            if not t.startswith("<reserved:"):
                self.english_ids[t] = i

        # 统一字符查询：SYMBOL + CJK（两区字符不相交）
        self.char_ids: dict[str, int] = {**self.symbol_ids, **self.cjk_ids}

    def token(self, idx: int) -> str:
        return self.id_to_token[idx]

    def __len__(self) -> int:
        return len(self.id_to_token)


_VOCAB: Vocab | None = None


def get_vocab() -> Vocab:
    """进程级懒加载单例。"""
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = Vocab()
    return _VOCAB
