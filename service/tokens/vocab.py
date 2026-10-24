# ===========================================================================
# GARGANTUA v1 — 词表数据结构的加载、查询与扩充
# ---------------------------------------------------------------------------
# 加载 tokens/vocab.json（由 build_vocab.py 生成），提供：
#   - 四大类区域映射（regular / fallback / special / system），全部为
#     {token 文本: id}；同一字符串在全词表内唯一，token_to_id 无反歧义。
#   - lookup(text)：enctok 查表优先级——先 regular 后 fallback；
#     special / system 不可见，不参与文本编码查表。
#   - add_token(text, category)：四类任选其一，全局查重，无冲突则占用
#     该类 reserved 尾段槽位并原子持久化（写回 vocab.json）。
#   - category_of(id)：ID → 大类（供 dectok / 调试 / 前端分类展示）。
#
# 所有区间常数一律取自 spec.py（冻结契约），禁止硬编码。
# 重新运行 build_vocab.py 会重建基础词表，并自动重放此前用户新增的
# token（记录在 meta["user_added"]），不会丢失。
# ===========================================================================
"""词表只读视图 + 四类查表 + 添加 token（全局查重、reserved 槽分配）。"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import spec

VOCAB_PATH = Path(__file__).resolve().parent / "vocab.json"

CAT_REGULAR = "regular"
CAT_FALLBACK = "fallback"
CAT_SPECIAL = "special"
CAT_SYSTEM = "system"
CATEGORIES = (CAT_REGULAR, CAT_FALLBACK, CAT_SPECIAL, CAT_SYSTEM)

_RESERVED_PREFIX = "<reserved:"

# 用户新增 token 的分配区间：各类的 reserved 尾段（基础 token 之后）
_ALLOC_RANGES = {
    CAT_REGULAR: (spec.EXTRA_BEGIN, spec.EXTRA_BEGIN + spec.EXTRA_SIZE),
    CAT_FALLBACK: (spec.FALLBACK_BEGIN + len(spec.FALLBACK_LETTERS),
                   spec.FALLBACK_BEGIN + spec.FALLBACK_SIZE),
    CAT_SPECIAL: (spec.SPECIAL_BEGIN + len(spec.SPECIAL_TOKENS),
                  spec.SPECIAL_BEGIN + spec.SPECIAL_SIZE),
    CAT_SYSTEM: (spec.SYSTEM_BEGIN + 3,
                 spec.SYSTEM_BEGIN + spec.SYSTEM_SIZE),
}

# regular 大类下的全部子区间（enctok 的常规查表范围）
_REGULAR_RANGES = (
    (spec.DIGIT_BEGIN, spec.DIGIT_BEGIN + spec.DIGIT_SIZE),
    (spec.WHITESPACE_BEGIN, spec.WHITESPACE_BEGIN + spec.WHITESPACE_SIZE),
    (spec.SYMBOL_BEGIN, spec.SYMBOL_BEGIN + spec.SYMBOL_SIZE),
    (spec.EMOJI_BEGIN, spec.EMOJI_BEGIN + spec.EMOJI_SIZE),
    (spec.CJK_BEGIN, spec.CJK_BEGIN + spec.CJK_SIZE),
    (spec.ENGLISH_BEGIN, spec.ENGLISH_BEGIN + spec.ENGLISH_SIZE),
    (spec.EXTRA_BEGIN, spec.EXTRA_BEGIN + spec.EXTRA_SIZE),
)


def _is_reserved(tok: str) -> bool:
    return tok.startswith(_RESERVED_PREFIX)


class Vocab:
    """词表视图。四类区域映射均为 {token 文本: id}。"""

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path or VOCAB_PATH)
        data = json.loads(
            self._path.read_text(encoding="utf-8")
        )
        self.id_to_token: list[str] = data["id_to_token"]
        self.token_to_id: dict[str, int] = data["token_to_id"]
        self.meta: dict = data["meta"]
        if len(self.id_to_token) != spec.VOCAB_SIZE_PADDED:
            raise ValueError(
                f"词表长度 {len(self.id_to_token)} != spec.VOCAB_SIZE_PADDED "
                f"{spec.VOCAB_SIZE_PADDED}"
            )

        # —— 四类区域映射（跳过 reserved 占位）——
        self.regular_ids: dict[str, int] = {}
        for begin, end in _REGULAR_RANGES:
            for i in range(begin, end):
                t = self.id_to_token[i]
                if not _is_reserved(t):
                    self.regular_ids[t] = i

        self.fallback_ids: dict[str, int] = {}
        for i in range(spec.FALLBACK_BEGIN,
                       spec.FALLBACK_BEGIN + spec.FALLBACK_SIZE):
            t = self.id_to_token[i]
            if not _is_reserved(t):
                self.fallback_ids[t] = i

        # special：SPECIAL 区 + MODAL（图元 DRAW 输出）区
        self.special_ids: dict[str, int] = {}
        for i in range(spec.SPECIAL_BEGIN, spec.INVISIBLE_END):
            if spec.SYSTEM_BEGIN <= i < spec.SYSTEM_BEGIN + spec.SYSTEM_SIZE:
                continue                                # system 区不属 special
            t = self.id_to_token[i]
            if not _is_reserved(t):
                self.special_ids[t] = i

        self.system_ids: dict[str, int] = {}
        for i in range(spec.SYSTEM_BEGIN,
                       spec.SYSTEM_BEGIN + spec.SYSTEM_SIZE):
            t = self.id_to_token[i]
            if not _is_reserved(t):
                self.system_ids[t] = i

    # ------------------------------------------------------------------
    # 基础查询
    # ------------------------------------------------------------------
    def token(self, idx: int) -> str:
        return self.id_to_token[idx]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def category_of(self, idx: int) -> str:
        """ID → 大类（regular / fallback / special / system）。"""
        if not (0 <= idx < len(self.id_to_token)):
            raise ValueError(f"token ID 越界: {idx}")
        if spec.SYSTEM_BEGIN <= idx < spec.SYSTEM_BEGIN + spec.SYSTEM_SIZE:
            return CAT_SYSTEM
        if idx < spec.INVISIBLE_END:
            return CAT_SPECIAL
        if idx < spec.REGULAR_END:
            return CAT_REGULAR
        return CAT_FALLBACK

    def lookup(self, text: str) -> int | None:
        """enctok 查表：先常规可见，后 fallback 可见；不可见两类不参与。"""
        tid = self.regular_ids.get(text)
        if tid is not None:
            return tid
        return self.fallback_ids.get(text)

    def free_slots(self, category: str) -> int:
        """该类别剩余可分配槽位数（reserved 尾段中的空位）。"""
        if category not in _ALLOC_RANGES:
            raise ValueError(f"未知 token 类别: {category!r}"
                             f"（可选 {CATEGORIES}）")
        begin, end = _ALLOC_RANGES[category]
        return sum(1 for i in range(begin, end)
                   if _is_reserved(self.id_to_token[i]))

    # ------------------------------------------------------------------
    # 添加 token（四类任选，全局查重，原子持久化）
    # ------------------------------------------------------------------
    def add_token(self, text: str, category: str) -> int:
        """添加一个 token，返回新 ID。

        规则：
          1. category ∈ {regular, fallback, special, system}；
          2. text 非空、不以 "<reserved:" 开头（占位符命名空间）；
          3. 全局查重——text 已存在于词表任何位置即拒绝（含四类之间）；
          4. 无冲突则占用该类 reserved 尾段的第一个空槽，写回 vocab.json。
        """
        if category not in _ALLOC_RANGES:
            raise ValueError(f"未知 token 类别: {category!r}"
                             f"（可选 {CATEGORIES}）")
        if not isinstance(text, str) or not text:
            raise ValueError("token 文本必须是非空字符串")
        if text.startswith(_RESERVED_PREFIX):
            raise ValueError(f"token 文本不得使用保留前缀 {_RESERVED_PREFIX!r}")
        if text in self.token_to_id:
            raise ValueError(
                f"token 冲突：{text!r} 已存在于词表"
                f"（ID {self.token_to_id[text]}，"
                f"类别 {self.category_of(self.token_to_id[text])}）")

        begin, end = _ALLOC_RANGES[category]
        slot = next((i for i in range(begin, end)
                     if _is_reserved(self.id_to_token[i])), None)
        if slot is None:
            raise ValueError(f"{category} 类预留槽位已满，无法添加新 token")

        with _LOCK:
            # 双检：持锁后再查一次（防并发重复添加）
            if text in self.token_to_id:
                raise ValueError(f"token 冲突：{text!r} 已被并发添加")
            self.id_to_token[slot] = text
            self.token_to_id[text] = slot
            if category == CAT_REGULAR:
                self.regular_ids[text] = slot
            elif category == CAT_FALLBACK:
                self.fallback_ids[text] = slot
            elif category == CAT_SPECIAL:
                self.special_ids[text] = slot
            else:
                self.system_ids[text] = slot
            self.meta.setdefault("user_added", []).append(
                {"text": text, "category": category, "id": slot})
            self._persist()
        return slot

    def _persist(self) -> None:
        """原子写回加载时的 vocab.json（临时文件 + replace）。"""
        path = self._path
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"id_to_token": self.id_to_token,
                        "token_to_id": self.token_to_id,
                        "meta": self.meta},
                       ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")
        tmp.replace(path)

    def reload(self) -> None:
        """丢弃内存映射，从磁盘重新加载（外部改动后同步）。"""
        self.__init__(self._path)


_LOCK = threading.Lock()
_VOCAB: Vocab | None = None


def get_vocab() -> Vocab:
    """进程级懒加载单例。"""
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = Vocab()
    return _VOCAB


def reset_vocab_cache() -> None:
    """清空单例（测试 / 外部改写 vocab.json 后调用）。"""
    global _VOCAB
    _VOCAB = None
