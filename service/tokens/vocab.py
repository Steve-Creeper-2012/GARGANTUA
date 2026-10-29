# ===========================================================================
# GARGANTUA v1 — 词表数据结构的加载、查询与扩充
# ---------------------------------------------------------------------------
# 两种词表来源：
#   1) 全局词表 tokens/vocab.json（由 build_vocab.py 生成）：四大类区域
#      布局（regular / fallback / special / system），全部为 {token: id}；
#      同一字符串在全词表内唯一，token_to_id 无反歧义。
#   2) 每模型词表 models/NAME/vocab.txt（2026-10-24，steve：每个模型可以
#      有不同的 vocab 词表）：扁平可见 token 列表，ID 由文件显式给出
#      （v2 格式，见 common.VOCAB_FILE_MAGIC）；全部按 regular 处理
#      （special/system 的不可见 token 由 spec 常量按 ID 定位，不依赖
#      词表文件）。契约见 common.read_model_vocab_entries 与 spec.py 第 9 节。
#
# 公共能力：
#   - lookup(text)：enctok 查表优先级——先 regular 后 fallback；
#     special / system 不可见，不参与文本编码查表。
#   - add_token(text, category)：四类任选其一，全局查重，无冲突则占用
#     该类 reserved 尾段槽位并原子持久化（json 写回 vocab.json /
#     txt 写回 vocab.txt）。
#   - category_of(id)：ID → 大类（供 dectok / 调试 / 前端分类展示）。
#
# 进程级覆盖：set_model_vocab(model_dir) 之后，本进程的 get_vocab()
# 返回该模型的词表（infer/train 服务进程在加载模型时调用；server 在
# 切换活动模型时调用）。没有 vocab.txt → 清退到全局词表。
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
from common import (model_vocab_path, read_flat_vocab_file,
                    read_model_vocab_entries, write_flat_vocab_file)

VOCAB_PATH = Path(__file__).resolve().parent / "vocab.json"

CAT_REGULAR = "regular"
CAT_FALLBACK = "fallback"
CAT_SPECIAL = "special"
CAT_SYSTEM = "system"
CATEGORIES = (CAT_REGULAR, CAT_FALLBACK, CAT_SPECIAL, CAT_SYSTEM)

_RESERVED_PREFIX = "<reserved:"

# 软删占位符前缀（2026-10-25）：扁平词表（每模型 / 词表预设）里被删除的
# token 原地替换为 <del:N>（N=行号），既保留 ID 位置（不令后续 token 漂移）
# 又不可被文本编码查到。from_tokens / read_model_vocab 仍把它留在 id_to_token
# 以维持行号→ID 映射，但不再登记进 token_to_id / regular_ids。
DEL_MARKER = "<del:"

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


def _is_hole(tok: str) -> bool:
    """空位：reserved 占位符或软删占位符，都不是可查表的真实 token。"""
    return _is_reserved(tok) or tok.startswith(DEL_MARKER)


# 全局词表的「不可见区」头部缓存：special/system/modal 三类固定 token
# （<stop>/<caps>/{ } |/图元输出 token……）。扁平词表原样复用这段——
# 它们的 ID 是架构契约（spec 常量按 ID 定位），自定义词表不得占用。
_HEAD_CACHE: list | None = None


def _global_head() -> list:
    global _HEAD_CACHE
    if _HEAD_CACHE is None:
        data = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
        _HEAD_CACHE = data["id_to_token"][:spec.INVISIBLE_END]
    return _HEAD_CACHE


class Vocab:
    """词表视图。四类区域映射均为 {token 文本: id}。

    flat=True 表示来自 models/NAME/vocab.txt 的每模型词表：全部 token
    按 regular 处理，无四类区间语义（special/system 由 spec 常量按 ID
    定位）。持久化写回 txt；json 词表持久化写回 vocab.json。
    """

    def __init__(self, path: str | Path | None = None):
        self.flat = False
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
    @classmethod
    def from_tokens(cls, tokens: list, path: str | Path | None = None) -> "Vocab":
        """位置式便捷构造：第 N 个 token → ID = INVISIBLE_END + N。

        仅用于「还没有 ID 的新词表」与测试；磁盘上的词表一律走
        from_entries（v2 格式带显式 ID，不会被行序影响）。
        """
        base = spec.INVISIBLE_END
        return cls.from_entries([(base + i, t) for i, t in enumerate(tokens)],
                                path)

    @classmethod
    def from_entries(cls, entries, path: str | Path | None = None,
                     legacy: bool = False) -> "Vocab":
        """每模型扁平词表（models/NAME/vocab.txt，见 common.read_model_vocab_entries）。

        entries 是 [(id, token)]，ID 必须在 [INVISIBLE_END,
        VOCAB_SIZE_PADDED) 内且不重复。不可见区 [0, INVISIBLE_END) 原样
        复用全局词表的固定 token——<stop>/<caps>/{ } |/图元输出 token 的
        ID 是架构契约（spec 常量按 ID 定位，decode 的停止/大小写逻辑靠
        它们），自定义词表绝不得占用。

        2026-10-27 审计修复（H2）：special / system / fallback 三类查表
        映射不再清空——
          - special / system 直接取固定 head 的对应区（<stop> 必须能被
            gts 识别，否则 Token 编辑器永远导不出停止标记）；
          - fallback 取词表里的单字母常规 token（a–z），让 enctok 在整词
            未命中时仍能逐字母拼写，而不是整段 skipped。
        """
        v = cls.__new__(cls)                # 绕开 json 加载路径
        v.flat = True
        v._path = Path(path) if path else None
        v.legacy_format = bool(legacy)
        head = _global_head()
        v.id_to_token = (list(head) + [f"<reserved:{i}>"
                                       for i in range(spec.INVISIBLE_END,
                                                      spec.VOCAB_SIZE_PADDED)])
        seen_tok = set()
        seen_ids = set()
        for tid, tok in entries:
            if not isinstance(tid, int) or isinstance(tid, bool):
                raise ValueError(f"token ID 必须是整数: {tid!r}")
            if not (spec.INVISIBLE_END <= tid < spec.VOCAB_SIZE_PADDED):
                raise ValueError(
                    f"token ID {tid} 越界 "
                    f"[{spec.INVISIBLE_END}, {spec.VOCAB_SIZE_PADDED})")
            if tid in seen_ids:
                raise ValueError(f"token ID 重复: {tid}")
            if not isinstance(tok, str) or not tok:
                raise ValueError(f"token 文本必须是非空字符串: {tok!r}")
            if tok in seen_tok:
                raise ValueError(f"vocab.txt token 重复: {tok!r}")
            seen_ids.add(tid)
            seen_tok.add(tok)
            v.id_to_token[tid] = tok

        # 查重覆盖全词表（含不可见区）；用户可见 token 优先——{ } | 这类
        # 双挂载字符按全局语义「文本→ID 解析到可见符号」
        v.token_to_id = {t: i for i, t in enumerate(head)
                         if not _is_reserved(t)}
        v.regular_ids = {}
        v.fallback_ids = {}
        for tid, tok in entries:
            if _is_hole(tok):
                continue                    # 软删/reserved 占位：留位不可查
            v.token_to_id[tok] = tid
            v.regular_ids[tok] = tid
            if len(tok) == 1 and "a" <= tok <= "z":
                v.fallback_ids[tok] = tid

        # special / system：固定 head 里的真实 token（H2）
        v.special_ids = {}
        for i in range(spec.SPECIAL_BEGIN, spec.INVISIBLE_END):
            if spec.SYSTEM_BEGIN <= i < spec.SYSTEM_BEGIN + spec.SYSTEM_SIZE:
                continue
            t = v.id_to_token[i]
            if not _is_reserved(t):
                v.special_ids[t] = i
        v.system_ids = {}
        for i in range(spec.SYSTEM_BEGIN, spec.SYSTEM_BEGIN + spec.SYSTEM_SIZE):
            t = v.id_to_token[i]
            if not _is_reserved(t):
                v.system_ids[t] = i

        v.meta = {"source": "vocab.txt", "n_tokens": len(v.regular_ids),
                  "format": "v1" if legacy else "v2"}
        v._n_user = len(v.regular_ids)
        # 扁平词表的最长 token 长度（encode_text 贪心匹配用）
        v._max_match = max((len(t) for t in v.regular_ids), default=1)
        return v

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
        if self.flat:
            # 不可见区沿用全局布局分类；可见部分一律 regular
            if idx < spec.INVISIBLE_END:
                if (spec.SYSTEM_BEGIN <= idx
                        < spec.SYSTEM_BEGIN + spec.SYSTEM_SIZE):
                    return CAT_SYSTEM
                return CAT_SPECIAL
            return CAT_REGULAR
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
        """该类别剩余可分配槽位数（空位 = 尚未被真实 token 占用的槽）。"""
        if self.flat:
            # 扁平词表没有区间语义：可见区里所有 reserved 槽都可回收
            return sum(1 for i in range(spec.INVISIBLE_END,
                                        spec.VOCAB_SIZE_PADDED)
                       if _is_reserved(self.id_to_token[i]))
        begin, end = _ALLOC_RANGES[category]
        return sum(1 for i in range(begin, end)
                   if _is_reserved(self.id_to_token[i]))

    # ------------------------------------------------------------------
    # 添加 token（四类任选，全局查重，原子持久化）
    # ------------------------------------------------------------------
    def _alloc_ranges(self, category: str):
        """按优先级返回候选分配区间列表。"""
        if self.flat:
            # 扁平词表：先吃 EXTRA（用户新增常规可见 token 的专用区），
            # 再退到可见区任意空位——这样 base 这种带散落空位的词表也
            # 能继续扩容，而不会因「尾部已被 padding 占满」直接失败。
            return [(spec.EXTRA_BEGIN, spec.REGULAR_END),
                    (spec.INVISIBLE_END, spec.VOCAB_SIZE_PADDED)]
        if category not in _ALLOC_RANGES:
            raise ValueError(f"未知 token 类别: {category!r}"
                             f"（可选 {CATEGORIES}）")
        return [_ALLOC_RANGES[category]]

    def add_token(self, text: str, category: str) -> int:
        """添加一个 token，返回新 ID。

        规则：
          1. category ∈ {regular, fallback, special, system}（扁平词表
             忽略 category，统一走空位回收）；
          2. text 非空、不以 "<reserved:" 开头（占位符命名空间）；
          3. 全局查重——text 已存在于词表任何位置即拒绝（含四类之间）；
          4. 无冲突则占用候选区间的第一个空槽，写回来源文件。
        """
        if not self.flat and category not in _ALLOC_RANGES:
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

        slot = None
        for begin, end in self._alloc_ranges(category):
            slot = next((i for i in range(begin, end)
                         if _is_reserved(self.id_to_token[i])), None)
            if slot is not None:
                break
        if slot is None:
            raise ValueError(f"{category} 类预留槽位已满，无法添加新 token")

        with _LOCK:
            # 双检：持锁后再查一次（防并发重复添加）
            if text in self.token_to_id:
                raise ValueError(f"token 冲突：{text!r} 已被并发添加")
            self.id_to_token[slot] = text
            self.token_to_id[text] = slot
            if self.flat:
                self.regular_ids[text] = slot
                self._n_user += 1
                self._max_match = max(self._max_match, len(text))
                if len(text) == 1 and "a" <= text <= "z":
                    self.fallback_ids[text] = slot
            elif category == CAT_REGULAR:
                self.regular_ids[text] = slot
            elif category == CAT_FALLBACK:
                self.fallback_ids[text] = slot
            elif category == CAT_SPECIAL:
                self.special_ids[text] = slot
            else:
                self.system_ids[text] = slot
            if not self.flat:
                self.meta.setdefault("user_added", []).append(
                    {"text": text, "category": category, "id": slot})
            self._persist()
        return slot

    def remove_token(self, text: str) -> int:
        """扁平词表软删一个可见 token：原地替换为 <del:N> 占位符，保留 ID
        位置（不令后续 token 漂移）。返回被删 token 的原 ID；其文本不再可被
        文本编码查到。非扁平（全局 json）词表不支持（走 build_vocab 重建）。"""
        if not self.flat:
            raise ValueError("remove_token 仅支持扁平（每模型/词表预设）词表")
        tid = self.regular_ids.get(text)
        if tid is None:
            raise ValueError(f"token 未收录于词表: {text!r}")
        idx = tid - spec.INVISIBLE_END
        with _LOCK:
            self.id_to_token[tid] = f"{DEL_MARKER}{idx}>"
            self.regular_ids.pop(text, None)
            self.token_to_id.pop(text, None)
            self.fallback_ids.pop(text, None)
            self._n_user -= 1
            self._persist()
        return tid

    def _persist(self) -> None:
        """原子写回加载时的来源文件（json → vocab.json / txt → vocab.txt）。"""
        if self._path is None:
            raise RuntimeError("该词表无来源文件路径，无法持久化")
        if self.flat:
            # 只写真实可见 token（显式 ID + JSON 转义）；reserved 空位不落盘，
            # 空位本身就是「文件里没有这一行」，可被后续 add_token 回收。
            entries = [(i, self.id_to_token[i])
                       for i in range(spec.INVISIBLE_END,
                                      spec.VOCAB_SIZE_PADDED)
                       if not _is_reserved(self.id_to_token[i])]
            write_flat_vocab_file(str(self._path), entries)
            self.legacy_format = False
            return
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"id_to_token": self.id_to_token,
                        "token_to_id": self.token_to_id,
                        "meta": self.meta},
                       ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8")
        tmp.replace(self._path)

    def reload(self) -> None:
        """丢弃内存映射，从磁盘重新加载（外部改动后同步）。"""
        if self.flat:
            entries, legacy = read_flat_vocab_file(str(self._path))
            fresh = Vocab.from_entries(entries, self._path, legacy=legacy)
        else:
            fresh = Vocab(self._path)
        self.__dict__.update(fresh.__dict__)


_LOCK = threading.Lock()
_VOCAB: Vocab | None = None
# 进程级「活动模型词表」覆盖：set_model_vocab 之后 get_vocab() 返回它。
# infer/train 服务进程在加载模型时设置；server 在切换活动模型时设置。
_MODEL_VOCAB: Vocab | None = None
_MODEL_VOCAB_DIR: str | None = None


def set_model_vocab(model_dir: str) -> None:
    """把进程的词表切换为 model_dir 的 vocab.txt（没有则清退到全局）。

    重复设置同一目录是廉价的（直接复用已加载实例）。"""
    global _MODEL_VOCAB, _MODEL_VOCAB_DIR
    if model_dir == _MODEL_VOCAB_DIR and _MODEL_VOCAB is not None:
        return
    got = read_model_vocab_entries(model_dir)
    if got is None:
        clear_model_vocab()
        return
    entries, legacy = got
    _MODEL_VOCAB = Vocab.from_entries(entries, model_vocab_path(model_dir),
                                      legacy=legacy)
    _MODEL_VOCAB_DIR = model_dir


def clear_model_vocab() -> None:
    """清退到全局词表（无活动模型时）。"""
    global _MODEL_VOCAB, _MODEL_VOCAB_DIR
    _MODEL_VOCAB = None
    _MODEL_VOCAB_DIR = None


def set_vocab_source(path: str) -> None:
    """CLI 便捷入口：path 是目录 → 用它的 vocab.txt；是文件 → 直接当
    vocab.txt 加载。供 enctok/dectok --vocab 使用。"""
    p = Path(path)
    if p.is_dir():
        set_model_vocab(str(p))
        return
    if not p.is_file():
        raise FileNotFoundError(f"词表来源不存在: {path}")
    global _MODEL_VOCAB, _MODEL_VOCAB_DIR
    entries, legacy = read_flat_vocab_file(str(p))
    _MODEL_VOCAB = Vocab.from_entries(entries, p, legacy=legacy)
    _MODEL_VOCAB_DIR = None


def get_vocab() -> Vocab:
    """进程级懒加载单例；有活动模型词表时优先返回它。"""
    global _VOCAB
    if _MODEL_VOCAB is not None:
        return _MODEL_VOCAB
    if _VOCAB is None:
        _VOCAB = Vocab()
    return _VOCAB


def reset_vocab_cache() -> None:
    """清空单例（测试 / 外部改写 vocab.json 后调用）。"""
    global _VOCAB
    _VOCAB = None
    clear_model_vocab()
