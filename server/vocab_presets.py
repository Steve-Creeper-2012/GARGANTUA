# ===========================================================================
# GARGANTUA v1 — vocab_presets.py：多词表预设管理（2026-10-25）
# ---------------------------------------------------------------------------
# 用户要求：允许用户创建多个词表预设，每个都从基础词表（现有全局词表）fork
# 出来，彼此独立隔离、可独立增删。创建模型时从选定预设 seed 出
# models/NAME/vocab.txt。
#
# 存储：vocabs/<name>/vocab.txt（扁平可见 token 列表，行 N → ID INVISIBLE_END+N，
# 与每模型词表同契约）。base 为系统预设，由全局 vocab.json 的可见区派生。
#
# 增删：复用 tokens.vocab.Vocab（flat 模式）的 add_token / remove_token；
# remove_token 走软删（<del:N> 占位符），保留 ID 位置。
# ===========================================================================
from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path

import spec
from tokens.vocab import DEL_MARKER, Vocab

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOCABS_DIR = PROJECT_ROOT / "vocabs"
BASE = "base"
GLOBAL_VOCAB_JSON = PROJECT_ROOT / "service" / "tokens" / "vocab.json"

_LOCK = threading.Lock()


def _valid_name(name) -> str | None:
    if not name or not isinstance(name, str):
        return "词表预设名不能为空"
    if len(name) > 64:
        return "词表预设名过长（>64）"
    import re
    if re.search(r'[^A-Za-z0-9_-]', name):
        return "词表预设名只能含字母/数字/下划线/连字符"
    if name in (".", ".."):
        return "非法名称"
    return None


# ---------------------------------------------------------------------------
# 读取 / 写入扁平词表文件
# ---------------------------------------------------------------------------
def preset_dir(name: str) -> Path:
    return VOCABS_DIR / name


def vocab_txt_path(name: str) -> Path:
    return preset_dir(name) / "vocab.txt"


def read_flat_tokens(name: str) -> list:
    """读预设的扁平 token 列表（跳过空行；保留 <del:N> 占位符）。"""
    p = vocab_txt_path(name)
    if not p.is_file():
        return []
    out = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        t = raw.rstrip("\n").rstrip("\r")
        if not t:
            continue
        out.append(t)
    return out


def _global_visible_tokens() -> list:
    data = json.loads(GLOBAL_VOCAB_JSON.read_text(encoding="utf-8"))
    return data["id_to_token"][spec.INVISIBLE_END:]


# ---------------------------------------------------------------------------
# 预设生命周期
# ---------------------------------------------------------------------------
def ensure_base() -> None:
    """确保 base 预设存在：从全局 vocab.json 的可见区派生。幂等。"""
    d = preset_dir(BASE)
    txt = vocab_txt_path(BASE)
    if txt.is_file():
        return
    d.mkdir(parents=True, exist_ok=True)
    tokens = _global_visible_tokens()
    txt.write_text("\n".join(tokens) + "\n", encoding="utf-8")


def list_presets() -> list:
    """列出所有预设：[{name, used, capacity, is_base}]。"""
    ensure_base()
    out = []
    if VOCABS_DIR.is_dir():
        for child in sorted(VOCABS_DIR.iterdir()):
            if not child.is_dir():
                continue
            txt = child / "vocab.txt"
            if not txt.is_file():
                continue
            used = sum(1 for t in read_flat_tokens(child.name)
                       if not t.startswith(DEL_MARKER))
            out.append({
                "name": child.name,
                "used": used,
                "capacity": spec.VOCAB_SIZE_PADDED - spec.INVISIBLE_END,
                "is_base": child.name == BASE,
            })
    return out


def create_preset(name: str, source: str = BASE) -> dict:
    """从 source 预设 fork 出一个新预设（独立隔离，复制 token 列表）。"""
    bad = _valid_name(name)
    if bad:
        raise ValueError(bad)
    if (preset_dir(name) / "vocab.txt").exists():
        raise ValueError(f"词表预设已存在: {name}")
    ensure_base()
    src = vocab_txt_path(source)
    if not src.is_file():
        raise ValueError(f"源预设不存在: {source}")
    preset_dir(name).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, vocab_txt_path(name))
    used = sum(1 for t in read_flat_tokens(name)
               if not t.startswith(DEL_MARKER))
    return {"name": name, "source": source, "used": used,
            "capacity": spec.VOCAB_SIZE_PADDED - spec.INVISIBLE_END}


def delete_preset(name: str) -> None:
    """删除一个用户预设（base 不可删）。"""
    if name == BASE:
        raise ValueError("base 预设不可删除")
    bad = _valid_name(name)
    if bad:
        raise ValueError(bad)
    d = preset_dir(name)
    if not d.is_dir():
        raise ValueError(f"词表预设不存在: {name}")
    shutil.rmtree(d)


# ---------------------------------------------------------------------------
# 增删 token（复用 tokens.vocab 的 flat 语义）
# ---------------------------------------------------------------------------
def add_token(name: str, text: str, category: str = "regular") -> dict:
    ensure_base()
    tokens = read_flat_tokens(name)
    v = Vocab.from_tokens(tokens, vocab_txt_path(name))
    tid = v.add_token(text, category)
    return {"token": text, "id": tid, "category": category,
            "vocab": name,
            "used": sum(1 for t in read_flat_tokens(name)
                       if not t.startswith(DEL_MARKER))}


def remove_token(name: str, text: str) -> dict:
    ensure_base()
    tokens = read_flat_tokens(name)
    v = Vocab.from_tokens(tokens, vocab_txt_path(name))
    tid = v.remove_token(text)
    return {"token": text, "id": tid, "vocab": name,
            "used": sum(1 for t in read_flat_tokens(name)
                       if not t.startswith(DEL_MARKER))}


# ---------------------------------------------------------------------------
# 模型 seed：把选定预设的 vocab.txt 复制进新模型目录
# ---------------------------------------------------------------------------
def seed_model_vocab(model_dir: str, preset_name: str | None = None) -> None:
    """把预设 vocab.txt 复制到 models/NAME/vocab.txt。
    preset_name 缺省/为 base → 用 base；其余用 vocabs/<name>/vocab.txt。"""
    name = preset_name or BASE
    ensure_base()
    src = vocab_txt_path(name)
    if not src.is_file():
        raise ValueError(f"词表预设不存在: {name}")
    os.makedirs(model_dir, exist_ok=True)
    shutil.copyfile(src, os.path.join(model_dir, "vocab.txt"))


# ---------------------------------------------------------------------------
# 区间占用（给前端「词表」页面）：全局 12 子区 → 4 大类，含 used/padded
# ---------------------------------------------------------------------------
def global_ranges() -> dict:
    """返回全局词表的区间占用：12 个子区，归类到 4 大类，含 used/padded。"""
    v = Vocab()  # 全局词表
    id_to_token = v.id_to_token

    def used_in(begin: int, end: int) -> int:
        return sum(1 for i in range(begin, end)
                   if not id_to_token[i].startswith("<reserved:"))

    # 子区定义（与 build_vocab.py meta["ranges"] 同源，全部取自 spec）
    sub = [
        ("special", "special", spec.SPECIAL_BEGIN, spec.SPECIAL_SIZE),
        ("system", "system", spec.SYSTEM_BEGIN, spec.SYSTEM_SIZE),
        ("modal(draw)", "special", spec.MODAL_BEGIN, spec.MODAL_SIZE),
        ("digit", "regular", spec.DIGIT_BEGIN, spec.DIGIT_SIZE),
        ("whitespace", "regular", spec.WHITESPACE_BEGIN, spec.WHITESPACE_SIZE),
        ("symbol", "regular", spec.SYMBOL_BEGIN, spec.SYMBOL_SIZE),
        ("emoji", "regular", spec.EMOJI_BEGIN, spec.EMOJI_SIZE),
        ("cjk", "regular", spec.CJK_BEGIN, spec.CJK_SIZE),
        ("english", "regular", spec.ENGLISH_BEGIN, spec.ENGLISH_SIZE),
        ("extra", "regular", spec.EXTRA_BEGIN, spec.EXTRA_SIZE),
        ("fallback", "fallback", spec.FALLBACK_BEGIN, spec.FALLBACK_SIZE),
    ]
    regions = []
    for name, cat, begin, size in sub:
        regions.append({
            "region": name, "category": cat,
            "begin": begin, "size": size,
            "used": used_in(begin, begin + size),
            "padded": size,
        })
    return {
        "vocab_size_padded": spec.VOCAB_SIZE_PADDED,
        "invisible_end": spec.INVISIBLE_END,
        "regular_end": spec.REGULAR_END,
        "categories": ["regular", "fallback", "special", "system"],
        "regions": regions,
        "category_totals": {
            "regular": sum(r["used"] for r in regions
                          if r["category"] == "regular"),
            "fallback": sum(r["used"] for r in regions
                           if r["category"] == "fallback"),
            "special": sum(r["used"] for r in regions
                          if r["category"] == "special"),
            "system": sum(r["used"] for r in regions
                         if r["category"] == "system"),
        },
    }


def preset_occupancy(name: str) -> dict:
    """某个预设的可见区占用（行号 → ID INVISIBLE_END+N）。"""
    tokens = read_flat_tokens(name)
    used = sum(1 for t in tokens if not t.startswith(DEL_MARKER))
    capacity = spec.VOCAB_SIZE_PADDED - spec.INVISIBLE_END
    return {"name": name, "used": used, "capacity": capacity,
            "free": capacity - used}


# ---------------------------------------------------------------------------
# 模型创建时：config 记录 vocab 来源 + dtype（common.build_model 据此施加）
# ---------------------------------------------------------------------------
def validate_preset(name: str | None) -> str:
    """解析模型创建时指定的词表预设名；缺省/base → base。"""
    if name is None or name == BASE:
        return BASE
    if not (preset_dir(name) / "vocab.txt").is_file():
        raise ValueError(f"词表预设不存在: {name}")
    return name
