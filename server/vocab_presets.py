# ===========================================================================
# GARGANTUA v1 — vocab_presets.py：多词表预设管理（2026-10-25）
# ---------------------------------------------------------------------------
# 用户要求：允许用户创建多个词表预设，每个都从基础词表（现有全局词表）fork
# 出来，彼此独立隔离、可独立增删。创建模型时从选定预设 seed 出
# models/NAME/vocab.txt。
#
# 存储：vocabs/<name>/vocab.txt（扁平可见 token 列表，v2 格式带显式 ID；
# 见 common.VOCAB_FILE_MAGIC）。base 为系统预设，由全局 vocab.json 的
# 可见区派生——只导出真实 token，reserved 占位符不落盘。
#
# 增删：复用 tokens.vocab.Vocab（flat 模式）的 add_token / remove_token；
# remove_token 走软删（<del:N> 占位符），保留 ID 位置。
# ===========================================================================
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from pathlib import Path

import spec
from common import (VOCAB_FILE_MAGIC, read_flat_vocab_file,
                    write_flat_vocab_file)
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


def read_flat_entries(name: str) -> list:
    """读预设的条目 [(id, token)]（v2 显式 ID；旧格式文件就地升级）。"""
    p = vocab_txt_path(name)
    if not p.is_file():
        return []
    return read_flat_vocab_file(str(p))[0]


def read_flat_tokens(name: str) -> list:
    """读预设的 token 文本列表（按 ID 升序；展示/断言用）。"""
    return [t for _, t in read_flat_entries(name)]


def _used_count(name: str) -> int:
    """已用槽位：真实 token（排除 reserved 空位与 <del:N> 软删占位）。"""
    return sum(1 for t in read_flat_tokens(name)
               if not t.startswith(DEL_MARKER)
               and not t.startswith("<reserved:"))


def _global_visible_entries() -> list:
    """全局词表可见区的真实 token（排除 reserved 占位符），带全局 ID。"""
    data = json.loads(GLOBAL_VOCAB_JSON.read_text(encoding="utf-8"))
    out = []
    for i in range(spec.INVISIBLE_END, spec.VOCAB_SIZE_PADDED):
        t = data["id_to_token"][i]
        if t.startswith("<reserved:"):
            continue
        out.append((i, t))
    return out


def _write_preset(name: str, entries) -> None:
    preset_dir(name).mkdir(parents=True, exist_ok=True)
    write_flat_vocab_file(str(vocab_txt_path(name)), entries)


def _is_unmodified_seed(entries) -> bool:
    """旧文件是否只是「全局可见区 join 出来的副本」（无用户改动）。"""
    g = _global_visible_entries()
    gset = {t for _, t in g}
    toks = [t for _, t in entries]
    if not toks:
        return True
    if len(toks) > len(g):
        return False                       # 多出来的只能是用户新增
    if not all(t in gset for t in toks):
        return False
    # 旧格式必然丢掉 "\n"/"\r" 这类无法用行表示的 token（数量极少）
    return len(g) - len(toks) <= 8


def _upgrade_preset(name: str) -> bool:
    """把旧格式（v1）预设升级为 v2；返回是否发生了升级。

    base 与「未编辑过的 fork」按全局可见区重写——它们的 ID 本就应与全局
    一致，旧文件里被空行吞掉的 "\\n"/"\\r" 与 reserved 占位符一并清掉。
    用户改过的预设只换表示法、不重排 ID（已训练的权重行对应关系不动）。
    """
    txt = vocab_txt_path(name)
    # 快路径：已是 v2 就只读首行，避免每次列预设都全文解析 6 万条
    with open(txt, encoding="utf-8") as f:
        if f.readline().strip() == VOCAB_FILE_MAGIC:
            return False
    entries, legacy = read_flat_vocab_file(str(txt))
    if not legacy:
        return False
    real = [(i, t) for i, t in entries if not t.startswith("<reserved:")]
    if _is_unmodified_seed(real):
        _write_preset(name, _global_visible_entries())
    else:
        print(f"[vocab_presets] {name}: 旧格式词表就地升级（ID 保持不变）",
              file=sys.stderr)
        _write_preset(name, real)
    return True


# ---------------------------------------------------------------------------
# 预设生命周期
# ---------------------------------------------------------------------------
def ensure_base() -> None:
    """确保 base 预设存在：从全局 vocab.json 的可见区派生。幂等。

    2026-10-27 审计修复（H1）：只导出真实 token（显式 ID + JSON 转义），
    不再把 9914 个 reserved 占位符写进文件——旧实现让可见区一上来就被
    占满（free_slots=0），任何「添加 token」都必然失败，且 token 自身的
    "\\n"/"\\r" 会把整段 ID 顶前 2 位。旧格式文件在首次访问时就地升级。
    """
    d = preset_dir(BASE)
    txt = vocab_txt_path(BASE)
    if txt.is_file():
        _upgrade_preset(BASE)
        return
    d.mkdir(parents=True, exist_ok=True)
    _write_preset(BASE, _global_visible_entries())


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
            _upgrade_preset(child.name)
            out.append({
                "name": child.name,
                "used": _used_count(child.name),
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
    if source != BASE:
        _upgrade_preset(source)
    preset_dir(name).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, vocab_txt_path(name))
    return {"name": name, "source": source, "used": _used_count(name),
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
    v = Vocab.from_entries(read_flat_entries(name), vocab_txt_path(name))
    tid = v.add_token(text, category)
    return {"token": text, "id": tid, "category": category,
            "vocab": name, "used": _used_count(name)}


def remove_token(name: str, text: str) -> dict:
    ensure_base()
    v = Vocab.from_entries(read_flat_entries(name), vocab_txt_path(name))
    tid = v.remove_token(text)
    return {"token": text, "id": tid, "vocab": name,
            "used": _used_count(name)}


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
    """某个预设的可见区占用（used = 真实 token 数）。"""
    used = _used_count(name)
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
