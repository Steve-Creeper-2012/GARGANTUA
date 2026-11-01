# ===========================================================================
# GARGANTUA v1 — vocab_presets.py：多词表管理（2026-10-30 契约重构）
# ---------------------------------------------------------------------------
# 每个词表是一个**文件夹**：
#     vocabs/<name>/
#       vocab.vfc      词表正文（.vfc 格式，见 tokens/vfc.py）
#       config.json    词表配置（含 vocab_size 与分区表）
#
# - base 是系统预设（main），**禁止直接编辑/删除**；要改就先 Fork 一份。
# - Fork = 整目录复制（vfc + config），新词表与源彼此独立、可独立增删。
# - 创建模型时把选定词表的**两个文件都复制**进 models/NAME/vocab/
#   （即使多个模型复用同一词表也各自持有一份——否则用户误删共享词表，
#    模型就彻底不能工作了）。
#
# 词表不变量（由 tokens/vfc.py 与 tokens/vocab.py 强制）：
#   - ID 严格连续 0..N-1，**不允许空位**（空位也要占一个 embedding 行）
#   - token 文本全表唯一
#   - 前 HEAD_SIZE 个 ID 是架构契约头（special/system/draw），不可删改
#
# 安全：所有路径拼接都先过 vocab_dir() 的名字校验 + realpath 前缀校验，
# 防止 HTTP 入参 name / new_name 穿越写出到任意 vocab.vfc。
# ===========================================================================
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

import spec
from tokens import vfc
from tokens.vocab import (VOCAB_DIR_NAME, VOCAB_FILE_NAME,
                          DEFAULT_APPEND_REGION, Vocab, model_vocab_dir)

# 词表目录下的配置文件名
VOCAB_CONFIG_NAME = spec.VOCAB_CONFIG_NAME

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOCABS_DIR = PROJECT_ROOT / "vocabs"
BASE = "base"
VOCAB_FOLDER_FORMAT = "gargantua-vocab-folder-v1"

GLOBAL_VOCAB_DIR = PROJECT_ROOT / "service" / "tokens"
GLOBAL_VOCAB_VFC = GLOBAL_VOCAB_DIR / VOCAB_FILE_NAME

# 词表级互斥：create/delete/write 会在内部互相调用（write_vocab_text →
# create_vocab），必须是可重入锁，否则同一线程自锁死。
_LOCK = threading.RLock()

# ---------------------------------------------------------------------------
# Vocab 实例缓存（2026-10-31 性能）
# ---------------------------------------------------------------------------
# load_vocab 每次全量解析 .vfc（base ≈ 6.9 万行，含转义/唯一性/契约头
# 校验，数百毫秒级），而 regions/query/occupancy/list 全都走它——一次
# 页面刷新能解析七八遍。按 (mtime_ns, size) 键控缓存：文件没变就直接复
# 用。写路径（add/remove/fork 写入）落盘后 mtime 必变 → 缓存自然失效，
# 所以失效逻辑是自洽的；只有读路径共享实例，且读路径不变异词表。
_CACHE_LIMIT = 8
_VCACHE: dict = {}
_VCACHE_ORDER: list = []


def _cache_get(key: str, sig: tuple):
    hit = _VCACHE.get(key)
    if hit and hit[0] == sig:
        return hit[1]
    return None


def _cache_put(key: str, sig: tuple, v) -> None:
    if key not in _VCACHE:
        _VCACHE_ORDER.append(key)
        while len(_VCACHE_ORDER) > _CACHE_LIMIT:
            _VCACHE.pop(_VCACHE_ORDER.pop(0), None)
    _VCACHE[key] = (sig, v)


# ---------------------------------------------------------------------------
# 名字校验 / 路径安全
# ---------------------------------------------------------------------------
def _valid_name(name) -> str | None:
    if not name or not isinstance(name, str):
        return "词表名不能为空"
    if len(name) > 64:
        return "词表名过长（>64）"
    if name in (".", ".."):
        return "非法名称"
    for ch in name:
        if not (ch.isascii() and (ch.isalnum() or ch in "_-")):
            return "词表名只能含字母/数字/下划线/连字符"
    return None


def _assert_within_vocabs(name, path) -> None:
    """realpath 前缀校验：解析符号链接后路径仍须落在 VOCABS_DIR 内。"""
    base = os.path.realpath(VOCABS_DIR)
    target = os.path.realpath(path)
    try:
        inside = os.path.commonpath([base, target]) == base
    except ValueError:          # 跨盘符 / 混合绝对相对路径：无公共前缀
        inside = False
    if not inside:
        raise ValueError(f"非法词表名: {name!r}")


def vocab_dir(name) -> Path:
    """vocabs/<name>/ —— 所有词表读写的唯一路径入口（统一在此校验）。"""
    if _valid_name(name) is not None:
        raise ValueError(f"非法词表名: {name!r}")
    p = VOCABS_DIR / name
    _assert_within_vocabs(name, p)   # 防 name 为符号链接目录绕过
    return p


def vocab_vfc_path(name) -> Path:
    """词表正文文件路径：vocabs/<name>/vocab.vfc。"""
    p = vocab_dir(name) / VOCAB_FILE_NAME
    # 目录层已校验；这里再对最终文件做一次前缀校验（纵深防御，防目录内
    # vocab.vfc 被换成指向外部文件的符号链接）。
    _assert_within_vocabs(name, p)
    return p


def config_path(name) -> Path:
    p = vocab_dir(name) / VOCAB_CONFIG_NAME
    _assert_within_vocabs(name, p)
    return p


# ---------------------------------------------------------------------------
# config.json 读写
# ---------------------------------------------------------------------------
def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def read_config(name: str) -> dict:
    """读 vocabs/<name>/config.json；缺失/损坏返回 {}。"""
    p = config_path(name)
    if not p.is_file():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception as e:
        print(f"[vocab_presets] config.json 读取失败（{name}）: {e}",
              file=sys.stderr)
        return {}


def write_config(name: str, cfg: dict) -> None:
    vocab_dir(name).mkdir(parents=True, exist_ok=True)
    _write_json_atomic(config_path(name), cfg)


def _regions_from_cfg(cfg: dict) -> list:
    rs = cfg.get("regions")
    if not isinstance(rs, list) or not rs:
        return []
    out = []
    for r in rs:
        if not isinstance(r, dict):
            continue
        out.append({"region": str(r.get("region", "?")),
                    "category": str(r.get("category", "regular")),
                    "size": max(int(r.get("size", 0)), 0)})
    return out


def _sync_config(name: str, n: int | None = None) -> dict:
    """按当前 vocab.vfc 的实际内容刷新 config.json。

    两件事：
      1. `vocab_size` 与正文条数对齐（正文是唯一事实源）；
      2. 各分区 size 之和必须等于正文条数。不等时**只调整 extra 区**的
         size 补齐差额，而不是把整张分区表塌成一个 "visible" 区——后者
         会让用户每次手工编辑正文（模式二）就丢掉全部分区信息。
         假定「正文编辑的增删落在 extra 区」，这是模式二最常见的情形；
         若用户删的是 cjk 区中间某行，分区边界会偏一点（表格仍自洽），
         属于可接受的近似，远好过整表消失。

    n 可传入已统计好的正文条数（调用方刚解析过 .vfc 时避免二次全量
    解析——base ≈ 6.9 万行，一次解析数百毫秒）。
    """
    if n is None:
        n = sum(1 for _ in vfc.read_vfc(str(vocab_vfc_path(name))))
    cfg = read_config(name)
    regions = _regions_from_cfg(cfg)
    total = sum(r["size"] for r in regions)
    if not regions:
        regions = [{"region": "visible", "category": "regular", "size": n}]
    elif total != n:
        delta = n - total
        target = next((r for r in regions if r["region"] == "extra"), None)
        if target is None:
            target = {"region": "extra", "category": "regular", "size": 0}
            regions.append(target)
        target["size"] = max(target["size"] + delta, 0)
        # 极端情况下 extra 被压到 0 还差（用户删得比 extra 还多）：把剩余
        # 差额摊到最后一个非空区，保证 sum == n 这条硬约束成立。
        rest = n - sum(r["size"] for r in regions)
        if rest:
            for r in reversed(regions):
                if r["size"] + rest >= 0:
                    r["size"] += rest
                    break
    cfg["name"] = name
    cfg["format"] = VOCAB_FOLDER_FORMAT
    cfg["is_main"] = (name == BASE)
    cfg["regions"] = regions
    cfg["vocab_size"] = n
    old = read_config(name)
    if old != cfg:                    # 内容没变就不写盘（list 每次都会调）
        write_config(name, cfg)
    return cfg


def save_vocab_config(name: str, v: Vocab) -> None:
    """把内存词表的分区表写进 config.json。

    调用前提：正文已经落盘（Vocab.add_token / remove_token 内部会写
    .vfc）。这里只同步元数据，顺序上永远「先正文、后配置」——配置里的
    vocab_size 不会指向一份还没写下去的正文。
    """
    cfg = read_config(name)
    cfg["name"] = name
    cfg["format"] = VOCAB_FOLDER_FORMAT
    cfg["is_main"] = (name == BASE)
    cfg["regions"] = [{"region": r["region"], "category": r["category"],
                       "size": r["size"]} for r in v.regions]
    cfg["vocab_size"] = len(v)
    write_config(name, cfg)


# ---------------------------------------------------------------------------
# 全局基础词表 → base 预设
# ---------------------------------------------------------------------------
def _global_config() -> dict:
    """全局词表的 config.json（分区表 / user_added 重放源）。"""
    p = GLOBAL_VOCAB_DIR / "config.json"
    if not p.is_file():
        return {}
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
        return cfg if isinstance(cfg, dict) else {}
    except (ValueError, OSError) as e:
        print(f"[vocab_presets] {p} 读取失败: {e}", file=sys.stderr)
        return {}


def _global_regions() -> list:
    """全局词表的分区表（从 build_vocab 产出的 config 里取）。"""
    rs = _global_config().get("regions")
    if isinstance(rs, list) and rs:
        return [{"region": str(r.get("region", "?")),
                 "category": str(r.get("category", "regular")),
                 "size": max(int(r.get("size", 0)), 0)}
                for r in rs if isinstance(r, dict)]
    return []


def ensure_base() -> None:
    """确保 base 词表存在（从全局基础词表派生）。幂等。

    派生规则：正文逐条复制全局 vocab.vfc（已经是连续编号且带头部契约），
    分区表取全局 meta 的 regions；两者条数不一致时以正文为准兜底。

    已存在时的健康检查（2026-10-31 性能版）：只做 mtime 比较而不是每次
    全量解析 .vfc —— config.json 比 vocab.vfc 新 = 上次同步后正文没被
    手改过，直接返回（本函数被每个词表 API 调用，全量解析会把每个请求
    拖慢数百毫秒）。mtime 拿不到时保守起见照旧同步。
    """
    d = vocab_dir(BASE)
    txt = vocab_vfc_path(BASE)
    if txt.is_file() and config_path(BASE).is_file():
        try:
            drift = txt.stat().st_mtime > config_path(BASE).stat().st_mtime
        except OSError:
            drift = True
        if not drift:
            return
        _sync_config(BASE)
        return
    d.mkdir(parents=True, exist_ok=True)
    entries = Vocab.global_vocab().to_entries()
    vfc.write_vfc(str(txt), entries)
    regions = _global_regions()
    if sum(r["size"] for r in regions) != len(entries):
        regions = [{"region": "visible", "category": "regular",
                    "size": len(entries)}]
    write_config(BASE, {
        "name": BASE,
        "format": VOCAB_FOLDER_FORMAT,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": None,
        "description": "基础词表（系统预设，不可直接编辑）",
        "is_main": True,
        "regions": regions,
        "vocab_size": len(entries),
    })


# ---------------------------------------------------------------------------
# 列出 / 创建（Fork）/ 复制 / 删除
# ---------------------------------------------------------------------------
def list_vocabs() -> list:
    """列出所有词表：[{name, vocab_size, visible_size, invisible_size,
    is_main, source, description}]。"""
    ensure_base()
    out = []
    if not VOCABS_DIR.is_dir():
        return out
    for child in sorted(VOCABS_DIR.iterdir()):
        if not child.is_dir():
            continue
        if _valid_name(child.name) is not None:
            continue                      # 跳过非法名目录（底层会拒这些名字）
        txt = child / VOCAB_FILE_NAME
        if not txt.is_file():
            continue
        cfg = read_config(child.name)
        # 快路径：config.json 比 vocab.vfc 新 → 上次同步后正文没被改过，
        # 直接信 config 的 vocab_size，省一次全量解析。条数以正文为事实
        # 源，所以任何疑似漂移（正文更新）都走慢路径重解析 + 重同步。
        n = None
        cfg_file = child / VOCAB_CONFIG_NAME
        try:
            drifted = txt.stat().st_mtime > cfg_file.stat().st_mtime
        except OSError:
            drifted = True
        decl = cfg.get("vocab_size")
        if not drifted and isinstance(decl, int) and not isinstance(
                decl, bool) and decl >= 1:
            n = decl
        else:
            try:
                n = sum(1 for _ in vfc.read_vfc(str(txt)))
            except vfc.VfcError as e:
                print(f"[vocab_presets] 跳过损坏词表 {child.name}: {e}",
                      file=sys.stderr)
                continue
            cfg = _sync_config(child.name, n=n)
        out.append({
            "name": child.name,
            "vocab_size": n,
            "visible_size": max(n - spec.HEAD_SIZE, 0),
            "invisible_size": min(n, spec.HEAD_SIZE),
            "is_main": child.name == BASE,
            "source": cfg.get("source"),
            "description": cfg.get("description", ""),
        })
    return out


def create_vocab(name: str, source: str = BASE) -> dict:
    """从 source 词表 Fork 出新词表（整目录复制，彼此独立）。"""
    bad = _valid_name(name)
    if bad:
        raise ValueError(bad)
    ensure_base()
    with _LOCK:
        src_dir = vocab_dir(source)
        if not (src_dir / VOCAB_FILE_NAME).is_file():
            raise ValueError(f"源词表不存在: {source}")
        dst_dir = vocab_dir(name)
        if dst_dir.exists():
            # 只认「已经是一张词表」的目录；空目录/残留目录不该阻止创建，
            # 但也不能 copytree 到已存在目录（会 FileExistsError → 500）。
            if (dst_dir / VOCAB_FILE_NAME).exists():
                raise ValueError(f"词表已存在: {name}")
            if any(dst_dir.iterdir()):
                raise ValueError(f"目标目录已存在且非空: {name}")
            dst_dir.rmdir()
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_dir, dst_dir)
        cfg = read_config(name)
        cfg["name"] = name
        cfg["source"] = source
        cfg["is_main"] = False
        cfg["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        # 描述不继承源词表：fork 出来的是一张新表，「基础词表（系统预设）」
        # 这种描述跟它没关系，还会误导用户以为它也是 main。
        cfg["description"] = f"从 {source} fork"
        write_config(name, cfg)
    return {"name": name, "source": source,
            "vocab_size": int(cfg.get("vocab_size", 0))}



def delete_vocab(name: str) -> None:
    """删除一个词表（base 不可删）。"""
    if name == BASE:
        raise ValueError(
            "base 词表不可删除：它是系统预设，其它词表都由它 fork 而来")
    bad = _valid_name(name)
    if bad:
        raise ValueError(bad)
    with _LOCK:
        d = vocab_dir(name)
        if not d.is_dir():
            raise ValueError(f"词表不存在: {name}")
        shutil.rmtree(d)


# ---------------------------------------------------------------------------
# 载入词表 / 读取与写回正文（编辑器模式二用）
# ---------------------------------------------------------------------------
def load_vocab(name: str, fresh: bool = False) -> Vocab:
    """载入词表（含头部契约与连续性校验）。

    fresh=True 跳过缓存拿全新实例：写路径（add/remove）会通过
    Vocab.add_token 的 __dict__.update **就地变异**实例——共享缓存实例
    会把变异暴露给并发读线程（属性逐个换新，中途读到的组合不一致）。
    读路径共享实例是安全的：它们不变异。
    """
    ensure_base()
    p = vocab_vfc_path(name)
    if not fresh:
        try:
            st = p.stat()
            sig = (st.st_mtime_ns, st.st_size)
            hit = _cache_get(str(p), sig)
            if hit is not None:
                return hit
        except OSError:
            pass
    v = Vocab.load(p)
    if not fresh:
        try:
            st = p.stat()
            _cache_put(str(p), (st.st_mtime_ns, st.st_size), v)
        except OSError:
            pass
    return v


def read_vocab_text(name: str) -> str:
    """返回 vocab.vfc 的原文（供文本编辑器显示）。"""
    ensure_base()
    return vocab_vfc_path(name).read_text(encoding="utf-8")


def write_vocab_text(name: str, text: str, mode: str = "this",
                     new_name: str | None = None) -> str:
    """校验并写回 .vfc 正文；返回实际写入的词表名。

    mode="this"  直接改 name 这个词表（name 是 main → 拒绝）
    mode="fork"  从 name fork 出新词表再写入（new_name 必填）

    校验失败抛 ValueError（→400），携带逐行错误列表（见 VfcError.errors）。
    """
    ensure_base()
    if mode not in ("this", "fork"):
        raise ValueError(f"mode 必须是 this/fork: {mode!r}")
    if not isinstance(text, str):
        raise ValueError(f"text 必须是字符串: {type(text).__name__}")
    if len(text) > spec.MAX_VOCAB_SIZE * 64:
        # 粗粒度守卫：正常文件每行远小于 64 字节
        raise ValueError("词表正文过长")

    errs = vfc.validate_vfc(text, source=str(vocab_vfc_path(name)))
    if errs:
        raise VfcListError(errs)

    with _LOCK:
        if mode == "fork":
            if not new_name:
                raise ValueError("fork 模式必须提供 new_name")
            create_vocab(new_name, source=name)
            target = new_name
        else:
            if name == BASE:
                raise ValueError(
                    "main 词表不可直接编辑：它是系统预设，被其它词表与模型共享。"
                    "请改用 Fork（先复制一份再改），或选择另一个派生词表。")
            target = name

        vfc.write_vfc(str(vocab_vfc_path(target)), vfc.parse_vfc(text))
        _sync_config(target)
    return target


class VfcListError(ValueError):
    """批量语法错误：携带 errors 列表，供前端逐行高亮。"""

    def __init__(self, errors):
        self.errors = errors
        super().__init__("; ".join(
            (f"第 {e['line']} 行：{e['message']}" if e.get("line")
             else e["message"])
            for e in errors[:5]))


# ---------------------------------------------------------------------------
# 增删 token
# ---------------------------------------------------------------------------
def _require_editable(name: str) -> None:
    """直接编辑的闸门：main 词表必须先 Fork。"""
    if name == BASE:
        raise ValueError(
            "main 词表不可直接编辑：它是系统预设，任何改动都会影响由它 fork "
            "出去的其它词表与已建模型。请先 Fork 出一份再编辑。")


def add_token_to(name: str, text: str, category: str = "regular") -> dict:
    """给词表加一个 token（紧凑插入指定子区末尾，不留空位）。"""
    ensure_base()
    _require_editable(name)
    with _LOCK:
        v = load_vocab(name, fresh=True)   # 写路径：实例会被就地变异
        tid = v.add_token(text, category)
        save_vocab_config(name, v)
    return {"vocab": name, "token": text, "category": category, "id": tid,
            "vocab_size": len(v)}


def remove_token_from(name: str, text: str) -> dict:
    """从词表删一个 token（其后 ID 整体前移，不留空位）。"""
    ensure_base()
    _require_editable(name)
    with _LOCK:
        v = load_vocab(name, fresh=True)   # 写路径：实例会被就地变异
        tid = v.remove_token(text)
        save_vocab_config(name, v)
    return {"vocab": name, "token": text, "id": tid,
            "vocab_size": len(v)}


def query_token(name: str, text: str) -> dict:
    """在词表里查一个 token（可显示与不可显示都查）。

    用 token_to_id 而不是 lookup()：后者只覆盖可显示 token（enctok 的
    编码语义），而「查询」这个功能对 <stop> / <caps> 同样有意义。
    """
    ensure_base()
    v = load_vocab(name)
    tid = v.token_to_id.get(text)
    if tid is None and not text.startswith("<"):
        # 允许用裸名查不可显示 token（stop → <stop>）
        tid = v.special_ids.get(f"<{text}>")
    return {"vocab": name, "token": text, "found": tid is not None,
            "id": tid, "category": (v.category_of(tid)
                                    if tid is not None else None),
            "invisible": (v.is_invisible(tid) if tid is not None else None)}


# ---------------------------------------------------------------------------
# 模型 seed：整目录复制
# ---------------------------------------------------------------------------
def seed_model_vocab(model_dir: str, preset_name: str | None = None) -> None:
    """把词表的 vocab.vfc + config.json 复制进 models/NAME/vocab/。

    即使多个模型复用同一词表也各自持有一份——用户误删共享词表不该让模型
    报废（steve 2026-10-30 定的契约）。
    """
    name = preset_name or BASE
    ensure_base()
    src_dir = vocab_dir(name)
    if not (src_dir / VOCAB_FILE_NAME).is_file():
        raise ValueError(f"词表不存在: {name}")
    dst_dir = model_vocab_dir(model_dir)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    os.makedirs(model_dir, exist_ok=True)
    shutil.copytree(src_dir, dst_dir)
    cfg = read_config(name)
    return {"preset": name, "vocab_size": int(cfg.get("vocab_size", 0))}


def validate_vocab(name: str | None) -> str:
    """解析创建模型时指定的词表名；缺省/base → base。"""
    if name is None or name == BASE:
        return BASE
    if not (vocab_dir(name) / VOCAB_FILE_NAME).is_file():
        raise ValueError(f"词表不存在: {name}")
    return name


def vocab_occupancy(name: str) -> dict:
    """某个词表的占用统计（前端标签页「分区」区用）。

    每个分区额外带 samples（该区前几个 token 文本）：前端表格要回答
    「这一块到底覆盖什么 token」，光给 ID 区间对用户没有意义。
    """
    ensure_base()
    v = load_vocab(name)
    n = len(v)
    regions = []
    for r in v.regions:
        begin, size = r["begin"], r["size"]
        samples = []
        for i in range(begin, min(begin + size, n)):
            t = v.id_to_token[i]
            if t.startswith("<reserved:"):
                continue
            samples.append(t if len(t) <= 16 else t[:15] + "…")
            if len(samples) >= 4:
                break
        regions.append({**r, "samples": samples})
    return {"name": name, "vocab_size": n,
            "visible_size": max(n - spec.HEAD_SIZE, 0),
            "invisible_size": min(n, spec.HEAD_SIZE),
            "is_main": name == BASE,
            "regions": regions,
            "category_totals": v.category_totals()}
