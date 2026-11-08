# ===========================================================================
# GARGANTUA v1 — gen_vocab.py：按「用户决定的预留槽位数」从 base 生成新词表
# ---------------------------------------------------------------------------
# 背景（2026-11-06 steve 定案）：spec **不定死**预留槽位数量，由词表自己的
# config.json 决定。**base 词表零预留**（slot_count=0）；要预留槽位就跑本脚本，
# 以 base 为基础生成一份新词表，新词表的 config 里写明 slot_count=N。
#
# 用法：
#   python service/tokens/gen_vocab.py --name mymodel --slots 128
#   python service/tokens/gen_vocab.py --name mymodel --slots 0 --force
#   python service/tokens/gen_vocab.py --name mymodel --slots 64 --activate
#
#   --base      模板词表目录（默认 vocabs/base1.3.0）
#   --force     目标目录已存在时覆盖
#   --activate  同时把结果同步为全局 service/tokens/{vocab.vfc, config.json}
#               （改的是「运行时布局」，需要重启 server 让 spec 重新 import）
#
# 生成结果：
#   vocabs/<name>/{vocab.vfc, config.json}
#   头部 = core 3 + boundary 6 + system 3 + slot N + typst 228 + draw 9605
#   DRAW_BEGIN = 240 + N，HEAD_SIZE = 9845 + N，draw / 可见区 ID 顺移 N。
#   （base1.3.0 起 typst 段在 slot 之后、draw 之前；它落在 tail 里，
#     所以本脚本只重排 slot，typst / unicode_ext 自动原样跟随。）
# ===========================================================================
"""gen_vocab: base + 用户指定的预留槽位数 → 一份新词表文件夹。"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "start"))
import boot  # noqa: E402  (把 server/ 放进 sys.path)
boot.bootstrap()

import spec  # noqa: E402
from tokens import vfc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VOCABS = ROOT / "vocabs"
GLOBAL = ROOT / "service" / "tokens"
NAME_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
FIXED_PREFIX = spec.CORE_SIZE + spec.BOUNDARY_SIZE + spec.SYSTEM_SIZE  # 12


def _validate_name(name: str) -> None:
    if not NAME_RE.fullmatch(name) or name in (".", "..") or ".." in name:
        raise ValueError(f"非法词表名: {name!r}（只允许 1..64 位 "
                         r"[A-Za-z0-9_.-]，且不能含 '..'）")
    if name == spec.BASE_VOCAB_NAME:
        raise ValueError("不能覆盖 base 词表；base 永远零预留")


def _read_base(base_dir: Path):
    vfc_path = base_dir / "vocab.vfc"
    cfg_path = base_dir / "config.json"
    if not vfc_path.is_file() or not cfg_path.is_file():
        raise ValueError(f"模板词表不完整: {base_dir}")
    text = vfc_path.read_text(encoding="utf-8")
    errs = vfc.validate_vfc(text)
    if errs:
        raise ValueError(f"模板词表 {vfc_path} 校验失败: {errs[:3]}")
    entries = vfc.parse_vfc(text)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return entries, cfg


def generate(name: str, slots: int, base_dir: Path, force: bool = False,
             activate: bool = False) -> dict:
    """返回生成摘要 dict；失败一律抛 ValueError。"""
    _validate_name(name)
    # 校验 N（spec.slot_tokens 会再校验一次上界，这里给出友好信息）
    if not isinstance(slots, int) or isinstance(slots, bool) or slots < 0:
        raise ValueError(f"--slots 必须是 ≥0 的整数，得到 {slots!r}")
    spec.slot_tokens(slots)

    entries, cfg = _read_base(base_dir)
    base_slots = vfc.infer_slot_count(entries)
    if len(entries) < FIXED_PREFIX + base_slots:
        raise ValueError("模板词表头部不完整")
    # 去掉模板自带的槽位，只保留「固定前缀 + typst + draw/可见区」。
    # typst 段在 slot 之后，天然落在 tail 里，无需单独处理（base1.3.0）。
    tail = entries[FIXED_PREFIX + base_slots:]
    new_entries = [(i, t, v) for i, (_old, t, v)
                   in enumerate(entries[:FIXED_PREFIX])]
    new_entries += [(FIXED_PREFIX + i, f"<s{i}>", False)
                    for i in range(slots)]
    new_entries += [(FIXED_PREFIX + slots + i, t, v)
                    for i, (_old, t, v) in enumerate(tail)]

    # .vfc 往返 + 头部/唯一性/连续性校验
    text = vfc.dumps_vfc(new_entries)
    errs = vfc.validate_vfc(text)
    if errs:
        raise ValueError(f"生成结果 .vfc 校验失败: {errs[:3]}")
    if vfc.parse_vfc(text) != new_entries:
        raise ValueError("生成结果 .vfc 往返不一致")

    layout = spec.head_layout(slots)
    regions = []
    for r in cfg.get("regions") or []:
        r = dict(r)
        if r.get("region") == "slot":
            r["size"] = slots
        regions.append(r)
    if not any(r["region"] == "slot" for r in regions):
        # 模板没有 slot 段（base 就是零预留）→ 插在 system 之后
        regions = [dict(r) for r in cfg.get("regions") or []]
        idx = next((i for i, r in enumerate(regions)
                    if r["region"] == "system"), 2)
        regions.insert(idx + 1, {"region": "slot",
                                 "category": "special", "size": slots})
    if sum(r["size"] for r in regions) != len(new_entries):
        raise ValueError("分区表合计与词表长度不一致（模板 config 可能已损坏）")

    out = dict(cfg)
    out.update({
        "name": name,
        "is_main": False,
        "vocab_size": len(new_entries),
        "head_size": layout["head_size"],
        "slot_count": slots,
        "regions": regions,
        "base_vocab": cfg.get("name") or base_dir.name,
        "base_slot_count": base_slots,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    dest = VOCABS / name
    if dest.exists():
        if not force:
            raise ValueError(f"目标词表已存在: {dest}（要覆盖加 --force）")
        if not dest.is_dir():
            raise ValueError(f"目标不是目录: {dest}")
        for f in dest.iterdir():
            f.unlink()
    else:
        dest.mkdir(parents=True)
    (dest / "vocab.vfc").write_text(text, encoding="utf-8")
    (dest / "config.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 用真正的加载器复核一遍（含 validate_head / regions / Vocab 实例）
    from tokens.vocab import Vocab
    v = Vocab.load(dest / "vocab.vfc")
    if len(v) != len(new_entries) or v.slot_count != slots \
            or v.head_size != layout["head_size"]:
        raise ValueError("生成结果加载校验失败（slot_count / head_size 不符）")

    if activate:
        # 全局词表仍是「base 那份文件」，只是布局换成了 N 个槽位；
        # vocabs/base1.3.0/ 里零预留的 base 副本不受影响。
        gcfg = dict(out)
        gcfg["name"] = spec.BASE_VOCAB_NAME
        gcfg["is_main"] = True
        shutil.copyfile(dest / "vocab.vfc", GLOBAL / "vocab.vfc")
        (GLOBAL / "config.json").write_text(
            json.dumps(gcfg, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    return {
        "name": name, "path": str(dest), "slot_count": slots,
        "base": out["base_vocab"], "base_slot_count": base_slots,
        "head_size": layout["head_size"],
        "draw_begin": layout["draw_begin"],
        "vocab_size": len(new_entries),
        "activated": activate,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="以 base 为基础，按预留槽位数生成一份新词表")
    ap.add_argument("--name", required=True, help="新词表名（vocabs/<name>/）")
    ap.add_argument("--slots", type=int, required=True,
                    help="预留槽位数量 N（0..%d）" % spec.MAX_SLOT_COUNT)
    ap.add_argument("--base", default=None,
                    help="模板词表目录（默认 vocabs/<BASE_VOCAB_NAME>）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的目标词表")
    ap.add_argument("--activate", action="store_true",
                    help="同步为全局词表布局（需重启 server 生效）")
    args = ap.parse_args(argv)
    base_dir = (Path(args.base) if args.base
                else VOCABS / spec.BASE_VOCAB_NAME)
    try:
        r = generate(args.name, args.slots, base_dir,
                     force=args.force, activate=args.activate)
    except (ValueError, OSError) as e:
        print(f"[gen_vocab] 失败: {e}", file=sys.stderr)
        return 1
    print(f"[gen_vocab] 已生成 {r['path']}")
    print(f"  base        : {r['base']}（slot_count={r['base_slot_count']}）")
    print(f"  slot_count  : {r['slot_count']}")
    print(f"  head_size   : {r['head_size']}  draw_begin: {r['draw_begin']}")
    print(f"  vocab_size  : {r['vocab_size']}")
    if r["activated"]:
        print("  已同步为全局词表布局；请重启 server 让 spec 重新读取 config")
    else:
        print("  提示：--activate 可把它设为运行时布局（重启 server 生效）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
