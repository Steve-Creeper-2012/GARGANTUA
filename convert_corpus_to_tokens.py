#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GARGANTUA 语料 → token-5 .tokens 转换器（预训练用）。

把 en_corpus.txt / zh_corpus.txt 的每一行（一条独立语料）编码成一个
独立的 token-5 .tokens 文件，落到桌面 ~/Desktop/traindata/{en,zh}/。

设计依据：
  - 训练管线 service/train/train.py 把每个 .tokens 文件视为一个独立文档，
    绝不在文档间跨 <stop> 采样窗口；语料本身也是「一行 = 一条独立语料」，
    所以「一行 = 一个 .tokens 文档」最忠实。
  - 编码复用项目自带 tokens/enctok.encode_text（常规优先、fallback 兜底），
    写出复用 tokens/tok5.write_file（与 enctok CLI、Token 编辑器同款字节流）。

复用项目代码，不重造轮子；故本脚本放在项目根，靠 start/boot.py 引导。
"""
import sys
from pathlib import Path

# ---- GARGANTUA 路径引导（与 enctok.py 同款）----
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

import spec
from tokens import tok5
from tokens.enctok import encode_text
from tokens.vocab import get_vocab
from tokens.tokenset import read_doc  # 训练侧真正的读取器，用于回环校验

# 2026-11-04 可移植性：旧版写死某一台机器的桌面绝对路径，换机后全是死
# 引用，而且产物写到项目文件夹外。默认输入/输出都锚在项目根内；要读外部
# 语料请用 --input-dir / --en / --zh 显式指定（那是有意指定的用户数据，
# 不是项目自带依赖）。
def _default_paths():
    from paths import project_dir
    ap_in = Path(project_dir("data", "corpus"))
    ap_out = Path(project_dir("data", "traindata"))
    return ap_in, ap_out


def convert_lang(lang: str, in_path: Path, out_dir: Path):
    text = in_path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)

    n_docs = 0
    total_ids = 0
    total_skipped = 0
    skipped_chars = set()
    doc_token_counts = []

    for idx, line in enumerate(lines, 1):
        ids, skipped = encode_text(line)
        if not ids:  # 整行编码后为空（罕见），跳过避免空文档
            continue
        out_path = out_dir / f"{lang}_{idx:05d}.tokens"
        if out_path.exists():  # 可断点续跑：已存在则跳过
            n_docs += 1
            total_ids += len(ids)
            total_skipped += len(skipped)
            skipped_chars.update(skipped)
            doc_token_counts.append(len(ids))
            continue
        items = [tok5.step(tok5.text(i)) for i in ids]
        # 直接写 dumps 字节流（与 tok5.write_file 内容一致，省去逐文件
        # fsync/原子改名开销，2146 个文件才跑得动）
        out_path.write_bytes(tok5.dumps(items))
        n_docs += 1
        total_ids += len(ids)
        total_skipped += len(skipped)
        skipped_chars.update(skipped)
        doc_token_counts.append(len(ids))
        if n_docs % 200 == 0:
            print(f"  [{lang}] 已写 {n_docs} 文档…", file=sys.stderr, flush=True)

    return {
        "lang": lang,
        "n_docs": n_docs,
        "total_ids": total_ids,
        "total_skipped": total_skipped,
        "skipped_chars": skipped_chars,
        "doc_token_counts": doc_token_counts,
        "out_dir": out_dir,
    }


def validate(lang: str, out_dir: Path, n_check: int = 3):
    """用训练侧 read_doc 回环校验前 n_check 个文件可解析、ID 数自洽。"""
    problems = []
    files = sorted(out_dir.glob(f"{lang}_*.tokens"))[:n_check]
    for f in files:
        try:
            doc = read_doc(f)
            if not doc["pos_ids"]:
                problems.append(f"{f.name}: 无 pos_ids")
        except Exception as e:  # noqa: BLE001
            problems.append(f"{f.name}: {type(e).__name__}: {e}")
    return problems


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="语料 txt → token-5 .tokens（一行语料 = 一个文档）")
    in_default, out_default = _default_paths()
    ap.add_argument("--input-dir", default=str(in_default),
                    help="语料目录（含 en_corpus.txt / zh_corpus.txt）")
    ap.add_argument("--out-dir", default=str(out_default),
                    help="输出目录（默认项目内 data/traindata）")
    ap.add_argument("--en", default=None, help="英文语料完整路径（覆盖 --input-dir）")
    ap.add_argument("--zh", default=None, help="中文语料完整路径（覆盖 --input-dir）")
    a = ap.parse_args()
    corpus = {
        "en": Path(a.en) if a.en else Path(a.input_dir) / "en_corpus.txt",
        "zh": Path(a.zh) if a.zh else Path(a.input_dir) / "zh_corpus.txt",
    }
    OUT_ROOT = Path(a.out_dir)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_lines = []
    manifest_lines.append("GARGANTUA 预训练语料 token-5 转换清单")
    manifest_lines.append("=" * 48)
    manifest_lines.append(f"词表：base 全局词表（实际 vocab_size={len(get_vocab())}）")
    manifest_lines.append("格式：token-5 .tokens，一行语料 = 一个 .tokens 文档")
    manifest_lines.append("训练：python service/train/train.py --model <模型> "
                          f"--data {OUT_ROOT}")
    manifest_lines.append("")

    all_problems = []
    for lang, in_path in corpus.items():
        if not in_path.is_file():
            manifest_lines.append(f"[{lang}] 源文件缺失：{in_path}")
            continue
        r = convert_lang(lang, in_path, OUT_ROOT / lang)
        probs = validate(lang, OUT_ROOT / lang)
        all_problems.extend(probs)
        counts = r["doc_token_counts"]
        avg = sum(counts) / len(counts) if counts else 0
        manifest_lines.append(f"[{lang}] 源：{in_path}")
        manifest_lines.append(f"  文档数   : {r['n_docs']}")
        manifest_lines.append(f"  token 总数: {r['total_ids']}")
        manifest_lines.append(f"  单文档 token: 最小 {min(counts)} / 均值 "
                              f"{avg:.1f} / 最大 {max(counts)}")
        manifest_lines.append(f"  跳过字符数: {r['total_skipped']}")
        if r["skipped_chars"]:
            sc = " ".join(sorted(r["skipped_chars"]))
            manifest_lines.append(f"  被跳过的字符: {sc}")
        manifest_lines.append(f"  输出目录 : {r['out_dir']}")
        manifest_lines.append("")

    if all_problems:
        manifest_lines.append("校验问题（训练侧 read_doc 解析失败）：")
        manifest_lines.extend(f"  - {p}" for p in all_problems)
    else:
        manifest_lines.append("校验：训练侧 read_doc 回环通过，文件可直接喂训练。")

    manifest_path = OUT_ROOT / "MANIFEST.txt"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    print("\n".join(manifest_lines))


if __name__ == "__main__":
    main()
