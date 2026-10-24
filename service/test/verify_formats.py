#!/usr/bin/env python3
# ===========================================================================
# GARGANTUA — 两种 token 存档格式的端到端验证脚本
# ---------------------------------------------------------------------------
# 验证内容：
#   A. tok-32（.tokens 二进制）：4 字节/字符的大端 uint32 流
#      - 与同一文本的 UTF-8 字节对比（证明是不同编码，文本编辑器打开
#        是乱码，必须用 tok32 解释器）
#      - 用【不依赖 tokens 包】的裸 struct 读取器解析（证明格式自描述）
#      - hex 文本形式 ↔ 二进制互转
#      - enctok → dectok roundtrip
#   B. .tokenvector（UTF-8 文本）：dim+precision 头 + { } | 结构
#      - f32 / f16 / bf16 三精度 roundtrip（逐位一致）
#      - 时间步结构解析（{} 之间、| 分隔）
#      - 非法输入报错（空时间步 / 坏头部 / 维度不符）
#
# 用法：cd ROOT && python verify_formats.py
# ===========================================================================
# ---- GARGANTUA 路径引导（2026-10-24 目录树重构：server/ service/ start/）----
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
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(boot.project_root())   # 2026-10-24：项目根

from tokens import dectok, enctok, text_codec, tokenvector, tok32  # noqa: E402


def section(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def verify_tok32():
    section("A. tok-32 / .tokens 二进制格式")
    text = "Hello 世界 123 😀"
    ids, skipped = enctok.encode_text(text)
    print(f"输入文本      : {text!r}")
    print(f"未收录字符    : {skipped or '无'}")
    print(f"token 数      : {len(ids)}")
    print(f"十六进制形式  : {tok32.ids_to_hex(ids)}")

    data = tok32.encode_ids(ids)
    utf8 = text.encode("utf-8")
    print(f"\n.tokens 二进制: {len(data)} 字节（4×{len(ids)}）")
    print(f"  前 32 字节  : {data[:32].hex(' ')}")
    print(f"同文本 UTF-8  : {len(utf8)} 字节")
    print(f"  前 32 字节  : {utf8[:32].hex(' ')}")
    assert data != utf8, "tok-32 不应该等于 UTF-8"
    print("  → 两种编码字节不同：tok-32 必须用专用解释器读写 ✓")

    # 独立读取器：不 import tokens 包，裸 struct 解析
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "demo.tokens"
        tok32.write_tokens_file(f, ids)
        raw = f.read_bytes()
        assert len(raw) % 4 == 0
        n = len(raw) // 4
        independent = list(struct.unpack(">%dI" % n, raw))
        print(f"\n独立读取器（裸 struct.unpack('>{n}I')）: {independent}")
        assert independent == ids, "独立读取结果与编码不一致"
        print("  → 格式自描述：任何程序按「大端 uint32 流」都能解析 ✓")

        # hex ↔ 二进制互转
        hex_path = Path(d) / "demo.hex"
        hex_path.write_text(tok32.ids_to_hex(ids), encoding="utf-8")
        back = tok32.hex_to_ids(hex_path.read_text(encoding="utf-8"))
        assert back == ids
        print("hex 文本 ↔ 二进制互转 ✓")

    # 编解码 roundtrip
    out = dectok.decode_ids(tok32.decode_ids(data), "text")
    print(f"\ndectok 还原   : {out!r}")
    assert out == text
    steps = text_codec.encode(text)
    assert text_codec.decode(steps) == text
    print("enctok → .tokens → dectok / text_codec 全链路 roundtrip ✓")


def verify_tokenvector():
    section("B. .tokenvector 向量存档格式（UTF-8）")
    rng = np.random.default_rng(42)
    for precision in ("f32", "f16", "bf16"):
        v1 = rng.normal(size=4).astype(np.float32)
        v2 = rng.normal(size=4).astype(np.float32)
        # 两个时间步：第一步含 2 个 token 向量，第二步含 1 个
        tv = tokenvector.dumps([[v1, v2], [v2]], dim=4,
                               precision=precision)
        dim, prec, steps = tokenvector.loads(tv)
        got = np.concatenate([np.concatenate(s) for s in steps])
        want = tokenvector._cast(np.concatenate([v1, v2, v2]), precision)
        assert (dim, prec) == (4, precision)
        assert got.tobytes() == want.tobytes(), f"{precision} roundtrip 失败"
        print(f"{precision}: 3 个时间步向量 roundtrip 逐位一致 ✓")
        if precision == "f32":
            print("\n文件内容示例（f32，前两行 + 第二个时间步）:")
            lines = tv.splitlines()
            print("  " + lines[0])
            print("  " + lines[1][:70] + " …")
            print("  " + lines[2])
            print("  → 头部 dim/precision；{} 包时间步；| 分步内向量；纯数字 ✓")

    # 错误处理
    print("\n非法输入拒绝:")
    for bad, why in (("dim=4 precision=f32\n{}\n", "空时间步"),
                     ("dim=4\n{1 2 3 4}\n", "头部缺 precision"),
                     ("dim=4 precision=f32\n{1 2 3}\n", "维度不符"),
                     ("dim=4 precision=f64\n{1 2 3 4}\n", "未知精度")):
        try:
            tokenvector.loads(bad)
            raise AssertionError(f"未拒绝: {why}")
        except ValueError:
            print(f"  {why:12s} → ValueError ✓")


def main():
    verify_tok32()
    verify_tokenvector()
    print("\n" + "=" * 62)
    print("两种格式全部验证通过：.tokens（tok-32 二进制）与 "
          ".tokenvector（UTF-8）均可正常读写解析")
    print("=" * 62)


if __name__ == "__main__":
    main()
