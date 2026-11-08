#!/usr/bin/env python3
# ===========================================================================
# GARGANTUA — 两种 token 存档格式的端到端验证脚本
# ---------------------------------------------------------------------------
# 验证内容：
#   A. token-5（.tokens 二进制）：27 个单字符 + 4 个原子绑定 token、5 bit/字符的位流
#      - 与同一文本的 UTF-8 字节对比（证明是不同编码）
#      - 用【不依赖 tokens 包】的裸位循环解析（证明格式自描述）
#      - 文本 / 音频 / 视觉三类载荷的定长约束与往返
#      - enctok → token-6 → dectok roundtrip
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
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(boot.project_root())   # 2026-10-24：项目根

import spec  # noqa: E402
from tokens import dectok, enctok, text_codec, tokenvector, tok6  # noqa: E402


def section(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def _independent_decode(data: bytes) -> str:
    """不 import tokens 的裸解码器：每 6 bit 一个字符，码 T6_END_CODE 终止。

    码 29-34 还原成原子边界 token 的展示名（与 spec 第 5b 节一致）。
    """
    chars = spec.T6_CHARS                            # 与 spec 第 5b 节一致
    bound = spec.T6_BOUND_TOKENS
    acc = nbits = 0
    out = []
    for byte in data:
        acc = (acc << 8) | byte
        nbits += 8
        while nbits >= spec.T6_BITS_PER_CHAR:
            nbits -= spec.T6_BITS_PER_CHAR
            code = (acc >> nbits) & 0x3F
            acc &= (1 << nbits) - 1
            if code == spec.T6_END_CODE:
                return "".join(out)
            if code in bound:
                out.append(bound[code])
                continue
            out.append(chars[code])
    raise AssertionError("缺少终止符")


def verify_tok6():
    section("A. token-6 / .tokens 位流格式")
    text = "Hello 世界 123 😀"
    ids, skipped = enctok.encode_text(text)
    items = [tok6.step(tok6.text(i)) for i in ids]
    chars = tok6.items_to_chars(items)
    print(f"输入文本      : {text!r}")
    print(f"未收录字符    : {skipped or '无'}")
    print(f"token 数      : {len(ids)}")
    print(f"字符流        : {chars}")
    print(f"可读形式      : {tok6.to_display(items)}")

    data = tok6.dumps(items)
    utf8 = text.encode("utf-8")
    print(f"\n.tokens 位流 : {len(data)} 字节"
          f"（{len(chars)} 字符 × 6 bit + 终止符 → 补零）")
    print(f"  前 32 字节  : {data[:32].hex(' ')}")
    print(f"同文本 UTF-8  : {len(utf8)} 字节")
    assert data != utf8, "token-6 不应该等于 UTF-8"
    print("  → 两种编码字节不同：token-6 必须用专用解释器读写 ✓")

    # 独立读取器：不 import tokens 包
    independent = _independent_decode(data)
    print(f"\n独立读取器（裸位循环）: {independent[:40]}…")
    assert independent == chars, "独立读取结果与编码不一致"
    print("  → 格式自描述：按「6 bit/字符 + 终止符 %d」即可解析 ✓" % spec.T6_END_CODE)

    # 文件读写
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "demo.tokens"
        tok6.write_file(f, items)
        assert tok6.read_file(f) == items
        print("文件写入 / 读回一致 ✓")

    # 编解码 roundtrip
    back_ids = [e["id"] for it in tok6.loads(data)
                for e in it["elements"]]
    out = dectok.decode_ids(back_ids, "text")
    print(f"\ndectok 还原   : {out!r}")
    assert out == text
    steps = text_codec.encode(text)
    assert text_codec.decode(steps) == text
    print("enctok → token-6 → dectok / text_codec 全链路 roundtrip ✓")

    # 载荷定长约束
    print("\n载荷尺寸:")
    wave = tok6.samples_to_hex([0] * spec.AUDIO_SAMPLES_PER_TOKEN)
    print(f"  音频 {spec.AUDIO_SAMPLES_PER_TOKEN} 采样 × "
          f"{spec.AUDIO_SAMPLE_BITS} bit → {len(wave)} hex 字符 ✓")
    assert len(wave) == spec.T6_WAVE_HEX
    patch = tok6.rgba8_to_hex([0] * spec.PATCH_VEC_DIM)
    print(f"  视觉 {spec.PATCH_PIXELS} 像素 × {spec.T6_PIXEL_BITS} bit "
          f"→ {len(patch)} hex 字符 ✓")
    assert len(patch) == spec.T6_PATCH_HEX
    for bad in ("{t:GG}", "{}", "{t:00001", "t:00001"):
        try:
            tok6.parse_items(bad)
            raise AssertionError(f"未拒绝: {bad}")
        except tok6.Tok6Error:
            pass
    print("  非法结构（4 位 ID / 空步 / 未闭合 / 游离 token）全部拒绝 ✓")


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
    verify_tok6()
    verify_tokenvector()
    print("\n" + "=" * 62)
    print("两种格式全部验证通过：.tokens（token-6 位流）与 "
          ".tokenvector（UTF-8）均可正常读写解析")
    print("=" * 62)


if __name__ == "__main__":
    main()
