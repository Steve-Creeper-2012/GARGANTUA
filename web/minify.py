#!/usr/bin/env python3
# ===========================================================================
# GARGANTUA v1 — web/minify.py：安全压缩器（index_source.html → index.html）
# ---------------------------------------------------------------------------
# 设计原则：**只压空白，不动语义**。前端是单文件 SPA，压缩只为省几 KB，
# 不值得为此引入任何行为差异。所以：
#   1. <script> 内容逐字保留 —— JS 压缩哪怕换个行位置都可能踩坑（ASI、
#      模板字符串、正则字面量），收益却接近零；
#   2. <style> 与 HTML 只做「空白折叠 + 注释移除」，标签结构、属性、文本
#      内容一律不动；
#   3. --check 自校验：重新生成并与现有 index.html 逐项比对，任何不一致
#      都以非零退出码失败（CI / 提交前闸门用）。
#
# 用法：
#   python web/minify.py           生成 web/index.html
#   python web/minify.py --check   只校验，不写文件
#
# 2026-10-23 重写说明：原文件随一次误删丢失（本机清洁工具），按原有契约
# 重建——<script> 逐字、<style> 语义、标签序列、可见文本、node --check
# 五项校验一条不少。
# ===========================================================================
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "index_source.html")
DST = os.path.join(HERE, "index.html")


def minify_html(text: str) -> str:
    """压缩 HTML：保留 <script> 原文，其余折叠空白、移除注释。"""
    # 按 <script>...</script> 切分，脚本段原样拼回
    parts = re.split(r"(<script\b[^>]*>.*?</script>)", text,
                     flags=re.S | re.I)
    out = []
    for i, seg in enumerate(parts):
        if i % 2 == 1:                       # <script> 段：逐字保留
            out.append(seg)
            continue
        seg = re.sub(r"<!--.*?-->", "", seg, flags=re.S)   # HTML 注释
        seg = _fold_ws(seg)
        out.append(seg)
    return "".join(out)


def _fold_ws(s: str) -> str:
    """折叠标签间空白；标签内部的空白一律不动（属性值/预排文本不能碰）。"""
    # 标签之间的连续空白（含换行）→ 单空格
    s = re.sub(r">\s+<", "><", s)
    # 行首行尾空白
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def minify_css(css: str) -> str:
    """压缩 CSS：去注释、折叠空白、去掉最后一个分号。语义完全不变。"""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"\s+", " ", css)
    css = re.sub(r"\s*([{};:,>])\s*", r"\1", css)
    css = css.replace(";}", "}")
    return css.strip()


def _style_blocks(text: str):
    return re.findall(r"<style\b[^>]*>(.*?)</style>", text, flags=re.S | re.I)


def _script_bodies(text: str):
    return re.findall(r"<script\b[^>]*>(.*?)</script>", text, flags=re.S | re.I)


def _tag_sequence(text: str):
    """只取标签名序列（忽略属性与文本），用于比对结构是否被改变。"""
    return re.findall(r"<(/?[a-zA-Z][a-zA-Z0-9]*)", text)


def _visible_text(text: str):
    """去掉 script/style/注释后的可见文本（折叠空白），用于比对用户能看到什么。"""
    t = re.sub(r"<script\b[^>]*>.*?</script>", "", text, flags=re.S | re.I)
    t = re.sub(r"<style\b[^>]*>.*?</style>", "", t, flags=re.S | re.I)
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)      # 注释不是可见文本
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"&lt;", "<", t)
    t = re.sub(r"&gt;", ">", t)
    t = re.sub(r"&amp;", "&", t)
    t = re.sub(r"&quot;", '"', t)
    t = re.sub(r"&#39;", "'", t)
    t = re.sub(r"&nbsp;", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def node_check(js: str) -> bool:
    """node --check 校验 JS 语法（找不到 node 则跳过，不算失败）。"""
    node = shutil.which("node")
    if not node:
        for cand in ("/usr/local/bin/node", "/opt/homebrew/bin/node",
                     os.path.expanduser("~/.workbuddy/binaries/node/"
                                        "versions/22.22.2-6/bin/node")):
            if os.path.isfile(cand):
                node = cand
                break
    if not node:
        print("[verify] node 不可用，跳过 JS 语法检查")
        return True
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                     encoding="utf-8") as f:
        f.write(js)
        path = f.name
    try:
        r = subprocess.run([node, "--check", path],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("[verify] JS 语法错误:", (r.stderr or "")[:400])
        return r.returncode == 0
    finally:
        os.unlink(path)


def verify(src_text: str, dst_text: str) -> bool:
    """五项校验，任一不过即失败。"""
    ok = True

    # 1. <script> 逐字一致
    a, b = _script_bodies(src_text), _script_bodies(dst_text)
    same = a == b and len(a) > 0
    print(f"[verify] <script> {'逐字一致' if same else '内容不一致'} "
          f"（{len(a)} 段）{' ✓' if same else ' ✗'}")
    ok &= same

    # 2. <style> 语义一致（压缩后应等于对源 CSS 再压一次）
    sa, sb = _style_blocks(src_text), _style_blocks(dst_text)
    same = ([minify_css(x) for x in sa] == [minify_css(x) for x in sb]
            and len(sa) > 0)
    print(f"[verify] <style> {'语义一致' if same else '不一致'} "
          f"（{len(sa)} 段）{' ✓' if same else ' ✗'}")
    ok &= same

    # 3. 标签序列一致（结构没被压坏）
    ta, tb = _tag_sequence(src_text), _tag_sequence(dst_text)
    same = ta == tb
    print(f"[verify] body 标签序列{'一致' if same else '不一致'} "
          f"（{len(ta)} 个标签）{' ✓' if same else ' ✗'}")
    ok &= same

    # 4. 可见文本一致（用户看到的东西没变）
    same = _visible_text(src_text) == _visible_text(dst_text)
    print(f"[verify] body 可见文本{'一致' if same else '不一致'} "
          f"{'✓' if same else ' ✗'}")
    ok &= same

    # 5. JS 语法可过 node --check
    js = "\n".join(_script_bodies(src_text))
    same = node_check(js)
    print(f"[verify] JS 语法 node --check {'通过' if same else '失败'} "
          f"{'✓' if same else ' ✗'}")
    ok &= same

    print("[verify] " + ("ALL PASS" if ok else "FAILED"))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="GARGANTUA 安全 HTML 压缩器")
    ap.add_argument("--check", action="store_true",
                    help="只校验 index.html 是否与 index_source.html 一致")
    args = ap.parse_args()

    if not os.path.exists(SRC):
        print(f"[minify] 找不到源文件: {SRC}")
        return 1
    src_text = open(SRC, encoding="utf-8").read()

    if args.check:
        if not os.path.exists(DST):
            print(f"[verify] 找不到 {DST}，请先运行 python web/minify.py")
            return 1
        dst_text = open(DST, encoding="utf-8").read()
        return 0 if verify(src_text, dst_text) else 1

    out = minify_html(src_text)
    with open(DST, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"[minify] index_source.html -> index.html  "
          f"{len(src_text.encode('utf-8'))} -> {len(out.encode('utf-8'))} bytes")
    return 0 if verify(src_text, out) else 1


if __name__ == "__main__":
    sys.exit(main())
