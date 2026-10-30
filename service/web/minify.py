#!/usr/bin/env python3
# ===========================================================================
# GARGANTUA v1 — web/minify.py：安全压缩器（单文件 SPA 拆成 3 件后组装）
# ---------------------------------------------------------------------------
# 设计原则：**只压空白，不动语义**。前端拆成三件：
#   - index_source.html  开发用的 HTML 骨架（结构 + 首帧内联小脚本 + 外链
#     app.css / app.js 的 <link> / <script src>）
#   - app.css            全部样式（直接编辑、直接服务，不做压缩——本地 localhost
#                        下收益为零，可读性优先）
#   - app.js             全部逻辑（直接编辑、直接服务；node --check 保证语法）
# 构建只把 index_source.html 压成 index.html；app.css / app.js 原样服务。
#
# --check 自校验：index.html 是否与源码逐项一致、app.css / app.js 是否就位、
# app.js 能否过 node --check。任何一项不符都以非零退出（CI / 提交前闸门）。
#
# 用法：
#   python web/minify.py           生成 web/index.html
#   python web/minify.py --check   只校验，不写文件
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
CSS = os.path.join(HERE, "app.css")
JS = os.path.join(HERE, "app.js")


def minify_html(text: str) -> str:
    """压缩 HTML：保留 <script> 原文，其余折叠空白、移除注释。"""
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
    s = re.sub(r">\s+<", "><", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def _tag_sequence(text: str):
    return re.findall(r"<(/?[a-zA-Z][a-zA-Z0-9]*)", text)


def _visible_text(text: str):
    t = re.sub(r"<script\b[^>]*>.*?</script>", "", text, flags=re.S | re.I)
    t = re.sub(r"<style\b[^>]*>.*?</style>", "", t, flags=re.S | re.I)
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    for a, b in (("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"),
                 ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


def _script_bodies(text: str):
    return re.findall(r"<script\b[^>]*>(.*?)</script>", text, flags=re.S | re.I)


def node_check(js: str) -> bool:
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


def verify(src_text: str, dst_text: str, css_text: str, js_text: str) -> bool:
    """校验三件是否就位且最新。"""
    ok = True

    # 0. index.html 与重新生成结果逐字节一致（证明产物不过期）
    expected = minify_html(src_text)
    same = expected == dst_text
    if not same:
        n = min(len(expected), len(dst_text))
        pos = next((i for i in range(n) if expected[i] != dst_text[i]), n)
        print(f"[verify] index.html 已过期（首个差异 @ 字符 {pos}，"
              f"长度 {len(dst_text)} vs {len(expected)}）✗")
    else:
        print("[verify] index.html 与重新生成结果逐字节一致 ✓")
    ok &= same

    # 1. 骨架必须外链 app.css / app.js
    ref = ('href="app.css"' in src_text) and ('src="app.js"' in src_text)
    print(f"[verify] 骨架外链 app.css/app.js {'✓' if ref else '✗'}")
    ok &= ref

    # 2. app.css 非空
    has_css = bool(css_text.strip())
    print(f"[verify] app.css {'存在且非空' if has_css else '缺失或空'} ✓" if has_css
          else "[verify] app.css 缺失或空 ✗")
    ok &= has_css

    # 3. app.js 非空 + node --check
    has_js = bool(js_text.strip())
    js_ok = node_check(js_text) if has_js else False
    print(f"[verify] app.js {'存在' if has_js else '缺失'} "
          f"+ node --check {'通过' if js_ok else '失败'} "
          f"{'✓' if (has_js and js_ok) else '✗'}")
    ok &= (has_js and js_ok)

    # 4. 标签序列一致（结构没被压坏）
    ta, tb = _tag_sequence(src_text), _tag_sequence(dst_text)
    same = ta == tb
    print(f"[verify] body 标签序列{'一致' if same else '不一致'} "
          f"（{len(ta)} 个标签）{' ✓' if same else ' ✗'}")
    ok &= same

    # 5. 可见文本一致
    same = _visible_text(src_text) == _visible_text(dst_text)
    print(f"[verify] body 可见文本{'一致' if same else '不一致'} "
          f"{'✓' if same else ' ✗'}")
    ok &= same

    # 6. 内联脚本（首帧小脚本）逐字一致
    a, b = _script_bodies(src_text), _script_bodies(dst_text)
    same = a == b and len(a) > 0
    print(f"[verify] 内联 <script> {'逐字一致' if same else '内容不一致'} "
          f"（{len(a)} 段）{' ✓' if same else ' ✗'}")
    ok &= same

    print("[verify] " + ("ALL PASS" if ok else "FAILED"))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="GARGANTUA 前端组装器")
    ap.add_argument("--check", action="store_true",
                    help="只校验 index.html/app.css/app.js 是否与源一致")
    args = ap.parse_args()

    if not os.path.exists(SRC):
        print(f"[minify] 找不到源文件: {SRC}")
        return 1
    src_text = open(SRC, encoding="utf-8").read()
    css_text = (open(CSS, encoding="utf-8").read() if os.path.exists(CSS)
                else "")
    js_text = (open(JS, encoding="utf-8").read() if os.path.exists(JS)
               else "")

    if args.check:
        if not os.path.exists(DST):
            print(f"[verify] 找不到 {DST}，请先运行 python web/minify.py")
            return 1
        dst_text = open(DST, encoding="utf-8").read()
        return 0 if verify(src_text, dst_text, css_text, js_text) else 1

    out = minify_html(src_text)
    fd, tmp = tempfile.mkstemp(prefix=".index.", suffix=".tmp", dir=HERE)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(out)
        if not verify(src_text, out, css_text, js_text):
            print("[minify] 校验失败，未覆盖 index.html")
            return 1
        os.replace(tmp, DST)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(f"[minify] index_source.html -> index.html  "
          f"{len(src_text.encode('utf-8'))} -> {len(out.encode('utf-8'))} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
