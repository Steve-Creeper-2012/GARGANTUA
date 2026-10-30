#!/usr/bin/env python3
# ===========================================================================
# GARGANTUA v1 — service/test/run_tests.py：全量回归（逐套件独立进程）
# ---------------------------------------------------------------------------
# 为什么一个套件一个进程（README「Testing」记录过）：
#   全尺寸 Gargantua() ≈ 527.9M 参数（fp32 ≈ 2.1GB），PyTorch 的 CPU
#   分配器把释放的块留在自己的池里不还 OS——同进程跑多个套件，RSS 只涨
#   不跌，`unittest discover` 会在 test_model 中途被 SIGKILL（exit 137）。
#
# 流程：
#   1. 静态门：py_compile 全部 .py / bash -n 全部 .sh / minify --check；
#   2. 逐套件独立子进程跑 unittest，超时长的套件给大超时；
#   3. 汇总表 + 总耗时；任一失败 → 退出码 1。
#
# 用法：
#   python service/test/run_tests.py            # 全量
#   python service/test/run_tests.py --list     # 只看套件清单
#   python service/test/run_tests.py --suite test_model --suite test_services
#   python service/test/run_tests.py --no-static # 跳过静态门
# ===========================================================================
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# ---- GARGANTUA 路径引导 ----------------
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

ROOT = boot.project_root()

# 套件 → (显示名, 目标, 超时秒, 类型)。重量级（建真模型/起子进程）的给大超时。
# 类型：unittest = `python -m unittest <模块>`；script = 独立可执行校验脚本。
SUITES = [
    ("test_model", "service.test.test_model", 1800, "unittest"),
    ("test_codecs", "service.test.test_codecs", 900, "unittest"),
    ("test_tokens", "service.test.test_tokens", 900, "unittest"),
    ("test_tok5", "service.test.test_tok5", 300, "unittest"),
    ("test_multimodal", "service.test.test_multimodal", 900, "unittest"),
    ("test_precision", "service.test.test_precision", 600, "unittest"),
    ("test_vocab_presets", "service.test.test_vocab_presets", 600, "unittest"),
    ("test_vocab_api", "service.test.test_vocab_api", 600, "unittest"),
    ("test_image2prim", "service.test.test_image2prim", 900, "unittest"),
    ("test_fixes", "service.test.test_fixes", 1800, "unittest"),
    ("test_audit_20261023", "service.test.test_audit_20261023", 1800, "unittest"),
    ("test_audit_20261027", "service.test.test_audit_20261027", 1800, "unittest"),
    ("test_services", "service.test.test_services", 600, "unittest"),
    ("test_server", "service.test.test_server", 2400, "unittest"),
    # 2026-10-29 新增：进程监视器 / CPU 拓扑 / 越权白名单（35 项，秒级）
    ("test_monitor_ui", "service.test.test_monitor_ui", 600, "unittest"),
    # 2026-10-29 新增：词表预设名路径穿越（任意 vocab.txt 写）防护（16 项）
    ("test_vocab_preset_path", "service.test.test_vocab_preset_path", 600,
     "unittest"),
    ("verify_formats", "service/test/verify_formats.py", 600, "script"),
]


def static_gates() -> list:
    """返回失败信息列表（空 = 全过）。"""
    failures = []

    # 1) py_compile：全部项目 .py（server/ service/ start/）
    py_files = []
    for d in ("server", "service", "start"):
        base = os.path.join(ROOT, d)
        for dirpath, _dirs, files in os.walk(base):
            if "__pycache__" in dirpath or "/." in dirpath:
                continue
            py_files += [os.path.join(dirpath, f) for f in files
                         if f.endswith(".py")]
    r = subprocess.run([sys.executable, "-m", "py_compile", *py_files],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        failures.append(f"py_compile 失败:\n{r.stdout}\n{r.stderr}")

    # 2) bash -n：全部 .sh
    sh_files = []
    for dirpath, _dirs, files in os.walk(ROOT):
        if "/." in dirpath or "__pycache__" in dirpath:
            continue
        sh_files += [os.path.join(dirpath, f) for f in files
                     if f.endswith(".sh")]
    for sh in sh_files:
        r = subprocess.run(["bash", "-n", sh], capture_output=True, text=True)
        if r.returncode != 0:
            failures.append(f"bash -n {sh} 失败:\n{r.stderr}")

    # 3) 前端 minify 自校验（index.html 与源码不得漂移）
    minify = os.path.join(ROOT, "service", "web", "minify.py")
    if os.path.isfile(minify):
        r = subprocess.run([sys.executable, minify, "--check"],
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            failures.append(f"minify --check 失败:\n{r.stdout}\n{r.stderr}")
    return failures


def run_suite(name: str, target: str, timeout: int, kind: str = "unittest") -> dict:
    t0 = time.time()
    if kind == "script":
        cmd = [sys.executable, os.path.join(ROOT, target)]
    else:
        cmd = [sys.executable, "-m", "unittest", "-v", target]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, cwd=ROOT)
        ok = r.returncode == 0
        out = (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        ok = False
        out = f"超时（{timeout}s）——套件被强制结束"
    return {"name": name, "ok": ok, "sec": time.time() - t0,
            "output": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GARGANTUA 全量回归")
    ap.add_argument("--suite", action="append", default=None,
                    help="只跑指定套件（可多次）")
    ap.add_argument("--list", action="store_true", help="列出套件后退出")
    ap.add_argument("--no-static", action="store_true",
                    help="跳过静态门（py_compile/bash -n/minify）")
    args = ap.parse_args(argv)

    if args.list:
        for name, target, timeout, kind in SUITES:
            print(f"{name:22s} {target:42s} {kind:9s} timeout={timeout}s")
        return 0

    suites = SUITES
    if args.suite:
        wanted = set(args.suite)
        suites = [s for s in SUITES if s[0] in wanted]
        if not suites:
            print(f"没有匹配的套件: {sorted(wanted)}")
            return 2

    print("=" * 64)
    print("GARGANTUA 全量回归（逐套件独立进程）")
    print("=" * 64)

    if not args.no_static:
        print("[static] py_compile / bash -n / minify --check ...")
        failures = static_gates()
        if failures:
            for f in failures:
                print(f"[static] ✗ {f}")
            return 1
        print("[static] ✓ 全过")

    results = []
    for name, target, timeout, kind in suites:
        print(f"\n[run] {name}（{target}，{kind}，timeout={timeout}s）...")
        res = run_suite(name, target, timeout, kind)
        results.append(res)
        mark = "✓" if res["ok"] else "✗"
        print(f"[run] {mark} {name}  {res['sec']:.1f}s")
        if not res["ok"]:
            # 失败套件打印尾部输出（头部通常是进度点）
            tail = "\n".join(res["output"].splitlines()[-40:])
            print("-" * 64)
            print(tail)
            print("-" * 64)

    print("\n" + "=" * 64)
    n_ok = sum(1 for r in results if r["ok"])
    for r in results:
        mark = "✓" if r["ok"] else "✗"
        print(f"  {mark} {r['name']:22s} {r['sec']:7.1f}s")
    total = sum(r["sec"] for r in results)
    print(f"\n合计: {n_ok}/{len(results)} 套件通过，总耗时 {total:.1f}s")
    print("=" * 64)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
