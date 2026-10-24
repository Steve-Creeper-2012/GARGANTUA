#!/usr/bin/env python3
# ===========================================================================
# GARGANTUA v1 — window.py：独立桌面窗口（不借道系统浏览器）
# ---------------------------------------------------------------------------
# 为什么不用浏览器打开 127.0.0.1:8712：
#   Chrome/Edge/Firefox 每个标签页都是独立渲染进程 + 一套扩展/GPU 管线，
#   常驻 300~800MB。本项目的内存预算是给训练（模型 + 优化器状态 + KV）
#   的，被浏览器吃掉几百 MB 直接影响 batch/上下文。pywebview 直接调
#   macOS WKWebView / Windows WebView2 / Linux WebKitGTK——系统组件，
#   不另带一个浏览器内核，实测依赖只有 ~18MB 磁盘。
#
# 进程模型（关键）：
#   server.py 是老大（HTTP + 训练/推理调度），本窗口是**子进程**。窗口崩了、被
#   kill 了、内存爆了都不影响训练；窗口关掉也只是少一个 UI，server 照跑
#   （和 Steam 关掉客户端下载还在继续一个语义）。server.py 负责起这个子进程，
#   并对 window.py 做 sha256 完整性校验（不一致弹窗，可跳过）。
#
# 用法（一般不用手敲，run.py 会自动带对参数）：
#   python window.py --port 8712 [--width 1280] [--height 860]
#   python window.py --port 8712 --smoke     只做导入/窗口对象冒烟，不进事件循环
# ===========================================================================
import argparse
import os
import sys
import urllib.request

APP_TITLE = "GARGANTUA"


def wait_for_server(port: int, host: str, timeout: float = 60.0) -> bool:
    """ 等 server 就绪再开窗口：避免窗口先起来看到「无法连接」。 """
    import time
    url = f"http://{host}:{port}/api/status"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def open_window(port: int, host: str, width: int, height: int):
    """ 创建并显示窗口；阻塞到用户关闭。返回进程退出码。 """
    try:
        import webview
    except ImportError:
        print("[window] 缺少 pywebview，无法创建独立窗口。")
        print("[window] 安装窗口依赖（仅 ~18MB，用系统 WebView，不带浏览器内核）：")
        print("  pip install -r start/requirements/window.txt")
        print("[window] 或者继续用浏览器访问，不影响 server：")
        print(f"  http://{host}:{port}")
        return 2

    url = f"http://{host}:{port}/"
    if not wait_for_server(port, host):
        print(f"[window] 等 server 就绪超时，仍尝试打开 {url}")

    win = webview.create_window(APP_TITLE, url, width=width, height=height,
                                min_size=(960, 640), text_select=True)
    # 关窗口 ≠ 关 server：这里只发通知，真正退出由 run.py 决定
    try:
        win.events.closed += lambda: print("[window] 窗口已关闭（server 继续运行）")
    except Exception:
        pass                                    # 老版本 pywebview 无 events
    webview.start()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GARGANTUA 独立窗口")
    ap.add_argument("--port", type=int, default=8712)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=860)
    ap.add_argument("--smoke", action="store_true",
                    help="只验证依赖与窗口对象可创建，不进入事件循环")
    args = ap.parse_args(argv)

    if args.smoke:
        # 无显示环境（CI / SSH）下也能验证：pywebview 可导入、create_window
        # 可用、参数链路通。真正显示需要图形会话。
        try:
            import webview
        except ImportError as e:
            print(f"[window] smoke FAIL: 缺少 pywebview ({e})")
            return 2
        if not (320 <= args.width and 320 <= args.height):
            print("[window] smoke FAIL: 窗口尺寸非法")
            return 2
        win = webview.create_window(APP_TITLE, "about:blank",
                                    width=args.width, height=args.height)
        print(f"[window] smoke OK: pywebview {os.path.basename(sys.executable)}"
              f" 窗口对象 {type(win).__name__} 已创建（未显示）")
        try:
            win.destroy()
        except Exception:
            pass
        return 0

    return open_window(args.port, args.host, args.width, args.height)


if __name__ == "__main__":
    sys.exit(main())
