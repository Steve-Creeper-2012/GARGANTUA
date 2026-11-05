# ===========================================================================
# GARGANTUA v1 — server/server_term.py：内置终端（项目目录内的交互式 shell）
# ---------------------------------------------------------------------------
# 为什么要它：environment/ 的自动安装偶尔会卡在镜像/证书/编译上，用户需要
# 一个能自己动手的地方（装依赖、跑测试、看日志），而不是每次来改代码。
#
# 边界（steve 2026-11-02 定）：**默认在项目根目录，禁止出去，内部随意**。
#   - 会话启动 cwd = ROOT；
#   - 每条命令送进 shell 之前先过 `_guard_line()`：`cd` 的目标解析成绝对
#     路径后必须落在 ROOT 内，否则整条命令不执行、直接回一段提示；
#   - 这是「防手滑」而不是安全沙箱——`bash -c 'cd /'` 这种子 shell 逃逸
#     拦不住（也不打算拦），拦的是 `cd ..` / `cd /tmp` / `cd ~` 这类误操作。
#
# 传输模型（为什么不上 WebSocket）：
#   本项目 HTTP 层是手写的 stdlib handler，加 WebSocket 要自己做握手 +
#   帧解析。终端是本地 localhost，200ms 轮询的延迟体感可接受，代码量少
#   一个数量级。每次 poll 返回**渲染后的尾部文本**（不是增量字节流），
#   前端直接整体替换 <pre>，天然免疫「漏一个包就永久错位」。
#
# 输出渲染：PTY 里是原始字节流（含 ANSI 转义、\r 回车覆盖、\b 退格）。
# 这里用一个极简终端模型（行数组 + 光标）把 pip 的进度条、apt 的覆盖式
# 输出还原成人能看的文本；颜色一律丢弃（前端是深色主题，保留转义反而乱）。
# ===========================================================================
from __future__ import annotations

import codecs
import fcntl
import os
import pty
import re
import shutil
import signal
import subprocess
import termios
import threading
import time

MAX_LINES = 3000           # 保留的滚动行数（超出从头部丢）
POLL_TAIL_LINES = 800      # 每次 poll 最多回多少行（约 100KB 量级）

_CSI_RE = re.compile(r"\[[0-9;?]*[ -/]*[@-~]")
_CD_RE = re.compile(r"^\s*cd(?:\s+(.*?))?\s*$")
# 密码提示：sudo / ssh / passwd 都会在最后一行留一个「…password…:」。
# 只用来给前端切换输入框类型（type=password），不参与鉴权。
_SECRET_RE = re.compile(
    r"(?:password|passphrase|口令|密码)[^\n]*[:：]\s*$", re.IGNORECASE)


class TermError(ValueError):
    """终端相关错误（路径越界、会话不存在等）。

    继承 ValueError 是**故意的**：server.py 的 POST 分发把 ValueError
    统一映射成 400，这里不用再写一套错误转换。
    """


class _Screen:
    """极简终端屏幕模型：按行保存字符，支持 \\r 覆盖 / \\b 退格 / CSI 清行。"""

    def __init__(self):
        self.lines: list[list[str]] = [[]]
        self.row = 0
        self.col = 0

    def _line(self, row: int) -> list[str]:
        while len(self.lines) <= row:
            self.lines.append([])
        return self.lines[row]

    def feed(self, text: str) -> None:
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch == "\x1b":
                # CSI：ESC [ … final；OSC：ESC ] … BEL/ST（直接跳过）
                if i + 1 < n and text[i + 1] == "[":
                    j = i + 2
                    while j < n and not ("@" <= text[j] <= "~"):
                        j += 1
                    seq = text[i:j + 1]
                    self._csi(seq)
                    i = j + 1
                elif i + 1 < n and text[i + 1] == "]":
                    j = i + 2
                    while j < n and text[j] not in ("\x07",):
                        if text[j] == "\x1b" and j + 1 < n and text[j + 1] == "\\":
                            j += 2
                            break
                        j += 1
                    i = j + 1
                else:
                    i += 2
                continue
            if ch == "\r":
                self.col = 0
            elif ch == "\n":
                self.row += 1
                self.col = 0
                self._line(self.row)
            elif ch == "\b":
                self.col = max(0, self.col - 1)
            elif ch == "\t":
                self.col += 4
            elif ch == "\x07":
                pass
            elif ch >= " " or ch == "\u00a0":
                line = self._line(self.row)
                if self.col > len(line):
                    line.extend([" "] * (self.col - len(line)))
                if self.col < len(line):
                    line[self.col] = ch
                else:
                    line.append(ch)
                self.col += 1
            i += 1
        self._trim()

    def _csi(self, seq: str) -> None:
        body = seq[2:-1]
        final = seq[-1]
        if final == "K":
            line = self._line(self.row)
            if body in ("", "0"):
                del line[self.col:]
            elif body == "2":
                line.clear()
        elif final == "J":
            if body in ("", "0", "2"):
                line = self._line(self.row)
                del line[self.col:]
                del self.lines[self.row + 1:]
        # 其它（颜色、光标移动、清屏等）一律忽略

    def _trim(self) -> None:
        if len(self.lines) > MAX_LINES:
            drop = len(self.lines) - MAX_LINES
            del self.lines[:drop]
            self.row = max(0, self.row - drop)

    def text(self, tail: int = POLL_TAIL_LINES) -> str:
        rows = self.lines[-tail:]
        return "\n".join("".join(r).rstrip() for r in rows)


def _child_session_setup() -> None:
    """子进程里：建新会话，并把 PTY 从设备设成它的**控制终端**。

    2026-11-04 修（sudo / Ctrl-C 全线失效的根因）：旧实现只 setsid()，
    POSIX 下新会话不会自动获得控制终端——`/dev/tty` 打不开，于是：
      · sudo / ssh / passwd 读密码时卡死在「Password:」；
      · 往 master 写 \\x03 没有前台进程组可投递，SIGINT 永远发不出去。
    TIOCSCTTY 之后 /dev/tty 指向这块 PTY，sudo 能正常读密码、终端驱动
    也能把 Ctrl-C 变成前台进程组的 SIGINT。
    """
    os.setsid()
    try:
        # fd 0 此时已经是 slave（Popen 在 preexec_fn 之前完成 dup2）
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    except OSError:
        pass
    try:
        os.tcsetpgrp(0, os.getpgrp())
    except OSError:
        pass


class TermSession:
    """一个 PTY shell 会话（行模式：一次送一整行命令）。"""

    def __init__(self, root: str):
        self.root = os.path.realpath(root)
        self.cwd = self.root
        self.screen = _Screen()
        self._lock = threading.Lock()
        self._master = -1
        self._proc: subprocess.Popen | None = None
        self._alive = False
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._thread: threading.Thread | None = None
        self.started_at = time.time()
        self.last_used = time.time()

    # ---------------- 生命周期 ----------------
    def start(self) -> None:
        if self._alive:
            return
        # 2026-11-04：shell 也走项目内优先（paths.shell_binary），不再直接
        # 继承 SHELL 环境变量或裸查系统 PATH。
        from paths import shell_binary
        shell = shell_binary()
        name = os.path.basename(shell)
        args = [shell]
        if name == "zsh":
            args += ["-f", "-i"]                     # -f：不读 .zshrc，启动快且可预期
            prompt = "%~ %# "
        elif name == "bash":
            args += ["--noprofile", "--norc", "-i"]
            prompt = "\\w \\$ "
        else:
            args += ["-i"]
            prompt = "$ "
        env = dict(os.environ)
        env.update({
            "TERM": "xterm-256color",
            "PS1": prompt,
            "PROMPT": prompt,
            "GARGANTUA_ROOT": self.root,
            "GARGANTUA_TERM": "1",
        })
        env.pop("RUSTUP_HOME", None)                 # 不把服务进程的隔离变量漏给用户
        env.pop("CARGO_HOME", None)
        master, slave = pty.openpty()
        try:
            self._proc = subprocess.Popen(
                args, stdin=slave, stdout=slave, stderr=slave,
                cwd=self.root, env=env, close_fds=True,
                preexec_fn=_child_session_setup)
        finally:
            os.close(slave)
        self._master = master
        self._alive = True
        self._thread = threading.Thread(target=self._pump, name="term-pump",
                                        daemon=True)
        self._thread.start()
        self.screen.feed(
            "GARGANTUA 终端 —— 工作目录固定在本项目内（cd 到外面会被拒绝）\n"
            f"根目录：{self.root}\n\n")

    def close(self) -> None:
        self._alive = False
        try:
            if self._proc and self._proc.poll() is None:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGHUP)
        except Exception:
            pass
        try:
            if self._master >= 0:
                os.close(self._master)
        except Exception:
            pass
        self._master = -1

    @property
    def alive(self) -> bool:
        if self._proc is not None and self._proc.poll() is not None:
            return False
        return self._alive

    # ---------------- 读 ----------------
    def _pump(self) -> None:
        while self._alive and self._master >= 0:
            try:
                data = os.read(self._master, 65536)
            except OSError:
                break
            if not data:
                break
            text = self._decoder.decode(data)
            if text:
                with self._lock:
                    self.screen.feed(text)
        self._alive = False
        with self._lock:
            self.screen.feed("\n[会话已结束]\n")

    # ---------------- 写 ----------------
    def _write(self, data: bytes) -> None:
        if not self.alive:
            raise TermError("终端会话已结束，请重开")
        try:
            os.write(self._master, data)
        except OSError as e:
            raise TermError(f"写入终端失败: {e}") from None

    def _guard_cd(self, arg: str | None) -> str:
        """把 cd 的目标解析成绝对路径并校验在项目内。"""
        if arg is None or arg == "" or arg == "~":
            target = self.root
        else:
            raw = arg.strip().strip("'\"")
            if raw == "~" or raw.startswith("~/"):
                raw = os.path.expanduser(raw)
            target = raw if os.path.isabs(raw) else os.path.join(self.cwd, raw)
        target = os.path.realpath(os.path.normpath(target))
        if target != self.root and not target.startswith(self.root + os.sep):
            raise TermError(
                "只能在项目目录内活动："
                f"{target} 不在 {self.root} 之下")
        if not os.path.isdir(target):
            raise TermError(f"不是目录：{target}")
        return target

    def submit(self, line: str, secret: bool = False) -> None:
        """送一整行（行模式，回车由这里补）。

        secret=True：正在回答 sudo / ssh 的密码提示。密码原样写进 PTY，
        不走 cd 守卫、也不做任何解释——否则密码恰好叫 "cd .." 会被拦下。
        """
        line = (line or "").rstrip("\n")
        self.last_used = time.time()
        if secret:
            self._write((line + "\n").encode())
            return
        m = _CD_RE.match(line)
        if m:
            target = self._guard_cd(m.group(1))
            self.cwd = target
            self._write(("cd -- " + _shquote(target) + "\n").encode())
            return
        if line.strip() in ("exit", "logout"):
            self._write(b"exit\n")
            return
        self._write((line + "\n").encode())

    # ---------------- 中断 ----------------
    def _foreground_pgid(self) -> int | None:
        """PTY 当前的前台进程组；拿不到返回 None。"""
        try:
            pgid = os.tcgetpgrp(self._master)
            return pgid if pgid > 0 else None
        except OSError:
            return None

    def _signal_foreground(self, sig: int) -> bool:
        """给前台进程组发信号（Ctrl-C 的兜底路径）。"""
        pgid = self._foreground_pgid()
        if pgid is None and self._proc is not None:
            try:
                pgid = os.getpgid(self._proc.pid)
            except OSError:
                pgid = None
        if pgid is None:
            return False
        try:
            os.killpg(pgid, sig)
            return True
        except OSError:
            return False

    def interrupt(self, force: bool = False) -> None:
        """Ctrl-C。

        force=False：只往 PTY 写 \\x03，交给终端驱动投递 SIGINT（最接近
        真实键盘，前台程序能自己收尾）。
        force=True：再直接对前台进程组补一发 SIGINT——某些程序把终端设成
        原始模式或屏蔽了 Ctrl-C 时，这条仍然能把它打断。
        """
        self._write(b"\x03")
        if force:
            self._signal_foreground(signal.SIGINT)

    def kill(self) -> None:
        """强制结束前台进程组（SIGKILL）。

        只杀前台作业，不杀 shell 本身；前台恰好是 shell（空闲态）时
        等同于结束整个会话，前端会提示重开。
        """
        if not self._signal_foreground(signal.SIGKILL):
            raise TermError("没有可结束的前台进程")

    def clear(self) -> None:
        with self._lock:
            self.screen = _Screen()

    def snapshot(self) -> str:
        with self._lock:
            return self.screen.text()

    def awaiting_secret(self) -> bool:
        """最后一行是不是密码提示（前端据此把输入框切成 password）。"""
        with self._lock:
            rows = self.screen.lines[-6:]
        for row in reversed(rows):
            line = "".join(row).rstrip()
            if line:
                return bool(_SECRET_RE.search(line))
        return False


def _shquote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


class TermManager:
    """全局单例：只维护**一个**终端会话（再开一个就把旧的关掉）。"""

    def __init__(self, root: str):
        self.root = os.path.realpath(root)
        self._lock = threading.Lock()
        self._session: TermSession | None = None

    def open(self) -> TermSession:
        with self._lock:
            if self._session is not None and self._session.alive:
                self._session.last_used = time.time()
                return self._session
            if self._session is not None:
                self._session.close()
            s = TermSession(self.root)
            s.start()
            self._session = s
            return s

    def get(self) -> TermSession:
        with self._lock:
            s = self._session
        if s is None:
            raise TermError("终端未打开")
        return s

    def close(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session.close()
                self._session = None

    def shutdown(self) -> None:
        self.close()
