# ===========================================================================
# GARGANTUA v1 — server_services.py：按需服务监督器（省内存的关键）
# ---------------------------------------------------------------------------
# 2026-10-24 重构（steve：所有服务按需启动，省内存）：
#   重活（推理/训练/建模）全部搬到独立子进程，server 本体只做 HTTP 路由 +
#   调度。模型权重（单个 safetensors 可达数 GB）只存在于服务进程里；
#   服务空闲超时自行退出，内存完整归还 OS——PyTorch 的 CPU 分配器在
#   进程内不归还内存（README 实测过），所以「按需」必须是进程级的。
#
# 协议（每个服务子进程 = 一条 JSON-lines 通道，stdin 请求 / stdout 响应）：
#   请求  {"id": "<uuid>", "op": "<op>", ...参数}
#   响应  {"id": "<uuid>", "ok": true,  "result": {...}}
#         {"id": "<uuid>", "ok": false, "error": "..."}
#   服务的 stdout 只准写 JSON 行；人类可读日志一律走 stderr。
#   服务退出条件：收到 {"op":"shutdown"}、stdin EOF（server 死了）、
#   或空闲超过 --idle-timeout。
#
# 已注册的服务（script 相对项目根）：
#   infer  service/infer/infer.py      加载模型 + 文本生成
#   train  service/train/auto_train.py 训练队列守护
# init 是罕见的一次性重活（建模落盘），不走常驻协议，用 run_oneshot。
# ===========================================================================
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid

SERVICE_REGISTRY = {
    "infer": os.path.join("service", "infer", "infer.py"),
    "train": os.path.join("service", "train", "auto_train.py"),
}


class ServiceError(RuntimeError):
    """服务调用失败（起不来 / 协议错误 / 服务内抛错 / 超时）。"""


def _service_env(root: str) -> dict:
    """服务子进程环境：强制隔离（缓存/临时/工具链 HOME 全在项目内）。"""
    # 2026-11-04：子进程不得继承会写系统目录的默认环境
    from paths import isolated_env
    env = isolated_env()
    start_dir = os.path.join(root, "start")
    if start_dir not in sys.path:
        sys.path.insert(0, start_dir)
    try:
        import boot
        env["PYTHONPATH"] = boot.pythonpath()
    except Exception:
        pass
    return env


class _Handle:
    """一个服务子进程的托管句柄（读线程 + 请求/响应配对）。"""

    def __init__(self, name: str, script: str, python_exe: str, root: str,
                 idle_timeout: int):
        self.name = name
        self.script = script
        self.python_exe = python_exe
        self.root = root
        self.idle_timeout = idle_timeout
        self.proc = None
        self.lock = threading.RLock()
        self.pending = {}            # req_id -> Queue（每请求一个）
        self.reader = None
        self.spawned_at = 0.0

    # ------------------------------------------------------------------
    def alive(self) -> bool:
        return (self.proc is not None and self.proc.poll() is None
                and self.reader is not None and self.reader.is_alive())

    def spawn(self) -> None:
        """（重）启服务子进程。调用方持锁。"""
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
        script_path = os.path.join(self.root, self.script)
        if not os.path.isfile(script_path):
            raise ServiceError(f"服务脚本不存在: {script_path}")
        cmd = [self.python_exe, script_path, "--serve",
               "--idle-timeout", str(self.idle_timeout)]
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=None,                 # 人类日志直接进 server 的 stderr
                text=True, encoding="utf-8", bufsize=1,
                env=_service_env(self.root), cwd=self.root)
        except Exception as e:
            raise ServiceError(f"服务 {self.name} 启动失败: {e}")
        self.pending = {}
        self.spawned_at = time.time()
        self.reader = threading.Thread(target=self._read_loop, daemon=True,
                                       name=f"svc-{self.name}")
        self.reader.start()

    def _read_loop(self) -> None:
        """读服务 stdout：JSON 行 → 配对到等待中的请求。EOF = 服务退出。"""
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    # 服务违规往 stdout 写了非 JSON 内容：不致命，记到 stderr
                    print(f"[services] {self.name} 非 JSON 输出: {line[:200]}",
                          file=sys.stderr)
                    continue
                rid = obj.get("id")
                q = self.pending.get(rid)
                if q is not None:
                    q.put(obj)
        except Exception as e:
            print(f"[services] {self.name} 读线程异常: {e}", file=sys.stderr)
        finally:
            # EOF：服务退出——把所有还在等的请求全部判失败
            for q in list(self.pending.values()):
                q.put({"id": None, "ok": False,
                       "error": f"服务 {self.name} 已退出"})

    def send(self, req: dict, timeout: float) -> dict:
        """发一个请求并等待响应（调用方持 h.lock，按服务串行）。

        2026-10-27 审计修复：超时不再只是放弃等待——超时的请求会让
        服务子进程继续执行（推理最长 600s，任务实际还在跑、还在写
        存档，状态与执行分裂）。现在超时即判定服务失控：杀掉子进程
        （迟到的响应随管道关闭被丢弃），由调用方决定是否重启。"""
        rid = uuid.uuid4().hex
        req = dict(req)
        req["id"] = rid
        q = queue.Queue()
        self.pending[rid] = q
        try:
            self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
        except Exception as e:
            self.pending.pop(rid, None)
            raise ServiceError(f"服务 {self.name} 写入失败: {e}")
        try:
            resp = q.get(timeout=timeout)
        except queue.Empty:
            self.pending.pop(rid, None)
            # 隔离失控服务：任务随进程终止，杜绝「接口已报错、后台还在跑」
            try:
                self.proc.kill()
            except Exception:
                pass
            raise ServiceError(
                f"服务 {self.name} 超时（{timeout:.0f}s）：{req.get('op')}"
                f"（服务已被终止）")
        finally:
            self.pending.pop(rid, None)
        if not resp.get("ok"):
            raise ServiceError(str(resp.get("error") or "未知服务错误"))
        return resp.get("result") or {}

    def send_stream(self, req: dict, timeout: float):
        """发一个流式请求：逐个 yield 服务发回的帧，直到 final 帧。

        与 send 的区别：pending 条目要保留到流结束（同一个 id 会连续来多
        帧），partial 帧原样交给调用方。超时与 send 一样判定服务失控——
        杀掉子进程，避免「接口报错、后台还在算」。

        2026-11-04 新增（推理流式显示）：协议本身没变，只是允许服务在
        final 结果之前先发若干 `{"partial": {...}}` 帧。
        """
        rid = uuid.uuid4().hex
        req = dict(req)
        req["id"] = rid
        q = queue.Queue()
        self.pending[rid] = q
        try:
            self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
        except Exception as e:
            self.pending.pop(rid, None)
            raise ServiceError(f"服务 {self.name} 写入失败: {e}")
        try:
            while True:
                try:
                    obj = q.get(timeout=timeout)
                except queue.Empty:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                    raise ServiceError(
                        f"服务 {self.name} 流式超时（{timeout:.0f}s）："
                        f"{req.get('op')}（服务已被终止）")
                if not obj.get("ok"):
                    raise ServiceError(str(obj.get("error") or "未知服务错误"))
                yield obj
                if obj.get("final"):
                    return
        finally:
            self.pending.pop(rid, None)

    def abort(self) -> None:
        """终止子进程（超时隔离用）。调用方持 h.lock。"""
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass


class ServiceSupervisor:
    """按需拉起/复用/收殓服务子进程。"""

    def __init__(self, root: str, python_exe: str = None,
                 idle_timeout: int = 300):
        self.root = root
        self.python_exe = python_exe or sys.executable
        self.idle_timeout = idle_timeout
        self._handles = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    def _handle(self, name: str) -> _Handle:
        rel = SERVICE_REGISTRY.get(name)
        if rel is None:
            raise ServiceError(f"未知服务: {name}（可选 {sorted(SERVICE_REGISTRY)}）")
        h = self._handles.get(name)
        if h is None:
            h = _Handle(name, rel, self.python_exe, self.root,
                        self.idle_timeout)
            self._handles[name] = h
        return h

    def running(self, name: str) -> bool:
        with self._lock:
            h = self._handles.get(name)
            return h is not None and h.alive()

    def kill_service(self, name: str) -> bool:
        """杀掉并注销某个常驻服务（不动其它服务）。

        用于「重置 infer 服务」：杀掉服务进程 → 下次推理重新加载模型。
        当前实现每次 generate 从零开始（state/kv 都是本次局部量），跨轮
        持久状态尚未接线，所以杀进程并不等于「清空记忆 / 重置 fast weight」。
        返回是否真有进程被终止。"""
        with self._lock:
            h = self._handles.get(name)
            if h is None:
                return False
            try:
                h.abort()
            finally:
                self._handles.pop(name, None)
            return True

    # ------------------------------------------------------------------
    def call(self, name: str, op: str, timeout: float = 120.0, **args) -> dict:
        """确保服务在跑 → 发请求 → 等响应。服务死亡时重启并重试一次。

        2026-10-27 审计修复：全局锁（self._lock）只保护句柄注册表，
        不再包住阻塞式 send——旧实现下 600s 的推理期间，训练停止、
        状态查询等全部拿不到锁。每个服务句柄有自己的锁，请求按服务
        串行、跨服务并行。"""
        with self._lock:
            h = self._handle(name)
        with h.lock:
            for attempt in (1, 2):
                if not h.alive():
                    # kill_service 可能已把该句柄从注册表摘掉：此时绝不能在旧
                    # 局部 h 上重新 spawn，否则造出 shutdown_all 收不到的孤儿进程
                    if attempt == 2:
                        with self._lock:
                            if self._handles.get(name) is not h:
                                raise ServiceError(
                                    f"服务 {name} 已被注销，取消重启")
                    h.spawn()
                try:
                    return h.send({"op": op, **args}, timeout)
                except ServiceError:
                    if attempt == 1 and not h.alive():
                        # 服务在请求期间死了（或超时被隔离）：已/将重启，重试一次
                        continue
                    raise
            raise ServiceError(f"服务 {name} 重试后仍失败")

    def stream(self, name: str, op: str, timeout: float = 600.0, **args):
        """流式调用：yield 服务的每一帧（partial… final）。

        句柄锁在整个迭代期间持有——推理本来就是按服务串行的，不能两条
        流同时进同一个模型。调用方必须把生成器耗尽或显式 close()，
        否则锁要等 GC 才释放（server 侧用 try/finally 包住）。
        """
        with self._lock:
            h = self._handle(name)
        h.lock.acquire()
        try:
            if not h.alive():
                h.spawn()
            yield from h.send_stream({"op": op, **args}, timeout)
        finally:
            h.lock.release()

    def query(self, name: str, op: str, timeout: float = 15.0, **args):
        """状态查询：服务没在跑就返回 None（不为查询而起进程）。

        句柄锁用带超时的 acquire：整个生成期 stream()/call() 持有 h.lock，
        拿不到锁就立刻返回 None（状态查询永远不阻塞 HTTP 线程），锁等待
        上限取 min(timeout, 2)。"""
        with self._lock:
            h = self._handles.get(name)
        if h is None or not h.alive():
            return None
        # 拿不到锁直接放弃，不阻塞：生成期锁被推理流持有，stats 走本地缓冲
        if not h.lock.acquire(timeout=min(timeout, 2)):
            return None
        try:
            if not h.alive():
                return None
            try:
                return h.send({"op": op, **args}, timeout)
            except ServiceError:
                return None
        finally:
            h.lock.release()

    # ------------------------------------------------------------------
    def run_oneshot(self, script: str, argv: list, timeout: float = 600.0) -> dict:
        """一次性子进程（init 建模）：跑完即退，stdout 末行 = JSON 结果。"""
        script_path = os.path.join(self.root, script)
        if not os.path.isfile(script_path):
            raise ServiceError(f"脚本不存在: {script_path}")
        cmd = [self.python_exe, script_path, *argv, "--json"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout, cwd=self.root,
                               env=_service_env(self.root))
        except subprocess.TimeoutExpired:
            raise ServiceError(f"{script} 执行超时（{timeout:.0f}s）")
        except Exception as e:
            raise ServiceError(f"{script} 执行失败: {e}")
        last = ""
        for line in (r.stdout or "").splitlines():
            if line.strip():
                last = line.strip()
        try:
            obj = json.loads(last)
        except Exception:
            raise ServiceError(
                f"{script} 未输出 JSON 结果（exit={r.returncode}）："
                f"{(r.stderr or r.stdout or '')[-400:]}")
        if not obj.get("ok"):
            raise ServiceError(str(obj.get("error") or "未知错误"))
        return obj.get("result") or {}

    # ------------------------------------------------------------------
    def shutdown_all(self, grace: float = 30.0) -> None:
        """server 退出时收尸：shutdown 协议 → 等自然退出 → SIGTERM → SIGKILL。

        2026-10-27 审计修复：旧实现收到 shutdown 应答后立即 terminate()，
        而训练服务的应答先于「停止并 join 训练线程」——关闭恰逢训练 step
        时，落盘前清理被 SIGTERM 中断。现在应答后先等待服务自然退出
        （训练服务在应答前已 stop+join，见 auto_train.serve），直至
        grace 到期才 escalate 到 terminate/kill。"""
        with self._lock:
            handles = list(self._handles.values())
        for h in handles:
            if h.alive():
                try:
                    with h.lock:
                        if h.alive():
                            h.send({"op": "shutdown"}, timeout=5.0)
                except Exception:
                    pass
        # 等自然退出（训练收尾/存档落盘需要时间）
        deadline = time.time() + grace
        for h in handles:
            if h.proc is None:
                continue
            while time.time() < deadline and h.proc.poll() is None:
                time.sleep(0.1)
        # grace 内没退的才 escalate
        for h in handles:
            if h.proc is not None and h.proc.poll() is None:
                try:
                    h.proc.terminate()
                except Exception:
                    pass
        deadline = time.time() + min(grace, 10.0)
        for h in handles:
            if h.proc is None:
                continue
            while time.time() < deadline and h.proc.poll() is None:
                time.sleep(0.1)
            if h.proc.poll() is None:
                try:
                    h.proc.kill()
                except Exception:
                    pass
