# ===========================================================================
# GARGANTUA v1 — server/server_prims.py：图元绘画的「任务列队」
# ---------------------------------------------------------------------------
# 为什么要有它：原来 /api/codec/image2prims 是「一次请求跑一张图」，跑的时候
# 界面只能干等，关掉页面就白跑，结果也不落盘。现在改成**任务制**：
#
#   建任务（图片 + 参数） → 入队 → 后台线程顺序执行 → 结果落到
#   service/prims/output/jobs/<id>/ 下，随时可以回看/对比/取 token。
#
# 目录契约（「每个 service 一个产物文件夹」的第一例）：
#   service/prims/output/
#     jobs/<id>/job.json      任务元信息（状态/参数/耗时/错误）
#     jobs/<id>/source.png    原始图片（渲染页的「原图」半边用它）
#     jobs/<id>/result.json   {canvas, prims, tokens, note}
#     jobs/<id>/log.txt       运行日志（界面右下角那个面板）
#
# 实时渲染：执行时把每个采纳的图元追加进内存里的 `live` 列表，前端按
# `since=<已拿到的条数>` 增量拉取——比整包轮询省得多，也让「边算边画」
# 天然成立（GPU 路径的图元来自转换器的 stream 文件，见 image_to_primitives）。
# ===========================================================================
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

import spec

# 产物根目录（每个 service 一个 output/，本服务的是图元任务）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "service", "prims", "output")
JOBS_DIR = os.path.join(OUTPUT_DIR, "jobs")

MAX_STEPS = 20000          # 与转换器的 SHAPE_RENDERER_MAX_ITERATIONS 上限一致
LIVE_LIMIT = 200000        # 单个任务最多缓存的实时图元数（防内存失控）


def _now() -> float:
    return time.time()


def _prim_dict(p) -> Dict[str, float]:
    return {"cx": float(p.cx), "cy": float(p.cy), "w": float(p.w),
            "h": float(p.h), "shape_type": int(p.shape_type),
            "theta": float(p.theta), "r": int(p.r), "g": int(p.g),
            "b": int(p.b)}


class PrimsJob:
    """一个转换任务（内存态 + 落盘态）。"""

    def __init__(self, job_id: str, name: str, params: Dict[str, Any]):
        self.id = job_id
        self.name = name
        self.params = params            # engine/quality/steps/early_stop/canvas
        self.status = "queued"          # queued|running|done|error|canceled
        self.created = _now()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.error: Optional[str] = None
        self.note: Optional[str] = None
        self.canvas: Optional[List[int]] = None
        self.progress_step = 0
        self.progress_total = int(params.get("steps") or 0)
        self.live: List[Dict[str, float]] = []
        self.result: Optional[Dict[str, Any]] = None
        self.log_lines: List[str] = []
        self.sink = None                # 由 PrimsStore 注入（写全局日志）
        self._lock = threading.Lock()

    # ---------- 目录 ----------
    @property
    def dir(self) -> str:
        return os.path.join(JOBS_DIR, self.id)

    @property
    def source_path(self) -> str:
        return os.path.join(self.dir, "source.png")

    @property
    def result_path(self) -> str:
        return os.path.join(self.dir, "result.json")

    @property
    def job_path(self) -> str:
        return os.path.join(self.dir, "job.json")

    def ensure_dir(self) -> None:
        os.makedirs(self.dir, exist_ok=True)

    # ---------- 日志 ----------
    def log(self, msg: str) -> None:
        line = time.strftime("%H:%M:%S") + "  " + msg
        with self._lock:
            self.log_lines.append(line)
            if len(self.log_lines) > 500:
                del self.log_lines[:100]
        try:
            with open(os.path.join(self.dir, "log.txt"), "a",
                      encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
        # 全局日志（所有任务汇总），供「日志」面板的「全局」页签用
        if self.sink is not None:
            try:
                self.sink(self.name, line)
            except Exception:
                pass

    def add_primitive(self, p, index: int) -> None:
        with self._lock:
            if len(self.live) < LIVE_LIMIT:
                self.live.append(_prim_dict(p))
            self.progress_step = max(self.progress_step, int(index))

    # ---------- 序列化 ----------
    def meta(self) -> Dict[str, Any]:
        elapsed = None
        if self.started:
            elapsed = (self.finished or _now()) - self.started
        return {
            "id": self.id, "name": self.name, "status": self.status,
            "params": self.params, "created": self.created,
            "started": self.started, "finished": self.finished,
            "elapsed": elapsed, "error": self.error, "note": self.note,
            "canvas": self.canvas,
            "progress": {"step": self.progress_step,
                         "total": self.progress_total},
            "n_live": len(self.live),
            "n_primitives": (self.result or {}).get("n_primitives"),
        }

    def save_meta(self) -> None:
        try:
            self.ensure_dir()
            with open(self.job_path, "w", encoding="utf-8") as f:
                json.dump(self.meta(), f, ensure_ascii=False, indent=1)
        except OSError:
            pass

    def load_meta(self, data: Dict[str, Any]) -> None:
        self.status = data.get("status") or "queued"
        self.created = data.get("created") or _now()
        self.started = data.get("started")
        self.finished = data.get("finished")
        self.error = data.get("error")
        self.note = data.get("note")
        self.canvas = data.get("canvas")
        prog = data.get("progress") or {}
        self.progress_step = int(prog.get("step") or 0)
        self.progress_total = int(prog.get("total") or 0)


class PrimsRunner:
    """单线程顺序执行队列（GPU 转换器本来就是独占的，并发没有意义）。"""

    def __init__(self, store: "PrimsStore"):
        self.store = store
        self._queue: List[str] = []
        self._cv = threading.Condition()
        self._thread = threading.Thread(target=self._loop, name="prims-runner",
                                        daemon=True)
        self._thread.start()

    def enqueue(self, job_id: str) -> None:
        with self._cv:
            if job_id not in self._queue:
                self._queue.append(job_id)
            self._cv.notify()

    def _next(self) -> Optional[str]:
        with self._cv:
            while not self._queue:
                self._cv.wait(timeout=1.0)
                if not self._queue:
                    return None
            return self._queue.pop(0)

    def _loop(self) -> None:
        while True:
            job_id = self._next()
            if job_id is None:
                continue
            job = self.store.get(job_id)
            if job is None or job.status not in ("queued",):
                continue
            try:
                self._run(job)
            except Exception as e:                  # 兜底：别让线程死掉
                job.status = "error"
                job.error = f"{e}"
                job.finished = _now()
                job.log("✗ 未捕获异常：" + traceback.format_exc(limit=3))
                job.save_meta()

    def _run(self, job: PrimsJob) -> None:
        from PIL import Image
        from prims.image_to_primitives import image_to_primitives
        from tokens import draw_codec
        from tokens.vocab import get_vocab

        job.status = "running"
        job.started = _now()
        job.log(f"开始：{job.name}")
        job.save_meta()

        img = Image.open(job.source_path).convert("RGB")
        params = job.params
        canvas_req = params.get("canvas")
        if canvas_req == "auto" or not canvas_req:
            canvas = list(spec.draw_canvas_auto(*img.size))
        else:
            canvas = [int(canvas_req[0]), int(canvas_req[1])]
        job.canvas = canvas
        job.progress_total = int(params.get("steps") or 100)
        job.log(f"引擎 {params.get('engine')} · 画布 {canvas[0]}×{canvas[1]} · "
                f"步数 {job.progress_total} · 早停 "
                f"{'开' if params.get('early_stop') else '关'}")

        last_log = 0
        def on_prim(p, index):
            nonlocal last_log
            job.add_primitive(p, index)
            if index and index - last_log >= max(50, job.progress_total // 20):
                last_log = index
                job.log(f"已采纳 {index} 个图元")
                job.save_meta()

        prims, note = image_to_primitives(
            img, max_steps=job.progress_total,
            quality=params.get("quality") or "fast",
            backend=params.get("engine") or "auto",
            canvas=tuple(canvas),
            early_stop=bool(params.get("early_stop", True)),
            on_primitive=on_prim)

        job.log("拟合完成：" + note)
        vocab = get_vocab()
        tokens = draw_codec.primitives_to_tokens(prims, tuple(canvas), vocab)
        job.result = {
            "canvas": canvas,
            "prims": [_prim_dict(p) for p in prims],
            "tokens": tokens,
            "note": note,
            "n_primitives": len(prims),
        }
        try:
            with open(job.result_path, "w", encoding="utf-8") as f:
                json.dump(job.result, f, ensure_ascii=False)
        except OSError as e:
            job.log(f"⚠ 结果落盘失败：{e}")

        job.status = "done"
        job.note = note
        job.finished = _now()
        job.progress_step = len(prims)
        job.log(f"✓ 完成，共 {len(prims)} 个图元，"
                f"耗时 {job.finished - job.started:.1f}s")
        job.save_meta()


class PrimsStore:
    """任务的增删查 + 磁盘恢复。"""

    def __init__(self):
        os.makedirs(JOBS_DIR, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: Dict[str, PrimsJob] = {}
        self._global: List[str] = []
        self._global_lock = threading.Lock()
        self._load_from_disk()
        self.runner = PrimsRunner(self)

    # ---------- 全局日志 ----------
    @property
    def global_log_path(self) -> str:
        return os.path.join(OUTPUT_DIR, "global.log")

    def log_global(self, job_name: str, line: str) -> None:
        text = f"{line}  [{job_name}]"
        with self._global_lock:
            self._global.append(text)
            if len(self._global) > 2000:
                del self._global[:500]
        try:
            with open(self.global_log_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            pass

    def global_tail(self, limit: int = 400) -> List[str]:
        with self._global_lock:
            if not self._global:
                # 首次访问从磁盘捞一次（服务重启后还在）
                try:
                    with open(self.global_log_path, encoding="utf-8") as f:
                        self._global = f.read().splitlines()[-2000:]
                except OSError:
                    pass
            return self._global[-limit:]

    # ---------- 磁盘 ----------
    def _load_from_disk(self) -> None:
        if not os.path.isdir(JOBS_DIR):
            return
        for name in sorted(os.listdir(JOBS_DIR)):
            meta_path = os.path.join(JOBS_DIR, name, "job.json")
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            params = data.get("params") or {}
            job = PrimsJob(name, data.get("name") or name, params)
            job.sink = self.log_global
            job.load_meta(data)
            # 上次没跑完的（进程被杀）标成 error，不自动重跑——避免开机就
            # 莫名其妙跑一堆 20 分钟的活。
            if job.status in ("running", "queued"):
                job.status = "error"
                job.error = "服务重启，任务已中断"
                job.finished = job.finished or _now()
                job.save_meta()
            result_path = os.path.join(JOBS_DIR, name, "result.json")
            if os.path.isfile(result_path):
                try:
                    with open(result_path, encoding="utf-8") as f:
                        job.result = json.load(f)
                except (OSError, ValueError):
                    job.result = None
            self._jobs[name] = job

    # ---------- CRUD ----------
    def list(self) -> List[PrimsJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created,
                          reverse=True)

    def get(self, job_id: str) -> Optional[PrimsJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def create(self, name: str, image_bytes: bytes,
               params: Dict[str, Any]) -> PrimsJob:
        job_id = "p" + time.strftime("%y%m%d%H%M%S") + uuid.uuid4().hex[:6]
        job = PrimsJob(job_id, name, params)
        job.sink = self.log_global
        job.ensure_dir()
        with open(job.source_path, "wb") as f:
            f.write(image_bytes)
        job.log("已加入队列")
        job.save_meta()
        with self._lock:
            self._jobs[job_id] = job
        return job

    def delete(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is not None and os.path.isdir(job.dir):
            shutil.rmtree(job.dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# API（挂在 server.py 的 App 上）
# ---------------------------------------------------------------------------
class PrimsApi:
    def _prims_store(self) -> PrimsStore:
        if not hasattr(self, "_prims"):
            self._prims = PrimsStore()
        return self._prims

    def api_prims_jobs(self, query):
        store = self._prims_store()
        jobs = [j.meta() for j in store.list()]
        return 200, {"jobs": jobs}

    def api_prims_job_create(self, body):
        import base64
        b64 = body.get("image_b64")
        if not b64:
            raise ValueError("缺少 image_b64")
        try:
            raw = base64.b64decode(b64, validate=False)
        except Exception:
            raise ValueError("image_b64 不是合法 base64") from None
        if not raw:
            raise ValueError("图片为空")
        steps = int(body.get("steps") or 100)
        if steps < 1 or steps > MAX_STEPS:
            raise ValueError(f"steps 必须在 1..{MAX_STEPS}")
        canvas = body.get("canvas")
        if canvas not in (None, "auto"):
            try:
                cw, ch = int(canvas[0]), int(canvas[1])
            except (TypeError, ValueError, IndexError):
                raise ValueError(f"画布格式不对: {canvas!r}") from None
            if not spec.draw_canvas_valid(cw, ch):
                raise ValueError(
                    f"画布 {cw}×{ch} 超出 1..{spec.DRAW_CANVAS_MAX_W} × "
                    f"1..{spec.DRAW_CANVAS_MAX_H}")
            canvas = [cw, ch]
        params = {
            "engine": body.get("engine") or "auto",
            "quality": body.get("quality") or "fast",
            "steps": steps,
            "early_stop": bool(body.get("early_stop", True)),
            "canvas": canvas or "auto",
        }
        name = (body.get("name") or "").strip() or "未命名"
        job = self._prims_store().create(name, raw, params)
        job.save_meta()
        self._prims_store().runner.enqueue(job.id)
        return 200, {"id": job.id, "job": job.meta()}

    def api_prims_job_update(self, body):
        """改一个**还没开始**的任务的参数（已开始的改了没意义）。"""
        job_id = body.get("id")
        if not job_id:
            raise ValueError("缺少 id")
        job = self._prims_store().get(job_id)
        if job is None:
            raise ValueError(f"任务不存在: {job_id}")
        if job.status != "queued":
            raise ValueError("只有排队中的任务可以改参数")
        params = dict(job.params)
        if body.get("engine") is not None:
            params["engine"] = body["engine"]
        if body.get("quality") is not None:
            params["quality"] = body["quality"]
        if body.get("steps") is not None:
            steps = int(body["steps"])
            if steps < 1 or steps > MAX_STEPS:
                raise ValueError(f"steps 必须在 1..{MAX_STEPS}")
            params["steps"] = steps
            job.progress_total = steps
        if body.get("early_stop") is not None:
            params["early_stop"] = bool(body["early_stop"])
        if body.get("canvas") is not None:
            params["canvas"] = body["canvas"]
        if body.get("name"):
            job.name = str(body["name"]).strip() or job.name
        job.params = params
        job.log("参数已更新")
        job.save_meta()
        return 200, {"ok": True, "job": job.meta()}

    def api_prims_job_delete(self, body):
        job_id = body.get("id")
        if not job_id:
            raise ValueError("缺少 id")
        self._prims_store().delete(job_id)
        return 200, {"ok": True}

    def api_prims_job_live(self, query):
        job_id = (query.get("id") or [""])[0]
        since = int((query.get("since") or ["0"])[0] or 0)
        job = self._prims_store().get(job_id)
        if job is None:
            raise ValueError(f"任务不存在: {job_id}")
        with job._lock:
            prims = job.live[since:]
            total = len(job.live)
            logs = job.log_lines[-40:]
        return 200, {
            "id": job.id, "status": job.status,
            "canvas": job.canvas,
            "progress": {"step": job.progress_step,
                         "total": job.progress_total},
            "prims": prims, "since": since, "next": total,
            "elapsed": (job.finished or _now()) - job.started if job.started else None,
            "error": job.error, "note": job.note,
            "log": logs,
        }

    def api_prims_job_result(self, query):
        job_id = (query.get("id") or [""])[0]
        job = self._prims_store().get(job_id)
        if job is None:
            raise ValueError(f"任务不存在: {job_id}")
        if job.result is None and os.path.isfile(job.result_path):
            try:
                with open(job.result_path, encoding="utf-8") as f:
                    job.result = json.load(f)
            except (OSError, ValueError):
                job.result = None
        if job.result is None:
            raise ValueError("结果还没生成")
        return 200, job.result

    def api_prims_log(self, query):
        limit = int((query.get("limit") or ["400"])[0] or 400)
        return 200, {"lines": self._prims_store().global_tail(limit)}

    def api_prims_job_source(self, query):
        """原图（PNG 字节）。返回 (code, obj) 里的 obj 是 bytes 时由路由层直发。"""
        job_id = (query.get("id") or [""])[0]
        job = self._prims_store().get(job_id)
        if job is None or not os.path.isfile(job.source_path):
            raise ValueError(f"任务不存在: {job_id}")
        return 200, {"path": job.source_path}
