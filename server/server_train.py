# ===========================================================================
# GARGANTUA v1 — server_train.py：训练编排 API（任务 + 启停 + 状态）
# ---------------------------------------------------------------------------
# 从 server.py 拆出。只负责「数据入队 / 任务元数据 / 训练服务生命周期 /
# 状态查询」，不碰模型构造、不碰 HTTP 框架。App 以 mixin 方式继承本类：
#
#     class App(TrainApi, CodecApi, ...): ...
#
# 2026-11-07 训练界面大改；2026-11-08 按 steve 裁决修正：
#   * 上传**只收 .tokens**：普通训练一个框可多选（每文件 = 一个任务），
#     对比训练两个框各一个 .tokens；对比的两段模态任意组合（不再要求
#     「一段文本 + 一段媒体」），由 clip_data 通用 A×B 配对承载。
#   * 训练口径回到原先的「一个文件固定 steps 步、每步随机采样一个窗口」，
#     token 级遍历整套移除，loss 按样本/步记录。
#   * 每个队列项配套一份**任务元数据** sidecar（data/task_meta/*.json）：
#     目标模型 / 优化器 / 上下文窗口 / steps 都随任务走，训练守护逐个
#     任务读取后临时覆盖参数（见 auto_train.py）。
#   * 队列分三态：pending（未完成）/ done（已完成）/ failed（隔离），
#     接口一次给全；删除可作用于任一态；未完成任务可编辑。
#   * 新增事件流（logs/train_events.jsonl）与 loss 数据接口，供训练日志
#     视图与「每个文件一个标签页」的 loss 曲线视图使用。
#   * 旧的 tokens_b64 + mode=lm/clip 入队口径逐字保留（CLI / 测试共用）。
#
# 训练服务仍是按需拉起的子进程（service/train/auto_train.py --serve），
# server 本体不持有模型权重。
# ===========================================================================
import base64
import glob
import json
import os
import time
import uuid

import spec
from common import model_init_lock, validate_block_size, validate_steps
from server_models import ConflictError
from tokens import tok6

# 对比训练的通用 A×B 配对契约与队列文件名元数据由 clip_data 统一提供，
# server / auto_train / 终端 CLI 三处共用，不各写一套解析。
import clip_data
import clip_folder
from clip_data import (MODES, OPTIMIZERS, PairError, parse_bytes,
                       parse_queue_name, make_queue_name,
                       DEFAULT_MAX_PAIR_POSITIONS, MAX_ENQUEUE_BYTES,
                       MAX_ENQUEUE_ELEMENTS)

# 训练服务未启动时 /api/train/status 的默认态（与服务内默认值一致）
_DEFAULT_STATUS = {
    "running": False, "step": 0, "block_size": 32,
    "steps_per_file": 50, "optimizer": spec.OPTIMIZER,
    "loss_tail": [], "log_tail": [], "current_file": None,
    # 2026-11-06：CLIP 训练参数（与服务内默认值一致）
    "clip_batch_size": 4, "clip_temperature": 0.07,
}

# 一次入队的文件数上限（普通训练多文件）
MAX_FILES_PER_ENQUEUE = 32
# 训练曲线 / 事件最多回灌的条数
MAX_LOSS_POINTS = 8000
MAX_EVENTS = 400

# 可编辑的任务字段（mode 不允许改：训练方式变了数据结构完全不同，
# 应当删除后重建，而不是让队列里躺着一个内容和文件名不符的任务）
EDITABLE_FIELDS = ("optimizer", "target", "steps", "block_size",
                   "clip_batch_size", "label")


def _validate_clip_batch_size(v):
    hi = spec.CLIP_FOLDER_MAX_BATCH
    if isinstance(v, bool) or not isinstance(v, int) or not (2 <= v <= hi):
        raise ValueError(f"clip_batch_size 必须是 2..{hi} 的整数")
    return v


def _validate_clip_temperature(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("clip_temperature 必须是数值")
    if not (0 < float(v) <= 100):
        raise ValueError("clip_temperature 必须落在 (0, 100]")
    return float(v)


def _validate_steps_limit(v):
    """steps = 每个文件固定训练多少步（每步随机采样一个窗口），≥1。"""
    if isinstance(v, bool) or v is None:
        raise ValueError("steps 必须是整数")
    try:
        v = int(v)
    except (TypeError, ValueError):
        raise ValueError(f"steps 必须是整数: {v!r}") from None
    if not (1 <= v <= 1_000_000):
        raise ValueError("steps 必须落在 [1, 1000000]")
    return v


class TrainApi:
    """ /api/train/* 全部端点。 """

    # ------------------------------------------------------------------
    # 任务元数据 sidecar（data/task_meta/<队列文件名>.json）
    # ------------------------------------------------------------------
    def _data_root(self) -> str:
        """真 App 有 data_dir；测试里的轻量宿主只给了 queue_dir，按它的
        父目录兜底——不让「只借用几个 API 方法的 fake」因缺属性而 500。"""
        return getattr(self, "data_dir", None) or os.path.dirname(
            os.path.abspath(getattr(self, "queue_dir", ".")))

    def _meta_dir(self) -> str:
        d = getattr(self, "meta_dir", None) or os.path.join(
            self._data_root(), "task_meta")
        os.makedirs(d, exist_ok=True)
        return d

    def _meta_path(self, name: str) -> str:
        return os.path.join(self._meta_dir(),
                            os.path.basename(name) + ".json")

    def _read_meta(self, name: str) -> dict:
        try:
            with open(self._meta_path(name), encoding="utf-8") as f:
                obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_meta(self, name: str, meta: dict) -> None:
        path = self._meta_path(name)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _drop_meta(self, name: str) -> None:
        try:
            os.unlink(self._meta_path(name))
        except OSError:
            pass

    def _events_path(self):
        lp = getattr(self, "log_path", None)
        if not lp:
            # 没有日志路径的轻量宿主：不写事件，也绝不往系统临时目录落文件
            return None
        d = os.path.dirname(os.path.abspath(lp))
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "train_events.jsonl")

    def _event(self, kind: str, text: str, **extra) -> None:
        path = self._events_path()
        if not path:
            return
        rec = {"ts": time.time(), "time": time.strftime("%H:%M:%S"),
               "kind": kind, "text": text}
        rec.update(extra)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _done_dir(self) -> str:
        return os.path.join(self._data_root(), "done")

    def _failed_dir(self) -> str:
        return os.path.join(self._data_root(), "failed")

    def _atomic_write(self, path: str, data: bytes) -> None:
        """先写隐藏临时文件 + fsync，再原子 rename 进队列。

        训练守护每秒扫描队列，直接写 .tokens 会让它读到不完整流。"""
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        tmp = os.path.join(directory, "." + os.path.basename(path) + ".tmp")
        try:
            with open(tmp, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def _queue_one(self, data: bytes, optimizer: str, mode: str,
                   meta: dict) -> dict:
        """校验 + 落盘 + 写 sidecar，返回任务描述。"""
        if len(data) > MAX_ENQUEUE_BYTES:
            raise ValueError(f"训练数据超限: {len(data)} 字节 > "
                             f"{MAX_ENQUEUE_BYTES}，请拆分后上传")
        try:
            items = tok6.loads(data)
        except tok6.Tok6Error as e:
            raise ValueError(f"不是合法的 token-6 .tokens: {e}")
        if not items:
            raise ValueError("训练数据为空（没有任何时间步）")
        n_steps = sum(1 for it in items if it["kind"] == "step")
        n_elements = sum(len(it["elements"]) for it in items
                         if it["kind"] == "step")
        if n_elements > MAX_ENQUEUE_ELEMENTS:
            raise ValueError(f"元素数超限: {n_elements} > {MAX_ENQUEUE_ELEMENTS}")
        bad = next((e["id"] for it in items if it["kind"] == "step"
                    for e in it["elements"]
                    if e["kind"] == "text"
                    and not (0 <= e["id"] < spec.MAX_VOCAB_SIZE)), None)
        if bad is not None:
            raise ValueError(f"token ID {bad} 超出词表"
                             f"（< {spec.MAX_VOCAB_SIZE}）")
        info = {}
        if mode == "clip":
            try:
                info = parse_bytes(data, DEFAULT_MAX_PAIR_POSITIONS)
            except PairError as e:
                raise ValueError(f"CLIP 配对 .tokens 非法: {e}")
        elif mode == "pair":
            try:
                info = parse_bytes(data, DEFAULT_MAX_PAIR_POSITIONS)
            except PairError as e:
                raise ValueError(f"对比训练配对数据非法: {e}")
        elif n_elements == 0:
            raise ValueError("训练数据没有可训练元素（只有扩展系统 token）")

        fn = make_queue_name(optimizer, mode)
        self._atomic_write(os.path.join(self.queue_dir, fn), data)
        full = {
            "name": fn, "mode": mode, "optimizer": optimizer,
            "label": meta.get("label") or fn,
            "target": meta.get("target") or "",
            # 旧 sidecar 可能残留 steps=0（token 级时代的「自动全量」），
            # 服务端也兜一层：0 / None 一律回落到固定步数默认值。
            "steps": int(meta.get("steps") or 20),
            "block_size": meta.get("block_size", 32),
            "clip_batch_size": meta.get("clip_batch_size", 4),
            "created": time.time(),
            "bytes": len(data), "n_steps": n_steps,
            "n_elements": n_elements,
        }
        # 数据集文件夹派生的额外元数据（批次归属 / 顺序 / 温度）
        for k in ("pair", "group", "pair_index", "clip_temperature"):
            if meta.get(k) is not None:
                full[k] = meta[k]
        self._write_meta(fn, full)
        self._event("enqueue", f"入队 {full['label']}（{mode} / {optimizer}）",
                    name=fn, mode=mode)
        out = {"name": fn, "bytes": len(data), "n_tokens": n_elements,
               "n_steps": n_steps, "mode": mode, "meta": full}
        if info:
            out.update({"media_kind": info.get("media_kind", ""),
                        "n_text_tokens": info.get("n_text_tokens", 0),
                        "n_media_positions": info.get("n_media_positions", 0)})
        return out

    # ------------------------------------------------------------------
    # 入队
    # ------------------------------------------------------------------
    def api_train_enqueue(self, body):
        """训练数据入队。

        * mode="lm"   普通训练：files = [{name, data_b64}]，只收 .tokens，
                      可多选，每个文件 = 一个队列任务。
        * mode="clip" 对比训练：**强制定死数据集文件夹格式 v1**
                      dataset = {files: [{path, data_b64}]}，由 clip_folder
                      严格校验后解析出 N 对真实配对，每对写成一个 mode=clip
                      队列文件；训练侧同批做对称 InfoNCE，负样本来自批内
                      其它配对（100% CLIP）。两文件直传已废弃。
        * 旧口径：tokens_b64（.tokens 二进制）+ mode=lm/clip，CLI / 测试保留。
        """
        mode = body.get("mode", "lm")
        if mode not in MODES:
            raise ValueError(f"未知训练方式: {mode}（可选 {', '.join(MODES)}）")
        optimizer = body.get("optimizer", spec.OPTIMIZER)
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"未知优化器: {optimizer}")
        target = str(body.get("target") or "")
        steps = _validate_steps_limit(body.get("steps", 20))
        block_size = validate_block_size(body.get("block_size", 32))
        clip_batch_size = _validate_clip_batch_size(
            body.get("clip_batch_size", 4))
        common_meta = {"target": target, "steps": steps,
                       "block_size": block_size,
                       "clip_batch_size": clip_batch_size}

        # ---- 对比训练：数据集文件夹（唯一合法入口）----
        dataset = body.get("dataset")
        if dataset is not None:
            if mode != "clip":
                raise ValueError("数据集文件夹必须用 mode=clip（对比训练）")
            if not isinstance(dataset, dict):
                raise ValueError("dataset 必须是 {files: [{path, data_b64}]}")
            fs = dataset.get("files")
            if not isinstance(fs, list) or not fs:
                raise ValueError("dataset.files 必须是非空数组")
            try:
                parsed = clip_folder.parse(fs)
            except clip_folder.FolderError as e:
                raise ValueError(f"数据集文件夹不合规: {e}") from None
            group = uuid.uuid4().hex[:12]
            made = []
            for i, p in enumerate(parsed["pairs"]):
                try:
                    data = clip_data.pair_bytes_generic(
                        p["a_items"], p["b_items"])
                except PairError as e:
                    raise ValueError(f"配对「{p['id']}」非法: {e}") from None
                meta = dict(common_meta,
                            label=f"{parsed['name']}/{p['id']}",
                            group=group, pair_index=i,
                            clip_batch_size=parsed["batch_size"],
                            clip_temperature=parsed["temperature"])
                made.append(self._queue_one(data, optimizer, "clip", meta))
            self._event("dataset",
                        f"解析数据集 {parsed['name']}：{parsed['n_pairs']} 对"
                        "配对入队（CLIP 批内负样本）", group=group)
            return 200, {
                "queued": [m["name"] for m in made],
                "tasks": [m["meta"] for m in made],
                "n_pairs": parsed["n_pairs"],
                "batch_size": parsed["batch_size"],
                "dataset": parsed["name"],
                "n_tokens": sum(m["n_tokens"] for m in made),
                "n_steps": sum(m["n_steps"] for m in made),
                "bytes": sum(m["bytes"] for m in made),
                "mode": "clip",
            }

        files = body.get("files")
        if files is not None:
            if mode != "lm":
                raise ValueError("对比训练必须上传数据集文件夹格式 v1；"
                                 "files 直传只用于普通训练（lm）")
            if not isinstance(files, list) or not files:
                raise ValueError("files 必须是非空数组")
            if len(files) > MAX_FILES_PER_ENQUEUE:
                raise ValueError(f"一次最多上传 {MAX_FILES_PER_ENQUEUE} 个文件")
            decoded = []
            for f in files:
                if not isinstance(f, dict):
                    raise ValueError("files 元素必须是 {name, data_b64}")
                name = str(f.get("name") or "")
                # 上传一律只收 .tokens：前端的 accept / onlyTokens 只是
                # 体验层，真正契约在这里（服务端不解析任何原始媒体文件）。
                if not name.lower().endswith(".tokens"):
                    raise ValueError(
                        f"训练数据只接受 .tokens 文件，收到「{name}」")
                b64 = f.get("data_b64")
                if not b64 or not isinstance(b64, str):
                    raise ValueError(f"文件「{name}」缺少 data_b64")
                try:
                    raw = base64.b64decode(b64, validate=True)
                except Exception as e:
                    raise ValueError(f"文件「{name}」的 data_b64 非法: {e}")
                if not raw:
                    raise ValueError(f"文件「{name}」为空")
                try:
                    items = tok6.loads(raw)
                except tok6.Tok6Error as e:
                    raise ValueError(f"「{name}」不是合法的 token-6 .tokens: {e}")
                if not items:
                    raise ValueError(f"「{name}」是空的 .tokens")
                decoded.append((name, raw, items))

            made = []
            for name, raw, _items in decoded:
                meta = dict(common_meta, label=name)
                made.append(self._queue_one(raw, optimizer, "lm", meta))
            out = {"queued": [m["name"] for m in made],
                   "tasks": [m["meta"] for m in made],
                   "n_tokens": sum(m["n_tokens"] for m in made),
                   "n_steps": sum(m["n_steps"] for m in made),
                   "bytes": sum(m["bytes"] for m in made),
                   "mode": mode}
            if made[0].get("media_kind"):
                out["media_kind"] = made[0]["media_kind"]
                out["n_text_tokens"] = made[0].get("n_text_tokens", 0)
                out["n_media_positions"] = made[0].get("n_media_positions", 0)
            return 200, out

        # ---- 旧口径：单个 .tokens 位流（CLI / 既有测试走这里）----
        if body.get("json_b64") is not None:
            raise ValueError("训练数据已改为 .tokens 格式（token-6 位流，"
                             "必须带 { } 时间步结构）；不再接受 JSON/JSONL")
        tokens_b64 = body.get("tokens_b64")
        if not tokens_b64:
            raise ValueError("缺少 tokens_b64（训练数据必须是 .tokens 文件）")
        if not isinstance(tokens_b64, str):
            raise ValueError("tokens_b64 必须是字符串")
        try:
            data = base64.b64decode(tokens_b64, validate=True)
        except Exception as e:
            raise ValueError(f"tokens_b64 不是合法 base64: {e}")
        if not data:
            raise ValueError("训练数据为空")
        made = self._queue_one(data, optimizer, mode, common_meta)
        out = {"queued": [made["name"]], "tasks": [made["meta"]],
               "n_tokens": made["n_tokens"], "n_steps": made["n_steps"],
               "bytes": made["bytes"], "mode": mode}
        for k in ("media_kind", "n_text_tokens", "n_media_positions"):
            if k in made:
                out[k] = made[k]
        return 200, out

    # ------------------------------------------------------------------
    # 训练服务生命周期（supervisor 调用；服务没起就返回默认态）
    # ------------------------------------------------------------------
    def _train_model_dir(self, name=None) -> str:
        name = name or self.mgr.name
        return os.path.join(self.mgr.model_dir, name)

    def _first_pending_meta(self) -> dict:
        try:
            names = sorted(f for f in os.listdir(self.queue_dir)
                           if f.endswith(".tokens"))
        except OSError:
            return {}
        return self._read_meta(names[0]) if names else {}

    def api_train_start(self, body):
        if self.mgr.name is None:
            raise ValueError("无活动模型，请先加载模型")
        # 目标模型：UI 显式传 target 优先；没传就按队首任务的目标模型
        # 走（任务级参数随任务走，不再有「当前活动模型」历史遗留）。
        target = body.get("target")
        if not target:
            target = self._first_pending_meta().get("target") or ""
        if target and target != self.mgr.name:
            with self._state_lock:
                if self.trainer.running:
                    raise ConflictError("训练进行中，请先停止训练再切换模型")
                self.mgr.load(target)
            from tokens.vocab import set_model_vocab
            try:
                set_model_vocab(self._train_model_dir())
            except Exception as e:
                raise ValueError(f"模型词表加载失败: {e}")
        block_size = None
        steps = None
        optimizer = None
        clip_batch_size = None
        clip_temperature = None
        if body.get("block_size") is not None:
            block_size = validate_block_size(body["block_size"])
        if body.get("steps") is not None:
            steps = validate_steps(body["steps"])
        if body.get("optimizer"):
            opt = body["optimizer"]
            if opt not in OPTIMIZERS:
                raise ValueError(f"未知优化器: {opt}")
            optimizer = opt
        if body.get("clip_batch_size") is not None:
            clip_batch_size = _validate_clip_batch_size(body["clip_batch_size"])
        if body.get("clip_temperature") is not None:
            clip_temperature = _validate_clip_temperature(
                body["clip_temperature"])
        lock = model_init_lock(self._train_model_dir(), timeout=120)
        try:
            lock.__enter__()
        except RuntimeError as e:
            raise ConflictError(str(e))
        try:
            res = self.supervisor.call(
                "train", "start", timeout=300,
                model_dir=self._train_model_dir(),
                queue_dir=self.queue_dir,
                done_dir=self._done_dir(),
                log_path=self.log_path,
                task_meta_dir=self._meta_dir(),
                block_size=block_size, steps_per_file=steps,
                optimizer=optimizer,
                clip_batch_size=clip_batch_size,
                clip_temperature=clip_temperature)
        finally:
            lock.__exit__(None, None, None)
        self._event("start", "开始训练" + (f" · 目标模型 {target}"
                                          if target else ""),
                    target=target)
        return 200, res

    def api_train_stop(self, _body):
        if not self.trainer.running:
            return 200, {"stopping": False, **_DEFAULT_STATUS}
        res = self.supervisor.call("train", "stop", timeout=30)
        self._event("stop", "停止训练请求已发送")
        return 200, {"stopping": True, **res}

    def api_train_status(self):
        res = self.supervisor.query("train", "status")
        if res is None:
            return dict(_DEFAULT_STATUS)
        out = dict(_DEFAULT_STATUS)
        out.update(res)
        return out

    # ---------------- 队列 / 任务管理 ----------------
    def _list_dir(self, directory: str, where: str, running_name: str):
        items = []
        for path in sorted(glob.glob(os.path.join(directory, "*"))):
            name = os.path.basename(path)
            if not name.endswith(".tokens"):
                continue
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            meta = parse_queue_name(name)
            saved = self._read_meta(name)
            items.append({
                "name": name, "size": size, "where": where,
                "optimizer": saved.get("optimizer") or meta["optimizer"],
                "mode": saved.get("mode") or meta["mode"],
                "label": saved.get("label") or name,
                "target": saved.get("target") or "",
                "steps": int(saved.get("steps") or 20),
                "block_size": saved.get("block_size", 32),
                "clip_batch_size": saved.get("clip_batch_size", 4),
                "created": saved.get("created", 0),
                "n_steps": saved.get("n_steps", 0),
                "n_elements": saved.get("n_elements", 0),
                "pair": saved.get("pair"),
                "running": where == "queue" and name == running_name,
            })
        return items

    def api_train_queue(self):
        status = None
        query = getattr(getattr(self, "supervisor", None), "query", None)
        if query is not None:
            try:
                status = query("train", "status")
            except Exception:
                status = None
        running_name = (status or {}).get("current_file") or ""
        pending = self._list_dir(self.queue_dir, "queue", running_name)
        done = self._list_dir(self._done_dir(), "done", "")
        failed = self._list_dir(self._failed_dir(), "failed", "")
        return 200, {"items": pending, "pending": pending, "done": done,
                     "failed": failed, "running": running_name or None}

    def api_train_delete(self, body):
        name = body.get("name")
        if not name:
            raise ValueError("缺少 name")
        name = os.path.basename(str(name))
        where = body.get("where", "queue")
        dirs = {"queue": self.queue_dir, "done": self._done_dir(),
                "failed": self._failed_dir()}
        if where not in dirs:
            raise ValueError(f"未知位置: {where}")
        path = os.path.join(dirs[where], name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"任务不存在: {name}")
        if not os.path.isfile(path):
            raise ValueError(f"任务不是文件: {name}")
        os.unlink(path)
        self._drop_meta(name)
        self._event("delete", f"删除任务 {name}（{where}）", name=name)
        return 200, {"deleted": name}

    def api_train_task_update(self, body):
        """编辑未完成任务（正在进行的任务拒绝）。

        optimizer 变化会同步改队列文件名——文件名是 mode/optimizer 的
        唯一携带者，训练守护靠它分流，不能只改 sidecar。
        """
        name = os.path.basename(str(body.get("name") or ""))
        if not name:
            raise ValueError("缺少 name")
        path = os.path.join(self.queue_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"未完成任务不存在: {name}")
        running = (self.api_train_queue()[1].get("running") or "")
        if running == name:
            raise ConflictError("正在进行的任务不能编辑，请先停止训练")
        meta = self._read_meta(name)
        if not meta:
            parsed = parse_queue_name(name)
            meta = {"name": name, "mode": parsed["mode"],
                    "optimizer": parsed["optimizer"] or spec.OPTIMIZER,
                    "label": name, "target": "", "steps": 20,
                    "block_size": 32,
                    "clip_batch_size": 4, "created": time.time(),
                    "bytes": os.path.getsize(path)}
        for k in EDITABLE_FIELDS:
            if k not in body:
                continue
            v = body[k]
            if k == "optimizer":
                if v not in OPTIMIZERS:
                    raise ValueError(f"未知优化器: {v}")
            elif k == "steps":
                v = _validate_steps_limit(v)
            elif k == "block_size":
                v = validate_block_size(v)
            elif k == "clip_batch_size":
                v = _validate_clip_batch_size(v)
            elif k in ("target", "label"):
                v = str(v or "")
            meta[k] = v
        new_name = name
        if meta["optimizer"] != parse_queue_name(name)["optimizer"]:
            parts = name.split("_")
            if len(parts) >= 3:
                parts[2] = meta["optimizer"]
                new_name = "_".join(parts)
                new_path = os.path.join(self.queue_dir, new_name)
                os.replace(path, new_path)
                self._drop_meta(name)
                meta["name"] = new_name
        self._write_meta(new_name, meta)
        self._event("edit", f"编辑任务 {meta.get('label') or new_name}",
                    name=new_name)
        return 200, {"name": new_name, "meta": meta}

    # ---------------- 数据集文件夹格式说明 ----------------
    def api_train_dataset_spec(self):
        """前端一律从这里读文件树与规则，不许抄常量 / 自己拼说明。"""
        return clip_folder.spec_payload()

    # ---------------- 日志视图：事件 + 文本日志 ----------------
    def api_train_events(self):
        path = self._events_path()
        if not path:
            return {"events": []}
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()[-MAX_EVENTS:]
        except OSError:
            return {"events": []}
        events = []
        for line in lines:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    events.append(obj)
            except ValueError:
                continue
        return {"events": events}

    # ---------------- loss 视图：按文件分组的原始点 ----------------
    def api_train_loss(self):
        """logs/auto_train.jsonl 是唯一 loss 真源；按队列文件名分组。

        每个点 = 一个样本窗口的独立 loss（step / loss / file / ts），
        前端每个文件一个标签页、横轴按训练步序铺开，不再分段。
        """
        lp = getattr(self, "log_path", None)
        if not lp:
            return {"files": [], "total": 0}
        try:
            with open(lp, encoding="utf-8") as f:
                lines = f.readlines()[-MAX_LOSS_POINTS:]
        except OSError:
            return {"files": [], "total": 0}
        groups = {}
        total = 0
        for line in lines:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict) or "loss" not in rec:
                continue
            key = rec.get("file") or "(未标注文件)"
            groups.setdefault(key, []).append(rec)
            total += 1
        files = [{"name": k, "points": v, "n": len(v)}
                 for k, v in groups.items()]
        return {"files": files, "total": total}
