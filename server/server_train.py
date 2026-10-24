# ===========================================================================
# GARGANTUA v1 — server_train.py：训练编排 API（队列 + 启停 + 状态）
# ---------------------------------------------------------------------------
# 从 server.py 拆出。只负责「数据入队 / 训练服务生命周期 / 状态查询」，
# 不碰模型构造、不碰 HTTP 框架。App 以 mixin 方式继承本类：
#
#     class App(TrainApi, CodecApi): ...
#
# 2026-10-24 重构（steve：服务按需启动，省内存）：训练不再是 server 进程
# 内的守护线程，而是 service/train/auto_train.py --serve 子进程（协议见
# server_services.py）。server 只做三件事：
#   1. 入队/队列管理/删除——纯文件系统操作，留在 server（便宜）；
#   2. 启停——supervisor.call("train", ...)（拉起/复用训练服务进程）；
#   3. 状态——supervisor.query("train", "status")；服务没起就返回默认态
#      （不为一次状态查询而起进程，那正是「按需」的意义）。
#
# self.trainer 是 App.__init__ 注入的轻量状态代理（.running 反映训练服务
# 是否在跑），供 409 冲突检查使用；训练参数（block_size/steps/optimizer）
# 随 start  Op 下发给训练服务，服务退出后自然回到默认值。
# ===========================================================================
import base64
import glob
import json
import os
import time
import uuid

import spec
from common import validate_block_size, validate_steps
from server_models import ConflictError

# 2026-10-23 审计修复（P3）：入队此前无任何数量/体积上限——512MB 的
# JSON 数组可拆出数百万个 .txt 小文件（inode 耗尽），且 json.loads 本身
# 就要吃掉数倍内存。设样本数与总字符双上限。
MAX_ENQUEUE_SAMPLES = 10_000        # 单次入队样本数上限
MAX_ENQUEUE_CHARS = 64 * 1024 * 1024   # 单次入队总字符上限（64M）

# 训练服务未启动时 /api/train/status 的默认态（与服务内默认值一致）
_DEFAULT_STATUS = {
    "running": False, "step": 0, "block_size": 32,
    "steps_per_file": 50, "optimizer": spec.OPTIMIZER,
    "loss_tail": [], "log_tail": [],
}


class TrainApi:
    """ /api/train/* 全部端点。 """

    def api_train_enqueue(self, body):
        """训练数据：json_b64 上传。内容必须是
        {"text": "..."} 对象 / 对象数组 / JSONL 每行一个对象。"""
        json_b64 = body.get("json_b64")
        if not json_b64:
            raise ValueError("缺少 json_b64（仅支持上传 JSON/JSONL 训练数据）")
        raw = base64.b64decode(json_b64).decode("utf-8")
        filename = body.get("filename") or ""
        samples = []
        # 大小写不敏感：Data.TXT 此前不走 JSONL 分支被当 JSON 解析失败
        if filename.lower().endswith(".jsonl"):
            for i, line in enumerate(raw.splitlines()):
                if not line.strip():
                    continue
                obj = json.loads(line)
                # 2026-10-23 审计修复（P2）：{"text": 123}/{"text": null}
                # 此前通过「键存在」检查，直到写盘 f.write() 才抛
                # TypeError → HTTP 500，违反 400 入参契约
                if not isinstance(obj, dict) or "text" not in obj:
                    raise ValueError(f"第 {i+1} 行缺少 text 字段，"
                                     f"训练数据必须为 {{\"text\": \"...\"}}")
                if not isinstance(obj["text"], str):
                    raise ValueError(f"第 {i+1} 行 text 必须是字符串: "
                                     f"{obj['text']!r}")
                samples.append(obj["text"])
        else:
            obj = json.loads(raw)
            items = obj if isinstance(obj, list) else [obj]
            for it in items:
                if not isinstance(it, dict) or "text" not in it:
                    raise ValueError("每条训练数据必须为 "
                                     "{\"text\": \"...\"} 对象")
                if not isinstance(it["text"], str):
                    raise ValueError(f"text 必须是字符串: {it['text']!r}")
                samples.append(it["text"])
        if not samples:
            raise ValueError("训练数据为空")
        if len(samples) > MAX_ENQUEUE_SAMPLES:
            raise ValueError(f"样本数超限: {len(samples)} > "
                             f"{MAX_ENQUEUE_SAMPLES}，请分批上传")
        total_chars = sum(len(t) for t in samples)
        if total_chars > MAX_ENQUEUE_CHARS:
            raise ValueError(f"总字符数超限: {total_chars} > "
                             f"{MAX_ENQUEUE_CHARS}，请分批上传")
        optimizer = body.get("optimizer", spec.OPTIMIZER)
        if optimizer not in ("muon", "adamw"):
            raise ValueError(f"未知优化器: {optimizer}")
        queued = []
        for text in samples:
            fn = f"{time.strftime('%Y%m%d_%H%M%S')}_{optimizer}_" \
                 f"{uuid.uuid4().hex[:8]}.txt"
            path = os.path.join(self.queue_dir, fn)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            queued.append(fn)
        return 200, {"queued": queued, "n_samples": len(samples)}

    # ------------------------------------------------------------------
    # 训练服务生命周期（supervisor 调用；服务没起就返回默认态）
    # ------------------------------------------------------------------
    def _train_model_dir(self, name=None) -> str:
        name = name or self.mgr.name
        return os.path.join(self.mgr.model_dir, name)

    def api_train_start(self, body):
        if self.mgr.name is None:
            raise ValueError("无活动模型，请先加载模型")
        # 可选：切换到训练页选择的模型（未在训练时才允许；此前该下拉
        # 框采集后不发送，是死 UI——2026-10-20 接线）
        target = body.get("target")
        if target and target != self.mgr.name:
            if self.trainer.running:
                raise ConflictError("训练进行中，请先停止训练再切换模型")
            self.mgr.load(target)
        # 统一校验（非法值 400）：奇数 block_size 令前向崩溃、0 步触发
        # 训练服务 _loss_tail[-1] IndexError（均实测复现过）。
        # 校验留在 server 侧：400 契约不依赖服务进程是否可达。
        block_size = None
        steps = None
        optimizer = None
        if body.get("block_size") is not None:
            block_size = validate_block_size(body["block_size"])
        if body.get("steps") is not None:
            steps = validate_steps(body["steps"])
        if body.get("optimizer"):
            opt = body["optimizer"]
            if opt not in ("muon", "adamw"):
                raise ValueError(f"未知优化器: {opt}")
            optimizer = opt
        # 拉起/复用训练服务进程并下发参数（建模加载权重可能耗时数十秒）
        res = self.supervisor.call(
            "train", "start", timeout=300,
            model_dir=self._train_model_dir(),
            queue_dir=self.queue_dir,
            done_dir=os.path.join(self.data_dir, "done"),
            log_path=self.log_path,
            block_size=block_size, steps_per_file=steps,
            optimizer=optimizer)
        return 200, res

    def api_train_stop(self, _body):
        if not self.trainer.running:
            return 200, {"stopping": False, **_DEFAULT_STATUS}
        res = self.supervisor.call("train", "stop", timeout=30)
        return 200, {"stopping": True, **res}

    def api_train_status(self):
        res = self.supervisor.query("train", "status")
        if res is None:
            return dict(_DEFAULT_STATUS)
        return res

    # ---------------- 队列管理 ----------------
    def api_train_queue(self):
        items = []
        for path in sorted(glob.glob(os.path.join(self.queue_dir, "*"))):
            name = os.path.basename(path)
            # 2026-10-23 审计修复（P2）：getsize 此前不设防——文件在 glob
            # 与 getsize 之间被 Trainer 归档/被用户删除时 FileNotFoundError
            # 逃逸 → HTTP 500（训练期间前端轮询该接口概率性触发）
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            # 文件名形如 20260928_153000_muon_ab12cd34.txt，中段为优化器
            opt = ""
            parts = name.split("_")
            if len(parts) >= 3 and parts[2] in ("muon", "adamw"):
                opt = parts[2]
            items.append({"name": name, "size": size, "optimizer": opt})
        return 200, {"items": items}

    def api_train_delete(self, body):
        name = body.get("name")
        if not name:
            raise ValueError("缺少 name")
        path = os.path.join(self.queue_dir, os.path.basename(name))
        if not os.path.exists(path):
            raise FileNotFoundError(f"队列项不存在: {name}")
        if not os.path.isfile(path):
            raise ValueError(f"队列项不是文件: {name}")
        os.unlink(path)
        return 200, {"deleted": name}
