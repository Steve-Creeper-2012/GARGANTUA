# ===========================================================================
# GARGANTUA v1 — server_train.py：训练编排 API（队列 + 启停 + 状态）
# ---------------------------------------------------------------------------
# 从 server.py 拆出。只负责「数据入队 / Trainer 生命周期 / 状态查询」，
# 不碰模型构造、不碰 HTTP 框架。App 以 mixin 方式继承本类：
#
#     class App(TrainApi, CodecApi): ...
#
# self.trainer / self.queue_dir 均由 App.__init__ 注入（Trainer 本体在
# auto_train.py），本模块不持有训练状态。
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

    def api_train_start(self, body):
        if self.mgr.model is None:
            raise ValueError("无活动模型，请先加载模型")
        # 可选：切换到训练页选择的模型（未在训练时才允许；此前该下拉
        # 框采集后不发送，是死 UI——2026-10-20 接线）
        target = body.get("target")
        if target and target != self.mgr.name:
            if self.trainer.running:
                raise ConflictError("训练进行中，请先停止训练再切换模型")
            self.mgr.load(target)
        # 统一校验（非法值 400）：奇数 block_size 令前向崩溃、0 步触发
        # Trainer _loss_tail[-1] IndexError（均实测复现）
        if body.get("block_size") is not None:
            self.trainer.block_size = validate_block_size(body["block_size"])
        if body.get("steps") is not None:
            self.trainer.steps_per_file = validate_steps(body["steps"])
        if body.get("optimizer"):
            opt = body["optimizer"]
            if opt not in ("muon", "adamw"):
                raise ValueError(f"未知优化器: {opt}")
            self.trainer.optimizer = opt
        started = self.trainer.start()
        return 200, {"started": started, **self.trainer.status()}

    def api_train_stop(self, _body):
        self.trainer.stop()
        return 200, {"stopping": True, **self.trainer.status()}

    def api_train_status(self):
        return self.trainer.status()

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
