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
import os
import time
import uuid

import spec
from common import validate_block_size, validate_steps
from server_models import ConflictError

# 2026-10-25（steve）：训练数据改为 .tokens（tok-32，必须带 { } | 时间步
# 结构），不再接受 JSON/JSONL——与 Token 编辑器导出的格式完全一致。
# 一次上传 = 一个 token 流文件（不再拆成多条样本）。
# ===========================================================================
import base64
import glob
import os
import time
import uuid

import spec
from common import validate_block_size, validate_steps
from server_models import ConflictError
from tokens import timestep, tok32

# 2026-10-23 审计修复（P3）：入队此前无任何数量/体积上限。改成字节上限
# （.tokens 4 字节/token）与 token 数上限双保险。
MAX_ENQUEUE_BYTES = 64 * 1024 * 1024     # 单次上传 64MB（≈16M tokens）
MAX_ENQUEUE_TOKENS = MAX_ENQUEUE_BYTES // 4

# 训练服务未启动时 /api/train/status 的默认态（与服务内默认值一致）
_DEFAULT_STATUS = {
    "running": False, "step": 0, "block_size": 32,
    "steps_per_file": 50, "optimizer": spec.OPTIMIZER,
    "loss_tail": [], "log_tail": [],
}


class TrainApi:
    """ /api/train/* 全部端点。 """

    def api_train_enqueue(self, body):
        """训练数据：tokens_b64 上传（.tokens 二进制，tok-32）。

        文件内容必须带 { } | 时间步结构（timestep.unwrap 可解析），
        所有 ID 必须 < VOCAB_SIZE_PADDED。JSON/JSONL 已废弃。"""
        if body.get("json_b64") is not None:
            raise ValueError("训练数据已改为 .tokens 格式（tok-32 二进制，"
                             "必须带 { } | 时间步结构）；不再接受 JSON/JSONL")
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
        if len(data) > MAX_ENQUEUE_BYTES:
            raise ValueError(f"训练数据超限: {len(data)} 字节 > "
                             f"{MAX_ENQUEUE_BYTES}，请拆分后上传")
        ids = tok32.decode_ids(data)            # 长度非 4 倍数 → 400
        if len(ids) > MAX_ENQUEUE_TOKENS:
            raise ValueError(f"token 数超限: {len(ids)} > {MAX_ENQUEUE_TOKENS}")
        try:
            steps = timestep.unwrap(ids)
        except timestep.TimestepError as e:
            raise ValueError(f"时间步结构非法（.tokens 必须带 {{ }} | 结构）: {e}")
        bad = next((t for t in ids if not (0 <= t < spec.VOCAB_SIZE_PADDED)), None)
        if bad is not None:
            raise ValueError(f"token ID {bad} 超出词表"
                             f"（< {spec.VOCAB_SIZE_PADDED}）")
        optimizer = body.get("optimizer", spec.OPTIMIZER)
        if optimizer not in ("muon", "adamw"):
            raise ValueError(f"未知优化器: {optimizer}")
        fn = f"{time.strftime('%Y%m%d_%H%M%S')}_{optimizer}_" \
             f"{uuid.uuid4().hex[:8]}.tokens"
        path = os.path.join(self.queue_dir, fn)
        with open(path, "wb") as f:
            f.write(data)
        return 200, {"queued": [fn], "n_tokens": len(ids),
                     "n_steps": len(steps), "bytes": len(data)}

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
