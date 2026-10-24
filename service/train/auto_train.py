# ===========================================================================
# GARGANTUA v1 — auto_train.py：训练队列守护
# ---------------------------------------------------------------------------
# 监控 data/queue/*.txt（按文件名序），逐个文件：text_codec 编码 → 按
# common.make_batch 统一的窗口切分 / forward_train / Muon+AdamW step 逻辑
# 训练 steps_per_file 步 → 检查点落盘 → 文件移入 data/done/。
# 每步把 {"step", "loss", "ts"} 追加到 logs/auto_train.jsonl。
# 空队列 1s 空转轮询；stop() 后当前 step 完成、落盘后退。
#
# 两种用法：
#   1) server.py 内嵌线程：Trainer(get_model=..., save_fn=..., lock=...)
#   2) 独立运行：python auto_train.py --model models/alpha --block-size 16
#
# 2026-10-20 修复（审查结论）：
#   1. main() 改用 common.build_model：非默认 block 数的模型可加载。
#   2. block_size / steps_per_file / optimizer 改为带校验的 property：
#      server 直设属性也不再可能传入奇数 block_size 或 0 步（二者分别会
#      令前向崩溃 / 触发 _loss_tail[-1] IndexError）。
#   3. 训练中模型被热替换（server 加载其他模型）时，按实例标识重建优化
#      器——否则 opts 仍绑定旧模型参数，新模型只算梯度不被更新，落盘的
#      是未训练的新模型。
#   4. stop 早停 / 0 步文件不再触发 _loss_tail[-1] IndexError。
#   5. 窗口切分统一走 common.make_batch，与 train.py 零漂移。
# ===========================================================================
import argparse
import glob
import json
import os
import random
import threading
import time
from collections import deque

import torch
import torch.nn.functional as F

# ---- GARGANTUA 路径引导（目录树重构：把 server/ 与 service/* 放上 path）----
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "start" / "boot.py").is_file():
        _sys.path.insert(0, str(_anc / "start"))
        break
else:
    raise SystemExit("找不到 GARGANTUA 项目根（start/boot.py）")
import boot as _boot
_boot.bootstrap()
import spec
from common import (CKPT_NAME, build_model, make_batch, read_config,
                    save_checkpoint, validate_block_size, validate_steps)
from muon import build_optimizers
from tokens.text_codec import encode as text_encode

_OPTIMIZERS = ("muon", "adamw")


class _NoModelError(RuntimeError):
    """ 无活动模型：文件应保留在队列而非被归档。 """


class Trainer:
    """训练队列守护线程。

    参数：
      get_model      回调 → 当前活动 Gargantua 实例（None 表示未加载）
      save_fn        回调 save_fn(model) → 检查点落盘
      lock           全局模型锁（threading.Lock/RLock），训练 step 与
                     推理互斥；standalone 模式下传一把新锁即可
      queue_dir      队列目录（data/queue）
      done_dir       归档目录（data/done）
      log_path       loss jsonl 路径（logs/auto_train.jsonl）
      block_size     窗口大小（偶数，≥2；user/model 各半）
      steps_per_file 每个队列文件训练的步数（≥1）
    """

    def __init__(self, get_model, save_fn, lock,
                 queue_dir="data/queue", done_dir="data/done",
                 log_path="logs/auto_train.jsonl",
                 block_size=32, steps_per_file=50, seed=0):
        self.get_model = get_model
        self.save_fn = save_fn
        self.lock = lock
        self.queue_dir = queue_dir
        self.done_dir = done_dir
        self.log_path = log_path
        self._block_size = validate_block_size(block_size)
        self._steps_per_file = validate_steps(steps_per_file)
        self._optimizer = spec.OPTIMIZER      # "muon"（Muon+AdamW 分工）或 "adamw"
        random.seed(seed)

        self._stop = threading.Event()
        self._thread = None
        self._running = False
        self._start_lock = threading.Lock()   # 2026-10-23：start() 的
        #                                     check-then-act 需要加锁，
        #                                     并发启动曾可能双开守护线程
        self._step = 0
        self._loss_tail = deque(maxlen=500)
        self._log_tail = deque(maxlen=200)
        os.makedirs(queue_dir, exist_ok=True)
        os.makedirs(done_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # 带校验的属性（server 直设 trainer.block_size 也安全）
    # ------------------------------------------------------------------
    @property
    def block_size(self) -> int:
        return self._block_size

    @block_size.setter
    def block_size(self, v) -> None:
        self._block_size = validate_block_size(v)

    @property
    def steps_per_file(self) -> int:
        return self._steps_per_file

    @steps_per_file.setter
    def steps_per_file(self, v) -> None:
        self._steps_per_file = validate_steps(v)

    @property
    def optimizer(self) -> str:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, v) -> None:
        if v not in _OPTIMIZERS:
            raise ValueError(f"未知优化器: {v}")
        self._optimizer = v

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self):
        """幂等启动；已在跑返回 False。

        2026-10-23 审计修复（P3）：check-then-act 全程加锁。此前
        `self._thread.is_alive()` 判断与 `self._thread = Thread(...)`
        赋值之间无同步，两个并发 start() 可能都看到「线程不在」从而
        双开守护线程（同模型双优化器步进，状态互相踩）。 """
        with self._start_lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name="auto-train")
            self._thread.start()
            return True

    def stop(self):
        """请求优雅停止：当前 step 完成、检查点落盘后退（立即返回）。"""
        self._stop.set()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def running(self):
        return self._running

    def status(self):
        return {
            "running": self._running,
            "step": self._step,
            "block_size": self._block_size,
            "steps_per_file": self._steps_per_file,
            # 2026-10-23 审计修复（P1）：前端训练页可显示当前优化器；
            # 此前该字段不返回，UI 的 muon/adamw 选择是否生效无从判断
            "optimizer": self._optimizer,
            "loss_tail": list(self._loss_tail),
            "log_tail": list(self._log_tail),
        }

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _log(self, msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        self._log_tail.append(line)
        print(line, flush=True)

    def _run(self):
        self._running = True
        self._log("[auto_train] 守护启动")
        try:
            while not self._stop.is_set():
                files = sorted(glob.glob(os.path.join(self.queue_dir, "*.txt")))
                if not files:
                    time.sleep(1)
                    continue
                path = files[0]
                try:
                    self._train_file(path)
                except _NoModelError as e:
                    # 无活动模型：不归档、不消费文件，退避重试（此前文件被
                    # 静默移入 done/，用户以为训练过——审查记录 P2-7）
                    self._log(f"[auto_train] {e}，{os.path.basename(path)} "
                              f"保留在队列")
                    time.sleep(2)
                    continue
                except Exception as e:                  # 单文件失败不崩守护
                    self._log(f"[auto_train] 训练失败 {os.path.basename(path)}: {e}")
                # 无论成败都归档，避免同一文件死循环
                # 2026-10-23 审计修复（P2）：归档的 os.replace 此前在
                # try 之外——文件被并发删除（UI 删队列项）时
                # FileNotFoundError 逃逸出 while，守护线程静默死亡、
                # running 翻 False 且无任何错误日志。现在失败只记日志，
                # 守护继续运转。
                try:
                    os.replace(path, os.path.join(self.done_dir,
                                                  os.path.basename(path)))
                    self._log(f"[auto_train] 归档 {os.path.basename(path)}")
                except OSError as e:
                    self._log(f"[auto_train] 归档失败 {os.path.basename(path)}: {e}")
        finally:
            self._running = False
            self._log("[auto_train] 守护退出")

    def _train_file(self, path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        ids = [t for st in text_encode(text) for t in st["token_ids"]]
        ids.append(spec.STOP_ID)            # 文档结束 → 停止 token，教模型学会终止
        B = self._block_size
        if len(ids) < B + 2:
            self._log(f"[auto_train] {os.path.basename(path)} 仅 {len(ids)} "
                      f"tokens < {B + 2}，跳过训练")
            return
        m = self.get_model()
        if m is None:
            raise _NoModelError("无活动模型")
        self._log(f"[auto_train] {os.path.basename(path)}: {len(ids)} tokens, "
                  f"{self._steps_per_file} steps, block={B}")

        # 与 train.py 相同的优化器分组；模型实例变化时按需重建（防热替换失效）
        opts = build_optimizers(m, self._optimizer, spec.ADAMW_LR,
                                spec.MUON_LR, spec.MUON_MOMENTUM)
        for _ in range(self._steps_per_file):
            if self._stop.is_set():
                break
            m_now = self.get_model()
            if m_now is not m:              # 训练中模型被替换：重建优化器
                m = m_now
                opts = build_optimizers(m, self._optimizer, spec.ADAMW_LR,
                                        spec.MUON_LR, spec.MUON_MOMENTUM)
                self._log("[auto_train] 检测到模型热替换，优化器已重建")
            i = random.randint(0, len(ids) - B - 2)
            x_in, s_in, dec_in, s_dec, y = make_batch(ids, i, B)

            with self.lock:                          # 训练与推理互斥
                m.train()
                logits = m.forward_train(x_in, s_in, dec_in, s_dec)
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]),
                                       y.view(-1))
                for o in opts:
                    o.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                for o in opts:
                    o.step()
                m.fast_weight.reset()                # 样本间截断持久记忆
                m.eval()

            self._step += 1
            # 2026-10-23 审计修复（P3）：补 file 字段——前端 drawLoss 的
            # hover 提示读 r.file 展示「样本: …」，此前后端从不发送，
            # 该提示是死代码
            rec = {"step": self._step, "loss": round(loss.item(), 6),
                   "ts": time.time(),
                   "file": os.path.basename(path)}
            self._loss_tail.append(rec)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        with self.lock:                              # 检查点落盘
            self.save_fn(m)
        if self._loss_tail:
            self._log(f"[auto_train] step={self._step} "
                      f"loss={self._loss_tail[-1]['loss']:.4f} 已落盘")
        else:
            self._log(f"[auto_train] step={self._step} 已落盘（无有效 step）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型目录（含 model.safetensors）")
    ap.add_argument("--data", default="data", help="数据根目录（queue/done 在其下）")
    ap.add_argument("--log", default="logs/auto_train.jsonl")
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--steps-per-file", type=int, default=50)
    args = ap.parse_args()

    m = build_model(args.model)
    m.eval()

    def _save(mod):
        # 保留既有 config.json（block 数等），仅刷新 params——不可写空配置
        cfg = dict(read_config(args.model))
        cfg["params"] = sum(p.numel() for p in mod.parameters())
        save_checkpoint(args.model, mod, cfg)

    lock = threading.RLock()
    trainer = Trainer(
        get_model=lambda: m,
        save_fn=_save,
        lock=lock,
        queue_dir=os.path.join(args.data, "queue"),
        done_dir=os.path.join(args.data, "done"),
        log_path=args.log,
        block_size=args.block_size,
        steps_per_file=args.steps_per_file,
    )
    trainer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[auto_train] 停止中…")
        trainer.stop()
        trainer.join()


if __name__ == "__main__":
    main()
