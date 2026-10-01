# ===========================================================================
# GARGANTUA v1 — auto_train.py：训练队列守护
# ---------------------------------------------------------------------------
# 监控 data/queue/*.txt（按文件名序），逐个文件：text_codec 编码 → 按
# train.py 相同的窗口切分 / forward_train / Muon+AdamW step 逻辑训练
# steps_per_file 步 → 检查点落盘 → 文件移入 data/done/。
# 每步把 {"step", "loss", "ts"} 追加到 logs/auto_train.jsonl。
# 空队列 1s 空转轮询；stop() 后当前 step 完成、落盘后退。
#
# 两种用法：
#   1) server.py 内嵌线程：Trainer(get_model=..., save_fn=..., lock=...)
#   2) 独立运行：python auto_train.py --model models/alpha --block-size 16
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
from safetensors.torch import load_file, save_file

import spec
from model import Gargantua
from muon import build_optimizers
from tokens.text_codec import encode as text_encode

CKPT_NAME = "model.safetensors"


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
      block_size     窗口大小（同 train.py --block-size）
      steps_per_file 每个队列文件训练的步数
    """

    def __init__(self, get_model, save_fn, lock,
                 queue_dir="data/queue", done_dir="data/done",
                 log_path="logs/auto_train.jsonl",
                 block_size=16, steps_per_file=50, seed=0):
        self.get_model = get_model
        self.save_fn = save_fn
        self.lock = lock
        self.queue_dir = queue_dir
        self.done_dir = done_dir
        self.log_path = log_path
        self.block_size = block_size
        self.steps_per_file = steps_per_file
        self.optimizer = spec.OPTIMIZER      # "muon"（Muon+AdamW 分工）或 "adamw"
        random.seed(seed)

        self._stop = threading.Event()
        self._thread = None
        self._running = False
        self._step = 0
        self._loss_tail = deque(maxlen=500)
        self._log_tail = deque(maxlen=200)
        os.makedirs(queue_dir, exist_ok=True)
        os.makedirs(done_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self):
        """幂等启动；已在跑返回 False。"""
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
            "block_size": self.block_size,
            "steps_per_file": self.steps_per_file,
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
                except Exception as e:                  # 单文件失败不崩守护
                    self._log(f"[auto_train] 训练失败 {os.path.basename(path)}: {e}")
                # 无论成败都归档，避免同一文件死循环
                os.replace(path, os.path.join(self.done_dir,
                                              os.path.basename(path)))
                self._log(f"[auto_train] 归档 {os.path.basename(path)}")
        finally:
            self._running = False
            self._log("[auto_train] 守护退出")

    def _train_file(self, path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        ids = [t for st in text_encode(text) for t in st["token_ids"]]
        ids.append(spec.STOP_ID)            # 文档结束 → 停止 token，教模型学会终止
        B, half = self.block_size, self.block_size // 2
        if len(ids) < B + 2:
            self._log(f"[auto_train] {os.path.basename(path)} 仅 {len(ids)} "
                      f"tokens < {B + 2}，跳过训练")
            return
        m = self.get_model()
        if m is None:
            raise RuntimeError("无活动模型")
        self._log(f"[auto_train] {os.path.basename(path)}: {len(ids)} tokens, "
                  f"{self.steps_per_file} steps, block={B}")

        # 与 train.py 相同的优化器分组与窗口切分逻辑
        opts = build_optimizers(m, self.optimizer, spec.ADAMW_LR,
                                spec.MUON_LR, spec.MUON_MOMENTUM)
        for _ in range(self.steps_per_file):
            if self._stop.is_set():
                break
            i = random.randint(0, len(ids) - B - 2)
            x_in = torch.tensor([ids[i:i + half]])
            s_in = torch.arange(half).unsqueeze(0)
            tgt = torch.tensor([ids[i + half:i + B]])
            y = torch.tensor([ids[i + half : i + B]])  
            # decoder 输入 = 目标段左移一位（首 token 用 user 段尾 token 播种）
            dec_in = torch.tensor([[ids[i + half - 1]] + ids[i + half:i + B - 1]])
            s_dec = torch.arange(half, half + half).unsqueeze(0)

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
            rec = {"step": self._step, "loss": round(loss.item(), 6),
                   "ts": time.time()}
            self._loss_tail.append(rec)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

        with self.lock:                              # 检查点落盘
            self.save_fn(m)
        self._log(f"[auto_train] step={self._step} "
                  f"loss={self._loss_tail[-1]['loss']:.4f} 已落盘")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型目录（含 model.safetensors）")
    ap.add_argument("--data", default="data", help="数据根目录（queue/done 在其下）")
    ap.add_argument("--log", default="logs/auto_train.jsonl")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--steps-per-file", type=int, default=50)
    args = ap.parse_args()

    m = Gargantua()
    ckpt = os.path.join(args.model, CKPT_NAME)
    if os.path.exists(ckpt):
        m.load_state_dict(load_file(ckpt))
        print(f"[auto_train] 载入 {ckpt}")
    m.eval()

    lock = threading.RLock()
    trainer = Trainer(
        get_model=lambda: m,
        save_fn=lambda mod: save_file(mod.state_dict(), ckpt),
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
