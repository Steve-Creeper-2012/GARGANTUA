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
#   1) 独立运行：python auto_train.py --model models/alpha --block-size 16
#   2) --serve（server 按需拉起的常驻模式，2026-10-24）：
#        python auto_train.py --serve [--idle-timeout 300]
#      stdin/stdout 走 JSON-lines 协议（见 server_services.py 文件头）：
#        请求 {"id","op","model_dir","queue_dir","done_dir","log_path",
#              "block_size","steps_per_file","optimizer"}
#        响应 {"id","ok",result|error}
#      op：start / stop / status / shutdown。
#      进程退出条件：shutdown / stdin EOF（server 死了）/ 空闲超时。
#      「空闲」= 一段时间没有协议请求，且（未在训练 或 队列已空）——
#      训练进行中（队列有文件）绝不退出，那是用户明确要的重活；
#      训练停止后模型不再常驻，内存归还 OS（server 本体零权重）。
#
# 2026-10-20 修复（审查结论）：
#   1. main() 改用 common.build_model：非默认 block 数的模型可加载。
#   2. block_size / steps_per_file / optimizer 改为带校验的 property：
#      server 直设属性也不再可能传入奇数 block_size 或 0 步（二者分别会
#      令前向崩溃 / 触发 _loss_tail[-1] IndexError）。
#   3. 训练中模型被热替换（server 加载其他模型）时，按实例标识重建优化
#      器——否则 opts 仍绑定旧模型参数，新模型只算梯度不被更新，落盘的
#      是未训练的新模型。（serve 模式下模型热替换被 server 409 禁止，
#      该防御保留。）
#   4. stop 早停 / 0 步文件不再触发 _loss_tail[-1] IndexError。
#   5. 窗口切分统一走 common.make_batch，与 train.py 零漂移。
# ===========================================================================
import argparse
import glob
import json
import os
import queue
import random
import sys
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
from tokens.vocab import set_model_vocab

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
        # stderr：stdout 是 --serve 模式的 JSON-lines 协议通道，
        # 人肉日志混进去会污染响应流（2026-10-24 实测踩过）
        print(line, file=sys.stderr, flush=True)

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


def _prefill_from_log(trainer: Trainer) -> None:
    """serve 模式启动时从 loss jsonl 回填 step 计数与 loss 尾部
    （2026-10-24）：服务进程按需退出后，前端 loss 曲线不应清零——
    日志是持久的，回填即延续。log_tail 不持久化，留空。"""
    try:
        with open(trainer.log_path, encoding="utf-8") as f:
            lines = f.readlines()[-500:]
    except OSError:
        return
    last_step = 0
    for line in lines:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        trainer._loss_tail.append(rec)
        if isinstance(rec.get("step"), int):
            last_step = max(last_step, rec["step"])
    trainer._step = last_step


# ---------------------------------------------------------------------------
# --serve：JSON-lines 常驻模式（server_services 协议）
# ---------------------------------------------------------------------------
def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class _ServeState:
    def __init__(self):
        self.trainer = None
        self.model_dir = None


def _serve_handle(req: dict, st: _ServeState) -> dict:
    rid = req.get("id")
    op = req.get("op")
    try:
        if op == "start":
            if st.trainer is not None and st.trainer.running:
                return {"id": rid, "ok": False,
                        "error": "训练进行中，请先 stop 再 start"}
            model_dir = req["model_dir"]
            if not os.path.exists(os.path.join(model_dir, CKPT_NAME)):
                raise ValueError(f"模型目录无 {CKPT_NAME}: {model_dir}")
            m = build_model(model_dir)
            m.eval()
            set_model_vocab(model_dir)
            lock = threading.RLock()

            def _save(mod):
                # 保留既有 config.json（block 数等），仅刷新 params
                cfg = dict(read_config(model_dir))
                cfg["params"] = sum(p.numel() for p in mod.parameters())
                save_checkpoint(model_dir, mod, cfg)

            trainer = Trainer(
                get_model=lambda: m,
                save_fn=_save,
                lock=lock,
                queue_dir=req["queue_dir"],
                done_dir=req["done_dir"],
                log_path=req["log_path"],
                block_size=(req["block_size"]
                            if req.get("block_size") is not None else 32),
                steps_per_file=(req["steps_per_file"]
                                if req.get("steps_per_file") is not None
                                else 50),
            )
            if req.get("optimizer"):
                trainer.optimizer = req["optimizer"]   # 属性带校验
            _prefill_from_log(trainer)
            started = trainer.start()
            st.trainer, st.model_dir = trainer, model_dir
            return {"id": rid, "ok": True,
                    "result": {"started": started, **trainer.status(),
                               "model_dir": model_dir}}
        if op == "stop":
            if st.trainer is None:
                return {"id": rid, "ok": True,
                        "result": {"stopping": False, "running": False}}
            st.trainer.stop()
            return {"id": rid, "ok": True,
                    "result": {"stopping": True, **st.trainer.status()}}
        if op == "status":
            if st.trainer is None:
                return {"id": rid, "ok": True,
                        "result": {"running": False, "model_dir": None}}
            return {"id": rid, "ok": True,
                    "result": {**st.trainer.status(),
                               "model_dir": st.model_dir}}
        if op == "shutdown":
            return {"id": rid, "ok": True, "result": {"bye": True}}
        return {"id": rid, "ok": False, "error": f"未知 op: {op!r}"}
    except Exception as e:
        return {"id": rid, "ok": False, "error": str(e)}


def _queue_empty(queue_dir: str) -> bool:
    try:
        return not any(f.endswith(".txt")
                       for f in os.listdir(queue_dir))
    except OSError:
        return True


def serve(idle_timeout: int = 300) -> int:
    """常驻模式。退出条件：shutdown / stdin EOF / 空闲超时。
    空闲 = 无协议请求超过 idle_timeout，且（未在训练 或 队列已空）。"""
    st = _ServeState()
    q: "queue.Queue" = queue.Queue()

    def _reader():
        try:
            for line in sys.stdin:
                line = line.strip()
                if line:
                    q.put(("line", line))
        except Exception:
            pass
        finally:
            q.put(("eof", None))

    threading.Thread(target=_reader, daemon=True,
                     name="train-reader").start()
    print(f"[auto_train] serve 模式就绪（idle-timeout={idle_timeout}s）",
          file=sys.stderr)

    last_req = time.time()
    while True:
        try:
            kind, payload = q.get(timeout=1.0)
        except queue.Empty:
            kind = None
        if kind == "eof":
            print("[auto_train] stdin EOF，退出", file=sys.stderr)
            break
        if kind == "line":
            last_req = time.time()
            try:
                req = json.loads(payload)
                if not isinstance(req, dict):
                    raise ValueError("请求必须是 JSON 对象")
            except Exception as e:
                _emit({"id": None, "ok": False,
                       "error": f"请求解析失败: {e}"})
                continue
            resp = _serve_handle(req, st)
            _emit(resp)
            if req.get("op") == "shutdown":
                print("[auto_train] 收到 shutdown，退出", file=sys.stderr)
                break
            continue

        # 空闲判定：没请求超过 idle_timeout，且没有「正在进行」的训练
        # （队列有文件 = 用户在等训练结果，绝不退）
        if time.time() - last_req >= idle_timeout:
            training_live = (st.trainer is not None and st.trainer.running
                             and not _queue_empty(st.trainer.queue_dir))
            if not training_live:
                print("[auto_train] 空闲超时，退出（模型内存已释放）",
                      file=sys.stderr)
                break

    # 收尾：训练中的话优雅停止（当前 step 完成 + 落盘）
    if st.trainer is not None and st.trainer.running:
        print("[auto_train] 停止中的训练……", file=sys.stderr)
        st.trainer.stop()
        st.trainer.join(timeout=120)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="模型目录（含 model.safetensors）")
    ap.add_argument("--data", default="data", help="数据根目录（queue/done 在其下）")
    ap.add_argument("--log", default="logs/auto_train.jsonl")
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--steps-per-file", type=int, default=50)
    ap.add_argument("--serve", action="store_true",
                    help="JSON-lines 常驻模式（server 按需拉起）")
    ap.add_argument("--idle-timeout", type=int, default=300,
                    help="serve 模式空闲多少秒后自行退出（默认 300）")
    args = ap.parse_args()

    if args.serve:
        return serve(idle_timeout=args.idle_timeout)

    if not args.model:
        ap.error("CLI 模式需要 --model（或用 --serve）")
    m = build_model(args.model)
    m.eval()
    set_model_vocab(args.model)

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
    sys.exit(main())
