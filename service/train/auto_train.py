# ===========================================================================
# GARGANTUA v1 — auto_train.py：训练队列守护
# ---------------------------------------------------------------------------
# 监控 data/queue/*.tokens（按文件名序），逐个文件：tokenset 展开成
# 「pos_ids + step_ids」→ 按 train_step 统一的窗口切分 / 前向
# / Muon+AdamW step 逻辑训练 steps_per_file 步 → 检查点落盘 → 文件移入
# data/done/。每步把 {"step", "loss", "ts"} 追加到 logs/auto_train.jsonl。
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
#   5. 窗口切分与前向统一走 train_step（train.py 同一实现），零漂移。
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

# ---- GARGANTUA 路径引导（目录树重构：把 server/ 与 service/* 放上 path）----
import sys as _sys
from pathlib import Path as _Path
for _anc in _Path(__file__).resolve().parents:
    if (_anc / "start" / "boot.py").is_file():
        _sys.path.insert(0, str(_anc / "start"))
        break
else:
    raise SystemExit("找不到 GARGANTUA 项目根（start/boot.py）")
import boot
boot.bootstrap()
import spec
from common import (build_model, model_exists,
                    read_config, save_checkpoint, validate_block_size,
                    validate_steps)
from muon import build_optimizers
from tokens.tokenset import read_doc
from train_step import forward_window, window_index, window_loss
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
        # 异常隔离目录：训练失败的文件移到这里（不归档为「已完成」，
        # 2026-10-27 审计修复——此前异常文件被当作成功样本归档，且文件中
        # 已完成的参数更新可能已进入内存但未落盘，状态与执行分裂）
        failed_dir = os.path.join(os.path.dirname(self.done_dir.rstrip("/")),
                                  "failed")
        try:
            while not self._stop.is_set():
                files = sorted(glob.glob(os.path.join(self.queue_dir, "*.tokens")))
                if not files:
                    time.sleep(1)
                    continue
                path = files[0]
                try:
                    status = self._train_file(path)
                except _NoModelError as e:
                    # 无活动模型：不归档、不消费文件，退避重试（此前文件被
                    # 静默移入 done/，用户以为训练过——审查记录 P2-7）
                    self._log(f"[auto_train] {e}，{os.path.basename(path)} "
                              f"保留在队列")
                    time.sleep(2)
                    continue
                except Exception as e:                  # 单文件失败不崩守护
                    self._log(f"[auto_train] 训练失败 {os.path.basename(path)}: {e}"
                              f"（已完成的参数更新未落盘，文件隔离到 failed/）")
                    os.makedirs(failed_dir, exist_ok=True)
                    try:
                        os.replace(path, os.path.join(
                            failed_dir, os.path.basename(path)))
                    except OSError as e2:
                        self._log(f"[auto_train] 隔离失败 {os.path.basename(path)}: {e2}")
                    continue
                if status == "stopped":
                    # 用户停止：文件保留在队列（不消费、不归档），后续
                    # start 会重新训练它（2026-10-27 审计修复：旧实现停止后
                    # 仍保存并归档，剩余步数永久跳过）
                    self._log(f"[auto_train] 已停止，{os.path.basename(path)} "
                              f"保留在队列")
                    continue
                # status == "ok" / "skipped"：完整执行完毕（或本就无步可训），
                # 归档到 done（os.replace 在 try 内——文件被并发删除时
                # 只记日志，守护继续运转，2026-10-23 审计修复 P2）
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
        """训练一个队列文件。返回状态：
          "ok"      — 全部步数执行完且检查点已成功落盘（可归档）；
          "stopped" — 用户中途停止：参数更新未落盘，文件应保留在队列；
          "skipped" — 文件过短无步可训：可归档（该数据永远无法训练）。
        抛异常 = 训练失败：调用方负责隔离文件。"""
        # read_doc 已追加 <stop> 时间步；多模态文档额外带装配用的 steps
        doc = read_doc(path)
        B = self._block_size
        n_pos = len(doc["pos_ids"])
        if n_pos < B + 2:
            self._log(f"[auto_train] {os.path.basename(path)} 仅 {n_pos} "
                      f"tokens < {B + 2}，跳过训练")
            return "skipped"
        m = self.get_model()
        if m is None:
            raise _NoModelError("无活动模型")
        self._log(f"[auto_train] {os.path.basename(path)}: {n_pos} tokens"
                  f"{'（多模态）' if doc['multimodal'] else ''}, "
                  f"{self._steps_per_file} steps, block={B}")

        # 与 train.py 相同的优化器分组；模型实例变化时按需重建（防热替换失效）
        opts = build_optimizers(m, self._optimizer, spec.ADAMW_LR,
                                spec.MUON_LR, spec.MUON_MOMENTUM)
        stop_hit = False        # 2026-10-27 审计修复：每文档至少一个窗口含 STOP
        for step in range(self._steps_per_file):
            if self._stop.is_set():
                break
            m_now = self.get_model()
            if m_now is not m:              # 训练中模型被替换：重建优化器
                m = m_now
                opts = build_optimizers(m, self._optimizer, spec.ADAMW_LR,
                                        spec.MUON_LR, spec.MUON_MOMENTUM)
                self._log("[auto_train] 检测到模型热替换，优化器已重建")
            # step 0 强制尾部窗口：STOP 在序列末尾，只有 i = n-B 的窗口
            # 能让 y 覆盖最后一个 token，单文件训练因此必有停止监督
            i = window_index(doc, B, random, force_last=(step == 0))
            if i is None:
                continue
            with self.lock:                          # 训练与推理互斥
                m.train()
                logits, y, n_valid = forward_window(m, doc, i, B)
                if n_valid == 0:                     # 整窗都是媒体，无目标
                    continue
                if spec.STOP_ID in y[0].tolist():
                    stop_hit = True
                loss = window_loss(logits, y)
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

        # 2026-10-27 审计修复：区分「完成」与「停止」——停止时不再保存
        # 检查点（部分参数更新直接丢弃，文件保留在队列由下次 start 重训），
        # 只有完整执行完 steps_per_file 步才落盘并归档。
        if self._stop.is_set():
            return "stopped"

        with self.lock:                              # 检查点落盘
            self.save_fn(m)
        if not stop_hit:
            self._log(f"[auto_train] 警告：{os.path.basename(path)} 本轮采样"
                      f"未覆盖 <stop> 目标（步数过少）")
        if self._loss_tail:
            self._log(f"[auto_train] step={self._step} "
                      f"loss={self._loss_tail[-1]['loss']:.4f} 已落盘")
        else:
            self._log(f"[auto_train] step={self._step} 已落盘（无有效 step）")
        return "ok"


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
                # 幂等：已在跑 → 返回现状（与 Trainer.start() 旧语义一致：
                # started=False + 200，不是错误——UI 的重复点击/轮询重试
                # 都走这条路）。但换了模型目录是编程错误，拒绝。
                if (req.get("model_dir")
                        and req["model_dir"] != st.model_dir):
                    return {"id": rid, "ok": False,
                            "error": "训练进行中，不能切换模型（请先 stop）"}
                return {"id": rid, "ok": True,
                        "result": {"started": False,
                                   **st.trainer.status(),
                                   "model_dir": st.model_dir}}
            model_dir = req["model_dir"]
            if not model_exists(model_dir):
                raise ValueError(f"模型目录无检查点: {model_dir}")
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
        return not any(f.endswith(".tokens")
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
            if req.get("op") == "shutdown":
                # 2026-10-27 审计修复：先停训练、等训练线程退出（含检查点
                # 落盘），再应答 shutdown——旧实现先应答、退出时才 join，
                # server 收到应答后立刻 terminate()，关闭恰逢训练 step 时
                # 落盘前清理被 SIGTERM 中断，留下部分参数更新。
                if st.trainer is not None and st.trainer.running:
                    print("[auto_train] shutdown：停止中的训练……",
                          file=sys.stderr)
                    st.trainer.stop()
                    st.trainer.join(timeout=120)
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
