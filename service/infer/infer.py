# ===========================================================================
# GARGANTUA v1 — infer.py：文本推理（自回归，共享 KV 逐步回写）
# ---------------------------------------------------------------------------
# 两种用法：
#   1) CLI（人肉调试）：
#        python infer.py --model models/alpha --prompt "你好" \
#            --max-new-tokens 32 --temperature 0.8 --top-p 0.95
#   2) --serve（server 按需拉起的常驻模式，2026-10-24）：
#        python infer.py --serve [--idle-timeout 300]
#      stdin/stdout 走 JSON-lines 协议（见 server_services.py 文件头）：
#        请求 {"id","op","model_dir","prompt","max_new","temperature","top_p"}
#        响应 {"id","ok",result|error}
#      op：load / infer / status / shutdown。模型按请求里的 model_dir 惰性
#      加载（换了模型才重载），空闲超时自行退出——推理是重活，模型不能
#      常驻 server 进程（省内存，见 README「服务按需启动」）。
#
# 2026-10-20 修复（审查结论）：
#   1. main() 改用 common.build_model：按模型目录 config.json 的 block 数
#      构造，非默认 block 数的模型不再 load_state_dict 崩溃。
#   2. fast_weight.reset() 移到 encode 之前：原实现在 encode 之后 reset，
#      把 encoder 阶段写入的 fast weight 状态清掉，违反 spec「enc/dec
#      共享 bank、状态跨调用推进」，且与 train.py 的 forward_train 不一致。
#   3. KV 回写结构与训练逐位置对齐（review-pipeline 探针实测）：
#      训练 KV 流 = enc(prompt) + [emb(seed)@n, emb(t0)@n+1, …]，即每个
#      decoder 输入的 embedding 放在【它要预测的那个 token 的步】上。
#      本实现 therefore 把「当前输入 emb」拼进视图（s_dec=cur_step），
#      decode 后把该 emb 持久化进 KV；不再单独写回生成 token 的 emb
#      （旧实现因此让种子的 MODEL 嵌入从未进入 KV，且每个生成 token 比
#      训练早一步——train/infer 分布偏差，已回归测试固定）。
# ===========================================================================
import argparse
import json
import os
import queue
import sys
import threading
import time
import uuid
from pathlib import Path

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
import boot
boot.bootstrap()
import spec
from common import CKPT_NAME, MAX_NEW_TOKENS, build_model
from tokens import tok32
from tokens.text_codec import encode as text_encode, decode as text_decode
from tokens.vocab import set_model_vocab


# ===========================================================================
# 推理侧 token id 存档（2026-10-25，steve：KV 可丢，id 必须落盘）
# ---------------------------------------------------------------------------
# 每次推理一个 .tokens 文件（tok-32，每个 token 包成一个 { id } 时间步），
# prompt 与生成 token 边产边写并 flush——KV 超出上下文上限只丢 KV 向量，
# 原始 id 已经在磁盘上；进程崩溃 / 空闲退出也不丢。
# 目录：server 传入 logs/infer_archive（随 log_path 走，测试用临时目录）；
# CLI 缺省用 <项目根>/logs/infer_archive。
# 存档失败绝不阻断推理：打印到 stderr 并自动降级为「不存档」。
# ===========================================================================
class IdArchive:
    """一次推理的 token id 存档（.tokens + .json 元数据）。 """

    def __init__(self, model_dir: str = "", root: str | None = None):
        self.model_dir = model_dir or ""
        self.root = Path(root) if root else (
            Path(boot.project_root()) / "logs" / "infer_archive")
        self.path: Path | None = None
        self._fh = None
        self._disabled = False
        self.n_tokens = 0

    def _ensure(self) -> bool:
        if self._fh is not None:
            return True
        if self._disabled:
            return False
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            name = (f"{time.strftime('%Y%m%d_%H%M%S')}_"
                    f"{uuid.uuid4().hex[:8]}.tokens")
            self.path = self.root / name
            self._fh = open(self.path, "wb")
            return True
        except OSError as e:
            print(f"[infer] id 存档不可用（不影响推理）: {e}", file=sys.stderr)
            self._disabled = True
            return False

    def add(self, ids) -> None:
        """把一批 token id 以「一 token 一步」写入存档并 flush。 """
        ids = [int(i) for i in ids]
        if not ids or not self._ensure():
            return
        try:
            flat = []
            for i in ids:
                flat.extend((spec.TS_OPEN_ID, i, spec.TS_CLOSE_ID))
            self._fh.write(tok32.encode_ids(flat))
            self._fh.flush()
            self.n_tokens += len(ids)
        except OSError as e:
            print(f"[infer] id 存档写入失败（不影响推理）: {e}", file=sys.stderr)
            self._disabled = True

    def close(self, n_prompt: int = 0, n_generated: int = 0,
              ctx_total: int | None = None) -> None:
        if self._fh is None:
            return
        try:
            self._fh.close()
        except OSError:
            pass
        self._fh = None
        if self.path is None:
            return
        # 元数据：说明这次推理有多少 token、KV 窗口多大、多少条已不在 KV
        meta = {
            "model": os.path.basename(os.path.normpath(self.model_dir))
            if self.model_dir else "",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_prompt_tokens": int(n_prompt),
            "n_generated_tokens": int(n_generated),
            "n_tokens": int(self.n_tokens),
            "ctx_total": None if ctx_total is None else int(ctx_total),
            "n_evicted": (max(0, self.n_tokens - int(ctx_total))
                          if ctx_total is not None else 0),
        }
        try:
            side = self.path.with_suffix(".json")
            side.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        except OSError as e:
            print(f"[infer] 存档元数据写入失败: {e}", file=sys.stderr)

    def rel_path(self) -> str | None:
        """存档路径（相对项目根；给 API / 日志用）。 """
        if self.path is None:
            return None
        try:
            return os.path.relpath(self.path, boot.project_root())
        except ValueError:              # 跨盘（Windows）时退回绝对路径
            return str(self.path)


@torch.no_grad()
def generate(m, prompt: str, max_new: int = 32,
             temperature: float = 0.8, top_p: float = 0.95,
             archive_dir: str | None = None, model_dir: str | None = None):
    """ 编码 prompt → encoder → KV；decoder 逐步生成并回写共享 KV。

    返回 (生成的时间步序列, id 存档路径)。时间步序列是 List[Step]，
    text codec 可直接 decode；存档路径是本次推理的 .tokens 文件
    （相对项目根），存档不可用时为 None。
    """
    max_new = int(max_new)
    if not (1 <= max_new <= MAX_NEW_TOKENS):
        raise ValueError(f"max_new 必须在 [1, {MAX_NEW_TOKENS}]: {max_new}")
    if not (temperature > 0):
        raise ValueError(f"temperature 必须 > 0: {temperature}")
    if not (0 < top_p <= 1):
        raise ValueError(f"top_p 必须在 (0, 1]: {top_p}")

    steps = text_encode(prompt)
    ids = [t for st in steps for t in st["token_ids"]]
    if not ids:
        raise ValueError("prompt 编码为空")

    # 2026-10-25：prompt 与生成 token 边产边落盘（.tokens），KV 溢出只丢
    # KV 向量、不丢 id；存档失败自动降级，不影响推理本身。
    archive = IdArchive(model_dir or "", archive_dir)
    out_ids: list = []
    try:
        archive.add(ids)

        # 新生成序列：fast weight 状态从零开始；encode 写入的状态由 decode 继续推进
        m.fast_weight.reset()
        x_in = torch.tensor([ids])
        s_in = torch.arange(len(ids)).unsqueeze(0)
        emb_in = m.embed_text(x_in, torch.full_like(x_in, spec.TYPE_USER))
        enc_out = m.encode(emb_in, s_in)

        # 共享 KV 初始 = encoder 输出；decoder 每步把当前输入 emb 拼进视图，
        # decode 后持久化进 KV（与 forward_train 的 KV 流逐位置同构）。
        # 超出上下文上限时 build_views 只按整步丢弃 KV 视图，原始 id 已在
        # archive 落盘（丢 KV 不丢 id）。
        kv_vecs = enc_out.detach()
        kv_steps = s_in
        cur = ids[-1]                            # 用 prompt 尾 token 播种
        cur_step = len(ids)                      # 即将预测的 token 的时间步
        for _ in range(max_new):
            # 输入 token 的 embedding 放在其预测步上（s_dec = cur_step），
            # 与训练 make_batch 的 s_dec=arange(half, …) 约定一致
            s_dec = torch.tensor([[cur_step]])
            dec_in = torch.tensor([[cur]])
            emb = m.embed_text(dec_in, torch.full_like(dec_in, spec.TYPE_MODEL))
            kv_hca, kv_csa = m.build_views(
                torch.cat([kv_vecs, emb], 1),
                torch.cat([kv_steps, s_dec], 1))
            dec_out = m.decode(emb, s_dec, kv_hca, kv_csa)
            logits = m.logits(dec_out)[0, -1].float() / max(temperature, 1e-5)
            # top-p 核采样
            probs = F.softmax(logits, dim=-1)
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_p, dim=0)
            keep = cum - sorted_p < top_p
            sorted_p = sorted_p * keep
            sorted_p = sorted_p / sorted_p.sum().clamp(min=1e-9)
            nxt = sorted_i[torch.multinomial(sorted_p, 1)].item()
            if nxt == spec.STOP_ID:             # 模型输出停止 token → 立即停止
                break
            out_ids.append(nxt)
            archive.add([nxt])
            # 持久化【当前输入】的 emb（训练 KV 流同构）；下一轮输入即 nxt
            kv_vecs = torch.cat([kv_vecs, emb.detach()], 1)
            kv_steps = torch.cat([kv_steps, s_dec], 1)
            cur, cur_step = nxt, cur_step + 1
        out_steps = [{"token_ids": [i], "type": spec.TYPE_MODEL,
                      "modality": "text", "meta": {}} for i in out_ids]
        return out_steps, archive.rel_path()
    finally:
        archive.close(
            n_prompt=len(ids), n_generated=len(out_ids),
            ctx_total=(getattr(m, "ctx_full", spec.CTX_FULL)
                       + getattr(m, "ctx_compress", spec.CTX_COMPRESS)))


def _load_model(model_dir: str):
    """加载模型 + 应用该模型的词表（vocab.txt 覆盖，2026-10-24）。

    缺权重文件必须显式报错：build_model 静默用随机初始化会让手误的
    路径产出「看似正常」的乱码输出（审查记录 P2）。"""
    if not os.path.exists(os.path.join(model_dir, CKPT_NAME)):
        raise ValueError(f"模型目录无 {CKPT_NAME}: {model_dir}")
    m = build_model(model_dir)
    m.eval()
    set_model_vocab(model_dir)
    return m


# ---------------------------------------------------------------------------
# --serve：JSON-lines 常驻模式（server_services 协议）
# ---------------------------------------------------------------------------
def _emit(obj: dict) -> None:
    """stdout 只准写 JSON 行——人类可读日志一律 stderr（协议硬约束）。"""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _serve_handle(req: dict, state: dict) -> dict:
    rid = req.get("id")
    op = req.get("op")
    try:
        if op == "load":
            model_dir = req["model_dir"]
            state["model"] = None            # 先放掉旧模型再加载新的
            state["dir"] = None
            m = _load_model(model_dir)
            state["model"], state["dir"] = m, model_dir
            return {"id": rid, "ok": True,
                    "result": {"loaded": model_dir,
                               "params": sum(p.numel()
                                             for p in m.parameters())}}
        if op == "infer":
            model_dir = req["model_dir"]
            if state["model"] is None or state["dir"] != model_dir:
                m = _load_model(model_dir)
                state["model"], state["dir"] = m, model_dir
            steps, archive = generate(state["model"], req["prompt"],
                                      req.get("max_new", 32),
                                      req.get("temperature", 0.8),
                                      req.get("top_p", 0.95),
                                      archive_dir=req.get("archive_dir"),
                                      model_dir=model_dir)
            return {"id": rid, "ok": True,
                    "result": {"text": text_decode(steps),
                               "token_ids": [s["token_ids"][0]
                                             for s in steps],
                               "archive": archive}}
        if op == "status":
            return {"id": rid, "ok": True,
                    "result": {"loaded": state["dir"]}}
        if op == "shutdown":
            return {"id": rid, "ok": True, "result": {"bye": True}}
        return {"id": rid, "ok": False, "error": f"未知 op: {op!r}"}
    except Exception as e:
        return {"id": rid, "ok": False, "error": str(e)}


def serve(idle_timeout: int = 300) -> int:
    """常驻模式：读线程收请求 → 主线程串行处理 → 空闲超时自行退出。"""
    state = {"model": None, "dir": None}
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
            q.put(("eof", None))              # stdin 关 = server 没了

    threading.Thread(target=_reader, daemon=True,
                     name="infer-reader").start()
    print(f"[infer] serve 模式就绪（idle-timeout={idle_timeout}s）",
          file=sys.stderr)

    last = time.time()
    while True:
        remaining = idle_timeout - (time.time() - last)
        if remaining <= 0:
            print("[infer] 空闲超时，退出（模型内存已释放）", file=sys.stderr)
            return 0
        try:
            kind, payload = q.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue
        if kind == "eof":
            print("[infer] stdin EOF，退出", file=sys.stderr)
            return 0
        last = time.time()
        try:
            req = json.loads(payload)
            if not isinstance(req, dict):
                raise ValueError("请求必须是 JSON 对象")
        except Exception as e:
            _emit({"id": None, "ok": False, "error": f"请求解析失败: {e}"})
            continue
        resp = _serve_handle(req, state)
        _emit(resp)
        if req.get("op") == "shutdown":
            print("[infer] 收到 shutdown，退出", file=sys.stderr)
            return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="模型目录（含 model.safetensors）")
    ap.add_argument("--prompt")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--serve", action="store_true",
                    help="JSON-lines 常驻模式（server 按需拉起）")
    ap.add_argument("--idle-timeout", type=int, default=300,
                    help="serve 模式空闲多少秒后自行退出（默认 300）")
    ap.add_argument("--archive-dir", default=None,
                    help="id 存档目录（默认 <项目根>/logs/infer_archive）")
    args = ap.parse_args()

    if args.serve:
        return serve(idle_timeout=args.idle_timeout)

    if not args.model or not args.prompt:
        ap.error("CLI 模式需要 --model 与 --prompt（或用 --serve）")
    m = _load_model(args.model)
    steps, archive = generate(m, args.prompt, args.max_new_tokens,
                              args.temperature, args.top_p,
                              archive_dir=args.archive_dir,
                              model_dir=args.model)
    print(text_decode(steps))
    if archive:
        print(f"[infer] id 存档: {archive}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
