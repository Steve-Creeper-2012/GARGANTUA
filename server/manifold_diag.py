# ===========================================================================
# GARGANTUA — manifold_diag.py：报告「阶段 0 零风险谱诊断」的只读观测实现
# ---------------------------------------------------------------------------
# 对应 docs/analysis/manifold_muon_mhc_feasibility.md §3.3 阶段 0：
#   在现有 `--optimizer muon` 训练里每 N 步（默认 50，step 1 也采）记录
#     · MLA 六个投影 + FFN 投影的 σ_max / σ_min / diag(WᵀW) /
#       ‖Off(WᵀW)‖_F / ‖W‖_F（对齐 arXiv:2606.13276 §4.1）
#     · 裁剪前全局 grad norm（clip_grad_norm_(1.0) 会掩盖退化，§3.4.9）
#     · MLA attention logits 的 absmax / softmax 熵 / J_softmax 的 ‖J‖_F
#     · KDA 持久状态 ρ(S) / ‖S‖_F / λ_max(C)（§3.4.4）
#     · Muon NS5 的实测 wall-clock 占比（§3.2.3）
#   日志落 data/tmp/manifold_diag.jsonl（只追加，不覆盖）。
#
# 零侵入硬约束（阶段 0 的立身之本）：
#   * 默认关闭：--diag 未开时 train.py 连本模块都不 import。
#   * 零参数 / 零 buffer / 零 nn.Module 注册：ModelDiagnostics 不向 model
#     注册任何 Module/Parameter/buffer；attach() 只把 MLAAttention 的普通
#     属性 diag_probe 指向一个纯读回调、把 Muon 的普通属性 profile 置 True。
#     这两者都不进 model.state_dict()，故 attach 前后 state_dict 键集合与
#     值逐位一致。
#   * 不改数值：探针收到的是 attn.detach()，只读、不回写，forward 数值不变；
#     unattach 时 diag_probe 保持 None，连探针回调都不执行。
#   * 绝不崩训练：observe 整体 try/except，任何诊断异常只往 stderr 打一行。
#
# 本模块不依赖 spec.py 的任何新增常量（诊断参数全部自带），只 import paths
# 取 data/tmp；并自带 boot 引导头，可从任意位置直接 import。
# ===========================================================================
"""阶段 0「零风险谱诊断」只读观测：默认关闭、零参数、零 buffer、不改数值。"""
from __future__ import annotations

# ---- GARGANTUA 路径引导（照抄 boot.py BOOT_HEADER / test_somn.py 头部）----
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

import json
import math
import os
import sys
import time

import torch

import paths

# ---- 本模块自带诊断参数（不依赖 spec 新增常量）----
_DEFAULT_EVERY = 50            # 采样间隔
_DEFAULT_MAX_DIM = 2048        # Gram 维超过该值不做 eigvalsh（只留基础 fro）
_MAX_MATRICES = 24             # 每次 observe 最多采集的矩阵数（默认限速；0 = 不限）
_TOPK = 10                     # 批复要求：每矩阵记录 top-10 奇异值
_COND_REL_EPS = 1e-8           # σ_min 相对 σ_max 的下限，避免÷0 造出无意义的 inf
_MLA_WEIGHTS = ("w_dq", "w_uq", "w_dkv", "w_uk", "w_uv", "w_o")
_FFN_WEIGHTS = ("ffn_up", "ffn_down")
_DEFAULT_LOG_NAME = "manifold_diag.jsonl"


def _fro(t: torch.Tensor) -> float:
    """ Frobenius 范数 sqrt(Σ t²)，逐元素，避免 torch.linalg.norm 的维数歧义。 """
    return float(t.pow(2).sum().sqrt())


def _is_nan(x) -> bool:
    return isinstance(x, float) and (math.isnan(x) or math.isinf(x))


def _json_default(o):
    """ json.dumps 遇到非法类型的兜底（Tensor / numpy 标量等）。 """
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    try:
        return float(o)
    except Exception:
        return str(o)


# ---------------------------------------------------------------------------
# 1) 谱统计
# ---------------------------------------------------------------------------
def spectrum_stats(W: torch.Tensor, max_dim: int = _DEFAULT_MAX_DIM) -> dict:
    """ 单个 2D 权重矩阵的只读谱统计（全程 no_grad / detach / python float）。

    wide→tall 自动转置：若 W.shape[0] < W.shape[1] 则转置，使 Gram 落在
    min(in,out) 维。sigma_* 由 eigvalsh(Gram) 取 sqrt(clamp≥0)，不做 SVD。
    批复要求补齐的三个量：
      sigma_max   —— 谱范数；
      sigma_top10 —— 最大的 10 个奇异值（降序）；
      cond        —— 条件数 σ_max/σ_min，σ_min 过小则夹到相对下限并
                     置 cond_clamped=True（fp32 Gram 会平方条件数，小奇异值
                     精度有限，cond 只用于「是否随时间膨胀」的趋势判定）。
    Gram 维 > max_dim 时只返回基础 fro 值 + skipped 说明；NaN/Inf 输入返回
    skipped="non-finite"。任何异常都不抛出。
    """
    out = {
        "shape": list(W.shape) if isinstance(W, torch.Tensor) else [],
        "transposed": False,
        "gram_dim": 0,
        "sigma_max": None,
        "sigma_min": None,
        "sigma_top10": [],
        "cond": None,
        "cond_clamped": None,
        "gram_diag_mean": None,
        "gram_diag_max": None,
        "offgram_fro": None,
        "fro": None,
        "method": "gram_eigvalsh_fp32",
        "skipped": None,
    }
    try:
        with torch.no_grad():
            X = W.detach().float()
            # 基础 fro：即使后续被 skip 也给出（O(numel)，很便宜）
            out["fro"] = _fro(X) if X.numel() else 0.0
            if X.dim() != 2:
                out["skipped"] = f"expected 2D tensor, got dim={X.dim()}"
                return out
            if X.numel() == 0:
                out["skipped"] = "empty"
                return out
            if not bool(torch.isfinite(X).all()):
                out["skipped"] = "non-finite"
                out["fro"] = float("nan")
                return out

            m, n = X.shape
            transposed = bool(m < n)
            Y = X.t() if transposed else X          # tall: [max(m,n), min(m,n)]
            out["transposed"] = transposed
            gd = int(min(m, n))
            out["gram_dim"] = gd
            if gd > max_dim:
                out["skipped"] = f"gram_dim {gd} > max_dim {max_dim}; skipped eigvalsh"
                return out

            gram = Y.t() @ Y                        # [gd, gd]，特征值 = σ²(W)
            diag = torch.diagonal(gram)
            off = gram - torch.diag(diag)
            out["offgram_fro"] = _fro(off)
            out["gram_diag_mean"] = float(diag.mean())
            out["gram_diag_max"] = float(diag.max())
            ev = torch.linalg.eigvalsh(gram).clamp_min(0.0)
            s = torch.sqrt(ev)                    # 升序
            out["sigma_max"] = float(s[-1]) if s.numel() else 0.0
            out["sigma_min"] = float(s[0]) if s.numel() else 0.0
            out["sigma_top10"] = [float(v) for v in torch.flip(s, [0])[:_TOPK]]
            # 条件数：σ_min 相对 σ_max 低于下限时按秩亏处理，避免 inf/nan
            floor = max(1e-12, _COND_REL_EPS * out["sigma_max"])
            smin = out["sigma_min"]
            out["cond_clamped"] = bool(smin <= floor)
            out["cond"] = out["sigma_max"] / (smin if smin > floor else floor)
    except Exception as e:                          # 谱分解偶发失败也不抛
        out["skipped"] = f"computation failed: {e}"
    return out


# ---------------------------------------------------------------------------
# 2) 裁剪前全局 grad norm（纯读，不修改梯度）
# ---------------------------------------------------------------------------
def grad_norm_before_clip(params) -> float:
    """ 与 torch.nn.utils.clip_grad_norm_ 的裁剪前返回值同定义：
    sqrt(Σ_p ‖p.grad‖₂²)。跳过 grad=None，绝不修改任何梯度。 """
    total = 0.0
    with torch.no_grad():
        for p in params:
            g = getattr(p, "grad", None)
            if g is None:
                continue
            total += float(g.detach().norm(2).item()) ** 2
    return total ** 0.5


# ---------------------------------------------------------------------------
# 3) attention logits / softmax 熵 / J_softmax
# ---------------------------------------------------------------------------
def attention_stats(logits: torch.Tensor, with_jacobian: bool = False) -> dict:
    """ 只在有限项上做 softmax（因果 mask 的 -inf / +inf / NaN 一律过滤）。

    返回 logits_absmax / entropy_mean / entropy_min / finite_frac /
    jacobian_fro_mean（默认 None）。全 -inf 行跳过；整个张量无有限项时返回
    全 0/None 且 finite_frac=0.0。with_jacobian=True 时按
    J = diag(p) - p pᵀ 计算 ‖J‖_F 的均值（闭式：Σp² + (Σp²)² - 2Σp³，
    不物化 K×K 矩阵，内存有界）。 """
    out = {
        "logits_absmax": 0.0,
        "entropy_mean": 0.0,
        "entropy_min": 0.0,
        "finite_frac": 0.0,
        "jacobian_fro_mean": None,
    }
    try:
        with torch.no_grad():
            x = logits.detach().float()
            if x.dim() == 0 or x.numel() == 0:
                return out
            K = x.shape[-1]
            if K == 0:
                return out
            x2 = x.reshape(-1, K)
            finite = torch.isfinite(x2)
            out["finite_frac"] = float(finite.float().mean())
            if bool(finite.any()):
                out["logits_absmax"] = float(x2[finite].abs().max())

            # 非有限项置 -inf；全 -inf 行的 softmax 会得 NaN，用 valid 掩码跳过
            x_masked = x2.masked_fill(~finite, float("-inf"))
            p = torch.softmax(x_masked, dim=-1)
            valid = finite.any(dim=-1)              # [N]：至少有一个有限项
            p = torch.where(valid.unsqueeze(-1), p, torch.zeros_like(p))

            logp = torch.log(p.clamp_min(1e-12))
            ent = -(p * logp).sum(dim=-1)           # [N] 每行 softmax 熵
            ent = ent[valid]
            if ent.numel() > 0:
                out["entropy_mean"] = float(ent.mean())
                out["entropy_min"] = float(ent.min())

            if with_jacobian:
                # J = diag(p) - p pᵀ  ⇒  ‖J‖_F² = Σp² + (Σp²)² - 2Σp³
                s2 = (p * p).sum(dim=-1)
                s3 = (p * p * p).sum(dim=-1)
                jf = (s2 + s2 * s2 - 2.0 * s3).clamp_min(0.0).sqrt()[valid]
                out["jacobian_fro_mean"] = (float(jf.mean())
                                            if jf.numel() > 0 else None)
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# 4) KDA 持久状态
# ---------------------------------------------------------------------------
def kda_state_stats(S) -> dict:
    """ S: [B,H,dk,dv]；返回 fro / max_abs / spectral_radius / blocks / shape。
    spectral_radius = 所有 [dk,dv] 块最大奇异值上界（svdvals）的最大值。
    S=None 返回 {"present": False}；异常返回带 nan 字段的 present=True。 """
    if S is None:
        return {"present": False}
    try:
        with torch.no_grad():
            x = S.detach().float()
            shape = list(S.shape)
            if x.dim() != 4:
                return {"present": True, "shape": shape, "fro": float("nan"),
                        "max_abs": float("nan"), "spectral_radius": float("nan"),
                        "blocks": 0}
            B, H, dk, dv = x.shape
            xf = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            fro = _fro(xf) if xf.numel() else 0.0
            max_abs = float(xf.abs().max()) if xf.numel() else 0.0
            mat = xf.reshape(-1, dk, dv)
            sv = torch.linalg.svdvals(mat)          # [blocks, min(dk,dv)]
            spectral_radius = float(sv.max()) if sv.numel() else 0.0
            return {"present": True, "shape": shape, "fro": fro,
                    "max_abs": max_abs, "spectral_radius": spectral_radius,
                    "blocks": int(B * H)}
    except Exception:
        return {"present": True,
                "shape": list(getattr(S, "shape", [])),
                "fro": float("nan"), "max_abs": float("nan"),
                "spectral_radius": float("nan"), "blocks": 0}


def cov_lambda_max(k: torch.Tensor) -> float:
    """ k: [B,T,D] 或 [B,H,T,d]（先展平到 [N,F]）；C = XᵀX / N，
    返回 eigvalsh(C) 的最大值。非有限 / 空 / 异常输入返回 nan，不抛。 """
    try:
        with torch.no_grad():
            if k is None:
                return float("nan")
            x = k.detach().float()
            if x.dim() < 2 or x.numel() == 0 or x.shape[-1] == 0:
                return float("nan")
            x = x.reshape(-1, x.shape[-1])
            if not bool(torch.isfinite(x).all()):
                return float("nan")
            n = x.shape[0]
            C = (x.t() @ x) / float(n)
            ev = torch.linalg.eigvalsh(C)
            return float(ev[-1]) if ev.numel() else float("nan")
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# 5) ModelDiagnostics：只读挂载 / 采样 / 落盘
# ---------------------------------------------------------------------------
class ModelDiagnostics:
    """ 阶段 0 只读观测器。默认关闭；attach 后只设置普通属性，不改 state_dict、
    不改 forward 数值。observe 整体 try/except，任何异常只打一行 stderr。 """

    def __init__(self, model, opts=None, every: int = _DEFAULT_EVERY,
                 log_path=None, max_matrices: int = _MAX_MATRICES):
        self.model = model
        self.opts = list(opts) if opts is not None else []
        self.every = max(1, int(every))
        # 0 = 全量采集（覆盖所有 MLA/FFN 矩阵）；负数按默认限速处理。
        mm = int(max_matrices)
        self.max_matrices = 0 if mm == 0 else (mm if mm > 0 else _MAX_MATRICES)
        if log_path is None:
            log_path = os.path.join(paths.project_tmp_dir(), _DEFAULT_LOG_NAME)
        self._path = log_path
        self._file = None                 # 延迟打开，detach 时关闭
        self._probes = []                # [(module, old_diag_probe)]
        self._muon_profiles = []         # [(opt, old_profile)]
        self._attached = False
        self._attn = self._new_accum()

    # -- 日志路径 ----------------------------------------------------------
    @property
    def path(self) -> str:
        return self._path

    @staticmethod
    def _new_accum() -> dict:
        # 固定字段的聚合器：内存有界，不会随步数增长
        return {"calls": 0, "logits_absmax": 0.0,
                "entropy_sum": 0.0, "entropy_min": float("inf"),
                "finite_frac_sum": 0.0,
                "jacobian_sum": 0.0, "jacobian_count": 0}

    # -- 探针回调（接收 detached logits，只读累积）-------------------------
    def _attn_probe(self, logits):
        try:
            st = attention_stats(logits, with_jacobian=True)
            if st["finite_frac"] <= 0.0:
                return                              # 整窗无有限项，忽略
            a = self._attn
            a["calls"] += 1
            a["logits_absmax"] = max(a["logits_absmax"], st["logits_absmax"])
            a["entropy_sum"] += st["entropy_mean"]
            a["entropy_min"] = min(a["entropy_min"], st["entropy_min"])
            a["finite_frac_sum"] += st["finite_frac"]
            if st["jacobian_fro_mean"] is not None:
                a["jacobian_sum"] += st["jacobian_fro_mean"]
                a["jacobian_count"] += 1
        except Exception:
            pass                                    # 探针绝不崩训练

    def _consume_attention(self) -> dict:
        a = self._attn
        if a["calls"] == 0:
            result = {"logits_absmax": 0.0, "entropy_mean": 0.0,
                      "entropy_min": 0.0, "finite_frac": 0.0,
                      "jacobian_fro_mean": None, "probe_calls": 0}
        else:
            result = {
                "logits_absmax": a["logits_absmax"],
                "entropy_mean": a["entropy_sum"] / a["calls"],
                "entropy_min": (a["entropy_min"]
                                if a["entropy_min"] != float("inf") else 0.0),
                "finite_frac": a["finite_frac_sum"] / a["calls"],
                "jacobian_fro_mean": (a["jacobian_sum"] / a["jacobian_count"]
                                      if a["jacobian_count"] else None),
                "probe_calls": a["calls"],
            }
        self._attn = self._new_accum()             # 复位，开始下个采样区间
        return result

    # -- attach / detach ---------------------------------------------------
    def attach(self):
        """ 对有 diag_probe 属性的 MLAAttention 设置回调；把 Muon profile 置
        True（仅当该属性存在）。只动普通属性，零参数 / 零 buffer / 零注册。 """
        if self._attached:
            return self
        self._attn = self._new_accum()
        for module in self.model.modules():
            if hasattr(module, "diag_probe"):
                old = getattr(module, "diag_probe", None)
                self._probes.append((module, old))
                module.diag_probe = self._attn_probe
        for o in self.opts:
            if hasattr(o, "profile"):
                old = getattr(o, "profile", None)
                self._muon_profiles.append((o, old))
                o.profile = True
        self._attached = True
        return self

    def detach(self):
        """ 恢复 diag_probe / profile 原值，关闭日志句柄。可重复调用。 """
        for module, old in self._probes:
            try:
                module.diag_probe = old
            except Exception:
                pass
        self._probes.clear()
        for o, old in self._muon_profiles:
            try:
                o.profile = old
            except Exception:
                pass
        self._muon_profiles.clear()
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
            self._file = None
        self._attached = False
        return self

    # -- 采集 --------------------------------------------------------------
    def _collect_spectra(self) -> dict:
        out = {}
        seen_names = set()
        seen_ids = set()
        count = 0
        limit = self.max_matrices                 # 0 = 不限
        for prefix, module in self.model.named_modules():
            children = dict(module.named_children())
            is_mla = "w_dq" in children             # MLAAttention 独有
            is_ffn = ("ffn_up" in children and "ffn_down" in children)
            targets = []
            if is_mla:
                targets += [(a, children.get(a)) for a in _MLA_WEIGHTS]
            if is_ffn:
                targets += [(a, children.get(a)) for a in _FFN_WEIGHTS]
            for attr, lin in targets:
                if limit and count >= limit:
                    break
                w = getattr(lin, "weight", None)
                if not isinstance(w, torch.Tensor):
                    continue
                name = f"{prefix}.{attr}" if prefix else attr
                if name in seen_names or id(w) in seen_ids:
                    continue                         # 按 name / 共享权重去重
                seen_names.add(name)
                seen_ids.add(id(w))
                out[name] = spectrum_stats(w, max_dim=_DEFAULT_MAX_DIM)
                count += 1
            if limit and count >= limit:
                break
        return out

    def _collect_kda(self) -> dict:
        fw = getattr(self.model, "fast_weight", None)
        state = getattr(fw, "_state", None)          # (S, conv_cache) 或 None
        S = None
        if isinstance(state, (tuple, list)) and len(state) >= 1:
            S = state[0]
        elif isinstance(state, torch.Tensor):
            S = state
        ks = kda_state_stats(S)
        if ks.get("present") and S is not None:
            try:
                ks["state_cov_lambda_max"] = cov_lambda_max(S)
            except Exception:
                ks["state_cov_lambda_max"] = float("nan")
        return ks

    def _collect_optimizer(self) -> dict:
        per = []
        tot_ns5 = 0.0
        tot_step = 0.0
        tot_calls = 0
        for o in self.opts:
            if not hasattr(o, "ns5_seconds"):        # 属性缺失则优雅跳过
                continue
            ns5 = float(getattr(o, "ns5_seconds", 0.0) or 0.0)
            stp = float(getattr(o, "step_seconds", 0.0) or 0.0)
            calls = int(getattr(o, "ns5_calls", 0) or 0)
            per.append({"ns5_seconds": ns5, "step_seconds": stp,
                        "ns5_calls": calls,
                        "ns5_ratio": (ns5 / stp) if stp > 0 else None})
            tot_ns5 += ns5
            tot_step += stp
            tot_calls += calls
        return {"per_opt": per, "ns5_seconds": tot_ns5,
                "step_seconds": tot_step, "ns5_calls": tot_calls,
                "ns5_ratio": (tot_ns5 / tot_step) if tot_step > 0 else None}

    def _write(self, rec: dict) -> None:
        if self._file is None:
            d = os.path.dirname(self._path)
            if d:
                os.makedirs(d, exist_ok=True)
            self._file = open(self._path, "a", encoding="utf-8")
        self._file.write(json.dumps(rec, ensure_ascii=False,
                                    default=_json_default) + "\n")
        self._file.flush()

    @staticmethod
    def _to_float(v):
        if v is None:
            return None
        try:
            if isinstance(v, torch.Tensor):
                return float(v.detach().item())
            return float(v)
        except Exception:
            return None

    # -- observe -----------------------------------------------------------
    def observe(self, step: int, loss=None, preclip_grad_norm=None,
                extra=None):
        """ 仅当 step % every == 0 或 step == 1 时采集并写一行 JSONL。
        整体 try/except：任何诊断异常只往 stderr 打一行，绝不让训练崩。 """
        if not (step % self.every == 0 or step == 1):
            return
        try:
            rec = {
                "ts": time.time(),
                "step": int(step),
                "loss": self._to_float(loss),
                "preclip_grad_norm": self._to_float(preclip_grad_norm),
                "spectra": self._collect_spectra(),
                "attention": self._consume_attention(),
                "kda_state": self._collect_kda(),
                "optimizer": self._collect_optimizer(),
                "extra": extra if isinstance(extra, dict) else {},
            }
            self._write(rec)
        except Exception as e:
            print(f"[manifold_diag] observe 异常（已忽略）: {e}",
                  file=sys.stderr)


# ---------------------------------------------------------------------------
# 6) 日志摘要（供报告 / 测试）
# ---------------------------------------------------------------------------
def summarize_log(path, last_n: int = 100) -> dict:
    """ 读 JSONL 返回摘要：步数、σ_max 序列、offgram_fro 序列、grad norm
    序列、NS5 占比均值、是否存在 NaN。文件不存在返回 {}。
    σ_max / offgram_fro 序列为「每步该指标在所有矩阵上的最大值」。 """
    if not os.path.exists(path):
        return {}
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return {}
    if last_n and last_n > 0:
        rows = rows[-last_n:]

    steps, sigma_seq, off_seq, grad_seq, ns5_ratios = [], [], [], [], []
    cond_seq = []
    has_nan = False
    for r in rows:
        steps.append(r.get("step"))
        spectra = r.get("spectra") or {}
        sm_step = None
        off_step = None
        cond_step = None
        for st in spectra.values():
            if not isinstance(st, dict):
                continue
            sm = st.get("sigma_max")
            if sm is not None:
                if _is_nan(sm):
                    has_nan = True
                elif sm_step is None or sm > sm_step:
                    sm_step = sm
            off = st.get("offgram_fro")
            if off is not None:
                if _is_nan(off):
                    has_nan = True
                elif off_step is None or off > off_step:
                    off_step = off
            # 批复的判定量：条件数是否随时间膨胀（每步取所有矩阵的最大值）
            cd = st.get("cond")
            if cd is not None:
                if _is_nan(cd):
                    has_nan = True
                elif cond_step is None or cd > cond_step:
                    cond_step = cd
        sigma_seq.append(sm_step)
        off_seq.append(off_step)
        cond_seq.append(cond_step)

        g = r.get("preclip_grad_norm")
        if g is not None:
            grad_seq.append(g)
            if _is_nan(g):
                has_nan = True
        else:
            grad_seq.append(None)
        nr = (r.get("optimizer") or {}).get("ns5_ratio")
        if nr is not None:
            ns5_ratios.append(nr)
            if _is_nan(nr):
                has_nan = True

    return {
        "n_records": len(rows),
        "n_steps": len(rows),
        "steps": steps,
        "sigma_max": sigma_seq,
        "cond_max": cond_seq,
        "offgram_fro": off_seq,
        "grad_norm": grad_seq,
        "ns5_ratio_mean": (sum(ns5_ratios) / len(ns5_ratios))
                          if ns5_ratios else None,
        "has_nan": has_nan,
    }


__all__ = [
    "spectrum_stats", "grad_norm_before_clip", "attention_stats",
    "kda_state_stats", "cov_lambda_max", "ModelDiagnostics", "summarize_log",
]
