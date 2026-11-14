# ===========================================================================
# GARGANTUA v1 — train.py：文本预训练（教师强制，B=1，FP32）
# ---------------------------------------------------------------------------
# 数据：data/ 下所有 .tokens 文件（token-6 位流，必须带 { } 时间步结构），
# 由 tokens/tokenset 展开成「扁平 token_ids + 并行 step_ids」；
# 每个窗口切成 user 段（encoder）+ model 段（decoder 预测目标）。
# 优化器：muon（隐藏层 2D Muon + 其余 AdamW，默认）/ adamw / allmuon，
# 三档取值唯一真源 = clip_data.OPTIMIZERS。
# 用法：
#   python train.py --model models/alpha --data data --steps 200 --block-size 32
#
# 2026-10-20 修复（审查结论）：
#   1. main() 改用 common.build_model：按模型目录 config.json 的 block 数
#      构造与加载，非默认 block 数的模型不再崩溃。
#   2. block_size 经 common.validate_block_size 校验（偶数 ≥2）——奇数会
#      令 s_dec 与 dec_in 长度不一致，forward_train 必崩（已实测复现）。
#   3. 删除死变量 tgt（与 y 完全相同）。
# 2026-10-25：训练数据改为 .tokens（不再读 .txt / JSON）。
# ===========================================================================
import argparse
import glob
import os
import random
import time

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
import clip_data
from common import (build_model, read_config, resolve_weights_path,
                    save_checkpoint,
                    validate_block_size)
from muon import build_optimizers
from tokens.tokenset import load_corpus
from tokens.vocab import set_model_vocab
from train_step import forward_window, window_index, window_loss


def load_corpus_ids(data_dir: str):
    """ data/ 下全部 .tokens → 按文档分组的 [(ids, step_ids), ...]。 """
    paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.tokens"),
                             recursive=True))
    if not paths:
        raise SystemExit(f"{data_dir} 下没有 .tokens 训练文件"
                         f"（训练数据必须带 {{ }} | 时间步结构）")
    return load_corpus(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="模型目录（config.json + weights/ + vocab/）")
    ap.add_argument("--data", default="data")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--block-size", type=int, default=32,
                    help="窗口大小（token/时间步，偶数）；user/model 各一半")
    ap.add_argument("--lr", type=float, default=spec.ADAMW_LR)
    ap.add_argument("--muon-lr", type=float, default=spec.MUON_LR)
    ap.add_argument("--optimizer", default=spec.OPTIMIZER,
                    choices=list(clip_data.OPTIMIZERS))
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--diag", action="store_true", default=False,
                    help="阶段 0 零风险谱诊断（只读观测，默认关闭）")
    ap.add_argument("--diag-every", type=int, default=50,
                    help="诊断采样间隔步数（step 1 也会采样）")
    ap.add_argument("--diag-log", type=str, default="",
                    help="诊断 JSONL 路径（缺省 data/tmp/manifold_diag.jsonl）")
    ap.add_argument("--diag-max-matrices", type=int, default=24,
                    help="每次采样最多采集的 MLA/FFN 矩阵数（0 = 全量，覆盖所有 block）")
    # 阶段 0 Muon 校准对照（默认全部不传 → build_optimizers 沿用类默认值，
    # 既有训练行为逐位不变）。只用于诊断「步子太大」假设，不进生产配置。
    ap.add_argument("--no-muon-rms-align", action="store_true",
                    help="关闭 Muon 的 RMS 对齐（去掉 ×√max(shape) 放大）")
    ap.add_argument("--muon-weight-decay", type=float, default=None,
                    help="覆盖 Muon 组的解耦 weight decay（默认 0）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    B = validate_block_size(args.block_size)
    m = build_model(args.model)
    set_model_vocab(args.model)          # 每模型词表（models/NAME/vocab/ 副本）
    m.train()

    opts = build_optimizers(
        m, args.optimizer, args.lr, args.muon_lr, spec.MUON_MOMENTUM,
        muon_rms_align=(False if args.no_muon_rms_align else None),
        muon_weight_decay=args.muon_weight_decay)
    docs = load_corpus_ids(args.data)
    # 文档边界保留：窗口只在单个文档内采样，绝不跨 <stop>（2026-10-27
    # 审计修复）；过短（装不下一个完整窗口）的文档跳过。
    usable = [d for d in docs if len(d["pos_ids"]) >= B + 2]
    if not usable:
        raise SystemExit(f"语料不足：所有文档都 < {B + 2} tokens")
    total = sum(len(d["pos_ids"]) for d in usable)
    n_mm = sum(1 for d in usable if d["multimodal"])
    print(f"[train] corpus={total} tokens/{len(usable)} docs"
          f"（多模态 {n_mm}）  steps={args.steps}  block={B}  "
          f"optimizer={args.optimizer}")

    diag = None
    t0 = time.time()
    skipped = 0
    try:
        if args.diag:
            # 阶段 0 零风险谱诊断：只在显式 --diag 时才 import / 计算，
            # 默认路径连 manifold_diag 的 bootstrap 都不触发（零额外开销）。
            from manifold_diag import ModelDiagnostics, grad_norm_before_clip
            diag = ModelDiagnostics(
                m, opts, every=args.diag_every,
                log_path=args.diag_log or None,
                max_matrices=args.diag_max_matrices).attach()
        for step in range(1, args.steps + 1):
            doc = random.choice(usable)            # 先选文档，再在文档内开窗
            i = window_index(doc, B, random, force_last=(step == 1))
            if i is None:
                skipped += 1
                continue
            logits, y, n_valid = forward_window(m, doc, i, B)
            if n_valid == 0:                       # 整窗都是媒体，无目标可训
                skipped += 1
                continue
            loss = window_loss(logits, y)
            for o in opts:
                o.zero_grad(set_to_none=True)
            loss.backward()
            # 阶段 0：裁剪前全局 grad norm（clip 到 1.0 会掩盖退化）
            diag_grad = None
            if diag is not None:
                diag_grad = grad_norm_before_clip(m.parameters())
            clip_return = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            for o in opts:
                o.step()
            # 阶段 0 交叉审计修复：observe 必须在 reset_memory 之前。
            # _collect_kda() 读 fast_weight._state，而 reset_memory() 会把它
            # 清成 None —— 旧顺序让 KDA 状态恒为 present=False，白采一批空数据。
            # 观测只读、不改数值；复位仍在同一步内、下一个 forward 之前完成。
            if diag is not None:
                diag.observe(step, loss.item(),
                             preclip_grad_norm=diag_grad,
                             extra={"clip_return": float(clip_return),
                                    "optimizer": args.optimizer})
            m.reset_memory()            # 统一出口：含 fast weight / MoMW / SOMN / 未来新增记忆组件

            if step % 10 == 0 or step == 1:
                el = time.time() - t0
                print(f"[train] step {step:4d}/{args.steps}  "
                      f"loss {loss.item():.4f}  {el/step:.2f}s/it"
                      + (f"  (跳过 {skipped} 窗)" if skipped else ""))
            if step % args.save_every == 0:
                save_checkpoint(args.model, m, _config_dict(args, m))
                print(f"[train] saved {resolve_weights_path(args.model)}")
    finally:
        if diag is not None:
            diag.detach()
    save_checkpoint(args.model, m, _config_dict(args, m))
    print(f"[train] done, saved {resolve_weights_path(args.model)}")


def _config_dict(args, m) -> dict:
    """ 与 init.py / server.py 同构的 config.json 内容（block 数以实例为准，
    保留首次创建的 created 时间戳）。

    2026-10-23：上下文长度从模型实例读回（建模时可配）——此前这里硬写
    spec 常量，训练一存盘就会把自定义 ctx_full/CSA/HCA 覆写成默认值，
    磁盘配置与真实结构脱节。 """
    cfg = dict(read_config(args.model))          # 保留既有字段（如 created）
    cfg.update({
        "name": os.path.basename(os.path.normpath(args.model)),
        "params": sum(p.numel() for p in m.parameters()),
        # 2026-11-04：以模型实例为准写回宽度（旧默认目录同样是 1024）
        "d_model": getattr(m, "d_model", spec.D_MODEL),
        "head_dim": spec.HEAD_DIM,
        "n_enc_blocks": len(m.enc_blocks),
        "n_dec_blocks": len(m.dec_blocks),
        # 2026-11-04：以模型实例为准写回权重模式。共享模型若丢了这个
        # 字段，下次 build_model 会按「分离」构造，strict load 直接失败。
        "share_enc_dec": bool(getattr(m, "share_enc_dec", False)),
        # 2026-10-30：词表长度是每个模型自己的属性（不再有全局 pad 值）。
        # 以模型实例为准——训练中途换过词表时，这里记录的是实际在训的形状。
        "vocab_size": int(getattr(m, "n_vocab", 0) or 0),
        "ctx_full": getattr(m, "ctx_full", spec.CTX_FULL),
        "ctx_compress": getattr(m, "ctx_compress", spec.CTX_COMPRESS),
        "csa_ratio": getattr(m, "csa_ratio", spec.CSA_RATIO),
        "hca_ratio": getattr(m, "hca_ratio", spec.HCA_RATIO),
        # 2026-10-27 审计修复：CSA 局部窗参数已删除，此前此处 getattr 回退
        # 读取该字段必抛 AttributeError，任何周期保存/最终保存都让训练
        # 结果无法落盘
        "optimizer": args.optimizer,
    })
    cfg.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
    return cfg


if __name__ == "__main__":
    main()
