# ===========================================================================
# GARGANTUA v1 — init.py：创建并初始化模型
# 用法：python init.py --name alpha [--out models/] [--force]
#       [--enc-blocks N] [--dec-blocks N] [--share-enc-dec]
#       [--d-model N]
#       [--ctx-full N] [--ctx-compress N]
#       [--csa-ratio N] [--hca-ratio N]
#       [--json]                    # 末行输出 JSON 结果（server 用）
#
# 2026-10-24：--json 模式——server 的 /api/model/init 把建模当作一次性
# 子进程调度（server_services.run_oneshot）：stdout 最后一行必须是
# {"ok": true, "result": {...}} 或 {"ok": false, "error": "..."}。
# 人类可读的摘要照常打印（在 JSON 行之前），互不干扰。
#
# 2026-10-20 修复（审查结论）：
#   1. 落盘改走 common.save_checkpoint（临时文件 + replace 原子写），
#      与 server.py / train.py 的 config.json 格式完全同构。
#   2. 默认拒绝覆盖已存在模型（防止误操作抹掉训练成果），--force 才允许。
# 2026-10-23：新增上下文长度选项（完整/压缩容量 + CSA/HCA 压缩比）。
#   语义：KV = 完整区（ctx_full 个原始 token）｜压缩区
#   （ctx_compress 容量，对外有 CSA/HCA 两个同源视图，压缩比分别可配）。
#   缺省即 spec 默认（1024/3072/4/32）。
# 2026-10-25：删除 CSA 局部窗选项——CSA 的保留区就是完整 KV 区。
# 2026-11-04：新增 --share-enc-dec（enc/dec 逐层共享权重）。共享模式下
#   两侧 block 数强制相等：只给一侧另一侧跟随、都不给取
#   spec.SHARED_DEFAULT_BLOCKS、给全但不等直接拒绝。归一化规则在
#   common.normalize_shared_blocks（server 的 Web 入口用同一份）。
# ===========================================================================
import argparse
import json
import os
import sys
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
from common import (MAX_BLOCKS, model_exists, model_init_lock,
                    normalize_shared_blocks, save_checkpoint,
                    valid_model_name, validate_ctx_config)
from model import Gargantua


def _run(args) -> dict:
    # --json 模式：人类摘要走 stderr，stdout 只留最后的 JSON 结果行
    _say = (lambda s: print(s, file=sys.stderr)) if args.json else print
    bad = valid_model_name(args.name)
    if bad:
        raise SystemExit(f"[init] 非法模型名 {args.name!r}: {bad}")
    # 2026-11-04：共享权重要求两侧等深（唯一规则在 common）
    try:
        enc_blocks, dec_blocks = normalize_shared_blocks(
            args.enc_blocks, args.dec_blocks, args.share_enc_dec)
    except ValueError as e:
        raise SystemExit(f"[init] {e}")
    for label, v in (("enc-blocks", enc_blocks),
                     ("dec-blocks", dec_blocks)):
        if v is not None and not (1 <= int(v) <= MAX_BLOCKS):
            # 0/负数会造出 0-block 实例，config 写默认值 → 目录再也无法
            # 被加载（state_dict 缺 key），属不可逆的坏目录
            raise SystemExit(f"[init] --{label} 必须在 [1, {MAX_BLOCKS}]: {v}")
    try:
        ctx = validate_ctx_config(ctx_full=args.ctx_full,
                                  ctx_compress=args.ctx_compress,
                                  csa_ratio=args.csa_ratio,
                                  hca_ratio=args.hca_ratio)
    except ValueError as e:
        raise SystemExit(f"[init] {e}")
    # 2026-11-04：模型宽度建模可选，规则唯一真源在 spec.check_d_model
    try:
        dm = (spec.D_MODEL if args.d_model is None
              else spec.check_d_model(args.d_model))
    except ValueError as e:
        raise SystemExit(f"[init] {e}")

    # 2026-10-25：模型精度（fp64/32/16/8/4 + int16/8/4/2）；非法名 → 拒绝
    from precision import validate_dtype
    if args.dtype is not None:
        try:
            validate_dtype(args.dtype)
        except ValueError as e:
            raise SystemExit(f"[init] {e}")
    dtype_name = args.dtype or "fp32"
    # 2026-10-25：词表预设（每个模型从选定预设 seed 出自己的词表）
    from vocab_presets import validate_vocab, vocab_dir
    from common import CONFIG_NAME, install_model_vocab
    try:
        preset_name = validate_vocab(args.vocab)
    except ValueError as e:
        raise SystemExit(f"[init] {e}")

    d = os.path.join(args.out, args.name)
    # 2026-10-27 审计修复（H4）：冲突检查—写入—发布全程持跨进程锁。
    # 并发提交两个同名初始化此前会并发写同一临时文件（replace 失败/
    # 权重与 config 不配套）；训练 start 在同一把锁上等待，不可能读到
    # 写了一半的模型。
    with model_init_lock(d):
        if model_exists(d) and not args.force:
            raise SystemExit(f"[init] {d} 已存在，拒绝覆盖（--force 强制）")
        # 2026-10-30 词表契约重构：先把词表整目录复制进 models/NAME/vocab/。
        # 顺序不能反——模型本体的 wte / lm_head 行数由词表长度决定，必须
        # 先落到 vocabs 目录里拿到 vocab_size，才能构造出形状正确的模型。
        src_vocab = str(vocab_dir(preset_name))
        vocab_size = install_model_vocab(d, src_vocab)
        if vocab_size is None:
            raise SystemExit(
                f"[init] 词表预设 {preset_name!r} 没有声明 vocab_size："
                f"{src_vocab}/{CONFIG_NAME} 缺失或字段非法。"
                f"该预设还是旧格式（只有 vocab.txt），请用 .vfc 生成器"
                f"重新导出后再建模")

        torch.manual_seed(args.seed)
        t0 = time.time()
        m = Gargantua(n_enc_blocks=enc_blocks,
                      n_dec_blocks=dec_blocks,
                      share_enc_dec=args.share_enc_dec,
                      d_model=dm,
                      vocab_size=vocab_size, **ctx)
        # parameters() 由 PyTorch 去重——共享模型的 n 只算一份参数
        n = sum(p.numel() for p in m.parameters())
        config = {
            "name": args.name,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "params": n,
            # 2026-11-04：以实例为准（旧目录 config 里本就有这个键）
            "d_model": m.d_model, "head_dim": spec.HEAD_DIM,
            # 以实例为准：共享模式下 common 会把两侧归一化成同一个数，
            # train.py 的 _config_dict 也是这么记的。
            "n_enc_blocks": len(m.enc_blocks),
            "n_dec_blocks": len(m.dec_blocks),
            # 2026-11-04：加载侧的开关（共享模型只落 enc 侧一份权重）
            "share_enc_dec": bool(args.share_enc_dec),
            # 2026-10-30：词表长度（每个模型可以不同；common.model_vocab_size
            # 据此构造 wte / lm_head 的形状）。旧的 "vocab_size_padded" 键随之
            # 退役——长度不再「pad」到一个全局固定值，就是词表的真实长度。
            "vocab_size": vocab_size,
            # 上下文长度（建模期可选；CSA/HCA 是同一段压缩区内容的两个视图）
            "ctx_full": ctx["ctx_full"],
            "ctx_compress": ctx["ctx_compress"],
            "csa_ratio": ctx["csa_ratio"],
            "hca_ratio": ctx["hca_ratio"],
            "optimizer": spec.OPTIMIZER,
            # 2026-10-25：模型精度 + 词表预设来源（build_model 据此施加）
            "dtype": dtype_name,
            "vocab_preset": preset_name,
        }
        save_checkpoint(d, m, config)
        # 词表已在建模之前复制进 models/NAME/vocab/（见上方 install_model_vocab）
        # ——这里不再重复 seed：重复复制会覆盖用户在建模期间可能做的改动。
    _say(f"[init] 词表预设: {preset_name}  dtype: {dtype_name}"
         f"  d_model: {m.d_model}")
    _say(f"[init] 权重模式: " + (
        f"共享（enc/dec 各 {len(m.enc_blocks)} 层逐层绑定，只落一份参数）"
        if args.share_enc_dec else "分离"))
    total_ctx = ctx["ctx_full"] + ctx["ctx_compress"]
    _say(f"[init] {d}  params={n/1e6:.1f}M  ({time.time()-t0:.1f}s)")
    _say(f"[init] 上下文: full={ctx['ctx_full']} + compress="
         f"{ctx['ctx_compress']} = {total_ctx}；"
         f"CSA {ctx['csa_ratio']}:1 · HCA {ctx['hca_ratio']}:1"
         f"（两视图都读完整区）")
    return {"name": args.name, "dir": d, "params": n,
            "d_model": m.d_model,
            "share_enc_dec": bool(args.share_enc_dec),
            "n_enc_blocks": len(m.enc_blocks),
            "n_dec_blocks": len(m.dec_blocks),
            "total_ctx": total_ctx, **ctx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="模型名（目录名）")
    ap.add_argument("--out", default="models", help="模型根目录")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enc-blocks", type=int, default=None,
                    help=f"encoder block 数（默认 {spec.N_ENC_BLOCKS}）")
    ap.add_argument("--dec-blocks", type=int, default=None,
                    help=f"decoder block 数（默认 {spec.N_DEC_BLOCKS}）")
    ap.add_argument("--share-enc-dec", action="store_true",
                    help="enc/dec 逐层共享权重（默认分离；共享时两侧 block "
                         "数强制相等，都不给时一起取 "
                         f"{spec.SHARED_DEFAULT_BLOCKS}）")
    ap.add_argument("--d-model", type=int, default=None,
                    help=f"模型宽度（默认 {spec.D_MODEL}；"
                         f"须 ≥ {spec.D_MODEL_MIN} 且为 {spec.HEAD_DIM} 的倍数，"
                         f"无上限；示例档位 {list(spec.D_MODEL_CHOICES)}）")
    ap.add_argument("--force", action="store_true",
                    help="允许覆盖已存在的同名模型")
    # ---- 上下文长度（2026-10-23 新功能）----
    ap.add_argument("--ctx-full", type=int, default=None,
                    help=f"完整 KV 区容量，token（默认 {spec.CTX_FULL}）")
    ap.add_argument("--ctx-compress", type=int, default=None,
                    help=f"压缩区容量（原始 token 计，默认 "
                         f"{spec.CTX_COMPRESS}；CSA/HCA 是它的两个视图）")
    ap.add_argument("--csa-ratio", type=int, default=None,
                    help=f"CSA 压缩比 N:1（默认 {spec.CSA_RATIO}，检索视图）")
    ap.add_argument("--hca-ratio", type=int, default=None,
                    help=f"HCA 压缩比 N:1（默认 {spec.HCA_RATIO}，"
                         f"全局概括视图；须 ≥ csa-ratio）")
    ap.add_argument("--json", action="store_true",
                    help="末行输出 JSON 结果（server run_oneshot 协议）")
    # ---- 词表预设 + 精度（2026-10-25）----
    ap.add_argument("--vocab", default=None,
                    help="词表预设名（默认 base，从 vocabs/ 派生）")
    ap.add_argument("--dtype", default=None,
                    help="模型精度：fp64/fp32/fp16/bf16/fp8/int8/int4/int2"
                         "（默认 fp32）")
    args = ap.parse_args()

    try:
        result = _run(args)
    except SystemExit as e:
        if args.json:
            msg = e.code if isinstance(e.code, str) else f"退出码 {e.code}"
            print(json.dumps({"ok": False, "error": msg},
                             ensure_ascii=False))
            return 1
        raise
    except Exception as e:
        if args.json:
            print(json.dumps({"ok": False, "error": str(e)},
                             ensure_ascii=False))
            return 1
        raise
    if args.json:
        # 必须是 stdout 的【最后】一行——run_oneshot 只解析末行
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
