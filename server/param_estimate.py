# ===========================================================================
# GARGANTUA v1 — server/param_estimate.py：建模参数量预估（纯标准库）
# ---------------------------------------------------------------------------
# 为什么要有它：创建模型前要先知道「大概多大」。默认 4+6 blocks 是 10 亿
# 参数级别，d_model / block 数一改就可能是 0.2B 也可能是 4B，盲建一次要等
# 几分钟、写几个 GB 才看见结果。这里按架构定义解析式地算，不建模型、不碰
# torch——服务端在没装 torch 的环境里也能回答。
#
# 唯一真源：所有维度都从 spec 取（BLOCK_ORDER 里的子层数、N_HEADS、
# FFN_HIDDEN、MLA 秩、PATCH_VEC_DIM、AUDIO_SAMPLES_PER_TOKEN、
# VISUAL_COORD_BINS、DRAW_KV_COMPRESS……），本文件只写「形状 → 元素个数」
# 的推导，不重复定义任何维度。
#
# 三个分组与 common.split_state_dict 严格对齐（保存/加载用的是同一套
# 前缀规则，预估不能另起一套）：
#   离散 embedding = wte.*
#   连续 embedding = visual_proj.* / visual_coord_x.* / visual_coord_y.* /
#                    audio_proj.*
#   其余全部 = 模型权重（type_emb、lm_head、压缩器、blocks、fast weight、
#              final norm……）
#
# 校验说明：入参假定已由调用方（server.py）按建模同一套规则校验过；
# 这里只做最后一道防御，非法值抛 ValueError（HTTP 层映射 400）。
# ===========================================================================
from __future__ import annotations

import spec


# ---------------------------------------------------------------------------
# 子层形状（与 server/model.py 一一对应）
# ---------------------------------------------------------------------------
def _kda_params(d: int) -> int:
    """KDAAttention：3 组 q/k/v 投影 + 3 条 depthwise 短卷积 + alpha/beta
    + 输出门控 + 输出投影。KDA_EXPAND_V=1.0 → val_dim = key_dim = H*head_dim。"""
    h = spec.N_HEADS
    dk = spec.KDA_HEAD_DIM
    dv = int(dk * spec.KDA_EXPAND_V)
    key_dim, val_dim = h * dk, h * dv
    k = spec.KDA_CONV_KERNEL
    n = (d * key_dim) * 2 + d * val_dim                 # w_q / w_k / w_v
    n += (key_dim * k + key_dim) * 2                    # conv_q / conv_k
    n += val_dim * k + val_dim                          # conv_v
    n += d * dk + dk * key_dim + d * h                  # alpha down/up、beta
    n += dv                                             # o_norm.weight
    n += d * val_dim + val_dim * d                      # w_gate / w_o
    return n


def _mla_params(d: int) -> int:
    """MLAAttention：Q 低秩、KV 低秩、V 低秩 + 输出投影（全无 bias）。"""
    inner = spec.N_HEADS * spec.HEAD_DIM
    return (d * spec.MLA_Q_LORA_RANK + spec.MLA_Q_LORA_RANK * inner
            + d * spec.MLA_KV_LORA_RANK + spec.MLA_KV_LORA_RANK * inner
            + spec.MLA_V_LORA_RANK * inner + inner * d)


def _ffn_params(d: int) -> int:
    """GLUFFN：up 是 d → 2*FFN_HIDDEN，down 是 FFN_HIDDEN → d。"""
    return d * spec.FFN_HIDDEN * 2 + spec.FFN_HIDDEN * d


def _block_params(d: int) -> int:
    """一个 Block：按 BLOCK_ORDER 数子层，外加每个子层一份 RMSNorm、
    AttnRes 的两套 proj/norm 与块末标量门。"""
    order = tuple(spec.BLOCK_ORDER)
    n_kda = order.count("kda")
    n_mla = order.count("mla_csa") + order.count("mla_hca")
    n_ffn = order.count("ffn")
    if n_ffn != spec.N_FFN_PER_BLOCK:
        raise ValueError(
            f"spec.BLOCK_ORDER 的 ffn 数 {n_ffn} 与 N_FFN_PER_BLOCK "
            f"{spec.N_FFN_PER_BLOCK} 不一致")
    n = n_kda * _kda_params(d) + n_mla * _mla_params(d)
    n += n_ffn * _ffn_params(d)
    n += spec.N_SUBL_LAYERS * d                  # 每个计算子层一份 RMSNorm
    n += d + d + d + d + 1                       # AttnRes: 2 proj + 2 norm + gate
    return n


def _compressor_params(ratio: int, d: int) -> int:
    """StepCompressor：ratio×d 位置嵌入 + 两层 D×D Linear（带 bias）。"""
    return ratio * d + 2 * (d * d + d)


# ---------------------------------------------------------------------------
# 对外：一次算出三组参数量与存储占用
# ---------------------------------------------------------------------------
def estimate(d_model: int = spec.D_MODEL,
             n_enc_blocks: int = spec.N_ENC_BLOCKS,
             n_dec_blocks: int = spec.N_DEC_BLOCKS,
             share_enc_dec: bool = spec.SHARE_ENC_DEC,
             csa_ratio: int = spec.CSA_RATIO,
             hca_ratio: int = spec.HCA_RATIO,
             vocab_size: int = spec.HEAD_SIZE,
             dtype: str = "fp32") -> dict:
    """返回三类参数量 / 理论存储 / fp32 落盘大小。

    参数：
      d_model       语义向量宽度（128 的倍数，128..2048）
      n_enc_blocks  encoder block 数
      n_dec_blocks  decoder block 数（共享权重时必须等于 enc）
      share_enc_dec enc/dec 是否逐层共享同一批参数
      csa_ratio     CSA 压缩比（决定 csa 压缩器位置嵌入行数）
      hca_ratio     HCA 压缩比
      vocab_size    词表长度（wte / lm_head 的行数）。缺省 HEAD_SIZE 只是
                    直接调用时的占位——HTTP 端点总是传入选中的真实词表长度。
      dtype         精度名（spec.DTYPE_BITS 的键）

    存储口径：
      bytes        = params × bits / 8，即「按该精度存」的理论下限；
      file_bytes   = params × 4，当前 init 一律以 fp32 容器落盘
                     （低比特只在加载后量化），所以磁盘占用按 fp32 算。
    """
    d = spec.check_d_model(d_model)
    if not isinstance(share_enc_dec, bool):
        raise ValueError(f"share_enc_dec 必须是布尔值: {share_enc_dec!r}")
    enc = int(n_enc_blocks)
    dec = int(n_dec_blocks)
    if enc < 1 or dec < 1:
        raise ValueError(f"block 数必须 >= 1（收到 {enc} / {dec}）")
    if share_enc_dec and enc != dec:
        raise ValueError(
            f"共享权重要求 enc blocks == dec blocks（收到 {enc} / {dec}）")
    if csa_ratio < 2 or hca_ratio < 2:
        raise ValueError(f"压缩比必须 >= 2（收到 {csa_ratio} / {hca_ratio}）")
    v = int(vocab_size)
    if v < 1:
        raise ValueError(f"词表长度非法: {vocab_size!r}")
    if dtype not in spec.DTYPE_BITS:
        raise ValueError(
            f"dtype 必须是 {list(spec.DTYPE_BITS)} 之一: {dtype!r}")
    bits = spec.DTYPE_BITS[dtype]

    # ---- 离散 embedding：wte（查表类，可放 CPU）----
    discrete = v * d

    # ---- 连续 embedding：视觉 patch / 2D 坐标 / 音频波形 → d_model ----
    continuous = ((spec.PATCH_VEC_DIM + 2 * spec.VISUAL_COORD_BINS
                   + spec.AUDIO_SAMPLES_PER_TOKEN) * d)

    # ---- 模型权重：其余全部（含 lm_head / type_emb / 压缩器 / blocks）----
    blk = _block_params(d)
    n_blocks = enc if share_enc_dec else enc + dec
    # 共享时两侧 final norm 共用同一批参数（见 model.retie_shared_weights）
    n_final_norm = 1 if share_enc_dec else 2
    weights = (
        v * d                                    # lm_head（与 wte 不绑定）
        + spec.NUM_TYPES * d                     # type_emb
        + _compressor_params(csa_ratio, d)
        + _compressor_params(hca_ratio, d)
        + _compressor_params(spec.DRAW_KV_COMPRESS, d)
        + _kda_params(d)                         # fast weight bank = 1 个 KDA
        + n_blocks * blk
        + n_final_norm * d
    )

    def part(key: str, label: str, params: int) -> dict:
        return {"key": key, "label": label, "params": params,
                "bytes": params * bits / 8,
                "file_bytes": params * 4}

    parts = [
        part("discrete", "离散 Embedding", discrete),
        part("continuous", "连续 Embedding", continuous),
        part("weights", "模型权重", weights),
    ]
    total_params = discrete + continuous + weights
    return {
        "ok": True,
        "config": {
            "d_model": d,
            "n_enc_blocks": enc,
            "n_dec_blocks": dec,
            "share_enc_dec": share_enc_dec,
            "csa_ratio": int(csa_ratio),
            "hca_ratio": int(hca_ratio),
            "vocab_size": v,
            "dtype": dtype,
            "bits": bits,
        },
        "parts": parts,
        "total": {
            "params": total_params,
            "bytes": total_params * bits / 8,
            "file_bytes": total_params * 4,
        },
    }
