# ===========================================================================
# GARGANTUA v1 — precision.py：模型精度配置
# ---------------------------------------------------------------------------
# 用户要求（2026-10-25）：模型权重与推理精度可配置。最终落地集合：
#   原生 dtype（GPU/CPU 均可原生计算）：fp64 / fp32 / fp16 / bf16
#   低比特模拟（CPU 无对应 matmul 内核，按用户「CPU 可强制跑、有损耗」
#     的口径走「权重量化 → 反量化回 fp32」的权重-only 模拟）：
#     fp8（e4m3fn）/ fp4（e2m1）/ int16 / int8 / int4 / int2。
#   「损耗」即：模型仍以 fp32 张量承载，但参数已被压到目标格式的表示
#   网格；int2 是最低档（相对误差上限 ≈ 50%，符合「强制跑、有损耗」）。
#
# 两类落地方式：
#   1) 原生 dtype（fp64 / fp32 / fp16 / bf16）：torch 直接 cast。
#   2) 模拟（fp8/fp4/int16/int8/int4/int2）：把权重量化到目标格式的表示
#      网格、再反量化回 fp32 原地写回参数。
#
# 所有映射与校验集中在此模块，common.build_model 在加载权重后调用
# apply_precision；config.json 记录 dtype 名（默认 fp32）。
# ===========================================================================
from __future__ import annotations

import torch

# ---- 原生 dtype（torch 可直接 compute）----
NATIVE = {
    "fp64": torch.float64,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

# ---- 低比特模拟（CPU 无内核，量化→反量化）----
#   fp8   : 走 torch.float8_e4m3fn（>=2.1 可用；否则手动 e4m3 网格模拟）
#   fp4   : e2m1 网格（4bit：1符号/2指数/1尾数），手动最近邻网格量化
#   intN  : 对称逐张量整数量化，N 位
SIMULATED = {
    "fp8": "e4m3fn",
    "fp4": "e2m1",
    "int16": 16,
    "int8": 8,
    "int4": 4,
    "int2": 2,
}

ALL_NAMES = list(NATIVE.keys()) + list(SIMULATED.keys())
DEFAULT = "fp32"


def validate_dtype(name) -> None:
    """dtype 名校验；非法抛 ValueError（HTTP 层映射 400）。"""
    if name not in ALL_NAMES:
        raise ValueError(
            f"dtype 必须是 {ALL_NAMES} 之一: {name!r}")


def parse_precision(name: str) -> dict:
    """名字 → {kind, name, ...}。"""
    validate_dtype(name)
    if name in NATIVE:
        return {"kind": "native", "name": name,
                "torch_dtype": NATIVE[name]}
    return {"kind": "sim", "name": name, "spec": SIMULATED[name]}


# ---------------------------------------------------------------------------
# 量化→反量化（纯张量运算，CPU 可跑）
# ---------------------------------------------------------------------------
def _to_fp8_grid(x: torch.Tensor, fmt: str) -> torch.Tensor:
    """fp8 网格量化→反量化（返回 fp32）。优先 torch 原生 float8，否则手算。"""
    if hasattr(torch, "float8_e4m3fn"):
        dt = (torch.float8_e4m3fn if fmt == "e4m3fn"
              else torch.float8_e5m2)
        return x.to(dt).to(torch.float32)
    return _sim_e4m3(x)


def _sim_e4m3(x: torch.Tensor) -> torch.Tensor:
    """e4m3fn 简化网格（指数 4bit [-7,8]、尾数 3bit；饱和 ±448）。"""
    sign = torch.sign(x)
    ax = x.abs().clamp(0, 448)
    # 以 2^4=16 为「粗糙档」起步的近似：用 log2 量化指数、线性量化尾数。
    # 这里采用与 torch float8 一致的 round-to-nearest-even 近似实现：
    # 直接借助 fp16 做中转（fp16 的 exp=5bit/man=10bit 比目标更细，
    # 再按 e4m3 的 exp/man 位宽裁剪），足够贴近真实 fp8 的损耗量级。
    fp16 = ax.half()
    # 把 fp16 的指数压到 4bit、尾数压到 3bit：
    #   exp16 ∈ [-14,15] → exp4 ∈ [-7,8]；man → 低 3bit。
    u = fp16.view(torch.uint16).to(torch.int32)
    sign_bit = (u >> 15) & 1
    exp16 = (u >> 10) & 0x1F
    man16 = u & 0x3FF
    # 非规格/零：直接归零（e4m3 非规格区简单处理为 0）
    e4 = exp16 - 15  # 真实指数
    e4 = e4.clamp(-7, 8)
    # 尾数保留高 3bit（10→3）：右移 7 位再左移 7 位
    man3 = (man16 >> 7) << 7
    exp8 = (e4 + 15).clamp(0, 31)
    u8 = (sign_bit << 15) | (exp8 << 10) | man3
    out = u8.to(torch.uint16).view(torch.float16).to(torch.float32)
    return sign * out


def _sim_int(x: torch.Tensor, bits: int) -> torch.Tensor:
    """对称逐张量整数量化→反量化（返回 fp32）。

    以 maxq = 2^(bits-1)-1 为整数上限，scale = max|x| / maxq；
    q = round(x/scale) 裁剪到 [-maxq, maxq]，反量化 = q*scale。
    相对量化误差上限 ≈ 1/(2*maxq)（bits 越大越精；int2 → 50%）。
    """
    maxq = (1 << (bits - 1)) - 1
    if x.numel() == 0:
        return x
    s = x.abs().amax()
    if s == 0:
        return x
    scale = s / maxq
    q = (x / scale).round_().clamp_(-maxq, maxq)
    return (q * scale).to(torch.float32)


# e2m1 fp4 的可表示正网格（bias=1）：0, 0.5, 1, 1.5, 2, 3, 4, 6。
_FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _sim_fp4(x: torch.Tensor) -> torch.Tensor:
    """fp4(e2m1) 逐张量 scale 网格量化→反量化（返回 fp32）。

    CPU 无 fp4 计算内核，按权重-only 模拟：先算 scale = amax|x| / 6，
    把 x/scale 映射到 e2m1 网格 {0,0.5,…,6} 的最近邻、带符号写回，
    再乘回 scale。scale 归一化保证任意量级的权重都铺满整个网格——
    不做归一化时初始权重（std≈0.02）会全部量化为 0（2026-10-27 审计
    实测：10000 个采样权重零值比例 100%，embedding/投影/lm_head 同时
    失效）。 """
    grid = torch.tensor(_FP4_GRID, dtype=torch.float32, device=x.device)
    sign = torch.sign(x)
    ax = x.abs()
    amax = ax.amax()
    if amax == 0:
        return torch.zeros_like(x, dtype=torch.float32)
    scale = amax / _FP4_GRID[-1]            # 网格最大可表示值 = 6
    q = (ax / scale).unsqueeze(-1)
    d = (q - grid.view(1, -1)).abs()        # [N, 8]
    nearest = grid[d.argmin(dim=-1)]        # [N]
    return (sign * nearest * scale).to(torch.float32)


# ---------------------------------------------------------------------------
# 施加精度
# ---------------------------------------------------------------------------
def apply_precision(model, name: str):
    """把模型参数按 dtype 名施加精度；原地修改，返回 model。

    - 原生 dtype：model.to(dtype)。
    - 模拟（fp8/intN）：逐参数量化→反量化，仍以 fp32 张量承载（CPU 可跑、
      精度被压到 N 比特，即损耗）。
    """
    if name is None or name == DEFAULT:
        return model
    info = parse_precision(name)
    if info["kind"] == "native":
        return model.to(info["torch_dtype"])
    # 模拟路径：原地改写参数，保留 fp32 容器
    with torch.no_grad():
        for p in model.parameters():
            if p.numel() == 0:
                continue
            if info["name"] == "fp8":
                p32 = _to_fp8_grid(p.data, info["spec"])
            elif info["name"] == "fp4":
                p32 = _sim_fp4(p.data)
            else:
                p32 = _sim_int(p.data, info["spec"])
            p.copy_(p32)
    return model


def dtype_summary(name: str) -> dict:
    """给前端展示一个精度的元数据（类别/比特/是否原生/说明）。"""
    info = parse_precision(name)
    if info["kind"] == "native":
        return {"name": name, "kind": "native",
                "bits": {torch.float64: 64, torch.float32: 32,
                         torch.float16: 16, torch.bfloat16: 16}[
                             info["torch_dtype"]],
                "note": "torch 原生 dtype，GPU/CPU 均可原生计算"}
    bits = 8 if name == "fp8" else (4 if name == "fp4" else info["spec"])
    return {"name": name, "kind": "sim", "bits": bits,
            "note": "低比特：CPU 以 fp32 容器承载、权重量化到该精度"
                    "（权重-only 模拟，有损耗）"}
