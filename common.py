# ===========================================================================
# GARGANTUA v1 — common.py：跨模块共享的检查点 IO 与运行参数校验
# ---------------------------------------------------------------------------
# 存在意义（2026-10-20 审查修复）：
#   此前 init.py / server.py 各自写 config.json + safetensors，而 train.py /
#   infer.py / auto_train.py 的 main() 一律硬编码 Gargantua() 默认 block 数，
#   导致用非默认 block 数创建的模型无法被 CLI 训练/推理加载
#   （load_state_dict 直接 RuntimeError，已实测复现）。
#   本模块把「模型目录契约」收敛成单一实现，所有入口共用：
#     models/NAME/
#       ├── model.safetensors   权重（safetensors）
#       └── config.json         {"n_enc_blocks": int, "n_dec_blocks": int, ...}
# ===========================================================================
from __future__ import annotations

import json
import os
import re
import sys
from typing import Optional

import torch
from safetensors.torch import load_file, save_file

import spec

CKPT_NAME = "model.safetensors"
CONFIG_NAME = "config.json"

# block 数合法范围（与 server.py 的 Web 表单 clamp 一致）
MAX_BLOCKS = 64


def _warn(msg: str) -> None:
    print(f"[common] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 模型名校验（init.py CLI 与 server.py Web 入口共用同一套规则）
# ---------------------------------------------------------------------------
# Windows 非法文件名字符 + 控制字符 + 首尾空白/点；不含路径分隔符即无穿越风险
_BAD_NAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def valid_model_name(name) -> Optional[str]:
    """返回 None 表示合法；否则返回人类可读的拒绝原因。"""
    if not name or not isinstance(name, str):
        return "模型名不能为空"
    if len(name) > 64:
        return "模型名过长（>64 字符）"
    if name != name.strip() or name in (".", ".."):
        return "模型名不能为 '.'/'..' 或含首尾空白"
    if _BAD_NAME_RE.search(name):
        return "模型名不能包含 < > : \" / \\ | ? * 或控制字符"
    if name.split(".")[0].upper() in (
            "CON", "PRN", "AUX", "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10))):
        return "模型名不能使用 Windows 保留设备名"
    return None


# ---------------------------------------------------------------------------
# config.json 读写
# ---------------------------------------------------------------------------
def read_config(model_dir: str) -> dict:
    """读模型目录的 config.json；缺失/损坏时返回 {}（调用方用默认值容错）。"""
    path = os.path.join(model_dir, CONFIG_NAME)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception as e:
        _warn(f"config.json 读取失败（{e}），按默认 block 数构造")
        return {}


def _block_count(cfg: dict, key: str, default: int) -> int:
    v = cfg.get(key)
    if v is None:
        return default
    try:
        v = int(v)
    except (TypeError, ValueError):
        _warn(f"config.json {key}={v!r} 非法，使用默认值 {default}")
        return default
    if not (1 <= v <= MAX_BLOCKS):
        _warn(f"config.json {key}={v} 越界 [1,{MAX_BLOCKS}]，使用默认值 {default}")
        return default
    return v


def build_model(model_dir: str):
    """按模型目录的 config.json 构造 Gargantua 并加载权重。

    这是 train / infer / auto_train / server 共用的唯一构建入口，
    保证「创建时的 block 数」与「加载时的 block 数」永远一致。
    """
    from model import Gargantua          # 延迟 import：避免 common → model 循环
    cfg = read_config(model_dir)
    m = Gargantua(n_enc_blocks=_block_count(cfg, "n_enc_blocks",
                                            spec.N_ENC_BLOCKS),
                  n_dec_blocks=_block_count(cfg, "n_dec_blocks",
                                            spec.N_DEC_BLOCKS))
    ckpt = os.path.join(model_dir, CKPT_NAME)
    if os.path.exists(ckpt):
        m.load_state_dict(load_file(ckpt))
    return m


def save_checkpoint(model_dir: str, model, config: dict) -> None:
    """权重 + config.json 原子落盘（先写临时文件再 replace，防半成品）。"""
    os.makedirs(model_dir, exist_ok=True)
    tmp = os.path.join(model_dir, CKPT_NAME + ".tmp")
    save_file(model.state_dict(), tmp)
    os.replace(tmp, os.path.join(model_dir, CKPT_NAME))
    tmp_cfg = os.path.join(model_dir, CONFIG_NAME + ".tmp")
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    os.replace(tmp_cfg, os.path.join(model_dir, CONFIG_NAME))


# ---------------------------------------------------------------------------
# 运行参数校验（train / auto_train / server 共用）
# ---------------------------------------------------------------------------
def validate_block_size(block_size) -> int:
    """窗口大小必须是偶数且 ≥2（user/model 各半；奇数会令 s_dec 与
    dec_in 长度不一致，前向必崩——2026-10-20 实测复现后收敛到此）。"""
    try:
        b = int(block_size)
    except (TypeError, ValueError):
        raise ValueError(f"block_size 必须是整数: {block_size!r}")
    if b < 2 or b > 1024:
        raise ValueError(f"block_size 必须在 [2, 1024]: {b}")
    if b % 2 != 0:
        raise ValueError(f"block_size 必须为偶数（user/model 各半）: {b}")
    return b


def validate_steps(steps) -> int:
    """训练步数：≥1。0 步会让 Trainer 的空 _loss_tail 触发 IndexError。"""
    try:
        s = int(steps)
    except (TypeError, ValueError):
        raise ValueError(f"steps 必须是整数: {steps!r}")
    if not (1 <= s <= 1_000_000):
        raise ValueError(f"steps 必须在 [1, 1000000]: {s}")
    return s


def make_batch(ids, i: int, block_size: int):
    """训练窗口切分——train.py 与 auto_train.py 共用的唯一实现。

    返回 (x_in, s_in, dec_in, s_dec, y)：
      user 段   = ids[i : i+half]        （encoder 输入）
      model 段  = ids[i+half : i+B]      （decoder 预测目标）
      dec_in    = model 段左移一位，首 token 用 user 段尾 token 播种
    调用方保证 len(ids) >= B + 2 且 0 <= i <= len(ids) - B - 2。
    """
    half = block_size // 2
    x_in = torch.tensor([ids[i:i + half]])
    s_in = torch.arange(half).unsqueeze(0)
    y = torch.tensor([ids[i + half:i + block_size]])
    dec_in = torch.tensor([[ids[i + half - 1]] + ids[i + half:i + block_size - 1]])
    s_dec = torch.arange(half, half + half).unsqueeze(0)
    return x_in, s_in, dec_in, s_dec, y

