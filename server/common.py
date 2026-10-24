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

# ---------------------------------------------------------------------------
# 上下文长度配置（2026-10-23 新功能：建模时可选）
# ---------------------------------------------------------------------------
# 语义（spec 第 3 节）：KV 序列流 = 完整区（未压缩，ctx_full 个原始
# token）｜压缩区（ctx_compress 个原始 token 的容量）。压缩区对外有两个
# 【同源视图】：CSA（csa_ratio:1 可学习压缩 + 完整区尾部 csa_local_window
# 局部窗）与 HCA（hca_ratio:1 可学习压缩 + 全局）。两层压缩的是同一段
# 内容，只是压缩率与注意力范围不同。压缩/滑窗/因果均以时间步为原子。
CTX_FULL_MAX = 65536             # 完整区容量上限（token）
CTX_COMPRESS_MAX = 1 << 20       # 压缩区容量上限（原始 token）
CTX_RATIO_MAX = 256              # 压缩比上限
CTX_KEYS = ("ctx_full", "ctx_compress", "csa_ratio", "hca_ratio",
            "csa_local_window")


def _ctx_int(v, name: str) -> int:
    """严格整数转换：拒 bool / 拒非整数值 float / 拒非数字字符串。"""
    if isinstance(v, bool):
        raise ValueError(f"{name} 必须是整数，不能是布尔值: {v!r}")
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not v.is_integer():
            raise ValueError(f"{name} 必须是整数: {v!r}")
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if s.lstrip("+-").isdigit():
            return int(s)
    raise ValueError(f"{name} 必须是整数: {v!r}")


def validate_ctx_config(ctx_full=None, ctx_compress=None, csa_ratio=None,
                        hca_ratio=None, csa_local_window=None) -> dict:
    """校验并规范化上下文长度配置；None 的项取 spec 默认值。

    返回可直接写进 config.json 的 dict（五个键齐全）。
    非法值抛 ValueError（HTTP 层映射为 400）。 """
    cf = (spec.CTX_FULL if ctx_full is None
          else _ctx_int(ctx_full, "ctx_full"))
    cc = (spec.CTX_COMPRESS if ctx_compress is None
          else _ctx_int(ctx_compress, "ctx_compress"))
    cr = (spec.CSA_RATIO if csa_ratio is None
          else _ctx_int(csa_ratio, "csa_ratio"))
    hr = (spec.HCA_RATIO if hca_ratio is None
          else _ctx_int(hca_ratio, "hca_ratio"))
    cw = (spec.CSA_LOCAL_WINDOW if csa_local_window is None
          else _ctx_int(csa_local_window, "csa_local_window"))
    if not (1 <= cf <= CTX_FULL_MAX):
        raise ValueError(f"ctx_full 必须在 [1, {CTX_FULL_MAX}]: {cf}")
    if not (0 <= cc <= CTX_COMPRESS_MAX):
        raise ValueError(f"ctx_compress 必须在 [0, {CTX_COMPRESS_MAX}]: {cc}")
    if not (2 <= cr <= CTX_RATIO_MAX):
        raise ValueError(f"csa_ratio 必须在 [2, {CTX_RATIO_MAX}]: {cr}")
    if not (2 <= hr <= CTX_RATIO_MAX):
        raise ValueError(f"hca_ratio 必须在 [2, {CTX_RATIO_MAX}]: {hr}")
    if hr < cr:
        # HCA 是「重压缩」视图，压缩比不应低于 CSA，否则两层语义倒挂
        raise ValueError(f"hca_ratio({hr}) 不能小于 csa_ratio({cr})："
                         f"HCA 是重压缩视图")
    if not (1 <= cw <= cf):
        raise ValueError(f"csa_local_window 必须在 [1, ctx_full={cf}]: {cw}")
    return {"ctx_full": cf, "ctx_compress": cc, "csa_ratio": cr,
            "hca_ratio": hr, "csa_local_window": cw}


def _ctx_from_cfg(cfg: dict) -> dict:
    """从 config.json 读上下文配置：逐字段容错（坏值告警 + 回退默认），
    与 _block_count 的策略一致；跨字段约束（hca≥csa、window≤full）
    不满足时两个比例一起回退 spec 默认。 """
    def _get(key, default, lo, hi):
        v = cfg.get(key)
        if v is None:
            return default
        try:
            v = _ctx_int(v, key)
        except ValueError:
            _warn(f"config.json {key}={v!r} 非法，使用默认值 {default}")
            return default
        if not (lo <= v <= hi):
            _warn(f"config.json {key}={v} 越界，使用默认值 {default}")
            return default
        return v

    cf = _get("ctx_full", spec.CTX_FULL, 1, CTX_FULL_MAX)
    cc = _get("ctx_compress", spec.CTX_COMPRESS, 0, CTX_COMPRESS_MAX)
    cr = _get("csa_ratio", spec.CSA_RATIO, 2, CTX_RATIO_MAX)
    hr = _get("hca_ratio", spec.HCA_RATIO, 2, CTX_RATIO_MAX)
    cw = _get("csa_local_window", spec.CSA_LOCAL_WINDOW, 1, cf)
    if hr < cr or cw > cf:
        _warn(f"config.json 上下文跨字段约束不满足"
              f"（hca={hr} < csa={cr} 或 window={cw} > full={cf}），"
              f"压缩比与窗口回退 spec 默认")
        cr, hr = spec.CSA_RATIO, spec.HCA_RATIO
        cw = min(spec.CSA_LOCAL_WINDOW, cf)
    return {"ctx_full": cf, "ctx_compress": cc, "csa_ratio": cr,
            "hca_ratio": hr, "csa_local_window": cw}


def read_ctx_config(model_dir: str) -> dict:
    """模型目录的上下文长度配置（config.json 驱动，缺失即默认）。 """
    return _ctx_from_cfg(read_config(model_dir))


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
    2026-10-23：上下文长度（ctx_full/ctx_compress/CSA/HCA）同样由
    config.json 驱动，缺省回退 spec 默认值。 """
    from model import Gargantua          # 延迟 import：避免 common → model 循环
    cfg = read_config(model_dir)
    m = Gargantua(n_enc_blocks=_block_count(cfg, "n_enc_blocks",
                                            spec.N_ENC_BLOCKS),
                  n_dec_blocks=_block_count(cfg, "n_dec_blocks",
                                            spec.N_DEC_BLOCKS),
                  **_ctx_from_cfg(cfg))
    ckpt = os.path.join(model_dir, CKPT_NAME)
    if os.path.exists(ckpt):
        try:
            m.load_state_dict(load_file(ckpt))
        except RuntimeError as e:
            # 2026-10-23 审计修复（P2）：config 与权重不一致（block 数/
            # 上下文配置被手改）此前抛 RuntimeError → HTTP 500 且报文
            # 是大段 missing keys；转成 ValueError（→400）并给结论
            first = str(e).splitlines()[0] if str(e) else "结构不匹配"
            raise ValueError(
                f"模型目录的 config.json 与权重不匹配"
                f"（block 数或上下文配置被修改过）: {first}")
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

