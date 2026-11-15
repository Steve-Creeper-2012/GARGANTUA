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
#       ├── config.json    manifest：{"weights": ..., "weights_discrete_embed":
#       │                  ..., "weights_continuous_embed": ..., 
#       │                  "vocab_size": int, "vocab_source": str, ...}
#       ├── weights/       三个 safetensors：主权重 / 离散 embed / 连续 embed
#       └── vocab/         vocab.vfc + config.json（每模型一份独立副本）
#   2026-10-30 词表契约重构：权重按「使用方式」拆成三个文件（见下方
#   WEIGHTS_DIRNAME 一节），词表从 vocab.txt 升级为 vocab/ 目录。
# ===========================================================================
from __future__ import annotations

import contextlib
import glob
import json
import math
import os
import re
import shutil
import sys
import time
import uuid
from typing import Optional

import torch
from safetensors.torch import load_file, save_file

import spec

CONFIG_NAME = "config.json"
# 初始化互斥锁文件（H4）：覆盖「冲突检查—写入—发布」全程，跨进程生效。
# init.py 子进程与 server 的 train start 用同一把锁，杜绝「初始化写到一半
# 训练已经开始读」的权重/config 不配套。
INIT_LOCK_NAME = ".model-init.lock"

# ---------------------------------------------------------------------------
# 模型文件夹布局（2026-10-30 词表契约重构）
# ---------------------------------------------------------------------------
# models/NAME/
#   config.json                          manifest：唯一事实源
#   weights/
#     model.<uuid>.safetensors            主权重（transformer 主体）
#     embed_discrete.<uuid>.safetensors    离散 token 的 embedding（wte）
#     embed_continuous.<uuid>.safetensors  连续 token 的 embedding（视觉/音频投影）
#   vocab/                               词表（即使与别人共用也各存一份副本）
#     vocab.vfc
#     config.json
#
# 为什么要拆三个文件：离散 embedding 是查表（forward 只取一行，放在 CPU
# RAM 里开销可接受），连续投影（视觉/音频 linear）每次前向都要真算矩阵乘、
# 必须留在 VRAM，主体权重又是优化器视角的另一回事。合成一个 blob 之后三
# 者被迫同生共死，加载侧无法按设备分层放置。
#
# 为什么 vocab 要整目录复制：多个模型复用同一个 vocabs/<preset>/ 时，用户
# 在前端删掉那个共享词表，依赖它的所有模型会一起报废（权重里 embedding
# 行数对得上一个都不剩）。每个模型各存一份副本，删源不影响已建模型。
WEIGHTS_DIRNAME = "weights"
VOCAB_DIRNAME = "vocab"
VOCAB_VFC_NAME = "vocab.vfc"
MAIN_WEIGHTS_PREFIX = "model"
DISCRETE_EMBED_PREFIX = "embed_discrete"
CONTINUOUS_EMBED_PREFIX = "embed_continuous"
# 全部允许出现在 weights/ 下的权重文件前缀——回收旧文件时按这张表枚举，
# 新增一类权重只需在这里加一个前缀。
CKPT_PREFIXES = (MAIN_WEIGHTS_PREFIX, DISCRETE_EMBED_PREFIX,
                 CONTINUOUS_EMBED_PREFIX)
# config.json 里指向三个权重文件的键名
WEIGHTS_KEYS = ("weights", "weights_discrete_embed",
                "weights_continuous_embed")

# state_dict 分组规则（按 key 前缀匹配，不硬编码完整 key 列表——将来新增
# 离散/连续 embedding 子模块只要沿用同一前缀就不会漏）:
#   离散 embedding：wte（词表 token 查表，forward 只取一行）
#   连续 embedding：视觉 patch 投影 + 2D 坐标嵌入、音频波形投影
#   （server/model.py 里的确切子模块名：visual_proj / visual_coord_x /
#    visual_coord_y / audio_proj）
_DISCRETE_EMBED_KEY_PREFIXES = ("wte.",)
_CONTINUOUS_EMBED_KEY_PREFIXES = ("visual_proj.", "visual_coord_x.",
                                  "visual_coord_y.", "audio_proj.")

# 每模型词表目录（2026-10-30 起）：models/NAME/vocab/{vocab.vfc, config.json}，
# 建模时从 vocabs/<name>/ 整目录复制，模型之间互不牵连；没有该目录才回落到
# 全局词表 service/tokens/{vocab.vfc, config.json}。
# 见 tokens/vfc.py 的 .vfc 契约（一行一个 token，ID 严格连续 0..N-1）。

# 推理 API/CLI 层硬上限：CPU 上单步是全序列重解码，2048 已是宽松保护值。
# 定义在 common（server 与服务进程共用），避免 server 为拿一个常量去
# import 重量级的 infer 模块。
MAX_NEW_TOKENS = 2048

# prompt token 数上限（2026-10-27 H10）：编码后的第二道闸门（第一道是
# server 的 MAX_PROMPT_CHARS）。超长 prompt 在进 encoder 之前就被拒，
# 避免全序列编码的内存/CPU 尖峰；8192 已远大于默认完整区 1024，
# 超出部分本就会被滑窗淘汰。
MAX_PROMPT_TOKENS = 8192

# block 数合法范围（与 server.py 的 Web 表单 clamp 一致）
MAX_BLOCKS = 64

# ---------------------------------------------------------------------------
# 上下文长度配置（2026-10-23 新功能：建模时可选）
# ---------------------------------------------------------------------------
# 语义（spec 第 3 节）：KV 序列流 = 完整区（未压缩，ctx_full 个原始
# token）｜压缩区（ctx_compress 个原始 token 的容量）。压缩区对外有两个
# 【同源视图】：CSA（csa_ratio:1 检索视图）与 HCA（hca_ratio:1 全局概括
# 视图），两者都读【压缩段 + 全部完整段】——处理未压缩 KV 时就是标准
# MLA 逻辑（2026-10-25 删除 CSA 局部窗）。压缩/滑窗/因果均以时间步为原子。
CTX_FULL_MAX = 65536             # 完整区容量上限（token）
CTX_COMPRESS_MAX = 1 << 20       # 压缩区容量上限（原始 token）
CTX_RATIO_MAX = 256              # 压缩比上限
CTX_KEYS = ("ctx_full", "ctx_compress", "csa_ratio", "hca_ratio")


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
                        hca_ratio=None) -> dict:
    """校验并规范化上下文长度配置；None 的项取 spec 默认值。

    返回可直接写进 config.json 的 dict（四个键齐全）。
    非法值抛 ValueError（HTTP 层映射为 400）。

    2026-10-25：csa_local_window 已删除——CSA 的保留区就是完整 KV 区，
    不再有独立的局部窗参数。旧 config.json 里的该键被忽略。
    """
    cf = (spec.CTX_FULL if ctx_full is None
          else _ctx_int(ctx_full, "ctx_full"))
    cc = (spec.CTX_COMPRESS if ctx_compress is None
          else _ctx_int(ctx_compress, "ctx_compress"))
    cr = (spec.CSA_RATIO if csa_ratio is None
          else _ctx_int(csa_ratio, "csa_ratio"))
    hr = (spec.HCA_RATIO if hca_ratio is None
          else _ctx_int(hca_ratio, "hca_ratio"))
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
    return {"ctx_full": cf, "ctx_compress": cc, "csa_ratio": cr,
            "hca_ratio": hr}


def model_vocab_dir(model_dir: str) -> str:
    """模型目录的词表目录 models/NAME/vocab（不一定存在）。"""
    return os.path.join(model_dir, VOCAB_DIRNAME)


def install_model_vocab(model_dir: str, src_dir: str) -> Optional[int]:
    """把词表目录 src 的 vocab.vfc + config.json 复制进 models/NAME/vocab/。

    返回该词表声明的 vocab_size（取自 vocab/config.json；读不到返回
    None）。即使多个模型共用同一个词表预设也各自复制一份——共享词表被
    用户在别处删掉，依赖它的模型会一起报废（权重里 embedding 的行数对
    应的就是这份词表）。复制是纯文件级的，不解析 .vfc 内容：格式归
    tokens/vfc.py 解释，本模块只负责「模型目录里必须有一份」。
    """
    dst = model_vocab_dir(model_dir)
    os.makedirs(dst, exist_ok=True)
    for fname in (VOCAB_VFC_NAME, CONFIG_NAME):
        src = os.path.join(src_dir, fname)
        if not os.path.isfile(src):
            _warn(f"词表源文件缺失，跳过: {src}")
            continue
        _atomic_copy(src, os.path.join(dst, fname))
    try:
        with open(os.path.join(dst, CONFIG_NAME), encoding="utf-8") as f:
            v = json.load(f).get("vocab_size")
    except Exception:
        return None
    # 正文是词表长度的唯一事实源：config 可能过期（用户手工改过 vfc 但没
    # 走 _sync_config）。能解析就以正文条数为准，解析失败才退回 config 声明
    # ——长度错了会造出一个 wte 行数不对的模型，几步训练之后才炸。
    try:
        from tokens import vfc as _vfc
        n = len(_vfc.read_vfc(os.path.join(dst, VOCAB_VFC_NAME)))
        if n >= 1:
            return n
    except Exception as e:
        _warn(f"{dst}/{VOCAB_VFC_NAME} 解析失败，改用 config 的 vocab_size: {e}")
    try:
        return _vocab_size_int(v)
    except ValueError:
        _warn(f"{dst}/{CONFIG_NAME} 的 vocab_size={v!r} 非法")
        return None


def _atomic_copy(src: str, dst: str) -> None:
    """tmp + fsync + replace 复制文件（复制过程中崩溃不会留下半个词表）。"""
    tmp = f"{dst}.{uuid.uuid4().hex}.tmp"
    try:
        with open(src, "rb") as fi:
            with open(tmp, "wb") as fo:
                shutil.copyfileobj(fi, fo)
                fo.flush()
                os.fsync(fo.fileno())
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _vocab_size_int(v) -> int:
    """词表长度必须是 ≥1 的整数（bool 拒绝；非整数 float 拒绝，别截断）。"""
    if isinstance(v, bool):
        raise ValueError(f"必须是整数，不能是布尔值: {v!r}")
    if isinstance(v, int):
        n = v
    elif isinstance(v, float):
        if not v.is_integer():
            raise ValueError(f"必须是整数: {v!r}")
        n = int(v)
    else:
        raise ValueError(f"必须是整数: {v!r}")
    if n < 1:
        raise ValueError(f"必须 ≥ 1: {n}")
    return n


def _wte_rows(model_dir: str) -> Optional[int]:
    """已有权重里 wte.weight 的行数 = 该模型实际在用的词表长度。

    给「config 里没写 vocab_size 的旧目录」兜底：权重自己是词表长度的
    最终事实源，比任何外部声明都准。safetensors 的 safe_open 只读文件
    头（几百字节），不必把几百 MB 的 embedding 读进内存。
    """
    from safetensors import safe_open
    cfg = read_config(model_dir)
    candidates = (_manifest_path(model_dir,
                                 cfg.get(WEIGHTS_KEYS[1])),   # 新布局：先查分散式
                  resolve_weights_path(model_dir))            # 旧布局：单文件
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            with safe_open(path, framework="pt") as f:
                if "wte.weight" in list(f.keys()):
                    return int(f.get_slice("wte.weight").get_shape()[0])
        except Exception as e:
            _warn(f"{path} 读取 wte.weight 形状失败（{e}）")
    return None


def model_vocab_size(model_dir: str) -> int:
    """模型词表长度（优先级从高到低）：

         1. models/NAME/vocab/config.json 的 "vocab_size"（词表自己声明）
         2. models/NAME/config.json 的 "vocab_size"（模型创建时记下的）
         3. 已有权重的 wte.weight 行数（旧目录没有前两项时的唯一可靠来源）
         4. spec 默认（老 spec 的 VOCAB_SIZE_PADDED / VOCAB_SIZE；都没有 →
            抛 ValueError，因为「默认多大」在新契约下是未定义问题）

    任一层读取失败 / 类型非法都只是告警并继续向下一层回退——词表长度是
    模型能不能加载的前提，不能因为某一个文件写错就让整个目录报废；但每
    一层都必须说出来，否则排障时不知道长度到底从哪来的。 """
    for path, label in ((os.path.join(model_vocab_dir(model_dir),
                                      CONFIG_NAME), "vocab/config.json"),
                        (os.path.join(model_dir, CONFIG_NAME),
                         "config.json")):
        try:
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
        except FileNotFoundError:
            continue
        except Exception as e:
            _warn(f"{label} 读取失败（{e}），词表长度向下一层回退")
            continue
        if not isinstance(cfg, dict):
            _warn(f"{label} 不是 JSON 对象，词表长度向下一层回退")
            continue
        if cfg.get("vocab_size") is None:
            continue
        try:
            return _vocab_size_int(cfg["vocab_size"])
        except ValueError:
            _warn(f"{label} 的 vocab_size={cfg['vocab_size']!r} 非法，"
                  f"词表长度向下一层回退")
    rows = _wte_rows(model_dir)
    if rows:
        return rows
    from model import default_vocab_size   # 延迟 import：避免 common → model 循环
    return default_vocab_size()


def _ctx_from_cfg(cfg: dict) -> dict:
    """从 config.json 读上下文配置：逐字段容错（坏值告警 + 回退默认），
    与 _block_count 的策略一致；跨字段约束（hca≥csa）不满足时两个比例
    一起回退 spec 默认。旧配置里的 csa_local_window 直接忽略。 """
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
    if hr < cr:
        _warn(f"config.json 上下文跨字段约束不满足"
              f"（hca={hr} < csa={cr}），压缩比回退 spec 默认")
        cr, hr = spec.CSA_RATIO, spec.HCA_RATIO
    return {"ctx_full": cf, "ctx_compress": cc, "csa_ratio": cr,
            "hca_ratio": hr}


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


# ---------------------------------------------------------------------------
# enc/dec 权重模式（2026-11-04）——创建入口与加载入口共用的唯一规则
# ---------------------------------------------------------------------------
SHARE_CONFIG_KEY = "share_enc_dec"


def read_share_enc_dec(cfg: dict) -> bool:
    """config.json 的 share_enc_dec → bool；旧目录缺字段 = 分离权重。

    只认真布尔、0/1 与 "true"/"false"：其余一律告警并回退分离。把 "yes"
    或 2 这类手改值解释成共享，会让加载侧按错误的键集合 strict load，
    报一堆看不懂的 missing keys，还不如明确拒绝。"""
    v = cfg.get(SHARE_CONFIG_KEY, False)
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    if v is not None:
        _warn(f"config.json {SHARE_CONFIG_KEY}={v!r} 非法，按分离权重处理")
    return False


def read_d_model(cfg: dict) -> int:
    """config.json 的 d_model → 合法宽度（2026-11-04）。

    缺省 / 非法都回退 spec.D_MODEL，并告警——与 _block_count 的容错
    口径一致：单个字段写错不该让整个模型目录报废。"""
    v = cfg.get("d_model", spec.D_MODEL)
    try:
        return spec.check_d_model(v)
    except ValueError as e:
        _warn(f"config.json d_model={v!r} 非法，回退 {spec.D_MODEL}: {e}")
        return spec.D_MODEL


# ---------------------------------------------------------------------------
# MoMW 配置（2026-11-08）——创建入口与加载入口共用的唯一规则
# ---------------------------------------------------------------------------
# 与 d_model / share_enc_dec 一样：config.json 的 "momw" 是唯一开关，
# 旧目录没这个字段 → spec.momw_default_config()（enabled=False），
# Gargantua 根本不构造 MoMW 子模块，state_dict 与行为逐字不变。
def _momw_int(v, name: str, lo: int, hi: int) -> int:
    n = _ctx_int(v, name)          # 复用 ctx 的严格整数口径（拒 bool 等）
    if not (lo <= n <= hi):
        raise ValueError(f"momw.{name} 必须在 [{lo}, {hi}]: {n}")
    return n


def _momw_bool(v, name: str) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, int) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    raise ValueError(f"momw.{name} 必须是布尔值: {v!r}")


def _momw_float(v, name: str, lo: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ValueError(f"momw.{name} 必须是数字: {v!r}")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"momw.{name} 必须是数字: {v!r}")
    if not math.isfinite(f) or f < lo:
        raise ValueError(f"momw.{name} 必须是 ≥ {lo} 的有限数: {f}")
    return f


def validate_momw_config(v) -> dict:
    """规范化 MoMW 配置（唯一规则，创建入口 / CLI / 测试共用）。

    None / 缺失 → spec 默认（关闭）；bool 简写只切开关；dict 逐字段严格
    校验（拒 bool 冒充整数、NaN/inf、越界、未知字段）。非法抛 ValueError，
    HTTP 层映射 400 —— 与 validate_ctx_config 同一口径。

    **B0 > threshold 是硬约束**，不是风格建议：新层零状态 ⇒ 语义分接近 0，
    只有 bonus 能把它抬过 threshold；B0 ≤ threshold 时新层永远不会被召回，
    动态扩展就成了纯浪费内存的无效扩容（规格里点名的「无训练扩展死锁」）。

    2026-11-08 第二版：新增 6 个三阶段参数（expand_delta / gamma_k /
    eigen_interval / load_lambda / anchor_sample / eigen_min_writes），
    v1 的 10 个字段全部保留 → 旧 config.json 无需迁移即可通过校验。
    """
    cfg = spec.momw_default_config()
    if v is None:
        return cfg
    if isinstance(v, bool):
        cfg["enabled"] = v
        return cfg
    if not isinstance(v, dict):
        raise ValueError(f"momw 必须是对象或布尔值: {type(v).__name__}")
    unknown = set(v) - set(spec.MOMW_CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"momw 未知字段: {', '.join(sorted(unknown))}")
    e = v.get("enabled", cfg["enabled"])
    if isinstance(e, bool):
        cfg["enabled"] = e
    elif isinstance(e, int) and e in (0, 1):
        cfg["enabled"] = bool(e)
    elif isinstance(e, str) and e.strip().lower() in ("true", "false"):
        cfg["enabled"] = e.strip().lower() == "true"
    else:
        raise ValueError(f"momw.enabled 必须是布尔值: {e!r}")
    for key, rng in (("N", spec.MOMW_TOP_N_RANGE),
                     ("M", spec.MOMW_WINDOW_RANGE),
                     ("initial_layers", spec.MOMW_INITIAL_LAYERS_RANGE),
                     ("max_layers", spec.MOMW_MAX_LAYERS_RANGE),
                     ("eigen_interval", spec.MOMW_EIGEN_INTERVAL_RANGE),
                     ("anchor_sample", spec.MOMW_ANCHOR_SAMPLE_RANGE),
                     ("eigen_min_writes",
                      spec.MOMW_EIGEN_MIN_WRITES_RANGE)):
        if key in v:
            cfg[key] = _momw_int(v[key], key, *rng)
    for key, lo in (("tau", spec.MOMW_TAU_MIN),
                    ("threshold", spec.MOMW_POSITIVE_MIN),
                    ("B0", spec.MOMW_POSITIVE_MIN),
                    ("starve_threshold", spec.MOMW_POSITIVE_MIN),
                    ("saturation_limit", spec.MOMW_POSITIVE_MIN),
                    ("expand_delta", spec.MOMW_POSITIVE_MIN),
                    ("gamma_k", spec.MOMW_GAMMA_K_MIN)):
        if key in v:
            cfg[key] = _momw_float(v[key], key, lo)
    if "load_lambda" in v:
        lam = _momw_float(v["load_lambda"], "load_lambda",
                          spec.MOMW_POSITIVE_MIN)
        if lam > spec.MOMW_LOAD_LAMBDA_MAX:
            raise ValueError(
                f"momw.load_lambda 必须 ≤ {spec.MOMW_LOAD_LAMBDA_MAX}："
                "软均衡不能盖过语义打分")
        cfg["load_lambda"] = lam
    # ---- v3：数据驱动阈值 / Oja / 门控 / 异步 / 持久化 ----
    if "threshold_mode" in v:
        mode = v["threshold_mode"]
        if not isinstance(mode, str) or \
                mode.strip().lower() not in spec.MOMW_THRESHOLD_MODES:
            raise ValueError(
                f"momw.threshold_mode 必须是 "
                f"{' / '.join(spec.MOMW_THRESHOLD_MODES)}: {mode!r}")
        cfg["threshold_mode"] = mode.strip().lower()
    if "quantile_window" in v:
        cfg["quantile_window"] = _momw_int(
            v["quantile_window"], "quantile_window",
            *spec.MOMW_QUANTILE_WINDOW_RANGE)
    if "quantile_p" in v:
        p = _momw_float(v["quantile_p"], "quantile_p",
                        spec.MOMW_QUANTILE_RANGE[0])
        if p > spec.MOMW_QUANTILE_RANGE[1]:
            raise ValueError(
                f"momw.quantile_p 必须 ≤ {spec.MOMW_QUANTILE_RANGE[1]}")
        cfg["quantile_p"] = p
    if "saturation_ratio" in v:
        r = _momw_float(v["saturation_ratio"], "saturation_ratio",
                        spec.MOMW_RATIO_MIN)
        cfg["saturation_ratio"] = r
    if "growth_ratio" in v:
        cfg["growth_ratio"] = _momw_float(
            v["growth_ratio"], "growth_ratio", spec.MOMW_POSITIVE_MIN)
    if "oja_eta" in v:
        eta = _momw_float(v["oja_eta"], "oja_eta",
                          spec.MOMW_OJA_ETA_RANGE[0])
        if eta > spec.MOMW_OJA_ETA_RANGE[1]:
            raise ValueError(
                f"momw.oja_eta 必须 ≤ {spec.MOMW_OJA_ETA_RANGE[1]}")
        cfg["oja_eta"] = eta
    if "fw_gate_c" in v:
        g = _momw_float(v["fw_gate_c"], "fw_gate_c",
                        spec.MOMW_FW_GATE_C_RANGE[0])
        if g > spec.MOMW_FW_GATE_C_RANGE[1]:
            raise ValueError(
                f"momw.fw_gate_c 必须 ≤ {spec.MOMW_FW_GATE_C_RANGE[1]}")
        cfg["fw_gate_c"] = g
    for key in ("async_write", "persist"):
        if key in v:
            cfg[key] = _momw_bool(v[key], key)
    if "persist_dtype" in v:
        d = v["persist_dtype"]
        if not isinstance(d, str) or \
                d.strip().lower() not in spec.MOMW_PERSIST_DTYPES:
            raise ValueError(
                f"momw.persist_dtype 必须是 "
                f"{' / '.join(spec.MOMW_PERSIST_DTYPES)}: {d!r}")
        cfg["persist_dtype"] = d.strip().lower()
    if cfg["initial_layers"] > cfg["max_layers"]:
        raise ValueError(
            f"momw.initial_layers({cfg['initial_layers']}) 不能超过 "
            f"max_layers({cfg['max_layers']})")
    if cfg["B0"] <= cfg["threshold"]:
        raise ValueError(
            f"momw.B0({cfg['B0']}) 必须 > threshold({cfg['threshold']})："
            "否则新层的 bonus 跨不过阈值，零状态层永远不会被召回")
    return cfg


def read_momw_config(cfg: dict) -> dict:
    """build_model 用：config 坏值只告警并按「关闭」处理，不让目录报废。

    与 read_d_model / _block_count 的容错口径一致：单个字段写错不该让整个
    模型目录无法加载；但必须把结论打出来，否则排障时不知道开关被谁关了。
    """
    v = cfg.get(spec.MOMW_CONFIG_KEY)
    if v is None:
        return spec.momw_default_config()
    # v1 → v2 迁移：v1 的 threshold/B0 是「能量范数」量级（1.0 / 2.0），
    # v2 改成余弦相似度（[-1,1]，0.2 / 0.6）。不迁移的话旧 config 会
    # 静默变成「永远不激活」。
    # 判定必须窄：余弦相似度恒 ≤ 1，所以只有 threshold ≥ 1 或 B0 ≥ 2
    # 才可能是 v1 遗留。若用「缺 v2 字段」判定，新建的
    # {"enabled": true} 会被误迁移，用户显式给的非法 B0 也会被悄悄覆盖
    # 成默认值（校验被绕过，坏值变成假阳性通过）。
    if isinstance(v, dict) and not any(
            k in v for k in ("gamma_k", "eigen_interval", "load_lambda")):
        def _numeric(x):
            return isinstance(x, (int, float)) and not isinstance(x, bool)
        legacy = ((_numeric(v.get("threshold")) and
                   float(v["threshold"]) >= 1.0) or
                  (_numeric(v.get("B0")) and float(v["B0"]) >= 2.0))
        if legacy:
            _warn("config.json momw 是 v1 配置（能量范数阈值），"
                  "已迁移到 v2 余弦打分默认值")
            v = dict(v)
            v["threshold"] = spec.MOMW_THRESHOLD
            v["B0"] = spec.MOMW_BONUS
    try:
        return validate_momw_config(v)
    except ValueError as e:
        _warn(f"config.json {spec.MOMW_CONFIG_KEY} 非法，按关闭处理: {e}")
        out = spec.momw_default_config()
        out["enabled"] = False
        return out


# ---------------------------------------------------------------------------
# Manifold Muon 配置（阶段 2）——与 MoMW 同款「缺字段即关闭」契约
# ---------------------------------------------------------------------------
# config.json 里 "manifold_muon" 是唯一开关。旧目录没这个字段 →
# spec.manifold_default_config()（enabled=False）；坏值只告警并按关闭处理，
# 绝不让单个目录报废。配置是普通 Python 属性，不进 state_dict，
# strict load 与三个 safetensors 契约逐字不变。
_MANIFOLD_MODES = ("stiefel", "dgram", "none")


def _manifold_bool(v, key: str) -> bool:
    # 只认真 bool：0/1/"false" 一律拒绝（bool("false") is True 的坑）。
    if not isinstance(v, bool):
        raise ValueError(
            f"manifold_muon.{key} 必须是布尔值(bool)，拒绝 0/1/字符串: {v!r}")
    return v


def _manifold_number(v, key: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"manifold_muon.{key} 必须是数字: {v!r}")
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"manifold_muon.{key} 必须是有限数: {v!r}")
    return f


def _manifold_int(v, key: str, lo: int, hi: int = None) -> int:
    # 严格整数：拒 bool / float（含 20.0）/ 字符串，避免类型悄悄漂移。
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"manifold_muon.{key} 必须是整数: {v!r}")
    if v < lo or (hi is not None and v > hi):
        if hi is not None:
            raise ValueError(f"manifold_muon.{key} 必须在 [{lo}, {hi}]: {v}")
        raise ValueError(f"manifold_muon.{key} 必须 ≥ {lo}: {v}")
    return v


def validate_manifold_config(v) -> dict:
    """规范化 Manifold Muon 配置（唯一规则，创建入口 / CLI / 测试共用）。

    - 缺的键用 spec.manifold_default_config() 补齐（用户可能只写 enabled）；
    - 已知键逐字段严格校验，非法抛 ValueError（说明键 / 收到值 / 期望）；
    - 未知键原样保留（不覆盖任何默认键，也不参与校验），保证 config 往返
      不丢用户字段；绝不 cfg.update(v) 整体灌入，避免坏值绕过校验；
    - 返回补齐后的新 dict，绝不修改入参。
    """
    cfg = spec.manifold_default_config()
    if not isinstance(v, dict):
        raise ValueError(
            f"manifold_muon 必须是对象(dict): {type(v).__name__}")
    if "enabled" in v:
        cfg["enabled"] = _manifold_bool(v["enabled"], "enabled")
    for key in ("attention", "mlp"):
        if key in v:
            mode = v[key]
            if not isinstance(mode, str) or mode not in _MANIFOLD_MODES:
                raise ValueError(
                    f"manifold_muon.{key} 必须是 "
                    f"{'/'.join(_MANIFOLD_MODES)}: {mode!r}")
            cfg[key] = mode
    if "K" in v:
        cfg["K"] = _manifold_int(v["K"], "K", 0, 1000)
    if "ns_steps" in v:
        cfg["ns_steps"] = _manifold_int(v["ns_steps"], "ns_steps", 1)
    if "max_matrices" in v:
        cfg["max_matrices"] = _manifold_int(
            v["max_matrices"], "max_matrices", 0)
    if "alpha" in v:
        a = _manifold_number(v["alpha"], "alpha")
        if a <= 0:
            raise ValueError(f"manifold_muon.alpha 必须 > 0: {a!r}")
        cfg["alpha"] = a
    if "lr" in v:
        lr = _manifold_number(v["lr"], "lr")
        if lr <= 0:
            raise ValueError(f"manifold_muon.lr 必须 > 0: {lr!r}")
        cfg["lr"] = lr
    if "momentum" in v:
        mom = _manifold_number(v["momentum"], "momentum")
        if not (0 <= mom < 1):
            raise ValueError(
                f"manifold_muon.momentum 必须在 [0, 1): {mom!r}")
        cfg["momentum"] = mom
    for key in ("exclude_kda", "exact_msign"):
        if key in v:
            cfg[key] = _manifold_bool(v[key], key)
    # 未知键原样保留：不同名，不会覆盖任何默认键；也不参与上面的校验。
    known = set(cfg)
    for key in v:
        if key not in known:
            cfg[key] = v[key]
    return cfg


def read_manifold_config(cfg: dict) -> dict:
    """build_model 用：config 坏值只告警并按「关闭」处理，不让目录报废。

    与 read_momw_config 完全同款容错口径：缺字段 → spec 默认（关闭）；
    非法 ValueError → _warn 后返回 enabled=False 的默认。
    """
    v = cfg.get(spec.MANIFOLD_CONFIG_KEY)
    if v is None:
        return spec.manifold_default_config()
    try:
        return validate_manifold_config(v)
    except ValueError as e:
        _warn(f"config.json {spec.MANIFOLD_CONFIG_KEY} 非法，按关闭处理: {e}")
        out = spec.manifold_default_config()
        out["enabled"] = False
        return out


# ---------------------------------------------------------------------------
# SOMN 配置（2026-11-14 主干接线）——与 MoMW / Manifold 同款容错契约
# ---------------------------------------------------------------------------
def read_somn_config(cfg: dict) -> dict:
    """build_model 用：读 config.json 的 "somn" 开关与调优字段。

    config.json 是配置的唯一事实源，Gargantua 不自己读文件。缺字段 →
    {"enabled": False}（主干零开销、state_dict 逐字不变）；坏值只告警并
    按关闭处理，绝不让单个模型目录报废。调优字段走 spec 白名单，避免
    未知键打穿 attach_somn(**kwargs)。
    """
    v = cfg.get(spec.SOMN_CONFIG_KEY)
    if v is None:
        return spec.somn_default_config()
    if isinstance(v, bool):
        return {"enabled": v}
    if not isinstance(v, dict):
        _warn(f"config.json {spec.SOMN_CONFIG_KEY} 必须是对象或布尔值，"
              "按关闭处理")
        return spec.somn_default_config()
    out = {"enabled": False}
    try:
        e = v.get("enabled", False)
        if isinstance(e, bool):
            out["enabled"] = e
        elif isinstance(e, int) and e in (0, 1):
            out["enabled"] = bool(e)
        elif isinstance(e, str) and e.strip().lower() in ("true", "false"):
            out["enabled"] = e.strip().lower() == "true"
        else:
            raise ValueError(f"somn.enabled 必须是布尔值: {e!r}")
        for key in spec.SOMN_INT_FIELDS:
            if key in v:
                n = _ctx_int(v[key], f"somn.{key}")
                if n < 0:
                    raise ValueError(f"somn.{key} 必须 ≥ 0: {n}")
                out[key] = n
        for key in spec.SOMN_BOOL_FIELDS:
            if key in v:
                b = v[key]
                if isinstance(b, bool):
                    out[key] = b
                elif isinstance(b, int) and b in (0, 1):
                    out[key] = bool(b)
                elif isinstance(b, str) and b.strip().lower() in ("true", "false"):
                    out[key] = b.strip().lower() == "true"
                else:
                    raise ValueError(f"somn.{key} 必须是布尔值: {b!r}")
        # 2026-11-14 复核修复：SOMN 内部还有自己的硬断言（EnergyElbowRouter
        # 的 elbow_alpha ∈ (0,1]、IndependentFWLayer 的 lam ∈ [0,1]），以及
        # eta·λ_max(C)<2 的稳定性上界 0.01*64/D。这些必须在这里就拦下，
        # 否则坏 config 会在 Gargantua.__init__ 的 attach_somn 抛 ValueError，
        # 直接把整个模型目录打崩（违背「坏值按关闭处理」的容错契约）。
        safe_eta = 0.01 * (64.0 / float(read_d_model(cfg)))
        for key in spec.SOMN_FLOAT_FIELDS:
            if key in v and v[key] is not None:
                f = float(v[key])
                if isinstance(v[key], bool) or not math.isfinite(f) or f < 0:
                    raise ValueError(f"somn.{key} 必须是 ≥ 0 的有限数: {v[key]!r}")
                if key == "elbow_alpha" and not (0.0 < f <= 1.0):
                    raise ValueError(f"somn.elbow_alpha 须在 (0,1]: {f}")
                if key == "lam" and f > 1.0:
                    raise ValueError(f"somn.lam 须在 [0,1]: {f}")
                if key == "eta" and not (0.0 < f <= safe_eta):
                    raise ValueError(
                        f"somn.eta 须在 (0,{safe_eta:g}]（0.01*64/d_model "
                        f"稳定性上界）: {f}")
                if key in ("min_score", "min_energy") and f > 1.0:
                    raise ValueError(
                        f"somn.{key} 须 ≤ 1（路由打分 ∈ [-1,1]）: {f}")
                out[key] = f
        unknown = set(v) - set(spec.SOMN_CONFIG_FIELDS)
        if unknown:
            _warn(f"config.json somn 忽略未知字段: "
                  f"{', '.join(sorted(unknown))}")
    except (TypeError, ValueError) as e:
        _warn(f"config.json {spec.SOMN_CONFIG_KEY} 非法，按关闭处理: {e}")
        return spec.somn_default_config()
    return out


def normalize_shared_blocks(enc, dec, share: bool):
    """建模入口共用的 block 数归一化（server.py 与 init.py 都走这里）。

    - 分离权重：原样返回（None 让 Gargantua 取 spec 默认 4 / 6）。
    - 共享权重：强制 enc == dec；只给一侧则另一侧跟随；两侧都没给则
      一起取 spec.SHARED_DEFAULT_BLOCKS（不能各取各的 spec 默认，那两个
      值不等，共享模型必须等深）。两侧都给但不等 → ValueError（HTTP 400）。
    """
    if not share:
        return enc, dec
    if enc is not None and dec is not None:
        if int(enc) != int(dec):
            raise ValueError(
                f"共享权重要求 enc blocks == dec blocks（收到 {enc} / {dec}）；"
                f"共享模式下两侧逐层一一绑定参数")
        return int(enc), int(dec)
    one = enc if enc is not None else dec
    if one is not None:
        return int(one), int(one)
    return spec.SHARED_DEFAULT_BLOCKS, spec.SHARED_DEFAULT_BLOCKS


def build_model(model_dir: str):
    """按模型目录的 config.json 构造 Gargantua 并加载权重。

    这是 train / infer / auto_train / server 共用的唯一构建入口，
    保证「创建时的 block 数」与「加载时的 block 数」永远一致。
    2026-10-23：上下文长度（ctx_full/ctx_compress/CSA/HCA）同样由
    config.json 驱动，缺省回退 spec 默认值。
    2026-10-30：词表长度同样由「模型目录里的词表配置」驱动
    （model_vocab_size）——build_model 被 infer/train 在 set_model_vocab()
    之前调用，所以它不能依赖进程里已加载的词表，必须自己读目录。 """
    from model import Gargantua          # 延迟 import：避免 common → model 循环
    cfg = read_config(model_dir)
    share = read_share_enc_dec(cfg)
    n_enc = _block_count(cfg, "n_enc_blocks", spec.N_ENC_BLOCKS)
    n_dec = _block_count(cfg, "n_dec_blocks", spec.N_DEC_BLOCKS)
    if share and n_enc != n_dec:
        # 共享模型的 config 被手改成不等深：先给结论，别落到
        # load_state_dict 的一堆 missing keys 里
        raise ValueError(
            f"共享权重模型的 config.json 要求 n_enc_blocks == n_dec_blocks"
            f"（读到 {n_enc} / {n_dec}）：配置与权重不匹配")
    # 2026-11-08：MoMW 配置同样由 config.json 驱动；旧目录缺字段 →
    # read_momw_config 返回 enabled=False，Gargantua 不构造子模块，
    # state_dict 一个 key 都不变（strict load 不受影响）。
    m = Gargantua(n_enc_blocks=n_enc, n_dec_blocks=n_dec,
                  vocab_size=model_vocab_size(model_dir),
                  share_enc_dec=share, d_model=read_d_model(cfg),
                  momw=read_momw_config(cfg),
                  # 2026-11-14：config.json 的 "somn.enabled" 是 SOMN 的
                  # 唯一开关；true 时 Gargantua.__init__ 显式 attach_somn，
                  # 旧目录缺字段 = 关闭（state_dict 零新增 key）。
                  somn=read_somn_config(cfg),
                  # 锚点用模型自己的词表副本编码类别代表文本（不依赖硬编码
                  # token ID；版本升级时全局 ID 位移也不影响）
                  vocab_dir=model_vocab_dir(model_dir),
                  **_ctx_from_cfg(cfg))
    sd = load_model_state_dict(model_dir)
    if sd:
        try:
            m.load_state_dict(sd)
        except RuntimeError as e:
            # 2026-10-23 审计修复（P2）：config 与权重不一致（block 数/
            # 上下文配置被手改）此前抛 RuntimeError → HTTP 500 且报文
            # 是大段 missing keys；转成 ValueError（→400）并给结论。
            # 2026-10-30：词表长度不一致（换过 vocab/ 但没重训）同样落到这里。
            first = str(e).splitlines()[0] if str(e) else "结构不匹配"
            raise ValueError(
                f"模型目录的 config.json 与权重不匹配"
                f"（block 数、上下文配置或词表长度被修改过）: {first}")
    # 2026-10-25：按 config["dtype"] 施加模型精度（fp64/32/16/bf16 原生；
    # fp8/int8/int4/int2 在 CPU 上以 fp32 容器 + 权重量化模拟——损耗但能跑）。
    # 坏值防御性回退 fp32 并告警，不让单个错 dtype 把服务进程打崩。
    dtype_name = cfg.get("dtype") or "fp32"
    try:
        from precision import apply_precision
        apply_precision(m, dtype_name)
    except Exception as e:
        _warn(f"config dtype={dtype_name!r} 施加失败，回退 fp32: {e}")
    finally:
        # 2026-11-04：model.to(dtype) 会替换 Parameter 对象，逐层共享的
        # 绑定可能因此断掉（共享模型会退化成两份独立权重、且 dec 侧的
        # 训练更新不再被保存）。转换前后数值相同，重建绑定无损。
        # hasattr 是为了兼容测试里替换掉的替身模型。
        if share and hasattr(m, "retie_shared_weights"):
            m.retie_shared_weights()
    # 2026-11-08：MoMW 池快照（独立子目录，不进 weights/config 契约）。
    # 有就按快照重建，没有/损坏就从零开始——它本来就是可重建的运行时数据。
    if getattr(m, "momw", None) is not None:
        try:
            m.momw.load_state(model_dir)
        except Exception as e:
            _warn(f"momw_state 恢复失败，按零状态启动: {e}")
    # Manifold Muon 配置是普通 Python 属性，不是 nn.Module / Parameter /
    # buffer —— 不进 state_dict，strict load 与三个 safetensors 契约不变。
    # 旧目录缺字段时 read_manifold_config 返回 enabled=False。
    m.manifold_config = read_manifold_config(cfg)
    return m


# ---------------------------------------------------------------------------
# 共享权重的 state_dict 键镜像（2026-11-04）
# ---------------------------------------------------------------------------
# spec.TIED_KEY_PAIRS 是唯一真源。save 侧剥掉 dec 侧 key（只落 enc 一份：
# safetensors 不允许两个 key 共享同一块 storage，且存两份会把省下的磁盘
# 又吃回去），load 侧按 config 把 enc 侧张量镜像成 dec 侧 key，让严格
# load_state_dict 拿到完整键集合。两边共用同一张前缀表。
def _tied_dec_key(k: str):
    """enc 侧 key → 对应的 dec 侧 key；不是绑定范围的返回 None。"""
    for enc_p, dec_p in spec.TIED_KEY_PAIRS:
        if k.startswith(enc_p):
            return dec_p + k[len(enc_p):]
    return None


def _dec_prefixes() -> tuple:
    return tuple(dec_p for _enc_p, dec_p in spec.TIED_KEY_PAIRS)


def strip_tied_dec_keys(sd: dict) -> dict:
    """共享模型落盘前：只保留 enc 侧那一份参数。"""
    dec_p = _dec_prefixes()
    return {k: v for k, v in sd.items() if not k.startswith(dec_p)}


def expand_tied_dec_keys(sd: dict) -> dict:
    """共享模型加载时：把 enc 侧张量**引用**镜像成 dec 侧 key。

    不 clone：内存里两个 key 指向同一张量，load_state_dict 对同一个
    Parameter copy_ 两次、值相同，无副作用；落盘仍只有一份。"""
    dec_p = _dec_prefixes()
    if any(k.startswith(dec_p) for k in sd):
        raise ValueError(
            "权重文件里已有 dec 侧参数，与 config.json 的 "
            f"{SHARE_CONFIG_KEY}=true 冲突（目录被手改过，或混用了"
            "分离/共享两套权重）")
    out = dict(sd)
    for k, v in sd.items():
        twin = _tied_dec_key(k)
        if twin is not None:
            out[twin] = v
    return out


# ---------------------------------------------------------------------------
# 权重分组（save / load 共用的唯一真相源，2026-10-30 词表契约重构）
# ---------------------------------------------------------------------------
def split_state_dict(sd: dict) -> tuple:
    """state_dict → (主权重, 离散 embedding, 连续 embedding)。

    三个分组必须【恰好覆盖】入参的每一个 key：不重（if/elif 天然互斥）、
    不漏（else 兜住其余全部）。save 侧按这个结果拆三个文件，load 侧再按
    同样的规则合回去——两边共用同一份规则才不会出现「写进去的参数读不
    回来」。断言把这条不变量固化下来，将来动前缀表时立刻暴露。
    """
    main, disc, cont = {}, {}, {}
    for k, v in sd.items():
        if k.startswith(_DISCRETE_EMBED_KEY_PREFIXES):
            disc[k] = v
        elif k.startswith(_CONTINUOUS_EMBED_KEY_PREFIXES):
            cont[k] = v
        else:
            main[k] = v
    if set(main) | set(disc) | set(cont) != set(sd) or len(sd) != \
            len(main) + len(disc) + len(cont):
        raise AssertionError("split_state_dict 分组必须不重不漏地覆盖所有 key")
    return main, disc, cont


# ---------------------------------------------------------------------------
# 检查点发布（2026-10-27 审计修复 M1/H4；2026-10-30 拆三文件）
# ---------------------------------------------------------------------------
# 旧实现先 replace 权重再写 config：进程在两次 replace 之间崩溃会留下
# 「新权重 + 旧 config」的结构不匹配目录。现在改为版本化快照 + manifest：
#   1. 权重按 split_state_dict 拆成三份，各自写进 weights/（tmp + fsync +
#      replace，单文件原子）；
#   2. config.json 作为 manifest 原子指向这三个文件（tmp + fsync + replace，
#      单文件原子）——(config, 三个权重) 通过单文件替换实现整体原子；
#   3. 回收 manifest 之外的旧权重文件。
# 兼容旧目录：config 无 "weights_discrete_embed" 时按「一个文件装全部」读。
def _atomic_save_file(sd: dict, path: str) -> None:
    """tmp + fsync + replace 写单个 safetensors（带 UUID，并发写互不覆盖）。

    save_file 自己 close 文件后就返回了，数据安全与否取决于 page cache 何
    时回写；这里重新打开同一个 inode fsync 一次，确保 replace 之前字节已
    在盘上——否则断电后会得到一个「manifest 指向了文件名、但内容还没落
    盘」的目录，比干脆没有这个文件更难排查。
    """
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        save_file(sd, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _manifest_path(model_dir: str, rel) -> Optional[str]:
    """config.json 里的相对路径 → 绝对路径；非字符串/绝对路径/含 ".." 一律
    拒绝（返回 None + 告警）。

    manifest 是被当路径用的字符串：一旦 config.json 被手改成
    "../../etc/passwd" 或某个绝对路径，拼接后的读权就跑出模型目录了。这
    里按「只允许向下的、干净的 POSIX 相对分量」收紧（不必要的安全加固，
    但成本一行：本来就要判断有没有值）。
    """
    if not isinstance(rel, str) or not rel:
        return None
    parts = rel.replace("\\", "/").split("/")
    if os.path.isabs(rel) or not parts or \
            any(p in ("", ".", "..") for p in parts):
        _warn(f"config.json 的权重路径非法，已忽略: {rel!r}")
        return None
    return os.path.join(model_dir, *parts)


def resolve_weights_path(model_dir: str) -> Optional[str]:
    """config.json 的 manifest → 主权重文件路径；manifest 缺失/非法/文件
    不存在时返回 None（调用方按「无权重」处理，不猜别的文件名）。"""
    p = _manifest_path(model_dir, read_config(model_dir).get("weights"))
    if p and os.path.exists(p):
        return p
    return None


def load_model_state_dict(model_dir: str) -> dict:
    """读模型目录的完整 state_dict：manifest 指向的三个权重文件合并。

    合并顺序无所谓——三个分组的 key 两两不相交（split_state_dict 的不变
    量）。缺少任何一份时该分组的参数保持随机初始化，最终的形状冲突由
    load_state_dict 的 strict 校验统一报出。没有可加载的权重时返回 {}。
    """
    main_path = resolve_weights_path(model_dir)
    if main_path is None:
        return {}
    sd = dict(load_file(main_path))
    cfg = read_config(model_dir)
    for key in WEIGHTS_KEYS[1:]:
        p = _manifest_path(model_dir, cfg.get(key))
        if p and os.path.exists(p):
            sd.update(load_file(p))
    # 2026-11-04：共享模型只落了 enc 侧一份，按 config 镜像出 dec 侧 key，
    # 让 build_model 的 strict load_state_dict 拿到完整键集合。
    if read_share_enc_dec(cfg):
        sd = expand_tied_dec_keys(sd)
    return sd


def _recycle_old_weights(model_dir: str, cfg: dict) -> None:
    """回收 manifest 之外的旧权重文件。

    每次 save 都生成新的 UUID 文件名（读者永远只读 manifest 指向的那一
    份，旧文件纯属垃圾）。不回收的话训练集每存一次就多一份几百 MB，很
    快撑爆磁盘。三个前缀全覆盖，漏一个前缀就等于漏一类文件。
    """
    keep = set()
    for key in WEIGHTS_KEYS:
        p = _manifest_path(model_dir, cfg.get(key))
        if p:
            keep.add(os.path.basename(p))
    wdir = os.path.join(model_dir, WEIGHTS_DIRNAME)
    for prefix in CKPT_PREFIXES:
        for old in glob.glob(os.path.join(wdir, f"{prefix}.*.safetensors")):
            if os.path.basename(old) not in keep:
                try:
                    os.unlink(old)
                except OSError:
                    pass


def model_exists(model_dir: str) -> bool:
    """模型目录是否包含可加载的模型（config.json + manifest 权重）。"""
    if not os.path.exists(os.path.join(model_dir, CONFIG_NAME)):
        return False
    return resolve_weights_path(model_dir) is not None


@contextlib.contextmanager
def model_init_lock(model_dir: str, timeout: float = 600.0,
                    stale_after: float = 1800.0):
    """跨进程初始化锁：覆盖「冲突检查—写入—发布」全程。

    init.py 子进程持锁建模落盘；server 的 train start 在同一把锁上
    等待/拒绝——训练绝不可能读到写了一半的模型。锁文件带 stale
    检测（持锁进程崩溃后 30 分钟自动回收）。"""
    os.makedirs(model_dir, exist_ok=True)
    lock_path = os.path.join(model_dir, INIT_LOCK_NAME)
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock_path)
            except OSError:
                age = stale_after + 1.0
            if age > stale_after:
                try:
                    os.unlink(lock_path)
                except OSError:
                    pass
                continue
            if time.time() > deadline:
                raise RuntimeError(
                    f"模型目录被占用（初始化进行中，>{timeout:.0f}s 未释放）: "
                    f"{model_dir}")
            time.sleep(0.2)
    try:
        yield
    finally:
        try:
            os.unlink(lock_path)
        except OSError:
            pass


def save_checkpoint(model_dir: str, model, config: dict) -> None:
    """权重（三个文件）+ config.json 以「版本化快照 + manifest」原子发布。

    config.json 是唯一事实源：它指向哪三个权重文件，读者就加载哪三个——
    单文件原子替换让 (config, weights) 永远自洽。临时文件带 UUID，并发
    写互不覆盖；写后回收旧权重。"""
    os.makedirs(model_dir, exist_ok=True)
    wdir = os.path.join(model_dir, WEIGHTS_DIRNAME)
    os.makedirs(wdir, exist_ok=True)
    sd = model.state_dict()
    # 2026-11-04：共享模型只落 enc 侧一份。两件事都不可省——safetensors
    # 不允许两个 key 共享 storage（会直接报错），而且重复落一份等于把
    # 共享省下来的磁盘又吃回去。config 是加载侧判断的唯一依据。
    tied = bool(getattr(model, "share_enc_dec", False))
    if tied:
        sd = strip_tied_dec_keys(sd)
    main_sd, disc_sd, cont_sd = split_state_dict(sd)
    cfg = dict(config)
    if tied or SHARE_CONFIG_KEY in cfg:
        cfg[SHARE_CONFIG_KEY] = tied
    plan = ((MAIN_WEIGHTS_PREFIX, main_sd, WEIGHTS_KEYS[0]),
            (DISCRETE_EMBED_PREFIX, disc_sd, WEIGHTS_KEYS[1]),
            (CONTINUOUS_EMBED_PREFIX, cont_sd, WEIGHTS_KEYS[2]))
    for prefix, sd, key in plan:
        # 空分组不写文件：写了也是个合法但无内容的 safetensors，反而让
        # manifest 多一个必须存在、却毫无意义的指针。
        if not sd:
            continue
        name = f"{prefix}.{uuid.uuid4().hex}.safetensors"
        _atomic_save_file(sd, os.path.join(wdir, name))
        cfg[key] = f"{WEIGHTS_DIRNAME}/{name}"
    tmp_cfg = os.path.join(model_dir,
                           f"{CONFIG_NAME}.{uuid.uuid4().hex}.tmp")
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_cfg, os.path.join(model_dir, CONFIG_NAME))
    _recycle_old_weights(model_dir, cfg)


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


def make_batch(ids, i: int, block_size: int, step_ids=None):
    """训练窗口切分——train.py 与 auto_train.py 共用的唯一实现。

    返回 (x_in, s_in, dec_in, s_dec, y)：
      user 段   = ids[i : i+half]        （encoder 输入）
      model 段  = ids[i+half : i+B]      （decoder 预测目标）
      dec_in    = model 段左移一位，首 token 用 user 段尾 token 播种
    调用方保证 len(ids) >= B + 1 且 0 <= i <= len(ids) - B。

    2026-10-27 审计修复：起点上界从 len(ids)-B-2 放宽到 len(ids)-B——
    旧上界把序列最后一个 token 永远排除在 y 之外，而 STOP 恰好被追加在
    序列末尾，导致单文件训练完全没有停止监督（模型学不会终止）。

    step_ids（2026-10-25）：与 ids 等长的「每 token 时间步编号」，由
    .tokens 流 unwrap 得到；缺省时退化为每 token 一步（旧行为）。
    窗口内的步号会重映射成 0..K（不同文件的绝对步号无关），并把
    decoder 首 token 的步设为「输入段最大步 + 1」——种子必须能看到
    完整输入段。
    """
    half = block_size // 2
    x_in = torch.tensor([ids[i:i + half]])
    y = torch.tensor([ids[i + half:i + block_size]])
    dec_ids = [ids[i + half - 1]] + ids[i + half:i + block_size - 1]
    dec_in = torch.tensor([dec_ids])
    if step_ids is None:
        s_in = torch.arange(half).unsqueeze(0)
        s_dec = torch.arange(half, half + half).unsqueeze(0)
        return x_in, s_in, dec_in, s_dec, y
    # 窗口内重映射（保序）：rank 只用于本窗口，跨文件不冲突
    rank = {}
    for t in range(i, i + block_size):
        v = step_ids[t]
        if v not in rank:
            rank[v] = len(rank)
    s_in = torch.tensor([[rank[step_ids[i + t]] for t in range(half)]])
    s_dec = torch.tensor(
        [[rank[step_ids[i + half - 1 + t]] + 1 for t in range(half)]])
    return x_in, s_in, dec_in, s_dec, y


def make_batch_emb(emb, pos_ids, i: int, block_size: int, step_ids):
    """多模态窗口切分——与 make_batch 同一刀法，只是 x/dec 用现成嵌入。

    emb:      [1, N, D] 整篇装配好的嵌入（assemble_embeddings 的产物）
    pos_ids:  长度 N；文本位置 = 词表 ID，媒体位置（视觉/音频）= None
    step_ids: 长度 N；每个位置所属的时间步编号

    返回 (emb_in, s_in, emb_dec, s_dec, y, n_valid)：
      - 切法、步号重映射与 make_batch 完全一致（同一窗口在两条路径上
        语义相同，纯文本数据切到哪条都等价）；
      - y 里媒体位置填 -100（F.cross_entropy 的 ignore_index）——视觉/
        音频没有词表 ID，lm_head 预测不了，只能当上下文；
      - n_valid = y 中可训练目标个数；为 0 说明整窗都是媒体，调用方
        必须跳过（否则 cross_entropy 全 ignore 会得到 NaN）。
    """
    half = block_size // 2
    emb_in = emb[:, i:i + half]
    emb_dec = emb[:, i + half - 1:i + block_size - 1]
    tgt = pos_ids[i + half:i + block_size]
    y = torch.tensor([[(-100 if t is None else int(t)) for t in tgt]])
    # 窗口内重映射（保序）：与 make_batch 逐行同构
    rank = {}
    for t in range(i, i + block_size):
        v = step_ids[t]
        if v not in rank:
            rank[v] = len(rank)
    s_in = torch.tensor([[rank[step_ids[i + t]] for t in range(half)]])
    s_dec = torch.tensor(
        [[rank[step_ids[i + half - 1 + t]] + 1 for t in range(half)]])
    n_valid = int((y != -100).sum())
    return emb_in, s_in, emb_dec, s_dec, y, n_valid

