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

import contextlib
import glob
import json
import os
import re
import sys
import time
import uuid
from typing import Optional

import torch
from safetensors.torch import load_file, save_file

import spec

CKPT_NAME = "model.safetensors"
CONFIG_NAME = "config.json"
# 初始化互斥锁文件（H4）：覆盖「冲突检查—写入—发布」全程，跨进程生效。
# init.py 子进程与 server 的 train start 用同一把锁，杜绝「初始化写到一半
# 训练已经开始读」的权重/config 不配套。
INIT_LOCK_NAME = ".model-init.lock"

# 每模型词表文件（2026-10-24，steve：每个模型可以有不同的 vocab 词表）。
# 语义：扁平可见 token 列表，ID 落在 [INVISIBLE_END, VOCAB_SIZE_PADDED)
# （不可见区 [0, INVISIBLE_END) 是 special/system/modal 固定 token 的架构
# 契约，自定义词表不占用）；不允许重复（反查歧义）。没有 vocab.txt 的
# 模型用全局词表 service/tokens/vocab.json。
# 2026-10-27：文件格式升级为 v2（显式 ID + JSON 转义），详见下方
# VOCAB_FILE_MAGIC 一节；v1 位置式文件读入时自动升级。
VOCAB_TXT_NAME = "vocab.txt"

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


def model_vocab_path(model_dir: str) -> str:
    """模型目录的 vocab.txt 路径（不一定存在）。"""
    return os.path.join(model_dir, VOCAB_TXT_NAME)


# ---------------------------------------------------------------------------
# 扁平词表文件格式（2026-10-27 审计修复 H1）
# ---------------------------------------------------------------------------
# v1（旧，位置式）：一行一个 token，第 N 行 → ID = INVISIBLE_END + N。
#   两个致命缺陷：
#     1. 无法表示 token 自身的 "\n" / "\r"——写出即变成空行，读回被跳过，
#        其后所有 ID 整体前移（base 预设实测前移 2 位）；
#     2. 把 reserved 占位符当作真实 token 导出，加载后可见区被占满
#        （free_slots=0），任何「添加 token」必然失败。
# v2（当前）：首行魔数，其后每行 "<ID>\t<JSON 字符串>"，显式 ID。
#   - 未列出的 ID 即空位（reserved），可被「添加 token」回收；
#   - token 文本走 JSON 转义，换行/回车/制表符/引号无损往返；
#   - 显式 ID 让「软删占位」「新增」「重排文件行序」都不会令其它 token
#     漂移，也不需要为占位符付出容量。
VOCAB_FILE_MAGIC = "#gargantua-vocab-v2"


def escape_vocab_token(tok: str) -> str:
    """token → 单行 JSON 字面量（换行等控制字符转义为 \\n 等）。"""
    return json.dumps(tok, ensure_ascii=False)


def unescape_vocab_token(text: str) -> str:
    """JSON 字面量 → token 文本。"""
    return json.loads(text)


def dumps_flat_vocab(entries) -> str:
    """[(id, token)] → v2 文件文本（含魔数，末尾换行）。"""
    lines = [VOCAB_FILE_MAGIC]
    lines.extend(f"{int(tid)}\t{escape_vocab_token(tok)}"
                 for tid, tok in entries)
    return "\n".join(lines) + "\n"


def parse_flat_vocab(text: str, source: str = "vocab.txt"):
    """解析扁平词表文本 → ([(id, token)], is_legacy)。

    带魔数 → v2，按显式 ID 解析；不带魔数 → v1，按行号位置解析
    （第 N 个非空行 → INVISIBLE_END + N），保证旧文件仍可读。
    重复 ID / 重复 token / 越界 / 条目超上限一律 ValueError。
    """
    raw_lines = text.split("\n")
    if raw_lines and raw_lines[0].strip() == VOCAB_FILE_MAGIC:
        entries = []
        seen_ids, seen_tok = set(), set()
        for lineno, raw in enumerate(raw_lines[1:], 2):
            line = raw.rstrip("\r")
            if not line.strip():
                continue
            tid_s, sep, payload = line.partition("\t")
            if not sep:
                raise ValueError(f"{source}:{lineno} 缺少制表符分隔: {line!r}")
            try:
                tid = int(tid_s)
                tok = unescape_vocab_token(payload)
            except Exception as e:
                raise ValueError(f"{source}:{lineno} 解析失败: {e}")
            if not (spec.INVISIBLE_END <= tid < spec.VOCAB_SIZE_PADDED):
                raise ValueError(
                    f"{source}:{lineno} token ID {tid} 越界 "
                    f"[{spec.INVISIBLE_END}, {spec.VOCAB_SIZE_PADDED})")
            if tid in seen_ids:
                raise ValueError(f"{source}:{lineno} token ID 重复: {tid}")
            if not isinstance(tok, str) or not tok:
                raise ValueError(f"{source}:{lineno} token 文本非法: {tok!r}")
            if tok in seen_tok:
                raise ValueError(
                    f"{source}:{lineno} token 重复: {tok!r}"
                    f"（反查歧义，拒绝加载）")
            seen_ids.add(tid)
            seen_tok.add(tok)
            entries.append((tid, tok))
        entries.sort(key=lambda e: e[0])
        return entries, False

    # ---- v1：位置式 ----
    entries = []
    seen = set()
    idx = 0
    for lineno, raw in enumerate(raw_lines, 1):
        tok = raw.rstrip("\r")
        if not tok:
            continue                        # 空行跳过，不占 ID（旧语义）
        if tok in seen:
            raise ValueError(
                f"{source}:{lineno} token 重复: {tok!r}"
                f"（反查歧义，拒绝加载）")
        seen.add(tok)
        entries.append((spec.INVISIBLE_END + idx, tok))
        idx += 1
    limit = spec.VOCAB_SIZE_PADDED - spec.INVISIBLE_END
    if len(entries) > limit:
        raise ValueError(
            f"{source}: {len(entries)} 个 token 超过上限 {limit}"
            f"（VOCAB_SIZE_PADDED - INVISIBLE_END）")
    return entries, True


def read_flat_vocab_file(path: str):
    """读任意扁平词表文件（vocab.txt）→ ([(id, token)], is_legacy)。"""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return parse_flat_vocab(text, source=path)


def write_flat_vocab_file(path: str, entries) -> None:
    """原子写 v2 扁平词表（唯一临时文件 + fsync + replace）。"""
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(dumps_flat_vocab(entries))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def migrate_flat_vocab_file(path: str):
    """把 v1 文件就地升级为 v2，**保持当前生效的 ID 映射不变**。

    返回 (entries, migrated)。升级只换表示法、不重排 ID：对已训练模型
    来说，ID ↔ 权重行的对应关系一个都不动（旧文件里被空行吞掉的
    "\\n"/"\\r" 保持丢失状态，因为它们本来就没进入过任何权重行）。
    """
    entries, legacy = read_flat_vocab_file(path)
    if not legacy:
        return entries, False
    try:
        write_flat_vocab_file(path, entries)
    except OSError as e:                    # 只读目录：内存里照常可用
        _warn(f"{path} 旧格式升级失败（只读？），按旧语义继续: {e}")
        return entries, False
    return entries, True


def read_model_vocab_entries(model_dir: str):
    """模型目录 vocab.txt → ([(id, token)], is_legacy)；不存在返回 None。

    旧格式（v1）会被就地升级为 v2（ID 映射不变），避免「读一次就漂移」
    的历史问题。条目数上限 VOCAB_SIZE_PADDED - INVISIBLE_END。
    """
    path = model_vocab_path(model_dir)
    if not os.path.exists(path):
        return None
    return migrate_flat_vocab_file(path)


def read_model_vocab(model_dir: str) -> Optional[list]:
    """读模型目录的 vocab.txt → token 列表（按 ID 升序）；不存在返回 None。"""
    got = read_model_vocab_entries(model_dir)
    if got is None:
        return None
    return [t for _, t in got[0]]


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
    ckpt = resolve_weights_path(model_dir)
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
    # 2026-10-25：按 config["dtype"] 施加模型精度（fp64/32/16/bf16 原生；
    # fp8/int8/int4/int2 在 CPU 上以 fp32 容器 + 权重量化模拟——损耗但能跑）。
    # 坏值防御性回退 fp32 并告警，不让单个错 dtype 把服务进程打崩。
    dtype_name = cfg.get("dtype") or "fp32"
    try:
        from precision import apply_precision
        apply_precision(m, dtype_name)
    except Exception as e:
        _warn(f"config dtype={dtype_name!r} 施加失败，回退 fp32: {e}")
    return m


# ---------------------------------------------------------------------------
# 检查点发布（2026-10-27 审计修复 M1/H4）
# ---------------------------------------------------------------------------
# 旧实现先 replace 权重再写 config：进程在两次 replace 之间崩溃会留下
# 「新权重 + 旧 config」的结构不匹配目录。现在改为版本化快照 + manifest：
#   1. 权重完整写入 model.<uuid>.safetensors（tmp + replace，单文件原子）；
#   2. config.json 作为 manifest 原子指向该权重文件（tmp + fsync + replace，
#      单文件原子）——(config, weights) 通过单文件替换实现整体原子；
#   3. 回收 manifest 之外的旧权重文件。
# 兼容旧目录：config 无 "weights" 键时回退 model.safetensors。
def resolve_weights_path(model_dir: str) -> str:
    """config.json 的 manifest → 权重文件路径；旧目录回退 model.safetensors。"""
    w = read_config(model_dir).get("weights")
    if isinstance(w, str) and w:
        p = os.path.join(model_dir, w)
        if os.path.exists(p):
            return p
    return os.path.join(model_dir, CKPT_NAME)


def model_exists(model_dir: str) -> bool:
    """模型目录是否包含可加载的模型（manifest 或旧式权重文件存在）。"""
    if os.path.exists(os.path.join(model_dir, CONFIG_NAME)):
        return True
    return bool(glob.glob(os.path.join(model_dir, "model*.safetensors")))


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
    """权重 + config.json 以「版本化快照 + manifest」原子发布。

    config.json 是唯一事实源：它指向哪个权重文件，读者就加载哪个——
    单文件原子替换让 (config, weights) 永远自洽。临时文件带 UUID，
    并发写互不覆盖；写后回收旧权重。"""
    os.makedirs(model_dir, exist_ok=True)
    weights_name = f"model.{uuid.uuid4().hex}.safetensors"
    tmp = os.path.join(model_dir, weights_name + ".tmp")
    save_file(model.state_dict(), tmp)
    os.replace(tmp, os.path.join(model_dir, weights_name))
    cfg = dict(config)
    cfg["weights"] = weights_name
    tmp_cfg = os.path.join(model_dir,
                           f"{CONFIG_NAME}.{uuid.uuid4().hex}.tmp")
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_cfg, os.path.join(model_dir, CONFIG_NAME))
    # 回收旧权重（manifest 之外的 model*.safetensors）
    keep = {weights_name}
    for old in glob.glob(os.path.join(model_dir, "model*.safetensors")):
        if os.path.basename(old) not in keep:
            try:
                os.unlink(old)
            except OSError:
                pass


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

