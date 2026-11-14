# ===========================================================================
# GARGANTUA v1 — server/momw_state.py：MoMW 池快照（独立于权重契约）
# ---------------------------------------------------------------------------
# 动机（2026-11-08 第三版）：层数 / S / E / γ / bonus 原本只活在一次会话里，
# 进程重启即丢。但把它们塞进模型权重会破坏 `config.json` manifest +
# 三个 safetensors 的严格契约（build_model 是 strict load，多一个 key 就炸）。
#
# 因此快照单独放一个子目录，**不进 weights/、不进 config.json 的 manifest**：
#     models/NAME/momw_state/
#       layout.json              布局版本 + 层元数据（类型 / writes / age / seed）
#       tensors.safetensors      L{i}.S / L{i}.v / L{i}.E / L{i}.P / L{i}.cc.*
#       meta.json                config 快照 / steps / expansions / saved_at
#
# 放在模型目录**内部**而不是外面：删模型时快照一起回收，不会留下孤儿状态；
# 同时与「模型目录自包含」的既有契约一致。快照缺失/损坏一律按「无快照」
# 处理（从零初始化），绝不让推理起不来——它本来就是可重建的运行时数据。
#
# 参照 LangGraph checkpointer 的语义：序列化协议与存储解耦，恢复时按
# 快照重建图结构；落盘时机是「一个写入批次之后」，不是每一步。
# ===========================================================================
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Optional

import torch
from safetensors.torch import load_file, save_file

import spec

STATE_DIRNAME = "momw_state"
LAYOUT_VERSION = 1
LAYOUT_NAME = "layout.json"
TENSORS_NAME = "tensors.safetensors"
META_NAME = "meta.json"


def state_dir(model_dir: str) -> str:
    return os.path.join(model_dir, STATE_DIRNAME)


def has_state(model_dir: str) -> bool:
    d = state_dir(model_dir)
    return (os.path.isfile(os.path.join(d, LAYOUT_NAME))
            and os.path.isfile(os.path.join(d, TENSORS_NAME)))


def _atomic_write_json(path: str, obj) -> None:
    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _atomic_save_tensors(sd: dict, path: str) -> None:
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


def _persist_torch_dtype(name: str):
    """落盘精度 → torch dtype；非法值按 fp32 处理（不让它破坏快照）。"""
    return getattr(torch, spec.MOMW_PERSIST_DTYPE_MAP.get(
        str(name).lower(), "float32"))


def save(dst_dir: str, pool, extra: Optional[dict] = None,
         persist_dtype: Optional[str] = None) -> str:
    """把一个 MoMWPool 的运行时状态写成快照（原子写每个文件）。

    `pool.cfg["persist_dtype"]` 决定落盘精度（fp32 默认 / fp16 减半）。
    所有状态张量在写 safetensors 之前统一降精度；恢复时按模型自身 dtype
    转回（见 MoMW.load_state），所以落盘精度只影响磁盘体积与重启舍入。
    """
    os.makedirs(dst_dir, exist_ok=True)
    tensors: dict = {}
    layers = []
    for i, l in enumerate(pool.kda_layers):
        layers.append({
            "index": i,
            "type_idx": int(l.type_idx),
            "writes": int(l._rt["writes"]),
            "bonus_age": int(l._rt["bonus_age"]),
            "prev_norm": float(l._rt["prev_norm"]),
            "growth": float(l._rt["growth"]),
            "seed": int(l._rt["seed"]),
            "oja_steps": int(l._rt["oja_steps"]),
            "has_state": l.S is not None,
            "has_E": l._rt["E"] is not None,
            "has_P_override": l._rt["P_override"] is not None,
        })
        st = l._rt["state"]
        if st is not None and st[0] is not None:
            tensors[f"L{i}.S"] = st[0].detach().cpu().contiguous()
            for k, v in (st[1] or {}).items():
                if v is not None:
                    tensors[f"L{i}.cc.{k}"] = v.detach().cpu().contiguous()
        tensors[f"L{i}.v"] = l._rt["eig_v"].detach().cpu().contiguous()
        if l._rt["E"] is not None:
            tensors[f"L{i}.E"] = l._rt["E"].detach().cpu().contiguous()
        if l._rt["P_override"] is not None:
            tensors[f"L{i}.P"] = l._rt["P_override"].detach().cpu().contiguous()
    # 显式参数优先（MoMW.cfg 是权威；pool.cfg 是建池时的快照，运行时改
    # 配置后两者可能不同步）
    persist_dtype = str(persist_dtype
                        or pool.cfg.get("persist_dtype", "fp32")).lower()
    tensors = {k: v.to(_persist_torch_dtype(persist_dtype)).contiguous()
               for k, v in tensors.items()}
    _atomic_save_tensors(tensors, os.path.join(dst_dir, TENSORS_NAME))
    _atomic_write_json(os.path.join(dst_dir, LAYOUT_NAME), {
        "layout_version": LAYOUT_VERSION,
        "expansions": int(pool.expansions),
        "serial": int(pool._serial),
        "next_layer_index": len(layers),
        "layers": layers,
    })
    meta = {
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": None,
        "config": dict(pool.cfg),
        "persist_dtype": persist_dtype,
        "tensor_bytes": sum(int(v.numel() * v.element_size())
                            for v in tensors.values()),
    }
    if extra:
        meta.update(extra)
    _atomic_write_json(os.path.join(dst_dir, META_NAME), meta)
    return dst_dir


def load(src_dir: str) -> Optional[dict]:
    """读快照；缺失 / 版本不符 / 损坏都返回 None（调用方从零初始化）。"""
    layout_path = os.path.join(src_dir, LAYOUT_NAME)
    tensors_path = os.path.join(src_dir, TENSORS_NAME)
    if not (os.path.isfile(layout_path) and os.path.isfile(tensors_path)):
        return None
    try:
        with open(layout_path, encoding="utf-8") as f:
            layout = json.load(f)
        if int(layout.get("layout_version", -1)) != LAYOUT_VERSION:
            return None
        tensors = load_file(tensors_path)
        meta = {}
        meta_path = os.path.join(src_dir, META_NAME)
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        return {"layout": layout, "tensors": tensors, "meta": meta}
    except Exception:
        return None


def clear(model_dir: str) -> bool:
    """删快照（只删本模块自己的三个文件，不做递归删除）。"""
    d = state_dir(model_dir)
    removed = False
    for name in (LAYOUT_NAME, TENSORS_NAME, META_NAME):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            try:
                os.unlink(p)
                removed = True
            except OSError:
                pass
    return removed
