# ===========================================================================
# GARGANTUA v1 — clip_folder.py：对比训练数据集文件夹格式 v1（唯一真源）
# ---------------------------------------------------------------------------
# 2026-11-08（steve）：对比训练不再「一次上传两个 .tokens」，而是强制
# 上传一个按下面文件树组织的**文件夹**。解析出的 N 对真实配对同批做
# 对称 InfoNCE，负样本来自批内其它配对 —— 100% 还原 CLIP，不再用
# (A,B)+(B,A) 那种自相似假负样本。
#
# 【强制文件树】多一个文件、少一个文件、命名不符一律拒绝（FolderError → 400）
#
#   <dataset>/                       ← 文件夹名任意，服务端自动剥离第一层
#   ├── dataset.json                 ← 必需：格式声明 + 配对清单 + 可选超参
#   └── pairs/                       ← 必需：只能有配对子目录，不能有散文件
#       ├── 001/                     ← 一个配对 = 一个子目录，名字见 dataset.json
#       │   ├── a.tokens             ← 必需：A 侧 token-6 位流（模态任意）
#       │   └── b.tokens             ← 必需：B 侧 token-6 位流（模态任意）
#       ├── 002/
#       │   ├── a.tokens
#       │   └── b.tokens
#       └── ...                      ← 至少 CLIP_FOLDER_MIN_PAIRS 个配对
#
# 【dataset.json】（UTF-8 JSON 对象）
#   {
#     "format": "gargantua-clip-folder",   // 必需，固定值
#     "version": 1,                        // 必需，固定值
#     "name": "my dataset",                // 可选，默认用文件夹名
#     "pairs": ["001", "002"],             // 可选：配对目录名与顺序；
#                                          // 省略则按目录名排序自动发现；
#                                          // 给出时必须与目录集合完全一致
#     "batch_size": 16,                    // 可选：每批多少对做 InfoNCE，
#                                          // 2..64，默认 min(N, 16)
#     "temperature": 0.07                  // 可选：(0, 100]，默认 0.07
#   }
#
# 【模态】a.tokens / b.tokens 内部可以是文本、图片、音频、视频或混合，
#   任意组合都行（图片×图片、视频×视频、文本×图片…）。不再要求
#   「一段文本 + 一段媒体」，也不再按内容推断 image/audio/video。
#
# 【为什么至少 2 对】InfoNCE 在 1×1 相似度矩阵上 softmax 只有一个元素，
#   log(1)=0，梯度恒为零；必须有批内负样本才能训练。
#
# 浏览器侧通过 webkitdirectory / 拖拽文件夹拿到带相对路径的 File 列表，
# 服务端在这里统一校验并解码，前端不许抄任何常量（走 spec_payload()）。
# ===========================================================================
"""clip_folder: CLIP 数据集文件夹的定义、严格校验与解码。"""
from __future__ import annotations

import base64
import json
import re

import spec
from tokens import tok6

# 请求体安全上限（不是架构常数，只防一次上传把 server 撑爆）
MAX_FOLDER_FILES = 512
MAX_FOLDER_BYTES = 64 * 1024 * 1024

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ALLOWED_PAIR_FILES = ("a.tokens", "b.tokens")


class FolderError(ValueError):
    """数据集文件夹不符合 v1 契约。server 侧映射 400。"""


def _norm(path) -> str:
    """相对路径归一化；拒绝绝对路径、.. 、空段与奇怪字符。"""
    p = str(path or "").replace("\\", "/").strip()
    if not p or p.startswith("/"):
        raise FolderError(f"非法文件路径: {path!r}")
    parts = [seg for seg in p.split("/") if seg]
    if any(seg in (".", "..") for seg in parts):
        raise FolderError(f"文件路径含 . 或 ..: {path!r}")
    if len(parts) < 1:
        raise FolderError(f"空文件路径: {path!r}")
    return "/".join(parts)


def _decode_one(item) -> tuple:
    """{path, data_b64} → (归一化相对路径, bytes)。"""
    if not isinstance(item, dict):
        raise FolderError("文件夹条目必须是 {path, data_b64}")
    raw_path = item.get("path", item.get("name"))
    b64 = item.get("data_b64")
    if not b64 or not isinstance(b64, str):
        raise FolderError(f"文件「{raw_path}」缺少 data_b64")
    try:
        data = base64.b64decode(b64, validate=True)
    except Exception as e:
        raise FolderError(f"文件「{raw_path}」的 data_b64 非法: {e}") from None
    if len(data) > MAX_FOLDER_BYTES:
        raise FolderError(
            f"文件「{raw_path}」超限: {len(data)} > {MAX_FOLDER_BYTES} 字节")
    return _norm(raw_path), data


def _strip_root(entries: list) -> tuple:
    """浏览器给的路径带一层文件夹名；统一剥掉，返回 (根名, [(rel,data)])。"""
    parts = [rel.split("/") for rel, _ in entries]
    root = ""
    if (parts and all(len(p) >= 2 for p in parts)
            and len({p[0] for p in parts}) == 1
            and parts[0][0] not in ("dataset.json", "pairs")):
        root = parts[0][0]
        entries = [(rel.split("/", 1)[1], data) for rel, data in entries]
    return root, entries


def _manifest(entries: dict) -> dict:
    if "dataset.json" not in entries:
        raise FolderError("缺少 dataset.json（文件夹格式 v1 必需）")
    try:
        obj = json.loads(entries["dataset.json"].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise FolderError(f"dataset.json 不是合法 UTF-8 JSON: {e}") from None
    if not isinstance(obj, dict):
        raise FolderError("dataset.json 必须是 JSON 对象")
    if obj.get("format") != spec.CLIP_FOLDER_FORMAT:
        raise FolderError(
            f"dataset.json 的 format 必须是 {spec.CLIP_FOLDER_FORMAT!r}")
    if obj.get("version") != spec.CLIP_FOLDER_VERSION:
        raise FolderError(
            f"dataset.json 的 version 必须是 {spec.CLIP_FOLDER_VERSION}")
    return obj


def _pair_dirs(entries: dict):
    """→ {id: {文件名: bytes}}；顶层只允许 dataset.json 与 pairs/。"""
    out = {}
    for rel, data in entries.items():
        seg = rel.split("/")
        if seg == ["dataset.json"]:
            continue
        if len(seg) != 3 or seg[0] != "pairs" or not _ID_RE.match(seg[1]):
            raise FolderError(
                f"文件树非法: {rel!r}；只允许 dataset.json 与 "
                "pairs/<id>/{a,b}.tokens")
        out.setdefault(seg[1], {})[seg[2]] = data
    if not out:
        raise FolderError("pairs/ 下没有任何配对目录")
    for pid, files in out.items():
        got = set(files)
        if got != set(_ALLOWED_PAIR_FILES):
            raise FolderError(
                f"配对「{pid}」必须恰好包含 a.tokens 与 b.tokens，"
                f"实际: {sorted(got)}")
    return out


def _items(data: bytes, where: str):
    try:
        items = tok6.loads(data)
    except tok6.Tok6Error as e:
        raise FolderError(f"{where} 不是合法的 token-6 .tokens: {e}") from None
    if not items:
        raise FolderError(f"{where} 是空的 .tokens")
    return items


def parse(files) -> dict:
    """文件夹 File 列表 → 可入队的配对描述；任何不合规都抛 FolderError。

    返回 {
      "name": str, "pairs": [{"id","a_items","b_items"}],
      "n_pairs": int, "batch_size": int, "temperature": float,
    }
    """
    if not isinstance(files, list) or not files:
        raise FolderError("数据集文件夹为空或不是数组")
    if len(files) > MAX_FOLDER_FILES:
        raise FolderError(f"文件夹文件数超限: {len(files)} > {MAX_FOLDER_FILES}")
    decoded = [_decode_one(f) for f in files]
    root, decoded = _strip_root(decoded)
    entries = dict(decoded)
    if len(entries) != len(decoded):
        raise FolderError("文件夹里出现重复路径")
    man = _manifest(entries)
    dirs = _pair_dirs(entries)

    listed = man.get("pairs")
    if listed is None:
        ids = sorted(dirs)
    else:
        if (not isinstance(listed, list) or not listed
                or not all(isinstance(x, str) and _ID_RE.match(x)
                           for x in listed)):
            raise FolderError("dataset.json 的 pairs 必须是非空字符串数组，"
                              "目录名只允许 [A-Za-z0-9_-]，最长 64")
        if len(set(listed)) != len(listed):
            raise FolderError("dataset.json 的 pairs 有重复项")
        ids = list(listed)
    if set(ids) != set(dirs):
        raise FolderError(
            "dataset.json 的 pairs 与 pairs/ 目录不一致；"
            f"声明 {sorted(ids)}，实际 {sorted(dirs)}")
    n = len(ids)
    if n < spec.CLIP_FOLDER_MIN_PAIRS:
        raise FolderError(
            f"CLIP 至少需要 {spec.CLIP_FOLDER_MIN_PAIRS} 对真实配对"
            "（InfoNCE 需要批内负样本），当前 " + str(n) + " 对")

    batch = man.get("batch_size", None)
    if batch is None:
        batch = min(n, spec.CLIP_FOLDER_DEFAULT_BATCH)
    if (isinstance(batch, bool) or not isinstance(batch, int)
            or not (2 <= batch <= spec.CLIP_FOLDER_MAX_BATCH)):
        raise FolderError(
            f"batch_size 必须是 2..{spec.CLIP_FOLDER_MAX_BATCH} 的整数")
    temp = man.get("temperature", spec.CLIP_FOLDER_TEMPERATURE)
    if (isinstance(temp, bool) or not isinstance(temp, (int, float))
            or not (0 < float(temp) <= 100)):
        raise FolderError("temperature 必须落在 (0, 100]")
    name = man.get("name")
    if name is not None and not isinstance(name, str):
        raise FolderError("dataset.json 的 name 必须是字符串")
    name = (name or root or "clip-dataset").strip()[:128] or "clip-dataset"

    pairs = []
    for pid in ids:
        pairs.append({
            "id": pid,
            "a_items": _items(dirs[pid]["a.tokens"], f"pairs/{pid}/a.tokens"),
            "b_items": _items(dirs[pid]["b.tokens"], f"pairs/{pid}/b.tokens"),
        })
    return {"name": name, "pairs": pairs, "n_pairs": n,
            "batch_size": batch, "temperature": float(temp)}


def spec_payload() -> dict:
    """给前端看的格式说明（前端一律从这里读，不抄常量 / 不拼文件树）。"""
    return {
        "format": spec.CLIP_FOLDER_FORMAT,
        "version": spec.CLIP_FOLDER_VERSION,
        "min_pairs": spec.CLIP_FOLDER_MIN_PAIRS,
        "max_batch": spec.CLIP_FOLDER_MAX_BATCH,
        "default_batch": spec.CLIP_FOLDER_DEFAULT_BATCH,
        "default_temperature": spec.CLIP_FOLDER_TEMPERATURE,
        "max_files": MAX_FOLDER_FILES,
        "tree": [
            "<数据集文件夹>/",
            "├── dataset.json",
            "└── pairs/",
            "    ├── 001/",
            "    │   ├── a.tokens",
            "    │   └── b.tokens",
            "    ├── 002/",
            "    │   ├── a.tokens",
            "    │   └── b.tokens",
            "    └── ...",
        ],
        "manifest": {
            "format": spec.CLIP_FOLDER_FORMAT,
            "version": spec.CLIP_FOLDER_VERSION,
            "name": "示例数据集",
            "pairs": ["001", "002"],
            "batch_size": spec.CLIP_FOLDER_DEFAULT_BATCH,
            "temperature": spec.CLIP_FOLDER_TEMPERATURE,
        },
        "rules": [
            f"顶层只允许 dataset.json 与 pairs/，至少 {spec.CLIP_FOLDER_MIN_PAIRS} 个配对目录",
            "每个配对目录恰好两个文件：a.tokens 与 b.tokens，不多不少",
            "a.tokens / b.tokens 都是 token-6 位流，内部模态任意（文本 / 图片 / 音频 / 视频 / 混合）",
            "N 对真实配对同批做对称 InfoNCE，负样本来自批内其它配对",
            "dataset.json 的 pairs 省略时按目录名排序自动发现；给出时必须与目录完全一致",
        ],
    }
