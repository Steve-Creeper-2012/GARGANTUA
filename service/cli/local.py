# ===========================================================================
# GARGANTUA v1 — service/cli/local.py：离线命令（不开 server 也能做转换与队列）
# ---------------------------------------------------------------------------
# 所有命令都返回一个可 JSON 序列化的 dict，由入口 gargantua.py 统一打印。
# 失败一律抛异常（ValueError / OSError / RuntimeError），入口转成「错误: ...」。
#
# 复用既有共享契约，绝不重写格式逻辑：
#   * 配对 .tokens / 队列文件名 → clip_data（service/train）
#   * token-6 编解码       → tokens.tok6
#   * 文本 / 图像 / 音频 / 视频 codec → tokens.enctok / image_codec / audio_codec
#   * 词表上界             → spec.MAX_VOCAB_SIZE
#   * 项目内路径           → paths.PROJECT_ROOT
#
# 落盘一律「隐藏临时文件 + fsync + os.replace」原子写，不写项目外的
# 系统临时目录，也不依赖任何系统级临时目录工具。
# ===========================================================================
"""local: 离线 .tokens 转换、训练队列管理、服务生命周期命令。"""
from __future__ import annotations

import os
import subprocess
import uuid

import spec
from tokens import tok6
import clip_data
import paths

# 这几个 group 默认走离线命令（入口据此分派）；--http 可强制改走命名 HTTP 命令
OFFLINE_GROUPS = ("tokens", "queue", "server")

USAGE = (
    "用法: gargantua [--server URL] [--timeout SEC] [--json] "
    "[--data-dir PATH] [--http] <group> <action> [参数...]"
)

OFFLINE_HELP = (
    "tokens encode --text STR | --text-file PATH --out PATH [--vocab NAME]",
    "tokens decode PATH                        .tokens → 显示文本",
    "tokens export --from PATH --out PATH      显示文本 → .tokens",
    "tokens inspect PATH                      .tokens 结构 / 配对检查",
    "tokens convert --source PATH --kind text|image|audio|video --out PATH",
    "tokens pair --text/--text-file/--text-tokens ... --media/--media-tokens ... --out PATH",
    "queue add PATH [--mode lm|clip] [--optimizer muon|adamw]",
    "queue list | remove NAME | clear --yes | inspect NAME",
    "queue start [--steps N] [--block-size N] [--optimizer ...] | stop | status  (HTTP)",
    "server start | stop | status | url",
    "api <METHOD> <PATH>                      任意端点直连（兜底）",
    "help                                     打印本帮助",
)


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------
def _parse_opts(argv, value_keys, flag_keys=()):
    """解析 --k v / --k=v / --flag；返回 (opts: dict, positionals: list)。

    只认声明过的 value_keys / flag_keys，未知选项抛 ValueError，便于入口
    统一输出「错误: ...」。key 保留原样（含连字符，如 max-media-positions）。
    """
    opts = {}
    pos = []
    i = 0
    n = len(argv)
    value_keys = set(value_keys)
    flag_keys = set(flag_keys)
    while i < n:
        t = argv[i]
        if not t.startswith("--"):
            pos.append(t)
            i += 1
            continue
        body = t[2:]
        if "=" in body:
            key, val = body.split("=", 1)
        else:
            key, val = body, None
        if key in flag_keys:
            # 布尔开关（如 queue clear --yes）：无值，出现即 True
            opts[key] = True
            i += 1
            continue
        if key not in value_keys:
            raise ValueError(f"未知参数 --{key}")
        if val is None:
            if i + 1 >= n:
                raise ValueError(f"--{key} 缺少值")
            val = argv[i + 1]
            i += 1
        opts[key] = val
        i += 1
    return opts, pos


def _read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _require_tokens_suffix(path):
    if not str(path).lower().endswith(".tokens"):
        raise ValueError(f"必须是 .tokens 文件: {path}")


def _atomic_write_bytes(path, data: bytes):
    """隐藏临时文件 + fsync + os.replace 原子落盘（不用 tempfile）。"""
    p = os.path.abspath(str(path))
    d = os.path.dirname(p)
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(
        d, "." + os.path.basename(p) + "." + uuid.uuid4().hex + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _set_vocab(name):
    if name:
        from tokens.vocab import set_vocab_source
        set_vocab_source(name)


def _queue_dir(ctx):
    return os.path.join(str(ctx["data_dir"]), "queue")


def _count(items):
    n_steps = sum(1 for it in items if it.get("kind") == "step")
    n_elements = sum(len(it.get("elements") or [])
                     for it in items if it.get("kind") == "step")
    n_text = sum(1 for it in items if it.get("kind") == "step"
                 for e in it["elements"] if e.get("kind") == "text")
    n_wave = sum(1 for it in items if it.get("kind") == "step"
                 for e in it["elements"] if e.get("kind") == "wave")
    n_visual = sum(1 for it in items if it.get("kind") == "step"
                   for e in it["elements"] if e.get("kind") == "visual")
    return n_steps, n_elements, n_text, n_wave, n_visual


# ---------------------------------------------------------------------------
# Step 构造（tokens convert / tokens pair 共用）
# ---------------------------------------------------------------------------
def _encode_text_steps(text):
    """文本字符串 → 旧式 text Step 列表（供 tok6.from_steps 消费）。"""
    from tokens import enctok
    ids, _skipped = enctok.encode_text(text)
    if not ids:
        raise ValueError("文本编码结果为空（没有可编码的字符）")
    return [{"token_ids": [int(i)], "type": spec.TYPE_USER,
             "modality": "text", "meta": {}} for i in ids]


def _auto_kind(src):
    if os.path.isdir(src):
        return "video"
    ext = os.path.splitext(str(src))[1].lower()
    if ext in (".wav",):
        return "audio"
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        return "image"
    return "video"          # gif 动图 / 其它一律按帧序列尝试


def _source_to_steps(src, kind):
    """单一来源 → 旧式 Step 列表（文本 / 图像 / 视频 / 音频）。"""
    if kind == "text":
        return _encode_text_steps(_read_text(src))
    if kind == "image":
        from tokens import image_codec
        img = image_codec.load_rgba(src)
        return [image_codec.image_to_step(img)]
    if kind == "video":
        from tokens import image_codec
        frames = image_codec.extract_frames(src)
        if not frames:
            raise ValueError(f"没有可抽取的帧: {src}")
        return image_codec.frames_to_steps(frames)
    if kind == "audio":
        from tokens import audio_codec
        samples = audio_codec.load_audio(src)
        return audio_codec.waveform_to_steps(samples)
    raise ValueError(f"未知媒体类型: {kind}")


# ---------------------------------------------------------------------------
# tokens 命令
# ---------------------------------------------------------------------------
def _tokens_encode(argv, ctx):
    opts, _pos = _parse_opts(argv, {"text", "text-file", "out", "vocab"})
    out = opts.get("out")
    if not out:
        raise ValueError("tokens encode 需要 --out PATH")
    _require_tokens_suffix(out)
    if "text" in opts:
        text = opts["text"]
    elif "text-file" in opts:
        text = _read_text(opts["text-file"])
    else:
        raise ValueError("tokens encode 需要 --text STR 或 --text-file PATH")
    _set_vocab(opts.get("vocab"))
    from tokens import enctok
    ids, skipped = enctok.encode_text(text)
    items = [tok6.step(tok6.text(i)) for i in ids]
    data = tok6.dumps(items)
    _atomic_write_bytes(out, data)
    return {"out": out, "n_tokens": len(ids), "n_steps": len(items),
            "bytes": len(data), "skipped": skipped}


def _tokens_decode(argv, ctx):
    _opts, pos = _parse_opts(argv, ())
    if not pos:
        raise ValueError("tokens decode 需要 PATH")
    path = pos[0]
    items = tok6.read_file(path)
    n_steps, n_elements, _t, _w, _v = _count(items)
    return {"path": path, "display": tok6.to_display(items),
            "n_steps": n_steps, "n_elements": n_elements}


def _tokens_export(argv, ctx):
    opts, _pos = _parse_opts(argv, {"from", "out"})
    src = opts.get("from")
    out = opts.get("out")
    if not src or not out:
        raise ValueError("tokens export 需要 --from PATH 和 --out PATH")
    _require_tokens_suffix(out)
    items = tok6.parse_display(_read_text(src))
    data = tok6.dumps(items)
    _atomic_write_bytes(out, data)
    n_steps, n_elements, _t, _w, _v = _count(items)
    return {"from": src, "out": out, "n_steps": n_steps,
            "n_elements": n_elements, "bytes": len(data)}


def _tokens_inspect(argv, ctx):
    _opts, pos = _parse_opts(argv, ())
    if not pos:
        raise ValueError("tokens inspect 需要 PATH")
    path = pos[0]
    _require_tokens_suffix(path)        # 只接受 .tokens 后缀
    data = _read_bytes(path)
    items = tok6.loads(data)
    n_steps, n_elements, n_text, n_wave, n_visual = _count(items)
    result = {"path": path, "bytes": len(data), "n_steps": n_steps,
              "n_elements": n_elements, "text": n_text, "wave": n_wave,
              "visual": n_visual}
    try:
        info = clip_data.parse_bytes(data)
        result.update({"pair": True, "media_kind": info["media_kind"],
                       "n_text_tokens": info["n_text_tokens"],
                       "n_media_positions": info["n_media_positions"]})
    except clip_data.PairError:
        result["pair"] = False
    return result


def _tokens_convert(argv, ctx):
    opts, _pos = _parse_opts(argv, {"source", "kind", "out", "vocab"})
    src = opts.get("source")
    kind = opts.get("kind")
    out = opts.get("out")
    if not src or not kind or not out:
        raise ValueError("tokens convert 需要 --source PATH --kind TYPE --out PATH")
    if kind not in ("text", "image", "audio", "video"):
        raise ValueError(f"未知 --kind: {kind}（可选 text|image|audio|video）")
    _require_tokens_suffix(out)
    _set_vocab(opts.get("vocab"))
    steps = _source_to_steps(src, kind)
    items = tok6.from_steps(steps)
    data = tok6.dumps(items)
    _atomic_write_bytes(out, data)
    return {"source": src, "kind": kind, "out": out, "n_steps": len(steps),
            "n_items": len(items), "bytes": len(data)}


def _pair_text_steps(opts):
    """文本源三选一 → 旧式 text Step 列表。"""
    if "text" in opts:
        return _encode_text_steps(opts["text"])
    if "text-file" in opts:
        return _encode_text_steps(_read_text(opts["text-file"]))
    # --text-tokens：必须是 .tokens，读入后取 text 步（跳过 ext/tool 标记步）
    path = opts["text-tokens"]
    _require_tokens_suffix(path)
    items = tok6.read_file(path)
    steps = tok6.to_steps(items)
    out = []
    for st in steps:
        meta = st.get("meta") or {}
        if meta.get("ext") or meta.get("tool"):
            continue
        if st.get("modality") == "text" and st.get("token_ids"):
            out.append(st)
    if not out:
        raise ValueError(f"--text-tokens 里没有可用的文本步: {path}")
    return out


def _pair_media_steps(opts, kind):
    """媒体源二选一 → 旧式 media Step 列表。"""
    if "media-tokens" in opts:
        path = opts["media-tokens"]
        _require_tokens_suffix(path)
        items = tok6.read_file(path)
        steps = tok6.to_steps(items)
        out = [st for st in steps
               if st.get("modality") in ("visual", "audio", "mixed")]
        if not out:
            raise ValueError(f"--media-tokens 里没有媒体步: {path}")
        return out
    src = opts["media"]
    eff_kind = _auto_kind(src) if kind == "auto" else kind
    return _source_to_steps(src, eff_kind)


def _tokens_pair(argv, ctx):
    opts, _pos = _parse_opts(argv, {
        "text", "text-file", "text-tokens", "media", "media-tokens",
        "kind", "out", "max-media-positions"})
    out = opts.get("out")
    if not out:
        raise ValueError("tokens pair 需要 --out PATH")
    _require_tokens_suffix(out)
    tsrc = [k for k in ("text", "text-file", "text-tokens") if k in opts]
    if len(tsrc) != 1:
        raise ValueError("tokens pair 需要 --text / --text-file / --text-tokens 三选一")
    msrc = [k for k in ("media", "media-tokens") if k in opts]
    if len(msrc) != 1:
        raise ValueError("tokens pair 需要 --media / --media-tokens 二选一")
    kind = opts.get("kind", "auto")
    if kind not in ("auto", "image", "audio", "video"):
        raise ValueError(f"未知 --kind: {kind}（可选 auto|image|audio|video）")
    cap = int(opts.get("max-media-positions",
                       clip_data.DEFAULT_MAX_PAIR_POSITIONS))
    text_steps = _pair_text_steps(opts)
    media_steps = _pair_media_steps(opts, kind)
    data = clip_data.pair_bytes(text_steps, media_steps, cap)
    _atomic_write_bytes(out, data)
    info = clip_data.parse_bytes(data, cap)
    return {"out": out, "media_kind": info["media_kind"],
            "n_text_tokens": info["n_text_tokens"],
            "n_media_positions": info["n_media_positions"],
            "bytes": len(data)}


_TOKENS = {
    "encode": _tokens_encode,
    "decode": _tokens_decode,
    "export": _tokens_export,
    "inspect": _tokens_inspect,
    "convert": _tokens_convert,
    "pair": _tokens_pair,
}


# ---------------------------------------------------------------------------
# queue 命令
# ---------------------------------------------------------------------------
def _validate_lm(data):
    """LM 模式入队校验（口径同 server_train.api_train_enqueue）。"""
    items = tok6.loads(data)
    if not items:
        raise ValueError("训练数据为空（没有任何时间步）")
    _steps, n_elements, _t, _w, _v = _count(items)
    if n_elements == 0:
        raise ValueError("训练数据没有可训练元素（只有扩展系统 token）")
    bad = next((e["id"] for it in items if it.get("kind") == "step"
                for e in it["elements"]
                if e.get("kind") == "text"
                and not (0 <= e["id"] < spec.MAX_VOCAB_SIZE)), None)
    if bad is not None:
        raise ValueError(f"token ID {bad} 超出词表（< {spec.MAX_VOCAB_SIZE}）")
    return n_elements


def _queue_add(argv, ctx):
    opts, pos = _parse_opts(argv, {"mode", "optimizer"})
    if not pos:
        raise ValueError("queue add 需要 PATH")
    path = pos[0]
    _require_tokens_suffix(path)
    mode = opts.get("mode", "lm")
    optimizer = opts.get("optimizer", spec.OPTIMIZER)
    if mode not in clip_data.MODES:
        raise ValueError(f"未知训练方式: {mode}（可选 {', '.join(clip_data.MODES)}）")
    if optimizer not in clip_data.OPTIMIZERS:
        raise ValueError(f"未知优化器: {optimizer}（可选 {', '.join(clip_data.OPTIMIZERS)}）")
    data = _read_bytes(path)
    if not data:
        raise ValueError("训练数据为空")
    # 与 server_train.api_train_enqueue 同一套上限（clip_data 单源）
    if len(data) > clip_data.MAX_ENQUEUE_BYTES:
        raise ValueError(
            f"训练数据超限: {len(data)} 字节 > "
            f"{clip_data.MAX_ENQUEUE_BYTES}，请拆分后入队")
    items = tok6.loads(data)
    if not items:
        raise ValueError("训练数据为空（没有任何时间步）")
    _steps, n_elements, _t, _w, _v = _count(items)
    if n_elements == 0:
        raise ValueError("训练数据没有可训练元素（只有扩展系统 token）")
    if n_elements > clip_data.MAX_ENQUEUE_ELEMENTS:
        raise ValueError(f"元素数超限: {n_elements} > "
                         f"{clip_data.MAX_ENQUEUE_ELEMENTS}")
    if mode == "clip":
        try:
            clip_data.parse_items(items,
                                  clip_data.DEFAULT_MAX_PAIR_POSITIONS)
        except clip_data.PairError as e:
            raise ValueError(f"CLIP 配对 .tokens 非法: {e}") from e
    else:
        _validate_lm(data)
    name = clip_data.make_queue_name(optimizer, mode)
    dst = os.path.join(_queue_dir(ctx), name)
    _atomic_write_bytes(dst, data)
    return {"name": name, "bytes": len(data), "mode": mode,
            "optimizer": optimizer}


def _queue_list(argv, ctx):
    _parse_opts(argv, ())
    qdir = _queue_dir(ctx)
    items = []
    if os.path.isdir(qdir):
        for name in sorted(os.listdir(qdir)):
            if name.startswith("."):        # 跳过隐藏临时文件（与 glob('*') 一致）
                continue
            p = os.path.join(qdir, name)
            if not os.path.isfile(p):
                continue
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            meta = clip_data.parse_queue_name(name)
            items.append({"name": name, "size": size,
                          "optimizer": meta["optimizer"],
                          "mode": meta["mode"]})
    return {"dir": qdir, "items": items}


def _queue_remove(argv, ctx):
    _opts, pos = _parse_opts(argv, ())
    if not pos:
        raise ValueError("queue remove 需要 NAME")
    name = pos[0]
    safe = os.path.basename(name)
    if safe != name:
        raise ValueError("NAME 不能包含路径分隔符")
    p = os.path.join(_queue_dir(ctx), safe)
    if not os.path.isfile(p):
        raise ValueError(f"队列项不存在: {safe}")
    os.unlink(p)
    return {"removed": safe}


def _queue_clear(argv, ctx):
    opts, _pos = _parse_opts(argv, (), {"yes"})
    if not opts.get("yes"):
        raise ValueError("queue clear 需要 --yes 确认才会执行")
    qdir = _queue_dir(ctx)
    removed = []
    if os.path.isdir(qdir):
        for name in sorted(os.listdir(qdir)):
            if name.startswith("."):
                continue
            p = os.path.join(qdir, name)
            if os.path.isfile(p):
                os.unlink(p)
                removed.append(name)
    return {"cleared": len(removed), "names": removed}


def _queue_inspect(argv, ctx):
    _opts, pos = _parse_opts(argv, ())
    if not pos:
        raise ValueError("queue inspect 需要 NAME")
    safe = os.path.basename(pos[0])
    p = os.path.join(_queue_dir(ctx), safe)
    data = _read_bytes(p)
    items = tok6.loads(data)
    n_steps, n_elements, n_text, n_wave, n_visual = _count(items)
    meta = clip_data.parse_queue_name(safe)
    result = {"name": safe, "bytes": len(data),
              "optimizer": meta["optimizer"], "mode": meta["mode"],
              "n_steps": n_steps, "n_elements": n_elements,
              "text": n_text, "wave": n_wave, "visual": n_visual}
    try:
        info = clip_data.parse_bytes(data)
        result["valid_clip_pair"] = True
        result["media_kind"] = info["media_kind"]
        result["n_text_tokens"] = info["n_text_tokens"]
        result["n_media_positions"] = info["n_media_positions"]
    except clip_data.PairError:
        result["valid_clip_pair"] = False
    return result


def _queue_start(argv, ctx):
    opts, _pos = _parse_opts(argv, {
        "steps", "block-size", "optimizer", "clip-batch-size",
        "clip-temperature", "target"})
    body = {}
    int_keys = {"steps": "steps", "block-size": "block_size",
                "clip-batch-size": "clip_batch_size"}
    for src, dst in int_keys.items():
        if src in opts:
            body[dst] = int(opts[src])
    if "clip-temperature" in opts:
        body["clip_temperature"] = float(opts["clip-temperature"])
    for k in ("optimizer", "target"):
        if k in opts:
            body[k] = opts[k]
    return ctx["http"]("POST", "/api/train/start", body=body)


def _queue_stop(argv, ctx):
    _parse_opts(argv, ())
    return ctx["http"]("POST", "/api/train/stop", body={})


def _queue_status(argv, ctx):
    _parse_opts(argv, ())
    return ctx["http"]("GET", "/api/train/status")


_QUEUE = {
    "add": _queue_add,
    "list": _queue_list,
    "remove": _queue_remove,
    "clear": _queue_clear,
    "inspect": _queue_inspect,
    "start": _queue_start,
    "stop": _queue_stop,
    "status": _queue_status,
}


# ---------------------------------------------------------------------------
# server 命令
# ---------------------------------------------------------------------------
def _server_run(extra):
    script = "启动.bat" if os.name == "nt" else "启动.sh"
    path = os.path.join(paths.PROJECT_ROOT, script)
    if not os.path.isfile(path):
        raise ValueError(f"找不到启动脚本: {script}")
    cmd = (["cmd", "/c", path] + extra) if os.name == "nt" else ([path] + extra)
    proc = subprocess.run(cmd, cwd=paths.PROJECT_ROOT,
                          capture_output=True, text=True)
    return {"action": "server", "args": extra, "script": script,
            "returncode": proc.returncode,
            "stdout": proc.stdout, "stderr": proc.stderr}


def _server_start(argv, ctx):
    _parse_opts(argv, ())
    return _server_run([])


def _server_stop(argv, ctx):
    _parse_opts(argv, ())
    return _server_run(["--stop"])


def _server_status(argv, ctx):
    _parse_opts(argv, ())
    return ctx["http"]("GET", "/api/status")


def _server_url(argv, ctx):
    _parse_opts(argv, ())
    return {"url": ctx["server"]}


_SERVER = {
    "start": _server_start,
    "stop": _server_stop,
    "status": _server_status,
    "url": _server_url,
}


# ---------------------------------------------------------------------------
# 分派 / 帮助
# ---------------------------------------------------------------------------
_DISPATCH = {"tokens": _TOKENS, "queue": _QUEUE, "server": _SERVER}


def run(group, action, argv, ctx):
    """执行离线命令 → 返回 dict。ctx 含 server / timeout / data_dir / http。"""
    table = _DISPATCH.get(group)
    if table is None:
        raise ValueError(f"未知离线命令组: {group}")
    fn = table.get(action)
    if fn is None:
        raise ValueError(f"未知命令: {group} {action}（可用: {', '.join(sorted(table))}）")
    return fn(argv, ctx)


def help_text(api_map) -> str:
    """生成完整帮助文本（命名命令清单从 api_map.API_COMMANDS 读取）。"""
    lines = [USAGE, ""]
    lines.append("全局选项: --server URL(默认环境变量 GARGANTUA_SERVER 或 "
                 "http://127.0.0.1:8712)  --timeout SEC  --json  "
                 "--data-dir PATH(默认项目内 data/)  --http  --help")
    lines.append("通用参数(命名命令): --data '<JSON对象>'  --param k=v  "
                 "--text k=str  --num k=number  --flag k  --file k=PATH")
    lines.append("  (GET 时 --data/--param/... 进 query；POST 时并入 body；"
                 "--file 读文件转 base64，frames_b64 可用逗号分隔多路径)")
    lines.append("")
    lines.append("== 命名命令（走 HTTP server；离线组默认本地执行，"
                 "加 --http 可强制走 HTTP）==")
    groups = {}
    for key, c in api_map.API_COMMANDS.items():
        group, _, action = key.partition(" ")
        groups.setdefault(group, []).append(
            (action or "-", c["method"], c["path"], c["help"]))
    for group in sorted(groups):
        lines.append(f"  [{group}]")
        for action, method, path, desc in sorted(groups[group]):
            lines.append(f"    {action:<22} {method:<5} {path:<34} {desc}")
    lines.append("")
    lines.append("== 离线命令（不开 server 也能用）==")
    for line in OFFLINE_HELP:
        lines.append("  " + line)
    lines.append("")
    lines.append("示例:")
    lines.append("  ./gargantua tokens encode --text 'hello' --out a.tokens")
    lines.append("  ./gargantua tokens pair --text 'a cat' --media-tokens m.tokens "
                 "--out pair.tokens")
    lines.append("  ./gargantua queue add a.tokens --mode lm --data-dir ./data")
    lines.append("  ./gargantua --json status")
    lines.append("  ./gargantua api POST /api/infer --data '{\"prompt\":\"hi\"}'")
    return "\n".join(lines)
