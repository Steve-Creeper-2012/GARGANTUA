# ===========================================================================
# GARGANTUA v1 — service/cli/gargantua.py：终端入口与调度
# ---------------------------------------------------------------------------
# 用法：
#   gargantua [--server URL] [--timeout SEC] [--json] [--data-dir PATH]
#             [--http] <group> <action> [参数...]
#
#   * 命名命令（api_map.API_COMMANDS）走 HTTP；离线命令（local.py）本地执行。
#   * 通用参数 --data / --param / --text / --num / --flag / --file 构造请求体。
#   * 兜底 `api <METHOD> <PATH>` 可直连任意端点（含将来新增的 API）。
#   * 成功打印 JSON，失败打印「错误: ...」并以非零码退出。
#
# 用标准库 urllib.request 作 HTTP 客户端：显式 Content-Length（server 拒绝
# chunked）、Content-Type: application/json；HTTPError 打印响应体 JSON。
# ===========================================================================
"""GARGANTUA 终端入口：BOOT_HEADER 引导 → 参数解析 → 分派 → 统一输出。"""
from __future__ import annotations

# ---- 标准路径引导头（放在 docstring / __future__ 之后）----
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

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from urllib.parse import quote, urlencode

import paths
from cli import api_map, local

DEFAULT_SERVER = "http://127.0.0.1:8712"


class ApiError(Exception):
    """HTTP 非 2xx：携带状态码与解析后的响应体（供入口打印）。"""

    def __init__(self, code, payload):
        super().__init__(f"HTTP {code}")
        self.code = code
        self.payload = payload


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# HTTP 客户端
# ---------------------------------------------------------------------------
def _http_call(method, path, *, query=None, body=None, server, timeout):
    url = server.rstrip("/") + path
    if query:
        url += "?" + urlencode(query, doseq=True)
    data = None
    headers = {"Accept": "application/json"}
    if method not in ("GET", "HEAD"):
        data = json.dumps(body if body is not None else {},
                          ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(data))   # 显式：server 拒绝 chunked
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "") or ""
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = raw.decode("utf-8", "replace")
        raise ApiError(e.code, payload) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            "服务未启动：先运行 ./启动.sh，或使用离线命令 tokens / queue"
            f"（{e.reason}）") from e
    if "application/json" in ctype or not ctype:
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            pass
    # 非 JSON 响应（如 /api/prims/job/source 返回 PNG）→ 结构化描述，不丢信息
    return {"_binary": True, "content_type": ctype, "size": len(raw),
            "base64": base64.b64encode(raw).decode("ascii")}


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
def _parse_global(argv):
    """从整个 argv 抽取全局选项（允许出现在命令前后），其余按原序保留。"""
    server = os.environ.get("GARGANTUA_SERVER") or DEFAULT_SERVER
    timeout = 30.0
    data_dir = os.environ.get("GARGANTUA_DATA_DIR") or paths.project_dir("data")
    as_json = False
    force_http = False
    help_flag = False
    rest = []
    i = 0
    n = len(argv)
    while i < n:
        t = argv[i]
        if t == "--server" and i + 1 < n:
            server = argv[i + 1]
            i += 2
        elif t.startswith("--server="):
            server = t[len("--server="):]
            i += 1
        elif t == "--timeout" and i + 1 < n:
            timeout = float(argv[i + 1])
            i += 2
        elif t.startswith("--timeout="):
            timeout = float(t[len("--timeout="):])
            i += 1
        elif t == "--data-dir" and i + 1 < n:
            data_dir = argv[i + 1]
            i += 2
        elif t.startswith("--data-dir="):
            data_dir = t[len("--data-dir="):]
            i += 1
        elif t == "--json":
            as_json = True
            i += 1
        elif t == "--http":
            force_http = True
            i += 1
        elif t in ("--help", "-h"):
            help_flag = True
            i += 1
        else:
            rest.append(t)
            i += 1
    return {
        "server": server, "timeout": timeout, "data_dir": data_dir,
        "as_json": as_json, "force_http": force_http, "help_flag": help_flag,
        "rest": rest,
    }


_GENERIC_OPTS = {"data", "param", "text", "num", "file"}


def _parse_generic(tail):
    """解析命名命令的通用参数 → (collected, positionals)。"""
    collected = {"data": [], "param": [], "text": [], "num": [],
                 "flag": [], "file": []}
    pos = []
    i = 0
    n = len(tail)
    while i < n:
        t = tail[i]
        if not t.startswith("--"):
            pos.append(t)
            i += 1
            continue
        raw = t[2:]
        if raw.startswith("data="):
            collected["data"].append(raw[len("data="):])
            i += 1
            continue
        if "=" in raw:
            opt, val = raw.split("=", 1)
        else:
            opt, val = raw, None
        if opt == "flag":
            if val is None:
                if i + 1 >= n:
                    raise ValueError("--flag 缺少 key")
                val = tail[i + 1]
                i += 1
            collected["flag"].append(val)
            i += 1
            continue
        if opt not in _GENERIC_OPTS:
            raise ValueError(f"未知通用参数 --{opt}")
        if val is None:
            if i + 1 >= n:
                raise ValueError(f"--{opt} 缺少值")
            val = tail[i + 1]
            i += 1
        collected[opt].append(val)
        i += 1
    return collected, pos


def _to_number(v):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError as e:
            raise ValueError(f"--num 值不是数字: {v}") from e


def _apply_generic(base, g):
    """把 --param/--text/--num/--flag/--file 并入 base（body 或 query）。"""
    for kv in g["param"]:
        k, v = kv.split("=", 1)
        base[k] = v
    for kv in g["text"]:
        k, v = kv.split("=", 1)
        base[k] = v
    for kv in g["num"]:
        k, v = kv.split("=", 1)
        base[k] = _to_number(v)
    for k in g["flag"]:
        base[k] = True
    for kv in g["file"]:
        k, paths_value = kv.split("=", 1)
        if k == "frames_b64":
            base[k] = [base64.b64encode(_read_bytes(p)).decode("ascii")
                       for p in paths_value.split(",") if p]
        else:
            base[k] = base64.b64encode(
                _read_bytes(paths_value)).decode("ascii")
    return base


def _build_base(c, g):
    base = dict(c.get("defaults", {}))
    for d in g["data"]:
        obj = json.loads(d)
        if not isinstance(obj, dict):
            raise ValueError("--data 必须是 JSON 对象")
        base.update(obj)
    return _apply_generic(base, g)


# ---------------------------------------------------------------------------
# 命令执行
# ---------------------------------------------------------------------------
def _cmd_named(key, c, tail, ctx):
    g, pos = _parse_generic(tail)
    base = _build_base(c, g)
    path = c["path"]
    positionals = c.get("positionals", [])
    if "<" in path:
        for idx, k in enumerate(positionals):
            if idx < len(pos):
                path = path.replace("<" + k + ">", quote(str(pos[idx]), safe=""))
    else:
        for idx, k in enumerate(positionals):
            if idx < len(pos):
                base[k] = pos[idx]
    if c["method"] == "GET":
        return ctx["http"]("GET", path, query=base, body=None)
    return ctx["http"](c["method"], path, query=None, body=base)


def _cmd_api(argv, ctx):
    """兜底：api <METHOD> <PATH> [通用参数...] 直连任意端点。"""
    if len(argv) < 2:
        raise ValueError("api 需要 <METHOD> <PATH>")
    method = argv[0].upper()
    path = argv[1]
    if not path.startswith("/"):
        path = "/" + path
    g, _pos = _parse_generic(argv[2:])
    base = _build_base({"method": method}, g)
    if method in ("GET", "HEAD"):
        return ctx["http"](method, path, query=base, body=None)
    return ctx["http"](method, path, query=None, body=base)


def _emit(obj, as_json):
    if as_json:
        print(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    else:
        print(json.dumps(obj, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    opt = _parse_global(argv)
    ctx = {
        "server": opt["server"],
        "timeout": opt["timeout"],
        "data_dir": opt["data_dir"],
        "http": lambda m, p, query=None, body=None: _http_call(
            m, p, query=query, body=body,
            server=opt["server"], timeout=opt["timeout"]),
    }
    try:
        rest = opt["rest"]
        if opt["help_flag"] or not rest or rest[0] == "help":
            print(local.help_text(api_map))
            return 0
        group = rest[0]
        action = rest[1] if len(rest) > 1 else ""
        if group == "api":
            result = _cmd_api(rest[1:], ctx)
        elif group in local.OFFLINE_GROUPS and not opt["force_http"]:
            result = local.run(group, action, rest[2:], ctx)
        else:
            key = (group + " " + action).strip() if action else group
            c = api_map.API_COMMANDS.get(key)
            if c is None:
                raise ValueError(f"未知命令: {key}（用 --help 查看全部命令）")
            tail = rest[2:] if " " in key else rest[1:]
            result = _cmd_named(key, c, tail, ctx)
        _emit(result, opt["as_json"])
        return 0
    except ApiError as e:
        detail = (json.dumps(e.payload, ensure_ascii=False)
                  if isinstance(e.payload, (dict, list)) else str(e.payload))
        print(f"错误: HTTP {e.code} {detail}")
        return 1
    except Exception as e:
        print(f"错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
