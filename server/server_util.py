# ===========================================================================
# GARGANTUA v1 — server_util.py：HTTP 层小工具（被 server/codecs 共用）
# ---------------------------------------------------------------------------
# 只放无状态、无依赖的取值/校验助手，避免 server.py ↔ server_codecs.py
# 之间为拿一个函数而互相 import（循环依赖）。
# ===========================================================================
import os
import re

_INT_RE = re.compile(r"[+-]?\d+")


def num_param(body, key, default, cast):
    """ 取数值参数：只有键缺失或显式为 None 时才用默认值。

    不能用 `body.get(key) or default`：0 / 0.0 / "" 是合法请求值，会被
    `or` 静默替换掉——用户拿到与请求不符的行为，且没有任何报错（实测
    temperature=0 被静默改成 0.8，返回的却是 202 而非 400）。

    2026-10-23 审计修复（P3）：此前 `int(True)==1` / `float("0.8")` 之类
    的类型混淆会被静默接受——JSON true 变成 max_new_tokens=1、字符串
    "0.8" 变成合法 temperature。现在：bool 一律拒绝；int 只接受整数
    （含整数值 float 与纯数字字符串）；float 只接受数值或可解析的
    数字字符串。显式 0 仍然是合法的 0（不被默认值吞掉）。 """
    v = body.get(key)
    if v is None:
        return cast(default)
    if isinstance(v, bool):
        raise ValueError(f"{key} 必须是数值，不能是布尔值: {v!r}")
    if cast is int:
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            if not v.is_integer():
                raise ValueError(f"{key} 必须是整数: {v!r}")
            return int(v)
        if isinstance(v, str) and _INT_RE.fullmatch(v.strip()):
            return int(v.strip())
        raise ValueError(f"{key} 必须是整数: {v!r}")
    # float
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            raise ValueError(f"{key} 必须是数值: {v!r}")
    raise ValueError(f"{key} 必须是数值: {v!r}")


def pip_index_args() -> list:
    """ pip 源参数：尊重用户已设的 PIP_INDEX_URL；否则默认清华镜像。

    实测（国内出口）：PyPI 官方 ~5KB/s，清华 ~18MB/s，差约 3000 倍。
    torch 2.2.2 的 macOS 轮子有 200MB+，用官方源能把首次启动拖到几十分钟。
    设了 PIP_INDEX_URL 就听用户的（公司内网源 / 官方直连快的环境）。
    run.py 装主依赖、server 装可选依赖走的是同一份逻辑。 """
    if os.environ.get("PIP_INDEX_URL"):
        return []
    return ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"]


def optional_install_cmd(py: str, pkgs: list, req_file: str) -> list:
    """ 装可选依赖（cv2 / soundfile）的 pip 命令。

    **必须带 `-c start/requirements/core.txt`**：实测不带约束时，pip 解析
    opencv-python / soundfile 会顺手把 numpy 升到 2.x，而 torch==2.2.2
    只支持 numpy<2——一个"可选"安装就把主依赖契约悄悄破坏了，事后极难
    排查（症状是 numpy 行为变化，没人会联想到上次装了 cv2）。
    constraints 文件只约束版本、不会额外安装里面的包，正好合适。 """
    cmd = [py, "-m", "pip", "install"]
    if req_file and os.path.exists(req_file):
        cmd += ["-c", req_file]
    cmd += pip_index_args() + list(pkgs)
    return cmd
