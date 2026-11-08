# ===========================================================================
# GARGANTUA v1 — service/cli/api_map.py：GUI 操作 → 终端命名命令映射
# ---------------------------------------------------------------------------
# 本表是「所有 GUI 操作都有对应终端命令」的可验证证据的唯一事实来源。
# key = "group action"（单段命令直接用 "status" 这类单词）；value 至少含
#   {"method": "GET"|"POST", "path": "/api/...", "help": "中文说明"}，
# 可选 {"positionals": [...], "defaults": {...}}。
#
# 覆盖 server/server.py do_GET 的全部 GET 分支与 do_POST 的 routes 字典，
# 一个端点都不漏。注意：
#   * /api/vocab/source 同时是 GET 与 POST → 用两个不同 key
#     （vocab source-get / vocab source-set）。
#   * /api/task/<id> 是动态路径，<id> 由 `task get <id>` 位置参数替换。
# endpoint_set() 返回被命名命令覆盖的 path 集合（/api/task/<id> 原样返回），
# 供静态测试与 server.py 路由表比对。
#
# 本模块不 import 任何项目内模块（纯数据），可被静态测试独立导入。
# ===========================================================================
"""api_map: 全部 GUI 端点 → 终端命名命令的映射表。"""
from __future__ import annotations


def _c(method: str, path: str, help_: str, positionals=None, defaults=None):
    d = {"method": method, "path": path, "help": help_}
    if positionals:
        d["positionals"] = list(positionals)
    if defaults:
        d["defaults"] = dict(defaults)
    return d


API_COMMANDS: dict[str, dict] = {
    # ------------------------------------------------------------------ GET
    "status":            _c("GET", "/api/status", "服务器总体状态"),
    "models list":       _c("GET", "/api/models", "列出已注册模型"),
    "session list":      _c("GET", "/api/sessions", "列出全部会话"),
    "session get":       _c("GET", "/api/session/get", "读取指定会话",
                            positionals=["id"]),
    "env deps":          _c("GET", "/api/env/deps", "依赖清单与安装状态"),
    "env binaries":      _c("GET", "/api/env/binaries", "内置二进制状态"),
    "env versions":      _c("GET", "/api/env/versions", "依赖版本信息"),
    "task get":          _c("GET", "/api/task/<id>", "查询异步任务",
                            positionals=["id"]),
    "train status":      _c("GET", "/api/train/status", "训练服务状态"),
    "train queue":       _c("GET", "/api/train/queue", "训练队列列表"),
    "train events":      _c("GET", "/api/train/events", "训练事件流"),
    "train loss":        _c("GET", "/api/train/loss", "按文件分组的 loss 曲线数据"),
    "train dataset-spec": _c("GET", "/api/train/dataset/spec",
                             "CLIP 数据集文件夹格式 v1 说明"),
    "system status":     _c("GET", "/api/system", "系统总览"),
    "system devices":    _c("GET", "/api/system/devices", "计算设备列表"),
    "system processes":  _c("GET", "/api/system/processes", "进程列表"),
    "term poll":         _c("GET", "/api/term/poll", "拉取终端输出"),
    "prims log":         _c("GET", "/api/prims/log", "图元日志"),
    "prims jobs":        _c("GET", "/api/prims/jobs", "图元任务列表"),
    "prims live":        _c("GET", "/api/prims/job/live", "图元任务实时画面"),
    "prims result":      _c("GET", "/api/prims/job/result", "图元任务结果"),
    "prims source":      _c("GET", "/api/prims/job/source", "图元任务源图（二进制）"),
    "system updates":    _c("GET", "/api/system/updates", "可更新依赖"),
    "system versions":   _c("GET", "/api/system/versions", "依赖可用版本"),
    "infer stats":       _c("GET", "/api/infer/stats", "推理统计"),
    "infer stream":      _c("GET", "/api/infer/stream", "流式推理输出"),
    "plugin list":       _c("GET", "/api/plugin/list", "插件列表"),
    "agent spec":        _c("GET", "/api/agent/spec", "Agent 规格"),
    "agent tools":       _c("GET", "/api/agent/tools", "Agent 工具列表"),
    "vocab get":         _c("GET", "/api/vocab", "当前词表信息"),
    "vocab list":        _c("GET", "/api/vocab/list", "词表文件列表"),
    "vocab regions":     _c("GET", "/api/vocab/regions", "词表分区"),
    "vocab source-get":  _c("GET", "/api/vocab/source", "读取词表源"),

    # ----------------------------------------------------------------- POST
    "model estimate":        _c("POST", "/api/model/estimate", "估算模型参数量"),
    "model init":            _c("POST", "/api/model/init", "初始化新模型"),
    "model load":            _c("POST", "/api/model/load", "加载模型",
                                positionals=["name"]),
    "model delete":          _c("POST", "/api/model/delete", "删除模型",
                                positionals=["name"]),
    "infer run":             _c("POST", "/api/infer", "执行推理"),
    "infer reset":           _c("POST", "/api/infer/reset", "重置推理上下文"),
    "plugin import":         _c("POST", "/api/plugin/import", "导入插件"),
    "plugin remove":         _c("POST", "/api/plugin/remove", "移除插件"),
    "train enqueue":         _c("POST", "/api/train/enqueue", "上传训练数据入队"),
    "train start":           _c("POST", "/api/train/start", "启动训练"),
    "train stop":            _c("POST", "/api/train/stop", "停止训练"),
    "train delete":          _c("POST", "/api/train/delete", "删除队列项",
                                positionals=["name"]),
    "train task-update":     _c("POST", "/api/train/task/update", "编辑未完成任务"),
    "codec image":           _c("POST", "/api/codec/image", "图片编码"),
    "codec frames":          _c("POST", "/api/codec/frames", "帧序列编码"),
    "codec audio":           _c("POST", "/api/codec/audio", "音频编码"),
    "codec draw":            _c("POST", "/api/codec/draw", "绘图编码"),
    "codec text":            _c("POST", "/api/codec/text", "文本编码"),
    "agent encode":          _c("POST", "/api/agent/encode", "Agent 编码"),
    "agent decode":          _c("POST", "/api/agent/decode", "Agent 解码"),
    "agent file":            _c("POST", "/api/agent/file", "Agent 文件"),
    "agent tool":            _c("POST", "/api/agent/tool", "Agent 工具调用"),
    "agent tool-result":     _c("POST", "/api/agent/tool_result", "Agent 工具结果"),
    "codec video":           _c("POST", "/api/codec/video", "视频编码"),
    "codec image2prims":     _c("POST", "/api/codec/image2prims", "图片转图元"),
    "tokens encode":         _c("POST", "/api/tokens/encode", "文本→.tokens（HTTP 版）"),
    "tokens decode":         _c("POST", "/api/tokens/decode", ".tokens→显示文本（HTTP 版）"),
    "tokens export":         _c("POST", "/api/tokens/export", "显示文本→.tokens（HTTP 版）"),
    "vocab token":           _c("POST", "/api/vocab/token", "词表 token 操作"),
    "vocab source-set":      _c("POST", "/api/vocab/source", "设置词表源"),
    "vocab fork":            _c("POST", "/api/vocab/fork", "派生词表"),
    "vocab copy":            _c("POST", "/api/vocab/copy", "复制词表"),
    "vocab create":          _c("POST", "/api/vocab/create", "新建词表"),
    "vocab delete":          _c("POST", "/api/vocab/delete", "删除词表"),
    "system install-optional": _c("POST", "/api/system/install-optional", "安装可选依赖"),
    "system install":        _c("POST", "/api/system/install", "安装依赖"),
    "system kill":           _c("POST", "/api/system/kill", "结束进程"),
    "term open":             _c("POST", "/api/term/open", "打开终端"),
    "term input":            _c("POST", "/api/term/input", "终端输入"),
    "term interrupt":        _c("POST", "/api/term/interrupt", "终端中断"),
    "term kill":             _c("POST", "/api/term/kill", "终止终端"),
    "term clear":            _c("POST", "/api/term/clear", "清空终端"),
    "term close":            _c("POST", "/api/term/close", "关闭终端"),
    "prims create":          _c("POST", "/api/prims/job/create", "创建图元任务"),
    "prims update":          _c("POST", "/api/prims/job/update", "更新图元任务"),
    "prims delete":          _c("POST", "/api/prims/job/delete", "删除图元任务"),
    "session create":        _c("POST", "/api/session/create", "新建会话"),
    "session save":          _c("POST", "/api/session/save", "保存会话"),
    "session append":        _c("POST", "/api/session/append", "追加会话消息"),
    "session delete":        _c("POST", "/api/session/delete", "删除会话"),
    "infer clear-fw":        _c("POST", "/api/infer/clear_fw", "清空快权重"),
    # 2026-11-08：MoMW 三阶段记忆阀门的会话级控制。action 位置参数取
    # clear（丢掉会话记忆池、下次前向从零重建）或 silent（路由与写入照常，
    # 但不把 ΔH 注入输出）；要关静默用 api 兜底传 value=false。
    "momw control":          _c("POST", "/api/momw/control",
                                "MoMW 记忆阀门控制（clear / silent）",
                                positionals=["action"]),
    "model preload":         _c("POST", "/api/model/preload", "预加载模型"),
    "model unload":          _c("POST", "/api/model/unload", "卸载模型"),
    "model select":          _c("POST", "/api/model/select", "选择模型"),
    "env install":           _c("POST", "/api/env/install", "安装环境依赖"),
    "env remove":            _c("POST", "/api/env/remove", "移除环境依赖"),
}


def endpoint_set() -> set[str]:
    """返回被命名命令覆盖的 path 集合（/api/task/<id> 原样返回）。

    GET 30 + POST 55 个 method+path 组合，去重后 84 个 path（因为
    /api/vocab/source 的 GET 与 POST 共享同一个 path）。
    """
    return {c["path"] for c in API_COMMANDS.values()}
