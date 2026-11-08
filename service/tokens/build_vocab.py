#!/usr/bin/env python3
# ===========================================================================
# GARGANTUA v1 — 构建全局基础词表 tokens/{vocab.vfc, config.json}
# ---------------------------------------------------------------------------
# 布局严格依照 ROOT/spec.py（冻结契约），禁止硬编码区间。
# 产物（词表文件夹契约，与 vocabs/<name>/ 完全同构）：
#   vocab.vfc    词表正文（一行一个 token，见 tokens/vfc.py）
#   config.json  vocab_size / regions 分区表 / user_added 重放源
# 总长度 = 架构契约头 + 各可见子区实际收录数（不 pad、不留空位）。
#
# 内容来源：
#   - emoji：内嵌 EMOJI_RANGES（Unicode 16.0.0 emoji-data.txt 的
#     Emoji=Yes 码点，非 ASCII；构建离线可复现）。
#   - 符号：运行时 unicodedata 的全部 P*/S* 类码点（排除 emoji 区与
#     CJK 区）+ 全角字母数字 62 个；构建结果随 UCD 版本可能微调，
#     meta 记录 unidata_version 与实际数量。
#   - 英文词：tokens/words_en.txt（bundled，hermitdave/FrequencyWords
#     词频序前 30000，单字母词除外）；fallback 链见 load_english_words。
#   - 用户新增 token：config.json 的 "user_added" 自动重放，重建不丢数据。
#
# 用法：
#   cd ROOT && python tokens/build_vocab.py [--words-file PATH]
# ===========================================================================
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---- GARGANTUA 路径引导（目录树重构：把 server/ 与 service/* 放上 path）----
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
import spec  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
VFC_PATH = Path(__file__).resolve().parent / spec.VOCAB_FILE_NAME

# 大类标签（与 tokens/vocab.py 同源，这里只用来标注分区表）
CAT_REGULAR = "regular"
CAT_FALLBACK = "fallback"
CAT_SPECIAL = "special"
CAT_SYSTEM = "system"
WORDS_PATH = Path(__file__).resolve().parent / "words_en.txt"

EN_FREQ_URL = (
    "https://raw.githubusercontent.com/hermitdave/FrequencyWords/"
    "master/content/2016/en/en_full.txt"
)
EN_FALLBACK_URL = (
    "https://raw.githubusercontent.com/first20hours/"
    "google-10000-english/master/google-10000-english.txt"
)
# 英文词形：全小写 [a-z]+，允许一个撇号后缀（don't / it's 等原形收录）
EN_WORD_RE = re.compile(r"^[a-z]+(?:'[a-z]+)?$")

# Unicode 16.0.0 emoji-data.txt 中 Emoji=Yes 的码点（已排除 ASCII 键帽
# 基字符 # * 0–9——它们属于 SYMBOL/DIGIT 区），压缩为闭区间列表。
EMOJI_RANGES = [
    (0xA9, 0xA9), (0xAE, 0xAE), (0x203C, 0x203C), (0x2049, 0x2049), (0x2122, 0x2122), (0x2139, 0x2139), (0x2194, 0x2199), (0x21A9, 0x21AA), (0x231A, 0x231B), (0x2328, 0x2328), (0x23CF, 0x23CF), (0x23E9, 0x23F3), (0x23F8, 0x23FA), (0x24C2, 0x24C2), (0x25AA, 0x25AB), (0x25B6, 0x25B6), (0x25C0, 0x25C0), (0x25FB, 0x25FE), (0x2600, 0x2604), (0x260E, 0x260E), (0x2611, 0x2611), (0x2614, 0x2615), (0x2618, 0x2618), (0x261D, 0x261D), (0x2620, 0x2620), (0x2622, 0x2623), (0x2626, 0x2626), (0x262A, 0x262A), (0x262E, 0x262F), (0x2638, 0x263A), (0x2640, 0x2640), (0x2642, 0x2642), (0x2648, 0x2653), (0x265F, 0x2660), (0x2663, 0x2663), (0x2665, 0x2666), (0x2668, 0x2668), (0x267B, 0x267B), (0x267E, 0x267F), (0x2692, 0x2697), (0x2699, 0x2699), (0x269B, 0x269C), (0x26A0, 0x26A1), (0x26A7, 0x26A7), (0x26AA, 0x26AB), (0x26B0, 0x26B1), (0x26BD, 0x26BE), (0x26C4, 0x26C5), (0x26C8, 0x26C8), (0x26CE, 0x26CF), (0x26D1, 0x26D1), (0x26D3, 0x26D4), (0x26E9, 0x26EA), (0x26F0, 0x26F5), (0x26F7, 0x26FA), (0x26FD, 0x26FD), (0x2702, 0x2702), (0x2705, 0x2705), (0x2708, 0x270D), (0x270F, 0x270F), (0x2712, 0x2712), (0x2714, 0x2714), (0x2716, 0x2716), (0x271D, 0x271D), (0x2721, 0x2721), (0x2728, 0x2728), (0x2733, 0x2734), (0x2744, 0x2744), (0x2747, 0x2747), (0x274C, 0x274C), (0x274E, 0x274E), (0x2753, 0x2755), (0x2757, 0x2757), (0x2763, 0x2764), (0x2795, 0x2797), (0x27A1, 0x27A1), (0x27B0, 0x27B0), (0x27BF, 0x27BF), (0x2934, 0x2935), (0x2B05, 0x2B07), (0x2B1B, 0x2B1C), (0x2B50, 0x2B50), (0x2B55, 0x2B55), (0x3030, 0x3030), (0x303D, 0x303D), (0x3297, 0x3297), (0x3299, 0x3299), (0x1F004, 0x1F004), (0x1F0CF, 0x1F0CF), (0x1F170, 0x1F171), (0x1F17E, 0x1F17F), (0x1F18E, 0x1F18E), (0x1F191, 0x1F19A), (0x1F1E6, 0x1F1FF), (0x1F201, 0x1F202), (0x1F21A, 0x1F21A), (0x1F22F, 0x1F22F), (0x1F232, 0x1F23A), (0x1F250, 0x1F251), (0x1F300, 0x1F321), (0x1F324, 0x1F393), (0x1F396, 0x1F397), (0x1F399, 0x1F39B), (0x1F39E, 0x1F3F0), (0x1F3F3, 0x1F3F5), (0x1F3F7, 0x1F4FD), (0x1F4FF, 0x1F53D), (0x1F549, 0x1F54E), (0x1F550, 0x1F567), (0x1F56F, 0x1F570), (0x1F573, 0x1F57A), (0x1F587, 0x1F587), (0x1F58A, 0x1F58D), (0x1F590, 0x1F590), (0x1F595, 0x1F596), (0x1F5A4, 0x1F5A5), (0x1F5A8, 0x1F5A8), (0x1F5B1, 0x1F5B2), (0x1F5BC, 0x1F5BC), (0x1F5C2, 0x1F5C4), (0x1F5D1, 0x1F5D3), (0x1F5DC, 0x1F5DE), (0x1F5E1, 0x1F5E1), (0x1F5E3, 0x1F5E3), (0x1F5E8, 0x1F5E8), (0x1F5EF, 0x1F5EF), (0x1F5F3, 0x1F5F3), (0x1F5FA, 0x1F64F), (0x1F680, 0x1F6C5), (0x1F6CB, 0x1F6D2), (0x1F6D5, 0x1F6D7), (0x1F6DC, 0x1F6E5), (0x1F6E9, 0x1F6E9), (0x1F6EB, 0x1F6EC), (0x1F6F0, 0x1F6F0), (0x1F6F3, 0x1F6FC), (0x1F7E0, 0x1F7EB), (0x1F7F0, 0x1F7F0), (0x1F90C, 0x1F93A), (0x1F93C, 0x1F945), (0x1F947, 0x1F9FF), (0x1FA70, 0x1FA7C), (0x1FA80, 0x1FA89), (0x1FA8F, 0x1FAC6), (0x1FACE, 0x1FADC), (0x1FADF, 0x1FAE9), (0x1FAF0, 0x1FAF8),
]

# emoji 组件字符（emoji-data.txt 的 Emoji_Component 属性中不属于 Emoji
# 属性、且未被其他区收录的码点）：ZWJ / VS16 / 分区旗帜标签（E0020–E007F）。
# ASCII 键帽基字符（# * 0–9）已在 SYMBOL/DIGIT 区；肤色修饰与区域指示符
# 已在 Emoji 属性内。不收录这些组件会导致 ZWJ 序列 / 分区旗帜有损。
EMOJI_COMPONENT_EXTRA = [0x200D, 0xFE0F] + list(range(0xE0020, 0xE0080))

# 全角字母数字（U+FF10–FF19 / U+FF21–FF3A / U+FF41–FF5B）：不属于 P*/S*
# 类，但为保证全角文本无损 roundtrip 显式收入 SYMBOL 区。
FULLWIDTH_ALNUM = (
    list(range(0xFF10, 0xFF1A))
    + list(range(0xFF21, 0xFF3B))
    + list(range(0xFF41, 0xFF5B))
)


# ---------------------------------------------------------------------------
# 代码专用 token 区（2026-11-06，base1.1.0 新增）
# ---------------------------------------------------------------------------
# 跨语言常用语法 / 标识符 / 运算符序列，作为「可见 regular」token 收入词表。
# 收益：代码语料里这些高频片段各占一个离散 token，模型不必把 `async`
# `await` `useState` `rgba` 之类拆成字母级碎片，提升代码建模效率。
#
# 设计约束（见 build_entries 的去重）：
#   - 文本全表唯一（除 { } | 三系统符）；所以本清单里若某 token 已存在于
#     英文词区（class/for/if/new/var/int/return…）或符号区，会被静默跳过。
#     实测 821 条候选 → 311 条撞名 + 36 条清单内部重复 → 实收 474 条。
#   - 唯一硬约束：**不得构成 `<名字>` / `[名字]` 整词形态**（vfc._is_marked），
#     否则加载时会被当成不可显示标记。可见 token 一律用引号包裹写出，文本
#     内容不进 token-6 字符流（那里是 {t:ID}），所以 `</`、`/>`、`>>`、`&&`
#     这类含尖括号/竖线的多字符片段都安全，实测 .vfc 往返零错误。
#   - 单字符运算符（( ) = + - * / < > ! ? : ; , # @ $ % & ^ ~ " ' ` \ 等）已在
#     symbol 区收录，本清单只放「多字符」序列。
#   - 清单内部重复是刻意允许的（跨语言关键字天然重名，如 readonly 在
#     TS/C#/HTML 各有），_emit_code 会去重，不必人工合并。
CODE_TOKENS = [
    # —— 多字符运算符 / 标点序列（安全字符）——
    "==", "!=", ">=", ">>", "&&", "||", "++", "--", "->", "::", "//",
    "+=", "-=", "*=", "/=", "===", "!==", "**", "...", "?.", "??",
    "@@", "##", "%%", "&>", "#!", "/*", "*/", "/**", "/>", "</", "/>",
    # —— Python ——
    "def", "class", "import", "from", "as", "lambda", "return", "yield",
    "global", "nonlocal", "raise", "except", "finally", "with", "elif",
    "assert", "pass", "break", "continue", "async", "await", "None",
    "True", "False", "self", "cls", "print", "len", "range", "open",
    "read", "write", "split", "join", "replace", "strip", "format",
    "type", "isinstance", "enumerate", "zip", "map", "filter", "sum",
    "sorted", "reversed", "list", "dict", "tuple", "set", "str", "int",
    "float", "bool", "super", "args", "kwargs", "init", "repr", "hash",
    "dir", "eval", "exec", "getattr", "setattr", "delattr", "hasattr",
    "ord", "chr", "bin", "hex", "oct", "abs", "round", "pow", "divmod",
    "iter", "next", "callable", "property", "staticmethod", "classmethod",
    "__init__", "__str__", "__repr__", "__len__", "__call__", "__main__",
    "__name__", "__file__", "numpy", "pandas", "torch", "tensorflow",
    "keras", "sklearn", "scipy", "matplotlib", "plt", "np", "pd", "json",
    "subprocess", "pathlib", "collections", "itertools", "functools",
    "dataclass", "typing", "Union", "Optional", "Callable", "Generator",
    "bytes", "bytearray", "memoryview", "frozenset", "slice", "complex",
    "ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError",
    "StopIteration", "NotImplementedError", "ImportError",
    "FileNotFoundError", "recursion",
    # —— JavaScript / TypeScript / Node ——
    "function", "var", "let", "const", "new", "this", "undefined", "NaN",
    "Promise", "fetch", "require", "module", "exports", "console", "window",
    "document", "JSON", "parseInt", "parseFloat", "toString",
    "querySelector", "querySelectorAll", "addEventListener",
    "getElementById", "createElement", "appendChild", "removeChild",
    "classList", "innerHTML", "textContent", "useState", "useEffect",
    "useMemo", "useRef", "useCallback", "useReducer", "useContext",
    "render", "component", "props", "hooks", "router", "store", "dispatch",
    "commit", "mutations", "getters", "actions", "axios", "react",
    "reactdom", "vue", "angular", "jquery", "dom", "event", "listener",
    "canvas", "ctx", "requestAnimationFrame", "setInterval", "setTimeout",
    "clearInterval", "clearTimeout", "Map", "Set", "Symbol", "Proxy",
    "Reflect", "BigInt", "WeakMap", "WeakSet", "arrow", "template",
    "spread", "destructure", "node", "npm", "yarn", "webpack", "babel",
    "eslint", "prettier", "typescript", "implements", "namespace",
    "readonly", "abstract", "declare", "keyof", "infer", "never",
    "unknown", "interface",
    # —— Java ——
    "package", "extends", "public", "private", "protected", "static",
    "final", "abstract", "void", "long", "double", "char", "byte", "short",
    "boolean", "String", "Integer", "Double", "Float", "Long", "Boolean",
    "Character", "Short", "null", "instanceof", "synchronized", "volatile",
    "transient", "native", "strictfp", "enum", "override", "Override",
    "equals", "hashCode", "getClass", "System", "println", "Scanner",
    "ArrayList", "HashMap", "HashSet", "LinkedList", "TreeMap", "TreeSet",
    "ConcurrentHashMap", "Arrays", "Collections", "Stream", "Runnable",
    "Thread", "Callable", "Future", "IOException", "SQLException",
    "NullPointerException", "main",
    # —— C / C++ ——
    "unsigned", "signed", "const", "extern", "auto", "register",
    "mutable", "struct", "union", "typedef", "sizeof", "inline", "virtual",
    "constexpr", "noexcept", "template", "typename", "using", "friend",
    "operator", "nullptr", "goto", "std", "cout", "cin", "cerr", "clog",
    "endl", "string", "vector", "unordered_map", "unordered_set", "array",
    "pair", "tuple", "deque", "priority_queue", "shared_ptr", "unique_ptr",
    "weak_ptr", "algorithm", "iterator", "emplace_back", "push_back",
    "printf", "scanf", "sprintf", "fprintf", "snprintf", "fopen", "fclose",
    "fread", "fwrite", "malloc", "free", "memcpy", "memset", "strlen",
    "strcpy", "strcat", "iostream", "fstream", "sstream", "iomanip",
    "cmath", "cstring", "cstdlib", "cstdio", "thread", "mutex", "atomic",
    "chrono", "functional", "define", "ifndef", "ifdef", "pragma", "argv",
    "argc", "NULL", "EXIT_SUCCESS", "EXIT_FAILURE",
    # —— C# ——
    "internal", "readonly", "sealed", "partial", "as", "is", "typeof",
    "decimal", "object", "dynamic", "foreach", "lock", "delegate", "event",
    "get", "set", "out", "ref", "params", "base", "Console", "WriteLine",
    "Write", "StringBuilder", "Dictionary", "HashSet", "Task", "LINQ",
    "Convert", "DateTime", "Int32", "Int64",
    # —— PHP ——
    "echo", "elseif", "foreach", "isset", "empty", "array", "global",
    "require_once", "include_once", "trait", "construct", "die", "exit",
    "json_encode", "json_decode", "mysqli", "PDO", "var_dump", "print_r",
    # —— SQL ——
    "SELECT", "INSERT", "VALUES", "UPDATE", "DELETE", "CREATE", "DROP",
    "ALTER", "INDEX", "INNER", "OUTER", "GROUP", "HAVING", "DISTINCT",
    "BETWEEN", "COUNT", "AVG", "UNION", "WHEN", "THEN", "PRIMARY",
    "FOREIGN", "REFERENCES", "AUTO_INCREMENT", "UNIQUE", "CONSTRAINT",
    "TRANSACTION", "ROLLBACK", "EXPLAIN", "DESCRIBE", "IFNULL", "COALESCE",
    "CAST", "CONVERT",
    # —— HTML / CSS ——
    "html", "head", "body", "div", "span", "img", "table", "select",
    "option", "textarea", "label", "nav", "header", "footer", "section",
    "article", "aside", "canvas", "svg", "path", "rect", "circle",
    "ellipse", "defs", "charset", "viewport", "rel", "srcset", "placeholder",
    "required", "disabled", "checked", "readonly", "multiple", "selected",
    "enctype", "selector", "background-color", "background-image",
    "background-size", "background-position", "background-repeat",
    "border-color", "border-width", "border-style", "border-radius",
    "margin-top", "margin-right", "margin-bottom", "margin-left",
    "padding-top", "padding-right", "padding-bottom", "padding-left",
    "inline-block", "position", "absolute", "relative", "fixed", "sticky",
    "translate", "rotate", "scale", "transition-duration",
    "transition-property", "transition-timing", "keyframes", "font-size",
    "font-weight", "font-family", "font-style", "line-height", "text-align",
    "text-decoration", "overflow-x", "overflow-y", "box-sizing", "pointer",
    "first-child", "last-child", "nth-child", "before", "after", "calc",
    "linear-gradient", "radial-gradient", "rgba", "hsl", "hsla", "vh", "vw",
    "important", "flex-direction", "justify-content", "align-content",
    "flex-wrap", "grid-template-columns", "grid-template-rows", "row-gap",
    "column-gap", "px", "em", "rem",
    # —— Shell / Bash ——
    "then", "elif", "fi", "done", "esac", "function", "export", "source",
    "readonly", "unset", "mkdir", "rmdir", "chmod", "chown", "gzip",
    "gunzip", "xargs", "select", "until",
    # —— Go ——
    "func", "chan", "defer", "range", "fallthrough", "iota", "nil",
    "error", "int64", "int32", "float64", "rune", "make", "cap", "append",
    "copy", "panic", "recover", "WaitGroup", "RWMutex", "Printf",
    # —— Rust ——
    "mut", "trait", "impl", "loop", "match", "move", "ref", "mod", "pub",
    "crate", "Self", "super", "where", "dyn", "unsafe", "macro", "vec",
    "println", "Option", "Some", "Result", "Ok", "Err", "Vec", "HashMap",
    "HashSet", "Rc", "Arc", "RefCell", "Cell", "Cow", "Clone", "Copy",
    "Drop", "Iterator", "iter", "collect", "unwrap", "expect", "derive",
    "Debug", "Display", "Send", "Sync", "usize", "u8", "u16", "u32", "u64",
    "i8", "i16", "i32", "i64", "f32", "f64",
    # —— Kotlin / Swift 片段 ——
    "val", "fun", "when", "companion", "suspend", "lateinit", "guard",
    "protocol", "extension", "deinit", "optional",
    # —— 通用工程词 ——
    "TODO", "FIXME", "NOTE", "DEBUG", "INFO", "WARN", "ERROR", "logger",
    "trace", "info", "warn", "fatal", "exception", "handler", "callback",
    "middleware", "factory", "singleton", "builder", "adapter", "observer",
    "iterator", "generator", "serializer", "deserializer", "compute",
    "execute", "invoke", "schedule", "register", "subscribe", "publish",
    "header", "status", "method", "url", "uri", "host", "port", "buffer",
    "reader", "writer", "channel", "stack", "config", "settings", "options",
    "argument", "session", "cookie", "payload", "deadline", "cancel",
    "signal", "process", "worker", "daemon", "cluster", "replica", "shard",
    "query", "scan", "aggregate", "cursor", "savepoint", "bind", "fetch",
    "encrypt", "decrypt", "sign", "verify", "auth", "watch", "observe",
    "reconcile", "diff", "patch", "clone", "vacuum", "analyze", "prepare",
    "endpoint", "middleware", "request", "response", "server", "client",
    "service", "gateway", "proxy", "token", "agent", "prompt", "dataset",
    "tensor", "matrix", "gradient", "optimizer", "epoch", "accuracy",
    "precision", "recall", "confusion", "pipeline", "workflow", "partition",
    "compression", "decompression", "serialization", "deserialization",
    "encrypt", "decrypt", "signature", "verification", "authentication",
    "authorization", "middleware",     "microservice", "container", "kubernetes",
    "docker", "registry", "repository", "controller", "viewmodel", "layout",
    "widget", "fragment", "activity", "intent", "broadcast", "provider",
    "sync",
]


def _emit_code(seq, region: str, category: str, known: set,
               entries: list, tokens: list, regions: list) -> int:
    """把代码候选 token 去重后追加进可见区（code 区）。

    去重两层：(1) 候选清单内部可能重复（手写冗余）；(2) 与已有词表文本
    撞名（英文词区 / 符号区 / 用户新增等已收录的，如 class/for/if 这类
    常见英文词）——撞名的直接跳过，不重复收录。返回实际新增条数。

    entries / tokens / regions 由调用方（build_entries）传入，因为本函数
    是模块级定义，无法闭包到 build_entries 的局部变量。
    """
    added = 0
    for t in seq:
        if not isinstance(t, str) or not t:
            continue
        if t in known:
            continue
        known.add(t)
        entries.append((len(entries), t, category != CAT_SPECIAL))
        tokens.append(t)
        added += 1
    regions.append({"region": region, "category": category, "size": added})
    return added


# ---------------------------------------------------------------------------
# Unicode 扩展字符区（2026-11-07，base1.3.0 新增）
# ---------------------------------------------------------------------------
# 唯一真源：桌面文档「词表新增内容.md」——31 个 Unicode 区块、合计 3537 字。
# 口径（已用 base1.2.0 词表逐块实测复现，31/31 全中）：
#   区块内 UCD 已分配（unicodedata.name 非空）、且**不在已构建词表**、
#   且不是大写 ASCII A–Z（大小写折叠由 fallback 小写覆盖）的单字符。
# 每块的期望数写进表里，生成时逐块断言：文档数字一旦与 UCD 版本漂移，
# 重建会当场报错，而不是静默少收 / 多收。区块范围互不重叠。
UNICODE_EXT_BLOCKS = (
    (0x0080, 0x00FF, "Latin-1 Supplement", 72),
    (0x0100, 0x017F, "Latin Extended-A", 128),
    (0x0180, 0x024F, "Latin Extended-B", 208),
    (0x2C60, 0x2C7F, "Latin Extended-C", 32),
    (0xA720, 0xA7FF, "Latin Extended-D", 189),
    (0xAB30, 0xAB6F, "Latin Extended-E", 57),
    (0x0250, 0x02AF, "IPA Extensions", 96),
    (0x02B0, 0x02FF, "Spacing Modifier Letters", 37),
    (0x1D00, 0x1D7F, "Phonetic Extensions", 128),
    (0x1D80, 0x1DBF, "Phonetic Extensions Supplement", 64),
    (0x0300, 0x036F, "Combining Diacritical Marks", 112),
    (0x1AB0, 0x1AFF, "Combining Diacritical Marks Extended", 31),
    (0x1DC0, 0x1DFF, "Combining Diacritical Marks Supplement", 64),
    (0x0400, 0x04FF, "Cyrillic", 255),
    (0x0500, 0x052F, "Cyrillic Supplement", 48),
    (0x2DE0, 0x2DFF, "Cyrillic Extended-A", 32),
    (0xA640, 0xA69F, "Cyrillic Extended-B", 94),
    (0x0370, 0x03FF, "Greek and Coptic", 129),
    (0x1F00, 0x1FFF, "Greek Extended", 218),
    (0x3040, 0x309F, "Hiragana", 91),
    (0x30A0, 0x30FF, "Katakana", 94),
    (0x31F0, 0x31FF, "Katakana Phonetic Extensions", 16),
    (0x1D400, 0x1D7FF, "Mathematical Alphanumeric Symbols", 986),
    (0x2070, 0x209F, "Superscripts and Subscripts", 32),
    (0x2100, 0x214F, "Letterlike Symbols", 45),
    (0x2150, 0x218F, "Number Forms", 58),
    (0x2460, 0x24FF, "Enclosed Alphanumerics", 82),
    (0x2000, 0x206F, "General Punctuation", 39),
    (0x2700, 0x27BF, "Dingbats", 30),
    (0x1F100, 0x1F1FF, "Enclosed Alphanumeric Supplement", 13),
    (0xFB00, 0xFB4F, "Alphabetic Presentation Forms", 57),
)
UNICODE_EXT_TOTAL = sum(b[3] for b in UNICODE_EXT_BLOCKS)   # 3537
assert UNICODE_EXT_TOTAL == 3537
# 区块不重叠：顺序按文档（法语/拉丁 → 西里尔 → 希腊 → 假名 → 数学/符号）
_spans = sorted((a, b) for a, b, _n, _c in UNICODE_EXT_BLOCKS)
assert all(_spans[i][1] < _spans[i + 1][0]
           for i in range(len(_spans) - 1)), "Unicode 扩展区块重叠"


def unicode_ext_chars(known: set) -> list:
    """31 个区块内 base 未逻辑覆盖的已分配字符（按文档顺序、块内码点升序）。

    known = 已构建进词表的全部文本（含契约头 / digit / symbol / emoji /
    cjk / english / extra / fallback / code），用它做去重，保证「同一字符串
    全表唯一」。大写 A–Z 走 enctok 的 <caps> + fallback 小写折叠，逻辑上
    已覆盖，不重复收录。
    """
    out = []
    for a, b, name, expected in UNICODE_EXT_BLOCKS:
        got = []
        for cp in range(a, b + 1):
            ch = chr(cp)
            if unicodedata.name(ch, None) is None:
                continue
            if ch in known or ("A" <= ch <= "Z"):
                continue
            got.append(ch)
        if len(got) != expected:
            raise ValueError(
                f"Unicode 扩展区块 {name} 期望 {expected} 字，实际 {len(got)} "
                f"（UCD {unicodedata.unidata_version}；文档计数与当前 UCD "
                f"或已构建词表不一致，请核对后再重建）")
        out.extend(got)
    if len(out) != len(set(out)):
        raise ValueError("Unicode 扩展区出现重复字符")
    return out


# ---------------------------------------------------------------------------
# emoji / 符号 码点集
# ---------------------------------------------------------------------------
def emoji_chars() -> list:
    """内嵌 emoji 区间 + 组件字符 → 字符列表（按码点升序）。"""
    cps = {cp for a, b in EMOJI_RANGES for cp in range(a, b + 1)}
    cps.update(EMOJI_COMPONENT_EXTRA)
    return [chr(cp) for cp in sorted(cps)]


def symbol_chars() -> list:
    """全部 Unicode P*/S* 类码点（排除 emoji 区）+ 全角字母数字。"""
    emoji = {ord(c) for c in emoji_chars()}
    out = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:          # 代理区，无字符
            continue
        if cp in emoji:
            continue
        if unicodedata.category(chr(cp))[0] in ("P", "S"):
            out.append(chr(cp))
    out.extend(chr(cp) for cp in FULLWIDTH_ALNUM)
    return sorted(set(out), key=ord)


# ---------------------------------------------------------------------------
# 英文词加载（bundled 文件优先，fallback 链与旧版一致）
# ---------------------------------------------------------------------------
def _iter_words(lines):
    """按出现顺序产出合法去重英文词（词频表每行 'word freq' 或纯词表）。"""
    seen = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        w = line.split()[0].lower()
        if EN_WORD_RE.match(w) and len(w) > 1 and w not in seen:
            # 单字母词（a / i）不收录：由 fallback 字母区承担，
            # 保证同一字符串在全词表内唯一
            seen.add(w)
            yield w


def _take(gen, limit: int):
    out = []
    for w in gen:
        out.append(w)
        if len(out) >= limit:
            break
    return out


ENGLISH_TARGET = 30000        # 用户规格：3 万个常用英文单词


def load_english_words(words_file: str | None = None,
                       source_desc: str | None = None):
    """返回 (words, source_desc)。按 fallback 链依次尝试。"""
    errors = []
    limit = ENGLISH_TARGET

    # (a) 显式指定的本地文件 / bundled 词表
    local = words_file or os.environ.get("GARGANTUA_WORDS_FILE")
    candidates = ([Path(local)] if local else []) + [WORDS_PATH]
    for p in candidates:
        if p.is_file():
            words = _take(
                _iter_words(
                    p.read_text(encoding="utf-8", errors="replace").splitlines()
                ),
                limit,
            )
            if words:
                if source_desc:
                    return words, source_desc
                # 默认 bundled 词表：只记相对名，避免把本机绝对路径写进会被
                # git 跟踪提交的 config.json（service/tokens/ 是仓库一部分）。
                if p.resolve() == WORDS_PATH.resolve():
                    return words, "file:words_en.txt"
                return words, f"file:{p}"
            errors.append(f"local file {p} yielded no valid words")
        else:
            errors.append(f"local file not found: {p}")

    # (b) hermitdave/FrequencyWords（在线）
    try:
        with urllib.request.urlopen(EN_FREQ_URL, timeout=120) as r:
            text = r.read().decode("utf-8", "replace")
        words = _take(_iter_words(text.splitlines()), limit)
        if words:
            return words, EN_FREQ_URL
        errors.append(f"{EN_FREQ_URL}: no valid words parsed")
    except Exception as e:  # noqa: BLE001
        errors.append(f"{EN_FREQ_URL}: {e!r}")

    # (c) wordfreq 包
    try:
        from wordfreq import top_n_list  # type: ignore

        words = _take(iter(top_n_list("en", 60000)), limit)
        if words:
            return words, "wordfreq:top_n_list('en', 60000)"
        errors.append("wordfreq: no valid words after filtering")
    except Exception as e:  # noqa: BLE001
        errors.append(f"wordfreq: {e!r}")

    # (d) google-10000-english
    try:
        with urllib.request.urlopen(EN_FALLBACK_URL, timeout=120) as r:
            text = r.read().decode("utf-8", "replace")
        words = _take(_iter_words(text.splitlines()), limit)
        if words:
            return words, EN_FALLBACK_URL
        errors.append(f"{EN_FALLBACK_URL}: no valid words parsed")
    except Exception as e:  # noqa: BLE001
        errors.append(f"{EN_FALLBACK_URL}: {e!r}")

    raise RuntimeError("所有英文词源均不可用：\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------------------
# 用户新增 token 重放（重建不丢失）
# ---------------------------------------------------------------------------
def load_user_added() -> list:
    """config.json 的 "user_added" → [{text, ...}, ...]（重建重放源）。"""
    if not CONFIG_PATH.is_file():
        return []
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []
    got = cfg.get("user_added")
    return list(got) if isinstance(got, list) else []


# 图元 DRAW 区的命名与 bin 数**不再在这里重复**：build_entries 直接调用
# tokens.vfc.head_entries()（唯一推导处）。这里只留一句说明，防止有人
# 又加一份副本。


def build_entries(english_words, user_added: list | None = None,
                  slot_count: int | None = None):
    """按 spec 布局拼装 id_to_token 列表（**长度随内容变化，不再 pad**）。

    2026-10-30 契约重构：删掉所有 `pad_to(...)` 预留。可见区各子区**连续**
    排布，长度就是它实际收录的条数，一个空位都不留——空位同样要占一个
    embedding 行（wte 与 lm_head 各一份 1024 维），白吃 RAM，还会在
    logits 里留下一堆永远不该被采样到的死行。
    于是**每个词表的总长度可以不同**，真值记在词表目录的 config.json。

    返回 (entries, regions, applied)：
      entries  [(id, text, visible)]  连续编号，可直接落 .vfc
      regions  [{region, category, size}]  各子区**大小**（begin 由累加得出）
      applied  实际生效的 user_added 条目
    """
    entries: list[tuple] = []
    tokens: list[str] = []
    regions: list[dict] = []

    def emit(seq, region: str, category: str):
        for _t in seq:
            entries.append((len(entries), _t, category != CAT_SPECIAL))
            tokens.append(_t)
            known.add(_t)
        regions.append({"region": region, "category": category,
                        "size": len(seq)})

    # ---- 架构契约头：core / boundary / system / slot / typst / draw ----
    # 直接复用 vfc.head_entries()：词表头只有**一处**推导，build 与加载
    # 永远不会分叉（曾经两边各写一份命名规则，spec 一改就一边对一边错）。
    # 2026-11-06：agent 区 53 个预定义工具退役，换成编号槽位；
    # core（caps/stop/draw:end）单开一段。
    # 2026-11-07（base1.3.0）：slot 之后插 typst 结构段（spec.TYPST_TOKENS）。
    # regions 顺序 = ID 顺序，必须与 head 严格一致，否则分区表会整体错位。
    # **base 零预留**：slot_count 由调用方给（默认 0），不受 spec 当前
    # 运行时布局影响；要预留请用 gen_vocab.py 从 base 派生新词表。
    n_slots = spec.BASE_SLOT_COUNT if slot_count is None else int(slot_count)
    layout = spec.head_layout(n_slots)
    from tokens import vfc as _vfc
    head = _vfc.head_entries(slot_count=n_slots)
    for _hid, _htext, _hvis in head:
        entries.append((_hid, _htext, _hvis))
        tokens.append(_htext)
    regions.append({"region": "core", "category": CAT_SPECIAL,
                    "size": spec.CORE_SIZE})
    regions.append({"region": "boundary", "category": CAT_SPECIAL,
                    "size": spec.BOUNDARY_SIZE})
    regions.append({"region": "system", "category": CAT_SYSTEM,
                    "size": spec.SYSTEM_SIZE})
    regions.append({"region": "slot", "category": CAT_SPECIAL,
                    "size": n_slots})
    regions.append({"region": "typst", "category": CAT_SPECIAL,
                    "size": spec.TYPST_SIZE})
    regions.append({"region": "draw", "category": CAT_SPECIAL,
                    "size": spec.MODAL_SIZE})
    assert len(tokens) == layout["head_size"], (
        f"架构契约头长度 {len(tokens)} != 布局 head_size "
        f"{layout['head_size']}（slot_count={n_slots}）")

    # ---- 可见区：各子区按 spec.VISIBLE_REGION_ORDER 连续排布 ----
    # 用户新增 token 统一落进 extra 区（config["append_region"]）。
    # 先按「已收录文本」去重：user_added 里若混进了基础词表本来就有的词
    # （历史上 meta 被写脏过一次），直接重放会造出同文本双 ID，整张词表
    # 会被 Vocab 的唯一性校验拒绝。
    known = {t for _i, t, _v in entries}
    extra = []
    dropped = 0
    for e in (user_added or []):
        if not isinstance(e, dict):
            continue
        t = e.get("text")
        if not isinstance(t, str) or not t or t in known:
            dropped += 1
            continue
        known.add(t)
        extra.append(t)
    emit([str(d) for d in range(10)], "digit", CAT_REGULAR)
    emit(list(spec.WHITESPACE_CHARS), "whitespace", CAT_REGULAR)
    emit(symbol_chars(), "symbol", CAT_REGULAR)
    emit(emoji_chars(), "emoji", CAT_REGULAR)
    emit([chr(cp) for cp in range(0x4E00, 0x9FFF + 1)] +
         [chr(cp) for cp in range(0x3400, 0x4DBF + 1)],
         "cjk", CAT_REGULAR)
    emit([str(w) for w in english_words], "english", CAT_REGULAR)
    emit(extra, "extra", CAT_REGULAR)
    emit(list(spec.FALLBACK_LETTERS), "fallback", CAT_FALLBACK)
    # 2026-11-06（base1.1.0）：代码专用 token 区追加在 fallback 之后、可见
    # 区末尾。known 此时已含 head + 全部可见区文本，撞名（def/class/for/
    # if/int/return 等已被英文词区收录的）会被静默跳过，不会造出双 ID。
    _emit_code(CODE_TOKENS, "code", CAT_REGULAR,
               known, entries, tokens, regions)
    # 2026-11-07（base1.3.0）：Unicode 扩展字符**真·末尾追加**，接在 code
    # 之后、可见区最后。绝不用 user_added→extra 通道（它插在 english 与
    # fallback 之间，会把 fallback+code 整体 +N 平移）。逐块断言 3537，
    # 与桌面文档「词表新增内容.md」精确对账。
    ext = unicode_ext_chars(known)
    emit(ext, "unicode_ext", CAT_REGULAR)
    assert len(ext) == UNICODE_EXT_TOTAL, (
        f"unicode_ext 实收 {len(ext)} != {UNICODE_EXT_TOTAL}")

    if len(entries) > spec.MAX_VOCAB_SIZE:
        raise ValueError(
            f"词表长度 {len(entries)} 超过安全上界 {spec.MAX_VOCAB_SIZE}")

    # applied 只能取 extra 区里的条目。旧实现用 `i >= sum(regions[:7])`
    # 当边界，而 regions[:7] 到 emoji 为止——cjk/english 全被误记成
    # 「用户新增」，下次重建就会把它们当 extra 重放一遍（文本双 ID）。
    extra_begin = 0
    extra_size = 0
    for r in regions:
        if r["region"] == "extra":
            extra_size = r["size"]
            break
        extra_begin += r["size"]
    applied = [{"text": t, "category": "regular", "id": i}
               for i, t, _v in entries[extra_begin:extra_begin + extra_size]]
    if dropped:
        print(f"[build_vocab] user_added 中 {dropped} 条与基础词表重复，"
              f"已跳过", file=sys.stderr)
    return entries, regions, applied


def build_vocab(words_file: str | None = None,
                source_desc: str | None = None,
                slot_count: int | None = None) -> dict:
    """完整构建：entries + regions + config.json 内容。

    slot_count 默认 0（base 零预留）。写进 config 的 slot_count 决定 spec
    下次 import 时的运行时布局——spec 本身不钉死这个数字。
    """
    n_slots = spec.BASE_SLOT_COUNT if slot_count is None else int(slot_count)
    layout = spec.head_layout(n_slots)
    english_words, source = load_english_words(words_file, source_desc)
    user_added = load_user_added()
    entries, regions, applied = build_entries(english_words, user_added,
                                              slot_count=n_slots)
    id_to_token = [t for _i, t, _v in entries]
    # 同一字符串在全词表内唯一，唯一例外是 3 个系统 token 的显示字符
    # （{ } |）与可见符号区撞名——设计使然：text→id 经 regular 查表
    # 永远解析到可见符号 token；id→显示时两者同为字面量，无歧义。
    seen: dict = {}
    for i, t in enumerate(id_to_token):
        if t in seen and t not in (spec.TS_OPEN, spec.TS_CLOSE, spec.TS_SEP):
            raise AssertionError(
                f"词表存在同文本双 ID: {t!r}（{seen[t]} 与 {i}）")
        seen.setdefault(t, i)
    config = {
        "name": spec.BASE_VOCAB_NAME,
        "format": "gargantua-vocab-folder-v1",
        "vocab_version": spec.VOCAB_VERSION,
        "token_encoding": spec.TOKEN_ENCODING,
        "is_main": True,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vocab_size": len(id_to_token),
        "head_size": layout["head_size"],
        "slot_count": n_slots,
        "regions": regions,
        "append_region": "extra",
        "user_added": applied,
        "sources": {
            "english": source,
            "english_word_count": len(english_words),
            "emoji": ("Unicode 16.0.0 emoji-data.txt (Emoji=Yes, 非 ASCII)"
                      " + Emoji_Component 组件字符"),
            "emoji_count": len(emoji_chars()),
            "symbol": (f"Unicode P*/S* 类码点（UCD "
                       f"{unicodedata.unidata_version}）排除 emoji + "
                       f"全角字母数字"),
            "symbol_count": len(symbol_chars()),
            "typst": ("gargantua-typst-token-spec.md §3 + §2（结构 token "
                      "唯一真源 spec.TYPST_CONSTRUCTS）"),
            "typst_construct_count": spec.TYPST_CONSTRUCT_COUNT,
            "typst_size": spec.TYPST_SIZE,
            "unicode_ext": (f"词表新增内容.md（{len(UNICODE_EXT_BLOCKS)} 个 "
                            f"区块，UCD {unicodedata.unidata_version} 已分配"
                            f"且 base 未逻辑覆盖）"),
            "unicode_ext_block_count": len(UNICODE_EXT_BLOCKS),
            "unicode_ext_count": next(
                r["size"] for r in regions if r["region"] == "unicode_ext"),
            "unidata_version": unicodedata.unidata_version,
        },
    }
    return {"entries": entries, "regions": regions, "config": config}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="构建全局基础词表 tokens/{vocab.vfc, config.json}")
    ap.add_argument("--words-file", default=None, help="本地英文词频文件路径")
    ap.add_argument(
        "--source-desc",
        default=None,
        help="config 中记录的英文词来源描述（默认按实际获取途径记录）",
    )
    ap.add_argument("--slots", type=int, default=spec.BASE_SLOT_COUNT,
                    help="预留槽位数量（默认 0 = base 零预留；"
                         "给派生词表用请改跑 gen_vocab.py）")
    args = ap.parse_args(argv)
    vocab = build_vocab(args.words_file, args.source_desc,
                        slot_count=args.slots)

    from tokens import vfc as _vfc
    # 直接用 build_entries 给出的 (id, text, visible)：visible 标记决定
    # .vfc 里是引号还是尖括号形态，system 的 { } | 必须是引号（尖括号
    # 里放不下非标识符），不能按「ID < HEAD_SIZE」重新推导。
    _vfc.write_vfc(str(VFC_PATH), vocab["entries"])
    cfg = vocab["config"]
    CONFIG_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    src = cfg["sources"]
    print(f"已写出 {VFC_PATH}")
    print(f"已写出 {CONFIG_PATH}")
    print(f"  词表长度      : {cfg['vocab_size']}")
    print(f"  契约头        : {cfg['head_size']}（含 typst "
          f"{src['typst_size']}）  DRAW_BEGIN: {spec.head_layout(cfg['slot_count'])['draw_begin']}")
    print(f"  分区数        : {len(cfg['regions'])}")
    print(f"  英文词实收    : {src['english_word_count']}（{src['english']}）")
    print(f"  emoji         : {src['emoji_count']}")
    print(f"  符号          : {src['symbol_count']}")
    print(f"  Typst 构造    : {src['typst_construct_count']} → "
          f"{src['typst_size']} token")
    print(f"  Unicode 扩展  : {src['unicode_ext_count']} "
          f"（{src['unicode_ext_block_count']} 区块）")
    print(f"  用户新增重放  : {len(cfg['user_added'])}")


if __name__ == "__main__":
    main()
