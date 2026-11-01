# ===========================================================================
# GARGANTUA v1 — 词表数据结构的加载、查询与扩充
# ---------------------------------------------------------------------------
# 【2026-10-30 词表契约重构】
#
# 单一格式：.vfc（定义在 tokens/vfc.py）。所有词表——全局 base、
# vocabs/<name>/、models/<NAME>/vocab/ —— 都用它，不再有两套表示法
# （旧的 vocab.json / vocab.txt 走兼容读取 + 就地迁移）。
#
# 核心不变量（构造 Vocab 时强制，任何一条不满足即拒绝加载）：
#   I1 连续性：ID 恰好是 0..N-1，一个空位都不留。
#              理由：每个 ID 都对应一个 embedding 行（wte 与 lm_head 各
#              一份 1024 维），空位纯属白吃 RAM，且会在 logits 里留下一
#              堆永远不该被采样的死行（副作用）。
#   I2 头部契约：前 HEAD_SIZE 个 ID 必须逐字等于 spec 冻结的
#              special / system / draw 序列（名字与顺序都不能变）。
#              理由：dectok 的停止/大小写判断、gts 的 { } | 结构、
#              draw_codec 的 bin 算术都按 ID 定位这些 token。
#              头对所有词表相同，所以压缩只改数值、不改语义。
#   I3 文本唯一：全表 token 文本唯一，反查无歧义。
#
# 由 I1 推出：删除 token 走**紧凑化**（其后所有 ID 整体前移），而不是旧的
# 软删占位 <del:N>。代价——改词表会让已训练权重里该行及之后的行错位。
# 所以「直接改」只允许在非 main 的派生词表上做，改 main 必须先 Fork
# （前端词表页的 Fork 滑块 / 弹窗即为此而设，逻辑见 vocab_presets）。
#
# 进程级覆盖：set_model_vocab(model_dir) 之后，本进程的 get_vocab()
# 返回该模型的词表（infer/train 服务进程在加载模型时调用；server 在
# 切换活动模型时调用）。没有词表就清退到全局。
# ===========================================================================
"""词表视图：连续 ID 空间 0..N-1，支持四种大类的查表与增删。"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import spec
from tokens import vfc

# 目录/文件命名取自 spec（目录树契约的唯一事实源）。这里不 import common
# 的同名常量，是因为 common.py 正被并行改造（权重三件套），尽量少动它。
VOCAB_DIR_NAME = spec.VOCAB_DIR_NAME
VOCAB_FILE_NAME = spec.VOCAB_FILE_NAME

GLOBAL_VOCAB_VFC = Path(__file__).resolve().parent / VOCAB_FILE_NAME

CAT_REGULAR = "regular"
CAT_FALLBACK = "fallback"
CAT_SPECIAL = "special"
CAT_SYSTEM = "system"
CATEGORIES = (CAT_REGULAR, CAT_FALLBACK, CAT_SPECIAL, CAT_SYSTEM)

# 新 token 默认落入的可见子区（见 config 的 "append_region"）
DEFAULT_APPEND_REGION = "extra"

def _read_sibling_config(vfc_path: Path) -> dict:
    """读 .vfc 同目录的 config.json（词表文件夹契约）；失败返回 {}。

    只做最小结构校验：必须是 JSON 对象。字段级的校验留给 Vocab（
    regions 里的非法项在 _build_regions 里按 0/默认值处理），这样一份
    手改坏的 config 不会让词表正文也读不出来。
    """
    p = Path(vfc_path).with_name("config.json")
    if not p.is_file():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:                       # noqa: BLE001
        _warn(f"{p} 读取失败，按无分区元数据处理: {e}")
        return {}
    if not isinstance(cfg, dict):
        _warn(f"{p} 不是 JSON 对象，按无分区元数据处理")
        return {}
    return cfg


# ---------------------------------------------------------------------------
# 条目规范化 / 紧凑化
# ---------------------------------------------------------------------------
def _bracket(name: str) -> str:
    """不可显示条目的**裸名** → 词表里的文本形态 '<name>'。

    .vfc 文件里尖括号是**语法分隔符**，不属于名字本身（`<stop>:5` 的名字
    是 `stop`）；而进程内 id_to_token 沿用历史形态 `<stop>`，dectok /
    gts / tok5 都按这个整词查表。两者在加载/落盘的边界上互转。
    """
    if len(name) >= 2 and name.startswith("<") and name.endswith(">"):
        return name
    return f"<{name}>"


def _unbracket(text: str) -> str:
    """'<name>' → 裸名 'name'（落盘用）。非尖括号形态原样返回。"""
    if len(text) >= 2 and text.startswith("<") and text.endswith(">"):
        return text[1:-1]
    return text


def validate_head(entries, source: str = "vocab.vfc") -> None:
    """校验架构契约头（不变量 I2）。

    头 = special + system + draw 三段，名字与顺序由 spec 冻结；头内不允许
    任何空位，所以第 k 个条目的 ID 必须恰好等于 k。
    """
    head = vfc.head_entries()
    if len(entries) < len(head):
        raise ValueError(
            f"{source}: 词表只有 {len(entries)} 条，短于架构契约头 "
            f"{len(head)} 条（special {spec.SPECIAL_SIZE} + "
            f"system {spec.SYSTEM_SIZE} + draw {spec.MODAL_SIZE}）")
    for k, (hid, htext, hvis) in enumerate(head):
        got = entries[k]
        # 两侧都用裸名比较：文件侧是裸名，内存侧可能是 '<name>' 形态
        if got[0] != hid or _unbracket(got[1]) != _unbracket(htext) \
                or bool(got[2]) != bool(hvis):
            raise ValueError(
                f"{source}: 架构契约头第 {k} 项应为 "
                f"id={hid} text={htext!r} visible={bool(hvis)}，"
                f"实际为 id={got[0]} text={got[1]!r} "
                f"visible={bool(got[2])}")


class Vocab:
    """词表视图：连续 ID 空间 0..N-1，每个 ID 恰好一个 token。

    布局：
        [0, HEAD_SIZE)            架构契约头（special + system + draw）
                                  名字与顺序由 spec 冻结，所有词表一致
        [HEAD_SIZE, vocab_size)   可见区，按 config["regions"] 连续排布

    两种 token 形态（.vfc 文件里就是引号包裹 vs 尖括号包裹）：
        可显示   → 引号包裹   走 enctok 查表；单字母 a–z 兼作 fallback 兜底
        不可显示 → 尖括号包裹  <stop>/<caps>/{ }| 这类标记，不参与文本编码
    """

    def __init__(self, entries, path=None, name=None, config=None):
        self._path = Path(path) if path else None
        self.name = name or (self._path.parent.name if self._path else None)
        self.config = dict(config or {})

        flat = sorted((tuple(e) for e in entries), key=lambda x: int(x[0]))
        n = len(flat)
        if n == 0:
            raise ValueError(f"{path or 'vocab'}: 词表为空")
        if n > spec.MAX_VOCAB_SIZE:
            raise ValueError(
                f"{path or 'vocab'}: 词表长度 {n} 超过安全上界 "
                f"{spec.MAX_VOCAB_SIZE}")
        # I1：连续性。报出全部断点，而不是撞到第一个就返回。
        gaps = [k for k, e in enumerate(flat) if int(e[0]) != k]
        if gaps:
            shown = ", ".join(str(g) for g in gaps[:5])
            more = f"…（共 {len(gaps)} 处）" if len(gaps) > 5 else ""
            raise ValueError(
                f"{path or 'vocab'}: 词表必须占满——ID 应严格连续 0..{n - 1}，"
                f"在 {shown} 处出现空位{more}")
        validate_head(flat, source=str(path or "vocab.vfc"))

        self._entries = [(k, t, bool(v)) for k, (_i, t, v) in enumerate(flat)]
        self.id_to_token: list[str] = [t for _i, t, _v in self._entries]
        # 不可显示 ID 集合：**头部以上**用尖括号包裹的那些。
        # [0, HEAD_SIZE) 整段都是架构契约头，恒为不可显示，不靠这个集合判断
        # （见 is_invisible 的注释）。
        self._invisible_above_head = frozenset(
            i for i, _t, v in self._entries if not v and i >= spec.HEAD_SIZE)
        self.invisible_ids = frozenset(i for i, _t, v in self._entries
                                       if not v)

        # 四个类别映射。system 区单独拎出来：`{` `}` `|` 在文件里是引号
        # 形态（visible=True），但绝不能进 regular_ids——否则文本里的 "{"
        # 会被编码成结构 token（ID 6），token-5 流当场破结构。它们按 ID
        # 区间判定，与包裹形态无关。
        self.regular_ids: dict[str, int] = {}
        self.fallback_ids: dict[str, int] = {}
        self.special_ids: dict[str, int] = {}
        self.system_ids: dict[str, int] = {}
        sys_begin = spec.SYSTEM_BEGIN
        sys_end = sys_begin + spec.SYSTEM_SIZE
        for i, t, v in self._entries:
            if sys_begin <= i < sys_end:
                self.system_ids[t] = i
            elif v:
                self.regular_ids[t] = i
                if len(t) == 1 and "a" <= t <= "z":
                    self.fallback_ids[t] = i
            else:
                self.special_ids[t] = i

        # I3：文本唯一（system 与可见符号同名是唯一豁免）。
        # token_to_id 供 add_token 查重与 /api/vocab 查询使用；system 字符
        # 不进去——反查「{」应当得到文本 token，而不是结构 token。
        seen: dict = {}
        dupes = []
        for i, t, _v in self._entries:
            if sys_begin <= i < sys_end:
                continue
            if t in seen:
                dupes.append((i, seen[t], t))
            else:
                seen[t] = i
        if dupes:
            i, j, t = dupes[0]
            raise ValueError(
                f"{path or 'vocab'}: token 文本重复 {t!r}"
                f"（ID {j} 与 {i}，共 {len(dupes)} 组重复）")
        self.token_to_id: dict[str, int] = seen

        self._max_match = max((len(t) for t in self.regular_ids), default=1)
        self.config["vocab_size"] = n
        self.regions = self._build_regions()

    # ------------------------------------------------------------------
    # 构造路径
    # ------------------------------------------------------------------
    @classmethod
    def from_entries(cls, entries, path=None, name=None, config=None) -> "Vocab":
        """[(id, text, visible)] → Vocab。ID 必须严格连续 0..N-1。"""
        return cls(entries, path=path, name=name, config=config)

    @classmethod
    def from_tokens(cls, tokens, path=None, name=None) -> "Vocab":
        """头部契约 + 可见 token 列表 → Vocab（测试/新词表用）。

        新契约下词表必须从 ID 0 连续占满，所以头部条目必须一起给——
        只给可见部分的话 ID 会从 HEAD_SIZE 起跳，连续性校验当场拒绝。
        """
        head = list(vfc.head_entries())
        base = len(head)                      # == spec.HEAD_SIZE
        return cls(head + [(base + i, t, True)
                           for i, t in enumerate(tokens)],
                   path=path, name=name)

    @classmethod
    def load(cls, path, name=None) -> "Vocab":
        """从 .vfc 文件加载（含头部契约校验与连续性校验）。

        同目录的 config.json 一并读入（分区表 / vocab_size 都在里面）。
        读不到或格式不对只告警、按「无分区元数据」处理——词表正文能加载
        才是硬要求，元数据坏了不该让整张词表不可用。
        """
        p = Path(path)
        return cls(vfc.read_vfc(str(p)), path=p, name=name,
                   config=_read_sibling_config(p))

    @classmethod
    def global_vocab(cls) -> "Vocab":
        """全局基础词表：service/tokens/{vocab.vfc, config.json}。

        目录名是 "tokens"，不是词表名；对外统一叫 base（= 系统预设名）。
        """
        if not GLOBAL_VOCAB_VFC.is_file():
            raise RuntimeError(
                f"{GLOBAL_VOCAB_VFC} 不存在，请先运行 "
                f"python tokens/build_vocab.py")
        return cls.load(GLOBAL_VOCAB_VFC, name="base")

    # ------------------------------------------------------------------
    # 分区（前端词表页「分区」表的数据源）
    # ------------------------------------------------------------------
    def _build_regions(self) -> list:
        """按 config["regions"] 的大小顺序展开成带 begin 的分区表。

        存的是**各区的大小**而不是绝对 begin——这样增删 token 时只要
        调整对应区的 size，后面所有区的 begin 自动跟随，不会出现
        「begin 忘改导致区间重叠或断裂」这类 bug。

        分区表覆盖**整个词表**（含头部三段 special/system/draw）：它们的
        大小也由 config 记录，第 0 区的 begin 就是 0。这样前端画占用条时
        不需要任何偏移假设。

        容错：config 的 size 之和与真实长度不一致（手工改过正文/旧配置）
        时，多出来的尾巴补一个 "extra" 区、超出部分截断——渲染不能因为
        一份过期配置就整块空白。
        """
        n = len(self.id_to_token)
        sizes = self.config.get("regions")
        out = []
        if not sizes:
            # 无分区元数据（旧文件）：整段可见区当一个区
            n_vis = n - spec.HEAD_SIZE
            out.append({"region": "visible", "category": CAT_REGULAR,
                        "begin": spec.HEAD_SIZE, "size": max(n_vis, 0),
                        "used": max(n_vis, 0)})
            return out
        begin = 0
        for r in sizes:
            if begin >= n:
                break
            size = max(int(r.get("size", 0)), 0)
            end = min(begin + size, n)
            used = end - begin
            out.append({"region": r.get("region", "?"),
                        "category": r.get("category", CAT_REGULAR),
                        "begin": begin, "size": end - begin, "used": used})
            begin = end
        if begin < n:
            out.append({"region": "extra", "category": CAT_REGULAR,
                        "begin": begin, "size": n - begin,
                        "used": n - begin})
        return out

    def region_of(self, idx: int) -> str | None:
        for r in self.regions:
            if r["begin"] <= idx < r["begin"] + r["size"]:
                return r["region"]
        return None

    def category_totals(self) -> dict:
        out = {c: 0 for c in CATEGORIES}
        for r in self.regions:
            out[r["category"]] = out.get(r["category"], 0) + r["used"]
        return out

    # ------------------------------------------------------------------
    # 基础查询
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        """词表长度（每个词表可以不同，由它的 config.json 声明）。"""
        return len(self.id_to_token)

    def token(self, idx: int) -> str:
        return self.id_to_token[idx]

    def is_invisible(self, idx: int) -> bool:
        """该 ID 是否为不可显示 token。

        判定 = 「落在架构契约头内」或「.vfc 里写成尖括号形态」。
        头部 [0, HEAD_SIZE) 是 special/system/draw 三段，恒为不可显示，
        所以这里直接按 ID 范围判，不依赖文件的包裹形态。

        **为什么不用 `idx < HEAD_SIZE` 一把梭**：HEAD_SIZE 现在等于
        HEAD_SIZE，头部内已无空位，两种写法对头部等价；但用户可以把新的不可
        显示 token 追加到头部之上的任意 ID（.vfc 用尖括号标注），那种情况
        只有后者能覆盖。两者取并集最稳。

        已知例外：`{` `}` `|` 语义上是 system（不可显示），但 `{` 不是合法
        标识符、无法写成 `<{>`，所以文件里是引号形态。它们靠 ID 落在
        SYSTEM 区被正确归类，不受影响。
        """
        return idx < spec.HEAD_SIZE or idx in self._invisible_above_head

    def category_of(self, idx: int) -> str:
        """ID → 大类（regular / fallback / special / system）。"""
        if not (0 <= idx < len(self.id_to_token)):
            raise ValueError(f"token ID 越界: {idx}")
        if self.is_invisible(idx):
            if spec.SYSTEM_BEGIN <= idx < spec.SYSTEM_BEGIN + spec.SYSTEM_SIZE:
                return CAT_SYSTEM
            return CAT_SPECIAL
        t = self.id_to_token[idx]
        if len(t) == 1 and "a" <= t <= "z" and self.regular_ids.get(t) == idx:
            return CAT_FALLBACK
        return CAT_REGULAR

    def lookup(self, text: str) -> int | None:
        """enctok 查表：先常规可显示，后 fallback 可显示；不可显示不参与。"""
        tid = self.regular_ids.get(text)
        if tid is not None:
            return tid
        return self.fallback_ids.get(text)

    def name_of(self, idx: int) -> str | None:
        """不可显示 token 的 ID → 名字（如 <stop> → "stop"）；可显示返回 None。"""
        if not self.is_invisible(idx):
            return None
        t = self.id_to_token[idx]
        return t[1:-1] if t.startswith("<") and t.endswith(">") else t

    def id_of(self, name: str) -> int | None:
        """名字（如 "stop" 或 "<stop>"）→ 不可显示 token 的 ID；查不到 None。"""
        cand = name if name.startswith("<") else f"<{name}>"
        tid = self.special_ids.get(cand)
        if tid is None:
            tid = self.system_ids.get(cand)
        if tid is not None:
            return tid
        # system 区的 { } | 文本不带尖括号
        return self.system_ids.get(name)

    # ------------------------------------------------------------------
    # 增删（紧凑化：删除后其后 ID 整体前移，绝不留下空位）
    # ----------------------------------------------------------------------
    def add_token(self, text: str, category: str = CAT_REGULAR) -> int:
        """追加一个 token，返回新 ID。

        - category 只能是 regular（可显示）或 special（不可显示）；
          fallback / system 由词表自动判定：单字母 a–z 兼作 fallback，
          system 属架构契约头、不接受用户添加。
        - 文本非空、不得使用 `<reserved:` 前缀、全表查重。
        - 新 token 插入 config["append_region"] 指定的可见子区末尾，
          其后所有 ID 自动前移/后移，保证连续。
        """
        if category not in (CAT_REGULAR, CAT_SPECIAL):
            raise ValueError(
                f"category 必须是 {CAT_REGULAR}/{CAT_SPECIAL}: {category!r}")
        if not isinstance(text, str) or not text:
            raise ValueError("token 文本必须是非空字符串")
        if text.startswith("<reserved:"):
            # 旧契约的软删占位符形态（<reserved:N>）。新契约没有占位符
            # 概念（占位即空位，空位违反对齐不变量 I1），且这种名字会让
            # build_vocab 的 user_added 重放把它当已删条目静默跳过——
            # 从入口就拒绝，避免造出「看得到但永远不会被重建」的 token。
            raise ValueError(
                f"<reserved:> 前缀是旧契约占位符形态，新契约不接受: "
                f"{text!r}")
        if text in self.token_to_id:
            raise ValueError(
                f"token 冲突：{text!r} 已存在于词表"
                f"（ID {self.token_to_id[text]}，"
                f"类别 {self.category_of(self.token_to_id[text])}）")

        region = (self.config.get("append_region")
                  or DEFAULT_APPEND_REGION)
        entries = list(self._entries)
        pos = self._insert_position(entries, region)
        entries.insert(pos, (pos, text, category == CAT_REGULAR))
        # 插入点之后的所有 ID 必须整体后移 1：entries 是按位置存的，
        # 直接 insert 会让「新条目」和「原 pos 处条目」撞同一个 ID，出现
        # 一个空位 + 一个重复（连续性校验当场拒绝）。重新按位置编号。
        entries = [(i, t, v) for i, (_old, t, v) in enumerate(entries)]
        # 分区表同步：新 token 落在 region 末尾 → 该区 size +1，后面各区的
        # begin 由累加自动跟随。不更新的话 config 里各 size 之和会比正文
        # 少 1，下一次 _sync_config 会以为配置过期、把整张分区表塌成一个
        # "visible" 区（分区信息就此丢失）。
        cfg = dict(self.config)
        sizes = [dict(r) for r in (cfg.get("regions") or [])]
        found = False
        for r in sizes:
            if r.get("region") == region:
                r["size"] = int(r.get("size", 0)) + 1
                found = True
                break
        if not found and sizes:
            sizes[-1]["size"] = int(sizes[-1].get("size", 0)) + 1
        cfg["regions"] = sizes
        fresh = Vocab(entries, path=self._path, name=self.name, config=cfg)
        self.__dict__.update(fresh.__dict__)
        self._persist()
        return self.token_to_id[text]

    def _insert_position(self, entries, region: str) -> int:
        """返回新 token 应插入的下标：指定 region 的末尾。

        没有该 region 时：插到最后一个**可见**条目之后（保持不可显示头
        与其后的可见区各自成段），全空则插到末尾。"""
        sizes = self.config.get("regions") or []
        begin = 0
        for r in sizes:
            size = int(r.get("size", 0))
            if r.get("region") == region:
                return begin + size
            begin += size
        # 兜底：最后一个可显示条目之后
        last_vis = max((i for i, _t, v in entries if v), default=-1)
        return last_vis + 1

    def remove_token(self, text: str) -> int:
        """删除一个 token 并紧凑化，返回被删 token 的原 ID。

        不可显示头（special/system/draw）是架构契约，**禁止删除**。
        """
        tid = self.token_to_id.get(text)
        if tid is None:
            raise ValueError(f"token 未收录于词表: {text!r}")
        if tid < spec.HEAD_SIZE:
            raise ValueError(
                f"不可删除架构契约头内的 token: {text!r}（ID {tid} < "
                f"HEAD_SIZE {spec.HEAD_SIZE}）")
        entries = [e for e in self._entries if int(e[0]) != tid]
        # 同步收缩该 token 所属子区的 size（其余区的 begin 自动跟随）
        cfg = dict(self.config)
        sizes = [dict(r) for r in (cfg.get("regions") or [])]
        begin = 0
        for r in sizes:
            size = int(r.get("size", 0))
            if begin <= tid < begin + size:
                r["size"] = size - 1
                break
            begin += size
        cfg["regions"] = sizes
        fresh = Vocab([(i, t, v) for i, (_o, t, v) in enumerate(entries)],
                      path=self._path, name=self.name, config=cfg)
        self.__dict__.update(fresh.__dict__)
        self._persist()
        return tid

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def to_entries(self) -> list:
        return [(i, t, v) for i, t, v in self._entries]

    def _persist(self) -> None:
        if self._path is None:
            raise RuntimeError("该词表无来源文件路径，无法持久化")
        vfc.write_vfc(str(self._path), self._entries)

    def reload(self) -> None:
        """丢弃内存映射，从磁盘重新加载（外部改动后同步）。"""
        fresh = (Vocab.load(self._path) if self._path
                 else Vocab.global_vocab())
        self.__dict__.update(fresh.__dict__)


# ---------------------------------------------------------------------------
# 进程级单例与模型词表覆盖
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
_VOCAB: Vocab | None = None
_MODEL_VOCAB: Vocab | None = None
_MODEL_VOCAB_DIR: str | None = None


def set_model_vocab(model_dir: str) -> None:
    """把进程的词表切换为模型的 vocab/vocab.vfc。

    没有词表文件就清退到全局。重复设置同一目录是廉价的（直接复用）。"""
    global _MODEL_VOCAB, _MODEL_VOCAB_DIR
    if model_dir == _MODEL_VOCAB_DIR and _MODEL_VOCAB is not None:
        return
    new = _model_vocab_file(model_dir)
    if os.path.isfile(new):
        _MODEL_VOCAB = Vocab.load(new)
        _MODEL_VOCAB_DIR = model_dir
        return
    clear_model_vocab()


def clear_model_vocab() -> None:
    """清退到全局词表（无活动模型时）。"""
    global _MODEL_VOCAB, _MODEL_VOCAB_DIR
    _MODEL_VOCAB = None
    _MODEL_VOCAB_DIR = None


def set_vocab_source(path: str) -> None:
    """CLI 便捷入口（enctok/dectok --vocab）：
    path 是目录 → 用它的 vocab/vocab.vfc（或旧 vocab.txt）；
    是文件 → 直接当 .vfc 加载。"""
    p = Path(path)
    if p.is_dir():
        set_model_vocab(str(p))
        return
    if not p.is_file():
        raise FileNotFoundError(f"词表来源不存在: {path}")
    global _MODEL_VOCAB, _MODEL_VOCAB_DIR
    _MODEL_VOCAB = Vocab.load(p)
    _MODEL_VOCAB_DIR = None


def get_vocab() -> Vocab:
    """进程级懒加载单例；有活动模型词表时优先返回它。"""
    global _VOCAB
    if _MODEL_VOCAB is not None:
        return _MODEL_VOCAB
    if _VOCAB is None:
        _VOCAB = Vocab.global_vocab()
    return _VOCAB


def reset_vocab_cache() -> None:
    """清空单例（测试 / 外部改写词表文件后调用）。"""
    global _VOCAB
    _VOCAB = None
    clear_model_vocab()


# ---------------------------------------------------------------------------
# 模型目录内的词表定位：models/NAME/vocab/vocab.vfc（+ 同目录 config.json）
# ---------------------------------------------------------------------------
def model_vocab_dir(model_dir: str) -> Path:
    """模型目录内的词表子目录路径（不一定存在）。

    返回 Path，与 server.vocab_presets.vocab_dir 保持一致——
    seed_model_vocab 等调用方会对其调用 .exists() / 用 / 拼接。
    """
    return Path(model_dir) / VOCAB_DIR_NAME


def _model_vocab_file(model_dir: str) -> str:
    return os.path.join(model_vocab_dir(model_dir), VOCAB_FILE_NAME)


def _warn(msg: str) -> None:
    import sys
    print(f"[vocab] {msg}", file=sys.stderr)
