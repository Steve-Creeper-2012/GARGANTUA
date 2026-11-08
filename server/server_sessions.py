# ===========================================================================
# GARGANTUA v1 — server_sessions.py：会话持久化（一会话一 JSON 文件）
# ---------------------------------------------------------------------------
# 2026-11-05 新增（会话 + 模型加载/选中拆分）：
#   前端推理页的「会话」落盘到 <项目根>/logs/conversations/<id>.json。
#   server 只做轻量 JSON 记账，token_ids 是唯一载荷（文本由前端插件解码）。
#   进程内 RLock 串行化读写 + 同目录临时文件 os.replace 原子写，杜绝半截文件。
#   项目根一律经 server/paths.py 推导，绝不依赖 cwd / HOME。
# ===========================================================================
import glob
import json
import os
import re
import tempfile
import threading
import time
import uuid

from paths import project_dir

# 一会话一文件，id = "s_" + 12 位小写 hex；正则同时挡住路径穿越
_ID_RE = re.compile(r"^s_[0-9a-f]{12}$")
_MAX_MESSAGES = 500          # 单会话消息条数上限
_MAX_TOKEN_IDS = 20000       # 单条 token_ids 长度上限

# schema 固定字段的默认值（不含 id / created / updated / messages）
_DEFAULTS = {
    "title": "",
    "plugin": "tokens",
    "model": "",
    "kv_enabled": True,
    "max_new": 32,
    "temperature": 0.8,
    "top_p": 0.95,
}
# create / save 允许写入的业务字段（id / created / updated 由 store 自己管）
_FIELDS = set(_DEFAULTS) | {"messages"}


class SessionStore:
    """会话文件仓库：一个 id 一个 .json，进程内用 RLock 串行化读写。"""

    def __init__(self, root_dir=None):
        self._dir = root_dir or project_dir("logs", "conversations")
        os.makedirs(self._dir, exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _check_id(sid):
        if not isinstance(sid, str) or not _ID_RE.fullmatch(sid):
            raise ValueError(f"非法会话 id: {sid!r}")
        return sid

    def _path(self, sid):
        return os.path.join(self._dir, sid + ".json")

    def _read(self, sid):
        """读一个会话；文件缺失 / 损坏一律返回 None（交给调用方判 404）。"""
        try:
            with open(self._path(sid), "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError, TypeError):
            return None
        return data if isinstance(data, dict) else None

    def _write(self, sid, sess):
        """原子写：同目录临时文件 → os.replace，绝不留下半截 JSON。"""
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(prefix="._sess-", suffix=".tmp",
                                       dir=self._dir)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(sess, f, ensure_ascii=False)
            os.replace(tmp, self._path(sid))
        except Exception:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            raise

    @staticmethod
    def _check_messages(messages):
        """校验整份消息列表：结构 + 两条限额。超限抛 ValueError（→ 400）。"""
        if not isinstance(messages, list):
            raise ValueError("messages 必须是数组")
        if len(messages) > _MAX_MESSAGES:
            raise ValueError(
                f"单会话消息数超限: {len(messages)} > {_MAX_MESSAGES}")
        for m in messages:
            if not isinstance(m, dict):
                raise ValueError("每条 message 必须是对象")
            if m.get("role") not in ("me", "bot"):
                raise ValueError("message.role 必须是 me 或 bot")
            ids = m.get("token_ids")
            if not isinstance(ids, list):
                raise ValueError("message.token_ids 必须是数组")
            if len(ids) > _MAX_TOKEN_IDS:
                raise ValueError(
                    f"单条 token_ids 超限: {len(ids)} > {_MAX_TOKEN_IDS}")
            if any(not isinstance(i, int) or isinstance(i, bool) for i in ids):
                raise ValueError("token_ids 只能包含整数")
            ts = m.get("ts")
            if ts is not None and (isinstance(ts, bool)
                                   or not isinstance(ts, (int, float))):
                raise ValueError("message.ts 必须是数值")
        return messages

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------
    def list(self):
        """摘要列表（不含 messages），按 updated 倒序。"""
        out = []
        with self._lock:
            for path in glob.glob(os.path.join(self._dir, "*.json")):
                sid = os.path.basename(path)[:-5]
                if not _ID_RE.fullmatch(sid):
                    continue
                sess = self._read(sid)
                if sess is None:
                    continue
                out.append({k: v for k, v in sess.items()
                            if k != "messages"})
        out.sort(key=lambda s: s.get("updated") or 0, reverse=True)
        return out

    def create(self, partial=None):
        """新建会话；partial 只采纳白名单业务字段，缺省走 schema 默认值。"""
        if partial is None:
            partial = {}
        if not isinstance(partial, dict):
            raise ValueError("会话参数必须是对象")
        sid = "s_" + uuid.uuid4().hex[:12]
        now = time.time()
        sess = dict(_DEFAULTS)
        sess.update({"id": sid, "created": now, "updated": now,
                     "messages": []})
        for k in _FIELDS:
            if k == "messages":
                continue                     # create 不接受初始消息
            if k in partial:
                sess[k] = partial[k]
        with self._lock:
            self._write(sid, sess)
        return sess

    def get(self, sid):
        """返回完整会话 dict；不存在 / 损坏返回 None。"""
        self._check_id(sid)
        with self._lock:
            return self._read(sid)

    def save(self, sid, patch):
        """合并 patch 到会话：messages 允许整体覆盖，刷新 updated。"""
        self._check_id(sid)
        if not isinstance(patch, dict):
            raise ValueError("patch 必须是对象")
        with self._lock:
            sess = self._read(sid)
            if sess is None:
                raise FileNotFoundError(f"会话不存在: {sid}")
            for k, v in patch.items():
                if k in ("id", "created"):
                    continue                 # 主键 / 创建时间不可改
                if k == "messages":
                    self._check_messages(v)
                    sess["messages"] = v
                elif k in _FIELDS:
                    sess[k] = v
            sess["updated"] = time.time()
            self._write(sid, sess)
            return sess

    def append(self, sid, role, token_ids):
        """追加一条消息；先做结构 + 限额校验，再整体落盘。"""
        self._check_id(sid)
        if role not in ("me", "bot"):
            raise ValueError("role 必须是 me 或 bot")
        if not isinstance(token_ids, list):
            raise ValueError("token_ids 必须是数组")
        if len(token_ids) > _MAX_TOKEN_IDS:
            raise ValueError(
                f"单条 token_ids 超限: {len(token_ids)} > {_MAX_TOKEN_IDS}")
        if any(not isinstance(i, int) or isinstance(i, bool)
               for i in token_ids):
            raise ValueError("token_ids 只能包含整数")
        with self._lock:
            sess = self._read(sid)
            if sess is None:
                raise FileNotFoundError(f"会话不存在: {sid}")
            msgs = sess.get("messages")
            if not isinstance(msgs, list):
                msgs = []
            if len(msgs) >= _MAX_MESSAGES:
                raise ValueError(
                    f"单会话消息数超限: {len(msgs)} >= {_MAX_MESSAGES}")
            msgs.append({"role": role, "token_ids": token_ids,
                         "ts": time.time()})
            sess["messages"] = msgs
            sess["updated"] = time.time()
            self._write(sid, sess)
            return sess

    def delete(self, sid):
        """删除会话文件；存在并删除返回 True，不存在返回 False。"""
        self._check_id(sid)
        with self._lock:
            try:
                os.unlink(self._path(sid))
                return True
            except FileNotFoundError:
                return False
