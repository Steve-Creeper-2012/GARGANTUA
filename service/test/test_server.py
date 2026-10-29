# ===========================================================================
# GARGANTUA v1 — tests/test_server.py：server.py 全 API 集成测试
# ---------------------------------------------------------------------------
# setUpClass：临时目录里用 Gargantua() 新建 safetensors 小模型替身
# （约 25s），子进程启动 server.py（随机高端口 + 临时 model/data 目录），
# 标准库 urllib 打全部 API。
# 运行：python -m unittest tests.test_server -v
# ===========================================================================
# ---- GARGANTUA 路径引导（2026-10-24 目录树重构：server/ service/ start/）----
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
import io
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
import urllib.error
import wave

ROOT = boot.project_root()   # 2026-10-24：项目根（旧布局曾是 service/）


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def http(method, port, path, body=None):
    url = f"http://127.0.0.1:{port}{path}"
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def make_png_b64(w, h, rgba):
    """内存造一张纯色 PNG（base64）。"""
    from PIL import Image
    img = Image.new("RGBA", (w, h), rgba)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_wav_b64(n_samples=1600, sr=16000):
    """内存造一段 16bit 单声道 wav（base64）。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = b"".join(struct.pack("<h", (i * 37) % 2000 - 1000)
                          for i in range(n_samples))
        wf.writeframes(frames)
    return base64.b64encode(buf.getvalue()).decode()


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="garg_test_")
        cls.model_dir = os.path.join(cls.tmp, "models")
        cls.data_dir = os.path.join(cls.tmp, "data")
        cls.log_file = os.path.join(cls.tmp, "logs", "auto_train.jsonl")
        os.makedirs(cls.model_dir)

        # 小模型替身：与 init.py 相同的 safetensors + config.json 方式。
        # 2026-10-20：改用具名 block 数（1+1）的小模型——默认构造是
        # 527M 参数（2.1GB safetensors），磁盘压力大且与本测试意图
        # （API 行为，与规模无关）不符；同时顺带覆盖 config.json 的
        # block 数经 server 加载路径。
        import torch
        from safetensors.torch import save_file
        from model import Gargantua
        torch.manual_seed(0)
        m = Gargantua(n_enc_blocks=1, n_dec_blocks=1)
        n = sum(p.numel() for p in m.parameters())
        d = os.path.join(cls.model_dir, "tiny")
        os.makedirs(d)
        save_file(m.state_dict(), os.path.join(d, "model.safetensors"))
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"name": "tiny", "params": n,
                       "n_enc_blocks": 1, "n_dec_blocks": 1}, f)
        cls.params = n
        del m

        # tok-32 词表备份：test_15_vocab_add 会经 API 改写仓库里的
        # tokens/vocab.json（server 子进程 cwd=ROOT），测后必须还原
        cls.vocab_path = os.path.join(ROOT, "service", "tokens", "vocab.json")
        cls.vocab_backup = os.path.join(cls.tmp, "vocab.json.bak")
        shutil.copy(cls.vocab_path, cls.vocab_backup)

        cls.port = free_port()
        cls.proc = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "server", "server.py"),
             "--port", str(cls.port),
             "--model-dir", cls.model_dir,
             "--data-dir", cls.data_dir,
             "--log-file", cls.log_file],
            cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # 等服务器可连接（模型后台加载，先能响应即可）
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                code, _ = http("GET", cls.port, "/api/status")
                if code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("服务器启动超时")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
        # 还原词表（add 测试的写入不落仓库）
        shutil.copy(cls.vocab_backup, cls.vocab_path)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    def test_00_index(self):
        code, _ = http("GET", self.port, "/api/status")  # 确认活着
        self.assertEqual(code, 200)
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/")
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read().decode("utf-8")
        self.assertIn("GARGANTUA", html)

    def test_01_status_and_model_autoload(self):
        # 唯一模型自动加载（后台线程）：轮询直到加载完成
        deadline = time.time() + 300
        st = {}
        while time.time() < deadline:
            code, st = http("GET", self.port, "/api/status")
            self.assertEqual(code, 200)
            if st.get("model"):
                break
            time.sleep(2)
        self.assertEqual(st.get("model"), "tiny", f"自动加载超时: {st}")
        self.assertEqual(st["params"], self.params)
        self.assertIn("training", st)
        self.assertIn("queue", st)
        self.assertIn("kv_note", st)

    def test_02_models_list(self):
        code, j = http("GET", self.port, "/api/models")
        self.assertEqual(code, 200)
        names = [m["name"] for m in j["models"]]
        self.assertIn("tiny", names)
        tiny = [m for m in j["models"] if m["name"] == "tiny"][0]
        self.assertEqual(tiny["params"], self.params)
        self.assertEqual(j["active"], "tiny")

    def test_03_vocab_query(self):
        code, j = http("GET", self.port, "/api/vocab?q=user")
        self.assertEqual(code, 200)
        self.assertTrue(any(h["token"] == "<user>" for h in j["hits"]))
        self.assertLessEqual(len(j["hits"]), 50)
        self.assertIn("categories", j)
        self.assertTrue(all("category" in h for h in j["hits"]))
        # 系统 token 的显示字符也能搜到（归类为 regular 可见符号）
        code, j = http("GET", self.port, "/api/vocab?q=" + "%7B")
        self.assertEqual(code, 200)
        self.assertTrue(any(h["token"] == "{" for h in j["hits"]))

    def test_04_codec_image(self):
        b64 = make_png_b64(1, 1, (255, 0, 0, 255))
        code, j = http("POST", self.port, "/api/codec/image",
                       {"image_b64": b64})
        self.assertEqual(code, 200)
        st = j["steps"][0]
        self.assertEqual(st["modality"], "visual")
        self.assertTrue(st["meta"]["full"])
        self.assertEqual(st["meta"]["grid"], [1, 1])   # 1×1 pad 到 32×32
        self.assertEqual(st["meta"]["n_patches"], 1)
        self.assertIn("visual", j["archive"])
        self.assertEqual(len(j["archive"]["visual"]), 1)

    def test_05_codec_frames_delta(self):
        f1 = make_png_b64(32, 32, (10, 20, 30, 255))
        f2 = make_png_b64(32, 32, (10, 20, 30, 255))   # 与 f1 相同
        f3 = make_png_b64(32, 32, (99, 99, 99, 255))   # 全变（末帧强制全量）
        code, j = http("POST", self.port, "/api/codec/frames",
                       {"frames_b64": [f1, f2, f3]})
        self.assertEqual(code, 200)
        frames = j["frames"]
        self.assertEqual(len(frames), 3)
        self.assertTrue(frames[0]["full"])
        self.assertEqual(frames[0]["n_patches"], 1)
        # 第 2 帧无变化 → delta 无 patch，但时间步必须非空（.tokenvector
        # 不允许空步），故送 1 个占位 patch 并标记 static
        self.assertFalse(frames[1]["full"])
        self.assertTrue(frames[1]["static"])
        self.assertEqual(frames[1]["n_patches"], 1)
        # 末帧强制全量
        self.assertTrue(frames[2]["full"])
        self.assertIn("delta_rule", j)

    def test_06_codec_audio(self):
        b64 = make_wav_b64(n_samples=1600)
        code, j = http("POST", self.port, "/api/codec/audio",
                       {"audio_b64": b64, "filename": "t.wav"})
        self.assertEqual(code, 200)
        # 1600 采样 / 533 = 3.001 → 4 音元（末尾补零）
        self.assertEqual(j["n_tokens"], 4)
        self.assertAlmostEqual(j["duration_s"], 4 * 533 / 16000, places=3)
        # 当前契约：step_to_json 把音元参数收在 meta 下（与前端
        # stepsToGts / image 页的 st.meta.* 访问一致）
        self.assertEqual(j["steps"][-1]["meta"]["n"], 1600 - 3 * 533)

    def _draw_and_wait(self, body, timeout=180):
        """2026-10-23：/api/codec/draw 改为异步任务（原为同步渲染，
        无图元数上限时可占死 HTTP handler 线程）→ 202 + task_id。"""
        code, j = http("POST", self.port, "/api/codec/draw", body)
        self.assertEqual(code, 202)
        self.assertTrue(j.get("task_id"))
        deadline = time.time() + timeout
        while time.time() < deadline:
            code, t = http("GET", self.port, "/api/task/" + j["task_id"])
            self.assertEqual(code, 200)
            if t["status"] == "done":
                return t["result"]
            if t["status"] == "error":
                self.fail(f"draw 任务报错: {t['error']}")
            time.sleep(0.5)
        self.fail("draw 任务超时")

    def test_08_codec_draw(self):
        prims = [
            {"x": 960, "y": 540, "shape": "background", "width": 0,
             "length": 0, "rot": 0, "r": 64, "g": 64, "b": 96},
            {"x": 480, "y": 400, "shape": "rectangle", "width": 200,
             "length": 150, "rot": 0, "r": 900, "g": 300, "b": 200},
        ]
        j = self._draw_and_wait({"primitives": prims})
        png = base64.b64decode(j["png_b64"])
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn("<svg", j["svg"])
        self.assertEqual(len(j["tokens"]), 2 * 9)     # 9 token/图元
        # 反向：tokens → 渲染
        j2 = self._draw_and_wait({"tokens": j["tokens"]})
        self.assertEqual(j2["n_primitives"], 2)
        self.assertTrue(base64.b64decode(j2["png_b64"]).startswith(b"\x89PNG"))

    def test_09_train_flow(self):
        # enqueue → queue 文件出现
        # 2026-10-25 契约：只接受 tokens_b64（.tokens 二进制，必须带
        # { } | 时间步结构）；一次上传 = 一条 token 流；JSON 上传被拒。
        from tokens import tok5
        from tokens.text_codec import encode as text_encode

        def make_tokens(text):
            steps = text_encode(text)
            items = [tok5.step(tok5.text(i)) for st in steps
                     for i in st["token_ids"]]
            return base64.b64encode(tok5.dumps(items)).decode()

        def enqueue(text):
            return http("POST", self.port, "/api/train/enqueue",
                        {"tokens_b64": make_tokens(text)})
        code, j = enqueue("你好世界 " * 40)
        self.assertEqual(code, 200)
        queued = j["queued"]
        self.assertEqual(len(queued), 1)
        self.assertTrue(queued[0].endswith(".tokens"))
        qpath = os.path.join(self.data_dir, "queue", queued[0])
        self.assertTrue(os.path.exists(qpath))
        self.assertIn("queue", qpath)
        # 第二条流
        code, j = enqueue("甲乙丙丁" * 20)
        self.assertEqual(code, 200)
        self.assertEqual(len(j["queued"]), 1)
        # JSON 上传已废弃 → 400
        code, j = http("POST", self.port, "/api/train/enqueue",
                       {"json_b64": base64.b64encode(
                           b'{"text": "x"}').decode()})
        self.assertEqual(code, 400)
        # 不是 token-5 位流（随便几个字节）→ 400
        code, j = http("POST", self.port, "/api/train/enqueue",
                       {"tokens_b64": base64.b64encode(b"\x00\x01\x02\x03").decode()})
        self.assertEqual(code, 400)
        # start → running
        code, j = http("POST", self.port, "/api/train/start",
                       {"steps": 1, "block_size": 16})
        self.assertEqual(code, 200)
        deadline = time.time() + 20
        st = {}
        while time.time() < deadline:
            code, st = http("GET", self.port, "/api/train/status")
            self.assertEqual(code, 200)
            if st["running"]:
                break
            time.sleep(0.3)
        self.assertTrue(st["running"], "训练线程未进入 running")
        self.assertIn("loss_tail", st)
        self.assertIn("log_tail", st)
        # start 幂等：已在跑返回现状
        code, j = http("POST", self.port, "/api/train/start", {})
        self.assertEqual(code, 200)
        self.assertFalse(j["started"])
        # stop → 优雅停止请求
        code, j = http("POST", self.port, "/api/train/stop", {})
        self.assertEqual(code, 200)
        self.assertTrue(j["stopping"])

    def test_10_infer_task_lifecycle(self):
        code, j = http("POST", self.port, "/api/infer",
                       {"prompt": "你好", "max_new_tokens": 1,
                        "temperature": 0.8, "top_p": 0.95})
        self.assertEqual(code, 202)
        tid = j["task_id"]
        self.assertTrue(tid)
        # 轮询任务机：允许超时不等完成，只验证 202→running/done 生命周期
        seen = set()
        result = None
        deadline = time.time() + 120
        while time.time() < deadline:
            code, t = http("GET", self.port, f"/api/task/{tid}")
            self.assertEqual(code, 200)
            self.assertIn(t["status"], ("running", "done", "error"))
            seen.add(t["status"])
            if t["status"] == "done":
                result = t["result"]
                break
            if t["status"] == "error":
                self.fail(f"推理任务报错: {t['error']}")
            time.sleep(2)
        self.assertTrue(seen, "任务从未被查询到")
        if result is not None:      # 若已完成则校验结果结构
            self.assertIn("text", result)
            self.assertEqual(len(result["token_ids"]), 1)
            # 2026-10-25：推理侧 token id 存档（.tokens 落盘）
            self.assertTrue(result.get("archive"), "缺少 archive 路径")
            apath = os.path.join(ROOT, result["archive"])
            self.assertTrue(os.path.exists(apath), apath)
            self.assertIn("infer_archive", apath)
            # 存档能按 token-5 结构读回（prompt + 1 个生成 token）
            from tokens import tok5
            items = tok5.read_file(apath)
            steps = [it for it in items if it["kind"] == "step"]
            meta_path = apath[:-7] + ".json"
            self.assertTrue(os.path.exists(meta_path))
            import json as _json
            with open(meta_path, encoding="utf-8") as f:
                meta = _json.load(f)
            self.assertEqual(meta["n_tokens"], len(steps))
            self.assertEqual(meta["n_generated_tokens"], 1)
            self.assertGreaterEqual(meta["n_prompt_tokens"], 1)
            self.assertEqual(meta["n_evicted"], 0)
        # 不存在的任务 → 404
        code, j = http("GET", self.port, "/api/task/" + "0" * 32)
        self.assertEqual(code, 404)

    def test_11_error_handling(self):
        # 缺字段 → 400 + {error}
        code, j = http("POST", self.port, "/api/infer", {})
        self.assertEqual(code, 400)
        self.assertIn("error", j)
        code, j = http("POST", self.port, "/api/codec/image", {})
        self.assertEqual(code, 400)
        self.assertIn("error", j)
        # 未知路径 → 404，服务器不崩
        code, j = http("GET", self.port, "/api/nonexistent")
        self.assertEqual(code, 404)
        code, _ = http("GET", self.port, "/api/status")
        self.assertEqual(code, 200)

    # ------------------------------------------------------------------
    # 2026-10-20 审查修复的参数校验回归
    # ------------------------------------------------------------------
    def test_12_param_validation(self):
        # 奇数 block_size → 400（此前令前向必崩）
        code, j = http("POST", self.port, "/api/train/start",
                       {"block_size": 15})
        self.assertEqual(code, 400)
        self.assertIn("偶数", j.get("error", ""))
        # 0 步 → 400（此前触发 Trainer _loss_tail[-1] IndexError）
        code, j = http("POST", self.port, "/api/train/start", {"steps": 0})
        self.assertEqual(code, 400)
        # max_new 越界 → 400
        code, j = http("POST", self.port, "/api/infer",
                       {"prompt": "hi", "max_new_tokens": 999999})
        self.assertEqual(code, 400)
        # temperature 非法 → 400
        code, j = http("POST", self.port, "/api/infer",
                       {"prompt": "hi", "temperature": 0})
        self.assertEqual(code, 400)
        # 显式 0 不被 or 默认值吞掉：max_new_tokens=0 此前静默变 32
        code, j = http("POST", self.port, "/api/infer",
                       {"prompt": "hi", "max_new_tokens": 0})
        self.assertEqual(code, 400)
        # 非数值 → 400（此前 float("x") 抛 ValueError 也会 400，但消息不可读）
        code, j = http("POST", self.port, "/api/infer",
                       {"prompt": "hi", "temperature": "abc"})
        self.assertEqual(code, 400)
        self.assertIn("temperature", j.get("error", ""))
        # 模型名路径穿越 → 400（此前可加载 model_dir 外目录）
        code, j = http("POST", self.port, "/api/model/load",
                       {"name": "../escaped"})
        self.assertEqual(code, 400)
        # image2prims：非法 quality 此前被静默改成 fast
        code, j = http("POST", self.port, "/api/codec/image2prims",
                       {"image_b64": "AAAA", "quality": "medium"})
        self.assertEqual(code, 400)
        self.assertIn("quality", j.get("error", ""))
        # max_steps=0 此前被静默改成 100
        code, j = http("POST", self.port, "/api/codec/image2prims",
                       {"image_b64": "AAAA", "max_steps": 0})
        self.assertEqual(code, 400)
        # 非法 JSON → 400
        import urllib.request
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/infer", data=b"not-json",
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=10)
            self.fail("应拒绝非法 JSON")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 400)

    def test_13_model_init_conflict(self):
        # 2026-10-23：先等训练真正停下（stop 是优雅停止，守护线程要跑完
        # 当前 step 才退出）——否则「训练中覆盖活动模型 → 409」的新守卫
        # 会抢在「已存在」错误前面，让本测试不稳定
        deadline = time.time() + 60
        while time.time() < deadline:
            code, st = http("GET", self.port, "/api/train/status")
            self.assertEqual(code, 200)
            if not st["running"]:
                break
            time.sleep(0.5)
        # 重名创建默认拒绝（任务报错），overwrite=True 才成功。
        # 用小 block 数：默认构造写 2.1GB safetensors，磁盘吃不消。
        def init_and_wait(name, overwrite=None, timeout=300):
            body = {"name": name, "n_enc_blocks": 1, "n_dec_blocks": 1}
            if overwrite is not None:
                body["overwrite"] = overwrite
            code, j = http("POST", self.port, "/api/model/init", body)
            self.assertEqual(code, 202)
            tid = j["task_id"]
            deadline = time.time() + timeout
            while time.time() < deadline:
                code, t = http("GET", self.port, f"/api/task/{tid}")
                if t["status"] in ("done", "error"):
                    return t
                time.sleep(1)
            self.fail("init 任务超时")
        t = init_and_wait("tiny")           # setUpClass 已建 tiny
        self.assertEqual(t["status"], "error")
        self.assertIn("已存在", t["error"])
        # 先清盘再覆盖：os.replace 前 .tmp 与旧文件并存，双倍体积
        shutil.rmtree(os.path.join(self.model_dir, "tiny"))
        t = init_and_wait("tiny", overwrite=True)
        self.assertEqual(t["status"], "done", t.get("error"))
        # 覆盖后磁盘 config.json 仍完整
        cfg_path = os.path.join(self.model_dir, "tiny", "config.json")
        self.assertTrue(os.path.exists(cfg_path))
        with open(cfg_path, encoding="utf-8") as f:
            self.assertIn("params", json.load(f))


    # ------------------------------------------------------------------
    # 2026-10-28 token-5 离散 token API
    # ------------------------------------------------------------------
    def test_14_tokens_encode_decode(self):
        from tokens import tok5
        # encode：默认返回 token-5 位流（b64）
        code, j = http("POST", self.port, "/api/tokens/encode",
                       {"text": "Hello World"})
        self.assertEqual(code, 200)
        self.assertEqual(j["n_tokens"], 5)
        self.assertEqual(j["n_steps"], 5)
        self.assertEqual(j["skipped"], [])
        items = tok5.loads(base64.b64decode(j["tokens_b64"]))
        self.assertEqual(len(items), 5)
        self.assertTrue(all(it["kind"] == "step" for it in items))
        # display 形式：{text: 00001}
        code, jd = http("POST", self.port, "/api/tokens/encode",
                        {"text": "Hello World", "format": "display"})
        self.assertEqual(code, 200)
        self.assertTrue(jd["display"].startswith("{text: "))
        self.assertEqual(jd["display"].count("{text: "), 5)
        # decode：b64 → display
        code, j3 = http("POST", self.port, "/api/tokens/decode",
                        {"tokens_b64": j["tokens_b64"]})
        self.assertEqual(code, 200)
        self.assertEqual(j3["n_tokens"], 5)
        self.assertEqual(j3["n_steps"], 5)
        self.assertEqual(j3["display"], jd["display"])
        # decode：ids 输入
        code, j4 = http("POST", self.port, "/api/tokens/decode",
                        {"ids": [51046]})
        self.assertEqual(code, 200)
        self.assertEqual(j4["n_tokens"], 1)
        # 中文 + 未收录字符跳过记录
        code, j6 = http("POST", self.port, "/api/tokens/encode",
                        {"text": "\u4f60\u597d\U00020000"})
        self.assertEqual(code, 200)
        self.assertEqual(j6["n_tokens"], 2)
        self.assertEqual(j6["skipped"], ["\U00020000"])
        # 入参错误 → 400
        code, j7 = http("POST", self.port, "/api/tokens/encode",
                        {"text": 123})
        self.assertEqual(code, 400)
        code, j8 = http("POST", self.port, "/api/tokens/decode", {})
        self.assertEqual(code, 400)
        code, j9 = http("POST", self.port, "/api/tokens/decode",
                        {"tokens_b64": "not-base64!!"})
        self.assertEqual(code, 400)
        # Token 编辑器 → .tokens 导出（token-5 文本）
        code, je = http("POST", self.port, "/api/tokens/export",
                        {"text": "<00001>{text: 51046}<00002>"})
        self.assertEqual(code, 200)
        self.assertEqual(je["n_steps"], 1)
        self.assertEqual(je["n_tokens"], 1)
        got = tok5.loads(base64.b64decode(je["tokens_b64"]))
        self.assertEqual([it["kind"] for it in got],
                         ["ext", "step", "ext"])
        self.assertEqual(got[1]["elements"][0]["id"], 51046)
        # 结构非法 / 空流 → 400（不产出坏 .tokens）
        code, _ = http("POST", self.port, "/api/tokens/export",
                       {"text": "{text: 0001"})
        self.assertEqual(code, 400)
        code, _ = http("POST", self.port, "/api/tokens/export", {"text": ""})
        self.assertEqual(code, 400)

    def test_15_vocab_add(self):
        # 添加 token（四类 + 查重）；server 子进程写仓库 vocab.json，
        # tearDownClass 负责还原
        code, j = http("POST", self.port, "/api/vocab/add",
                       {"token": "\U00020000", "category": "regular"})
        self.assertEqual(code, 200)
        self.assertEqual(j["token"], "\U00020000")
        self.assertEqual(j["category"], "regular")
        self.assertIsInstance(j["id"], int)
        self.assertEqual(j["vocab_size"], 88704)
        # 重复添加 → 400
        code, j2 = http("POST", self.port, "/api/vocab/add",
                        {"token": "\U00020000", "category": "regular"})
        self.assertEqual(code, 400)
        self.assertIn("\u51b2\u7a81", j2.get("error", ""))
        # 与基础词表撞名 → 400
        code, j3 = http("POST", self.port, "/api/vocab/add",
                        {"token": "hello", "category": "regular"})
        self.assertEqual(code, 400)
        # 非法类别 / 空文本 → 400
        code, _ = http("POST", self.port, "/api/vocab/add",
                       {"token": "x", "category": "bogus"})
        self.assertEqual(code, 400)
        code, _ = http("POST", self.port, "/api/vocab/add",
                       {"token": "", "category": "system"})
        self.assertEqual(code, 400)
        # 新增后可被 enctok 收录（常规优先）
        code, j4 = http("POST", self.port, "/api/tokens/encode",
                        {"text": "\U00020000"})
        self.assertEqual(code, 200)
        self.assertEqual(j4["n_tokens"], 1)
        self.assertEqual(j4["skipped"], [])
        # 系统类添加
        code, j5 = http("POST", self.port, "/api/vocab/add",
                        {"token": "<ts>", "category": "system"})
        self.assertEqual(code, 200)
        self.assertEqual(j5["category"], "system")


if __name__ == "__main__":
    unittest.main()
