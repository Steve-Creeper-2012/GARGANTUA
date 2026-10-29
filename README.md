# GARGANTUA v1

> A multimodal early-fusion autoregressive model for world perception and reliable understanding.

GARGANTUA is a next-generation AI framework built from the ground up for **multimodal understanding and generation**. It unifies text, video, and audio within a single transformer architecture, sharing the same `d_model` and KV cache while maintaining modality-specific embedding projections and mechanisms.

## Architecture Overview

### Core Design

- **Encoder-Decoder**: Standard encoder-decoder with shared KV cache. The encoder processes user input bidirectionally; the decoder generates autoregressively.
- **Early Fusion**: All modalities (text, image, video, audio) are embedded into the same `d_model=1024` dimensional space before entering the transformer, but each modality uses its own projection mechanism.
- **Shared KV**: A unified KV sequence stream split into two segments — full-resolution KV + a compressed segment that carries two independent compressions (CSA retrieval view, HCA global-summary view).
- **No Cross-Attention / No Prefill**: The encoder writes the input into the shared KV; the decoder reads it directly and starts from the last input token. There is no cross-attention and no separate prefill pass.

### Block Structure

Each block follows the order:

```
KDA → KDA → KDA → MLA-CSA → MLA-HCA → FFN(GLU) → AttnRes
```

- **KDA** (Kimi-style Delta Attention): Gated linear recurrence with delta-state updates, inspired by Moonshot AI. Three consecutive KDA layers carry the recurrent/positional path.
- **MLA** (Multi-head Latent Attention): Low-rank joint KV compression, NoPE (no RoPE branch), inspired by DeepSeek-V2.
  - **MLA-CSA**: the retrieval view — 4:1 compressed segment + the whole full-KV segment.
  - **MLA-HCA**: the global-summary view — 32:1 compressed segment + the whole full-KV segment.
  - Both layers are plain MLA when the KV they read is uncompressed; the difference is which compressed segment they see.
- **FFN**: GLU variant, `d_ff=4096`.
- **AttnRes** (Kimi/Moonshot, [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals), arXiv 2603.15031): before every sub-layer, the block aggregates the token embedding, all completed block representations and the current partial sum with a softmax pseudo-query (`h = softmax(norm(V)·w)ᵀV`); block-internal accumulation stays standard residual. A zero-initialized scalar gate at the block end mixes the block input back in (both residual forms are kept, complementarily).

### Fast Weight Memory

A global `FastWeightBank` serves as long-term memory. It is **structurally one KDA layer** (same q/k/v projections, causal short conv, L2 norm, channel-wise forget gate, output gate, RMSNorm); the only difference is that its state persists across calls. It is invoked once after the first block in both encoder and decoder, reading with the old state and then writing every token (no selective filtering yet). Based on Schmidhuber's 1991 fast weight paper.

### KV Cache Design

```
Full KV (uncompressed) | Compressed KV
                       ├── CSA view  (4:1, retrieval)      → MLA-CSA
                       └── HCA view  (32:1, global summary) → MLA-HCA
```

- **Full KV**: uncompressed, the most recent `ctx_full` tokens.
- **Compressed KV**: everything older, kept as **two independent compressions of the same content** — CSA and HCA.
- Both views read **the compressed segment + the entire full segment**; when the input is still inside the full region there is nothing to compress and the layers behave as standard MLA (the old `csa_local_window` knob was removed on 2026-10-25).
- **Compression Rule**: a time step is atomic. A group never straddles a step boundary, and an oversized step may be compressed internally into several tokens that all keep that step's id. Each group targets the compression ratio (4 or 32) tokens; when whole steps cannot hit it exactly, the grouping with the smallest error wins (a slight overshoot is allowed, ties prefer the smaller side).
- **Eviction**: at time-step granularity (whole steps discarded together). Evicted KV vectors are dropped, but their **original token ids are archived** (`SharedKV.archive`) — context overflow loses the KV view, never the raw ids.
- **Buffer**: compensates for length changes when primitive/audio tokens are compressed.

### Context Length (model-creation-time option)

The unified KV stream is split into two regions, and the split is chosen when the model is created (recorded in `config.json`, honored by every entry point via `common.build_model`):

| Region | Meaning | Default |
|---|---|---|
| **Full** (`ctx_full`) | Uncompressed recent tokens | 1024 |
| **Compressed** (`ctx_compress`) | Capacity, counted in **original** tokens, of everything older | 3072 |

Total context = `ctx_full + ctx_compress` (default 4096). The compressed region is served by **two views over the same content** — this is the "two compression layers" of the spec:

- **CSA** (`csa_ratio`:1, default 4): the retrieval view — learnable step-grouped compression of the compressed segment, then standard MLA over the whole full segment.
- **HCA** (`hca_ratio`:1, default 32): the global-summary view — heavier compression of the same segment, then the same whole full segment.

Compression, eviction and the causal mask are all **time-step atomic** — a group never straddles a step boundary, and eviction discards whole steps. `csa_ratio` is an architectural parameter (the compressor's position embedding is `[ratio, d_model]`), so changing it changes the checkpoint layout; `hca_ratio ≥ csa_ratio` is enforced. There is no separate CSA window: the retained region of CSA **is** the full-KV region.

Choose it three ways (all validated, all landing in the same `config.json` contract):

```bash
python service/init/init.py --name alpha --ctx-full 2048 --ctx-compress 8192 \
    --csa-ratio 8 --hca-ratio 64                      # total 10240
# 或走 Web UI 的建模表单 / POST /api/model/init（同一套参数）
```

or in the web UI's model-creation form (four number inputs, defaults shown), or `POST /api/model/init` with `ctx_full` / `ctx_compress` / `csa_ratio` / `hca_ratio` (omitted = spec default, illegal value = 400). `GET /api/status` and `GET /api/system` return the active model's real configuration (`ctx` + `kv_note`), so the UI never shows a stale spec constant.

## Token System

All text tokens are discrete. The `.tokens` binary is **token-5**: a 5-bit
bitstream over an alphabet of exactly 27 symbols
(`0-9 { } < > | : w t v A-F x y`), terminated by code 27 and zero-padded to a
byte boundary. Any other program opening it sees garbage; `tokens/tok5.py` is
the interpreter.

Inside the stream, every element is wrapped in a time step:

| Element | Meaning |
|---|---|
| `{t:00001}` | text token — visible, has a vector, enters the loss |
| `{w:<hex>}` | audio token — 533 samples × 16 bit = 2132 hex chars |
| `{v:<hex>x:10y:24}` | visual patch — 32×32 px, RGBA 10 bit each = 10240 hex chars |
| `<00005>` | extended system token — has an embedding, no position coordinate, never in the loss |
| `{ } \|` | structure characters — **not** tokens (no position, no loss, no embedding) |

Multiple elements in one step are separated by `|`; `<...>` lives at stream
level, outside `{}`. The human-readable form is `{text: 00001}`,
`{wave: <hex>}`, `{visual: <hex> x: 10 y: 24}` — the editor accepts both.

Why extended tokens carry the "position embedding" role: KDA lets the model
learn relative order itself, so the standard Transformer position-embedding
job is delegated to these extensible tokens; plain structure characters need
no embedding at all.

The old tok-32 (big-endian `uint32`) format is retired; `tokens/tok32.py`
remains only as a reader for historical files.

The vocabulary (88,704 slots) is split into **four categories**; `enctok`
only ever looks up the two visible ones:

| Category | Contents | Example |
|---|---|---|
| `regular` (visible) | digits, whitespace, all Unicode P*/S* symbols (half+full width), all emoji (incl. ZWJ/VS16/tag components), all CJK basic + ext-A chars, 30K common English words | `你`, `😀`, `hello`, `！` |
| `fallback` (visible) | 26 lowercase letters `a-z` (OOV words are spelled letter-by-letter) | `q` |
| `special` (invisible) | `<caps>`, `<user>`, `<user/>`, `<model>`, `<model/>`, `<stop>`; primitive (draw) output tokens — geometrize_256_v1 layout | `<caps>` |
| `system` (invisible) | time-step structure: `{` open, `}` close, `|` intra-step separator | `{` |

- Every surface string is unique across the whole vocabulary (single-letter
  words `a`/`i` are *not* word tokens — the fallback letters cover them), so
  `token_to_id` reverse lookup is unambiguous. The only intentional exception:
  the three system display characters `{ } |` also exist as visible symbols;
  text→ID always resolves to the visible symbol, ID→display shows the literal.
- **Adding tokens**: `POST /api/vocab/add` (or `Vocab.add_token`) with one of
  the four categories. Duplicates are rejected against the entire vocabulary;
  new tokens consume reserved tail slots of their category, so the model
  output dimension never changes. User additions persist in `vocab.json`
  (`meta["user_added"]`) and are replayed on rebuild.
- **enctok** (`python service/tokens/enctok.py "text"`): word segments `[A-Za-z]+`
  hit the word list (`<caps>` + word for `Hello`; ALL-CAPS/mixed case is
  spelled letter-by-letter with `<caps>` per capital — lossless); other chars
  look up regular tokens, then fallback letters; unmapped chars are skipped
  and reported. Output: hex string or `.tokens` binary.
  `--vocab <vocab.txt|模型目录>` 可用指定词表编码（见「每模型词表」）。
- **tok5** (`tokens/tok5.py`): the token-5 codec — characters ↔ 5-bit
  bitstream ↔ elements ↔ time-step stream, plus payload helpers
  (`samples_to_hex`, `rgba8_to_hex`) and `to_display` / `parse_display`.
- **dectok** (`python service/tokens/dectok.py stream.tokens`): renders a
  token-5 file as `{text: 00001}` …; `--text` renders surface text.
- **gts** (`tokens/gts.py`): *legacy* tok-32-era editor parser, kept read-only.
  The editor now speaks token-5 directly (`POST /api/tokens/export`).
- **tokenset** (`tokens/tokenset.py`): the only reader used by training —
  token-5 `.tokens` → `(ids, step_ids)`, rejecting files without a valid
  time-step structure and files carrying `{w:}`/`{v:}` elements (the current
  `forward_train` only consumes discrete IDs).

### Time-Step Format (.gts / { } |)

A time-step is the atomic unit for KDA recurrent-state recursion and logical
parallelism (compression, sliding-window eviction and causality never cross a
step boundary). Each step is wrapped by the system tokens `{ }`; multiple
tokens inside one step are separated by `|`:

```
{<user>}{h|e|l|l|o}{<user/>}
```

- Text: one token per time-step (`{ } |` stay implicit in the Step list).
- Image: one time-step (all patch tokens inside).
- Audio: 533 samples = one token = one time-step.
- Video: each frame = one time-step (all patch tokens + that frame's audio
  token inside).
- `.tokenvector` (UTF-8) archives audio/visual vectors — no discrete IDs for
  those modalities. Header declares `dim=` and `precision=` (`f32`/`f16`/
  `bf16`); steps are `{ }`-delimited and intra-step token vectors are
  `|`-separated; the body is numbers only. Video frames store patch and audio
  vectors projected to the same `d_model` in a single file.

## Embedding Strategy

Embedding is **not unified** — each modality has its own projection path to `d_model=1024`:

| Modality | Projection | Position Info |
|---|---|---|
| Text | `wte` lookup table | None |
| Visual | `visual_proj` (4096 → 1024) | `visual_coord_x` + `visual_coord_y` (2D absolute) |
| Audio | `audio_proj` (533 → 1024) | None |
| Drawing | `wte` (in-vocab IDs) | None |

**Segment marker = type embedding.** `type_emb` is a learned table with one
entry per segment kind (user / model / none), applied to **every token inside
that segment** — there is no start/end distinction. It plays the same role as
a position embedding (a learned additive vector), not a token. The boundary
tokens `<user>`, `<user/>`, `<model>`, `<model/>` are system-level markers:
they do not enter the shared KV and are not ordinary tokens in the loss.

Within a single time-step (e.g., one video frame), all patches and audio tokens share the same `step_id` and receive bidirectional attention (no causal mask within the step).

## Visual Processing

### Image → Patches
- Input is padded to multiples of 32, split into 32×32 RGBA patches.
- Each patch = 1024 pixels × 4 channels = 4096 bytes, flattened into a vector.
- `visual_proj` maps the 4096-byte vector to `d_model=1024`.
- `(gx, gy)` grid coordinates are added via learned position embeddings (not tokenized).

### Video → Token Stream
- Fixed 30fps.
- **Frame 1**: Full input (all patches).
- **Frames 2–30**: Only patches with pixel-level changes.
- **Frame 31** (per second): Full input.
- **All-changed frame**: If all patches change, full input + reset counter.
- **Last frame**: Always full input.
- Audio is aligned at 533 samples/frame (16kHz / 30fps).

## Audio Processing

- **Input**: 16kHz mono, any format (mp3/flac/ogg/wav via ffmpeg).
- **Tokenization**: 533 samples → 1 token (~30 tok/s).
- **Embedding**: 533-sample vector → `audio_proj` → `d_model=1024`.
- **Output**: MIDI discrete tokens temporarily removed (2026-10-24,
  steve: to be redesigned; midi_codec and /api/codec/midi were removed).

## Primitive Pipeline (image → primitive tokens)

Adopted from [wonderfulearth/primitive-operation-painter](https://github.com/wonderfulearth/primitive-operation-painter)
(vendored **verbatim** under `service/prims/` — the converter is copied
byte-for-byte, never modified; the rest of the upstream repo is kept under
`service/prims/upstream/` for reference).
**Languages**: the image→sequence converter (`fast_shape_render/`) is
**Rust + wgpu** (GPU compute; Metal on macOS, DirectX 12 on Windows,
Vulkan on Linux); the driver + CPU fallback is **Python + numpy**.

- **Converter (照搬第三方代码)**: reads images, fits a sequence of
  rectangle/ellipse operations on the GPU (275 search steps), and writes
  CSV (`image_name,cx,cy,w,h,shape_type,theta,r,g,b`; first row per image
  is the background). Driven by `service/prims/image_to_primitives.py`
  (`convert_with_renderer`, subprocess).
- **CPU fallback**: the same greedy algorithm in pure numpy (no GPU
  required), outputting the identical primitive schema.
- **Tokens (照搬第三方布局, `geometrize_256_v1`)**: 9 tokens per primitive
  in field order `cx, cy, theta, w, h, shape, r, g, b`, each field an
  independent bin range (token ID = `DRAW_BEGIN + field offset + bin`);
  canvas 256×256; a step-first token in the special bin range ends the
  sequence. See `server/spec.py` (DRAW_*) and
  `service/tokens/draw_codec.py`.
- **Quality modes** (CPU path): `fast` = 20 candidates/step, no
  hill-climbing (~2–5s/100 steps); `slow` = 50 candidates + short
  hill-climbing (~30–60s/100 steps).
- **CSA Compression**: after a primitive's 9 tokens are written, the KV
  cache compresses them 9:1 in place.

### Startup dependencies

```sh
sh service/prims/setup.sh
```

installs the Rust toolchain **into `environment/rust/`** (fully isolated
from the system: `RUSTUP_HOME`/`CARGO_HOME` are redirected, `--no-modify-path`
keeps your shell config untouched), then builds the GPU converter
(`cargo build --release`, first run ~5–15 min). Download mirrors are
auto-selected by live speed test (USTC → TUNA → official; override with
`RUSTUP_DIST_SERVER` / `CARGO_CRATES_MIRROR`). macOS needs Xcode Command
Line Tools for the linker/Metal SDK. Without the build the pipeline
automatically falls back to the CPU fitter; `GARGANTUA_SHAPE_RENDERER`
overrides the binary path. Converter env vars (`SHAPE_RENDERER_INPUT_DIR`,
`_OUTPUT_DIR`, `_NUM_VERSIONS`, `_MAX_OUTPUT_STEPS`) are documented in
`service/prims/fast_shape_render/README.md`.

## Training

### Optimizer Strategy
- **Muon**: Applied to all hidden-layer 2D matrices (attention, FFN, projections).
- **AdamW**: Applied to embedding tables, prediction head, and RMSNorm parameters.
- Muon can be disabled — falls back to full AdamW.

### Data Format
Training data is a **`.tokens` file** — the token-5 bitstream, with the
`{ }` time-step structure written out explicitly. The Token editor exports
exactly this format (Export .tokens), and it is the only accepted training
input; JSON/JSONL uploads are rejected with `400`.

```bash
# queue one stream (CLI side, same file the editor produces)
cp stream.tokens data/queue/
```

Each file is one token stream. The trainer expands it into flat positions
plus a parallel `step_ids` array (`tokens/tokenset.py`), and appends a
`<stop>` step at the end of the document. Files without a valid `{ }`
structure are rejected before training starts.

**Multimodal documents.** A file may mix `{t:}`, `{w:}` and `{v:}` elements.
Text-only docs go through `make_batch` + `forward_train` (embedding lookup,
cheapest). Docs containing media go through `make_batch_emb` +
`forward_train_embeddings`: the whole document is projected once
(`visual_proj` / `audio_proj`, both plain `Linear(…, d_model)`), and media
positions are given target `-100` — the `lm_head` can only predict vocabulary
IDs, so images and audio act purely as context while the text targets drive
the loss. Both paths share `service/train/train_step.py`, so the queue daemon
and the CLI trainer can never diverge.

### Training Script
```bash
python service/train/train.py --model models/my_model --data data/
```

## Inference

```bash
python service/infer/infer.py --model models/my_model --prompt "Hello"
```

The decoder stops immediately when `<stop>` (ID 5) is sampled.

**Multimodal prompts.** `POST /api/infer` accepts `media: [{kind, b64,
filename}]` with `kind` ∈ `image | audio | video` (up to 4 attachments, 64 MB
total). The media is decoded in the inference service (patches / 533-sample
tokens), placed *before* the text prompt, projected into the same `d_model`,
and encoded into the KV cache — generation itself stays textual, so at least
one text token is required as the decode seed.

### Token-id archive (inference)

Every inference call persists its token ids before/while generating:

```
logs/infer_archive/<YYYYmmdd_HHMMSS>_<id>.tokens   # tok-32, one { id } step per token
logs/infer_archive/<...>.json                      # model / counts / ctx_total / n_evicted
```

The `.tokens` file receives the prompt at start and each generated token as it
is sampled (flushed per token), so a crash, an idle exit or KV eviction can
never lose ids — only KV *vectors* are dropped when the context overflows.
The sidecar records `n_tokens`, `n_prompt_tokens`, `n_generated_tokens`,
`ctx_total` and `n_evicted`. `--archive-dir` overrides the location; the
server passes `<log dir>/infer_archive` and returns the path as `archive` in
the inference result. Archiving failure is logged to stderr and degrades to
"no archive" — it never blocks inference.

## Web Interface

A built-in web UI provides:
- **Model Management**: Create models with configurable encoder/decoder block counts **and context length** (full / compressed capacity + CSA/HCA compression ratios — see "Context Length" above).
- **Training Queue**: Upload `.tokens` streams, queue training tasks with automatic execution.
- **Inference Chat**: Standard AI chat interface with collapsible debug panel (temperature, top-p, max_new_tokens).
- **Codec Visualizers**:
  - Text → Token conversion
  - Audio → Wave token conversion
  - Image/Video → Patch token conversion
  - Image → Primitive sequence conversion
  - Token editor with a spec-aligned palette and **Export .tokens** (writes the same tok-32 stream training consumes)
  - **Vocabulary page**: ID-region occupancy, presets, plus global search / add-token (the editor itself no longer carries vocabulary operations)

### Launch

终端脚本只负责「把隔离环境准备好，然后让位给 server」；环境检查、窗口
拉起、服务调度全部由 server 自己做：

```bash
./start/terminal/start.sh        # macOS / Linux
./start/terminal/start.command   # macOS: double-click in Finder (Terminal → app window)
start\terminal\start.bat         # Windows (double-click)
```

首次启动 `setup.sh` / `setup.bat` 会：

1. 从 python-build-standalone 发现并下载 **3.12 系列最新版**（atom feed
   取最新 release tag + HEAD 探测补 patch 号），**镜像自动测速选取**
   （对 gh-proxy / GitHub 直连 / SJTU 镜像并行发 Range 请求前 2MB 比速率，
   选最快的下载；也可用 `GARGANTUA_PY_BASE_URL` 手动指定）；
2. 解包到 `environment/python/`，用它建虚拟环境 `environment/venv/`
   （**不碰系统 Python**；项目目录被移动后符号链接悬空会自动重建）；
3. 按分类装依赖 `start/requirements/{core,window,dev}.txt`（core 必装，
   window 桌面环境才装，dev 仅测试需要；pip 源同样测速选取：
   TUNA vs PyPI 官方，默认快者）；
4. 写依赖清单哈希 stamp——之后启动秒过，只有清单变了才重装。

随后 `server/server.py` 启动：先做环境预检（解释器 ≥3.12 且 ≤3.12 系列、
主依赖可导入——torch 2.2.2 没有 3.13 预编译轮子，版本不对会给可执行提示
而不是 20 分钟源码编译），再起 HTTP 服务与独立桌面窗口。

所有依赖与工具链（Python / venv / Rust）统一放在 `environment/`，与系统
完全隔离——契约见 `environment/README.md`。

| Flag | Effect |
| --- | --- |
| `--port N` | Port for the HTTP server (default 8712) |
| `--no-window` | Server only — no GUI window (SSH / servers / CI) |
| `--idle-timeout S` | Service subprocess idle self-exit (default 300s) |
| `--model-dir` / `--data-dir` / `--log-file` | Override default paths |

环境变量：`GARGANTUA_PY_RELEASE` / `GARGANTUA_PY_VERSION`（锁定
python-build-standalone 版本）、`GARGANTUA_PY_BASE_URL`（指定 Python
下载镜像）、`GARGANTUA_NO_WINDOW=1` / `GARGANTUA_INSTALL_DEV=1`（setup
阶段）、`PIP_INDEX_URL`（自定义 pip 源）。

### The window is not a browser

The UI opens in a **standalone desktop window** via
[pywebview](https://pywebview.flowrl.com/) (macOS `WKWebView` / Windows
`WebView2` / Linux `WebKitGTK`) — the OS web component, not a second browser
engine. Chrome/Firefox spend 300–800 MB per tab on renderer processes,
extensions and their own GPU pipeline; that budget belongs to training here
(model + optimizer state + KV cache). The window dependency is deliberately
**not** in the core requirements:

```bash
pip install -r start/requirements/window.txt   # ~18 MB, optional
```

Without it the server still starts and prints the URL — you can open it in
a browser by hand.

Process model (deliberate, Steam-like):

```
server.py ──┬── infer/train/init 服务子进程   按需拉起，空闲自行退出
            └── window.py                    UI 窗口（崩了不影响 server）
```

- Closing or crashing the window leaves the server running; the log prints
  the command to reopen the UI.
- Exiting the server takes the window and all service subprocesses down
  with it (no orphans pointing at a dead port). Ctrl-C / SIGTERM both go
  through the cleanup path.

### Services start on demand (memory)

推理 / 训练 / 建模是重活（模型权重单个 safetensors 可达数 GB），全部以
**独立子进程**按需拉起，server 本体只做 HTTP 路由 + 记账，常驻不带权重：

- 协议 = JSON-lines over stdin/stdout（请求 `{"id","op",...}` → 响应
  `{"id","ok",result|error}`；服务 stdout 只准写 JSON 行，人肉日志走
  stderr）；服务退出条件 = `shutdown` / stdin EOF（server 死了）/ 空闲超时。
- infer 服务：请求带 `model_dir`，换了模型才重载；空闲 300s 自行退出。
- train 服务：`start` 下发模型/队列/日志路径与 block_size/steps/optimizer；
  启动时从 `logs/auto_train.jsonl` 回填 loss 曲线；训练进行中（队列有文件）
  绝不因空闲退出。
- init 是一次性重活：`run_oneshot` 起进程、`--json` 末行协议拿结果、跑完即退。
- PyTorch 的 CPU 分配器在进程内不向 OS 归还内存——所以「按需」必须是
  进程级的，进程退出才是真省内存。

### HTTP contract

All `/api/*` endpoints share one error convention, so the UI never has to
guess:

| Status | Meaning |
| --- | --- |
| `400` | Invalid/missing parameter (odd `block_size`, `steps < 1`, `temperature <= 0`, illegal model name, unknown `quality`, malformed JSON) |
| `404` | Unknown path |
| `409` | State conflict — training running while switching/creating a model, or overwriting an existing model without `overwrite:true` |
| `411` | `Content-Length` required (chunked encoding rejected) |
| `413` | Body larger than 512 MB |

Long-running work (inference, model init, image→primitives, **drawing render**) returns `202` with a `task_id`; poll `GET /api/task/<task_id>` until `status == "done"`. Numeric parameters are never silently defaulted: an explicit `0` is validated as `0` rather than being replaced by the default value. Booleans, non-numeric strings and negative `Content-Length` are rejected with `400` (the last one used to bypass the 512 MB cap and turn into an unbounded read).

## Project Structure

```
.
├── start/                     # 启动链：终端脚本 + 分类依赖 + 路径引导
│   ├── boot.py                # Path bootstrap: server/ + service/* onto sys.path
│   ├── terminal/              # Terminal entry points
│   │   ├── setup.sh / .bat    # Isolated Python 3.12.x + venv + deps (idempotent,
│   │   │                      #   mirrors auto-selected by live speed test)
│   │   └── start.sh / .bat / .command   # setup → exec server.py
│   └── requirements/          # Categorized dependencies
│       ├── core.txt           # torch / numpy / pillow / safetensors (required)
│       ├── window.txt         # pywebview — standalone window (~18 MB, optional)
│       └── dev.txt            # pytest (tests only)
├── environment/               # ALL dependencies & toolchains (system-isolated)
│   ├── python/                # standalone CPython 3.12.x (downloaded by setup)
│   ├── venv/                  # virtual env on the above interpreter
│   ├── rust/                  # Rust toolchain (installed by service/prims/setup.sh)
│   └── README.md              # the contract (see it before touching this dir)
├── server/                    # All server files
│   ├── spec.py                # Frozen architecture contract (single source of truth)
│   ├── common.py              # Shared checkpoint I/O + validation (config.json
│   │                          #   contract, incl. vocab.txt reader)
│   ├── model.py               # Gargantua model (KDA, MLA, FastWeight, SharedKV)
│   ├── muon.py                # Muon optimizer implementation
│   ├── server.py              # App assembly + routing + startup orchestration
│   ├── server_env.py          # Env precheck + subprocess helpers
│   ├── server_services.py     # ServiceSupervisor: on-demand subprocesses (JSON-lines)
│   ├── server_models.py       # ModelManager (registry, no weights) / TaskPool
│   ├── server_media.py        # Uploaded-media decode + size guards (b64_to_pil / ffmpeg)
│   ├── server_codecs.py       # CodecApi mixin — all /api/codec/* endpoints
│   ├── server_train.py        # TrainApi mixin — all /api/train/* endpoints
│   ├── server_tokens.py       # TokensApi mixin — /api/tokens/* + /api/vocab/add
│   └── server_util.py         # num_param / pip source / optional-install command
├── service/                   # One folder per service (started on demand)
│   ├── infer/infer.py         # Autoregressive generation (CLI + --serve mode)
│   ├── train/train.py         # Training loop with Muon/AdamW optimizer routing
│   ├── train/auto_train.py    # Queue trainer daemon (CLI + --serve mode)
│   ├── init/init.py           # Model initialization (CLI + --json one-shot mode)
│   ├── tokens/                # Codecs for all modalities
│   │   ├── tok32.py           # tok-32 interpreter: .tokens binary ↔ IDs ↔ hex
│   │   ├── gts.py             # editor text (.gts) → IDs, backs Export .tokens
│   │   ├── tokenset.py        # .tokens training streams → (ids, step_ids)
│   │   ├── enctok.py          # Text → token ID stream (regular first, fallback next)
│   │   ├── dectok.py          # Token ID stream → readable ({ } | + hex / surface text)
│   │   ├── timestep.py        # { } | time-step structure (wrap/unwrap/render)
│   │   ├── tokenvector.py     # .tokenvector vector archive format (dim+precision)
│   │   ├── text_codec.py      # Text ↔ List[Step] (one token per step)
│   │   ├── audio_codec.py     # Audio → wave tokens (+ .tokenvector export)
│   │   ├── image_codec.py     # Image/Video → patch tokens (+ .tokenvector export)
│   │   ├── draw_codec.py      # Primitive sequence ↔ drawing tokens + PNG/SVG render
│   │   ├── build_vocab.py     # vocab.json generator (layout from spec.py)
│   │   ├── vocab.py           # Vocab view: global + per-model (vocab.txt) + add_token
│   │   ├── vocab.json         # Built global vocabulary (88,704 slots)
│   │   └── words_en.txt       # 30K English word list (frequency order)
│   ├── prims/                 # Primitive pipeline (image → primitive sequence)
│   │   ├── image_to_primitives.py  # Driver: GPU subprocess + CPU fallback fitter
│   │   ├── setup.sh           # Build GPU converter (Rust into environment/rust/)
│   │   ├── fast_shape_render/ # ★ vendored Rust/wgpu converter, verbatim
│   │   └── upstream/          # the rest of the upstream repo, verbatim (reference)
│   ├── web/                   # Frontend
│   │   ├── index_source.html  # Single-page app source (11 pages)
│   │   ├── index.html         # Minified artifact (regenerate via web/minify.py)
│   │   └── minify.py          # Safe minifier with --check self-test
│   ├── window/window.py       # Standalone desktop window (pywebview, no browser)
│   └── test/                  # Unit + integration tests
│       ├── test_model.py
│       ├── test_codecs.py
│       ├── test_tokens.py
│       ├── test_image2prim.py
│       ├── test_server.py
│       ├── test_fixes.py      # Regression tests for the 2026-10-20 audit
│       ├── test_audit_20261023.py  # 2026-10-23 audit fixes + context-length feature
│       ├── test_services.py   # 2026-10-24: supervisor / vocab.txt / env precheck
│       ├── verify_formats.py  # Format round-trip verifier
│       └── run_tests.py       # Full regression, one process per suite
├── models/                    # models/NAME/{model.safetensors, config.json, vocab.txt?}
├── data/                      # data/queue/ (pending) + data/done/ (trained)
└── logs/                      # auto_train.jsonl and friends
```

`server.py` is routing + startup orchestration (~600 lines) plus focused
modules: `server_env` (env precheck + window integrity), `server_services`
(on-demand service supervision), `server_models` (registry + task pool),
`server_media`, `server_codecs`, `server_train`, `server_tokens`,
`server_util`; `App` composes the three API mixins. All previous
`from server import X` paths still work through re-exports.

### Model directory contract

Every model lives in `models/NAME/` with `model.safetensors` (weights) and
`config.json` (records `n_enc_blocks` / `n_dec_blocks` and more). All entry
points (`service/init/init.py`, `service/train/train.py`,
`service/infer/infer.py`, `service/train/auto_train.py`) construct and load
models through `common.build_model()` so a model created with custom block
counts loads identically everywhere.

**Per-model vocabulary** (optional): drop a `vocab.txt` into the model
directory and that model gets its own token↔ID mapping — one token per
line, line *N* → ID `INVISIBLE_END + N` (the invisible region below
`INVISIBLE_END` holds the fixed special/system/modal tokens: `<stop>`,
`<caps>`, `{ } |`, draw tokens — their IDs are an architectural contract and
a custom vocab never occupies them). Empty lines are skipped; duplicates are
rejected; at most `VOCAB_SIZE_PADDED - INVISIBLE_END` entries. Models
without `vocab.txt` use the global vocabulary
`service/tokens/vocab.json`. When the active model changes, the server and
the infer/train service processes each call `set_model_vocab()`, so
`/api/tokens/*`, `/api/vocab`, training and inference all speak that
model's vocabulary. Export a starting point with
`python service/tokens/build_vocab.py --export-txt models/NAME/vocab.txt`.

## Testing

```bash
python service/test/run_tests.py                    # all suites, one process per suite
python service/test/run_tests.py --list             # list suites
python service/test/run_tests.py --suite test_model # a single suite
python -m unittest service.test.test_model -v       # unittest directly
```

`run_tests.py` first runs static gates (`py_compile` on every `.py`,
`bash -n` on every `.sh`, `minify.py --check`), then each suite in its own
subprocess. Suites run in **separate processes** on purpose: a full
`Gargantua()` is ~527.9M params (~2.1 GB fp32), and PyTorch's CPU allocator
keeps freed blocks in its own pool, so RSS never drops within a process —
running every suite under one `unittest discover` gets SIGKILLed (exit 137)
partway through `test_model`.

## Technical References

- **MLA**: DeepSeek-V2 — low-rank joint KV compression, NoPE.
- **KDA**: Moonshot AI / Kimi — linear gated delta recurrence.
- **CSA/HCA Compression**: DeepSeek NSA — learnable MLP compression within time-steps.
- **AttnRes**: Moonshot AI / Kimi — [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals) (arXiv 2603.15031), Block AttnRes over depth plus a block-level scalar gate.
- **Fast Weight**: Schmidhuber 1991 — persistent associative memory layer.
- **Primitive Rendering**: [wonderfulearth/primitive-operation-painter](https://github.com/wonderfulearth/primitive-operation-painter) — GPU image-to-sequence converter with hill-climbing shape fitting.

## License

GPLv3.0

---

**Note**: GARGANTUA v1 is a research-oriented framework. The architecture is fully implemented and forward-pass functional, but meaningful generation requires training on domain-specific data.
