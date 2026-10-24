# GARGANTUA v1

> A multimodal early-fusion autoregressive model for world perception and reliable understanding.

GARGANTUA is a next-generation AI framework built from the ground up for **multimodal understanding and generation**. It unifies text, video, and audio within a single transformer architecture, sharing the same `d_model` and KV cache while maintaining modality-specific embedding projections and mechanisms.

## Architecture Overview

### Core Design

- **Encoder-Decoder**: Standard encoder-decoder with shared KV cache. The encoder processes user input bidirectionally; the decoder generates autoregressively.
- **Early Fusion**: All modalities (text, image, video, audio) are embedded into the same `d_model=1024` dimensional space before entering the transformer, but each modality uses its own projection mechanism.
- **Shared KV**: A unified KV sequence stream with dual-layer compression — full-resolution KV + compressed CSA/HCA views.
- **No Cross-Attention**: The decoder reads the shared KV directly without cross-attention layers.

### Block Structure

Each block follows the order:

```
KDA → MLA-HCA → KDA → MLA-CSA → KDA → FFN(GLU)
```

- **KDA** (Kimi-style Delta Attention): Gated linear recurrence with delta-state updates, inspired by Moonshot AI.
- **MLA** (Multi-head Latent Attention): Low-rank joint KV compression, NoPE (no RoPE branch), inspired by DeepSeek-V2.
  - **MLA-HCA**: Heavily compressed (32:1) with global attention.
  - **MLA-CSA**: Lightly compressed (4:1) with local sliding-window attention.
- **FFN**: GLU variant, `d_ff=4096`.

### Fast Weight Memory

A global `FastWeightBank` serves as long-term memory, structured identically to a KDA layer but with persistent state. It is invoked once after the first block in both encoder and decoder, updating its weights token-by-token. Based on Schmidhuber's 1991 fast weight paper.

### KV Cache Design

```
MLA1/2: [Full KV] | [Buffer] | [CSA]
MLA1/2: [Full KV] | [Buffer] | [HCA]
```

- **Full KV**: Uncompressed, first layer.
- **CSA**: Small compression + local attention (4:1 ratio).
- **HCA**: Heavy compression + global attention (32:1 ratio).
- **Compression Rule**: Compression happens within time-steps, never across time-step boundaries.
- **Sliding Window**: Eviction happens at time-step granularity (whole steps discarded together).
- **Buffer**: Compensates for length changes when primitive/audio tokens are compressed.

### Context Length (model-creation-time option)

The unified KV stream is split into two regions, and the split is chosen when the model is created (recorded in `config.json`, honored by every entry point via `common.build_model`):

| Region | Meaning | Default |
|---|---|---|
| **Full** (`ctx_full`) | Uncompressed recent tokens | 1024 |
| **Compressed** (`ctx_compress`) | Capacity, counted in **original** tokens, of everything older | 3072 |

Total context = `ctx_full + ctx_compress` (default 4096). The compressed region is served by **two views over the same content** — this is the "two compression layers" of the spec:

- **CSA** (`csa_ratio`:1, default 4): learnable step-grouped compression + a local sliding window (`csa_local_window`, default 512) over the tail of the full region.
- **HCA** (`hca_ratio`:1, default 32): heavier compression + global attention.

Compression, sliding-window eviction and the causal mask are all **time-step atomic** — a group never straddles a step boundary, and eviction discards whole steps. `csa_ratio` is an architectural parameter (the compressor's position embedding is `[ratio, d_model]`), so changing it changes the checkpoint layout; `hca_ratio ≥ csa_ratio` and `csa_local_window ≤ ctx_full` are enforced.

Choose it three ways (all validated, all landing in the same `config.json` contract):

```bash
python init.py --name alpha --ctx-full 2048 --ctx-compress 8192 \
    --csa-ratio 8 --hca-ratio 64 --csa-window 1024   # total 10240
python run.py --init alpha --ctx-full 2048            # same flags via launcher
```

or in the web UI's model-creation form (five number inputs, defaults shown), or `POST /api/model/init` with `ctx_full` / `ctx_compress` / `csa_ratio` / `hca_ratio` / `csa_local_window` (omitted = spec default, illegal value = 400). `GET /api/status` and `GET /api/system` return the active model's real configuration (`ctx` + `kv_note`), so the UI never shows a stale spec constant.

## Token System

All text tokens are discrete. Token IDs are **32-bit**; the `.tokens` binary
format (tok-32) is a plain big-endian `uint32` stream — 4 bytes per
"character", one character per token — so any other program opening it sees
garbage. Hex text form uses 8 hex digits per token.

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
- **enctok** (`python tokens/enctok.py "text"`): word segments `[A-Za-z]+`
  hit the word list (`<caps>` + word for `Hello`; ALL-CAPS/mixed case is
  spelled letter-by-letter with `<caps>` per capital — lossless); other chars
  look up regular tokens, then fallback letters; unmapped chars are skipped
  and reported. Output: hex string or `.tokens` binary.
- **dectok** (`python tokens/dectok.py stream.tokens`): `{ } |` render as
  literals, everything else as 8-hex IDs; `--text` renders surface text.

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

| Modality | Projection | Position Info | Type Embedding |
|---|---|---|---|
| Text | `wte` lookup table | None | `type_emb(TYPE_TEXT)` |
| Visual | `visual_proj` (4096 → 1024) | `visual_coord_x` + `visual_coord_y` (2D absolute) | `type_emb(TYPE_VISUAL)` |
| Audio | `audio_proj` (533 → 1024) | None | `type_emb(TYPE_AUDIO)` |
| Drawing | `wte` (in-vocab IDs) | None | `type_emb(TYPE_DRAW)` |

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
(vendored verbatim under `third_party/primitive-operation-painter/`).
**Languages**: the image→sequence converter (`fast_shape_render/`) is
**Rust + wgpu** (GPU compute; Metal on macOS, DirectX 12 on Windows,
Vulkan on Linux); the model / training / generation code is **Python +
PyTorch**.

- **Converter (照搬第三方代码)**: reads images, fits a sequence of
  rectangle/ellipse operations on the GPU (275 search steps), and writes
  CSV (`image_name,cx,cy,w,h,shape_type,theta,r,g,b`; first row per image
  is the background). Driven by `tokens/image_to_primitives.py`
  (`convert_with_renderer`, subprocess).
- **CPU fallback**: the same greedy algorithm in pure numpy (no GPU
  required), outputting the identical primitive schema.
- **Tokens (照搬第三方布局, `geometrize_256_v1`)**: 9 tokens per primitive
  in field order `cx, cy, theta, w, h, shape, r, g, b`, each field an
  independent bin range (token ID = `DRAW_BEGIN + field offset + bin`);
  canvas 256×256; a step-first token in the special bin range ends the
  sequence. See `spec.py` (DRAW_*) and `tokens/draw_codec.py`.
- **Quality modes** (CPU path): `fast` = 20 candidates/step, no
  hill-climbing (~2–5s/100 steps); `slow` = 50 candidates + short
  hill-climbing (~30–60s/100 steps).
- **CSA Compression**: after a primitive's 9 tokens are written, the KV
  cache compresses them 9:1 in place.

### Startup dependencies

```sh
sh third_party/primitive-operation-painter/setup.sh
```

installs the Rust toolchain (rustup), builds the GPU converter
(`cargo build --release`, first run ~5–15 min), and pip-installs the
Python-side deps (`pandas`, `matplotlib`; torch/numpy come from the
GARGANTUA venv). Without the build the pipeline automatically falls back
to the CPU fitter; `GARGANTUA_SHAPE_RENDERER` overrides the binary path.
Converter env vars (`SHAPE_RENDERER_INPUT_DIR`, `_OUTPUT_DIR`,
`_NUM_VERSIONS`, `_MAX_OUTPUT_STEPS`) are documented in
`third_party/primitive-operation-painter/fast_shape_render/README.md`.

## Training

### Optimizer Strategy
- **Muon**: Applied to all hidden-layer 2D matrices (attention, FFN, projections).
- **AdamW**: Applied to embedding tables, prediction head, and RMSNorm parameters.
- Muon can be disabled — falls back to full AdamW.

### Data Format
Training data must be uploaded as JSON files with one of these formats:
```json
{"text": "hello world"}
```
Or object arrays / JSONL (one object per line).

### Training Script
```bash
python train.py --model models/my_model --data data/
```

## Inference

```bash
python infer.py --model models/my_model --prompt "Hello"
```

The decoder stops immediately when `<stop>` (ID 27) is sampled.

## Web Interface

A built-in web UI provides:
- **Model Management**: Create models with configurable encoder/decoder block counts **and context length** (full / compressed capacity + CSA/HCA compression ratios — see "Context Length" above).
- **Training Queue**: Upload JSON datasets, queue training tasks with automatic execution.
- **Inference Chat**: Standard AI chat interface with collapsible debug panel (temperature, top-p, max_new_tokens).
- **Codec Visualizers**:
  - Text → Token conversion
  - Audio → Wave token conversion
  - Image/Video → Patch token conversion
  - Image → Primitive sequence conversion
  - Token editor with spec-aligned token panel and vocab search

### Launch

One `run.py` holds every bit of logic; the per-platform entry points only
locate the system Python and delegate to it:

```bash
./run.sh          # macOS / Linux
./run.command     # macOS: double-click in Finder (opens Terminal, then the app window)
run.bat           # Windows (double-click)
python run.py     # any platform, direct
```

On first launch `run.py` creates `.venv/`, installs `requirements.txt`,
and verifies the imports before starting anything. torch 2.2.2 has no wheel
above Python 3.12, so a newer interpreter is refused with an actionable
message instead of a 20-minute source build.

| Flag | Effect |
| --- | --- |
| `--port N` | Port for the HTTP server (default 8712) |
| `--no-window` | Server only — no GUI window (SSH / servers / CI) |
| `--test` | Full regression, one process per suite |
| `--init NAME` / `--train NAME` / `--infer NAME` | CLI shortcuts, no server |

### The window is not a browser

The UI opens in a **standalone desktop window** via
[pywebview](https://pywebview.flowrl.com/) (macOS `WKWebView` / Windows
`WebView2` / Linux `WebKitGTK`) — the OS web component, not a second browser
engine. Chrome/Firefox spend 300–800 MB per tab on renderer processes,
extensions and their own GPU pipeline; that budget belongs to training here
(model + optimizer state + KV cache). The window dependency is deliberately
**not** in `requirements.txt`:

```bash
pip install -r requirements-window.txt   # ~18 MB, optional
```

Without it `run.py` still starts the server and prints the URL — you can
open it in a browser by hand.

Process model (deliberate, Steam-like):

```
run.py ──┬── server.py    training + inference + HTTP  (closing the window does not stop it)
         └── window.py    the UI window               (crash/kill does not stop training)
```

- Closing or crashing the window leaves the server running; the log prints
  the command to reopen the UI.
- Exiting the server takes the window down with it (no orphan pointing at a
  dead port). Ctrl-C is forwarded to both.
- `PYTHONUNBUFFERED=1` is forced on children so startup logs appear
  immediately instead of after 8 KB of buffering.

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
├── spec.py              # Frozen architecture contract (single source of truth)
├── common.py            # Shared checkpoint I/O + validation (config.json contract,
│                        #   incl. context-length config: validate/read_ctx_config)
├── model.py             # Gargantua model (KDA, MLA, FastWeight, SharedKV)
├── train.py             # Training loop with Muon/AdamW optimizer routing
├── infer.py             # Autoregressive generation
├── init.py              # Model initialization
├── auto_train.py        # Trainer class with queue support
├── server.py            # HTTP layer only: App assembly + routing + Handler
├── server_models.py     # ModelManager / TaskPool / step_to_json / ConflictError
├── server_media.py      # Uploaded-media decode + size guards (b64_to_pil / ffmpeg)
├── server_codecs.py     # CodecApi mixin — all /api/codec/* endpoints
├── server_train.py      # TrainApi mixin — all /api/train/* endpoints
├── server_tokens.py     # TokensApi mixin — /api/tokens/* + /api/vocab/add
├── server_util.py       # num_param (explicit 0 is never swallowed by a default)
├── window.py            # Standalone desktop window (pywebview, no browser)
├── run.py               # Cross-platform launcher: venv, deps, server + window
├── run.sh               # Linux/macOS entry — finds system Python, delegates
├── run.command          # macOS double-click entry (Finder → Terminal → window)
├── run.bat              # Windows entry — finds system Python, delegates
├── muon.py              # Muon optimizer implementation
├── requirements.txt     # Pinned runtime dependencies (single source)
├── requirements-window.txt  # Optional: pywebview only (~18 MB, OS webview)
├── tokens/              # Codecs for all modalities
│   ├── tok32.py         # tok-32 interpreter: .tokens binary ↔ IDs ↔ hex
│   ├── enctok.py        # Text → token ID stream (regular first, fallback next)
│   ├── dectok.py        # Token ID stream → readable ({ } | + hex / surface text)
│   ├── timestep.py      # { } | time-step structure (wrap/unwrap/render)
│   ├── tokenvector.py   # .tokenvector vector archive format (dim+precision)
│   ├── text_codec.py    # Text ↔ List[Step] (one token per step)
│   ├── audio_codec.py   # Audio → wave tokens (+ .tokenvector export)
│   ├── image_codec.py   # Image/Video → patch tokens (+ .tokenvector export)
│   ├── draw_codec.py    # Primitive sequence ↔ drawing tokens + PNG/SVG render
│   ├── image_to_primitives.py  # Image → primitive sequence (fast_shape_render)
│   ├── build_vocab.py   # vocab.json generator (layout from spec.py)
│   ├── vocab.py         # Vocab view: 4-category maps + add_token
│   ├── vocab.json       # Built vocabulary (88,704 slots)
│   └── words_en.txt     # 30K English word list (frequency order)
├── web/                 # Frontend
│   ├── index_source.html  # Single-page app source (11 pages)
│   ├── index.html        # Minified artifact (regenerate via web/minify.py)
│   └── minify.py         # Safe minifier with --check self-test
└── tests/               # Unit + integration tests
    ├── test_model.py
    ├── test_codecs.py
    ├── test_tokens.py
    ├── test_image2prim.py
    ├── test_server.py
    ├── test_fixes.py    # Regression tests for the 2026-10-20 audit
    └── test_audit_20261023.py  # 2026-10-23 audit fixes + context-length feature
```

`server.py` used to be one 986-line file holding the HTTP layer, the model
manager, the task pool, the ffmpeg plumbing and every API handler. It is now
366 lines of routing plus six focused modules (`server_models`,
`server_media`, `server_codecs`, `server_train`, `server_tokens`,
`server_util`); `App` composes the three API mixins. All previous `from server import X` paths still
work through re-exports, and `tests/test_server.py` passes unchanged.

### Model directory contract

Every model lives in `models/NAME/` with `model.safetensors` (weights) and
`config.json` (records `n_enc_blocks` / `n_dec_blocks` and more). All entry
points (`init.py`, `train.py`, `infer.py`, `auto_train.py`, `server.py`)
construct and load models through `common.build_model()` so a model created
with custom block counts loads identically everywhere.

## Testing

```bash
python run.py --test          # all suites, one process per suite
python -m unittest tests.test_model -v   # a single suite
```

Suites run in **separate processes** on purpose: a full `Gargantua()` is
~527.9M params (~2.1 GB fp32), and PyTorch's CPU allocator keeps freed blocks
in its own pool, so RSS never drops within a process — running every suite
under one `unittest discover` gets SIGKILLed (exit 137) partway through
`test_model`.

## Technical References

- **MLA**: DeepSeek-V2 — low-rank joint KV compression, NoPE.
- **KDA**: Moonshot AI / Kimi — linear gated delta recurrence.
- **CSA/HCA Compression**: DeepSeek NSA — learnable MLP compression within time-steps.
- **Fast Weight**: Schmidhuber 1991 — persistent associative memory layer.
- **Primitive Rendering**: [wonderfulearth/primitive-operation-painter](https://github.com/wonderfulearth/primitive-operation-painter) — GPU image-to-sequence converter with hill-climbing shape fitting.

## License

MIT

---

**Note**: GARGANTUA v1 is a research-oriented framework. The architecture is fully implemented and forward-pass functional, but meaningful generation requires training on domain-specific data.
