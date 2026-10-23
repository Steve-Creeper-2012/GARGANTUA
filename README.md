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

## Token System

All tokens are discrete. The vocabulary is organized into continuous intervals:

### Content Tokens
- **Chinese**: All Unicode CJK basic + A-extension characters (single character, no phrases).
- **English**: 40K common words + proper nouns (lowercase), with `<caps>` / `<allcaps>` modifiers.
- **Symbols**: All full-width/half-width ASCII symbols.
- **Fallback**: 26 lowercase letters `a-z` for OOV English words.
- **Digits**: `0-9`.

### Special Tokens
- **Formatting**: `<caps>`, `<allcaps>`, `<italic>`, `<bold>`, `<under>`, `<mid>`, `<high>` (with matching close tokens).
- **Boundary**: `<think>`, `<answer>`, `<text>`, `<visual>`, `<audio>`, `<model>`, `<user>` (with matching close tokens).
- **Stop**: `<stop>` (ID 27) — generation terminates immediately when sampled.

### Modality-Specific Tokens
- **Visual Input (not in vocab)**: `<x-axis:n>`, `<y-axis:n>`, `<Rn>`, `<Gn>`, `<Bn>`, `<An>` — one token per 32×32 patch, RGB 1024-level, A 1000-level.
- **Audio Input (not in vocab)**: `<wave:hex>` — 533 samples compressed to one token (~30 tok/s).
- **Drawing Output (in vocab)**: `<x-axis>`, `<y-axis>`, `<width>`, `<length>`, `<rot>` — 9 parameters per primitive (rectangle/ellipse/triangle).
- **Audio Output (in vocab)**: MIDI discrete tokens.

### Time-Step Format (.gts)

GARGANTUA uses a custom token stream format where `{}` denotes a time-step:

```
{<user>}{<visual>}{<x-axis:0,y-axis:0,R1015,G0,B0,A999>,<wave:00000000>}
{<x-axis:3,y-axis:5,R105,G43,B50,A99>,<wave:00110000>}{<visual/>}{<user/>}
```

- Boundary tokens (`<visual>`, `<visual/>`, etc.) occupy their own time-steps.
- Within a time-step, multiple tokens are comma-separated, no spaces.
- Video frames: each frame is one time-step containing all its patch tokens + corresponding audio wave tokens.
- Text: one token per time-step.

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
- **Output**: MIDI discrete tokens (not raw waveform).

## Image-to-Primitives (Drawing Output)

Based on [wonderfulearth/primitive-operation-painter](https://github.com/wonderfulearth/primitive-operation-painter)'s fast_shape_render algorithm:

- **CPU-adapted greedy fitting**: Background average color → residual block fitting with rectangle/ellipse primitives.
- **Two quality modes**:
  - `fast`: 20 candidates/step, no hill-climbing (~2–5s for 100 steps).
  - `slow`: 50 candidates/step + short hill-climbing (~30–60s for 100 steps).
- **Parameters per primitive**: `(x, y, shape, width, length, rot, R, G, B)` — 9 tokens.
- **CSA Compression**: After a sequence of primitives is output, the 9 tokens per primitive are compressed 10:1 into the KV cache.

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
- **Model Management**: Create models with configurable encoder/decoder block counts.
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

Long-running work (inference, model init, image→primitives) returns `202`
with a `task_id`; poll `GET /api/task/<task_id>` until `status == "done"`.
Numeric parameters are never silently defaulted: an explicit `0` is validated
as `0` rather than being replaced by the default value.

## Project Structure

```
.
├── spec.py              # Frozen architecture contract (single source of truth)
├── common.py            # Shared checkpoint I/O + validation (config.json contract)
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
│   ├── text_codec.py    # Text tokenization
│   ├── audio_codec.py   # Audio → wave tokens
│   ├── image_codec.py   # Image/Video → patch tokens
│   ├── draw_codec.py    # Primitive sequence ↔ drawing tokens + PNG/SVG render
│   ├── midi_codec.py    # MIDI ↔ MIDI tokens
│   ├── image_to_primitives.py  # Image → primitive sequence (fast_shape_render)
│   ├── build_vocab.py   # vocab.json generator (layout from spec.py)
│   └── vocab.py         # Vocab view with region-specific maps
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
    └── test_fixes.py    # Regression tests for the 2026-10-20 audit
```

`server.py` used to be one 986-line file holding the HTTP layer, the model
manager, the task pool, the ffmpeg plumbing and every API handler. It is now
366 lines of routing plus five focused modules (`server_models`,
`server_media`, `server_codecs`, `server_train`, `server_util`); `App`
composes the two API mixins. All previous `from server import X` paths still
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
