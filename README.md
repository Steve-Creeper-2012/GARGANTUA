# GARGANTUA v1

> A multimodal early-fusion autoregressive model for world perception and reliable understanding.

GARGANTUA is a next-generation AI framework built from the ground up for **multimodal understanding and generation**. It unifies text, image, video, audio and primitive drawings within a single transformer architecture, sharing the same `d_model` and KV stream while maintaining modality-specific embedding projections.

This README describes the code as it actually is (verified 2026-11-12). Every architecture constant has one owner — `server/spec.py` — and every runtime path has one owner — `server/paths.py`. The UI reads IDs / layouts from `/api/agent/spec`, `/api/status` and `/api/train/dataset/spec` instead of copying constants.

## Documentation map

| Document | What it owns |
| --- | --- |
| `README.md` | Architecture, formats, CLI/API surface, directory layout (this file) |
| `docs/README.md` | Index of every document and the rule for where new docs go |
| `docs/VOCAB_EXPANSION_EVAL.md` | **Historical** vocabulary-expansion evaluation (superseded by base1.3.0; do not implement from it) |
| `service/web/plugins/README.md` | Infer display-plugin protocol (8 sections) |
| `service/web/plugins/AGENT_TOKENS.md` | Full contract-head / slot / core / typst token contract |
| `service/web/plugins/TYPST_TOKENS.md` | Generated 114-construct / 228-token Typst table (do not hand-edit IDs) |
| `service/prims/README.md` | GPU vs CPU primitive paths and their environment variables |
| `service/README.md` | One-folder-per-service convention, including `output/` artifacts |
| `environment/README.md` | Zero-system-dependency contract: one dependency = one folder + `pack.json` |
| `start/requirements/README.md` | `must.txt` / `necessary.txt` and the single install entry point |
| `experiments/README.md` | Reserved folder for future comparison/test data (intentionally empty) |

## Architecture Overview

### Core Design

- **Encoder-Decoder with a shared KV stream**: the encoder processes user input bidirectionally; the decoder generates autoregressively. Both write into / read from the same `SharedKV` — there is **no cross-attention and no separate prefill pass**.
- **Early Fusion**: all modalities are projected into the same semantic width (`d_model`, default 1024) before entering the transformer, but each modality has its own projection.
- **Shared KV, two views**: the stream is split into a full-resolution recent region and an older compressed region carrying two *independent* compressions of the same content (CSA retrieval view, HCA global-summary view).
- **Time step is atomic**: compression, sliding-window eviction and causality never split a step; a step is never empty.

### Block Structure

```
KDA → FFN → KDA → FFN → KDA → FFN → MLA-CSA → FFN → MLA-HCA → FFN → AttnRes
```

`spec.BLOCK_ORDER` is the single source of truth; `Block.__init__` asserts that this order matches the hand-written `forward`. Since 2026-11-04 each recurrence/attention sub-layer is immediately followed by **its own** FFN: five independent `GLUFFN` instances and ten `RMSNorm`s per block, not one FFN reused five times. Checkpoints created before that date have a different `state_dict` layout (`ffn_up`/`ffn_down` vs `ffns.i.*`) and must be re-initialized.

- **KDA** (Kimi-style Delta Attention): gated linear recurrence with delta-state updates; three KDA layers carry the recurrent/positional path.
- **MLA** (DeepSeek-V2): low-rank joint KV compression, NoPE. **MLA-CSA** reads a 4:1 compressed segment plus the whole full-KV segment; **MLA-HCA** reads a 32:1 compressed segment plus the whole full-KV segment. When nothing is compressed yet, both are plain MLA.
- **FFN**: GLU variant, `FFN_HIDDEN = 4096`. Default 4+6 blocks ≈ **1.03B params** (~4.1 GB fp32); use `d_model`, block counts or shared weights to scale down.
- **AttnRes** ([MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals), arXiv 2603.15031): before every sub-layer the block aggregates token embedding, completed block representations and the current partial sum with a softmax pseudo-query; a zero-initialized scalar gate at the block end mixes the block input back in.

**Configurable width**: `d_model` is chosen at creation (`spec.check_d_model`): any multiple of `HEAD_DIM = 128`, **≥ 128 with no upper bound**, default 1024. It only changes semantic width; head count/head dim, MLA ranks and `FFN_HIDDEN` stay frozen. `config.json` is the only input to `common.build_model()`.

**Encoder/decoder weight sharing** (optional): `share_enc_dec=true` ties `enc_blocks[i]`/`dec_blocks[i]` `nn.Parameter` objects and both final norms, so both block counts must be equal. Only *parameters* are shared — the two `Block` instances keep their bidirectional/causal behaviour. Tied weights are stored once on the encoder side (safetensors forbids shared storage) and mirrored on load by `common.strip/expand_tied_dec_keys`. Old model folders without the field load as separate-weight models.

### Fast Weight Memory

A global `FastWeightBank` is structurally one KDA layer (same q/k/v projections, causal short conv, L2 norm, channel-wise forget gate, output gate, RMSNorm); the only difference is that its state `(S, conv_cache)` persists across calls. `S` is `[B,H,dk,dv]`, **not** `[D,D]`. It is invoked once after the first block in both encoder and decoder, reading with the old state and then writing every token. Based on Schmidhuber's 1991 fast-weight paper.

**Sample boundaries are mandatory**: all training / inference / CLIP entry points call `Gargantua.reset_memory()`. Skipping it leaks state across samples (measured logits gap 20.19). Inference session state lives in an external `state` dict so multi-turn chat can carry memory intentionally.

### MoMW — optional three-stage memory valve

`MoMW` (Mixture-of-Memory-Weights, `server/momw.py`) is **off by default** and sits next to the fast-weight bank after the first block. Every pool layer shares the same `KDAAttention` reference and keeps only its own recurrence state `S_i`, so enabling it adds **zero parameters and zero persistent buffers**: `state_dict()` is byte-identical with the valve on or off.

1. **Cold start (anchor seeding)**: each layer is bound to a class anchor `P_k`, computed lazily on the first forward as the mean embedding of representative tokens from `wte.weight`; `set_anchor_provider()` can inject real text-embedding means.
2. **Drift from anchor to state feature**: the principal direction of `S_hᵀS_h` is tracked online with **Oja's rule** (one fixed-cost update per write) and projected back through the KDA `w_o` to form `E_i`; `γ_i = 1 − exp(−writes/k)` moves the score from `cos(H_t, P_i)` toward `cos(H_t, E_i)`.
3. **Expansion**: overload is judged on *relative* quantities — `‖S_i‖` above `saturation_ratio ×` the same-type median, or growing faster than `growth_ratio ×` its own previous norm. A **zero-state** layer is appended (states are never cloned); new layers get an exponentially decaying bonus `B₀·e^(−age/M)` and same-type layers are softly load-balanced.

The admission threshold is **data-driven** (`threshold_mode="quantile"`, default p=0.9): a sliding window of recent scores, with a fixed fallback while the window is cold. `MoMW.calibrate(batches)` returns suggested percentiles but never writes them back to config. Nothing activates below the threshold — the block then returns `H_1` unchanged and un-recalled layers are never forwarded. The whole valve runs under `torch.no_grad()`, is isolated per session, supports a **silent mode** (write but do not inject `ΔH`), and applies a cost-aware gate `exp(−c·E_fw)` so the two memory paths do not both spike the residual norm.

Pool snapshots go to `models/NAME/momw_state/{layout,tensors,meta}` (separate from `weights/` and `config.json`); a corrupt snapshot just means starting from zero. `async_write=true` moves writes to a single-writer thread; `flush()` joins before persisting/unloading. The inference panel shows per-layer γ / writes / ‖S‖ / Oja steps / activation plus the live threshold and gate; `/api/momw/control` and the `momw control` CLI command offer the same `clear` / `silent` operations. `spec.momw_default_config()` owns the 25 config keys.

### SOMN — optional self-organising memory network

`SOMN` (`server/somn.py`, v4-somn, merged 2026-11-12) is a second, **independent** fast-weight system that can be attached alongside the fast-weight bank and MoMW after the first block. It is **off by default**: `attach_somn()` must be called explicitly; without it `Gargantua` returns `None`, the numerical output is unchanged and `state_dict()` gains no keys. Layers and their `S` matrices live in a plain dict `_rt['somn']` (callable entry at `['module']`), never in the module registration tree, so the "state in ordinary dicts" invariant holds.

- `IndependentFWLayer`: an `S = [D,D]` delta-rule memory (read `H@S` with the old state, then update). Stability requires `eta·λ_max(C) < 2`; the implementation uses **`eta = 0.01·64/D`** (equals the spec value at D=64, scales down for larger widths) with the split threshold linked as `5.0·eta/0.01`.
- `EnergyElbowRouter`: scores a layer by **cosine reconstruction match** `cos(H, H@S).mean()` (the old `‖H@S‖` norm could not distinguish structured from random inputs). `min_score=0.5` admits a layer, empty layers score 0.5, cold start forces a write. `min_energy` remains an alias.
- **Alternating SVD split**: on expansion the parent takes singular values at even indices and the child at odd indices, so `S_main + S_sec = S` stays lossless while both children get comparable energy (measured energy ratio 1.0241 vs 7.6×10⁶ for the old top/bottom split).
- **Warm-up birth privilege**: new layers are unconditionally active for N steps, receive an **additive** bonus decaying linearly 1→0, and are exempt from re-splitting. This removes the "zombie layer" failure where a newborn layer never gets a write.

`reset_memory()` also resets SOMN unless `persist_across_samples=True` is set. `service/test/test_somn.py` (30 cases) is registered in `SUITES`; `python server/somn.py --demo` prints layer growth over 100 samples. SOMN is a code-level opt-in feature — there is no UI toggle and no HTTP endpoint yet; document any future UI/API addition here.

### KV Cache Design

```
Full KV (uncompressed) | Compressed KV
                       ├── CSA view  (4:1, retrieval)      → MLA-CSA
                       └── HCA view  (32:1, global summary) → MLA-HCA
```

- **Full KV**: the most recent `ctx_full` tokens, uncompressed.
- **Compressed KV**: everything older, kept as two independent compressions of the same content. Both views read the compressed segment **plus the entire full segment**.
- **Compression rule**: a time step is atomic; a group never straddles a step boundary and an oversized step may be compressed internally into several tokens that all keep that step's id. Each group targets the ratio (4 or 32); when whole steps cannot hit it exactly, the grouping with the smallest error wins.
- **Eviction**: whole steps are discarded together. Evicted KV *vectors* are dropped, but their original token ids are archived in `SharedKV.archive` — context overflow loses the KV view, never the raw ids.
- **Causality**: MLA's causal mask is step-aware **and token-aware** — within one step a token only sees earlier tokens of that step (`within_step_pos`), and compressed aggregate entries (`pos = -1`) are hidden from same-step queries. A causal MLA without `step_ids`/`pos` raises instead of silently leaking the future. Encoder blocks stay bidirectional.
- Drawing primitives are packed 9:1 in the KV after a primitive's 9 tokens are written (`DRAW_KV_COMPRESS = 9`).

### Context Length (model-creation-time option)

| Region | Meaning | Default |
|---|---|---|
| **Full** (`ctx_full`) | Uncompressed recent tokens | 1024 |
| **Compressed** (`ctx_compress`) | Capacity, counted in **original** tokens, of everything older | 3072 |

Total = `ctx_full + ctx_compress` (default 4096). `csa_ratio` (default 4) / `hca_ratio` (default 32, must be ≥ `csa_ratio`) are architectural — the compressor's position embedding is `[ratio, d_model]`, so changing them changes the checkpoint layout. There is no separate CSA window: CSA's retained region **is** the full-KV region (the old `csa_local_window` knob was removed).

```bash
python service/init/init.py --name alpha --ctx-full 2048 --ctx-compress 8192 \
    --csa-ratio 8 --hca-ratio 64                      # total 10240
```

The same four values are accepted by the Web UI's creation form and `POST /api/model/init` (omitted = spec default, illegal = 400). `GET /api/status` / `GET /api/system` report the active model's real configuration (`ctx` + `kv_note`).

## Token System

### token-6 stream

`.tokens` is a **token-6** bitstream (`service/tokens/tok6.py`, 2026-11-05): 6 bits per symbol over 29 single characters

```
0 1 2 3 4 5 6 7 8 9 { } < > | : w t v A B C D E F x y [ ]
```

plus 6 atomic boundary tokens (`[model] [user] [model/] [user/] [sys] [sys/]`, codes 29–34). Codes 35–62 are reserved, **63 terminates** the stream, which is then zero-padded to a byte boundary. IDs inside the stream are uppercase, self-delimiting hexadecimal. `tok5.py` is only an import alias for token-6; token-5 files are **not readable**, and `tok32.py` remains solely as a reader for historical uint32/hex files. `spec.py` is the single source of truth for every code and ID.

| Element | Meaning |
| --- | --- |
| `{t:00001}` | text token — visible, has a vector, enters the loss |
| `{w:<hex>}` | audio token — 533 samples × 16 bit = 2132 hex chars |
| `{v:<hex>x:10y:24}` | visual patch — 32×32 px, RGBA 10 bit each = 10240 hex chars |
| `[00005]` / `[user]` | extended token / atomic boundary — has an embedding, not displayed as content, **still enters** the main AR loss |
| `<00000>` / `<caps>` | `core` mechanism marker — `<>` shaped, no closing form, enters the loss, has KV |
| `<0000C>` / `<s0>` / `<DOC>` | slot / Typst tool element — enters the loss, has KV |
| `{ } \| [ ] < >` | structure characters — **not** tokens (no position, no loss, no embedding) |

Multiple elements in one step are separated by `|`; `[...]` and `<...>` live at stream level, outside `{}`. The human-readable form is `{text: 00001}`, `{wave: <hex>}`, `{visual: <hex> x: 10 y: 24}`, `[user]`, `<caps>`, `<s0>`; the editor accepts both compact and display forms. Typst names take precedence over `<HEX>` parsing, otherwise `<B>` (bold) would be misread as ID 0xB.

**Wrapping semantics / loss**: `[]` is for the six `boundary` segment markers; `<>` is for `core`, `slot` and `typst` tool elements. Both shapes **enter the main AR loss and have KV + fast weight**. Only media positions (no vocabulary ID) are given target `-100` and excluded from the loss. `spec.segment_type_of()` defines four segment types — `TYPE_USER=0`, `TYPE_MODEL=1`, `TYPE_NONE=2`, `TYPE_SYS=3`.

### base1.3.0 vocabulary

A vocabulary is a **folder** `<name>/{vocab.vfc, config.json}`. `.vfc` holds one token per line (`内容:ID`, invisible tokens wrapped in `<>`, IDs uppercase hexadecimal), strictly continuous `0..N-1`; deletion compacts the file. `config.json` declares `vocab_size`, `head_size`, `slot_count` and the ordered `regions`. The versioned base is `vocabs/base1.3.0/` (`vocab_version: "1.3.0"`, `token_encoding: "token-6"`); `service/tokens/{vocab.vfc, config.json}` is the active global runtime copy, and each model gets its own `vocab/` copy at creation.

The contract head has **six segments**:

```
core 3 + boundary 6 + system 3 + slot N + typst 228 + draw 9605 = 9845 + N
```

- `core` = `<caps>`(0) `<stop>`(**1**) `<draw:end>`(2) — earliest mechanism markers, single-handled, not `[]` boundaries or tool calls.
- `boundary` = `[user] [user/] [model] [model/] [sys] [sys/]` (IDs 3–8); `[sys]…[sys/]` is the `TYPE_SYS` system segment used for tool results / system prompts.
- `system` = `{ } |` (IDs 9–11), time-step structure.
- `slot` = `<s0>…<sN-1>` (IDs `12 … 11+N`), numbered placeholders with **no preset semantics**. `N` comes from that vocabulary's `config.json`, not from `spec.py`; base has **zero** reserved slots. Since 1.2.0 every predefined agent tool / formatting token is gone (`TOOL_SPECS`, `TOOL_EXECUTABLE` are empty, `SLOT_DEFS = {}`) and tool execution is disabled in this tooling/test bench. Define names, kinds and open/close pairs yourself; `GET /api/agent/spec` is the only source for IDs.
- `typst` = 114 Typst syntax constructs × explicit open/close pairs = 228 `<>` tools (`<DOC>`/`</DOC>` … `<ATTR>`/`</ATTR>`), `TYPE_NONE`, invisible, in loss, with KV. `TYPST_BEGIN = 12+N`, `TYPST_END = 240+N` — it **must** follow `slot`, because `vfc.infer_slot_count()` derives N by counting from ID 12 to the first Typst token.
- `draw` = 9605 output-side tokens; `DRAW_BEGIN = 240 + N`.

For base (N=0): `HEAD_SIZE = 9845`, `DRAW_BEGIN = 240`, visible IDs start at 9845 and **`vocab_size = 80262`**. Ten visible regions follow the head: `digit 10 / whitespace 6 / symbol 7263 / emoji 1517 / cjk 27584 / english 30000 / extra 0 / fallback 26 / code 474 / unicode_ext 3537`. `code` is 474 cross-language code tokens appended after `fallback`; `unicode_ext` adds 3537 characters from 31 Unicode blocks (Latin extended, Cyrillic, Greek, kana, math symbols, …). `MAX_VOCAB_SIZE = 100000` is only a defensive upper bound for IDs in the bitstream, not the vocabulary length.

To reserve slots, derive a new vocabulary (base itself is never edited):

```bash
python service/tokens/gen_vocab.py --name mymodel --slots 128
# vocabs/mymodel/{vocab.vfc, config.json}, slot_count=128
# --activate makes it the global runtime layout (restart the server)
```

`spec.head_layout(n)` is the pure layout function; `vfc.infer_slot_count()` reads N back from any `.vfc`, so a derived vocabulary loads correctly even when the global layout differs. Full contracts: `service/web/plugins/AGENT_TOKENS.md` and `TYPST_TOKENS.md`.

**Version history / compatibility**: 1.0.0 introduced token-6; 1.1.0 appended `code`; 1.2.0 rewrote the head; 1.3.0 inserted `typst` and `unicode_ext`. IDs shifted wholesale in 1.2.0 **and again in 1.3.0** (draw and the whole visible area +228) — old `.tokens` files, weights and archives are not readable, and no compatibility layer is provided. Changing the head requires synchronizing `vfc.head_entries()`, `build_vocab.build_entries()`, the `.vfc`/`config.json` copies, the "About" page SVG, `minify.py` rebuild and tests.

### Regions, add-token and codecs

Surface strings are unique across the whole vocabulary (single-letter words `a`/`i` are not word tokens — fallback letters cover them), so reverse lookup is unambiguous. Adding a token is **region-scoped**: `spec.REGION_ADD_POLICY` / `check_add_region()` are the policy source — `core`/`boundary`/`system`/`typst`/`draw` are locked, `slot` accepts `<angle>` names, the ten visible regions accept text.

- `POST /api/vocab/token` with `{name, action: add|remove|query, token, region?}` is the canonical endpoint; `POST /api/vocab/add` is a thin compatibility alias that forces `action=add` (no duplicated logic).
- `Vocab.add_token(text, category, region=None)` / `remove_token()` keep `config.json` regions, `head_size` and `slot_count` in sync. Only `slot` tokens can be removed.
- Other vocabulary endpoints: `GET /api/vocab`, `/api/vocab/list`, `/api/vocab/regions`, `/api/vocab/source`; `POST /api/vocab/source`, `/fork`, `/copy`, `/create`, `/delete`. The **base** vocabulary is protected — fork before editing.
- **enctok** (`python service/tokens/enctok.py "text"`): `[A-Za-z]+` word segments hit the 30K word list (`<caps>` + word for `Hello`; ALL-CAPS/mixed case is spelled letter-by-letter, lossless); code tokens are matched verbatim first and by longest-match; other chars look up regular tokens then fallback letters; unmapped chars are skipped and reported. Output is hex text or a token-6 `.tokens` file. `--vocab` accepts a `.vfc` file, a vocabulary folder or a model folder (reads `vocab/`).
- **dectok** (`python service/tokens/dectok.py stream.tokens`): renders token-6 as `{text: 00001} …`; `--text` renders surface text.
- **gts** (`service/tokens/gts.py`): legacy tok-32-era editor parser, kept read-only; the editor now speaks token-6 directly via `POST /api/tokens/export`.
- **tokenset** (`service/tokens/tokenset.py`): the only training reader. `read_doc()` returns `pos_ids` (text ID / `None` for media), `step_ids`, assembled `steps` and a `multimodal` flag; `read_stream()` is the text-only convenience entry and rejects media elements. `MAX_DOC_POSITIONS = 32768` guards memory. `.tokenvector` (UTF-8) archives audio/visual vectors with `dim=`/`precision=` headers for modalities that have no discrete IDs.

### Time-Step Format (`{ } |`)

```
{[user]}{h|e|l|l|o}{[user/]}
```

A time step is the atomic unit for KDA recursion and logical parallelism. `{ }` wraps a step, `|` separates tokens inside it. Text: one token per step (compact form). Image: one step holding all patch tokens. Audio: 533 samples = one token = one step. Video: one frame = one step holding all patches + that frame's audio token(s). Inside a step all tokens share `step_id`; attention is still **token-level causal** (see KV Cache Design), not bidirectional.

## Embedding Strategy

Embedding is **not unified** — each modality has its own projection to `d_model`:

| Modality | Projection | Position info |
| --- | --- | --- |
| Text | `wte` lookup | none |
| Visual | `visual_proj` (4096 → `d_model`) | `visual_coord_x` + `visual_coord_y` (2D absolute) |
| Audio | `audio_proj` (533 → `d_model`) | none |
| Drawing | `wte` (in-vocabulary IDs) | none |

**Segment marker = type embedding**: `type_emb` is a learned table with one entry per segment kind (`user / model / none / sys`, `NUM_TYPES = 4`), added to **every token inside that segment** — there is no start/end distinction. It plays the learned-additive-vector role of a position embedding; KDA learns relative order itself. The `[]` boundary markers are ordinary tokens in the loss with KV (see Token System), not "system-level markers excluded from KV".

## Visual Processing

- Input is padded to multiples of 32 and split into 32×32 RGBA patches.
- Each patch = 1024 pixels × 4 channels × 10 bit, flattened; `visual_proj` maps it to `d_model`.
- `(gx, gy)` grid coordinates are added via learned position embeddings (not tokenized).
- **Video**: fixed 30 fps. Frame 1 = full input; frames 2–30 = only patches with pixel-level changes; frame 31 (per second) = full input; an all-changed frame = full input + counter reset; last frame = always full. Audio is aligned at 533 samples/frame (16 kHz / 30 fps).

## Audio Processing

- Input: 16 kHz mono, any format (mp3/flac/ogg/wav via the bundled ffmpeg).
- Tokenization: 533 samples → 1 token (~30 tok/s).
- Embedding: 533-sample vector → `audio_proj` → `d_model`.
- Output: MIDI discrete tokens were removed on 2026-10-24 (`midi_codec` and `/api/codec/midi` are gone; recoverable from git history).

## Primitive Pipeline (image → primitive tokens)

Adopted from [wonderfulearth/primitive-operation-painter](https://github.com/wonderfulearth/primitive-operation-painter); the converter under `service/prims/fast_shape_render/` is vendored verbatim (Rust + wgpu: Metal / DX12 / Vulkan). The rest of the upstream repo (~11 MB of model/training code) was removed on 2026-10-31 and is recoverable from git history. Driver + CPU fallback are Python + numpy.

- **GPU converter**: fits rectangle/ellipse operations in 275 search steps and writes CSV (`image_name,cx,cy,w,h,shape_type,theta,r,g,b`; first row = background). Driven by `service/prims/image_to_primitives.py`.
- **CPU fallback**: the same greedy algorithm in pure numpy, identical schema, any canvas up to 1920×1080. `quality=fast` (20 candidates/step, no hill-climbing) or `slow` (50 candidates + short hill-climbing); `engine=auto|gpu|cpu`.
- **Tokens (`geometrize_256_v1`)**: a canvas declaration (`<x> digits <x/> <y> digits <y/>`) opens the sequence, then N × 9 tokens in field order `x, y, theta, w, h, shape, r, g, b`. `x/y/w/h` are 1 token = 1 px, `theta` 3 bins/degree, `r/g/b` **10 bit (0..1023)** internally. CSV / PNG / SVG are 8-bit boundaries converted with `color_to8` / `color_to10`. `/api/status.draw_layout` (built solely by `server.draw_layout_payload()`) is the front-end's layout source.
- Rendering endpoints (`POST /api/codec/draw`, `/api/codec/image2prims`) return `202` with a `task_id`; poll `GET /api/task/<task_id>`.

### Startup dependencies

```sh
sh service/prims/setup.sh
```

installs the Rust toolchain into `environment/rust/` (redirected `RUSTUP_HOME`/`CARGO_HOME`, `--no-modify-path`) and builds the GPU converter (`cargo build --release`, 5–15 min). Mirrors are selected by speed test; macOS needs Xcode Command Line Tools, Windows has no POSIX `setup.sh` and simply uses the CPU fallback (or a prebuilt binary via `GARGANTUA_SHAPE_RENDERER`). Without a built binary the pipeline automatically falls back to CPU. Converter env vars are documented in `service/prims/fast_shape_render/README.md`.

## Training

### Optimizer Strategy

`clip_data.OPTIMIZERS` is the only source for the three modes, shared by the UI, the queue filename, the CLI and the trainer:

| Mode | Behaviour |
| --- | --- |
| `muon` (default) | hidden-layer 2D matrices → Muon; embeddings / head / norms / 1D params → AdamW |
| `adamw` | all parameters → AdamW |
| `allmuon` | **all** 2D matrices → Muon; 1D params still AdamW (Muon orthogonalizes matrices, not vectors) |

Implemented once in `muon.build_optimizers()`; `service/train/train.py`, `auto_train.py`, `server_train.py` and the CLI all validate against the same tuple.

### Data formats

**Language-model training** accepts only a token-6 **`.tokens` file** with explicit `{ }` time-step structure — the Token editor's "Export .tokens" produces exactly this, and JSON/JSONL uploads are rejected with 400. `tokenset` appends a `<stop>` step at the end of each document; files without valid structure are rejected before training. A file may mix `{t:}`, `{w:}` and `{v:}`: text-only docs go through `make_batch` + `forward_train`, multimodal docs through `make_batch_emb` + `forward_train_embeddings` (media positions get target `-100`, so media is context only). Both paths share `service/train/train_step.py`.

**Contrastive training = CLIP dataset folder v1** (`service/train/clip_folder.py` is the single source; `GET /api/train/dataset/spec` renders the spec for the UI):

```
<dataset>/
├── dataset.json          # format=gargantua-clip-folder, version=1
└── pairs/
    ├── 001/{a.tokens, b.tokens}
    ├── 002/{a.tokens, b.tokens}
    └── ...                # at least 2 pairs
```

The top level allows only `dataset.json` and `pairs/`; each pair directory holds exactly `a.tokens` and `b.tokens`, whose internal modalities are arbitrary (image×image, video×video, text×image, …). `dataset.json` may optionally set `name`, `pairs` (order), `batch_size` (2..64) and `temperature` (0,100]. `clip.clip_loss` is standard symmetric InfoNCE with negatives drawn from the other pairs in the batch — **no** synthetic `(A,B)+(B,A)` negatives. Four pairs start at loss ≈ ln(4) = 1.386386, a useful regression anchor. The legacy two-file `mode=pair` queue format is read-only compatibility; the new UI never creates it.

**Training granularity**: one file = a fixed number of `steps`; each step samples one random window, computes one average loss over the whole window and takes one optimizer step. Token-level sequential traversal was removed. Loss records carry `step/loss/ts/file` (plus `mode/n_pairs` for CLIP); the loss curve has one tab per file with no vertical token-segment lines.

### Queue and task metadata

`data/queue/` holds pending `.tokens` files, `data/done/` holds trained ones (`data/failed/` on error), and `data/task_meta/<queue-name>.json` sidecars store target / optimizer / block_size / steps / CLIP batch size / temperature. Queue filenames encode mode + optimizer (`clip_data.parse_queue_name`). `auto_train._task_scope()` applies each task's parameters temporarily and restores them afterwards. Editing an unfinished task from the UI (`POST /api/train/task/update`) rewrites its sidecar and, when the optimizer changes, renames the queue file.

Training endpoints: `/api/train/{enqueue,start,stop,delete,status,queue,events,loss,task/update,dataset/spec}`. `queue` reports `pending / done / failed / running`; `delete` takes a `where`. Every one of them has a named terminal command in `service/cli/api_map.py` (a static test asserts full coverage).

### Running training

```bash
python service/train/train.py --model models/my_model --data data/ \
    --steps 200 --block-size 32 --optimizer muon
cp stream.tokens data/queue/                       # queue daemon path
python service/train/auto_train.py --model models/my_model
./gargantua queue add --file stream.tokens --mode lm
```

The Web UI's training page has one block for creating/editing a task, the pending queue above the completed queue on the right, and a three-view top bar (tasks / logs / loss). CLIP upload is a single folder (`webkitdirectory` + recursive drag-and-drop); LM upload accepts multiple `.tokens` files, one task each.

## Inference

```bash
python service/infer/infer.py --model models/my_model --prompt "Hello"
```

The decoder stops as soon as `<stop>` (**ID 1**) is sampled. `POST /api/infer` accepts `media: [{kind, b64, filename}]` with `kind ∈ image | audio | video` (up to 4 attachments, 64 MB total). Media is decoded into patches / 533-sample tokens, placed before the text prompt, projected into `d_model` and encoded into the KV cache; generation stays textual, so at least one text token is needed as decode seed. Streaming goes through `GET /api/infer/stream`, which returns each new token with `char` / `tool` metadata.

### Token-id archive

```
logs/infer_archive/<YYYYmmdd_HHMMSS>_<id>.tokens   # token-6，每步一个 id
logs/infer_archive/<...>.json                      # model / counts / ctx_total / n_evicted
```

The `.tokens` file receives the prompt at start and each generated token as it is sampled (flushed per token), so a crash or KV eviction can never lose ids. The sidecar records `n_tokens`, `n_prompt_tokens`, `n_generated_tokens`, `ctx_total`, `n_evicted`. `--archive-dir` overrides the location; the server passes `<log dir>/infer_archive` and returns the path as `archive`. Archiving failure degrades to "no archive" and never blocks inference.

## Web Interface

A built-in single-page UI (`service/web/index_source.html` → minified `index.html`) provides:

- **Model management**: create / load / delete models with block counts, context length, `d_model`, weight-sharing and MoMW options; per-model parameter estimates.
- **Training queue**: `.tokens` upload, CLIP folder upload, task editing, live events and per-file loss curves.
- **Inference chat**: the whole middle block (including the input box) is rendered by a **pluggable display plugin**, because token display differs per vocabulary. Default is **Infer-Plug v1.0.0** (contenteditable with inline file chips, canvas SVG preview with a draggable timeline, streaming text, tool badges); the built-in `tokens` plugin keeps the converter-style `[]`/`<>` multi-turn view. Selector / import entry: Developer options → 显示插件. Protocol: `service/web/plugins/README.md`.
- **Developer options**: context usage, KV cache, fast-weight heatmap + per-step write magnitude, MoMW panel (per-layer type / γ / writes / ‖S‖ / activation, clear pool, silent toggle), plugin selector, sampling parameters.
- **Codec visualizers**: text / audio / image / video → token, image → primitive sequence, token editor with spec-aligned palette and **Export .tokens**, and a vocabulary page (region occupancy, presets, search, region-scoped add-token).
- **Environment & dependencies**: read-only cards plus a resource monitor; installs/updates are terminal-only (see `start/requirements/README.md`).
- **Built-in terminal**: a PTY shell locked to the project root (POSIX only; on Windows the endpoint returns a clear message and the external terminal is used).

### Launch

```bash
./启动.sh                        # macOS / Linux 终端
./启动.command                   # macOS: 双击启动（Finder）
启动.bat                         # Windows: 双击启动（调用 start\terminal\start.bat）
./gargantua --help               # 终端接口（POSIX）；Windows 用 gargantua.bat
```

The terminal scripts only prepare the isolated environment and **hand control to the server, then exit**. `launcher/start-server.sh` checks the bundled Python + venv + pywebview, calls `launcher/setup-environment.sh` if anything is missing, then spawns the server into a detached background process (`nohup` + a one-shot Python `os.setsid()` + `disown`, stdin `/dev/null`, stdout appended to `logs/server.log`, PID in `logs/server.pid`), waits for `/api/status` and prints the address.

- Closing the terminal or an SSH session does not kill the server; closing/crashing the *window* also leaves the server running.
- Double-clicking twice does not start a second server; if the server is alive but the window is gone, the launcher reopens just the window.
- Stop: `./启动.sh --stop` (falls back to finding the listener by port if the pidfile is gone). Debug/foreground: `./启动.sh --foreground`.
- Exiting the server takes the window and all service subprocesses with it (no orphans pointing at a dead port).

First-run setup: download standalone CPython 3.12 into `environment/python/` with automatic mirror speed-probing (never reuses system Python or a system package manager), create `environment/venv/` via `start/setup_env.py`, install `must.txt`, and write a dependency-hash + OS/arch fingerprint to `environment/venv/.gargantua_setup_stamp`. Set `GARGANTUA_FULL_SETUP=1` to also install `necessary.txt` in one pass. All downloads, pip caches and temp files stay inside the project (`environment/_cache/`, `data/tmp/`).

| Flag | Effect |
| --- | --- |
| `--port N` | HTTP port (default 8712) |
| `--host H` | Bind address (default 127.0.0.1) |
| `--no-window` | Server only — SSH / CI / headless |
| `--idle-timeout S` | Service subprocess idle self-exit (default 300s) |
| `--model-dir` / `--data-dir` / `--log-file` | Override default paths |
| `--foreground` / `--stop` (launcher) | Foreground exec / stop the background server |

Environment variables: `GARGANTUA_PY_RELEASE` / `GARGANTUA_PY_VERSION` (pin python-build-standalone), `GARGANTUA_PY_BASE_URL` (mirror), `GARGANTUA_FULL_SETUP=1`, `PIP_INDEX_URL`, `GARGANTUA_SHAPE_RENDERER`.

### The window is not a browser

The UI opens in a standalone desktop window via [pywebview](https://pywebview.flowrl.com/) (macOS `WKWebView` / Windows `WebView2` / Linux `WebKitGTK`) — the OS web component, not a second browser engine. The window dependency lives in the lightweight `must.txt`; without it the server still starts and prints a URL you can open manually.

```
启动脚本 ──→ 预检 + 交接（nohup，随后自己退出）
server.py ──┬── infer/train/init 服务子进程   按需拉起，空闲自行退出
            └── window.py                    UI 窗口（崩了不影响 server）
```

### Services start on demand

Inference / training / model-init are heavy (weights can be several GB), so they run as **independent subprocesses**; the server itself only routes HTTP and does bookkeeping. Protocol = JSON-lines over stdin/stdout (`{"id","op",…}` → `{"id","ok",result|error}`); service stdout carries JSON lines only, human logs go to stderr; exit conditions are `shutdown` / stdin EOF / idle timeout. The infer service reloads only when `model_dir` changes (idle 300s); the train service restores the loss curve from `logs/auto_train.jsonl` on start and never idles out while the queue has files; init is a one-shot `run_oneshot` with a trailing `--json` line. PyTorch's CPU allocator does not return memory to the OS within a process, so "on demand" must be process-level.

### HTTP contract

All `/api/*` endpoints share one error convention:

| Status | Meaning |
| --- | --- |
| `400` | Invalid/missing parameter (odd `block_size`, `steps < 1`, `temperature <= 0`, illegal model name, unknown `quality`, malformed JSON) |
| `403` | Illegal source / path traversal |
| `404` | Unknown path or task |
| `409` | State conflict — training running while switching/creating a model, or overwriting an existing model without `overwrite:true` |
| `411` | `Content-Length` required (chunked encoding rejected) |
| `413` | Body larger than 512 MB |

Long-running work (inference, model init, image→primitives, drawing render) returns `202` with a `task_id`; poll `GET /api/task/<task_id>` until `status == "done"`. Numeric parameters are never silently defaulted (an explicit `0` is validated as `0`); booleans, non-numeric strings and negative `Content-Length` are rejected with 400.

The complete endpoint inventory lives in **`service/cli/api_map.py`** (90 distinct paths: 33 GET + 58 POST method/path pairs). Every GUI endpoint has a named `./gargantua …` command; `api <METHOD> <PATH>` is the generic escape hatch. Endpoint families: `/api/status`, `/api/system*`, `/api/env/*`, `/api/models*`, `/api/model/*`, `/api/infer*`, `/api/train/*`, `/api/codec/*`, `/api/tokens/*`, `/api/vocab*`, `/api/agent/*`, `/api/plugin/*`, `/api/prims/*`, `/api/session*`, `/api/term/*`, `/api/task/<id>`, `/api/momw/control`.

## Project Structure

```
.
├── README.md                     # 本文：架构 / 格式 / API / 目录树
├── VERSION                       # 关于页版本徽章的数据源
├── LICENSE                       # GPLv3
├── docs/                         # 文档索引 + 历史评估归档
│   ├── README.md
│   └── VOCAB_EXPANSION_EVAL.md   # 历史评估稿（base1.2.0 时代，未实施）
├── experiments/                  # 预留：后续测试对比数据（暂为空）
│   └── README.md
├── 启动.sh / 启动.command / 启动.bat   # 三平台中文名入口（薄包装）
├── gargantua / gargantua.bat     # 终端 CLI 入口（POSIX / Windows）
├── launcher/                     # POSIX 启动器 + 环境搭建
│   ├── start-server.sh           #   预检 → 后台交接 → 退出
│   ├── start-server.command      #   历史路径兼容壳 → 根目录 启动.command
│   └── setup-environment.sh      #   镜像测速 + 隔离 Python/venv
├── start/                        # 启动链
│   ├── boot.py                   #   sys.path 引导（唯一路径引导）
│   ├── setup_env.py              #   唯一安装器
│   ├── requirements/{must,necessary}.txt
│   ├── requirements/README.md
│   └── terminal/{start,setup}.bat  # Windows 入口
├── environment/                  # 全部依赖与工具链（系统隔离，运行时生成）
│   ├── README.md                 #   契约：一个依赖一个文件夹 + pack.json
│   └── python/, rust/, venv/, tools/, _cache/ …
├── server/                       # 常驻 HTTP 服务 + 冻结架构契约
│   ├── spec.py                   # ★ 所有区间/维度常数的唯一真源
│   ├── paths.py                  # ★ 所有项目内路径的唯一真源
│   ├── model.py                  #   Gargantua / KDA / MLA / SharedKV / Block
│   ├── common.py                 #   检查点 I/O + config.json + build_model
│   ├── muon.py / precision.py / param_estimate.py / reqs.py
│   ├── momw.py / momw_state.py   #   MoMW 三阶段记忆阀门 + 快照
│   ├── somn.py                   #   SOMN 自组织记忆（可选、默认关闭）
│   ├── agent_tools.py            #   纯槽位协议（工具执行禁用）
│   ├── vocab_presets.py          #   vocabs/ 预设增删查改
│   ├── server.py                 #   装配 + 路由 + 启动编排
│   ├── server_*.py               #   env / services / models / media / codecs /
│   │                             #   train / tokens / agent / monitor / prims /
│   │                             #   sessions / term / util  各一个职责
│   └── env_manager.py / env_install.py
├── service/                      # 一个服务一个目录，按需拉起
│   ├── README.md                 #   服务层约定（含 output/ 产物目录）
│   ├── infer/infer.py            #   自回归生成 + id 存档
│   ├── init/init.py              #   建模（CLI + --json 一次性）
│   ├── train/                    #   train.py / auto_train.py / train_step.py /
│   │                             #   clip.py / clip_data.py / clip_folder.py
│   ├── tokens/                   #   tok6.py / vfc.py / vocab.py / build_vocab.py /
│   │                             #   gen_vocab.py / gen_typst_map.py / enctok.py /
│   │                             #   dectok.py / tokenset.py / timestep.py /
│   │                             #   *_codec.py / tok32.py(历史) / tok5.py(shim)
│   │                             #   + 全局 vocab.vfc / config.json / words_en.txt
│   ├── prims/                    #   image_to_primitives.py / setup.sh /
│   │                             #   fast_shape_render/(vendored Rust) / README.md
│   ├── web/                      #   index_source.html / index.html / minify.py /
│   │                             #   app.css / app.js / plugins/(js,css,3 份 md)
│   ├── cli/                      #   gargantua.py / local.py / api_map.py
│   ├── window/window.py          #   pywebview 独立窗口
│   ├── calib/                    #   本地标定脚本（工作区产物）
│   └── test/                     #   run_tests.py + 31 个套件 / verify_formats.py
├── vocabs/base1.3.0/             # 版本化基础词表（进仓库）
│   └── vocab.vfc / config.json / typst_map.json
├── models/                       # 运行时：models/NAME/…（.gitignore）
├── data/                         # 运行时：queue / done / failed / task_meta / tmp
├── logs/                         # 运行时：server.log / auto_train.jsonl / …
└── service/*/output/             # 运行时：各服务产物（.gitignore）
```

`server.py` composes the API mixins (`TrainApi`, `CodecApi`, `AgentApi`, `MonitorApi`, `PrimsApi`, `TokensApi`) and delegates startup, subprocess supervision, model registry, media decoding and utilities to focused modules; `App` is the single assembly point. Runtime directories (`models/`, `data/`, `logs/`, `environment/*`, `service/*/output/`) are git-ignored and created on demand — the checked-out tree contains only source and contracts.

### Model directory contract

```
models/NAME/
├── config.json       # manifest: blocks / d_model / ctx / share_enc_dec / momw /
│                     #   vocab_source / weights 三文件路径 …
├── weights/          # 三个 safetensors：主权重 / wte. / visual_proj.+coords+audio_proj.
├── vocab/            # 建模时从 vocabs/<name>/ 整目录复制的词表副本
└── momw_state/       # 可选：MoMW 池快照（layout / tensors / meta），不进权重契约
```

`common.build_model(model_dir)` is the **only** model construction entry point (used by init / train / auto_train / infer / server); it builds from `config.json` and does a strict `load_state_dict`. `save_checkpoint` writes versioned snapshots with atomic single-file replace; `split_state_dict` guarantees the three files are disjoint and complete. After `apply_precision()` it must call `retie_shared_weights()` because `model.to(dtype)` replaces `Parameter` objects and breaks tying. A missing `d_model` / `share_enc_dec` / `momw` field falls back to the documented default (1024 / false / off), so old folders still load.

**Per-model vocabulary**: `vocab/` is a full copy of a vocabulary folder, so deleting or editing another model's vocabulary can never invalidate a checkpoint. Without it, the global `service/tokens/vocab.vfc` is used. When the active model changes, the server and the infer/train subprocesses each call `set_model_vocab()` so `/api/tokens/*`, `/api/vocab/*`, training and inference all speak the same mapping. Encode with a specific vocabulary via `--vocab <.vfc|vocab folder|model folder>` on `enctok`/`dectok`.

## Portability and isolation

Everything the app writes stays **inside the project folder**. `server/paths.py` derives the root from `__file__` — never from cwd, `HOME`, or a baked-in absolute path. Launchers resolve the root from their own location, so the folder can be moved or renamed freely. Runtime temp files use `paths.project_tmp_dir()` (`data/tmp/`), downloads and pip caches use `paths.project_cache_dir()` (`environment/_cache/`), and subprocesses inherit `paths.isolated_env()`. `service/test/test_portability.py` is a static gate that fails if a machine-specific absolute path, an unpinned `tempfile.*` call, a `shutil.which()` fallback, a system package manager, or a venv bound to a system Python reappears.

**Zero system dependency (hard rule)**: the only Python base is the standalone CPython downloaded into `environment/python/`; `pyvenv.cfg` must point inside the project (`paths.venv_is_isolated()`), otherwise `start/setup_env.py` rebuilds the venv (migrating its site-packages with a same-partition `os.replace`, never a copy). All installs go through `start/setup_env.py`; runtime binaries are resolved only through `paths.bundled_tool()` — no `shutil.which()` / PATH fallback for tools, no system package manager. The only system binaries ever *read* are `curl`/`tar` for the one-time bootstrap (plus the platform linker when building the Rust renderer); nothing is installed or written outside the project.

**Cross-platform behaviour**: POSIX uses `launcher/*.sh` + `启动.sh` / `启动.command`; Windows uses `start/terminal/*.bat` + `启动.bat` and `gargantua.bat`. `server_monitor.py` branches on `platform.system()` for device/process queries; `server_term.py` detects PTY support and degrades the built-in terminal to a clear error on Windows instead of failing server startup; `paths._exe()` appends `.exe` on Windows. The Rust GPU renderer and `setup.sh` are optional — the pure-numpy CPU fallback works everywhere.

**Hard limit — one folder cannot run unchanged on every OS/CPU**: native wheels (torch 2.2.2 macOS x86_64 here), the vendored Rust renderer binary and the system WebView (WKWebView / WebView2 / WebKitGTK) are platform-specific. Moving between OSes or CPU architectures requires one `environment/` built per target platform (or a container image) sharing the same sources — the code paths, API and model formats are identical, but native binaries must be rebuilt/reinstalled for the target.

## Testing

```bash
python service/test/run_tests.py                    # all suites, one process per suite
python service/test/run_tests.py --list             # list suites
python service/test/run_tests.py --suite test_model # a single suite
python -m unittest service.test.test_model -v       # unittest directly
```

`run_tests.py` first runs static gates (`py_compile` on every `.py`, `bash -n` on every `.sh`, `minify.py --check`), then each of the 31 registered suites in its own subprocess. Suites run in separate processes on purpose: a full `Gargantua()` is ~1.03B params (~4.1 GB fp32 since the 2026-11-04 block reorder; shape tests use 1+1 blocks to stay light) and PyTorch's CPU allocator keeps freed blocks in its pool, so RSS never drops within a process. A full regression takes ≈13.5 minutes on this machine and requires explicit authorization; day-to-day changes are checked with `py_compile` / `node --check` / `minify.py --check` plus the affected suite only. New suites must be registered in `run_tests.SUITES`. Changing `index_source.html` requires rebuilding `index.html` via `service/web/minify.py` and re-running `--check`.

## Technical References

- **MLA**: DeepSeek-V2 — low-rank joint KV compression, NoPE.
- **KDA**: Moonshot AI / Kimi — linear gated delta recurrence.
- **CSA/HCA Compression**: DeepSeek NSA — learnable MLP compression within time-steps.
- **AttnRes**: Moonshot AI / Kimi — [Attention Residuals](https://github.com/MoonshotAI/Attention-Residuals) (arXiv 2603.15031).
- **Fast Weight**: Schmidhuber 1991 — persistent associative memory layer.
- **Muon**: matrix-orthogonalizing optimizer for hidden 2D weights.
- **CLIP**: symmetric InfoNCE over in-batch negatives.
- **Primitive Rendering**: [wonderfulearth/primitive-operation-painter](https://github.com/wonderfulearth/primitive-operation-painter).

## License

GPLv3.0

---

**Note**: GARGANTUA v1 is a research-oriented framework. The architecture is fully implemented and forward-pass functional, but meaningful generation requires training on domain-specific data.
