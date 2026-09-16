# Hearth — Model Recommendations

This guide recommends specific Ollama chat and embedding models, quantizations, context sizes and server settings for each hardware tier.

> **Freshness warning.** These picks were checked against the Ollama library in **mid-September 2026**. Open models ship very quickly: Qwen 3.5, 3.6 and 3.8, the Gemma 4 family and Meta's Muse Glimmer all appeared within roughly seven months. Vendor benchmark tables are self-reported. Treat everything below as a **shortlist to validate on your own hardware** with `hearth bench` and `hearth eval`, and re-check monthly. Model choice is configuration, never code.

---

## 0. Your Development Machine — ASUS ROG Strix G15 G513IM

| Component | Spec | What it means for Hearth |
|---|---|---|
| GPU | GeForce RTX 3060 Laptop, **6 GB GDDR6** | Hard ceiling. The chat model plus its KV cache must fit in ~5 GB to stay 100% on GPU. |
| CPU | AMD Ryzen 7 4800H (8 cores / 16 threads) | Good for indexing, tree-sitter parsing and CPU-side query embeddings. Too slow for CPU prefill of long prompts. |
| RAM | 16 GB DDR4 | Shared by the OS, WSL2, the IDE, Python and any model layers that spill off the GPU. Partial offload is possible but tight. |
| Storage | 512 GB SSD | Plenty for 2–3 models (~11 GB) plus indexes, but don't hoard models. |

**Tier placement:** lower edge of **Tier 1**. The Tier 1 table below assumes an 8 GB GPU; 6 GB means one step smaller on the chat model.

### 0.1 Models to pull

| Role | Model | Size | Why |
|---|---|---|---|
| **Daily chat and agent model** | **`qwen3.5:4b`** (default tag = Q4_K_M) | 3.4 GB | Fits entirely on GPU with room for KV cache at 12–16K context. Supports tools, thinking and vision. |
| **Embedding model** | **`qwen3-embedding:0.6b`** | ~0.64 GB | Same default as every tier, so your index stays valid if you upgrade hardware |
| Optional stretch / eval model | `qwen3.5:9b` (Q4_K_M) | 6.6 GB | Slightly larger than VRAM, so some layers spill to CPU and it runs noticeably slower. Use it for occasional eval runs and hard questions, not interactive work. Close the browser first. |
| Optional tiny fallback | `qwen3.5:2b` (default tag = Q8_0) | 2.7 GB | When you need longer context or very fast simple Q&A |

```bash
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
# later, only if needed:
# ollama pull qwen3.5:9b
```

**Don't pull these for this laptop:**
- **`qwen3.5:4b-q8_0`** (5.3 GB). Higher precision, but it leaves almost no VRAM for context.
- **Anything ≥ 12B** (`gemma4:12b` is 7.6 GB). It spills more to CPU than `qwen3.5:9b`, so `qwen3.5:9b` is the better stretch model.

### 0.2 VRAM budget (target: `ollama ps` shows `100% GPU`)

| Item | Approx. |
|---|---|
| CUDA runtime / driver overhead | 0.5–0.8 GB |
| `qwen3.5:4b` weights (incl. vision encoder) | 3.4 GB |
| KV cache at 12K context, `q8_0` | Small (Qwen 3.5 uses hybrid attention); confirm with `ollama ps` |
| Headroom for display / other GPU apps | ~0.5–1 GB |

Start at **`num_ctx = 12288`**. If `ollama ps` still shows `100% GPU` with the model loaded, try `16384`. If you see a CPU/GPU split, go back down.

The G513IM pairs the RTX 3060 with the Ryzen's integrated graphics. When Windows drives the display from the integrated GPU, most of the RTX 3060's VRAM stays free for models. Check `nvidia-smi` before starting a session. Browsers, games or GPU-accelerated apps holding VRAM will push the model partly onto the CPU.

### 0.3 Operating system layout

Hearth's MVP targets Linux and macOS, and Windows runs it through WSL2. Pick one layout:

| Option | Layout | Pros | Cons |
|---|---|---|---|
| **A (recommended on Windows 11)** | **Ollama for Windows** (native GPU driver path) + **Hearth inside WSL2 (Ubuntu)** with **mirrored networking** | Best GPU compatibility; `127.0.0.1` inside WSL reaches Windows Ollama, so Hearth's loopback check passes | Two environments to understand |
| B | Ollama **and** Hearth both inside WSL2 (NVIDIA's Windows driver exposes CUDA to WSL2) | Everything in one Linux environment | Ollama must run as a service inside WSL; models live inside the WSL virtual disk |
| C | Native Linux (dual boot) | Simplest match for the project's Linux-first design | Requires a Linux install and NVIDIA driver setup |

**Option A setup:**

1. **Enable mirrored networking.** Create or edit `C:\Users\<you>\.wslconfig`:
   ```ini
   [wsl2]
   networkingMode=mirrored
   memory=6GB        # leave the rest of your 16 GB to Windows, Ollama and your editor
   swap=8GB
   ```
   Then run `wsl --shutdown` in PowerShell and reopen Ubuntu. In default NAT mode, Windows appears to WSL as a non-loopback IP, and Hearth refuses to connect by design. Don't set `allow_remote_host = true` just to work around it; switch to mirrored mode instead.

2. **Set Ollama's server variables on Windows** (PowerShell), then quit Ollama from the system tray and start it again:
   ```powershell
   setx OLLAMA_NO_CLOUD 1
   setx OLLAMA_HOST 127.0.0.1:11434
   setx OLLAMA_FLASH_ATTENTION 1
   setx OLLAMA_KV_CACHE_TYPE q8_0
   setx OLLAMA_NUM_PARALLEL 1
   setx OLLAMA_MAX_LOADED_MODELS 2
   setx OLLAMA_KEEP_ALIVE 20m
   ```

3. **Keep repositories inside the WSL filesystem** (e.g., `~/code/…`), not under `/mnt/c/…`. Windows-drive paths from WSL are much slower for indexing and file watching, and they are case-insensitive, which complicates path safety checks.

4. **Verify from inside WSL:**
   ```bash
   curl -s http://127.0.0.1:11434/api/version    # should print Ollama's version
   ```
   Then run on the Windows side:
   ```powershell
   ollama run qwen3.5:4b "say hi"
   ollama ps                                     # PROCESSOR should read 100% GPU
   ```

### 0.4 Laptop-specific practicalities

- **Power and thermals.** Stay on AC power and use the Performance/Turbo profile in Armoury Crate during indexing and evals. Long embedding runs heat the GPU, and thermal throttling shows up as falling tokens per second.
- **RAM discipline.** With 16 GB, close Chrome and other heavy apps during eval runs or when using `qwen3.5:9b`. Keep WSL's memory cap (above) so Windows and Ollama don't get squeezed.
- **Embedding placement.** Hearth's `embed_placement = "auto"` uses the GPU for bulk indexing, when no chat model is loaded. During chat sessions it keeps query embeddings on the CPU, so the 4B chat model is never evicted from VRAM.
- **Disk.** Budget ~11 GB if you pull all three chat/embedding models, 1–2 GB for Python environments, and well under 1 GB per repository index. On Windows, models live in `C:\Users\<you>\.ollama\models`. Set `OLLAMA_MODELS` to move them if the SSD fills up.

### 0.5 Expectations on this laptop

- ✅ Codebase Q&A with citations, explanations, docs for single modules, small single-file edits.
- ⚠️ Writing or fixing a single test file. Expect retries, and keep thinking off in chat to save context.
- ❌ Autonomous multi-file refactors. Use `/plan`, then execute one step at a time with approvals.

The indexing and retrieval engine, which is the hardest engineering in the project, runs the same as on bigger machines. Only the model's reasoning depth is limited.

### 0.6 Hearth configuration for this laptop

```toml
# ~/.config/hearth/config.toml   (inside WSL2 for Option A/B)
[ollama]
host = "http://127.0.0.1:11434"
min_version = "0.34.0"

[models]
tier = "tier1"
chat = "qwen3.5:4b"
eval_chat = "qwen3.5:9b"          # used by `hearth eval --model eval_chat`; never loaded alongside chat
embed = "qwen3-embedding:0.6b"
embed_dimensions = 1024
num_ctx = 12288
keep_alive = "20m"
embed_placement = "auto"          # auto | gpu | cpu

[agent]
default_mode = "chat"
permission_level = "supervised"
```

---

## 1. Hardware Tiers

| Tier | Label (your request) | Typical hardware | Realistic `num_ctx` | What to expect |
|---|---|---|---|---|
| **Tier 1** | Low-end | 6–8 GB VRAM GPU + 16 GB RAM · 16 GB Apple Silicon · CPU-only with 16–32 GB RAM | 8K (CPU) · 12K (6 GB GPU) · 16K (8 GB GPU) | Excellent codebase Q&A with citations; single-file edits; short supervised agent tasks (≤ ~8 steps) |
| **Tier 2** | Mid-range | 12–16 GB VRAM + 32 GB RAM · 24–36 GB Apple Silicon | 32K | Good Q&A including cross-file flow; multi-file edits with plan mode; test writing with a fix loop |
| **Tier 3** | High-end | 24–32 GB VRAM (RTX 3090/4090/5090) + 32–64 GB RAM · 48–64 GB Apple Silicon | 64K (32K on 24 GB with larger dense models) | Solid agentic coding: plan → multi-file change → tests → commit; documentation generation |
| **Tier 4** | Workstation (bonus) | 96–128 GB+ unified memory (Apple Max/Ultra, AMD Ryzen AI Max-class, NVIDIA DGX Spark-class) · ≥48 GB multi-GPU | 128K | Longest autonomous runs, large refactors, second "fast" model loaded concurrently |

**Rule of thumb:** memory decides which models *fit*. For MoE models, **active** parameters decide generation *speed*. Your prompt size and GPU versus CPU placement decide *prefill* latency, and prefill often dominates agent latency.

---

## 2. Recommended Models per Tier

### 2.1 Tier 1 — Low-end

| Role | Default | Alternatives | Notes |
|---|---|---|---|
| Chat (6 GB GPU laptop, e.g., RTX 3060 Laptop) — **your machine** | **`qwen3.5:4b`** (≈ 3.4 GB, Q4_K_M) | `qwen3.5:2b` (≈ 2.7 GB, Q8_0) for longer contexts; `qwen3.5:9b` for evals only (partial offload) | Fits fully on GPU with room for KV cache at 12K. See §0. |
| Chat (8 GB GPU / 16 GB Mac) | **`qwen3.5:9b`** (default tag ≈ 6.6 GB) | `gemma4:12b` (≈ 7.6 GB) if you have 10–12 GB VRAM | Both support tools, thinking and vision with long native context windows. Keep `num_ctx` at 16K so the KV cache fits. |
| Chat (CPU-only, 16 GB RAM) | **`qwen3.5:4b`** (≈3 GB class) | `qwen3.5:2b` for very weak machines (Q&A only) | Prefill on CPU is slow: keep `num_ctx` ≈ 8K and prefer single-shot chat mode. |
| Chat (CPU-only, ≥32 GB RAM) | **`gemma4:26b`** (MoE, ~4B active, ≈ 19 GB) | `qwen3.5:9b` | An MoE model generates at small-model speed but needs large RAM. Prefill is still CPU-bound, so keep prompts small. |
| Embedding | **`qwen3-embedding:0.6b`** (≈ 639 MB) | `embeddinggemma` (≈ 622 MB) | A/B both on the retrieval eval. Pin the embedding model to CPU if VRAM is tight. |

**Hearth profile defaults for Tier 1:**
- `tool_reliability = "medium"` for 9–12B models and `"low"` for ≤4B.
- Thinking off in chat and on only for `/plan`.
- Agent step limit of 8.
- Chat mode may disable tools entirely for ≤4B models (pure RAG).

### 2.2 Tier 2 — Mid-range

| Role | Default | Alternatives | Notes |
|---|---|---|---|
| Chat (16 GB VRAM PC) | **`gemma4:12b`** at a higher-precision tag if one fits (see §3), else the default tag | `qwen3.5:9b` (Q8 tag), `gpt-oss:20b` (MoE, fits 16 GB, older generation but fast and proven) | Everything fits in VRAM, which keeps prefill fast at 32K context. |
| Chat, "MoE stretch" (16 GB VRAM + ≥32 GB RAM) | **`qwen3.6:35b`** (35B MoE, ~3B active, ≈ 23 GB) with partial CPU offload | `gemma4:26b` (≈ 19 GB) with partial offload | Stronger agentic coding than the in-VRAM options. Generation stays usable because few parameters are active, but prefill slows. Decide with `hearth bench` + `hearth eval tasks`. |
| Chat (36 GB Apple Silicon) | **`qwen3.6:35b-mlx`** (≈ 24 GB) | `gemma4:26b`, `qwen3.6:27b-mlx` (≈ 19 GB) | Prefer `-mlx` tags on Apple Silicon: Ollama's MLX engine is optimized for them. |
| Chat (24 GB Apple Silicon) | **`gemma4:12b`** | `qwen3.5:9b` | macOS reserves part of unified memory for the system, so 20 GB+ models are too tight. |
| Embedding | **`qwen3-embedding:0.6b`** | `embeddinggemma` | Same default as Tier 1, so an index stays valid if you upgrade hardware. |

**Hearth profile defaults for Tier 2:**
- `tool_reliability = "medium"` (12B) or `"high"` (35B MoE).
- Thinking `low`/`medium` in plan and agent modes.
- Agent step limit of 20.

### 2.3 Tier 3 — High-end

| Role | Default | Alternatives to evaluate | Notes |
|---|---|---|---|
| Chat — quality | **`qwen3.8:27b`** (dense, ≈ 18 GB, 256K native) | `muse-glimmer:30b` (≈ 18 GB, 128K, Apache-2.0, tuned for tool use and failure recovery) | Dense 27–32B models are the strongest single-GPU options per their vendors' benchmarks, but they are several times slower to generate than MoE models of similar size. |
| Chat — speed | **`gemma4:26b`** (MoE, ~4B active, ≈ 19 GB) | `nemotron-3.5-lightning:30b` (MoE, ~3B active) | Community reports on RTX 3090-class cards for Gemma 4 26B vary widely with quantization and harness; measure yourself. |
| Chat (32 GB VRAM or 48–64 GB Mac) | **`qwen3.6:35b`** / `qwen3.6:35b-mlx` | `qwen3.8:27b` with 64K+ context | Headroom allows MoE speed *and* long context. |
| Embedding | **`qwen3-embedding:0.6b`** | `qwen3-embedding:4b` (≈ 2.5 GB) | Upgrade only if the retrieval eval shows a real recall gain. Changing it means re-embedding. |

**Hearth profile defaults for Tier 3:**
- `tool_reliability = "high"`.
- Thinking `medium` for plan and agent, off for chat.
- Agent step limit of 40.
- Optional LLM rerank if the eval supports it.

**Suggested approach on a 24 GB card:** configure *two* profiles, `quality` = `qwen3.8:27b` and `speed` = `gemma4:26b`. Switch with `/model` and don't load both at once. Use `speed` for chat and quick edits, and `quality` for `/plan` and hard debugging.

### 2.4 Tier 4 — Workstation

| Role | Default | Alternatives | Notes |
|---|---|---|---|
| Chat — main | **`qwen3.5:122b`** (MoE, ~10B active) | `gpt-oss:120b` (proven, native MXFP4), `mistral-medium-3.5:128b` (dense; slow, quality-critical tasks only) | Check the tag's size against your memory before pulling. |
| Chat — experimental (≥128 GB Apple Silicon) | **`qwen3.8-flash-next:125b-mlx`** (≈ 105 GB, ~6B active) | — | Explicitly an *experimental preview* of a next-generation architecture. Strong self-reported agentic scores and very fast per token, but validate stability before relying on it. |
| Chat — fast secondary | `qwen3.6:35b` or `gemma4:26b` | — | Loaded concurrently for summaries, commit messages, reranking and compaction. |
| Embedding | **`qwen3-embedding:4b`** | `qwen3-embedding:8b` (≈ 4.7 GB at Q4_K_M) | Larger embedders help most on big, heterogeneous repos. |

### 2.5 Models and variants to avoid

- **Anything with a `cloud` tag** (e.g., `gemma4:31b-cloud`, `glm-5.3`, `deepseek-v4-flash`, `kimi-k2.7-code`). These run on Ollama's servers. Hearth refuses them, and you should also disable Ollama's cloud features server-side (§5).
- **Sub-4B models for agent mode.** They are fine for chat or RAG but unreliable at multi-step tool use.
- **Unofficial community re-uploads** (`someuser/model`) as defaults. Chat templates and tool-call parsing are often subtly wrong, and provenance is unclear. Prefer official library models; use community models only after they pass your evals.
- **Mixing embedding models or dimensions** between indexing and querying. Hearth treats each model + dimension pair as a separate index.

---

## 3. Quantization Recommendations

Ollama's default library tags are usually 4-bit (typically `Q4_K_M` for GGUF models). MLX tags use MLX-native low-bit formats. Some models ship native formats (gpt-oss uses MXFP4) or quantization-aware-trained variants (e.g., Gemma 4 `*-it-qat` tags). Exact tag names differ per model, so check each model's **Tags** page.

| Model size | Recommended | Acceptable minimum | Reasoning |
|---|---|---|---|
| ≤ 4B | **Q8_0** (FP16 if it fits) | Q6_K | Small models lose proportionally more to quantization; tool-call JSON fidelity and exact identifiers suffer first |
| 7–14B | **Q6_K or Q8_0** if memory allows | Q4_K_M | The step from Q4 to Q6 is noticeable in code correctness on small models; memory cost is modest |
| 20–35B (dense or MoE) | **Q4_K_M** (default tags) | Q4_K_M; prefer QAT tags when available | Best quality per GB at this size; spend spare memory on context before going above Q5 |
| 70B+ / large MoE | **Q4_K_M** or native format (MXFP4) | Avoid Q3 and below unless unavoidable | Below 4 bits, quality tends to fall off sharply, especially for code |
| Embedding models | **Q8_0 or F16** | Q8_0 | They're small; retrieval quality is the product's foundation |

**Priority when memory is tight:** (1) keep the whole model on GPU, (2) keep adequate context (≥16K for agent mode), (3) then raise weight precision.

### 3.1 KV cache quantization

| Setting | Memory vs f16 | Recommendation |
|---|---|---|
| `f16` (default) | 1× | Use when memory is plentiful |
| **`q8_0`** | ~½ | **Default for all tiers.** Ollama's docs describe the quality impact as usually unnoticeable. |
| `q4_0` | ~¼ | Last resort. Loss can become noticeable at larger contexts, which is exactly the RAG-heavy regime Hearth runs in. |

KV cache quantization requires flash attention and is a **global** server setting (it applies to every model the server runs).

---

## 4. Memory Estimation

### 4.1 Formulas

```text
weights_GB     ≈ params_B × bits_per_weight / 8          (+ vision/audio encoder if the tag includes one)
                  Q4_K_M ≈ 4.8–4.9 bpw · Q5_K_M ≈ 5.7 · Q6_K ≈ 6.6 · Q8_0 ≈ 8.5

kv_bytes/token ≈ 2 × n_full_attention_layers × n_kv_heads × head_dim × bytes_per_element
                  bytes_per_element: f16 = 2 · q8_0 ≈ 1 · q4_0 ≈ 0.5

total          ≈ weights + kv_bytes/token × num_ctx × OLLAMA_NUM_PARALLEL + ~0.5–1.5 GB runtime overhead
```

### 4.2 Worked example (illustrative architecture)

Take a 27B dense model with full attention in every layer: 64 layers, 8 KV heads, head dim 128.

| Item | f16 KV | q8_0 KV |
|---|---|---|
| Weights at Q4_K_M | ≈ 27 × 4.85 / 8 ≈ **16.4 GB** | same |
| KV per token | 2 × 64 × 8 × 128 × 2 B ≈ 0.25 MiB | ≈ 0.125 MiB |
| KV at 32K context | ≈ **8.0 GiB** | ≈ **4.0 GiB** |
| KV at 64K context | ≈ 16.0 GiB | ≈ 8.0 GiB |
| Total at 32K, q8_0 | — | ≈ 16.4 + 4.0 + 1.0 ≈ **21.4 GB** → fits a 24 GB card |

**Hybrid-attention models change this math.** Recent Qwen generations interleave linear-attention layers with full-attention layers, and only the full-attention layers store a growing KV cache. KV memory can be a fraction of the example above, which is why long contexts are more practical on these families. Don't hand-compute for specific models: **load the model at your target `num_ctx` and read `ollama ps`**. It shows total size, the GPU/CPU split and the allocated context.

### 4.3 Checklist for fitting a model

1. `ollama ps` shows **`100% GPU`** (any CPU split means much slower prefill).
2. The `CONTEXT` column equals the `num_ctx` Hearth requested.
3. You leave ~1 GB of VRAM free for the desktop and browser.
4. `OLLAMA_NUM_PARALLEL=1` (parallel slots multiply context memory).
5. `hearth bench` reports prefill and generation speeds you can live with at a realistic prompt size (e.g., 12K tokens).

---

## 5. Ollama Server Configuration

Set these on the Ollama **server**. On macOS use `launchctl setenv`, on Linux use `systemctl edit ollama.service`, and on Windows use user environment variables. Restart Ollama afterwards.

```bash
# Privacy (all tiers)
OLLAMA_HOST=127.0.0.1:11434     # loopback only (the default; make it explicit)
OLLAMA_NO_CLOUD=1               # or {"disable_ollama_cloud": true} in ~/.ollama/server.json

# Memory & performance (all tiers)
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0
OLLAMA_NUM_PARALLEL=1
OLLAMA_KEEP_ALIVE=30m           # Hearth also sets keep_alive per request
OLLAMA_MAX_LOADED_MODELS=2      # chat + embedding without eviction (3 on Tier 4 for a fast secondary model)

# Fallback default context (Hearth always sends num_ctx explicitly anyway)
OLLAMA_CONTEXT_LENGTH=32768     # Tier 1: 16384 · Tier 2: 32768 · Tier 3: 65536 · Tier 4: 131072
```

**Why set context explicitly:** Ollama's default context length depends on detected VRAM and can be as small as 4K on sub-24 GB GPUs. Ollama's own guidance is to use at least 64K for coding tools. Hearth sends `num_ctx` on every request and keeps it constant within a session, because changing it forces a model reload.

After restarting, confirm the cloud setting took effect: the Ollama logs should report cloud disabled. `hearth doctor` warns if cloud features appear to be enabled.

---

## 6. Hearth Configuration per Tier

```toml
# ~/.config/hearth/config.toml — choose one block

# ---- Tier 1, 6 GB GPU (your laptop — full block in §0.6) ----
[models]
tier = "tier1"
chat = "qwen3.5:4b"
eval_chat = "qwen3.5:9b"
embed = "qwen3-embedding:0.6b"
num_ctx = 12288
embed_placement = "auto"

# ---- Tier 1, 8 GB GPU ----
# chat = "qwen3.5:9b"
# embed = "qwen3-embedding:0.6b"
# num_ctx = 16384
# embed_placement = "auto"

# ---- Tier 2 (36 GB Mac) ----
# chat = "qwen3.6:35b-mlx"
# embed = "qwen3-embedding:0.6b"
# num_ctx = 32768

# ---- Tier 3 (24 GB GPU) ----
# chat = "qwen3.8:27b"
# fast_profile = "gemma4:26b"      # switch with /model speed
# embed = "qwen3-embedding:0.6b"
# num_ctx = 32768                  # try 65536 and check `ollama ps`

# ---- Tier 4 (128 GB unified) ----
# chat = "qwen3.5:122b"
# fast = "qwen3.6:35b"
# embed = "qwen3-embedding:4b"
# num_ctx = 131072
```

### 6.1 Model profile examples (`profiles.toml`)

Profiles encode per-family behavior. Ollama honors default sampling parameters defined in the model file, so only override sampling when the model card recommends different values for your use.

```toml
[[profile]]
match = ["qwen3.5:*", "qwen3.6:*", "qwen3.8:*"]
family = "qwen3.x"
tools = "native"                    # native | text_fallback | none
tool_reliability = "high"           # overridden to "medium" for ≤9B via size_rules
think = { chat = "off", plan = "on", agent = "on" }
preserve_thinking = false           # keep reasoning only within a turn's tool loop
edit_format = "search_replace"
max_steps = { chat = 6, plan = 15, agent = 40 }
size_rules = [ { max_params_b = 9, tool_reliability = "medium", max_steps_agent = 12 } ]

[[profile]]
match = ["gemma4:*"]
family = "gemma4"
tools = "native"
tool_reliability = "high"
sampling = { temperature = 1.0, top_p = 0.95, top_k = 64 }   # per the Gemma 4 model card
think = { chat = "off", plan = "on", agent = "on" }
edit_format = "search_replace"

[[profile]]
match = ["qwen3-embedding:*"]
family = "qwen3-embedding"
kind = "embedding"
query_template = "Instruct: Given a question about a software repository, retrieve the code or documentation that answers it\nQuery: {query}"
document_template = "{text}"
matryoshka = true
max_input_tokens = 8192            # keep chunks far below the model's limit

[[profile]]
match = ["embeddinggemma*"]
family = "embeddinggemma"
kind = "embedding"
# Task-prefix templates: copy the exact query/document prefixes from the EmbeddingGemma model card,
# then confirm with `hearth eval retrieval` before switching defaults.
matryoshka = true
max_input_tokens = 2048
```

---

## 7. Choosing Your Model Empirically

1. **Pick your tier**, then pull the default plus one or two alternatives.
2. **`hearth bench --models A,B --ctx <num_ctx> --prompt-tokens 12000`** measures time to first token, prefill tokens/s, generation tokens/s, `ollama ps` placement and memory headroom.
3. **`hearth eval retrieval --embed X,Y`** compares recall@5, recall@10 and MRR on the fixture sets plus your own questions.
4. **`hearth eval tasks --models A,B`** measures success rate, steps, tool-call errors and wall time on the agent task suite.
5. **Choose by task success first, then latency.** A model that's 2× faster but fails twice as often costs you more time.
6. **Re-run monthly** or when a new model family lands. Commit the results table to `docs/benchmarks/`.

### 7.1 Reference throughput data point

One published cheat sheet from August 2026 measured generation speed on a single 24 GB consumer GPU:
- MoE models (`qwen3-coder:30b`, `gpt-oss:20b`, `gemma4:26b`) generated at 7B-class speeds or faster, well above 100 tokens/s.
- Dense 7–9B models landed around 130–175 tokens/s, and dense 12B around 90.
- Dense 27–32B models dropped to roughly 40 tokens/s.

Your numbers will differ with GPU, quantization, context and Ollama version, but the *shape* holds: **MoE for speed, dense 27B+ for maximum quality per GPU.**

---

## 8. Capability Expectations (Be Honest with Yourself)

| Task | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---|---|---|---|---|
| "Where/how is X implemented?" with citations | ✅ Strong (retrieval does the heavy lifting) | ✅ | ✅ | ✅ |
| Architecture overview / docs generation | ⚠️ Summaries slow; short docs OK | ✅ | ✅ | ✅ |
| Single-file bug fix with tests | ⚠️ Often, with guidance | ✅ | ✅ | ✅ |
| Write unit tests + fix loop | ⚠️ Simple targets | ✅ | ✅ | ✅ |
| Multi-file refactor (3–10 files) | ❌ Use plan mode + manual steps | ⚠️ With plan mode | ✅ With plan mode | ✅ |
| Long unattended tasks (30+ steps) | ❌ | ❌ | ⚠️ | ⚠️–✅ |

Local models in 2026 are dramatically better than a year earlier, but they still trail frontier cloud models on long-horizon agentic work. Hearth's design (retrieval, plan mode, deterministic workflows, verification loops, human approval) exists to close as much of that gap as possible.

---

## 9. Sources Checked (September 2026)

- Ollama releases (v0.34.0, v0.33.x, v0.32.x notes): https://github.com/ollama/ollama/releases
- Ollama FAQ (cloud disable, KV cache types, concurrency settings): https://docs.ollama.com/faq
- Ollama context length defaults: https://docs.ollama.com/context-length.md
- Ollama embeddings API (`dimensions`, `truncate`, normalized vectors): https://docs.ollama.com/api/embed.md
- Ollama tool calling: https://docs.ollama.com/capabilities/tool-calling.md
- Model pages: https://ollama.com/library/qwen3.8 · https://ollama.com/library/qwen3.6 · https://ollama.com/library/qwen3.8-flash-next · https://ollama.com/library/muse-glimmer · https://ollama.com/library/gemma4/tags · https://ollama.com/library/qwen3.5/tags · https://ollama.com/library/qwen3-embedding
- Embedding model comparison (sizes, MTEB tracks): https://www.morphllm.com/ollama-embedding-models
- Throughput cheat sheet: https://computingforgeeks.com/ollama-models-cheat-sheet/
