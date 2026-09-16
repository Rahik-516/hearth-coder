# Hearth — Tech Stack

**Every library, tool and external binary Hearth depends on, why it was chosen, and what was rejected.**

| | |
|---|---|
| Document status | v1.0 — derived from the design baseline |
| Date | September 2026 |
| Companion docs | `system-design.md`, `project-structure.md`, `implementation-roadmap.md`, `model-recommendations.md`, `safety-and-tool-use.md`, `future-extensions.md` |

This document expands the decision table in `system-design.md` §2 (D1–D10) and inventories everything else the
project depends on. It is the authority for one rule: **do not add a dependency that isn't listed in §19 without
writing an ADR first.**

Decisions here are traceable to the companion docs, with section references given throughout. Writing it surfaced
four places where the baseline was silent or inconsistent; all four are now decided, and §19.4 records each one and
points at the document that owns it.

---

## 1. Dependency Policy

### 1.1 Why this is strict

`system-design.md` §1.5 states the constraint plainly: every dependency is a liability for an offline tool. It can
break, add telemetry, or phone home. Every framework layer hides prompts and control flow that the developer will
eventually need to debug. The target is **about 12 runtime dependencies plus tree-sitter grammar wheels**, one
language, one storage engine, and one inference backend behind a thin interface.

### 1.2 Admission criteria

A new runtime dependency must clear all six:

1. **No network at import or runtime.** Nothing that downloads models, grammars, rules or updates on first use.
2. **No telemetry**, opt-out or otherwise. Check the source, not the README.
3. **It replaces real work.** If a competent implementation is under ~200 lines of code the project already
   understands, write it instead.
4. **Its own dependency tree is small and inspectable.** A package that pulls torch, pyarrow or a browser engine
   is a different decision than the package itself.
5. **It is installable as a wheel for the supported platforms** (Linux x86-64/arm64, macOS arm64, Windows x86-64
   for the Phase 3 native port). No compiler required at install time.
6. **The licence is permissive** (MIT/BSD/Apache-2.0) and compatible with Hearth's MIT licence.

Dev-only dependencies are held to 1, 2 and 6 only.

### 1.3 Process

New runtime dependency → **ADR in `docs/adr/NNNN-title.md`** (`implementation-roadmap.md` "Cross-Cutting
Practices") → ask the user → add to `pyproject.toml` → `uv lock` → update §19 of this document. The ADR records the
alternatives considered and the rejection reasons, in the same shape as the sections below.

Removing a dependency needs no ADR. It needs a superseding note on the ADR that introduced it.

---

## 2. D1 — Language and Runtime

**Chosen: Python 3.12+.**

| Why | Detail |
|---|---|
| Tree-sitter | `py-tree-sitter` is the reference binding, and grammar wheels ship on PyPI (§6) |
| Ollama | The official Python SDK is first-party and exposes the native API surface Hearth needs (§4) |
| Numerics | NumPy makes exact vector search a few lines instead of a vector database (§5, §7) |
| Iteration speed | The agent loop, prompts and retrieval weights change constantly during tuning |
| Developer fit | `system-design.md` D1: "the developer's strength" — a solo-maintained project runs on this |

**Minimum 3.12, specifically:**
- `tomllib` in the stdlib — config loading needs no dependency for reading (§15).
- `asyncio.TaskGroup` and `except*` — structured concurrency for parallel retrievers and bounded embed batches.
- `Path.walk()`, faster interpreter startup, and improved error messages (which matter when the traceback is what
  the agent shows a user).
- Typing that `mypy --strict` can actually use: PEP 695 syntax is available but **not** required — keep to
  `TypeVar`/`Protocol` spelling that older tooling reads cleanly.

Pinned in `.python-version` (3.12) and `requires-python = ">=3.12"`.

### Rejected

| Alternative | Why not |
|---|---|
| **TypeScript / Node** | Best editor-integration story and a real tree-sitter binding, but the numeric and data tooling is weaker, and the eventual VS Code extension is a *thin client over stdio* (§14) so it gains little. Would also mean two languages. |
| **Go** | Excellent single-binary distribution and process control. But tree-sitter bindings are CGo, the ML/numeric ecosystem is thin, and prompt/retrieval iteration is slower. Distribution was never the binding constraint — a `uv tool install` is fine. |
| **Rust** | Best runtime characteristics and the native home of tree-sitter. Wrong trade for a solo developer iterating on prompts and retrieval weights daily. Revisit only if indexing throughput becomes the bottleneck, and then only for the indexer. |

**Revisit trigger:** CLI startup misses the < 1.5 s target (`system-design.md` §15) for reasons lazy imports can't
fix, or indexing a 50K LOC repo misses the < 2 min target on the dev laptop.

---

## 3. Packaging, Environment and Distribution

### 3.1 Toolchain

| Concern | Choice | Notes |
|---|---|---|
| Environment + resolver | **uv** | Fast, lockfile-first, manages the Python toolchain itself |
| Lockfile | **`uv.lock`, committed** | Reproducibility is a security property here, not a convenience |
| Build backend | **hatchling** | Pure-Python, no build-time plugins, standard `pyproject.toml` |
| Layout | **`src/` layout** | Tests import the installed package, not a shadowed working directory |
| Entry point | `hearth = "hearth.cli.app:main"` plus `python -m hearth` | `project-structure.md` §1 |
| Hooks | **pre-commit** | Runs ruff, mypy, lint-imports, pytest (§12) |

Every command in the project is `uv run …`. `project-structure.md` §5 lists the canonical set; they are repeated in
the repository's `CLAUDE.md`.

### 3.2 Version pinning

- **Runtime dependencies are floor-pinned in `pyproject.toml`** (`pydantic>=2`) and **exactly pinned in
  `uv.lock`**. The lock is what ships and what CI uses.
- **Tree-sitter grammar wheels are pinned exactly in both places.** The grammar ABI is coupled to the
  `tree-sitter` runtime version, and a silent grammar bump changes chunk boundaries — which invalidates every
  embedding and every chunker snapshot test. `project-structure.md` §6 already requires a unit test that loads
  every bundled grammar so an incompatible upgrade fails at test time rather than at index time.
- **Dependency updates are a deliberate act**, run with the full check suite plus `hearth eval retrieval`. Prompt
  and retrieval behaviour can move without any Hearth code changing.

### 3.3 Offline installation

The offline guarantee covers *runtime*, but a first install still needs packages from somewhere.
`scripts/build_wheelhouse.sh` (`project-structure.md` §1) exists for this: on a connected machine, download every
wheel for the target platform, then install on the target with `--no-index --find-links`. The README documents both
paths (online and wheelhouse) as an MVP exit requirement (`implementation-roadmap.md` MVP exit checklist).

Models are a separate concern: `ollama pull` on a connected machine, or copy `~/.ollama/models`
(`model-recommendations.md` §0.1).

### 3.4 Platform support

| Platform | Status |
|---|---|
| **Linux (native)** | Supported |
| **Windows 11 + WSL2 (Ubuntu), Ollama on Windows** | The primary development target (`model-recommendations.md` §0.3). Requires WSL **mirrored networking** so loopback reaches Ollama (§4.4) |
| **macOS** | Supported by design, best-effort until hardware is available (`implementation-roadmap.md` MVP exit) |
| **Native Windows (no WSL2)** | Phase 3. Needs a PowerShell/cmd command classifier, path semantics and process-tree kill (`implementation-roadmap.md` Phase 3) |

---

## 4. D2 — Inference API

**Chosen: the native Ollama API through the official `ollama` Python SDK (async client).**

### 4.1 Why native, not OpenAI-compatible

Ollama exposes an OpenAI-compatible endpoint. Hearth does not use it, because every one of these is either absent
or unreliable through the compatibility layer, and each one is load-bearing:

| Need | Native API | Why it matters |
|---|---|---|
| `options.num_ctx` per request | ✅ | **The single most important parameter.** Ollama's default context depends on detected VRAM and can be as small as 4K; exceeding it silently truncates the prompt, and the system prompt or tool definitions can vanish (`system-design.md` §1.1, `model-recommendations.md` §5) |
| `keep_alive` per request | ✅ | Model residency control on a 6 GB GPU — avoids multi-second reload thrash (`system-design.md` §16) |
| `think` | ✅ | Per-mode thinking level; thinking is rendered collapsed and capped (`system-design.md` §5.3, §16) |
| `format` (JSON schema) | ✅ | Structured plan output and listwise rerank (`system-design.md` §7, `implementation-roadmap.md` I3) |
| `/api/show` capabilities | ✅ | Startup validation: does this model have `tools`? does that one have `embedding`? (`system-design.md` §5.4) |
| `/api/ps` | ✅ | Verify GPU placement and residency; feeds `hearth doctor` |
| Prompt/cached token counts + timings | ✅ | Calibrates the TokenEstimator and detects silent truncation (`system-design.md` §5.3, §16) |
| `/api/embed` with `dimensions` and array input | ✅ | Batched, Matryoshka-aware embedding (§8) |

The SDK is first-party, thin, dependency-light, and ships an async client. Using it rather than raw HTTP means no
`httpx` dependency of Hearth's own and no hand-maintained request models.

### 4.2 The provider boundary

The SDK is confined to `llm/ollama_provider.py` behind the `LLMProvider` protocol (`system-design.md` §5.4):

```python
class LLMProvider(Protocol):
    async def chat_stream(self, req: ChatRequest) -> AsyncIterator[ChatChunk]: ...
    async def embed(self, texts: list[str], *, dimensions: int | None) -> np.ndarray: ...
    async def show(self, model: str) -> ModelInfo: ...
    async def running(self) -> list[RunningModel]: ...
    async def version(self) -> str: ...
```

Two implementations: `OllamaProvider` (production) and `ScriptedProvider` (deterministic tests and demos, §11). A
second backend — llama.cpp's server, vLLM, LM Studio — is an additive implementation of the same protocol, not a
rewrite. Nothing above `hearth.llm` imports the SDK.

### 4.3 Egress control lives here

`system-design.md` §4.1 layering rule 5: **only `infra.llm` may open network sockets, and only to the configured
loopback Ollama host.** Two refusals are implemented in the provider and asserted by tests
(`implementation-roadmap.md` Phase 0 acceptance):

- **Cloud model tags** — anything ending in `cloud` or `-cloud` is refused outright.
- **Non-loopback hosts** — refused unless the user sets `allow_remote_host = true` explicitly.

`pytest-socket` (§11) enforces the same property across the whole test suite.

### 4.4 The WSL2 loopback subtlety

Hearth commonly runs inside WSL2 while Ollama runs on Windows. In WSL's **mirrored networking mode**, `127.0.0.1`
inside WSL reaches the Windows-side Ollama and the loopback check passes. In the default **NAT mode**, the Windows
host appears as a private, non-loopback IP — which Hearth deliberately refuses. `hearth doctor` detects this and
explains the two fixes: enable mirrored mode, or run Ollama inside WSL2 (`system-design.md` §5.4).

### 4.5 Server-side configuration

Not a dependency, but part of the stack. `model-recommendations.md` §5 specifies the Ollama server environment:
`OLLAMA_HOST=127.0.0.1:11434`, `OLLAMA_NO_CLOUD=1`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`,
`OLLAMA_NUM_PARALLEL=1`, `OLLAMA_KEEP_ALIVE=30m`, `OLLAMA_MAX_LOADED_MODELS=2`. `hearth doctor` checks what it can
observe and warns when cloud features appear enabled.

### Rejected

| Alternative | Why not |
|---|---|
| **Ollama's OpenAI-compatible endpoint** | Loses `num_ctx`, `keep_alive`, `think`, capabilities and timing data. The portability it buys is already covered by the `LLMProvider` protocol. |
| **Raw HTTP via `httpx`** | Adds a dependency to hand-maintain request/response models that the first-party SDK already maintains. Reconsider only if the SDK lags the server API. |
| **`llama-cpp-python`** | In-process inference means Hearth owns GGUF loading, GPU layer splitting and VRAM management — exactly the work Ollama does well. Also turns a pure-Python install into a compiled one (criterion 1.2.5). |
| **vLLM / TGI / Ray Serve** | Server-class infrastructure for concurrent users. Wrong shape for a single-user laptop tool with a 6 GB GPU. |
| **LiteLLM / LangChain model wrappers** | Abstraction over ~100 providers Hearth will never use, with a large dependency tree and the lowest-common-denominator parameter set. The `LLMProvider` protocol is 5 methods. |
| **LM Studio / Jan / KoboldCpp** | Fine tools, but their APIs are less complete for this purpose and their headless stories are weaker. Addable behind the protocol if a user asks. |

---

## 5. D3 — Storage

**Chosen: SQLite (stdlib `sqlite3`), WAL mode, FTS5, with vectors as float16 BLOBs searched exactly by NumPy.**

### 5.1 Why one engine

`system-design.md` §0 decision 2: SQLite is the single system of record. Two databases per project
(`system-design.md` §13):

| DB | Contents | Disposable? |
|---|---|---|
| `index.db` | `meta`, `files`, `chunks`, `chunks_fts` (FTS5), `embeddings`, `symbols`, `refs`, `imports`, `summaries` | **Yes** — always rebuildable from the working tree |
| `state.db` | `sessions`, `messages`, `tool_calls`, `checkpoints`, `grants`, `trust` | **No** — this is the user's history |

The decisive property is **transactional consistency across all of it**. The chunk row, its contentless FTS5 row
and its symbol rows are written in one transaction (`system-design.md` §13.1). A separate vector store would make
"chunk exists but its vector doesn't, or vice versa" a permanent class of bug, reachable by Ctrl+C at the wrong
moment — and the embedding phase is explicitly interruptible and resumable (`system-design.md` §6.7).

Settings: `PRAGMA journal_mode = WAL` and `PRAGMA foreign_keys = ON` (`system-design.md` §13.1). FTS5 uses a
**contentless** table (`content=''`) with `tokenize = "unicode61 remove_diacritics 2 tokenchars '_'"`, keeping the
database small; the writer maintains FTS rowids explicitly.

**FTS5 availability** is not universal in every Python build, so `hearth doctor` checks for it explicitly
(`system-design.md` §15).

### 5.2 Migrations

Hand-rolled numbered SQL migrations (`storage/migrate.py` + `storage/migrations/{index,state}/NNNN_*.sql`) with a
`schema_version` key in `meta`. No Alembic: there is no ORM, the schemas are small and hand-written, and `index.db`
can simply be dropped and rebuilt when a migration would be complex. `state.db` migrations are the only ones that
must genuinely preserve data.

**No ORM at all.** `system-design.md` §13 is written as SQL, the queries are hand-tuned (BM25 column weights,
candidate masks), and SQLAlchemy would add a large dependency to hide ~15 tables.

### 5.3 Vector search

`VectorIndex` is a protocol — `upsert`, `delete`, `search(query_vec, k, candidate_filter)` — with
`NumpyVectorIndex` as the default implementation (`system-design.md` §5.6):

- Vectors stored as **float16, L2-normalized, little-endian BLOBs** in the `embeddings` table, keyed by
  `(embed_text_hash, model_id, dims)`.
- An **in-memory NumPy matrix cache**, rebuilt lazily when dirty.
- Search is an **exact dot product** — Ollama's embed endpoint returns normalized vectors, so cosine similarity is
  a matmul.
- `candidate_filter` masks rows for filtered searches (by language, path glob) without a second index.

At the target scale this is fast enough by a wide margin: 100K chunks × 1024 dims × float16 is ~200 MB, and the
retrieval budget is < 150 ms for the *whole* pipeline at that size (`system-design.md` §15). An approximate index
would add a dependency, an index-build step and a recall cliff, to solve a problem that does not exist yet.

**Revisit trigger** (`system-design.md` §18): a repository exceeds ~1M chunks, or vector RAM exceeds 1.5 GB. Then
add `SqliteVecIndex` or `LanceDbIndex` behind the same protocol and require equal retrieval eval scores at lower
memory (`future-extensions.md` §2.6).

### Rejected

| Alternative | Why not |
|---|---|
| **sqlite-vec** | The closest call, and the designated escape hatch. Rejected *for now* only because exact NumPy search is simpler, has zero recall loss, and needs no extension loading (which some Python builds disable). Kept as the first scale backend. |
| **LanceDB** | Pulls pyarrow and a large dependency tree; adds a second on-disk format with its own consistency story. Reasonable at monorepo scale, wrong at 50K LOC. |
| **Chroma** | Heavy dependency tree, a server-ish model, and historically telemetry-on-by-default — disqualifying under criterion 1.2.2 even though it can be disabled. |
| **Qdrant / Weaviate / Milvus** | Separate server processes. An offline single-user laptop tool should not require a running database daemon. |
| **FAISS** | Excellent ANN, but a large binary dependency, no persistence story of its own, and it solves a scale problem Hearth doesn't have. |
| **DuckDB** | Great analytics engine, but Hearth's workload is point lookups, FTS and a matmul — not analytics — and it would sit *next to* SQLite rather than replacing it. |
| **Postgres + pgvector** | A server. Ends the "one file, zero setup" property. |

---

## 6. D4 — Parsing

**Chosen: `py-tree-sitter` with pinned grammar wheels from PyPI, plus tag queries vendored in the repository.**

### 6.1 The three parts

1. **`tree-sitter`** — the Python binding to the C runtime.
2. **Grammar wheels, pinned exactly**: `tree-sitter-python`, `tree-sitter-javascript`, `tree-sitter-typescript`
   (which provides TS and TSX) in the MVP; `tree-sitter-go`, `tree-sitter-rust`, `tree-sitter-java` behind the
   `langs-extra` extra in Phase 2 (`project-structure.md` §6).
3. **Vendored `tags.scm` queries** under `src/hearth/indexing/queries/<lang>/` (`system-design.md` §6.3), with
   `scripts/update_tags_queries.md` documenting the manual, reviewed update procedure.

### 6.2 Why runtime grammar downloaders are rejected

This is the question `system-design.md` §6.3 defers here. The older and still-common tree-sitter flow builds
grammars by cloning grammar repositories and compiling them with `Language.build_library()` at first use. That is
disqualifying three times over:

1. **It performs network I/O at runtime** — a direct violation of the product's central guarantee
   (`system-design.md` §3.3: "Zero egress. Verifiable by tests").
2. **It requires a C compiler on the user's machine**, turning a wheel install into a build.
3. **It is non-deterministic.** A grammar fetched today differs from one fetched last month; chunk boundaries move;
   every embedding in the index is silently invalidated, along with every chunker snapshot test.

Pinned wheels give the opposite of all three: no network, no compiler, and a version number in `uv.lock`.

### 6.3 Why the queries are vendored rather than imported

Tag queries decide what counts as a definition, a reference and an import (`system-design.md` §6.6) — they are the
symbol graph's schema. Vendoring them means:

- **They are reviewable.** A query is code; it belongs in the diff.
- **They are stable.** An upstream query change can't silently alter symbol extraction.
- **They can be extended.** Hearth needs signature capture and export flags that upstream tags files don't always
  provide.
- **Upstream licences and provenance are recorded once**, in the vendoring procedure, rather than assumed.

### Rejected

| Alternative | Why not |
|---|---|
| **`tree-sitter-language-pack` / `tree-sitter-languages`** | Convenient — one dependency, ~100 grammars. But it bundles far more than Hearth supports, couples the grammar ABI to a third package's release cadence, and supplies its own unvetted queries. Hearth needs 4 grammars at MVP with reviewed queries, and pinning each one is more precise. |
| **Runtime `build_library()` from cloned grammars** | §6.2. |
| **universal-ctags** | An external binary, weaker structure (no full AST for chunking), and one more thing to install per platform. Tree-sitter gives symbols *and* the chunk boundaries from one parse. |
| **LSP servers as the primary source** | Far more precise, and it's the Phase 3 addition (`implementation-roadmap.md` Phase 3, `future-extensions.md` §2.3). Rejected as the *primary* mechanism because it requires per-language server installation and management, adds seconds of startup per language, and cannot index a repository the user hasn't configured a toolchain for. Tree-sitter always works. |
| **Language-specific parsers** (`ast`, `@babel/parser`, …) | Python's `ast` is better for Python only. Hearth needs one uniform mechanism across 5+ languages, including files with syntax errors — which tree-sitter parses anyway, and which `ast` refuses. |

---

## 7. D5 — Retrieval Libraries

**Chosen: nothing new.** The hybrid pipeline (`system-design.md` §7) is built from what is already present.

| Stage | Implementation | Dependency |
|---|---|---|
| Lexical / BM25 | **SQLite FTS5 `bm25()`** with per-column weights | stdlib |
| Symbol and path search | Indexed SQL over `symbols` / `files` | stdlib |
| Dense | `NumpyVectorIndex` (§5.3) | numpy |
| Fusion | **Weighted Reciprocal Rank Fusion**, ~40 lines, plus boosts and penalties | — |
| Graph expansion | Parent skeletons, callee signatures, caller hints over `refs`/`imports` | stdlib |
| Repo map ranking | **Personalized PageRank** (`system-design.md` §7) | networkx |
| Packing | Diversity, range merging, budgeted ordering, citation rendering | — |
| Rerank (optional, off) | Listwise LLM rerank via `format` JSON schema | — |

**`networkx`** is the one retrieval-specific dependency. It exists for `pagerank` with a personalization vector on
the file graph, where a hand-rolled power iteration would be a correctness liability for no real gain. It is
imported lazily (`project-structure.md` §4) because it is slow to import and is not needed on the chat path.

### Rejected

| Alternative | Why not |
|---|---|
| **`rank_bm25`** | Requires holding the corpus in memory and reimplementing what FTS5 already does inside the database Hearth already queries, transactionally. |
| **`sentence-transformers`** | Pulls a multi-gigabyte torch install, duplicates the inference stack Ollama already provides, and would compete for the same VRAM. Embeddings go through `/api/embed` (§8). |
| **A cross-encoder reranker as a dependency** | Same torch problem. When reranking is enabled it is done by the chat model with structured output, behind a per-profile flag, only if the eval shows a ≥5 point recall@5 gain (`system-design.md` §18). |
| **LlamaIndex / LangChain retrievers** | The retrieval pipeline *is* the product's differentiator (`system-design.md` §1.3). Hiding fusion weights, chunk boundaries and packing behind a framework makes the thing that most needs tuning the hardest thing to tune. |
| **`scikit-learn`** (for similarity, clustering) | A large dependency for operations that are one NumPy line each. |
| **`igraph` / `graph-tool`** | Faster than networkx, but both are compiled dependencies, and PageRank on a repo-sized file graph is not a bottleneck. |

---

## 8. Embeddings

**Chosen: Ollama's `/api/embed`, default model `qwen3-embedding:0.6b`.**

No embedding library is a Hearth dependency. The embedder (`indexing/embedder.py`) is batching, caching and
scheduling logic over the provider protocol:

- **Batched** 16–64 items per request, adaptive to latency, with bounded concurrency of 1–2 because Ollama
  serializes per model (`system-design.md` §6.7).
- **Cached by `(embedding_model_id, dimensions, sha256(embedded_text))`** — unchanged chunks are never re-embedded,
  including across branch switches and file moves.
- **Resumable**, committed per batch, so Ctrl+C is safe.
- **Model-specific prompt templates from the profile**, not from code. Qwen3-Embedding takes an instruction prefix
  on queries only; other models define their own (`system-design.md` §6.7).
- **Matryoshka `dimensions`** where the model supports it, with re-normalization after truncation.
- **Placement** (`embed_placement = "auto" | "gpu" | "cpu"`): bulk indexing on GPU, interactive query embedding
  pinned to CPU so the chat model is never evicted from a 6 GB card (`system-design.md` §6.7).

The embedding model and dimensions are recorded in `meta`; changing either invalidates vectors while keeping the
lexical index (`system-design.md` §6.7, §16).

Model selection per tier is `model-recommendations.md` §6, not this document. The default is the same on every tier
including the dev laptop, and the acceptance bar is **hybrid recall@10 ≥ 0.80** on the fixture eval sets
(`implementation-roadmap.md` M2).

---

## 9. D6 — Agent Framework

**Chosen: a custom loop, about 400 lines, with Pydantic v2 for validation. No agent framework.**

### 9.1 Why

`system-design.md` D6 puts it in five words: **the loop is the product.** Everything that makes Hearth work on
local models is a property of the loop itself:

- **Prefix-cache-stable prompt layout** (`system-design.md` §9.2) — static system prompt, then project
  instructions, then append-only history, then the current user message carrying the retrieved context. A framework
  that reorders or re-renders messages destroys the KV cache, and on a CPU-split prefill that is the difference
  between seconds and minutes.
- **Approvals are mid-loop, asynchronous and blocking** (`system-design.md` §5.9). `ApprovalRequested` suspends
  tool execution until an `ApprovalResponse` arrives, and edited arguments are re-prepared and re-evaluated by
  policy. No framework models this.
- **Tolerant tool-call parsing** (§9.2 below) — small models emit tool calls as text, invent arguments, and loop.
  The recovery path is model-profile-specific.
- **Step limits, retry budgets and loop detection** scale with a per-model reliability profile
  (`system-design.md` §1.1, `core/limits.py`).
- **Context budgeting and compaction** are first-class (`system-design.md` §9), not a callback.

### 9.2 Validation and tool-call parsing

**Pydantic v2** is used for every boundary object: events and commands, tool `Args`, config, model profiles, the
plan schema (`project-structure.md` §4). Tool schemas are generated from the `Args` models, so the schema the model
sees and the schema that validates its call can't drift. `model_config = ConfigDict(extra="forbid")` on tool args
makes invented parameters a validation error rather than a silent no-op (`safety-and-tool-use.md` §2.3).

`llm/tool_call_parser.py` reads native `tool_calls` first, then falls back tolerantly to common text formats —
bare JSON objects, `<tool_call>…</tool_call>` blocks, fenced JSON — and validates before accepting
(`system-design.md` §5.4). Failures return **corrective** error messages the model can act on, against a retry
budget.

### Rejected

| Alternative | Why not |
|---|---|
| **LangGraph** | Real graph semantics and checkpointing, but a large dependency tree, and its state/message handling fights the byte-identical-prefix requirement. Debugging a local model's bad tool call through a framework's abstraction is worse than debugging 400 lines. |
| **PydanticAI** | The closest fit philosophically, and it shares the Pydantic validation approach. Rejected because approvals-as-events, per-profile fallback parsing and cache-stable layout would all require fighting or bypassing it — leaving a framework that provides little. |
| **smolagents** | Code-execution-centric agents are the wrong safety model here: Hearth's entire design is preview-then-approve per discrete tool call (`safety-and-tool-use.md` §3). |
| **AutoGen / CrewAI** | Multi-agent orchestration. Hearth runs one model at a time on a 6 GB GPU; sub-agents are a `future-extensions.md` §3.1 item. |
| **LlamaIndex agents** | Same objection as its retrievers (§7): the parts Hearth most needs to control are the parts it abstracts. |
| **OpenAI Agents SDK / provider-native agent loops** | Assume a frontier hosted model with reliable tool calling — precisely the assumption Hearth cannot make. |

---

## 10. D7 — Edit Engine

**Chosen: `old_string` / `new_string` search-and-replace, implemented directly. Stdlib only.**

| Concern | Implementation |
|---|---|
| Matching | Exact → normalized (line endings, trailing whitespace) → indentation-insensitive with re-indentation, flagged `FUZZY-MATCH` (`system-design.md` §10.1) |
| No-match feedback | **`difflib`** returns the closest candidate region so the model can retry |
| Syntax guard | Re-parse with tree-sitter; new `ERROR`/`MISSING` nodes raise a `PARSE-ERRORS-INTRODUCED` badge (`safety-and-tool-use.md` §7.2) |
| Diff preview | **`difflib.unified_diff`**, rendered by Rich (§13) |
| Atomic write | Temp file in the same directory → **`os.replace`**; mode, encoding, BOM and dominant line ending preserved (`system-design.md` §10.1) |
| Hashing | **`hashlib.blake2b`** for file content and change detection; `sha256` for blobs and embed-text cache keys (`system-design.md` §13) |

`difflib` is stdlib and adequate: it is used for *hints to the model* and for *display to the human*, never to
apply changes.

### Rejected

| Alternative | Why not |
|---|---|
| **Unified-diff application** (`patch`, `unidiff`, `whatthepatch`) | Small models produce diffs with wrong line numbers and wrong hunk headers constantly. Search-replace has no line numbers to get wrong. This is the single most reliability-critical format choice in the tool set. |
| **Whole-file rewrite** | Reliable but expensive in tokens on a 12K context, and it makes every edit a full-file diff to review. Retained only as a fallback for small files and for new files (`write_file`). |
| **`rapidfuzz` / `Levenshtein`** for fuzzy matching | A dependency for a bounded problem `difflib.SequenceMatcher` already solves. Revisit only if fuzzy matching becomes a measured bottleneck on large files. |
| **AST-level rewriting** (`libcst`, `jscodeshift`) | Precise and language-specific — one dependency and one implementation per language, and it can't express edits to comments, strings, Markdown or config. A `future-extensions.md`-class idea for specific refactor workflows, not the general edit path. |

---

## 11. Testing Stack

The test suite must be **fast, deterministic, offline, and adversarial about safety** (`system-design.md` §17).

| Dependency | Purpose |
|---|---|
| **pytest** | Runner |
| **pytest-asyncio** | `asyncio_mode = "auto"` — the core is async throughout |
| **pytest-socket** | **The egress guard.** Blocks every socket except loopback for the entire run |
| **hypothesis** | Property tests where adversarial inputs matter: path jail against generated `..`, symlinks, absolute paths and Unicode tricks; command classifier against generated metacharacter injections; edit-engine round-trips |
| **syrupy** | Snapshots for chunker output per language, assembled prompts, and repo maps |

### 11.1 pytest-socket is a product requirement, not a convenience

`system-design.md` §3.3 makes zero egress the top quality attribute, "verifiable by tests". The configuration in
`project-structure.md` §6 is therefore part of the specification:

```toml
addopts = "--disable-socket --allow-hosts=127.0.0.1,::1 -q -m \"not live and not slow\""
```

`implementation-roadmap.md` lists "pytest-socket green: no egress beyond loopback" as an MVP exit criterion, and
`safety-and-tool-use.md` §15 lists it in the security checklist. **Do not disable it**, and do not add
`--allow-hosts` entries.

### 11.2 Markers

| Marker | Meaning |
|---|---|
| `live` | Requires a running local Ollama with `qwen3.5:4b`. Excluded by default. Prompts stay under ~8K tokens, run **sequentially**, and never load two chat models at once |
| `slow` | Large synthetic indexes and benchmarks. Excluded by default; run deliberately on AC power |

### 11.3 Determinism without a model

`ScriptedProvider` (§4.2) replays scripted responses, including tool calls and fake deterministic embeddings, so
the indexer, retrieval, agent loop and approval flow are all testable with no Ollama running
(`implementation-roadmap.md` §0 principle 4). A test frontend auto-responds to `ApprovalRequested` events
(`system-design.md` §5.1). This is what keeps CI offline and the inner loop fast.

### 11.4 Fixture repositories

All development and testing runs against `tests/fixtures/repos/` — `py_small`, `ts_small`, `polyglot`, and
`malicious` (instruction-injection comments, symlink escapes, fake secrets). Agent tools are **never** run against
real repositories in tests; fixtures are copied to tmp directories first (`project-structure.md` §5).

### 11.5 Test categories

Beyond unit and integration: a **security suite** asserting zero side effects without approval and headless
fail-closed behaviour (`tests/security/`, `safety-and-tool-use.md` §15), and the **eval harness** (§18) which is
model-in-the-loop and runs on demand, not in CI.

---

## 12. Lint, Typing and Architecture Enforcement

| Tool | Role |
|---|---|
| **ruff** | Lint and format. `line-length = 110`, `target-version = "py312"`, rules `["E","F","I","B","UP","SIM","ASYNC","S","PTH","RUF"]` (`project-structure.md` §6) |
| **mypy** | `--strict` for `hearth.core.*`, `hearth.safety.*`, `hearth.tools.*`, `hearth.llm.*`; standard mode elsewhere |
| **import-linter** | Layer contracts — the architecture is executable, not aspirational |
| **pre-commit** | Runs all of the above plus pytest before every commit |

Two ruff selections are deliberate: **`S`** (flake8-bandit) because this project executes commands and touches
paths for a living, and **`PTH`** because `pathlib` use is uniform in the path-safety code.

The three `import-linter` contracts are specified verbatim in `project-structure.md` §2 — the layer ordering, plus
`tools-no-llm` (tools never call models) and `policy-pure` (the policy engine has no I/O dependencies). These
encode two of the design's load-bearing invariants; a violation is an architecture bug, not a style issue.

**One further guard** that isn't a tool but a test: a grep-based CI check fails if `open(`, `Path.write_text` or
`os.remove` appear in `tools/` outside the approved helper, because every filesystem access must go through
`safety.paths.resolve_in_workspace()` (`project-structure.md` §4).

### 12.1 The `S` ruleset and process execution

Ruff's `S` rules fire constantly on code whose entire purpose is executing processes. `S603`
(subprocess-without-shell-equals-true) and `S607` (start-process-with-partial-path) hit nearly every line of
`safety/sandbox/subprocess_runner.py` and `git/runner.py`. The policy distinguishes rules that flag **a module's
job** from the rule that flags **a genuinely dangerous call**.

**Per-file-ignores for the modules allowed to spawn processes:**

```toml
[tool.ruff.lint.per-file-ignores]
"src/hearth/safety/sandbox/*_runner.py" = ["S603", "S607"]
"src/hearth/git/runner.py"              = ["S603", "S607"]
"tests/**"                              = ["S101"]          # assert is the point
```

**`S602` (`shell=True`) is never per-file-ignored.** There is exactly one call site — the `/bin/sh -c` path, taken
only after a human approves a command carrying the `SHELL` badge (`safety-and-tool-use.md` §8.2) — and it carries a
per-line `# noqa: S602` with a justification comment. One dangerous call, one annotation, reviewed on its own.

**The allowlist is enforced, not merely configured.** A CI test, following the same grep-guard pattern that keeps
filesystem access inside `resolve_in_workspace` (§12 above), fails if `subprocess`, `os.system`, `os.exec*`,
`Popen` or `create_subprocess_*` appears anywhere outside those modules. That is the property actually worth
having: process execution is confined to two reviewed files, and adding a third is a visible change to both
`pyproject.toml` and the guard test.

Per-line `noqa` throughout was the alternative. Rejected: in a module where every line spawns a process, the
annotations are noise, and noise trains reviewers to skim exactly the code that most needs reading.

### Rejected

| Alternative | Why not |
|---|---|
| **black + isort + flake8 + pylint** | Four tools, four configs, slower. Ruff covers all of it. |
| **pyright** | Excellent, but a Node dependency for a Python-only project. Mypy's strict mode is sufficient and its plugin story with Pydantic is well-trodden. |
| **Hand-written layering docs** | Documented layers rot silently. Contract violations should fail CI. |

---

## 13. D9 — CLI and Rendering

**Chosen: a headless core with a typed event protocol; the CLI is one frontend among several.**

| Dependency | Role |
|---|---|
| **Typer** | The command surface: `init`, `index`, `chat`, `run`, `search`, `doctor`, `trust`, `eval`, `bench`, `resume`, `undo`, `stats`, `audit`, `gc` |
| **Rich** | Event rendering — streaming Markdown, collapsed thinking, tool panels, coloured diffs, approval panels, risk badges, progress, the per-turn stats line |
| **prompt_toolkit** | The REPL input line: history, `/command` and `@path` completion from the index, multiline editing, Ctrl+C semantics |

### 13.1 Why the split matters more than the libraries

`system-design.md` D9 and §5.1: all interaction flows through typed Pydantic events (core → frontend) and commands
(frontend → core). The CLI renders them; a future IDE extension receives the same objects as JSON-RPC
notifications (§14). `project-structure.md` §2 forbids `cli`/`server` from importing `tools`, `safety`, `llm` or
`storage` at all.

The payoff is testability as much as portability: because approvals are just events, a test frontend can
auto-respond and the entire approval flow becomes deterministically testable without a terminal.

### 13.2 Inline streaming, not a full-screen app

`implementation-roadmap.md` M3 is explicit: **inline streaming, not full-screen.** Output scrolls in the user's
terminal and stays in their scrollback, copyable and greppable, alongside their other work.

### Rejected

| Alternative | Why not |
|---|---|
| **Textual** | A genuinely good TUI framework, and the natural choice *if* Hearth wanted a full-screen app. It doesn't: taking over the screen loses scrollback and makes copy-paste awkward for a tool whose output is code. Reconsider as an optional alternate frontend — the event protocol makes it additive. |
| **click** (directly) | Typer is a thin typed layer over click; using click directly means writing the type plumbing by hand. |
| **argparse** | Free, but the command surface is large and the help output matters for a tool with `doctor` and `trust` flows. |
| **curses / blessed** | Too low-level; Rich already handles the terminal capability matrix. |
| **A web UI first** | Phase 3 (§14). A terminal-first tool for a terminal-first user is the fastest path to daily dogfooding (`implementation-roadmap.md` Cross-Cutting Practices). |

---

## 14. D10 — IDE and Remote Protocol

**Chosen: JSON-RPC 2.0 over stdio (Phase 2), with an optional loopback WebSocket (Phase 3).**

### 14.1 Why stdio

`system-design.md` D10: **stdio has no open port**, and therefore no localhost CSRF exposure, no DNS-rebinding
attack surface, and no authentication to get wrong. The editor spawns `hearth serve --stdio` as a child process;
the OS provides the trust boundary. For a tool whose primary promise is privacy, "there is no listening socket" is
a much stronger statement than "the listening socket is authenticated".

The protocol is a direct serialization of the existing event and command models (`server/protocol.py` maps JSON-RPC
method names to core commands and events), so it needs no new framework — `json` from the stdlib plus the Pydantic
models that already exist. **No new runtime dependency.**

Acceptance (`implementation-roadmap.md` I4): a minimal stdio client script can run a chat turn *with approvals*.

### 14.2 The Phase 3 WebSocket

The optional web UI (`server/ws_app.py`) uses **FastAPI + uvicorn**, isolated behind the `server` optional extra so
a CLI-only install never sees them (`project-structure.md` §6). Requirements when it lands
(`implementation-roadmap.md` Phase 3): bind loopback only, require a token, and check `Host`/`Origin` headers —
the defences the stdio path doesn't need.

### 14.3 The VS Code extension

TypeScript, a **thin client over the stdio protocol** (`implementation-roadmap.md` Phase 3). It renders diffs and
approvals natively; it contains no agent logic. Because the core speaks JSON-RPC over stdio, Neovim, JetBrains,
Zed and Emacs clients are all feasible later (`future-extensions.md` §5.2).

### Rejected

| Alternative | Why not |
|---|---|
| **HTTP REST on localhost** | Opens a port. Inherits localhost CSRF and DNS-rebinding concerns, and needs an auth scheme — all to solve a problem stdio doesn't have. |
| **gRPC** | Adds protobuf codegen and a large dependency to a single-machine, single-client protocol. |
| **LSP as the transport** | LSP models documents and diagnostics, not streaming chat with blocking approvals. Hearth *consumes* LSP later for references (`future-extensions.md` §2.3); it shouldn't impersonate one. |
| **MCP as the primary protocol** | Phase 3+ and additive in both directions: a local-stdio MCP *client* wraps external tools into Hearth's policy engine, and an MCP *server* exposes retrieval read-only (`future-extensions.md` §5.3). Remote MCP servers stay disabled by default — they are a privacy boundary. |

---

## 15. Filesystem, Configuration and Platform

| Dependency | Role | Why not stdlib |
|---|---|---|
| **pathspec** | `.gitignore` semantics for the non-git walk path and for `.hearthignore` (`system-design.md` §6.2) | Gitignore matching has many edge cases (negation, anchoring, directory-only patterns); getting it subtly wrong means indexing `node_modules` or missing source files |
| **watchfiles** | Debounced (500 ms) incremental reindexing during a session; handles branch-switch bursts (`system-design.md` §6.8) | Rust-backed, correct across platforms, far fewer spurious events than hand-rolled polling |
| **platformdirs** | Config and data locations on Linux/macOS/Windows (`project-structure.md` §3) | Three platforms, three conventions, plus XDG overrides |
| **tomli-w** | *Writing* TOML — `hearth init` scaffolds `.hearth/config.toml`; `hearth trust` records config hashes | `tomllib` reads only |

**Reading TOML uses stdlib `tomllib`** (Python 3.11+), which is part of why 3.12 is the floor (§2).

Configuration precedence is `defaults.toml` → `~/.config/hearth/config.toml` → `<repo>/.hearth/config.toml`, with
the project layer **trust-aware**: permission-relaxing rules from project config are ignored until `hearth trust`
records the file's SHA-256, while deny rules always apply because they only make things stricter
(`system-design.md` §5.12). The model is VS Code Workspace Trust and direnv's `allow`.

### Rejected

| Alternative | Why not |
|---|---|
| **watchdog** | The historical default, but noisier event streams and more platform-specific quirks. watchfiles is smaller and more predictable. |
| **appdirs** | Unmaintained predecessor of platformdirs. |
| **PyYAML for configuration** | TOML is the Python ecosystem's configuration format, `tomllib` is stdlib, and YAML's implicit typing is a footgun in a file that controls permissions. (YAML *data* for evals is a separate open question — §19.4.) |
| **pydantic-settings** | Config comes from files with a three-layer trust-aware merge, not from environment variables. The loader is ~100 lines and the precedence logic is security-relevant enough to own outright. |
| **Hand-rolled gitignore matching** | Tried by many, correct by few. |

---

## 16. Process Execution and Sandboxing

**Chosen: stdlib `asyncio.create_subprocess_exec` behind a `ProcessRunner` protocol. OS sandboxing is additive and later.**

No dependency. The MVP `SubprocessRunner` (`safety-and-tool-use.md` §8.2) is stdlib, and the settings *are* the
security control:

| Aspect | Setting |
|---|---|
| Invocation | `create_subprocess_exec(*argv)`; `/bin/sh -c` **only** after explicit approval of a shell-flagged command |
| stdin | `DEVNULL` — commands that want input fail fast instead of hanging |
| Environment | Allowlist (`PATH`, `HOME`, `LANG`, `LC_*`, `TERM=dumb`, `VIRTUAL_ENV`, configured toolchain vars) plus a scrub denylist (`*_TOKEN`, `*_SECRET`, `*_KEY`, `*PASSWORD*`, `AWS_*`, `GITHUB_*`, `OPENAI_*`, `ANTHROPIC_*`, `OLLAMA_API_KEY`, `SSH_AUTH_SOCK`, `GPG_AGENT_INFO`) |
| Non-interactive | `CI=1`, `GIT_TERMINAL_PROMPT=0`, `PAGER=cat`, `GIT_PAGER=cat`, `NO_COLOR=1`, `PIP_NO_INPUT=1`, `DEBIAN_FRONTEND=noninteractive` |
| Process group | `start_new_session=True`; on timeout or cancel, `SIGINT` → 5 s → `SIGTERM` → 3 s → `SIGKILL` to the group |
| Timeout | 120 s default, 600 s max; `run_tests` 300 s |
| Resource hints | Optional `ulimit`-style limits via `resource` on POSIX |

Command classification before execution uses **`shlex`** (stdlib) for POSIX splitting plus metacharacter detection
(`safety-and-tool-use.md` §8.1).

### 16.1 Sandboxing is external binaries, not Python packages

`safety-and-tool-use.md` §12:

| Level | Phase | Mechanism |
|---|---|---|
| **L0** | MVP | Policy + approval + checkpoints + scrubbed env + timeouts + process groups |
| **L1** | Phase 3 | Linux/WSL2: **`bwrap`** (`--ro-bind / /`, `--bind <workspace>`, `--tmpfs /tmp`, `--unshare-net`, `--unshare-pid`, `--die-with-parent`) · macOS: **`sandbox-exec`** profile |
| **L2** | Phase 3+ | Container mode (**Podman/Docker**): workspace bind-mounted, network `none`, non-root |
| **L3** | Future | Disposable git worktree + container per task |

Each is an external binary invoked behind the same `ProcessRunner` protocol — `bwrap_runner.py`,
`seatbelt_runner.py` (`project-structure.md` §1). Nothing enters `pyproject.toml`.

**The WSL2 caveat is load-bearing:** Windows interop can escape a Linux sandbox, so sandboxed runs require interop
disabled (`safety-and-tool-use.md` §16.2). Until then, `*.exe` and `/mnt/<drive>/…` executables are classified
`WIN-INTEROP` — always Ask, never grantable.

### Rejected

| Alternative | Why not |
|---|---|
| **Container-first / sandbox-first (D8)** | Stronger isolation, but it addresses the *third* most likely harm. The most likely are a wrong edit, a destructive command and an accidental secret commit (`system-design.md` §1.4) — all of which policy, preview, approval and checkpoints address immediately, at a fraction of the complexity. Sandboxing is defence in depth, added on top, never instead. |
| **`sh=True` convenience** | Shell interpretation is the attack surface. It happens only after a human approves a command explicitly flagged as using a shell. |
| **Docker SDK for Python** | The container runtime is invoked as a binary through the same runner protocol; an SDK would add a dependency for `subprocess`. |

---

## 17. Git and External Binaries

**Chosen: the `git` CLI as a subprocess, hardened.**

### 17.1 Why the CLI

1. **Behaviour parity.** The user's `.gitconfig`, hooks, credential helpers, LFS and worktrees all behave exactly as
   they do in their own terminal. A library reimplements a subset and diverges at the edges.
2. **The hardening is expressible.** `git/runner.py` invokes (`safety-and-tool-use.md` §9.1):

   ```text
   git -c core.fsmonitor=false \
       -c core.hooksPath=<unchanged for commit; /dev/null for read ops> \
       -c diff.external= \
       -c core.pager=cat \
       -c color.ui=false \
       --no-pager <subcommand> [--no-ext-diff --no-textconv for diff/show/log -p]
   ```

   These flags matter: repository-local git configuration can make an ordinary *read* command execute programs, via
   fsmonitor hooks, external diff drivers and textconv filters. A repository obtained as an archive can carry that
   config. Read operations disable those mechanisms; **commits deliberately keep hooks enabled** and never use
   `--no-verify` (`safety-and-tool-use.md` §9.2).
3. **Discovery reuse.** `git ls-files -co --exclude-standard -z` is the fastest correct file discovery available,
   and it is already the indexing fast path (`system-design.md` §6.2).

Output parsing is confined to `git/porcelain.py` using stable machine formats (`status --porcelain=v2`, `-z`
separators, explicit `--format`).

### 17.2 External binaries

| Binary | Required? | Role |
|---|---|---|
| **git** | Effectively yes | Discovery, read tools, write tools. Hearth degrades to a directory walk without it |
| **ripgrep (`rg`)** | Optional | The `grep` tool prefers it, with a pure-Python fallback (`system-design.md` §10) |
| **nvidia-smi** | Optional | VRAM headroom checks in `hearth doctor` |
| **bwrap / sandbox-exec / podman** | Phase 3 | §16.1 |

`hearth doctor` reports on `git` and `rg` presence (`system-design.md` §15).

### Rejected

| Alternative | Why not |
|---|---|
| **GitPython** | Shells out for much of its work anyway, adds a dependency, and has a history of CVEs in its argument handling — a poor trade for a security-sensitive path. |
| **pygit2 (libgit2)** | Fast and proper, but a compiled dependency, and it *doesn't* run hooks or use credential helpers the way the user's git does. The hardening flags above have no libgit2 equivalent. |
| **dulwich** | Pure-Python git, but a partial reimplementation — divergence risk on exactly the operations where fidelity matters. |

---

## 18. Observability, Logging and Evaluation

### 18.1 Local-only by construction

`system-design.md` §15: observability is local only. There is no telemetry, no crash reporting, no update check.
`hearth doctor` warns if Ollama's own cloud features appear enabled.

| Surface | Implementation |
|---|---|
| Per-turn stats line | Rich; prompt tokens, cached tokens, prefill ms, gen tok/s, context used/budget — from the `ContextStats` event |
| App logs | stdlib `logging` to `~/.local/share/hearth/logs/`, rotated |
| `--debug` prompt dumps | **Off by default**, with its privacy implications stated at the point of use |
| **Audit log** | `~/.local/share/hearth/audit/YYYY-MM.jsonl` — append-only (`O_APPEND`), one JSON object per line, **fsync per record**, secret-redacted before writing (`safety-and-tool-use.md` §13) |
| `hearth search --explain` | Retrieval score breakdowns |

The audit log uses stdlib `json` deliberately: append-only JSONL with per-record fsync is ~30 lines, and the format
must stay readable by `jq` and by a human five years from now.

### 18.2 The eval harness

Model-in-the-loop, run on demand, never in CI (`system-design.md` §17.2):

| Command | Measures |
|---|---|
| `hearth eval retrieval` | recall@5, recall@10, MRR against question → expected-files/symbols sets |
| `hearth eval tasks` | Success rate, steps, tool errors, tokens, wall time — each task runs inside a **disposable copy** of a fixture repo with approvals auto-granted |
| `hearth bench` | Prefill and generation throughput, TTFT, memory headroom at the configured `num_ctx` |

Eval *data* lives in `evals/` separate from the harness code in `src/hearth/evals/` (`project-structure.md` §1).
Results are committed to `docs/benchmarks/` and `docs/adr/`. `implementation-roadmap.md` requires a monthly model
review, since open model families ship frequently.

**Eval data stays YAML, parsed by `pyyaml` under the `evals` optional extra.** Question sets are nested,
hand-authored and reviewed in diffs — 25+ entries per fixture repo with expected files and symbols
(`implementation-roadmap.md` M2) — and YAML is the readable format for that. Confining it to an extra keeps a normal
`uv tool install hearth` at the §19.1 dependency count, since nobody running `hearth chat` needs a YAML parser.

Three consequences for the implementation:

1. **Import `yaml` lazily, inside the eval harness only.** Never at module import time, and never anywhere else in
   the package. A missing extra must not break `hearth --version`.
2. **`hearth eval` degrades with a clear message**, not an `ImportError` traceback: *"Eval data requires the evals
   extra. Install with `uv sync --extra evals`."* A CLI smoke test asserts this holds with the extra absent.
3. **Use `yaml.safe_load`, always.** Eval files are repository data, and `yaml.load` with the default loader
   constructs arbitrary Python objects. A ruff `S506` violation here is not one to silence.

The YAML dependency is scoped to eval data. It never becomes a configuration format — configuration is TOML
(§15), and `.hearth/config.toml` controls permissions.

### Rejected

| Alternative | Why not |
|---|---|
| **OpenTelemetry / Prometheus** | Built for distributed systems and for exporting. Both assumptions are wrong here; an exporter is an egress path waiting to be misconfigured. |
| **Sentry / any crash reporting** | Egress. Disqualifying. |
| **structlog** | Genuinely nice, but stdlib `logging` plus a JSONL audit writer covers both needs, and the audit format is security-relevant enough to own. |
| **MLflow / Weights & Biases / LangSmith** | Experiment tracking with a server and an account. Eval results are a table in a Markdown file committed to the repository. |
| **`promptfoo` / generic LLM eval frameworks** | Hearth's evals are repo-specific: retrieval recall against expected files, and agent tasks verified by running a command in a disposable repo copy. The harness is small and the metrics are domain-specific. |

---

## 19. Complete Dependency Inventory

### 19.1 Runtime

| Package | Purpose | Section |
|---|---|---|
| `ollama` | Inference client (native API, async) | §4 |
| `pydantic` (>=2) | Events, tool args, config, profiles, plan schema | §9.2 |
| `typer` | CLI command surface | §13 |
| `rich` | Event rendering, diffs, approval panels | §13 |
| `prompt_toolkit` | REPL input, completion, history | §13 |
| `tree-sitter` | Parser runtime | §6 |
| `tree-sitter-python` | Grammar (pinned) | §6 |
| `tree-sitter-javascript` | Grammar (pinned) | §6 |
| `tree-sitter-typescript` | Grammar (pinned; TS + TSX) | §6 |
| `numpy` | Vector search, fusion math | §5.3, §7 |
| `networkx` | Personalized PageRank for the repo map | §7 |
| `watchfiles` | Debounced incremental reindexing | §15 |
| `pathspec` | `.gitignore` / `.hearthignore` semantics | §15 |
| `platformdirs` | Cross-platform config and data paths | §15 |
| `tomli-w` | Writing TOML | §15 |

**12 libraries plus 3 grammar wheels** — on target for `system-design.md` §1.5.

**Optional extras:**

| Extra | Contents | Phase |
|---|---|---|
| `evals` | `pyyaml` — eval data only; see §18.2 | 1 (M2) |
| `langs-extra` | `tree-sitter-go`, `tree-sitter-rust`, `tree-sitter-java` | 2 |
| `server` | `fastapi`, `uvicorn` | 3 |

### 19.2 Development

`pytest`, `pytest-asyncio`, `pytest-socket`, `hypothesis`, `syrupy`, `ruff`, `mypy`, `import-linter`, `pre-commit`
(§11, §12).

### 19.3 Carrying the weight from the stdlib

Worth stating explicitly, because each one is a dependency deliberately *not* taken: `sqlite3` (§5), `asyncio`
(§16), `subprocess`/`shlex` (§16), `difflib` (§10), `hashlib` (blake2b, sha256), `tomllib` (§15), `json` (§14,
§18), `logging` (§18), `pathlib`, `os.replace` (§10), `struct`/`array` for vector BLOB packing.

### 19.4 Gaps found and resolved

Gaps in the design baseline that this document surfaced. All four are now decided; each entry records where the
decision lives, so the resolution is reviewable rather than buried.

1. **Eval data format vs. the dependency list.** `project-structure.md` §1 specified eval data as YAML while no
   YAML parser appeared in the dependency list. **Decided: `pyyaml` under an `evals` optional extra** — eval data
   stays YAML, a default install is unaffected, the harness imports lazily and `safe_load`s (§18.2, §19.1).
   → ADR `0003-pyyaml-evals-extra.md` at scaffolding time, per §1.3.
2. **Project-config commands vs. `hearth trust`.** `.hearth/config.toml` names test and lint commands that become
   argv, and `safety-and-tool-use.md` §5.5 did not say whether trust governs them. **Decided: configured commands
   are data, not permissions** — classified and approved like any other command, with trust granting nothing
   (`safety-and-tool-use.md` §5.6). This also fixed a real hole: the `run_tests` grant key was scoped to the
   *target*, so a session grant would have survived a change to the underlying command. Grant keys now bind the
   resolved argv digest (`safety-and-tool-use.md` §5.4), and a new `PROJECT-CONFIG` badge shows the argv's
   provenance (§6.2).
3. **Ruff `S` ruleset vs. process execution.** **Decided: narrow per-file-ignores for the two modules whose job is
   spawning processes, `S602` never ignored, and a CI grep guard making the allowlist enforceable** (§12.1;
   config in `project-structure.md` §6).
4. **Headless output schema.** **Decided: JSONL over the existing event models under `--json`, with a terminal
   `run_result` object and human-readable text by default** (`safety-and-tool-use.md` §14.1). The summary ships
   with headless mode in M6; the full event stream lands with the stdio server in I4.

---

## 20. Revisit Triggers

Consolidated from `system-design.md` §18 and the sections above.

| Decision | Revisit when |
|---|---|
| NumPy exact vector search (§5.3) | A repo exceeds ~1M chunks, or vector RAM exceeds 1.5 GB → add `sqlite-vec` or LanceDB behind `VectorIndex` |
| Python (§2) | Startup misses < 1.5 s or indexing misses < 2 min for reasons lazy imports can't fix |
| Native Ollama SDK (§4) | The SDK lags the server API on a parameter Hearth needs |
| Name-based symbol graph (§6) | Refactor tasks fail on ambiguous names in the eval suite → LSP references |
| Rerank off by default (§7) | Eval shows ≥5 point recall@5 gain at acceptable latency on a tier |
| No file summaries on Tier 1 (§8) | Conceptual-question recall stays low with embeddings alone |
| CLI-only frontend (§13) | Demand for an editor experience → the stdio protocol already exists |
| L0 sandboxing (§16) | Users want unattended test runs, or `offline.enforce` needs to be real rather than heuristic |
| Default models | Monthly (`model-recommendations.md`); new families ship frequently |
