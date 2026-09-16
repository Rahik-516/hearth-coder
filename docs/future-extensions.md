# Hearth — Future Extensions

This document collects ideas beyond the Phase 3 roadmap. Each entry covers:
- the value,
- the approach that fits Hearth's architecture,
- prerequisites,
- rough complexity (S / M / L / XL), and
- privacy or safety considerations.

The guiding rule stays the same: **offline by default, human-in-the-loop for side effects, measured by evals before becoming a default.**

---

## 1. Prioritization at a Glance

| # | Extension | Value | Complexity | Prerequisites | Suggested order |
|---|---|---|---|---|---|
| 1 | Sub-agents with isolated context (explorer, tester, reviewer) | High | M | Stable runner, compaction | First |
| 2 | Dependency source and offline docs indexing | High | M | Indexer, retrieval namespaces | First |
| 3 | Git worktree task isolation + parallel tasks | High | M | Sandboxing L1/L2 | Early |
| 4 | Project memory and decision log | High | S–M | state.db, AGENTS.md workflow | Early |
| 5 | Inline autocomplete (fill-in-the-middle) | High | L | VS Code extension, FIM-capable model | Mid |
| 6 | Multi-repo / workspace support | Medium–High | M | Index namespaces, repo map federation | Mid |
| 7 | Model cascades and speculative decoding | Medium | M | Profiles, evals, runtime support | Mid |
| 8 | Test impact analysis and verification agents | Medium–High | M | Code graph (LSP ideal), coverage data | Mid |
| 9 | Vision: screenshots, diagrams, UI bugs | Medium | S–M | Multimodal chat models (already common) | Mid |
| 10 | MCP server mode (expose Hearth to other local tools) | Medium | S | stdio protocol | Mid |
| 11 | Architecture knowledge graph + drift detection | Medium | L | Summaries, import graph, LSP | Later |
| 12 | Local fine-tuning / adapters on your codebase | Medium (uncertain) | XL | Eval suite, data pipeline, GPU | Later (research) |
| 13 | Team / LAN mode | Medium | L | Auth, multi-user state, threat model rework | Later |
| 14 | Preference learning from approvals | Medium | M | Audit data, evals | Later |
| 15 | CI log triage, pre-commit review bot | Medium | S–M | Headless mode | Anytime |
| 16 | Notebook, SQL and data-project support | Niche–Medium | M | Chunkers for `.ipynb`/SQL | As needed |

---

## 1a. Feasibility on the Current Dev Laptop (RTX 3060 Laptop 6 GB · 16 GB RAM)

Many extensions depend more on hardware than on code. This table shows what's realistic on your machine today, so you can prioritize work you can actually use and test.

| Extension | On your laptop | Why |
|---|---|---|
| Sub-agents with isolated context | ✅ **Especially valuable** | Run sequentially with the same `qwen3.5:4b`. Isolating exploration from the main conversation is the best way to stretch a 12K window. |
| Dependency source / offline docs indexing | ✅ | CPU and SSD bound; keep an eye on disk usage |
| Project memory, decision log, preferences | ✅ | Negligible compute |
| Multi-repo workspace | ✅ | Retrieval-side feature; indexes stay small |
| Test impact analysis, self-review step | ✅ | Deterministic analysis; the self-review costs one extra short model call |
| MCP server mode, CI log triage, pre-commit review | ✅ | Light workloads |
| Background indexing daemon | ✅ | Embeds on GPU when no chat session is active |
| Vision workflows (screenshots) | ✅ with care | Qwen 3.5 models accept images, but each image consumes a significant share of a 12K context |
| Git worktree task isolation | ✅ sequential · ❌ parallel | One 6 GB GPU can serve one agent at a time |
| Inline autocomplete (FIM) | ⚠️ | Needs its own small model. Can't run beside the chat model in 6 GB, so it would mean switching modes, not running both. |
| Model cascades / speculative decoding | ⚠️ | Two chat models don't fit in VRAM together, and swapping costs seconds per switch. A tiny draft model may fit; verify with `ollama ps`. |
| Architecture knowledge graph (summaries) | ⚠️ | Summarizing many files with a 4B model is slow; run as an overnight batch on AC power |
| Containers for sandboxing (L2) | ⚠️ | Docker/Podman inside WSL2 costs RAM you don't have much of. bubblewrap (L1) inside WSL2 is lighter. |
| Embedding model adaptation | ⚠️ | A 0.6B embedder with small batches is plausible, but verify memory before investing |
| Local fine-tuning of chat models | ❌ (not recommended here) | 6 GB VRAM is borderline even for adapters on small models; retrieval plus `AGENTS.md` conventions is the better investment |
| Team / LAN server | ❌ as the server · ✅ as a client | A stronger machine on your network could serve models to this laptop, but code then leaves the laptop, so the privacy tradeoff must be explicit (§7.5) |

**If you upgrade later:** a 16 GB+ desktop GPU or a 36 GB+ Apple Silicon machine turns most ⚠️ rows into ✅. The design needs no changes, only config, as covered in `model-recommendations.md`.

---

## 2. Context and Codebase Intelligence

### 2.1 Dependency source indexing (offline API knowledge)
**Problem:** without internet, the model can't look up library documentation, and models hallucinate APIs, especially for newer library versions.

**Approach:**
- Build a separate, read-only **"deps" index namespace** from installed dependency sources: `.venv/lib/**/site-packages` public modules, `node_modules/*/dist/*.d.ts` type declarations, Go module cache, Cargo registry sources.
- Index **signatures and docstrings only** (skeleton chunks), not full bodies, to keep the index small.
- Retrieval gets a `scope` parameter (`project` | `deps` | `all`). Query analysis routes "how do I use X from library Y" to `deps`, and `find_symbol` can fall back to `deps`.

**Complexity:** M. **Privacy:** fully local.

### 2.2 Offline documentation packs
**Approach:**
- Let users import documentation dumps while online, e.g., DevDocs-style exports, framework docs sites saved as Markdown, or generated `pydoc`/TypeDoc output, into a `docs` namespace with version tags.
- `hearth docs add <name> <path>` indexes them.
- Chunks carry `library@version` metadata, and retrieval prefers versions that match the project's lockfile.

**Complexity:** M.

### 2.3 LSP-powered precision (beyond the Phase 3 basics)
- **Semantic operations:** rename, call hierarchy, type hierarchy and implementations.
- **Diagnostics after every edit**, fed to the model before tests run (cheaper and faster than a full test cycle).
- **"Safe rename" workflow:** the LSP computes all edits, and Hearth shows one batch approval.

**Complexity:** L (per-language server management is the cost).

### 2.4 Architecture knowledge graph and drift detection
- **Graph:** a persistent graph of components (packages/modules), their responsibilities (from summaries), dependencies (import graph), entry points and data stores.
- **Uses:**
  - generate and refresh `docs/ARCHITECTURE.md` and Mermaid diagrams,
  - answer "what would break if I change X" by combining the graph with references, and
  - **drift detection:** flag new imports that violate declared layering rules. Users declare rules much like Hearth's own import-linter contracts, and a `/review` check reports violations.

**Complexity:** L.

### 2.5 Multi-repo and workspace support
- **Workspace file:** `hearth.workspace.toml` lists repositories (e.g., backend, frontend, shared protos).
- **Indexes:** each repository keeps its own index. Retrieval federates queries and fuses results across indexes with RRF plus a repo boost.
- **Cross-repo edges:** matched from API contracts, such as OpenAPI specs, protobuf definitions, GraphQL schemas and shared type packages.
- **Tools and trust:** tools gain an explicit `repo` argument, and the workspace jail becomes a set of roots, each with its own trust record.

**Complexity:** M. **Safety:** writes still confined per repo root, and approvals show which repo is affected.

### 2.6 Scale backends
When a monorepo exceeds about 1M chunks, add a `SqliteVecIndex` (in-DB approximate search) or `LanceDbIndex` behind the existing `VectorIndex` protocol. Validate by requiring equal retrieval eval scores at lower memory. **Complexity:** S–M.

---

## 3. Agents and Reasoning

### 3.1 Sub-agents with isolated context
**Why this matters more locally than in the cloud:** small context windows fill quickly during exploration. A sub-agent explores with its *own* context and returns a compact report, so the main conversation stays lean and prefix-cache-friendly.

| Sub-agent | Tools | Returns |
|---|---|---|
| **Explorer** | Read-only search/read/symbols | ≤ 1K-token findings with citations |
| **Test runner / fixer** | run_tests, read, edit (approval) | Pass/fail summary + diffs applied |
| **Reviewer** | Read-only + git diff | Findings list with severity and citations |
| **Doc writer** | Read + write_file (approval) | Document path + outline |

**Implementation:**
- Add a `delegate(agent, task, budget)` meta-tool. It runs a nested `run_turn` with a separate history and tool subset, and returns only the final report.
- Nested approvals surface through the same event bus, tagged with the sub-agent name.

**Complexity:** M.

### 3.2 Parallel tasks in git worktrees
- Each task gets `git worktree add .hearth-worktrees/<task>`, its own session and index overlay (base index plus a delta for changed files), and ideally its own sandbox (L2).
- The user reviews the resulting branch diff and merges manually.
- **Hardware note:** parallel agents compete for the same GPU, and Ollama processes requests for a model sequentially unless parallel slots are configured, which multiplies KV memory. This is realistic mainly on Tier 4 or with a smaller model per worker.

**Complexity:** M.

### 3.3 Model cascades and routing
- **Cascade:** a fast model (e.g., a 4–12B or small-active MoE) handles query rewriting, summaries, commit messages, simple edits and reranking. The main model handles planning and hard debugging. Escalate automatically on tool-error rate, failed verification, or low self-consistency.
- **Speculative decoding** using a small draft model of the same family can speed up generation substantially when the runtime supports it. Track Ollama's support. Its MLX engine has already shipped speculative-style acceleration for some models. Expose it through profiles when available.

**Complexity:** M. **Must be eval-driven:** cascades can silently reduce quality.

### 3.4 Verification-first agents
- **Test impact analysis:** map changed symbols to tests that import or call them (code graph, optionally coverage data from a previous `pytest --cov` run) and run the smallest relevant test set first.
- **Self-review step:** before presenting a batch of edits, the model reviews its own diff against the plan and repository conventions. This costs one extra call and catches many obvious mistakes.
- **Property-based test suggestions** for pure functions (Hypothesis/fast-check).

**Complexity:** M.

### 3.5 Long-running task mode (with strong containment only)
- Unattended multi-hour tasks are allowed **only** with L2/L3 sandboxing: a container plus a disposable worktree.
- The task runs with a budget (steps, wall time) and a pre-approved policy profile.
- The human reviews the final branch, the full audit trail and test results before merging.

**Complexity:** L.

---

## 4. Memory and Personalization

### 4.1 Project memory and decision log
- **`AGENTS.md` stays the human-curated source of truth.** Hearth can *propose* additions, with approval, when it discovers stable facts: test commands, conventions, "never edit generated client code".
- **Decision log:** `docs/decisions.log.md` or a `state.db` table records "decided X because Y" entries extracted at compaction time. The user confirms each one before it's saved.
- **Retrieval integration:** memory entries become a small, always-retrieved namespace with strict token caps.

**Complexity:** S–M. **Safety:** memory never contains permissions; it is data, not policy.

### 4.2 Personal preferences
Style preferences such as comment density, test naming, or "prefer functional style" are stored in global config and injected into the stable system prefix. **Complexity:** S.

### 4.3 Learning from approvals and rejections
- **Data:** the audit log already records rejections with feedback, and edited commands and messages.
- **Near term:** mine this data into **new eval cases** and **profile adjustments**, for example lowering autonomy for a model that often gets rejected.
- **Longer term:** assemble a local preference dataset for adapter fine-tuning (§6.2).
- **Privacy:** it never leaves the machine and is opt-in.

**Complexity:** M.

---

## 5. Interfaces and Integrations

### 5.1 Inline autocomplete (Cursor-style Tab)
- **Why it's separate:** it needs roughly 100–300 ms latency, a fill-in-the-middle-capable model, and tight editor integration. The agent path has none of these requirements.
- **Approach:**
  - The VS Code extension sends prefix and suffix plus a few retrieved snippets (from Hearth's index: same-file symbols, imported signatures) to a small FIM model served by Ollama.
  - Debounce, cancellation and caching are required. It needs its own context builder, with a tiny budget and no tools.
- **Hardware:** a 1.5–7B coder model on GPU. It runs poorly while a large chat model occupies the same GPU, so provide a profile switch or scheduling.

**Complexity:** L.

### 5.2 Editors beyond VS Code
Because the core speaks JSON-RPC over stdio, thin clients for **Neovim** (Lua), **JetBrains** (Kotlin plugin), **Zed** and **Emacs** are feasible. **Complexity:** M each.

### 5.3 MCP integration
- **Client:** connect *local stdio* MCP servers (database schema inspectors, issue trackers running on the LAN, custom internal tools). Each server's tools are wrapped as Hearth tools with configured risk levels, so they flow through the same policy engine and approvals. Remote MCP servers stay disabled by default, since they are a privacy boundary.
- **Server:** expose Hearth's retrieval (`search_code`, `find_symbol`, `repo_map`) as an MCP server, so other local tools and agents can use its index. Read-only by default.

**Complexity:** S (server) to M (client with policy mapping).

### 5.4 Web dashboard
A local browser UI for:
- session history and audit browsing,
- retrieval debugging (score breakdowns),
- index health,
- eval and bench reports, and
- model comparisons.

Loopback plus token authentication, as specified in the tech stack. **Complexity:** M.

### 5.5 Voice and hands-free (low priority)
Local speech-to-text through audio-capable models or a small local STT model, for dictating questions. **Complexity:** S–M. Nice-to-have only.

---

## 6. Models and Learning

### 6.1 Vision workflows
Several current local model families accept images. Useful workflows:
- **UI bug from screenshot:** the model maps the screenshot to components via retrieval (text labels, CSS class names).
- **Whiteboard or diagram to code skeleton or ADR.**
- **Render and compare:** generated Mermaid diagrams are rendered locally and checked by the model for layout and readability.

**Complexity:** S–M.

### 6.2 Local fine-tuning or adapters (research track)
**Hypothesis:** a LoRA adapter trained on your repository's conventions, and on your accepted diffs, improves edit acceptance rate.

**Reality check:**
- It needs a solid eval suite first. Without one, you can't tell whether it helped.
- The data pipeline must be careful about leaking secrets into weights.
- Retrieval plus good `AGENTS.md` conventions often capture most of the benefit at far lower cost.

**Approach, if pursued:**
- Train an adapter with a local tool such as Unsloth, MLX or Axolotl.
- Export it to GGUF and import it into Ollama with a Modelfile.
- A/B test it through `hearth eval tasks`.

**Complexity:** XL. Treat as research.

### 6.3 Embedding model adaptation
Lightweight fine-tuning of a small embedding model on (question → relevant code) pairs mined from your own sessions. Citations the user clicked or kept serve as positives. It is cheaper and more likely to help than chat-model fine-tuning. **Complexity:** L.

---

## 7. Safety and Operations

### 7.1 Stronger containment
- **Per-task containers** with network disabled and resource quotas (L2), and a disposable worktree plus container per task (L3).
- **Egress monitor:** a local check that alerts if any Hearth-spawned process opens non-loopback connections, useful even without full sandboxing, using OS tools where available.

**Complexity:** M–L.

### 7.2 Tamper-evident audit log
Hash-chain each audit record, where each record includes the previous record's hash. Optionally sign periodic checkpoints with a local key. This matters for regulated environments. **Complexity:** S.

### 7.3 Policy packs
Shareable, reviewable policy presets (e.g., `python-web-strict`, `data-science-relaxed`, `regulated-readonly`), installable into global config with a diff preview. **Complexity:** S.

### 7.4 Background indexing daemon
- A per-user `hearthd` keeps indexes warm across terminals and editors. It watches configured repositories, embeds during idle time, and serves retrieval over a local socket (Unix domain socket with file permissions; no TCP).
- The CLI and IDE share it, avoiding duplicate indexing.

**Complexity:** M.

### 7.5 Team / LAN mode
- A shared inference box (e.g., a Tier 4 workstation) serves several developers over the LAN.
- **This changes the threat model.** Code now leaves each laptop, even though it stays inside the office. Requirements:
  - TLS,
  - per-user authentication,
  - per-user isolation of sessions and indexes,
  - an explicit opt-in (`allow_remote_host = true`) with a prominent banner, and
  - organizational policy approval.

**Complexity:** L.

---

## 8. Workflow Extensions

| Workflow | Description | Complexity |
|---|---|---|
| `/triage-ci <log file>` | Parse a pasted or local CI log, locate failing code via retrieval, propose a fix plan | S |
| `hearth review --staged --headless` as a pre-commit hook | Read-only review that prints findings; never blocks unless configured | S |
| `/migrate <from> <to>` | Guided framework or library migrations using the deps index for the new API, plan mode and batch edits | M |
| `/perf <target>` | Run a profiler command (approved), summarize hotspots, propose optimizations with benchmarks | M |
| `/security-review` | Local static checks (e.g., Semgrep with local rules, if installed) + model triage of findings | M |
| `/onboard` | Generate a guided tour of the codebase: key flows, entry points, a glossary | S–M |
| `/changelog` | Draft release notes from commits since the last tag | S |
| Notebook support | Chunk `.ipynb` code and markdown cells; edits via cell-aware tools | M |
| SQL and data projects | Schema-aware retrieval from migrations or DDL; query explanation; dbt model graph | M |

---

## 9. Research Questions Worth Tracking

1. **Retrieval versus agentic search balance per model tier.** At what model capability does pre-retrieval stop helping, or start hurting by crowding context? Measure with the eval suite across tiers.
2. **Optimal chunk granularity for code** under small contexts: function-level versus skeleton plus on-demand bodies.
3. **Summary-augmented retrieval ROI** on local hardware: recall gain versus summarization cost.
4. **Edit format reliability** by model family: search/replace versus whole-file versus structured AST edits.
5. **Approval UX:** which badge designs reduce rubber-stamping without slowing experts? This can be measured from local audit data (time-to-approve, rejection rates).
6. **Hybrid-attention and sparse-attention models:** how much longer can local contexts get before quality degrades? Re-tune budgets as architectures evolve.
