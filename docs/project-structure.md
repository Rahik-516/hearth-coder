# Hearth — Project Structure

This layout follows the layered architecture in `system-design.md` §4.
- It uses a `src/` layout, with one package per layer or concern.
- Tests mirror the source tree.
- Everything security-critical is isolated in a small number of modules that get the strictest typing and review.

---

## 1. Repository Tree

```text
hearth/
├── pyproject.toml                  # uv project, deps, ruff/mypy/pytest/import-linter config
├── uv.lock                         # pinned, reproducible (commit it)
├── README.md                       # user-facing: install, quickstart, offline guarantees
├── CLAUDE.md                       # instructions for Claude Code while building Hearth (template in §5)
├── LICENSE
├── .python-version                 # 3.12
├── .pre-commit-config.yaml
├── .importlinter                   # layer contracts (or [tool.importlinter] in pyproject)
│
├── docs/
│   ├── system-design.md
│   ├── tech-stack.md
│   ├── project-structure.md
│   ├── implementation-roadmap.md
│   ├── model-recommendations.md
│   ├── safety-and-tool-use.md
│   ├── future-extensions.md
│   ├── adr/                        # Architecture Decision Records, one per significant choice
│   │   ├── 0001-sqlite-numpy-vector-search.md
│   │   ├── 0002-custom-agent-loop.md
│   │   └── template.md
│   └── user/                       # end-user docs (config reference, permissions guide, FAQ)
│
├── src/hearth/
│   ├── __init__.py                 # version only; no side effects
│   ├── __main__.py                 # `python -m hearth` → cli.app
│   │
│   ├── cli/                        # FRONTEND: terminal UI (depends on core only)
│   │   ├── app.py                  # Typer app: init, index, chat, run, search, doctor, trust, eval, bench, resume, undo
│   │   ├── repl.py                 # interactive loop: prompt_toolkit session ↔ event bus
│   │   ├── render.py               # Rich renderers for events (markdown stream, thinking, tool panels, stats)
│   │   ├── approval.py             # approval panel, key handling, $EDITOR edit flow, typed confirmation
│   │   ├── diff_view.py            # colored unified diff, per-file accept/reject for batches
│   │   ├── completers.py           # /command and @path completion (from index)
│   │   ├── slash_commands.py       # /mode /model /plan /execute /test /doc /review /commit /undo /compact /context …
│   │   └── doctor.py               # environment checks and report formatting
│   │
│   ├── server/                     # FRONTEND (Phase 2+): machine protocol adapters
│   │   ├── stdio_jsonrpc.py        # `hearth serve --stdio` (IDE integration)
│   │   ├── ws_app.py               # Phase 3: FastAPI + WebSocket (loopback, token, Host/Origin checks)
│   │   └── protocol.py             # JSON-RPC method names ↔ core commands/events
│   │
│   ├── core/                       # APPLICATION CORE
│   │   ├── events.py               # Pydantic event + command models (the frontend contract)
│   │   ├── bus.py                  # async event bus; request/response for approvals
│   │   ├── session.py              # Session state, modes, permission level, grants, read-hash registry, todo list
│   │   ├── session_store.py        # persistence to state.db (resume)
│   │   ├── runner.py               # the agent loop (system-design §8)
│   │   ├── limits.py               # step limits, retry budgets, wall-clock limits per profile/mode
│   │   ├── loop_detector.py
│   │   ├── modes.py                # chat | plan | agent → tool subsets, pre-retrieval policy
│   │   ├── plan.py                 # structured plan schema, rendering, plan-scoped grants
│   │   └── context/
│   │       ├── tokens.py           # calibrated TokenEstimator
│   │       ├── budget.py           # per-tier segment budgets
│   │       ├── builder.py          # prefix-cache-stable request assembly
│   │       ├── compactor.py        # structured summarization, cache epochs
│   │       └── truncation.py       # insertion-time tool-output truncation
│   │
│   ├── prompts/                    # prompt text as data (Markdown + string.Template placeholders)
│   │   ├── system_core.md          # identity, rules, citation format (short)
│   │   ├── system_tools.md         # tool usage guidance, edit protocol
│   │   ├── mode_chat.md
│   │   ├── mode_plan.md
│   │   ├── mode_agent.md
│   │   ├── compaction.md
│   │   ├── rerank.md
│   │   └── workflows/
│   │       ├── test.md
│   │       ├── doc_readme.md
│   │       ├── doc_architecture.md
│   │       ├── review.md
│   │       ├── commit.md
│   │       └── refactor.md
│   │
│   ├── workflows/                  # deterministic scaffolding around the LLM
│   │   ├── base.py                 # Workflow dataclass, Step protocol, runner integration
│   │   ├── plan_execute.py
│   │   ├── test_writer.py          # detect framework & conventions → write → run → parse → fix loop
│   │   ├── docs.py                 # facts gathering → summaries → outline → sections → assemble
│   │   ├── review.py               # diff + related context → findings (read-only)
│   │   ├── commit.py               # staged diff → message → approval → commit
│   │   ├── refactor.py             # target analysis (refs) → plan → edits → tests
│   │   └── test_parsers/           # pytest.py, jest.py, vitest.py, go_test.py, cargo_test.py
│   │
│   ├── llm/                        # INFRA: model access
│   │   ├── types.py                # ChatRequest, ChatChunk, Usage, ModelInfo, Message types
│   │   ├── provider.py             # LLMProvider protocol
│   │   ├── ollama_provider.py      # native API via official SDK; cloud-tag refusal; loopback check
│   │   ├── scripted_provider.py    # deterministic fake for tests and demos
│   │   ├── profiles.py             # ModelProfile model + matching logic
│   │   ├── profiles.toml           # bundled profiles (families, sampling, thinking, reliability, embed templates)
│   │   └── tool_call_parser.py     # native tool_calls + tolerant text fallbacks + validation
│   │
│   ├── indexing/                   # INFRA: building the index
│   │   ├── pipeline.py             # orchestrates stages, two-phase availability, progress events
│   │   ├── scanner.py              # git ls-files / walk
│   │   ├── filters.py              # ignore layers, size, binary, generated, secret files
│   │   ├── change_detector.py      # stat fast path + hashing
│   │   ├── languages.py            # extension/shebang → language; support levels
│   │   ├── parser.py               # tree-sitter parser pool (grammar wheels)
│   │   ├── symbols.py              # tags-query extraction: defs, refs, imports, signatures
│   │   ├── chunker.py              # AST split-merge, skeletons, fallback windows, doc sections
│   │   ├── enrich.py               # context headers, identifier splitting
│   │   ├── embedder.py             # batching, cache, resumability, dimensions, CPU pinning
│   │   ├── summaries.py            # Phase 2: lazy file/dir summaries
│   │   ├── watcher.py              # watchfiles-based incremental updates
│   │   └── queries/                # vendored, versioned tree-sitter queries
│   │       ├── python/tags.scm
│   │       ├── typescript/tags.scm
│   │       ├── tsx/tags.scm
│   │       ├── javascript/tags.scm
│   │       └── …                   # go, rust, java in Phase 2
│   │
│   ├── retrieval/                  # SERVICE: finding context
│   │   ├── engine.py               # public API: retrieve(query, budget, session) → ContextBlock
│   │   ├── query_analysis.py       # identifiers, paths, intent
│   │   ├── dense.py
│   │   ├── lexical.py              # FTS5 query building + bm25 column weights
│   │   ├── symbol_search.py
│   │   ├── path_search.py
│   │   ├── fusion.py               # weighted RRF + boosts/penalties
│   │   ├── rerank.py               # optional LLM listwise rerank (structured output)
│   │   ├── expansion.py            # parent skeletons, callee signatures, caller hints
│   │   ├── packing.py              # diversity, merge ranges, budgeted ordering, citation rendering
│   │   ├── repomap.py              # file graph, personalized PageRank, budgeted rendering
│   │   └── explain.py              # score breakdowns for `hearth search --explain`
│   │
│   ├── storage/                    # INFRA: persistence
│   │   ├── db.py                   # connections, pragmas, transactions
│   │   ├── migrate.py              # numbered migrations + schema_version
│   │   ├── migrations/
│   │   │   ├── index/0001_init.sql
│   │   │   └── state/0001_init.sql
│   │   ├── index_repo.py           # files/chunks/fts/symbols/refs/imports/summaries access
│   │   ├── vector_index.py         # VectorIndex protocol + NumpyVectorIndex
│   │   ├── state_repo.py           # sessions/messages/tool_calls/grants/trust
│   │   └── blobs.py                # content-addressed blob store
│   │
│   ├── tools/                      # SERVICE: things the model can do
│   │   ├── base.py                 # Tool ABC: Args, risk, prepare(), execute(), render_for_model()
│   │   ├── registry.py             # registration, schemas per mode/profile
│   │   ├── gateway.py              # validate → prepare → policy → approve → execute → record
│   │   ├── results.py              # ToolResult, error codes, model-facing formatting
│   │   ├── read_fs.py              # read_file, list_dir, find_files
│   │   ├── search.py               # search_code, grep, find_symbol, find_references, repo_map
│   │   ├── edit_engine.py          # matching (exact/normalized/fuzzy), parse guard, diff generation
│   │   ├── write_fs.py             # edit_file, multi_edit, write_file, move_file, delete_file
│   │   ├── shell.py                # run_command
│   │   ├── tests.py                # run_tests
│   │   ├── git_read.py             # git_status, git_diff, git_log, git_show, git_blame
│   │   ├── git_write.py            # git_add, git_commit, git_branch_create, git_switch
│   │   └── meta.py                 # todo_write, ask_user
│   │
│   ├── safety/                     # SERVICE: SECURITY-CRITICAL (strict typing, tests first)
│   │   ├── policy.py               # pure decision function
│   │   ├── rules.py                # rule models, matching (argv prefix, path globs)
│   │   ├── invariants.py           # hard-deny checks (workspace escape, protected paths, sudo…)
│   │   ├── command_classifier.py   # shlex parse, metacharacters, families, network/destructive detection
│   │   ├── paths.py                # workspace jail, protected & sensitive path sets
│   │   ├── env.py                  # environment scrubbing for subprocesses
│   │   ├── secrets.py              # secret patterns for index exclusion, edit/commit scanning, redaction
│   │   ├── injection.py            # instruction-like content heuristics → badges
│   │   ├── checkpoints.py          # snapshot/restore/undo/rewind
│   │   ├── audit.py                # append-only JSONL audit writer with redaction
│   │   ├── trust.py                # project config trust (hash-based)
│   │   └── sandbox/
│   │       ├── runner.py           # ProcessRunner protocol
│   │       ├── subprocess_runner.py# MVP: process groups, timeouts, output capture, no stdin
│   │       ├── bwrap_runner.py     # Phase 3 (Linux)
│   │       └── seatbelt_runner.py  # Phase 3 (macOS)
│   │
│   ├── git/                        # INFRA: git CLI adapter
│   │   ├── runner.py               # hardened invocation (safe -c flags, env, timeouts)
│   │   └── porcelain.py            # status v2 / log / diff-stat parsers
│   │
│   ├── config/                     # INFRA: configuration
│   │   ├── schema.py               # Pydantic config models
│   │   ├── loader.py               # defaults → global → project (trust-aware merge)
│   │   ├── defaults.toml           # built-in defaults incl. per-tier presets
│   │   └── paths.py                # platformdirs locations, project id
│   │
│   ├── evals/                      # model-in-the-loop evaluation harness
│   │   ├── retrieval_eval.py       # recall@k, MRR
│   │   ├── task_eval.py            # disposable repo copies, verification commands
│   │   ├── bench.py                # prefill/gen throughput, TTFT, memory headroom
│   │   └── report.py
│   │
│   └── util/
│       ├── aio.py                  # cancellation helpers, timeouts
│       ├── hashing.py
│       ├── text.py                 # identifier splitting, line-ending/encoding detection
│       ├── atomic.py               # atomic file writes
│       └── logging.py
│
├── tests/
│   ├── conftest.py                 # pytest-socket (loopback only), tmp workspaces, scripted provider fixtures
│   ├── unit/                       # mirrors src/hearth/*
│   │   ├── indexing/
│   │   ├── retrieval/
│   │   ├── core/
│   │   ├── tools/
│   │   └── safety/                 # heaviest coverage + Hypothesis property tests
│   ├── integration/
│   │   ├── test_index_fixture_repos.py
│   │   ├── test_agent_loop_scripted.py
│   │   └── test_cli_smoke.py
│   ├── security/
│   │   ├── test_no_side_effects_without_approval.py
│   │   ├── test_headless_fails_closed.py
│   │   ├── test_prompt_injection_repo.py
│   │   └── test_path_escape_attempts.py
│   ├── snapshots/                  # syrupy snapshots (chunks, prompts, repo maps)
│   └── fixtures/
│       └── repos/
│           ├── py_small/           # ~2K LOC Python service with tests
│           ├── ts_small/           # ~2K LOC TypeScript app with tests
│           ├── polyglot/           # Python + TS + Markdown + YAML
│           └── malicious/          # instruction-injection comments, symlink escapes, fake secrets
│
├── evals/                          # eval DATA (kept separate from code); YAML — needs the `evals` extra
│   ├── retrieval/
│   │   ├── py_small.yaml           # question → expected files/symbols
│   │   └── ts_small.yaml
│   └── tasks/
│       ├── add_unit_test.yaml
│       ├── rename_symbol.yaml
│       └── fix_failing_test.yaml
│
└── scripts/
    ├── build_wheelhouse.sh         # offline install bundle
    ├── gen_fixture_repo.py
    └── update_tags_queries.md      # procedure for updating vendored queries (manual, reviewed)
```

---

## 2. Module Responsibilities and Boundaries

| Package | Layer | May import | Must not import | Notes |
|---|---|---|---|---|
| `cli`, `server` | Frontend | `core`, `config`, `util` | `tools`, `safety`, `llm`, `storage` directly | All interaction via events/commands |
| `core` | Application | `retrieval`, `tools`, `safety`, `llm`, `storage`, `config`, `prompts`, `util` | `cli`, `server` | Owns orchestration only |
| `workflows` | Application | `core`, `tools`, `retrieval`, `git`, `util` | `cli`, `server` | Recipes over the runner |
| `tools` | Service | `retrieval`, `indexing` (reindex hook), `safety`, `storage`, `git`, `config`, `util` | `llm`, `core` | Tools never call models |
| `retrieval` | Service | `storage`, `llm` (embed + optional rerank), `config`, `util` | `tools`, `core`, `indexing` | Deterministic given index + embeddings |
| `indexing` | Infra | `safety` (paths, secrets), `storage`, `llm` (embed only), `config`, `util` | `core`, `tools`, `retrieval` | |
| `safety` | Service | `storage` (checkpoints, trust), `config`, `util` | `llm`, `core`, `tools`, `retrieval` | `policy.py` itself performs no I/O (separate contract) |
| `llm`, `storage`, `git` | Infra | `config`, `util`, stdlib, third-party | Everything above; each other | Only `llm` opens sockets |
| `config`, `util` | Foundation | stdlib, third-party | All other Hearth packages | |

Enforce with `import-linter` contracts, run in pre-commit and CI. Higher layers may import lower ones, and modules separated by `|` must not import each other:

```ini
[importlinter]
root_package = hearth

[importlinter:contract:layers]
name = Layered architecture
type = layers
layers =
    hearth.cli | hearth.server
    hearth.workflows
    hearth.core
    hearth.tools
    hearth.retrieval
    hearth.indexing
    hearth.safety
    hearth.llm | hearth.storage | hearth.git
    hearth.config
    hearth.util

[importlinter:contract:tools-no-llm]
name = Tools never call models
type = forbidden
source_modules = hearth.tools
forbidden_modules = hearth.llm

[importlinter:contract:policy-pure]
name = Policy engine has no I/O dependencies
type = forbidden
source_modules = hearth.safety.policy
forbidden_modules = hearth.storage, hearth.llm, hearth.tools, hearth.git
```

---

## 3. Runtime Data Locations (outside the repository)

| Path (Linux; macOS/Windows via platformdirs) | Contents | Lifetime |
|---|---|---|
| `~/.config/hearth/config.toml` | Global config and permission rules | User-managed; **protected from agent writes** |
| `~/.local/share/hearth/projects/<slug>-<id>/index.db` | Index (disposable) | Rebuilt on demand |
| `~/.local/share/hearth/projects/<slug>-<id>/state.db` | Sessions, checkpoints metadata, grants, trust | Kept; `hearth gc` prunes old sessions |
| `~/.local/share/hearth/projects/<slug>-<id>/blobs/` | Content-addressed file snapshots, full tool outputs | Pruned with sessions |
| `~/.local/share/hearth/projects/<slug>-<id>/trash/` | Files "deleted" by the agent | Pruned after N days |
| `~/.local/share/hearth/audit/YYYY-MM.jsonl` | Append-only audit log | Kept (user-managed) |
| `~/.local/share/hearth/logs/` | App logs; `--debug` prompt dumps (off by default) | Rotated |

**On the dev laptop (Hearth inside WSL2):**
- All paths above are inside the WSL Linux filesystem.
- Ollama's models are stored on the Windows side in `C:\Users\<you>\.ollama\models`. Set the Windows `OLLAMA_MODELS` variable to relocate them if the 512 GB SSD fills up.
- Keep project checkouts in `~/code` inside WSL for indexing speed, reliable file watching and correct path-safety semantics.

`<id>` is the first 16 hex characters of the SHA-256 of the canonical, resolved repository root path. `<slug>` is the directory name, for human readability.

**In the repository** (all optional, user-authored):

| Path | Purpose |
|---|---|
| `.hearth/config.toml` | Project settings (test/lint commands, index excludes, project rules; allow-rules need `hearth trust`) |
| `.hearthignore` | Extra index exclusions (gitignore syntax) |
| `AGENTS.md` (preferred) or `HEARTH.md` | Project conventions and instructions injected into the system context |

Hearth never writes index or state data into the repository. `hearth init` offers to add `.hearth/local.toml` (personal, uncommitted overrides) to `.gitignore`.

---

## 4. Naming and Coding Conventions

- **Modules** are nouns (`chunker.py`, `policy.py`). **Tools** are verbs in snake_case (`edit_file`, `run_tests`), matching the names the model sees.
- **Pydantic models** are used for every boundary object: events, tool arguments, config, profiles, plan schema. Plain dataclasses are used for internal value objects.
- **No module-level side effects.** Heavy imports (numpy, tree_sitter, networkx) are imported lazily inside functions in CLI paths to keep startup fast.
- **Errors** are domain exceptions in `<package>/errors.py`. Tool errors are *returned* as `ToolResult(error=...)`, never raised into the runner.
- **Prompts live in `src/hearth/prompts/`**, never inline in Python. Snapshot tests catch accidental prompt changes.
- **All filesystem access by tools** goes through `safety.paths.resolve_in_workspace()`. A grep-based test in CI fails if `open(`, `Path.write_text` or `os.remove` appear inside `tools/` outside the approved helper module.
- **Type checking:** `mypy --strict` for `core`, `safety`, `tools`, `llm`; standard mode elsewhere.
- **Docstrings** go on public functions. The doc must state invariants in security-critical code (e.g., "`prepare()` must not mutate the filesystem").

---

## 5. `CLAUDE.md` Template (for building Hearth with Claude Code)

Place this at the repository root. Keep it short and update it as the project evolves.

````markdown
# Hearth — instructions for Claude Code

## What this project is
Hearth is a fully offline, local AI coding assistant (Python 3.12, uv) that uses local Ollama models.
Design docs live in `docs/`. Read the relevant section before implementing anything:
- Architecture & data model: docs/system-design.md
- Library choices (do not add deps outside this list without an ADR): docs/tech-stack.md
- Where code goes: docs/project-structure.md
- Current milestone & acceptance criteria: docs/implementation-roadmap.md
- Tool/permission rules: docs/safety-and-tool-use.md

## Commands
- Install: `uv sync`
- Tests: `uv run pytest -q` (unit: `uv run pytest tests/unit -q`)
- Lint/format: `uv run ruff check --fix . && uv run ruff format .`
- Types: `uv run mypy src/hearth`
- Layer contracts: `uv run lint-imports`
- Live model evals (needs local Ollama): `uv run hearth eval retrieval`

## Dev machine (design every live test and benchmark for this)
- ASUS ROG Strix G15: Ryzen 7 4800H, RTX 3060 Laptop **6 GB VRAM**, 16 GB RAM, 512 GB SSD.
- Windows 11 + Ollama for Windows; Hearth is developed inside WSL2 (Ubuntu) with mirrored networking.
  Repos live in `~/code` inside WSL, never under `/mnt/c`.
- Models: `qwen3.5:4b` (daily, num_ctx 12288), `qwen3.5:9b` (eval-only), `qwen3-embedding:0.6b`.
- Implications:
  - Live tests (`-m live`) use `qwen3.5:4b`, keep prompts < 8K tokens, and run sequentially.
  - Never load two chat models at once.
  - Mark heavy tests (large synthetic indexes, benchmarks) `-m slow`; they don't run by default.
  - Don't generate large fixture data into the repo (disk is limited).

## Non-negotiable rules
1. No runtime network access except the configured loopback Ollama host. Never add telemetry, update checks,
   or libraries that download at runtime. Tests run with pytest-socket; do not disable it.
2. Respect layer boundaries (see .importlinter). Tools never call the LLM. safety/policy.py does no I/O.
3. Security-critical packages (`src/hearth/safety/`, `tools/write_fs.py`, `tools/edit_engine.py`,
   `tools/shell.py`, `tools/git_write.py`): write tests FIRST, include adversarial cases, keep mypy --strict clean.
4. Every filesystem access in tools goes through `safety.paths.resolve_in_workspace`.
5. Prompts live in `src/hearth/prompts/*.md`; update snapshot tests intentionally when changing them.
6. New runtime dependency → propose an ADR in docs/adr/ and ask before adding.
7. Never run Hearth's agent tools against real repositories in tests — use `tests/fixtures/repos/*` copied to tmp dirs.

## Style
- Pydantic v2 for boundary models; dataclasses for internal values.
- Async-first in core/llm/tools; no blocking calls in the event loop (use `asyncio.to_thread` for heavy CPU work).
- Small functions, explicit names, docstrings stating invariants in safety code.

## Workflow
- Start each task in plan mode: restate the acceptance criteria from the roadmap, list files to touch, then implement.
- Keep changes scoped to one milestone task; run tests + ruff + mypy + lint-imports before finishing.
````

> **Privacy note while building:** Claude Code is a cloud service. Developing *Hearth's own source code* with it is fine. Don't point it at the private codebases Hearth is meant to protect. Build and test against the public fixture repositories in `tests/fixtures/repos/`.

---

## 6. `pyproject.toml` Skeleton

```toml
[project]
name = "hearth"
version = "0.1.0"
description = "Fully offline AI coding assistant and project agent for local Ollama models"
requires-python = ">=3.12"
readme = "README.md"
license = { text = "MIT" }
dependencies = [
  "ollama",            # pin exact versions via uv.lock
  "pydantic>=2",
  "typer",
  "rich",
  "prompt_toolkit",
  "tree-sitter",
  "tree-sitter-python",
  "tree-sitter-javascript",
  "tree-sitter-typescript",
  "numpy",
  "networkx",
  "watchfiles",
  "pathspec",
  "platformdirs",
  "tomli-w",
]

[project.optional-dependencies]
evals = ["pyyaml"]                                                          # eval DATA only; import lazily
langs-extra = ["tree-sitter-go", "tree-sitter-rust", "tree-sitter-java"]   # Phase 2
server = ["fastapi", "uvicorn"]                                             # Phase 3 web UI

[project.scripts]
hearth = "hearth.cli.app:main"

[dependency-groups]
dev = [
  "pytest", "pytest-asyncio", "pytest-socket", "hypothesis", "syrupy",
  "ruff", "mypy", "import-linter", "pre-commit",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
addopts = "--disable-socket --allow-hosts=127.0.0.1,::1 -q -m \"not live and not slow\""
asyncio_mode = "auto"
markers = [
  "live: requires a running local Ollama with qwen3.5:4b (run with: pytest -m live)",
  "slow: heavy tests (large synthetic indexes, benchmarks) — run deliberately on AC power",
]

[tool.ruff]
line-length = 110
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "ASYNC", "S", "PTH", "RUF"]

[tool.ruff.lint.per-file-ignores]
# Only these modules may spawn processes; S602 (shell=True) is never ignored here —
# its single call site carries a justified per-line noqa. See tech-stack.md §12.1.
"src/hearth/safety/sandbox/*_runner.py" = ["S603", "S607"]
"src/hearth/git/runner.py"              = ["S603", "S607"]
"tests/**"                              = ["S101"]

[tool.mypy]
python_version = "3.12"
warn_unused_ignores = true

[[tool.mypy.overrides]]
module = ["hearth.core.*", "hearth.safety.*", "hearth.tools.*", "hearth.llm.*"]
strict = true
```

Confirm that grammar wheel versions match the ABI supported by your pinned `tree-sitter` version when you first lock dependencies. Add a unit test that loads every bundled grammar so an incompatible upgrade fails fast.
