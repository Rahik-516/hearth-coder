# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current state

**Phase 0 through M6 (mostly) implemented.** The full unit suite, `ruff check`, `mypy src/hearth` and
`lint-imports` all run clean under WSL2. Chat mode with RAG works; the safety spine, write tools,
checkpoints, exec tools, git writes and agent turns are in.

**M6 is code-complete except batch review.** Landed: the command classifier, environment scrubbing,
`SubprocessRunner`, `run_command`, `run_tests`, the four git write tools, `todo_write`,
`prompts/system_tools.md` + `mode_agent.md`, `ChatRunner.run_agent_turn`, `/mode agent`,
`hearth run` with headless fail-closed, and the task-eval harness.

**The MVP loop works against a real model.** `hearth eval tasks` scores **3/3** on `qwen3.5:4b`
([docs/benchmarks/m6-task-eval.md](docs/benchmarks/m6-task-eval.md)); the M6 bar was ≥1/3. The live
suite (`-m live`) is green except the embedding test.

**Phase 2 (I1) is underway.** `retrieval/repomap.py` is done and wired into `ChatRunner` behind a
per-epoch cache: personalised PageRank over a use graph, budget-fitted by binary search over whole
definitions. It lands at 1200/1200 tokens against a 1200 budget on this repository (the I1 bar is ±5%),
and architecture questions now name real components across the codebase.

`retrieval/expansion.py` is done too: parent skeletons, callee signatures and caller hints, all
signature-only, wired into the engine after diversity. A callee name defined in more than one file is
skipped rather than guessed — a wrong signature is worse than none, since the model cannot tell.

**The eval data was lost in the disaster and is restored.** `evals/retrieval/` was empty, so
`hearth eval retrieval` could not run at all. `py_small.yaml` is rebuilt, and `global_questions.yaml`
is the I1 acceptance subset — architecture questions, run against Hearth itself because a 10-file
fixture has no architecture to ask about. Baseline: **recall@10 0.90** (bar is 0.80).

Its header states the labelling rule, which matters: **a file counts as ground truth if a developer
reading it would learn the answer, including documentation.** The first run marked
`docs/system-design.md` a miss for "what are the main components", which measured the labels rather
than retrieval. The rule cuts both ways — same-package neighbours that do not answer the question are
still misses, and one remains.

**Go, Rust and Java are at FULL support**, with vendored `tags.scm` queries and a
`tests/fixtures/repos/polyglot` fixture. They live in the optional `langs-extra` extra, so run
`uv sync --extra langs-extra` (or `uv run --extra langs-extra …`) to exercise them — a plain `uv run`
re-syncs to the default set and removes them, and the tests then skip rather than fail.

`GRAMMAR_AVAILABLE` used to be a hardcoded frozenset, so an optional grammar could never register:
the query loaded, the grammar imported, and parsing still returned `skipped`. Availability is now
derived from `GRAMMAR_MODULES` in `languages.py` via `find_spec`, which is also the single source of
truth `parser.py` loads through.

**Structured chunking for JSON/YAML/TOML is done**, which completes I1's listed work. Files past
`STRUCTURED_MIN_TOKENS` split on their own structure — TOML table headers, YAML column-zero keys,
JSON keys at brace depth 1 — rather than at arbitrary window boundaries. The split is textual on
purpose: `json` and `tomllib` report no line numbers and PyYAML is not a runtime dependency (rule 7),
so a parse would buy correctness the chunker cannot use, since every chunk needs a citable range.

**The index records what built it.** `EXTRACTION_VERSION` in `pipeline.py` is stamped into `meta` and
compared each run; a mismatch forces a full re-index and says so in the summary. **Bump it whenever
chunking, symbol extraction or a tag query changes** — otherwise the index keeps chunks the current
code would never produce, with embeddings keyed to them, and nothing reports it. A missing stamp
counts as a mismatch, since an index written before the key existed has unknown provenance.

**Outstanding in I1:** only structural support for C/C++/C#/Ruby/PHP, which needs grammars no extra
currently ships. `hearth doctor` lists which languages have grammars and warns only when one is
installable, so the gap is visible without being nagging.

**Also outstanding:** (1) **batch review** (§6.4) — needs a loop pre-pass that prepares every write in a
multi-write step and one review screen; `cli/approval.batch_blockers` is already written and tested,
and the roadmap names this the first thing to cut. (2) **The 9b half of the task eval** — now possible,
since the GPU works. (3) I2–I4 and Phase 3 are not started.

**The GPU works, after a fix worth remembering.** Ollama's device discovery was crashing
(`0xc0000005`) on every backend because `llama-server.exe` loaded the old system
`C:\Windows\System32\MSVCP140.dll` (14.28) instead of the 14.44 copy Ollama bundles in its backend
folders. The fix was copying those bundled runtime DLLs into `E:\DevTools\Ollama\lib\ollama\` beside
`llama-server.exe`. Effect: 61 tok/s instead of 7.6, and the task eval drops from ~5 minutes to ~20
seconds per task. If an Ollama update ever puts the model back on CPU, `hearth doctor` will say so and
the same copy fixes it. Diagnosis method, if it recurs: run `llama-server.exe --list-devices` by hand
and read the faulting module from the Windows Application event log (Event ID 1000).

**Run the tests the way CLAUDE.md documents.** `uv run pytest -q` was broken for months — it collected
the fixture repos' own suites and died on import — and nobody noticed because every session ran
`pytest tests/unit` instead. The integration tests were never executed at all, and they were hiding a
real bug (re-indexing raised "cannot start a transaction within a transaction"). Narrow commands give
narrow assurances.

**One hard lesson, recorded because it cost the entire working tree.** A test in
`tests/unit/tools/test_shell.py` once called `tool.prepare()` and `tool.execute()` directly — bypassing
`ToolGateway` — to assert that the classifier flags `pytest; rm -rf ~`. The assertion was about a
predicate, but with policy skipped the command *ran* and deleted `/home/hearth`, including the
uncommitted M1–M6 work. Roughly 260 files were recovered from git objects found in the ext4 image; the
rest was rewritten. The file now carries two standing rules in its header: **nothing executes except
through the gateway, and every adversarial payload is inert.** The path tools may use the
prepare/execute shortcut safely because `resolve_in_workspace` jails them; a command string has no jail,
so for exec, policy *is* the containment. Commit before any risky step.

**Toolchain lives under `E:\DevTools\`, one subdirectory per tool** — `uv\`, `Git\`, `Ollama\`, matching the
existing convention. `uv`'s own Python installs and cache also live under `E:\DevTools\uv\python` and
`...\cache` rather than the default `AppData\Roaming\uv\`, because that path sits next to this machine's
OneDrive sync and broke uv's directory-junction creation for managed Python installs (confirmed: deleting
the AppData cache and pointing `UV_PYTHON_INSTALL_DIR`/`UV_CACHE_DIR` at DevTools instead fixed it
immediately). If `uv`/`git`/`ollama` aren't found as bare commands in a *new* terminal, the user-level PATH
was updated but the current shell predates that change — open a fresh terminal, or use the full paths
above.

## What Hearth is

A fully offline, terminal-first AI coding agent (Python 3.12, uv) that indexes a local repo and answers
questions with `path:line` citations, edits files, runs commands/tests and uses git — every side effect
gated on human approval. Models are served by a loopback-only Ollama instance; the index is a local
SQLite file; nothing ever leaves the machine.

| Doc | Read it before |
|---|---|
| [docs/system-design.md](docs/system-design.md) | Any architecture, data-model or pipeline work. §6 indexing, §7 retrieval, §8 agent loop, §9 context, §13 schema |
| [docs/tech-stack.md](docs/tech-stack.md) | Adding or questioning any library. §19 is the dependency inventory — **do not add a runtime dep that isn't listed there without an ADR** |
| [docs/project-structure.md](docs/project-structure.md) | Deciding where code goes. §2 has the import-linter contracts; §6 the `pyproject.toml` skeleton |
| [docs/implementation-roadmap.md](docs/implementation-roadmap.md) | Starting a milestone — it holds the acceptance criteria and a per-milestone starter prompt |
| [docs/safety-and-tool-use.md](docs/safety-and-tool-use.md) | Anything touching tools, policy, approvals, edits or exec |
| [docs/model-recommendations.md](docs/model-recommendations.md) | Model/quantization/`num_ctx` choices and profile tables |
| [docs/future-extensions.md](docs/future-extensions.md) | Confirming something is deliberately deferred (autocomplete, MCP, LSP, web UI) |
| [docs/adr/](docs/adr/) | The decision record. New dependency or significant choice → add one, using `template.md` |

## Commands

```bash
uv sync                                          # install (--extra evals for the eval harness)
uv run pytest -q                                 # tests (excludes live + slow by default)
uv run pytest tests/unit/safety -q               # one directory
uv run pytest tests/unit/safety/test_policy.py::test_name -q   # one test
uv run pytest -m live                            # needs a running local Ollama + qwen3.5:4b
uv run pytest -m slow                            # benchmarks, large synthetic indexes
uv run ruff check --fix . && uv run ruff format .
uv run mypy src/hearth                           # --strict for core, safety, tools, llm
uv run lint-imports                              # layer contracts
uv run hearth eval retrieval                     # live model eval
```

Run `pytest`, `ruff`, `mypy` and `lint-imports` before finishing any task.

## Architecture in one pass

Strict layering, enforced by `import-linter` (contracts live in [.importlinter](.importlinter), explained in
[docs/project-structure.md](docs/project-structure.md) §2):

```
cli | server  →  workflows  →  core  →  tools  →  retrieval  →  indexing  →  safety  →  llm | storage | git  →  config  →  util
```

The load-bearing consequences:

- **Frontends never touch services.** `cli/` and `server/` talk to `core/` only through the typed event
  bus (`core/events.py`, `core/bus.py`). An approval request is just an event that waits for a response —
  that is what makes CLI, IDE and web frontends interchangeable.
- **Tools never call the LLM.** Tools are deterministic; only `core/runner.py` talks to models.
- **`safety/policy.py` is a pure function** — no filesystem, network or clock — so it can be tested
  exhaustively with tables and Hypothesis. Keep it that way.
- **Every filesystem access in `tools/` goes through `safety.paths.resolve_in_workspace()`.** A CI grep
  test fails if `open(`, `Path.write_text` or `os.remove` appear in `tools/` outside that helper.
- **SQLite is the single system of record**: `index.db` (files, chunks, FTS5, symbols, refs, float16
  vector BLOBs) and `state.db` (sessions, messages, checkpoints, grants, trust). No vector database —
  exact NumPy search behind the `VectorIndex` protocol.

Two design constraints drive almost everything and are easy to accidentally violate:

1. **Prompt layout must be prefix-cache-stable.** Static system prompt → project instructions →
   append-only history → current user message containing the retrieved `<context>` block. Never put
   volatile content (timestamps, fresh retrieval) early in the request. Always set `num_ctx` explicitly
   and keep it constant within a session — changing it forces a model reload, and exceeding it makes
   Ollama silently truncate the system prompt.
2. **Retrieval is hybrid by design.** BM25 + symbol lookup + dense embeddings + path search, fused with
   weighted RRF, then graph-expanded — *plus* agentic `grep`/`read_file` navigation. Neither alone
   covers the four question classes in [docs/system-design.md](docs/system-design.md) §1.2.

## Non-negotiable rules

1. **No runtime network access** except the configured loopback Ollama host. Never add telemetry, update
   checks, or libraries that download at runtime (tree-sitter grammars are pinned wheels, installed once).
   Tests run under `pytest-socket`; do not disable it.
2. **Build the safety spine before any write tool.** Path jail, policy engine, approval flow, checkpoints
   and audit land in M5 *before* `edit_file` can touch a file. Never retrofit safety.
3. **Security-critical modules get tests first, including adversarial cases**, and stay `mypy --strict`
   clean: all of `src/hearth/safety/`, plus `tools/write_fs.py`, `tools/edit_engine.py`, `tools/shell.py`,
   `tools/git_write.py`.
4. **Fail closed.** Unknown tool, invalid args, policy error, or a headless "ask" all resolve to *deny*.
   Approvals never time out into approval. The agent can never modify its own permissions — Hearth config,
   trust records and data dirs are protected paths.
5. **Deliberately absent, don't add them:** `git push/pull/fetch`, `git reset --hard`, `git clean`,
   `git rebase`, `git commit --amend`, force operations, `sudo`, package installation, arbitrary HTTP.
6. **Prompts live in `src/hearth/prompts/*.md`**, never inline in Python. Snapshot tests guard them;
   update snapshots deliberately and re-run the eval before merging a prompt change.
7. **A new runtime dependency needs an ADR** in `docs/adr/` and a question to the user first. The target
   is ~12 runtime deps.
8. **Never run Hearth's agent tools against real repositories in tests** — copy `tests/fixtures/repos/*`
   to tmp dirs. Also: develop and test against those fixtures rather than pointing this cloud assistant at
   the private codebases Hearth exists to protect.

## Dev machine — calibrate every live test and benchmark for it

ASUS ROG Strix G15: Ryzen 7 4800H, **RTX 3060 Laptop 6 GB VRAM**, 16 GB RAM, 512 GB SSD.
Windows 11 + Ollama for Windows, with Hearth developed inside WSL2 (Ubuntu, mirrored networking).
Repos live in `~/code` inside WSL, **never under `/mnt/c`** — indexing speed and path-safety semantics
both depend on it.

Models: `qwen3.5:4b` (daily, `num_ctx` 12288), `qwen3.5:9b` (eval-only, partial CPU offload),
`qwen3-embedding:0.6b`.

- Live tests use `qwen3.5:4b`, keep prompts under ~8K tokens, and run sequentially — never in parallel.
- Never load two chat models at once.
- Mark heavy tests `-m slow`; don't generate large fixture data into the repo (disk is limited).

## Style

- Pydantic v2 for every boundary object (events, tool args, config, profiles, plan schema); plain
  dataclasses for internal values.
- Async-first in `core`/`llm`/`tools`; no blocking calls in the event loop (`asyncio.to_thread` for heavy
  CPU work).
- No module-level side effects. Import numpy, tree_sitter and networkx lazily inside functions on CLI
  paths to keep startup fast.
- Modules are nouns (`chunker.py`, `policy.py`); tools are snake_case verbs (`edit_file`, `run_tests`)
  matching the names the model sees.
- Tool errors are *returned* as `ToolResult(error=...)`, never raised into the runner. Domain exceptions
  go in `<package>/errors.py`.
- Docstrings on public functions state invariants in security-critical code (e.g. "`prepare()` must not
  mutate the filesystem").

## Workflow

One milestone task per session. Start in plan mode, restate that milestone's acceptance criteria from
[docs/implementation-roadmap.md](docs/implementation-roadmap.md), list the files you will touch, then implement.
Build against `ScriptedProvider` first so the suite stays fast, offline and reproducible; connect real
models after. Keep this file current as milestones land.
