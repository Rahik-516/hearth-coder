# Hearth

**A fully offline, private AI coding assistant and project agent built on local Ollama models.**

Hearth indexes a local repository and answers questions about it with `file:line` citations. It proposes
refactors, writes tests and documentation, and can edit files, run commands, run tests and use git. Every side
effect needs explicit human approval.

Everything runs on your machine. Models are served by a loopback-only Ollama instance, the index is a local SQLite
file, and no code, prompt or telemetry ever leaves the machine.

---

## Status: pre-alpha — scaffolding only

**Nothing works yet.** The repository currently contains the design baseline (`docs/`) and the project skeleton.
`hearth --version` runs; no other command is implemented.

Progress tracks [docs/implementation-roadmap.md](docs/implementation-roadmap.md):

| Phase | Delivers | Status |
|---|---|---|
| **0** | Scaffolding, config, LLM gateway, event bus, `hearth doctor` | Scaffolding done; config/LLM/events/doctor remain |
| **1 (M1–M6)** | Indexing, hybrid retrieval, chat with citations, tools, safety spine, agent mode | Not started |
| **2** | Repo map, more languages, context management, plan mode, workflows | Not started |
| **3** | Sandboxing, VS Code extension, LSP, MCP, web UI | Not started |

---

## Requirements

- **Python 3.12+** and [uv](https://docs.astral.sh/uv/)
- **git** (used for file discovery and the git tools)
- **[Ollama](https://ollama.com/)** running on loopback
- Optional: **ripgrep** (`rg`) for faster search, `nvidia-smi` for VRAM checks

Hardware guidance and per-tier model choices are in
[docs/model-recommendations.md](docs/model-recommendations.md). The reference development machine is a laptop with
a 6 GB GPU — Hearth is designed to be useful there, not only on workstations.

## Install

```bash
uv sync
```

With the eval harness:

```bash
uv sync --extra evals
```

Offline installation from a wheelhouse is described in `scripts/build_wheelhouse.sh`.

## Models

```bash
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
```

Then configure the Ollama server for privacy and memory behaviour —
[docs/model-recommendations.md §5](docs/model-recommendations.md) has the exact environment variables, including
`OLLAMA_HOST=127.0.0.1:11434` and `OLLAMA_NO_CLOUD=1`.

## Quickstart (once implemented)

```bash
hearth doctor          # check Ollama, models, VRAM, FTS5, git, rg
hearth init            # create the project data directory
hearth index           # build the lexical index, then embeddings in the background
hearth chat            # ask questions, with path:line citations
```

## Offline guarantees

1. **Zero egress.** Hearth connects only to the configured Ollama host, and only on loopback. Non-loopback hosts
   and cloud model tags are refused.
2. **Enforced by tests.** The whole suite runs under `pytest-socket` with every socket blocked except loopback.
3. **No telemetry, no update checks, no runtime downloads.** Tree-sitter grammars ship as pinned wheels.
4. **Verifiable.** `hearth doctor` reports on host, models and cloud settings.

## Safety model

Reads are allowed. Every side effect asks, unless a rule you wrote covers it.

- Edits are previewed as diffs, checkpointed, and undoable with `/undo`.
- Commands are classified, shown as exact argv, and run with a scrubbed environment.
- Writes are confined to the workspace; config, git internals and credential paths are protected.
- The agent can never modify its own permissions.
- Headless mode fails closed: every "ask" becomes "deny".

Full specification: [docs/safety-and-tool-use.md](docs/safety-and-tool-use.md).

## Development

```bash
uv run pytest -q                                    # fast suite (excludes live + slow)
uv run ruff check --fix . && uv run ruff format .
uv run mypy src/hearth
uv run lint-imports                                 # layer contracts
```

Architecture and conventions live in [docs/](docs/); read
[docs/project-structure.md](docs/project-structure.md) before adding a module and
[docs/tech-stack.md](docs/tech-stack.md) §19 before adding a dependency. [CLAUDE.md](CLAUDE.md) is the working
brief for AI-assisted development of this repository.

## License

MIT — see [LICENSE](LICENSE).
