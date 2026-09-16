# 0003 — pyyaml under an `evals` optional extra

| | |
|---|---|
| Status | Accepted |
| Date | 2026-09-16 |
| Deciders | Project owner |

## Context

`docs/project-structure.md` §1 specifies eval data as YAML — `evals/retrieval/py_small.yaml`,
`evals/tasks/*.yaml` — and `hearth eval` ships as part of the CLI. But no YAML parser appeared in the
dependency list in `docs/tech-stack.md` §19.1, so the eval harness could not have parsed its own data.

The constraint in tension: `docs/system-design.md` §1.5 targets about 12 runtime dependencies, because
every dependency is a liability for an offline tool.

## Decision

Add **`pyyaml` under a new `evals` optional extra**. Eval data stays YAML. A default install does not
get a YAML parser.

Three implementation rules follow:

1. Import `yaml` **lazily, inside the eval harness only** — never at module import time, and nowhere
   else in the package. A missing extra must not break `hearth --version`.
2. `hearth eval` **degrades with an install hint**, not an `ImportError` traceback. A CLI smoke test
   asserts this with the extra absent.
3. **Always `yaml.safe_load`.** Eval files are repository data, and the default loader constructs
   arbitrary Python objects. A ruff `S506` violation here is not one to silence.

YAML is scoped to eval data. It never becomes a configuration format — configuration is TOML, and
`.hearth/config.toml` controls permissions.

## Alternatives considered

| Alternative | Why not |
|---|---|
| **Convert eval data to TOML, read with stdlib `tomllib`** | Genuinely tempting: zero new dependencies, and TOML is already the project's config format. Rejected because eval data is nested and hand-authored — 25+ questions per fixture repo, each with expected files and symbols — and TOML's array-of-tables syntax makes that materially harder to read and review in a diff. |
| **`pyyaml` as a dev dependency, `hearth eval` removed from the shipped CLI** | Keeps the runtime tree smallest, but `docs/implementation-roadmap.md` M2 and I4 both invoke `hearth eval` as a user-facing command, and running evals on your own repository is a documented workflow. Moving it out of the CLI would break that. |
| **`ruamel.yaml`** | Round-trip fidelity and comment preservation, neither of which eval data needs. Larger surface for no gain. |

## Consequences

- A normal `uv tool install hearth` stays at the §19.1 dependency count.
- `uv sync --extra evals` becomes part of the documented developer setup, and CI that runs evals must
  use it.
- One more failure mode to handle well: the harness must detect the missing extra and say so clearly.

## Revisit when

Eval data grows a second consumer outside the harness, or a future feature needs YAML on the runtime
path — at which point `pyyaml` should move into the main dependency list with its own ADR, rather than
being reached through the extra.
