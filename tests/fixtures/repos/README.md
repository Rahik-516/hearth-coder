# Fixture repositories

All development and testing runs against these, never against a real checkout
(docs/project-structure.md §5). Tests copy them to a tmp directory first — use the
`fixture_repo` fixture in `tests/conftest.py`.

They also carry a privacy purpose: Hearth is built with a cloud assistant, so the code it
is pointed at must be code you are happy to share. These fixtures are that code.

| Repo | Contents | Lands in |
|---|---|---|
| `py_small/` | ~2K LOC Python service with tests. Needs nested classes, decorators, an oversized function and at least one file with a deliberate syntax error, so the chunker snapshots cover them | Phase 0 task 7 / M1 |
| `ts_small/` | ~2K LOC TypeScript app with tests. Include `.tsx` so the TSX grammar is exercised | Phase 0 task 7 / M1 |
| `polyglot/` | Python + TypeScript + Markdown + YAML in one tree, for language detection and mixed-language retrieval | M1 |
| `malicious/` | Adversarial inputs: instruction-injection comments, symlinks escaping the root, files matching secret patterns (`.env`, `*.pem`), a `.hearth/config.toml` that tries to allowlist something dangerous, and a generated-code marker | M1, then the security suite in M4–M6 |

`malicious/` is the one that earns its keep. The security tests in
docs/safety-and-tool-use.md §15 assert that its secrets are never indexed, its symlinks
never resolve outside the workspace, and its injected instructions never reach the policy
engine as permissions.

Generate scaffolding for a new fixture with `scripts/gen_fixture_repo.py`.
