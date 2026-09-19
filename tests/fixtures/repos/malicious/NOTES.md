# Fixture notes

| Path | What it tests |
|---|---|
| `.env`, `.env.production`, `server.pem`, `id_rsa`, `.npmrc`, `terraform.tfstate` | Secret files must never appear in `files` or `chunks` (M1 acceptance criterion) |
| `src/helpers.py`, `README.md` | Instruction-like content must be treated as data and badged, never followed |
| `.hearth/config.toml` | Allow rules from an untrusted project config must be dropped |
| `src/schema_pb2.py` | Generated-code detection: metadata only, not chunked |
| `escape_symlink`, `config/relative_escape` | Symlinks pointing outside the workspace must not be followed |
| `src/safe.py` | A control: ordinary code that *should* be indexed |

Every credential here is fabricated. Nothing here is a real key or token.
