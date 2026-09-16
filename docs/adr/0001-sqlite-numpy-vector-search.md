# 0001 — SQLite as the single store, with exact NumPy vector search

| | |
|---|---|
| Status | Accepted |
| Date | 2026-09-16 |
| Deciders | Project owner |

## Context

Hearth needs lexical search, a symbol graph, dense vectors, session history and checkpoints, on a
single-user laptop, fully offline. The obvious shape — SQLite for metadata plus a vector database —
introduces two stores that must agree with each other.

They will not always agree. The embedding phase is explicitly interruptible and resumable
(`docs/system-design.md` §6.7), so "chunk exists but its vector does not, or vice versa" becomes a
permanent class of bug reachable by Ctrl+C at the wrong moment.

## Decision

**SQLite is the single system of record**, in WAL mode with FTS5. Vectors are stored as float16,
L2-normalized BLOBs in the same database, and searched exactly with NumPy behind a `VectorIndex`
protocol (`upsert`, `delete`, `search`).

A chunk row, its contentless FTS5 row and its symbol rows are written in one transaction. Search is an
exact dot product — Ollama's embed endpoint returns normalized vectors, so cosine similarity is a matmul
over an in-memory matrix cache that is rebuilt lazily when dirty.

Full analysis and the rejected alternatives: `docs/tech-stack.md` §5.

## Alternatives considered

| Alternative | Why not |
|---|---|
| **sqlite-vec** | The closest call, and the designated escape hatch. Rejected for now only because exact search is simpler, loses no recall, and needs no extension loading — which some Python builds disable. |
| **LanceDB** | Pulls pyarrow and a large tree; adds a second on-disk format with its own consistency story. |
| **Chroma** | Heavy dependencies and historically telemetry-on-by-default — disqualifying for an offline tool even though it can be disabled. |
| **Qdrant / Weaviate / Milvus** | Separate server processes. Ends the "one file, zero setup" property. |
| **FAISS** | Large binary dependency, no persistence of its own, solving a scale problem Hearth does not have. |

## Consequences

- Transactional consistency across metadata, lexical index, symbols and vectors, for free.
- `index.db` stays disposable: it can always be rebuilt from the working tree.
- Vector RAM grows linearly. At 100K chunks × 1024 dims × float16 this is roughly 200 MB, against a
  retrieval budget of < 150 ms for the whole pipeline (`docs/system-design.md` §15).
- No approximate-search recall cliff to reason about, and no index build step.

## Measurements

To be recorded at M2 (`docs/implementation-roadmap.md`): lexical-only, dense-only and hybrid
recall@5/recall@10/MRR side by side on the fixture eval sets, plus retrieval latency at 100K synthetic
chunks. The acceptance bar is hybrid recall@10 ≥ 0.80, with hybrid beating both single-signal baselines.

## Revisit when

A repository exceeds ~1M chunks, or vector RAM exceeds 1.5 GB. Then add `SqliteVecIndex` or
`LanceDbIndex` behind the existing protocol, and require equal retrieval eval scores at lower memory
before switching the default.
