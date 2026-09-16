# Updating vendored tree-sitter tag queries

Tag queries decide what counts as a definition, a reference and an import
(`docs/system-design.md` §6.6). They are the symbol graph's schema, so they are **vendored
and reviewed**, never fetched at runtime (`docs/tech-stack.md` §6.3).

This procedure is deliberately manual. A query change alters symbol extraction across the
whole index, and that should never arrive as a transitive dependency bump.

## When

- Adding a language.
- A grammar upgrade changes node names the queries rely on.
- Symbol extraction is measurably wrong — missing definitions, references landing on the
  wrong node kind.

## Procedure

1. **Find the upstream source.** Grammar repositories publish `queries/tags.scm`. Note the
   repository, the commit SHA and the licence.
2. **Copy it to `src/hearth/indexing/queries/<lang>/tags.scm`.** Add a header comment with
   the source URL, the commit SHA and the date.
3. **Reconcile with Hearth's captures.** Hearth needs more than upstream tags files usually
   provide:
   - `@definition.*` with enough surrounding nodes to recover a signature,
   - `@reference.call`, `@reference.type`, `@reference.attribute`,
   - imports with the module spec and imported names, and
   - an exported/public marker where the language expresses one.
4. **Check the grammar ABI.** The query must load against the pinned `tree-sitter` runtime.
   The grammar-loading test fails fast if it does not.
5. **Re-run the chunker snapshots.** `uv run pytest tests/unit/indexing -q`. Review every
   snapshot diff by hand — a changed chunk boundary invalidates embeddings for that file.
6. **Re-run the retrieval eval.** `uv run hearth eval retrieval`. Symbol extraction feeds
   the symbol retriever and the repo map, so a regression shows up here.
7. **Record the numbers** in the commit message: before and after recall@10.

## Licences

Grammar repositories are typically MIT, but confirm per language and record it in the file
header. The vendored query is a copy of someone else's work; treat it that way.
