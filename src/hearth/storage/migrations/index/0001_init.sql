-- Index schema (docs/system-design.md §13.1).
--
-- index.db is DISPOSABLE: it can always be rebuilt from the working tree. That is what
-- lets `hearth index --rebuild` simply drop the file, and why migrations here may be
-- destructive when a future change is awkward.
--
-- Summaries (§6.9) are Phase 2 and deliberately absent.

CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
-- keys: schema_version, embed_model, embed_dims, created_at, hearth_version, root_path

CREATE TABLE files (
  id            INTEGER PRIMARY KEY,
  path          TEXT    NOT NULL UNIQUE,      -- POSIX, relative to the workspace root
  language      TEXT,
  size_bytes    INTEGER NOT NULL,
  mtime_ns      INTEGER NOT NULL,
  content_hash  TEXT    NOT NULL,             -- blake2b-128 hex
  is_generated  INTEGER NOT NULL DEFAULT 0,
  parse_status  TEXT    NOT NULL,             -- ok | partial | failed | skipped
  indexed_at    INTEGER NOT NULL
);
CREATE INDEX files_language ON files(language);

CREATE TABLE chunks (
  id              INTEGER PRIMARY KEY,
  file_id         INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  kind            TEXT    NOT NULL,           -- function | method | class | class_skeleton
                                              -- | module_skeleton | preamble | section
                                              -- | window | summary
  symbol_path     TEXT,                       -- "InvoiceService > finalize"
  start_line      INTEGER NOT NULL,
  end_line        INTEGER NOT NULL,
  start_byte      INTEGER NOT NULL,
  end_byte        INTEGER NOT NULL,
  text            TEXT    NOT NULL,
  embed_text_hash TEXT    NOT NULL,           -- sha256 of header+text; the embed cache key
  token_estimate  INTEGER NOT NULL
);
CREATE INDEX chunks_file ON chunks(file_id);
CREATE INDEX chunks_symbol_path ON chunks(symbol_path);

-- Contentless FTS5: the text lives in `chunks`, so it is not duplicated here. The writer
-- inserts and deletes FTS rows with a matching rowid, inside the same transaction as the
-- chunks themselves.
--
-- contentless_delete=1 (SQLite 3.43+) is what makes the delete side safe. Without it, a
-- contentless table can only delete a row by replaying its *original* column values, so
-- any later change to how search_text is built would leave old rows permanently
-- undeletable -- an index that silently accumulates hits for code that no longer exists.
-- It costs a little extra stored state per row; correctness by construction is worth it.
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  search_text,
  symbol_path,
  path,
  content='',
  contentless_delete=1,
  tokenize = "unicode61 remove_diacritics 2 tokenchars '_'"
);

CREATE TABLE embeddings (
  embed_text_hash TEXT    NOT NULL,
  model_id        TEXT    NOT NULL,
  dims            INTEGER NOT NULL,
  vector          BLOB    NOT NULL,           -- float16 little-endian, L2-normalized
  PRIMARY KEY (embed_text_hash, model_id, dims)
);

CREATE TABLE symbols (
  id          INTEGER PRIMARY KEY,
  file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  name        TEXT    NOT NULL,
  kind        TEXT    NOT NULL,               -- function | method | class | interface
                                              -- | type | const | var
  parent_id   INTEGER REFERENCES symbols(id),
  start_line  INTEGER NOT NULL,
  end_line    INTEGER NOT NULL,
  signature   TEXT,
  exported    INTEGER
);
CREATE INDEX symbols_name ON symbols(name);
CREATE INDEX symbols_name_nocase ON symbols(name COLLATE NOCASE);
CREATE INDEX symbols_file ON symbols(file_id);

CREATE TABLE refs (
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  name    TEXT    NOT NULL,
  line    INTEGER NOT NULL,
  kind    TEXT    NOT NULL                    -- call | type | attribute | other
);
CREATE INDEX refs_name ON refs(name);
CREATE INDEX refs_file ON refs(file_id);

CREATE TABLE imports (
  file_id          INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  module_spec      TEXT    NOT NULL,
  names            TEXT,                      -- JSON array
  resolved_file_id INTEGER REFERENCES files(id)
);
CREATE INDEX imports_file ON imports(file_id);
CREATE INDEX imports_spec ON imports(module_spec);
