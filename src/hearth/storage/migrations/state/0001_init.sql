-- Session state (docs/system-design.md §13.2).
--
-- Unlike index.db, this is NOT disposable: it holds the user's conversation history,
-- their checkpoints, and the trust records that gate project permissions. Migrations here
-- must preserve data, and `--rebuild` must never touch this file.
--
-- M3 uses sessions and messages. The remaining tables are created now because they are
-- part of the same schema version and creating them later would mean a migration that
-- edits a file people already have data in.

CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Timestamps are epoch MILLISECONDS. Seconds tie when two sessions are created in the
-- same second, which makes `hearth resume` pick non-deterministically between them.
CREATE TABLE sessions (
  id               TEXT PRIMARY KEY,
  created_at       INTEGER NOT NULL,  -- epoch ms
  updated_at       INTEGER NOT NULL,  -- epoch ms
  title            TEXT,
  mode             TEXT    NOT NULL DEFAULT 'chat',
  permission_level TEXT    NOT NULL DEFAULT 'supervised',
  model            TEXT,
  num_ctx          INTEGER,
  workspace        TEXT
);
CREATE INDEX sessions_updated ON sessions(updated_at DESC);

CREATE TABLE messages (
  session_id     TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  seq            INTEGER NOT NULL,
  role           TEXT    NOT NULL,          -- system | user | assistant | tool
  content_json   TEXT    NOT NULL,          -- the serialized Message
  token_estimate INTEGER NOT NULL DEFAULT 0,
  -- Cache epoch. History is append-only within an epoch; compaction starts a new one
  -- (docs/system-design.md §9.2). Stored so a resumed session rebuilds the same prefix.
  epoch          INTEGER NOT NULL DEFAULT 0,
  created_at     INTEGER NOT NULL,
  PRIMARY KEY (session_id, seq)
);

CREATE TABLE tool_calls (
  id          TEXT PRIMARY KEY,
  session_id  TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  step        INTEGER NOT NULL,
  tool        TEXT    NOT NULL,
  args_json   TEXT    NOT NULL,
  decision    TEXT,                          -- allow | ask | deny
  decided_by  TEXT,                          -- rule | user | grant | default
  rule_id     TEXT,
  started_at  INTEGER,
  finished_at INTEGER,
  exit_code   INTEGER,
  output_blob TEXT
);
CREATE INDEX tool_calls_session ON tool_calls(session_id, step);

CREATE TABLE checkpoints (
  id          INTEGER PRIMARY KEY,
  session_id  TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  step        INTEGER NOT NULL,
  path        TEXT    NOT NULL,
  before_blob TEXT,                          -- NULL means the file did not exist
  after_blob  TEXT,
  created_at  INTEGER NOT NULL,
  reverted    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX checkpoints_session ON checkpoints(session_id, step DESC);

CREATE TABLE grants (
  session_id TEXT    NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  grant_key  TEXT    NOT NULL,
  created_at INTEGER NOT NULL,
  PRIMARY KEY (session_id, grant_key)
);

-- Trust is keyed by the hash of the project config, not by path: any edit to that file
-- invalidates trust and the allow rules stop applying until re-trusted
-- (docs/safety-and-tool-use.md §5.5).
CREATE TABLE trust (
  project_config_sha256 TEXT PRIMARY KEY,
  trusted_at            INTEGER NOT NULL
);
