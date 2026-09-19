-- fts5vocab, in 'row' mode: one row per term with its document count and total occurrence
-- count across `chunks_fts`. Retrieval's query analysis and RRF fusion weighting use this
-- for term document-frequency statistics without a full-table scan
-- (docs/system-design.md §7).
CREATE VIRTUAL TABLE chunks_vocab USING fts5vocab(chunks_fts, 'row');
