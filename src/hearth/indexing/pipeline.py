"""The indexing pipeline.

``Scan -> Filter -> ChangeDetect -> Parse -> Extract -> Chunk -> Enrich -> Write``
(docs/system-design.md §6.1).

Two properties this module is responsible for:

* **One transaction per batch.** Chunks, their FTS rows and their symbols land together or
  not at all, so a Ctrl+C never leaves the index internally inconsistent.
* **Counters that prove the work.** ``IndexStats.parsed`` backs the M1 acceptance
  criterion that re-running on an unchanged tree performs zero parses. A counter is the
  only honest way to assert that; timing would be a proxy.

This is the lexical phase only. Embeddings fill in afterwards, in the background
(M2) — retrieval works on whatever is available, which is what makes a large repo usable
within a minute or two rather than after a full embed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from hearth.indexing.change_detector import ChangeSet, detect_changes
from hearth.indexing.chunker import chunk_file
from hearth.indexing.enrich import enrich_all
from hearth.indexing.filters import PathFilter, is_too_large, looks_generated
from hearth.indexing.languages import detect_language, grammar_available, is_parseable
from hearth.indexing.parser import ParseResult, parse
from hearth.indexing.scanner import ScannedFile, scan
from hearth.indexing.symbols import ExtractionResult, extract
from hearth.storage.index_repo import (
    ChunkRecord,
    FilePayload,
    FileRecord,
    ImportRecord,
    IndexRepository,
    RefRecord,
    SymbolRecord,
)
from hearth.util.hashing import content_hash
from hearth.util.text import decode_text, is_probably_binary

#: Files written per transaction. Large enough to amortise commit cost, small enough that
#: a cancelled run loses little work.
DEFAULT_BATCH_SIZE = 64

ProgressCallback = Callable[["IndexProgress"], None]


@dataclass(frozen=True)
class IndexProgress:
    phase: str
    done: int
    total: int
    current_path: str | None = None


@dataclass
class IndexStats:
    """What an indexing run actually did."""

    scanned: int = 0
    added: int = 0
    modified: int = 0
    deleted: int = 0
    unchanged: int = 0

    #: Files handed to tree-sitter. Zero on a no-op re-run — the M1 criterion.
    parsed: int = 0
    #: Files opened and hashed by change detection.
    hashed: int = 0

    chunks_written: int = 0
    symbols_written: int = 0

    skipped_binary: int = 0
    skipped_large: int = 0
    skipped_unreadable: int = 0
    generated_metadata_only: int = 0
    parse_failures: int = 0

    skipped_by_reason: dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def files_written(self) -> int:
        return self.added + self.modified

    def summary(self) -> str:
        return (
            f"{self.files_written} file(s) indexed "
            f"({self.added} added, {self.modified} modified, {self.deleted} deleted), "
            f"{self.chunks_written} chunks, {self.symbols_written} symbols"
        )


class Indexer:
    """Runs the lexical indexing phase over a workspace."""

    def __init__(
        self,
        *,
        root: Path,
        repository: IndexRepository,
        path_filter: PathFilter | None = None,
        max_file_bytes: int = 1_000_000,
        batch_size: int = DEFAULT_BATCH_SIZE,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self._root = root.resolve()
        self._repo = repository
        self._filter = path_filter
        self._max_file_bytes = max_file_bytes
        self._batch_size = batch_size
        self._on_progress = on_progress

    def run(self, *, force: bool = False) -> IndexStats:
        """Index the workspace, returning counters describing the work done."""
        import time

        started = time.monotonic()
        stats = IndexStats()

        self._emit("scanning", 0, 0)
        scan_result = scan(self._root, path_filter=self._filter)
        stats.scanned = len(scan_result.files)
        stats.skipped_by_reason = dict(scan_result.skipped)

        changes = detect_changes(
            scan_result.files,
            self._repo.file_states(),
            force=force,
            # A grammar installed since the last run makes previously skipped files
            # parseable without touching their bytes, which change detection cannot see.
            recheck=self._repo.paths_now_parseable(grammar_available()),
        )
        stats.added = len(changes.added)
        stats.modified = len(changes.modified)
        stats.unchanged = len(changes.unchanged)
        stats.hashed = changes.hashed

        self._remove_deleted(changes, stats)
        self._index_files(changes.needs_indexing, stats)

        self._repo.set_meta("root_path", self._root.as_posix())
        stats.duration_s = time.monotonic() - started
        self._emit("done", stats.files_written, stats.files_written)
        return stats

    def index_one(self, relative_path: str) -> bool:
        """Re-index a single file, synchronously. Returns whether anything was written.

        Called by the write tools immediately after an edit lands
        (docs/safety-and-tool-use.md §7.2, execute step 6). Synchronous on purpose: the
        model's very next step is often a `search_code` or `find_symbol` that must see the
        edit it just made, and an index that lags one step behind a write teaches the
        model that its own changes did not take effect.

        A deleted file is removed from the index rather than treated as an error, so this
        is also the right call after a write that removes something.
        """
        absolute = self._root / relative_path
        if not absolute.is_file():
            self._repo.delete_files([relative_path])
            return True

        try:
            stat = absolute.stat()
        except OSError:
            return False

        file = ScannedFile(
            relative_path=relative_path,
            absolute_path=absolute,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )

        stats = IndexStats()
        payload = self._build_payload(file, stats)
        if payload is None:
            return False

        self._flush([payload], stats)
        return True

    # ------------------------------------------------------------- internals

    def _remove_deleted(self, changes: ChangeSet, stats: IndexStats) -> None:
        if not changes.deleted:
            return
        stats.deleted = self._repo.delete_files(changes.deleted)

    def _index_files(self, files: list[ScannedFile], stats: IndexStats) -> None:
        total = len(files)
        batch: list[FilePayload] = []

        for position, file in enumerate(files, start=1):
            self._emit("indexing", position, total, file.relative_path)

            payload = self._build_payload(file, stats)
            if payload is None:
                continue

            batch.append(payload)
            if len(batch) >= self._batch_size:
                self._flush(batch, stats)
                batch = []

        if batch:
            self._flush(batch, stats)

    def _flush(self, batch: list[FilePayload], stats: IndexStats) -> None:
        self._repo.replace_files(batch)
        for payload in batch:
            stats.chunks_written += len(payload.chunks)
            stats.symbols_written += len(payload.symbols)

    def _build_payload(self, file: ScannedFile, stats: IndexStats) -> FilePayload | None:
        """Read, parse, extract and chunk one file. None when it should be skipped."""
        if is_too_large(file.size_bytes, max_bytes=self._max_file_bytes):
            stats.skipped_large += 1
            return None

        try:
            raw = file.absolute_path.read_bytes()
        except OSError:
            stats.skipped_unreadable += 1
            return None

        if is_probably_binary(raw):
            stats.skipped_binary += 1
            return None

        text = decode_text(raw)
        first_line = text.split("\n", 1)[0] if text else None
        language = detect_language(file.relative_path, first_line=first_line)

        record = FileRecord(
            path=file.relative_path,
            language=language,
            size_bytes=file.size_bytes,
            mtime_ns=file.mtime_ns,
            content_hash=content_hash(raw),
            parse_status="skipped",
        )

        # Generated files are indexed as metadata only: they bloat the index and rarely
        # answer a question worth asking (docs/system-design.md §6.2).
        if looks_generated(text):
            record.is_generated = True
            stats.generated_metadata_only += 1
            return FilePayload(file=record)

        parse_result = self._parse(raw, language, stats)
        record.parse_status = parse_result.status if parse_result else "skipped"

        extraction = self._extract(parse_result, raw, language)
        chunking = chunk_file(
            source=raw,
            text=text,
            language=language,
            parse_result=parse_result,
            extraction=extraction,
        )

        enriched = enrich_all(chunking.chunks, path=file.relative_path, language=language)

        return FilePayload(
            file=record,
            chunks=[
                ChunkRecord(
                    kind=item.chunk.kind,
                    symbol_path=item.chunk.symbol_path,
                    start_line=item.chunk.start_line,
                    end_line=item.chunk.end_line,
                    start_byte=item.chunk.start_byte,
                    end_byte=item.chunk.end_byte,
                    text=item.chunk.text,
                    embed_text_hash=item.embed_text_hash,
                    token_estimate=item.chunk.token_estimate,
                )
                for item in enriched
            ],
            symbols=_symbol_records(extraction),
            refs=[
                RefRecord(name=ref.name, line=ref.line, kind=ref.kind)
                for ref in (extraction.references if extraction else [])
            ],
            imports=[
                ImportRecord(module_spec=imp.module_spec, names=list(imp.names))
                for imp in (extraction.imports if extraction else [])
            ],
        )

    def _parse(self, raw: bytes, language: str | None, stats: IndexStats) -> ParseResult | None:
        if language is None or not is_parseable(language):
            return None

        stats.parsed += 1
        result = parse(raw, language)
        if result.status == "failed":
            stats.parse_failures += 1
        return result

    @staticmethod
    def _extract(
        parse_result: ParseResult | None, raw: bytes, language: str | None
    ) -> ExtractionResult | None:
        if parse_result is None or parse_result.root is None or language is None:
            return None
        return extract(parse_result.root, raw, language)

    def _emit(self, phase: str, done: int, total: int, path: str | None = None) -> None:
        if self._on_progress is not None:
            self._on_progress(IndexProgress(phase=phase, done=done, total=total, current_path=path))


def _symbol_records(extraction: ExtractionResult | None) -> list[SymbolRecord]:
    """Flatten definitions into rows, resolving parent links to batch indices."""
    if extraction is None:
        return []

    index_by_definition = {id(d): i for i, d in enumerate(extraction.definitions)}
    records: list[SymbolRecord] = []

    for definition in extraction.definitions:
        parent_index = (
            index_by_definition.get(id(definition.parent)) if definition.parent is not None else None
        )
        records.append(
            SymbolRecord(
                name=definition.name,
                kind=definition.kind,
                start_line=definition.start_line,
                end_line=definition.end_line,
                signature=definition.signature,
                exported=definition.exported,
                parent_index=parent_index,
            )
        )
    return records
