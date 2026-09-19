"""Chunk enrichment: the context header and the search text.

Two different texts are derived from every chunk, for two different consumers
(docs/system-design.md §6.5):

* **Embedding text** — the chunk prefixed with a compact context header. This matters
  disproportionately for short methods, whose meaning depends on the class and file around
  them: ``def finalize(self, invoice_id)`` embedded alone is nearly contentless, while the
  same code under ``scope: class InvoiceService`` is not.

* **Search text** — the chunk plus expanded identifier forms, so FTS5 can match
  ``invoice service`` against ``InvoiceService``. Built by ``hearth.util.text``.

The embedding text is also what the embed cache is keyed on, so the header must be stable:
changing its format invalidates every cached vector.
"""

from __future__ import annotations

from dataclasses import dataclass

from hearth.indexing.chunker import Chunk
from hearth.util.hashing import blob_hash
from hearth.util.text import build_search_text

_HEADER_SEPARATOR = "---"


@dataclass(frozen=True)
class EnrichedChunk:
    """A chunk with its derived texts and cache key."""

    chunk: Chunk
    embed_text: str
    search_text: str
    embed_text_hash: str


def build_context_header(
    *,
    path: str,
    language: str | None,
    symbol_path: str | None,
    kind: str,
) -> str:
    """The compact header prepended to embedding text.

    Kept short and in a fixed field order. It is part of the embed cache key, so a format
    change re-embeds the entire repository.
    """
    lines = [f"path: {path}"]
    if language:
        lines.append(f"language: {language}")
    if symbol_path:
        lines.append(f"scope: {symbol_path}")
    elif kind in ("module_skeleton", "preamble"):
        lines.append(f"scope: {kind.replace('_', ' ')}")
    return "\n".join(lines)


def compose_embed_text(header: str, body: str) -> str:
    """Join a header and a body into embedding text.

    **The only place that knows this format.** ``embed_text_hash`` is the hash of the
    result, and the embedder reconstructs the same string from the database rather than
    storing it twice. If the two sides ever disagreed, the cache would look populated
    while holding vectors for text that was never embedded — silently, and only visible as
    poor retrieval.
    """
    return f"{header}\n{_HEADER_SEPARATOR}\n{body}"


def build_embed_text(chunk: Chunk, *, path: str, language: str | None) -> str:
    """Header plus chunk body, as sent to the embedding model."""
    header = build_context_header(
        path=path,
        language=language,
        symbol_path=chunk.symbol_path,
        kind=chunk.kind,
    )
    return compose_embed_text(header, chunk.text)


def enrich(chunk: Chunk, *, path: str, language: str | None) -> EnrichedChunk:
    """Derive both texts and the cache key for one chunk."""
    embed_text = build_embed_text(chunk, path=path, language=language)
    return EnrichedChunk(
        chunk=chunk,
        embed_text=embed_text,
        search_text=build_search_text(chunk.text),
        embed_text_hash=blob_hash(embed_text),
    )


def enrich_all(chunks: list[Chunk], *, path: str, language: str | None) -> list[EnrichedChunk]:
    return [enrich(chunk, path=path, language=language) for chunk in chunks]
