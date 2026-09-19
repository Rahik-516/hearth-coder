"""Shared retrieval types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Retriever(StrEnum):
    """Which signal produced a candidate. Kept on results for `--explain`."""

    DENSE = "dense"
    BM25 = "bm25"
    SYMBOL = "symbol"
    PATH = "path"


@dataclass
class Candidate:
    """One chunk proposed by one retriever."""

    chunk_id: int
    path: str
    kind: str
    symbol_path: str | None
    start_line: int
    end_line: int
    text: str
    score: float
    rank: int
    retriever: Retriever
    language: str | None = None


@dataclass
class FusedResult:
    """A chunk after fusion, carrying enough provenance to explain its rank."""

    chunk_id: int
    path: str
    kind: str
    symbol_path: str | None
    start_line: int
    end_line: int
    text: str
    language: str | None = None

    score: float = 0.0
    #: Per-retriever rank, for `--explain`. Absent means that retriever did not find it.
    ranks: dict[Retriever, int] = field(default_factory=dict)
    #: Per-retriever contribution to the fused score.
    contributions: dict[Retriever, float] = field(default_factory=dict)
    #: Multiplicative adjustments applied, as (label, factor).
    adjustments: list[tuple[str, float]] = field(default_factory=list)

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"

    @property
    def retrievers(self) -> list[Retriever]:
        return sorted(self.ranks, key=lambda r: self.ranks[r])
