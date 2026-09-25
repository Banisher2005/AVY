"""Data models for AVY retrieval, decomposition, fusion, and citations."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A raw document loaded from the corpus."""

    doc_id: str
    title: str
    source: str
    filepath: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A chunked segment of a document ready for embedding."""

    chunk_id: str
    doc_id: str
    source: str
    title: str
    chunk_index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class Subquery:
    """An individual search query produced by multi-intent decomposition."""

    query_id: str
    subquery_text: str
    intent_label: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "query_id": self.query_id,
            "subquery_text": self.subquery_text,
            "intent_label": self.intent_label,
        }


@dataclass
class QueryDecomposition:
    """Result of multi-intent query decomposition."""

    original_utterance: str
    subqueries: list[Subquery] = field(default_factory=list)
    is_compound: bool = False
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_utterance": self.original_utterance,
            "subqueries": [sq.to_dict() for sq in self.subqueries],
            "is_compound": self.is_compound,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class RetrievalCandidate:
    """An individual chunk retrieved for a specific subquery."""

    query_id: str
    subquery: str
    document_id: str
    source: str
    chunk_id: str
    chunk_text: str
    rank: int
    score: float
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "subquery": self.subquery,
            "document_id": self.document_id,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "chunk_text": self.chunk_text,
            "rank": self.rank,
            "score": round(self.score, 4),
            "title": self.title,
            "metadata": self.metadata,
        }


@dataclass
class FusedEvidence:
    """A deduplicated, fused, and reranked evidence candidate."""

    chunk_id: str
    document_id: str
    source: str
    title: str
    text: str
    rrf_score: float
    rerank_score: float | None = None
    rank: int = 1
    matched_queries: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source": self.source,
            "title": self.title,
            "text": self.text,
            "rrf_score": round(self.rrf_score, 4),
            "rerank_score": round(self.rerank_score, 4) if self.rerank_score is not None else None,
            "rank": self.rank,
            "matched_queries": self.matched_queries,
            "metadata": self.metadata,
        }


@dataclass
class Citation:
    """Traceable citation link connecting an answer reference to source evidence."""

    index: int  # e.g., 1 for [1]
    document_id: str
    source: str
    title: str
    chunk_id: str
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "citation_label": f"[{self.index}]",
            "document_id": self.document_id,
            "source": self.source,
            "title": self.title,
            "chunk_id": self.chunk_id,
            "snippet": self.snippet,
        }
