"""Tests for Evidence Fusion and Reciprocal Rank Fusion (RRF)."""

from avy.retrieval.fusion import EvidenceFusion
from avy.retrieval.models import RetrievalCandidate


def test_rrf_deduplication_and_boosting() -> None:
    fusion = EvidenceFusion(rrf_k=60)

    # Chunk A appears in both Query 1 (rank 1) and Query 2 (rank 1)
    # Chunk B appears only in Query 1 (rank 2)
    # Chunk C appears only in Query 2 (rank 2)
    candidates = [
        RetrievalCandidate(
            query_id="q1",
            subquery="query 1",
            document_id="doc_a",
            source="source_a.md",
            chunk_id="chunk_a",
            chunk_text="Text of chunk A",
            rank=1,
            score=0.9,
            title="Title A",
        ),
        RetrievalCandidate(
            query_id="q1",
            subquery="query 1",
            document_id="doc_b",
            source="source_b.md",
            chunk_id="chunk_b",
            chunk_text="Text of chunk B",
            rank=2,
            score=0.7,
            title="Title B",
        ),
        RetrievalCandidate(
            query_id="q2",
            subquery="query 2",
            document_id="doc_a",
            source="source_a.md",
            chunk_id="chunk_a",
            chunk_text="Text of chunk A",
            rank=1,
            score=0.88,
            title="Title A",
        ),
        RetrievalCandidate(
            query_id="q2",
            subquery="query 2",
            document_id="doc_c",
            source="source_c.md",
            chunk_id="chunk_c",
            chunk_text="Text of chunk C",
            rank=2,
            score=0.65,
            title="Title C",
        ),
    ]

    fused, duration_ms = fusion.fuse(candidates, top_n=5)

    assert len(fused) == 3  # Chunk A, B, C deduplicated
    assert duration_ms >= 0.0

    # Chunk A was retrieved by both queries, so it MUST be ranked #1
    top_evidence = fused[0]
    assert top_evidence.chunk_id == "chunk_a"
    assert top_evidence.rank == 1
    assert "query 1" in top_evidence.matched_queries
    assert "query 2" in top_evidence.matched_queries
    assert top_evidence.rrf_score == 1.0  # normalized max


def test_rrf_empty_candidates() -> None:
    fusion = EvidenceFusion(rrf_k=60)
    fused, duration_ms = fusion.fuse([], top_n=5)
    assert len(fused) == 0
    assert duration_ms >= 0.0
