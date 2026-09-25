"""Tests for Evidence Reranking and Latency Measurement."""

from avy.retrieval.models import FusedEvidence
from avy.retrieval.reranker import EvidenceReranker


def test_reranker_reordering_and_latency() -> None:
    reranker = EvidenceReranker(enabled=True)

    candidates = [
        FusedEvidence(
            chunk_id="c1",
            document_id="doc1",
            source="src1.md",
            title="General Samsung Information",
            text="Samsung manufactures smartphones and home electronics globally.",
            rrf_score=0.9,
            rank=1,
        ),
        FusedEvidence(
            chunk_id="c2",
            document_id="doc2",
            source="src2.md",
            title="AMD Xclipse GPU Specs",
            text="The Xclipse 940 GPU is built on AMD RDNA 3 architecture featuring hardware ray tracing.",
            rrf_score=0.85,
            rank=2,
        ),
    ]

    query = "Tell me about AMD Xclipse GPU ray tracing"
    reranked, duration_ms = reranker.rerank(query, candidates, top_n=2)

    assert len(reranked) == 2
    assert duration_ms >= 0.0

    # The chunk mentioning Xclipse, GPU, ray tracing must rank first after reranking
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].rank == 1
    assert reranked[0].rerank_score is not None


def test_reranker_disabled() -> None:
    reranker = EvidenceReranker(enabled=False)
    candidates = [
        FusedEvidence(
            chunk_id="c1",
            document_id="doc1",
            source="src1.md",
            title="Title",
            text="Text",
            rrf_score=0.9,
            rank=1,
        )
    ]

    reranked, duration_ms = reranker.rerank("Query", candidates, top_n=1)
    assert len(reranked) == 1
    assert duration_ms >= 0.0
    assert reranked[0].chunk_id == "c1"
