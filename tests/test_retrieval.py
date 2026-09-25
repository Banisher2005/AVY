"""Tests for Vector Store, Embeddings, and Multi-Query Retrieval."""

from pathlib import Path

from avy.retrieval.embeddings import LocalDenseEmbedding
from avy.retrieval.models import QueryDecomposition, Subquery
from avy.retrieval.multi_query import MultiQueryRetriever
from avy.retrieval.vectorstore import FAISSVectorStore


def test_vectorstore_search(sample_vectorstore: FAISSVectorStore, local_embedding: LocalDenseEmbedding) -> None:
    q_vec = local_embedding.embed_query("Exynos CPU ray tracing")
    hits = sample_vectorstore.search(q_vec, top_k=2)

    assert len(hits) > 0
    top_chunk, score = hits[0]
    assert "exynos" in top_chunk.text.lower() or "ray tracing" in top_chunk.text.lower()
    assert score > 0.0


def test_multi_query_retrieval(sample_vectorstore: FAISSVectorStore, local_embedding: LocalDenseEmbedding) -> None:
    retriever = MultiQueryRetriever(
        vectorstore=sample_vectorstore,
        embeddings=local_embedding,
        top_k=3,
        threshold=0.0,
    )

    decomposition = QueryDecomposition(
        original_utterance="Tell me about Exynos and battery",
        subqueries=[
            Subquery(query_id="sq_1", subquery_text="Exynos 2400 processor specifications"),
            Subquery(query_id="sq_2", subquery_text="Samsung battery power charging"),
        ],
        is_compound=True,
    )

    candidates, duration_ms = retriever.retrieve(decomposition)

    assert len(candidates) > 0
    assert duration_ms >= 0.0

    # Verify provenance preservation
    for c in candidates:
        assert c.query_id in ("sq_1", "sq_2")
        assert c.subquery is not None
        assert c.document_id is not None
        assert c.source is not None
        assert c.chunk_id is not None
        assert c.rank >= 1
        assert c.score >= 0.0


def test_vectorstore_save_and_load(
    sample_vectorstore: FAISSVectorStore,
    local_embedding: LocalDenseEmbedding,
    tmp_path: Path,
) -> None:
    idx_file = tmp_path / "test.index"
    meta_file = tmp_path / "test.json"

    sample_vectorstore.save(idx_file, meta_file)
    assert meta_file.is_file()

    loaded_store = FAISSVectorStore.load(idx_file, meta_file)
    assert len(loaded_store._chunks) == len(sample_vectorstore._chunks)

    q_vec = local_embedding.embed_query("Galaxy battery 5000 mAh")
    hits = loaded_store.search(q_vec, top_k=2)
    assert len(hits) > 0
    assert "battery" in hits[0][0].text.lower()
