"""Multi-Query Retrieval engine executing subquery searches."""

from avy.core.timing import Stopwatch
from avy.retrieval.embeddings import BaseEmbedding
from avy.retrieval.models import QueryDecomposition, RetrievalCandidate
from avy.retrieval.vectorstore import FAISSVectorStore


class MultiQueryRetriever:
    """Executes independent vector searches across all decomposed subqueries."""

    def __init__(
        self,
        vectorstore: FAISSVectorStore,
        embeddings: BaseEmbedding,
        top_k: int = 5,
        threshold: float = 0.20,
    ) -> None:
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.top_k = top_k
        self.threshold = threshold

    def retrieve(
        self,
        decomposition: QueryDecomposition,
        top_k: int | None = None,
    ) -> tuple[list[RetrievalCandidate], float]:
        """Execute vector search for each subquery, preserving complete metadata."""
        k = top_k or self.top_k
        sw = Stopwatch()
        all_candidates: list[RetrievalCandidate] = []

        for subquery in decomposition.subqueries:
            # 1. Generate query embedding vector
            q_vec = self.embeddings.embed_query(subquery.subquery_text)

            # 2. Search FAISS index
            search_hits = self.vectorstore.search(
                query_vector=q_vec,
                top_k=k,
                threshold=self.threshold,
            )

            # 3. Preserve provenance
            for rank, (chunk, score) in enumerate(search_hits, start=1):
                candidate = RetrievalCandidate(
                    query_id=subquery.query_id,
                    subquery=subquery.subquery_text,
                    document_id=chunk.doc_id,
                    source=chunk.source,
                    chunk_id=chunk.chunk_id,
                    chunk_text=chunk.text,
                    rank=rank,
                    score=score,
                    title=chunk.title,
                    metadata=chunk.metadata,
                )
                all_candidates.append(candidate)

        elapsed_ms = sw.stop()
        return all_candidates, elapsed_ms
