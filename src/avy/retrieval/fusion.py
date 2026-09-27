"""Evidence Fusion and Deduplication using Reciprocal Rank Fusion (RRF)."""

from collections import defaultdict

from avy.core.timing import Stopwatch
from avy.retrieval.models import FusedEvidence, RetrievalCandidate


class EvidenceFusion:
    """Combines, deduplicates, and fuses multi-query retrieval candidates via RRF."""

    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k

    def fuse(
        self,
        candidates: list[RetrievalCandidate],
        top_n: int = 5,
    ) -> tuple[list[FusedEvidence], float]:
        """Apply Reciprocal Rank Fusion and deduplicate multi-query results.

        Formula:
            RRF_score(d) = sum_{q} (1 / (k + rank_q(d)))
        """
        sw = Stopwatch()
        if not candidates:
            return [], sw.stop()

        # Group by chunk_id
        chunk_map: dict[str, RetrievalCandidate] = {}
        rrf_scores: dict[str, float] = defaultdict(float)
        matched_queries: dict[str, list[str]] = defaultdict(list)

        for cand in candidates:
            cid = cand.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = cand

            # Add RRF contribution: 1 / (k + rank)
            contribution = 1.0 / (self.rrf_k + cand.rank)
            rrf_scores[cid] += contribution
            if cand.subquery not in matched_queries[cid]:
                matched_queries[cid].append(cand.subquery)

        # Sort candidates descending by accumulated RRF score
        sorted_chunks = sorted(
            rrf_scores.keys(),
            key=lambda cid: rrf_scores[cid],
            reverse=True,
        )

        # Max score for normalization
        max_score = max(rrf_scores.values()) if rrf_scores else 1.0

        fused_list: list[FusedEvidence] = []
        for rank, cid in enumerate(sorted_chunks[:top_n], start=1):
            cand = chunk_map[cid]
            norm_score = rrf_scores[cid] / max_score if max_score > 0 else rrf_scores[cid]
            best_vector_score = max(c.score for c in candidates if c.chunk_id == cid)

            fused_list.append(
                FusedEvidence(
                    chunk_id=cid,
                    document_id=cand.document_id,
                    source=cand.source,
                    title=cand.title,
                    text=cand.chunk_text,
                    rrf_score=norm_score,
                    vector_score=best_vector_score,
                    rank=rank,
                    matched_queries=matched_queries[cid],
                    metadata=cand.metadata,
                )
            )

        elapsed_ms = sw.stop()
        return fused_list, elapsed_ms
