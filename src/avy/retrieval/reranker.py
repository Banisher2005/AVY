"""Second-stage cross-encoder and lexical reranker for fused evidence."""

import re

from avy.core.timing import Stopwatch
from avy.retrieval.models import FusedEvidence


class EvidenceReranker:
    """Reranks fused evidence candidates using cross-attention or lexical scoring."""

    def __init__(
        self,
        enabled: bool = True,
        model_name: str = "BAAI/bge-reranker-base",
    ) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self._cross_encoder = None
        self._init_cross_encoder()

    def _init_cross_encoder(self) -> None:
        """Attempt to lazily load ONNX cross-encoder if enabled."""
        if not self.enabled:
            return

        try:
            from fastembed import TextCrossEncoder

            self._cross_encoder = TextCrossEncoder(model_name=self.model_name)
        except Exception:
            # Fall back to lexical cross-matching if model download is unavailable
            self._cross_encoder = None

    def rerank(
        self,
        query: str,
        candidates: list[FusedEvidence],
        top_n: int = 5,
    ) -> tuple[list[FusedEvidence], float]:
        """Rerank candidates and measure execution latency."""
        sw = Stopwatch()

        if not self.enabled or not candidates:
            return candidates[:top_n], sw.stop()

        # 1. Neural Cross-Encoder Reranking
        if self._cross_encoder is not None:
            try:
                pairs = [(query, c.text) for c in candidates]
                scores = list(self._cross_encoder.rerank(pairs))
                for c, s in zip(candidates, scores):
                    c.rerank_score = float(s)

                candidates.sort(key=lambda x: x.rerank_score if x.rerank_score is not None else 0.0, reverse=True)
                for rank, c in enumerate(candidates, start=1):
                    c.rank = rank
                elapsed_ms = sw.stop()
                return candidates[:top_n], elapsed_ms
            except Exception:
                pass

        # 2. Lexical Term-Overlap / BM25-style Fallback Reranking
        q_tokens = set(re.findall(r"\w+", query.lower()))
        for c in candidates:
            c_tokens = re.findall(r"\w+", c.text.lower())
            total_doc = len(c_tokens) or 1
            overlap_count = sum(1 for t in c_tokens if t in q_tokens)
            # Combine term frequency with unique token coverage
            unique_coverage = len(set(c_tokens).intersection(q_tokens)) / (len(q_tokens) or 1)
            tf = overlap_count / total_doc
            lexical_score = (unique_coverage * 0.7) + (tf * 0.3)
            # Interpolate with base RRF score
            combined = (c.rrf_score * 0.4) + (lexical_score * 0.6)
            c.rerank_score = round(combined, 4)

        candidates.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        for rank, c in enumerate(candidates, start=1):
            c.rank = rank

        elapsed_ms = sw.stop()
        return candidates[:top_n], elapsed_ms
