"""Evaluation metrics for Streaming Live RAG benchmark."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TestResult:
    """Outcome for a single evaluation test case."""

    test_id: str
    scenario_name: str
    decision_correct: bool
    actual_state: str
    expected_state: str
    recall: float
    groundedness: float
    citations_count: int
    ttft_ms: float
    total_latency_ms: float
    estimated_cost_usd: float = 0.0
    decomposed_queries: list[str] = field(default_factory=list)
    retrieved_docs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "scenario": self.scenario_name,
            "decision_correct": self.decision_correct,
            "actual_state": self.actual_state,
            "expected_state": self.expected_state,
            "recall": round(self.recall, 4),
            "groundedness": round(self.groundedness, 4),
            "citations_count": self.citations_count,
            "ttft_ms": round(self.ttft_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "cost_usd": self.estimated_cost_usd,
            "subqueries": self.decomposed_queries,
            "retrieved_docs": self.retrieved_docs,
        }


def calculate_retrieval_recall(retrieved_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    """Calculate recall ratio of expected documents retrieved in candidate set."""
    if not expected_doc_ids:
        return 1.0  # Not a retrieval task (e.g. WAIT or NO_RETRIEVE)

    ret_set = set(retrieved_doc_ids)
    exp_set = set(expected_doc_ids)
    hits = exp_set.intersection(ret_set)
    return len(hits) / len(exp_set)


def estimate_turn_cost(prompt_text: str, answer_text: str, is_local: bool = True) -> float:
    """Estimate cost per turn in USD based on token counts."""
    if is_local:
        return 0.0  # 100% free local inference on Ollama

    # Approximate 4 chars per token for typical models ($0.15/1M input, $0.60/1M output)
    prompt_tokens = len(prompt_text) / 4.0
    completion_tokens = len(answer_text) / 4.0
    cost = (prompt_tokens * 0.00000015) + (completion_tokens * 0.00000060)
    return round(cost, 6)
