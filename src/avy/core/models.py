"""Core data models and state representations for AVY."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ControllerState(str, Enum):
    """The three definitive states of the AVY Retrieval Controller."""

    WAIT = "WAIT"
    RETRIEVE = "RETRIEVE"
    NO_RETRIEVE = "NO_RETRIEVE"


@dataclass
class ControllerDecision:
    """Decision output produced by the Retrieval Controller."""

    state: ControllerState
    reason: str
    confidence: float = 1.0
    accumulated_transcript: str = ""
    is_early_retrieval: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert decision to dictionary representation."""
        return {
            "state": self.state.value,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "accumulated_transcript": self.accumulated_transcript,
            "is_early_retrieval": self.is_early_retrieval,
            "metadata": self.metadata,
        }


@dataclass
class PipelineTimings:
    """End-to-end performance and latency measurements."""

    ttft_ms: float = 0.0
    refinement_ms: float = 0.0
    decomposition_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    synthesis_ms: float = 0.0
    total_ms: float = 0.0
    early_lead_time_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert timings to rounded ms dictionary."""
        return {
            "ttft_ms": round(self.ttft_ms, 2),
            "refinement_ms": round(self.refinement_ms, 2),
            "decomposition_ms": round(self.decomposition_ms, 2),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "reranking_ms": round(self.reranking_ms, 2),
            "synthesis_ms": round(self.synthesis_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "early_lead_time_ms": round(self.early_lead_time_ms, 2),
        }
