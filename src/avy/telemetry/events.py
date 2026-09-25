"""Telemetry event schemas and standard event constants."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class TelemetryEventNames:
    """Standardized event names required for Theme 04 telemetry."""

    INPUT_CHUNK = "input.chunk"
    RETRIEVAL_WAIT = "retrieval.wait"
    RETRIEVAL_TRIGGERED = "retrieval.triggered"
    RETRIEVAL_SKIPPED = "retrieval.skipped"
    QUERY_DECOMPOSED = "query.decomposed"
    RETRIEVAL_STARTED = "retrieval.started"
    RETRIEVAL_COMPLETED = "retrieval.completed"
    FUSION_COMPLETED = "fusion.completed"
    RERANK_COMPLETED = "rerank.completed"
    SYNTHESIS_STARTED = "synthesis.started"
    SYNTHESIS_FIRST_TOKEN = "synthesis.first_token"
    SYNTHESIS_COMPLETED = "synthesis.completed"
    CITATION_GENERATED = "citation.generated"
    PIPELINE_ERROR = "pipeline.error"


@dataclass
class TelemetryEvent:
    """Structured telemetry event record."""

    event: str
    session_id: str
    utterance_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "utterance_id": self.utterance_id,
            "metadata": self.metadata,
        }
