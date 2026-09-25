"""Context and session models for AVY."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ConversationTurn:
    """Represents a single conversational interaction in a session."""

    turn_id: int
    user_utterance: str
    controller_state: str
    decomposed_queries: list[str] = field(default_factory=list)
    retrieved_evidence_ids: list[str] = field(default_factory=list)
    assistant_response: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_utterance": self.user_utterance,
            "controller_state": self.controller_state,
            "decomposed_queries": self.decomposed_queries,
            "retrieved_evidence_ids": self.retrieved_evidence_ids,
            "assistant_response": self.assistant_response,
            "citations": self.citations,
            "timestamp": self.timestamp,
        }


@dataclass
class SessionContext:
    """Session-scoped context container tracking conversational state and evidence."""

    session_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    active_intent: str = ""
    fused_evidence_pool: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def last_turn(self) -> ConversationTurn | None:
        return self.turns[-1] if self.turns else None

    def get_recent_history_summary(self, max_turns: int = 3) -> str:
        """Format concise conversation history for model context."""
        recent = self.turns[-max_turns:]
        if not recent:
            return ""
        lines = []
        for t in recent:
            lines.append(f"User: {t.user_utterance}")
            if t.assistant_response:
                # Truncate to keep context concise
                resp_preview = t.assistant_response[:300].strip()
                if len(t.assistant_response) > 300:
                    resp_preview += "..."
                lines.append(f"Assistant: {resp_preview}")
        return "\n".join(lines)
