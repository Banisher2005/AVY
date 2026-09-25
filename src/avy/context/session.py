"""Session manager and conversation refinement for AVY."""

import re
import uuid
from typing import Any

from avy.context.models import ConversationTurn, SessionContext

_FOLLOWUP_INDICATORS = re.compile(
    r"\b(it|its|their|they|this|that|these|those|the\s+previous|earlier|now\s+compare|what\s+about|how\s+about|and\s+its|and\s+the)\b",
    re.IGNORECASE,
)


class SessionManager:
    """Manages active conversational sessions and query refinement."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionContext] = {}

    def get_or_create(self, session_id: str | None = None) -> SessionContext:
        """Get an existing session by ID or create a fresh one."""
        sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        if sid not in self._sessions:
            self._sessions[sid] = SessionContext(session_id=sid)
        return self._sessions[sid]

    def add_turn(
        self,
        session_id: str,
        user_utterance: str,
        controller_state: str,
        decomposed_queries: list[str] | None = None,
        retrieved_evidence_ids: list[str] | None = None,
        assistant_response: str = "",
        citations: list[dict[str, Any]] | None = None,
    ) -> ConversationTurn:
        """Append a completed turn to the session."""
        session = self.get_or_create(session_id)
        turn_num = len(session.turns) + 1
        turn = ConversationTurn(
            turn_id=turn_num,
            user_utterance=user_utterance,
            controller_state=controller_state,
            decomposed_queries=decomposed_queries or [],
            retrieved_evidence_ids=retrieved_evidence_ids or [],
            assistant_response=assistant_response,
            citations=citations or [],
        )
        session.turns.append(turn)
        session.active_intent = user_utterance
        return turn

    def is_followup_refinement(self, session: SessionContext, current_utterance: str) -> bool:
        """Check if current utterance is an anaphoric or comparative follow-up to prior turn."""
        if not session.turns:
            return False

        # Pattern match pronoun or comparative indicators
        if _FOLLOWUP_INDICATORS.search(current_utterance):
            return True

        # Very short queries (< 4 words) often act as late qualifiers or refinements
        words = current_utterance.strip().split()
        if len(words) <= 3 and any(w.lower() in ("battery", "screen", "camera", "price", "ram") for w in words):
            return True

        return False

    def contextualize_query(self, session: SessionContext, current_utterance: str) -> str:
        """Synthesize a stand-alone search query incorporating past session context if needed."""
        if not self.is_followup_refinement(session, current_utterance):
            return current_utterance

        last_turn = session.last_turn
        if not last_turn:
            return current_utterance

        # Contextual synthesis combining last intent with current refinement
        prior_intent = last_turn.user_utterance.strip().rstrip("?.!")
        refinement = current_utterance.strip()
        return f"{prior_intent} {refinement}"

    def update_evidence_pool(self, session_id: str, new_evidence_items: list[Any]) -> None:
        """Cache evidence items in session pool for subsequent multi-turn fusion."""
        session = self.get_or_create(session_id)
        for item in new_evidence_items:
            chunk_id = getattr(item, "chunk_id", None) or getattr(item, "id", None)
            if chunk_id:
                session.fused_evidence_pool[chunk_id] = item

    def get_evidence_pool(self, session_id: str) -> list[Any]:
        """Retrieve existing evidence pool for this session."""
        session = self.get_or_create(session_id)
        return list(session.fused_evidence_pool.values())

    def clear(self, session_id: str) -> None:
        """Reset session memory."""
        self._sessions.pop(session_id, None)
