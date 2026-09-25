"""Three-state Retrieval Controller for Streaming Live RAG.

Determines whether to WAIT, RETRIEVE, or skip retrieval (NO_RETRIEVE)
based on semantic completeness, information need, and session state.
"""

import re

from avy.config import AVYConfig
from avy.context.models import SessionContext
from avy.core.models import ControllerDecision, ControllerState

# Conversational non-retrieval patterns
_NO_RETRIEVE_PATTERNS = [
    re.compile(r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy)\b", re.IGNORECASE),
    re.compile(r"^(thanks|thank\s+you|that\'?s\s+(helpful|great|awesome)|got\s+it|ok\s+thanks)\b", re.IGNORECASE),
    re.compile(r"^(bye|goodbye|see\s+you|have\s+a\s+good\s+day)\b", re.IGNORECASE),
    re.compile(r"^(who\s+are\s+you|what\s+is\s+your\s+name|what\s+can\s+you\s+do)\??$", re.IGNORECASE),
]

# Incomplete trailing prepositions/conjunctions indicating user is in mid-sentence
_DANGLING_ENDINGS = re.compile(
    r"\b(about|the|a|an|of|for|with|and|or|in|to|at|on|from|by|is|are|was|were|its|their|whose|which|that|than)\s*$",
    re.IGNORECASE,
)

# Incomplete question starters without predicates
_INCOMPLETE_STARTERS = [
    re.compile(r"^(tell\s+me(\s+about)?|what(\s+is|\'s)?|how(\s+about|\s+does)?|can\s+you(\s+tell\s+me|\s+explain)?|show\s+me|compare(\s+the)?)\s*$", re.IGNORECASE),
    re.compile(r"^(tell\s+me\s+about\s+samsung\'?s?)\s*$", re.IGNORECASE),
    re.compile(r"^(what\s+is\s+the|how\s+is\s+the|explain\s+the)\s*$", re.IGNORECASE),
]

# Keywords indicating substantive domain information need
_DOMAIN_TOPIC_KEYWORDS = {
    "samsung", "processor", "exynos", "snapdragon", "galaxy", "battery", "display",
    "semiconductor", "gaa", "finfet", "camera", "npu", "gpu", "cpu", "chip",
    "ultrathin", "hinge", "fold", "flip", "s24", "s23", "s22", "ram", "power",
    "efficiency", "specifications", "features", "charging", "watt", "mah", "nm",
}


class RetrievalController:
    """Evaluates incoming streaming transcript chunks to govern retrieval triggers."""

    def __init__(self, config: AVYConfig | None = None) -> None:
        self.config = config or AVYConfig.load()
        self.min_tokens = self.config.wait_token_min

    def evaluate(
        self,
        transcript_chunk: str,
        accumulated_transcript: str | None = None,
        session_context: SessionContext | None = None,
        has_already_retrieved: bool = False,
    ) -> ControllerDecision:
        """Evaluate current transcript state and return ControllerDecision."""
        text = (accumulated_transcript or transcript_chunk).strip()

        # 1. Check for empty or whitespace-only transcript
        if not text:
            return ControllerDecision(
                state=ControllerState.WAIT,
                reason="Empty transcript chunk; awaiting audio input",
                confidence=1.0,
                accumulated_transcript=text,
            )

        # 2. Check for Conversational / Non-Retrieval Requests (NO_RETRIEVE)
        clean_text = text.rstrip(".!?").strip()
        for pat in _NO_RETRIEVE_PATTERNS:
            if pat.search(clean_text):
                return ControllerDecision(
                    state=ControllerState.NO_RETRIEVE,
                    reason=f"Conversational utterance ('{clean_text}') does not require corpus retrieval",
                    confidence=0.98,
                    accumulated_transcript=text,
                )

        # 3. Check for Incomplete Query / Trailing Dangling Tokens (WAIT)
        tokens = text.split()

        # Trailing ellipsis explicitly signifies streaming pause
        if text.endswith("...") or text.endswith(".."):
            return ControllerDecision(
                state=ControllerState.WAIT,
                reason="Transcript ends with continuation ellipsis",
                confidence=0.95,
                accumulated_transcript=text,
            )

        # Check incomplete phrase starters
        for starter_pat in _INCOMPLETE_STARTERS:
            if starter_pat.match(clean_text):
                return ControllerDecision(
                    state=ControllerState.WAIT,
                    reason=f"Transcript starter '{clean_text}' lacks predicate and semantic object",
                    confidence=0.94,
                    accumulated_transcript=text,
                )

        # Dangling preposition or conjunction at the end
        if _DANGLING_ENDINGS.search(text):
            return ControllerDecision(
                state=ControllerState.WAIT,
                reason=f"Transcript ends with dangling connective/preposition ('{tokens[-1]}')",
                confidence=0.92,
                accumulated_transcript=text,
            )

        # Token count threshold check: If too short (< min_tokens) and no domain keyword
        has_domain_kw = any(t.lower().strip(",.?!") in _DOMAIN_TOPIC_KEYWORDS for t in tokens)
        if len(tokens) < self.min_tokens and not has_domain_kw:
            return ControllerDecision(
                state=ControllerState.WAIT,
                reason=f"Transcript too brief ({len(tokens)} tokens) and lacks domain anchor",
                confidence=0.88,
                accumulated_transcript=text,
            )

        # 4. Check for Session Refinement / Late Details
        if session_context and session_context.turns and not has_already_retrieved:
            # If follow-up or late qualification in active session
            return ControllerDecision(
                state=ControllerState.RETRIEVE,
                reason="Session follow-up contains actionable refinement for retrieval",
                confidence=0.95,
                accumulated_transcript=text,
                is_early_retrieval=False,
                metadata={"refinement": True},
            )

        # 5. Early Retrieval Detection:
        # If the transcript already contains a self-contained informational subject
        # (e.g. "Tell me about Samsung's latest processor"), we trigger early retrieval
        is_early = False
        if self.config.early_retrieval_enabled and len(tokens) >= 4 and has_domain_kw:
            # Even if the user continues speaking, we can initiate vector search early
            is_early = True

        return ControllerDecision(
            state=ControllerState.RETRIEVE,
            reason="Transcript possesses sufficient semantic entity and informational predicate",
            confidence=0.96,
            accumulated_transcript=text,
            is_early_retrieval=is_early,
        )
