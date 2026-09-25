"""Grounding prompt assembly, citation mapping, and factual verification."""

import re

from avy.context.models import SessionContext
from avy.retrieval.models import Citation, FusedEvidence

_CITATION_REGEX = re.compile(r"\[(\d+)\]")

_STRICT_GROUNDING_SYSTEM_PROMPT = (
    "You are AVY, an accurate, grounded live conversational RAG assistant. "
    "Your answers MUST be strictly synthesized from the provided numbered evidence blocks. "
    "Do NOT fabricate facts, extrapolate unverifiable claims, or invent citations. "
    "When stating factual details from an evidence source, cite the source number using [1], [2], etc. "
    "If the provided evidence does not contain sufficient information to answer the question, "
    "clearly state that the information is not available in the corpus."
)


class GroundingEngine:
    """Constructs grounded synthesis prompts and traces citations back to source chunks."""

    def build_synthesis_prompt(
        self,
        utterance: str,
        evidence: list[FusedEvidence],
        session_context: SessionContext | None = None,
    ) -> tuple[str, str]:
        """Assemble structured system and user prompts with evidence blocks and session context."""
        evidence_blocks = []
        for idx, ev in enumerate(evidence, start=1):
            block = (
                f"[{idx}] Source: {ev.source} | Document: {ev.document_id} | Chunk: {ev.chunk_id}\n"
                f"Title: {ev.title}\n"
                f"Content: {ev.text}"
            )
            evidence_blocks.append(block)

        evidence_section = "\n\n".join(evidence_blocks) if evidence_blocks else "NO RETRIEVED EVIDENCE."

        session_section = ""
        if session_context and session_context.turns:
            history_summary = session_context.get_recent_history_summary(max_turns=2)
            if history_summary:
                session_section = f"Conversation History:\n{history_summary}\n\n"

        user_prompt = (
            f"{session_section}"
            f"Retrieved Evidence:\n"
            f"{evidence_section}\n\n"
            f"User Question: {utterance}\n\n"
            f"Provide a concise, grounded response citing evidence using [1], [2] where appropriate."
        )

        return _STRICT_GROUNDING_SYSTEM_PROMPT, user_prompt

    def extract_citations(
        self,
        generated_text: str,
        evidence: list[FusedEvidence],
    ) -> list[Citation]:
        """Parse [1], [2] markers from generated text and map to exact evidence metadata."""
        found_indices = sorted(
            list({int(m.group(1)) for m in _CITATION_REGEX.finditer(generated_text)})
        )

        citations: list[Citation] = []
        for idx in found_indices:
            if 1 <= idx <= len(evidence):
                ev = evidence[idx - 1]
                snippet = ev.text[:220].strip()
                if len(ev.text) > 220:
                    snippet += "..."

                citations.append(
                    Citation(
                        index=idx,
                        document_id=ev.document_id,
                        source=ev.source,
                        title=ev.title,
                        chunk_id=ev.chunk_id,
                        snippet=snippet,
                    )
                )

        return citations

    def calculate_groundedness_score(
        self,
        generated_text: str,
        evidence: list[FusedEvidence],
    ) -> float:
        """Evaluate overlap ratio of content tokens between answer and cited evidence."""
        if not evidence or not generated_text:
            return 0.0

        answer_tokens = set(re.findall(r"\w{4,}", generated_text.lower()))
        if not answer_tokens:
            return 1.0

        evidence_text = " ".join(e.text for e in evidence).lower()
        evidence_tokens = set(re.findall(r"\w{4,}", evidence_text))

        supported_tokens = answer_tokens.intersection(evidence_tokens)
        score = len(supported_tokens) / len(answer_tokens)
        return min(1.0, max(0.0, score))
