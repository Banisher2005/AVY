"""Multi-Intent Query Decomposition for complex spoken utterances."""

import json
import re
import uuid

from avy.core.timing import Stopwatch
from avy.providers.base import BaseProvider
from avy.providers.models import AgentRequest
from avy.retrieval.models import QueryDecomposition, Subquery

_DECOMPOSITION_SYSTEM_PROMPT = (
    "You are a specialized search query decomposer for a conversational retrieval system. "
    "Given a spoken user utterance, analyze whether it contains one or multiple search intents. "
    "Decompose compound questions into 1 to 3 distinct, concise, standalone search-ready queries. "
    "Do NOT answer the question. Return ONLY a valid JSON object in this exact schema:\n"
    "{\"queries\": [\"independent search query 1\", \"independent search query 2\"]}"
)

_CONJUNCTION_SPLITTERS = re.compile(
    r"\b(?:and\s+compare|and\s+explain|and\s+also|and\s+tell|and\s+describe|and\s+what|and\s+how|as\s+well\s+as|along\s+with|also\s+explain|plus|while\s+also|additionally)\b",
    re.IGNORECASE,
)


class MultiIntentDecomposer:
    """Decomposes compound or multi-intent utterances into standalone retrieval queries."""

    def __init__(self, provider: BaseProvider) -> None:
        self.provider = provider

    def decompose(self, utterance: str) -> QueryDecomposition:
        """Decompose utterance into structured search subqueries."""
        sw = Stopwatch()
        clean_text = utterance.strip()

        # Selective decomposition: single intent queries skip provider dispatch
        is_potentially_compound = bool(
            _CONJUNCTION_SPLITTERS.search(clean_text)
            or "?" in clean_text[:-1]
            or " vs " in clean_text.lower()
            or " versus " in clean_text.lower()
        )

        if not is_potentially_compound:
            subquery_strings = [clean_text]
        else:
            # Dispatch prompt to provider with tight token limit
            req = AgentRequest(
                prompt=f"Decompose the following user query into search queries:\n\"{clean_text}\"",
                system_prompt=_DECOMPOSITION_SYSTEM_PROMPT,
                temperature=0.0,
                stream=False,
                extra_options={"num_predict": 80},
            )
            try:
                resp = self.provider.send(req)
                subquery_strings = self._parse_provider_output(resp.text)
                if not subquery_strings:
                    subquery_strings = self._rule_based_fallback(clean_text)
            except Exception:
                subquery_strings = self._rule_based_fallback(clean_text)

        # Post-process, clean, and deduplicate queries
        valid_queries = self._clean_and_deduplicate(subquery_strings, fallback=clean_text)

        elapsed_ms = sw.stop()

        subqueries = [
            Subquery(
                query_id=f"sq_{uuid.uuid4().hex[:8]}",
                subquery_text=q,
                intent_label=f"intent_{idx + 1}",
            )
            for idx, q in enumerate(valid_queries)
        ]

        return QueryDecomposition(
            original_utterance=clean_text,
            subqueries=subqueries,
            is_compound=len(subqueries) > 1,
            duration_ms=elapsed_ms,
        )

    def _parse_provider_output(self, raw_text: str) -> list[str]:
        """Extract and parse query array from LLM response."""
        text = raw_text.strip()

        # 1. Remove markdown code fences if wrapped
        if "```" in text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                text = match.group(1).strip()

        # 2. Try JSON parsing
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "queries" in data and isinstance(data["queries"], list):
                return [str(q).strip() for q in data["queries"] if str(q).strip()]
            if isinstance(data, list):
                return [str(q).strip() for q in data if str(q).strip()]
        except json.JSONDecodeError:
            pass

        # 3. Fallback: Parse bulleted or numbered lines
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        extracted = []
        for line in lines:
            line_clean = re.sub(r"^[-*•\d\.\)]+\s*", "", line).strip().strip('"\'')
            if len(line_clean) > 5 and not line_clean.startswith("{") and not line_clean.startswith("}"):
                extracted.append(line_clean)

        if extracted:
            return extracted

        return []

    def _rule_based_fallback(self, text: str) -> list[str]:
        """Syntactic heuristic splitter when model parsing fails."""
        clauses = _CONJUNCTION_SPLITTERS.split(text)
        if len(clauses) > 1:
            return [c.strip() for c in clauses if c.strip()]
        return [text]

    def _clean_and_deduplicate(self, queries: list[str], fallback: str) -> list[str]:
        """Eliminate redundant queries, remove punctuation, filter filler phrases."""
        if not queries:
            queries = [fallback]

        seen = set()
        cleaned: list[str] = []

        # Filler prefix cleaner
        prefix_pattern = re.compile(
            r"^(tell\s+me(\s+about)?|what(\s+is|\'s)?|can\s+you(\s+explain)?|show\s+me|compare)\s+",
            re.IGNORECASE,
        )

        for q in queries:
            c = prefix_pattern.sub("", q).strip().rstrip("?.!")
            if not c:
                c = q.strip().rstrip("?.!")

            c_lower = c.lower()
            if c_lower not in seen and len(c) >= 3:
                seen.add(c_lower)
                cleaned.append(c)

        return cleaned or [fallback]
