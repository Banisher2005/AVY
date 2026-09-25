"""Deterministic MockProvider for testing, offline benchmarks, and CI."""

import json
import re
import time
from typing import Iterator

from avy.providers.base import BaseProvider
from avy.providers.models import (
    AgentRequest,
    AgentResponse,
    ProviderCapabilities,
    ResponseMetrics,
)


class MockProvider(BaseProvider):
    """Deterministic, zero-latency provider for reproducible tests."""

    def __init__(self, model_name: str = "mock-model") -> None:
        self.model_name = model_name
        self.custom_responses: dict[str, str] = {}
        self._last_metrics: ResponseMetrics | None = None

    def get_model_name(self) -> str:
        return self.model_name

    def is_available(self) -> bool:
        return True

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(streaming=True, structured_output=True, local=True)

    @property
    def last_metrics(self) -> ResponseMetrics | None:
        return self._last_metrics

    def set_response(self, prompt_keyword: str, response_text: str) -> None:
        """Register canned response for prompts containing keyword."""
        self.custom_responses[prompt_keyword.lower()] = response_text

    def _generate_text(self, request: AgentRequest) -> str:
        """Generate deterministic text matching prompt intent."""
        prompt_lower = request.prompt.lower()
        sys_lower = (request.system_prompt or "").lower()

        # Check explicit custom responses
        for k, v in self.custom_responses.items():
            if k in prompt_lower:
                return v

        # Multi-intent decomposition prompt detection
        if "decompose" in sys_lower or "queries" in sys_lower or "subqueries" in prompt_lower:
            if "battery" in prompt_lower and ("exynos" in prompt_lower or "processor" in prompt_lower):
                return json.dumps({
                    "queries": [
                        "Samsung latest processor specifications",
                        "Samsung processor battery efficiency and power management",
                    ]
                })
            elif "s24" in prompt_lower and "s23" in prompt_lower:
                return json.dumps({
                    "queries": [
                        "Samsung Galaxy S24 specifications and features",
                        "Samsung Galaxy S23 battery and hardware comparison",
                    ]
                })
            else:
                m = re.search(r'"([^"]+)"', request.prompt)
                clean_q = m.group(1).strip() if m else request.prompt.splitlines()[-1].strip()
                return json.dumps({"queries": [clean_q]})

        # Grounded answer synthesis detection
        if "evidence" in prompt_lower or "sources" in prompt_lower:
            return (
                "Based on the provided Samsung specifications, the Exynos 2400 processor features a 10-core CPU "
                "architecture and Xclipse 940 GPU [1]. The Galaxy S24 series includes advanced power management "
                "and an enhanced 4,000 mAh to 5,000 mAh battery capacity across the lineup [2]."
            )

        # Conversational / Greetings / Thanks fallback
        if any(w in prompt_lower for w in ["hello", "hi", "hey"]):
            return "Hello! How can I assist you with Samsung technologies and specifications today?"
        if any(w in prompt_lower for w in ["thank", "thanks"]):
            return "You're very welcome! Let me know if you need more details."

        return f"Mock response answering: {request.prompt[:50]}..."

    def send(self, request: AgentRequest) -> AgentResponse:
        t0 = time.perf_counter()
        text = self._generate_text(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        metrics = ResponseMetrics(
            total_duration_ms=elapsed_ms,
            ttft_ms=elapsed_ms / 2.0,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(text.split()),
        )
        self._last_metrics = metrics
        return AgentResponse(text=text, metrics=metrics)

    def stream(self, request: AgentRequest) -> Iterator[str]:
        t0 = time.perf_counter()
        text = self._generate_text(request)
        tokens = text.split(" ")
        first_token_time = None

        for idx, token in enumerate(tokens):
            if idx > 0:
                chunk = " " + token
            else:
                chunk = token
            if first_token_time is None:
                first_token_time = time.perf_counter()
            yield chunk

        total_ms = (time.perf_counter() - t0) * 1000.0
        ttft_ms = (first_token_time - t0) * 1000.0 if first_token_time else total_ms
        self._last_metrics = ResponseMetrics(
            total_duration_ms=total_ms,
            ttft_ms=ttft_ms,
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(tokens),
        )
