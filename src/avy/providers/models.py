"""Provider models and request/response dataclasses for AVY."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderCapabilities:
    """Capabilities supported by a model provider backend."""

    streaming: bool = True
    structured_output: bool = True
    context_size: int = 8192
    local: bool = True


@dataclass
class ResponseMetrics:
    """Performance telemetry for a provider inference call."""

    total_duration_ms: float = 0.0
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class AgentRequest:
    """Normalized request dispatched to an AI model provider."""

    prompt: str
    system_prompt: str | None = None
    temperature: float = 0.1
    stream: bool = True
    extra_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Normalized response returned from an AI model provider."""

    text: str
    metrics: ResponseMetrics | None = None
    raw: Any | None = None
