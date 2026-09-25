"""Base provider abstract class for AVY."""

from abc import ABC, abstractmethod
from typing import Iterator

from avy.providers.models import (
    AgentRequest,
    AgentResponse,
    ProviderCapabilities,
    ResponseMetrics,
)


class BaseProvider(ABC):
    """Abstract base class for all AVY AI model providers."""

    @abstractmethod
    def send(self, request: AgentRequest) -> AgentResponse:
        """Send a synchronous inference request and return the complete text."""
        pass

    @abstractmethod
    def stream(self, request: AgentRequest) -> Iterator[str]:
        """Send a streaming inference request and yield token chunks."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider backend is reachable and ready."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the active model name."""
        pass

    def capabilities(self) -> ProviderCapabilities:
        """Return backend capability flags."""
        return ProviderCapabilities()

    @property
    def last_metrics(self) -> ResponseMetrics | None:
        """Return telemetry metrics from the latest request."""
        return None
