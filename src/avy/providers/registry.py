"""Provider factory registry for AVY."""

from typing import Any, Callable

from avy.config import AVYConfig
from avy.core.exceptions import ProviderError
from avy.providers.base import BaseProvider

ProviderFactory = Callable[[AVYConfig, dict[str, Any]], BaseProvider]


class ProviderRegistry:
    """Registry maintaining AI model provider factories."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., BaseProvider]] = {}

    def register(self, name: str, factory: Callable[..., BaseProvider]) -> None:
        """Register a provider factory under a given key."""
        self._factories[name.lower().strip()] = factory

    def get(self, name: str, config: AVYConfig | None = None, **kwargs: Any) -> BaseProvider:
        """Retrieve and instantiate provider by name."""
        canonical_name = name.lower().strip()
        factory = self._factories.get(canonical_name)
        if factory is None:
            available = ", ".join(sorted(self._factories.keys()))
            raise ProviderError(f"Unsupported provider '{name}'. Available: {available}")
        cfg = config or AVYConfig.load()
        return factory(cfg, **kwargs)

    def list_providers(self) -> list[str]:
        return sorted(self._factories.keys())


def _create_ollama_provider(config: AVYConfig, **kwargs: Any) -> BaseProvider:
    from avy.providers.ollama import OllamaProvider

    return OllamaProvider(
        host=kwargs.get("host", config.ollama_host),
        model=kwargs.get("model", config.model),
        timeout=kwargs.get("timeout", config.timeout),
        temperature=kwargs.get("temperature", config.temperature),
    )


def _create_openai_provider(config: AVYConfig, **kwargs: Any) -> BaseProvider:
    from avy.providers.openai_compatible import OpenAICompatibleProvider

    return OpenAICompatibleProvider(
        base_url=kwargs.get("base_url", config.base_url or "https://api.openai.com/v1"),
        api_key=kwargs.get("api_key", config.api_key),
        model=kwargs.get("model", config.model),
        timeout=kwargs.get("timeout", config.timeout),
        temperature=kwargs.get("temperature", config.temperature),
    )


def _create_mock_provider(config: AVYConfig, **kwargs: Any) -> BaseProvider:
    from avy.providers.mock import MockProvider

    return MockProvider(model_name=kwargs.get("model", "mock-model"))


_DEFAULT_REGISTRY: ProviderRegistry | None = None


def get_default_registry() -> ProviderRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        registry = ProviderRegistry()
        registry.register("ollama", _create_ollama_provider)
        registry.register("local", _create_ollama_provider)
        registry.register("openai", _create_openai_provider)
        registry.register("mock", _create_mock_provider)
        _DEFAULT_REGISTRY = registry
    return _DEFAULT_REGISTRY


def get_provider(name: str, config: AVYConfig | None = None, **kwargs: Any) -> BaseProvider:
    return get_default_registry().get(name, config=config, **kwargs)
