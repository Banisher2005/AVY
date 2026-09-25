"""Core exception hierarchy for AVY."""


class AVYError(Exception):
    """Base exception for all AVY errors."""

    pass


class ConfigurationError(AVYError):
    """Raised when configuration is invalid or missing."""

    pass


class ControllerError(AVYError):
    """Raised when the Retrieval Controller encounters an unrecoverable state."""

    pass


class DecompositionError(AVYError):
    """Raised when query decomposition fails to produce valid search queries."""

    pass


class RetrievalError(AVYError):
    """Raised when vector search or corpus retrieval fails."""

    pass


class FusionError(AVYError):
    """Raised when reciprocal rank fusion or evidence deduplication fails."""

    pass


class GroundingError(AVYError):
    """Raised when grounded answer synthesis or citation mapping fails."""

    pass


class ProviderError(AVYError):
    """Base exception for model provider errors."""

    pass


class ProviderConnectionError(ProviderError):
    """Raised when provider backend is unreachable or connection is refused."""

    pass


class ProviderTimeoutError(ProviderError):
    """Raised when an inference call times out."""

    pass
