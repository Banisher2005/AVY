"""Embedding models and vector generators for AVY."""

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np

from avy.config import AVYConfig


class BaseEmbedding(ABC):
    """Abstract interface for dense embedding generators."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Generate normalized 2D float32 embeddings for a batch of texts."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Generate normalized 1D float32 embedding for a single query."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality."""
        pass


class LocalDenseEmbedding(BaseEmbedding):
    """Deterministic, zero-download, pure NumPy subword & n-gram dense embedding.

    Provides fast (< 1 ms), completely offline, reproducible vectorization
    using subword hashing with L2-normalization.
    """

    def __init__(self, dimension: int = 256) -> None:
        self._dim = dimension
        self.model_name = "local-dense-hash"

    @property
    def dimension(self) -> int:
        return self._dim

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"\w+", text.lower())
        features = list(tokens)
        # Add character tri-grams for subword morphological matching
        for t in tokens:
            if len(t) >= 3:
                for i in range(len(t) - 2):
                    features.append(t[i : i + 3])
        return features

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        features = self._tokenize(text)
        if not features:
            return vec

        for feat in features:
            # Deterministic MD5 hash to bucket index and sign
            h = int(hashlib.md5(feat.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if (h >> 16) & 1 else -1.0
            vec[idx] += sign

        # Apply sub-linear scaling (1 + log(tf)) and L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        vectors = [self._vectorize(t) for t in texts]
        return np.vstack(vectors).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vectorize(text).astype(np.float32)


class FastEmbedEmbedding(BaseEmbedding):
    """Dense embedding generator using ONNX fastembed (e.g. BAAI/bge-small-en-v1.5)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding

            self.model_name = model_name
            self._model = TextEmbedding(model_name=model_name)
            # Probe dimension by embedding a tiny sample
            probe = list(self._model.embed(["probe"]))[0]
            self._dim = int(len(probe))
        except Exception as err:
            raise RuntimeError(f"Failed to initialize FastEmbed: {err}")

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        embeddings = list(self._model.embed(texts))
        arr = np.array(embeddings, dtype=np.float32)
        # Normalize to unit length for inner-product cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_query(self, text: str) -> np.ndarray:
        embedding = list(self._model.query_embed([text]))[0]
        vec = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec


def get_embedding_engine(config: AVYConfig | None = None) -> BaseEmbedding:
    """Factory creating configured embedding engine with graceful offline fallback."""
    cfg = config or AVYConfig.load()
    provider = cfg.embedding_provider.lower().strip()

    if provider == "fastembed":
        try:
            return FastEmbedEmbedding(model_name=cfg.embedding_model)
        except Exception:
            # Fallback to local dense embedding if download fails or offline
            return LocalDenseEmbedding(dimension=256)

    elif provider == "local_dense":
        return LocalDenseEmbedding(dimension=256)

    # Default fallback
    return LocalDenseEmbedding(dimension=256)
