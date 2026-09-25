"""FAISS Vector Store and chunk metadata index for AVY."""

import json
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore

from avy.core.exceptions import RetrievalError
from avy.retrieval.models import Chunk


class FAISSVectorStore:
    """Vector database backed by FAISS IndexFlatIP (Cosine Similarity)."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._chunks: dict[int, Chunk] = {}
        self._vectors: list[np.ndarray] = []  # Backup in-memory vector store

        if faiss is not None:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = None

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks and corresponding float32 embeddings into the index."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")

        start_id = len(self._chunks)
        for i, chunk in enumerate(chunks):
            self._chunks[start_id + i] = chunk
            self._vectors.append(embeddings[i])

        if self.index is not None:
            # Ensure float32 contiguous array
            matrix = np.ascontiguousarray(embeddings, dtype=np.float32)
            self.index.add(matrix)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        """Search top-K most similar chunks by cosine similarity."""
        if not self._chunks:
            return []

        q_vec = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)

        if self.index is not None and self.index.ntotal > 0:
            k = min(top_k, self.index.ntotal)
            scores, indices = self.index.search(q_vec, k)
            results: list[tuple[Chunk, float]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                if float(score) >= threshold:
                    chunk = self._chunks.get(int(idx))
                    if chunk:
                        results.append((chunk, float(score)))
            return results

        # Fallback pure NumPy cosine search
        if self._vectors:
            matrix = np.vstack(self._vectors)
            scores = np.dot(matrix, query_vector.flatten())
            sorted_indices = np.argsort(scores)[::-1][:top_k]
            results = []
            for idx in sorted_indices:
                score = float(scores[idx])
                if score >= threshold:
                    chunk = self._chunks.get(int(idx))
                    if chunk:
                        results.append((chunk, score))
            return results

        return []

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        """Serialize FAISS index and metadata json to disk."""
        ipath = Path(index_path)
        mpath = Path(metadata_path)
        ipath.parent.mkdir(parents=True, exist_ok=True)
        mpath.parent.mkdir(parents=True, exist_ok=True)

        if self.index is not None and faiss is not None:
            faiss.write_index(self.index, str(ipath))

        # Serialize metadata
        meta = {
            "dimension": self.dimension,
            "count": len(self._chunks),
            "chunks": {str(k): v.to_dict() for k, v in self._chunks.items()},
        }
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, index_path: str | Path, metadata_path: str | Path) -> "FAISSVectorStore":
        """Deserialize FAISS index and metadata from disk."""
        ipath = Path(index_path)
        mpath = Path(metadata_path)

        if not mpath.is_file():
            raise RetrievalError(f"Vector store metadata file not found at {mpath}")

        with open(mpath, "r", encoding="utf-8") as f:
            meta = json.load(f)

        dimension = meta.get("dimension", 256)
        store = cls(dimension=dimension)

        if ipath.is_file() and faiss is not None:
            try:
                store.index = faiss.read_index(str(ipath))
            except Exception:
                pass

        chunks_data = meta.get("chunks", {})
        for k_str, cdict in chunks_data.items():
            k = int(k_str)
            chunk = Chunk(
                chunk_id=cdict["chunk_id"],
                doc_id=cdict["doc_id"],
                source=cdict["source"],
                title=cdict["title"],
                chunk_index=cdict["chunk_index"],
                text=cdict["text"],
                metadata=cdict.get("metadata", {}),
            )
            store._chunks[k] = chunk

        return store
