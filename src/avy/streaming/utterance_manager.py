"""Utterance lifecycle management.

Buffers incremental transcript chunks into complete utterances.
A complete utterance is one logical user turn — the unit fed into the RAG pipeline.

KEY PRINCIPLE: chunks are NOT queries. An utterance is.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any


@dataclass
class UtteranceBuffer:
    """Mutable buffer accumulating a single user utterance from sequential chunks."""

    utterance_id: str
    session_id: str
    chunks: list[str] = field(default_factory=list)
    is_finalized: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def accumulated_text(self) -> str:
        """Return full text of all chunks joined."""
        return " ".join(c.strip() for c in self.chunks if c.strip())

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def add_chunk(self, chunk: str) -> None:
        """Append a transcript chunk to the buffer."""
        if chunk.strip():
            self.chunks.append(chunk.strip())

    def finalize(self) -> str:
        """Mark utterance as complete and return full text."""
        self.is_finalized = True
        return self.accumulated_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "utterance_id": self.utterance_id,
            "session_id": self.session_id,
            "chunk_count": self.chunk_count,
            "accumulated_text": self.accumulated_text,
            "is_finalized": self.is_finalized,
            "created_at": self.created_at,
        }


class UtteranceManager:
    """Per-session utterance buffer registry.

    Typical usage:
        1. Client sends partial transcript chunks (e.g., from STT interim results).
        2. UtteranceManager.add_chunk() accumulates them.
        3. On final transcript / silence / explicit finalization, call finalize_utterance().
        4. The returned complete text is sent to the RAG engine exactly ONCE.

    This ensures:
        - ONE utterance → ONE RAG pipeline execution → ONE answer.
        - Duplicate pipeline executions are impossible.
    """

    def __init__(self) -> None:
        self._buffers: dict[str, UtteranceBuffer] = {}
        self._lock = Lock()

    def start_utterance(self, session_id: str) -> UtteranceBuffer:
        """Begin a new utterance buffer for a session (clears any prior buffer)."""
        with self._lock:
            buf = UtteranceBuffer(
                utterance_id=f"utt_{uuid.uuid4().hex[:8]}",
                session_id=session_id,
            )
            self._buffers[session_id] = buf
            return buf

    def add_chunk(self, session_id: str, chunk: str) -> UtteranceBuffer:
        """Add a transcript chunk to the active utterance buffer, creating one if needed."""
        with self._lock:
            if session_id not in self._buffers:
                self._buffers[session_id] = UtteranceBuffer(
                    utterance_id=f"utt_{uuid.uuid4().hex[:8]}",
                    session_id=session_id,
                )
            self._buffers[session_id].add_chunk(chunk)
            return self._buffers[session_id]

    def finalize_utterance(self, session_id: str) -> str | None:
        """Finalize and return the accumulated utterance text. Returns None if already finalized."""
        with self._lock:
            buf = self._buffers.get(session_id)
            if buf and not buf.is_finalized:
                text = buf.finalize()
                return text if text.strip() else None
            return None

    def get_current(self, session_id: str) -> UtteranceBuffer | None:
        """Return the current (possibly in-progress) utterance buffer for a session."""
        return self._buffers.get(session_id)

    def clear(self, session_id: str) -> None:
        """Clear the utterance buffer for a session (e.g., on session reset)."""
        with self._lock:
            self._buffers.pop(session_id, None)

    def clear_all(self) -> None:
        """Clear all utterance buffers."""
        with self._lock:
            self._buffers.clear()
