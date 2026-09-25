"""Protocol-agnostic Live RAG Gateway for AVY."""

from typing import Any, Iterator

from avy.config import AVYConfig
from avy.streaming.engine import StreamingLiveRAGEngine


class LiveRAGGateway:
    """Gateway interface for streaming live RAG operations."""

    def __init__(
        self,
        engine: StreamingLiveRAGEngine | None = None,
        config: AVYConfig | None = None,
    ) -> None:
        self.config = config or AVYConfig.load()
        self.engine = engine or StreamingLiveRAGEngine(config=self.config)

    def stream_query(
        self,
        text: str,
        session_id: str | None = None,
        accumulated_text: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Process incoming query and yield streaming pipeline events."""
        yield from self.engine.process_transcript_stream(
            transcript_chunk=text,
            session_id=session_id,
            accumulated_transcript=accumulated_text,
            is_final_chunk=True,
        )

    def execute_query(
        self,
        text: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Synchronously execute complete RAG pipeline and aggregate result."""
        accumulated_text = []
        controller_decision = None
        decomposition = None
        evidence = []
        citations = []
        timings = None

        for event in self.stream_query(text=text, session_id=session_id):
            etype = event.get("type")
            if etype == "controller_decision":
                controller_decision = event
            elif etype == "decomposition":
                decomposition = event.get("decomposition")
            elif etype == "evidence":
                evidence = event.get("evidence", [])
            elif etype == "token":
                accumulated_text.append(event.get("token", ""))
            elif etype == "citations":
                citations = event.get("citations", [])
            elif etype == "timings":
                timings = event.get("timings")

        return {
            "query": text,
            "session_id": session_id,
            "controller_decision": controller_decision,
            "decomposition": decomposition,
            "evidence": evidence,
            "answer": "".join(accumulated_text),
            "citations": citations,
            "timings": timings,
        }

    def simulate_streaming_chunks(
        self,
        chunks: list[str],
        session_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Simulate incremental speech-to-text audio chunks arriving sequentially."""
        accumulated = ""
        for i, chunk in enumerate(chunks):
            accumulated = f"{accumulated} {chunk}".strip()
            is_final = (i == len(chunks) - 1)
            for event in self.engine.process_transcript_stream(
                transcript_chunk=chunk,
                session_id=session_id,
                accumulated_transcript=accumulated,
                is_final_chunk=is_final,
            ):
                yield event

    def get_health(self) -> dict[str, Any]:
        """Aggregate health status across model provider and vector store."""
        provider_ok = self.engine.provider.is_available()
        return {
            "status": "healthy" if provider_ok else "degraded",
            "provider": self.engine.provider.get_model_name(),
            "provider_available": provider_ok,
            "vectorstore_indexed_chunks": len(self.engine.vectorstore._chunks),
            "embedding_dimension": self.engine.embeddings.dimension,
            "reranker_enabled": self.engine.config.reranker_enabled,
        }
