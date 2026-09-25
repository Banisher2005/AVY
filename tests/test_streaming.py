"""Tests for Streaming Live RAG Engine and Latency Measurements."""

from avy.core.models import ControllerState
from avy.streaming.engine import StreamingLiveRAGEngine


def test_streaming_retrieval_flow(test_engine: StreamingLiveRAGEngine) -> None:
    events = list(
        test_engine.process_transcript_stream(
            transcript_chunk="Tell me about the Exynos 2400 processor specifications",
            session_id="stream_test_1",
            is_final_chunk=True,
        )
    )

    event_types = [e.get("type") for e in events]

    assert "transcript" in event_types
    assert "controller_decision" in event_types
    assert "decomposition" in event_types
    assert "evidence" in event_types
    assert "token" in event_types
    assert "timings" in event_types

    # Find controller decision
    cd = next(e for e in events if e.get("type") == "controller_decision")
    assert cd.get("state") == ControllerState.RETRIEVE.value

    # Find timings
    timing_event = next(e for e in events if e.get("type") == "timings")
    timings = timing_event.get("timings", {})
    assert "ttft_ms" in timings
    assert "total_ms" in timings
    assert timings["total_ms"] >= timings["ttft_ms"]


def test_streaming_conversational_no_retrieve_flow(test_engine: StreamingLiveRAGEngine) -> None:
    events = list(
        test_engine.process_transcript_stream(
            transcript_chunk="Hello AVY",
            session_id="stream_test_2",
            is_final_chunk=True,
        )
    )

    event_types = [e.get("type") for e in events]

    assert "controller_decision" in event_types
    cd = next(e for e in events if e.get("type") == "controller_decision")
    assert cd.get("state") == ControllerState.NO_RETRIEVE.value

    # Conversational flow should not have decomposition or evidence
    assert "decomposition" not in event_types
    assert "evidence" not in event_types
    assert "token" in event_types


def test_streaming_wait_state(test_engine: StreamingLiveRAGEngine) -> None:
    events = list(
        test_engine.process_transcript_stream(
            transcript_chunk="Tell me about Samsung's...",
            session_id="stream_test_3",
            is_final_chunk=False,
        )
    )

    event_types = [e.get("type") for e in events]
    assert "controller_decision" in event_types
    cd = next(e for e in events if e.get("type") == "controller_decision")
    assert cd.get("state") == ControllerState.WAIT.value

    # Must NOT generate tokens when waiting for more speech input
    assert "token" not in event_types
