"""Tests for structured Telemetry logging and event schema validation."""

import json
from pathlib import Path

from avy.telemetry.events import TelemetryEventNames
from avy.telemetry.logger import TelemetryLogger, _scrub_secrets


def test_telemetry_event_logging_and_file_persistence(tmp_path: Path) -> None:
    log_file = tmp_path / "events.jsonl"
    logger = TelemetryLogger(log_path=log_file, enabled=True)

    captured_events = []
    logger.subscribe(lambda ev: captured_events.append(ev))

    logger.log(
        event=TelemetryEventNames.INPUT_CHUNK,
        session_id="sess_t1",
        utterance_id="utt_t1",
        metadata={"chunk": "Tell me about Samsung"},
    )
    logger.log(
        event=TelemetryEventNames.RETRIEVAL_TRIGGERED,
        session_id="sess_t1",
        utterance_id="utt_t1",
        metadata={"reason": "Sufficient semantic predicate"},
    )

    assert len(captured_events) == 2
    assert log_file.is_file()

    with open(log_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 2
    assert lines[0]["event"] == TelemetryEventNames.INPUT_CHUNK
    assert lines[1]["event"] == TelemetryEventNames.RETRIEVAL_TRIGGERED
    assert "timestamp" in lines[0]


def test_telemetry_secret_scrubbing() -> None:
    raw_metadata = {
        "query": "Tell me about Exynos",
        "api_key": "sk-1234567890abcdef",
        "nested": {
            "token": "secret_token_val",
            "safe_field": 42,
        },
    }

    scrubbed = _scrub_secrets(raw_metadata)

    assert scrubbed["query"] == "Tell me about Exynos"
    assert scrubbed["api_key"] == "[REDACTED]"
    assert scrubbed["nested"]["token"] == "[REDACTED]"
    assert scrubbed["nested"]["safe_field"] == 42
