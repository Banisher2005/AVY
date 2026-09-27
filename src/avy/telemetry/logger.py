"""Thread-safe structured JSON/JSONL telemetry logger."""

import json
import threading
from collections import deque
from pathlib import Path
from typing import Any, Callable

from avy.telemetry.events import TelemetryEvent

_SENSITIVE_KEYS = {"api_key", "token", "password", "secret", "authorization", "auth"}


def _scrub_secrets(obj: Any) -> Any:
    """Recursively scrub any sensitive credentials from metadata before logging."""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if any(s in k.lower() for s in _SENSITIVE_KEYS):
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = _scrub_secrets(v)
        return cleaned
    elif isinstance(obj, list):
        return [_scrub_secrets(item) for item in obj]
    return obj


class TelemetryLogger:
    """Thread-safe telemetry logger writing structured JSONL."""

    def __init__(self, log_path: str | Path | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self.log_path = Path(log_path) if log_path else None
        if self.log_path and self.enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._listeners: list[Callable[[TelemetryEvent], None]] = []
        self._recent_events: deque[TelemetryEvent] = deque(maxlen=2000)

    def subscribe(self, callback: Callable[[TelemetryEvent], None]) -> None:
        """Register a callback for real-time telemetry streaming."""
        self._listeners.append(callback)

    def log(
        self,
        event: str,
        session_id: str,
        utterance_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> TelemetryEvent:
        """Record and emit a telemetry event."""
        meta = _scrub_secrets(metadata or {})
        record = TelemetryEvent(
            event=event,
            session_id=session_id,
            utterance_id=utterance_id,
            metadata=meta,
        )

        with self._lock:
            self._recent_events.append(record)

        if not self.enabled:
            return record

        # Write to JSONL file
        if self.log_path:
            with self._lock:
                try:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record.to_dict()) + "\n")
                except Exception:
                    pass

        # Notify in-memory subscribers
        for listener in self._listeners:
            try:
                listener(record)
            except Exception:
                pass

        return record

    def get_events(self, session_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve recent telemetry records, optionally filtered by session_id."""
        with self._lock:
            events = list(self._recent_events)

        if session_id:
            events = [e for e in events if e.session_id == session_id]

        events = events[-limit:]
        return [e.to_dict() for e in events]


_GLOBAL_TELEMETRY: TelemetryLogger | None = None


def get_telemetry_logger(log_path: str | Path | None = None) -> TelemetryLogger:
    global _GLOBAL_TELEMETRY
    if _GLOBAL_TELEMETRY is None:
        _GLOBAL_TELEMETRY = TelemetryLogger(log_path=log_path)
    return _GLOBAL_TELEMETRY
