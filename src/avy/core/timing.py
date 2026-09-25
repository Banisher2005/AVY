"""Precise execution timer utilities for AVY."""

import time
from contextlib import contextmanager
from typing import Generator


class Stopwatch:
    """High-resolution stopwatch for measuring component latencies."""

    def __init__(self) -> None:
        self._start: float = time.perf_counter()
        self._stop: float | None = None

    def stop(self) -> float:
        """Stop the stopwatch and return elapsed milliseconds."""
        if self._stop is None:
            self._stop = time.perf_counter()
        return (self._stop - self._start) * 1000.0

    @property
    def elapsed_ms(self) -> float:
        """Return current elapsed time in milliseconds."""
        end = self._stop if self._stop is not None else time.perf_counter()
        return (end - self._start) * 1000.0


@contextmanager
def measure_latency() -> Generator[Stopwatch, None, None]:
    """Context manager yielding a Stopwatch that records block duration."""
    sw = Stopwatch()
    try:
        yield sw
    finally:
        sw.stop()
