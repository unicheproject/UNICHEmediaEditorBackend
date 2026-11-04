"""Duration-based progress for FFmpeg work inside one Job execution.

``execute_job`` activates a tracker (via a context variable) when the Job's
expected output duration is known. ``runner.run`` then asks FFmpeg for
machine-readable progress and feeds its output time here, so the roughly 40
tool call sites need no changes. Several sequential FFmpeg runs accumulate
(e.g. split segments, concat normalization); tools whose output length differs
from the input (trim) override the expectation.

Progress is advisory: it is monotonic, capped below completion (publication
sets 100), and a failing sink never fails media processing.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from app.core.logging import get_logger

logger = get_logger(__name__)

ProgressSink = Callable[[float], Awaitable[None]]
_CAP = 0.99


@dataclass
class ProgressTracker:
    expected_seconds: float
    sink: ProgressSink
    completed_seconds: float = 0.0
    _last_fraction: float = field(default=0.0, repr=False)

    def expect(self, seconds: float) -> None:
        """Replace the expected total output duration (e.g. a trim window)."""
        if seconds > 0:
            self.expected_seconds = seconds

    async def update(self, current_run_seconds: float) -> None:
        if self.expected_seconds <= 0 or current_run_seconds < 0:
            return
        fraction = (self.completed_seconds + current_run_seconds) / self.expected_seconds
        fraction = min(fraction, _CAP)
        if fraction <= self._last_fraction:
            return
        self._last_fraction = fraction
        try:
            await self.sink(fraction)
        except Exception as exc:  # noqa: BLE001 - progress must never fail a Job
            logger.warning("Progress update failed (%s)", type(exc).__name__)

    def finish_run(self, run_seconds: float) -> None:
        if run_seconds > 0:
            self.completed_seconds += run_seconds


_current: ContextVar[ProgressTracker | None] = ContextVar("media_progress", default=None)


def current() -> ProgressTracker | None:
    return _current.get()


@contextmanager
def tracking(expected_seconds: float | None, sink: ProgressSink) -> Iterator[bool]:
    """Activate progress for this task; yields whether tracking is active."""
    if expected_seconds is None or expected_seconds <= 0:
        yield False
        return
    token = _current.set(ProgressTracker(expected_seconds, sink))
    try:
        yield True
    finally:
        _current.reset(token)


def expect_output_seconds(seconds: float) -> None:
    """Called by tools that know their output length differs from the input."""
    tracker = _current.get()
    if tracker is not None:
        tracker.expect(seconds)


def parse_out_time_seconds(line: str) -> float | None:
    """Read FFmpeg ``-progress`` output time; ``out_time_us`` is microseconds."""
    key, _, value = line.strip().partition("=")
    if key not in {"out_time_us", "out_time_ms"}:  # out_time_ms is also µs
        return None
    try:
        microseconds = int(value)
    except ValueError:
        return None  # e.g. "N/A" before the first frame
    return microseconds / 1_000_000 if microseconds >= 0 else None
