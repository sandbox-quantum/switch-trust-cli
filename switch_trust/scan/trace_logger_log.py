"""LogTraceLogger — emits trace events via Python logging."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager

from switch_trust.scan.trace_logger import (
    CallContext,
    TraceLogger,
    estimate_tokens,
)

logger = logging.getLogger(__name__)


class LogTraceLogger(TraceLogger):
    """
    Trace logger that emits structured information via Python's logging module
    instead of writing to a file. Useful for environments where file output
    is unavailable or when trace data should flow through the standard
    log pipeline.
    """

    def __init__(self, session_id: str | None = None, log_level: int = logging.INFO):
        self.session_id = session_id or f"AGT-SCAN-{uuid.uuid4().hex[:8].upper()}"
        self._log_level = log_level
        self._start_time: float | None = None
        self._provider = "unknown"
        self._iterations = 0
        self._call_count = 0
        self._total_tokens = 0

    def start(self, provider_model: str) -> None:
        self._start_time = time.monotonic()
        self._provider = provider_model
        logger.log(
            self._log_level,
            "[trace:%s] session_start | model=%s",
            self.session_id,
            provider_model,
        )

    @contextmanager
    def record_call(self, tool_name: str, tool_args: dict, iteration: int):
        call_start = time.monotonic()
        ctx = CallContext()
        try:
            yield ctx
        finally:
            duration_ms = int((time.monotonic() - call_start) * 1000)
            result_text = ctx.result or ""
            tokens = estimate_tokens(result_text)
            self._total_tokens += tokens
            self._call_count += 1

            logger.log(
                self._log_level,
                "[trace:%s] tool_call #%d | iter=%d tool=%s "
                "duration=%dms tokens=%d success=%s%s",
                self.session_id,
                self._call_count,
                iteration,
                tool_name,
                duration_ms,
                tokens,
                ctx.success,
                f" error={ctx.error}" if ctx.error else "",
            )

    def set_iteration(self, n: int) -> None:
        self._iterations = n

    def finish(self, findings_count: int, exit_reason: str = "completed") -> None:
        if self._start_time is None:
            return
        wall_ms = int((time.monotonic() - self._start_time) * 1000)
        logger.log(
            self._log_level,
            "[trace:%s] session_end | exit=%s iterations=%d calls=%d "
            "tokens≈%d wall=%dms findings=%d",
            self.session_id,
            exit_reason,
            self._iterations,
            self._call_count,
            self._total_tokens,
            wall_ms,
            findings_count,
        )

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "provider_model": self._provider,
            "total_iterations": self._iterations,
            "total_tool_calls": self._call_count,
            "total_tokens": self._total_tokens,
        }
