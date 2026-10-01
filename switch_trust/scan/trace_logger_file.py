"""FileTraceLogger — writes a JSONL trace file to disk."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager

from switch_trust.scan.trace_logger import (
    CallContext,
    TraceLogger,
    estimate_tokens,
    now_iso,
)

logger = logging.getLogger(__name__)


class FileTraceLogger(TraceLogger):
    """
    Writes a structured JSONL trace of all tool calls made during an agentic scan.

    Output is written to ``<output_path>.trace.jsonl``.
    """

    def __init__(self, session_id: str | None = None, output_path: str | None = None):
        self.session_id = session_id or f"AGT-SCAN-{uuid.uuid4().hex[:8].upper()}"
        self.output_path = output_path
        self._trace_path = f"{output_path}.trace.jsonl" if output_path else None
        self._start_time: float | None = None
        self._provider = "unknown"
        self._iterations = 0
        self._call_count = 0
        self._total_tokens = 0
        self._tool_calls: list[dict] = []
        self._file = None

    def start(self, provider_model: str) -> None:
        self._start_time = time.monotonic()
        self._provider = provider_model
        self._write_line(
            {
                "event": "session_start",
                "session_id": self.session_id,
                "model": provider_model,
                "timestamp": now_iso(),
            }
        )
        logger.info(
            "Trace session started: %s | model: %s", self.session_id, provider_model
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

            entry = {
                "event": "tool_call",
                "session_id": self.session_id,
                "iteration": iteration,
                "call_number": self._call_count,
                "tool": tool_name,
                "args": tool_args,
                "result_preview": result_text[:200],
                "result_length": len(result_text),
                "tokens_consumed": tokens,
                "tokens_total": self._total_tokens,
                "duration_ms": duration_ms,
                "timestamp": now_iso(),
                "success": ctx.success,
                "error": ctx.error,
            }
            self._tool_calls.append(entry)
            self._write_line(entry)

    def set_iteration(self, n: int) -> None:
        self._iterations = n

    def finish(self, findings_count: int, exit_reason: str = "completed") -> None:
        if self._start_time is None:
            return
        wall_ms = int((time.monotonic() - self._start_time) * 1000)
        summary = {
            "event": "session_end",
            "session_id": self.session_id,
            "model": self._provider,
            "exit_reason": exit_reason,
            "total_iterations": self._iterations,
            "total_tool_calls": self._call_count,
            "total_tokens": self._total_tokens,
            "wall_clock_ms": wall_ms,
            "findings_count": findings_count,
            "timestamp": now_iso(),
        }
        self._write_line(summary)
        if self._file:
            self._file.close()
            self._file = None
        logger.info(
            "Trace session complete: %s | iterations=%d | calls=%d | "
            "tokens≈%d | %dms | exit=%s",
            self.session_id,
            self._iterations,
            self._call_count,
            self._total_tokens,
            wall_ms,
            exit_reason,
        )
        if self._trace_path:
            logger.info("Trace written to: %s", self._trace_path)

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "provider_model": self._provider,
            "total_iterations": self._iterations,
            "total_tool_calls": self._call_count,
            "total_tokens": self._total_tokens,
        }

    def _write_line(self, obj: dict) -> None:
        """Append a JSON line to the trace file. Opens on first write."""
        if self._trace_path is None:
            return
        try:
            if self._file is None:
                self._file = open(self._trace_path, "w", encoding="utf-8")
            self._file.write(json.dumps(obj) + "\n")
            self._file.flush()
        except Exception as e:
            # Trace writes are best-effort — never crash the scanner
            logger.warning("Could not write trace entry: %s", e)
