"""
trace_logger.py — Structured Observability for the Agentic Reasoning Loop
=========================================================================
Writes a JSONL trace log of every tool call made during an agentic scan session.

WHY THIS EXISTS:
  The agentic scanner itself exhibits the behaviours it scans for in customer agents:
  autonomous tool-calling, adaptive control flow, incremental decision-making.
  Per ASI10 (missing_agent_monitoring), any agent that takes consequential actions
  without an audit trail is a security risk. This module is the scanner's own
  compliance with that rule.

OUTPUT FORMAT:
  One JSON object per line, written to <output_path>.trace.jsonl
  Each line represents one tool call with timing, token estimates, and a result preview.
  The final line is a session summary record.

USAGE:
  logger = FileTraceLogger(session_id="AGT-SCAN-abc123", output_path="report.json")
  logger.start(provider_model="claude-sonnet-4-6")

  with logger.record_call("fetch_file", {"path": "tools/exec.py"}) as ctx:
      result = do_fetch(...)
      ctx.set_result(result)

  logger.finish(findings_count=5, exit_reason="completed")
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

import tiktoken

# ── Token estimation ──────────────────────────────────────────────────────────

_TOKEN_REFERENCE_MODEL = "gpt-4o"


def estimate_tokens(text: str) -> int:
    """Estimate token count using tiktoken (accurate) or char heuristic (fallback).

    Returns:
        Estimated number of tokens (minimum 1).
    """
    try:
        return max(
            1, len(tiktoken.encoding_for_model(_TOKEN_REFERENCE_MODEL).encode(text))
        )
    except Exception:
        return max(1, len(text) // 4)


def now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


class CallContext:
    """Mutable context shared between the record_call context manager and the caller."""

    def __init__(self):
        self.result: str | None = None
        self.success: bool = True
        self.error: str | None = None

    def set_result(self, result: str) -> None:
        self.result = result

    def set_error(self, error: str) -> None:
        self.error = error
        self.success = False
        self.result = f"ERROR: {error}"


# ── Abstract interface ────────────────────────────────────────────────────────


class TraceLogger(abc.ABC):
    """
    Abstract interface for structured trace logging of tool calls
    during an agentic scan session.

    Thread-safety: single-threaded use only (scanner runs synchronously).
    """

    @abc.abstractmethod
    def start(self, provider_model: str) -> None:
        """Begin a trace session. Call once before the reasoning loop starts."""

    @abc.abstractmethod
    @contextmanager
    def record_call(
        self, tool_name: str, tool_args: dict, iteration: int
    ) -> Iterator[CallContext]:
        """Context manager that times a tool call and writes the trace entry.

        Usage:
            with logger.record_call("fetch_file", {"path": "x.py"}, iteration=2) as ctx:
                result = fetch_file(...)
                ctx.set_result(result)
        """
        yield CallContext()  # pragma: no cover

    @abc.abstractmethod
    def set_iteration(self, n: int) -> None:
        """Update the current iteration count."""

    @abc.abstractmethod
    def finish(self, findings_count: int, exit_reason: str = "completed") -> None:
        """Write the session summary and close. Call once after the reasoning loop exits."""

    @abc.abstractmethod
    def as_dict(self) -> dict:
        """Return a summary dict suitable for embedding in ScanReport.scan_metadata."""
