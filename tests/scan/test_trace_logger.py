"""
Tests for the trace_logger module family:
  trace_logger.py      — abstract base + utilities
  trace_logger_file.py — JSONL file writer
  trace_logger_log.py  — Python logging emitter
"""

import json
import os
import tempfile
import unittest

from switch_trust.scan.trace_logger import (
    CallContext,
    TraceLogger,
    estimate_tokens,
    now_iso,
)
from switch_trust.scan.trace_logger_file import FileTraceLogger
from switch_trust.scan.trace_logger_log import LogTraceLogger

# ── Utilities ───────────────────────────────────────────────────────


class TestEstimateTokens(unittest.TestCase):
    def test_short_text(self):
        result = estimate_tokens("hello")
        self.assertGreaterEqual(result, 1)

    def test_empty_text(self):
        self.assertEqual(estimate_tokens(""), 1)

    def test_longer_text(self):
        result = estimate_tokens("a" * 400)
        self.assertGreaterEqual(result, 10)


class TestNowIso(unittest.TestCase):
    def test_returns_iso_string(self):
        result = now_iso()
        self.assertIn("T", result)
        self.assertTrue(result.endswith("+00:00"))


class TestCallContext(unittest.TestCase):
    def test_initial_state(self):
        ctx = CallContext()
        self.assertIsNone(ctx.result)
        self.assertTrue(ctx.success)
        self.assertIsNone(ctx.error)

    def test_set_result(self):
        ctx = CallContext()
        ctx.set_result("some result")
        self.assertEqual(ctx.result, "some result")
        self.assertTrue(ctx.success)

    def test_set_error(self):
        ctx = CallContext()
        ctx.set_error("something broke")
        self.assertEqual(ctx.error, "something broke")
        self.assertFalse(ctx.success)
        self.assertEqual(ctx.result, "ERROR: something broke")


class TestTraceLoggerIsAbstract(unittest.TestCase):
    def test_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            TraceLogger()


# ── FileTraceLogger ─────────────────────────────────────────────────


class TestFileTraceLogger(unittest.TestCase):
    def test_init_with_defaults(self):
        logger = FileTraceLogger()
        self.assertTrue(
            logger.session_id.startswith("AGT-SCAN-"),
        )
        self.assertIsNone(logger.output_path)

    def test_init_with_output_path(self):
        logger = FileTraceLogger(
            session_id="test-123",
            output_path="/tmp/report.json",
        )
        self.assertEqual(logger.session_id, "test-123")
        self.assertEqual(
            logger._trace_path,
            "/tmp/report.json.trace.jsonl",
        )

    def test_as_dict_before_start(self):
        logger = FileTraceLogger(session_id="test-123")
        d = logger.as_dict()
        self.assertEqual(d["session_id"], "test-123")
        self.assertEqual(d["total_tool_calls"], 0)
        self.assertEqual(d["total_tokens"], 0)

    def test_full_session_writes_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "report.json")
            logger = FileTraceLogger(
                session_id="test-session",
                output_path=output,
            )
            logger.start(provider_model="test-model")

            with logger.record_call(
                "read_source",
                {"path": "f.py"},
                iteration=1,
            ) as ctx:
                ctx.set_result("file content here")

            logger.set_iteration(1)
            logger.finish(
                findings_count=3,
                exit_reason="completed",
            )

            trace_path = output + ".trace.jsonl"
            self.assertTrue(os.path.exists(trace_path))

            with open(trace_path) as f:
                lines = f.readlines()

            self.assertEqual(len(lines), 3)

            start_event = json.loads(lines[0])
            self.assertEqual(
                start_event["event"],
                "session_start",
            )
            self.assertEqual(start_event["model"], "test-model")

            tool_event = json.loads(lines[1])
            self.assertEqual(tool_event["event"], "tool_call")
            self.assertEqual(tool_event["tool"], "read_source")
            self.assertTrue(tool_event["success"])
            self.assertGreater(tool_event["tokens_consumed"], 0)

            end_event = json.loads(lines[2])
            self.assertEqual(
                end_event["event"],
                "session_end",
            )
            self.assertEqual(end_event["findings_count"], 3)
            self.assertEqual(
                end_event["exit_reason"],
                "completed",
            )

    def test_record_call_with_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "report.json")
            logger = FileTraceLogger(
                session_id="err-test",
                output_path=output,
            )
            logger.start(provider_model="test")

            with logger.record_call(
                "bad_tool",
                {},
                iteration=0,
            ) as ctx:
                ctx.set_error("tool failed")

            logger.finish(findings_count=0)

            trace_path = output + ".trace.jsonl"
            with open(trace_path) as f:
                lines = f.readlines()

            tool_event = json.loads(lines[1])
            self.assertFalse(tool_event["success"])
            self.assertEqual(tool_event["error"], "tool failed")

    def test_finish_without_start_is_noop(self):
        logger = FileTraceLogger(session_id="no-start")
        logger.finish(findings_count=0)

    def test_no_trace_when_no_output_path(self):
        logger = FileTraceLogger(session_id="no-output")
        logger.start(provider_model="test")
        with logger.record_call(
            "tool",
            {},
            iteration=0,
        ) as ctx:
            ctx.set_result("ok")
        logger.finish(findings_count=0)
        d = logger.as_dict()
        self.assertEqual(d["total_tool_calls"], 1)


# ── LogTraceLogger ──────────────────────────────────────────────────


class TestLogTraceLogger(unittest.TestCase):
    def test_full_session(self):
        logger = LogTraceLogger(session_id="log-test")
        logger.start(provider_model="test-model")

        with logger.record_call(
            "read_source",
            {"path": "f.py"},
            iteration=1,
        ) as ctx:
            ctx.set_result("content")

        logger.set_iteration(1)
        logger.finish(findings_count=2)

        d = logger.as_dict()
        self.assertEqual(d["session_id"], "log-test")
        self.assertEqual(d["total_tool_calls"], 1)
        self.assertGreater(d["total_tokens"], 0)

    def test_finish_without_start_is_noop(self):
        logger = LogTraceLogger(session_id="no-start")
        logger.finish(findings_count=0)

    def test_record_call_with_error(self):
        logger = LogTraceLogger(session_id="err-test")
        logger.start(provider_model="test")

        with logger.record_call(
            "bad_tool",
            {},
            iteration=0,
        ) as ctx:
            ctx.set_error("failed")

        d = logger.as_dict()
        self.assertEqual(d["total_tool_calls"], 1)

    def test_as_dict_before_start(self):
        logger = LogTraceLogger(session_id="fresh")
        d = logger.as_dict()
        self.assertEqual(d["total_tool_calls"], 0)


if __name__ == "__main__":
    unittest.main()
