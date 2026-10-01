"""Tests for CLI telemetry privacy and failure isolation.

Two guarantees are locked in here, both requested during security review of the
telemetry PR:

1. Attribute allowlist — every attribute set that can reach an exporter passes
   through ``_sanitize`` against an explicit allowlist. Disallowed keys and
   sensitive values (exception messages, filenames, paths, prompts, provider
   output) cannot be exported.
2. Failure isolation — telemetry initialization, export, and shutdown failures
   never propagate and never change the CLI's result or exit status.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider

import switch_trust.cli.main as main_mod
import switch_trust.cli.telemetry as telemetry

# A value that stands in for the kinds of user-controlled / sensitive content
# that must never leave the process: absolute file paths, prompts, provider
# output. Assertions check this substring never appears in any exported field.
_SECRET = "/Users/alice/secret-project/prompt-and-provider-output.py"

_TELEMETRY_GLOBALS = (
    "_tracer_provider",
    "_logger_provider",
    "_meter_provider",
    "_client_id",
    "_command_counter",
    "_duration_histogram",
)


@pytest.fixture(autouse=True)
def _reset_telemetry_globals():
    """Snapshot and restore telemetry module globals around every test."""
    saved = {name: getattr(telemetry, name) for name in _TELEMETRY_GLOBALS}
    yield
    for name, value in saved.items():
        setattr(telemetry, name, value)


class _InMemoryTelemetry:
    """In-memory OTel providers wired into the telemetry module globals.

    Exercises the real emit code paths (``command_span`` -> ``on_end`` ->
    ``logger.emit`` / metric recording, and ``emit_event``) while capturing
    everything that would have been exported, without touching the network.
    """

    def __init__(self) -> None:
        self.log_exporter = InMemoryLogExporter()
        self.logger_provider = LoggerProvider()
        self.logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(self.log_exporter)
        )

        self.metric_reader = InMemoryMetricReader()
        self.meter_provider = MeterProvider(metric_readers=[self.metric_reader])
        meter = self.meter_provider.get_meter("test")
        self.counter = meter.create_counter("switch_trust.cli.command")
        self.histogram = meter.create_histogram("switch_trust.cli.command.duration")

        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            telemetry._SpanToLogProcessor(self.logger_provider)
        )

    def install(self) -> None:
        telemetry._tracer_provider = self.tracer_provider
        telemetry._logger_provider = self.logger_provider
        telemetry._meter_provider = self.meter_provider
        telemetry._command_counter = self.counter
        telemetry._duration_histogram = self.histogram
        telemetry._client_id = "test-client-id"

    def log_attributes(self) -> dict:
        records = self.log_exporter.get_finished_logs()
        assert records, "expected at least one exported log record"
        return dict(records[-1].log_record.attributes or {})

    def log_body(self) -> str:
        records = self.log_exporter.get_finished_logs()
        assert records
        return str(records[-1].log_record.body)

    def metric_point_attributes(self) -> list[dict]:
        data = self.metric_reader.get_metrics_data()
        points: list[dict] = []
        if not data:
            return points
        for resource_metric in data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    for point in metric.data.data_points:
                        points.append(dict(point.attributes))
        return points


@pytest.fixture
def inmem() -> _InMemoryTelemetry:
    env = _InMemoryTelemetry()
    env.install()
    return env


# ---------------------------------------------------------------------------
# Item 1: attribute allowlist enforced at the emit boundary
# ---------------------------------------------------------------------------


class TestAttributeAllowlist:
    def test_sanitize_drops_disallowed_keys_and_their_values(self):
        attrs = {
            "switch_trust.command": "scan",
            "exception.message": _SECRET,
            "arbitrary.injected": _SECRET,
        }

        out = telemetry._sanitize(attrs, telemetry._ALLOWED_LOG_ATTRIBUTES)

        assert out == {"switch_trust.command": "scan"}
        assert _SECRET not in "".join(str(v) for v in out.values())

    def test_span_path_strips_disallowed_key_and_exception_message(self, inmem):
        # command_span -> on_end -> logger.emit is the primary path that carries
        # user-controlled attributes (anything set on the span) to the exporter.
        with telemetry.command_span("scan", subcommand=None) as span:
            # An attribute an attacker/future-dev might set directly on the span.
            span.set_attribute("secret.path", _SECRET)
            # record_error records only exception.type; the message (which
            # embeds the path) is never captured, and exception.message is
            # excluded from the allowlist as a backstop.
            telemetry.record_error(
                span, FileNotFoundError(f"Path does not exist: {_SECRET}")
            )
            span.set_attribute("exception.message", str(FileNotFoundError(_SECRET)))

        attrs = inmem.log_attributes()

        assert "secret.path" not in attrs
        assert "exception.message" not in attrs
        assert attrs.get("exception.type") == "FileNotFoundError"
        # The sensitive value must not survive in any exported attribute or body.
        assert _SECRET not in "".join(str(v) for v in attrs.values())
        assert _SECRET not in inmem.log_body()

    def test_span_path_keeps_all_allowlisted_keys(self, inmem):
        with telemetry.command_span(
            "eval", subcommand="models list", is_ci=True, is_first_time=True
        ):
            pass

        attrs = inmem.log_attributes()

        assert attrs["switch_trust.command"] == "eval"
        assert attrs["switch_trust.subcommand"] == "models list"
        assert attrs["switch_trust.is_ci"] is True
        assert attrs["switch_trust.is_first_time"] is True
        assert attrs["switch_trust.client_id"] == "test-client-id"
        assert attrs["switch_trust.status"] == "ok"
        assert attrs["status"] == "ok"
        assert "switch_trust.duration_ms" in attrs
        # Every exported key is on the allowlist.
        assert set(attrs) <= telemetry._ALLOWED_LOG_ATTRIBUTES

    def test_emit_event_only_exports_allowlisted_keys(self, inmem):
        telemetry.emit_event("init", is_ci=False, is_first_time=True)

        attrs = inmem.log_attributes()

        assert set(attrs) <= telemetry._ALLOWED_LOG_ATTRIBUTES
        assert attrs["switch_trust.command"] == "init"
        assert attrs["switch_trust.status"] == "ok"

    def test_metric_path_only_exports_allowlisted_keys(self, inmem):
        with telemetry.command_span("scan", subcommand=None):
            pass

        point_attrs = inmem.metric_point_attributes()

        assert point_attrs, "expected at least one metric data point"
        for attrs in point_attrs:
            assert set(attrs) <= telemetry._ALLOWED_METRIC_ATTRIBUTES

    def test_resource_only_carries_allowlisted_keys(self):
        # The resource is attached to every exported record; it must be bounded
        # by the allowlist too.
        assert telemetry._ALLOWED_RESOURCE_ATTRIBUTES == frozenset(
            {
                "service.name",
                "service.version",
                "switch_trust.client_id",
                "python.version",
                "os.type",
                "os.version",
            }
        )
        assert "exception.message" not in telemetry._ALLOWED_LOG_ATTRIBUTES


# ---------------------------------------------------------------------------
# Item 2: failure isolation (unit level)
# ---------------------------------------------------------------------------


class _BrokenLogger:
    def emit(self, *args, **kwargs):
        raise RuntimeError("emit boom")


class _BrokenLoggerProvider:
    def get_logger(self, *args, **kwargs):
        return _BrokenLogger()


class _BrokenProvider:
    def shutdown(self, *args, **kwargs):
        raise RuntimeError("shutdown boom")


class TestFailureIsolationUnit:
    def test_init_failure_is_swallowed_and_disables_telemetry(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("exporter construction failed")

        monkeypatch.setattr(telemetry, "OTLPLogExporter", boom)

        # Must not raise.
        telemetry.init_telemetry("client-id", "1.2.3")

        # All providers are left disabled so downstream calls are no-ops.
        assert telemetry._tracer_provider is None
        assert telemetry._logger_provider is None
        assert telemetry._meter_provider is None
        assert telemetry._command_counter is None
        assert telemetry._duration_histogram is None

    def test_export_failure_is_swallowed(self, monkeypatch):
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            telemetry._SpanToLogProcessor(_BrokenLoggerProvider())
        )
        monkeypatch.setattr(telemetry, "_tracer_provider", tracer_provider)

        # span.end() -> on_end -> broken emit; must not propagate.
        with telemetry.command_span("scan", subcommand=None) as span:
            telemetry.record_error(span, RuntimeError("scan failed"))

    def test_shutdown_failure_is_swallowed(self, monkeypatch):
        monkeypatch.setattr(telemetry, "_tracer_provider", _BrokenProvider())
        monkeypatch.setattr(telemetry, "_logger_provider", _BrokenProvider())
        monkeypatch.setattr(telemetry, "_meter_provider", _BrokenProvider())

        # Must not raise even though every provider.shutdown() throws.
        telemetry.shutdown_telemetry()

        # Providers are cleared regardless of shutdown failures.
        assert telemetry._tracer_provider is None
        assert telemetry._logger_provider is None
        assert telemetry._meter_provider is None

    def test_record_helpers_are_noops_without_span(self):
        # When telemetry is disabled command_span yields None; the recorders
        # must tolerate a None span.
        telemetry.record_error(None, RuntimeError("boom"))
        telemetry.record_interruption(None)

    def test_emit_event_noop_without_provider(self, monkeypatch):
        monkeypatch.setattr(telemetry, "_logger_provider", None)
        # Must not raise.
        telemetry.emit_event("init")


# ---------------------------------------------------------------------------
# Item 2: failure isolation (end-to-end via main())
# ---------------------------------------------------------------------------

# The exit-code invariant: for each command outcome the code is fixed, and every
# telemetry state below must reproduce it exactly. success -> 0 (main returns),
# exception -> 1, KeyboardInterrupt -> 130.
_OUTCOMES = [
    pytest.param({"result": None}, 0, id="success"),
    pytest.param({"exc": RuntimeError("boom")}, 1, id="exception"),
    pytest.param({"exc": KeyboardInterrupt()}, 130, id="interrupt"),
]


def _configure_main_env(monkeypatch, tmp_path, *, consent):
    monkeypatch.setattr(main_mod, "silence_noisy_loggers", lambda: None)
    monkeypatch.setattr(main_mod, "_load_environment", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "setup_file_logging", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "_print_logo", lambda: None)
    monkeypatch.setattr(main_mod, "is_ci", lambda: True)
    monkeypatch.setattr(main_mod, "ensure_client_id", lambda: None)
    monkeypatch.setattr(main_mod, "ensure_telemetry_consent", lambda: None)
    monkeypatch.setattr(main_mod, "get_telemetry_consent", lambda: consent)
    monkeypatch.setattr(main_mod, "get_client_id", lambda: "test-client-id")

    env = tmp_path / ".switch-trust.env"
    env.write_text("")
    monkeypatch.setattr(main_mod.init_cli, "get_switch_trust_env_path", lambda: env)


def _set_dispatch(monkeypatch, *, result=None, exc=None):
    def _dispatch(args):
        if exc is not None:
            raise exc
        return result

    monkeypatch.setattr(main_mod, "_dispatch", _dispatch)


def _invoke_main(argv) -> int:
    try:
        main_mod.main(argv)
        return 0
    except SystemExit as exc:
        return exc.code if exc.code is not None else 0


class TestFailureIsolationEndToEnd:
    @pytest.mark.parametrize("dispatch_kwargs, expected_code", _OUTCOMES)
    def test_baseline_telemetry_disabled(
        self, monkeypatch, tmp_path, dispatch_kwargs, expected_code
    ):
        _configure_main_env(monkeypatch, tmp_path, consent=False)
        _set_dispatch(monkeypatch, **dispatch_kwargs)

        assert _invoke_main(["eval", "models", "list"]) == expected_code

    @pytest.mark.parametrize("dispatch_kwargs, expected_code", _OUTCOMES)
    def test_exit_code_unchanged_when_init_fails(
        self, monkeypatch, tmp_path, dispatch_kwargs, expected_code
    ):
        _configure_main_env(monkeypatch, tmp_path, consent=True)
        _set_dispatch(monkeypatch, **dispatch_kwargs)

        def boom(*args, **kwargs):
            raise RuntimeError("exporter construction failed")

        monkeypatch.setattr(telemetry, "OTLPLogExporter", boom)

        assert _invoke_main(["eval", "models", "list"]) == expected_code

    @pytest.mark.parametrize("dispatch_kwargs, expected_code", _OUTCOMES)
    def test_exit_code_unchanged_with_working_telemetry(
        self, monkeypatch, tmp_path, inmem, dispatch_kwargs, expected_code
    ):
        _configure_main_env(monkeypatch, tmp_path, consent=True)
        _set_dispatch(monkeypatch, **dispatch_kwargs)
        # Keep the in-memory providers installed by the `inmem` fixture; don't
        # let main() replace them with real network exporters. main() imports
        # init_telemetry from the telemetry module lazily, so patch it there.
        monkeypatch.setattr(telemetry, "init_telemetry", lambda *a, **k: None)

        assert _invoke_main(["eval", "models", "list"]) == expected_code

    @pytest.mark.parametrize("dispatch_kwargs, expected_code", _OUTCOMES)
    def test_exit_code_unchanged_when_export_fails_midflow(
        self, monkeypatch, tmp_path, dispatch_kwargs, expected_code
    ):
        _configure_main_env(monkeypatch, tmp_path, consent=True)
        _set_dispatch(monkeypatch, **dispatch_kwargs)
        # main() imports init_telemetry lazily from the telemetry module.
        monkeypatch.setattr(telemetry, "init_telemetry", lambda *a, **k: None)

        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            telemetry._SpanToLogProcessor(_BrokenLoggerProvider())
        )
        monkeypatch.setattr(telemetry, "_tracer_provider", tracer_provider)

        assert _invoke_main(["eval", "models", "list"]) == expected_code
