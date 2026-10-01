"""CLI usage telemetry via OpenTelemetry.

Spans are used internally as data accumulators (attributes, errors, latency),
then converted to OTel log records and metrics on close via SpanToLogProcessor.
Only log records and metrics leave the process — no traces are exported.

Providers are kept private (never registered globally via set_tracer_provider /
set_logger_provider / set_meter_provider) so that other OTel instrumentation
in the process (e.g. Google ADK, GenAI) cannot route its data through our
exporter.

Telemetry is strictly isolated from the CLI's local file logging: OTel log
records go to the OTLP exporter, Python logging.Logger records go to the file
handler. There is no LoggingHandler bridge between the two.

All telemetry operations are wrapped in try/except so that a failing telemetry
endpoint never disrupts the CLI. Errors are logged to the local file log.

Every attribute that leaves the process passes through an explicit allowlist
(``_sanitize``) at the emit boundary: log records against
``_ALLOWED_LOG_ATTRIBUTES``, metrics against ``_ALLOWED_METRIC_ATTRIBUTES``,
and the resource against ``_ALLOWED_RESOURCE_ATTRIBUTES``. Any key not on the
relevant list is dropped before it can reach an exporter, regardless of where
on the span it was set. In particular ``exception.message`` is never recorded
(free-text error strings routinely embed file paths, prompts, and provider
output) and is also excluded from the allowlist as a backstop — only the
bounded ``exception.type`` class name is exported. This is the control that
backs the consent prompt's promise that no code, prompts, keys, or personal
data is ever collected.
"""

from __future__ import annotations

import atexit
import logging
import platform
import sys
from collections.abc import Generator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry._logs import SeverityNumber
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import (
    Counter,
    Histogram,
    MeterProvider,
    ObservableCounter,
    ObservableGauge,
    ObservableUpDownCounter,
    UpDownCounter,
)
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.trace import Span

global_logger = logging.getLogger(__name__)

_BASE_ENDPOINT = "https://telemetry.flintai.dev"
_SERVICE_NAME = "switch-trust-cli"

# The event bridge that forwards OTel logs to Amplitude keys off the log
# record's `event_name` field (passed to Logger.emit, not an attribute) — a
# record without it is dropped by the bridge and only lands in Datadog. Every
# emitted record sets this to a single stable name so all CLI invocations
# surface as one Amplitude event type, with switch_trust.command / switch_trust.subcommand /
# switch_trust.status available as breakdown properties (mirrors the
# switch_trust.cli.command counter metric).
_EVENT_NAME = "switch_trust.cli.command"

# Explicit allowlists enforced at the emit boundary by _sanitize(). Any
# attribute key not present in the relevant set is dropped before it can reach
# an exporter, no matter where on the span it was set. These are the only keys
# that ever leave the process.
#
# exception.message is deliberately excluded: str(error) commonly embeds file
# paths, prompts, and provider output. Only the bounded exception.type class
# name is exported. switch_trust.command / switch_trust.subcommand are fixed enum-like values
# (scan, eval, init, ...), never user input.
_ALLOWED_LOG_ATTRIBUTES = frozenset(
    {
        "switch_trust.command",
        "switch_trust.subcommand",
        "switch_trust.is_ci",
        "switch_trust.is_first_time",
        "switch_trust.client_id",
        "switch_trust.status",
        "switch_trust.duration_ms",
        "switch_trust.interrupted",
        "status",
        "exception.type",
    }
)

_ALLOWED_METRIC_ATTRIBUTES = frozenset(
    {
        "switch_trust.command",
        "switch_trust.subcommand",
        "switch_trust.status",
        "switch_trust.client_id",
    }
)

_ALLOWED_RESOURCE_ATTRIBUTES = frozenset(
    {
        "service.name",
        "service.version",
        "switch_trust.client_id",
        "python.version",
        "os.type",
        "os.version",
    }
)

# Datadog turns cumulative monotonic sums into deltas and treats the first
# point as a baseline, so a CLI that increments once and exits would report
# nothing. DELTA temporality avoids this.
_DELTA_TEMPORALITY = {
    Counter: AggregationTemporality.DELTA,
    UpDownCounter: AggregationTemporality.DELTA,
    Histogram: AggregationTemporality.DELTA,
    ObservableCounter: AggregationTemporality.DELTA,
    ObservableUpDownCounter: AggregationTemporality.DELTA,
    ObservableGauge: AggregationTemporality.DELTA,
}

_tracer_provider: TracerProvider | None = None
_logger_provider: LoggerProvider | None = None
_meter_provider: MeterProvider | None = None
_client_id: str | None = None
_command_counter: Counter | None = None
_duration_histogram: Histogram | None = None


def _sanitize(attrs: dict, allowed: frozenset[str]) -> dict:
    """Return a copy of ``attrs`` containing only allowlisted keys.

    This is the single choke point every exported attribute set passes
    through: any key not on ``allowed`` is dropped before it can reach an
    exporter.
    """
    return {k: v for k, v in attrs.items() if k in allowed}


class _SpanToLogProcessor(SpanProcessor):
    """Converts completed spans into OTel log records and records metrics."""

    def __init__(self, logger_provider: LoggerProvider) -> None:
        self._logger = logger_provider.get_logger(_SERVICE_NAME)

    def on_start(self, span: ReadableSpan, parent_context: object = None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        try:
            attrs = dict(span.attributes or {})

            duration_ms = None
            if span.start_time and span.end_time:
                duration_ms = (span.end_time - span.start_time) / 1_000_000
                attrs["switch_trust.duration_ms"] = duration_ms

            is_error = span.status and span.status.status_code == trace.StatusCode.ERROR
            is_interrupted = attrs.get("switch_trust.interrupted", False)

            if is_error:
                status = "error"
            elif is_interrupted:
                status = "interrupted"
            else:
                status = "ok"
            attrs["switch_trust.status"] = status
            # Duplicate status as a root (unprefixed) attribute.
            attrs["status"] = status

            severity = SeverityNumber.ERROR if is_error else SeverityNumber.INFO
            severity_text = "ERROR" if is_error else "INFO"

            command = str(attrs.get("switch_trust.command", "unknown"))
            subcommand_raw = attrs.get("switch_trust.subcommand")
            subcommand = str(subcommand_raw) if subcommand_raw is not None else None
            uid = _client_id or "unknown"

            parts = [f"CLI: {uid} called {command}"]
            if subcommand:
                parts[0] += f" ({subcommand})"
            parts.append(status.upper())
            if duration_ms is not None:
                parts.append(f"{duration_ms / 1000:.1f}s")
            body = " - ".join(parts)

            self._logger.emit(
                severity_number=severity,
                severity_text=severity_text,
                body=body,
                attributes=_sanitize(attrs, _ALLOWED_LOG_ATTRIBUTES),
                # Routes this record to Amplitude via the OTel->Amplitude bridge.
                event_name=_EVENT_NAME,
            )

            self._record_metrics(command, subcommand, status, duration_ms)
        except Exception:
            global_logger.debug("Failed to convert span to log record", exc_info=True)

    def _record_metrics(
        self,
        command: str,
        subcommand: str | None,
        status: str,
        duration_ms: float | None,
    ) -> None:
        metric_attrs: dict[str, str] = {
            "switch_trust.command": command,
            "switch_trust.status": status,
        }
        if subcommand:
            metric_attrs["switch_trust.subcommand"] = subcommand
        if _client_id:
            metric_attrs["switch_trust.client_id"] = _client_id

        metric_attrs = _sanitize(metric_attrs, _ALLOWED_METRIC_ATTRIBUTES)
        if _command_counter:
            _command_counter.add(1, metric_attrs)
        if _duration_histogram and duration_ms is not None:
            _duration_histogram.record(duration_ms, metric_attrs)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def init_telemetry(client_id: str, version: str) -> None:
    global _tracer_provider, _logger_provider, _meter_provider, _client_id
    global _command_counter, _duration_histogram

    try:
        _client_id = client_id

        resource = Resource.create(
            _sanitize(
                {
                    "service.name": _SERVICE_NAME,
                    "service.version": version,
                    "switch_trust.client_id": client_id,
                    "python.version": platform.python_version(),
                    "os.type": sys.platform,
                    "os.version": platform.release(),
                },
                _ALLOWED_RESOURCE_ATTRIBUTES,
            )
        )

        _logger_provider = LoggerProvider(resource=resource)
        log_exporter = OTLPLogExporter(endpoint=_BASE_ENDPOINT + "/v1/logs")
        _logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))

        metric_exporter = OTLPMetricExporter(
            endpoint=_BASE_ENDPOINT + "/v1/metrics",
            preferred_temporality=_DELTA_TEMPORALITY,
            timeout=10,
        )
        metric_reader = PeriodicExportingMetricReader(
            metric_exporter,
            export_interval_millis=600_000,
        )
        _meter_provider = MeterProvider(
            metric_readers=[metric_reader],
            resource=resource,
        )
        meter = _meter_provider.get_meter(_SERVICE_NAME)

        _command_counter = meter.create_counter(
            "switch_trust.cli.command",
            unit="1",
            description="Switch Trust CLI commands invoked",
        )
        _duration_histogram = meter.create_histogram(
            "switch_trust.cli.command.duration",
            unit="ms",
            description="Wall-clock duration of a Switch Trust CLI command",
        )
        _tracer_provider = TracerProvider(resource=resource)
        _tracer_provider.add_span_processor(_SpanToLogProcessor(_logger_provider))

        atexit.register(shutdown_telemetry)
    except Exception:
        global_logger.debug("Failed to initialize telemetry", exc_info=True)
        _tracer_provider = None
        _logger_provider = None
        _meter_provider = None
        _command_counter = None
        _duration_histogram = None


@contextmanager
def command_span(
    command: str,
    subcommand: str | None = None,
    is_ci: bool = False,
    is_first_time: bool = False,
) -> Generator[Span | None]:
    if _tracer_provider is None:
        yield None
        return

    try:
        tracer = _tracer_provider.get_tracer(_SERVICE_NAME)
        span = tracer.start_span(f"cli.{command}")
        span.set_attribute("switch_trust.command", command)
        if subcommand:
            span.set_attribute("switch_trust.subcommand", subcommand)
        span.set_attribute("switch_trust.is_ci", is_ci)
        span.set_attribute("switch_trust.is_first_time", is_first_time)
        if _client_id:
            span.set_attribute("switch_trust.client_id", _client_id)
    except Exception:
        global_logger.debug("Failed to create telemetry span", exc_info=True)
        yield None
        return

    try:
        yield span
    finally:
        try:
            span.end()
        except Exception:
            global_logger.debug("Failed to end telemetry span", exc_info=True)


def emit_event(command: str, is_ci: bool = False, is_first_time: bool = False) -> None:
    """Emit a log record directly, without a span. Used for init on first run."""
    if _logger_provider is None:
        return
    try:
        uid = _client_id or "unknown"
        body = f"CLI: {uid} called {command} - OK"
        attrs: dict[str, str | bool] = {
            "switch_trust.command": command,
            "switch_trust.is_ci": is_ci,
            "switch_trust.is_first_time": is_first_time,
            # This path always represents a successful invocation. Set both the
            # prefixed and root status attributes to match on_end.
            "switch_trust.status": "ok",
            "status": "ok",
        }
        if _client_id:
            attrs["switch_trust.client_id"] = _client_id

        logger = _logger_provider.get_logger(_SERVICE_NAME)
        logger.emit(
            severity_number=SeverityNumber.INFO,
            severity_text="INFO",
            body=body,
            attributes=_sanitize(attrs, _ALLOWED_LOG_ATTRIBUTES),
            # Routes this record to Amplitude via the OTel->Amplitude bridge.
            event_name=_EVENT_NAME,
        )

        if _command_counter:
            metric_attrs: dict[str, str] = {
                "switch_trust.command": command,
                "switch_trust.status": "ok",
            }
            if _client_id:
                metric_attrs["switch_trust.client_id"] = _client_id
            _command_counter.add(1, _sanitize(metric_attrs, _ALLOWED_METRIC_ATTRIBUTES))
    except Exception:
        global_logger.debug("Failed to emit telemetry event", exc_info=True)


def record_error(span: Span | None, error: Exception) -> None:
    if span is None:
        return
    try:
        span.set_status(trace.StatusCode.ERROR, type(error).__name__)
        # Only the bounded exception.type (class name) is recorded. The
        # exception message (str(error)) is deliberately NOT set: it routinely
        # embeds PII — file paths, argument values, prompts, provider output —
        # so we never capture it. The stack trace is omitted for the same
        # reason. exception.message is also excluded from _ALLOWED_LOG_ATTRIBUTES
        # so any future attempt to attach it is stripped at the emit boundary.
        span.set_attribute("exception.type", type(error).__name__)
    except Exception:
        global_logger.debug("Failed to record error on telemetry span", exc_info=True)


def record_interruption(span: Span | None) -> None:
    if span is None:
        return
    try:
        span.set_attribute("switch_trust.interrupted", True)
    except Exception:
        global_logger.debug(
            "Failed to record interruption on telemetry span", exc_info=True
        )


def shutdown_telemetry() -> None:
    global _tracer_provider, _logger_provider, _meter_provider

    tracer, _tracer_provider = _tracer_provider, None
    logger, _logger_provider = _logger_provider, None
    meter, _meter_provider = _meter_provider, None

    try:
        if tracer is not None and hasattr(tracer, "shutdown"):
            tracer.shutdown()
    except Exception:
        global_logger.debug("Failed to shut down tracer provider", exc_info=True)

    try:
        if logger is not None and hasattr(logger, "shutdown"):
            logger.shutdown()
    except Exception:
        global_logger.debug("Failed to shut down logger provider", exc_info=True)

    try:
        if meter is not None and hasattr(meter, "shutdown"):
            meter.shutdown()
    except Exception:
        global_logger.debug("Failed to shut down meter provider", exc_info=True)
