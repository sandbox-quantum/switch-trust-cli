"""Exit 62: a provider call timing out — retry it.

`eval run` and `switch-trust scan` detect this differently, so — like
`test_rate_limited.py` — the two are tested separately rather than through a
shared harness that would hide that.

`eval` reads it off `ErrorKind.TIMEOUT`, mapped directly in
`eval_cli._EXIT_FOR_ERROR_KIND` — unlike `UNREACHABLE`/`QUOTA_EXCEEDED`, every
status `classify_model_error` sorts into `TIMEOUT` (a bare `TimeoutError`, a
408/504, or google.rpc's `DEADLINE_EXCEEDED`) means the same thing, so no
status sub-split is needed. `scan` reads it off the reasoner's
`error_reason == "timeout"` marker, set via
`exception_classification.is_timeout_error` — see `test_auth_failed.py` for
the "auth"/"model" markers this is alongside.
"""

from types import SimpleNamespace

import pytest

from switch_trust.cli import eval_cli, scan_cli
from switch_trust.cli.exit_codes import CliExit, ExitCode
from switch_trust.eval.core.models.model import ModelErrorAccumulator
from switch_trust.eval.core.models.model_errors import ErrorKind


def _accumulator(
    *kinds_and_statuses: tuple[ErrorKind, object],
) -> ModelErrorAccumulator:
    """See `test_wrong_url.py::_accumulator`."""
    accumulator = ModelErrorAccumulator()
    failures = [
        Exception(f"failure {index}") for index in range(len(kinds_and_statuses))
    ]
    for (kind, status), exc in zip(kinds_and_statuses, failures, strict=True):
        accumulator.record_once(exc, "openai", status, None, kind)
    return accumulator


class TestEvalEndpoint:
    def test_a_dominant_timeout_is_sixty_two(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.TIMEOUT, None)),
                "out.json",
            )

        assert exc_info.value.code == ExitCode.TIMEOUT
        assert exc_info.value.code.is_retryable

    def test_a_dominant_504_is_sixty_two(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.TIMEOUT, 504)),
                "out.json",
            )

        assert exc_info.value.code == ExitCode.TIMEOUT

    def test_the_message_is_actionable_and_counts_the_failures(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator(
                    (ErrorKind.TIMEOUT, None),
                    (ErrorKind.TIMEOUT, None),
                ),
                "out.json",
            )

        message = exc_info.value.message
        assert "2 model calls" in message

    def test_the_report_path_rides_along(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.TIMEOUT, None)),
                "eval_20260101T000000.json",
            )

        assert exc_info.value.output_path == "eval_20260101T000000.json"

    def test_it_is_retryable_unlike_the_fix_config_band(self):
        assert ExitCode.TIMEOUT.is_retryable


def _report(error_reason: str | None = None, findings: list | None = None):
    agentic_trace = {}
    if error_reason:
        agentic_trace = {"exit_reason": "error", "error_reason": error_reason}
    return SimpleNamespace(
        findings=findings or [],
        pre_triage_findings=None,
        category_summary={},
        framework_detected="openai",
        scan_metadata={"agentic_trace": agentic_trace},
    )


class TestScanEndpoint:
    def test_a_timed_out_provider_is_sixty_two(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(_report("timeout"), "scan.json")

        assert exc_info.value.code == ExitCode.TIMEOUT
        assert exc_info.value.code.is_retryable

    def test_the_message_says_the_findings_are_partial(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(_report("timeout"), "scan.json")

        assert "static-analysis only" in exc_info.value.message

    def test_the_report_path_rides_along(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(
                _report("timeout"),
                "scan_20260101T000000.json",
            )

        assert exc_info.value.output_path == "scan_20260101T000000.json"
