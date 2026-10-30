"""Exit 64: a real HTTP 429 the provider answered with — retry it.

`eval run` and `switch-trust scan` detect this differently, so — like
`test_network_failure.py` — the two are tested separately rather than through
a shared harness that would hide that.

`eval` reads it off `ErrorKind.QUOTA_EXCEEDED`'s dominant status: a literal
`"429"` is this; anything else — most notably Gemini's statusless
`RESOURCE_EXHAUSTED` — is the less specific exit 61 (`test_rate_limited.py`);
see the comment above `eval_cli._check_model_credentials` and
`_RESOURCE_EXHAUSTED_STATUS`. `scan` reads it off the reasoner's
`error_reason == "resource_exhausted"` marker, set via
`exception_classification.is_resource_exhausted` — see `test_auth_failed.py`
for the "auth"/"model" markers this is alongside.
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
    def test_a_dominant_literal_429_is_resource_exhausted(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.QUOTA_EXCEEDED, 429)),
                "out.json",
            )

        assert exc_info.value.code == ExitCode.RESOURCE_EXHAUSTED
        assert exc_info.value.code.is_retryable

    def test_a_statusless_quota_error_does_not_borrow_sixty_four(self):
        """No HTTP status at all is the less specific exit 61 instead."""
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.QUOTA_EXCEEDED, None)),
                "out.json",
            )

        assert exc_info.value.code == ExitCode.RATE_LIMITED

    def test_the_message_is_actionable_and_counts_the_failures(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator(
                    (ErrorKind.QUOTA_EXCEEDED, 429),
                    (ErrorKind.QUOTA_EXCEEDED, 429),
                ),
                "out.json",
            )

        message = exc_info.value.message
        assert "2 model calls" in message
        assert "429" in message

    def test_the_report_path_rides_along(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.QUOTA_EXCEEDED, 429)),
                "eval_20260101T000000.json",
            )

        assert exc_info.value.output_path == "eval_20260101T000000.json"

    def test_429_wins_when_it_dominates_a_mixed_quota_run(self):
        accumulator = _accumulator(
            (ErrorKind.QUOTA_EXCEEDED, 429),
            (ErrorKind.QUOTA_EXCEEDED, 429),
            (ErrorKind.QUOTA_EXCEEDED, None),
        )

        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(accumulator, "out.json")

        assert exc_info.value.code == ExitCode.RESOURCE_EXHAUSTED

    def test_it_is_retryable_unlike_the_fix_config_band(self):
        assert ExitCode.RESOURCE_EXHAUSTED.is_retryable


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
    def test_a_resource_exhausted_provider_is_sixty_four(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(_report("resource_exhausted"), "scan.json")

        assert exc_info.value.code == ExitCode.RESOURCE_EXHAUSTED
        assert exc_info.value.code.is_retryable

    def test_the_message_says_the_findings_are_partial(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(_report("resource_exhausted"), "scan.json")

        assert "static-analysis only" in exc_info.value.message

    def test_the_report_path_rides_along(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(
                _report("resource_exhausted"),
                "scan_20260101T000000.json",
            )

        assert exc_info.value.output_path == "scan_20260101T000000.json"
