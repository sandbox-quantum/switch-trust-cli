"""Exit 60: a connection refusal or unresolvable host — retry it.

`eval run` and `switch-trust scan` detect this differently, so — like
`test_auth_failed.py` — the two are tested separately rather than through a
shared harness that would hide that.

`eval` reads it off `ErrorKind.UNREACHABLE`'s dominant status (no HTTP status
at all — a bare connection refusal, or the `host_not_found` sentinel for an
NXDOMAIN); see the comment above `eval_cli._check_model_credentials` and
`test_wrong_url.py` for the 404 sub-case this is split from. `scan` reads it
off the reasoner's `error_reason == "network"` marker, set via
`exception_classification.is_network_error` — see `test_auth_failed.py` for
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
    def test_a_dominant_connection_refusal_is_sixty(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.UNREACHABLE, None)),
                "out.json",
            )

        assert exc_info.value.code == ExitCode.NETWORK_FAILURE
        assert exc_info.value.code.is_retryable

    def test_a_dominant_host_not_found_is_sixty(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.UNREACHABLE, "host_not_found")),
                "out.json",
            )

        assert exc_info.value.code == ExitCode.NETWORK_FAILURE

    def test_the_message_is_actionable_and_counts_the_failures(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator(
                    (ErrorKind.UNREACHABLE, None), (ErrorKind.UNREACHABLE, None)
                ),
                "out.json",
            )

        message = exc_info.value.message
        assert "2 model calls" in message

    def test_the_report_path_rides_along(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.UNREACHABLE, None)),
                "eval_20260101T000000.json",
            )

        assert exc_info.value.output_path == "eval_20260101T000000.json"

    def test_a_dominant_5xx_does_not_borrow_sixty(self):
        """503 is UPSTREAM_ERROR (63), not this — see `test_upstream_error.py`."""
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.UNREACHABLE, 503)),
                "out.json",
            )

        assert exc_info.value.code == ExitCode.UPSTREAM_ERROR

    def test_network_failure_wins_when_it_dominates_a_mixed_unreachable_run(self):
        accumulator = _accumulator(
            (ErrorKind.UNREACHABLE, None),
            (ErrorKind.UNREACHABLE, None),
            (ErrorKind.UNREACHABLE, 404),
        )

        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(accumulator, "out.json")

        assert exc_info.value.code == ExitCode.NETWORK_FAILURE

    def test_404_wins_when_it_dominates_a_mixed_unreachable_run(self):
        accumulator = _accumulator(
            (ErrorKind.UNREACHABLE, 404),
            (ErrorKind.UNREACHABLE, 404),
            (ErrorKind.UNREACHABLE, None),
        )

        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(accumulator, "out.json")

        assert exc_info.value.code == ExitCode.WRONG_URL

    def test_it_is_retryable_unlike_the_fix_config_band(self):
        assert ExitCode.NETWORK_FAILURE.is_retryable
        assert not ExitCode.WRONG_URL.is_retryable


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
    def test_an_unreachable_provider_is_sixty(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(_report("network"), "scan.json")

        assert exc_info.value.code == ExitCode.NETWORK_FAILURE
        assert exc_info.value.code.is_retryable

    def test_the_message_says_the_findings_are_partial(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(_report("network"), "scan.json")

        assert "static-analysis only" in exc_info.value.message

    def test_the_report_path_rides_along(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(
                _report("network"),
                "scan_20260101T000000.json",
            )

        assert exc_info.value.output_path == "scan_20260101T000000.json"
