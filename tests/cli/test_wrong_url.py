"""Exit 24: a 404 from the model endpoint — the configured URL is wrong.

`ErrorKind.UNREACHABLE` collapses a 404 (fix the URL) with a 5xx (retry it,
exit 63 — see `test_upstream_error.py`), a bare connection refusal, and an
unresolvable host (both: retry it, exit 60 — see `test_network_failure.py`)
into one kind, so only the unambiguous 404 sub-case gets a code here.

`eval run` is the only producer: the scan path's LLM call has no
user-configurable endpoint (fixed-provider SDKs only), so a 404 there is
always "unknown model", not "wrong URL" — see
`inventory/common/exception_classification.py::is_model_error`.
"""

import pytest

from switch_trust.cli import eval_cli
from switch_trust.cli.exit_codes import CliExit, ExitCode
from switch_trust.eval.core.models.model import ModelErrorAccumulator
from switch_trust.eval.core.models.model_errors import ErrorKind


def _accumulator(
    *kinds_and_statuses: tuple[ErrorKind, object],
) -> ModelErrorAccumulator:
    """An accumulator carrying one recorded failure per (kind, status) given.

    See `test_auth_failed.py::_accumulator` for why the exceptions are built
    up front rather than inline.
    """
    accumulator = ModelErrorAccumulator()
    failures = [
        Exception(f"failure {index}") for index in range(len(kinds_and_statuses))
    ]
    for (kind, status), exc in zip(kinds_and_statuses, failures, strict=True):
        accumulator.record_once(exc, "openai", status, None, kind)
    return accumulator


class TestEvalEndpoint:
    def test_a_dominant_404_is_twenty_four(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.UNREACHABLE, 404)),
                "out.json",
            )

        assert exc_info.value.code == ExitCode.WRONG_URL
        assert exc_info.value.code.needs_intervention

    def test_the_message_is_actionable_and_counts_the_failures(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator(
                    (ErrorKind.UNREACHABLE, 404), (ErrorKind.UNREACHABLE, 404)
                ),
                "out.json",
            )

        message = exc_info.value.message
        assert "2 model calls" in message
        assert "URL" in message

    def test_the_report_path_rides_along(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.UNREACHABLE, 404)),
                "eval_20260101T000000.json",
            )

        assert exc_info.value.output_path == "eval_20260101T000000.json"

    def test_a_dominant_5xx_does_not_trip(self):
        """503 is UPSTREAM_ERROR (63), not this — see `test_upstream_error.py`."""
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.UNREACHABLE, 503)),
                "out.json",
            )

        assert exc_info.value.code != ExitCode.WRONG_URL

    def test_a_dominant_connection_refusal_is_not_wrong_url(self):
        """No HTTP status at all — not a 404. It is exit 60 instead, see
        `test_network_failure.py`."""
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.UNREACHABLE, None)),
                "out.json",
            )

        assert exc_info.value.code != ExitCode.WRONG_URL

    def test_a_dominant_host_not_found_is_not_wrong_url(self):
        """NXDOMAIN carries the `host_not_found` sentinel, never `"404"`.
        It is exit 60 instead, see `test_network_failure.py`."""
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.UNREACHABLE, "host_not_found")),
                "out.json",
            )

        assert exc_info.value.code != ExitCode.WRONG_URL

    def test_404_wins_when_it_dominates_a_mixed_unreachable_run(self):
        accumulator = _accumulator(
            (ErrorKind.UNREACHABLE, 404),
            (ErrorKind.UNREACHABLE, 404),
            (ErrorKind.UNREACHABLE, 503),
        )

        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(accumulator, "out.json")

        assert exc_info.value.code == ExitCode.WRONG_URL

    def test_5xx_wins_when_it_dominates_a_mixed_unreachable_run(self):
        accumulator = _accumulator(
            (ErrorKind.UNREACHABLE, 503),
            (ErrorKind.UNREACHABLE, 503),
            (ErrorKind.UNREACHABLE, 404),
        )

        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(accumulator, "out.json")

        assert exc_info.value.code == ExitCode.UPSTREAM_ERROR

    def test_it_outranks_the_fail_under_gate(self):
        assert ExitCode.WRONG_URL.needs_intervention
        assert not ExitCode.FINDINGS_THRESHOLD.needs_intervention

    def test_auth_and_unreachable_do_not_interfere(self):
        """The two splits key off different kinds; a 404-auth collision can't happen."""
        accumulator = _accumulator(
            (ErrorKind.UNREACHABLE, 404),
            (ErrorKind.UNREACHABLE, 404),
            (ErrorKind.AUTH_FAILED, 403),
        )

        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(accumulator, "out.json")

        assert exc_info.value.code == ExitCode.WRONG_URL
