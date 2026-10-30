"""Exit 21: a credential the provider accepted but refused as unpermitted.

A 403 is not a 401. Both used to collapse into the same `auth_failed`
classification and the same exit 20, telling the user their key was wrong when
the real problem is a plan/quota/scope limit on an otherwise-valid credential.
The two commands detect it the same way exit 20 does — differently from each
other — so they are tested separately, mirroring `test_auth_failed.py`.
"""

import argparse
from types import SimpleNamespace

import pytest

from switch_trust.cli import eval_cli, scan_cli
from switch_trust.cli.exit_codes import CliExit, ExitCode
from switch_trust.eval.core.models.model import ModelErrorAccumulator
from switch_trust.eval.core.models.model_errors import ErrorKind


def _accumulator(*kinds_and_statuses: tuple[ErrorKind, int]) -> ModelErrorAccumulator:
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


class TestEvalCredentials:
    def test_a_dominant_403_is_twenty_one(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.AUTH_FAILED, 403)),
                "out.json",
            )

        assert exc_info.value.code == ExitCode.NOT_ENTITLED
        assert exc_info.value.code.needs_intervention

    def test_the_message_is_actionable_and_counts_the_failures(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator(
                    (ErrorKind.AUTH_FAILED, 403), (ErrorKind.AUTH_FAILED, 403)
                ),
                "out.json",
            )

        message = exc_info.value.message
        assert "2 model calls" in message
        assert "plan/quota" in message

    def test_the_report_path_rides_along(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.AUTH_FAILED, 403)),
                "eval_20260101T000000.json",
            )

        assert exc_info.value.output_path == "eval_20260101T000000.json"

    def test_a_dominant_401_stays_twenty(self):
        """403 is the carve-out; a plain rejected key is unaffected."""
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.AUTH_FAILED, 401)),
                "out.json",
            )

        assert exc_info.value.code == ExitCode.AUTH_FAILED

    def test_403_wins_when_it_dominates_a_mixed_auth_run(self):
        accumulator = _accumulator(
            (ErrorKind.AUTH_FAILED, 403),
            (ErrorKind.AUTH_FAILED, 403),
            (ErrorKind.AUTH_FAILED, 401),
        )

        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(accumulator, "out.json")

        assert exc_info.value.code == ExitCode.NOT_ENTITLED

    def test_401_wins_when_it_dominates_a_mixed_auth_run(self):
        accumulator = _accumulator(
            (ErrorKind.AUTH_FAILED, 401),
            (ErrorKind.AUTH_FAILED, 401),
            (ErrorKind.AUTH_FAILED, 403),
        )

        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(accumulator, "out.json")

        assert exc_info.value.code == ExitCode.AUTH_FAILED

    def test_it_outranks_the_fail_under_gate(self):
        assert ExitCode.NOT_ENTITLED.needs_intervention
        assert not ExitCode.FINDINGS_THRESHOLD.needs_intervention


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


class TestScanCredentials:
    def test_a_refused_key_is_twenty_one(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(_report("not_entitled"), "scan.json")

        assert exc_info.value.code == ExitCode.NOT_ENTITLED

    def test_the_message_says_the_findings_are_partial(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(_report("not_entitled"), "scan.json")

        assert "static-analysis only" in exc_info.value.message
        assert "plan/quota" in exc_info.value.message

    def test_the_report_path_rides_along(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(
                _report("not_entitled"),
                "scan_20260101T000000.json",
            )

        assert exc_info.value.output_path == "scan_20260101T000000.json"

    def test_a_rejected_key_stays_twenty(self):
        """`not_entitled` is the carve-out; a plain rejected key is unaffected."""
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(_report("auth"), "scan.json")

        assert exc_info.value.code == ExitCode.AUTH_FAILED


@pytest.fixture
def stubbed_scan(monkeypatch, tmp_path):
    """Mirrors `test_auth_failed.py::stubbed_scan` for the 403 path."""

    def _configure(report, *, fail_on=None, written=None):
        target = tmp_path / "agent.py"
        target.write_text("import openai\n")

        def _run_core(*args, **kwargs):
            return report

        monkeypatch.setattr(
            "switch_trust.scan.agent_scanner.run_core",
            _run_core,
        )
        monkeypatch.setattr(scan_cli, "print_report", lambda _: None)

        def _write(report_, output_path, fmt="json"):
            with open(output_path, "w") as handle:
                handle.write("{}")
            if written is not None:
                written.append(output_path)

        monkeypatch.setattr(scan_cli, "write_output", _write)
        monkeypatch.delenv("GENERATOR_MODEL", raising=False)

        args = argparse.Namespace(
            path=str(target),
            output=str(tmp_path / "scan.json"),
            format="json",
            fail_on=fail_on,
        )
        return args

    return _configure


class TestHandleScanEndToEnd:
    def test_a_refused_key_raises_twenty_one_and_keeps_the_report(
        self,
        stubbed_scan,
        tmp_path,
    ):
        written: list[str] = []
        args = stubbed_scan(_report("not_entitled"), written=written)

        with pytest.raises(CliExit) as exc_info:
            scan_cli.handle_scan(args)

        assert exc_info.value.code == ExitCode.NOT_ENTITLED
        assert written == [args.output]
        assert (tmp_path / "scan.json").exists()
