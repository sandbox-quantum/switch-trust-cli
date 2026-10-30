"""Exit 25: the config was found and read, but isn't usable as it stands.

Three independent producers, each already classifying the failure for its own
reasons before this ticket — wiring 25 is reading markers that already exist:

* `eval run`'s model-call failures: `ErrorKind.INVALID_CONFIG` (any residual
  4xx/`INVALID_ARGUMENT`/`FAILED_PRECONDITION` not already claimed by auth,
  timeout, quota or unreachable) maps directly, unlike `UNREACHABLE` — every
  status it covers means the same thing, so there's no sub-split to get wrong.
* `eval`'s `--config` file: parses and reads fine as a *file*, but its content
  doesn't parse as JSON or doesn't match the config schema.
* `scan`'s `GENERATOR_MODEL`: parses to *something* regardless of shape (no
  validation at construction time), so a typo'd/unsupported model id is only
  discovered once the provider rejects it mid-scan.
"""

import argparse
import json
from types import SimpleNamespace

import pytest

from switch_trust.cli import eval_cli, main, scan_cli
from switch_trust.cli.exit_codes import CliExit, ExitCode
from switch_trust.eval.core.models.model import ModelErrorAccumulator
from switch_trust.eval.core.models.model_errors import ErrorKind


def _accumulator(
    *kinds_and_statuses: tuple[ErrorKind, object],
) -> ModelErrorAccumulator:
    """See `test_auth_failed.py::_accumulator`."""
    accumulator = ModelErrorAccumulator()
    failures = [
        Exception(f"failure {index}") for index in range(len(kinds_and_statuses))
    ]
    for (kind, status), exc in zip(kinds_and_statuses, failures, strict=True):
        accumulator.record_once(exc, "openai", status, None, kind)
    return accumulator


class TestEvalModelCalls:
    def test_a_dominant_invalid_config_is_twenty_five(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.INVALID_CONFIG, 400)),
                "out.json",
            )

        assert exc_info.value.code == ExitCode.INVALID_CONFIG
        assert exc_info.value.code.needs_intervention

    def test_it_maps_regardless_of_the_specific_status(self):
        """Unlike UNREACHABLE, every status under this kind means the same thing."""
        for status in (400, 409, 422, None):
            with pytest.raises(CliExit) as exc_info:
                eval_cli._check_model_credentials(
                    _accumulator((ErrorKind.INVALID_CONFIG, status)),
                    "out.json",
                )
            assert exc_info.value.code == ExitCode.INVALID_CONFIG

    def test_the_message_counts_the_failures(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator(
                    (ErrorKind.INVALID_CONFIG, 400), (ErrorKind.INVALID_CONFIG, 400)
                ),
                "out.json",
            )

        assert "2 model calls" in exc_info.value.message

    def test_the_report_path_rides_along(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator((ErrorKind.INVALID_CONFIG, 400)),
                "eval_20260101T000000.json",
            )

        assert exc_info.value.output_path == "eval_20260101T000000.json"

    def test_it_outranks_the_fail_under_gate(self):
        assert ExitCode.INVALID_CONFIG.needs_intervention
        assert not ExitCode.FINDINGS_THRESHOLD.needs_intervention


class TestEvalConfigFile:
    def test_malformed_json_is_a_config_error(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text("{not valid json")
        args = argparse.Namespace(command="eval", config=str(config))

        with pytest.raises(CliExit) as exc_info:
            main._dispatch_eval(args)

        assert exc_info.value.code == ExitCode.INVALID_CONFIG
        assert str(config) in exc_info.value.message

    def test_an_unrecognized_evaluation_type_is_a_config_error(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text(
            json.dumps(
                {
                    "evaluations": [
                        {"id": "ev-1", "name": "e", "type": "not_a_real_type"}
                    ]
                }
            )
        )
        args = argparse.Namespace(command="eval", config=str(config))

        with pytest.raises(CliExit) as exc_info:
            main._dispatch_eval(args)

        assert exc_info.value.code == ExitCode.INVALID_CONFIG

    def test_a_model_missing_a_required_field_is_a_config_error(self, tmp_path):
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"models": [{"type": "openai"}]}))
        args = argparse.Namespace(command="eval", config=str(config))

        with pytest.raises(CliExit) as exc_info:
            main._dispatch_eval(args)

        assert exc_info.value.code == ExitCode.INVALID_CONFIG

    def test_a_missing_config_file_stays_a_target_error(self, tmp_path):
        """22 and 25 are different fixes; a bad path must not become 25."""
        args = argparse.Namespace(command="eval", config=str(tmp_path / "nope.json"))

        with pytest.raises(CliExit) as exc_info:
            main._dispatch_eval(args)

        assert exc_info.value.code == ExitCode.TARGET_ERROR


def _report(error_reason: str | None = None):
    agentic_trace = {}
    if error_reason:
        agentic_trace = {"exit_reason": "error", "error_reason": error_reason}
    return SimpleNamespace(
        findings=[],
        pre_triage_findings=None,
        category_summary={},
        framework_detected="openai",
        scan_metadata={"agentic_trace": agentic_trace},
    )


class TestScanModel:
    def test_an_invalid_model_is_twenty_five(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(_report("model"), "scan.json")

        assert exc_info.value.code == ExitCode.INVALID_CONFIG

    def test_the_message_says_the_findings_are_partial(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(_report("model"), "scan.json")

        assert "static-analysis only" in exc_info.value.message
        assert "GENERATOR_MODEL" in exc_info.value.message

    def test_the_report_path_rides_along(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(
                _report("model"),
                "scan_20260101T000000.json",
            )

        assert exc_info.value.output_path == "scan_20260101T000000.json"

    def test_a_rejected_key_stays_twenty(self):
        """`model` is the carve-out; a rejected key is unaffected."""
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(_report("auth"), "scan.json")

        assert exc_info.value.code == ExitCode.AUTH_FAILED
