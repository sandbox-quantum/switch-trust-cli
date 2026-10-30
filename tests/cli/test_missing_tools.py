"""Exit 26: a tool the run needed (an optional `full`-extra dependency, or a
scan analysis tool) is not installed.

Two independent producers:

* `eval`: `transformers`/`torch` and `garak` ship behind the optional `full`
  extra (`core/optional_deps.py`) rather than as core dependencies, so a
  model/detector that needs one (`HuggingFaceModel`, `ToxicityDetector`,
  anything garak-backed) raises `MissingToolError` if it is absent.
  `runner.run_cli_evaluation` classifies that at every point it can surface —
  building the evaluation/model, and the lazy garak import during
  `init()`/`run()` — rather than letting it fall into the swallow-and-continue
  `except Exception` that per-prompt run failures use.
* `scan`: Bandit, OpenGrep, detect-secrets and pip-audit are all load-bearing —
  OpenGrep is the only source of agent-specific findings — so `run_static_scan`
  reporting any of them in `tools_skipped` means the scan is incomplete
  (`scan_cli._check_missing_tools`).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from switch_trust.cli import runner, scan_cli
from switch_trust.cli.exit_codes import CliExit, ExitCode
from switch_trust.eval.core.optional_deps import MissingToolError


class TestEvalConstruction:
    """A missing tool discovered while building the evaluation or model —
    e.g. a `HuggingFaceModel` or `ToxicityDetector` importing `transformers`
    eagerly in `__init__` — is a one-time setup failure, not a run failure."""

    @pytest.mark.asyncio
    async def test_a_missing_tool_building_the_evaluation_is_exit_26(self):
        store = MagicMock()
        store.evaluations.get.return_value = MagicMock(name="an evaluation")
        store.evaluations.get_evaluation.side_effect = MissingToolError(
            "garak is required for garak probes and detectors. "
            "Install it with: pip install switch-trust-cli[full]"
        )
        model_evaluation = MagicMock(evaluation_id="ev-1", model_id="model-1")

        with pytest.raises(CliExit) as exc_info:
            await runner.run_cli_evaluation(
                model_evaluation,
                store,
                concurrency=1,
                progress=MagicMock(),
                desc_width=20,
            )

        assert exc_info.value.code == ExitCode.MISSING_TOOLS
        assert exc_info.value.code.needs_intervention
        assert "switch-trust-cli[full]" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_a_missing_tool_building_the_model_is_exit_26(self):
        store = MagicMock()
        store.evaluations.get.return_value = MagicMock(name="an evaluation")
        store.evaluations.get_evaluation.return_value = MagicMock()
        store.models.get_model.side_effect = MissingToolError(
            "transformers is required for HuggingFace models and toxicity "
            "detection. Install it with: pip install switch-trust-cli[full]"
        )
        model_evaluation = MagicMock(evaluation_id="ev-1", model_id="model-1")

        with pytest.raises(CliExit) as exc_info:
            await runner.run_cli_evaluation(
                model_evaluation,
                store,
                concurrency=1,
                progress=MagicMock(),
                desc_width=20,
            )

        assert exc_info.value.code == ExitCode.MISSING_TOOLS
        assert "transformers" in exc_info.value.message


class TestEvalLazyGarakImport:
    """`GarakMessageCollection`/`_ensure_garak_config` import garak lazily,
    during `evaluation.init()` or `evaluation.run()` rather than at
    construction — still a configuration problem, not a per-prompt failure."""

    def _store(self):
        store = MagicMock()
        store.evaluations.get.return_value = MagicMock(name="an evaluation")
        store.models.get_model.return_value = MagicMock()
        return store

    @pytest.mark.asyncio
    async def test_a_missing_tool_during_init_is_exit_26(self):
        store = self._store()
        evaluation = MagicMock()
        evaluation.init = AsyncMock(
            side_effect=MissingToolError(
                "garak is required for garak probes and detectors. "
                "Install it with: pip install switch-trust-cli[full]"
            )
        )
        store.evaluations.get_evaluation.return_value = evaluation
        model_evaluation = MagicMock(evaluation_id="ev-1", model_id="model-1")

        with pytest.raises(CliExit) as exc_info:
            await runner.run_cli_evaluation(
                model_evaluation,
                store,
                concurrency=1,
                progress=MagicMock(),
                desc_width=20,
            )

        assert exc_info.value.code == ExitCode.MISSING_TOOLS

    @pytest.mark.asyncio
    async def test_a_missing_tool_during_run_is_exit_26(self):
        store = self._store()
        evaluation = MagicMock()
        evaluation.init = AsyncMock()
        evaluation.get_summary.return_value = MagicMock(total_evaluations=1)
        evaluation.run = AsyncMock(
            side_effect=MissingToolError(
                "garak is required for garak probes and detectors. "
                "Install it with: pip install switch-trust-cli[full]"
            )
        )
        store.evaluations.get_evaluation.return_value = evaluation
        model_evaluation = MagicMock(evaluation_id="ev-1", model_id="model-1")

        with pytest.raises(CliExit) as exc_info:
            await runner.run_cli_evaluation(
                model_evaluation,
                store,
                concurrency=1,
                progress=MagicMock(),
                desc_width=20,
            )

        assert exc_info.value.code == ExitCode.MISSING_TOOLS

    @pytest.mark.asyncio
    async def test_the_observer_is_still_cleaned_up(self):
        """The `finally` in `run_cli_evaluation` must run even though the
        except clause raises rather than falling through."""
        store = self._store()
        evaluation = MagicMock()
        evaluation.init = AsyncMock(side_effect=MissingToolError("missing"))
        store.evaluations.get_evaluation.return_value = evaluation
        model_evaluation = MagicMock(evaluation_id="ev-1", model_id="model-1")

        with pytest.raises(CliExit):
            await runner.run_cli_evaluation(
                model_evaluation,
                store,
                concurrency=1,
                progress=MagicMock(),
                desc_width=20,
            )

        evaluation.remove_observer.assert_called_once()


def _report(tools_skipped: list[dict] | None = None):
    return SimpleNamespace(
        scan_metadata={"tools_skipped": tools_skipped or []},
    )


class TestScanTools:
    def test_a_missing_bandit_is_exit_26(self):
        report = _report([{"tool": "bandit", "reason": "Bandit is not installed."}])

        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_missing_tools(report, "scan.json")

        assert exc_info.value.code == ExitCode.MISSING_TOOLS
        assert exc_info.value.code.needs_intervention
        assert "bandit" in exc_info.value.message

    def test_a_missing_detect_secrets_is_exit_26(self):
        report = _report(
            [{"tool": "detect-secrets", "reason": "detect-secrets is not installed."}]
        )

        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_missing_tools(report, "scan.json")

        assert exc_info.value.code == ExitCode.MISSING_TOOLS

    def test_a_missing_pip_audit_is_exit_26(self):
        report = _report(
            [{"tool": "pip-audit", "reason": "pip-audit is not installed."}]
        )

        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_missing_tools(report, "scan.json")

        assert exc_info.value.code == ExitCode.MISSING_TOOLS

    def test_a_missing_opengrep_is_exit_26(self):
        """OpenGrep is the only source of agent-specific findings, so its
        absence is treated the same as the other three tools, not excused."""
        report = _report([{"tool": "opengrep", "reason": "OpenGrep binary not found."}])

        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_missing_tools(report, "scan.json")

        assert exc_info.value.code == ExitCode.MISSING_TOOLS
        assert "opengrep" in exc_info.value.message

    def test_no_tools_skipped_does_not_raise(self):
        scan_cli._check_missing_tools(_report([]), "scan.json")  # does not raise

    def test_the_report_path_rides_along(self):
        report = _report([{"tool": "bandit", "reason": "x"}])

        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_missing_tools(report, "scan_20260101T000000.json")

        assert exc_info.value.output_path == "scan_20260101T000000.json"

    def test_multiple_missing_tools_are_all_named(self):
        report = _report(
            [
                {"tool": "bandit", "reason": "x"},
                {"tool": "detect-secrets", "reason": "y"},
            ]
        )

        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_missing_tools(report, "scan.json")

        assert "bandit" in exc_info.value.message
        assert "detect-secrets" in exc_info.value.message
