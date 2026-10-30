"""Exit 22: the user's target — a config file or a referenced data file —
is missing, unreadable, or not something the CLI can open.

`scan`'s path refusals already carry this code (`test_scan_cli.py`,
`test_file_filter.py`); this file covers the two `eval` producers: the
`--config` file itself (and anything it references eagerly, like a detector's
`prompt_file`), and a message collection's CSV file, which is resolved lazily
at run time rather than at config-load time.
"""

import argparse
from unittest.mock import MagicMock

import pytest

from switch_trust.cli import main, runner
from switch_trust.cli.exit_codes import CliExit, ExitCode


class TestEvalConfigTarget:
    def test_a_missing_config_file_is_a_target_error(self, tmp_path):
        args = argparse.Namespace(
            command="eval",
            config=str(tmp_path / "nope.json"),
        )

        with pytest.raises(CliExit) as exc_info:
            main._dispatch_eval(args)

        assert exc_info.value.code == ExitCode.TARGET_ERROR
        assert "nope.json" in exc_info.value.message

    def test_a_missing_prompt_file_referenced_by_config_is_a_target_error(
        self, tmp_path
    ):
        """`prompt_file` is resolved eagerly, inside the same JsonRepository call."""
        config = tmp_path / "config.json"
        config.write_text(
            '{"detectors": [{"id": "det-1", "name": "d", '
            '"prompt_file": "missing_prompt.txt"}]}'
        )
        args = argparse.Namespace(command="eval", config=str(config))

        with pytest.raises(CliExit) as exc_info:
            main._dispatch_eval(args)

        assert exc_info.value.code == ExitCode.TARGET_ERROR
        assert "missing_prompt.txt" in exc_info.value.message


class TestEvalMessageCollectionTarget:
    """`run_cli_evaluation` resolves a CSV message collection's file lazily,
    one evaluation at a time — a missing/unreadable file is the one thing
    discovered there that is not a per-prompt run failure."""

    def _store(self, *, get_evaluation_error: OSError):
        store = MagicMock()
        store.evaluations.get.return_value = MagicMock(name="an evaluation")
        store.models.get.return_value = MagicMock()
        store.evaluations.get_evaluation.side_effect = get_evaluation_error
        return store

    @pytest.mark.asyncio
    async def test_a_missing_csv_file_is_a_target_error(self):
        missing = "/no/such/prompts.csv"
        store = self._store(
            get_evaluation_error=FileNotFoundError(
                2, "No such file or directory", missing
            )
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

        assert exc_info.value.code == ExitCode.TARGET_ERROR
        assert missing in exc_info.value.message

    @pytest.mark.asyncio
    async def test_an_unreadable_csv_file_is_a_target_error(self):
        unreadable = "/no/permission/prompts.csv"
        store = self._store(
            get_evaluation_error=PermissionError(13, "Permission denied", unreadable)
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

        assert exc_info.value.code == ExitCode.TARGET_ERROR
        assert unreadable in exc_info.value.message

    @pytest.mark.asyncio
    async def test_a_column_mismatch_is_not_a_target_error(self):
        """The file was fine; its shape is the problem — out of scope for 22."""
        store = self._store(get_evaluation_error=OSError())
        store.evaluations.get_evaluation.side_effect = ValueError(
            "Column 'prompt' not found in 'prompts.csv'. Available: ['text']"
        )
        model_evaluation = MagicMock(evaluation_id="ev-1", model_id="model-1")

        with pytest.raises(ValueError):
            await runner.run_cli_evaluation(
                model_evaluation,
                store,
                concurrency=1,
                progress=MagicMock(),
                desc_width=20,
            )
