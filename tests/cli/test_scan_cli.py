"""`scan`'s refusals carry the exit code a caller can act on.

Both cases below used to raise a bare `FileNotFoundError` / `ValueError` and
land on the catch-all 1, which is indistinguishable from the scanner crashing.
They are different problems with different fixes, so they get different codes.
"""

import argparse
import os

import pytest

from switch_trust.cli.scan_cli import handle_scan
from switch_trust.cli.exit_codes import CliExit, ExitCode


def _args(path) -> argparse.Namespace:
    return argparse.Namespace(path=str(path), output=None, format="json")


class TestScanTargetRefusals:
    def test_a_missing_path_is_a_target_error(self, tmp_path):
        with pytest.raises(CliExit) as exc_info:
            handle_scan(_args(tmp_path / "nope"))

        assert exc_info.value.code == ExitCode.TARGET_ERROR
        assert "nope" in exc_info.value.message

    def test_an_unsupported_path_is_a_target_error(self, tmp_path):
        """Readable, present, and not something a scanner can open."""
        fifo = tmp_path / "fifo"
        os.mkfifo(fifo)

        with pytest.raises(CliExit) as exc_info:
            handle_scan(_args(fifo))

        assert exc_info.value.code == ExitCode.TARGET_ERROR

    def test_a_directory_with_nothing_to_scan_is_not_a_target_error(self, tmp_path):
        """The path was fine; there is simply nothing here to analyse.

        This is the CHOO-3479 shape — the run did no work, and before the exit
        codes landed it was reported the same way as a crash.
        """
        (tmp_path / "notes.md").write_text("no code here\n")

        with pytest.raises(CliExit) as exc_info:
            handle_scan(_args(tmp_path))

        assert exc_info.value.code == ExitCode.NO_TARGETS

    @pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses permission bits")
    def test_an_unreadable_directory_is_a_target_error(self, tmp_path):
        """Unreadable is not the same as empty — see file_filter's onerror."""
        os.chmod(tmp_path, 0o000)
        try:
            with pytest.raises(CliExit) as exc_info:
                handle_scan(_args(tmp_path))
        finally:
            os.chmod(tmp_path, 0o755)

        assert exc_info.value.code == ExitCode.TARGET_ERROR

    @pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses permission bits")
    def test_an_unreadable_file_is_a_target_error(self, tmp_path):
        """A file that passes the walk but can't actually be opened for the scan."""
        target = tmp_path / "agent.py"
        target.write_text("import openai\n")
        os.chmod(target, 0o000)
        try:
            with pytest.raises(CliExit) as exc_info:
                handle_scan(_args(target))
        finally:
            os.chmod(target, 0o644)

        assert exc_info.value.code == ExitCode.TARGET_ERROR
