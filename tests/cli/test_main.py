import argparse
import io
import os
import runpy
import sys
from unittest.mock import MagicMock, patch

import pytest

from switch_trust.cli import eval_cli
from switch_trust.cli import main as main_mod
from switch_trust.cli.main import (
    _dispatch,
    _dispatch_eval,
    _load_environment,
    _print_error,
)
from switch_trust.cli.exit_codes import CliExit, ExitCode


class TestPrintError:
    def test_displays_error_type_and_message(self, capsys):
        error = ValueError("something went wrong")

        _print_error(error)

        captured = capsys.readouterr().out
        assert "ValueError" in captured
        assert "something went wrong" in captured

    def test_empty_message_shows_fallback(self, capsys):
        error = RuntimeError()

        _print_error(error)

        captured = capsys.readouterr().out
        assert "(no message)" in captured


class TestLoadEnvironment:
    """Env loading precedence: real environment > local `.env` > global config.

    The local `.env` in the cwd is discovered and wins over the global config,
    but neither dotfile clobbers a variable already present in `os.environ`
    (e.g. one supplied on the command line as `FOO=bar switch-trust ...`).

    The `.env` discovery is anchored at the cwd (`find_dotenv(usecwd=True)`),
    not at this module's directory — a bare `load_dotenv()` searches from the
    module's location, which is site-packages once installed, so a local `.env`
    would never be found. Each test chdirs into a `project` dir to stand in for
    the user's working directory.

    The global config dir is resolved via `get_switch_trust_dir()`, so it is
    monkeypatched (via the `global_dir` fixture) to a temp dir separate from the
    cwd — otherwise the two `.env` files would collapse onto the same path and
    the override behavior couldn't be observed.

    `os.environ` is snapshotted and restored so loaded vars don't leak between
    tests.
    """

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for key in ("HOOT_GLOBAL_VAR", "HOOT_SHARED_VAR", "HOOT_LOCAL_VAR"):
            monkeypatch.delenv(key, raising=False)

    @pytest.fixture
    def global_dir(self, tmp_path, monkeypatch):
        """A temp stand-in for ``~/.switch-trust``; its ``.env`` is the global file."""
        d = tmp_path / "global"
        d.mkdir()
        monkeypatch.setattr("switch_trust.cli.main.get_switch_trust_dir", lambda: d)
        return d

    @pytest.fixture
    def project(self, tmp_path, monkeypatch):
        """A temp working directory, separate from the global dir."""
        d = tmp_path / "project"
        d.mkdir()
        monkeypatch.chdir(d)
        return d

    def test_global_env_is_loaded(self, global_dir, project):
        (global_dir / ".env").write_text("HOOT_GLOBAL_VAR=from_global\n")

        _load_environment()

        assert os.environ["HOOT_GLOBAL_VAR"] == "from_global"

    def test_missing_global_env_is_tolerated(self, global_dir, project):
        # No .env in the global dir — must not raise.
        _load_environment()

        assert "HOOT_GLOBAL_VAR" not in os.environ

    def test_local_env_in_cwd_is_loaded(self, global_dir, project):
        (project / ".env").write_text("HOOT_LOCAL_VAR=from_local\n")

        _load_environment()

        assert os.environ["HOOT_LOCAL_VAR"] == "from_local"

    def test_local_env_overrides_global(self, global_dir, project):
        (global_dir / ".env").write_text(
            "HOOT_SHARED_VAR=from_global\nHOOT_GLOBAL_VAR=g\n"
        )
        (project / ".env").write_text("HOOT_SHARED_VAR=from_local\n")

        _load_environment()

        # Local wins for the shared key; global-only keys still come through.
        assert os.environ["HOOT_SHARED_VAR"] == "from_local"
        assert os.environ["HOOT_GLOBAL_VAR"] == "g"

    def test_startup_does_not_override_existing_env(
        self, global_dir, project, monkeypatch
    ):
        """On normal startup the global file must not clobber a value already
        set in the environment (e.g. a one-off shell/CI override)."""
        monkeypatch.setenv("HOOT_GLOBAL_VAR", "from_shell")
        (global_dir / ".env").write_text("HOOT_GLOBAL_VAR=from_global\n")

        _load_environment()

        assert os.environ["HOOT_GLOBAL_VAR"] == "from_shell"

    def test_local_env_does_not_override_existing_env(
        self, global_dir, project, monkeypatch
    ):
        """The bugfix: a variable supplied on the command line / shell (already
        in `os.environ`) must win over the local `.env`, not be clobbered by
        it."""
        monkeypatch.setenv("HOOT_LOCAL_VAR", "from_cli")
        (project / ".env").write_text("HOOT_LOCAL_VAR=from_local\n")

        _load_environment()

        assert os.environ["HOOT_LOCAL_VAR"] == "from_cli"

    def test_reload_overrides_existing_env_from_global(
        self, global_dir, project, monkeypatch
    ):
        """After init (re)writes the global file, the reload must pick up the
        freshly written value even if a stale one is already in the environment."""
        monkeypatch.setenv("HOOT_GLOBAL_VAR", "stale")
        (global_dir / ".env").write_text("HOOT_GLOBAL_VAR=fresh\n")

        _load_environment(override=True)

        assert os.environ["HOOT_GLOBAL_VAR"] == "fresh"

    def test_reload_overrides_existing_env_from_local(
        self, global_dir, project, monkeypatch
    ):
        """`init` writes to the local `.env` when one exists in the cwd, so the
        reload must also pick up freshly written values from the local file, not
        only the global one."""
        monkeypatch.setenv("HOOT_LOCAL_VAR", "stale")
        (project / ".env").write_text("HOOT_LOCAL_VAR=fresh\n")

        _load_environment(override=True)

        assert os.environ["HOOT_LOCAL_VAR"] == "fresh"

    def test_local_env_found_from_cwd_not_module_dir(
        self, global_dir, tmp_path, monkeypatch
    ):
        """The bugfix: discovery is anchored at the cwd. A `.env` sitting in an
        arbitrary working directory (unrelated to where the CLI is installed) is
        found."""
        nested = tmp_path / "some" / "nested" / "project"
        nested.mkdir(parents=True)
        (nested / ".env").write_text("HOOT_LOCAL_VAR=from_cwd\n")
        monkeypatch.chdir(nested)

        _load_environment()

        assert os.environ["HOOT_LOCAL_VAR"] == "from_cwd"


class TestMainErrorHandling:
    @patch("switch_trust.cli.main._dispatch", side_effect=RuntimeError("test crash"))
    @patch("switch_trust.cli.main.setup_file_logging")
    @patch("switch_trust.cli.main._print_logo")
    @patch("switch_trust.cli.main._load_environment")
    @patch("switch_trust.cli.main.get_telemetry_consent", return_value=False)
    @patch("switch_trust.cli.main.ensure_telemetry_consent")
    @patch("switch_trust.cli.main.ensure_client_id")
    @patch("switch_trust.cli.main.init_cli")
    def test_exception_prints_error_and_exits(
        self,
        mock_init_cli,
        mock_ensure_client_id,
        mock_ensure_telemetry_consent,
        mock_get_telemetry_consent,
        mock_load_env,
        mock_logo,
        mock_logging,
        mock_dispatch,
        capsys,
        tmp_path,
    ):
        mock_init_cli.get_switch_trust_env_path.return_value = tmp_path / "nonexistent"

        from switch_trust.cli.main import main

        # An exception we never classified is exit 1, as it has always been.
        assert main(["eval", "models", "list"]) == ExitCode.ERROR
        captured = capsys.readouterr().out
        assert "RuntimeError" in captured
        assert "test crash" in captured
        assert "Completed in" in captured


@pytest.fixture
def stubbed_startup(monkeypatch, tmp_path):
    """Neuter everything `main` does before dispatch, so only the exit path is
    under test: logging setup, the logo, env loading, telemetry consent and
    `init`'s first-run detection."""
    monkeypatch.setattr(main_mod, "setup_file_logging", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "_print_logo", lambda: None)
    monkeypatch.setattr(main_mod, "_load_environment", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "get_telemetry_consent", lambda: False)
    monkeypatch.setattr(main_mod, "ensure_telemetry_consent", lambda: None)
    monkeypatch.setattr(main_mod, "ensure_client_id", lambda: None)
    monkeypatch.setattr(main_mod, "is_ci", lambda: False)

    init_cli = MagicMock()
    init_cli.get_switch_trust_env_path.return_value = tmp_path / "nonexistent"
    monkeypatch.setattr(main_mod, "init_cli", init_cli)
    return monkeypatch


class TestExitCodes:
    """`main` returns the code; it never exits. See `exit_codes.py`."""

    def test_a_clean_run_returns_zero(self, stubbed_startup):
        stubbed_startup.setattr(main_mod, "_dispatch", lambda args, ci: None)

        assert main_mod.main(["eval", "models", "list"]) == ExitCode.OK

    def test_a_cli_exit_returns_its_own_code(self, stubbed_startup, capsys):
        stubbed_startup.setattr(
            main_mod,
            "_dispatch",
            MagicMock(side_effect=CliExit(ExitCode.AUTH_FAILED, "API key missing")),
        )

        assert main_mod.main(["eval", "models", "list"]) == ExitCode.AUTH_FAILED

        # One classification, two renderings: the raiser's own sentence and the
        # code beside it, never a second diagnosis.
        captured = capsys.readouterr().out
        assert "API key missing" in captured
        assert "Exit code: 20" in captured
        assert "CliExit" not in captured

    def test_a_tripped_gate_is_not_rendered_as_an_error(self, stubbed_startup, capsys):
        """Exit 5 means the scan worked and the user's gate fired."""
        stubbed_startup.setattr(
            main_mod,
            "_dispatch",
            MagicMock(
                side_effect=CliExit(
                    ExitCode.FINDINGS_THRESHOLD,
                    "3 findings at or above `high`",
                )
            ),
        )

        assert main_mod.main(["scan", "."]) == ExitCode.FINDINGS_THRESHOLD

        captured = capsys.readouterr().out
        assert "3 findings at or above" in captured
        assert "Error" not in captured

    def test_a_gate_still_reports_the_report_it_wrote(self, stubbed_startup, capsys):
        """A gate trips after its artifact lands, so the footer must name it.

        The exception unwound past `_run`'s own `output_path` assignment, so
        the only record of the file is the one the raiser attached.
        """
        stubbed_startup.setattr(
            main_mod,
            "_dispatch",
            MagicMock(
                side_effect=CliExit(
                    ExitCode.FINDINGS_THRESHOLD,
                    "1 of 2 evaluations scored below 0.80: jailbreak (0.42)",
                    output_path="eval_20260101T000000.json",
                )
            ),
        )

        assert main_mod.main(["eval", "run", "me_1"]) == ExitCode.FINDINGS_THRESHOLD
        assert "eval_20260101T000000.json" in capsys.readouterr().out

    def test_a_refusal_with_no_artifact_reports_none(self, stubbed_startup, capsys):
        """The default stays `None` — a refusal has no report to point at."""
        stubbed_startup.setattr(
            main_mod,
            "_dispatch",
            MagicMock(side_effect=CliExit(ExitCode.TARGET_ERROR, "no such path")),
        )

        assert main_mod.main(["scan", "."]) == ExitCode.TARGET_ERROR
        assert "Results:" not in capsys.readouterr().out

    def test_keyboard_interrupt_returns_130(self, stubbed_startup):
        stubbed_startup.setattr(
            main_mod, "_dispatch", MagicMock(side_effect=KeyboardInterrupt)
        )

        assert main_mod.main(["eval", "models", "list"]) == ExitCode.INTERRUPTED

    def test_an_argparse_error_still_exits_two(self):
        """argparse exits for us, and 2 is the code the table reserves."""
        with pytest.raises(SystemExit) as exc_info:
            main_mod.main(["no-such-command"])

        assert exc_info.value.code == ExitCode.USAGE

    def test_a_malformed_tag_filter_is_a_usage_error(self):
        """A `--tag` without `=` is a bad invocation, so 2 rather than 1."""
        with pytest.raises(CliExit) as exc_info:
            eval_cli._parse_tags(["no-equals-sign"])

        assert exc_info.value.code == ExitCode.USAGE
        assert "no-equals-sign" in exc_info.value.message

    def test_a_malformed_tag_filter_does_not_escape_as_system_exit(
        self, stubbed_startup
    ):
        """`_parse_tags` used to `sys.exit(1)`.

        `SystemExit` is a `BaseException`, so it unwound past every handler in
        `_run` and past `record_exit_code` — the failed invocation reached
        telemetry as a success. `main` must return the code instead.
        """
        stubbed_startup.setattr(
            main_mod,
            "_dispatch",
            lambda args, ci: eval_cli._parse_tags(args.tag),
        )

        assert (
            main_mod.main(["eval", "models", "list", "--tag", "bad"]) == ExitCode.USAGE
        )

    @pytest.mark.parametrize(
        "dispatch",
        [_dispatch, _dispatch_eval],
        ids=["_dispatch", "_dispatch_eval"],
    )
    def test_an_unknown_command_is_a_usage_error(self, dispatch):
        """Reached only past argparse (a stale dispatch branch), but a bad
        invocation is still 2, not the generic 1 it used to be."""
        with pytest.raises(CliExit) as exc_info:
            dispatch(argparse.Namespace(command="no-such-command"))

        assert exc_info.value.code == ExitCode.USAGE


class TestEntrypointsAgree:
    """`python -m switch_trust.cli`, the Bazel binary (same `__main__.py`) and
    the installed console script (`sys.exit(main())`) must report one code."""

    def test_python_m_propagates_the_return_value(self, monkeypatch):
        monkeypatch.setattr(
            "switch_trust.cli.main.main", lambda: ExitCode.AUTH_FAILED
        )

        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("switch_trust.cli.__main__", run_name="__main__")

        assert exc_info.value.code == ExitCode.AUTH_FAILED

    def test_python_m_propagates_success(self, monkeypatch):
        monkeypatch.setattr("switch_trust.cli.main.main", lambda: ExitCode.OK)

        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("switch_trust.cli.__main__", run_name="__main__")

        assert exc_info.value.code == ExitCode.OK


class _TtyStdin(io.StringIO):
    def isatty(self):
        return True


class TestFirstRunWithoutTerminal:
    @pytest.mark.parametrize(
        ("stdin", "runs_init"),
        [(io.StringIO("\n"), False), (None, False), (_TtyStdin(""), True)],
        ids=["piped", "closed", "tty"],
    )
    @patch("switch_trust.cli.main._dispatch")
    @patch("switch_trust.cli.main.setup_file_logging")
    @patch("switch_trust.cli.main._print_logo")
    @patch("switch_trust.cli.main._load_environment")
    @patch("switch_trust.cli.main.get_telemetry_consent", return_value=False)
    @patch("switch_trust.cli.main.ensure_telemetry_consent")
    @patch("switch_trust.cli.main.ensure_client_id")
    @patch("switch_trust.cli.main.is_ci", return_value=False)
    @patch("switch_trust.cli.main.init_cli")
    def test_first_run_skips_interactive_init_without_a_terminal(
        self,
        mock_init_cli,
        mock_is_ci,
        mock_ensure_client_id,
        mock_ensure_telemetry_consent,
        mock_get_telemetry_consent,
        mock_load_env,
        mock_logo,
        mock_logging,
        mock_dispatch,
        stdin,
        runs_init,
        monkeypatch,
        tmp_path,
    ):
        mock_init_cli.get_switch_trust_env_path.return_value = tmp_path / "nonexistent"
        monkeypatch.setattr(sys, "stdin", stdin)

        from switch_trust.cli.main import main

        main(["eval", "models", "list"])

        assert mock_init_cli.run_init.called is runs_init
