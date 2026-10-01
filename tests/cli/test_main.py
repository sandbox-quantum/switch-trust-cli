import os
from unittest.mock import patch

import pytest

from switch_trust.cli.main import _load_environment, _print_error


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

        with pytest.raises(SystemExit) as exc_info:
            main(["eval", "models", "list"])

        assert exc_info.value.code == 1
        captured = capsys.readouterr().out
        assert "RuntimeError" in captured
        assert "test crash" in captured
        assert "Completed in" in captured
