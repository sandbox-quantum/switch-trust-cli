import io
import os
import sys
from unittest.mock import patch

import pytest

from switch_trust.cli import utils
from switch_trust.cli.utils import (
    CLIENT_ID_ENV_VAR,
    TELEMETRY_CONSENT_ENV_VAR,
    ensure_telemetry_consent,
    get_client_id,
    get_telemetry_consent,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (CLIENT_ID_ENV_VAR, TELEMETRY_CONSENT_ENV_VAR):
        monkeypatch.delenv(var, raising=False)


class TestClientId:
    def test_none_when_unset(self):
        assert get_client_id() is None

    def test_reads_current_env_var(self, monkeypatch):
        monkeypatch.setenv(CLIENT_ID_ENV_VAR, "new")
        assert get_client_id() == "new"


class TestTelemetryConsent:
    def test_false_when_unset(self):
        assert get_telemetry_consent() is False

    def test_reads_current_env_var(self, monkeypatch):
        monkeypatch.setenv(TELEMETRY_CONSENT_ENV_VAR, "true")
        assert get_telemetry_consent() is True


class _TtyStdin(io.StringIO):
    def isatty(self):
        return True


class TestEnsureTelemetryConsent:
    @pytest.fixture
    def env_file(self, tmp_path, monkeypatch):
        env_path = tmp_path / ".env"
        env_path.write_text("GENERATOR_MODEL=x\n")
        monkeypatch.setattr(utils, "get_switch_trust_env_path", lambda: env_path)
        monkeypatch.setattr(utils, "is_ci", lambda: False)
        return env_path

    @pytest.mark.parametrize(
        "stdin",
        [io.StringIO("\n"), io.StringIO(""), None],
        ids=["newline", "eof", "closed"],
    )
    def test_non_tty_stdin_is_no_consent_and_not_persisted(
        self, env_file, monkeypatch, stdin
    ):
        monkeypatch.setattr(sys, "stdin", stdin)
        ensure_telemetry_consent()
        assert os.environ[TELEMETRY_CONSENT_ENV_VAR] == "false"
        assert get_telemetry_consent() is False
        assert TELEMETRY_CONSENT_ENV_VAR not in env_file.read_text()

    def test_tty_stdin_prompts_with_no_default_and_persists(
        self, env_file, monkeypatch
    ):
        monkeypatch.setattr(sys, "stdin", _TtyStdin())
        with patch.object(utils, "select", return_value="Y") as mock_select:
            ensure_telemetry_consent()
        assert mock_select.call_args.kwargs == {
            "options": ["Y", "N"],
            "default_index": 0,
        }
        assert os.environ[TELEMETRY_CONSENT_ENV_VAR] == "true"
        assert f"{TELEMETRY_CONSENT_ENV_VAR}=true" in env_file.read_text()
