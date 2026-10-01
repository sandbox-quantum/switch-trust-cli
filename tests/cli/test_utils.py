from unittest.mock import patch

import pytest

from switch_trust.cli import utils
from switch_trust.cli.utils import (
    CLIENT_ID_ENV_VAR,
    TELEMETRY_CONSENT_ENV_VAR,
    get_client_id,
    get_telemetry_consent,
    legacy_env_vars_used,
    resolve_env_var,
)

_LEGACY_CLIENT_ID = "FLINTAI_CLIENT_ID"
_LEGACY_CONSENT = "FLINTAI_TELEMETRY_CONSENT"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate the env vars under test and reset the one-time notice tracker."""
    for var in (
        CLIENT_ID_ENV_VAR,
        TELEMETRY_CONSENT_ENV_VAR,
        _LEGACY_CLIENT_ID,
        _LEGACY_CONSENT,
    ):
        monkeypatch.delenv(var, raising=False)
    utils._deprecation_notices_shown.clear()


class TestResolveEnvVar:
    def test_prefers_current_name(self, monkeypatch):
        monkeypatch.setenv(CLIENT_ID_ENV_VAR, "new")
        monkeypatch.setenv(_LEGACY_CLIENT_ID, "old")
        with patch.object(utils.console, "print") as mock_print:
            assert resolve_env_var(CLIENT_ID_ENV_VAR) == "new"
        mock_print.assert_not_called()

    def test_falls_back_to_legacy_and_warns(self, monkeypatch):
        monkeypatch.setenv(_LEGACY_CLIENT_ID, "old")
        with patch.object(utils.console, "print") as mock_print:
            assert resolve_env_var(CLIENT_ID_ENV_VAR) == "old"
        mock_print.assert_called_once()
        note = mock_print.call_args.args[0]
        assert _LEGACY_CLIENT_ID in note
        assert CLIENT_ID_ENV_VAR in note

    def test_none_when_neither_set(self):
        with patch.object(utils.console, "print") as mock_print:
            assert resolve_env_var(CLIENT_ID_ENV_VAR) is None
        mock_print.assert_not_called()

    def test_deprecation_note_printed_once(self, monkeypatch):
        monkeypatch.setenv(_LEGACY_CONSENT, "true")
        with patch.object(utils.console, "print") as mock_print:
            resolve_env_var(TELEMETRY_CONSENT_ENV_VAR)
            resolve_env_var(TELEMETRY_CONSENT_ENV_VAR)
        mock_print.assert_called_once()

    def test_legacy_env_vars_used_tracks_fallback(self, monkeypatch):
        assert legacy_env_vars_used() is False
        monkeypatch.setenv(_LEGACY_CLIENT_ID, "old")
        with patch.object(utils.console, "print"):
            resolve_env_var(CLIENT_ID_ENV_VAR)
        assert legacy_env_vars_used() is True

    def test_legacy_env_vars_used_false_for_current(self, monkeypatch):
        monkeypatch.setenv(CLIENT_ID_ENV_VAR, "new")
        with patch.object(utils.console, "print"):
            resolve_env_var(CLIENT_ID_ENV_VAR)
        assert legacy_env_vars_used() is False


class TestBackwardsCompatAccessors:
    def test_get_client_id_uses_legacy(self, monkeypatch):
        monkeypatch.setenv(_LEGACY_CLIENT_ID, "legacy-id")
        with patch.object(utils.console, "print"):
            assert get_client_id() == "legacy-id"

    def test_get_telemetry_consent_uses_legacy(self, monkeypatch):
        monkeypatch.setenv(_LEGACY_CONSENT, "true")
        with patch.object(utils.console, "print"):
            assert get_telemetry_consent() is True

    def test_current_consent_overrides_legacy(self, monkeypatch):
        monkeypatch.setenv(TELEMETRY_CONSENT_ENV_VAR, "false")
        monkeypatch.setenv(_LEGACY_CONSENT, "true")
        with patch.object(utils.console, "print") as mock_print:
            assert get_telemetry_consent() is False
        mock_print.assert_not_called()
