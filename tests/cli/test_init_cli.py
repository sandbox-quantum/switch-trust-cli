import io
import sys
from unittest.mock import MagicMock, patch

from switch_trust.cli import init_cli, utils
from switch_trust.cli.utils import TELEMETRY_CONSENT_ENV_VAR


class _TtyStdin(io.StringIO):
    def isatty(self):
        return True


def _run_init(tmp_path, monkeypatch, stdin, consent_choice="Y"):
    st_dir = tmp_path / ".switch-trust"
    monkeypatch.setattr(init_cli, "get_switch_trust_dir", lambda: st_dir)
    monkeypatch.setattr(init_cli, "get_switch_trust_env_path", lambda: st_dir / ".env")
    monkeypatch.setattr(
        init_cli, "get_switch_trust_config_path", lambda: st_dir / "config.json"
    )
    monkeypatch.setattr(sys, "stdin", stdin)
    prompt = MagicMock()
    prompt.ask.side_effect = lambda *a, default=None, **kw: default or "fake-key"
    with (
        patch.object(init_cli, "select", return_value="gemini"),
        patch.object(init_cli, "Prompt", prompt),
        patch.object(utils, "select", return_value=consent_choice) as consent_select,
    ):
        init_cli.run_init()
    return (st_dir / ".env").read_text(), consent_select


def test_init_with_piped_stdin_skips_consent_and_omits_it(tmp_path, monkeypatch):
    env_text, consent_select = _run_init(tmp_path, monkeypatch, io.StringIO("\n"))
    consent_select.assert_not_called()
    assert TELEMETRY_CONSENT_ENV_VAR not in env_text
    assert "GENERATOR_MODEL=gemini:" in env_text


def test_init_with_tty_records_consent_answer(tmp_path, monkeypatch):
    env_text, consent_select = _run_init(
        tmp_path, monkeypatch, _TtyStdin(), consent_choice="N"
    )
    consent_select.assert_called_once()
    assert f"{TELEMETRY_CONSENT_ENV_VAR}=false" in env_text
