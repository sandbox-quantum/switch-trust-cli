"""The CI-detection warning: the other half of `--fail-on` / `--fail-under`
defaulting to off for backwards compatibility.

Without it, a CI user who never passes either flag silently keeps the
pre-CHOO-3479 always-exit-0 behaviour and never learns the gate exists. The
warning must fire exactly when CI is detected and the matching gate was
omitted — not when the gate was set, not outside CI, and not for eval
subcommands other than `run` (`eval models list` etc. would just be noise).
"""

import argparse
import json

import pytest

from switch_trust.cli import main, scan_cli
from switch_trust.cli.main import _warn_if_ci_without_gate


class TestWarnIfCiWithoutGate:
    """The pure gating logic, independent of any particular command."""

    @pytest.mark.parametrize(
        "ci, gate_set, should_warn",
        [
            (True, False, True),
            (True, True, False),
            (False, False, False),
            (False, True, False),
        ],
    )
    def test_fires_only_in_ci_with_no_gate(self, capsys, ci, gate_set, should_warn):
        _warn_if_ci_without_gate(
            ci, gate_set, "--fail-on", "this scan will always exit 0"
        )

        captured = capsys.readouterr().out
        assert ("--fail-on" in captured) == should_warn


class TestScanWarns:
    def test_ci_without_fail_on_warns(self, monkeypatch, capsys):
        monkeypatch.setattr(scan_cli, "handle_scan", lambda args: "scan_out.json")

        main._dispatch_scan(
            argparse.Namespace(path=".", fail_on=None),
            ci=True,
        )

        captured = capsys.readouterr().out
        assert "--fail-on" in captured
        assert "CI detected" in captured

    def test_ci_with_fail_on_is_silent(self, monkeypatch, capsys):
        monkeypatch.setattr(scan_cli, "handle_scan", lambda args: "scan_out.json")

        main._dispatch_scan(
            argparse.Namespace(path=".", fail_on="high"),
            ci=True,
        )

        assert "--fail-on" not in capsys.readouterr().out

    def test_outside_ci_is_silent(self, monkeypatch, capsys):
        monkeypatch.setattr(scan_cli, "handle_scan", lambda args: "scan_out.json")

        main._dispatch_scan(
            argparse.Namespace(path=".", fail_on=None),
            ci=False,
        )

        assert "--fail-on" not in capsys.readouterr().out


class TestEvalRunWarns:
    """Scoped to `eval run` — `models`/`evaluations`/`model-evaluations` stay quiet."""

    def _config(self, tmp_path) -> str:
        config = tmp_path / "config.json"
        config.write_text(json.dumps({}))
        return str(config)

    def test_ci_without_fail_under_warns_on_run(self, monkeypatch, tmp_path, capsys):
        async def fake_handle_run(args, store):
            return "eval_out.json"

        monkeypatch.setattr(main.eval_cli, "handle_run", fake_handle_run)

        main._dispatch_eval(
            argparse.Namespace(
                command="eval",
                config=self._config(tmp_path),
                eval_cmd="run",
                fail_under=None,
            ),
            ci=True,
        )

        captured = capsys.readouterr().out
        assert "--fail-under" in captured
        assert "CI detected" in captured

    def test_ci_with_fail_under_is_silent(self, monkeypatch, tmp_path, capsys):
        async def fake_handle_run(args, store):
            return "eval_out.json"

        monkeypatch.setattr(main.eval_cli, "handle_run", fake_handle_run)

        main._dispatch_eval(
            argparse.Namespace(
                command="eval",
                config=self._config(tmp_path),
                eval_cmd="run",
                fail_under=0.8,
            ),
            ci=True,
        )

        assert "--fail-under" not in capsys.readouterr().out

    def test_other_eval_subcommands_never_warn(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(main.eval_cli, "handle_models", lambda args, store: None)

        main._dispatch_eval(
            argparse.Namespace(
                command="eval",
                config=self._config(tmp_path),
                eval_cmd="models",
            ),
            ci=True,
        )

        assert "--fail-under" not in capsys.readouterr().out
