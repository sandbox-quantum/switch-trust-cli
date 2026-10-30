"""`scan --fail-on`: the severity gate that makes exit 5 reachable.

Two policies carry most of the risk here and so most of the tests: the gate
counts the post-triage findings, and a severity it cannot read fails closed.
Both are choices rather than consequences of the code, which is why each one
is pinned rather than left to be rediscovered from behaviour.
"""

import argparse
from types import SimpleNamespace

import pytest

from switch_trust.cli import scan_cli
from switch_trust.cli.exit_codes import CliExit, ExitCode
from switch_trust.scan.constants import SEVERITY_ORDER


def _finding(severity: str):
    """A finding as a local scan produces it — an object, not a dict."""
    return SimpleNamespace(ai_spm_severity=severity)


def _report(*severities: str, pre_triage: list | None = None):
    return SimpleNamespace(
        findings=[_finding(s) for s in severities],
        pre_triage_findings=pre_triage,
    )


def _args(fail_on: str | None) -> argparse.Namespace:
    return argparse.Namespace(fail_on=fail_on)


def _gate(fail_on: str | None, *severities: str, output_path: str = "scan.json"):
    scan_cli._apply_fail_on(_args(fail_on), _report(*severities), output_path)


class TestTheSeverityVocabulary:
    """The flag's choices come from the scanner, so the two cannot drift."""

    def test_choices_are_the_scanner_severities(self):
        assert set(scan_cli._SEVERITY_CHOICES) == {
            name.lower() for name in SEVERITY_ORDER
        }

    def test_the_flag_reaches_the_parser(self):
        parser = argparse.ArgumentParser()
        scan_cli.register(parser.add_subparsers(dest="command"))

        args = parser.parse_args(["scan", ".", "--fail-on", "high"])

        assert args.fail_on == "high"

    def test_the_flag_is_optional(self):
        parser = argparse.ArgumentParser()
        scan_cli.register(parser.add_subparsers(dest="command"))

        assert parser.parse_args(["scan", "."]).fail_on is None

    def test_an_unknown_severity_is_a_usage_error(self):
        """argparse rejects it before the scan burns a single model call."""
        parser = argparse.ArgumentParser()
        scan_cli.register(parser.add_subparsers(dest="command"))

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["scan", ".", "--fail-on", "severe"])

        assert exc_info.value.code == ExitCode.USAGE


class TestReadingASeverity:
    """Dataclass findings and round-tripped dicts must read identically."""

    def test_an_object_finding(self):
        assert scan_cli._severity_of(_finding("High")) == "High"

    def test_a_dict_finding(self):
        assert scan_cli._severity_of({"ai_spm_severity": "High"}) == "High"

    @pytest.mark.parametrize(
        "finding",
        [{}, {"ai_spm_severity": None}, SimpleNamespace(), _finding(None)],
        ids=["missing-key", "null-dict", "no-attr", "null-attr"],
    )
    def test_an_absent_severity_reads_as_empty_never_none(self, finding):
        """`None` would blow up `.strip()` in the gate and `severity_style`."""
        assert scan_cli._severity_of(finding) == ""

    def test_a_dict_finding_trips_the_gate_like_an_object_one(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._apply_fail_on(
                _args("high"),
                SimpleNamespace(
                    findings=[{"ai_spm_severity": "Critical"}],
                    pre_triage_findings=None,
                ),
                "scan.json",
            )

        assert exc_info.value.code == ExitCode.FINDINGS_THRESHOLD


class TestTheGate:
    def test_no_flag_never_trips(self):
        """The default is ungated: a critical finding still exits 0."""
        _gate(None, "Critical", "Critical")

    def test_no_findings_passes(self):
        _gate("low")

    def test_a_finding_below_the_bar_passes(self):
        _gate("high", "Medium", "Low")

    def test_a_finding_at_the_bar_trips(self):
        """'At or above' — the bar itself is inside the gate."""
        with pytest.raises(CliExit) as exc_info:
            _gate("high", "High")

        assert exc_info.value.code == ExitCode.FINDINGS_THRESHOLD

    def test_a_finding_above_the_bar_trips(self):
        with pytest.raises(CliExit) as exc_info:
            _gate("high", "Critical")

        assert exc_info.value.code == ExitCode.FINDINGS_THRESHOLD

    @pytest.mark.parametrize("severity", ["critical", "CRITICAL", "Critical"])
    def test_severity_casing_does_not_matter(self, severity):
        """The field is free text; `Critical` and `critical` are one severity."""
        with pytest.raises(CliExit):
            _gate("high", severity)

    def test_whitespace_is_tolerated(self):
        with pytest.raises(CliExit):
            _gate("high", "  High  ")

    def test_the_lowest_bar_still_excludes_nothing_below_it(self):
        """`--fail-on info` is the strictest gate the vocabulary allows."""
        with pytest.raises(CliExit) as exc_info:
            _gate("info", "Info")

        assert exc_info.value.code == ExitCode.FINDINGS_THRESHOLD

    def test_the_message_names_the_gate_and_the_counts(self):
        with pytest.raises(CliExit) as exc_info:
            _gate("high", "Critical", "High", "High", "Low")

        message = exc_info.value.message
        assert "3 findings at or above `high`" in message
        assert "1 critical, 2 high" in message
        assert "low" not in message  # below the bar, not the gate's business

    def test_the_breakdown_is_ordered_most_severe_first(self):
        with pytest.raises(CliExit) as exc_info:
            _gate("medium", "Medium", "Critical", "High")

        message = exc_info.value.message
        assert "1 critical, 1 high, 1 medium" in message

    def test_a_single_finding_is_not_pluralised(self):
        with pytest.raises(CliExit) as exc_info:
            _gate("high", "Critical")

        assert "1 finding at or above" in exc_info.value.message

    def test_the_report_path_rides_along(self):
        """The gate trips after the write, so the footer can still name it."""
        with pytest.raises(CliExit) as exc_info:
            _gate("high", "Critical", output_path="scan_20260101T000000.json")

        assert exc_info.value.output_path == "scan_20260101T000000.json"


class TestUnreadableSeverities:
    """Fail closed: a severity nobody can parse must not be silently dropped.

    `severity_style` renders anything unrecognised as green, so the failure
    this prevents is invisible in the findings table too.
    """

    @pytest.mark.parametrize(
        "severity",
        ["Severe", "", "   ", "P1", "high-ish"],
        ids=["unknown-word", "empty", "whitespace", "ticket-scale", "fuzzy"],
    )
    def test_an_unreadable_severity_trips_the_gate(self, severity):
        with pytest.raises(CliExit) as exc_info:
            _gate("critical", severity)

        assert exc_info.value.code == ExitCode.FINDINGS_THRESHOLD

    def test_it_trips_even_at_the_strictest_threshold(self):
        """Unknown is never assumed to be below the bar, whatever the bar is."""
        with pytest.raises(CliExit):
            _gate("info", "Severe")

    def test_the_message_names_the_value_it_could_not_read(self):
        with pytest.raises(CliExit) as exc_info:
            _gate("critical", "Severe")

        message = exc_info.value.message
        assert "'Severe'" in message
        assert "unreadable severity" in message

    def test_an_empty_severity_is_named_legibly(self):
        with pytest.raises(CliExit) as exc_info:
            _gate("critical", "")

        assert "(empty)" in exc_info.value.message

    def test_both_causes_are_reported_together(self):
        """A run can be over the bar *and* partly unreadable; say both."""
        with pytest.raises(CliExit) as exc_info:
            _gate("high", "Critical", "Severe")

        message = exc_info.value.message
        assert "1 finding at or above `high` (1 critical)" in message
        assert "1 finding with an unreadable severity" in message

    def test_unreadable_values_are_deduplicated_and_capped(self):
        """The message goes in a fixed-width panel, so it cannot be unbounded."""
        with pytest.raises(CliExit) as exc_info:
            _gate("critical", *[f"bogus-{i}" for i in range(10)])

        message = exc_info.value.message
        assert "10 findings with an unreadable severity" in message
        assert message.count("bogus-") == 3

    def test_an_ungated_scan_ignores_them(self):
        _gate(None, "Severe", "")


class TestTriagePolicy:
    """The gate counts post-triage findings. This is a choice; pin it."""

    def test_pre_triage_findings_are_not_counted(self):
        """A finding triage dropped does not trip the gate.

        Triage exists to remove false positives, so gating on the pre-triage
        set would fire CI on findings the product has already judged wrong.
        """
        report = SimpleNamespace(
            findings=[_finding("Low")],
            pre_triage_findings=[_finding("Critical"), _finding("Critical")],
        )

        scan_cli._apply_fail_on(_args("high"), report, "scan.json")

    def test_a_downgraded_finding_is_judged_on_its_new_severity(self):
        report = SimpleNamespace(
            findings=[_finding("Low")],
            pre_triage_findings=[_finding("Critical")],
        )

        scan_cli._apply_fail_on(_args("critical"), report, "scan.json")
