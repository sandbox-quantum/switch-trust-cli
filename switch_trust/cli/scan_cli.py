"""
Subcommands for `switch-trust scan`.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table

from switch_trust.cli.console import CLI_WIDTH, console, severity_style
from switch_trust.cli.file_filter import FileType, RelevantFile, find_relevant_files
from switch_trust.cli.output_formatters import (
    OutputFormat,
    get_scan_output_formatter,
)
from switch_trust.cli.exit_codes import CliExit, ExitCode

# The scanner's own ranked vocabulary, not a second copy of it: the gate has
# to order severities exactly the way the scanner assigns them, and a private
# ranking here would drift the first time one is added. `constants.py` imports
# nothing, so this costs none of the ADK load the scanner import below defers.
from switch_trust.scan.constants import SEVERITY_ORDER

if TYPE_CHECKING:
    from switch_trust.scan.schema import ScanReport

logger = logging.getLogger(__name__)

# Lowercase for the command line, canonical title-case internally. Derived
# from SEVERITY_ORDER so a severity added to the scanner becomes gateable
# without a second edit here.
_SEVERITY_CHOICES = tuple(name.lower() for name in SEVERITY_ORDER)


def register(subparsers: argparse._SubParsersAction) -> None:
    scan_parser = subparsers.add_parser(
        "scan",
        help="Agent security scanning commands",
    )
    scan_parser.add_argument(
        "path",
        help="Path to a file or folder to scan",
    )
    scan_parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: scan_<timestamp>.<format>)",
    )
    scan_parser.add_argument(
        "--format",
        "-f",
        choices=["json", "sarif"],
        default="json",
        help="Output format (default: json)",
    )
    scan_parser.add_argument(
        "--log",
        help="Log file path (default: switch_trust_<timestamp>.log)",
    )
    scan_parser.add_argument(
        "--fail-on",
        choices=_SEVERITY_CHOICES,
        help=(
            "Exit 5 if any finding is at or above this severity. "
            "Omit to always exit 0 on a completed scan."
        ),
    )


def _check_configured_model(
    model: str | None,
    api_key_present: Callable[[str], bool],
) -> None:
    """Refuse a configured model whose key is missing, before the scan starts.

    A preflight rather than a post-mortem: `run_core` would quietly fall back
    to static findings, and the user would pay minutes of CI to learn nothing.
    Refusing here costs them nothing, since no scan has begun — hence no
    `output_path` on the raise.

    **A model left unconfigured is not an error.** The user did not ask for AI
    reasoning, and the static layer is the whole product they get; turning
    that into exit 20 would break every static-only run.

    `api_key_present` is passed in rather than imported here because importing
    it loads the ADK stack, which `handle_scan` defers to call time.
    """
    if model and not api_key_present(model):
        raise CliExit(
            ExitCode.AUTH_FAILED,
            f"No API key is configured for the scanner model '{model}'.",
        )


def _check_llm_credentials(report: ScanReport, output_path: str) -> None:
    """Stop on an LLM credential the provider rejected or refused mid-scan.

    `run_core` deliberately swallows an AI-reasoning failure and finishes with
    static findings only — a customer's scan must not die because the model
    did. The cost is that a rejected/unentitled key produces a clean-looking
    report with the entire reasoning layer silently missing, and exit 0 to go
    with it.

    The reasoner classifies the failure where the exception still exists and
    carries a bounded `error_reason` out in its trace, precisely so a caller
    can say which config is wrong without re-reading `str(exc)`. Reading that
    marker is what turns the silent degradation into exit
    20/21/25/60/61/62/63/64.
    """
    meta = report.scan_metadata or {}
    trace = meta.get("agentic_trace") or {}
    reason = trace.get("error_reason")

    if reason == "not_entitled":
        raise CliExit(
            ExitCode.NOT_ENTITLED,
            "The provider accepted the configured credential but refused it "
            "as not permitted, so the AI reasoning layer was skipped and "
            "these findings are static-analysis only. Check the plan/quota "
            "for the scanner model's provider account and re-run.",
            output_path=output_path,
        )

    if reason == "auth":
        raise CliExit(
            ExitCode.AUTH_FAILED,
            "The provider rejected the configured credential, so the AI reasoning "
            "layer was skipped and these findings are static-analysis only. Check "
            "the API key for the scanner model and re-run.",
            output_path=output_path,
        )

    if reason == "model":
        # `GENERATOR_MODEL` parses regardless of shape (`make_model` never
        # validates the provider/model string), so a typo'd or unsupported
        # value is never caught before the scan starts — it surfaces only
        # once the provider itself rejects it mid-reasoning. The credential
        # was fine; the configured model id is the thing to fix.
        raise CliExit(
            ExitCode.INVALID_CONFIG,
            "The configured model is invalid or unsupported, so the AI "
            "reasoning layer was skipped and these findings are "
            "static-analysis only. Check GENERATOR_MODEL and re-run.",
            output_path=output_path,
        )

    if reason == "timeout":
        # The provider call itself timed out — not one of our own budgets
        # (ScannerTimeout/LOOP_TIMEOUT_SECS), which never reach here. Like
        # "rate_limited" below, nothing about the configuration is wrong, so
        # this is retryable rather than a 20-39 fix-the-config stop.
        raise CliExit(
            ExitCode.TIMEOUT,
            "The scanner model's provider call timed out, so the AI "
            "reasoning layer was skipped and these findings are "
            "static-analysis only. This is often transient — re-run.",
            output_path=output_path,
        )

    if reason == "upstream":
        # The provider's own backend answered with a 5xx — not a timeout
        # (504 is claimed there) and not a config problem, so this is
        # retryable rather than a 20-39 fix-the-config stop.
        raise CliExit(
            ExitCode.UPSTREAM_ERROR,
            "The scanner model's provider backend failed, so the AI "
            "reasoning layer was skipped and these findings are "
            "static-analysis only. This is often transient — re-run.",
            output_path=output_path,
        )

    if reason == "resource_exhausted":
        # A literal HTTP 429 the provider answered with — distinct from
        # "rate_limited" below (which has no status behind it at all), but
        # equally nothing about the configuration is wrong, so this is
        # retryable rather than a 20-39 fix-the-config stop.
        raise CliExit(
            ExitCode.RESOURCE_EXHAUSTED,
            "The scanner model's provider rate-limited the request with "
            "HTTP 429, so the AI reasoning layer was skipped and these "
            "findings are static-analysis only. This is often transient — "
            "wait a moment and re-run.",
            output_path=output_path,
        )

    if reason == "rate_limited":
        # A provider-declared rate limit with no HTTP 429 behind it (e.g. a
        # gRPC RESOURCE_EXHAUSTED) — like "network" below, nothing about the
        # configuration is wrong, so this is retryable rather than a 20-39
        # fix-the-config stop.
        raise CliExit(
            ExitCode.RATE_LIMITED,
            "The scanner model's provider declared a rate limit, so the AI "
            "reasoning layer was skipped and these findings are "
            "static-analysis only. This is often transient — wait a moment "
            "and re-run.",
            output_path=output_path,
        )

    if reason == "network":
        # A connection refusal or unresolvable host, not a status the
        # provider answered with — unlike the three cases above, nothing
        # about the configuration is wrong, so this is retryable rather than
        # a 20-39 fix-the-config stop.
        raise CliExit(
            ExitCode.NETWORK_FAILURE,
            "Could not reach the configured scanner model's provider, so "
            "the AI reasoning layer was skipped and these findings are "
            "static-analysis only. This is often transient — re-run.",
            output_path=output_path,
        )


def _check_missing_tools(report: ScanReport, output_path: str) -> None:
    """Stop when any static-analysis tool the scan needed could not run.

    `run_static_scan` runs four tools — Bandit, OpenGrep, detect-secrets,
    pip-audit — and `tools_skipped` already distinguishes which ones it tried
    to use from which it couldn't. All four are load-bearing: OpenGrep is the
    *only* source of agent-specific findings (every missing-guardrail,
    unbounded-loop and excessive-agency rule lives in
    `agent_opengrep_rules.yaml`), so a scan missing any of them is not the
    clean verdict exit 0 would report.
    """
    meta = report.scan_metadata or {}
    skipped = meta.get("tools_skipped") or []
    missing = sorted({s.get("tool", "unknown") for s in skipped})
    if not missing:
        return

    plural = "s are" if len(missing) > 1 else " is"
    raise CliExit(
        ExitCode.MISSING_TOOLS,
        f"{', '.join(missing)} {plural} missing, so the scan is incomplete. "
        "Install the missing tool(s) and re-run.",
        output_path=output_path,
    )


def _severity_of(finding) -> str:
    """Read a finding's severity, whichever shape it arrived in.

    Findings reach the CLI as dataclasses from a local scan and as plain
    dicts when a report has been round-tripped through JSON, and the gate has
    to agree with the findings table about what it is looking at — so both
    read severities through here rather than branching twice.
    """
    if isinstance(finding, dict):
        return finding.get("ai_spm_severity") or ""
    return getattr(finding, "ai_spm_severity", "") or ""


def _apply_fail_on(
    args: argparse.Namespace,
    report: ScanReport,
    output_path: str,
) -> None:
    """Trip the `--fail-on` gate, if one was asked for and the scan cleared it.

    Two policies this settles, because neither has a safe default:

    **It counts post-triage `report.findings`** — the set the summary prints,
    the output file contains, and the scan stands behind. Triage exists to
    drop false positives, so gating on `pre_triage_findings` would fire CI on
    findings the product has already judged wrong. The consequence is worth
    stating plainly: a finding the reasoning layer called `critical` and
    triage downgraded to `low` does not trip `--fail-on critical`.

    **An unreadable severity trips the gate, whatever the threshold.** The
    field is free text written partly by a model, and `severity_style` renders
    anything it does not recognise as green — so the quiet failure here is a
    finding that is dropped from a security gate because nobody could parse
    its severity. Failing closed converts that into a message naming the value
    that could not be read. A gate that silently ignores what it cannot
    understand is not a gate.
    """
    threshold = getattr(args, "fail_on", None)
    if threshold is None:
        return

    limit = SEVERITY_ORDER[threshold.capitalize()]

    counts: dict[str, int] = {}
    unreadable: list[str] = []
    for finding in report.findings:
        severity = _severity_of(finding)
        rank = SEVERITY_ORDER.get(severity.strip().capitalize())
        if rank is None:
            unreadable.append(severity.strip() or "(empty)")
        elif rank <= limit:
            counts[severity.strip().capitalize()] = (
                counts.get(severity.strip().capitalize(), 0) + 1
            )

    if not counts and not unreadable:
        return

    raise CliExit(
        ExitCode.FINDINGS_THRESHOLD,
        _gate_message(threshold, counts, unreadable),
        output_path=output_path,
    )


def _gate_message(
    threshold: str,
    counts: dict[str, int],
    unreadable: list[str],
) -> str:
    """Name the gate and the values that tripped it, never the exit code."""
    parts = []
    if counts:
        total = sum(counts.values())
        plural = "s" if total > 1 else ""
        breakdown = ", ".join(
            f"{counts[name]} {name.lower()}"
            for name in SEVERITY_ORDER
            if name in counts
        )
        parts.append(
            f"{total} finding{plural} at or above `{threshold}` ({breakdown})",
        )
    if unreadable:
        plural = "s" if len(unreadable) > 1 else ""
        shown = ", ".join(sorted({f"{value!r}" for value in unreadable})[:3])
        parts.append(
            f"{len(unreadable)} finding{plural} with an unreadable severity "
            f"({shown}), counted as tripping the gate",
        )
    return "; ".join(parts) + "."


def write_output(report: ScanReport, output_path: str, fmt: str = "json"):

    formatter = get_scan_output_formatter(OutputFormat(fmt))
    content = formatter.format(report)
    with open(output_path, "w") as f:
        f.write(content)


def print_report(report: ScanReport) -> None:
    console.print()

    # ── Summary panel ────────────────────────────────────────────
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="key", justify="right")
    grid.add_column()

    grid.add_row("Framework", report.framework_detected or "unknown")

    meta = report.scan_metadata or {}
    py_files = meta.get("python_files", 0)
    total_files = meta.get("total_files_scanned", 0)
    grid.add_row("Files", f"{py_files} Python, {total_files} total scanned")

    total = len(report.findings)
    pre_triage = (
        len(report.pre_triage_findings) if report.pre_triage_findings else total
    )
    if pre_triage != total:
        grid.add_row("Findings", f"{total} [dim]({pre_triage} pre-triage)[/dim]")
    else:
        grid.add_row("Findings", str(total))

    tools = meta.get("tools_used", [])
    if tools:
        grid.add_row("Tools", ", ".join(tools))

    skipped = meta.get("tools_skipped") or []
    if skipped:
        grid.add_row(
            "Skipped",
            "[bold yellow]"
            + ", ".join(s.get("tool", "unknown") for s in skipped)
            + "[/bold yellow]",
        )

    if meta.get("ai_summary"):
        grid.add_row("AI summary", meta["ai_summary"])

    console.print(Panel(grid, title="[bold]Scan Summary[/bold]", width=CLI_WIDTH))

    # ── Partial-scan warning ─────────────────────────────────────
    # A skipped tool means whole rule families never ran, so the findings
    # below are incomplete. Say so loudly — the log file is not enough.
    if skipped:
        warn = Table.grid(padding=(0, 1))
        warn.add_column()
        plural = "s were" if len(skipped) > 1 else " was"
        warn.add_row(
            f"[bold yellow]This scan is incomplete — "
            f"{len(skipped)} analysis tool{plural} skipped.[/bold yellow]"
        )
        for s in skipped:
            warn.add_row("")
            warn.add_row(
                f"[bold]{s.get('tool', 'unknown')}[/bold] — {s.get('reason', '')}"
            )
        console.print(
            Panel(
                warn,
                title="[bold yellow]Warning[/bold yellow]",
                border_style="yellow",
                width=CLI_WIDTH,
            )
        )

    # ── Findings table ───────────────────────────────────────────
    if not report.findings:
        console.print("[dim]No findings.[/dim]")
        console.print()
        return

    table = Table(
        title="Findings",
        show_lines=True,
        width=CLI_WIDTH,
    )
    table.add_column("Sev", max_width=10, ratio=1)
    table.add_column("CVSS", max_width=6, ratio=1)
    table.add_column("Title", style="bold", ratio=5)
    table.add_column("File", ratio=3)
    table.add_column("Source", style="dim", ratio=1)

    for f in report.findings:
        # One reader, shared with the gate, so the table and the exit code
        # can never disagree about a finding's severity.
        sev = _severity_of(f)
        cvss = ""
        title = ""
        source = ""
        file_name = ""

        if isinstance(f, dict):
            cvss_scores = f.get("cvss_scores", {})
            cvss = str(cvss_scores.get("base_score", "")) if cvss_scores else ""
            title = f.get("title", "") or f.get("description", "")[:80]
            source = f.get("source", "")
            components = f.get("affected_components", [])
            if components:
                file_name = os.path.basename(components[0].get("name", ""))
        else:
            cvss = str(f.cvss_scores.base_score) if f.cvss_scores else ""
            title = f.title or f.description[:80]
            source = f.source
            if f.affected_components:
                file_name = os.path.basename(f.affected_components[0].name)

        sev_display = f"[{severity_style(sev)}]{sev}[/]"
        source_label = (
            source.replace("_", " ")
            .replace("ai reasoning", "AI")
            .replace("static ", "")
        )

        table.add_row(sev_display, cvss, title, file_name, source_label)

    console.print(table)

    # ── Category summary ─────────────────────────────────────────
    non_zero = {
        k: v
        for k, v in (report.category_summary or {}).items()
        if (v.get("count", 0) if isinstance(v, dict) else getattr(v, "count", 0)) > 0
    }

    if non_zero:
        cat_table = Table(
            title="Category Summary",
            show_lines=False,
            width=CLI_WIDTH,
        )
        cat_table.add_column("Category", style="bold", ratio=5)
        cat_table.add_column("Total", justify="right", ratio=1)
        cat_table.add_column("Crit", justify="right", ratio=1)
        cat_table.add_column("High", justify="right", ratio=1)
        cat_table.add_column("Med", justify="right", ratio=1)
        cat_table.add_column("Low", justify="right", ratio=1)

        for v in non_zero.values():
            if isinstance(v, dict):
                asi_title = v.get("asi_title", "")
                count = v.get("count", 0)
                crit = v.get("critical", 0)
                high = v.get("high", 0)
                med = v.get("medium", 0)
                low = v.get("low", 0)
            else:
                asi_title = getattr(v, "asi_title", "")
                count = getattr(v, "count", 0)
                crit = getattr(v, "critical", 0)
                high = getattr(v, "high", 0)
                med = getattr(v, "medium", 0)
                low = getattr(v, "low", 0)

            cat_table.add_row(
                asi_title,
                str(count),
                f"[{severity_style('critical')}]{crit}[/]" if crit else str(crit),
                f"[{severity_style('high')}]{high}[/]" if high else str(high),
                f"[{severity_style('medium')}]{med}[/]" if med else str(med),
                str(low),
            )

        console.print(cat_table)

    console.print()


def handle_scan(args: argparse.Namespace) -> str:
    # Imported lazily: pulling in the agent scanner loads the Google ADK stack,
    # which is expensive and only needed when a scan actually runs.
    from switch_trust.scan.agent_scanner import (  # noqa: PLC0415 - deferred import cost
        run_core,
    )
    from switch_trust.scan.llm_provider import (  # noqa: PLC0415 - deferred import cost
        api_key_present,
    )
    from switch_trust.schema import RepoFile

    path: str = os.path.abspath(os.path.expanduser(args.path))

    # Both checks below raise CliExit rather than calling sys.exit: the error
    # reaches main's unified handler (record_error -> telemetry, error panel,
    # logging) with the exit code the table gives it, where sys.exit would
    # raise SystemExit — a BaseException that bypasses that path entirely.
    if not os.path.exists(path):
        raise CliExit(ExitCode.TARGET_ERROR, f"Path does not exist: {path}")

    if not os.path.isdir(path) and not os.path.isfile(path):
        raise CliExit(
            ExitCode.TARGET_ERROR,
            f"Path is neither a file nor a directory: {path}",
        )

    try:
        files: list[RelevantFile] = find_relevant_files(path)
    except OSError as e:
        raise CliExit(
            ExitCode.TARGET_ERROR,
            f"Could not read {path}: {e.strerror or e}",
        ) from e

    if not files:
        # Distinct from a bad path: the target was readable, it simply holds
        # nothing this scanner can analyse. A caller fixes those two different
        # ways, so they do not share a code.
        raise CliExit(
            ExitCode.NO_TARGETS,
            f"No scannable targets found in {path}",
        )

    model = os.environ.get("GENERATOR_MODEL")

    _check_configured_model(model, api_key_present)

    logger.info(
        "Starting agent scan on %s files with model '%s'", len(files), model or "none"
    )

    python_files: list[RepoFile] = []
    requirements_files: list[RepoFile] = []
    for rf in files:
        try:
            with open(rf.path, encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError as e:
            raise CliExit(
                ExitCode.TARGET_ERROR,
                f"Could not read {rf.path}: {e.strerror or e}",
            ) from e
        repo_file = RepoFile(path=rf.path, content=content, size=len(content))
        if rf.type == FileType.REQUIREMENTS:
            requirements_files.append(repo_file)
        elif rf.type == FileType.PYTHON:
            python_files.append(repo_file)

    logger.info(
        "Files categorized: %d Python, %d requirements",
        len(python_files),
        len(requirements_files),
    )

    fw_counts: dict[str, int] = {}
    for rf in files:
        if rf.framework:
            fw_counts[rf.framework] = fw_counts.get(rf.framework, 0) + 1
    frameworks = list(fw_counts)
    primary = max(fw_counts, key=fw_counts.get) if fw_counts else "unknown"

    logger.info("Frameworks detected: %s (primary: %s)", ", ".join(frameworks), primary)

    # Let a scan failure propagate to main's unified exception handler so it is
    # recorded to telemetry; previously it was swallowed into sys.exit(1).
    report = run_core(
        python_files,
        requirements_files,
        skip_triage=False,
        agentic=True,
        model_string=model,
        repo_name=os.path.basename(path),
        primary_framework=primary,
        frameworks_detected=frameworks,
        total_files_scanned=len(files),
    )

    print_report(report)
    fmt = getattr(args, "format", "json") or "json"
    ext = fmt if fmt != "json" else "json"
    output_path = args.output or (
        f"scan_{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}.{ext}"
    )
    # All three checks run after the report lands, so a stop still leaves its
    # artifact. Both the missing-tools and the credentials check outrank the
    # gate because the 20-39 band outranks the gate's 5 in the documented
    # precedence — and rightly: findings counted from a scan that is missing
    # part of its own tooling, or whose reasoning layer never ran, are not the
    # verdict the user asked the gate to judge.
    write_output(report, output_path, fmt=fmt)
    _check_missing_tools(report, output_path)
    _check_llm_credentials(report, output_path)
    _apply_fail_on(args, report, output_path)
    return output_path
