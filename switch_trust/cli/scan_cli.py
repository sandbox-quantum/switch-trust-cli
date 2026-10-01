"""
Subcommands for `switch-trust scan`.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import logging
import os
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.table import Table

from switch_trust.cli.console import CLI_WIDTH, console, severity_style
from switch_trust.cli.file_filter import FileType, RelevantFile, find_relevant_files
from switch_trust.cli.output_formatters import (
    OutputFormat,
    get_scan_output_formatter,
)

if TYPE_CHECKING:
    from switch_trust.scan.schema import ScanReport

logger = logging.getLogger(__name__)


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
        sev = (
            f.ai_spm_severity
            if isinstance(f, dataclasses.Field) or hasattr(f, "ai_spm_severity")
            else f.get("ai_spm_severity", "")
        )
        cvss = ""
        title = ""
        source = ""
        file_name = ""

        if isinstance(f, dict):
            sev = f.get("ai_spm_severity", "")
            cvss_scores = f.get("cvss_scores", {})
            cvss = str(cvss_scores.get("base_score", "")) if cvss_scores else ""
            title = f.get("title", "") or f.get("description", "")[:80]
            source = f.get("source", "")
            components = f.get("affected_components", [])
            if components:
                file_name = os.path.basename(components[0].get("name", ""))
        else:
            sev = f.ai_spm_severity
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
    from switch_trust.schema import RepoFile

    path: str = os.path.abspath(os.path.expanduser(args.path))

    if not os.path.exists(path):
        # Raise instead of sys.exit so the error flows through main's unified
        # exception handler (record_error -> telemetry, error panel, logging).
        # sys.exit raises SystemExit, a BaseException that bypasses that path.
        raise FileNotFoundError(f"Path does not exist: {path}")

    files: list[RelevantFile] = find_relevant_files(path)

    if not files:
        # Raise (rather than sys.exit) so the error reaches main's unified
        # handler and is recorded to telemetry. See the path check above.
        raise ValueError("No scannable targets found")

    model = os.environ.get("GENERATOR_MODEL")

    logger.info(
        "Starting agent scan on %s files with model '%s'", len(files), model or "none"
    )

    python_files: list[RepoFile] = []
    requirements_files: list[RepoFile] = []
    for rf in files:
        with open(rf.path, encoding="utf-8", errors="replace") as fh:
            content = fh.read()
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
    write_output(report, output_path, fmt=fmt)
    return output_path
