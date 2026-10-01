"""
CLI evaluation runner.

Runs evaluations without a database, using the JsonRepository for
configuration and printing progress to the console.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from dataclasses_json import dataclass_json
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from switch_trust.cli.console import CLI_WIDTH, console, score_style, status_style
from switch_trust.cli.output_formatters import (
    OutputFormat,
    get_eval_output_formatter,
)
from switch_trust.cli.rich_observer import RichObserver, pad_description
from switch_trust.eval.core.eval.evaluation import (
    EvaluationResult,
    EvaluationStatus,
    EvaluationSummary,
)
from switch_trust.eval.db.base.eval.model_eval_types import DbModelEvaluation
from switch_trust.eval.db.json.repository_json import JsonRepository

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_RUN_FAILURES = 3


@dataclass_json
@dataclass
class CliRunResult:
    model_evaluation_id: str
    model_evaluation_name: str
    model: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    summary: EvaluationSummary | None = None
    results: list[EvaluationResult] = field(
        default_factory=list,
    )


async def run_cli_evaluation(
    model_evaluation: DbModelEvaluation,
    store: JsonRepository,
    concurrency: int,
    progress: Progress,
    desc_width: int,
) -> CliRunResult:
    db_eval = store.evaluations.get(
        model_evaluation.evaluation_id,
    )
    db_model = store.models.get(
        model_evaluation.model_id,
    )

    logger.info("Running evaluation: %s", db_eval.name)

    desc = pad_description(db_eval.name, desc_width)
    task_id = progress.add_task(desc, total=None)

    evaluation = store.evaluations.get_evaluation(
        model_evaluation.evaluation_id,
        message_collection_repo=store.message_collections,
        detector_repo=store.detectors,
    )
    model = store.models.get_model(model_evaluation.model_id)

    observer = RichObserver(progress, task_id)

    try:
        await evaluation.init()
        total = evaluation.get_summary().total_evaluations
        progress.update(task_id, total=total)
        evaluation.add_observer(observer)
        await evaluation.run(model, concurrency)
    except Exception as e:
        logger.error(
            "Evaluation failed: %s (%s: %s)",
            db_eval.name,
            type(e).__name__,
            e,
        )
        progress.console.print(
            f"[red]Error ({db_eval.name}): {e}[/red]",
        )
    finally:
        evaluation.remove_observer(observer)
        observer.finish(evaluation)

    summary = evaluation.get_summary()
    results = evaluation.get_results()

    logger.info(
        "Evaluation finished: %s (score=%s)",
        db_eval.name,
        summary.score if summary else "N/A",
    )

    return CliRunResult(
        model_evaluation_id=model_evaluation.id,
        model_evaluation_name=model_evaluation.name,
        model={
            "id": db_model.id,
            "name": db_model.name,
            "type": db_model.type.value,
        },
        evaluation={
            "id": db_eval.id,
            "name": db_eval.name,
            "type": db_eval.type.value,
        },
        summary=summary,
        results=results,
    )


def _aggregate_summary(
    results: list[CliRunResult],
) -> EvaluationSummary:
    total = 0
    finished = 0
    errors = 0
    max_score = 0.0
    achieved = 0.0
    error_msgs: list[str] = []
    has_error = False
    has_non_error = False

    for r in results:
        s = r.summary
        if s is None:
            continue
        total += s.total_evaluations
        finished += s.finished_evaluations
        errors += s.error_evaluations
        error_msgs.extend(s.error_messages)

        if s.status == EvaluationStatus.ERROR:
            has_error = True
        else:
            has_non_error = True
            max_score += s.max_score
            achieved += s.achieved_score

    if has_error and not has_non_error:
        status = EvaluationStatus.ERROR
    elif total > 0 and finished + errors >= total:
        status = EvaluationStatus.FINISHED
    else:
        status = EvaluationStatus.RUNNING

    return EvaluationSummary(
        status=status,
        total_evaluations=total,
        finished_evaluations=finished,
        error_evaluations=errors,
        max_score=max_score,
        achieved_score=achieved,
        error_messages=error_msgs,
    )


def _make_summary_panel(
    summary: EvaluationSummary,
    title: str,
    width: int | None = None,
) -> Panel:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="key", justify="right")
    grid.add_column()

    st = summary.status.value.upper()
    grid.add_row("Status", f"[{status_style(summary.status.value)}]{st}[/]")

    if summary.score is not None:
        pct = summary.score * 100
        style = score_style(summary.score)
        grid.add_row("Score", f"[{style}]{summary.score:.2f} ({pct:.0f}%)[/]")

    completed = summary.finished_evaluations
    total = summary.total_evaluations
    errors = summary.error_evaluations
    grid.add_row(
        "Completed",
        f"{completed} [dim]/ {total}[/dim]",
    )
    if errors:
        grid.add_row("Errors", f"[red]{errors}[/red]")

    return Panel(grid, title=f"[bold]{title}[/bold]", width=width)


def log_run_summary(results: list[CliRunResult]) -> None:
    for r in results:
        s = r.summary
        name = r.evaluation.get("name", "?")
        if s is None:
            logger.info("Evaluation: %s | status=no_result", name)
            continue

        score_str = f"{s.score:.2f}" if s.score is not None else "N/A"
        error_str = f" | error={s.error_messages[0]}" if s.error_messages else ""
        logger.info(
            "Evaluation: %s | status=%s | score=%s | completed=%d/%d%s",
            name,
            s.status.value,
            score_str,
            s.finished_evaluations,
            s.total_evaluations,
            error_str,
        )

    overall = _aggregate_summary(results)
    overall_score = f"{overall.score:.2f}" if overall.score is not None else "N/A"
    logger.info(
        "Overall: status=%s | score=%s | completed=%d/%d | errors=%d",
        overall.status.value,
        overall_score,
        overall.finished_evaluations,
        overall.total_evaluations,
        overall.error_evaluations,
    )


def print_results(results: list[CliRunResult]) -> None:
    with_summary = [r for r in results if r.summary is not None]
    console.print()

    if with_summary:
        cols_per_row = min(len(with_summary), 2)
        col_width = CLI_WIDTH // cols_per_row

        renderables = [
            _make_summary_panel(
                r.summary,
                r.evaluation.get("name", "?"),
                width=col_width,
            )
            for r in with_summary
        ]

        for row_start in range(0, len(renderables), cols_per_row):
            row = renderables[row_start : row_start + cols_per_row]
            grid = Table.grid(padding=0)
            for _ in row:
                grid.add_column()
            grid.add_row(*row)
            console.print(grid)

    overall = _aggregate_summary(results)
    console.print(
        _make_summary_panel(overall, "Overall", width=CLI_WIDTH),
    )


def write_output(
    runs: list[CliRunResult],
    config_path: str,
    output_path: str,
    fmt: str = "json",
) -> None:

    formatter = get_eval_output_formatter(OutputFormat(fmt))
    content = formatter.format(runs, config_path)
    with open(output_path, "w") as f:
        f.write(content)
