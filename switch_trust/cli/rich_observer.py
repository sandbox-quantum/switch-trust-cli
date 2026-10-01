from __future__ import annotations

from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.spinner import Spinner
from rich.text import Text

from switch_trust.cli.console import console, score_style
from switch_trust.eval.core.eval.evaluation import Evaluation


class StatusColumn(ProgressColumn):
    """Spinner while running, green check when done, red X on error."""

    def __init__(self) -> None:
        super().__init__()
        self._spinner = Spinner("dots")

    def render(self, task: object) -> Text:
        if task.finished:
            status = task.fields.get("final_status", "finished")
            if status == "error":
                return Text("✗", style="bold red")
            return Text("✔", style="bold green")
        return self._spinner.render(task.get_time())


class ProgressOrScoreColumn(ProgressColumn):
    """Shows M/N while running, score percentage when finished."""

    def render(self, task: object) -> Text:
        if task.finished:
            score = task.fields.get("score")
            if score is not None:
                pct = score * 100
                return Text(f"{pct:.0f}%", style=score_style(score))
            return Text("—", style="dim")
        completed = int(task.completed)
        if task.total is not None:
            return Text(f"{completed}/{int(task.total)}")
        return Text(f"{completed}/?")


def create_progress() -> Progress:
    return Progress(
        StatusColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        ProgressOrScoreColumn(),
        TextColumn("[dim]|[/dim]"),
        TimeElapsedColumn(),
        console=console,
    )


def pad_description(name: str, width: int) -> str:
    if len(name) > width:
        return name[: width - 1] + "…"
    return name.ljust(width)


class RichObserver:
    """Evaluation observer that updates a task in a shared Progress."""

    def __init__(self, progress: Progress, task_id: TaskID):
        self._progress = progress
        self._task_id = task_id

    def __call__(self, evaluation: Evaluation) -> None:
        summary = evaluation.get_summary()
        done = summary.finished_evaluations + summary.error_evaluations
        self._progress.update(self._task_id, completed=done)

    def finish(self, evaluation: Evaluation) -> None:
        summary = evaluation.get_summary()
        total = summary.total_evaluations or 1
        self._progress.update(
            self._task_id,
            completed=total,
            total=total,
            final_status=summary.status.value,
            score=summary.score,
        )
