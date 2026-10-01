from __future__ import annotations

import sys
import threading
from collections.abc import Callable

from switch_trust.eval.core.eval.evaluation import (
    Evaluation,
    EvaluationStatus,
    EvaluationSummary,
)

# -- Formatting -----------------------------------------------------------


def format_summary(summary: EvaluationSummary) -> str:
    done = summary.finished_evaluations + summary.error_evaluations
    parts = [
        f"[{summary.status.value}]",
        f"{done}/{summary.total_evaluations} evaluations",
        f"({summary.progress * 100:.0f}%)",
    ]
    if summary.score is not None:
        parts.append(f"| score: {summary.score:.2f}")
    if summary.error_evaluations > 0:
        parts.append(f"| {summary.error_evaluations} errors")
    return " ".join(parts)


def print_summary(
    summary: EvaluationSummary,
    include_errors: bool = False,
    file=None,
) -> None:
    file = file or sys.stdout
    print(format_summary(summary), file=file)
    if include_errors and summary.error_messages:
        for msg in summary.error_messages:
            print(f"  Error: {msg}", file=file)


# -- Callback observer ----------------------------------------------------


class ConsoleObserver:
    """Callback observer that prints evaluation progress inline.

    Add an instance to ``evaluation.observers`` to get live
    console updates after each child completes.
    """

    def __init__(self, file=None):
        self._file = file or sys.stdout

    def __call__(self, evaluation: Evaluation) -> None:
        line = format_summary(evaluation.get_summary())
        print(f"\r{line}", end="", flush=True, file=self._file)

    def finish(self) -> None:
        print(file=self._file)


# -- Polling observer ------------------------------------------------------


class PollingObserver:
    """Polls an Evaluation on a background daemon thread and calls
    a callback with the current summary at each interval.
    """

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(
        self,
        evaluation: Evaluation,
        callback: Callable[[EvaluationSummary], None],
    ) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll,
            args=(evaluation, callback),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join()
            self._thread = None

    def _poll(
        self,
        evaluation: Evaluation,
        callback: Callable[[EvaluationSummary], None],
    ) -> None:
        while not self._stop_event.is_set():
            summary = evaluation.get_summary()
            callback(summary)
            if summary.status in (
                EvaluationStatus.FINISHED,
                EvaluationStatus.ERROR,
            ):
                break
            self._stop_event.wait(self.interval)
        else:
            callback(evaluation.get_summary())
        self._thread = None


class ConsolePollingObserver:
    """Polling observer that prints evaluation progress to the
    console on a background thread.
    """

    def __init__(self, interval: float = 5.0, file=None):
        self._file = file or sys.stdout
        self._poller = PollingObserver(interval=interval)

    def start(self, evaluation: Evaluation) -> None:
        def _print(summary: EvaluationSummary) -> None:
            line = format_summary(summary)
            print(
                f"\r{line}",
                end="",
                flush=True,
                file=self._file,
            )

        self._poller.start(evaluation, _print)

    def stop(self) -> None:
        self._poller.stop()
        print(file=self._file)
