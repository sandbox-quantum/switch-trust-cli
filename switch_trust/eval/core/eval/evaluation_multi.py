from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from dataclasses import dataclass, field

from dataclasses_json import dataclass_json

from switch_trust.eval.core.eval.evaluation import (
    DEFAULT_MIN_SUCCESS_RATE,
    Evaluation,
    EvaluationObserver,
    EvaluationResult,
    EvaluationStatus,
    EvaluationSummary,
)
from switch_trust.eval.core.models.model import Model

logger = logging.getLogger(__name__)


PER_CHILD_TIMEOUT = 60  # seconds per child
MIN_TIMEOUT = 30 * 60  # 30 minutes minimum
MAX_CONSECUTIVE_FAILURES = 5


@dataclass_json
@dataclass
class MultiEvaluation(Evaluation):
    """An evaluation that produces and runs a list of child evaluations.

    Subclasses implement ``get_children`` to define the children.
    ``init`` calls ``get_children`` and then initializes each child.
    ``run`` executes all initialized children concurrently using
    asyncio with a semaphore for concurrency control.

    Observer propagation: rather than forwarding observers directly
    to children (which would cause the observer to receive the
    child as the ``evaluation`` argument and report per-child
    progress), we attach a relay observer to each child that
    triggers ``_notify_observers`` on **this** parent evaluation.
    The observer therefore always sees the top-level aggregated
    summary.
    """

    children: list[Evaluation] = field(default_factory=list)
    _child_relay: EvaluationObserver | None = field(
        default=None,
        repr=False,
    )

    def __init__(self):
        super().__init__()
        self.children = []
        self._child_relay = None

    def _ensure_relay(self) -> EvaluationObserver:
        """Return a relay observer that notifies this parent."""
        if self._child_relay is None:

            def relay(_child: Evaluation) -> None:
                self._notify_observers()

            self._child_relay = relay
        return self._child_relay

    async def init(self):
        try:
            self.status = EvaluationStatus.INITIALIZING
            self.children = await self.get_children()
            logger.debug(
                "%s initialized: %d children", type(self).__name__, len(self.children)
            )
            relay = self._ensure_relay()
            for child in self.children:
                child.add_observer(relay)
                await child.init()
            self.status = EvaluationStatus.INITIALIZED
        except Exception as e:
            self.error_message = str(e)
            self.status = EvaluationStatus.ERROR
            logger.error(
                "%s init failed (%s: %s)",
                type(self).__name__,
                type(e).__name__,
                e,
                exc_info=True,
            )
        finally:
            self._notify_observers()

    @abstractmethod
    async def get_children(self) -> list[Evaluation]:
        """Return the child evaluations to run."""
        pass

    def get_summary(self) -> EvaluationSummary:
        summaries = [c.get_summary() for c in self.children]
        total = sum(s.total_evaluations for s in summaries)
        finished = sum(s.finished_evaluations for s in summaries)
        errors = sum(s.error_evaluations for s in summaries)
        max_score = sum(s.max_score for s in summaries)
        achieved = sum(s.achieved_score for s in summaries)
        error_msgs = [msg for s in summaries for msg in s.error_messages]
        if self.error_message:
            error_msgs.append(self.error_message)

        return EvaluationSummary(
            status=self.status,
            total_evaluations=total,
            finished_evaluations=finished,
            error_evaluations=errors,
            max_score=max_score,
            achieved_score=achieved,
            error_messages=error_msgs,
        )

    def get_results(self) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        for child in self.children:
            results.extend(child.get_results())
        return results

    async def run(
        self,
        model: Model,
        concurrency: int = 50,
        min_success_rate: float = DEFAULT_MIN_SUCCESS_RATE,
    ) -> None:
        if self.status == EvaluationStatus.ERROR:
            return
        self.status = EvaluationStatus.RUNNING
        self._notify_observers()

        semaphore = asyncio.Semaphore(concurrency)
        abort = asyncio.Event()
        consecutive_errors = 0
        error_lock = asyncio.Lock()

        initialized = [
            c
            for c in self.children
            if c.get_summary().status == EvaluationStatus.INITIALIZED
        ]
        timeout = max(
            MIN_TIMEOUT,
            PER_CHILD_TIMEOUT * len(initialized),
        )

        name = type(self).__name__
        logger.debug(
            "%s running: %d children, concurrency=%d",
            name,
            len(initialized),
            concurrency,
        )

        async def run_child(child: Evaluation) -> None:
            nonlocal consecutive_errors
            async with semaphore:
                if abort.is_set():
                    child.status = EvaluationStatus.ERROR
                    child.error_message = "Aborted: too many consecutive failures"
                    child._notify_observers()
                    return
                await child.run(model, concurrency)
                async with error_lock:
                    if child.get_summary().status == EvaluationStatus.ERROR:
                        consecutive_errors += 1
                        if consecutive_errors >= MAX_CONSECUTIVE_FAILURES:
                            abort.set()
                            logger.warning(
                                "%s: aborting after %d consecutive failures",
                                name,
                                consecutive_errors,
                            )
                    else:
                        consecutive_errors = 0

        try:
            tasks = [asyncio.create_task(run_child(child)) for child in initialized]
            if tasks:
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=timeout,
                    )
                    errors = [r for r in results if isinstance(r, Exception)]
                    if errors:
                        self.error_message = (
                            f"Evaluation errors: {'; '.join(str(e) for e in errors)}"
                        )
                        logger.warning("%s: %s", name, self.error_message)
                except TimeoutError:
                    for task in tasks:
                        task.cancel()
                    unfinished = sum(
                        1
                        for c in self.children
                        if c.get_summary().status
                        not in (
                            EvaluationStatus.FINISHED,
                            EvaluationStatus.ERROR,
                        )
                    )
                    self.error_message = (
                        f"{unfinished} (of {len(initialized)}) "
                        f"evaluations timed out after "
                        f"{timeout}s"
                    )
                    logger.warning("%s: %s", name, self.error_message)

            if abort.is_set() and not self.error_message:
                self.error_message = (
                    f"aborted after {MAX_CONSECUTIVE_FAILURES} consecutive failures"
                )

            errored = [
                c
                for c in self.children
                if c.get_summary().status == EvaluationStatus.ERROR
            ]
            finished = [
                c
                for c in self.children
                if c.get_summary().status == EvaluationStatus.FINISHED
            ]
            total = len(self.children)
            if errored:
                logger.warning(
                    "%s: %d of %d prompts errored (%.0f%%)",
                    name,
                    len(errored),
                    total,
                    100.0 * len(errored) / total,
                )

            # ERROR on a structural failure (error_message), no successful prompt,
            # or a success rate below the minimum. Otherwise FINISHED.
            #
            # Split into one branch per cause so each sets error_message. The
            # worker reports that field and falls back to a generic "evaluation
            # produced no results" when it is empty, which is what every one of
            # these used to look like from the outside.
            success_rate = len(finished) / total if total else 0.0
            if self.error_message:
                self.status = EvaluationStatus.ERROR
            elif not finished:
                self.error_message = (
                    "evaluation had no prompts to run"
                    if not total
                    else f"all {total} prompts failed"
                )
                self.status = EvaluationStatus.ERROR
            elif success_rate < min_success_rate:
                self.error_message = (
                    f"only {len(finished)} of {total} prompts succeeded "
                    # .1%, not .0%: at 200+ prompts a rate under the minimum can
                    # round up to it (179/200 shows as 90%, below a 90% minimum),
                    # and the message then contradicts itself.
                    f"({success_rate:.1%}), below the "
                    f"{min_success_rate:.1%} minimum"
                )
                self.status = EvaluationStatus.ERROR
            else:
                self.status = EvaluationStatus.FINISHED
        except Exception as e:
            self.error_message = str(e)
            self.status = EvaluationStatus.ERROR
            logger.error(
                "%s run failed (%s: %s)", name, type(e).__name__, e, exc_info=True
            )
        finally:
            logger.info("%s finished: status=%s", name, self.status.value)
            self._notify_observers()
