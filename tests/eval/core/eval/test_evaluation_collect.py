import unittest
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

from switch_trust.eval.core.eval.evaluation import (
    Evaluation,
    EvaluationResult,
    EvaluationStatus,
    EvaluationSummary,
)
from switch_trust.eval.core.eval.evaluation_multi import MultiEvaluation
from switch_trust.eval.core.models.model import Model


@dataclass
class LeafEvaluation(Evaluation):
    target_score: float = 1.0
    should_fail: bool = False
    score: float = 0.0

    async def init(self):
        self.status = EvaluationStatus.INITIALIZED

    def get_summary(self) -> EvaluationSummary:
        return EvaluationSummary(
            status=self.status,
            total_evaluations=1,
            finished_evaluations=1 if self.status == EvaluationStatus.FINISHED else 0,
            error_evaluations=1 if self.status == EvaluationStatus.ERROR else 0,
            max_score=0.0 if self.status == EvaluationStatus.ERROR else 1.0,
            achieved_score=self.score
            if self.status == EvaluationStatus.FINISHED
            else 0.0,
            error_messages=[self.error_message] if self.error_message else [],
        )

    def get_results(self) -> list[EvaluationResult]:
        return [
            EvaluationResult(
                score=self.score,
                status=self.status,
                error_message=self.error_message,
            )
        ]

    async def run(self, model: Model, concurrency: int = 50) -> None:
        self.status = EvaluationStatus.RUNNING
        if self.should_fail:
            self.error_message = "intentional failure"
            self.status = EvaluationStatus.ERROR
        else:
            self.score = self.target_score
            self.status = EvaluationStatus.FINISHED
        self._notify_observers()


@dataclass
class SimpleCollect(MultiEvaluation):
    leaf_scores: list[float] = field(default_factory=list)
    fail_indices: list[int] = field(default_factory=list)

    async def get_children(self) -> list[Evaluation]:
        return [
            LeafEvaluation(
                target_score=s,
                should_fail=(i in self.fail_indices),
            )
            for i, s in enumerate(self.leaf_scores)
        ]


class TestMultiEvaluation(unittest.IsolatedAsyncioTestCase):
    async def test_all_children_succeed(self):
        c = SimpleCollect(leaf_scores=[1.0, 0.5])
        await c.init()
        self.assertEqual(len(c.children), 2)

        await c.run(AsyncMock(), concurrency=2)

        self.assertEqual(c.status, EvaluationStatus.FINISHED)
        summary = c.get_summary()
        self.assertAlmostEqual(summary.achieved_score, 1.5)
        self.assertAlmostEqual(summary.max_score, 2.0)
        self.assertEqual(summary.finished_evaluations, 2)

    async def test_partial_error_finishes_and_scores_successes(self):
        c = SimpleCollect(
            leaf_scores=[1.0, 0.5],
            fail_indices=[1],
        )
        await c.init()

        # min_success_rate=0 isolates the scoring behaviour from the threshold.
        await c.run(AsyncMock(), concurrency=2, min_success_rate=0.0)

        # One prompt errored, one succeeded: the run finishes and is scored over
        # the successful prompt only.
        self.assertEqual(c.status, EvaluationStatus.FINISHED)
        summary = c.get_summary()
        self.assertAlmostEqual(summary.score, 1.0)
        self.assertTrue(
            any("intentional failure" in m for m in summary.error_messages),
        )

    async def test_empty_children_is_error(self):
        c = SimpleCollect(leaf_scores=[])
        await c.init()

        await c.run(AsyncMock(), concurrency=1)

        # A run with no measurable prompt is unscorable, so it errors rather than
        # finishing with a (coerced) zero score that would count in weighted health.
        self.assertEqual(c.status, EvaluationStatus.ERROR)
        summary = c.get_summary()
        self.assertIsNone(summary.score)

    async def test_progress(self):
        c = SimpleCollect(leaf_scores=[1.0, 1.0])
        await c.init()

        summary_before = c.get_summary()
        self.assertEqual(summary_before.total_evaluations, 2)
        self.assertEqual(summary_before.finished_evaluations, 0)

        await c.run(AsyncMock(), concurrency=2)

        summary_after = c.get_summary()
        self.assertEqual(summary_after.finished_evaluations, 2)

    async def test_observer_called_per_child(self):
        notifications = []
        c = SimpleCollect(
            leaf_scores=[1.0, 1.0, 1.0],
            observers=[lambda ev: notifications.append(True)],
        )
        await c.init()

        await c.run(AsyncMock(), concurrency=3)

        # At least one notification per child + final
        self.assertGreaterEqual(len(notifications), 3)

    async def test_skips_non_initialized_children(self):
        c = SimpleCollect(leaf_scores=[1.0])
        await c.init()
        c.children[0].status = EvaluationStatus.FINISHED

        await c.run(AsyncMock(), concurrency=1)

        self.assertEqual(c.status, EvaluationStatus.FINISHED)

    async def test_skips_if_already_errored(self):
        c = SimpleCollect(leaf_scores=[1.0])
        await c.init()
        c.status = EvaluationStatus.ERROR

        await c.run(AsyncMock(), concurrency=1)

        # Children should not have run
        self.assertEqual(
            c.children[0].get_summary().status,
            EvaluationStatus.INITIALIZED,
        )

    async def test_get_summary_aggregates_children(self):
        c = SimpleCollect(leaf_scores=[1.0, 0.5, 0.8])
        await c.init()

        await c.run(AsyncMock(), concurrency=3)

        summary = c.get_summary()
        self.assertEqual(summary.total_evaluations, 3)
        self.assertEqual(summary.finished_evaluations, 3)
        self.assertEqual(summary.error_evaluations, 0)
        self.assertAlmostEqual(summary.max_score, 3.0)
        self.assertAlmostEqual(summary.achieved_score, 2.3)
        self.assertAlmostEqual(summary.score, 2.3 / 3.0)

    async def test_get_summary_with_errors(self):
        c = SimpleCollect(
            leaf_scores=[1.0, 0.5],
            fail_indices=[0],
        )
        await c.init()

        # min_success_rate=0 isolates the scoring behaviour from the threshold.
        await c.run(AsyncMock(), concurrency=2, min_success_rate=0.0)

        # One prompt succeeded, so the run finishes and is scored over that
        # prompt only — the errored prompt contributes to neither side of the mean.
        self.assertEqual(c.status, EvaluationStatus.FINISHED)
        summary = c.get_summary()
        self.assertEqual(summary.error_evaluations, 1)
        self.assertEqual(summary.finished_evaluations, 1)
        self.assertAlmostEqual(summary.achieved_score, 0.5)
        self.assertAlmostEqual(summary.max_score, 1.0)
        self.assertAlmostEqual(summary.score, 0.5)


if __name__ == "__main__":
    unittest.main()
