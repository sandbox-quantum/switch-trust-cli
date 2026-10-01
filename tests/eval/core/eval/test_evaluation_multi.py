import asyncio
import unittest
from unittest.mock import MagicMock

from switch_trust.eval.core.eval.evaluation import (
    Evaluation,
    EvaluationResult,
    EvaluationStatus,
    EvaluationSummary,
)
from switch_trust.eval.core.eval.evaluation_multi import (
    MAX_CONSECUTIVE_FAILURES,
    MultiEvaluation,
)
from switch_trust.eval.core.models.model import Model


class StubMultiEvaluation(MultiEvaluation):
    """Concrete MultiEvaluation for testing."""

    def __init__(self, children: list[Evaluation] | None = None):
        super().__init__()
        self._test_children = children or []

    async def get_children(self) -> list[Evaluation]:
        return self._test_children


class FakeChild(Evaluation):
    """A fake leaf evaluation for testing."""

    def __init__(
        self,
        summary: EvaluationSummary | None = None,
        results: list[EvaluationResult] | None = None,
        run_error: Exception | None = None,
    ):
        super().__init__()
        self._summary = summary or EvaluationSummary(
            status=EvaluationStatus.INITIALIZED,
            total_evaluations=1,
            finished_evaluations=0,
            error_evaluations=0,
            max_score=1.0,
            achieved_score=0.0,
        )
        self._results = results or []
        self._run_error = run_error

    async def init(self):
        pass

    def get_summary(self) -> EvaluationSummary:
        return self._summary

    def get_results(self) -> list[EvaluationResult]:
        return self._results

    async def run(self, model: Model, concurrency: int = 50):
        if self._run_error:
            raise self._run_error
        self._summary = EvaluationSummary(
            status=EvaluationStatus.FINISHED,
            total_evaluations=1,
            finished_evaluations=1,
            error_evaluations=0,
            max_score=1.0,
            achieved_score=0.8,
        )


class GracefulChild(Evaluation):
    """A fake child that handles errors internally (like SingleEvaluation).

    Unlike FakeChild which raises on error, this child catches its own
    failures and sets status to ERROR — matching real SingleEvaluation
    behavior and needed for circuit-breaker tests.
    """

    def __init__(self, should_fail: bool = False):
        super().__init__()
        self.status = EvaluationStatus.INITIALIZED
        self._should_fail = should_fail

    async def init(self):
        pass

    def get_summary(self) -> EvaluationSummary:
        return EvaluationSummary(
            status=self.status,
            total_evaluations=1,
            finished_evaluations=1 if self.status == EvaluationStatus.FINISHED else 0,
            error_evaluations=1 if self.status == EvaluationStatus.ERROR else 0,
            max_score=1.0,
            achieved_score=0.8 if self.status == EvaluationStatus.FINISHED else 0.0,
            error_messages=[self.error_message] if self.error_message else [],
        )

    def get_results(self) -> list[EvaluationResult]:
        return [EvaluationResult(score=0.0, status=self.status)]

    async def run(self, model: Model, concurrency: int = 50):
        if self._should_fail:
            self.status = EvaluationStatus.ERROR
            self.error_message = "Connection refused"
        else:
            self.status = EvaluationStatus.FINISHED
        self._notify_observers()


class TestMultiEvaluation(unittest.TestCase):
    def test_get_summary_aggregates_children(self):
        child1 = FakeChild(
            summary=EvaluationSummary(
                status=EvaluationStatus.FINISHED,
                total_evaluations=2,
                finished_evaluations=2,
                error_evaluations=0,
                max_score=2.0,
                achieved_score=1.5,
            ),
        )
        child2 = FakeChild(
            summary=EvaluationSummary(
                status=EvaluationStatus.FINISHED,
                total_evaluations=3,
                finished_evaluations=3,
                error_evaluations=1,
                max_score=3.0,
                achieved_score=2.0,
            ),
        )
        multi = StubMultiEvaluation(children=[child1, child2])
        multi.children = [child1, child2]
        multi.status = EvaluationStatus.FINISHED

        summary = multi.get_summary()
        self.assertEqual(summary.total_evaluations, 5)
        self.assertEqual(summary.finished_evaluations, 5)
        self.assertEqual(summary.error_evaluations, 1)
        self.assertAlmostEqual(summary.max_score, 5.0)
        self.assertAlmostEqual(summary.achieved_score, 3.5)

    def test_get_summary_includes_own_error_message(self):
        multi = StubMultiEvaluation()
        multi.status = EvaluationStatus.ERROR
        multi.error_message = "init failed"

        summary = multi.get_summary()
        self.assertIn("init failed", summary.error_messages)

    def test_get_results_collects_child_results(self):
        result1 = EvaluationResult(
            score=0.9,
            status=EvaluationStatus.FINISHED,
        )
        result2 = EvaluationResult(
            score=0.5,
            status=EvaluationStatus.FINISHED,
        )
        child1 = FakeChild(results=[result1])
        child2 = FakeChild(results=[result2])
        multi = StubMultiEvaluation(children=[child1, child2])
        multi.children = [child1, child2]

        results = multi.get_results()
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0].score, 0.9)
        self.assertAlmostEqual(results[1].score, 0.5)

    def test_run_skips_if_status_is_error(self):
        multi = StubMultiEvaluation()
        multi.status = EvaluationStatus.ERROR
        model = MagicMock(spec=Model)

        asyncio.run(multi.run(model))
        # Should return immediately; status stays ERROR
        self.assertEqual(multi.status, EvaluationStatus.ERROR)

    def test_run_with_successful_children(self):
        child = FakeChild()
        multi = StubMultiEvaluation(children=[child])
        asyncio.run(multi.init())
        model = MagicMock(spec=Model)

        asyncio.run(multi.run(model))
        self.assertEqual(multi.status, EvaluationStatus.FINISHED)

    def test_run_with_child_that_raises(self):
        child = FakeChild(run_error=RuntimeError("boom"))
        multi = StubMultiEvaluation(children=[child])
        asyncio.run(multi.init())
        model = MagicMock(spec=Model)

        asyncio.run(multi.run(model))
        self.assertEqual(multi.status, EvaluationStatus.ERROR)
        self.assertIsNotNone(multi.error_message)

    def test_init_failure_sets_error_status(self):
        class FailingMulti(MultiEvaluation):
            async def get_children(self):
                raise RuntimeError("cannot init")

        multi = FailingMulti()
        asyncio.run(multi.init())

        self.assertEqual(multi.status, EvaluationStatus.ERROR)
        self.assertEqual(multi.error_message, "cannot init")

    def test_init_sets_children_and_status(self):
        child = FakeChild()
        multi = StubMultiEvaluation(children=[child])
        asyncio.run(multi.init())

        self.assertEqual(multi.status, EvaluationStatus.INITIALIZED)
        self.assertEqual(len(multi.children), 1)

    def test_partial_failure_finishes_below_threshold_when_disabled(self):
        children = [GracefulChild(should_fail=True), GracefulChild(should_fail=False)]
        multi = StubMultiEvaluation(children=children)
        asyncio.run(multi.init())
        model = MagicMock(spec=Model)

        # Threshold disabled: any success finishes the run.
        asyncio.run(multi.run(model, concurrency=1, min_success_rate=0.0))

        self.assertEqual(multi.status, EvaluationStatus.FINISHED)

    def test_below_min_success_rate_is_error(self):
        # 1 of 2 succeed (50%) < default 0.9 -> ERROR.
        children = [GracefulChild(should_fail=True), GracefulChild(should_fail=False)]
        multi = StubMultiEvaluation(children=children)
        asyncio.run(multi.init())
        model = MagicMock(spec=Model)

        asyncio.run(multi.run(model, concurrency=1))

        self.assertEqual(multi.status, EvaluationStatus.ERROR)

    def test_below_min_success_rate_names_the_shortfall(self):
        children = [GracefulChild(should_fail=True), GracefulChild(should_fail=False)]
        multi = StubMultiEvaluation(children=children)
        asyncio.run(multi.init())
        model = MagicMock(spec=Model)

        asyncio.run(multi.run(model, concurrency=1))

        # Reported as a shortfall, not as "no results" -- one prompt did score.
        self.assertEqual(
            multi.error_message,
            "only 1 of 2 prompts succeeded (50.0%), below the 90.0% minimum",
        )

    def test_every_prompt_failing_says_so(self):
        children = [GracefulChild(should_fail=True), GracefulChild(should_fail=True)]
        multi = StubMultiEvaluation(children=children)
        asyncio.run(multi.init())
        model = MagicMock(spec=Model)

        asyncio.run(multi.run(model, concurrency=1))

        self.assertEqual(multi.status, EvaluationStatus.ERROR)
        self.assertEqual(multi.error_message, "all 2 prompts failed")

    def test_no_prompts_says_so(self):
        multi = StubMultiEvaluation(children=[])
        asyncio.run(multi.init())
        model = MagicMock(spec=Model)

        asyncio.run(multi.run(model, concurrency=1))

        self.assertEqual(multi.status, EvaluationStatus.ERROR)
        self.assertEqual(multi.error_message, "evaluation had no prompts to run")

    def test_meets_min_success_rate_finishes(self):
        # 9 of 10 succeed (90%) >= default 0.9 -> FINISHED.
        children = [GracefulChild(should_fail=(i == 0)) for i in range(10)]
        multi = StubMultiEvaluation(children=children)
        asyncio.run(multi.init())
        model = MagicMock(spec=Model)

        asyncio.run(multi.run(model, concurrency=1))

        self.assertEqual(multi.status, EvaluationStatus.FINISHED)

    def test_min_success_rate_is_configurable(self):
        # 1 of 2 succeed (50%) >= a lowered 0.4 threshold -> FINISHED.
        children = [GracefulChild(should_fail=True), GracefulChild(should_fail=False)]
        multi = StubMultiEvaluation(children=children)
        asyncio.run(multi.init())
        model = MagicMock(spec=Model)

        asyncio.run(multi.run(model, concurrency=1, min_success_rate=0.4))

        self.assertEqual(multi.status, EvaluationStatus.FINISHED)

    def test_all_errored_is_error(self):
        children = [GracefulChild(should_fail=True), GracefulChild(should_fail=True)]
        multi = StubMultiEvaluation(children=children)
        asyncio.run(multi.init())
        model = MagicMock(spec=Model)

        asyncio.run(multi.run(model, concurrency=1))

        self.assertEqual(multi.status, EvaluationStatus.ERROR)

    def test_warns_on_errored_prompts_with_rate(self):
        children = [GracefulChild(should_fail=True), GracefulChild(should_fail=False)]
        multi = StubMultiEvaluation(children=children)
        asyncio.run(multi.init())
        model = MagicMock(spec=Model)

        with self.assertLogs(
            "switch_trust.eval.core.eval.evaluation_multi", level="WARNING"
        ) as logs:
            asyncio.run(multi.run(model, concurrency=1))

        self.assertTrue(any("1 of 2 prompts errored" in m for m in logs.output))

    def test_run_with_errored_child_status(self):
        child = FakeChild(
            summary=EvaluationSummary(
                status=EvaluationStatus.ERROR,
                total_evaluations=1,
                finished_evaluations=0,
                error_evaluations=1,
                max_score=1.0,
                achieved_score=0.0,
                error_messages=["child error"],
            ),
        )
        multi = StubMultiEvaluation(children=[child])
        asyncio.run(multi.init())
        model = MagicMock(spec=Model)

        # The child status is ERROR so it won't be in 'initialized' list
        # but the multi checks for errored children at the end
        asyncio.run(multi.run(model))
        self.assertEqual(multi.status, EvaluationStatus.ERROR)


class TestCircuitBreaker(unittest.TestCase):
    def test_aborts_after_consecutive_failures(self):
        num_children = MAX_CONSECUTIVE_FAILURES + 10
        children = [GracefulChild(should_fail=True) for _ in range(num_children)]
        multi = StubMultiEvaluation(children=children)
        asyncio.run(multi.init())
        model = MagicMock(spec=Model)

        asyncio.run(multi.run(model, concurrency=1))

        self.assertEqual(multi.status, EvaluationStatus.ERROR)
        aborted = [
            c
            for c in children
            if c.error_message == "Aborted: too many consecutive failures"
        ]
        self.assertGreater(len(aborted), 0)

    def test_aborts_regardless_of_concurrency(self):
        children = [
            GracefulChild(should_fail=True) for _ in range(MAX_CONSECUTIVE_FAILURES + 5)
        ]
        multi = StubMultiEvaluation(children=children)
        asyncio.run(multi.init())
        model = MagicMock(spec=Model)

        asyncio.run(multi.run(model, concurrency=50))

        self.assertEqual(multi.status, EvaluationStatus.ERROR)
        aborted = [
            c
            for c in children
            if c.error_message == "Aborted: too many consecutive failures"
        ]
        self.assertGreater(len(aborted), 0)

    def test_success_resets_consecutive_errors(self):
        children = []
        for i in range(MAX_CONSECUTIVE_FAILURES * 3):
            children.append(GracefulChild(should_fail=(i % 2 == 0)))
        multi = StubMultiEvaluation(children=children)
        asyncio.run(multi.init())
        model = MagicMock(spec=Model)

        asyncio.run(multi.run(model, concurrency=1))

        aborted = [
            c
            for c in children
            if c.error_message == "Aborted: too many consecutive failures"
        ]
        self.assertEqual(len(aborted), 0)


if __name__ == "__main__":
    unittest.main()
