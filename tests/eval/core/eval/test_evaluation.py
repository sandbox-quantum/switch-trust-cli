import unittest

from switch_trust.eval.core.eval.evaluation import (
    Evaluation,
    EvaluationResult,
    EvaluationStatus,
    EvaluationSummary,
)


class TestEvaluation(unittest.IsolatedAsyncioTestCase):
    async def test_notify_observers(self):
        class DummyEval(Evaluation):
            async def init(self):
                pass

            def get_summary(self) -> EvaluationSummary:
                return EvaluationSummary(
                    status=EvaluationStatus.WAITING,
                    total_evaluations=0,
                    finished_evaluations=0,
                    error_evaluations=0,
                    max_score=0.0,
                    achieved_score=0.0,
                )

            def get_results(self) -> list[EvaluationResult]:
                return []

            async def run(self, model, concurrency=50):
                self._notify_observers()

        called = []
        e = DummyEval()
        e.add_observer(lambda ev: called.append(ev))
        await e.run(None)
        self.assertEqual(len(called), 1)
        self.assertIs(called[0], e)


if __name__ == "__main__":
    unittest.main()
