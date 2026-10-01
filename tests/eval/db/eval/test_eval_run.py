import unittest
from unittest.mock import AsyncMock, MagicMock

from switch_trust.eval.core.eval.evaluation import (
    Evaluation,
    EvaluationResult,
    EvaluationStatus,
    EvaluationSummary,
)
from switch_trust.eval.db.base.eval.eval_run import run_model_evaluation
from switch_trust.eval.db.base.eval.model_eval_run_types import DbModelEvaluationRun
from switch_trust.eval.db.base.eval.model_eval_types import DbModelEvaluation


def _make_model_evaluation(
    id: str = "me-1",
    model_id: str = "mod-1",
    evaluation_id: str = "eval-1",
) -> DbModelEvaluation:
    return DbModelEvaluation(
        model_id=model_id,
        evaluation_id=evaluation_id,
        name="test",
        id=id,
    )


def _make_summary(
    status: EvaluationStatus = EvaluationStatus.FINISHED,
) -> EvaluationSummary:
    return EvaluationSummary(
        status=status,
        total_evaluations=2,
        finished_evaluations=2,
        error_evaluations=0,
        max_score=2.0,
        achieved_score=1.5,
    )


class TestRunModelEvaluation(unittest.IsolatedAsyncioTestCase):
    def _setup_mocks(self):
        me = _make_model_evaluation()
        me_repo = MagicMock()
        me_repo.get.return_value = me

        mock_evaluation = MagicMock(spec=Evaluation)
        mock_evaluation.observers = []
        mock_evaluation.init = AsyncMock()
        mock_evaluation.run = AsyncMock()
        mock_evaluation.get_summary.return_value = _make_summary()
        mock_evaluation.get_results.return_value = [
            EvaluationResult(
                score=0.8,
                status=EvaluationStatus.FINISHED,
            ),
            EvaluationResult(
                score=0.7,
                status=EvaluationStatus.FINISHED,
            ),
        ]

        def fake_add_observer(obs):
            mock_evaluation.observers.append(obs)

        def fake_remove_observer(obs):
            mock_evaluation.observers.remove(obs)

        mock_evaluation.add_observer.side_effect = fake_add_observer
        mock_evaluation.remove_observer.side_effect = fake_remove_observer

        eval_repo = MagicMock()
        eval_repo.get_evaluation.return_value = mock_evaluation

        mock_model = MagicMock()
        model_repo = MagicMock()
        model_repo.get_model.return_value = mock_model

        run_repo = MagicMock()

        return (
            me,
            me_repo,
            mock_evaluation,
            eval_repo,
            mock_model,
            model_repo,
            run_repo,
        )

    async def test_creates_and_returns_run(self):
        (
            me,
            me_repo,
            mock_eval,
            eval_repo,
            mock_model,
            model_repo,
            run_repo,
        ) = self._setup_mocks()

        result = await run_model_evaluation(
            model_evaluation_id="me-1",
            concurrency=1,
            model_evaluation_repo=me_repo,
            evaluation_repo=eval_repo,
            model_repo=model_repo,
            run_repo=run_repo,
        )

        self.assertIsInstance(result, DbModelEvaluationRun)
        self.assertEqual(
            result.model_evaluation_id,
            "me-1",
        )
        self.assertIsNotNone(result.started)
        self.assertIsNotNone(result.finished)
        self.assertIsNotNone(result.summary)

    async def test_looks_up_model_evaluation(self):
        (
            me,
            me_repo,
            mock_eval,
            eval_repo,
            mock_model,
            model_repo,
            run_repo,
        ) = self._setup_mocks()

        await run_model_evaluation(
            model_evaluation_id="me-1",
            concurrency=1,
            model_evaluation_repo=me_repo,
            evaluation_repo=eval_repo,
            model_repo=model_repo,
            run_repo=run_repo,
        )

        me_repo.get.assert_called_once_with("me-1")

    async def test_creates_evaluation_and_model(self):
        (
            me,
            me_repo,
            mock_eval,
            eval_repo,
            mock_model,
            model_repo,
            run_repo,
        ) = self._setup_mocks()
        mc_repo = MagicMock()
        det_repo = MagicMock()

        await run_model_evaluation(
            model_evaluation_id="me-1",
            concurrency=1,
            model_evaluation_repo=me_repo,
            evaluation_repo=eval_repo,
            model_repo=model_repo,
            run_repo=run_repo,
            message_collection_repo=mc_repo,
            detector_repo=det_repo,
        )

        eval_repo.get_evaluation.assert_called_once_with(
            "eval-1",
            message_collection_repo=mc_repo,
            detector_repo=det_repo,
        )
        model_repo.get_model.assert_called_once_with("mod-1")

    async def test_persists_run_on_create(self):
        (
            me,
            me_repo,
            mock_eval,
            eval_repo,
            mock_model,
            model_repo,
            run_repo,
        ) = self._setup_mocks()

        await run_model_evaluation(
            model_evaluation_id="me-1",
            concurrency=1,
            model_evaluation_repo=me_repo,
            evaluation_repo=eval_repo,
            model_repo=model_repo,
            run_repo=run_repo,
        )

        run_repo.create.assert_called_once()
        created_run = run_repo.create.call_args[0][0]
        self.assertEqual(
            created_run.model_evaluation_id,
            "me-1",
        )
        self.assertIsNotNone(created_run.started)

    async def test_inits_and_runs_evaluation(self):
        (
            me,
            me_repo,
            mock_eval,
            eval_repo,
            mock_model,
            model_repo,
            run_repo,
        ) = self._setup_mocks()

        await run_model_evaluation(
            model_evaluation_id="me-1",
            concurrency=1,
            model_evaluation_repo=me_repo,
            evaluation_repo=eval_repo,
            model_repo=model_repo,
            run_repo=run_repo,
        )

        mock_eval.init.assert_called_once()
        mock_eval.run.assert_called_once_with(
            mock_model,
            1,
        )

    async def test_registers_observer(self):
        (
            me,
            me_repo,
            mock_eval,
            eval_repo,
            mock_model,
            model_repo,
            run_repo,
        ) = self._setup_mocks()

        await run_model_evaluation(
            model_evaluation_id="me-1",
            concurrency=1,
            model_evaluation_repo=me_repo,
            evaluation_repo=eval_repo,
            model_repo=model_repo,
            run_repo=run_repo,
        )

        mock_eval.add_observer.assert_called_once()

    async def test_observer_writes_summary_on_progress(self):
        (
            me,
            me_repo,
            mock_eval,
            eval_repo,
            mock_model,
            model_repo,
            run_repo,
        ) = self._setup_mocks()

        running_summary = _make_summary(
            status=EvaluationStatus.RUNNING,
        )
        mock_eval.get_summary.return_value = running_summary

        # Capture the observer and invoke it during run
        async def fake_run(model, concurrency):
            observer = mock_eval.observers[0]
            observer(mock_eval)

        mock_eval.run = AsyncMock(side_effect=fake_run)

        await run_model_evaluation(
            model_evaluation_id="me-1",
            concurrency=1,
            model_evaluation_repo=me_repo,
            evaluation_repo=eval_repo,
            model_repo=model_repo,
            run_repo=run_repo,
        )

        # Observer should have triggered an update
        # plus the final update after run completes
        self.assertGreaterEqual(
            run_repo.update.call_count,
            2,
        )

    async def test_observer_sets_finished_on_completion(self):
        (
            me,
            me_repo,
            mock_eval,
            eval_repo,
            mock_model,
            model_repo,
            run_repo,
        ) = self._setup_mocks()

        finished_summary = _make_summary(
            status=EvaluationStatus.FINISHED,
        )

        async def fake_run(model, concurrency):
            mock_eval.get_summary.return_value = finished_summary
            observer = mock_eval.observers[0]
            observer(mock_eval)

        mock_eval.run = AsyncMock(side_effect=fake_run)

        result = await run_model_evaluation(
            model_evaluation_id="me-1",
            concurrency=1,
            model_evaluation_repo=me_repo,
            evaluation_repo=eval_repo,
            model_repo=model_repo,
            run_repo=run_repo,
        )

        self.assertIsNotNone(result.finished)
        self.assertEqual(
            result.summary.status,
            EvaluationStatus.FINISHED,
        )

    async def test_persists_results_when_repo_provided(self):
        (
            me,
            me_repo,
            mock_eval,
            eval_repo,
            mock_model,
            model_repo,
            run_repo,
        ) = self._setup_mocks()

        result_repo = MagicMock()
        await run_model_evaluation(
            model_evaluation_id="me-1",
            concurrency=1,
            model_evaluation_repo=me_repo,
            evaluation_repo=eval_repo,
            model_repo=model_repo,
            run_repo=run_repo,
            result_repo=result_repo,
        )

        result_repo.create_batch.assert_called_once()
        batch = result_repo.create_batch.call_args[0][0]
        self.assertEqual(len(batch), 2)
        self.assertAlmostEqual(batch[0].score, 0.8)
        self.assertAlmostEqual(batch[1].score, 0.7)

    async def test_no_error_without_result_repo(self):
        (
            me,
            me_repo,
            mock_eval,
            eval_repo,
            mock_model,
            model_repo,
            run_repo,
        ) = self._setup_mocks()

        result = await run_model_evaluation(
            model_evaluation_id="me-1",
            concurrency=1,
            model_evaluation_repo=me_repo,
            evaluation_repo=eval_repo,
            model_repo=model_repo,
            run_repo=run_repo,
        )

        self.assertIsNotNone(result.finished)

    async def test_cleans_up_observer_on_error(self):
        (
            me,
            me_repo,
            mock_eval,
            eval_repo,
            mock_model,
            model_repo,
            run_repo,
        ) = self._setup_mocks()

        mock_eval.run = AsyncMock(
            side_effect=RuntimeError("boom"),
        )

        await run_model_evaluation(
            model_evaluation_id="me-1",
            concurrency=1,
            model_evaluation_repo=me_repo,
            evaluation_repo=eval_repo,
            model_repo=model_repo,
            run_repo=run_repo,
        )

        # Observer should be removed even on error
        self.assertEqual(len(mock_eval.observers), 0)
        # Final update should still be called
        run_repo.update.assert_called()


if __name__ == "__main__":
    unittest.main()
