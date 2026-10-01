import unittest
from unittest.mock import AsyncMock

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.detectors.detector import DetectorResult
from switch_trust.eval.core.eval.evaluation import EvaluationStatus
from switch_trust.eval.core.eval.evaluation_single_prompt import (
    SinglePromptEvaluation,
)
from switch_trust.eval.core.models.model import ModelResponse, ResponseStatus


def _make_prompt(text: str = "test prompt") -> Message:
    return Message(content=Content.text(Role.USER, text))


def _make_response(
    text: str = "response",
    status: ResponseStatus = ResponseStatus.OK,
) -> ModelResponse:
    msg = Message(content=Content.text(Role.ASSISTANT, text))
    return ModelResponse(message=msg, status=status)


class TestSinglePromptEvaluation(unittest.IsolatedAsyncioTestCase):
    async def test_successful_detection(self):
        model = AsyncMock()
        model.generate = AsyncMock(return_value=_make_response())
        detector = AsyncMock()
        detector.detect = AsyncMock(
            return_value=DetectorResult(score=0.8),
        )

        e = SinglePromptEvaluation(
            prompt=_make_prompt(),
            detector=detector,
        )
        await e.init()
        await e.run(model, concurrency=1)

        self.assertEqual(e.status, EvaluationStatus.FINISHED)
        self.assertAlmostEqual(e.score, 0.8)

    async def test_blocked_response_scores_one(self):
        model = AsyncMock()
        model.generate = AsyncMock(
            return_value=_make_response(
                status=ResponseStatus.BLOCKED_SAFETY,
            ),
        )
        detector = AsyncMock()

        e = SinglePromptEvaluation(
            prompt=_make_prompt(),
            detector=detector,
        )
        await e.init()
        await e.run(model, concurrency=1)

        self.assertEqual(e.status, EvaluationStatus.FINISHED)
        self.assertAlmostEqual(e.score, 1.0)
        detector.detect.assert_not_called()

    async def test_model_error_sets_error_status(self):
        model = AsyncMock()
        model.generate = AsyncMock(
            side_effect=RuntimeError("model down"),
        )
        detector = AsyncMock()

        e = SinglePromptEvaluation(
            prompt=_make_prompt(),
            detector=detector,
        )
        await e.init()
        await e.run(model, concurrency=1)

        self.assertEqual(e.status, EvaluationStatus.ERROR)
        self.assertIn("model down", e.error_message)

    async def test_observer_notified(self):
        model = AsyncMock()
        model.generate = AsyncMock(return_value=_make_response())
        detector = AsyncMock()
        detector.detect = AsyncMock(
            return_value=DetectorResult(score=1.0),
        )

        notified = []
        e = SinglePromptEvaluation(
            prompt=_make_prompt(),
            detector=detector,
        )
        e.add_observer(lambda ev: notified.append(ev.status))
        await e.init()
        await e.run(model, concurrency=1)

        self.assertIn(EvaluationStatus.FINISHED, notified)

    async def test_get_summary(self):
        model = AsyncMock()
        model.generate = AsyncMock(return_value=_make_response())
        detector = AsyncMock()
        detector.detect = AsyncMock(
            return_value=DetectorResult(score=0.6),
        )

        e = SinglePromptEvaluation(
            prompt=_make_prompt(),
            detector=detector,
        )
        await e.init()
        await e.run(model, concurrency=1)

        summary = e.get_summary()
        self.assertEqual(summary.status, EvaluationStatus.FINISHED)
        self.assertEqual(summary.total_evaluations, 1)
        self.assertEqual(summary.finished_evaluations, 1)
        self.assertAlmostEqual(summary.achieved_score, 0.6)
        self.assertAlmostEqual(summary.max_score, 1.0)

    async def test_errored_summary_excluded_from_score(self):
        model = AsyncMock()
        model.generate = AsyncMock(side_effect=RuntimeError("model down"))
        detector = AsyncMock()

        e = SinglePromptEvaluation(
            prompt=_make_prompt(),
            detector=detector,
        )
        await e.init()
        await e.run(model, concurrency=1)

        summary = e.get_summary()
        self.assertEqual(summary.status, EvaluationStatus.ERROR)
        self.assertEqual(summary.error_evaluations, 1)
        # Errored prompt contributes to neither side of the mean.
        self.assertAlmostEqual(summary.max_score, 0.0)
        self.assertAlmostEqual(summary.achieved_score, 0.0)

    def test_init_validates_fields(self):
        with self.assertRaises(TypeError):
            SinglePromptEvaluation()


if __name__ == "__main__":
    unittest.main()
