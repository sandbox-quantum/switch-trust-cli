import unittest
from unittest.mock import AsyncMock

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.detectors.detector import DetectorResult
from switch_trust.eval.core.eval.evaluation import EvaluationStatus
from switch_trust.eval.core.eval.evaluation_single_prompt import (
    SinglePromptEvaluation,
)
from switch_trust.eval.core.models.model import (
    ModelResponse,
    ResponseStatus,
    track_model_errors,
)
from switch_trust.eval.core.models.model_errors import (
    ErrorKind,
    mark_generator_failure,
)


class _StatusError(Exception):
    """Stand-in for a classified tenant-caused failure (e.g. a 500 from the
    agent endpoint), as opposed to an unclassifiable bare exception."""

    def __init__(self, status: int):
        self.status = status
        super().__init__(f"status={status}")


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

    async def test_tenant_caused_failure_logs_as_warning_not_error(self):
        model = AsyncMock()
        model.generate = AsyncMock(side_effect=_StatusError(500))
        detector = AsyncMock()

        e = SinglePromptEvaluation(prompt=_make_prompt(), detector=detector)
        await e.init()
        with self.assertLogs(
            "switch_trust.eval.core.eval.evaluation_single", level="WARNING"
        ) as cm:
            await e.run(model, concurrency=1)

        self.assertFalse(any(r.levelname == "ERROR" for r in cm.records))

    async def test_unclassifiable_failure_still_logs_as_error(self):
        # A bare exception with no HTTP signal is as likely to be our own
        # detector/scoring bug as an unusual tenant failure -- must alert.
        model = AsyncMock()
        model.generate = AsyncMock(side_effect=RuntimeError("unexpected"))
        detector = AsyncMock()

        e = SinglePromptEvaluation(prompt=_make_prompt(), detector=detector)
        await e.init()
        with self.assertLogs(
            "switch_trust.eval.core.eval.evaluation_single", level="ERROR"
        ) as cm:
            await e.run(model, concurrency=1)

        self.assertTrue(any(r.levelname == "ERROR" for r in cm.records))

    async def test_internal_error_still_logs_as_error(self):
        model = AsyncMock()
        exc = RuntimeError("judge exploded")
        mark_generator_failure(exc)
        model.generate = AsyncMock(side_effect=exc)
        detector = AsyncMock()

        e = SinglePromptEvaluation(prompt=_make_prompt(), detector=detector)
        await e.init()
        with self.assertLogs(
            "switch_trust.eval.core.eval.evaluation_single", level="ERROR"
        ) as cm:
            await e.run(model, concurrency=1)

        self.assertTrue(any(r.levelname == "ERROR" for r in cm.records))

    async def test_generator_failure_recorded_on_accumulator(self):
        # A judge parse failure is raised after the judge's own generate()
        # call already returned, so it never passes through Model.generate()'s
        # own accumulator recording -- SingleEvaluation.execute() must record
        # it itself, or the run's error_kind can never resolve past "unknown".
        model = AsyncMock()
        exc = ValueError("judge returned no parseable score")
        mark_generator_failure(exc)
        model.generate = AsyncMock(side_effect=exc)
        detector = AsyncMock()

        e = SinglePromptEvaluation(prompt=_make_prompt(), detector=detector)
        await e.init()
        with track_model_errors() as error_acc:
            await e.run(model, concurrency=1)

        self.assertEqual(error_acc.dominant_kind(), ErrorKind.INTERNAL_ERROR)

    async def test_non_generator_failure_not_recorded_here(self):
        # Model.generate() already records its own failures; SingleEvaluation
        # must not double-record an ordinary (non-judge) failure.
        model = AsyncMock()
        model.generate = AsyncMock(side_effect=RuntimeError("model down"))
        detector = AsyncMock()

        e = SinglePromptEvaluation(prompt=_make_prompt(), detector=detector)
        await e.init()
        with track_model_errors() as error_acc:
            await e.run(model, concurrency=1)

        self.assertEqual(error_acc.counts(), {})

    def test_init_validates_fields(self):
        with self.assertRaises(TypeError):
            SinglePromptEvaluation()


if __name__ == "__main__":
    unittest.main()
