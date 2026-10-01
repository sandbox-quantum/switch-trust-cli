import unittest
from unittest.mock import AsyncMock

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.detectors.detector import DetectorResult
from switch_trust.eval.core.eval.evaluation import EvaluationStatus
from switch_trust.eval.core.eval.evaluation_message_list import MessageListEvaluation
from switch_trust.eval.core.eval.evaluation_single_prompt import (
    SinglePromptEvaluation,
)
from switch_trust.eval.core.models.model import ModelResponse, ResponseStatus


def _msg(text: str) -> Message:
    return Message(content=Content.text(Role.USER, text))


class TestMessageListEvaluation(unittest.IsolatedAsyncioTestCase):
    async def test_init_creates_children(self):
        detector = AsyncMock()
        messages = [_msg("a"), _msg("b"), _msg("c")]

        e = MessageListEvaluation(messages=messages, detector=detector)
        await e.init()

        self.assertEqual(e.status, EvaluationStatus.INITIALIZED)
        self.assertEqual(len(e.children), 3)
        for child in e.children:
            self.assertIsInstance(child, SinglePromptEvaluation)

    async def test_run(self):
        detector = AsyncMock()
        detector.detect = AsyncMock(
            return_value=DetectorResult(score=0.5),
        )
        messages = [_msg("a"), _msg("b")]

        e = MessageListEvaluation(messages=messages, detector=detector)
        await e.init()

        model = AsyncMock()
        resp = ModelResponse(
            message=Message(content=Content.text(Role.ASSISTANT, "ok")),
            status=ResponseStatus.OK,
        )
        model.generate = AsyncMock(return_value=resp)

        await e.run(model, concurrency=4)

        self.assertEqual(e.status, EvaluationStatus.FINISHED)
        summary = e.get_summary()
        self.assertEqual(summary.total_evaluations, 2)
        self.assertEqual(summary.finished_evaluations, 2)
        self.assertAlmostEqual(summary.achieved_score, 1.0)
        self.assertAlmostEqual(summary.max_score, 2.0)

    async def test_empty_messages_errors(self):
        detector = AsyncMock()
        e = MessageListEvaluation(messages=[], detector=detector)
        await e.init()

        self.assertEqual(e.status, EvaluationStatus.ERROR)


if __name__ == "__main__":
    unittest.main()
