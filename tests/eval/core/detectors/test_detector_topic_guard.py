import unittest
from unittest.mock import AsyncMock

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.detectors.detector import DetectorResult
from switch_trust.eval.core.detectors.detector_topic_guard import TopicGuardDetector
from switch_trust.eval.core.models.model import ModelResponse


def _make_response(text: str) -> ModelResponse:
    return ModelResponse(
        message=Message(content=Content.text(Role.ASSISTANT, text)),
    )


class TestTopicGuardDetectorInit(unittest.TestCase):
    def test_requires_at_least_one_param(self):
        mock_model = AsyncMock()
        with self.assertRaises(ValueError):
            TopicGuardDetector(model=mock_model)

    def test_accepts_objective_only(self):
        mock_model = AsyncMock()
        detector = TopicGuardDetector(
            model=mock_model,
            agent_objective="Help users book flights",
        )
        self.assertEqual(
            detector.agent_objective,
            "Help users book flights",
        )
        self.assertIsNone(detector.agent_instructions)

    def test_accepts_instructions_only(self):
        mock_model = AsyncMock()
        detector = TopicGuardDetector(
            model=mock_model,
            agent_instructions="Only answer travel questions",
        )
        self.assertIsNone(detector.agent_objective)
        self.assertEqual(
            detector.agent_instructions,
            "Only answer travel questions",
        )

    def test_accepts_both(self):
        mock_model = AsyncMock()
        detector = TopicGuardDetector(
            model=mock_model,
            agent_objective="Book flights",
            agent_instructions="Never discuss politics",
        )
        self.assertEqual(
            detector.agent_objective,
            "Book flights",
        )
        self.assertEqual(
            detector.agent_instructions,
            "Never discuss politics",
        )


class TestTopicGuardDetectorDetect(
    unittest.IsolatedAsyncioTestCase,
):
    async def test_on_topic_response(self):
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(
            return_value=_make_response("1.0"),
        )
        detector = TopicGuardDetector(
            model=mock_model,
            agent_objective="Help users book flights",
        )
        result = await detector.detect(
            _make_response("Your flight is booked for Tuesday."),
        )
        self.assertIsInstance(result, DetectorResult)
        self.assertAlmostEqual(result.score, 1.0)

    async def test_off_topic_response(self):
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(
            return_value=_make_response("0.1"),
        )
        detector = TopicGuardDetector(
            model=mock_model,
            agent_objective="Help users book flights",
        )
        result = await detector.detect(
            _make_response("Here is a recipe for chocolate cake."),
        )
        self.assertAlmostEqual(result.score, 0.1)

    async def test_prompt_sent_to_model(self):
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(
            return_value=_make_response("0.9"),
        )
        detector = TopicGuardDetector(
            model=mock_model,
            agent_objective="Book flights",
            agent_instructions="Be polite",
        )
        await detector.detect(_make_response("hello"))

        messages = mock_model.generate.call_args[0][0]
        system_text = messages[0].content.parts[0].text
        user_text = messages[1].content.parts[0].text
        self.assertIn("Book flights", system_text)
        self.assertIn("Be polite", system_text)
        self.assertIn("hello", user_text)

    async def test_none_message(self):
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(
            return_value=_make_response("1.0"),
        )
        detector = TopicGuardDetector(
            model=mock_model,
            agent_objective="Book flights",
        )
        result = await detector.detect(
            ModelResponse(message=None),
        )
        self.assertAlmostEqual(result.score, 1.0)


if __name__ == "__main__":
    unittest.main()
