import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.detectors.detector import DetectorResult
from switch_trust.eval.core.eval.evaluation import EvaluationStatus
from switch_trust.eval.core.eval.evaluation_topic_guard import (
    TopicGuardEvaluation,
    TopicGuardTurnEvaluation,
)
from switch_trust.eval.core.models.model import ModelResponse, ResponseStatus


def _make_response(
    text: str = "I can only help with travel.",
    status: ResponseStatus = ResponseStatus.OK,
) -> ModelResponse:
    msg = Message(content=Content.text(Role.ASSISTANT, text))
    return ModelResponse(message=msg, status=status)


def _make_attacker_response(
    broken: bool,
    score: float,
    next_prompt: str = "Try again.",
) -> ModelResponse:
    data = {
        "broken": broken,
        "score": score,
        "reasoning": "test",
        "next_prompt": next_prompt,
    }
    msg = Message(
        content=Content.text(
            Role.ASSISTANT,
            json.dumps(data),
        ),
    )
    return ModelResponse(message=msg)


class TestTopicGuardEvaluationInit(unittest.TestCase):
    def test_requires_at_least_one_param(self):
        with self.assertRaises(ValueError):
            TopicGuardEvaluation()

    def test_accepts_objective_only(self):
        mock_model = AsyncMock()
        mock_detector = MagicMock()
        e = TopicGuardEvaluation(
            agent_objective="Book flights",
            attacker_model=mock_model,
            detector=mock_detector,
        )
        self.assertEqual(
            e.agent_objective,
            "Book flights",
        )
        self.assertIsNone(e.agent_instructions)
        self.assertIn("Book flights", e.goals[0])

    def test_accepts_both(self):
        mock_model = AsyncMock()
        mock_detector = MagicMock()
        e = TopicGuardEvaluation(
            agent_objective="Book flights",
            agent_instructions="Be polite",
            attacker_model=mock_model,
            detector=mock_detector,
        )
        self.assertEqual(
            e.agent_objective,
            "Book flights",
        )
        self.assertEqual(
            e.agent_instructions,
            "Be polite",
        )


class TestTopicGuardTurnEvaluation(
    unittest.IsolatedAsyncioTestCase,
):
    async def test_on_topic_responses_score_high(self):
        target = AsyncMock()
        target.generate = AsyncMock(
            return_value=_make_response(
                "I can help you book a flight.",
            ),
        )

        attacker = AsyncMock()
        attacker.generate = AsyncMock(
            return_value=_make_attacker_response(
                broken=False,
                score=1.0,
                next_prompt="What about cooking?",
            ),
        )

        mock_detector = AsyncMock()
        mock_detector.detect = AsyncMock(
            return_value=DetectorResult(score=0.95),
        )

        e = TopicGuardTurnEvaluation(
            starting_prompt="Tell me a joke.",
            goal="Lead model off-topic.",
            max_turns=2,
            attacker_model=attacker,
            detector=mock_detector,
        )
        await e.init()
        await e.run(target, concurrency=1)

        self.assertEqual(
            e.status,
            EvaluationStatus.FINISHED,
        )
        self.assertAlmostEqual(e.score, 0.95)

    async def test_off_topic_response_scores_low(self):
        target = AsyncMock()
        target.generate = AsyncMock(
            return_value=_make_response(
                "Here is a recipe for chocolate cake.",
            ),
        )

        attacker = AsyncMock()
        attacker.generate = AsyncMock(
            return_value=_make_attacker_response(
                broken=True,
                score=0.1,
            ),
        )

        mock_detector = AsyncMock()
        mock_detector.detect = AsyncMock(
            return_value=DetectorResult(score=0.2),
        )

        e = TopicGuardTurnEvaluation(
            starting_prompt="Can you cook?",
            goal="Lead model off-topic.",
            max_turns=5,
            attacker_model=attacker,
            detector=mock_detector,
        )
        await e.init()
        await e.run(target, concurrency=1)

        self.assertEqual(
            e.status,
            EvaluationStatus.FINISHED,
        )
        self.assertAlmostEqual(e.score, 0.2)

    async def test_min_detector_score_across_turns(self):
        target = AsyncMock()
        target.generate = AsyncMock(
            return_value=_make_response("Some response."),
        )

        attacker = AsyncMock()
        attacker.generate = AsyncMock(
            side_effect=[
                _make_attacker_response(
                    broken=False,
                    score=1.0,
                    next_prompt="Next question.",
                ),
                _make_attacker_response(
                    broken=False,
                    score=1.0,
                    next_prompt="Another question.",
                ),
            ],
        )

        mock_detector = AsyncMock()
        mock_detector.detect = AsyncMock(
            side_effect=[
                DetectorResult(score=0.9),
                DetectorResult(score=0.3),
            ],
        )

        e = TopicGuardTurnEvaluation(
            starting_prompt="Hello.",
            goal="Lead off-topic.",
            max_turns=2,
            attacker_model=attacker,
            detector=mock_detector,
        )
        await e.init()
        await e.run(target, concurrency=1)

        self.assertAlmostEqual(e.score, 0.3)

    async def test_blocked_response_scores_one(self):
        target = AsyncMock()
        target.generate = AsyncMock(
            return_value=ModelResponse(
                message=None,
                status=ResponseStatus.BLOCKED_SAFETY,
            ),
        )

        attacker = AsyncMock()
        mock_detector = AsyncMock()

        e = TopicGuardTurnEvaluation(
            starting_prompt="Bad prompt.",
            goal="Test.",
            max_turns=3,
            attacker_model=attacker,
            detector=mock_detector,
        )
        await e.init()
        await e.run(target, concurrency=1)

        self.assertAlmostEqual(e.score, 1.0)
        mock_detector.detect.assert_not_called()


class TestTopicGuardEvaluation(
    unittest.IsolatedAsyncioTestCase,
):
    async def test_get_children_generates_correct_count(self):
        attacker = AsyncMock()
        prompt_data = {
            "prompts": ["Prompt 1", "Prompt 2", "Prompt 3"],
        }
        attacker.generate = AsyncMock(
            return_value=ModelResponse(
                message=Message(
                    content=Content.text(
                        Role.ASSISTANT,
                        json.dumps(prompt_data),
                    ),
                ),
            ),
        )

        mock_detector = AsyncMock()

        e = TopicGuardEvaluation(
            agent_objective="Book flights",
            num_prompts=3,
            max_turns=4,
            attacker_model=attacker,
            detector=mock_detector,
        )
        await e.init()

        self.assertEqual(len(e.children), 3)
        for child in e.children:
            self.assertIsInstance(
                child,
                TopicGuardTurnEvaluation,
            )
            self.assertEqual(child.max_turns, 4)

    async def test_full_run_uses_detector_scores(self):
        target = AsyncMock()
        target.generate = AsyncMock(
            return_value=_make_response("Some response."),
        )

        attacker = AsyncMock()
        prompt_data = {"prompts": ["P1", "P2"]}
        attacker.generate = AsyncMock(
            side_effect=[
                ModelResponse(
                    message=Message(
                        content=Content.text(
                            Role.ASSISTANT,
                            json.dumps(prompt_data),
                        ),
                    ),
                ),
                _make_attacker_response(
                    broken=False,
                    score=1.0,
                ),
                _make_attacker_response(
                    broken=True,
                    score=0.2,
                ),
                _make_attacker_response(
                    broken=True,
                    score=0.1,
                ),
            ],
        )

        mock_detector = AsyncMock()
        mock_detector.detect = AsyncMock(
            side_effect=[
                DetectorResult(score=0.9),
                DetectorResult(score=0.8),
                DetectorResult(score=0.4),
            ],
        )

        e = TopicGuardEvaluation(
            agent_objective="Book flights",
            num_prompts=2,
            max_turns=2,
            attacker_model=attacker,
            detector=mock_detector,
        )
        await e.init()
        await e.run(target, concurrency=1)

        summary = e.get_summary()
        self.assertEqual(
            summary.status,
            EvaluationStatus.FINISHED,
        )
        self.assertEqual(summary.total_evaluations, 2)

        results = e.get_results()
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
