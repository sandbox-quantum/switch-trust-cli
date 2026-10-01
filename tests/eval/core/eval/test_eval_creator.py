import json
import unittest
from unittest.mock import AsyncMock

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.eval.eval_creator import (
    CreationContext,
    EvaluationPlan,
    _build_user_message,
    _parse_response,
    create_evaluation,
)
from switch_trust.eval.core.message.message_collection_memory import (
    InMemoryMessageCollection,
)
from switch_trust.eval.core.models.model import ModelResponse


def _mock_model_with_response(text: str) -> AsyncMock:
    response_msg = Message(
        content=Content.text(Role.ASSISTANT, text),
    )
    mock = AsyncMock()
    mock.generate = AsyncMock(
        return_value=ModelResponse(
            message=response_msg,
        ),
    )
    return mock


class TestCreationContext(unittest.TestCase):
    def test_minimal(self):
        c = CreationContext(
            purpose="A chatbot",
            evaluation_goal="test injection",
        )
        self.assertEqual(c.purpose, "A chatbot")
        self.assertEqual(c.evaluation_goal, "test injection")
        self.assertEqual(c.num_prompts, 10)
        self.assertIsNone(c.instructions)
        self.assertEqual(c.tool_names, [])

    def test_full(self):
        c = CreationContext(
            purpose="Customer support agent",
            evaluation_goal="test PII leakage",
            num_prompts=20,
            instructions="Be polite",
            tool_names=["search", "lookup_order"],
            additional_context="Handles PII",
        )
        self.assertEqual(c.num_prompts, 20)
        self.assertEqual(len(c.tool_names), 2)


class TestEvaluationPlan(unittest.TestCase):
    def test_defaults(self):
        p = EvaluationPlan()
        self.assertIsNone(p.prompts)
        self.assertEqual(p.detector_prompt, "")

    def test_with_data(self):
        collection = InMemoryMessageCollection(
            [
                Message(content=Content.text(Role.USER, "p1")),
            ]
        )
        p = EvaluationPlan(
            prompts=collection,
            detector_prompt="Check for leaks",
        )
        self.assertEqual(p.prompts.size(), 1)
        self.assertEqual(p.detector_prompt, "Check for leaks")


class TestBuildUserMessage(unittest.TestCase):
    def test_minimal(self):
        c = CreationContext(
            purpose="A chatbot",
            evaluation_goal="test injection",
            num_prompts=5,
        )
        msg = _build_user_message(c)
        self.assertIn("A chatbot", msg)
        self.assertIn("test injection", msg)
        self.assertIn("5 test prompts", msg)
        self.assertNotIn("Instructions:", msg)

    def test_full(self):
        c = CreationContext(
            purpose="Support bot",
            evaluation_goal="test PII leakage",
            num_prompts=10,
            instructions="Be helpful",
            tool_names=["search", "email"],
            additional_context="Handles PII",
        )
        msg = _build_user_message(c)
        self.assertIn("Instructions: Be helpful", msg)
        self.assertIn("search, email", msg)
        self.assertIn("Handles PII", msg)


def _response(text: str) -> ModelResponse:
    return ModelResponse(message=Message(content=Content.text(Role.ASSISTANT, text)))


class TestParseResponse(unittest.TestCase):
    def test_valid_json(self):
        data = {
            "prompts": ["p1", "p2", "p3"],
            "detector_prompt": "Check safety",
        }
        result = _parse_response(_response(json.dumps(data)))
        self.assertIsInstance(result, EvaluationPlan)
        self.assertIsInstance(
            result.prompts,
            InMemoryMessageCollection,
        )
        self.assertEqual(result.prompts.size(), 3)
        self.assertEqual(
            result.detector_prompt,
            "Check safety",
        )

    def test_json_with_markdown_fences(self):
        raw = '```json\n{"prompts": ["a"], "detector_prompt": "b"}\n```'
        result = _parse_response(_response(raw))
        self.assertEqual(result.prompts.size(), 1)
        self.assertEqual(result.detector_prompt, "b")

    def test_missing_prompts_raises(self):
        # The schema requires both fields, so a missing one is a hard error
        # rather than a silent default.
        with self.assertRaises(ValueError):
            _parse_response(_response('{"detector_prompt": "x"}'))

    def test_missing_detector_prompt_raises(self):
        with self.assertRaises(ValueError):
            _parse_response(_response('{"prompts": ["a"]}'))

    def test_invalid_json_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _parse_response(_response("not json at all"))
        self.assertIn("did not match schema", str(ctx.exception))

    def test_prompts_not_list_raises(self):
        with self.assertRaises(ValueError):
            _parse_response(
                _response('{"prompts": "not a list", "detector_prompt": "x"}')
            )

    def test_prompt_messages_are_user_role(self):
        data = {
            "prompts": ["hello", "world"],
            "detector_prompt": "check",
        }
        result = _parse_response(_response(json.dumps(data)))
        messages = result.prompts.load()
        for msg in messages:
            self.assertEqual(msg.content.role, Role.USER)


class TestCreateEvaluation(unittest.IsolatedAsyncioTestCase):
    async def test_calls_model_and_parses(self):
        response_json = json.dumps(
            {
                "prompts": ["prompt1", "prompt2"],
                "detector_prompt": "Evaluate safety",
            }
        )
        mock_model = _mock_model_with_response(response_json)

        context = CreationContext(
            purpose="A chatbot",
            evaluation_goal="test injection",
            num_prompts=2,
        )
        result = await create_evaluation(context, mock_model)

        self.assertIsInstance(result, EvaluationPlan)
        self.assertEqual(result.prompts.size(), 2)
        self.assertEqual(
            result.detector_prompt,
            "Evaluate safety",
        )
        mock_model.generate.assert_called_once()

    async def test_raises_on_empty_response(self):
        mock_model = AsyncMock()
        mock_model.generate = AsyncMock(
            return_value=ModelResponse(
                message=None,
            ),
        )

        context = CreationContext(
            purpose="Bot",
            evaluation_goal="test",
        )
        with self.assertRaises(ValueError) as ctx:
            await create_evaluation(context, mock_model)
        self.assertIn(
            "did not return a response",
            str(ctx.exception),
        )

    async def test_passes_system_instruction(self):
        response_json = json.dumps(
            {
                "prompts": ["p1"],
                "detector_prompt": "judge",
            }
        )
        mock_model = _mock_model_with_response(response_json)

        context = CreationContext(
            purpose="Bot",
            evaluation_goal="test",
            num_prompts=1,
        )
        await create_evaluation(context, mock_model)

        call_args = mock_model.generate.call_args
        messages = call_args[0][0]
        self.assertIsInstance(messages, list)
        self.assertEqual(len(messages), 2)
        self.assertEqual(
            messages[0].content.role,
            Role.SYSTEM,
        )
        self.assertIn(
            "red-teaming",
            messages[0].content.parts[0].text,
        )
        self.assertEqual(
            messages[1].content.role,
            Role.USER,
        )


if __name__ == "__main__":
    unittest.main()
