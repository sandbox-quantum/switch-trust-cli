import asyncio
import unittest

from switch_trust.eval.common.schema import Content, Message, Part, PartType, Role
from switch_trust.eval.core.models.model import (
    Model,
    ModelResponse,
    ResponseStatus,
    flatten_messages,
)


class StubModel(Model):
    """Concrete Model for testing the base class behavior."""

    def __init__(
        self,
        response: ModelResponse | None = None,
        error: Exception | None = None,
    ):
        self._response = response
        self._error = error

    async def _generate(self, messages, **kwargs):
        if self._error:
            raise self._error
        return self._response


class TestModelGenerate(unittest.TestCase):
    def test_generate_with_string_input(self):
        expected = ModelResponse(
            message=Message(
                content=Content.text(Role.ASSISTANT, "reply"),
            ),
        )
        model = StubModel(response=expected)
        result = asyncio.run(model.generate("hello"))
        self.assertEqual(result.status, ResponseStatus.OK)
        self.assertEqual(
            result.message.content.parts[0].text,
            "reply",
        )

    def test_generate_with_message_input(self):
        expected = ModelResponse(
            message=Message(
                content=Content.text(Role.ASSISTANT, "reply"),
            ),
        )
        model = StubModel(response=expected)
        msg = Message(content=Content.text(Role.USER, "hi"))
        result = asyncio.run(model.generate(msg))
        self.assertEqual(result.status, ResponseStatus.OK)

    def test_generate_with_message_list(self):
        expected = ModelResponse(
            message=Message(
                content=Content.text(Role.ASSISTANT, "reply"),
            ),
        )
        model = StubModel(response=expected)
        msgs = [Message(content=Content.text(Role.USER, "hi"))]
        result = asyncio.run(model.generate(msgs))
        self.assertEqual(result.status, ResponseStatus.OK)

    def test_generate_error_logs_and_reraises(self):
        error = ValueError("model failure")
        model = StubModel(error=error)
        with self.assertRaises(ValueError) as ctx:
            asyncio.run(model.generate("test"))
        self.assertIn("model failure", str(ctx.exception))

    def test_generate_non_ok_status(self):
        response = ModelResponse(
            message=Message(
                content=Content.text(Role.ASSISTANT, "blocked"),
            ),
            status=ResponseStatus.BLOCKED_SAFETY,
        )
        model = StubModel(response=response)
        result = asyncio.run(model.generate("test"))
        self.assertEqual(result.status, ResponseStatus.BLOCKED_SAFETY)

    def test_generate_none_message(self):
        response = ModelResponse(message=None)
        model = StubModel(response=response)
        result = asyncio.run(model.generate("test"))
        self.assertIsNone(result.message)


class TestFlattenMessages(unittest.TestCase):
    def test_single_message(self):
        msgs = [Message(content=Content.text(Role.USER, "hello world"))]
        self.assertEqual(flatten_messages(msgs), "hello world")

    def test_multiple_messages(self):
        msgs = [
            Message(content=Content.text(Role.USER, "Hello")),
            Message(content=Content.text(Role.ASSISTANT, "Hi!")),
            Message(content=Content.text(Role.USER, "How are you?")),
        ]
        result = flatten_messages(msgs)
        self.assertIn("USER: Hello", result)
        self.assertIn("ASSISTANT: Hi!", result)
        self.assertIn("USER: How are you?", result)

    def test_single_message_multiple_parts(self):
        msg = Message(
            content=Content(
                role=Role.USER,
                parts=[
                    Part(part_type=PartType.TEXT, text="part one"),
                    Part(part_type=PartType.TEXT, text="part two"),
                ],
            ),
        )
        result = flatten_messages([msg])
        self.assertEqual(result, "part one part two")


if __name__ == "__main__":
    unittest.main()
