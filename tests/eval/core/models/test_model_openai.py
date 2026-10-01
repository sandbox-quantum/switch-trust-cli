import unittest
from unittest.mock import AsyncMock, MagicMock

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from pydantic import BaseModel

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.models.model_openai import OpenAIModel


class _Score(BaseModel):
    score: float
    reason: str


def _make_completion(text: str) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-1",
        created=1700000000,
        model="gpt-4o",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=text),
            ),
        ],
    )


class TestOpenAIModel(unittest.IsolatedAsyncioTestCase):
    async def test_generate_text(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_completion("Hello!")
        )

        model = OpenAIModel(client=mock_client, model="gpt-4o")
        msg = Message(content=Content.text(Role.USER, "Hi"))
        resp = await model.generate(msg)

        self.assertIsNotNone(resp.message)
        self.assertEqual(resp.message.content.role, Role.ASSISTANT)
        self.assertEqual(resp.message.content.parts[0].text, "Hello!")
        mock_client.chat.completions.create.assert_called_once()

    async def test_generate_passes_kwargs(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_completion("Hi")
        )

        model = OpenAIModel(client=mock_client, model="gpt-4o")
        msg = Message(content=Content.text(Role.USER, "Hi"))
        await model.generate(msg, temperature=0.5, max_tokens=100)

        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs["temperature"], 0.5)
        self.assertEqual(call_kwargs.kwargs["max_tokens"], 100)

    async def test_output_schema_sets_response_format(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_completion('{"score": 1.0, "reason": "ok"}')
        )

        model = OpenAIModel(client=mock_client, model="gpt-4o")
        msg = Message(content=Content.text(Role.USER, "Hi"))
        await model.generate(msg, output_schema=_Score)

        response_format = mock_client.chat.completions.create.call_args.kwargs[
            "response_format"
        ]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["name"], "_Score")
        self.assertTrue(response_format["json_schema"]["strict"])
        schema = response_format["json_schema"]["schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(sorted(schema["required"]), ["reason", "score"])

    async def test_generate_system_message(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_completion("OK")
        )

        model = OpenAIModel(client=mock_client, model="gpt-4o")
        msg = Message(content=Content.text(Role.SYSTEM, "You are helpful."))
        resp = await model.generate(msg)

        self.assertIsNotNone(resp.message)
        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs["messages"][0]["role"], "system")


if __name__ == "__main__":
    unittest.main()
