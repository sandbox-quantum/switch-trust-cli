import unittest
from unittest.mock import AsyncMock, patch

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from pydantic import BaseModel

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.models.model_litellm import LiteLLMModel


class _Score(BaseModel):
    score: float


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


class TestLiteLLMModel(unittest.IsolatedAsyncioTestCase):
    @patch("switch_trust.eval.core.models.model_litellm.litellm")
    async def test_generate_text(self, mock_litellm):
        mock_litellm.acompletion = AsyncMock(return_value=_make_completion("Hello!"))

        model = LiteLLMModel(model="gpt-4o")
        msg = Message(content=Content.text(Role.USER, "Hi"))
        resp = await model.generate(msg)

        self.assertIsNotNone(resp.message)
        self.assertEqual(resp.message.content.role, Role.ASSISTANT)
        self.assertEqual(resp.message.content.parts[0].text, "Hello!")
        mock_litellm.acompletion.assert_called_once()

    @patch("switch_trust.eval.core.models.model_litellm.litellm")
    async def test_generate_passes_kwargs(self, mock_litellm):
        mock_litellm.acompletion = AsyncMock(return_value=_make_completion("Hi"))

        model = LiteLLMModel(model="gpt-4o")
        msg = Message(content=Content.text(Role.USER, "Hi"))
        await model.generate(msg, temperature=0.3)

        call_kwargs = mock_litellm.acompletion.call_args
        self.assertEqual(call_kwargs.kwargs["temperature"], 0.3)

    @patch("switch_trust.eval.core.models.model_litellm.litellm")
    async def test_output_schema_sets_response_format(self, mock_litellm):
        mock_litellm.acompletion = AsyncMock(return_value=_make_completion("Hi"))

        model = LiteLLMModel(model="gpt-4o")
        msg = Message(content=Content.text(Role.USER, "Hi"))
        await model.generate(msg, output_schema=_Score)

        call_kwargs = mock_litellm.acompletion.call_args
        self.assertIs(call_kwargs.kwargs["response_format"], _Score)

    @patch("switch_trust.eval.core.models.model_litellm.litellm")
    async def test_model_name_passed(self, mock_litellm):
        mock_litellm.acompletion = AsyncMock(return_value=_make_completion("Hi"))

        model = LiteLLMModel(model="anthropic/claude-sonnet-4-6-20250514")
        msg = Message(content=Content.text(Role.USER, "Hi"))
        await model.generate(msg)

        call_kwargs = mock_litellm.acompletion.call_args
        self.assertEqual(
            call_kwargs.kwargs["model"], "anthropic/claude-sonnet-4-6-20250514"
        )


if __name__ == "__main__":
    unittest.main()
