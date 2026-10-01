import unittest
from unittest.mock import AsyncMock, MagicMock

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.models.model_openai_compatible import (
    OpenAICompatibleModel,
)


def _make_completion(text: str) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-1",
        created=1700000000,
        model="my-model",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=text,
                ),
            ),
        ],
    )


class TestOpenAICompatibleModel(unittest.IsolatedAsyncioTestCase):
    async def test_generate_text(self):
        model = OpenAICompatibleModel(
            base_url="http://localhost:8000/v1",
            model="my-model",
        )
        model._client = MagicMock()
        model._client.chat.completions.create = AsyncMock(
            return_value=_make_completion("Hello!"),
        )

        msg = Message(
            content=Content.text(Role.USER, "Hi"),
        )
        resp = await model.generate(msg)

        self.assertIsNotNone(resp.message)
        self.assertEqual(
            resp.message.content.role,
            Role.ASSISTANT,
        )
        self.assertEqual(
            resp.message.content.parts[0].text,
            "Hello!",
        )

    async def test_passes_model_name(self):
        model = OpenAICompatibleModel(
            base_url="http://vllm:8000/v1",
            model="meta-llama/Llama-3-8B",
        )
        model._client = MagicMock()
        model._client.chat.completions.create = AsyncMock(
            return_value=_make_completion("Hi"),
        )

        msg = Message(
            content=Content.text(Role.USER, "Hello"),
        )
        await model.generate(msg)

        call_kwargs = model._client.chat.completions.create.call_args
        self.assertEqual(
            call_kwargs.kwargs["model"],
            "meta-llama/Llama-3-8B",
        )

    async def test_multi_turn_messages(self):
        model = OpenAICompatibleModel(
            base_url="http://localhost:8000/v1",
            model="test",
        )
        model._client = MagicMock()
        model._client.chat.completions.create = AsyncMock(
            return_value=_make_completion("Response"),
        )

        msgs = [
            Message(
                content=Content.text(Role.USER, "Hello"),
            ),
            Message(
                content=Content.text(
                    Role.ASSISTANT,
                    "Hi!",
                ),
            ),
            Message(
                content=Content.text(
                    Role.USER,
                    "How are you?",
                ),
            ),
        ]
        resp = await model.generate(msgs)

        self.assertIsNotNone(resp.message)
        call_kwargs = model._client.chat.completions.create.call_args
        sent_messages = call_kwargs.kwargs["messages"]
        self.assertEqual(len(sent_messages), 3)
        self.assertEqual(
            sent_messages[0]["role"],
            "user",
        )
        self.assertEqual(
            sent_messages[1]["role"],
            "assistant",
        )

    async def test_temperature_default(self):
        model = OpenAICompatibleModel(
            base_url="http://localhost:8000/v1",
            model="test",
            temperature=0.5,
        )
        model._client = MagicMock()
        model._client.chat.completions.create = AsyncMock(
            return_value=_make_completion("Hi"),
        )

        msg = Message(
            content=Content.text(Role.USER, "Hi"),
        )
        await model.generate(msg)

        call_kwargs = model._client.chat.completions.create.call_args
        self.assertEqual(
            call_kwargs.kwargs["temperature"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
