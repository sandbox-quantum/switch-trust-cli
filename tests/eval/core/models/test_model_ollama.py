import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from pydantic import BaseModel

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.models.model_ollama import (
    OllamaModel,
    discover_ollama_models,
)


class _Score(BaseModel):
    score: float


def _make_completion(text: str) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-1",
        created=1700000000,
        model="llama3",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=text),
            ),
        ],
    )


class TestOllamaModel(unittest.IsolatedAsyncioTestCase):
    async def test_generate_text(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_completion("Hello!"),
        )

        model = OllamaModel(model="llama3")
        model._client = mock_client

        msg = Message(content=Content.text(Role.USER, "Hi"))
        resp = await model.generate(msg)

        self.assertIsNotNone(resp.message)
        self.assertEqual(resp.message.content.role, Role.ASSISTANT)
        self.assertEqual(resp.message.content.parts[0].text, "Hello!")
        mock_client.chat.completions.create.assert_called_once()

    async def test_generate_passes_temperature(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_completion("Hi"),
        )

        model = OllamaModel(model="llama3", temperature=0.5)
        model._client = mock_client

        msg = Message(content=Content.text(Role.USER, "Hi"))
        await model.generate(msg)

        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs["temperature"], 0.5)

    async def test_generate_kwargs_override_temperature(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_completion("Hi"),
        )

        model = OllamaModel(model="llama3", temperature=0.0)
        model._client = mock_client

        msg = Message(content=Content.text(Role.USER, "Hi"))
        await model.generate(msg, temperature=0.9)

        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs["temperature"], 0.9)

    async def test_generate_uses_correct_model_name(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_completion("Hi"),
        )

        model = OllamaModel(model="mistral:7b")
        model._client = mock_client

        msg = Message(content=Content.text(Role.USER, "Hi"))
        await model.generate(msg)

        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(call_kwargs.kwargs["model"], "mistral:7b")

    async def test_output_schema_sets_native_format(self):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_completion('{"score": 1.0}'),
        )

        model = OllamaModel(model="llama3")
        model._client = mock_client

        msg = Message(content=Content.text(Role.USER, "Hi"))
        await model.generate(msg, output_schema=_Score)

        extra_body = mock_client.chat.completions.create.call_args.kwargs["extra_body"]
        self.assertEqual(extra_body["format"]["type"], "object")
        self.assertIn("score", extra_body["format"]["properties"])

    def test_init_sets_base_url(self):
        with patch("switch_trust.eval.core.models.model_ollama.AsyncOpenAI") as mock_cls:
            OllamaModel(model="llama3", host="http://myhost:11434")
            mock_cls.assert_called_once_with(
                base_url="http://myhost:11434/v1",
                api_key="ollama",
            )

    def test_init_default_host(self):
        with patch("switch_trust.eval.core.models.model_ollama.AsyncOpenAI") as mock_cls:
            OllamaModel(model="llama3")
            mock_cls.assert_called_once_with(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )


class TestDiscoverOllamaModels(unittest.TestCase):
    @patch("switch_trust.eval.core.models.model_ollama.requests.get")
    def test_returns_model_names(self, mock_get: MagicMock):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3"},
                {"name": "mistral:7b"},
            ],
        }
        mock_get.return_value = mock_response

        result = discover_ollama_models()
        self.assertEqual(result, ["llama3", "mistral:7b"])
        mock_get.assert_called_once_with("http://localhost:11434/api/tags")

    @patch("switch_trust.eval.core.models.model_ollama.requests.get")
    def test_empty_models_list(self, mock_get: MagicMock):
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response

        result = discover_ollama_models()
        self.assertEqual(result, [])

    @patch("switch_trust.eval.core.models.model_ollama.requests.get")
    def test_custom_host(self, mock_get: MagicMock):
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": [{"name": "phi3"}]}
        mock_get.return_value = mock_response

        result = discover_ollama_models(host="http://remote:11434")
        self.assertEqual(result, ["phi3"])
        mock_get.assert_called_once_with("http://remote:11434/api/tags")

    @patch("switch_trust.eval.core.models.model_ollama.requests.get")
    def test_http_error_raises(self, mock_get: MagicMock):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("connection refused")
        mock_get.return_value = mock_response

        with self.assertRaisesRegex(Exception, "connection refused"):
            discover_ollama_models()

    @patch("switch_trust.eval.core.models.model_ollama.requests.get")
    def test_missing_models_key(self, mock_get: MagicMock):
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_get.return_value = mock_response

        result = discover_ollama_models()
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
