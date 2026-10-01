import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.models.model import ResponseStatus
from switch_trust.eval.core.models.model_generic_http import (
    GenericHttpModel,
    _resolve_path,
)


def _make_aiohttp_mocks(json_data: dict):
    """Create mock aiohttp session and response for a single POST call."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value=json_data)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    return mock_session


class TestResolvePath(unittest.TestCase):
    def test_simple_key(self):
        self.assertEqual(
            _resolve_path({"output": "hi"}, "output"),
            "hi",
        )

    def test_nested_path(self):
        data = {"data": {"response": {"text": "hello"}}}
        self.assertEqual(
            _resolve_path(data, "data.response.text"),
            "hello",
        )

    def test_list_index(self):
        data = {"choices": [{"message": "hi"}]}
        self.assertEqual(
            _resolve_path(data, "choices.0.message"),
            "hi",
        )

    def test_missing_key_returns_none(self):
        self.assertIsNone(
            _resolve_path({"a": 1}, "b"),
        )

    def test_invalid_index_returns_none(self):
        self.assertIsNone(
            _resolve_path({"a": [1]}, "a.5"),
        )


class TestGenericHttpModel(unittest.IsolatedAsyncioTestCase):
    @patch("switch_trust.eval.core.models.model_generic_http.aiohttp.ClientSession")
    async def test_generate_text(self, mock_session_cls):
        mock_session = _make_aiohttp_mocks({"output": "Hello!"})
        mock_session_cls.return_value = mock_session

        model = GenericHttpModel(
            url="http://localhost:8000/chat",
        )
        msg = Message(
            content=Content.text(Role.USER, "Hi"),
        )
        resp = await model.generate(msg)

        self.assertIsNotNone(resp.message)
        self.assertEqual(
            resp.message.content.parts[0].text,
            "Hello!",
        )

    @patch("switch_trust.eval.core.models.model_generic_http.aiohttp.ClientSession")
    async def test_custom_field_names(self, mock_session_cls):
        mock_session = _make_aiohttp_mocks({"answer": "42"})
        mock_session_cls.return_value = mock_session

        model = GenericHttpModel(
            url="http://localhost:8000/api",
            input_path="question",
            output_path="answer",
        )
        msg = Message(
            content=Content.text(Role.USER, "What?"),
        )
        resp = await model.generate(msg)

        self.assertEqual(
            resp.message.content.parts[0].text,
            "42",
        )
        call_kwargs = mock_session.post.call_args
        payload = call_kwargs.kwargs["json"]
        self.assertIn("question", payload)

    @patch("switch_trust.eval.core.models.model_generic_http.aiohttp.ClientSession")
    async def test_nested_output_path(self, mock_session_cls):
        mock_session = _make_aiohttp_mocks(
            {
                "choices": [{"message": {"content": "deep"}}],
            }
        )
        mock_session_cls.return_value = mock_session

        model = GenericHttpModel(
            url="http://localhost:8000",
            output_path="choices.0.message.content",
        )
        msg = Message(
            content=Content.text(Role.USER, "Hi"),
        )
        resp = await model.generate(msg)

        self.assertEqual(
            resp.message.content.parts[0].text,
            "deep",
        )

    @patch("switch_trust.eval.core.models.model_generic_http.aiohttp.ClientSession")
    async def test_empty_output(self, mock_session_cls):
        mock_session = _make_aiohttp_mocks({"output": ""})
        mock_session_cls.return_value = mock_session

        model = GenericHttpModel(
            url="http://localhost:8000",
        )
        msg = Message(
            content=Content.text(Role.USER, "Hi"),
        )
        resp = await model.generate(msg)

        self.assertIsNone(resp.message)
        self.assertEqual(
            resp.status,
            ResponseStatus.EMPTY_RESPONSE,
        )

    @patch("switch_trust.eval.core.models.model_generic_http.aiohttp.ClientSession")
    async def test_custom_headers(self, mock_session_cls):
        mock_session = _make_aiohttp_mocks({"output": "ok"})
        mock_session_cls.return_value = mock_session

        model = GenericHttpModel(
            url="http://localhost:8000",
            headers={"X-Api-Key": "secret123"},
        )
        msg = Message(
            content=Content.text(Role.USER, "Hi"),
        )
        await model.generate(msg)

        call_kwargs = mock_session.post.call_args
        self.assertEqual(
            call_kwargs.kwargs["headers"]["X-Api-Key"],
            "secret123",
        )

    @patch("switch_trust.eval.core.models.model_generic_http.aiohttp.ClientSession")
    async def test_multi_message_flattened(self, mock_session_cls):
        mock_session = _make_aiohttp_mocks({"output": "ok"})
        mock_session_cls.return_value = mock_session

        model = GenericHttpModel(
            url="http://localhost:8000",
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
        await model.generate(msgs)

        call_kwargs = mock_session.post.call_args
        prompt = call_kwargs.kwargs["json"]["input"]
        self.assertIn("USER: Hello", prompt)
        self.assertIn("ASSISTANT: Hi!", prompt)
        self.assertIn("USER: How are you?", prompt)


if __name__ == "__main__":
    unittest.main()
