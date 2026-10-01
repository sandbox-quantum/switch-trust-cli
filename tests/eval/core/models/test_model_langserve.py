import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.models.model import ResponseStatus
from switch_trust.eval.core.models.model_langserve import LangServeModel


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


class TestLangServeModel(unittest.IsolatedAsyncioTestCase):
    @patch("switch_trust.eval.core.models.model_langserve.aiohttp.ClientSession")
    async def test_generate_text(self, mock_session_cls):
        mock_session = _make_aiohttp_mocks({"output": "Hello!"})
        mock_session_cls.return_value = mock_session

        model = LangServeModel(
            base_url="http://localhost:8000",
            chain_path="my-chain",
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

    @patch("switch_trust.eval.core.models.model_langserve.aiohttp.ClientSession")
    async def test_url_composition_with_chain(self, mock_session_cls):
        mock_session = _make_aiohttp_mocks({"output": "ok"})
        mock_session_cls.return_value = mock_session

        model = LangServeModel(
            base_url="http://localhost:8000",
            chain_path="joke",
        )
        msg = Message(
            content=Content.text(Role.USER, "Hi"),
        )
        await model.generate(msg)

        call_args = mock_session.post.call_args
        self.assertEqual(
            call_args.args[0],
            "http://localhost:8000/joke/invoke",
        )

    @patch("switch_trust.eval.core.models.model_langserve.aiohttp.ClientSession")
    async def test_url_composition_no_chain(self, mock_session_cls):
        mock_session = _make_aiohttp_mocks({"output": "ok"})
        mock_session_cls.return_value = mock_session

        model = LangServeModel(
            base_url="http://localhost:8000",
        )
        msg = Message(
            content=Content.text(Role.USER, "Hi"),
        )
        await model.generate(msg)

        call_args = mock_session.post.call_args
        self.assertEqual(
            call_args.args[0],
            "http://localhost:8000/invoke",
        )

    @patch("switch_trust.eval.core.models.model_langserve.aiohttp.ClientSession")
    async def test_empty_output(self, mock_session_cls):
        mock_session = _make_aiohttp_mocks({"output": ""})
        mock_session_cls.return_value = mock_session

        model = LangServeModel(
            base_url="http://localhost:8000",
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

    @patch("switch_trust.eval.core.models.model_langserve.aiohttp.ClientSession")
    async def test_dict_output_extracts_content(
        self,
        mock_session_cls,
    ):
        mock_session = _make_aiohttp_mocks(
            {
                "output": {"content": "The answer is 42"},
            }
        )
        mock_session_cls.return_value = mock_session

        model = LangServeModel(
            base_url="http://localhost:8000",
        )
        msg = Message(
            content=Content.text(Role.USER, "Hi"),
        )
        resp = await model.generate(msg)

        self.assertEqual(
            resp.message.content.parts[0].text,
            "The answer is 42",
        )

    @patch("switch_trust.eval.core.models.model_langserve.aiohttp.ClientSession")
    async def test_custom_headers(self, mock_session_cls):
        mock_session = _make_aiohttp_mocks({"output": "ok"})
        mock_session_cls.return_value = mock_session

        model = LangServeModel(
            base_url="http://localhost:8000",
            headers={"Authorization": "Bearer tok123"},
        )
        msg = Message(
            content=Content.text(Role.USER, "Hi"),
        )
        await model.generate(msg)

        call_kwargs = mock_session.post.call_args
        self.assertEqual(
            call_kwargs.kwargs["headers"]["Authorization"],
            "Bearer tok123",
        )

    @patch("switch_trust.eval.core.models.model_langserve.aiohttp.ClientSession")
    async def test_sends_input_format(self, mock_session_cls):
        mock_session = _make_aiohttp_mocks({"output": "ok"})
        mock_session_cls.return_value = mock_session

        model = LangServeModel(
            base_url="http://localhost:8000",
            chain_path="qa",
        )
        msg = Message(
            content=Content.text(Role.USER, "question"),
        )
        await model.generate(msg)

        call_kwargs = mock_session.post.call_args
        payload = call_kwargs.kwargs["json"]
        self.assertEqual(payload, {"input": "question"})


if __name__ == "__main__":
    unittest.main()
