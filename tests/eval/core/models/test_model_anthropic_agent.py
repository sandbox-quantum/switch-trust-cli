import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.models.model_anthropic_agent import AnthropicAgentModel


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


class TestAnthropicAgentModel(unittest.IsolatedAsyncioTestCase):
    @patch(
        "switch_trust.eval.core.models.model_anthropic_agent.aiohttp.ClientSession"
    )
    async def test_generate_text(self, mock_session_cls):
        mock_session = _make_aiohttp_mocks({"response": "Hello!"})
        mock_session_cls.return_value = mock_session

        model = AnthropicAgentModel(
            host="http://localhost:8000",
        )
        msg = Message(content=Content.text(Role.USER, "Hi"))
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

    @patch(
        "switch_trust.eval.core.models.model_anthropic_agent.aiohttp.ClientSession"
    )
    async def test_generate_empty_response(self, mock_session_cls):
        mock_session = _make_aiohttp_mocks({"response": ""})
        mock_session_cls.return_value = mock_session

        model = AnthropicAgentModel()
        msg = Message(content=Content.text(Role.USER, "Hi"))
        resp = await model.generate(msg)

        self.assertIsNone(resp.message)

    @patch(
        "switch_trust.eval.core.models.model_anthropic_agent.aiohttp.ClientSession"
    )
    async def test_custom_endpoint(self, mock_session_cls):
        mock_session = _make_aiohttp_mocks({"response": "ok"})
        mock_session_cls.return_value = mock_session

        model = AnthropicAgentModel(
            host="http://myhost:9000",
            endpoint="/agent/query",
        )
        msg = Message(content=Content.text(Role.USER, "Hi"))
        await model.generate(msg)

        call_args = mock_session.post.call_args
        self.assertEqual(
            call_args.args[0],
            "http://myhost:9000/agent/query",
        )
        payload = call_args.kwargs["json"]
        self.assertEqual(payload["prompt"], "Hi")


if __name__ == "__main__":
    unittest.main()
