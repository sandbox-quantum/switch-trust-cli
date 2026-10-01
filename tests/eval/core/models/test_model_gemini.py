import unittest
from unittest.mock import AsyncMock, MagicMock

from google.genai import types as genai_types
from pydantic import BaseModel

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.models.model import ResponseStatus
from switch_trust.eval.core.models.model_gemini import SAFETY_OFF, GeminiModel


class _Score(BaseModel):
    score: float


def _make_response(text: str):
    response = MagicMock()
    content = genai_types.Content(
        role="model",
        parts=[genai_types.Part(text=text)],
    )
    candidate = MagicMock()
    candidate.content = content
    response.candidates = [candidate]
    return response


def _make_blocked_response():
    """Simulate a response blocked by safety filters."""
    response = MagicMock()
    response.candidates = None
    return response


def _make_blocked_response_empty_content():
    """Simulate a blocked response with a candidate but no content."""
    response = MagicMock()
    candidate = MagicMock()
    candidate.content = None
    response.candidates = [candidate]
    return response


class TestGeminiModel(unittest.IsolatedAsyncioTestCase):
    async def test_generate_text(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_response("Hello!")
        )

        model = GeminiModel(client=mock_client, model="gemini-2.5-flash")
        msg = Message(content=Content.text(Role.USER, "Hi"))
        resp = await model.generate(msg)

        self.assertIsNotNone(resp.message)
        self.assertEqual(resp.status, ResponseStatus.OK)
        self.assertEqual(resp.message.content.role, Role.ASSISTANT)
        self.assertEqual(
            resp.message.content.parts[0].text,
            "Hello!",
        )

    async def test_output_schema_sets_config(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_response('{"score": 1.0}')
        )

        model = GeminiModel(client=mock_client, model="gemini-2.5-flash")
        msg = Message(content=Content.text(Role.USER, "Hi"))
        await model.generate(msg, output_schema=_Score)

        config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertIs(config.response_schema, _Score)

    async def test_blocked_response_returns_none_message(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_blocked_response()
        )

        model = GeminiModel(client=mock_client, model="gemini-2.5-flash")
        msg = Message(content=Content.text(Role.USER, "bad prompt"))
        resp = await model.generate(msg)

        self.assertIsNone(resp.message)
        self.assertEqual(resp.status, ResponseStatus.BLOCKED_SAFETY)

    async def test_blocked_response_empty_content(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_blocked_response_empty_content()
        )

        model = GeminiModel(client=mock_client, model="gemini-2.5-flash")
        msg = Message(content=Content.text(Role.USER, "bad prompt"))
        resp = await model.generate(msg)

        self.assertIsNone(resp.message)
        self.assertEqual(resp.status, ResponseStatus.BLOCKED_SAFETY)

    async def test_blocked_with_safety_finish_reason(self):
        mock_client = MagicMock()
        response = MagicMock()
        candidate = MagicMock()
        candidate.content = None
        candidate.finish_reason = "SAFETY"
        response.candidates = [candidate]
        response.prompt_feedback = None
        mock_client.aio.models.generate_content = AsyncMock(return_value=response)

        model = GeminiModel(client=mock_client, model="gemini-2.5-flash")
        msg = Message(content=Content.text(Role.USER, "bad"))
        resp = await model.generate(msg)

        self.assertIsNone(resp.message)
        self.assertEqual(resp.status, ResponseStatus.BLOCKED_SAFETY)

    async def test_blocked_with_recitation_finish_reason(self):
        mock_client = MagicMock()
        response = MagicMock()
        candidate = MagicMock()
        candidate.content = None
        candidate.finish_reason = "RECITATION"
        response.candidates = [candidate]
        response.prompt_feedback = None
        mock_client.aio.models.generate_content = AsyncMock(return_value=response)

        model = GeminiModel(client=mock_client, model="gemini-2.5-flash")
        msg = Message(content=Content.text(Role.USER, "copy"))
        resp = await model.generate(msg)

        self.assertIsNone(resp.message)
        self.assertEqual(
            resp.status,
            ResponseStatus.BLOCKED_RECITATION,
        )


class TestGeminiSafetySettings(unittest.IsolatedAsyncioTestCase):
    def _client_capturing_config(self):
        """A mock client whose generate_content records the config it was given."""
        captured = {}

        async def generate_content(**kwargs):
            captured["config"] = kwargs.get("config")
            return _make_response("ok")

        client = MagicMock()
        client.aio.models.generate_content = AsyncMock(side_effect=generate_content)
        return client, captured

    async def test_default_leaves_safety_settings_unset(self):
        client, captured = self._client_capturing_config()
        model = GeminiModel(client=client, model="gemini-2.5-flash")
        await model.generate(Message(content=Content.text(Role.USER, "hi")))
        self.assertIsNone(captured["config"].safety_settings)

    async def test_safety_off_is_applied_to_config(self):
        client, captured = self._client_capturing_config()
        model = GeminiModel(
            client=client, model="gemini-2.5-flash", safety_settings=SAFETY_OFF
        )
        await model.generate(Message(content=Content.text(Role.USER, "hi")))
        self.assertEqual(captured["config"].safety_settings, SAFETY_OFF)
        self.assertTrue(
            all(s.threshold == "OFF" for s in captured["config"].safety_settings)
        )

    async def test_explicit_config_safety_settings_are_not_overridden(self):
        client, captured = self._client_capturing_config()
        model = GeminiModel(
            client=client, model="gemini-2.5-flash", safety_settings=SAFETY_OFF
        )
        caller_config = genai_types.GenerateContentConfig(
            safety_settings=[
                genai_types.SafetySetting(
                    category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold="BLOCK_LOW_AND_ABOVE",
                )
            ]
        )
        await model.generate(
            Message(content=Content.text(Role.USER, "hi")), config=caller_config
        )
        self.assertEqual(len(captured["config"].safety_settings), 1)
        self.assertEqual(
            captured["config"].safety_settings[0].threshold, "BLOCK_LOW_AND_ABOVE"
        )


if __name__ == "__main__":
    unittest.main()
