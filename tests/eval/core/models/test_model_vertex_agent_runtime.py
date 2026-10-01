import asyncio
import json
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.models.model import ResponseStatus
from switch_trust.eval.core.models.model_vertex_agent_runtime import (
    VertexAgentRuntimeModel,
    _extract_final,
    _extract_first,
    _parse_events,
    _service_account_credentials,
)

MIXED_EVENT = {
    "content": {
        "parts": [{"text": "answer"}, {"function_call": {"name": "search"}}],
        "role": "model",
    }
}

URL = (
    "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/"
    "us-central1/reasoningEngines/123:streamQuery"
)


def _event(text: str, role: str = "model") -> dict:
    return {"content": {"parts": [{"text": text}], "role": role}}


def _tool_call_event(name: str) -> dict:
    return {"content": {"parts": [{"function_call": {"name": name}}], "role": "model"}}


def _make_aiohttp_mocks(body: str):
    """Create mock aiohttp session and response for a single POST call."""
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value=body)
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    return mock_session


class TestParseEvents(unittest.TestCase):
    def test_single_json_object(self):
        body = json.dumps(_event("hi"))
        self.assertEqual(len(_parse_events(body)), 1)

    def test_newline_delimited(self):
        body = "\n".join(json.dumps(_event(t)) for t in ("a", "b", "c"))
        self.assertEqual(len(_parse_events(body)), 3)

    def test_json_array(self):
        body = json.dumps([_event("a"), _event("b")])
        self.assertEqual(len(_parse_events(body)), 2)

    def test_empty_body(self):
        self.assertEqual(_parse_events("   "), [])

    def test_trailing_partial_line_is_skipped(self):
        # A truncated final chunk must not discard the events already read.
        body = json.dumps(_event("a")) + "\n" + '{"content": {"parts": ['
        events = _parse_events(body)
        self.assertEqual(len(events), 1)


class TestExtract(unittest.TestCase):
    def test_final_skips_tool_events(self):
        events = [_event("thinking"), _tool_call_event("search"), _event("answer")]
        self.assertEqual(_extract_final(events), "answer")

    def test_final_ignores_trailing_tool_event(self):
        events = [_event("answer"), _tool_call_event("search")]
        self.assertEqual(_extract_final(events), "answer")

    def test_camel_case_tool_event_also_skipped(self):
        events = [
            _event("answer"),
            {"content": {"parts": [{"functionCall": {"name": "s"}}]}},
        ]
        self.assertEqual(_extract_final(events), "answer")

    def test_mixed_text_and_tool_call_keeps_the_text(self):
        # A turn holding both narration and a tool call must not lose the text.
        # Gemini emits this shape routinely.
        self.assertEqual(_extract_final([MIXED_EVENT]), "answer")

    def test_mixed_event_does_not_shadow_a_later_answer(self):
        events = [MIXED_EVENT, _event("final")]
        self.assertEqual(_extract_final(events), "final")

    def test_first_returns_earliest_text(self):
        events = [_event("first"), _event("second")]
        self.assertEqual(_extract_first(events), "first")

    def test_no_content_returns_none(self):
        self.assertIsNone(_extract_final([{"usage_metadata": {}}]))


class TestServiceAccountCredentials(unittest.TestCase):
    def test_raw_access_token_is_not_a_service_account(self):
        self.assertIsNone(_service_account_credentials("ya29.a0Af_notjson"))

    def test_empty_credential(self):
        self.assertIsNone(_service_account_credentials(None))
        self.assertIsNone(_service_account_credentials(""))

    def test_non_service_account_json_falls_back_to_token(self):
        self.assertIsNone(_service_account_credentials('{"type": "authorized_user"}'))


class TestVertexAgentRuntimeModel(unittest.IsolatedAsyncioTestCase):
    @patch(
        "switch_trust.eval.core.models.model_vertex_agent_runtime.aiohttp.ClientSession"
    )
    async def test_request_body_shape(self, mock_session_cls):
        mock_session_cls.return_value = _make_aiohttp_mocks(json.dumps(_event("hi")))

        model = VertexAgentRuntimeModel(url=URL, user_id="tester")
        await model.generate(Message(content=Content.text(Role.USER, "Hello")))

        payload = mock_session_cls.return_value.post.call_args.kwargs["json"]
        # The nested Struct + sibling class_method that GenericHttpModel cannot
        # express, and the reason this model type exists.
        self.assertEqual(payload["class_method"], "stream_query")
        self.assertEqual(payload["input"]["message"], "Hello")
        self.assertEqual(payload["input"]["user_id"], "tester")
        # No session_id: AdkApp creates a fresh session per call, so eval turns
        # stay independent.
        self.assertNotIn("session_id", payload["input"])

    @patch(
        "switch_trust.eval.core.models.model_vertex_agent_runtime.aiohttp.ClientSession"
    )
    async def test_streamed_events_return_final_text(self, mock_session_cls):
        body = "\n".join(
            json.dumps(e)
            for e in (_event("let me look"), _tool_call_event("search"), _event("done"))
        )
        mock_session_cls.return_value = _make_aiohttp_mocks(body)

        model = VertexAgentRuntimeModel(url=URL)
        resp = await model.generate(Message(content=Content.text(Role.USER, "Hi")))

        self.assertEqual(resp.message.content.parts[0].text, "done")

    @patch(
        "switch_trust.eval.core.models.model_vertex_agent_runtime.aiohttp.ClientSession"
    )
    async def test_immediate_result_returns_first_text(self, mock_session_cls):
        body = "\n".join(json.dumps(e) for e in (_event("first"), _event("last")))
        mock_session_cls.return_value = _make_aiohttp_mocks(body)

        model = VertexAgentRuntimeModel(url=URL, immediate_result=True)
        resp = await model.generate(Message(content=Content.text(Role.USER, "Hi")))

        self.assertEqual(resp.message.content.parts[0].text, "first")

    @patch(
        "switch_trust.eval.core.models.model_vertex_agent_runtime.aiohttp.ClientSession"
    )
    async def test_no_text_yields_empty_response(self, mock_session_cls):
        mock_session_cls.return_value = _make_aiohttp_mocks(
            json.dumps(_tool_call_event("search"))
        )

        model = VertexAgentRuntimeModel(url=URL)
        resp = await model.generate(Message(content=Content.text(Role.USER, "Hi")))

        self.assertIsNone(resp.message)
        self.assertEqual(resp.status, ResponseStatus.EMPTY_RESPONSE)

    @patch(
        "switch_trust.eval.core.models.model_vertex_agent_runtime.aiohttp.ClientSession"
    )
    async def test_raw_access_token_sent_as_bearer(self, mock_session_cls):
        mock_session_cls.return_value = _make_aiohttp_mocks(json.dumps(_event("hi")))

        model = VertexAgentRuntimeModel(url=URL, credential="ya29.token")
        await model.generate(Message(content=Content.text(Role.USER, "Hi")))

        headers = mock_session_cls.return_value.post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer ya29.token")

    @patch(
        "switch_trust.eval.core.models.model_vertex_agent_runtime.aiohttp.ClientSession"
    )
    async def test_service_account_token_is_refreshed(self, mock_session_cls):
        mock_session_cls.return_value = _make_aiohttp_mocks(json.dumps(_event("hi")))

        model = VertexAgentRuntimeModel(url=URL)
        credentials = MagicMock()
        credentials.valid = False
        credentials.token = "minted-token"
        model._google_credentials = credentials

        with patch(
            "google.auth.transport.requests.Request",
            MagicMock(),
        ):
            await model.generate(Message(content=Content.text(Role.USER, "Hi")))

        credentials.refresh.assert_called_once()
        headers = mock_session_cls.return_value.post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer minted-token")

    @patch(
        "switch_trust.eval.core.models.model_vertex_agent_runtime.aiohttp.ClientSession"
    )
    async def test_valid_token_is_not_refreshed(self, mock_session_cls):
        mock_session_cls.return_value = _make_aiohttp_mocks(json.dumps(_event("hi")))

        model = VertexAgentRuntimeModel(url=URL)
        credentials = MagicMock()
        credentials.valid = True
        credentials.token = "cached-token"
        model._google_credentials = credentials

        await model.generate(Message(content=Content.text(Role.USER, "Hi")))

        credentials.refresh.assert_not_called()
        headers = mock_session_cls.return_value.post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer cached-token")

    @patch(
        "switch_trust.eval.core.models.model_vertex_agent_runtime.aiohttp.ClientSession"
    )
    async def test_concurrent_prompts_refresh_once(self, mock_session_cls):
        # An expiring token makes every in-flight prompt see valid == False at
        # the same moment; the lock must collapse that into one refresh.
        mock_session_cls.return_value = _make_aiohttp_mocks(json.dumps(_event("hi")))

        model = VertexAgentRuntimeModel(url=URL)
        credentials = MagicMock()
        credentials.valid = False
        credentials.token = "minted-token"

        # The refresh has to be slow enough to hold the window open. An
        # instantaneous one flips `valid` before the other tasks reach their
        # check, so the test would pass with or without the lock.
        def _refresh(_request):
            time.sleep(0.05)
            credentials.valid = True

        credentials.refresh.side_effect = _refresh
        model._google_credentials = credentials

        with patch("google.auth.transport.requests.Request", MagicMock()):
            await asyncio.gather(
                *(
                    model.generate(Message(content=Content.text(Role.USER, f"Hi {i}")))
                    for i in range(8)
                )
            )

        credentials.refresh.assert_called_once()

    @patch(
        "switch_trust.eval.core.models.model_vertex_agent_runtime.aiohttp.ClientSession"
    )
    async def test_custom_class_method(self, mock_session_cls):
        mock_session_cls.return_value = _make_aiohttp_mocks(json.dumps(_event("hi")))

        model = VertexAgentRuntimeModel(
            url=URL,
            class_method="streaming_agent_run_with_events",
        )
        await model.generate(Message(content=Content.text(Role.USER, "Hi")))

        payload = mock_session_cls.return_value.post.call_args.kwargs["json"]
        self.assertEqual(payload["class_method"], "streaming_agent_run_with_events")

    @patch(
        "switch_trust.eval.core.models.model_vertex_agent_runtime.aiohttp.ClientSession"
    )
    async def test_multi_message_flattened(self, mock_session_cls):
        mock_session_cls.return_value = _make_aiohttp_mocks(json.dumps(_event("ok")))

        model = VertexAgentRuntimeModel(url=URL)
        await model.generate(
            [
                Message(content=Content.text(Role.USER, "Hello")),
                Message(content=Content.text(Role.ASSISTANT, "Hi!")),
            ]
        )

        prompt = mock_session_cls.return_value.post.call_args.kwargs["json"]["input"][
            "message"
        ]
        self.assertIn("USER: Hello", prompt)
        self.assertIn("ASSISTANT: Hi!", prompt)


if __name__ == "__main__":
    unittest.main()
