import unittest
from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import APIStatusError, BadRequestError
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import ChatCompletion, Choice
from pydantic import BaseModel

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.models.model import (
    ResponseStatus,
    track_model_errors,
)
from switch_trust.eval.core.models.model_errors import ErrorKind
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


def _bad_request(code: str) -> BadRequestError:
    return BadRequestError(
        "rejected",
        response=httpx.Response(
            400, request=httpx.Request("POST", "https://api.openai.com/v1/x")
        ),
        body={"code": code},
    )


def _api_status_error(status: int) -> APIStatusError:
    return APIStatusError(
        f"http {status}",
        response=httpx.Response(
            status, request=httpx.Request("POST", "https://api.openai.com/v1/x")
        ),
        body=None,
    )


class TestOpenAIModelErrors(unittest.IsolatedAsyncioTestCase):
    """Only a content block becomes a response; everything else propagates.

    This adapter used to swallow every `APIError` into a message-less
    `ResponseStatus.ERROR`, which cost the caller the HTTP status: nothing
    reached `ModelErrorAccumulator`, so the CLI's 20-39/60-79 exit codes and
    `MultiEvaluation`'s auth fail-fast — both of which read that accumulator —
    could never fire for an `openai` model.
    """

    def _model(self, error: Exception) -> OpenAIModel:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=error)
        return OpenAIModel(client=mock_client, model="gpt-4o")

    async def _generate(self, error: Exception):
        model = self._model(error)
        return await model.generate(Message(content=Content.text(Role.USER, "Hi")))

    async def test_a_prohibited_prompt_is_a_blocked_response(self):
        resp = await self._generate(_bad_request("cyber_policy"))
        self.assertIsNone(resp.message)
        self.assertEqual(resp.status, ResponseStatus.BLOCKED_PROHIBITED)

    async def test_an_unsafe_prompt_is_a_blocked_response(self):
        for code in ("invalid_prompt", "bio_policy"):
            with self.subTest(code=code):
                resp = await self._generate(_bad_request(code))
                self.assertIsNone(resp.message)
                self.assertEqual(resp.status, ResponseStatus.BLOCKED_SAFETY)

    async def test_an_unrecognised_bad_request_propagates(self):
        """A 400 is only a block when it carries one of the block codes; any
        other is a request the provider refused, and the caller must see it."""
        error = _bad_request("context_length_exceeded")
        with self.assertRaises(BadRequestError):
            await self._generate(error)

    async def test_a_missing_model_propagates_with_its_status(self):
        """The reported case: `model_name` the provider does not have. A 404
        has to reach `Model.generate()` to be recorded and classified."""
        with self.assertRaises(APIStatusError) as ctx:
            await self._generate(_api_status_error(404))
        self.assertEqual(ctx.exception.status_code, 404)

    async def test_a_rejected_key_propagates(self):
        with self.assertRaises(APIStatusError) as ctx:
            await self._generate(_api_status_error(401))
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_a_propagated_error_is_recorded_for_classification(self):
        """The accumulator is what every classified exit code is read from,
        so recording is the point of propagating rather than returning."""
        with track_model_errors() as accumulator:
            with self.assertRaises(APIStatusError):
                await self._generate(_api_status_error(404))

        self.assertEqual(accumulator.dominant_kind(), ErrorKind.UNREACHABLE)
        statuses = {key.status for key in accumulator.counts()}
        self.assertEqual(statuses, {"404"})


if __name__ == "__main__":
    unittest.main()
