from typing import Any

from openai import APIError, AsyncOpenAI, BadRequestError
from pydantic import BaseModel

from switch_trust.eval.common import converter_openai
from switch_trust.eval.common.schema import Message
from switch_trust.eval.core.models.model import (
    Model,
    ModelResponse,
    ResponseStatus,
)
from switch_trust.eval.core.models.response_schema import to_openai_response_format


class OpenAIModel(Model):
    _client: AsyncOpenAI
    _model: str
    _temperature: float

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        temperature: float = 0.0,
    ):
        self._client = client
        self._model = model
        self._temperature = temperature

    async def _generate(
        self,
        messages: list[Message],
        *,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        openai_messages = [converter_openai.from_message(m) for m in messages]
        if output_schema is not None and "response_format" not in kwargs:
            kwargs["response_format"] = to_openai_response_format(output_schema)
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=openai_messages,
                temperature=kwargs.pop("temperature", self._temperature),
                **kwargs,
            )
        except APIError as e:
            return ModelResponse(
                None,
                _classify_block_reason(e),
            )
        choice = response.choices[0]
        message = converter_openai.to_message(choice.message)
        return ModelResponse(message)


def _classify_block_reason(api_error: APIError) -> ResponseStatus:
    """Determine why a OpenAI response was blocked."""

    # Check prompt-level blocking
    if isinstance(api_error, BadRequestError):
        code = (
            api_error.body.get("code", "") if isinstance(api_error.body, dict) else ""
        )
        if code == "cyber_policy":
            return ResponseStatus.BLOCKED_PROHIBITED
        if code in ["invalid_prompt", "bio_policy"]:
            return ResponseStatus.BLOCKED_SAFETY

    return ResponseStatus.ERROR
