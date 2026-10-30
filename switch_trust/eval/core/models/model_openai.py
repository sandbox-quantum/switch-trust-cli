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
            blocked = _classify_block_reason(e)
            if blocked is None:
                raise
            return ModelResponse(
                None,
                blocked,
            )
        choice = response.choices[0]
        message = converter_openai.to_message(choice.message)
        return ModelResponse(message)


def _classify_block_reason(api_error: APIError) -> ResponseStatus | None:
    """Why the provider's own safety layer declined, or None if it did not.

    Only a *prompt-level content block* becomes a response: the request
    reached the model, the model refused it, and that refusal is the
    measurement — `BLOCKED_STATUSES` scores it as a pass rather than an error.

    Everything else re-raises, and the `None` return is what says so. This
    used to fall through to `ResponseStatus.ERROR`, which turned a 401, a 404
    for a model name the provider does not have, a 429 or a 5xx into a
    message-less response that no caller could tell apart from a block:
    `Model.generate()` never saw an exception, so it recorded nothing on
    `ModelErrorAccumulator`, and the whole 20-39/60-79 half of the exit table
    (plus `MultiEvaluation`'s auth fail-fast, which reads the same
    accumulator) was unreachable for every `openai` model. The run instead
    errored every prompt on `SinglePromptEvaluation`'s "did not return a
    message or an error status" and exited 6. Every other adapter lets the
    provider's exception propagate; this one is why it must.
    """
    # Check prompt-level blocking
    if isinstance(api_error, BadRequestError):
        code = (
            api_error.body.get("code", "") if isinstance(api_error.body, dict) else ""
        )
        if code == "cyber_policy":
            return ResponseStatus.BLOCKED_PROHIBITED
        if code in ["invalid_prompt", "bio_policy"]:
            return ResponseStatus.BLOCKED_SAFETY

    return None
