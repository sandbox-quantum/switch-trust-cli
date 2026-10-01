"""Model for any OpenAI-compatible chat completions endpoint.

Works with vLLM, SGLang, Ollama, LiteLLM proxy, Azure
OpenAI, Amazon Bedrock, TGI, and any other server that
implements the ``/v1/chat/completions`` API.
"""

from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from switch_trust.eval.common import converter_openai
from switch_trust.eval.common.schema import Message
from switch_trust.eval.core.models.model import (
    Model,
    ModelResponse,
)
from switch_trust.eval.core.models.response_schema import to_openai_response_format


class OpenAICompatibleModel(Model):
    _client: AsyncOpenAI
    _model: str
    _temperature: float

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        headers: dict[str, str] | None = None,
    ):
        self._client = AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            default_headers=headers or {},
        )
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
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=openai_messages,
            temperature=kwargs.pop(
                "temperature",
                self._temperature,
            ),
            **kwargs,
        )
        choice = response.choices[0]
        message = converter_openai.to_message(
            choice.message,
        )
        return ModelResponse(message)
