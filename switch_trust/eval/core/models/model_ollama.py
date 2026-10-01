from typing import Any

import requests
from openai import AsyncOpenAI
from pydantic import BaseModel

from switch_trust.eval.common import converter_openai
from switch_trust.eval.common.schema import Message
from switch_trust.eval.core.models.model import (
    Model,
    ModelResponse,
)
from switch_trust.eval.core.models.response_schema import to_json_schema


class OllamaModel(Model):
    _client: AsyncOpenAI
    _model: str
    _temperature: float

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
    ):
        self._client = AsyncOpenAI(
            base_url=f"{host}/v1",
            api_key="ollama",
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
        if output_schema is not None:
            # Ollama constrains output via its native top-level ``format`` field
            # (a JSON schema), passed through the OpenAI-compatible body.
            extra_body = kwargs.setdefault("extra_body", {})
            extra_body.setdefault("format", to_json_schema(output_schema))
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=openai_messages,
            temperature=kwargs.pop("temperature", self._temperature),
            **kwargs,
        )
        choice = response.choices[0]
        message = converter_openai.to_message(choice.message)
        return ModelResponse(message)


def discover_ollama_models(
    host: str = "http://localhost:11434",
) -> list[str]:
    """Return a list of model names available on an Ollama instance."""
    response = requests.get(f"{host}/api/tags")
    response.raise_for_status()
    data = response.json()
    return [m["name"] for m in data.get("models", [])]
