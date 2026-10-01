from typing import Any

import litellm
from pydantic import BaseModel

from switch_trust.eval.common import converter_openai
from switch_trust.eval.common.schema import Message
from switch_trust.eval.core.models.model import (
    Model,
    ModelResponse,
)


class LiteLLMModel(Model):
    """Model adapter that delegates to LiteLLM.

    LiteLLM uses the OpenAI message format for all providers, so this
    adapter reuses ``converter_openai`` for serialisation.
    """

    _model: str
    _temperature: float

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
    ):
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
            # LiteLLM accepts a Pydantic class and translates it to each
            # provider's structured-output mechanism.
            kwargs["response_format"] = output_schema
        response = await litellm.acompletion(
            model=self._model,
            messages=openai_messages,
            temperature=kwargs.pop("temperature", self._temperature),
            **kwargs,
        )
        choice = response.choices[0]
        message = converter_openai.to_message(choice.message)
        return ModelResponse(message)
