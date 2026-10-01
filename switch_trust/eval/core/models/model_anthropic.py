import json
from typing import Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from switch_trust.eval.common import converter_anthropic
from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.models.model import Model, ModelResponse
from switch_trust.eval.core.models.response_schema import to_anthropic_tool


class AnthropicModel(Model):
    _client: AsyncAnthropic
    _model: str
    _max_tokens: int
    _temperature: float

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    async def _generate(
        self,
        messages: list[Message],
        *,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        system_blocks = None
        api_messages = []
        for msg in messages:
            anthropic_msg = converter_anthropic.from_message(msg)
            if msg.content.role == Role.SYSTEM:
                system_blocks = anthropic_msg["content"]
            else:
                api_messages.append(anthropic_msg)

        if not api_messages:
            api_messages = [{"role": "user", "content": ""}]

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": kwargs.pop("max_tokens", self._max_tokens),
            "temperature": kwargs.pop("temperature", self._temperature),
            "messages": api_messages,
            **kwargs,
        }
        if system_blocks is not None:
            create_kwargs["system"] = system_blocks

        if output_schema is not None and "tools" not in create_kwargs:
            # Anthropic has no cross-version response_format: force a single
            # tool whose input_schema is the target schema and read the
            # tool_use input back as the JSON payload.
            tool, tool_choice = to_anthropic_tool(output_schema)
            create_kwargs["tools"] = [tool]
            create_kwargs["tool_choice"] = tool_choice

        response = await self._client.messages.create(**create_kwargs)

        if output_schema is not None:
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    payload = json.dumps(block.input)
                    return ModelResponse(
                        Message(content=Content.text(Role.ASSISTANT, payload)),
                    )

        message = converter_anthropic.to_message(response)
        return ModelResponse(message)
