"""Model for any HTTP endpoint with configurable JSON fields.

Connects to a generic REST API where the request sends
a prompt in a configurable JSON field and the response
returns the output in another configurable field.

Default format::

    POST /your-endpoint
    {"input": "Hello"}

    {"output": "Hi there!"}

The ``output_path`` supports dot-separated paths for
nested responses, e.g. ``"data.response.text"`` or
``"choices.0.message.content"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import aiohttp

from switch_trust.eval.common.schema import Content, Message, Part, Role
from switch_trust.eval.core.models.model import (
    Model,
    ModelResponse,
    ResponseStatus,
    flatten_messages,
)
from switch_trust.eval.core.models.model_http import raise_for_status

if TYPE_CHECKING:
    from pydantic import BaseModel


class GenericHttpModel(Model):
    _url: str
    _headers: dict[str, str]
    _input_path: str
    _output_path: str

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        input_path: str = "input",
        output_path: str = "output",
    ):
        self._url = url
        self._headers = headers or {}
        self._input_path = input_path
        self._output_path = output_path

    async def _generate(
        self,
        messages: list[Message],
        *,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        # A generic REST endpoint under test cannot enforce a JSON schema;
        # output_schema is accepted for interface parity and ignored.
        prompt_text = flatten_messages(messages)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._url,
                json={self._input_path: prompt_text},
                headers=self._headers,
            ) as resp:
                await raise_for_status(resp)
                data = await resp.json()

        output = _resolve_path(data, self._output_path)
        if not output:
            return ModelResponse(
                message=None,
                status=ResponseStatus.EMPTY_RESPONSE,
            )

        message = Message(
            content=Content(
                role=Role.ASSISTANT,
                parts=[Part.text_part(str(output))],
            ),
        )
        return ModelResponse(message=message)


def _resolve_path(data: Any, path: str) -> Any:
    """Traverse a nested dict/list by dot-separated path.

    Supports integer indices for lists, e.g.
    ``"choices.0.message.content"`` resolves
    ``data["choices"][0]["message"]["content"]``.
    """
    current = data
    for key in path.split("."):
        if current is None:
            return None
        if isinstance(current, list):
            try:
                current = current[int(key)]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current
