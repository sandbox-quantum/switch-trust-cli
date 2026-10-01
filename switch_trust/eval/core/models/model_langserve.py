"""Model for LangServe endpoints.

LangServe exposes LangChain runnables as REST APIs with
a standard ``/{chain}/invoke`` endpoint.

Request format::

    POST /my-chain/invoke
    {"input": "Hello"}

Response format::

    {"output": "Hi there!", "metadata": {...}}
"""

from __future__ import annotations

from collections.abc import Callable
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


class LangServeModel(Model):
    _url: str
    _headers: dict[str, str]
    # Optional aiohttp connector factory; the worker passes the SSRF-guarded one
    # (platform/ssrf.py). None keeps the default connector.
    _connector_factory: Callable[[], aiohttp.BaseConnector] | None

    def __init__(
        self,
        base_url: str,
        chain_path: str = "",
        headers: dict[str, str] | None = None,
        connector_factory: Callable[[], aiohttp.BaseConnector] | None = None,
    ):
        base = base_url.rstrip("/")
        path = chain_path.strip("/")
        if path:
            self._url = f"{base}/{path}/invoke"
        else:
            self._url = f"{base}/invoke"
        self._headers = headers or {}
        self._connector_factory = connector_factory

    async def _generate(
        self,
        messages: list[Message],
        *,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        # A LangServe endpoint under test cannot enforce a JSON schema;
        # output_schema is accepted for interface parity and ignored.
        prompt_text = flatten_messages(messages)

        connector = self._connector_factory() if self._connector_factory else None
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                self._url,
                json={"input": prompt_text},
                headers=self._headers,
            ) as resp:
                await raise_for_status(resp)
                data = await resp.json()

        output = data.get("output")
        if not output:
            return ModelResponse(
                message=None,
                status=ResponseStatus.EMPTY_RESPONSE,
            )

        if isinstance(output, dict):
            output = output.get("content", str(output))

        message = Message(
            content=Content(
                role=Role.ASSISTANT,
                parts=[Part.text_part(str(output))],
            ),
        )
        return ModelResponse(message=message)
