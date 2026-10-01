"""Model implementation for OpenAI Agents SDK agents served over HTTP.

OpenAI Agents SDK agents don't have a built-in HTTP server.
The expected pattern is a FastAPI wrapper with a POST endpoint
that accepts ``{"input": "<prompt>"}`` and returns
``{"output": "<response>"}``.

Example server::

    from fastapi import FastAPI
    from pydantic import BaseModel
    from agents import Agent, Runner

    app = FastAPI()
    agent = Agent(name="my_agent", instructions="...")

    class Request(BaseModel):
        input: str

    @app.post("/run")
    async def run(req: Request):
        result = await Runner.run(agent, req.input)
        return {"output": result.final_output}
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import aiohttp

from switch_trust.eval.common.schema import Content, Message, Part, Role
from switch_trust.eval.core.models.model import Model, ModelResponse, ResponseStatus
from switch_trust.eval.core.models.model_http import raise_for_status

if TYPE_CHECKING:
    from pydantic import BaseModel


class OpenAIAgentModel(Model):
    """Sends prompts to an OpenAI Agents SDK agent over HTTP."""

    def __init__(
        self,
        host: str = "http://localhost:8000",
        endpoint: str = "/run",
        connector_factory: Callable[[], aiohttp.BaseConnector] | None = None,
    ):
        self._host = host.rstrip("/")
        self._endpoint = endpoint
        # Optional aiohttp connector factory; the worker passes the SSRF-guarded
        # one (platform/ssrf.py). None keeps the default connector.
        self._connector_factory = connector_factory

    async def _generate(
        self,
        messages: list[Message],
        *,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        # An agent under test cannot enforce a JSON schema; output_schema is
        # accepted for interface parity and ignored.
        if len(messages) > 1:
            raise ValueError("OpenAIAgentModel does not support multiple messages")
        text_parts = [p.text for p in messages[0].content.parts if p.text]
        prompt_text = " ".join(text_parts)

        connector = self._connector_factory() if self._connector_factory else None
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                f"{self._host}{self._endpoint}",
                json={"input": prompt_text},
            ) as resp:
                await raise_for_status(resp)
                data = await resp.json()

        output = data.get("output")
        if not output:
            return ModelResponse(
                message=None,
                status=ResponseStatus.EMPTY_RESPONSE,
            )

        message = Message(
            content=Content(
                role=Role.ASSISTANT,
                parts=[Part.text_part(output)],
            ),
        )
        return ModelResponse(message=message)
