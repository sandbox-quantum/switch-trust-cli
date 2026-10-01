"""Model implementation for Google ADK agents.

ADK agents are served via ``adk api_server`` or ``adk web`` and
expose a REST API at a configurable host/port.  This model sends
prompts to the ``/run`` endpoint and extracts the agent's text
response from the returned events.

Typical ADK server URL: ``http://localhost:8000``
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import aiohttp
import requests

from switch_trust.eval.common.schema import Content, Message, Part, Role
from switch_trust.eval.core.models.model import Model, ModelResponse, ResponseStatus
from switch_trust.eval.core.models.model_http import raise_for_status

if TYPE_CHECKING:
    from pydantic import BaseModel


class ADKModel(Model):
    """Sends prompts to a Google ADK agent served over HTTP.

    Args:
        app_name: The ADK app/agent name.
        host: The HTTP host where the ADK server is running.
        user_id: User identifier for sessions.
        immediate_result: If True, return the first event's
            content (e.g. a tool call). If False (default),
            return the final text response after all tool
            calls complete.
        headers: Extra HTTP headers sent on every request, e.g. an
            ``Authorization`` header for a deployment gated by an
            app-level bearer token rather than Cloud Run IAM.
    """

    def __init__(
        self,
        app_name: str,
        host: str = "http://localhost:8000",
        user_id: str = "aired",
        immediate_result: bool = False,
        headers: dict[str, str] | None = None,
        connector_factory: Callable[[], aiohttp.BaseConnector] | None = None,
    ):
        self._app_name = app_name
        self._host = host.rstrip("/")
        self._user_id = user_id
        self._immediate_result = immediate_result
        self._headers = headers or {}
        # Optional aiohttp connector factory; the worker passes the SSRF-guarded
        # one (platform/ssrf.py). None keeps the default connector.
        self._connector_factory = connector_factory

    async def _create_session(
        self,
        session: aiohttp.ClientSession,
    ) -> str:
        """Create a new session for this request."""
        url = f"{self._host}/apps/{self._app_name}/users/{self._user_id}/sessions"
        async with session.post(url, json={}, headers=self._headers) as resp:
            await raise_for_status(resp)
            data = await resp.json()
            return data["id"]

    async def _generate(
        self,
        messages: list[Message],
        *,
        output_schema: type[BaseModel] | None = None,
        **_kwargs: Any,
    ) -> ModelResponse:
        # An agent under test cannot enforce a JSON schema; output_schema is
        # accepted for interface parity and ignored.
        if len(messages) > 1:
            raise ValueError(
                "ADKModel does not support multiple messages; use session-based history"
            )

        connector = self._connector_factory() if self._connector_factory else None
        async with aiohttp.ClientSession(connector=connector) as session:
            session_id = await self._create_session(session)

            text_parts = [p.text for p in messages[0].content.parts if p.text]
            prompt_text = " ".join(text_parts)

            payload = {
                "appName": self._app_name,
                "userId": self._user_id,
                "sessionId": session_id,
                "newMessage": {
                    "role": "user",
                    "parts": [{"text": prompt_text}],
                },
            }

            async with session.post(
                f"{self._host}/run",
                json=payload,
                headers=self._headers,
            ) as resp:
                await raise_for_status(resp)
                events = await resp.json()

        if self._immediate_result:
            response_text = self._extract_first(events)
        else:
            response_text = self._extract_final(events)

        if response_text is None:
            return ModelResponse(
                message=None,
                status=ResponseStatus.EMPTY_RESPONSE,
            )

        message = Message(
            content=Content(
                role=Role.ASSISTANT,
                parts=[Part.text_part(response_text)],
            ),
        )
        return ModelResponse(message=message)

    def _extract_final(
        self,
        events: list[dict],
    ) -> str | None:
        """Extract the last text response, skipping tool
        call/response events.

        FIXME: the skip below drops a *mixed* event — one turn carrying both
        narration and a tool call, e.g. ``[{"text": "Let me look that up"},
        {"functionCall": ...}]``, which Gemini emits routinely — losing the text
        with it. The test is also redundant for pure-tool events, since those
        carry no ``text`` key and the ``if texts`` check below already skips
        them. Deleting the whole ``any(...)`` block both fixes and simplifies
        this, as done in ``model_vertex_agent_runtime._extract_final``. Left
        unchanged here only to keep that PR scoped; needs its own change plus a
        mixed-event test.
        """
        for event in reversed(events):
            content = event.get("content")
            if content is None:
                continue
            parts = content.get("parts", [])
            # Skip events that are tool calls or responses
            if any(
                "functionCall" in p or "functionResponse" in p
                for p in parts
                if isinstance(p, dict)
            ):
                continue
            texts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
            if texts:
                return " ".join(texts)
        return None

    def _extract_first(
        self,
        events: list[dict],
    ) -> str | None:
        """Extract the first event's text content."""
        for event in events:
            content = event.get("content")
            if content is None:
                continue
            parts = content.get("parts", [])
            texts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
            if texts:
                return " ".join(texts)
        return None


def discover_adk_agents(
    host: str = "http://localhost:8000",
) -> list[str]:
    """Return a list of agent app names on an ADK server."""
    resp = requests.get(f"{host.rstrip('/')}/list-apps")
    resp.raise_for_status()
    return resp.json()
