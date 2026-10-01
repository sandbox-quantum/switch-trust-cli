"""Model implementation for Google Agent Runtime (Vertex AI reasoning engines).

Agents deployed to Agent Runtime are invoked by POSTing to the
engine resource with a ``:streamQuery`` suffix::

    POST .../reasoningEngines/{engine_id}:streamQuery
    {"class_method": "stream_query", "input": {"message": "…", "user_id": "…"}}

The product has been renamed twice — Reasoning Engine, then Vertex AI Agent
Engine, now Agent Runtime under Gemini Enterprise Agent Platform — but Google
kept the old API identifiers for backwards compatibility. So ``reasoningEngines``
in the URL is correct, and the service still returns ``Agent Engine Error: …``
on a bad ``class_method``. See ``platform/MODEL_TYPES.md``.

Two things make this incompatible with :class:`GenericHttpModel`, which is
why it needs its own type:

* the body is **nested** and carries a sibling ``class_method`` discriminator.
  ``input`` must be a ``google.protobuf.Struct`` — passing the flat
  ``{"input": "<prompt>"}`` that ``GenericHttpModel`` sends is rejected with
  ``400 Invalid value at 'input' (…Struct)``;
* the response is a **stream of newline-delimited JSON** events, not one JSON
  object, so a single ``resp.json()`` cannot read it.

The non-streaming ``:query`` sibling is not an alternative for ADK-deployed
agents: their ``AdkApp`` wrapper marks every agent-invoking method
(``stream_query``, ``async_stream_query``, ``streaming_agent_run_with_events``)
as streaming, so ``:query`` exposes only session and memory management. Calling
it for ``stream_query`` returns ``404 … method not found``.

``session_id`` is deliberately omitted from the request: ``AdkApp`` creates a
fresh session per call when it is absent, which is what an eval turn wants —
each prompt is scored independently, with no history bleeding between them.

Authentication accepts either form of ``endpoint_credential``:

* a **service-account key** (raw or base64 JSON), from which a scoped access
  token is minted and transparently refreshed — the durable option, since
  Agent Runtime sits behind Google OAuth and a pasted token expires in an hour;
* a **raw OAuth access token** (e.g. ``gcloud auth print-access-token``), sent
  as-is. Convenient for a one-off local run, but it will start returning
  ``401 Unauthorized`` once it expires.
"""

from __future__ import annotations

import asyncio
import json
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

# Agent Runtime is a Cloud Platform API; the broad scope is what service-account
# tokens for aiplatform.googleapis.com are issued against.
CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

DEFAULT_CLASS_METHOD = "stream_query"
DEFAULT_USER_ID = "eval"


class VertexAgentRuntimeModel(Model):
    """Sends prompts to a Google Agent Runtime reasoning engine.

    Args:
        url: The full engine URL including the ``:streamQuery`` suffix.
        credential: A service-account key (raw or base64 JSON) or a raw OAuth
            access token. When None, no ``Authorization`` header is added and
            ``headers`` is expected to carry one.
        headers: Extra headers sent with every request.
        user_id: Identifies the caller to the agent; also scopes the
            auto-created session.
        class_method: The engine method to invoke. Engines that expose a
            differently named streaming entry point (e.g.
            ``streaming_agent_run_with_events``) can override it.
        immediate_result: If True, return the first event's text (e.g. a tool
            call). If False (default), return the final text response after all
            tool calls complete.
        connector_factory: Optional aiohttp connector factory; the worker passes
            the SSRF-guarded one (platform/ssrf.py).
    """

    _url: str
    _headers: dict[str, str]
    _user_id: str
    _class_method: str
    _immediate_result: bool
    _connector_factory: Callable[[], aiohttp.BaseConnector] | None
    # Set when the credential was a service-account key; None when a raw access
    # token (or nothing) was supplied.
    _google_credentials: Any | None
    _static_token: str | None
    _refresh_lock: asyncio.Lock

    def __init__(
        self,
        url: str,
        credential: str | None = None,
        headers: dict[str, str] | None = None,
        user_id: str = DEFAULT_USER_ID,
        class_method: str = DEFAULT_CLASS_METHOD,
        immediate_result: bool = False,
        connector_factory: Callable[[], aiohttp.BaseConnector] | None = None,
    ):
        self._url = url
        self._headers = dict(headers or {})
        self._user_id = user_id
        self._class_method = class_method
        self._immediate_result = immediate_result
        self._connector_factory = connector_factory
        self._google_credentials = _service_account_credentials(credential)
        self._static_token = (
            credential if credential and self._google_credentials is None else None
        )
        self._refresh_lock = asyncio.Lock()

    def _refresh_credentials(self) -> None:
        """Refresh the access token in place.

        Blocking: google-auth's transport is synchronous, so callers run this
        off the event loop.
        """
        from google.auth.transport.requests import (  # noqa: PLC0415 - blocking transport, imported at use
            Request,
        )

        self._google_credentials.refresh(Request())

    async def _auth_headers(self) -> dict[str, str]:
        credentials = self._google_credentials
        if credentials is not None:
            # One model instance serves every prompt in a run concurrently, so
            # the moment the token expires all of them see `valid == False` at
            # once. Without the lock that is N simultaneous refreshes — N token
            # requests, and a race on the token/expiry pair, which google-auth
            # does not guard. Re-check inside the lock so only the first waiter
            # refreshes; the rest fall through to the fresh token. The valid
            # path stays a plain attribute read, off the lock and off a thread.
            if not credentials.valid:
                async with self._refresh_lock:
                    if not credentials.valid:
                        await asyncio.to_thread(self._refresh_credentials)
            return {"Authorization": f"Bearer {credentials.token}"}
        if self._static_token:
            return {"Authorization": f"Bearer {self._static_token}"}
        return {}

    async def _generate(
        self,
        messages: list[Message],
        *,
        output_schema: type[BaseModel] | None = None,
        **_kwargs: Any,
    ) -> ModelResponse:
        # An agent under test cannot enforce a JSON schema; output_schema is
        # accepted for interface parity and ignored.
        prompt_text = flatten_messages(messages)

        payload = {
            "class_method": self._class_method,
            "input": {
                "message": prompt_text,
                "user_id": self._user_id,
            },
        }
        headers = {**self._headers, **await self._auth_headers()}

        connector = self._connector_factory() if self._connector_factory else None
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                self._url,
                json=payload,
                headers=headers,
            ) as resp:
                await raise_for_status(resp)
                # Read the whole stream rather than parsing incrementally: an
                # eval turn needs the complete event list to pick the final
                # text, and responses are a handful of KB.
                body = await resp.text()

        events = _parse_events(body)
        if self._immediate_result:
            response_text = _extract_first(events)
        else:
            response_text = _extract_final(events)

        if not response_text:
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


def _service_account_credentials(credential: str | None) -> Any | None:
    """Build google-auth credentials when ``credential`` is a service-account key.

    Returns None for a raw access token (or no credential), which is the signal
    to send it verbatim as a bearer token. Reuses the generator's parser so the
    base64 handling and the structural/PEM validation stay in one place — a
    newline-mangled key then fails with the same actionable message here.
    """
    if not credential:
        return None

    from google.oauth2 import (  # noqa: PLC0415 - only needed on the service-account path
        service_account,
    )

    from switch_trust.eval.core.models.generator_model import (  # noqa: PLC0415 - avoids a module-level cycle
        _parse_service_account,
    )

    try:
        info = _parse_service_account(credential, source="endpoint_credential")
    except ValueError:
        # Not a service-account key — treat it as a raw access token.
        return None

    return service_account.Credentials.from_service_account_info(
        info,
        scopes=[CLOUD_PLATFORM_SCOPE],
    )


def _parse_events(body: str) -> list[dict]:
    """Parse an Agent Runtime response into a list of event objects.

    The success path is newline-delimited JSON, one event per line, but a
    single-event response is indistinguishable from a bare JSON object and
    errors come back as a JSON array — so try whole-body JSON first and fall
    back to per-line parsing. Unparseable lines are skipped rather than
    raising: a trailing partial line must not discard the events already read.
    """
    text = body.strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        return _as_event_list(parsed)

    events: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.extend(_as_event_list(json.loads(line)))
        except json.JSONDecodeError:
            continue
    return events


def _as_event_list(parsed: Any) -> list[dict]:
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def _texts(event: dict) -> list[str] | None:
    """Return an event's text parts, or None if it carries no content."""
    content = event.get("content")
    if not isinstance(content, dict):
        return None
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None
    return [
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]


def _extract_final(events: list[dict]) -> str | None:
    """Extract the last text response, ignoring tool call/response events."""
    for event in reversed(events):
        texts = _texts(event)
        if texts:
            return " ".join(texts)
    return None


def _extract_first(events: list[dict]) -> str | None:
    """Extract the first event's text content."""
    for event in events:
        texts = _texts(event)
        if texts:
            return " ".join(texts)
    return None
