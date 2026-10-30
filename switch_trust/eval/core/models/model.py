import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import Any

from pydantic import BaseModel

from switch_trust.eval.common.log import redact
from switch_trust.eval.common.schema import Content, Message, PartType, Role
from switch_trust.eval.core.models.model_errors import (
    BLOCKED_STATUS,
    HOST_NOT_FOUND,
    ErrorKind,
    ModelErrorKey,
    SsrfBlockedError,
    classify_model_error,
    error_status_of,
    in_generator_scope,
    is_host_not_found,
    log_level,
    mark_generator_failure,
    status_of,
)

logger = logging.getLogger(__name__)

# Decorators call the inner Model.generate(), so mark failures to log them once.
_LOGGED_ATTR = "_flint_generate_logged"


class ResponseStatus(str, Enum):
    """Outcome of a model generation request."""

    OK = "ok"
    BLOCKED_SAFETY = "blocked_safety"
    BLOCKED_RECITATION = "blocked_recitation"
    BLOCKED_PROHIBITED = "blocked_prohibited"
    EMPTY_RESPONSE = "empty_response"
    ERROR = "error"


# A response in one of these statuses means the model/agent's own safety layer
# declined to answer, not that it produced nothing to evaluate — callers should
# treat it as a pass rather than feeding it to a detector.
BLOCKED_STATUSES = {
    ResponseStatus.BLOCKED_SAFETY,
    ResponseStatus.BLOCKED_RECITATION,
    ResponseStatus.BLOCKED_PROHIBITED,
    ResponseStatus.EMPTY_RESPONSE,
}


ModelContent = str | Message | list[Message]


# Set by PerEndpointLimiter around the call it delegates, so the choke point
# below can name the endpoint without every model carrying a URL accessor. The
# limiter wraps exactly the model types that have a tenant endpoint, so an unset
# value means "no endpoint to name" rather than "we lost it".
_endpoint_url: ContextVar[str | None] = ContextVar("_endpoint_url", default=None)


@contextmanager
def endpoint_scope(endpoint_url: str) -> Iterator[None]:
    """Attribute failures raised inside this scope to `endpoint_url`."""
    token = _endpoint_url.set(endpoint_url)
    try:
        yield
    finally:
        _endpoint_url.reset(token)


# Caps one stored provider message. Long enough for a real error, short enough
# that a provider echoing the request body can't bloat the summary JSONB.
_MAX_LAST_ERROR_CHARS = 500


class ModelErrorAccumulator:
    """Counts model/agent call failures recorded during a `track_model_errors()`
    scope, keyed by `ModelErrorKey`, and keeps the last message per key.

    Mirrors `TokenUsageAccumulator`: every `Model.generate()` call funnels
    through the one choke point below, so this is the one place that can
    capture failures for a whole run without threading a return value through
    `Evaluation`/`MultiEvaluation`/every `Detector` subclass.
    """

    def __init__(self) -> None:
        self._counts: dict[ModelErrorKey, int] = {}
        self._last_errors: dict[ModelErrorKey, str] = {}
        self._recorded_ids: set[int] = set()

    def record_once(
        self,
        exc: Exception,
        model_type: str,
        status: object,
        error_status: str | None = None,
        kind: ErrorKind = ErrorKind.UNKNOWN,
    ) -> None:
        """Like `record`, but skips an exception object already recorded in
        this scope.

        A decorated model (`RetryModel`, `GeneratorModel`) re-raises the
        *same* exception object through each wrapper layer's own
        `Model.generate()` call as it propagates up, so without this a single
        physical failure would be recorded once per decorator layer.
        Deduping by `id(exc)` keeps the *first* (innermost) recording — the
        concrete provider/agent class where the failure actually originated
        — which is the useful signal, not the `RetryModel`/`GeneratorModel`
        wrapper re-observing it.

        `error_status` is `ModelHttpError`'s fixed-vocabulary `error.status`
        (e.g. `RESOURCE_EXHAUSTED`), when the failing exception carries one --
        it distinguishes a quota error from an ordinary rate limit under the
        same HTTP status code. `kind` is the customer-facing classification
        the same signals resolve to (see `model_errors.classify_model_error`).

        The stored message is redacted and truncated: it is persisted on the
        run summary and read back by the tenant, so a credential echoed in a
        provider error must not survive the trip.
        """
        if id(exc) in self._recorded_ids:
            return
        self._recorded_ids.add(id(exc))
        key = ModelErrorKey(
            model_type=model_type,
            status=str(status),
            error_status=error_status,
            kind=kind,
            endpoint_url=_endpoint_url.get(),
        )
        self._counts[key] = self._counts.get(key, 0) + 1
        self._last_errors[key] = redact(f"{type(exc).__name__}: {exc}")[
            :_MAX_LAST_ERROR_CHARS
        ]

    def counts(self) -> dict[ModelErrorKey, int]:
        return dict(self._counts)

    def last_errors(self) -> dict[ModelErrorKey, str]:
        """The most recent message per key, for the run summary's breakdown."""
        return dict(self._last_errors)

    def dominant_kind(self) -> ErrorKind | None:
        """The kind that best explains why this run's calls failed.

        A run whose every prompt errored surfaces as a bare
        `EvaluationRunError` carrying only a sentence, so without this the
        CHOO-2833 case ("agent configured wrong, run stuck at 25%") would
        report `unknown` -- the one outcome the taxonomy exists to avoid.

        `internal_error` is excluded while anything else is present: a handful
        of judge hiccups alongside hundreds of 401s from the agent must not
        bury the 401s. Ties break on the first kind inserted. Under asyncio
        that is usually the earliest failure, but concurrent calls make the
        order only roughly chronological, so treat it as deterministic rather
        than as a claim about which failure came first.
        """
        totals: dict[ErrorKind, int] = {}
        for key, count in self._counts.items():
            totals[key.kind] = totals.get(key.kind, 0) + count
        if not totals:
            return None
        ranked = {
            k: n for k, n in totals.items() if k != ErrorKind.INTERNAL_ERROR
        } or totals
        return max(ranked, key=lambda k: ranked[k])


_error_recorder: ContextVar[ModelErrorAccumulator | None] = ContextVar(
    "_error_recorder", default=None
)


def recorded_error_kinds() -> set[ErrorKind]:
    """Kinds recorded so far in the enclosing `track_model_errors()` scope.

    Lets a running evaluation see *why* its calls are failing while it is still
    deciding whether to keep going, which the per-child status alone cannot
    say. Empty outside such a scope (the CLI never opens one), so a caller gets
    "nothing known" rather than an error.
    """
    accumulator = _error_recorder.get()
    if accumulator is None:
        return set()
    return {key.kind for key in accumulator.counts()}


def record_model_error(
    exc: Exception,
    model_type: str,
    status: object = None,
    error_status: str | None = None,
) -> None:
    """Record a failure that didn't go through `Model.generate()` onto the
    enclosing `track_model_errors()` accumulator, if one is open.

    `Model.generate()` records every call it makes itself, but a detector can
    raise its own exception after its judge's `generate()` call already
    returned successfully (e.g. an unparseable verdict) -- that failure has
    no other way to reach the run's `error_kind`. No-op outside a
    `track_model_errors()` scope.
    """
    accumulator = _error_recorder.get()
    if accumulator is not None:
        accumulator.record_once(
            exc, model_type, status, error_status, classify_model_error(exc)
        )


@contextmanager
def track_model_errors() -> Iterator[ModelErrorAccumulator]:
    """Scope in which every `Model.generate()` failure is counted by
    `(model_type, status)`. See `track_token_usage()`; this is the same
    pattern applied to failures instead of token usage."""
    accumulator = ModelErrorAccumulator()
    token = _error_recorder.set(accumulator)
    try:
        yield accumulator
    finally:
        _error_recorder.reset(token)


class ModelResponse:
    message: Message | None
    status: ResponseStatus

    def __init__(
        self,
        message: Message | None,
        status: ResponseStatus = ResponseStatus.OK,
    ):
        self.message = message
        self.status = status
        self._status_logged = False


def _message_text(msg: Message) -> str:
    parts = [p.text for p in msg.content.parts if p.text]
    return " ".join(parts) if parts else ""


def _prompt_length(contents: ModelContent) -> int:
    if isinstance(contents, str):
        return len(contents)
    if isinstance(contents, Message):
        return len(_message_text(contents))
    return sum(len(_message_text(m)) for m in contents) if contents else 0


class Model(ABC):
    async def generate(
        self,
        contents: ModelContent,
        *,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response from the model.

        Args:
            contents: The input to the model. Can be:
                - A string (converted to a user message)
                - A single Message
                - A list of Messages
            output_schema: Optional Pydantic model describing the JSON shape
                the response must take. When set, the adapter constrains the
                output using the provider's native structured-output mechanism.
                Models under test that cannot enforce a schema ignore it.
        """
        if isinstance(contents, str):
            messages = [
                Message(
                    content=Content.text(Role.USER, contents),
                ),
            ]
        elif isinstance(contents, Message):
            messages = [contents]
        else:
            messages = contents

        prompt_len = _prompt_length(contents)
        model_name = type(self).__name__

        try:
            response = await self._generate(
                messages, output_schema=output_schema, **kwargs
            )
        except Exception as e:
            # Stamp before classifying: this is the innermost layer and the
            # only point still inside the generator's scope, and the stamp is
            # what carries that fact to the terminal classification later.
            if in_generator_scope():
                mark_generator_failure(e)
            kind = classify_model_error(e)
            # Only internal_error is ours to fix; every other kind is the
            # tenant's agent/endpoint and must not page us as a platform error.
            if not getattr(e, _LOGGED_ATTR, False):
                setattr(e, _LOGGED_ATTR, True)
                logger.log(
                    log_level(kind),
                    "%s: prompt=%d chars (%s: %s)",
                    model_name,
                    prompt_len,
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
            error_accumulator = _error_recorder.get()
            if error_accumulator is not None:
                if is_host_not_found(e):
                    status = HOST_NOT_FOUND
                elif isinstance(e, SsrfBlockedError):
                    status = BLOCKED_STATUS
                else:
                    status = status_of(e)
                # Both fields come from the two readers `classify_model_error`
                # consults, never a bare `getattr`: a key derived here its own
                # way disagrees with the kind stored beside it, and the
                # consumers sub-split on these values —
                # `eval_cli._check_model_credentials` reads "404" as a wrong
                # endpoint and a statusless failure as a transient network one
                # — so a dropped field does not merely lose detail, it moves a
                # run into the retry band. A statusless exception needs no
                # fallback: `status_of` returns None and `record_once`
                # stringifies it to the "None" `_NETWORK_FAILURE_STATUSES`
                # matches on.
                error_accumulator.record_once(
                    e,
                    model_name,
                    status,
                    error_status_of(e),
                    kind,
                )
            raise

        response_len = len(_message_text(response.message)) if response.message else 0
        if response.status != ResponseStatus.OK:
            if response._status_logged:
                return response
            response._status_logged = True
            logger.warning(
                "%s: prompt=%d chars, status=%s",
                model_name,
                prompt_len,
                response.status.value,
            )
        else:
            logger.debug(
                "%s: prompt=%d chars, response=%d chars",
                model_name,
                prompt_len,
                response_len,
            )

        return response

    @abstractmethod
    async def _generate(
        self,
        messages: list[Message],
        *,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        pass


class EndpointScopedModel(Model):
    """Names the endpoint on failures without throttling, for SDK types that
    rate limit themselves (Ollama, OpenAI-compatible)."""

    def __init__(self, model: Model, endpoint_url: str):
        self._model = model
        self._endpoint_url = endpoint_url

    async def _generate(
        self,
        messages: list[Message],
        *,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        with endpoint_scope(self._endpoint_url):
            return await self._model.generate(
                messages, output_schema=output_schema, **kwargs
            )


# -- Helper functions -----------------------------------------------


def flatten_messages(messages: list[Message]) -> str:
    if len(messages) == 1:
        parts = [p.text for p in messages[0].content.parts if p.text]
        return " ".join(parts)

    lines: list[str] = []
    for msg in messages:
        role = msg.content.role.value.upper()
        text = " ".join(p.text for p in msg.content.parts if p.text)
        lines.append(f"{role}: {text}")
    return "\n\n".join(lines)


def extract_text_from_message(message: Message) -> str:
    return "".join(part.text for part in message.content.parts if part.text is not None)


def extract_final_text(message: Message) -> str:
    """Extract only the final text output, excluding
    thinking, tool calls, and tool results."""
    text_parts = [
        part.text
        for part in message.content.parts
        if part.text is not None and part.part_type == PartType.TEXT
    ]
    if text_parts:
        return "".join(text_parts)
    return extract_text_from_message(message)


def extract_text_from_conversation(
    messages: list[Message],
) -> str:
    lines: list[str] = []
    for msg in messages:
        role = msg.content.role.value.upper()
        text = extract_final_text(msg)
        lines.append(f"{role}: {text}")
    return "\n\n".join(lines)
