import asyncio
import logging
import random
from typing import Any

from pydantic import BaseModel

from switch_trust.eval.common.schema import Message
from switch_trust.eval.core.models.model import Model, ModelResponse

logger = logging.getLogger(__name__)

_TRANSIENT_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
# Terminal statuses stop retries immediately. 401/403 mean bad credentials
# and 404 means the endpoint is wrong - none recover by waiting.
_TERMINAL_STATUS_CODES = {401, 403, 404}
# Upper bound on the delay we accept from a tenant-controlled Retry-After
# header. Anything larger is clamped so a misconfigured or hostile endpoint
# can't park a worker coroutine indefinitely (defeats the pacing goal).
# 60s covers typical 429 recovery windows for the providers we've seen.
_MAX_RETRY_AFTER_SECONDS = 60.0


def _status_of(exc: Exception) -> int | None:
    for attr in ("status_code", "status", "code"):
        status = getattr(exc, attr, None)
        if isinstance(status, int):
            return status
    return None


def _is_terminal(exc: Exception) -> bool:
    status = _status_of(exc)
    return status is not None and status in _TERMINAL_STATUS_CODES


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, _TRANSIENT_EXCEPTIONS):
        return True
    status = _status_of(exc)
    return status is not None and status in _TRANSIENT_STATUS_CODES


def _extract_retry_after(exc: Exception) -> float | None:
    """Parse a Retry-After (integer seconds) header off an exception.

    Duck-types across ``aiohttp.ClientResponseError`` (``exc.headers``) and
    SDK errors like ``openai.APIStatusError`` (``exc.response.headers``).
    HTTP-date form is not supported. The parsed value is clamped to
    ``[0, _MAX_RETRY_AFTER_SECONDS]`` because the endpoint is
    tenant-controlled and cannot be trusted to name a sane delay.
    """
    for source in (exc, getattr(exc, "response", None)):
        headers = getattr(source, "headers", None) if source is not None else None
        if headers is None:
            continue
        try:
            raw = headers.get("Retry-After") or headers.get("retry-after")
        except AttributeError:
            continue
        if raw is None:
            continue
        try:
            delay = float(int(str(raw).strip()))
        except (ValueError, TypeError):
            return None
        return max(0.0, min(delay, _MAX_RETRY_AFTER_SECONDS))
    return None


class ExponentialRetryModel(Model):
    """Wraps a Model and retries on transient errors with
    exponential backoff."""

    def __init__(
        self,
        model: Model,
        max_retries: int = 5,
        base_delay: float = 2.0,
    ):
        self._model = model
        self._max_retries = max_retries
        self._base_delay = base_delay

    async def _generate(
        self,
        messages: list[Message],
        *,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return await self._model.generate(
                    messages, output_schema=output_schema, **kwargs
                )
            except Exception as exc:
                if _is_terminal(exc):
                    raise
                if not _is_transient(exc) or attempt == self._max_retries:
                    raise
                last_exc = exc
                retry_after = _extract_retry_after(exc)
                if retry_after is not None:
                    # Add jitter on top of the server-suggested delay: 100
                    # concurrent 429s all carry the same Retry-After, so
                    # sleeping exactly that value has every task fire again
                    # at the same instant and hit the same limit. Same
                    # shape as the exponential-backoff branch below.
                    delay = retry_after + random.uniform(0, retry_after)
                    delay_source = "retry-after"
                else:
                    base = self._base_delay * (2**attempt)
                    delay = base + random.uniform(0, base)
                    delay_source = "backoff"
                logger.warning(
                    "Transient error (attempt %d/%d), retrying in %.1fs [%s] (%s: %s)",
                    attempt + 1,
                    self._max_retries,
                    delay,
                    delay_source,
                    type(exc).__name__,
                    exc,
                )
                await asyncio.sleep(delay)
        raise last_exc  # unreachable, but keeps type checker happy


def _fibonacci_delays(max_value: float):
    """Yield fibonacci-sequence delays capped at max_value."""
    a, b = 1.0, 1.0
    while True:
        yield min(a, max_value)
        a, b = b, a + b


class FibonacciRetryModel(Model):
    """Wraps a Model and retries on transient errors with
    fibonacci backoff and jitter, capped at a maximum delay."""

    def __init__(
        self,
        model: Model,
        max_retries: int = 5,
        max_delay: float = 70.0,
    ):
        self._model = model
        self._max_retries = max_retries
        self._max_delay = max_delay

    async def _generate(
        self,
        messages: list[Message],
        *,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        last_exc: Exception | None = None
        delays = _fibonacci_delays(self._max_delay)
        for attempt in range(self._max_retries + 1):
            try:
                return await self._model.generate(
                    messages, output_schema=output_schema, **kwargs
                )
            except Exception as exc:
                if _is_terminal(exc):
                    raise
                if not _is_transient(exc) or attempt == self._max_retries:
                    raise
                last_exc = exc
                base = next(delays)
                retry_after = _extract_retry_after(exc)
                if retry_after is not None:
                    # Add jitter on top of the server-suggested delay: 100
                    # concurrent 429s all carry the same Retry-After, so
                    # sleeping exactly that value has every task fire again
                    # at the same instant and hit the same limit. Same
                    # shape as the exponential-backoff branch below.
                    delay = retry_after + random.uniform(0, retry_after)
                    delay_source = "retry-after"
                else:
                    delay = random.uniform(0, base)
                    delay_source = "backoff"
                logger.warning(
                    "Transient error (attempt %d/%d), retrying in %.1fs [%s] (%s: %s)",
                    attempt + 1,
                    self._max_retries,
                    delay,
                    delay_source,
                    type(exc).__name__,
                    exc,
                )
                await asyncio.sleep(delay)
        raise last_exc  # unreachable, but keeps type checker happy


RetryModel = ExponentialRetryModel
