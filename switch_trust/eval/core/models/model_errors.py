"""Customer-facing taxonomy for a failed model/agent call (CHOO-2936).

`error_message` on a run is `str(exc)` — fine for a human reading a log, useless
for a UI that wants to say "check the credential" rather than "something went
wrong". This module maps an exception to a small fixed vocabulary the UI can
render an actionable sentence from.

The vocabulary is the one Test Connection already ships (`ok`, `timeout`,
`blocked`, `invalid_config`, `unreachable`, `internal_error` — see
`platform/test_connection_server.py` and the UI's `getTestConnectionReasonKey`),
extended with the two cases a real run hits that a connectivity check does not:
a provider quota and a rejected credential. Values stay lowercase snake_case to
match `admission.Reason` and the entitlement wire codes.

Classification is deliberately content-free: it reads an HTTP status and a
fixed google.rpc vocabulary (`ModelHttpError.error_status`, or Gemini's
`APIError.status`), never response-body text. See `model_http.py` for why
that body is untrusted. The one case with no status to read is a transport
failure the OpenAI/Anthropic SDKs wrapped in their own exception tree, keyed
by class name in `_SDK_TRANSPORT_KINDS`.
"""

from __future__ import annotations

import logging
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import NamedTuple

from switch_trust.cli.http_status_kinds import (
    StatusKind,
    require_total,
    status_kind,
)

# The judge/generator runs on *our* provider key, so its failures are ours, not
# the tenant's — telling a customer their agent is rate-limited when our Gemini
# judge ran out of quota is worse than saying nothing.
#
# Two halves, because the two consumers read at different times.
# `generator_scope()` is what `GeneratorModel` enters, and it is live while the
# innermost `Model.generate()` records the failure. The stamp that same choke
# point then puts on the exception is what carries the fact out, for the
# terminal classification in `platform/errors.py` that runs long after the
# scope has unwound.
_in_generator: ContextVar[bool] = ContextVar("_in_generator", default=False)
_GENERATOR_ATTR = "_flintai_generator_failure"


class ErrorKind(str, Enum):
    """Why a model/agent call or an evaluation run failed."""

    OK = "ok"
    QUOTA_EXCEEDED = "quota_exceeded"
    AUTH_FAILED = "auth_failed"
    UNREACHABLE = "unreachable"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    INVALID_CONFIG = "invalid_config"
    PLAN_LIMIT_EXCEEDED = "plan_limit_exceeded"
    # Ours to fix: a parse failure, a judge/generator outage, an unexpected bug.
    INTERNAL_ERROR = "internal_error"
    # A target-model failure we could not place. Distinct from internal_error so
    # a growing `unknown` rate reads as "the taxonomy has a gap", not "we broke".
    UNKNOWN = "unknown"


class ModelErrorKey(NamedTuple):
    """What distinguishes one bucket of failures from another.

    A NamedTuple rather than a bare tuple so the fields are read by name at the
    several call sites that unpack it, and so adding a dimension doesn't mean
    re-counting positions in each of them.

    Lives here, beside ErrorKind, rather than in model.py: platform/common's
    ingestion builds these and the scheduler's target excludes the eval engine,
    so defining it next to the heavy Model ABC put torch and garak on the
    scheduler's import path. Nothing here needs them. CHOO-3546.
    """

    model_type: str
    status: str
    error_status: str | None
    kind: ErrorKind
    # The tenant endpoint the failing call went to, for any model built with a
    # host. None for hostless SDK providers like our judge, which is what keeps
    # a judge failure separable from the agent's own.
    endpoint_url: str | None


# google.rpc.Code -> kind, for the statuses that say more than the HTTP code
# does. RESOURCE_EXHAUSTED is the reason this lookup runs first: it separates a
# real quota error from an ordinary rate limit, both of which arrive as a 429.
# A code absent here (INTERNAL, DATA_LOSS, ...) falls through to the status.
_ERROR_STATUS_KINDS = {
    "RESOURCE_EXHAUSTED": ErrorKind.QUOTA_EXCEEDED,
    "UNAUTHENTICATED": ErrorKind.AUTH_FAILED,
    "PERMISSION_DENIED": ErrorKind.AUTH_FAILED,
    "DEADLINE_EXCEEDED": ErrorKind.TIMEOUT,
    "UNAVAILABLE": ErrorKind.UNREACHABLE,
    "INVALID_ARGUMENT": ErrorKind.INVALID_CONFIG,
    "FAILED_PRECONDITION": ErrorKind.INVALID_CONFIG,
}

# What each shared status meaning becomes in *this* vocabulary. `ErrorKind` is
# the coarser, wire-visible one, so two pairs collapse here and that is
# deliberate rather than an oversight: a 401 and a 403 are both "the
# credential did not work" to a UI that offers one remediation for them, and
# a 404 and a 5xx are both `UNREACHABLE`. The CLI re-splits exactly those
# pairs on the recorded status (`eval_cli._check_model_credentials`) because
# an exit code can afford the distinction a UI string table cannot.
_ERROR_KIND_FOR_STATUS_KIND = {
    StatusKind.AUTH: ErrorKind.AUTH_FAILED,
    StatusKind.NOT_ENTITLED: ErrorKind.AUTH_FAILED,
    StatusKind.NOT_FOUND: ErrorKind.UNREACHABLE,
    StatusKind.TIMEOUT: ErrorKind.TIMEOUT,
    StatusKind.RATE_CAP: ErrorKind.QUOTA_EXCEEDED,
    StatusKind.UPSTREAM: ErrorKind.UNREACHABLE,
}
require_total(_ERROR_KIND_FOR_STATUS_KIND, "model_errors._kind_for_status")

# Transport failures the OpenAI and Anthropic SDKs raise, keyed by class name.
#
# Both SDKs are generated from the same template and wrap a transport failure
# in their own tree — `APITimeoutError` -> `APIConnectionError` -> `APIError`
# -> `Exception` — which no other branch of `classify_model_error` can see:
# it is not a `TimeoutError`, not an `OSError`, and `APIConnectionError`
# passes `body=None` so `APIError` leaves `code` unset and `status_of` finds
# nothing. Without this a provider timeout classified as `unknown`, which the
# CLI's exit table has no entry for — a run that timed out exited
# `EVAL_INCOMPLETE` (6, "a human has to look at it") instead of `TIMEOUT`
# (62, "retry the identical invocation"), so every wrapper branching on the
# band got the opposite answer to the right one.
#
# Keyed on the name rather than the type because `core/` must not import a
# provider SDK to classify an error (`model_errors.py` is imported by the
# platform's `lib_platform_common`, which deliberately carries neither).
_SDK_TRANSPORT_KINDS = {
    "APITimeoutError": ErrorKind.TIMEOUT,
    "APIConnectionError": ErrorKind.UNREACHABLE,
}


def _sdk_transport_kind(exc: Exception) -> ErrorKind | None:
    """The kind an OpenAI/Anthropic-style transport exception stands for.

    Walks the MRO so a subclass is classified by its nearest known ancestor,
    and so `APITimeoutError` wins over the `APIConnectionError` it derives
    from. Returns None for anything unrecognised, leaving the caller's
    remaining branches to decide.
    """
    for klass in type(exc).__mro__:
        kind = _SDK_TRANSPORT_KINDS.get(klass.__name__)
        if kind is not None:
            return kind
    return None


# Recorded as the status of a call whose host does not exist. There is no HTTP
# status to record, and unlike a transient lookup failure no retry fixes it.
HOST_NOT_FOUND = "host_not_found"


# Recorded as the status of a call the SSRF guard refused. No retry fixes it.
BLOCKED_STATUS = "blocked"


class SsrfBlockedError(Exception):
    """Raised when a request target resolves to a disallowed address.

    Deliberately not an ``OSError``/``ConnectionError`` subclass: those are
    treated as transient by ``RetryModel`` and the worker's ``_is_retriable``,
    which would retry a blocked request instead of failing it once. As a plain
    ``Exception`` it is non-transient (no retries) and terminal at the worker.
    Lives in ``core`` so ``classify_model_error`` can see it; the guard itself is
    ``platform/ssrf.py``.
    """


def is_host_not_found(exc: Exception) -> bool:
    """Whether DNS answered that the host does not exist (NXDOMAIN).

    aiohttp keeps the resolver's `socket.gaierror` as `os_error`. Only
    EAI_NONAME counts: EAI_AGAIN is a resolver hiccup that can recover.
    """
    os_error = getattr(exc, "os_error", None)
    return isinstance(os_error, socket.gaierror) and os_error.errno == socket.EAI_NONAME


def status_of(exc: Exception) -> int | None:
    """The HTTP status an exception carries, across the shapes we see.

    Duck-typed rather than keyed on a type: `aiohttp.ClientResponseError` and
    `ModelHttpError` expose `status`, the OpenAI/Anthropic SDKs
    `status_code`, and some wrappers only `code`.

    `code` is the loose one: an exception carrying an unrelated small int
    there, a gRPC code say, is read as an HTTP status and falls through
    `_kind_for_status` to `unknown`. Harmless, and the same net `model_retry`
    has always used, but worth knowing now that two callers share it.
    """
    for attr in ("status_code", "status", "code"):
        status = getattr(exc, attr, None)
        if isinstance(status, int):
            return status
    return None


def error_status_of(exc: Exception) -> str | None:
    """The google.rpc-style status string an exception carries, if any.

    `ModelHttpError` exposes it as `.error_status`. `google.genai.errors.APIError`
    (Gemini) carries the same kind of value but under `.status` instead — and
    unlike the HTTP-status shapes `status_of` reads, it never puts a numeric
    code there (that's `.code`), so reading `.status` as a string here is
    unambiguous and doesn't collide with `status_of`'s own use of the name.

    Public, and paired with `status_of`: `classify_model_error` reads both to
    decide the kind, and `Model.generate()`'s choke point records both onto
    `ModelErrorKey`. A record site that derives either field its own way puts
    a key in the accumulator that disagrees with the kind stored beside it —
    which is how Gemini's `RESOURCE_EXHAUSTED` came to be classified from a
    string that was then dropped from the record.
    """
    for attr in ("error_status", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, str):
            return value
    return None


@contextmanager
def generator_scope() -> Iterator[None]:
    """Scope in which a model failure is attributed to us, not the tenant."""
    token = _in_generator.set(True)
    try:
        yield
    finally:
        _in_generator.reset(token)


def in_generator_scope() -> bool:
    return _in_generator.get()


def mark_generator_failure(exc: Exception) -> None:
    """Tag an exception as having come from the judge/generator model."""
    setattr(exc, _GENERATOR_ATTR, True)


def is_generator_failure(exc: Exception) -> bool:
    return getattr(exc, _GENERATOR_ATTR, False) is True


def _kind_for_status(status: int) -> ErrorKind:
    kind = status_kind(status)
    if kind is not None:
        return _ERROR_KIND_FOR_STATUS_KIND[kind]
    if status >= 400:
        # The endpoint answered and rejected the request shape — a wrong
        # input_path/output_path or a payload it didn't accept.
        return ErrorKind.INVALID_CONFIG
    return ErrorKind.UNKNOWN


def log_level(kind: ErrorKind) -> int:
    """Log severity for a classified model/agent failure.

    Only `internal_error` is ours to fix and should alert; every other kind
    reflects a problem with the tenant's agent/endpoint and must not look
    like a platform failure in Datadog. Scoped to the narrow `Model.generate()`
    choke point, where an unclassifiable failure is realistically the tenant's
    endpoint doing something odd, not our code -- see `log_level_for_run_failure`
    for the wider scope.
    """
    return logging.ERROR if kind == ErrorKind.INTERNAL_ERROR else logging.WARNING


# Kinds confidently caused by the tenant's agent/endpoint rather than by our
# own code.
_CONFIDENTLY_TENANT_CAUSED = frozenset(
    {
        ErrorKind.TIMEOUT,
        ErrorKind.BLOCKED,
        ErrorKind.AUTH_FAILED,
        ErrorKind.QUOTA_EXCEEDED,
        ErrorKind.UNREACHABLE,
        ErrorKind.INVALID_CONFIG,
        ErrorKind.PLAN_LIMIT_EXCEEDED,
    }
)


def log_level_for_run_failure(kind: ErrorKind) -> int:
    """Log severity for a failure caught around a whole evaluation's run.

    Unlike `log_level()`, defaults to ERROR rather than WARNING: this scope
    (`SingleEvaluation.execute`) also wraps our own detector and scoring code,
    so an unclassifiable failure here is as likely to be our bug as the
    tenant's.
    """
    return logging.WARNING if kind in _CONFIDENTLY_TENANT_CAUSED else logging.ERROR


def classify_model_error(exc: Exception) -> ErrorKind:
    """Map a model/agent call failure to its customer-facing kind.

    Covers HTTP and transport failures, plus the SSRF guard's refusal. Other
    platform-level causes (an invalid eval config) are layered on by
    `platform/errors.py::classify_run_error`, which `core` cannot import.
    """
    if is_generator_failure(exc):
        return ErrorKind.INTERNAL_ERROR
    if isinstance(exc, SsrfBlockedError):
        return ErrorKind.BLOCKED
    # Before the OSError branch below: TimeoutError subclasses OSError, and
    # asyncio.TimeoutError has been an alias for it since 3.11.
    if isinstance(exc, TimeoutError):
        return ErrorKind.TIMEOUT
    error_status = error_status_of(exc)
    if error_status is not None:
        kind = _ERROR_STATUS_KINDS.get(error_status)
        if kind is not None:
            return kind
    status = status_of(exc)
    if status is not None:
        return _kind_for_status(status)
    # After the status checks, so a provider that did answer is classified by
    # what it answered; before the `OSError` branch, which these never reach.
    sdk_kind = _sdk_transport_kind(exc)
    if sdk_kind is not None:
        return sdk_kind
    if isinstance(exc, OSError):
        return ErrorKind.UNREACHABLE
    return ErrorKind.UNKNOWN
