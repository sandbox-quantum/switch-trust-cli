"""Maps exceptions to the party responsible for them (``FailureDomain``).

Lives in ``common/`` rather than beside the OTEL/httpx-heavy metric and span
instrumentation so it can be reused without pulling that machinery in: the OSS
CLI export (``cli_exporter/export.py``) copies this module alongside the agent
scanner, and ``agent_scanner/reasoner.py`` needs `classify_exception` at the
point an LLM call fails but has no use for metrics or spans. The metrics
module imports from here, never the reverse — same direction as
``llm_usage.py``.
"""

from enum import StrEnum

import httpx

from switch_trust.cli.http_status_kinds import (
    StatusKind,
    require_total,
    status_kind,
)


class FailureDomain(StrEnum):
    """Who is at fault. Only `internal` and `provider` should ever page.

    `provider` is a third party across a network boundary — the LLM API, or the
    flint backend we upload to. `user` is the customer's repo or workflow
    config. `internal` is our own code, including budgets we chose ourselves.
    """

    NONE = "none"
    INTERNAL = "internal"
    PROVIDER = "provider"
    USER = "user"
    UNKNOWN = "unknown"


class ScannerTimeout(Exception):
    """A stage stopped because it hit one of *our* budgets.

    Distinct from a provider-side timeout: exceeding a wall-clock, iteration or
    token budget we chose ourselves is an `internal` fault, not a `provider`
    one, even though both surface as `outcome=timeout`.
    """


class PreconditionNotMet(Exception):
    """The run cannot start (no token, no workspace, invalid instance)."""


# Provider SDKs are matched by module prefix rather than imported: litellm is
# expensive to import and the google-genai error module is not a declared dep
# of every target that needs to classify an exception.
_PROVIDER_MODULE_PREFIXES = (
    "google.api_core",
    "google.auth",
    "google.genai",
    "litellm",
    "openai",
    "anthropic",
)

_USER_EXCEPTIONS = (
    FileNotFoundError,
    NotADirectoryError,
    PermissionError,
    PreconditionNotMet,
)

_PROVIDER_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    # httpx.TransportError covers connect/read/write/pool failures and does NOT
    # derive from the builtin ConnectionError, so it needs listing explicitly —
    # otherwise every unreachable-backend run is misattributed to `internal`.
    httpx.TransportError,
)

# A missing or rejected LLM credential surfaces in many shapes: google-genai
# raises a bare `builtins.ValueError` with a message naming the missing
# credential, openai a keyless `OpenAIError`, others a 401. The 401 variant is
# already `user` (via `http_status_of`), but the keyless ones carry no status
# and no provider module we match, so they fell through to `internal` — which
# pages. A missing/rejected credential is the customer's CI config, so it is
# `user`. Matched on the message (never exported — same as ScanSummary's
# timeout sniff), lower-cased.
_AUTH_ERROR_MARKERS = (
    "api key",
    "api_key",
    "apikey",
    "credential",
    "authentication",
    "unauthorized",
)


def _looks_like_auth_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _AUTH_ERROR_MARKERS)


def is_rejected_credential(exc: BaseException) -> bool:
    """Whether a credential was supplied and the provider refused it.

    A rejection always carries a 401/403: the request reached the provider for
    it to be refused. That status is the discriminator, not the wording — read
    through ``status_kind`` rather than as literals, so the shared table keeps
    owning what a status means.
    """
    return status_kind(http_status_of(exc)) in (
        StatusKind.AUTH,
        StatusKind.NOT_ENTITLED,
    )


def is_missing_credential(exc: BaseException) -> bool:
    """Whether no credential was supplied at all.

    The mirror of :func:`is_rejected_credential`: a keyless SDK never leaves the
    process, so its error carries no status and only its message identifies it.
    Telling the two apart matters because they ask the customer for different
    things — set a key, versus fix the key you have — and the sensor logs them
    at different levels for exactly that reason.
    """
    return http_status_of(exc) is None and _looks_like_auth_error(exc)


def is_auth_error(exc: BaseException) -> bool:
    """Whether ``exc`` is a rejected *or* missing LLM credential.

    Narrower than ``classify_exception(...) == USER``: a 400/404 (bad request,
    unknown model) is also `user` but is *not* an auth failure. Callers that
    must distinguish the two cases want the pair above instead.
    """
    return is_rejected_credential(exc) or is_missing_credential(exc)


def is_not_entitled(exc: BaseException) -> bool:
    """Whether ``exc`` is a *valid* credential the provider refused to honour.

    The narrow half of ``is_auth_error``, which deliberately claims this case
    too for the console's coarser "your credential is wrong" line. Callers that
    must tell the customer *which* fix applies — the plan/quota rather than the
    token — check this first and fall back to ``is_auth_error``; a status is
    required, since no message shape distinguishes the two reliably.

    It exists as a predicate rather than an inline ``== 403`` at each of those
    call sites so the status stays owned by the shared table: a second status
    joining `NOT_ENTITLED` reaches every caller at once.
    """
    return status_kind(http_status_of(exc)) is StatusKind.NOT_ENTITLED


# An unknown/unsupported model surfaces as a 404 or one of these message shapes.
# Matched lower-cased, like the auth markers, and never exported.
_MODEL_ERROR_MARKERS = (
    "model not found",
    "model_not_found",
    "does not exist",
    "no such model",
    "unknown model",
    "invalid model",
    "unsupported model",
)


def _looks_like_model_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _MODEL_ERROR_MARKERS)


def is_model_error(exc: BaseException) -> bool:
    """Whether ``exc`` is an unknown/unsupported *model*, not a bad credential.

    Also `user`, and also narrower than the domain: it lets the customer be told
    "the configured model is invalid" specifically. Auth wins — a 401 is a key
    problem, not a model one — so this returns False for anything ``is_auth_error``
    already claims. A 404 (the provider rejecting the model id) or a
    model-not-found message counts.
    """
    if is_auth_error(exc):
        return False
    is_not_found = status_kind(http_status_of(exc)) is StatusKind.NOT_FOUND
    return is_not_found or _looks_like_model_error(exc)


# A provider-declared rate limit surfaces as a bare `RateLimitError` (litellm
# mirrors the name across every backend it wraps) or a message naming the
# limit, for the one case that carries no HTTP status to key off at all: a
# gRPC `RESOURCE_EXHAUSTED` from a provider SDK that never touches HTTP (e.g.
# Gemini), which litellm re-raises with no `.response` attached. Matched lower-
# cased, like the auth/model markers, and never exported.
_RATE_LIMIT_ERROR_MARKERS = (
    "rate limit",
    "rate_limit",
    "resource_exhausted",
    "too many requests",
)


def _looks_like_rate_limit_error(exc: BaseException) -> bool:
    return type(exc).__name__ == "RateLimitError" or any(
        marker in str(exc).lower() for marker in _RATE_LIMIT_ERROR_MARKERS
    )


def is_rate_limited(exc: BaseException) -> bool:
    """Whether ``exc`` is a provider-declared rate limit that arrived with no
    HTTP 429 to key off.

    Narrower than ``classify_exception(...) == PROVIDER``: a real 429 already
    carries a status and is ``is_resource_exhausted`` instead. Checked only
    when no HTTP status exists, the same guard ``is_network_error`` uses, so a
    429 can never be double-counted as this.
    """
    if http_status_of(exc) is not None:
        return False
    return _looks_like_rate_limit_error(exc)


def is_resource_exhausted(exc: BaseException) -> bool:
    """Whether ``exc`` is a literal HTTP 429 the provider answered with.

    The mirror image of ``is_rate_limited``: that one fires when no status
    exists at all, this one requires the provider to have actually answered
    with 429. Kept distinct because a real 429 is a request-rate cap the
    provider counted and reported, worth distinguishing from a statusless
    provider-declared limit (e.g. Gemini's gRPC ``RESOURCE_EXHAUSTED``) even
    though both are equally retryable.
    """
    return status_kind(http_status_of(exc)) is StatusKind.RATE_CAP


# litellm and openai re-wrap a provider-side timeout into their own type
# (`litellm.exceptions.Timeout`, `openai.APITimeoutError`) rather than raising
# the builtin `TimeoutError`/`httpx.TimeoutException` — neither derives from
# either, so they are matched by message instead, the same way
# `is_rate_limited` matches a statusless `RESOURCE_EXHAUSTED`. Matched
# lower-cased, like the other markers, and never exported.
_TIMEOUT_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "deadline exceeded",
)


def _looks_like_timeout_error(exc: BaseException) -> bool:
    return any(marker in str(exc).lower() for marker in _TIMEOUT_ERROR_MARKERS)


def is_timeout_error(exc: BaseException) -> bool:
    """Whether ``exc`` is a provider call timing out, not one of our own budgets.

    A `408`/`504` — the provider answered that *it* gave up — counts, via the
    shared `StatusKind` table rather than a local list, so the eval engine
    cannot disagree about which statuses those are. So does a bare
    `TimeoutError`/`httpx.TimeoutException`, and so does a timeout-shaped
    message for the re-wrapped SDK types neither of those catches. Never
    `ScannerTimeout`: that is our own wall-clock/iteration/token budget, an
    `internal` fault (see its docstring), not a `provider` one, even though
    both can surface as `exit_reason="timeout"` upstream.
    """
    if isinstance(exc, ScannerTimeout):
        return False
    if status_kind(http_status_of(exc)) is StatusKind.TIMEOUT:
        return True
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return True
    return _looks_like_timeout_error(exc)


def is_upstream_error(exc: BaseException) -> bool:
    """Whether ``exc`` is the provider's or platform's own backend failing (5xx).

    A 504 is a timeout, not a generic backend fault, and that exclusion is
    now structural: the shared table maps it to `StatusKind.TIMEOUT`, so the
    two kinds are mutually exclusive by construction rather than by a
    hand-written `and status != 504` that one of the four copies of this
    knowledge had already lost. Unlike `is_rate_limited`/`is_network_error`, a
    status is required — a statusless 5xx-shaped message is not sniffed for,
    since nothing short of the provider actually answering with one says the
    backend is the problem rather than, say, a dropped connection.
    """
    return status_kind(http_status_of(exc)) is StatusKind.UPSTREAM


def is_network_error(exc: BaseException) -> bool:
    """Whether ``exc`` is a connection failure — refused, reset, or an
    unresolvable host — rather than a timeout or a status the provider
    actually answered with.

    Narrower than ``classify_exception(...) == PROVIDER``: a 5xx, a 429, or a
    generic provider-SDK exception with no status all classify as `provider`
    too, but none of those is "the network dropped" specifically. Mirrors
    ``standalone_inventory_action._upload_exit_code``'s ordering — a status
    always wins, and ``httpx.TimeoutException`` is excluded before the
    ``httpx.TransportError`` check since it is a subclass of it, so a timeout
    is never misreported as a network failure.
    """
    if http_status_of(exc) is not None:
        return False
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return False
    return isinstance(exc, (ConnectionError, httpx.TransportError))


# A provider that answered with no HTTP status can still name the condition in
# its message. `is_rate_limited` already covers the rate-limit wording; these
# are the transient-backend equivalents, which no predicate above sniffs for
# because a statusless 5xx says nothing on its own about whose fault it is.
# Here the question is only whether another attempt is worth making, which is
# a lower bar. Matched lower-cased, like every other marker tuple.
_RETRYABLE_MESSAGE_MARKERS = ("429", "503", "unavailable")


def is_retryable(exc: BaseException) -> bool:
    """Whether retrying ``exc`` could plausibly succeed.

    Composed from the status predicates rather than restating their numbers,
    so a status joining `RATE_CAP`/`UPSTREAM`/`TIMEOUT` becomes retryable
    everywhere at once. The status is read off the exception the way
    `classify_exception` reads it: a `RateLimitError` carries 429 but spells
    none of the markers, so a text-only check never retried a rate limit.

    Never `ScannerTimeout`. That is our own wall-clock budget, and a second
    attempt would only burn it again; `is_timeout_error` already excludes it,
    and the explicit guard keeps that true if its message ever changes.

    A statusless timeout is deliberately *not* retried: nothing distinguishes
    the provider giving up from a local stall, and the caller's budget is the
    better place to decide. A timeout the provider actually answered with
    (408/504) is.
    """
    if isinstance(exc, ScannerTimeout):
        return False
    if http_status_of(exc) is not None:
        return (
            is_resource_exhausted(exc)
            or is_upstream_error(exc)
            or is_timeout_error(exc)
        )
    return is_rate_limited(exc) or any(
        marker in str(exc).lower() for marker in _RETRYABLE_MESSAGE_MARKERS
    )


# Which predicate claims each shared status meaning. Nothing reads this map —
# it exists so `require_total` fails at import if a `StatusKind` is added and
# no predicate here answers for it, which would otherwise surface as a
# silently less specific `error_reason` (and a vaguer exit code) rather than
# as a build failure. `AUTH` and `NOT_ENTITLED` overlap on purpose:
# `is_auth_error` claims both, for the console's coarser "credential is wrong"
# diagnostic, while `is_not_entitled` is the narrow half the callers that need
# them apart (`reasoner.py`, `inventory_metrics.py`) check first.
_PREDICATE_FOR_KIND = {
    StatusKind.AUTH: is_auth_error,
    StatusKind.NOT_ENTITLED: is_not_entitled,
    StatusKind.NOT_FOUND: is_model_error,
    StatusKind.TIMEOUT: is_timeout_error,
    StatusKind.RATE_CAP: is_resource_exhausted,
    StatusKind.UPSTREAM: is_upstream_error,
}
require_total(_PREDICATE_FOR_KIND, "exception_classification's predicates")


def classify_exception(exc: BaseException) -> FailureDomain:
    """Map an exception to the party responsible for it."""
    if isinstance(exc, ScannerTimeout):
        return FailureDomain.INTERNAL
    if isinstance(exc, _USER_EXCEPTIONS):
        return FailureDomain.USER
    if isinstance(exc, _PROVIDER_EXCEPTIONS):
        return FailureDomain.PROVIDER

    status = http_status_of(exc)
    if status is not None:
        # 429 and 5xx are the backend having a bad day; other 4xx means the
        # caller sent something wrong (expired token, bad instance).
        if status == 429 or status >= 500:
            return FailureDomain.PROVIDER
        if status >= 400:
            return FailureDomain.USER

    # Before the provider-module check: a keyless openai error lives in the
    # `openai` module but is the customer's missing key, not a provider outage.
    # A real 401/403 already returned `user` above, so this only catches the
    # statusless credential shapes (google-genai's ValueError, keyless SDKs).
    if _looks_like_auth_error(exc):
        return FailureDomain.USER

    module = type(exc).__module__
    if any(module.startswith(prefix) for prefix in _PROVIDER_MODULE_PREFIXES):
        return FailureDomain.PROVIDER

    if type(exc) is Exception:
        # A bare `raise Exception(...)` carries no signal either way.
        return FailureDomain.UNKNOWN
    return FailureDomain.INTERNAL


def http_status_of(exc: BaseException) -> int | None:
    """Best-effort HTTP status from an httpx/requests-style exception.

    Public because the CI entrypoint needs a finer split than ``FailureDomain``
    gives it — 401 and 429 are both `user`/`provider` respectively but map to
    different exit codes — and a second copy of this getattr dance there would
    be free to drift from the one the classifier uses.
    """
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None
