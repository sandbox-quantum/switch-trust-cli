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
# already `user` (via `_http_status_of`), but the keyless ones carry no status
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


def is_auth_error(exc: BaseException) -> bool:
    """Whether ``exc`` is a rejected/missing LLM credential specifically.

    Narrower than ``classify_exception(...) == USER``: a 400/404 (bad request,
    unknown model) is also `user` but is *not* an auth failure. Used to tell the
    customer their **credential** is invalid, so it must not fire on other
    config mistakes. A 401/403 or an auth-shaped message (keyless SDK /
    google-genai's statusless ValueError) counts; everything else does not.
    """
    return _http_status_of(exc) in (401, 403) or _looks_like_auth_error(exc)


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
    return _http_status_of(exc) == 404 or _looks_like_model_error(exc)


def classify_exception(exc: BaseException) -> FailureDomain:
    """Map an exception to the party responsible for it."""
    if isinstance(exc, ScannerTimeout):
        return FailureDomain.INTERNAL
    if isinstance(exc, _USER_EXCEPTIONS):
        return FailureDomain.USER
    if isinstance(exc, _PROVIDER_EXCEPTIONS):
        return FailureDomain.PROVIDER

    status = _http_status_of(exc)
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


def _http_status_of(exc: BaseException) -> int | None:
    """Best-effort HTTP status from an httpx/requests-style exception."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None
