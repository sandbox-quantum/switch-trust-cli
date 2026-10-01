"""Shared HTTP-error handling for the aiohttp-backed model/agent types.

`resp.raise_for_status()` raises `aiohttp.ClientResponseError` from just
`status` + `reason` + `url` — the response body, where the useful detail
lives (e.g. a Google quota error's `error.status`/`consumer`), is discarded.

The body is never logged here, deliberately: `endpoint_url`/`db_model.host`
(consumed by every caller of `raise_for_status()` below) is tenant-controlled
(the same field `platform/ssrf.py` guards), so the body can carry arbitrary
tenant/user content — echoed prompts, internal stack traces, anything the
tenant's server chooses to put in an error response. There is no reliable way
to tell a safe platform-template error apart from arbitrary tenant content
from the outside, so instead of logging the body, this module extracts two
narrow, content-free signals from it and discards the rest.
"""

from __future__ import annotations

import json
import re
from urllib.parse import urlsplit, urlunsplit

import aiohttp

# Bodies are only read for their two content-free signals -- there is never a
# reason to buffer more than a small error payload needs, and an unbounded
# read lets a misbehaving tenant server exhaust worker memory with one huge
# response.
_MAX_BODY_BYTES = 64 * 1024

# The fixed google.rpc.Code vocabulary. Only a status string that is an exact
# match is ever surfaced -- this makes the field safe to log even though it
# came from a tenant-reachable response: a server can put arbitrary text in a
# field called "status", but it can't make arbitrary text collide with one of
# these 17 fixed names, so anything else is treated as unrecognized and
# dropped.
_KNOWN_ERROR_STATUSES = frozenset(
    {
        "OK",
        "CANCELLED",
        "UNKNOWN",
        "INVALID_ARGUMENT",
        "DEADLINE_EXCEEDED",
        "NOT_FOUND",
        "ALREADY_EXISTS",
        "PERMISSION_DENIED",
        "UNAUTHENTICATED",
        "RESOURCE_EXHAUSTED",
        "FAILED_PRECONDITION",
        "ABORTED",
        "OUT_OF_RANGE",
        "UNIMPLEMENTED",
        "INTERNAL",
        "UNAVAILABLE",
        "DATA_LOSS",
    }
)
_PROJECT_NUMBER_RE = re.compile(r"/projects/(\d+)/")


class ModelHttpError(Exception):
    """HTTP error from a model/agent endpoint.

    Deliberately carries no response-body content (see module docstring).
    `error_status` and `same_project_as_url` are the only signals derived
    from the body, and both are content-free.

    Duck-types `.status` like `aiohttp.ClientResponseError` so the existing
    `_is_transient` (and any future `_is_terminal`) checks in
    `model_retry.py`, which read `status_code`/`status`/`code` via
    `getattr`, keep working unmodified.
    """

    def __init__(
        self,
        status: int,
        url: str,
        error_status: str | None,
        same_project_as_url: bool | None,
    ):
        self.status = status
        self.url = url
        self.error_status = error_status
        self.same_project_as_url = same_project_as_url
        parts = [f"{status} from {url}"]
        if error_status is not None:
            parts.append(f"error_status={error_status}")
        if same_project_as_url is not None:
            parts.append(f"same_project_as_url={same_project_as_url}")
        super().__init__(", ".join(parts))


async def raise_for_status(resp: aiohttp.ClientResponse) -> None:
    """Like `resp.raise_for_status()`, but derives two safe signals from the
    body first. The body itself is never stored on the raised exception or
    returned -- only `_extract_error_status()`/`_same_project_as_url()`'s
    outputs are."""
    if resp.status < 400:
        return
    raw = await resp.content.read(_MAX_BODY_BYTES)
    body = raw.decode(resp.charset or "utf-8", errors="replace")
    # Query/fragment can carry tenant-supplied secrets (e.g. an API key on a
    # GenericHttpModel/LangServe endpoint URL) -- strip them before this URL
    # flows into the exception string, logs, and the OTel span.
    parts = urlsplit(str(resp.url))
    url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    raise ModelHttpError(
        resp.status,
        url,
        _extract_error_status(body),
        _same_project_as_url(body, url),
    )


def _extract_error_status(body: str) -> str | None:
    """The fixed-vocabulary ``error.status`` enum (e.g. RESOURCE_EXHAUSTED),
    if the body has that shape *and* the value is one of the known names.
    Never returns ``error.message`` or any other free text -- that's exactly
    the field a tenant server could use to smuggle arbitrary content."""
    try:
        status = json.loads(body)["error"]["status"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return status if status in _KNOWN_ERROR_STATUSES else None


def _same_project_as_url(body: str, url: str) -> bool | None:
    """Whether the GCP project number already visible in `url` also appears
    as `project_number:<n>` in the body -- as a boolean only, never quoting
    the body itself. `None` when `url` has no project number (the check
    doesn't apply), distinct from `False` (checked, and it didn't match)."""
    match = _PROJECT_NUMBER_RE.search(url)
    if not match:
        return None
    return f"project_number:{match.group(1)}" in body
