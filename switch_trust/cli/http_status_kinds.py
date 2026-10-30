"""What an HTTP status *means*, in one table, for every caller that asks.

Four places used to answer this question independently — the scanner's
exception predicates (`inventory/common/exception_classification.py`), the
Action's upload handler (`inventory/standalone_inventory_action.py`), the eval
engine's error taxonomy (`eval/core/models/model_errors.py`) and the eval
CLI's post-mortem over the error accumulator (`cli/eval_cli.py`) — each with
its own spelling of "504 is a timeout" and "5xx is the backend". They had
already drifted: the upload handler called a 504 an upstream fault and had no
answer at all for a 408, which put a retryable failure in the non-retryable
band.

`StatusKind` is the shared half, and only the shared half. It deliberately
stops short of the two things that are *not* shareable:

- **What a kind means for you.** A 404 from an LLM provider is a bad model id;
  a 404 from our own ingestion host is a wrong instance URL. Same number,
  different remediation. Each caller keeps its own `StatusKind -> <its own
  vocabulary>` mapping, so the divergence is a declared cell rather than two
  `if status == 404` branches that happen to differ.
- **Exceptions that carry no status.** The LLM path has to sniff types and
  messages (litellm re-wraps a timeout into a class deriving from neither
  `TimeoutError` nor `httpx.TimeoutException`); the upload path must not,
  since `httpx.TimeoutException` covers it precisely there and a substring
  match on "timeout" would move an unrelated permanent failure into the retry
  band. `status_kind` takes an `int`, so that sniffing physically cannot leak
  across.

**Drift is prevented by `require_total`, not by this docstring.** A caller
that maps kinds to its own vocabulary validates the mapping at import, so
adding a member to `StatusKind` fails every consumer's import until it
declares an answer. There is no static type checker in CI, so an exhaustive
`match`/`assert_never` would only fire at runtime on the branch nobody
exercises — the import-time check fires on any import, which every test in
these packages already does.

Lives in `common/` for the same reason `exit_codes.py` and
`llm_credentials.py` do: `cli:lib_cli -> inventory:lib_agent_scanner` already
exists, so either package would close the loop. Stdlib-only, which is what
lets the eval engine's narrow `lib_model_errors` slice depend on it.
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import Any


class StatusKind(StrEnum):
    """The meaning every caller agrees on, independent of who answered.

    Not `ErrorKind`: that one is wire-visible (proto field 9 ->
    `eval_evaluation_runs.error_kind` -> the UI's string table -> Test
    Connection's `_REPORTABLE_KINDS`) and deliberately coarser, collapsing a
    404 and a 5xx into `UNREACHABLE`. Widening it to carry this distinction
    would change a published contract to tidy up an internal one.
    """

    AUTH = "auth"
    NOT_ENTITLED = "not_entitled"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    RATE_CAP = "rate_cap"
    UPSTREAM = "upstream"


# 504 is listed rather than left to the `>= 500` fallback below, and that is
# the whole point of the table: a gateway timing out is a timeout, not a
# generic backend fault. Expressing it here makes the two kinds mutually
# exclusive by construction — callers no longer carry a hand-written
# `and status != 504`, which is exactly the clause that had gone missing in
# one of the four copies.
_KINDS: Mapping[int, StatusKind] = {
    401: StatusKind.AUTH,
    403: StatusKind.NOT_ENTITLED,
    404: StatusKind.NOT_FOUND,
    408: StatusKind.TIMEOUT,
    429: StatusKind.RATE_CAP,
    504: StatusKind.TIMEOUT,
}


def status_kind(status: int | None) -> StatusKind | None:
    """The shared meaning of `status`, or `None` if it has none.

    `None` is not "unclassified, treat as benign" — it means this table has no
    opinion and the caller's own fallback applies. The eval engine reads a
    remaining 4xx as the provider rejecting the request *shape*
    (`INVALID_CONFIG`); the upload handler falls through to its transport
    checks and then to the non-retryable catch-all. Those answers genuinely
    differ, so neither belongs here.
    """
    if status is None:
        return None
    kind = _KINDS.get(status)
    if kind is not None:
        return kind
    if status >= 500:
        return StatusKind.UPSTREAM
    return None


def require_total(mapping: Mapping[StatusKind, Any], owner: str) -> None:
    """Fail at import unless `mapping` answers for every `StatusKind`.

    Call this at module scope, right under the mapping it guards. A contract
    test can only assert agreement on the kinds someone thought to list; this
    catches the kind nobody thought about, which is the drift mode that
    actually happens.
    """
    missing = sorted(kind.value for kind in set(StatusKind) - set(mapping))
    if missing:
        raise RuntimeError(
            f"{owner} does not map every StatusKind: missing {', '.join(missing)}. "
            "Every status meaning needs an answer here — see the module "
            "docstring in http_status_kinds.py."
        )
