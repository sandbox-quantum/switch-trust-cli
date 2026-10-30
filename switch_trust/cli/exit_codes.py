"""Switch Trust's exit-code contract: the machine-readable half of every outcome.

A caller that wants a CI/CD quality gate should be able to branch on `$?`
rather than parse a report, so every way a run can stop maps to exactly one
code here. The table is **banded**, so a wrapper branches on a range instead of
enumerating values::

    if   code == 0:          proceed
    elif 60 <= code < 80:    retry the identical invocation
    else:                    a human has to look at it

| code    | meaning                                          |
|---------|--------------------------------------------------|
| 0       | success; no findings, or none above the threshold |
| 1       | general error                                     |
| 2       | usage error (argparse)                            |
| 3-4     | reserved for the CI wrappers (see below)          |
| 5       | `scan --fail-on` or `eval run --fail-under` tripped |
| 6       | eval incomplete — nothing scored, so no verdict   |
| 20-39   | fix the configuration, then re-run                |
| 40-59   | unused — spare                                    |
| 60-79   | temporary failure — retry                         |
| 80-125  | unused — spare                                    |
| 126-255 | never emitted by us; shell and signal territory   |

126-255 is the shell's (`126` cannot execute, `127` not found, `128+N` killed
by signal N), so nothing here may land in it. `130` is the one value we do
emit, and only for its reserved meaning: SIGINT, i.e. Ctrl-C.

**One classification, two renderings.** The code and the sentence the user
reads must come from the same decision. `CliExit` carries both, which is what
stops a job exiting "retry me" under a message saying the API key is wrong — a
gate branching on the band would then be confidently wrong, which is strictly
worse than the single flat code this replaces.

**A failed call is not a failed run.** For `eval run`, the 20-39 and 60-79
codes that come from the model's own failures are reported only for a run that
produced *no* score — they name the cause behind exit 6 rather than standing
beside it. A run that reached every target it resolved and scored at least
one of them exits on its verdict (0, or 5 if a gate was set and missed),
however many individual calls failed along the way, because the CLI retries
transient model failures itself and a recovered 429 is not an outcome:
telling a wrapper to retry a complete, valid report is the same false signal
in the other direction as CHOO-3479's false green. A run that gave up partway
through the suite has neither property and exits 6. See
`eval_cli._check_run_completed`.

**Shared by two independent entrypoints.** The CLI (`CliExit`'s raiser and
catcher) and the inventory scanner's CI entrypoint (the GitHub Action / GitLab
CI/CD component wrapper, `standalone_inventory_action.py`) both emit this
table, and each used to carry its own copy of the numbers. This module lives
in a package neither of those two depends on the other through, so both can
import the single definition instead of keeping their copies in step by hand.

**Stdlib only, deliberately.** This module is the contract shared by the CLI,
its OSS export (`cli_exporter/export.py` copies it alongside `cli/`) and the
scanner's CI entrypoints, so it must stay importable without dragging in
httpx, OTEL or the scanner. Exception-to-code mapping belongs at the layer that
owns the exception, not here.
"""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    """Every status a run may exit with. The enum *is* the documented table."""

    # The scan or eval ran to completion and nothing tripped a gate. A findings
    # count of zero and a findings count below `--fail-on` are the same answer:
    # the gate the user asked for is green.
    OK = 0

    # Catchall. An exception we never classified lands here, which is why the
    # mapping from exception to code is an explicit allowlist rather than a
    # fallthrough — see the precedence note below.
    ERROR = 1

    # Bad invocation. argparse already exits 2 on an unparsable command line;
    # anything we reject ourselves for the same reason uses the same code.
    USAGE = 2

    # 3 and 4 are reserved for the CI wrappers and are never emitted by the
    # CLI itself. 3 is an upload failure the action could not classify (the
    # CLI does not upload). 4 was the action's single "precondition not met"
    # and is now unused: those causes were split across the 20-39 band, which
    # says which precondition failed. Reserved rather than recycled — a value
    # a released wrapper once emitted must not come to mean something else.
    UPLOAD_FAILED = 3
    PRECONDITION = 4

    # The run succeeded and the user's own gate tripped. Each command names
    # its gate after the measure it produces, and the two read in opposite
    # directions: `scan --fail-on <severity>` trips on findings at or *above*
    # a badness level, `eval run --fail-under <score>` on an evaluation
    # falling *below* a goodness bar. Not a CLI failure — the tool did its job.
    FINDINGS_THRESHOLD = 5

    # The eval produced no verdict on the suite the user asked for. Two
    # shortfalls count: no evaluation scored, or the run was cut short and
    # some evaluations never ran at all. A *finished* run where some errored
    # and the rest did score still exits 0, because how much error is too
    # much is a policy nobody has set — but a target that was never put to
    # the model is not an evaluation that passed, whatever the ones that ran
    # scored (see `eval_cli._check_run_completed`). It is the floor for the
    # scoreless case rather than the first answer: when the run's recorded
    # model failures name a cause, the 20-39 or 60-79 code for it is raised
    # instead, which tells the user what to change. Unlike 5, this is *not*
    # gated behind `--fail-under`: a run with nothing to report is the
    # CHOO-3479 false green, not a verdict the user opted in to having
    # judged — and a gate judging only the fragment that ran is the same
    # false green wearing the user's own threshold. Distinct from
    # 60-79 because the CLI has already exhausted the retries it does on its
    # own (per-prompt timeout / unreachable / quota); re-running the whole
    # eval is a decision for the user, not something a wrapper should
    # automate.
    EVAL_INCOMPLETE = 6

    # 20-39 — a human has to change something before a re-run can work.
    # Missing or rejected API key. The two are found at different points and
    # both have to be covered: a *rejected* key takes a call to discover, so
    # it is read off the run's recorded failures afterwards, while a *missing*
    # one stops the provider SDK building its client at all — no call, nothing
    # recorded, so it is refused by a preflight instead
    # (`scan_cli._check_configured_model`, `eval_cli._check_target_credentials`,
    # and `GeneratorCredentialError` for the eval judge). Without the
    # preflights the missing-key case fell through to 1.
    AUTH_FAILED = 20
    NOT_ENTITLED = 21  # valid credentials, not permitted (plan, quota, scope)
    TARGET_ERROR = 22  # path missing, unreadable, or unsupported
    NO_TARGETS = 23  # nothing scannable matched
    WRONG_URL = 24  # 404, or a host that does not resolve
    INVALID_CONFIG = 25  # config present but unusable
    MISSING_TOOLS = 26  # a tool the scan or eval needed is not installed

    # 60-79 — transient: the identical invocation may well succeed next time.
    NETWORK_FAILURE = 60
    RATE_LIMITED = 61  # a provider-declared rate limit arriving without a 429
    TIMEOUT = 62
    UPSTREAM_ERROR = 63  # 5xx from the platform or a model provider
    RESOURCE_EXHAUSTED = 64  # HTTP 429

    # Ctrl-C. The shell's own 128+SIGINT, kept for its reserved meaning only.
    INTERRUPTED = 130

    @property
    def is_retryable(self) -> bool:
        """Whether re-running the identical invocation could succeed."""
        return self in _RETRY_BAND

    @property
    def needs_intervention(self) -> bool:
        """Whether a human has to change something before a re-run can work."""
        return self in _FIX_CONFIG_BAND

    @property
    def reports_failure(self) -> bool:
        """Whether the command itself failed, as opposed to reporting a verdict.

        `FINDINGS_THRESHOLD` is the one non-zero code that is not a failure:
        the scan completed and the gate the user configured did exactly what
        they asked. Telemetry keys on this, so that enabling `--fail-on` /
        `--fail-under` in CI does not turn every gated job into a CLI error.
        """
        return self not in (ExitCode.OK, ExitCode.FINDINGS_THRESHOLD)


_RETRY_BAND = range(60, 80)
_FIX_CONFIG_BAND = range(20, 40)

# Precedence, most severe first: 1-4 > 20-39 > 60-79 > 6 > 5 > 0.
#
# A run can have more than one thing to say — an eval can be incomplete *and*
# score below the threshold — but there is no code here that adjudicates it at
# runtime: a run only ever produces one code, and this order is realised by
# the order each dispatch function runs its checks in
# (`scan_cli.py::handle_scan`, `eval_cli.py::handle_run`). The one `CliExit`
# that is not raised where it is diagnosed is the stop that ends an eval batch
# early: `handle_run` holds it so the evaluations that already scored still
# reach `write_output`, then re-raises it ahead of the three checks, which is
# the same order — every code `run_cli_evaluation` raises is 20-39. This
# comment is the documentation of that
# contract, not a spec for code that exists elsewhere — if a call path ever
# needs to pick a winner among several candidate codes from one run, a reducer
# over this order belongs here, but nothing today calls for one.


class CliExit(Exception):
    """A classified, deliberate stop: the exit code *and* the line explaining it.

    Raised where the problem is diagnosed and translated exactly once, in the
    CLI's `main`'s unified handler — which keeps telemetry recording, the error
    panel and the "Results:" footer on one path, and leaves the `output_path`
    every dispatch function already returns alone.

    `sys.exit` is not an alternative here: `SystemExit` derives from
    `BaseException`, so it flies past the `except Exception` that records the
    run to telemetry, and the exit would go unreported.

    The message is user-facing and must name the gate that failed and the value
    that tripped it ("3 findings at or above `high`"), not restate the code.

    `output_path` carries the report a stop may have produced before it was
    raised. A refusal (a missing scan target) has none, but a *gate* trips only
    after the run finished and wrote its artifact — and the artifact is the
    thing a gated CI job most wants to collect. Without this the "Results:"
    footer names nothing, because `main`'s handler never reaches the assignment
    of the dispatch function's return value.
    """

    def __init__(
        self,
        code: ExitCode,
        message: str,
        output_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = ExitCode(code)
        self.message = message
        self.output_path = output_path

    def __str__(self) -> str:
        return self.message
