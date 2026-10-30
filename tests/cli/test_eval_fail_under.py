"""`eval run --fail-under`: the gate that makes exit 5 reachable, and the
completeness check beside it that is not gated at all.

Three outcomes have to stay distinguishable, because a CI job branches on
them: the bar was met (0), the bar was missed (5), and there was no verdict to
judge (23 / 6). Collapsing the third into either of the others is the false
green — or the false red — these checks exist to prevent. Only the gate is
opt-in; exit 6 fires with or without the flag (CHOO-3479).

`TestTheModelClassificationIsGatedOnTheVerdict` covers the fourth outcome the
other three are easiest to confuse with: a run that scored fine while some of
its model calls failed and recovered. `TestAnAbortedSuite` and
`TestTheConsecutiveFailureAbortEndToEnd` cover the fifth, which scores and is
still not a verdict: a run the CLI gave up on with targets it never reached.
`TestAStopMidBatch` covers the ordering all of them depend on from the other
side: whatever ends a run, the report is written before the exit code leaves
`handle_run`.
"""

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from switch_trust.cli import eval_cli, rich_observer, runner
from switch_trust.cli.runner import CliRunResult
from switch_trust.cli.exit_codes import CliExit, ExitCode
from switch_trust.eval.core.eval.evaluation import EvaluationStatus, EvaluationSummary
from switch_trust.eval.core.models import model as model_module
from switch_trust.eval.core.models.model import ModelErrorAccumulator
from switch_trust.eval.core.models.model_errors import ErrorKind


def _accumulator(
    *kinds_and_statuses: tuple[ErrorKind, object],
) -> ModelErrorAccumulator:
    """See `test_wrong_url.py::_accumulator`."""
    accumulator = ModelErrorAccumulator()
    failures = [
        Exception(f"failure {index}") for index in range(len(kinds_and_statuses))
    ]
    for (kind, status), exc in zip(kinds_and_statuses, failures, strict=True):
        accumulator.record_once(exc, "openai", status, None, kind)
    return accumulator


def _no_model_errors() -> ModelErrorAccumulator:
    """The accumulator of a run whose every model call succeeded."""
    return ModelErrorAccumulator()


def _result(name: str, achieved: float, maximum: float = 1.0) -> CliRunResult:
    """A finished evaluation scoring `achieved / maximum`."""
    return CliRunResult(
        model_evaluation_id=f"me_{name}",
        model_evaluation_name=f"{name} on a model",
        evaluation={"id": f"ev_{name}", "name": name, "type": "probe"},
        summary=EvaluationSummary(
            status=EvaluationStatus.FINISHED,
            total_evaluations=1,
            finished_evaluations=1,
            error_evaluations=0,
            max_score=maximum,
            achieved_score=achieved,
        ),
    )


def _errored(name: str) -> CliRunResult:
    """An evaluation that never produced a score — `summary.score` is None."""
    return CliRunResult(
        model_evaluation_id=f"me_{name}",
        model_evaluation_name=f"{name} on a model",
        evaluation={"id": f"ev_{name}", "name": name, "type": "probe"},
        summary=EvaluationSummary(
            status=EvaluationStatus.ERROR,
            total_evaluations=1,
            finished_evaluations=0,
            error_evaluations=1,
            max_score=1.0,
            achieved_score=0.0,
            error_messages=["provider unreachable"],
        ),
    )


def _args(fail_under: float | None) -> argparse.Namespace:
    return argparse.Namespace(fail_under=fail_under)


def _check_completed(
    results: list[CliRunResult],
    model_errors: ModelErrorAccumulator,
    output_path: str = "out.json",
    planned: int | None = None,
) -> None:
    """`_check_run_completed` for a suite that ran every target it resolved.

    `planned` defaults to the number that ran, which is what `handle_run`
    passes when the loop went all the way through. `TestAnAbortedSuite` is the
    one caller that passes a larger count, because that shortfall is the only
    record the abort leaves.
    """
    eval_cli._check_run_completed(
        results,
        model_errors,
        len(results) if planned is None else planned,
        output_path,
    )


class TestTheThreshold:
    """argparse rejects a bad value, so a bad gate never costs a model call."""

    @pytest.mark.parametrize("raw", ["0", "0.5", "1", "1.0"])
    def test_a_score_in_range_is_accepted(self, raw):
        assert eval_cli._score_threshold(raw) == float(raw)

    @pytest.mark.parametrize("raw", ["-0.1", "1.1", "80", "100"])
    def test_a_score_out_of_range_is_rejected(self, raw):
        """`--fail-under 80` is the percentage mistake; it must not mean 'never'."""
        with pytest.raises(argparse.ArgumentTypeError):
            eval_cli._score_threshold(raw)

    @pytest.mark.parametrize("raw", ["high", "", "0.8.1"])
    def test_a_non_number_is_rejected(self, raw):
        with pytest.raises(argparse.ArgumentTypeError):
            eval_cli._score_threshold(raw)

    def test_the_flag_reaches_the_parser(self):
        """End to end through argparse, not just the type function."""
        parser = argparse.ArgumentParser()
        eval_cli.register(parser.add_subparsers(dest="command"))

        args = parser.parse_args(["eval", "run", "me_1", "--fail-under", "0.8"])

        assert args.fail_under == 0.8

    def test_the_flag_is_optional(self):
        parser = argparse.ArgumentParser()
        eval_cli.register(parser.add_subparsers(dest="command"))

        assert parser.parse_args(["eval", "run", "me_1"]).fail_under is None

    def test_a_rejected_value_exits_two(self):
        """The table's usage code, via argparse's own SystemExit."""
        parser = argparse.ArgumentParser()
        eval_cli.register(parser.add_subparsers(dest="command"))

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["eval", "run", "me_1", "--fail-under", "80"])

        assert exc_info.value.code == ExitCode.USAGE


class TestTheGate:
    def test_no_flag_never_trips(self):
        """The default is ungated: a terrible score still exits 0."""
        eval_cli._apply_fail_under(
            _args(None), [_result("jailbreak", 0.01)], "out.json"
        )

    def test_a_score_above_the_bar_passes(self):
        eval_cli._apply_fail_under(_args(0.8), [_result("jailbreak", 0.9)], "out.json")

    def test_a_score_exactly_on_the_bar_passes(self):
        """Strictly-below: `--fail-under 0.8` is the bar a score must *meet*."""
        eval_cli._apply_fail_under(_args(0.8), [_result("jailbreak", 0.8)], "out.json")

    def test_a_score_below_the_bar_trips(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._apply_fail_under(
                _args(0.8),
                [_result("jailbreak", 0.42)],
                "out.json",
            )

        assert exc_info.value.code == ExitCode.FINDINGS_THRESHOLD

    def test_one_bad_evaluation_trips_a_passing_average(self):
        """The gate is per-evaluation, not on the aggregate.

        A mean would let four strong scores average away a total failure on a
        fifth, which is the one thing a safety gate must not do.
        """
        results = [
            _result("a", 1.0),
            _result("b", 1.0),
            _result("c", 1.0),
            _result("d", 1.0),
            _result("jailbreak", 0.0),
        ]

        with pytest.raises(CliExit) as exc_info:
            eval_cli._apply_fail_under(_args(0.8), results, "out.json")

        assert exc_info.value.code == ExitCode.FINDINGS_THRESHOLD
        assert "1 of 5" in exc_info.value.message

    def test_the_message_names_the_gate_and_what_tripped_it(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._apply_fail_under(
                _args(0.8),
                [_result("jailbreak", 0.42), _result("clean", 0.95)],
                "out.json",
            )

        message = exc_info.value.message
        assert "0.80" in message  # the bar
        assert "jailbreak (0.42)" in message  # what missed it
        assert "clean" not in message  # what did not

    def test_the_worst_offenders_are_named_first(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._apply_fail_under(
                _args(0.9),
                [_result("mid", 0.5), _result("worst", 0.1), _result("near", 0.8)],
                "out.json",
            )

        message = exc_info.value.message
        assert message.index("worst") < message.index("mid") < message.index("near")

    def test_a_long_list_is_truncated(self):
        """The message goes in a fixed-width panel, so it cannot be unbounded."""
        results = [_result(f"eval-{i}", 0.1) for i in range(10)]

        with pytest.raises(CliExit) as exc_info:
            eval_cli._apply_fail_under(_args(0.8), results, "out.json")

        assert "+7 more" in exc_info.value.message

    def test_the_report_path_rides_along(self):
        """The gate trips after the write, so the footer can still name it."""
        with pytest.raises(CliExit) as exc_info:
            eval_cli._apply_fail_under(
                _args(0.8),
                [_result("jailbreak", 0.42)],
                "eval_20260101T000000.json",
            )

        assert exc_info.value.output_path == "eval_20260101T000000.json"


class TestUnscorableRuns:
    """A run that produced no verdict is neither a pass nor a tripped gate."""

    def test_an_all_errored_run_is_incomplete_not_a_tripped_gate(self):
        """Exit 6, not 5: a provider outage is not a model failing the bar.

        Scoring an errored evaluation as zero would report the outage as the
        worst possible result, and a wrapper branching on 5 would act on a
        verdict that was never measured.
        """
        with pytest.raises(CliExit) as exc_info:
            _check_completed(
                [_errored("jailbreak"), _errored("pii")],
                _no_model_errors(),
                "out.json",
            )

        assert exc_info.value.code == ExitCode.EVAL_INCOMPLETE
        assert "2 of 2" in exc_info.value.message
        assert exc_info.value.output_path == "out.json"

    def test_an_all_errored_run_is_incomplete_without_the_flag_too(self):
        """CHOO-3479, in one assertion.

        Exit 6 is the deliberate backwards-compatibility break: unlike the
        gate, it is not opt-in, because a run with no verdict at all was
        never the "clean scan" exit 0 reported it as.
        """
        with pytest.raises(CliExit) as exc_info:
            _check_completed([_errored("jailbreak")], _no_model_errors(), "out.json")

        assert exc_info.value.code == ExitCode.EVAL_INCOMPLETE

    def test_the_message_does_not_name_a_gate_that_may_not_be_set(self):
        """It fires with or without `--fail-under`, so it cannot cite one."""
        with pytest.raises(CliExit) as exc_info:
            _check_completed([_errored("jailbreak")], _no_model_errors(), "out.json")

        assert "--fail-under" not in exc_info.value.message

    def test_one_score_is_enough_to_make_a_run_complete(self):
        _check_completed(
            [_errored("jailbreak"), _result("clean", 0.95)],
            _no_model_errors(),
            "out.json",
        )

    def test_the_gate_never_reports_a_scoreless_run_as_passing(self):
        """The ordering in `handle_run` is load-bearing, so the gate guards it.

        If the completeness check were ever moved after the gate, a bare
        `return` here would report the bar as met for a run that measured
        nothing — exactly the false green being fixed.
        """
        with pytest.raises(CliExit) as exc_info:
            eval_cli._apply_fail_under(_args(0.8), [_errored("pii")], "out.json")

        assert exc_info.value.code == ExitCode.EVAL_INCOMPLETE

    def test_a_partially_errored_run_is_judged_on_what_scored(self):
        """Documented scope limit: 'some errored' has no policy yet.

        The one that ran cleared the bar, so the run passes. If this ever
        becomes exit 6, it is a deliberate product decision — see the note in
        `eval_cli.py`.
        """
        eval_cli._apply_fail_under(
            _args(0.8),
            [_errored("jailbreak"), _result("clean", 0.95)],
            "out.json",
        )

    def test_a_partially_errored_run_still_trips_on_a_real_score(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._apply_fail_under(
                _args(0.8),
                [_errored("jailbreak"), _result("weak", 0.3)],
                "out.json",
            )

        assert exc_info.value.code == ExitCode.FINDINGS_THRESHOLD

    def test_a_zero_max_score_is_unscorable_not_zero(self):
        """`score` is None when max_score is 0 — an empty evaluation.

        Treating the division-by-zero guard as a 0.0 score would fail a gate
        on an evaluation that asked no questions.
        """
        with pytest.raises(CliExit) as exc_info:
            _check_completed(
                [_result("empty", achieved=0.0, maximum=0.0)],
                _no_model_errors(),
                "out.json",
            )

        assert exc_info.value.code == ExitCode.EVAL_INCOMPLETE
        # Unscored, but not *errored* — the count must not claim otherwise.
        assert "0 of 1" in exc_info.value.message


class TestAnAbortedSuite:
    """`handle_run` gives up after `MAX_CONSECUTIVE_RUN_FAILURES`, and the
    evaluations it never reached must not be reported as a clean run.

    The abort itself leaves no flag — only the shortfall between the targets
    resolved and the results collected — so every assertion here is about a
    `planned` larger than the list. One early score used to satisfy the
    "did anything score" question and carry the whole run to exit 0.
    """

    def test_a_run_that_gave_up_partway_is_not_a_pass(self):
        """The reviewer's case: 1 of 10 scored, 3 errored, 6 never ran."""
        results = [_result("clean", 0.95)] + [_errored(f"failed-{i}") for i in range(3)]

        with pytest.raises(CliExit) as exc_info:
            _check_completed(results, _no_model_errors(), planned=10)

        assert exc_info.value.code == ExitCode.EVAL_INCOMPLETE
        assert "6 of 10" in exc_info.value.message

    def test_the_skipped_targets_do_not_depend_on_the_failure_kind(self):
        """`UNKNOWN` was the reviewer's example, but the kind is beside the
        point — the evaluations that never ran are missing from the report
        whatever the ones that did ran into.
        """
        with pytest.raises(CliExit) as exc_info:
            _check_completed(
                [_result("clean", 0.95), _errored("jailbreak")],
                _accumulator((ErrorKind.UNKNOWN, 500)),
                planned=9,
            )

        assert exc_info.value.code == ExitCode.EVAL_INCOMPLETE

    def test_the_report_path_rides_along(self):
        """It stops after `write_output`, so the footer can still name it."""
        with pytest.raises(CliExit) as exc_info:
            _check_completed(
                [_result("clean", 0.95)],
                _no_model_errors(),
                "eval_20260101T000000.json",
                planned=4,
            )

        assert exc_info.value.output_path == "eval_20260101T000000.json"

    def test_a_scoreless_aborted_run_counts_the_targets_it_never_reached(self):
        """`len(results)` alone makes a 3-of-10 abort read as '3 of 3 errored',
        which tells the user the suite ran and failed rather than that most of
        it never ran.
        """
        with pytest.raises(CliExit) as exc_info:
            _check_completed(
                [_errored(f"failed-{i}") for i in range(3)],
                _no_model_errors(),
                planned=10,
            )

        assert exc_info.value.code == ExitCode.EVAL_INCOMPLETE
        assert "3 of 3 errored" in exc_info.value.message
        assert "7 never ran" in exc_info.value.message

    def test_a_finished_suite_says_nothing_about_skipped_targets(self):
        """The clause is absent, not zero — 'and 0 never ran' is noise on the
        ordinary all-errored run, which is the common case for this message.
        """
        with pytest.raises(CliExit) as exc_info:
            _check_completed([_errored("jailbreak")], _no_model_errors())

        assert "never ran" not in exc_info.value.message

    def test_a_scoreless_aborted_run_is_still_named_by_its_kind(self):
        """Precedence is unchanged: 20-39 beats 6 when the accumulator knows
        why, and an abort is usually a run where it does.
        """
        with pytest.raises(CliExit) as exc_info:
            _check_completed(
                [_errored("jailbreak")],
                _accumulator((ErrorKind.AUTH_FAILED, 401)),
                planned=10,
            )

        assert exc_info.value.code == ExitCode.AUTH_FAILED


class TestTheModelClassificationIsGatedOnTheVerdict:
    """`dominant_kind()` ranks the recorded kinds against each other, never
    against the calls that succeeded, so it is non-`None` whenever a single
    call failed anywhere in the run — including one `RetryModel` recovered.
    It may therefore only be consulted once the run is known to have scored
    nothing; otherwise it reports a healthy run as broken.
    """

    def test_a_recovered_transient_does_not_break_a_scored_run(self):
        """The reviewer's case: three 429s the retry absorbed, 997 clean
        prompts, every score valid. A wrapper branching on the 60-79 band
        would re-run the lot for nothing.
        """
        _check_completed(
            [_result("clean", 0.95)],
            _accumulator(
                (ErrorKind.QUOTA_EXCEEDED, 429),
                (ErrorKind.QUOTA_EXCEEDED, 429),
                (ErrorKind.QUOTA_EXCEEDED, 429),
            ),
            "out.json",
        )

    def test_a_rejected_credential_behind_a_scored_run_is_still_exit_zero(self):
        """The same gate, at its least comfortable: most of the run failed on
        a bad key and it *still* exits 0 because something scored.

        That is the documented scope limit, not an oversight — 'some errored'
        has no policy yet, and the per-evaluation errors are already printed
        above. Pinned here so the next change to it is deliberate.
        """
        _check_completed(
            [_errored("jailbreak"), _result("clean", 0.95)],
            _accumulator(*[(ErrorKind.AUTH_FAILED, 401)] * 9),
            "out.json",
        )

    def test_a_scoreless_run_is_named_by_its_dominant_kind(self):
        """With no verdict, the prevailing kind *is* why — so exit 20 rather
        than the bare 'no verdict' of exit 6, which tells the user nothing.
        """
        with pytest.raises(CliExit) as exc_info:
            _check_completed(
                [_errored("jailbreak")],
                _accumulator((ErrorKind.AUTH_FAILED, 401)),
                "out.json",
            )

        assert exc_info.value.code == ExitCode.AUTH_FAILED
        assert exc_info.value.output_path == "out.json"

    def test_a_kind_the_exit_table_cannot_name_falls_back_to_six(self):
        """`internal_error` has no entry, and a scoreless run must not leave
        on the strength of that — exit 6 is the floor, not a fallthrough.
        """
        with pytest.raises(CliExit) as exc_info:
            _check_completed(
                [_errored("jailbreak")],
                _accumulator((ErrorKind.INTERNAL_ERROR, 500)),
                "out.json",
            )

        assert exc_info.value.code == ExitCode.EVAL_INCOMPLETE


@pytest.fixture
def stubbed_run(monkeypatch, tmp_path):
    """Drive the real `handle_run` with the eval framework stubbed out.

    Everything the gate depends on stays real — the flag, the ordering of the
    write against the check, and the value `handle_run` returns or raises. Only
    the parts that would reach a model provider are replaced.
    """

    def _configure(
        results: list[CliRunResult | BaseException],
        fail_under: float | None,
        model_errors: list[tuple[ErrorKind, object]] | None = None,
    ):
        model_evaluations = [
            SimpleNamespace(
                id=f"me_{i}",
                evaluation_id=f"ev_{i}",
                model_id=f"model_{i}",
            )
            for i, _ in enumerate(results)
        ]
        monkeypatch.setattr(
            eval_cli,
            "_resolve_run_targets",
            lambda *a, **k: model_evaluations,
        )
        monkeypatch.setattr(rich_observer, "create_progress", MagicMock())
        monkeypatch.setattr(runner, "print_results", lambda *a, **k: None)
        monkeypatch.setattr(runner, "log_run_summary", lambda *a, **k: None)

        remaining = list(results)
        # Held in the closure rather than built per call: `record_once`
        # dedupes on `id(exc)`, so a temporary would let a recycled id drop a
        # recording. Same reason as `_accumulator` above.
        failures = [
            Exception(f"failure {index}") for index in range(len(model_errors or []))
        ]

        async def _run_one(*args, **kwargs):
            # Record onto the live accumulator `handle_run` has open, the way
            # a real `Model.generate()` inside the evaluation would. This is
            # the only way to exercise the verdict gate end to end: the
            # accumulator is created inside `handle_run` and never passed in.
            accumulator = model_module._error_recorder.get()
            if accumulator is not None:
                for (kind, status), exc in zip(
                    model_errors or [], failures, strict=True
                ):
                    accumulator.record_once(exc, "openai", status, None, kind)
            # An entry may be an exception, standing in for the evaluation
            # that raises partway through the batch.
            outcome = remaining.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        monkeypatch.setattr(runner, "run_cli_evaluation", _run_one)

        written: list[str] = []

        def _write(runs, config_path, output_path, fmt="json"):
            # Write for real: the point of checking the gate after this call
            # is that a tripped job still has its report.
            with open(output_path, "w") as handle:
                handle.write("{}")
            written.append(output_path)

        monkeypatch.setattr(runner, "write_output", _write)

        store = MagicMock()
        store.evaluations.get.return_value = SimpleNamespace(name="an evaluation")

        args = argparse.Namespace(
            config="config.json",
            model_evaluation_id="me_1",
            model=None,
            output=str(tmp_path / "eval.json"),
            format="json",
            concurrency=1,
            model_tag=[],
            eval_tag=[],
            fail_under=fail_under,
        )
        return args, store, written

    return _configure


class TestHandleRunEndToEnd:
    """The gate as `main` reaches it, not as a helper called directly."""

    @pytest.mark.asyncio
    async def test_a_passing_run_returns_its_output_path(self, stubbed_run):
        args, store, written = stubbed_run([_result("clean", 0.95)], fail_under=0.8)

        output_path = await eval_cli.handle_run(args, store)

        assert output_path == args.output
        assert written == [args.output]

    @pytest.mark.asyncio
    async def test_a_tripped_gate_raises_five_and_keeps_the_report(
        self,
        stubbed_run,
        tmp_path,
    ):
        """Exit 5 reaches `main`, and the artifact survives the stop."""
        args, store, written = stubbed_run([_result("jailbreak", 0.42)], fail_under=0.8)

        with pytest.raises(CliExit) as exc_info:
            await eval_cli.handle_run(args, store)

        assert exc_info.value.code == ExitCode.FINDINGS_THRESHOLD
        # Written before the gate was checked, so CI can still collect it.
        assert written == [args.output]
        assert (tmp_path / "eval.json").exists()
        assert exc_info.value.output_path == args.output

    @pytest.mark.asyncio
    async def test_an_ungated_run_with_a_bad_score_still_returns_normally(
        self,
        stubbed_run,
    ):
        args, store, _ = stubbed_run([_result("jailbreak", 0.01)], fail_under=None)

        assert await eval_cli.handle_run(args, store) == args.output

    @pytest.mark.asyncio
    async def test_an_ungated_run_that_scored_nothing_raises_six(
        self,
        stubbed_run,
        tmp_path,
    ):
        """CHOO-3479 end to end: no flag, no score, and no longer exit 0."""
        args, store, written = stubbed_run(
            [_errored("jailbreak"), _errored("pii")],
            fail_under=None,
        )

        with pytest.raises(CliExit) as exc_info:
            await eval_cli.handle_run(args, store)

        assert exc_info.value.code == ExitCode.EVAL_INCOMPLETE
        # Written before the check, so CI can still collect the errors.
        assert written == [args.output]
        assert (tmp_path / "eval.json").exists()
        assert exc_info.value.output_path == args.output

    @pytest.mark.asyncio
    async def test_a_recovered_transient_leaves_a_good_run_at_zero(
        self,
        stubbed_run,
    ):
        """The gate through the real `track_model_errors()` scope.

        Every attempt is recorded at the `Model.generate()` choke point and
        `RetryModel` retries through that same public method, so a 429 the
        retry absorbed is in the accumulator of a run that then scored
        perfectly. Before the gate this exited 64.
        """
        args, store, _ = stubbed_run(
            [_result("clean", 0.95)],
            fail_under=0.8,
            model_errors=[(ErrorKind.QUOTA_EXCEEDED, 429)],
        )

        assert await eval_cli.handle_run(args, store) == args.output

    @pytest.mark.asyncio
    async def test_a_scoreless_run_still_exits_on_its_dominant_kind(
        self,
        stubbed_run,
        tmp_path,
    ):
        """The gate must not cost the classification it guards: with nothing
        scored, exit 20 still beats the bare exit 6.
        """
        args, store, written = stubbed_run(
            [_errored("jailbreak")],
            fail_under=None,
            model_errors=[(ErrorKind.AUTH_FAILED, 401)],
        )

        with pytest.raises(CliExit) as exc_info:
            await eval_cli.handle_run(args, store)

        assert exc_info.value.code == ExitCode.AUTH_FAILED
        assert written == [args.output]
        assert (tmp_path / "eval.json").exists()
        assert exc_info.value.output_path == args.output

    @pytest.mark.asyncio
    async def test_completeness_outranks_the_gate(self, stubbed_run):
        """6 before 5 in the documented precedence.

        Nothing scored, so there is no verdict for the gate to judge — a
        wrapper branching on 5 must not be told the bar was missed.
        """
        args, store, _ = stubbed_run([_errored("jailbreak")], fail_under=0.8)

        with pytest.raises(CliExit) as exc_info:
            await eval_cli.handle_run(args, store)

        assert exc_info.value.code == ExitCode.EVAL_INCOMPLETE


class TestTheConsecutiveFailureAbortEndToEnd:
    """The abort in `handle_run` itself, rather than a `planned` count passed
    in by hand.

    `MAX_CONSECUTIVE_RUN_FAILURES` errored evaluations in a row end the loop
    and the targets behind them are never attempted. Unlike the `CliExit`
    stop in `TestAStopMidBatch`, the abort carries nothing out of the loop —
    only the shortfall in the results list — so these drive the real loop.
    """

    @pytest.mark.asyncio
    async def test_a_suite_abandoned_after_one_score_does_not_exit_zero(
        self,
        stubbed_run,
        tmp_path,
    ):
        """One score, three failures, six never attempted: the report covers
        well under half the suite and used to leave as a clean run.
        """
        args, store, written = stubbed_run(
            [_result("clean", 0.95)]
            + [_errored(f"failed-{i}") for i in range(3)]
            + [_result(f"never-reached-{i}", 0.95) for i in range(6)],
            fail_under=None,
        )

        with pytest.raises(CliExit) as exc_info:
            await eval_cli.handle_run(args, store)

        assert exc_info.value.code == ExitCode.EVAL_INCOMPLETE
        assert "6 of 10" in exc_info.value.message
        # Written first either way: the four that ran are what gets debugged.
        assert written == [args.output]
        assert (tmp_path / "eval.json").exists()
        assert exc_info.value.output_path == args.output

    @pytest.mark.asyncio
    async def test_the_gate_is_not_judged_on_the_fragment_that_ran(self, stubbed_run):
        """6 before 5 for the other shortfall. The evaluation that happened to
        run first cleared the bar; six that never ran cannot have.
        """
        args, store, _ = stubbed_run(
            [_result("clean", 0.95)]
            + [_errored(f"failed-{i}") for i in range(3)]
            + [_result(f"never-reached-{i}", 0.01) for i in range(6)],
            fail_under=0.8,
        )

        with pytest.raises(CliExit) as exc_info:
            await eval_cli.handle_run(args, store)

        assert exc_info.value.code == ExitCode.EVAL_INCOMPLETE

    @pytest.mark.asyncio
    async def test_an_intermittent_failure_still_finishes_and_exits_zero(
        self,
        stubbed_run,
    ):
        """The counter resets on a success, so a suite that limps through
        every target is a *finished* run — the documented scope note, which
        this change must not quietly turn red.
        """
        args, store, _ = stubbed_run(
            [
                _errored("a"),
                _errored("b"),
                _result("clean", 0.95),
                _errored("c"),
                _errored("d"),
            ],
            fail_under=None,
        )

        assert await eval_cli.handle_run(args, store) == args.output


class TestAStopMidBatch:
    """An evaluation that raises must not take the whole batch's report with it.

    `run_cli_evaluation` turns a missing judge credential, an unreadable
    prompt file and an absent optional tool into a `CliExit`. Any of them can
    first surface on the third evaluation of five — the judge is built per
    evaluation, by whichever type needs one — and an uncaught raise from
    inside the loop unwinds past `write_output`, so the two that already
    scored are measured, paid for, and then discarded.
    """

    @pytest.mark.asyncio
    async def test_an_earlier_score_still_reaches_the_report(
        self,
        stubbed_run,
        tmp_path,
    ):
        args, store, written = stubbed_run(
            [
                _result("clean", 0.95),
                CliExit(
                    ExitCode.AUTH_FAILED,
                    "The judge/generator model has no API key configured.",
                ),
                _result("never_reached", 0.10),
            ],
            fail_under=None,
        )

        with pytest.raises(CliExit) as exc_info:
            await eval_cli.handle_run(args, store)

        assert exc_info.value.code == ExitCode.AUTH_FAILED
        assert written == [args.output]
        assert (tmp_path / "eval.json").exists()
        # The artifact is what a gated CI job collects, so the stop has to
        # name it — `handle_run`'s return value never happened.
        assert exc_info.value.output_path == args.output

    @pytest.mark.asyncio
    async def test_the_stop_outranks_the_gate_the_partial_run_would_trip(
        self,
        stubbed_run,
    ):
        """20-39 before 5: the run never finished, so its scores are not a verdict.

        The one evaluation that ran is below the bar, so without this the job
        would exit 5 — "your model failed" — for a missing credential.
        """
        args, store, _ = stubbed_run(
            [
                _result("weak", 0.42),
                CliExit(ExitCode.MISSING_TOOLS, "garak is not installed."),
            ],
            fail_under=0.8,
        )

        with pytest.raises(CliExit) as exc_info:
            await eval_cli.handle_run(args, store)

        assert exc_info.value.code == ExitCode.MISSING_TOOLS

    @pytest.mark.asyncio
    async def test_a_stop_on_the_first_evaluation_carries_nothing(
        self,
        stubbed_run,
    ):
        """Nothing scored, so nothing was written and there is no path to carry.

        Gated, because the empty `results` this leaves also reaches
        `_gate_targets`: 23 would tell the user to fix a selector that was
        never the problem.
        """
        args, store, written = stubbed_run(
            [CliExit(ExitCode.TARGET_ERROR, "Could not read prompts.csv")],
            fail_under=0.8,
        )

        with pytest.raises(CliExit) as exc_info:
            await eval_cli.handle_run(args, store)

        assert exc_info.value.code == ExitCode.TARGET_ERROR
        assert exc_info.value.output_path is None
        assert written == []


class TestNothingToRun:
    """A gated run that resolved no targets must not report success."""

    def test_no_targets_under_a_gate_is_a_config_error(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._gate_targets(_args(0.8), [])

        assert exc_info.value.code == ExitCode.NO_TARGETS
        assert exc_info.value.code.needs_intervention

    def test_no_targets_without_a_gate_is_unchanged(self):
        """Still a yellow note and exit 0 for everyone not using --fail-under."""
        eval_cli._gate_targets(_args(None), [])

    def test_targets_present_never_trips(self):
        eval_cli._gate_targets(_args(0.8), [object()])
