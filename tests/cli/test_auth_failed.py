"""Exit 20: a credential the user has to fix before re-running.

Both commands used to report a rejected key as success or as the catch-all 1.
`eval` ground through every prompt and exited 0 with a report full of errors;
`scan` swallowed the failure inside `run_core`, dropped the whole AI reasoning
layer, and returned a clean-looking static-only report, also 0. Neither told
the user the one thing they could act on.

The two commands detect it differently because their failures surface
differently, so they are tested separately rather than through a shared
harness that would hide that.
"""

import argparse
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from switch_trust.cli import eval_cli, runner, scan_cli
from switch_trust.cli.exit_codes import CliExit, ExitCode
from switch_trust.eval.core.models.generator_model import GeneratorCredentialError
from switch_trust.eval.core.models.model import ModelErrorAccumulator
from switch_trust.eval.core.models.model_errors import ErrorKind
from switch_trust.eval.db.base.models.model_types import ModelType


def _accumulator(*kinds: ErrorKind, status: int = 401) -> ModelErrorAccumulator:
    """An accumulator carrying one recorded failure per kind given.

    The exceptions are built up front and held in a list for the whole loop
    on purpose. `record_once` dedupes on `id(exc)`, and a temporary created
    inline is freed as soon as the call returns — CPython then hands the next
    allocation the same address, and every failure after the first is
    silently dropped as a duplicate. In production the exception is alive on
    the propagating stack, so this is a test-authoring hazard only.
    """
    accumulator = ModelErrorAccumulator()
    failures = [Exception(f"failure {index}") for index in range(len(kinds))]
    for kind, exc in zip(kinds, failures, strict=True):
        accumulator.record_once(exc, "openai", status, None, kind)
    return accumulator


class TestEvalCredentials:
    """The classifier in isolation — accumulator in, exit code out.

    It is only ever reached for a run that scored nothing, so "dominant" here
    means "the kind that best explains an empty run", never "most of the run
    failed this way". `test_eval_fail_under.py` owns that gate; these cases
    deliberately say nothing about it.
    """

    def test_a_clean_run_does_not_trip(self):
        eval_cli._check_model_credentials(ModelErrorAccumulator(), "out.json")

    def test_a_dominant_auth_failure_is_twenty(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator(ErrorKind.AUTH_FAILED),
                "out.json",
            )

        assert exc_info.value.code == ExitCode.AUTH_FAILED
        assert exc_info.value.code.needs_intervention

    def test_the_message_is_actionable_and_counts_the_failures(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator(ErrorKind.AUTH_FAILED, ErrorKind.AUTH_FAILED),
                "out.json",
            )

        message = exc_info.value.message
        assert "2 model calls" in message
        assert "API key" in message

    def test_the_report_path_rides_along(self):
        """Auth is checked after the write, so the footer can still name it."""
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(
                _accumulator(ErrorKind.AUTH_FAILED),
                "eval_20260101T000000.json",
            )

        assert exc_info.value.output_path == "eval_20260101T000000.json"

    @pytest.mark.parametrize(
        "kind",
        [
            ErrorKind.INTERNAL_ERROR,
            ErrorKind.UNKNOWN,
        ],
    )
    def test_an_unmapped_kind_is_left_to_the_catch_all(self, kind):
        """Only an unambiguous kind is mapped; the rest still exit 1.

        `UNREACHABLE`, `INVALID_CONFIG`, `QUOTA_EXCEEDED` and `TIMEOUT` are
        covered in their own test files (`test_wrong_url.py`,
        `test_invalid_config.py`, `test_rate_limited.py`, `test_timeout.py`)
        since `UNREACHABLE`/`QUOTA_EXCEEDED` each need a specific status, not
        just the kind, to exercise fully.
        """
        eval_cli._check_model_credentials(_accumulator(kind), "out.json")

    def test_unreachable_at_the_default_401_status_is_not_wrong_url(self):
        """`_accumulator`'s default status (401) is not a 404 — sanity check
        that `UNREACHABLE` isn't mapped by kind alone (see `test_wrong_url.py`
        for the full behavior)."""
        eval_cli._check_model_credentials(
            _accumulator(ErrorKind.UNREACHABLE), "out.json"
        )

    def test_an_isolated_auth_blip_is_not_the_dominant_kind(self):
        """One 401 among many unknowns does not get to name the run.

        Note what this does *not* say: the unknowns outrank it, not the calls
        that succeeded — the accumulator cannot see those. A run where the
        401s were the only failures but 997 prompts scored is held off this
        function entirely, by `_check_run_completed`.
        """
        accumulator = _accumulator(
            ErrorKind.UNKNOWN,
            ErrorKind.UNKNOWN,
            ErrorKind.UNKNOWN,
            ErrorKind.AUTH_FAILED,
        )

        eval_cli._check_model_credentials(accumulator, "out.json")

    def test_auth_wins_when_it_dominates_a_mixed_run(self):
        accumulator = _accumulator(
            ErrorKind.AUTH_FAILED,
            ErrorKind.AUTH_FAILED,
            ErrorKind.UNKNOWN,
        )

        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_model_credentials(accumulator, "out.json")

        assert exc_info.value.code == ExitCode.AUTH_FAILED

    def test_it_outranks_the_fail_under_gate(self):
        """20-39 beats 5 for a run with nothing to judge: "your key was
        rejected" is actionable where "the bar was missed" would be a verdict
        nobody measured. The two bands have to stay distinguishable for a
        wrapper to act on that difference at all.

        The precedence only applies when the run scored nothing — a run that
        *did* score is judged on its verdict, however many calls were
        rejected along the way. See
        `test_eval_fail_under.py::TestTheModelClassificationIsGatedOnTheVerdict`.
        """
        assert ExitCode.AUTH_FAILED.needs_intervention
        assert not ExitCode.FINDINGS_THRESHOLD.needs_intervention


class _Models:
    """The slice of `JsonRepository.models` the preflight consults."""

    def __init__(self, *models):
        self._by_id = {m.id: m for m in models}

    def get(self, model_id):
        return self._by_id[model_id]


def _store(*models):
    return SimpleNamespace(models=_Models(*models))


def _model(model_type, key=None, name="Target", model_id="m1"):
    return SimpleNamespace(id=model_id, name=name, type=model_type, key=key)


def _assignment(model_id="m1"):
    return SimpleNamespace(model_id=model_id)


class TestTheTargetPreflight:
    """A model under test with no key is caught before the run, not after.

    The accumulator the post-run check reads is only filled at the
    `Model.generate()` choke point, so a key that is simply *absent* never
    reaches it: the provider SDK refuses to build its client and the run used
    to exit 1. These cover the other half — refusing early, and not refusing
    anything that would have worked.
    """

    @pytest.fixture(autouse=True)
    def _no_ambient_credentials(self, monkeypatch):
        """Clear every credential the preflight reads.

        An autouse fixture rather than per-test `delenv` calls because the
        set is no longer just the two API-key names: a developer or CI runner
        with `GOOGLE_GENAI_USE_VERTEXAI` or `ANTHROPIC_AUTH_TOKEN` exported
        would otherwise silently turn the refusal tests green for the wrong
        reason. Each test sets back only what it is about.
        """
        for var in (
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_GENAI_USE_VERTEXAI",
            "ANTHROPIC_AUTH_TOKEN",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_a_keyless_gemini_model_is_twenty(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_target_credentials(
                _store(_model(ModelType.GEMINI)), [_assignment()]
            )

        assert exc_info.value.code == ExitCode.AUTH_FAILED
        assert "Target" in exc_info.value.message
        assert "GEMINI_API_KEY" in exc_info.value.message

    def test_nothing_has_run_so_no_path_is_reported(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_target_credentials(
                _store(_model(ModelType.GEMINI)), [_assignment()]
            )

        assert exc_info.value.output_path is None

    def test_an_empty_env_var_counts_as_missing(self, monkeypatch):
        """The reported case: `GEMINI_API_KEY=` resolves to `""`, which the
        adapter treats as no key at all rather than as a credential."""
        monkeypatch.setenv("GEMINI_API_KEY", "")

        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_target_credentials(
                _store(_model(ModelType.GEMINI, key="${GEMINI_API_KEY}")),
                [_assignment()],
            )

        assert exc_info.value.code == ExitCode.AUTH_FAILED

    def test_an_unset_referenced_var_is_also_twenty(self):
        """`resolve_env` raises rather than returning empty here, and the
        fix is the same one — so it must not fall through to the catch-all."""
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_target_credentials(
                _store(_model(ModelType.GEMINI, key="${GEMINI_API_KEY}")),
                [_assignment()],
            )

        assert exc_info.value.code == ExitCode.AUTH_FAILED
        assert "GEMINI_API_KEY" in exc_info.value.message

    def test_a_literal_key_passes(self):
        eval_cli._check_target_credentials(
            _store(_model(ModelType.GEMINI, key="sk-literal")), [_assignment()]
        )

    @pytest.mark.parametrize("var", ["GOOGLE_API_KEY", "GEMINI_API_KEY"])
    def test_either_provider_env_var_satisfies_gemini(self, monkeypatch, var):
        """The SDK reads both, so refusing on one alone would be wrong."""
        monkeypatch.setenv(var, "k")

        eval_cli._check_target_credentials(
            _store(_model(ModelType.GEMINI)), [_assignment()]
        )

    @pytest.mark.parametrize("value", ["true", "True", "1"], ids=["true", "True", "1"])
    def test_vertex_mode_satisfies_gemini_with_no_key(self, monkeypatch, value):
        """The regression: a Gemini target on Vertex ADC needs no API key.

        `model_helpers._create_inner_model` omits `api_key=` when the model's
        key is empty, and google-genai's Vertex branch then authenticates from
        ADC — the "No API key was provided" raise is in the Developer-API
        branch it never reaches. Refusing here was a hard stop, with no
        override, on a configuration that runs.
        """
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", value)

        eval_cli._check_target_credentials(
            _store(_model(ModelType.GEMINI)), [_assignment()]
        )

    @pytest.mark.parametrize("value", ["false", "0", "", "yes"])
    def test_vertex_mode_off_leaves_gemini_refused(self, monkeypatch, value):
        """The SDK reads only `true`/`1`, so neither may anything else here —
        counting `false` as a credential would turn the refusal off for the
        people most likely to have the variable set."""
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", value)

        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_target_credentials(
                _store(_model(ModelType.GEMINI)), [_assignment()]
            )

        assert exc_info.value.code == ExitCode.AUTH_FAILED

    def test_application_credentials_alone_do_not_satisfy_gemini(self, monkeypatch):
        """Without the Vertex flag the client stays in Developer-API mode and
        raises, so this must keep failing. The scanner's preflight answers the
        same way for the same reason — it builds that client through ADK."""
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/sa.json")

        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_target_credentials(
                _store(_model(ModelType.GEMINI)), [_assignment()]
            )

        assert exc_info.value.code == ExitCode.AUTH_FAILED

    def test_an_auth_token_satisfies_anthropic(self, monkeypatch):
        """The SDK turns it into an `Authorization: Bearer` header, which is
        how a target behind a gateway is configured."""
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "t")

        eval_cli._check_target_credentials(
            _store(_model(ModelType.ANTHROPIC)), [_assignment()]
        )

    def test_an_anthropic_model_with_neither_credential_is_twenty(self):
        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_target_credentials(
                _store(_model(ModelType.ANTHROPIC)), [_assignment()]
            )

        assert exc_info.value.code == ExitCode.AUTH_FAILED

    def test_openai_has_no_keyless_path(self, monkeypatch):
        """`AsyncOpenAI()` raises when no key resolves, so neither env var
        above may leak across into the OpenAI branch."""
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "t")

        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_target_credentials(
                _store(_model(ModelType.OPENAI)), [_assignment()]
            )

        assert exc_info.value.code == ExitCode.AUTH_FAILED

    @pytest.mark.parametrize(
        "model_type",
        [ModelType.OLLAMA, ModelType.LITELLM, ModelType.GENERIC_HTTP],
        ids=["ollama", "litellm", "generic_http"],
    )
    def test_a_type_with_no_key_to_miss_is_not_an_auth_failure(self, model_type):
        """Turning these into exit 20 would break every local-model run."""
        eval_cli._check_target_credentials(_store(_model(model_type)), [_assignment()])

    def test_a_model_the_config_does_not_define_is_left_to_the_run(self):
        """The run loop raises on it with the context to say which
        assignment referenced it; guessing here would say less."""
        eval_cli._check_target_credentials(_store(), [_assignment("missing")])

    def test_every_distinct_model_is_checked(self, monkeypatch):
        """A keyed model behind a keyless one must still be caught."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")

        store = _store(
            _model(ModelType.OPENAI, model_id="ok", name="Fine"),
            _model(ModelType.GEMINI, model_id="bad", name="Broken"),
        )

        with pytest.raises(CliExit) as exc_info:
            eval_cli._check_target_credentials(
                store, [_assignment("ok"), _assignment("bad")]
            )

        assert "Broken" in exc_info.value.message


class TestTheJudgeCredential:
    """A missing judge/generator key is exit 20 too, in its own words.

    It cannot be preflighted alongside the target: whether a run needs a judge
    at all depends on the evaluation and detector types, and duplicating those
    two dispatch tables outside `eval_helpers`/`detector_helpers` would drift.
    So the classification rides on the typed error, raised where the judge is
    actually built — which fires for exactly the runs that need one.
    """

    def _store(self):
        store = MagicMock()
        store.evaluations.get.return_value = MagicMock(name="an evaluation")
        store.models.get_model.return_value = MagicMock()
        return store

    @pytest.mark.asyncio
    async def test_a_judge_built_at_construction_is_exit_20(self):
        """Adversarial probes and LLM-as-judge detectors build theirs in
        `create_evaluation`/`create_detector`, before anything runs."""
        store = self._store()
        store.evaluations.get_evaluation.side_effect = GeneratorCredentialError(
            "The judge/generator model has no API key configured."
        )

        with pytest.raises(CliExit) as exc_info:
            await runner.run_cli_evaluation(
                MagicMock(evaluation_id="ev-1", model_id="model-1"),
                store,
                concurrency=1,
                progress=MagicMock(),
                desc_width=20,
            )

        assert exc_info.value.code == ExitCode.AUTH_FAILED
        assert exc_info.value.code.needs_intervention
        assert "judge/generator" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_a_judge_built_lazily_during_init_is_exit_20(self):
        """The metric evaluations defer theirs to first use, so it must not
        fall into the swallow-and-continue per-prompt handler."""
        store = self._store()
        evaluation = MagicMock()
        evaluation.init = AsyncMock(
            side_effect=GeneratorCredentialError(
                "The judge/generator model has no API key configured."
            )
        )
        store.evaluations.get_evaluation.return_value = evaluation

        with pytest.raises(CliExit) as exc_info:
            await runner.run_cli_evaluation(
                MagicMock(evaluation_id="ev-1", model_id="model-1"),
                store,
                concurrency=1,
                progress=MagicMock(),
                desc_width=20,
            )

        assert exc_info.value.code == ExitCode.AUTH_FAILED

    @pytest.mark.asyncio
    async def test_nothing_has_run_so_no_path_is_reported(self):
        store = self._store()
        store.evaluations.get_evaluation.side_effect = GeneratorCredentialError(
            "The judge/generator model has no API key configured."
        )

        with pytest.raises(CliExit) as exc_info:
            await runner.run_cli_evaluation(
                MagicMock(evaluation_id="ev-1", model_id="model-1"),
                store,
                concurrency=1,
                progress=MagicMock(),
                desc_width=20,
            )

        assert exc_info.value.output_path is None


def _report(error_reason: str | None = None, findings: list | None = None):
    agentic_trace = {}
    if error_reason:
        agentic_trace = {"exit_reason": "error", "error_reason": error_reason}
    return SimpleNamespace(
        findings=findings or [],
        pre_triage_findings=None,
        category_summary={},
        framework_detected="openai",
        scan_metadata={"agentic_trace": agentic_trace},
    )


class TestScanCredentials:
    def test_a_clean_scan_does_not_trip(self):
        scan_cli._check_llm_credentials(_report(), "scan.json")

    def test_a_rejected_key_is_twenty(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(_report("auth"), "scan.json")

        assert exc_info.value.code == ExitCode.AUTH_FAILED

    def test_the_message_says_the_findings_are_partial(self):
        """The user must know the static layer is all they got."""
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(_report("auth"), "scan.json")

        assert "static-analysis only" in exc_info.value.message

    def test_a_model_error_is_not_an_auth_error(self):
        """The reasoner distinguishes them; 'invalid model' is not exit 20.

        It is exit 25 instead — see `test_invalid_config.py::TestScanModel`.
        """
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(_report("model"), "scan.json")

        assert exc_info.value.code != ExitCode.AUTH_FAILED

    @pytest.mark.parametrize(
        "metadata",
        [{}, {"agentic_trace": None}, {"agentic_trace": {}}],
        ids=["no-trace", "null-trace", "empty-trace"],
    )
    def test_a_report_without_a_trace_is_tolerated(self, metadata):
        """A static-only scan never ran the reasoner, so there is no trace."""
        report = SimpleNamespace(findings=[], scan_metadata=metadata)

        scan_cli._check_llm_credentials(report, "scan.json")

    def test_absent_metadata_is_tolerated(self):
        report = SimpleNamespace(findings=[], scan_metadata=None)

        scan_cli._check_llm_credentials(report, "scan.json")

    def test_the_report_path_rides_along(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_llm_credentials(
                _report("auth"),
                "scan_20260101T000000.json",
            )

        assert exc_info.value.output_path == "scan_20260101T000000.json"


class TestThePreflight:
    """A missing key is caught before the scan runs, not after."""

    def test_a_configured_model_without_a_key_refuses(self):
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_configured_model("openai:gpt-5", lambda _: False)

        assert exc_info.value.code == ExitCode.AUTH_FAILED
        assert "openai:gpt-5" in exc_info.value.message

    def test_nothing_is_written_so_no_path_is_reported(self):
        """Unlike the other stops, this one precedes the report entirely."""
        with pytest.raises(CliExit) as exc_info:
            scan_cli._check_configured_model("openai:gpt-5", lambda _: False)

        assert exc_info.value.output_path is None

    def test_a_configured_model_with_a_key_passes(self):
        scan_cli._check_configured_model("openai:gpt-5", lambda _: True)

    @pytest.mark.parametrize("model", [None, ""], ids=["unset", "empty"])
    def test_no_model_configured_is_not_an_auth_failure(self, model):
        """Static-only is a legitimate way to run; never turn it into exit 20.

        The key check is not even consulted — asking about the credential for
        a model nobody configured is how a static-only run would break.
        """
        scan_cli._check_configured_model(model, _never_called)


def _never_called(model):
    raise AssertionError(f"api_key_present should not be consulted for {model!r}")


@pytest.fixture
def stubbed_scan(monkeypatch, tmp_path):
    """Drive the real `handle_scan` with the scanner itself stubbed out.

    The ordering under test is `write_output` → auth → gate, which no unit
    test of the individual checks can observe.
    """

    def _configure(report, *, fail_on=None, written=None):
        target = tmp_path / "agent.py"
        target.write_text("import openai\n")

        calls = {"run_core": 0}

        def _run_core(*args, **kwargs):
            calls["run_core"] += 1
            return report

        monkeypatch.setattr(
            "switch_trust.scan.agent_scanner.run_core",
            _run_core,
        )
        monkeypatch.setattr(scan_cli, "print_report", lambda _: None)

        def _write(report_, output_path, fmt="json"):
            with open(output_path, "w") as handle:
                handle.write("{}")
            if written is not None:
                written.append(output_path)

        monkeypatch.setattr(scan_cli, "write_output", _write)
        monkeypatch.delenv("GENERATOR_MODEL", raising=False)

        args = argparse.Namespace(
            path=str(target),
            output=str(tmp_path / "scan.json"),
            format="json",
            fail_on=fail_on,
        )
        return args, calls

    return _configure


class TestHandleScanEndToEnd:
    def test_a_clean_scan_returns_its_output_path(self, stubbed_scan):
        args, calls = stubbed_scan(_report())

        assert scan_cli.handle_scan(args) == args.output
        assert calls["run_core"] == 1

    def test_a_missing_key_refuses_before_the_scanner_runs(
        self,
        stubbed_scan,
        monkeypatch,
    ):
        """The preflight's whole point: no model call, no minutes of CI."""
        args, calls = stubbed_scan(_report())
        monkeypatch.setenv("GENERATOR_MODEL", "openai:gpt-5")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(CliExit) as exc_info:
            scan_cli.handle_scan(args)

        assert exc_info.value.code == ExitCode.AUTH_FAILED
        assert calls["run_core"] == 0

    def test_a_rejected_key_raises_twenty_and_keeps_the_report(
        self,
        stubbed_scan,
        tmp_path,
    ):
        written: list[str] = []
        args, _ = stubbed_scan(_report("auth"), written=written)

        with pytest.raises(CliExit) as exc_info:
            scan_cli.handle_scan(args)

        assert exc_info.value.code == ExitCode.AUTH_FAILED
        assert written == [args.output]
        assert (tmp_path / "scan.json").exists()

    def test_a_tripped_gate_raises_five_and_keeps_the_report(
        self,
        stubbed_scan,
        tmp_path,
    ):
        """Exit 5 reaching `main` from a real `handle_scan`, not a helper."""
        written: list[str] = []
        args, _ = stubbed_scan(
            _report(findings=[SimpleNamespace(ai_spm_severity="Critical")]),
            fail_on="high",
            written=written,
        )

        with pytest.raises(CliExit) as exc_info:
            scan_cli.handle_scan(args)

        assert exc_info.value.code == ExitCode.FINDINGS_THRESHOLD
        assert written == [args.output]
        assert (tmp_path / "scan.json").exists()

    def test_auth_outranks_the_gate_when_both_apply(self, stubbed_scan):
        """20-39 before 5, as the precedence table says.

        Findings counted from a scan whose reasoning layer never ran are not
        the verdict the gate was asked to judge, so reporting 5 here would
        send the user to the wrong problem.
        """
        args, _ = stubbed_scan(
            _report("auth", findings=[SimpleNamespace(ai_spm_severity="Critical")]),
            fail_on="high",
        )

        with pytest.raises(CliExit) as exc_info:
            scan_cli.handle_scan(args)

        assert exc_info.value.code == ExitCode.AUTH_FAILED
