from switch_trust.cli.runner import CliRunResult, _aggregate_summary
from switch_trust.eval.core.eval.evaluation import EvaluationStatus, EvaluationSummary


def _make_run(
    summary: EvaluationSummary,
) -> CliRunResult:
    return CliRunResult(
        model_evaluation_id="me-1",
        model_evaluation_name="test",
        summary=summary,
    )


def _make_summary(
    status: EvaluationStatus = EvaluationStatus.FINISHED,
    total: int = 5,
    finished: int = 5,
    errors: int = 0,
    max_score: float = 5.0,
    achieved: float = 3.0,
    error_messages: list[str] | None = None,
) -> EvaluationSummary:
    return EvaluationSummary(
        status=status,
        total_evaluations=total,
        finished_evaluations=finished,
        error_evaluations=errors,
        max_score=max_score,
        achieved_score=achieved,
        error_messages=error_messages or [],
    )


class TestAggregateScoreExcludesErrors:
    def test_errored_evaluation_excluded_from_score(self):
        runs = [
            _make_run(_make_summary(achieved=4.0, max_score=5.0)),
            _make_run(
                _make_summary(
                    status=EvaluationStatus.ERROR,
                    achieved=0.0,
                    max_score=5.0,
                )
            ),
        ]
        overall = _aggregate_summary(runs)
        assert overall.score == 4.0 / 5.0

    def test_all_succeeded_includes_all_in_score(self):
        runs = [
            _make_run(_make_summary(achieved=5.0, max_score=5.0)),
            _make_run(_make_summary(achieved=3.0, max_score=3.0)),
        ]
        overall = _aggregate_summary(runs)
        assert overall.score == 1.0

    def test_all_errored_no_score(self):
        runs = [
            _make_run(
                _make_summary(
                    status=EvaluationStatus.ERROR,
                    max_score=5.0,
                    achieved=0.0,
                )
            ),
        ]
        overall = _aggregate_summary(runs)
        assert overall.score is None
        assert overall.status == EvaluationStatus.ERROR

    def test_status_finished_when_some_errored(self):
        runs = [
            _make_run(_make_summary()),
            _make_run(_make_summary(status=EvaluationStatus.ERROR)),
        ]
        overall = _aggregate_summary(runs)
        assert overall.status == EvaluationStatus.FINISHED
