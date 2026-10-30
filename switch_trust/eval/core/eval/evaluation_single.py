import logging
from abc import abstractmethod
from dataclasses import dataclass

from dataclasses_json import dataclass_json

from switch_trust.eval.common.schema import Session
from switch_trust.eval.core.eval.evaluation import (
    DEFAULT_MIN_SUCCESS_RATE,
    Evaluation,
    EvaluationResult,
    EvaluationStatus,
    EvaluationSummary,
)
from switch_trust.eval.core.models.model import Model, record_model_error
from switch_trust.eval.core.models.model_errors import (
    classify_model_error,
    is_generator_failure,
    log_level_for_run_failure,
)

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class SingleEvaluation(Evaluation):
    score: float = 0.0
    session: Session | None = None

    def __init__(self):
        super().__init__()
        self.score = 0.0
        self.session = None

    @abstractmethod
    async def init(self):
        pass

    def get_summary(self) -> EvaluationSummary:
        return EvaluationSummary(
            status=self.status,
            total_evaluations=1,
            finished_evaluations=1 if self.status == EvaluationStatus.FINISHED else 0,
            error_evaluations=1 if self.status == EvaluationStatus.ERROR else 0,
            # An errored prompt is a measurement failure, not a zero result:
            # contribute nothing to either side of the mean so it drops out of
            # the score instead of dragging it down.
            max_score=0.0 if self.status == EvaluationStatus.ERROR else 1.0,
            achieved_score=self.score
            if self.status == EvaluationStatus.FINISHED
            else 0.0,
            error_messages=[self.error_message] if self.error_message else [],
        )

    async def run(
        self,
        model: Model,
        concurrency: int = 50,
        min_success_rate: float = DEFAULT_MIN_SUCCESS_RATE,
        max_timeout: int | None = None,
    ):
        # min_success_rate and max_timeout are unused for a leaf; kept to match
        # the base signature.
        await self.execute(model)

    async def execute(self, model: Model):
        if self.status == EvaluationStatus.ERROR:
            return
        self.status = EvaluationStatus.RUNNING
        name = type(self).__name__

        try:
            self.score = await self.execute_internal(model)
            self.status = EvaluationStatus.FINISHED
            logger.debug("%s finished: score=%.2f", name, self.score)
        except Exception as e:
            self.error_message = str(e)
            self.status = EvaluationStatus.ERROR
            if is_generator_failure(e):
                # A judge/generator failure raised after its own generate()
                # call already returned (e.g. an unparseable verdict) never
                # passes through Model.generate()'s own accumulator recording.
                record_model_error(
                    e,
                    "judge",
                    getattr(e, "status", None),
                    getattr(e, "error_status", None),
                )
            # Only a confidently tenant-caused kind should page as a warning;
            # this scope also wraps our own detector/scoring code, so an
            # unclassifiable failure defaults to alerting.
            logger.log(
                log_level_for_run_failure(classify_model_error(e)),
                "%s failed (%s: %s)",
                name,
                type(e).__name__,
                e,
                exc_info=True,
            )
        finally:
            self._notify_observers()

    def get_results(self) -> list[EvaluationResult]:
        return [
            EvaluationResult(
                score=self.score,
                status=self.status,
                error_message=self.error_message,
                session=self.session,
            )
        ]

    @abstractmethod
    async def execute_internal(self, model: Model) -> float:
        pass
