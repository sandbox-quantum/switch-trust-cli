"""
Standalone function to run a model evaluation end-to-end.

Looks up the DbModelEvaluation, creates the core Evaluation and
Model, creates a DbModelEvaluationRun, and executes the evaluation
with an observer that persists progress back to the database.
"""

import logging

from switch_trust.eval.common.utils import now_utc
from switch_trust.eval.core.eval.evaluation import Evaluation, EvaluationStatus
from switch_trust.eval.db.base.detectors.detector_repository import DetectorRepository
from switch_trust.eval.db.base.eval.eval_repository import EvaluationRepository
from switch_trust.eval.db.base.eval.model_eval_repository import (
    ModelEvaluationRepository,
)
from switch_trust.eval.db.base.eval.model_eval_run_repository import (
    ModelEvaluationRunRepository,
)
from switch_trust.eval.db.base.eval.model_eval_run_result_repository import (
    ModelEvaluationRunResultRepository,
)
from switch_trust.eval.db.base.eval.model_eval_run_result_types import (
    DbModelEvaluationRunResult,
)
from switch_trust.eval.db.base.eval.model_eval_run_types import DbModelEvaluationRun
from switch_trust.eval.db.base.message.message_collection_repository import (
    MessageCollectionRepository,
)
from switch_trust.eval.db.base.models.model_repository import ModelRepository

logger = logging.getLogger(__name__)


async def run_model_evaluation(
    model_evaluation_id: str,
    concurrency: int,
    model_evaluation_repo: ModelEvaluationRepository,
    evaluation_repo: EvaluationRepository,
    model_repo: ModelRepository,
    run_repo: ModelEvaluationRunRepository,
    message_collection_repo: MessageCollectionRepository | None = None,
    detector_repo: DetectorRepository | None = None,
    result_repo: ModelEvaluationRunResultRepository | None = None,
    run: DbModelEvaluationRun | None = None,
) -> DbModelEvaluationRun:
    """Run a model evaluation and persist progress to the database.

    1. Looks up the DbModelEvaluation by ID
    2. Creates the core Evaluation and Model from their configs
    3. Creates a new DbModelEvaluationRun and persists it
    4. Initialises and runs the evaluation with an observer
       that writes the summary back after each progress update
    5. Returns the completed run

    Args:
        model_evaluation_id: ID of the DbModelEvaluation to run.
        concurrency: Max concurrent evaluation tasks.
        model_evaluation_repo: Repository for DbModelEvaluation.
        evaluation_repo: Repository for DbEvaluation.
        model_repo: Repository for DbModel.
        run_repo: Repository for DbModelEvaluationRun.
        message_collection_repo: Required for MESSAGE_COLLECTION
            evaluation types.
        detector_repo: Required for evaluations that use detectors.
    """
    model_evaluation = model_evaluation_repo.get(
        model_evaluation_id,
    )

    evaluation = evaluation_repo.get_evaluation(
        model_evaluation.evaluation_id,
        message_collection_repo=message_collection_repo,
        detector_repo=detector_repo,
    )
    model = model_repo.get_model(model_evaluation.model_id)

    if run is None:
        run = DbModelEvaluationRun(
            model_evaluation_id=model_evaluation_id,
            started=now_utc(),
        )
        run_repo.create(run)

    logger.info(
        "Run %s started for model evaluation %s",
        run.id,
        model_evaluation_id,
    )

    def _on_progress(e: Evaluation) -> None:
        summary = e.get_summary()
        run.summary = summary
        if summary.status in (
            EvaluationStatus.FINISHED,
            EvaluationStatus.ERROR,
        ):
            run.finished = now_utc()
        run_repo.update(run)
        logger.info(
            "Run %s: %s (%.0f%%)",
            run.id,
            summary.status.value,
            summary.progress * 100,
        )

    evaluation.add_observer(_on_progress)
    try:
        await evaluation.init()
        await evaluation.run(model, concurrency)
    except Exception as e:
        logger.error("Run %s failed: %s", run.id, e)
    finally:
        evaluation.remove_observer(_on_progress)

    run.summary = evaluation.get_summary()
    run.finished = now_utc()
    run_repo.update(run)

    if result_repo is not None:
        core_results = evaluation.get_results()
        db_results = [
            DbModelEvaluationRunResult(
                run_id=run.id,
                score=r.score,
                status=r.status,
                error_message=r.error_message,
                session=r.session,
            )
            for r in core_results
        ]
        result_repo.create_batch(db_results)
        logger.info(
            "Run %s: stored %d results",
            run.id,
            len(db_results),
        )

    logger.info("Run %s finished", run.id)

    return run
