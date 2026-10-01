"""
Abstract interface for a repository of ModelEvaluationRuns.

A ModelEvaluationRunRepository manages runs of model-evaluation
assignments, providing create and search operations.
"""

from abc import ABC, abstractmethod

from switch_trust.eval.db.base.eval.model_eval_run_types import (
    DbModelEvaluationRun,
    ModelEvaluationRunListView,
    ModelEvaluationRunSortOrder,
)


class ModelEvaluationRunRepository(ABC):
    """Repository that manages ModelEvaluationRun records."""

    @abstractmethod
    def create(
        self,
        run: DbModelEvaluationRun,
    ) -> DbModelEvaluationRun:
        """Persist a new run."""
        pass

    @abstractmethod
    def get(self, id: str) -> DbModelEvaluationRun:
        """Return the run with the given id.

        Raises KeyError if it does not exist.
        """
        pass

    @abstractmethod
    def update(
        self,
        run: DbModelEvaluationRun,
    ) -> DbModelEvaluationRun:
        """Update the summary and finished timestamp of a run.

        The run is identified by ``run.id``.

        Raises KeyError if the run does not exist.
        """
        pass

    @abstractmethod
    def search(
        self,
        model_evaluation_id: str | None = None,
        model_id: str | None = None,
        evaluation_id: str | None = None,
        order: ModelEvaluationRunSortOrder | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> ModelEvaluationRunListView:
        """Search runs with optional filter and pagination.

        Args:
            model_evaluation_id: Filter by model-evaluation ID.
            order: Sort field and direction.
            offset: Number of items to skip.
            limit: Maximum number of items to return.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all runs."""
        pass
