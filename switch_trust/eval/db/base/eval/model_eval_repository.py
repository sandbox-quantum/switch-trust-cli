"""
Abstract interface for a repository of ModelEvaluations.

A ModelEvaluationRepository manages the assignments of
Evaluations to Models.
"""

from abc import ABC, abstractmethod

from switch_trust.eval.db.base.eval.model_eval_types import (
    DbModelEvaluation,
    ModelEvaluationListView,
    ModelEvaluationSortOrder,
)


class ModelEvaluationRepository(ABC):
    """Repository that manages ModelEvaluation assignments."""

    @abstractmethod
    def list_by_model(
        self,
        model_id: str,
        order: ModelEvaluationSortOrder | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> ModelEvaluationListView:
        """Return assignments for a given model with pagination.

        Args:
            model_id: The model to list assignments for.
            order: Sort field and direction.
            offset: Number of items to skip.
            limit: Maximum number of items to return.
        """
        pass

    @abstractmethod
    def get(self, id: str) -> DbModelEvaluation:
        """Return the assignment with the given id.

        Raises KeyError if it does not exist.
        """
        pass

    @abstractmethod
    def create(
        self,
        config: DbModelEvaluation,
    ) -> DbModelEvaluation:
        """Persist a new model-evaluation assignment.

        Raises ValueError if an assignment with that name
        already exists.
        """
        pass

    @abstractmethod
    def list_by_evaluation(
        self,
        evaluation_id: str,
        order: ModelEvaluationSortOrder | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> ModelEvaluationListView:
        """Return assignments for a given evaluation with pagination.

        Args:
            evaluation_id: The evaluation to list assignments for.
            order: Sort field and direction.
            offset: Number of items to skip.
            limit: Maximum number of items to return.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all model-evaluation assignments."""
        pass
