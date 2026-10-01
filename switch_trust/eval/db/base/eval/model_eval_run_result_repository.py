"""
Abstract interface for a repository of ModelEvaluationRunResults.

A ModelEvaluationRunResultRepository manages the leaf-evaluation
results that belong to a model-evaluation run.
"""

from abc import ABC, abstractmethod

from switch_trust.eval.db.base.eval.model_eval_run_result_types import (
    DbModelEvaluationRunResult,
)


class ModelEvaluationRunResultRepository(ABC):
    """Repository that manages ModelEvaluationRunResult records."""

    @abstractmethod
    def create_batch(
        self,
        results: list[DbModelEvaluationRunResult],
    ) -> None:
        """Persist a batch of results for a run."""
        pass

    @abstractmethod
    def get_by_run_id(
        self,
        run_id: str,
    ) -> list[DbModelEvaluationRunResult]:
        """Return all results for a given run."""
        pass

    @abstractmethod
    def delete_by_run_id(self, run_id: str) -> None:
        """Remove all results for a given run."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all results."""
        pass
