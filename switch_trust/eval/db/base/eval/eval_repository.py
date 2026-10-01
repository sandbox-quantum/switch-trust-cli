"""
Abstract interface for a repository of Evaluations.

An EvaluationRepository manages evaluation configurations,
providing CRUD operations over them.  Evaluations are
always referenced by ID.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from switch_trust.eval.core.eval.evaluation import Evaluation
from switch_trust.eval.db.base.detectors.detector_repository import DetectorRepository
from switch_trust.eval.db.base.eval.eval_types import (
    DbEvaluation,
    EvaluationListView,
    EvaluationSortOrder,
)
from switch_trust.eval.db.base.message.message_collection_repository import (
    MessageCollectionRepository,
)


class EvaluationRepository(ABC):
    """Repository that manages Evaluation configurations."""

    @abstractmethod
    def list(self) -> list[DbEvaluation]:
        """Return all evaluation configurations in the repository."""
        pass

    @abstractmethod
    def search(
        self,
        query: str | None = None,
        types: list[str] | None = None,
        order: EvaluationSortOrder | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> EvaluationListView:
        """Search evaluations by name with offset-based pagination.

        Args:
            query: Text to match against the evaluation name
                (case-insensitive substring match).
            order: Sort field and direction.
            offset: Number of items to skip.
            limit: Maximum number of items to return.
        """
        pass

    @abstractmethod
    def get(self, id: str) -> DbEvaluation:
        """Return the evaluation configuration with the given id.

        Raises KeyError if the evaluation does not exist.
        """
        pass

    @abstractmethod
    def create(
        self,
        config: DbEvaluation,
    ) -> DbEvaluation:
        """Persist a new evaluation configuration.

        Raises ValueError if an evaluation with that name already
        exists.
        """
        pass

    @abstractmethod
    def update(
        self,
        config: DbEvaluation,
    ) -> DbEvaluation:
        """Replace an existing evaluation configuration.

        The evaluation is identified by ``config.id``.

        Raises KeyError if the evaluation does not exist.
        """
        pass

    def get_evaluation(
        self,
        id: str,
        message_collection_repo: MessageCollectionRepository | None = None,
        detector_repo: DetectorRepository | None = None,
    ) -> Evaluation:
        """Return an Evaluation instance for the given id.

        Depending on the evaluation type, additional repositories
        may be required to resolve referenced objects.

        Raises KeyError if the evaluation does not exist.
        """
        from switch_trust.eval.db.base.eval.eval_helpers import (  # noqa: PLC0415 - deferred import cost
            create_evaluation,
        )

        return create_evaluation(
            self.get(id),
            message_collection_repo=message_collection_repo,
            detector_repo=detector_repo,
        )

    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete the evaluation with the given id.

        Raises KeyError if the evaluation does not exist.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all evaluations from the repository."""
        pass
