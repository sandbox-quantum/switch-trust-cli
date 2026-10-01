"""
Abstract interface for a repository of Models.

A ModelRepository manages model configurations,
providing CRUD operations over them.  Models are
always referenced by ID.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from switch_trust.eval.core.models.model import Model
from switch_trust.eval.db.base.models.model_types import (
    DbModel,
    DbModelListView,
    ModelSortOrder,
)


class ModelRepository(ABC):
    """Repository that manages Model configurations."""

    @abstractmethod
    def list(self) -> list[DbModel]:
        """Return all model configurations in the repository."""
        pass

    @abstractmethod
    def search(
        self,
        query: str | None = None,
        types: list[str] | None = None,
        order: ModelSortOrder | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> DbModelListView:
        """Search models by name with offset-based pagination.

        Args:
            query: Text to match against the model name
                (case-insensitive substring match).
            order: Sort field and direction.
            offset: Number of items to skip.
            limit: Maximum number of items to return.
        """
        pass

    @abstractmethod
    def get(self, id: str) -> DbModel:
        """Return the model configuration with the given id.

        Raises KeyError if the model does not exist.
        """
        pass

    @abstractmethod
    def create(
        self,
        config: DbModel,
    ) -> DbModel:
        """Persist a new model configuration.

        Raises ValueError if a model with that name already
        exists.
        """
        pass

    @abstractmethod
    def update(
        self,
        config: DbModel,
    ) -> DbModel:
        """Replace an existing model configuration.

        The model is identified by ``config.id``.

        Raises KeyError if the model does not exist.
        """
        pass

    def get_model(self, id: str) -> Model:
        """Return a Model instance for the given id.

        Raises KeyError if the model does not exist.
        """
        from switch_trust.eval.db.base.models.model_helpers import (  # noqa: PLC0415 - deferred import cost
            create_model,
        )

        return create_model(self.get(id))

    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete the model with the given id.

        Raises KeyError if the model does not exist.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all models from the repository."""
        pass
