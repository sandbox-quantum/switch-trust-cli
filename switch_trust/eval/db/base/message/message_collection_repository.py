"""
Abstract interface for a repository of MessageCollections.

A MessageCollectionRepository manages multiple collections,
providing CRUD operations over them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from switch_trust.eval.core.message.message_collection import MessageCollection
from switch_trust.eval.db.base.message.message_collection_types import (
    DbMessageCollection,
    MessageCollectionListView,
    MessageCollectionSortOrder,
)


class MessageCollectionRepository(ABC):
    """Repository that manages multiple MessageCollections."""

    @abstractmethod
    def list(self) -> list[DbMessageCollection]:
        """Return all collections in the repository."""
        pass

    @abstractmethod
    def search(
        self,
        query: str | None = None,
        types: list[str] | None = None,
        order: MessageCollectionSortOrder | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> MessageCollectionListView:
        """Search collections with pagination and sorting."""
        pass

    @abstractmethod
    def get(self, id: str) -> DbMessageCollection:
        """Return the collection descriptor with the given id.

        Raises KeyError if the collection does not exist.
        """
        pass

    @abstractmethod
    def get_message_collection(
        self,
        id: str,
    ) -> MessageCollection:
        """Return a MessageCollection instance for the given id.

        Raises KeyError if the collection does not exist.
        """
        pass

    @abstractmethod
    def create(
        self,
        db_collection: DbMessageCollection,
    ) -> DbMessageCollection:
        """Persist a new collection.

        Raises ValueError if a collection with that name already
        exists.
        """
        pass

    @abstractmethod
    def update(
        self,
        db_collection: DbMessageCollection,
    ) -> DbMessageCollection:
        """Replace an existing collection.

        The collection is identified by ``db_collection.id``.

        Raises KeyError if the collection does not exist.
        """
        pass

    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete the collection with the given id.

        Raises KeyError if the collection does not exist.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all collections from the repository."""
        pass
