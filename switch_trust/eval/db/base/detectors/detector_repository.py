"""
Abstract interface for a repository of Detectors.

A DetectorRepository manages detector configurations,
providing CRUD operations over them.  Detectors are
always referenced by ID.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from switch_trust.eval.core.detectors.detector import Detector
from switch_trust.eval.db.base.detectors.detector_types import (
    DbDetector,
    DetectorListView,
    DetectorSortOrder,
)


class DetectorRepository(ABC):
    """Repository that manages Detector configurations."""

    @abstractmethod
    def list(self) -> list[DbDetector]:
        """Return all detector configurations in the repository."""
        pass

    @abstractmethod
    def search(
        self,
        query: str | None = None,
        types: list[str] | None = None,
        order: DetectorSortOrder | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> DetectorListView:
        """Search detectors with pagination and sorting."""
        pass

    @abstractmethod
    def get(self, id: str) -> DbDetector:
        """Return the detector configuration with the given id.

        Raises KeyError if the detector does not exist.
        """
        pass

    @abstractmethod
    def create(
        self,
        config: DbDetector,
    ) -> DbDetector:
        """Persist a new detector configuration.

        Raises ValueError if a detector with that name already
        exists.
        """
        pass

    @abstractmethod
    def update(
        self,
        config: DbDetector,
    ) -> DbDetector:
        """Replace an existing detector configuration.

        The detector is identified by ``config.id``.

        Raises KeyError if the detector does not exist.
        """
        pass

    def get_detector(self, id: str) -> Detector:
        """Return a Detector instance for the given id.

        Raises KeyError if the detector does not exist.
        """
        from switch_trust.eval.db.base.detectors.detector_helpers import (  # noqa: PLC0415 - deferred import cost
            create_detector,
        )

        return create_detector(self.get(id))

    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete the detector with the given id.

        Raises KeyError if the detector does not exist.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all detectors from the repository."""
        pass
