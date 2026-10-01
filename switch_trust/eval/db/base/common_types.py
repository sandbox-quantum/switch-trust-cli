"""Types shared by every repository's `*_types` module.

Each entity keeps its own ``*SortField`` (the column set differs per table) and
its own ``*SortOrder`` default, but the direction is the same two values
everywhere, so it lives here rather than being restated per module.

The per-entity modules re-export what they use, so
``from ...detectors.detector_types import SortDirection`` keeps working.
"""

from enum import Enum


class SortDirection(str, Enum):
    ASC = "asc"
    DESC = "desc"
