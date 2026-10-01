from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dataclasses_json import dataclass_json

from switch_trust.eval.common.reference import Reference, ReferenceType
from switch_trust.eval.common.utils import datetime_config, generate_id, now_utc
from switch_trust.eval.db.base.common_types import SortDirection


class DetectorType(str, Enum):
    """Discriminator for the kind of Detector."""

    GARAK = "garak"
    MODEL = "model"
    ADVERSARIAL_MODEL = "adversarial_model"
    PII = "pii"
    SECRET = "secret"
    TOPIC_GUARD = "topic_guard"


@dataclass_json
@dataclass
class DbDetector:
    """JSON-serialisable description of a Detector."""

    type: DetectorType
    name: str
    id: str = field(default_factory=generate_id)
    created: datetime = field(
        default_factory=now_utc,
        metadata=datetime_config,
    )
    description: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    include_conversation: bool = False

    # -- garak type --
    detector_name: str | None = None

    # -- model type --
    prompt: str | None = None

    # -- topic_guard type --
    agent_objective: str | None = None
    agent_instructions: str | None = None

    def get_ref(self) -> Reference:
        return Reference(
            self.id,
            self.name,
            ReferenceType.DETECTOR,
            self.description,
        )


class DetectorSortField(str, Enum):
    NAME = "name"
    CREATED = "created"
    TYPE = "type"


@dataclass_json
@dataclass
class DetectorSortOrder:
    field: DetectorSortField = DetectorSortField.NAME
    direction: SortDirection = SortDirection.ASC


@dataclass_json
@dataclass
class DetectorListView:
    """Paginated list of detector configurations."""

    items: list[DbDetector] = field(default_factory=list)
    total: int = 0
