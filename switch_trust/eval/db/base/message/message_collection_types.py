from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dataclasses_json import dataclass_json

from switch_trust.eval.common.reference import Reference, ReferenceType
from switch_trust.eval.common.utils import datetime_config, generate_id, now_utc
from switch_trust.eval.db.base.common_types import SortDirection


class MessageCollectionType(str, Enum):
    """Discriminator for the kind of MessageCollection."""

    IN_MEMORY = "in-memory"
    POSTGRES = "postgres"
    GARAK = "garak"
    CSV = "csv"


@dataclass_json
@dataclass
class DbMessageCollection:
    """JSON-serialisable description of a MessageCollection."""

    type: MessageCollectionType
    name: str
    id: str = field(default_factory=generate_id)
    created: datetime = field(
        default_factory=now_utc,
        metadata=datetime_config,
    )
    description: str | None = None
    tags: dict[str, str] = field(default_factory=dict)

    # -- in-memory type --
    messages: list[dict] | None = None
    prompts: list[str] | None = None

    # -- garak type --
    probe_name: str | None = None

    # -- csv type --
    filename: str | None = None
    column: str | None = None

    def get_ref(self) -> Reference:
        return Reference(
            self.id,
            self.name,
            ReferenceType.MESSAGE_COLLECTION,
            self.description,
        )


class MessageCollectionSortField(str, Enum):
    NAME = "name"
    CREATED = "created"
    TYPE = "type"


@dataclass_json
@dataclass
class MessageCollectionSortOrder:
    field: MessageCollectionSortField = MessageCollectionSortField.NAME
    direction: SortDirection = SortDirection.ASC


@dataclass_json
@dataclass
class MessageCollectionListView:
    """Paginated list of message collections."""

    items: list[DbMessageCollection] = field(default_factory=list)
    total: int = 0


@dataclass_json
@dataclass
class MessageCollectionDetailView:
    """A message collection with its message count."""

    id: str = ""
    name: str = ""
    description: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    message_count: int = 0
