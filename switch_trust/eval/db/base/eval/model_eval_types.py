from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dataclasses_json import dataclass_json

from switch_trust.eval.common.reference import Reference, ReferenceType
from switch_trust.eval.common.utils import datetime_config, generate_id, now_utc
from switch_trust.eval.db.base.common_types import SortDirection


@dataclass_json
@dataclass
class DbModelEvaluation:
    """Links a Model to an Evaluation with a weight."""

    model_id: str
    evaluation_id: str
    name: str
    id: str = field(default_factory=generate_id)
    created: datetime = field(
        default_factory=now_utc,
        metadata=datetime_config,
    )
    description: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    weight: float = 0.5

    def get_ref(self) -> Reference:
        return Reference(
            self.id,
            self.name,
            ReferenceType.MODEL_EVALUATION_SUITE,
            self.description,
        )


class ModelEvaluationSortField(str, Enum):
    NAME = "name"
    CREATED = "created"
    WEIGHT = "weight"


@dataclass_json
@dataclass
class ModelEvaluationSortOrder:
    field: ModelEvaluationSortField = ModelEvaluationSortField.NAME
    direction: SortDirection = SortDirection.ASC


@dataclass_json
@dataclass
class ModelEvaluationListItem:
    """A model-evaluation assignment with resolved references."""

    config: DbModelEvaluation
    model_ref: Reference | None = None
    evaluation_ref: Reference | None = None


@dataclass_json
@dataclass
class ModelEvaluationListView:
    """Paginated list of model-evaluation assignments."""

    items: list[ModelEvaluationListItem] = field(
        default_factory=list,
    )
    total: int = 0
