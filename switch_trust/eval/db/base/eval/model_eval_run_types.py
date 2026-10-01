from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dataclasses_json import dataclass_json

from switch_trust.eval.common.reference import Reference
from switch_trust.eval.common.utils import datetime_config, generate_id, now_utc
from switch_trust.eval.core.eval.evaluation import EvaluationSummary
from switch_trust.eval.db.base.common_types import SortDirection


@dataclass_json
@dataclass
class DbModelEvaluationRun:
    """A single run of a DbModelEvaluation."""

    model_evaluation_id: str
    id: str = field(default_factory=generate_id)
    created: datetime = field(
        default_factory=now_utc,
        metadata=datetime_config,
    )
    started: datetime | None = field(
        default=None,
        metadata=datetime_config,
    )
    finished: datetime | None = field(
        default=None,
        metadata=datetime_config,
    )
    tags: dict[str, str] = field(default_factory=dict)
    summary: EvaluationSummary | None = None


class ModelEvaluationRunSortField(str, Enum):
    CREATED = "created"
    STARTED = "started_at"
    FINISHED = "finished_at"
    SCORE = "score"


@dataclass_json
@dataclass
class ModelEvaluationRunSortOrder:
    field: ModelEvaluationRunSortField = ModelEvaluationRunSortField.CREATED
    direction: SortDirection = SortDirection.DESC


@dataclass_json
@dataclass
class ModelEvaluationRunDetailView:
    """A model-evaluation run with resolved references."""

    run: DbModelEvaluationRun
    model_ref: Reference | None = None
    evaluation_ref: Reference | None = None


@dataclass_json
@dataclass
class ModelEvaluationRunListItem:
    """A model-evaluation run with resolved references."""

    run: DbModelEvaluationRun
    model_ref: Reference | None = None
    evaluation_ref: Reference | None = None


@dataclass_json
@dataclass
class ModelEvaluationRunListView:
    """Paginated list of model-evaluation runs."""

    items: list[ModelEvaluationRunListItem] = field(
        default_factory=list,
    )
    total: int = 0
