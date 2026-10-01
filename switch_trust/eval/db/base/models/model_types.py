from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dataclasses_json import dataclass_json

from switch_trust.eval.common.reference import Reference, ReferenceType
from switch_trust.eval.common.utils import datetime_config, generate_id, now_utc
from switch_trust.eval.db.base.common_types import SortDirection
from switch_trust.eval.db.base.eval.model_eval_run_types import DbModelEvaluationRun


class ModelType(str, Enum):
    """Discriminator for the kind of Model."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"
    LITELLM = "litellm"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    ADK = "adk"
    OPENAI_AGENT = "openai_agent"
    ANTHROPIC_AGENT = "anthropic_agent"
    OPENAI_COMPATIBLE = "openai_compatible"
    GENERIC_HTTP = "generic_http"
    LANGSERVE = "langserve"
    VERTEX_AGENT_RUNTIME = "vertex_agent_runtime"


@dataclass_json
@dataclass
class DbModel:
    """JSON-serialisable description of a Model."""

    type: ModelType
    name: str
    model_name: str
    id: str = field(default_factory=generate_id)
    created: datetime = field(
        default_factory=now_utc,
        metadata=datetime_config,
    )
    description: str | None = None

    tags: dict[str, str] = field(default_factory=dict)

    # -- common optional --
    key: str | None = None
    temperature: float = 0.0

    # -- ollama / agent types --
    host: str | None = None
    endpoint: str | None = None

    # -- adk / vertex_agent_runtime --
    immediate_result: bool = False

    # -- vertex_agent_runtime --
    user_id: str | None = None
    class_method: str | None = None

    # -- generic_http / openai_compatible / langserve --
    headers: dict[str, str] = field(default_factory=dict)
    input_path: str | None = None
    output_path: str | None = None

    def get_ref(self) -> Reference:
        return Reference(
            self.id,
            self.name,
            ReferenceType.MODEL,
            self.description,
        )


class ModelSortField(str, Enum):
    NAME = "name"
    CREATED = "created"
    TYPE = "type"
    SCORE = "score"


@dataclass_json
@dataclass
class ModelSortOrder:
    field: ModelSortField = ModelSortField.NAME
    direction: SortDirection = SortDirection.ASC


@dataclass_json
@dataclass
class ModelScoreSummary:
    """Aggregate score and run status for a model."""

    overall_score: float | None = None
    finished_evaluations: int = 0
    running_evaluations: int = 0
    errored_evaluations: int = 0
    pending_evaluations: int = 0


@dataclass_json
@dataclass
class ModelListViewItem:
    """A model with its aggregate score summary."""

    model: DbModel | None = None
    score_summary: ModelScoreSummary = field(
        default_factory=ModelScoreSummary,
    )


@dataclass_json
@dataclass
class ModelListView:
    """Paginated list of model configurations with scores."""

    items: list[ModelListViewItem] = field(
        default_factory=list,
    )
    total: int = 0


@dataclass_json
@dataclass
class DbModelListView:
    """Raw paginated list from the repository."""

    items: list[DbModel] = field(default_factory=list)
    total: int = 0


@dataclass_json
@dataclass
class ModelDetailEvaluationView:
    """A single ModelEvaluation with references and last run."""

    id: str
    model_ref: Reference
    evaluation_ref: Reference
    created: datetime = field(metadata=datetime_config)
    weight: float = 0.5
    evaluation_approach: str = "Probe"
    evaluation_tags: dict[str, str] = field(
        default_factory=dict,
    )
    last_run: DbModelEvaluationRun | None = None


@dataclass_json
@dataclass
class ModelDetailView:
    """A model configuration with its evaluations."""

    model: DbModel | None = None
    evaluations: list[ModelDetailEvaluationView] = field(
        default_factory=list,
    )
