from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dataclasses_json import dataclass_json

from switch_trust.eval.common.reference import Reference, ReferenceType
from switch_trust.eval.common.utils import datetime_config, generate_id, now_utc
from switch_trust.eval.db.base.common_types import SortDirection
from switch_trust.eval.db.base.eval.model_eval_run_types import DbModelEvaluationRun


class EvaluationType(str, Enum):
    """Discriminator for the kind of Evaluation."""

    MESSAGE_COLLECTION = "message_collection"
    GARAK_PROBE = "garak_probe"
    GARAK_MODULE = "garak_module"
    METRIC_TOXICITY = "metric_toxicity"
    METRIC_CONCISENESS = "metric_conciseness"
    METRIC_FACTUAL_ACCURACY = "metric_factual_accuracy"
    METRIC_INSTRUCTION_ADHERENCE = "metric_instruction_adherence"
    METRIC_TONE = "metric_tone"
    ADVERSARIAL_PROBE = "adversarial_probe"
    TOPIC_GUARD = "topic_guard"


class EvaluationApproach(str, Enum):
    """Whether an evaluation actively probes the model
    or passively measures a quality metric."""

    PROBE = "Probe"
    METRIC = "Metric"


@dataclass_json
@dataclass
class DbEvaluation:
    """JSON-serialisable description of an Evaluation."""

    type: EvaluationType
    name: str
    id: str = field(default_factory=generate_id)
    created: datetime = field(
        default_factory=now_utc,
        metadata=datetime_config,
    )
    description: str | None = None
    approach: EvaluationApproach = EvaluationApproach.PROBE
    tags: dict[str, str] = field(default_factory=dict)

    # -- shared (message_collection, adversarial_probe) --
    message_collection_id: str | None = None
    detector_id: str | None = None

    # -- garak_probe type --
    probe_name: str | None = None

    # -- garak_module type --
    module_name: str | None = None
    probe_names: list[str] | None = None

    # -- adversarial_probe type --
    adversarial_goals: list[str] | None = None
    attack_techniques: list[str] | None = None
    num_prompts: int | None = None
    max_turns: int | None = None

    # -- topic_guard type --
    agent_objective: str | None = None
    agent_instructions: str | None = None

    def get_ref(self) -> Reference:
        return Reference(
            self.id,
            self.name,
            ReferenceType.EVALUATION,
            self.description,
        )


class EvaluationSortField(str, Enum):
    NAME = "name"
    CREATED = "created"
    TYPE = "type"


@dataclass_json
@dataclass
class EvaluationSortOrder:
    field: EvaluationSortField = EvaluationSortField.NAME
    direction: SortDirection = SortDirection.ASC


@dataclass_json
@dataclass
class EvaluationListView:
    """Paginated list of evaluation configurations."""

    items: list[DbEvaluation] = field(default_factory=list)
    total: int = 0


@dataclass_json
@dataclass
class EvaluationDetailModelView:
    """A model assigned to an evaluation, with last run info."""

    id: str
    model_ref: Reference
    evaluation_ref: Reference
    created: datetime = field(metadata=datetime_config)
    weight: float = 0.5
    last_run: DbModelEvaluationRun | None = None


@dataclass_json
@dataclass
class EvaluationDetailView:
    """An evaluation with its assigned models."""

    config: DbEvaluation | None = None
    models: list[EvaluationDetailModelView] = field(
        default_factory=list,
    )
