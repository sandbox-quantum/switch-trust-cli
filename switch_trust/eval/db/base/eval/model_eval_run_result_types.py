from dataclasses import dataclass, field

from dataclasses_json import dataclass_json

from switch_trust.eval.common.schema import Session
from switch_trust.eval.common.utils import generate_id
from switch_trust.eval.core.eval.evaluation import EvaluationStatus


@dataclass_json
@dataclass
class DbModelEvaluationRunResult:
    """A single leaf-evaluation result within a run."""

    run_id: str
    score: float
    status: EvaluationStatus
    id: str = field(default_factory=generate_id)
    error_message: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    session: Session | None = None
