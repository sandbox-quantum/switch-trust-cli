from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from dataclasses_json import dataclass_json

from switch_trust.eval.common.schema import Session


class EvaluationStatus(str, Enum):
    WAITING = "waiting"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"


@dataclass_json
@dataclass
class EvaluationSummary:
    status: EvaluationStatus
    total_evaluations: int
    finished_evaluations: int
    error_evaluations: int
    max_score: float
    achieved_score: float
    error_messages: list[str] = field(default_factory=list)

    @property
    def score(self) -> float | None:
        if self.status != EvaluationStatus.FINISHED:
            return None
        if self.max_score == 0:
            return None
        return self.achieved_score / self.max_score

    @property
    def progress(self) -> float:
        if self.total_evaluations == 0:
            return 1.0
        return (
            self.finished_evaluations + self.error_evaluations
        ) / self.total_evaluations


@dataclass_json
@dataclass
class EvaluationResult:
    """Result of a single leaf evaluation."""

    score: float
    status: EvaluationStatus
    error_message: str | None = None
    session: Session | None = None
