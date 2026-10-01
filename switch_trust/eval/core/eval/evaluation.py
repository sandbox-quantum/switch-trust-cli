from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from dataclasses_json import dataclass_json

from switch_trust.eval.common.schema import Session
from switch_trust.eval.core.models.model import Model


class EvaluationStatus(str, Enum):
    WAITING = "waiting"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"


EvaluationObserver = Callable[["Evaluation"], None]

# Min fraction of prompts that must succeed for a run to be scored (else ERROR).
DEFAULT_MIN_SUCCESS_RATE = 0.9


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


@dataclass_json
@dataclass
class Evaluation(ABC):
    status: EvaluationStatus = EvaluationStatus.WAITING
    error_message: str | None = None
    observers: list[EvaluationObserver] = field(
        default_factory=list,
        repr=False,
    )

    def __init__(self):
        self.status = EvaluationStatus.WAITING
        self.error_message = None
        self.observers = []

    def add_observer(self, observer: EvaluationObserver):
        self.observers.append(observer)

    def remove_observer(self, observer: EvaluationObserver):
        self.observers.remove(observer)

    def _notify_observers(self):
        for observer in self.observers:
            observer(self)

    @abstractmethod
    async def init(self):
        pass

    @abstractmethod
    def get_summary(self) -> EvaluationSummary:
        pass

    @abstractmethod
    async def run(
        self,
        model: Model,
        concurrency: int = 50,
        min_success_rate: float = DEFAULT_MIN_SUCCESS_RATE,
    ):
        pass

    @abstractmethod
    def get_results(self) -> list[EvaluationResult]:
        pass
