from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from dataclasses_json import dataclass_json

from switch_trust.eval.core.eval.eval_result_types import (
    EvaluationResult,
    EvaluationStatus,
    EvaluationSummary,
)
from switch_trust.eval.core.models.model import Model
from switch_trust.eval.core.models.model_errors import ErrorKind

EvaluationObserver = Callable[["Evaluation"], None]

# Min fraction of prompts that must succeed for a run to be scored (else ERROR).
DEFAULT_MIN_SUCCESS_RATE = 0.9


@dataclass_json
@dataclass
class Evaluation(ABC):
    status: EvaluationStatus = EvaluationStatus.WAITING
    error_message: str | None = None
    # Set only where the engine itself knows why it failed; model-call failures
    # are classified from the error accumulator instead.
    error_kind: ErrorKind | None = None
    observers: list[EvaluationObserver] = field(
        default_factory=list,
        repr=False,
    )

    def __init__(self):
        self.status = EvaluationStatus.WAITING
        self.error_message = None
        self.error_kind = None
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
        max_timeout: int | None = None,
    ):
        pass

    @abstractmethod
    def get_results(self) -> list[EvaluationResult]:
        pass
