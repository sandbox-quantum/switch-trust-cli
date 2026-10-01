from abc import ABC, abstractmethod
from dataclasses import dataclass

from dataclasses_json import dataclass_json

from switch_trust.eval.core.models.model import ModelResponse


@dataclass_json
@dataclass
class DetectorResult:
    """Result of a detector evaluation.

    score: A value between 0.0 and 1.0.
           1.0 means fully successful (no issues detected).
           0.0 means failed (issue detected).
    """

    score: float


class Detector(ABC):
    @abstractmethod
    async def detect(self, response: ModelResponse) -> DetectorResult:
        pass
