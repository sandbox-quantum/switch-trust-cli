import logging
from dataclasses import dataclass

from switch_trust.eval.common.schema import Message, Session
from switch_trust.eval.core.detectors.detector import Detector
from switch_trust.eval.core.eval.evaluation import EvaluationStatus
from switch_trust.eval.core.eval.evaluation_single import SingleEvaluation
from switch_trust.eval.core.models.model import Model, ResponseStatus

logger = logging.getLogger(__name__)


_BLOCKED_STATUSES = {
    ResponseStatus.BLOCKED_SAFETY,
    ResponseStatus.BLOCKED_RECITATION,
    ResponseStatus.BLOCKED_PROHIBITED,
    ResponseStatus.EMPTY_RESPONSE,
}


@dataclass
class SinglePromptEvaluation(SingleEvaluation):
    """Evaluates a single prompt against a model and detector."""

    prompt: Message | None = None
    detector: Detector | None = None

    def __init__(self, prompt: Message, detector: Detector):
        super().__init__()
        self.prompt = prompt
        self.detector = detector

    async def init(self):
        if self.prompt is None or self.detector is None:
            raise ValueError("prompt and detector must be set")
        self.status = EvaluationStatus.INITIALIZED

    async def execute_internal(self, model: Model) -> float:
        if self.prompt is None or self.detector is None:
            raise ValueError("prompt and detector must be set")

        logger.debug("SinglePromptEvaluation: sending prompt to model")
        response = await model.generate(self.prompt)

        messages = [self.prompt]
        if response.message:
            messages.append(response.message)
        self.session = Session(messages=messages)

        if response.status in _BLOCKED_STATUSES:
            logger.debug(
                "SinglePromptEvaluation: response blocked (%s)", response.status.value
            )
            return 1.0
        if response.message:
            detector_result = await self.detector.detect(response)
            logger.debug(
                "SinglePromptEvaluation: detector score=%.2f", detector_result.score
            )
            return detector_result.score

        raise ValueError("Model did not return a message or an error status")
