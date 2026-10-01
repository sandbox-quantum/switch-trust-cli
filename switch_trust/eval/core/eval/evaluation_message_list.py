import random
from dataclasses import dataclass

from dataclasses_json import dataclass_json

from switch_trust.eval.common.schema import Message
from switch_trust.eval.core.detectors.detector import Detector
from switch_trust.eval.core.eval.evaluation import Evaluation
from switch_trust.eval.core.eval.evaluation_multi import MultiEvaluation
from switch_trust.eval.core.eval.evaluation_single_prompt import (
    SinglePromptEvaluation,
)


@dataclass_json
@dataclass
class MessageListEvaluation(MultiEvaluation):
    """Evaluates a list of messages against a detector."""

    messages: list[Message] | None = None
    detector: Detector | None = None
    num_prompts: int | None = None

    def __init__(
        self,
        messages: list[Message],
        detector: Detector,
        num_prompts: int | None = None,
    ):
        super().__init__()
        self.messages = messages
        self.detector = detector
        self.num_prompts = num_prompts

    async def get_children(self) -> list[Evaluation]:
        if not self.messages or self.detector is None:
            raise ValueError("messages and detector must be set")

        messages = self.messages
        if self.num_prompts is not None and self.num_prompts < len(messages):
            messages = random.sample(messages, self.num_prompts)

        return [
            SinglePromptEvaluation(prompt=msg, detector=self.detector)
            for msg in messages
        ]
