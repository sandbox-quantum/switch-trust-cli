from dataclasses import dataclass
from enum import Enum

from dataclasses_json import dataclass_json


class ReferenceType(str, Enum):
    """The kind of object a Reference points to."""

    DETECTOR = "detector"
    EVALUATION = "evaluation"
    EVALUATION_SUITE = "evaluation_suite"
    MESSAGE_COLLECTION = "message_collection"
    MODEL = "model"
    MODEL_EVALUATION_SUITE = "model_evaluation_suite"
    MODEL_EVALUATION_SUITE_RUN = "model_evaluation_suite_run"


@dataclass_json
@dataclass
class Reference:
    id: str
    name: str
    type: ReferenceType
    description: str | None = None
