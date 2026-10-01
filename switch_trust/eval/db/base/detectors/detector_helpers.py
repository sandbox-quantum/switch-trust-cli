from __future__ import annotations

from switch_trust.eval.core.detectors.detector import Detector
from switch_trust.eval.db.base.detectors.detector_types import (
    DbDetector,
    DetectorType,
)


def create_detector(
    db_detector: DbDetector,
) -> Detector:
    """Create a Detector instance from a DbDetector."""
    if db_detector.type == DetectorType.GARAK:
        if not db_detector.detector_name:
            raise ValueError("detector_name is required for GARAK detectors")
        from switch_trust.eval.core.detectors.detector_garak import (  # noqa: PLC0415 - deferred import cost
            GarakDetector,
        )

        return GarakDetector(db_detector.detector_name)

    elif db_detector.type == DetectorType.MODEL:
        from switch_trust.eval.core.detectors.detector_model import (  # noqa: PLC0415 - patched at source in tests
            ModelDetector,
        )
        from switch_trust.eval.core.models.generator_model import (  # noqa: PLC0415 - patched at source in tests
            get_generator_model,
        )

        model = get_generator_model()
        kwargs = {}
        if db_detector.prompt is not None:
            kwargs["prompt"] = db_detector.prompt
        return ModelDetector(model=model, **kwargs)

    elif db_detector.type == DetectorType.ADVERSARIAL_MODEL:
        from switch_trust.eval.core.detectors.detector_model_adversarial import (  # noqa: PLC0415 - patched at source in tests
            AdversarialModelDetector,
        )
        from switch_trust.eval.core.models.generator_model import (  # noqa: PLC0415 - patched at source in tests
            get_generator_model,
        )

        model = get_generator_model()
        kwargs = {}
        kwargs["include_conversation"] = db_detector.include_conversation
        if db_detector.prompt is not None:
            kwargs["prompt"] = db_detector.prompt
        return AdversarialModelDetector(model=model, **kwargs)

    elif db_detector.type == DetectorType.PII:
        from switch_trust.eval.core.detectors.detector_pii import (  # noqa: PLC0415 - patched at source in tests
            PIIDetector,
        )

        return PIIDetector()

    elif db_detector.type == DetectorType.SECRET:
        from switch_trust.eval.core.detectors.detector_secret import (  # noqa: PLC0415 - patched at source in tests
            SecretDetector,
        )

        return SecretDetector()

    elif db_detector.type == DetectorType.TOPIC_GUARD:
        from switch_trust.eval.core.detectors.detector_topic_guard import (  # noqa: PLC0415 - patched at source in tests
            TopicGuardDetector,
        )
        from switch_trust.eval.core.models.generator_model import (  # noqa: PLC0415 - patched at source in tests
            get_generator_model,
        )

        if not db_detector.agent_objective and not db_detector.agent_instructions:
            raise ValueError(
                "at least one of agent_objective or "
                "agent_instructions is required for "
                "TOPIC_GUARD detectors"
            )
        model = get_generator_model()
        return TopicGuardDetector(
            model=model,
            agent_objective=db_detector.agent_objective,
            agent_instructions=db_detector.agent_instructions,
        )

    else:
        raise ValueError(f"unknown detector type: {db_detector.type}")
