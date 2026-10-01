from __future__ import annotations

from switch_trust.eval.core.eval.evaluation import Evaluation
from switch_trust.eval.core.models.generator_model import get_generator_model
from switch_trust.eval.db.base.detectors.detector_repository import DetectorRepository
from switch_trust.eval.db.base.eval.eval_types import DbEvaluation, EvaluationType
from switch_trust.eval.db.base.message.message_collection_repository import (
    MessageCollectionRepository,
)


def create_evaluation(
    db_evaluation: DbEvaluation,
    message_collection_repo: MessageCollectionRepository | None = None,
    detector_repo: DetectorRepository | None = None,
) -> Evaluation:
    """Create an Evaluation instance from a DbEvaluation."""
    if db_evaluation.type == EvaluationType.MESSAGE_COLLECTION:
        if message_collection_repo is None:
            raise ValueError(
                "message_collection_repo is required for MESSAGE_COLLECTION evaluations"
            )
        if db_evaluation.message_collection_id is None:
            raise ValueError("message_collection_id must be set on the DbEvaluation")
        if detector_repo is None:
            raise ValueError(
                "detector_repo is required for MESSAGE_COLLECTION evaluations"
            )
        if db_evaluation.detector_id is None:
            raise ValueError("detector_id must be set on the DbEvaluation")
        from switch_trust.eval.core.eval.evaluation_message_list import (  # noqa: PLC0415 - patched at source in tests
            MessageListEvaluation,
        )

        message_collection = message_collection_repo.get_message_collection(
            db_evaluation.message_collection_id
        )
        messages = message_collection.load()
        detector = detector_repo.get_detector(
            db_evaluation.detector_id,
        )
        return MessageListEvaluation(
            messages=messages,
            detector=detector,
            num_prompts=db_evaluation.num_prompts,
        )

    elif db_evaluation.type == EvaluationType.GARAK_PROBE:
        from switch_trust.eval.core.eval.evaluation_garak_probe import (  # noqa: PLC0415 - patched at source in tests
            GarakProbeEvaluation,
        )

        if db_evaluation.probe_name is None:
            raise ValueError("probe_name must be set on the DbEvaluation")
        return GarakProbeEvaluation(
            probe_name=db_evaluation.probe_name,
        )

    elif db_evaluation.type == EvaluationType.GARAK_MODULE:
        from switch_trust.eval.core.eval.evaluation_garak_module import (  # noqa: PLC0415 - patched at source in tests
            GarakModuleEvaluation,
        )

        if db_evaluation.module_name is None:
            raise ValueError("module_name must be set on the DbEvaluation")
        return GarakModuleEvaluation(
            module_name=db_evaluation.module_name,
            probe_names=db_evaluation.probe_names,
        )

    elif db_evaluation.type == EvaluationType.METRIC_TOXICITY:
        from switch_trust.eval.core.eval.metric_toxicity import (  # noqa: PLC0415 - patched at source in tests
            ToxicityMetricEvaluation,
        )

        return ToxicityMetricEvaluation()

    elif db_evaluation.type == EvaluationType.METRIC_CONCISENESS:
        from switch_trust.eval.core.eval.metric_conciseness import (  # noqa: PLC0415 - patched at source in tests
            ConcisenessMetricEvaluation,
        )

        return ConcisenessMetricEvaluation(
            judge_model=get_generator_model(),
        )

    elif db_evaluation.type == EvaluationType.METRIC_FACTUAL_ACCURACY:
        from switch_trust.eval.core.eval.metric_factual_accuracy import (  # noqa: PLC0415 - patched at source in tests
            FactualAccuracyMetricEvaluation,
        )

        return FactualAccuracyMetricEvaluation(
            judge_model=get_generator_model(),
        )

    elif db_evaluation.type == EvaluationType.METRIC_INSTRUCTION_ADHERENCE:
        from switch_trust.eval.core.eval.metric_instruction_adherence import (  # noqa: PLC0415 - patched at source in tests
            InstructionAdherenceMetricEvaluation,
        )

        return InstructionAdherenceMetricEvaluation(
            judge_model=get_generator_model(),
        )

    elif db_evaluation.type == EvaluationType.METRIC_TONE:
        from switch_trust.eval.core.eval.metric_tone import (  # noqa: PLC0415 - patched at source in tests
            ToneMetricEvaluation,
        )

        return ToneMetricEvaluation(
            judge_model=get_generator_model(),
        )

    elif db_evaluation.type == EvaluationType.ADVERSARIAL_PROBE:
        from switch_trust.eval.core.eval.evaluation_adversarial import (  # noqa: PLC0415 - patched at source in tests
            AdversarialEvaluation,
        )

        goals: list[str] = []
        if db_evaluation.message_collection_id and message_collection_repo:
            mc = message_collection_repo.get_message_collection(
                db_evaluation.message_collection_id,
            )
            messages = mc.load()
            goals = [
                part.text
                for msg in messages
                for part in msg.content.parts
                if part.text is not None
            ]

        if db_evaluation.adversarial_goals:
            goals.extend(db_evaluation.adversarial_goals)

        if not goals:
            raise ValueError(
                "adversarial_goals or "
                "message_collection_id must be set on "
                "the DbEvaluation"
            )
        if not db_evaluation.attack_techniques:
            raise ValueError(
                "attack_techniques must be set on "
                "the DbEvaluation for adversarial probes"
            )
        if detector_repo is None:
            raise ValueError(
                "detector_repo is required for ADVERSARIAL_PROBE evaluations"
            )
        if not db_evaluation.detector_id:
            raise ValueError(
                "detector_id must be set on the DbEvaluation for adversarial probes"
            )
        detector = detector_repo.get_detector(
            db_evaluation.detector_id,
        )
        return AdversarialEvaluation(
            goals=goals,
            attack_techniques=db_evaluation.attack_techniques,
            detector=detector,
            num_prompts=db_evaluation.num_prompts or 5,
            max_turns=db_evaluation.max_turns or 5,
            attacker_model=get_generator_model(),
        )

    elif db_evaluation.type == EvaluationType.TOPIC_GUARD:
        from switch_trust.eval.core.eval.evaluation_topic_guard import (  # noqa: PLC0415 - patched at source in tests
            TopicGuardEvaluation,
        )

        if not db_evaluation.agent_objective and not db_evaluation.agent_instructions:
            raise ValueError(
                "at least one of agent_objective or "
                "agent_instructions must be set on "
                "the DbEvaluation"
            )
        return TopicGuardEvaluation(
            agent_objective=db_evaluation.agent_objective,
            agent_instructions=db_evaluation.agent_instructions,
            num_prompts=db_evaluation.num_prompts or 5,
            max_turns=db_evaluation.max_turns or 5,
            attacker_model=get_generator_model(),
        )

    else:
        raise ValueError(f"unknown evaluation type: {db_evaluation.type}")
