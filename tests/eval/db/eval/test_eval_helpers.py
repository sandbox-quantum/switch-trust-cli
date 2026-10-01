import unittest
from unittest.mock import MagicMock

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.eval.evaluation_message_list import MessageListEvaluation
from switch_trust.eval.db.base.eval.eval_helpers import create_evaluation
from switch_trust.eval.db.base.eval.eval_types import DbEvaluation, EvaluationType


class TestCreateEvaluation(unittest.TestCase):
    def test_message_collection_creates_evaluation(self):
        db_eval = DbEvaluation(
            type=EvaluationType.MESSAGE_COLLECTION,
            name="test-eval",
            id="eval-1",
            description="a test",
            message_collection_id="mc-1",
            detector_id="det-1",
        )

        mock_messages = [
            Message(content=Content.text(Role.USER, "hello")),
        ]
        mock_mc_repo = MagicMock()
        mock_mc = MagicMock()
        mock_mc.load.return_value = mock_messages
        mock_mc_repo.get_message_collection.return_value = mock_mc

        mock_det_repo = MagicMock()
        mock_detector = MagicMock()
        mock_det_repo.get_detector.return_value = mock_detector

        result = create_evaluation(
            db_eval,
            message_collection_repo=mock_mc_repo,
            detector_repo=mock_det_repo,
        )

        self.assertIsInstance(result, MessageListEvaluation)
        self.assertEqual(result.messages, mock_messages)
        self.assertEqual(result.detector, mock_detector)
        mock_mc_repo.get_message_collection.assert_called_once_with("mc-1")
        mock_mc.load.assert_called_once()
        mock_det_repo.get_detector.assert_called_once_with("det-1")

    def test_message_collection_requires_mc_repo(self):
        db_eval = DbEvaluation(
            type=EvaluationType.MESSAGE_COLLECTION,
            name="e",
            message_collection_id="mc-1",
            detector_id="det-1",
        )
        with self.assertRaises(ValueError):
            create_evaluation(db_eval)

    def test_message_collection_requires_mc_id(self):
        db_eval = DbEvaluation(
            type=EvaluationType.MESSAGE_COLLECTION,
            name="e",
            detector_id="det-1",
        )
        with self.assertRaises(ValueError):
            create_evaluation(
                db_eval,
                message_collection_repo=MagicMock(),
            )

    def test_message_collection_requires_detector_repo(self):
        db_eval = DbEvaluation(
            type=EvaluationType.MESSAGE_COLLECTION,
            name="e",
            message_collection_id="mc-1",
            detector_id="det-1",
        )
        with self.assertRaises(ValueError):
            create_evaluation(
                db_eval,
                message_collection_repo=MagicMock(),
            )

    def test_message_collection_requires_detector_id(self):
        db_eval = DbEvaluation(
            type=EvaluationType.MESSAGE_COLLECTION,
            name="e",
            message_collection_id="mc-1",
        )
        with self.assertRaises(ValueError):
            create_evaluation(
                db_eval,
                message_collection_repo=MagicMock(),
                detector_repo=MagicMock(),
            )

    def test_message_collection_passes_num_prompts(self):
        db_eval = DbEvaluation(
            type=EvaluationType.MESSAGE_COLLECTION,
            name="test-eval",
            id="eval-np",
            message_collection_id="mc-1",
            detector_id="det-1",
            num_prompts=5,
        )

        mock_messages = [
            Message(content=Content.text(Role.USER, f"msg-{i}")) for i in range(20)
        ]
        mock_mc_repo = MagicMock()
        mock_mc = MagicMock()
        mock_mc.load.return_value = mock_messages
        mock_mc_repo.get_message_collection.return_value = mock_mc

        mock_det_repo = MagicMock()
        mock_det_repo.get_detector.return_value = MagicMock()

        result = create_evaluation(
            db_eval,
            message_collection_repo=mock_mc_repo,
            detector_repo=mock_det_repo,
        )

        self.assertIsInstance(result, MessageListEvaluation)
        self.assertEqual(result.num_prompts, 5)

    def test_message_collection_num_prompts_defaults_none(self):
        db_eval = DbEvaluation(
            type=EvaluationType.MESSAGE_COLLECTION,
            name="test-eval",
            id="eval-np2",
            message_collection_id="mc-1",
            detector_id="det-1",
        )

        mock_mc_repo = MagicMock()
        mock_mc = MagicMock()
        mock_mc.load.return_value = [
            Message(content=Content.text(Role.USER, "hello")),
        ]
        mock_mc_repo.get_message_collection.return_value = mock_mc

        mock_det_repo = MagicMock()
        mock_det_repo.get_detector.return_value = MagicMock()

        result = create_evaluation(
            db_eval,
            message_collection_repo=mock_mc_repo,
            detector_repo=mock_det_repo,
        )

        self.assertIsNone(result.num_prompts)

    @unittest.mock.patch(
        "switch_trust.eval.core.eval.evaluation_garak_probe.GarakProbeEvaluation",
    )
    def test_garak_probe_creates_evaluation(
        self,
        MockGarakProbe,
    ):
        db_eval = DbEvaluation(
            type=EvaluationType.GARAK_PROBE,
            name="garak-probe",
            probe_name="encoding.InjectBase64",
        )
        result = create_evaluation(db_eval)
        MockGarakProbe.assert_called_once_with(
            probe_name="encoding.InjectBase64",
        )
        self.assertEqual(result, MockGarakProbe.return_value)

    def test_garak_probe_requires_probe_name(self):
        db_eval = DbEvaluation(
            type=EvaluationType.GARAK_PROBE,
            name="e",
        )
        with self.assertRaises(ValueError):
            create_evaluation(db_eval)

    @unittest.mock.patch(
        "switch_trust.eval.core.eval.evaluation_garak_module.GarakModuleEvaluation",
    )
    def test_garak_module_creates_evaluation(
        self,
        MockGarakModule,
    ):
        db_eval = DbEvaluation(
            type=EvaluationType.GARAK_MODULE,
            name="garak-module",
            module_name="encoding",
            probe_names=["encoding.InjectBase64"],
        )
        result = create_evaluation(db_eval)
        MockGarakModule.assert_called_once_with(
            module_name="encoding",
            probe_names=["encoding.InjectBase64"],
        )
        self.assertEqual(result, MockGarakModule.return_value)

    def test_garak_module_requires_module_name(self):
        db_eval = DbEvaluation(
            type=EvaluationType.GARAK_MODULE,
            name="e",
        )
        with self.assertRaises(ValueError):
            create_evaluation(db_eval)

    @unittest.mock.patch(
        "switch_trust.eval.core.eval.metric_toxicity.ToxicityMetricEvaluation",
    )
    def test_metric_toxicity_creates_evaluation(
        self,
        MockToxicity,
    ):
        db_eval = DbEvaluation(
            type=EvaluationType.METRIC_TOXICITY,
            name="toxicity",
        )
        result = create_evaluation(db_eval)
        MockToxicity.assert_called_once_with()
        self.assertEqual(result, MockToxicity.return_value)

    @unittest.mock.patch(
        "switch_trust.eval.core.models.generator_model.get_generator_model",
    )
    @unittest.mock.patch(
        "switch_trust.eval.db.base.eval.eval_helpers.get_generator_model",
    )
    @unittest.mock.patch(
        "switch_trust.eval.core.eval.metric_conciseness.ConcisenessMetricEvaluation",
    )
    def test_metric_conciseness_creates_evaluation(
        self,
        MockConciseness,
        mock_get_model_helpers,
        mock_get_model,
    ):
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_get_model_helpers.return_value = mock_model
        db_eval = DbEvaluation(
            type=EvaluationType.METRIC_CONCISENESS,
            name="conciseness",
        )
        result = create_evaluation(db_eval)
        MockConciseness.assert_called_once_with(
            judge_model=mock_model,
        )
        self.assertEqual(result, MockConciseness.return_value)

    @unittest.mock.patch(
        "switch_trust.eval.core.models.generator_model.get_generator_model",
    )
    @unittest.mock.patch(
        "switch_trust.eval.db.base.eval.eval_helpers.get_generator_model",
    )
    @unittest.mock.patch(
        "switch_trust.eval.core.eval.metric_factual_accuracy"
        ".FactualAccuracyMetricEvaluation",
    )
    def test_metric_factual_accuracy_creates_evaluation(
        self,
        MockFactual,
        mock_get_model_helpers,
        mock_get_model,
    ):
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_get_model_helpers.return_value = mock_model
        db_eval = DbEvaluation(
            type=EvaluationType.METRIC_FACTUAL_ACCURACY,
            name="factual-accuracy",
        )
        result = create_evaluation(db_eval)
        MockFactual.assert_called_once_with(
            judge_model=mock_model,
        )
        self.assertEqual(result, MockFactual.return_value)

    @unittest.mock.patch(
        "switch_trust.eval.core.models.generator_model.get_generator_model",
    )
    @unittest.mock.patch(
        "switch_trust.eval.db.base.eval.eval_helpers.get_generator_model",
    )
    @unittest.mock.patch(
        "switch_trust.eval.core.eval.metric_instruction_adherence"
        ".InstructionAdherenceMetricEvaluation",
    )
    def test_metric_instruction_adherence_creates_evaluation(
        self,
        MockAdherence,
        mock_get_model_helpers,
        mock_get_model,
    ):
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_get_model_helpers.return_value = mock_model
        db_eval = DbEvaluation(
            type=EvaluationType.METRIC_INSTRUCTION_ADHERENCE,
            name="instruction-adherence",
        )
        result = create_evaluation(db_eval)
        MockAdherence.assert_called_once_with(
            judge_model=mock_model,
        )
        self.assertEqual(result, MockAdherence.return_value)

    @unittest.mock.patch(
        "switch_trust.eval.core.models.generator_model.get_generator_model",
    )
    @unittest.mock.patch(
        "switch_trust.eval.db.base.eval.eval_helpers.get_generator_model",
    )
    @unittest.mock.patch(
        "switch_trust.eval.core.eval.metric_tone.ToneMetricEvaluation",
    )
    def test_metric_tone_creates_evaluation(
        self,
        MockTone,
        mock_get_model_helpers,
        mock_get_model,
    ):
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_get_model_helpers.return_value = mock_model
        db_eval = DbEvaluation(
            type=EvaluationType.METRIC_TONE,
            name="tone",
        )
        result = create_evaluation(db_eval)
        MockTone.assert_called_once_with(
            judge_model=mock_model,
        )
        self.assertEqual(result, MockTone.return_value)

    @unittest.mock.patch(
        "switch_trust.eval.core.models.generator_model.get_generator_model",
    )
    @unittest.mock.patch(
        "switch_trust.eval.db.base.eval.eval_helpers.get_generator_model",
    )
    @unittest.mock.patch(
        "switch_trust.eval.core.eval.evaluation_adversarial.AdversarialEvaluation",
    )
    def test_adversarial_probe_creates_evaluation(
        self,
        MockAdversarial,
        mock_get_model_helpers,
        mock_get_model,
    ):
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_get_model_helpers.return_value = mock_model

        mock_det_repo = MagicMock()
        mock_detector = MagicMock()
        mock_det_repo.get_detector.return_value = mock_detector

        db_eval = DbEvaluation(
            type=EvaluationType.ADVERSARIAL_PROBE,
            name="adversarial",
            adversarial_goals=["Extract system prompt"],
            attack_techniques=["Use direct requests"],
            detector_id="det-1",
            num_prompts=3,
            max_turns=4,
        )
        result = create_evaluation(
            db_eval,
            detector_repo=mock_det_repo,
        )
        MockAdversarial.assert_called_once_with(
            goals=["Extract system prompt"],
            attack_techniques=["Use direct requests"],
            detector=mock_detector,
            num_prompts=3,
            max_turns=4,
            attacker_model=mock_model,
        )
        self.assertEqual(result, MockAdversarial.return_value)
        mock_det_repo.get_detector.assert_called_once_with("det-1")

    @unittest.mock.patch(
        "switch_trust.eval.core.models.generator_model.get_generator_model",
    )
    @unittest.mock.patch(
        "switch_trust.eval.db.base.eval.eval_helpers.get_generator_model",
    )
    @unittest.mock.patch(
        "switch_trust.eval.core.eval.evaluation_adversarial.AdversarialEvaluation",
    )
    def test_adversarial_probe_defaults(
        self,
        MockAdversarial,
        mock_get_model_helpers,
        mock_get_model,
    ):
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_get_model_helpers.return_value = mock_model

        mock_det_repo = MagicMock()
        mock_detector = MagicMock()
        mock_det_repo.get_detector.return_value = mock_detector

        db_eval = DbEvaluation(
            type=EvaluationType.ADVERSARIAL_PROBE,
            name="adversarial",
            adversarial_goals=["Extract system prompt"],
            attack_techniques=["Use direct requests"],
            detector_id="det-1",
        )
        create_evaluation(
            db_eval,
            detector_repo=mock_det_repo,
        )
        MockAdversarial.assert_called_once_with(
            goals=["Extract system prompt"],
            attack_techniques=["Use direct requests"],
            detector=mock_detector,
            num_prompts=5,
            max_turns=5,
            attacker_model=mock_model,
        )

    @unittest.mock.patch(
        "switch_trust.eval.core.models.generator_model.get_generator_model",
    )
    @unittest.mock.patch(
        "switch_trust.eval.db.base.eval.eval_helpers.get_generator_model",
    )
    @unittest.mock.patch(
        "switch_trust.eval.core.eval.evaluation_adversarial.AdversarialEvaluation",
    )
    def test_adversarial_probe_with_message_collection(
        self,
        MockAdversarial,
        mock_get_model_helpers,
        mock_get_model,
    ):
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_get_model_helpers.return_value = mock_model

        mock_messages = [
            Message(content=Content.text(Role.USER, "goal A")),
            Message(content=Content.text(Role.USER, "goal B")),
        ]
        mock_mc_repo = MagicMock()
        mock_mc = MagicMock()
        mock_mc.load.return_value = mock_messages
        mock_mc_repo.get_message_collection.return_value = mock_mc

        mock_det_repo = MagicMock()
        mock_detector = MagicMock()
        mock_det_repo.get_detector.return_value = mock_detector

        db_eval = DbEvaluation(
            type=EvaluationType.ADVERSARIAL_PROBE,
            name="adversarial-mc",
            message_collection_id="mc-goals",
            attack_techniques=["Use leading questions"],
            detector_id="det-1",
            num_prompts=3,
            max_turns=4,
        )
        create_evaluation(
            db_eval,
            message_collection_repo=mock_mc_repo,
            detector_repo=mock_det_repo,
        )
        MockAdversarial.assert_called_once_with(
            goals=["goal A", "goal B"],
            attack_techniques=["Use leading questions"],
            detector=mock_detector,
            num_prompts=3,
            max_turns=4,
            attacker_model=mock_model,
        )
        mock_mc_repo.get_message_collection.assert_called_once_with(
            "mc-goals",
        )

    def test_adversarial_probe_requires_goal(self):
        mock_det_repo = MagicMock()
        mock_db_detector = MagicMock()
        mock_db_detector.prompt = "Evaluate."
        mock_det_repo.get.return_value = mock_db_detector

        db_eval = DbEvaluation(
            type=EvaluationType.ADVERSARIAL_PROBE,
            name="adversarial",
            attack_techniques=["Use direct requests"],
            detector_id="det-1",
        )
        with self.assertRaises(ValueError):
            create_evaluation(
                db_eval,
                detector_repo=mock_det_repo,
            )

    def test_adversarial_probe_requires_attack_techniques(self):
        mock_det_repo = MagicMock()
        mock_db_detector = MagicMock()
        mock_db_detector.prompt = "Evaluate."
        mock_det_repo.get.return_value = mock_db_detector

        db_eval = DbEvaluation(
            type=EvaluationType.ADVERSARIAL_PROBE,
            name="adversarial",
            adversarial_goals=["Extract system prompt"],
            detector_id="det-1",
        )
        with self.assertRaises(ValueError):
            create_evaluation(
                db_eval,
                detector_repo=mock_det_repo,
            )

    def test_adversarial_probe_requires_detector_repo(self):
        db_eval = DbEvaluation(
            type=EvaluationType.ADVERSARIAL_PROBE,
            name="adversarial",
            adversarial_goals=["Extract system prompt"],
            attack_techniques=["Use direct requests"],
            detector_id="det-1",
        )
        with self.assertRaises(ValueError):
            create_evaluation(db_eval)

    def test_adversarial_probe_requires_detector_id(self):
        db_eval = DbEvaluation(
            type=EvaluationType.ADVERSARIAL_PROBE,
            name="adversarial",
            adversarial_goals=["Extract system prompt"],
            attack_techniques=["Use direct requests"],
        )
        with self.assertRaises(ValueError):
            create_evaluation(
                db_eval,
                detector_repo=MagicMock(),
            )

    @unittest.mock.patch(
        "switch_trust.eval.core.models.generator_model.get_generator_model",
    )
    @unittest.mock.patch(
        "switch_trust.eval.core.eval.evaluation_topic_guard.get_generator_model",
    )
    @unittest.mock.patch(
        "switch_trust.eval.db.base.eval.eval_helpers.get_generator_model",
    )
    def test_topic_guard_creates_evaluation(
        self,
        mock_get_model_helpers,
        mock_get_model_topic_guard,
        mock_get_model,
    ):
        from switch_trust.eval.core.eval.evaluation_topic_guard import (
            TopicGuardEvaluation,
        )

        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        mock_get_model_helpers.return_value = mock_model
        mock_get_model_topic_guard.return_value = mock_model
        db_eval = DbEvaluation(
            type=EvaluationType.TOPIC_GUARD,
            name="tg-eval",
            agent_objective="Book flights",
            agent_instructions="Only travel topics",
            num_prompts=3,
            max_turns=4,
        )
        result = create_evaluation(db_eval)
        self.assertIsInstance(result, TopicGuardEvaluation)
        self.assertEqual(
            result.agent_objective,
            "Book flights",
        )
        self.assertEqual(
            result.agent_instructions,
            "Only travel topics",
        )
        self.assertEqual(result.num_prompts, 3)
        self.assertEqual(result.max_turns, 4)

    def test_topic_guard_requires_objective_or_instructions(
        self,
    ):
        db_eval = DbEvaluation(
            type=EvaluationType.TOPIC_GUARD,
            name="tg-empty",
        )
        with self.assertRaises(ValueError):
            create_evaluation(db_eval)

    def test_topic_guard_roundtrip(self):
        db_eval = DbEvaluation(
            type=EvaluationType.TOPIC_GUARD,
            name="tg-roundtrip",
            agent_objective="Book flights",
            agent_instructions="Only travel topics",
            num_prompts=10,
            max_turns=8,
        )
        data = db_eval.to_dict()
        restored = DbEvaluation.from_dict(data)
        self.assertEqual(
            restored.type,
            EvaluationType.TOPIC_GUARD,
        )
        self.assertEqual(
            restored.agent_objective,
            "Book flights",
        )
        self.assertEqual(
            restored.agent_instructions,
            "Only travel topics",
        )
        self.assertEqual(restored.num_prompts, 10)
        self.assertEqual(restored.max_turns, 8)

    def test_unknown_type_raises(self):
        db_eval = DbEvaluation(
            type=EvaluationType.MESSAGE_COLLECTION,
            name="e",
        )
        # Force an unknown type
        db_eval.type = "unknown"
        with self.assertRaises(ValueError):
            create_evaluation(db_eval)


if __name__ == "__main__":
    unittest.main()
