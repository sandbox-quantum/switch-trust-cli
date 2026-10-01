import unittest
from unittest.mock import MagicMock, patch

from switch_trust.eval.db.base.detectors.detector_helpers import create_detector
from switch_trust.eval.db.base.detectors.detector_types import (
    DbDetector,
    DetectorType,
)


def _db_detector(**overrides) -> DbDetector:
    defaults = {
        "type": DetectorType.MODEL,
        "name": "test-detector",
    }
    defaults.update(overrides)
    return DbDetector(**defaults)


class TestCreateDetectorGarak(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.detectors.detector_garak.GarakDetector",
    )
    def test_garak_creates_detector(self, MockGarak):
        db = _db_detector(
            type=DetectorType.GARAK,
            detector_name="toxicity.ToxicityDetector",
        )
        result = create_detector(db)
        MockGarak.assert_called_once_with(
            "toxicity.ToxicityDetector",
        )
        self.assertEqual(result, MockGarak.return_value)

    def test_garak_without_detector_name_raises(self):
        db = _db_detector(
            type=DetectorType.GARAK,
            detector_name=None,
        )
        with self.assertRaises(ValueError):
            create_detector(db)


class TestCreateDetectorModel(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.models.generator_model.get_generator_model",
    )
    @patch(
        "switch_trust.eval.core.detectors.detector_model.ModelDetector",
    )
    def test_model_creates_detector(
        self,
        MockDetector,
        mock_get_model,
    ):
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        db = _db_detector(type=DetectorType.MODEL)
        result = create_detector(db)
        mock_get_model.assert_called_once()
        MockDetector.assert_called_once_with(model=mock_model)
        self.assertEqual(result, MockDetector.return_value)

    @patch(
        "switch_trust.eval.core.models.generator_model.get_generator_model",
    )
    @patch(
        "switch_trust.eval.core.detectors.detector_model.ModelDetector",
    )
    def test_model_with_prompt(
        self,
        MockDetector,
        mock_get_model,
    ):
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        db = _db_detector(
            type=DetectorType.MODEL,
            prompt="Check for safety",
        )
        create_detector(db)
        MockDetector.assert_called_once_with(
            model=mock_model,
            prompt="Check for safety",
        )

    @patch(
        "switch_trust.eval.core.models.generator_model.get_generator_model",
    )
    @patch(
        "switch_trust.eval.core.detectors.detector_model.ModelDetector",
    )
    def test_model_without_prompt(
        self,
        MockDetector,
        mock_get_model,
    ):
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        db = _db_detector(
            type=DetectorType.MODEL,
            prompt=None,
        )
        create_detector(db)
        MockDetector.assert_called_once_with(model=mock_model)


class TestCreateDetectorPII(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.detectors.detector_pii.PIIDetector",
    )
    def test_pii_creates_detector(self, MockPII):
        db = _db_detector(type=DetectorType.PII)
        result = create_detector(db)
        MockPII.assert_called_once_with()
        self.assertEqual(result, MockPII.return_value)


class TestCreateDetectorSecret(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.detectors.detector_secret.SecretDetector",
    )
    def test_secret_creates_detector(self, MockSecret):
        db = _db_detector(type=DetectorType.SECRET)
        result = create_detector(db)
        MockSecret.assert_called_once_with()
        self.assertEqual(result, MockSecret.return_value)


class TestCreateDetectorTopicGuard(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.models.generator_model.get_generator_model",
    )
    @patch(
        "switch_trust.eval.core.detectors.detector_topic_guard.TopicGuardDetector",
    )
    def test_topic_guard_creates_detector(
        self,
        MockTG,
        mock_get_model,
    ):
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        db = _db_detector(
            type=DetectorType.TOPIC_GUARD,
            agent_objective="Book flights",
            agent_instructions="Only travel",
        )
        result = create_detector(db)
        MockTG.assert_called_once_with(
            model=mock_model,
            agent_objective="Book flights",
            agent_instructions="Only travel",
        )
        self.assertEqual(result, MockTG.return_value)

    @patch(
        "switch_trust.eval.core.models.generator_model.get_generator_model",
    )
    @patch(
        "switch_trust.eval.core.detectors.detector_topic_guard.TopicGuardDetector",
    )
    def test_topic_guard_with_only_objective(
        self,
        MockTG,
        mock_get_model,
    ):
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        db = _db_detector(
            type=DetectorType.TOPIC_GUARD,
            agent_objective="Book flights",
        )
        create_detector(db)
        MockTG.assert_called_once_with(
            model=mock_model,
            agent_objective="Book flights",
            agent_instructions=None,
        )

    @patch(
        "switch_trust.eval.core.models.generator_model.get_generator_model",
    )
    @patch(
        "switch_trust.eval.core.detectors.detector_topic_guard.TopicGuardDetector",
    )
    def test_topic_guard_with_only_instructions(
        self,
        MockTG,
        mock_get_model,
    ):
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model
        db = _db_detector(
            type=DetectorType.TOPIC_GUARD,
            agent_instructions="Only travel topics",
        )
        create_detector(db)
        MockTG.assert_called_once_with(
            model=mock_model,
            agent_objective=None,
            agent_instructions="Only travel topics",
        )

    def test_topic_guard_without_objective_or_instructions_raises(
        self,
    ):
        db = _db_detector(type=DetectorType.TOPIC_GUARD)
        with self.assertRaises(ValueError):
            create_detector(db)


class TestCreateDetectorUnknown(unittest.TestCase):
    def test_unknown_type_raises(self):
        db = _db_detector()
        db.type = "unknown_type"
        with self.assertRaises(ValueError):
            create_detector(db)


if __name__ == "__main__":
    unittest.main()
