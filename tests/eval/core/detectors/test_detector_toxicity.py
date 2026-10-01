import unittest
from unittest.mock import MagicMock, patch

from switch_trust.eval.common.schema import Content, Message, Part, Role
from switch_trust.eval.core.models.model import ModelResponse


def _make_response(text: str) -> ModelResponse:
    msg = Message(content=Content.text(Role.ASSISTANT, text))
    return ModelResponse(msg)


def _make_response_no_message() -> ModelResponse:
    return ModelResponse(message=None)


def _make_response_no_text_parts() -> ModelResponse:
    """Response with only non-text parts (e.g. tool calls)."""
    from switch_trust.eval.common.schema import ToolCall

    part = Part.tool_call_part(ToolCall(id="tc1", name="fn", arguments={}))
    content = Content(role=Role.ASSISTANT, parts=[part])
    msg = Message(content=content)
    return ModelResponse(msg)


@patch(
    "switch_trust.eval.core.detectors.detector_toxicity.require_transformers_pipeline"
)
class TestToxicityDetector(unittest.IsolatedAsyncioTestCase):
    async def test_toxic_label_returns_inverted_confidence(self, mock_require):
        # require_transformers_pipeline() returns the pipeline *factory*, which
        # is then called to build the classifier.
        mock_factory = mock_require.return_value
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "toxic", "score": 0.9}]
        mock_factory.return_value = mock_pipeline

        from switch_trust.eval.core.detectors.detector_toxicity import (
            ToxicityDetector,
        )

        detector = ToxicityDetector()

        result = await detector.detect(_make_response("you are terrible"))
        self.assertAlmostEqual(result.score, 0.1)

    async def test_neutral_label_returns_confidence_directly(self, mock_require):
        mock_factory = mock_require.return_value
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "neutral", "score": 0.95}]
        mock_factory.return_value = mock_pipeline

        from switch_trust.eval.core.detectors.detector_toxicity import (
            ToxicityDetector,
        )

        detector = ToxicityDetector()

        result = await detector.detect(_make_response("hello world"))
        self.assertAlmostEqual(result.score, 0.95)

    async def test_none_message_returns_safe(self, mock_require):
        mock_pipeline = MagicMock()
        mock_require.return_value.return_value = mock_pipeline

        from switch_trust.eval.core.detectors.detector_toxicity import (
            ToxicityDetector,
        )

        detector = ToxicityDetector()

        result = await detector.detect(_make_response_no_message())
        self.assertEqual(result.score, 1.0)
        mock_pipeline.assert_not_called()

    async def test_no_text_parts_returns_safe(self, mock_require):
        mock_pipeline = MagicMock()
        mock_require.return_value.return_value = mock_pipeline

        from switch_trust.eval.core.detectors.detector_toxicity import (
            ToxicityDetector,
        )

        detector = ToxicityDetector()

        result = await detector.detect(_make_response_no_text_parts())
        self.assertEqual(result.score, 1.0)
        mock_pipeline.assert_not_called()

    async def test_toxic_high_confidence(self, mock_require):
        """Toxic label with very high confidence should produce near-zero score."""
        mock_factory = mock_require.return_value
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"label": "toxic", "score": 0.99}]
        mock_factory.return_value = mock_pipeline

        from switch_trust.eval.core.detectors.detector_toxicity import (
            ToxicityDetector,
        )

        detector = ToxicityDetector()

        result = await detector.detect(_make_response("horrible stuff"))
        self.assertAlmostEqual(result.score, 0.01)

    async def test_custom_model_name(self, mock_require):
        from switch_trust.eval.core.detectors.detector_toxicity import (
            ToxicityDetector,
        )

        mock_factory = mock_require.return_value
        ToxicityDetector(model_name="custom/model")
        mock_factory.assert_called_once_with(
            "text-classification",
            model="custom/model",
            truncation=True,
        )


class TestExtractText(unittest.TestCase):
    def test_single_text_part(self):
        from switch_trust.eval.core.detectors.detector_toxicity import _extract_text

        response = _make_response("hello")
        self.assertEqual(_extract_text(response), "hello")

    def test_multiple_text_parts(self):
        from switch_trust.eval.core.detectors.detector_toxicity import _extract_text

        parts = [Part.text_part("hello"), Part.text_part("world")]
        content = Content(role=Role.ASSISTANT, parts=parts)
        msg = Message(content=content)
        response = ModelResponse(msg)
        self.assertEqual(_extract_text(response), "hello\nworld")

    def test_none_message(self):
        from switch_trust.eval.core.detectors.detector_toxicity import _extract_text

        response = _make_response_no_message()
        self.assertEqual(_extract_text(response), "")

    def test_mixed_parts_only_text_extracted(self):
        from switch_trust.eval.common.schema import ToolCall
        from switch_trust.eval.core.detectors.detector_toxicity import _extract_text

        parts = [
            Part.text_part("hello"),
            Part.tool_call_part(ToolCall(id="t1", name="fn", arguments={})),
            Part.text_part("world"),
        ]
        content = Content(role=Role.ASSISTANT, parts=parts)
        msg = Message(content=content)
        response = ModelResponse(msg)
        self.assertEqual(_extract_text(response), "hello\nworld")


if __name__ == "__main__":
    unittest.main()
