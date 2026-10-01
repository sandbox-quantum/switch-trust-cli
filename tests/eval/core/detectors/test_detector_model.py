import unittest
from unittest.mock import AsyncMock, MagicMock

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.detectors.detector_model import (
    _EVIDENCE_END,
    _EVIDENCE_START,
    ModelDetector,
    _extract_text,
    _parse_score,
)
from switch_trust.eval.core.models.model import ModelResponse


def _make_response(text: str) -> ModelResponse:
    return ModelResponse(message=Message(content=Content.text(Role.ASSISTANT, text)))


class TestExtractText(unittest.TestCase):
    def test_extracts_text(self):
        response = _make_response("hello")
        self.assertEqual(_extract_text(response), "hello")

    def test_none_message(self):
        response = ModelResponse(message=None)
        self.assertEqual(_extract_text(response), "")


class TestParseScore(unittest.TestCase):
    def test_parses_start_anchored_number(self):
        self.assertEqual(_parse_score(_make_response("0.8")), 0.8)

    def test_parses_named_score_field(self):
        self.assertEqual(_parse_score(_make_response("SCORE: 0.7")), 0.7)

    def test_parses_named_score_field_case_insensitive(self):
        self.assertEqual(_parse_score(_make_response("Score = 0.4")), 0.4)

    def test_parses_named_score_field_ignoring_prose_prefix(self):
        response = _make_response("Here is my verdict:\nSCORE: 0.9\nrationale...")
        self.assertEqual(_parse_score(response), 0.9)

    def test_parses_one(self):
        self.assertEqual(_parse_score(_make_response("1.0")), 1.0)

    def test_parses_zero(self):
        self.assertEqual(_parse_score(_make_response("0.0")), 0.0)

    def test_clamps_above_one(self):
        self.assertEqual(_parse_score(_make_response("SCORE: 99.0")), 1.0)

    def test_clamps_below_zero(self):
        self.assertEqual(_parse_score(_make_response("SCORE: -0.5")), 0.0)

    def test_rejects_number_in_prose(self):
        # Guards against the original first-match-anywhere regex, which would
        # have picked "0" or "1" out of the judge's rationale and reported it
        # as the verdict.
        response = _make_response("The response mentions 1 unsafe action.")
        self.assertIsNone(_parse_score(response))

    def test_returns_none_for_unparseable(self):
        self.assertIsNone(_parse_score(_make_response("unsafe")))

    def test_returns_none_for_none_message(self):
        response = ModelResponse(message=None)
        self.assertIsNone(_parse_score(response))


class TestModelDetector(unittest.IsolatedAsyncioTestCase):
    def _make_detector(self, score_text="1.0", prompt=None):
        model = MagicMock()
        model.generate = AsyncMock(return_value=_make_response(score_text))
        kwargs = {}
        if prompt is not None:
            kwargs["prompt"] = prompt
        detector = ModelDetector(model=model, **kwargs)
        return detector, model

    async def test_safe_response(self):
        detector, _ = self._make_detector("1.0")
        result = await detector.detect(_make_response("safe text"))
        self.assertEqual(result.score, 1.0)

    async def test_unsafe_response(self):
        detector, _ = self._make_detector("0.0")
        result = await detector.detect(_make_response("unsafe text"))
        self.assertEqual(result.score, 0.0)

    async def test_partial_score(self):
        detector, _ = self._make_detector("0.6")
        result = await detector.detect(_make_response("borderline"))
        self.assertAlmostEqual(result.score, 0.6)

    async def test_untrusted_response_is_fenced_in_a_user_turn(self):
        detector, model = self._make_detector("1.0")
        await detector.detect(_make_response("test content"))
        messages = model.generate.call_args[0][0]
        self.assertEqual(len(messages), 2)
        system_text = messages[0].content.parts[0].text
        user_text = messages[1].content.parts[0].text
        # Instruction lives in the system turn; the untrusted response lives in
        # a separate user turn between the evidence markers.
        self.assertIn(_EVIDENCE_START, system_text)
        self.assertIn(_EVIDENCE_END, system_text)
        self.assertNotIn("test content", system_text)
        self.assertEqual(user_text, f"{_EVIDENCE_START}\ntest content\n{_EVIDENCE_END}")

    async def test_custom_prompt_lives_in_system_turn(self):
        detector, model = self._make_detector("0.5", prompt="Rate toxicity:")
        await detector.detect(_make_response("some text"))
        messages = model.generate.call_args[0][0]
        system_text = messages[0].content.parts[0].text
        self.assertTrue(system_text.startswith("Rate toxicity:"))

    async def test_unparseable_model_output_raises(self):
        # No default 0.0: an unparseable verdict errors the leaf so the run
        # does not record a false zero as a real "unsafe" finding.
        detector, _ = self._make_detector("I think it's somewhat safe")
        with self.assertRaises(ValueError):
            await detector.detect(_make_response("text"))

    async def test_model_exception_propagates(self):
        model = MagicMock()
        model.generate = AsyncMock(side_effect=RuntimeError("model error"))
        detector = ModelDetector(model=model)
        with self.assertRaises(RuntimeError):
            await detector.detect(_make_response("text"))


if __name__ == "__main__":
    unittest.main()
