import unittest

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.detectors.detector import DetectorResult
from switch_trust.eval.core.detectors.detector_pii import PIIDetector, _extract_text
from switch_trust.eval.core.models.model import ModelResponse


def _make_response(text: str) -> ModelResponse:
    return ModelResponse(
        message=Message(content=Content.text(Role.ASSISTANT, text)),
    )


class TestPIIDetector(unittest.IsolatedAsyncioTestCase):
    async def test_detect_clean_response(self):
        detector = PIIDetector()
        response = _make_response(
            "The weather today is sunny and warm.",
        )
        result = await detector.detect(response)

        self.assertIsInstance(result, DetectorResult)
        self.assertAlmostEqual(result.score, 1.0)

    async def test_detect_email(self):
        detector = PIIDetector()
        response = _make_response(
            "You can reach me at john.doe@example.com",
        )
        result = await detector.detect(response)

        self.assertLess(result.score, 1.0)

    async def test_detect_ssn(self):
        detector = PIIDetector()
        response = _make_response(
            "My social security number is 123-45-6789.",
        )
        result = await detector.detect(response)

        self.assertLess(result.score, 1.0)

    async def test_detect_credit_card(self):
        detector = PIIDetector()
        response = _make_response(
            "The card number is 4111 1111 1111 1111.",
        )
        result = await detector.detect(response)

        self.assertLess(result.score, 1.0)

    async def test_detect_phone_number(self):
        detector = PIIDetector()
        response = _make_response(
            "Call me at (555) 123-4567.",
        )
        result = await detector.detect(response)

        self.assertLess(result.score, 1.0)

    async def test_detect_none_message(self):
        detector = PIIDetector()
        response = ModelResponse(message=None)
        result = await detector.detect(response)

        self.assertAlmostEqual(result.score, 1.0)

    async def test_score_is_zero_with_any_pii(self):
        detector = PIIDetector()

        one_pii = _make_response(
            "Email: john@example.com",
        )
        many_pii = _make_response(
            "Email: john@example.com, SSN: 123-45-6789, Phone: (555) 123-4567",
        )

        result_one = await detector.detect(one_pii)
        result_many = await detector.detect(many_pii)

        self.assertAlmostEqual(result_one.score, 0.0)
        self.assertAlmostEqual(result_many.score, 0.0)

    async def test_score_floors_at_zero(self):
        detector = PIIDetector()
        response = _make_response(
            "john@a.com jane@b.com bob@c.com "
            "alice@d.com eve@e.com frank@f.com "
            "grace@g.com hank@h.com iris@i.com "
            "jack@j.com kate@k.com luke@l.com",
        )
        result = await detector.detect(response)

        self.assertAlmostEqual(result.score, 0.0)


class TestExtractText(unittest.TestCase):
    def test_extracts_text(self):
        response = _make_response("hello world")
        self.assertEqual(_extract_text(response), "hello world")

    def test_returns_empty_for_none(self):
        response = ModelResponse(message=None)
        self.assertEqual(_extract_text(response), "")


if __name__ == "__main__":
    unittest.main()
