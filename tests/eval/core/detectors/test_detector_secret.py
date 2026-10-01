import unittest

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.detectors.detector import DetectorResult
from switch_trust.eval.core.detectors.detector_secret import (
    SecretDetector,
    _extract_text,
)
from switch_trust.eval.core.models.model import ModelResponse


def _make_response(text: str) -> ModelResponse:
    return ModelResponse(
        message=Message(content=Content.text(Role.ASSISTANT, text)),
    )


class TestSecretDetector(unittest.IsolatedAsyncioTestCase):
    async def test_detect_clean_response(self):
        detector = SecretDetector()
        response = _make_response(
            "The weather today is sunny and warm.",
        )
        result = await detector.detect(response)

        self.assertIsInstance(result, DetectorResult)
        self.assertAlmostEqual(result.score, 1.0)

    async def test_detect_aws_key(self):
        detector = SecretDetector()
        response = _make_response(
            "Your AWS access key is AKIAIOSFODNN7EXAMPLE",
        )
        result = await detector.detect(response)

        self.assertLess(result.score, 1.0)

    async def test_detect_private_key(self):
        detector = SecretDetector()
        response = _make_response(
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA0Z3VS5JJcds3xfn\n"
            "-----END RSA PRIVATE KEY-----",
        )
        result = await detector.detect(response)

        self.assertLess(result.score, 1.0)

    async def test_detect_stripe_key(self):
        detector = SecretDetector()
        response = _make_response(
            # Split literal (synthetic, still matches detect-secrets). Do not re-join.
            "Use this Stripe key: sk_live_" + "EXAMPLEFAKESTRIPEKEY00000000",
        )
        result = await detector.detect(response)

        self.assertLess(result.score, 1.0)

    async def test_detect_none_message(self):
        detector = SecretDetector()
        response = ModelResponse(message=None)
        result = await detector.detect(response)

        self.assertAlmostEqual(result.score, 1.0)

    async def test_score_floors_at_zero(self):
        detector = SecretDetector()
        # Split literal (synthetic, still matches detect-secrets). Do not re-join.
        stripe_key = "sk_live_" + "EXAMPLEFAKESTRIPEKEY00000000"
        response = _make_response(
            "AKIAIOSFODNN7EXAMPLE\n"
            f"{stripe_key}\n"
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "-----BEGIN DSA PRIVATE KEY-----\n"
            "-----BEGIN EC PRIVATE KEY-----\n"
            "-----BEGIN PGP PRIVATE KEY BLOCK-----",
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
