import unittest
from unittest.mock import AsyncMock

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.detectors.detector_model import (
    _EVIDENCE_END,
    _EVIDENCE_START,
)
from switch_trust.eval.core.detectors.detector_model_adversarial import (
    AdversarialModelDetector,
    DetectorLLMResponse,
)
from switch_trust.eval.core.models.model import ModelResponse


def _model_returning(text: str) -> AsyncMock:
    model = AsyncMock()
    model.generate = AsyncMock(
        return_value=ModelResponse(
            message=Message(content=Content.text(Role.ASSISTANT, text))
        )
    )
    return model


def _target_response(text: str) -> ModelResponse:
    return ModelResponse(message=Message(content=Content.text(Role.ASSISTANT, text)))


class TestAdversarialModelDetector(unittest.IsolatedAsyncioTestCase):
    async def test_parses_score_and_passes_schema(self):
        model = _model_returning('{"score": 0.2, "reason": "complied"}')
        detector = AdversarialModelDetector(
            model=model, prompt="judge", include_conversation=False
        )

        result = await detector.detect(
            _target_response("sure, here you go"), goal="jailbreak", conversation=[]
        )

        self.assertAlmostEqual(result.score, 0.2)
        # The detector requests structured output.
        self.assertIs(
            model.generate.await_args.kwargs["output_schema"], DetectorLLMResponse
        )

    async def test_parses_fenced_json(self):
        model = _model_returning('```json\n{"score": 0.9, "reason": "resisted"}\n```')
        detector = AdversarialModelDetector(
            model=model, prompt="judge", include_conversation=False
        )

        result = await detector.detect(
            _target_response("no"), goal="jailbreak", conversation=[]
        )
        self.assertAlmostEqual(result.score, 0.9)

    async def test_invalid_output_raises(self):
        model = _model_returning("not json at all")
        detector = AdversarialModelDetector(
            model=model, prompt="judge", include_conversation=False
        )

        with self.assertRaises(ValueError):
            await detector.detect(
                _target_response("hmm"), goal="jailbreak", conversation=[]
            )

    async def test_untrusted_response_is_fenced_in_a_user_turn(self):
        model = _model_returning('{"score": 0.1, "reason": "ok"}')
        detector = AdversarialModelDetector(
            model=model, prompt="judge", include_conversation=False
        )
        await detector.detect(
            _target_response("attacker-controlled reply"),
            goal="jailbreak",
            conversation=[],
        )

        messages = model.generate.call_args[0][0]
        system_text = messages[0].content.parts[0].text
        user_text = messages[-1].content.parts[0].text
        # The evidence-fencing contract is declared in the system turn; the
        # untrusted response is only present between the markers in the user
        # turn, never inlined into the judge instruction.
        self.assertIn(_EVIDENCE_START, system_text)
        self.assertIn(_EVIDENCE_END, system_text)
        self.assertNotIn("attacker-controlled reply", system_text)
        self.assertEqual(
            user_text,
            f"{_EVIDENCE_START}\nattacker-controlled reply\n{_EVIDENCE_END}",
        )

    async def test_conversation_is_fenced_when_included(self):
        model = _model_returning('{"score": 0.5, "reason": "ok"}')
        detector = AdversarialModelDetector(
            model=model, prompt="judge", include_conversation=True
        )
        conversation = [
            Message(content=Content.text(Role.USER, "attacker turn")),
            Message(content=Content.text(Role.ASSISTANT, "target turn")),
        ]
        await detector.detect(
            _target_response("last reply"),
            goal="jailbreak",
            conversation=conversation,
        )

        messages = model.generate.call_args[0][0]
        conversation_text = messages[1].content.parts[0].text
        self.assertIn(_EVIDENCE_START, conversation_text)
        self.assertIn(_EVIDENCE_END, conversation_text)


if __name__ == "__main__":
    unittest.main()
