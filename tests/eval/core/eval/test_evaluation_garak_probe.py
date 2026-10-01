import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.detectors.detector import DetectorResult
from switch_trust.eval.core.eval.evaluation import EvaluationStatus
from switch_trust.eval.core.eval.evaluation_garak_probe import (
    GarakMultiTurnEvaluation,
    GarakProbeEvaluation,
)
from switch_trust.eval.core.eval.evaluation_single_prompt import (
    SinglePromptEvaluation,
)
from switch_trust.eval.core.models.model import ModelResponse, ResponseStatus


class TestGarakProbeEvaluation(unittest.IsolatedAsyncioTestCase):
    @patch("garak._plugins.load_plugin")
    async def test_static_probe_creates_children(self, mock_load):
        probe = MagicMock()
        probe.prompts = [
            "prompt one",
            "prompt two",
            "prompt three",
        ]
        probe.primary_detector = "always.Pass"
        mock_load.return_value = probe

        e = GarakProbeEvaluation(
            probe_name="probes.test.Dummy",
        )
        await e.init()

        self.assertEqual(e.status, EvaluationStatus.INITIALIZED)
        self.assertEqual(len(e.children), 3)
        for child in e.children:
            self.assertIsInstance(child, SinglePromptEvaluation)

    @patch("garak._plugins.load_plugin")
    async def test_static_probe_runs_children(self, mock_load):
        probe = MagicMock()
        probe.prompts = ["prompt one"]
        probe.primary_detector = "always.Pass"
        mock_load.return_value = probe

        e = GarakProbeEvaluation(
            probe_name="probes.test.Dummy",
        )
        await e.init()

        model = AsyncMock()
        response_msg = Message(
            content=Content.text(Role.ASSISTANT, "ok"),
        )
        model.generate = AsyncMock(
            return_value=ModelResponse(
                message=response_msg,
                status=ResponseStatus.OK,
            ),
        )

        detector = AsyncMock()
        detector.detect = AsyncMock(
            return_value=DetectorResult(score=0.9),
        )
        e.children[0].detector = detector

        await e.run(model, concurrency=4)

        self.assertEqual(e.status, EvaluationStatus.FINISHED)
        self.assertAlmostEqual(
            e.get_summary().achieved_score,
            0.9,
        )

    @patch("garak._plugins.load_plugin")
    async def test_multiturn_probe_creates_single_child(
        self,
        mock_load,
    ):
        probe = MagicMock(spec=["primary_detector", "goal"])
        mock_load.return_value = probe

        e = GarakProbeEvaluation(
            probe_name="probes.atkgen.Tox",
        )
        await e.init()

        self.assertEqual(e.status, EvaluationStatus.INITIALIZED)
        self.assertEqual(len(e.children), 1)
        self.assertIsInstance(
            e.children[0],
            GarakMultiTurnEvaluation,
        )
        self.assertEqual(
            e.children[0].probe_name,
            "probes.atkgen.Tox",
        )

    @patch("garak._plugins.load_plugin")
    async def test_empty_prompts_treated_as_multiturn(
        self,
        mock_load,
    ):
        probe = MagicMock()
        probe.prompts = []
        mock_load.return_value = probe

        e = GarakProbeEvaluation(
            probe_name="probes.topic.WordnetControversial",
        )
        await e.init()

        self.assertEqual(len(e.children), 1)
        self.assertIsInstance(
            e.children[0],
            GarakMultiTurnEvaluation,
        )

    def test_init_without_probe_name_errors(self):
        with self.assertRaises(TypeError):
            GarakProbeEvaluation()


if __name__ == "__main__":
    unittest.main()
