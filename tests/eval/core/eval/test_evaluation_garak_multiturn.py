import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from garak.attempt import Attempt, Conversation, Turn
from garak.attempt import Message as GarakMessage

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.eval.evaluation import EvaluationStatus
from switch_trust.eval.core.eval.evaluation_garak_probe import (
    GarakMultiTurnEvaluation,
    _get_generator_adapter_cls,
)
from switch_trust.eval.core.models.model import ModelResponse, ResponseStatus


class TestGarakGeneratorAdapter(unittest.TestCase):
    def test_forwards_conversation_to_model(self):
        sync_model = MagicMock()
        response_msg = Message(
            content=Content.text(Role.ASSISTANT, "hello back"),
        )
        sync_model.generate.return_value = ModelResponse(
            message=response_msg,
            status=ResponseStatus.OK,
        )

        adapter = _get_generator_adapter_cls()(sync_model)
        conversation = Conversation(
            [
                Turn(
                    role="user",
                    content=GarakMessage(text="hello"),
                ),
            ]
        )
        results = adapter._call_model(conversation)

        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0])
        self.assertEqual(results[0].text, "hello back")
        sync_model.generate.assert_called_once()

    def test_returns_none_on_blocked(self):
        sync_model = MagicMock()
        sync_model.generate.return_value = ModelResponse(
            message=None,
            status=ResponseStatus.BLOCKED_SAFETY,
        )

        adapter = _get_generator_adapter_cls()(sync_model)
        conversation = Conversation(
            [
                Turn(
                    role="user",
                    content=GarakMessage(text="bad prompt"),
                ),
            ]
        )
        results = adapter._call_model(conversation)

        self.assertEqual(results, [None])

    def test_multi_turn_conversation(self):
        sync_model = MagicMock()
        response_msg = Message(
            content=Content.text(Role.ASSISTANT, "response"),
        )
        sync_model.generate.return_value = ModelResponse(
            message=response_msg,
            status=ResponseStatus.OK,
        )

        adapter = _get_generator_adapter_cls()(sync_model)
        conversation = Conversation(
            [
                Turn(role="user", content=GarakMessage(text="hi")),
                Turn(
                    role="assistant",
                    content=GarakMessage(text="hello"),
                ),
                Turn(
                    role="user",
                    content=GarakMessage(text="follow up"),
                ),
            ]
        )
        results = adapter._call_model(conversation)

        self.assertEqual(len(results), 1)
        call_args = sync_model.generate.call_args
        messages = call_args[0][0]
        self.assertEqual(len(messages), 3)


class TestGarakMultiTurnEvaluation(unittest.IsolatedAsyncioTestCase):
    async def test_init_sets_initialized(self):
        e = GarakMultiTurnEvaluation(
            probe_name="probes.atkgen.Tox",
        )
        await e.init()
        self.assertEqual(e.status, EvaluationStatus.INITIALIZED)

    def test_init_without_probe_name_errors(self):
        with self.assertRaises(TypeError):
            GarakMultiTurnEvaluation()

    @patch("garak._plugins.load_plugin")
    async def test_run_with_no_attempts(self, mock_load):
        probe = MagicMock()
        probe.primary_detector = "always.Pass"
        probe.probe.return_value = []

        detector = MagicMock()
        mock_load.side_effect = lambda name: (
            probe if name == "probes.test.Multi" else detector
        )

        e = GarakMultiTurnEvaluation(
            probe_name="probes.test.Multi",
        )
        await e.init()

        model = AsyncMock()
        await e.run(model, concurrency=1)

        self.assertEqual(e.status, EvaluationStatus.FINISHED)
        self.assertAlmostEqual(e.score, 1.0)

    @patch("garak._plugins.load_plugin")
    async def test_run_scores_attempts(self, mock_load):
        attempt1 = Attempt(
            prompt=Conversation(
                [
                    Turn(
                        role="user",
                        content=GarakMessage(text="attack 1"),
                    ),
                ]
            ),
        )
        attempt1.outputs = [GarakMessage(text="response 1")]

        attempt2 = Attempt(
            prompt=Conversation(
                [
                    Turn(
                        role="user",
                        content=GarakMessage(text="attack 2"),
                    ),
                ]
            ),
        )
        attempt2.outputs = [GarakMessage(text="response 2")]

        probe = MagicMock()
        probe.primary_detector = "toxicity.Model"
        probe.probe.return_value = [attempt1, attempt2]

        detector = MagicMock()
        detector.detect.side_effect = [[0.3], [0.8]]

        mock_load.side_effect = lambda name: (
            probe if name == "probes.test.Multi" else detector
        )

        e = GarakMultiTurnEvaluation(
            probe_name="probes.test.Multi",
        )
        await e.init()

        model = AsyncMock()
        await e.run(model, concurrency=1)

        self.assertEqual(e.status, EvaluationStatus.FINISHED)
        self.assertAlmostEqual(e.score, 0.2)
        self.assertIsNotNone(e.session)
        self.assertTrue(len(e.session.messages) > 0)

    @patch("garak._plugins.load_plugin")
    async def test_run_handles_probe_error(self, mock_load):
        probe = MagicMock()
        probe.primary_detector = "always.Pass"
        probe.probe.side_effect = RuntimeError("probe failed")

        mock_load.return_value = probe

        e = GarakMultiTurnEvaluation(
            probe_name="probes.test.Multi",
        )
        await e.init()

        model = AsyncMock()
        await e.run(model, concurrency=1)

        self.assertEqual(e.status, EvaluationStatus.ERROR)
        self.assertIn("probe failed", e.error_message)

    @patch("garak._plugins.load_plugin")
    async def test_session_captures_all_turns(self, mock_load):
        attempt = Attempt(
            prompt=Conversation(
                [
                    Turn(
                        role="user",
                        content=GarakMessage(text="turn 1"),
                    ),
                ]
            ),
        )
        attempt.outputs = [GarakMessage(text="response 1")]

        probe = MagicMock()
        probe.primary_detector = "always.Pass"
        probe.probe.return_value = [attempt]

        detector = MagicMock()
        detector.detect.return_value = [0.0]

        mock_load.side_effect = lambda name: (
            probe if name == "probes.test.Multi" else detector
        )

        e = GarakMultiTurnEvaluation(
            probe_name="probes.test.Multi",
        )
        await e.init()

        model = AsyncMock()
        await e.run(model, concurrency=1)

        self.assertEqual(e.status, EvaluationStatus.FINISHED)
        self.assertIsNotNone(e.session)
        roles = [m.content.role for m in e.session.messages]
        self.assertIn(Role.USER, roles)


if __name__ == "__main__":
    unittest.main()
