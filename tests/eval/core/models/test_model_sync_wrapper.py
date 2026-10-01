import unittest
from unittest.mock import AsyncMock, MagicMock

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.models.model import ModelResponse
from switch_trust.eval.core.models.model_sync_wrapper import SyncModelWrapper


class TestSyncModelWrapper(unittest.TestCase):
    def test_generate_bridges_async_to_sync(self):
        expected_response = ModelResponse(
            message=Message(content=Content.text(Role.ASSISTANT, "Hello!")),
        )
        mock_model = MagicMock()
        mock_model.generate = AsyncMock(return_value=expected_response)

        wrapper = SyncModelWrapper(mock_model)
        msg = Message(content=Content.text(Role.USER, "Hi"))
        result = wrapper.generate(msg)

        self.assertIs(result, expected_response)
        mock_model.generate.assert_called_once_with(msg)
        wrapper.close()

    def test_generate_passes_kwargs(self):
        expected_response = ModelResponse(
            message=Message(content=Content.text(Role.ASSISTANT, "Hi")),
        )
        mock_model = MagicMock()
        mock_model.generate = AsyncMock(return_value=expected_response)

        wrapper = SyncModelWrapper(mock_model)
        msg = Message(content=Content.text(Role.USER, "Hi"))
        wrapper.generate(msg, temperature=0.5, max_tokens=100)

        call_kwargs = mock_model.generate.call_args
        self.assertEqual(call_kwargs.kwargs["temperature"], 0.5)
        self.assertEqual(call_kwargs.kwargs["max_tokens"], 100)
        wrapper.close()

    def test_generate_with_string_input(self):
        expected_response = ModelResponse(
            message=Message(content=Content.text(Role.ASSISTANT, "Hi")),
        )
        mock_model = MagicMock()
        mock_model.generate = AsyncMock(return_value=expected_response)

        wrapper = SyncModelWrapper(mock_model)
        result = wrapper.generate("Hi")

        self.assertIs(result, expected_response)
        mock_model.generate.assert_called_once_with("Hi")
        wrapper.close()

    def test_close_closes_loop(self):
        mock_model = MagicMock()
        wrapper = SyncModelWrapper(mock_model)

        loop = wrapper._loop
        self.assertFalse(loop.is_closed())

        wrapper.close()
        self.assertTrue(loop.is_closed())

    def test_del_closes_loop(self):
        mock_model = MagicMock()
        wrapper = SyncModelWrapper(mock_model)
        loop = wrapper._loop
        self.assertFalse(loop.is_closed())
        wrapper.__del__()
        self.assertTrue(loop.is_closed())

    def test_del_after_close_is_safe(self):
        mock_model = MagicMock()
        wrapper = SyncModelWrapper(mock_model)
        wrapper.close()
        wrapper.__del__()

    def test_generate_after_close_raises(self):
        expected_response = ModelResponse(
            message=Message(content=Content.text(Role.ASSISTANT, "Hi")),
        )
        mock_model = MagicMock()
        mock_model.generate = AsyncMock(return_value=expected_response)

        wrapper = SyncModelWrapper(mock_model)
        wrapper.close()

        with self.assertRaises(RuntimeError):
            wrapper.generate("Hi")


if __name__ == "__main__":
    unittest.main()
