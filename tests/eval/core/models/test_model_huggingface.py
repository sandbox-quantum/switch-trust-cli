import unittest
from unittest.mock import MagicMock, patch

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.models.model_huggingface import HuggingFaceModel


class TestHuggingFaceModel(unittest.IsolatedAsyncioTestCase):
    async def test_generate_text(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "Hello!"}]

        model = HuggingFaceModel(model=mock_pipeline)
        msg = Message(content=Content.text(Role.USER, "Hi"))
        resp = await model.generate(msg)

        self.assertIsNotNone(resp.message)
        self.assertEqual(resp.message.content.role, Role.ASSISTANT)
        self.assertEqual(resp.message.content.parts[0].text, "Hello!")
        mock_pipeline.assert_called_once()

    async def test_generate_passes_kwargs(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "Hi"}]

        model = HuggingFaceModel(model=mock_pipeline)
        msg = Message(content=Content.text(Role.USER, "Hi"))
        await model.generate(msg, max_new_tokens=512, temperature=0.8)

        call_kwargs = mock_pipeline.call_args
        self.assertEqual(call_kwargs.kwargs["max_new_tokens"], 512)
        self.assertEqual(call_kwargs.kwargs["temperature"], 0.8)

    async def test_default_return_full_text_false(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "output"}]

        model = HuggingFaceModel(model=mock_pipeline)
        msg = Message(content=Content.text(Role.USER, "prompt"))
        await model.generate(msg)

        call_kwargs = mock_pipeline.call_args
        self.assertFalse(call_kwargs.kwargs["return_full_text"])

    async def test_collects_multipart_text(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [{"generated_text": "response"}]

        from switch_trust.eval.common.schema import Part

        content = Content(
            role=Role.USER,
            parts=[
                Part.text_part("Hello"),
                Part.text_part("World"),
            ],
        )
        model = HuggingFaceModel(model=mock_pipeline)
        msg = Message(content=content)
        await model.generate(msg)

        prompt = mock_pipeline.call_args.args[0]
        self.assertEqual(prompt, "Hello World")

    @patch(
        "switch_trust.eval.core.models.model_huggingface."
        "require_transformers_pipeline"
    )
    def test_init_from_model_name(self, mock_require):
        mock_pipeline_fn = mock_require.return_value
        mock_pipeline_fn.return_value = MagicMock()
        HuggingFaceModel(model="gpt2")
        mock_pipeline_fn.assert_called_once_with("text-generation", model="gpt2")


if __name__ == "__main__":
    unittest.main()
