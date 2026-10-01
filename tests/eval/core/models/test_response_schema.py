import unittest

from pydantic import BaseModel

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.models.model import ModelResponse
from switch_trust.eval.core.models.response_schema import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    parse_json_text,
    parse_model_response,
    to_anthropic_tool,
    to_openai_response_format,
    to_strict_json_schema,
)


class _Score(BaseModel):
    score: float
    reason: str


def _response(text: str) -> ModelResponse:
    return ModelResponse(message=Message(content=Content.text(Role.ASSISTANT, text)))


class TestStrictJsonSchema(unittest.TestCase):
    def test_adds_additional_properties_and_required(self):
        schema = to_strict_json_schema(_Score)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(sorted(schema["required"]), ["reason", "score"])


class TestOpenAIResponseFormat(unittest.TestCase):
    def test_shape(self):
        rf = to_openai_response_format(_Score)
        self.assertEqual(rf["type"], "json_schema")
        self.assertEqual(rf["json_schema"]["name"], "_Score")
        self.assertTrue(rf["json_schema"]["strict"])


class TestAnthropicTool(unittest.TestCase):
    def test_shape(self):
        tool, tool_choice = to_anthropic_tool(_Score)
        self.assertEqual(tool["name"], STRUCTURED_OUTPUT_TOOL_NAME)
        self.assertIn("properties", tool["input_schema"])
        self.assertEqual(tool_choice["type"], "tool")
        self.assertEqual(tool_choice["name"], STRUCTURED_OUTPUT_TOOL_NAME)


class TestParseModelResponse(unittest.TestCase):
    def test_direct_json(self):
        result = parse_model_response(
            _response('{"score": 0.5, "reason": "ok"}'), _Score
        )
        self.assertEqual(result.score, 0.5)
        self.assertEqual(result.reason, "ok")

    def test_fenced_json_fallback(self):
        text = 'Here is the result:\n```json\n{"score": 1.0, "reason": "safe"}\n```'
        result = parse_model_response(_response(text), _Score)
        self.assertEqual(result.score, 1.0)

    def test_prose_wrapped_json_fallback(self):
        text = 'Reasoning first. {"score": 0.0, "reason": "unsafe"} done.'
        result = parse_json_text(text, _Score)
        self.assertEqual(result.reason, "unsafe")

    def test_invalid_raises_value_error(self):
        with self.assertRaises(ValueError):
            parse_model_response(_response("not json at all"), _Score)

    def test_missing_field_raises(self):
        with self.assertRaises(ValueError):
            parse_model_response(_response('{"score": 0.5}'), _Score)

    def test_none_message_raises(self):
        with self.assertRaises(ValueError):
            parse_model_response(ModelResponse(message=None), _Score)


if __name__ == "__main__":
    unittest.main()
