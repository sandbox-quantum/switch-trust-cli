import unittest

from google.genai import types as genai_types

from switch_trust.eval.common import converter_genai as genai_converter
from switch_trust.eval.common.schema import (
    Content,
    Part,
    PartType,
    Role,
    ToolCall,
    ToolResult,
)


class TestGenAIConverter(unittest.TestCase):
    # -- to_content from dict -------------------------------------------------

    def test_user_text_dict(self):
        msg = {"role": "user", "parts": [{"text": "Hello"}]}
        content = genai_converter.to_content(msg)
        self.assertEqual(content.role, Role.USER)
        self.assertEqual(content.parts[0].text, "Hello")

    def test_model_text_dict(self):
        msg = {"role": "model", "parts": [{"text": "Hi there"}]}
        content = genai_converter.to_content(msg)
        self.assertEqual(content.role, Role.ASSISTANT)

    def test_string_parts_dict(self):
        msg = {"role": "user", "parts": ["Hello world"]}
        content = genai_converter.to_content(msg)
        self.assertEqual(content.parts[0].text, "Hello world")

    def test_function_call_dict(self):
        msg = {
            "role": "model",
            "parts": [
                {
                    "function_call": {
                        "id": "fc_1",
                        "name": "get_weather",
                        "args": {"city": "Paris"},
                    },
                },
            ],
        }
        content = genai_converter.to_content(msg)
        self.assertEqual(content.parts[0].part_type, PartType.TOOL_CALL)
        self.assertEqual(content.parts[0].tool_call.name, "get_weather")
        self.assertEqual(
            content.parts[0].tool_call.arguments,
            {"city": "Paris"},
        )

    def test_function_response_dict(self):
        msg = {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "id": "fc_1",
                        "name": "get_weather",
                        "response": {"temp": "22C"},
                    },
                },
            ],
        }
        content = genai_converter.to_content(msg)
        self.assertEqual(content.parts[0].part_type, PartType.TOOL_RESULT)
        self.assertEqual(content.parts[0].tool_result.tool_call_id, "fc_1")

    def test_thought_dict(self):
        msg = {
            "role": "model",
            "parts": [{"text": "Reasoning...", "thought": True}],
        }
        content = genai_converter.to_content(msg)
        self.assertEqual(content.parts[0].part_type, PartType.THINKING)
        self.assertEqual(content.parts[0].text, "Reasoning...")

    # -- to_content from object -----------------------------------------------

    def test_content_object(self):
        obj = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text="Hello")],
        )
        content = genai_converter.to_content(obj)
        self.assertEqual(content.role, Role.USER)
        self.assertEqual(content.parts[0].text, "Hello")

    def test_content_object_function_call(self):
        obj = genai_types.Content(
            role="model",
            parts=[
                genai_types.Part(
                    function_call=genai_types.FunctionCall(
                        id="fc_2",
                        name="search",
                        args={"q": "test"},
                    ),
                ),
            ],
        )
        content = genai_converter.to_content(obj)
        self.assertEqual(content.parts[0].part_type, PartType.TOOL_CALL)
        self.assertEqual(content.parts[0].tool_call.name, "search")

    def test_content_object_thought(self):
        obj = genai_types.Content(
            role="model",
            parts=[genai_types.Part(text="thinking...", thought=True)],
        )
        content = genai_converter.to_content(obj)
        self.assertEqual(content.parts[0].part_type, PartType.THINKING)

    # -- from_content ---------------------------------------------------------

    def test_from_user_text(self):
        content = Content.text(Role.USER, "Hello")
        result = genai_converter.from_content(content)
        self.assertEqual(result["role"], "user")
        self.assertEqual(result["parts"][0]["text"], "Hello")

    def test_from_assistant_text(self):
        content = Content.text(Role.ASSISTANT, "Hi")
        result = genai_converter.from_content(content)
        self.assertEqual(result["role"], "model")

    def test_from_tool_call(self):
        tc = ToolCall(id="c1", name="fn", arguments={"x": 1})
        content = Content(
            role=Role.ASSISTANT,
            parts=[Part.tool_call_part(tc)],
        )
        result = genai_converter.from_content(content)
        fc = result["parts"][0]["function_call"]
        self.assertEqual(fc["name"], "fn")
        self.assertEqual(fc["args"], {"x": 1})

    def test_from_tool_result(self):
        tr = ToolResult(tool_call_id="c1", content={"result": "ok"})
        content = Content(
            role=Role.USER,
            parts=[Part.tool_result_part(tr)],
        )
        result = genai_converter.from_content(content)
        fr = result["parts"][0]["function_response"]
        self.assertEqual(fr["id"], "c1")
        self.assertEqual(fr["response"], {"result": "ok"})

    def test_from_thinking(self):
        content = Content.thinking("deep thoughts")
        result = genai_converter.from_content(content)
        part = result["parts"][0]
        self.assertEqual(part["text"], "deep thoughts")
        self.assertTrue(part["thought"])

    def test_from_tool_result_string(self):
        tr = ToolResult(tool_call_id="c1", content="plain text")
        content = Content(
            role=Role.USER,
            parts=[Part.tool_result_part(tr)],
        )
        result = genai_converter.from_content(content)
        fr = result["parts"][0]["function_response"]
        self.assertEqual(fr["response"], {"result": "plain text"})

    # -- from_content_obj -----------------------------------------------------

    def test_from_content_obj_text(self):
        content = Content.text(Role.USER, "Hello")
        obj = genai_converter.from_content_obj(content)
        self.assertIsInstance(obj, genai_types.Content)
        self.assertEqual(obj.role, "user")
        self.assertEqual(obj.parts[0].text, "Hello")

    def test_from_content_obj_function_call(self):
        tc = ToolCall(id="c1", name="fn", arguments={"x": 1})
        content = Content(
            role=Role.ASSISTANT,
            parts=[Part.tool_call_part(tc)],
        )
        obj = genai_converter.from_content_obj(content)
        self.assertIsInstance(obj, genai_types.Content)
        self.assertEqual(obj.parts[0].function_call.name, "fn")

    def test_from_content_obj_thought(self):
        content = Content.thinking("deep thoughts")
        obj = genai_converter.from_content_obj(content)
        self.assertTrue(obj.parts[0].thought)
        self.assertEqual(obj.parts[0].text, "deep thoughts")

    # -- from_content with system role ----------------------------------------

    def test_from_system_text(self):
        content = Content.text(Role.SYSTEM, "Be helpful")
        result = genai_converter.from_content(content)
        # SYSTEM is not in _ROLE_MAP_REVERSE, falls back to "user"
        self.assertEqual(result["role"], "user")
        self.assertEqual(result["parts"][0]["text"], "Be helpful")

    # -- from_message_obj -----------------------------------------------------

    def test_from_message_obj(self):
        from switch_trust.eval.common.schema import Message

        msg = Message(content=Content.text(Role.USER, "hi"))
        obj = genai_converter.from_message_obj(msg)
        self.assertIsInstance(obj, genai_types.Content)
        self.assertEqual(obj.role, "user")
        self.assertEqual(obj.parts[0].text, "hi")

    # -- to_content with Content object with empty/None parts ----------------

    def test_content_object_empty_parts(self):
        obj = genai_types.Content(role="user", parts=[])
        content = genai_converter.to_content(obj)
        self.assertEqual(content.role, Role.USER)
        self.assertEqual(len(content.parts), 1)
        self.assertEqual(content.parts[0].text, "")

    def test_content_object_none_parts(self):
        obj = genai_types.Content(role="user", parts=None)
        content = genai_converter.to_content(obj)
        self.assertEqual(content.role, Role.USER)
        self.assertEqual(len(content.parts), 1)
        self.assertEqual(content.parts[0].text, "")

    # -- to_content dict with genai Part objects in parts list ----------------

    def test_dict_with_genai_part_objects(self):
        msg = {
            "role": "user",
            "parts": [genai_types.Part(text="from obj")],
        }
        content = genai_converter.to_content(msg)
        self.assertEqual(content.parts[0].text, "from obj")

    # -- to_content dict with empty parts ------------------------------------

    def test_dict_with_empty_parts(self):
        msg = {"role": "user", "parts": []}
        content = genai_converter.to_content(msg)
        self.assertEqual(len(content.parts), 1)
        self.assertEqual(content.parts[0].text, "")

    # -- to_content: Content object with function_response -------------------

    def test_content_object_function_response(self):
        obj = genai_types.Content(
            role="user",
            parts=[
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        id="fr_1",
                        name="get_temp",
                        response={"celsius": 22},
                    ),
                ),
            ],
        )
        content = genai_converter.to_content(obj)
        self.assertEqual(content.parts[0].part_type, PartType.TOOL_RESULT)
        self.assertEqual(content.parts[0].tool_result.tool_call_id, "fr_1")
        self.assertEqual(
            content.parts[0].tool_result.content,
            {"celsius": 22},
        )

    # -- part dict with no recognized keys -----------------------------------

    def test_unknown_part_dict(self):
        msg = {
            "role": "model",
            "parts": [{"inline_data": {"mime_type": "image/png"}}],
        }
        content = genai_converter.to_content(msg)
        self.assertEqual(content.parts[0].text, "")

    # -- _part_to_genai fallback for unknown part type -----------------------

    def test_from_content_unknown_part_type(self):
        part = Part(
            part_type=PartType.TEXT, text=None, tool_call=None, tool_result=None
        )
        # Force an impossible part_type to hit the fallback
        part.part_type = "unknown_type"  # type: ignore
        content = Content(role=Role.USER, parts=[part])
        result = genai_converter.from_content(content)
        self.assertEqual(result["parts"][0], {"text": ""})

    # -- roundtrip ------------------------------------------------------------

    def test_roundtrip_user_text(self):
        original = {"role": "user", "parts": [{"text": "Hello"}]}
        content = genai_converter.to_content(original)
        back = genai_converter.from_content(content)
        self.assertEqual(back["role"], "user")
        self.assertEqual(back["parts"][0]["text"], "Hello")


if __name__ == "__main__":
    unittest.main()
