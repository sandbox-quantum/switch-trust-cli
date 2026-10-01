import json
import unittest

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from switch_trust.eval.common import converter_openai as openai_converter
from switch_trust.eval.common.schema import (
    Content,
    Part,
    PartType,
    Role,
    ToolCall,
    ToolResult,
)


class TestOpenAIConverter(unittest.TestCase):
    # -- to_content -----------------------------------------------------------

    def test_user_text_param(self):
        msg = {"role": "user", "content": "Hello"}
        content = openai_converter.to_content(msg)
        self.assertEqual(content.role, Role.USER)
        self.assertEqual(len(content.parts), 1)
        self.assertEqual(content.parts[0].text, "Hello")

    def test_system_text_param(self):
        msg = {"role": "system", "content": "You are helpful."}
        content = openai_converter.to_content(msg)
        self.assertEqual(content.role, Role.SYSTEM)
        self.assertEqual(content.parts[0].text, "You are helpful.")

    def test_assistant_response_object(self):
        msg = ChatCompletionMessage(
            role="assistant",
            content="Hi there!",
        )
        content = openai_converter.to_content(msg)
        self.assertEqual(content.role, Role.ASSISTANT)
        self.assertEqual(content.parts[0].text, "Hi there!")

    def test_assistant_with_tool_calls_response(self):
        msg = ChatCompletionMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(
                        name="get_weather",
                        arguments='{"city": "Paris"}',
                    ),
                ),
            ],
        )
        content = openai_converter.to_content(msg)
        self.assertEqual(content.role, Role.ASSISTANT)
        self.assertEqual(len(content.parts), 1)
        self.assertEqual(content.parts[0].part_type, PartType.TOOL_CALL)
        self.assertEqual(content.parts[0].tool_call.name, "get_weather")
        self.assertEqual(
            content.parts[0].tool_call.arguments,
            {"city": "Paris"},
        )

    def test_tool_result_param(self):
        msg = {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "22°C and sunny",
        }
        content = openai_converter.to_content(msg)
        self.assertEqual(content.role, Role.USER)
        self.assertEqual(content.parts[0].part_type, PartType.TOOL_RESULT)
        self.assertEqual(
            content.parts[0].tool_result.tool_call_id,
            "call_1",
        )
        self.assertEqual(
            content.parts[0].tool_result.content,
            "22°C and sunny",
        )

    def test_assistant_text_and_tool_calls(self):
        msg = ChatCompletionMessage(
            role="assistant",
            content="Let me check.",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_2",
                    type="function",
                    function=Function(
                        name="search",
                        arguments='{"q": "test"}',
                    ),
                ),
            ],
        )
        content = openai_converter.to_content(msg)
        self.assertEqual(len(content.parts), 2)
        self.assertEqual(content.parts[0].part_type, PartType.TEXT)
        self.assertEqual(content.parts[1].part_type, PartType.TOOL_CALL)

    # -- from_content ---------------------------------------------------------

    def test_from_system_content(self):
        content = Content.text(Role.SYSTEM, "Be helpful.")
        result = openai_converter.from_content(content)
        self.assertEqual(result["role"], "system")
        self.assertEqual(result["content"], "Be helpful.")

    def test_from_user_content(self):
        content = Content.text(Role.USER, "Hi")
        result = openai_converter.from_content(content)
        self.assertEqual(result["role"], "user")
        self.assertEqual(result["content"], "Hi")

    def test_from_assistant_with_tool_calls(self):
        tc = ToolCall(id="c1", name="fn", arguments={"a": 1})
        content = Content(
            role=Role.ASSISTANT,
            parts=[
                Part.text_part("thinking..."),
                Part.tool_call_part(tc),
            ],
        )
        result = openai_converter.from_content(content)
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["content"], "thinking...")
        self.assertEqual(len(result["tool_calls"]), 1)
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "fn")
        self.assertEqual(
            json.loads(result["tool_calls"][0]["function"]["arguments"]),
            {"a": 1},
        )

    def test_from_tool_result_content(self):
        tr = ToolResult(tool_call_id="c1", content="result")
        content = Content(
            role=Role.USER,
            parts=[Part.tool_result_part(tr)],
        )
        result = openai_converter.from_content(content)
        self.assertEqual(result["role"], "tool")
        self.assertEqual(result["tool_call_id"], "c1")
        self.assertEqual(result["content"], "result")

    # -- from_content_obj -----------------------------------------------------

    def test_from_content_obj_user(self):
        content = Content.text(Role.USER, "Hi")
        result = openai_converter.from_content_obj(content)
        self.assertEqual(result["role"], "user")
        self.assertEqual(result["content"], "Hi")

    def test_from_content_obj_assistant(self):
        tc = ToolCall(id="c1", name="fn", arguments={"a": 1})
        content = Content(
            role=Role.ASSISTANT,
            parts=[Part.tool_call_part(tc)],
        )
        result = openai_converter.from_content_obj(content)
        self.assertEqual(result["role"], "assistant")

    # -- to_content: list content items --------------------------------------

    def test_list_content_items(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
                {"type": "image_url", "image_url": {"url": "http://x"}},
            ],
        }
        content = openai_converter.to_content(msg)
        self.assertEqual(content.role, Role.USER)
        # Only text items are converted; image_url is ignored
        self.assertEqual(len(content.parts), 2)
        self.assertEqual(content.parts[0].text, "first")
        self.assertEqual(content.parts[1].text, "second")

    # -- to_content: dict with tool_calls ------------------------------------

    def test_dict_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "Sure",
            "tool_calls": [
                {
                    "id": "call_99",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"key": "val"}',
                    },
                },
            ],
        }
        content = openai_converter.to_content(msg)
        self.assertEqual(content.role, Role.ASSISTANT)
        self.assertEqual(len(content.parts), 2)
        self.assertEqual(content.parts[0].part_type, PartType.TEXT)
        self.assertEqual(content.parts[0].text, "Sure")
        self.assertEqual(content.parts[1].part_type, PartType.TOOL_CALL)
        self.assertEqual(content.parts[1].tool_call.name, "lookup")
        self.assertEqual(
            content.parts[1].tool_call.arguments,
            {"key": "val"},
        )

    # -- to_content: empty dict content --------------------------------------

    def test_empty_content(self):
        msg = {"role": "assistant"}
        content = openai_converter.to_content(msg)
        self.assertEqual(content.role, Role.ASSISTANT)
        self.assertEqual(len(content.parts), 1)
        self.assertEqual(content.parts[0].text, "")

    # -- from_content_obj returns ChatCompletionMessageParam ----------------

    def test_from_content_obj_returns_param(self):
        content = Content.text(Role.USER, "test")
        result = openai_converter.from_content_obj(content)
        self.assertEqual(result["role"], "user")
        self.assertEqual(result["content"], "test")

    # -- from_message_obj ---------------------------------------------------

    def test_from_message_obj(self):
        from switch_trust.eval.common.schema import Message

        msg = Message(content=Content.text(Role.USER, "hello"))
        result = openai_converter.from_message_obj(msg)
        self.assertEqual(result["role"], "user")
        self.assertEqual(result["content"], "hello")

    # -- roundtrip ------------------------------------------------------------

    def test_roundtrip_user_text(self):
        original = {"role": "user", "content": "Hello"}
        content = openai_converter.to_content(original)
        back = openai_converter.from_content(content)
        self.assertEqual(back["role"], "user")
        self.assertEqual(back["content"], "Hello")


if __name__ == "__main__":
    unittest.main()
