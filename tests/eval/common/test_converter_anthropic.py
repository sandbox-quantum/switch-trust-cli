import unittest

from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, ThinkingBlock, ToolUseBlock, Usage

from switch_trust.eval.common import converter_anthropic as anthropic_converter
from switch_trust.eval.common.schema import (
    Content,
    Part,
    PartType,
    Role,
    ToolCall,
    ToolResult,
)


class TestAnthropicConverter(unittest.TestCase):
    # -- to_content -----------------------------------------------------------

    def test_user_text_param(self):
        msg = {"role": "user", "content": "Hello"}
        content = anthropic_converter.to_content(msg)
        self.assertEqual(content.role, Role.USER)
        self.assertEqual(content.parts[0].text, "Hello")

    def test_user_content_blocks_param(self):
        msg = {
            "role": "user",
            "content": [{"type": "text", "text": "Hi there"}],
        }
        content = anthropic_converter.to_content(msg)
        self.assertEqual(content.parts[0].text, "Hi there")

    def test_assistant_response_object(self):
        msg = AnthropicMessage(
            id="msg_1",
            type="message",
            role="assistant",
            content=[TextBlock(type="text", text="Hello!", citations=None)],
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage(input_tokens=10, output_tokens=5),
        )
        content = anthropic_converter.to_content(msg)
        self.assertEqual(content.role, Role.ASSISTANT)
        self.assertEqual(content.parts[0].text, "Hello!")

    def test_assistant_with_tool_use_response(self):
        msg = AnthropicMessage(
            id="msg_2",
            type="message",
            role="assistant",
            content=[
                TextBlock(type="text", text="Let me check.", citations=None),
                ToolUseBlock(
                    type="tool_use",
                    id="tu_1",
                    name="get_weather",
                    input={"city": "Paris"},
                ),
            ],
            model="claude-sonnet-4-20250514",
            stop_reason="tool_use",
            stop_sequence=None,
            usage=Usage(input_tokens=10, output_tokens=20),
        )
        content = anthropic_converter.to_content(msg)
        self.assertEqual(len(content.parts), 2)
        self.assertEqual(content.parts[0].part_type, PartType.TEXT)
        self.assertEqual(content.parts[1].part_type, PartType.TOOL_CALL)
        self.assertEqual(content.parts[1].tool_call.name, "get_weather")
        self.assertEqual(
            content.parts[1].tool_call.arguments,
            {"city": "Paris"},
        )

    def test_thinking_response(self):
        msg = AnthropicMessage(
            id="msg_3",
            type="message",
            role="assistant",
            content=[
                ThinkingBlock(
                    type="thinking",
                    thinking="Let me reason...",
                    signature="sig123",
                ),
                TextBlock(type="text", text="The answer is 42.", citations=None),
            ],
            model="claude-sonnet-4-20250514",
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage(input_tokens=10, output_tokens=15),
        )
        content = anthropic_converter.to_content(msg)
        self.assertEqual(len(content.parts), 2)
        self.assertEqual(content.parts[0].part_type, PartType.THINKING)
        self.assertEqual(content.parts[0].text, "Let me reason...")
        self.assertEqual(content.parts[1].part_type, PartType.TEXT)

    def test_tool_result_param(self):
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_1",
                    "content": "22°C and sunny",
                },
            ],
        }
        content = anthropic_converter.to_content(msg)
        self.assertEqual(content.parts[0].part_type, PartType.TOOL_RESULT)
        self.assertEqual(
            content.parts[0].tool_result.tool_call_id,
            "tu_1",
        )

    def test_tool_result_with_error(self):
        msg = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tu_2",
                    "content": "Not found",
                    "is_error": True,
                },
            ],
        }
        content = anthropic_converter.to_content(msg)
        self.assertTrue(content.parts[0].tool_result.is_error)

    # -- from_content ---------------------------------------------------------

    def test_from_user_text(self):
        content = Content.text(Role.USER, "Hello")
        result = anthropic_converter.from_content(content)
        self.assertEqual(result["role"], "user")
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertEqual(result["content"][0]["text"], "Hello")

    def test_from_assistant_with_tool_use(self):
        tc = ToolCall(id="t1", name="search", arguments={"q": "test"})
        content = Content(
            role=Role.ASSISTANT,
            parts=[Part.tool_call_part(tc)],
        )
        result = anthropic_converter.from_content(content)
        self.assertEqual(result["role"], "assistant")
        block = result["content"][0]
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["name"], "search")

    def test_from_thinking(self):
        content = Content.thinking("deep thoughts")
        result = anthropic_converter.from_content(content)
        block = result["content"][0]
        self.assertEqual(block["type"], "thinking")
        self.assertEqual(block["thinking"], "deep thoughts")

    def test_from_tool_result(self):
        tr = ToolResult(tool_call_id="t1", content="done")
        content = Content(
            role=Role.USER,
            parts=[Part.tool_result_part(tr)],
        )
        result = anthropic_converter.from_content(content)
        block = result["content"][0]
        self.assertEqual(block["type"], "tool_result")
        self.assertEqual(block["tool_use_id"], "t1")

    # -- from_content_obj -----------------------------------------------------

    def test_from_content_obj_user(self):
        content = Content.text(Role.USER, "Hello")
        result = anthropic_converter.from_content_obj(content)
        self.assertEqual(result["role"], "user")
        self.assertEqual(result["content"][0]["text"], "Hello")

    def test_from_content_obj_assistant(self):
        tc = ToolCall(id="t1", name="search", arguments={"q": "test"})
        content = Content(
            role=Role.ASSISTANT,
            parts=[Part.tool_call_part(tc)],
        )
        result = anthropic_converter.from_content_obj(content)
        self.assertEqual(result["role"], "assistant")
        self.assertEqual(result["content"][0]["type"], "tool_use")

    # -- roundtrip ------------------------------------------------------------

    def test_roundtrip_user_text(self):
        original = {"role": "user", "content": "Hello"}
        content = anthropic_converter.to_content(original)
        back = anthropic_converter.from_content(content)
        self.assertEqual(back["role"], "user")
        self.assertEqual(back["content"][0]["text"], "Hello")


if __name__ == "__main__":
    unittest.main()
