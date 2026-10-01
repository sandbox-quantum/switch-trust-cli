"""Convert between Anthropic Messages API types and internal schema."""

from __future__ import annotations

from typing import Any

from anthropic.types import Message as AnthropicMessage
from anthropic.types import MessageParam, TextBlock, ThinkingBlock, ToolUseBlock

from switch_trust.eval.common.schema import (
    Content,
    Message,
    Part,
    PartType,
    Role,
    ToolCall,
    ToolResult,
)

# -- To internal --------------------------------------------------------------


def to_content(message: dict[str, Any] | AnthropicMessage) -> Content:
    """Convert an Anthropic message (param dict or response object) to Content."""
    if isinstance(message, AnthropicMessage):
        return _response_to_content(message)
    return _param_to_content(message)


def to_message(message: dict[str, Any] | AnthropicMessage) -> Message:
    return Message(content=to_content(message))


# -- From internal ------------------------------------------------------------


def from_content(content: Content) -> dict[str, Any]:
    """Convert internal Content to an Anthropic MessageParam dict.

    System content is returned as ``{"role": "system", "content": ...}``
    — the caller should extract system messages and pass them via the
    ``system`` parameter of ``messages.create()``.
    """
    role = content.role.value

    if role == "system":
        return {"role": "system", "content": _parts_to_blocks(content.parts)}

    return {"role": role, "content": _parts_to_blocks(content.parts)}


def from_message(message: Message) -> dict[str, Any]:
    return from_content(message.content)


def from_content_obj(content: Content) -> MessageParam:
    """Convert internal Content to a typed Anthropic MessageParam."""
    return from_content(content)  # type: ignore[return-value]


def from_message_obj(message: Message) -> MessageParam:
    return from_content_obj(message.content)


# -- Private: to internal -----------------------------------------------------


def _response_to_content(msg: AnthropicMessage) -> Content:
    parts: list[Part] = []
    for block in msg.content:
        if isinstance(block, TextBlock):
            parts.append(Part.text_part(block.text))
        elif isinstance(block, ToolUseBlock):
            parts.append(
                Part.tool_call_part(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )
            )
        elif isinstance(block, ThinkingBlock):
            parts.append(Part.thinking_part(block.thinking))

    if not parts:
        parts.append(Part.text_part(""))
    return Content(role=Role(msg.role), parts=parts)


def _param_to_content(msg: dict[str, Any]) -> Content:
    role = Role(msg["role"])
    raw_content = msg.get("content", "")

    parts: list[Part] = []

    if isinstance(raw_content, str):
        parts.append(Part.text_part(raw_content))
    elif isinstance(raw_content, list):
        for block in raw_content:
            block_type = block.get("type", "text")
            if block_type == "text":
                parts.append(Part.text_part(block["text"]))
            elif block_type == "thinking":
                parts.append(Part.thinking_part(block["thinking"]))
            elif block_type == "tool_use":
                parts.append(
                    Part.tool_call_part(
                        ToolCall(
                            id=block["id"],
                            name=block["name"],
                            arguments=block.get("input", {}),
                        )
                    )
                )
            elif block_type == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    # Extract text from content blocks
                    texts = [
                        b["text"] for b in result_content if b.get("type") == "text"
                    ]
                    result_content = " ".join(texts) if texts else ""
                parts.append(
                    Part.tool_result_part(
                        ToolResult(
                            tool_call_id=block["tool_use_id"],
                            content=result_content,
                            is_error=block.get("is_error", False),
                        )
                    )
                )

    if not parts:
        parts.append(Part.text_part(""))

    return Content(role=role, parts=parts)


# -- Private: from internal ---------------------------------------------------


def _parts_to_blocks(parts: list[Part]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for p in parts:
        if p.part_type == PartType.TEXT:
            blocks.append({"type": "text", "text": p.text or ""})
        elif p.part_type == PartType.THINKING:
            blocks.append({"type": "thinking", "thinking": p.text or ""})
        elif p.part_type == PartType.TOOL_CALL:
            tc = p.tool_call
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                }
            )
        elif p.part_type == PartType.TOOL_RESULT:
            tr = p.tool_result
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tr.tool_call_id,
                    "content": tr.content
                    if isinstance(tr.content, str)
                    else str(tr.content),
                    "is_error": tr.is_error,
                }
            )
    return blocks
