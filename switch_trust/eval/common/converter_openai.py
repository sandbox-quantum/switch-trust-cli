"""Convert between OpenAI ChatCompletion messages and internal schema."""

from __future__ import annotations

import json
from typing import Any

from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam

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


def to_content(message: dict[str, Any] | ChatCompletionMessage) -> Content:
    """Convert an OpenAI message (dict param or response object) to Content."""
    if isinstance(message, ChatCompletionMessage):
        return _response_to_content(message)
    return _param_to_content(message)


def to_message(message: dict[str, Any] | ChatCompletionMessage) -> Message:
    return Message(content=to_content(message))


# -- From internal ------------------------------------------------------------


def from_content(content: Content) -> dict[str, Any]:
    """Convert internal Content to an OpenAI message param dict."""
    role = content.role.value

    if role == "system":
        return _content_to_system(content)
    if role == "user":
        return _content_to_user(content)
    return _content_to_assistant_or_tool(content)


def from_message(message: Message) -> dict[str, Any]:
    return from_content(message.content)


def from_content_obj(content: Content) -> ChatCompletionMessageParam:
    """Convert internal Content to a typed OpenAI message param."""
    return from_content(content)  # type: ignore[return-value]


def from_message_obj(message: Message) -> ChatCompletionMessageParam:
    return from_content_obj(message.content)


# -- Private: to internal -----------------------------------------------------


def _response_to_content(msg: ChatCompletionMessage) -> Content:
    parts: list[Part] = []
    if msg.content:
        parts.append(Part.text_part(msg.content))
    if msg.tool_calls:
        for tc in msg.tool_calls:
            args = (
                json.loads(tc.function.arguments)
                if isinstance(tc.function.arguments, str)
                else tc.function.arguments
            )
            parts.append(
                Part.tool_call_part(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )
            )
    if not parts:
        parts.append(Part.text_part(""))
    return Content(role=Role.ASSISTANT, parts=parts)


def _param_to_content(msg: dict[str, Any]) -> Content:
    role_str = msg["role"]
    role = Role(role_str) if role_str != "tool" else Role.USER

    parts: list[Part] = []

    if role_str == "tool":
        parts.append(
            Part.tool_result_part(
                ToolResult(
                    tool_call_id=msg["tool_call_id"],
                    content=msg.get("content", ""),
                )
            )
        )
        return Content(role=Role.USER, parts=parts)

    content = msg.get("content")
    if isinstance(content, str):
        parts.append(Part.text_part(content))
    elif isinstance(content, list):
        for item in content:
            if item.get("type") == "text":
                parts.append(Part.text_part(item["text"]))

    tool_calls = msg.get("tool_calls")
    if tool_calls:
        for tc in tool_calls:
            func = tc["function"] if isinstance(tc, dict) else tc
            args = func["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            parts.append(
                Part.tool_call_part(
                    ToolCall(
                        id=tc["id"],
                        name=func["name"],
                        arguments=args,
                    )
                )
            )

    if not parts:
        parts.append(Part.text_part(""))

    return Content(role=role, parts=parts)


# -- Private: from internal ---------------------------------------------------


def _content_to_system(content: Content) -> dict[str, Any]:
    text = _collect_text(content)
    return {"role": "system", "content": text}


def _content_to_user(content: Content) -> dict[str, Any] | list[dict[str, Any]]:
    # If content has tool results, emit them as separate tool messages.
    tool_results = [p for p in content.parts if p.part_type == PartType.TOOL_RESULT]
    if tool_results:
        messages = []
        for p in tool_results:
            tr = p.tool_result
            result_content = (
                tr.content if isinstance(tr.content, str) else json.dumps(tr.content)
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tr.tool_call_id,
                    "content": result_content,
                }
            )
        return messages if len(messages) > 1 else messages[0]

    text = _collect_text(content)
    return {"role": "user", "content": text}


def _content_to_assistant_or_tool(content: Content) -> dict[str, Any]:
    text_parts = [p for p in content.parts if p.part_type == PartType.TEXT]
    tool_call_parts = [p for p in content.parts if p.part_type == PartType.TOOL_CALL]

    msg: dict[str, Any] = {"role": "assistant"}
    msg["content"] = (
        " ".join(p.text for p in text_parts if p.text) if text_parts else None
    )

    if tool_call_parts:
        msg["tool_calls"] = []
        for p in tool_call_parts:
            tc = p.tool_call
            msg["tool_calls"].append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
            )

    return msg


def _collect_text(content: Content) -> str:
    return " ".join(
        p.text for p in content.parts if p.part_type == PartType.TEXT and p.text
    )
