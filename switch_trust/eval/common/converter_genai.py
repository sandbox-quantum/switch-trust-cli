"""Convert between Google GenAI Content/Part types and internal schema."""

from __future__ import annotations

from typing import Any

from google.genai import types as genai_types

from switch_trust.eval.common.schema import (
    Content,
    Message,
    Part,
    PartType,
    Role,
    ToolCall,
    ToolResult,
)

_ROLE_MAP = {"user": Role.USER, "model": Role.ASSISTANT}
_ROLE_MAP_REVERSE = {Role.USER: "user", Role.ASSISTANT: "model"}


# -- To internal --------------------------------------------------------------


def to_content(
    content: dict[str, Any] | genai_types.Content,
) -> Content:
    """Convert a GenAI Content (object or dict) to internal Content."""
    if isinstance(content, genai_types.Content):
        return _content_obj_to_internal(content)
    return _content_dict_to_internal(content)


def to_message(
    content: dict[str, Any] | genai_types.Content,
) -> Message:
    return Message(content=to_content(content))


# -- From internal ------------------------------------------------------------


def from_content(content: Content) -> dict[str, Any]:
    """Convert internal Content to a GenAI-compatible dict."""
    role = _ROLE_MAP_REVERSE.get(content.role, "user")
    parts = [_part_to_genai(p) for p in content.parts]
    return {"role": role, "parts": parts}


def from_message(message: Message) -> dict[str, Any]:
    return from_content(message.content)


def from_content_obj(content: Content) -> genai_types.Content:
    """Convert internal Content to a GenAI Content object."""
    return genai_types.Content(**from_content(content))


def from_message_obj(message: Message) -> genai_types.Content:
    return from_content_obj(message.content)


# -- Private: to internal -----------------------------------------------------


def _content_obj_to_internal(
    content: genai_types.Content,
) -> Content:
    role = _ROLE_MAP.get(content.role, Role.USER)
    parts = [_genai_part_to_internal(p) for p in (content.parts or [])]
    if not parts:
        parts = [Part.text_part("")]
    return Content(role=role, parts=parts)


def _content_dict_to_internal(d: dict[str, Any]) -> Content:
    role = _ROLE_MAP.get(d.get("role", "user"), Role.USER)
    parts: list[Part] = []
    for p in d.get("parts", []):
        if isinstance(p, str):
            parts.append(Part.text_part(p))
        elif isinstance(p, dict):
            parts.append(_genai_part_dict_to_internal(p))
        else:
            parts.append(_genai_part_to_internal(p))
    if not parts:
        parts = [Part.text_part("")]
    return Content(role=role, parts=parts)


def _genai_part_to_internal(part: genai_types.Part) -> Part:
    if part.thought and part.text is not None:
        return Part.thinking_part(part.text)
    if part.text is not None:
        return Part.text_part(part.text)
    if part.function_call is not None:
        fc = part.function_call
        return Part.tool_call_part(
            ToolCall(
                id=fc.id or "",
                name=fc.name or "",
                arguments=dict(fc.args) if fc.args else {},
            )
        )
    if part.function_response is not None:
        fr = part.function_response
        return Part.tool_result_part(
            ToolResult(
                tool_call_id=fr.id or "",
                content=dict(fr.response) if fr.response else "",
            )
        )
    return Part.text_part("")


def _genai_part_dict_to_internal(d: dict[str, Any]) -> Part:
    if "thought" in d and d.get("thought") and "text" in d:
        return Part.thinking_part(d["text"])
    if "text" in d:
        return Part.text_part(d["text"])
    if "function_call" in d:
        fc = d["function_call"]
        return Part.tool_call_part(
            ToolCall(
                id=fc.get("id", ""),
                name=fc.get("name", ""),
                arguments=fc.get("args", {}),
            )
        )
    if "function_response" in d:
        fr = d["function_response"]
        return Part.tool_result_part(
            ToolResult(
                tool_call_id=fr.get("id", ""),
                content=fr.get("response", ""),
            )
        )
    return Part.text_part("")


# -- Private: from internal ---------------------------------------------------


def _part_to_genai(part: Part) -> dict[str, Any]:
    if part.part_type == PartType.THINKING:
        return {"text": part.text or "", "thought": True}
    if part.part_type == PartType.TEXT:
        return {"text": part.text or ""}
    if part.part_type == PartType.TOOL_CALL:
        tc = part.tool_call
        return {
            "function_call": {
                "id": tc.id,
                "name": tc.name,
                "args": tc.arguments,
            },
        }
    if part.part_type == PartType.TOOL_RESULT:
        tr = part.tool_result
        response = (
            tr.content if isinstance(tr.content, dict) else {"result": tr.content}
        )
        return {
            "function_response": {
                "id": tr.tool_call_id,
                "name": "",
                "response": response,
            },
        }
    return {"text": ""}
