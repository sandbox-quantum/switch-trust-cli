"""
Internal schema for representing agentic traffic.

Structure mirrors Google's GenAI Content/Part model:
  Session -> Message[] -> Content -> Part[]

Each Content carries a role and a list of typed Parts. Content types cover
the full lifecycle of an agentic interaction: system prompts, user/assistant
text, tool calls and their results, and intermediate reasoning steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from dataclasses_json import dataclass_json

from switch_trust.eval.common.utils import datetime_config, generate_id

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Role(str, Enum):
    """Who produced this content block."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class PartType(str, Enum):
    """Discriminator for the payload of a Part."""

    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"


# ---------------------------------------------------------------------------
# Content primitives
# ---------------------------------------------------------------------------


@dataclass_json
@dataclass
class ToolCall:
    """A request by the model to invoke a tool/function."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass_json
@dataclass
class ToolResult:
    """The outcome of executing a tool call."""

    tool_call_id: str
    content: Any  # str, dict, list — whatever the tool returned
    is_error: bool = False


# ---------------------------------------------------------------------------
# Part — atomic unit within a Content block
# ---------------------------------------------------------------------------


@dataclass_json
@dataclass
class Part:
    """
    A single typed piece of content within a Content block.

    ``part_type`` is the discriminator. Exactly one of the payload fields
    (``text``, ``tool_call``, ``tool_result``) should be populated,
    matching ``part_type``. TEXT and THINKING both use ``text``.
    """

    part_type: PartType

    # TEXT and THINKING share a plain string payload.
    text: str | None = None

    # Structured tool interaction payloads.
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None

    # Convenience constructors ------------------------------------------------
    @classmethod
    def text_part(cls, text: str) -> Part:
        return cls(part_type=PartType.TEXT, text=text)

    @classmethod
    def thinking_part(cls, thinking: str) -> Part:
        return cls(part_type=PartType.THINKING, text=thinking)

    @classmethod
    def tool_call_part(cls, tool_call: ToolCall) -> Part:
        return cls(part_type=PartType.TOOL_CALL, tool_call=tool_call)

    @classmethod
    def tool_result_part(cls, tool_result: ToolResult) -> Part:
        return cls(part_type=PartType.TOOL_RESULT, tool_result=tool_result)


# ---------------------------------------------------------------------------
# Content — a role-attributed block of Parts
# ---------------------------------------------------------------------------


@dataclass_json
@dataclass
class Content:
    """
    A role-attributed block of content, grouping one or more Parts.

    Mirrors Google's GenAI ``Content`` type: a single role (system / user /
    assistant) paired with a list of typed parts. A model response that
    includes a thinking block, text, and a tool call is represented as one
    Content with three Parts.
    """

    role: Role
    parts: list[Part]

    # Convenience constructors ------------------------------------------------

    @classmethod
    def text(cls, role: Role, text: str) -> Content:
        return cls(role=role, parts=[Part.text_part(text)])

    @classmethod
    def thinking(cls, thinking: str) -> Content:
        return cls(role=Role.ASSISTANT, parts=[Part.thinking_part(thinking)])


# ---------------------------------------------------------------------------
# Message — one turn in a conversation
# ---------------------------------------------------------------------------


@dataclass_json
@dataclass
class Message:
    """
    A single turn in an agentic session.

    Wraps a Content block with transport-level metadata: timestamp, model,
    provider, and token usage.
    """

    content: Content
    id: str = field(default_factory=generate_id)
    timestamp: datetime = field(
        default_factory=datetime.now,
        metadata=datetime_config,
    )
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Session — a full agentic interaction
# ---------------------------------------------------------------------------


@dataclass_json
@dataclass
class Session:
    """
    A complete agentic session — the top-level capture unit.

    A session groups all messages belonging to one end-to-end agent run,
    including system prompt, user turns, model responses, and tool
    interactions.
    """

    id: str = field(default_factory=generate_id)
    timestamp: datetime = field(
        default_factory=datetime.now,
        metadata=datetime_config,
    )
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
