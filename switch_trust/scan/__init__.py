"""Shared infrastructure for agent_scanner and mcp_scanner."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ADKModel(Protocol):
    """Anything ADK accepts as a model argument (str or LiteLlm wrapper)."""

    def __str__(self) -> str: ...
