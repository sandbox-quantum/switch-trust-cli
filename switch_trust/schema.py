"""
Minimal stub for inventory schema types used by the scan module.
"""

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class RepoFile:
    path: str
    content: str
    size: int


@dataclass
class DiscoveryResult:
    repo_url: str
    repo_name: str
    default_branch: str = "main"
    frameworks_detected: list[str] = field(default_factory=list)
    primary_framework: str = "unknown"
    files: list[RepoFile] = field(default_factory=list)
    python_files: list[RepoFile] = field(default_factory=list)
    requirements_files: list[RepoFile] = field(
        default_factory=list,
    )
    framework_evidence: dict[str, str] = field(
        default_factory=dict,
    )
    total_files_fetched: int = 0
    error: str | None = None
    source_backend: str = "unknown"


@dataclass
class AffectedComponent:
    """A specific package, file, or module affected by a vulnerability."""

    name: str
    version: str = ""
    path: str = ""


@dataclass
class Evidence:
    """Structured evidence supporting a vulnerability finding."""

    file: str
    code_snippet: str = ""
    context: str = ""
    confidence: str = ""  # HIGH | MEDIUM | LOW
    line: int = 0
    column: int = 0


class DependencyType(Enum):
    LIBRARY_USAGE = "LIBRARY_USAGE"
    MODEL_USAGE = "MODEL_USAGE"
    SECRET_USAGE = "SECRET_USAGE"
    SUBAGENT_USAGE = "SUBAGENT_USAGE"
    SUPPLIER = "SUPPLIER"
    VULNERABILITY_USAGE = "VULNERABILITY_USAGE"
    TOOL_USAGE = "TOOL_USAGE"
    TOOL_DEFINITION = "TOOL_DEFINITION"
    MCP_CLIENT_USAGE = "MCP_CLIENT_USAGE"
    MCP_SERVER_USAGE = "MCP_SERVER_USAGE"


class EntityType(Enum):
    LIBRARY = "LIBRARY"
    MODEL = "MODEL"
    AGENT = "AGENT"
    MCP_SERVER = "MCP_SERVER"
    SECRET = "SECRET"
    ORGANIZATION = "ORGANIZATION"
    MCP_CLIENT = "MCP_CLIENT"
    VULNERABILITY = "VULNERABILITY"
    TOOL = "TOOL"
