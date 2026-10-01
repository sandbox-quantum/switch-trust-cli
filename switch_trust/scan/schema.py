"""
schema.py — Data models for the AI Agent Security Scanner
Mirrors the existing AI-SPM schema.py patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from switch_trust.schema import (
    AffectedComponent,  # noqa: F401
    Evidence,  # noqa: F401
)

# ── Protocols ────────────────────────────────────────────────────────────────


class InventoryLike(Protocol):
    """Minimal interface for an Inventory object used by ScanConfig.

    The full Inventory lives in the parent package
    (switch_trust.schema). This protocol captures only what
    the agent scanner reads, keeping the agent_scanner package decoupled.
    """

    @property
    def agents(self) -> dict[str, Any]: ...


# ── TypedDicts ───────────────────────────────────────────────────────────────


class RawFinding(TypedDict):
    """Shape of the finding dict produced by ToolDispatcher.report_finding()."""

    category: str
    subcategory: str
    title: str
    description: str
    impact: str
    remediation: str
    affected_component: str
    evidence: str
    confidence: str
    hallucination_flag: bool
    evidence_file: str
    evidence_line: int
    agent_name: str


class PackageInfo(TypedDict):
    """A pinned package extracted from requirements.txt."""

    name: str
    version: str


class CategorySummaryEntry(TypedDict):
    """Per-category finding counts in ScanReport.category_summary."""

    asi_code: str
    asi_title: str
    count: int
    critical: int
    high: int
    medium: int
    low: int


class TraceToolCall(TypedDict):
    """Shape of a tool call entry recorded by TraceLogger."""

    event: str
    session_id: str
    iteration: int
    call_number: int
    tool: str
    args: dict[str, Any]
    result_preview: str
    result_length: int
    tokens_consumed: int
    tokens_total: int
    duration_ms: int
    timestamp: str
    success: bool
    error: str | None


# ── Config ───────────────────────────────────────────────────────────────────


@dataclass
class ScanConfig:
    """Configuration for a single scan run.

    Exactly one target must be set: local_path, files, or inventory.
    When inventory is set, local_path should also be set (the folder the
    inventory was parsed from) so agent files can be read from disk.
    """

    local_path: str | None = None
    files: list[str] | None = None
    inventory: InventoryLike | None = None

    skip_triage: bool = False
    agentic: bool = True

    model_string: str | None = None


# ── Core data models ────────────────────────────────────────────────────────


@dataclass
class AgentProfile:
    """
    Extracted profile of a single AI agent from source code.
    This is the core scan target — not the framework, the agent itself.
    """

    agent_id: str  # Derived identifier (file + variable name)
    framework: str  # "crewai" | "autogen" | "langchain" | "custom_http"
    source_file: str  # Relative path in repo
    role: str | None = None  # Agent's stated role/purpose
    goal: str | None = None  # Agent's objective
    backstory: str | None = None  # Agent's backstory (CrewAI specific)
    tools: list[str] = field(default_factory=list)
    tool_details: list[str] = field(default_factory=list)
    llm: str | None = None  # LLM used (e.g. "gpt-4", "claude-3")
    memory: bool | None = None  # Memory enabled?
    allow_delegation: bool | None = None  # Can delegate to other agents?
    verbose: bool | None = None
    system_prompt: str | None = None  # Inline system prompt if found
    raw_code: str = ""  # Relevant code block for AI reasoning
    # ── v2 fields: guardrails and graph structure ─────────────────────────────
    input_guardrails: list[str] = field(default_factory=list)
    output_guardrails: list[str] = field(default_factory=list)
    function_signatures: list[str] = field(default_factory=list)
    graph_nodes: list[str] = field(default_factory=list)
    has_recursion_limit: bool | None = None  # StateGraph: recursion_limit set?


@dataclass
class TaskProfile:
    """Extracted profile of a CrewAI Task or AutoGen task."""

    task_id: str
    source_file: str
    description: str | None = None
    expected_output: str | None = None
    agent_ref: str | None = None  # Which agent handles this task
    tools: list[str] = field(default_factory=list)
    human_input: bool | None = None
    raw_code: str = ""


@dataclass
class CrewProfile:
    """Extracted profile of a CrewAI Crew or AutoGen GroupChat."""

    crew_id: str
    source_file: str
    agents: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    process: str | None = None  # e.g. "sequential", "hierarchical"
    memory: bool | None = None
    raw_code: str = ""


@dataclass
class CvssScores:
    """CVSS v4 scoring details."""

    base_score: float = 0.0
    vector: str = ""


@dataclass
class Finding:
    """
    A single security finding — unified format shared with MCP scanner.
    Category and subcategory carry the taxonomy info (OWASP ASI or MCP taxonomy).
    """

    id: str  # Unique finding identifier
    category: str  # Taxonomy category
    subcategory: str  # Specific vulnerability type
    ai_spm_severity: str  # Critical | High | Medium | Low
    cvss_v4_severity: str  # Critical | High | Medium | Low | None
    cvss_scores: CvssScores  # {base_score, vector}
    description: str  # Technical description of the issue
    impact: str  # What an attacker could achieve
    likelihood: str  # High | Medium | Low
    remediation: str  # Concrete fix guidance
    affected_components: list[AffectedComponent]  # Affected packages/files/modules
    evidence: list[Evidence]  # Structured evidence items
    hallucination_flag: bool = False  # True if AI finding lacks strong evidence
    title: str = ""  # Short human-readable title
    source: str = ""  # "static_bandit" | "static_opengrep" | "static_secrets"
    # | "static_pip" | "ai_reasoning" | "mcp_scan"
    agent_fingerprints: list[str] = field(
        default_factory=list
    )  # Agent fingerprints this finding belongs to
    tool_fingerprints: list[str] = field(
        default_factory=list
    )  # Tools this finding's evidence falls within
    agent_name: str = ""  # Agent name/ID reported by the AI reasoning agent


# ── Trace models ─────────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    """
    Records a single tool call made by the agentic reasoning loop.
    Written to the scan trace for observability and debugging.
    """

    iteration: int
    tool_name: str
    tool_args: dict[str, Any]
    result_preview: str  # First 200 chars of the tool result
    tokens_consumed: int  # Approximate tokens in the result
    duration_ms: int
    timestamp: str


@dataclass
class ScanTrace:
    """
    Full observability record for an agentic reasoning session.
    Written to <output_path>.trace.jsonl alongside the main report.
    Allows replay of exactly what the agent investigated and in what order.
    """

    session_id: str
    scan_timestamp: str
    provider_model: str
    total_iterations: int
    total_tool_calls: int
    total_tokens: int
    wall_clock_ms: int
    findings_count: int
    tool_calls: list[ToolCall] = field(default_factory=list)
    exit_reason: str = "completed"  # completed | max_iterations | timeout | error


# ── Triage models ────────────────────────────────────────────────────────────


@dataclass
class TriageDismissed:
    """
    Records a finding that the triage agent dismissed as expected behaviour
    for this tool/agent's declared purpose.
    """

    finding_id: str  # Matches Finding.id
    reason: str  # Why the triage agent dismissed it


@dataclass
class TriageDowngraded:
    """
    Records a finding whose severity was reduced by the triage agent
    due to lack of direct exploitability evidence or mitigating context.
    """

    finding_id: str  # Matches Finding.id
    original_severity: str  # Severity assigned by the primary AI reasoning layer
    new_severity: str  # Recalibrated severity after triage
    reason: str  # Justification for the downgrade


# ── Report ───────────────────────────────────────────────────────────────────


@dataclass
class ScanReport:
    """
    Top-level output — one report per scanned repository.

    Triage fields:
      pre_triage_findings   — raw findings list BEFORE triage runs (for auditability).
                              Populated only when triage runs; None otherwise.
      triage_dismissed      — findings the triage agent dismissed as expected behaviour.
      triage_downgraded     — findings whose severity was recalibrated downward.

    The `findings` list always reflects the POST-triage state (filtered + recalibrated).
    `category_summary` is also updated to reflect only post-triage findings.

    Fix #18: schema_version is bumped whenever a breaking field change is made so that
    downstream consumers (the AI-SPM platform database layer) can detect and migrate
    old reports without guessing at the schema. Increment the minor version for additive
    changes (new optional fields), the major version for breaking changes (removed or
    renamed required fields).
    """

    # Serialises first so it always appears at the top of JSON output.
    schema_version: str = "2.0"
    repo_url: str = ""
    repo_name: str = ""
    scan_timestamp: str = ""
    framework_detected: str = ""
    agents_found: int = 0
    findings: list[Finding] = field(default_factory=list)
    category_summary: dict[str, CategorySummaryEntry] = field(default_factory=dict)
    agent_profiles: list[dict[str, Any]] = field(default_factory=list)
    scan_metadata: dict[str, Any] = field(default_factory=dict)
    # ── Triage fields — None until triage.py runs ─────────────────────────────
    pre_triage_findings: list[RawFinding] | None = None
    triage_dismissed: list[TriageDismissed] | None = None
    triage_downgraded: list[TriageDowngraded] | None = None
