"""
scorer.py — CVSS v4 Scoring Engine
Maps findings to CVSS v4 vectors and computes scores deterministically.
All category/subcategory keys align with the OWASP ASI taxonomy in taxonomy.py.
"""

import logging

from cvss import CVSS4

from switch_trust.scan.taxonomy import get_finding_metadata

# CVSS v4 official severity bands (FIRST.org spec)
CVSS_SEVERITY_BANDS = [
    (9.0, "critical"),
    (7.0, "high"),
    (4.0, "medium"),
    (0.1, "low"),
    (0.0, "info"),
]


def severity_from_score(score: float) -> str:
    """Derive severity label from CVSS v4 numeric score per FIRST.org bands.

    Args:
        score: CVSS v4 base score (0.0-10.0).

    Returns:
        Severity label: 'critical', 'high', 'medium', 'low', or 'info'.
        Returns 'info' if score does not match any band.
    """
    for threshold, label in CVSS_SEVERITY_BANDS:
        if score >= threshold:
            return label
    return "info"


def compute_cvss_v4_score(vector: str) -> float:
    """
    Compute CVSS v4 Base Score using the official FIRST.org library.

    Args:
        vector: CVSS v4 vector string (e.g., "CVSS:4.0/AV:N/AC:L/...")

    Returns:
        Base score (0.0–10.0) per official FIRST.org standard.
        Returns 0.0 on invalid input; logs a warning in that case.
    """
    _log = logging.getLogger("scorer")

    if not vector.startswith("CVSS:4.0/"):
        _log.warning(
            "[scorer] Invalid CVSS vector (wrong prefix): %r — defaulting to 0.0",
            vector[:80],
        )
        return 0.0

    try:
        cvss_obj = CVSS4(vector)
        return round(cvss_obj.base_score, 1)
    except Exception as exc:
        _log.warning(
            "[scorer] CVSS4 computation failed for vector %r: %s — defaulting to 0.0",
            vector[:80],
            exc,
        )
        return 0.0


def score_finding(subcategory: str, source: str = "static") -> tuple[float, str, str]:
    """
    Look up CVSS vector for a subcategory, compute score, and return (score, vector, severity).

    Severity label from taxonomy is authoritative — computed score is clamped into its band.

    Args:
        subcategory: ASI taxonomy subcategory key (e.g., 'hardcoded_credentials').
        source: Finding source for logging context ('static', 'ai_reasoning', etc.).

    Returns:
        Tuple of (cvss_score, cvss_vector, severity_label).
        cvss_score: float between 0.0 and 10.0 per CVSS v4 spec.
        cvss_vector: CVSS:4.0 vector string.
        severity_label: 'critical', 'high', 'medium', 'low', or 'info'.

    Raises:
        ValueError: If severity from taxonomy is not in recognized band_ranges.
    """
    meta = get_finding_metadata(subcategory)
    vector = meta.get(
        "cvss_vector", "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:L/VI:L/VA:L/SC:N/SI:N/SA:N"
    )
    severity = meta.get("severity", "medium").lower()

    band_ranges = {
        "critical": (9.0, 10.0),
        "high": (7.0, 8.9),
        "medium": (4.0, 6.9),
        "low": (0.1, 3.9),
        "info": (0.0, 0.0),
    }

    # Fix #7: Validate severity is in expected set before using it
    if severity not in band_ranges:
        logging.getLogger("scorer").warning(
            f"[scorer] Invalid severity '{severity}' for {subcategory}. "
            f"Valid: {list(band_ranges.keys())}. Defaulting to 'medium'."
        )
        severity = "medium"

    band_min, band_max = band_ranges.get(severity, (4.0, 6.9))
    raw_score = compute_cvss_v4_score(vector)
    score = round(max(band_min, min(band_max, raw_score)), 1)
    if score == 0.0 and severity != "info":
        score = round((band_min + band_max) / 2, 1)

    return score, vector, severity


# ── Bandit rule → (ASI category key, subcategory) ────────────────────────────


def map_bandit_to_taxonomy(bandit_rule_id: str, message: str) -> tuple[str, str]:
    """Map a Bandit rule ID to the ASI-aligned taxonomy category/subcategory."""
    rule_map = {
        # ASI05 — Unexpected Code Execution
        "B102": ("asi05_unexpected_code_execution", "arbitrary_code_execution"),
        "B307": ("asi05_unexpected_code_execution", "arbitrary_code_execution"),
        "B601": ("asi05_unexpected_code_execution", "arbitrary_code_execution"),
        "B602": ("asi05_unexpected_code_execution", "arbitrary_code_execution"),
        "B604": ("asi05_unexpected_code_execution", "arbitrary_code_execution"),
        "B605": ("asi05_unexpected_code_execution", "arbitrary_code_execution"),
        "B606": ("asi05_unexpected_code_execution", "arbitrary_code_execution"),
        "B607": ("asi05_unexpected_code_execution", "arbitrary_code_execution"),
        # ASI05 — unsafe deserialization
        "B301": ("asi05_unexpected_code_execution", "unsafe_deserialization"),
        "B302": ("asi05_unexpected_code_execution", "unsafe_deserialization"),
        "B403": ("asi05_unexpected_code_execution", "unsafe_deserialization"),
        # ASI02 — Tool misuse
        "B603": ("asi02_tool_misuse", "unvalidated_tool_input"),
        "B108": ("asi02_tool_misuse", "path_traversal"),
        # ASI03 — Identity / credentials
        "B105": ("asi03_identity_privilege_abuse", "hardcoded_credentials"),
        "B106": ("asi03_identity_privilege_abuse", "hardcoded_credentials"),
        "B107": ("asi03_identity_privilege_abuse", "hardcoded_credentials"),
        "B501": ("asi03_identity_privilege_abuse", "missing_auth_on_endpoint"),
        "B502": ("asi03_identity_privilege_abuse", "missing_auth_on_endpoint"),
        "B503": ("asi03_identity_privilege_abuse", "missing_auth_on_endpoint"),
        "B504": ("asi03_identity_privilege_abuse", "missing_auth_on_endpoint"),
        # ASI01 — Goal hijack via template injection
        "B701": ("asi01_agent_goal_hijack", "indirect_prompt_injection"),
        "B702": ("asi01_agent_goal_hijack", "indirect_prompt_injection"),
        # ASI06 — YAML unsafe load → memory/RAG poisoning vector
        "B506": ("asi06_memory_context_poisoning", "memory_poisoning"),
    }

    # Keyword fallbacks
    msg_lower = message.lower()
    if (
        "subprocess" in msg_lower
        or "shell=true" in msg_lower
        or "os.system" in msg_lower
    ):
        return ("asi05_unexpected_code_execution", "arbitrary_code_execution")
    if "eval" in msg_lower or "exec" in msg_lower:
        return ("asi05_unexpected_code_execution", "arbitrary_code_execution")
    if "pickle" in msg_lower or "deserializ" in msg_lower or "yaml.load" in msg_lower:
        return ("asi05_unexpected_code_execution", "unsafe_deserialization")
    if "hardcoded" in msg_lower or "password" in msg_lower or "secret" in msg_lower:
        return ("asi03_identity_privilege_abuse", "hardcoded_credentials")
    if "tls" in msg_lower or "ssl" in msg_lower or "certificate" in msg_lower:
        return ("asi03_identity_privilege_abuse", "missing_auth_on_endpoint")
    if "sql" in msg_lower or "injection" in msg_lower:
        return ("asi02_tool_misuse", "unvalidated_tool_input")

    return rule_map.get(bandit_rule_id, ("asi02_tool_misuse", "unvalidated_tool_input"))


# ── OpenGrep rule → (ASI category key, subcategory) ──────────────────────────


def map_opengrep_to_taxonomy(rule_id: str) -> tuple[str, str]:
    """Map an OpenGrep rule ID to the ASI-aligned taxonomy category/subcategory."""
    rule_map = {
        # ASI10
        "crewai-allow-delegation-true": (
            "asi10_rogue_agents",
            "unchecked_agent_delegation",
        ),
        # ASI06
        "crewai-memory-enabled": (
            "asi06_memory_context_poisoning",
            "persistent_memory_no_sanitization",
        ),
        "crewai-crew-memory-enabled": (
            "asi06_memory_context_poisoning",
            "cross_session_contamination",
        ),
        # ASI09
        "autogen-human-input-never": (
            "asi09_human_agent_trust_exploitation",
            "no_human_in_the_loop",
        ),
        # ASI05
        "autogen-code-execution-enabled": (
            "asi05_unexpected_code_execution",
            "arbitrary_code_execution",
        ),
        "subprocess-shell-true": (
            "asi05_unexpected_code_execution",
            "arbitrary_code_execution",
        ),
        "os-system-call": (
            "asi05_unexpected_code_execution",
            "arbitrary_code_execution",
        ),
        "eval-call": ("asi05_unexpected_code_execution", "arbitrary_code_execution"),
        "unsafe-yaml-load": (
            "asi05_unexpected_code_execution",
            "unsafe_deserialization",
        ),
        "pickle-load": ("asi05_unexpected_code_execution", "unsafe_deserialization"),
        # ASI01
        "llm-input-fstring": ("asi01_agent_goal_hijack", "direct_prompt_injection"),
        # ASI03
        "hardcoded-api-key": (
            "asi03_identity_privilege_abuse",
            "hardcoded_credentials",
        ),
        "no-tls-server": ("asi03_identity_privilege_abuse", "missing_auth_on_endpoint"),
        "server-binding-all-interfaces": (
            "asi03_identity_privilege_abuse",
            "missing_auth_on_endpoint",
        ),
        # ASI04
        "unpinned-requirements": ("asi04_supply_chain", "unpinned_dependencies"),
        "unpinned-ai-dependency": ("asi04_supply_chain", "unpinned_dependencies"),
        # ASI07
        "no-agent-message-auth": (
            "asi07_insecure_interagent_comms",
            "unauthenticated_agent_communication",
        ),
        "http-agent-endpoint": (
            "asi07_insecure_interagent_comms",
            "unencrypted_agent_channel",
        ),
        "unvalidated-agent-trust": (
            "asi07_insecure_interagent_comms",
            "unvalidated_agent_message",
        ),
        # ASI08
        "no-max-iter": ("asi08_cascading_failures", "unbounded_agent_loop"),
        "no-agent-timeout": ("asi08_cascading_failures", "unbounded_agent_loop"),
        # ASI09
        "no-confirmation-gate": (
            "asi09_human_agent_trust_exploitation",
            "missing_action_confirmation",
        ),
        # ASI10
        "no-kill-switch": ("asi10_rogue_agents", "missing_kill_switch"),
        "missing-agent-monitoring": ("asi10_rogue_agents", "missing_agent_monitoring"),
        # Google ADK
        "adk-loop-no-max-iterations": (
            "asi08_cascading_failures",
            "unbounded_agent_loop",
        ),
        "adk-agent-code-executor": (
            "asi05_unexpected_code_execution",
            "arbitrary_code_execution",
        ),
        "adk-agent-unchecked-delegation": (
            "asi10_rogue_agents",
            "unchecked_agent_delegation",
        ),
        "adk-agent-no-instruction": (
            "asi01_agent_goal_hijack",
            "goal_manipulation_via_rag",
        ),
        # OpenAI Agents SDK
        "openai-runner-no-max-turns": (
            "asi08_cascading_failures",
            "unbounded_agent_loop",
        ),
        "openai-agent-no-input-guardrails": (
            "asi01_agent_goal_hijack",
            "direct_prompt_injection",
        ),
        "openai-agent-no-output-guardrails": (
            "asi10_rogue_agents",
            "missing_behavioral_guardrails",
        ),
        "openai-agent-unchecked-handoffs": (
            "asi10_rogue_agents",
            "unchecked_agent_delegation",
        ),
        # Anthropic
        "anthropic-tool-loop-no-limit": (
            "asi08_cascading_failures",
            "unbounded_agent_loop",
        ),
        "anthropic-no-max-tokens": (
            "asi08_cascading_failures",
            "unbounded_agent_loop",
        ),
        "anthropic-tool-choice-any": (
            "asi02_tool_misuse",
            "excessive_tool_permissions",
        ),
        # LangGraph / LangChain
        "langgraph-react-no-interrupt": (
            "asi09_human_agent_trust_exploitation",
            "missing_action_confirmation",
        ),
        "langchain-agent-no-max-iterations": (
            "asi08_cascading_failures",
            "unbounded_agent_loop",
        ),
        "langgraph-no-recursion-limit": (
            "asi08_cascading_failures",
            "unbounded_agent_loop",
        ),
        # Claude Agents SDK
        "claude-query-no-max-turns": (
            "asi08_cascading_failures",
            "unbounded_agent_loop",
        ),
        "claude-client-no-max-turns": (
            "asi08_cascading_failures",
            "unbounded_agent_loop",
        ),
        "claude-options-no-system-prompt": (
            "asi01_agent_goal_hijack",
            "goal_manipulation_via_rag",
        ),
        "claude-options-bash-tool": (
            "asi05_unexpected_code_execution",
            "arbitrary_code_execution",
        ),
        "claude-subagent-no-prompt": (
            "asi01_agent_goal_hijack",
            "goal_manipulation_via_rag",
        ),
        "claude-subagent-bash-tool": (
            "asi05_unexpected_code_execution",
            "arbitrary_code_execution",
        ),
        # CrewAI (new rules)
        "crewai-crew-no-max-iter": (
            "asi08_cascading_failures",
            "unbounded_agent_loop",
        ),
        "crewai-agent-verbose-false": (
            "asi10_rogue_agents",
            "missing_agent_monitoring",
        ),
        # AutoGen (new rules)
        "autogen-groupchat-no-max-round": (
            "asi08_cascading_failures",
            "unbounded_agent_loop",
        ),
        "autogen-roundrobin-no-termination": (
            "asi08_cascading_failures",
            "unbounded_agent_loop",
        ),
    }

    # Strip path prefix from rule ID (opengrep adds the rules file path)
    short_id = rule_id.split(".")[-1] if "." in rule_id else rule_id
    return rule_map.get(
        short_id, rule_map.get(rule_id, ("asi02_tool_misuse", "unvalidated_tool_input"))
    )
