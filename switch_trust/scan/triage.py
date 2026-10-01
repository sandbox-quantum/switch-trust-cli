"""
triage.py — Layer 2d: Triage Agent
Sits downstream of the primary AI reasoning layer (ai_reasoner.py) and applies
two layers of judgment to the raw findings list:

  1. Contextual relevance filtering — dismisses findings that describe expected
     behaviour for this specific agent's declared purpose.

  2. Severity calibration — downgrades findings whose severity is disproportionate
     to the realistic exploitability and impact given the available evidence.

This is a single, focused LLM call. No ADK, no streaming, no tools.
It receives findings + agent context, and returns a filtered + recalibrated
version of the same findings list along with triage metadata.

The LLM provider is configurable via environment variables — see llm_provider.py.

Usage (called from agent_scanner.py after AI reasoning step):

    from triage import run_triage

    triage_result = run_triage(
        raw_findings=all_findings_as_dicts,
        agent_context=build_agent_context(discovery, agents, crews),
    )

    if triage_result:
        kept       = triage_result["kept_findings"]
        dismissed  = triage_result["triage_dismissed"]
        downgraded = triage_result["triage_downgraded"]
        summary    = triage_result["triage_summary"]
"""

import json
import logging
import os
import re
from typing import Any

from switch_trust.scan.schema import RawFinding
from switch_trust.scan import ADKModel
from switch_trust.scan.llm_provider import complete_text, make_model

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

MAX_TOKENS = (
    16000  # Must accommodate triage output for large finding sets (49+ findings)
)
TEMPERATURE = (
    0.0  # Fully deterministic — same input must always produce same triage result
)
TOP_P = 1.0

# Maximum CVSS base_score per severity band (FIRST.org CVSS v4 upper bounds)
_SEVERITY_SCORE_CAP: dict[str, float] = {
    "Critical": 10.0,
    "High": 8.9,
    "Medium": 6.9,
    "Low": 3.9,
    "Info": 0.0,
}

# Severity levels and ranking for enforcement
_SEV_LEVELS = ["Critical", "High", "Medium", "Low"]
_SEV_RANK = {s: i for i, s in enumerate(_SEV_LEVELS)}

# Patterns that prove a flaw directly — if matched, block downgrades.
_ANCHOR_PATTERNS = [
    re.compile(r"\beval\s*\("),
    re.compile(r"\bexec\s*\("),
    re.compile(r"shell\s*=\s*True"),
    re.compile(r"pickle\.load"),
    re.compile(r"yaml\.load\s*\("),
    re.compile(
        r"(?:password|api_key|secret|token)\s*=\s*['\"][^'\"]{8,}['\"]",
        re.IGNORECASE,
    ),
]

# Path to the prompt file
_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "config", "triage_prompt.txt")


# ── CVSS score capping ─────────────────────────────────────────────────────────


def _cap_cvss_score(finding: dict[str, Any], severity: str) -> None:
    """Cap a finding's cvss_scores.base_score to the max of the given severity band.

    Mutates the finding dict in place. If the finding has no cvss_scores or the
    score is already within the band, this is a no-op.
    """
    cap = _SEVERITY_SCORE_CAP.get(severity)
    if cap is None:
        return
    scores = finding.get("cvss_scores")
    if not isinstance(scores, dict):
        return
    base = scores.get("base_score")
    if isinstance(base, (int, float)) and base > cap:
        logger.info(
            "Capping CVSS score for %s from %.1f to %.1f (severity %s)",
            finding.get("id", "?"),
            base,
            cap,
            severity,
        )
        scores["base_score"] = cap


# ── Post-triage severity enforcement ──────────────────────────────────────────


def _has_anchor_evidence(finding: dict[str, Any]) -> bool:
    """Return True if evidence matches a Direct Code Evidence Anchor pattern."""
    for ev in finding.get("evidence", []):
        snippet = ev.get("code_snippet", "") if isinstance(ev, dict) else ""
        if not snippet:
            continue
        for pat in _ANCHOR_PATTERNS:
            if pat.search(snippet):
                return True
    return False


def _enforce_severity(
    kept_findings: list[dict[str, Any]],
    original_findings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic post-LLM severity enforcement.

    Rules:
    1. ANCHOR — if evidence matches a code pattern, block downgrades.
    2. NO UPGRADE — severity may never exceed cvss_v4_severity.
    3. MAX ONE-STEP — severity may drop at most one level from CVSS.
    """
    originals_by_id = {f.get("id"): f for f in original_findings}
    enforced: list[dict[str, Any]] = []

    for f in kept_findings:
        fid = f.get("id")
        original = originals_by_id.get(fid, {})
        cvss_sev = original.get("cvss_v4_severity", "").capitalize()

        if cvss_sev not in _SEV_RANK:
            continue

        llm_sev = f.get("ai_spm_severity", "").capitalize()
        if llm_sev not in _SEV_RANK:
            llm_sev = cvss_sev

        cvss_rank = _SEV_RANK[cvss_sev]
        llm_rank = _SEV_RANK[llm_sev]

        if _has_anchor_evidence(original) and llm_rank > cvss_rank:
            enforced.append(
                {
                    "id": fid,
                    "rule": "anchor",
                    "llm_severity": llm_sev,
                    "enforced_severity": cvss_sev,
                }
            )
            f["ai_spm_severity"] = cvss_sev
            continue

        if llm_rank < cvss_rank:
            enforced.append(
                {
                    "id": fid,
                    "rule": "no_upgrade",
                    "llm_severity": llm_sev,
                    "enforced_severity": cvss_sev,
                }
            )
            f["ai_spm_severity"] = cvss_sev
            continue

        if llm_rank > cvss_rank + 1:
            one_step = _SEV_LEVELS[cvss_rank + 1]
            enforced.append(
                {
                    "id": fid,
                    "rule": "max_one_step",
                    "llm_severity": llm_sev,
                    "enforced_severity": one_step,
                }
            )
            f["ai_spm_severity"] = one_step

    return kept_findings, enforced


# ── Prompt loader ──────────────────────────────────────────────────────────────


def _load_triage_prompt() -> str:
    """Load the triage system prompt from config/triage_prompt.txt."""
    try:
        with open(_PROMPT_PATH, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"[triage] Triage prompt not found at: {_PROMPT_PATH}\n"
            "Ensure config/triage_prompt.txt exists before running triage."
        ) from e


# ── Context builder ────────────────────────────────────────────────────────────


def build_agent_context(
    repo_name: str,
    framework: str,
    agent_profiles: list[dict[str, Any]],
    description: str = "",
) -> dict[str, Any]:
    """
    Build the agent_context dict that gives the triage agent enough information
    to make contextual relevance decisions.

    Called from agent_scanner.py — pass in the values already available after
    the discovery and extraction steps.

    Parameters
    ----------
    repo_name      : repo name string from DiscoveryResult.repo_name
    framework      : primary framework string from DiscoveryResult.primary_framework
    agent_profiles : list of agent profile dicts (from ScanReport.agent_profiles)
    description    : optional free-text description of the repo/agent purpose
                     (from README first paragraph or directory listing description)

    Returns
    -------
    dict suitable for passing directly to run_triage() as agent_context.
    """
    tool_inventory = []
    for profile in agent_profiles:
        agent_id = profile.get("agent_id", "unknown")
        for tool in profile.get("tools", []):
            if tool not in tool_inventory:
                tool_inventory.append(f"{agent_id}: {tool}")

    compact_profiles = []
    for p in agent_profiles:
        compact_profiles.append(
            {
                "agent_id": p.get("agent_id"),
                "role": p.get("role"),
                "goal": p.get("goal"),
                "backstory": p.get("backstory"),
                "tools": p.get("tools", []),
                "memory": p.get("memory"),
                "allow_delegation": p.get("allow_delegation"),
            }
        )

    return {
        "repo_name": repo_name,
        "description": description,
        "framework": framework,
        "agent_profiles": compact_profiles,
        "tool_inventory": tool_inventory,
    }


# ── Triage caller ──────────────────────────────────────────────────────────────


def run_triage(
    raw_findings: list[RawFinding],
    agent_context: dict[str, Any],
    model: ADKModel | None = None,
) -> dict[str, Any] | None:
    """
    Run the triage agent over a list of raw findings.

    Parameters
    ----------
    raw_findings  : list of finding dicts — the combined static + AI findings
                    produced by agent_scanner.py before triage. Each dict must
                    include at minimum: id, ai_spm_severity, source,
                    evidence (list with confidence), hallucination_flag.
    agent_context : dict built by build_agent_context() — describes the agent's
                    declared purpose, framework, and tool inventory.
    model         : ADK model object from make_model(). If None, resolved
                    automatically via make_model() using environment variables.

    Returns
    -------
    dict with keys:
        kept_findings     -- findings that survive triage (with updated severities)
        triage_dismissed  -- list of {id, reason}
        triage_downgraded -- list of {id, original_severity, new_severity, reason}
        triage_summary    -- counts and severity distribution

    Returns None if triage fails (caller should fall back to raw_findings).
    """
    if not raw_findings:
        logger.info("No findings to triage — skipping.")
        return None

    # ── Resolve provider ───────────────────────────────────────────────────────
    if model is None:
        try:
            model = make_model(scanner="agent", phase="triage")
        except (ValueError, ImportError) as e:
            logger.error("Failed to create model: %s", e)
            return None

    # ── Load prompt ────────────────────────────────────────────────────────────
    try:
        system_prompt = _load_triage_prompt()
    except FileNotFoundError as e:
        logger.error("Triage prompt load failed: %s", e)
        return None

    # CVE deduplication: OSV returns one finding per CVE per package.
    # For triage, all known_vulnerable_dependency findings for the same
    # package carry the same contextual decision (keep/dismiss). Collapse
    # them to one representative per package, then expand back after triage.
    collapsed_findings = []
    cve_groups = {}  # package_name -> list of original finding dicts
    representative_ids = {}  # package_name -> finding_id of the representative

    for f in raw_findings:
        is_cve_finding = (
            f.get("subcategory") == "known_vulnerable_dependency"
            and f.get("source", "") == "static_pip_audit"
        )
        if is_cve_finding:
            # Extract package name from affected_components list
            _ac = f.get("affected_components", [])
            pkg_name = (
                _ac[0].get("name", "") if _ac and isinstance(_ac[0], dict) else ""
            ).strip()
            if not pkg_name:
                collapsed_findings.append(f)
                continue
            if pkg_name not in cve_groups:
                cve_groups[pkg_name] = []
                # Fix #7: start with the first finding as the representative;
                # severity will be upgraded to the maximum across the group below.
                representative = dict(f)
                representative["description"] = (
                    f"{pkg_name}: multiple CVEs detected (representative finding — "
                    f"triage decision applies to all CVEs for this package)"
                )
                collapsed_findings.append(representative)
                representative_ids[pkg_name] = representative["id"]
            cve_groups[pkg_name].append(f)
        else:
            collapsed_findings.append(f)

    # For each CVE group, set the representative's severity to the maximum
    # severity across all CVEs in the group so a medium representative
    # cannot mask a critical CVE in the same package.
    _SEV_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4}
    for pkg_name, cves in cve_groups.items():
        rep_id = representative_ids[pkg_name]
        max_sev = min(
            (c.get("ai_spm_severity", "Info") for c in cves),
            key=lambda s: _SEV_ORDER.get(s, 99),
        )
        for rep in collapsed_findings:
            if rep.get("id") == rep_id:
                rep["ai_spm_severity"] = max_sev
                break

    if cve_groups:
        saved = sum(len(v) - 1 for v in cve_groups.values())
        logger.info(
            "CVE dedup: collapsed %d CVE findings into %d package representatives "
            "(saving %d tokens)",
            sum(len(v) for v in cve_groups.values()),
            len(cve_groups),
            saved,
        )

    triage_input = collapsed_findings

    # ── Build user message ─────────────────────────────────────────────────────
    finding_id_list = [f.get("id") for f in triage_input]
    user_message = (
        f"You must account for ALL {len(triage_input)} findings listed in "
        f"ids_to_process. Every ID must appear in either "
        f"kept_finding_ids or triage_dismissed. Do not omit any.\n\n"
        + json.dumps(
            {
                "agent_context": agent_context,
                "ids_to_process": finding_id_list,
                "raw_findings": triage_input,
            },
            indent=2,
        )
    )

    logger.info(
        "Sending %d findings to triage agent (%s) [%d total before CVE dedup]...",
        len(triage_input),
        model,
        len(raw_findings),
    )

    # ── LLM call ───────────────────────────────────────────────────────────────
    raw_text = complete_text(
        model=model,
        system_prompt=system_prompt,
        user_message=user_message,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    if raw_text is None:
        logger.info("Model returned no response — using raw findings unchanged")
        return None

    # Strip code fences if the model wraps output despite json mode
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        ).strip()
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("JSON parse error: %s", e)
        logger.error("Raw response (first 500 chars): %s", text[:500])
        # On JSON parse failure, return an identity result (keep all findings,
        # dismiss none) — returning None causes the caller to silently discard
        # all triage work.
        logger.info(
            "Falling back to identity result (all findings kept, none dismissed)"
        )
        return {
            "kept_findings": list(raw_findings),
            "triage_dismissed": [],
            "triage_downgraded": [],
            "triage_summary": {
                "parse_error": True,
                "total_input": len(raw_findings),
                "total_kept": len(raw_findings),
                "total_dismissed": 0,
                "total_downgraded": 0,
                "severity_distribution": {},
            },
        }

    # ── Validate required keys ─────────────────────────────────────────────────
    required_keys = {
        "kept_finding_ids",
        "triage_dismissed",
        "triage_downgraded",
        "triage_summary",
    }
    missing = required_keys - set(result.keys())
    if missing:
        # Backward-compat: accept kept_findings if model used old schema
        if "kept_findings" in result and "kept_finding_ids" not in result:
            result["kept_finding_ids"] = [
                {"id": f.get("id"), "ai_spm_severity": f.get("ai_spm_severity")}
                for f in result["kept_findings"]
            ]
        else:
            logger.error("Response missing required keys: %s", missing)
            return None

    # ── Reconstruct full finding objects from kept_finding_ids ────────────────
    triage_input_by_id = {f.get("id"): f for f in triage_input}
    kept_findings = []
    for item in result.get("kept_finding_ids", []):
        fid = item.get("id")
        new_sev = item.get("ai_spm_severity")
        original = triage_input_by_id.get(fid)
        if original is None:
            logger.warning("kept_finding_ids contains unknown id %s — skipping", fid)
            continue
        reconstructed = dict(original)
        if new_sev:
            reconstructed["ai_spm_severity"] = new_sev
            _cap_cvss_score(reconstructed, new_sev)
        kept_findings.append(reconstructed)
    result["kept_findings"] = kept_findings

    # ── Severity enforcement ──────────────────────────────────────────────────
    result["kept_findings"], severity_overrides = _enforce_severity(
        result["kept_findings"],
        triage_input,
    )
    if severity_overrides:
        logger.info(
            "Severity enforcement: %d finding(s) clamped to CVSS baseline",
            len(severity_overrides),
        )
        for ov in severity_overrides:
            logger.info(
                "  %s: %s -> %s (rule: %s)",
                ov["id"],
                ov["llm_severity"],
                ov["enforced_severity"],
                ov["rule"],
            )

    # ── Integrity check ────────────────────────────────────────────────────────
    input_ids = {f.get("id") for f in triage_input}
    kept_ids = {f.get("id") for f in result["kept_findings"]}
    dismissed_ids = {d.get("id") for d in result.get("triage_dismissed", [])}
    unaccounted = input_ids - (kept_ids | dismissed_ids)

    if unaccounted:
        missing_findings = [f for f in triage_input if f.get("id") in unaccounted]
        for mf in missing_findings:
            logger.warning(
                "finding %s (%s / %s) was silently dropped — re-adding as kept unchanged",
                mf.get("id"),
                mf.get("subcategory", "?"),
                mf.get("ai_spm_severity", "?"),
            )
        result["kept_findings"].extend(missing_findings)

    # ── CVE expansion ──────────────────────────────────────────────────────────
    if cve_groups:
        rep_severity_map = {
            f.get("id"): f.get("ai_spm_severity")
            for f in result.get("kept_findings", [])
        }
        expanded_kept = []
        for f in result.get("kept_findings", []):
            fid = f.get("id")
            pkg_for_rep = next(
                (pkg for pkg, rid in representative_ids.items() if rid == fid), None
            )
            if pkg_for_rep and pkg_for_rep in cve_groups:
                triaged_sev = rep_severity_map.get(fid, f.get("ai_spm_severity"))
                for cve_f in cve_groups[pkg_for_rep]:
                    expanded = dict(cve_f)
                    expanded["ai_spm_severity"] = triaged_sev
                    _cap_cvss_score(expanded, triaged_sev)
                    expanded_kept.append(expanded)
            else:
                expanded_kept.append(f)

        expanded_dismissed = list(result.get("triage_dismissed", []))
        for d in result.get("triage_dismissed", []):
            did = d.get("id")
            pkg_for_rep = next(
                (pkg for pkg, rid in representative_ids.items() if rid == did), None
            )
            if pkg_for_rep and pkg_for_rep in cve_groups:
                for cve_f in cve_groups[pkg_for_rep]:
                    if cve_f.get("id") != did:
                        expanded_dismissed.append(
                            {
                                "id": cve_f.get("id"),
                                "reason": d.get("reason", "")
                                + " (CVE group dismissal)",
                            }
                        )

        expanded_downgraded = list(result.get("triage_downgraded", []))
        for d in result.get("triage_downgraded", []):
            did = d.get("id")
            pkg_for_rep = next(
                (pkg for pkg, rid in representative_ids.items() if rid == did), None
            )
            if pkg_for_rep and pkg_for_rep in cve_groups:
                for cve_f in cve_groups[pkg_for_rep]:
                    if cve_f.get("id") != did:
                        expanded_downgraded.append(
                            {
                                "id": cve_f.get("id"),
                                "original_severity": d.get("original_severity"),
                                "new_severity": d.get("new_severity"),
                                "reason": d.get("reason", "")
                                + " (CVE group downgrade)",
                            }
                        )

        result["kept_findings"] = expanded_kept
        result["triage_dismissed"] = expanded_dismissed
        result["triage_downgraded"] = expanded_downgraded
        logger.info(
            "CVE expansion: %d final kept findings (%d dismissed)",
            len(expanded_kept),
            len(result.get("triage_dismissed", [])),
        )

    # ── Log summary ────────────────────────────────────────────────────────────
    summary = result.get("triage_summary", {})
    dist = summary.get("severity_distribution", {})
    logger.info(
        "Complete — input: %d | kept: %d | dismissed: %d | downgraded: %d",
        summary.get("total_input", len(raw_findings)),
        summary.get("total_kept", len(result["kept_findings"])),
        summary.get("total_dismissed", len(result["triage_dismissed"])),
        summary.get("total_downgraded", len(result["triage_downgraded"])),
    )
    logger.info(
        "Post-triage severity distribution — critical: %d | high: %d | medium: %d | low: %d",
        dist.get("critical", 0),
        dist.get("high", 0),
        dist.get("medium", 0),
        dist.get("low", 0),
    )

    return result
