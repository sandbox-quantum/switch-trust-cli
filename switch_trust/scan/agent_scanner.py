"""
agent_scanner.py — Main Orchestrator
Ties together all layers:
  Layer 1: Discovery (file collection)
  Layer 2: Static scan (Bandit, OpenGrep, detect-secrets, pip-audit)
  Layer 3: AI reasoning (agentic semantic analysis)
  Layer 4: Triage (contextual filtering + severity calibration)
Output: Structured JSON report.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import tempfile
import uuid
from datetime import UTC, datetime
from typing import Any

from switch_trust.scan.constants import (
    REQUIREMENT_FILE_NAMES,
    SEVERITY_ORDER,
    VALID_SEVERITIES,
)
from switch_trust.schema import RepoFile
from switch_trust.scan.reasoner import run_agentic_reasoning
from switch_trust.scan.schema import (
    AffectedComponent,
    AgentProfile,
    CategorySummaryEntry,
    CvssScores,
    Evidence,
    Finding,
    InventoryLike,
    RawFinding,
    ScanReport,
    TriageDismissed,
    TriageDowngraded,
)
from switch_trust.scan.scorer import (
    map_bandit_to_taxonomy,
    map_opengrep_to_taxonomy,
    score_finding,
)
from switch_trust.scan.static_scanner import (
    StaticFinding,
    run_static_scan,
)
from switch_trust.scan.taxonomy import (
    AGENT_TAXONOMY,
    get_finding_metadata,
    resolve_subcategory,
)
from switch_trust.scan.triage import (
    build_agent_context as build_triage_context,
)
from switch_trust.scan.triage import run_triage
from switch_trust.scan.llm_provider import get_model_name, make_model
from switch_trust.secret_anonymizer import anonymize_secrets

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_finding_id() -> str:
    return f"AGT-{uuid.uuid4().hex[:8].upper()}"


def _cvss_v4_severity_from_score(score: float) -> str:
    """Derive CVSS v4 qualitative severity label from a numeric score."""
    if score >= 9.0:
        return "Critical"
    elif score >= 7.0:
        return "High"
    elif score >= 4.0:
        return "Medium"
    elif score > 0.0:
        return "Low"
    return "None"


def _derive_likelihood(confidence: str, severity: str) -> str:
    """Derive exploitation likelihood from confidence and severity."""
    severity_l = severity.lower()
    confidence_l = confidence.lower()
    if confidence_l == "high" and severity_l in ("critical", "high"):
        return "High"
    if confidence_l == "low" or severity_l in ("low", "info"):
        return "Low"
    return "Medium"


def _dicts_to_findings(finding_dicts: list[dict[str, Any]]) -> list[Finding]:
    """Reconstruct Finding objects from serialised dicts (post-triage kept_findings)."""
    findings = []
    for d in finding_dicts:
        try:
            if not isinstance(d, dict):
                continue
            # Reconstruct nested dataclasses from dicts
            if isinstance(d.get("cvss_scores"), dict):
                d["cvss_scores"] = CvssScores(**d["cvss_scores"])
            if isinstance(d.get("affected_components"), list):
                d["affected_components"] = [
                    AffectedComponent(**ac) if isinstance(ac, dict) else ac
                    for ac in d["affected_components"]
                ]
            if isinstance(d.get("evidence"), list):
                d["evidence"] = [
                    Evidence(**e) if isinstance(e, dict) else e for e in d["evidence"]
                ]
            findings.append(Finding(**d))
        except (TypeError, ValueError, KeyError) as e:
            finding_id = d.get("id", "?") if isinstance(d, dict) else "?"
            logger.error("Skipping malformed finding dict %s: %s", finding_id, e)
            continue
    if len(findings) < len(finding_dicts):
        logger.warning(
            "%d of %d findings were skipped due to schema mismatches",
            len(finding_dicts) - len(findings),
            len(finding_dicts),
        )
    return findings


# ── Static findings → Finding objects ────────────────────────────────────────


def convert_static_findings(static_findings: list[StaticFinding]) -> list[Finding]:
    """Convert StaticFinding objects to canonical Finding objects with CVSS scoring."""
    findings = []
    seen = set()  # Deduplicate by (rule_id, filepath, line)
    conversion_errors = 0

    for sf in static_findings:
        try:
            # Validate StaticFinding object has required attributes
            if (
                not hasattr(sf, "tool")
                or not hasattr(sf, "rule_id")
                or not hasattr(sf, "filepath")
            ):
                logger.error("StaticFinding missing required attributes: %s", sf)
                conversion_errors += 1
                continue

            # Map to taxonomy first — subcategory is used in the dedup key.
            # Fix #24: dedup by (subcategory, filepath, line) instead of
            # (rule_id, filepath, line). Bandit B602 and OpenGrep
            # subprocess-shell-true both map to the same subcategory on the
            # same line. The old key treated them as distinct findings;
            # the new key correctly collapses them into one.
            if sf.tool == "bandit":
                category, subcategory = map_bandit_to_taxonomy(sf.rule_id, sf.message)
            elif sf.tool == "opengrep":
                category, subcategory = map_opengrep_to_taxonomy(sf.rule_id)
            elif sf.tool == "detect_secrets":
                category, subcategory = (
                    "asi03_identity_privilege_abuse",
                    "hardcoded_credentials",
                )
            elif sf.tool == "pip_audit":
                category, subcategory = (
                    "asi04_supply_chain",
                    "known_vulnerable_dependency",
                )
            elif sf.tool == "internal":
                category, subcategory = "asi04_supply_chain", "unpinned_dependencies"
            else:
                category, subcategory = "asi04_supply_chain", "unpinned_dependencies"

            dedup_key = (subcategory, sf.filepath, sf.line)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Score the finding with error handling
            try:
                cvss_score, cvss_vector, severity = score_finding(subcategory, sf.tool)
            except Exception as e:
                logger.warning(
                    "Could not score finding %s: %s. Defaulting to medium severity.",
                    sf.rule_id,
                    e,
                )
                cvss_score, cvss_vector, severity = (
                    5.5,
                    "CVSS:4.0/AV:N/AC:L/AT:N/PR:L/UI:N/VC:L/VI:L/VA:L/SC:N/SI:N/SA:N",
                    "medium",
                )

            meta = get_finding_metadata(subcategory)

            # For pip_audit findings, affected_component is the package name
            # (e.g. "langchain") not the filepath — more meaningful in the report
            # and enables correct per-package CVE grouping in triage.
            if sf.tool == "pip_audit":
                affected = sf.message.split("==")[0].strip() or sf.filepath
                # OSV already computed severity from the actual CVE CVSS score.
                # Use it directly rather than letting scorer.py override with taxonomy default.
                if hasattr(sf, "severity") and sf.severity:
                    severity = sf.severity
                cvss_score, cvss_vector, _ = score_finding(subcategory, sf.tool)
            else:
                affected = sf.filepath

            # Validate severity is in expected set
            valid_severities = VALID_SEVERITIES
            if severity not in valid_severities:
                logger.warning(
                    "Invalid severity '%s' for %s. Defaulting to 'medium'.",
                    severity,
                    sf.rule_id,
                )
                severity = "medium"

            confidence = "High" if severity in ("high", "critical") else "Medium"

            findings.append(
                Finding(
                    id=make_finding_id(),
                    category=category,
                    subcategory=subcategory,
                    ai_spm_severity=severity.capitalize(),
                    cvss_v4_severity=_cvss_v4_severity_from_score(cvss_score),
                    cvss_scores=CvssScores(base_score=cvss_score, vector=cvss_vector),
                    title=meta.get("title", sf.rule_id),
                    description=sf.message,
                    impact=meta.get(
                        "description",
                        f"Security issue detected in {sf.filepath}:{sf.line}",
                    ),
                    likelihood=_derive_likelihood(confidence, severity),
                    remediation=meta.get(
                        "remediation",
                        f"See OWASP {meta.get('asi_code', 'ASI')} guidance for {subcategory}.",
                    ),
                    affected_components=[
                        AffectedComponent(name=affected, path=sf.filepath)
                    ],
                    evidence=[
                        Evidence(
                            file=sf.filepath,
                            code_snippet=anonymize_secrets(sf.evidence[:200])
                            if sf.evidence
                            else "",
                            confidence=confidence,
                            line=sf.line,
                        )
                    ],
                    source=f"static_{sf.tool}",
                    hallucination_flag=False,
                )
            )
        except Exception as e:
            logger.error(
                "Failed to convert finding %s: %s",
                getattr(sf, "rule_id", "?"),
                e,
            )
            conversion_errors += 1
            continue

    if conversion_errors > 0:
        logger.warning(
            "%d static findings failed to convert. Continuing with %d valid findings.",
            conversion_errors,
            len(findings),
        )

    return findings


# ── AI findings → Finding objects ─────────────────────────────────────────────


def convert_ai_findings(ai_findings: list[RawFinding]) -> list[Finding]:
    """Convert raw AI reasoning finding dicts to canonical Finding objects."""
    findings = []
    conversion_errors = 0

    if not isinstance(ai_findings, list):
        logger.error("ai_findings is not a list, got %s", type(ai_findings).__name__)
        return findings

    for raw in ai_findings:
        try:
            if not isinstance(raw, dict):
                logger.error("AI finding is not a dict, got %s", type(raw).__name__)
                conversion_errors += 1
                continue

            subcategory = raw.get("subcategory", "unvalidated_tool_input")
            # The category is derived, never read off the raw finding. See
            # taxonomy.category_for_subcategory: the reasoning model was asked
            # for a key the prompt never showed it, so it inferred one from the
            # human title and got it wrong for five of the ten categories.
            subcategory = resolve_subcategory(subcategory) or subcategory
            meta = get_finding_metadata(subcategory)
            category = meta["category"]

            is_beyond = category == "beyond_asi"

            if is_beyond:
                # BEYOND-ASI findings: use a generic medium vector
                cvss_score = 5.5
                cvss_vector = (
                    "CVSS:4.0/AV:N/AC:L/AT:N/PR:L/UI:N/VC:L/VI:L/VA:L/SC:N/SI:N/SA:N"
                )
                severity = "medium"
            else:
                try:
                    cvss_score, cvss_vector, severity = score_finding(
                        subcategory, "ai_reasoning"
                    )
                except Exception as e:
                    logger.warning(
                        "Could not score AI finding '%s': %s. Defaulting to medium severity.",
                        raw.get("title", "?"),
                        e,
                    )
                    cvss_score = 5.5
                    cvss_vector = "CVSS:4.0/AV:N/AC:L/AT:N/PR:L/UI:N/VC:L/VI:L/VA:L/SC:N/SI:N/SA:N"  # noqa: B950
                    severity = "medium"

            # Hallucinated findings are capped at medium
            is_hallucinated = raw.get("hallucination_flag", False)
            if is_hallucinated and severity in ("critical", "high"):
                severity = "medium"
                cvss_score = min(cvss_score, 6.9)

            # Validate severity
            valid_severities = VALID_SEVERITIES
            if severity not in valid_severities:
                logger.warning(
                    "Invalid severity '%s' for AI finding. Defaulting to 'medium'.",
                    severity,
                )
                severity = "medium"

            # Validate confidence value
            valid_confidences = {"high", "medium", "low"}
            confidence = raw.get("confidence", "medium").lower()
            if confidence not in valid_confidences:
                confidence = "medium"

            affected_component = raw.get("affected_component", "")

            findings.append(
                Finding(
                    id=make_finding_id(),
                    category=category,
                    subcategory=subcategory,
                    ai_spm_severity=severity.capitalize(),
                    cvss_v4_severity=_cvss_v4_severity_from_score(cvss_score),
                    cvss_scores=CvssScores(base_score=cvss_score, vector=cvss_vector),
                    title=raw.get("title", meta.get("title", subcategory)),
                    description=raw.get("description", ""),
                    impact=raw.get("impact", ""),
                    likelihood=_derive_likelihood(confidence, severity),
                    remediation=raw.get("remediation", "")
                    or meta.get("remediation", ""),
                    affected_components=[AffectedComponent(name=affected_component)],
                    evidence=[
                        Evidence(
                            file=raw.get("evidence_file", affected_component),
                            code_snippet=anonymize_secrets(
                                raw.get("evidence", "")[:200]
                            ),
                            confidence=confidence.capitalize(),
                            line=int(raw.get("evidence_line", 0)),
                        )
                    ],
                    source="ai_reasoning",
                    hallucination_flag=is_hallucinated,
                    agent_name=raw.get("agent_name", ""),
                )
            )
        except Exception as e:
            logger.error("Failed to convert AI finding: %s", e)
            conversion_errors += 1
            continue

    if conversion_errors > 0:
        logger.warning(
            "%d AI findings failed to convert. Continuing with %d valid findings.",
            conversion_errors,
            len(findings),
        )

    return findings


# ── Findings consolidation ──────────────────────────────────────────────────


def _consolidate_findings(findings: list[Finding]) -> list[Finding]:
    """Deduplicate findings from different sources at the same location.

    Groups by (subcategory, file, line). When static and AI findings overlap,
    keeps the AI finding (richer description) and merges the source field.
    """
    groups: dict[tuple, list[Finding]] = {}
    for f in findings:
        ev = f.evidence[0] if f.evidence else None
        if ev and ev.file and ev.line:
            key: tuple = (f.subcategory, ev.file, ev.line)
        else:
            key = (f.id,)
        groups.setdefault(key, []).append(f)

    consolidated = []
    for group in groups.values():
        if len(group) == 1:
            consolidated.append(group[0])
            continue
        # Prefer ai_reasoning over static; merge source names.
        ai = [f for f in group if f.source == "ai_reasoning"]
        keep = ai[0] if ai else group[0]
        sources = sorted({f.source for f in group})
        consolidated.append(dataclasses.replace(keep, source="+".join(sources)))

    return consolidated


# ── Category summary builder ──────────────────────────────────────────────────


def build_category_summary(findings: list[Finding]) -> dict[str, CategorySummaryEntry]:
    """
    Count findings per ASI category plus a beyond_asi bucket.
    Initialized with all 10 ASI categories so the report always shows
    the full OWASP framework coverage even for zero-finding categories.
    """
    summary = {}

    # Pre-initialize all ASI categories with zero counts and their ASI metadata
    for cat_key, cat_data in AGENT_TAXONOMY.items():
        summary[cat_key] = {
            "asi_code": cat_data["asi_code"],
            "asi_title": cat_data["asi_title"],
            "count": 0,
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

    # Pre-initialize the beyond_asi bucket for findings outside the framework
    summary["beyond_asi"] = {
        "asi_code": "BEYOND-ASI",
        "asi_title": "Beyond Current OWASP ASI Framework",
        "count": 0,
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0,
    }

    for f in findings:
        cat = f.category.lower() if f.category else "beyond_asi"
        if cat not in summary:
            # Unknown category from old data or edge case — bucket into beyond_asi
            cat = "beyond_asi"
        summary[cat]["count"] += 1
        sev = (
            f.ai_spm_severity.lower()
            if f.ai_spm_severity.lower() in ("critical", "high", "medium", "low")
            else "low"
        )
        summary[cat][sev] += 1

    return summary


# ── Main scan pipeline ────────────────────────────────────────────────────────


def _read_inventory_files(
    inventory: InventoryLike, scan_root: str
) -> tuple[list[RepoFile], list[RepoFile]]:
    """Read Python files referenced by inventory agents and requirements files."""
    agent_file_paths: set[str] = set()
    for agent in inventory.agents.values():
        for occ in agent.occurrences:
            if hasattr(occ, "file_path") and occ.file_path:
                agent_file_paths.add(occ.file_path)

    python_files = []
    for raw_path in sorted(agent_file_paths):
        # occ.file_path may be absolute or relative — normalize to
        # a relative path from scan_root for consistent reporting.
        if os.path.isabs(raw_path) and raw_path.startswith(scan_root):
            rel_path = os.path.relpath(raw_path, scan_root)
        else:
            rel_path = raw_path

        abs_path = os.path.join(scan_root, rel_path)
        if os.path.isfile(abs_path):
            try:
                with open(abs_path, encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
                python_files.append(
                    RepoFile(path=rel_path, content=content, size=len(content))
                )
            except Exception as exc:
                logger.warning("Could not read %s: %s", abs_path, exc)

    requirements_files = []
    for name in REQUIREMENT_FILE_NAMES:
        req_path = os.path.join(scan_root, name)
        if os.path.isfile(req_path):
            try:
                with open(req_path, encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
                requirements_files.append(
                    RepoFile(path=name, content=content, size=len(content))
                )
            except Exception:
                pass

    return python_files, requirements_files


def run_core(
    python_files: list[RepoFile],
    requirements_files: list[RepoFile],
    *,
    agent_profiles: list[AgentProfile] | None = None,
    skip_triage: bool = False,
    agentic: bool = True,
    model_string: str | None = None,
    repo_name: str = "",
    primary_framework: str = "unknown",
    frameworks_detected: list[str] | None = None,
    total_files_scanned: int = 0,
) -> ScanReport:
    """Core scanning pipeline (layers 2-4). Takes already-resolved files.

    This is the main scanning engine. Discovery is handled by callers
    (run_scan or direct invocation).
    """
    timestamp = datetime.now(UTC).isoformat()
    agents = agent_profiles or []

    # ── LAYER 2: Static Scan ─────────────────────────────────────────────────
    logger.info("Layer 2: Static Scan")
    all_findings: list[Finding] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        static_result = run_static_scan(python_files, requirements_files, tmp_dir)

    static_raw = static_result.findings
    static_findings = convert_static_findings(static_raw)
    all_findings.extend(static_findings)
    logger.info("Static findings converted: %d", len(static_findings))

    # ── LAYER 3: AI Reasoning ────────────────────────────────────────────────
    logger.info("Layer 3: AI Reasoning")
    ai_findings: list[Finding] = []
    ai_summary = "AI reasoning skipped (no provider configured)."

    llm_model = None
    try:
        if model_string:
            os.environ["SCANNER_MODEL"] = model_string
        llm_model = make_model(scanner="agent", phase="triage")
        logger.info("LLM model: %s", llm_model)
    except Exception as e:
        logger.info("Could not initialise LLM model (%s): %s", type(e).__name__, e)
        logger.info("AI reasoning and triage will be skipped.")

    agentic_trace: dict[str, str] = {}

    if llm_model is not None:
        try:
            if agentic:
                logger.info("Mode: AGENTIC v2 (Google ADK reasoning loop)")
                ai_raw, ai_summary, agentic_trace = run_agentic_reasoning(
                    agents,
                    python_files,
                    static_findings=static_raw,
                )
                logger.info(
                    "ADK session: %s | exit: %s | model: %s",
                    agentic_trace.get("session_id", "?"),
                    agentic_trace.get("exit_reason", "?"),
                    agentic_trace.get("adk_model", "?"),
                )
            else:
                ai_raw = []
                ai_summary = "Agentic reasoning disabled."
            ai_findings = convert_ai_findings(ai_raw)
            all_findings.extend(ai_findings)
            logger.info("AI findings converted: %d", len(ai_findings))
        except Exception as e:
            logger.warning(
                "AI reasoning failed (%s): %s — continuing with static findings only",
                type(e).__name__,
                e,
            )
            ai_summary = f"AI reasoning failed: {e}"
    else:
        logger.info("No LLM provider — skipping AI reasoning layer")

    # ── Consolidate static + AI duplicates ────────────────────────────────────
    before = len(all_findings)
    all_findings = _consolidate_findings(all_findings)
    if len(all_findings) < before:
        logger.info(
            "Consolidated %d -> %d findings (removed %d duplicates)",
            before,
            len(all_findings),
            before - len(all_findings),
        )

    # ── Build agent profile dicts (needed by both triage and report) ─────────
    agent_profile_dicts = []
    for a in agents:
        agent_profile_dicts.append(
            {
                "agent_id": a.agent_id,
                "framework": a.framework,
                "source_file": a.source_file,
                "role": a.role,
                "goal": a.goal,
                "tools": a.tools,
                "memory": a.memory,
                "allow_delegation": a.allow_delegation,
                "llm": a.llm,
                "graph_nodes": a.graph_nodes,
                "has_recursion_limit": a.has_recursion_limit,
                "input_guardrails": a.input_guardrails,
                "output_guardrails": a.output_guardrails,
                "function_signatures": a.function_signatures,
            }
        )

    # ── LAYER 4: Triage ──────────────────────────────────────────────────────
    logger.info("Layer 4: Triage")

    pre_triage_findings = None
    triage_dismissed = None
    triage_downgraded = None
    triage_meta: dict = {}

    if skip_triage:
        logger.info("Triage skipped")
    elif llm_model is None:
        logger.info("No LLM provider — triage requires a configured provider")
    elif not all_findings:
        logger.info("No findings to triage")
    else:
        pre_triage_dicts = [dataclasses.asdict(f) for f in all_findings]

        triage_context = build_triage_context(
            repo_name=repo_name,
            framework=primary_framework,
            agent_profiles=agent_profile_dicts,
        )

        triage_result = run_triage(
            raw_findings=pre_triage_dicts,
            agent_context=triage_context,
            model=llm_model,
        )

        if triage_result:
            try:
                pre_triage_findings = pre_triage_dicts

                all_findings = _dicts_to_findings(
                    triage_result.get("kept_findings", [])
                )

                triage_dismissed = []
                for d in triage_result.get("triage_dismissed", []):
                    try:
                        if (
                            not isinstance(d, dict)
                            or "id" not in d
                            or "reason" not in d
                        ):
                            logger.warning("Malformed triage_dismissed entry: %s", d)
                            continue
                        triage_dismissed.append(
                            TriageDismissed(finding_id=d["id"], reason=d["reason"])
                        )
                    except (KeyError, TypeError) as e:
                        logger.warning("Could not construct TriageDismissed: %s", e)
                        continue

                triage_downgraded = []
                for d in triage_result.get("triage_downgraded", []):
                    try:
                        if not isinstance(d, dict) or not all(
                            k in d
                            for k in [
                                "id",
                                "original_severity",
                                "new_severity",
                                "reason",
                            ]
                        ):
                            logger.warning("Malformed triage_downgraded entry: %s", d)
                            continue
                        triage_downgraded.append(
                            TriageDowngraded(
                                finding_id=d["id"],
                                original_severity=d["original_severity"],
                                new_severity=d["new_severity"],
                                reason=d["reason"],
                            )
                        )
                    except (KeyError, TypeError) as e:
                        logger.warning("Could not construct TriageDowngraded: %s", e)
                        continue

                triage_meta = triage_result.get("triage_summary", {})
                if not isinstance(triage_meta, dict):
                    logger.warning(
                        "triage_summary is not a dict, got %s",
                        type(triage_meta).__name__,
                    )
                    triage_meta = {}

                logger.info(
                    "Triage complete — kept: %d | dismissed: %d | downgraded: %d",
                    len(all_findings),
                    len(triage_dismissed),
                    len(triage_downgraded),
                )
            except Exception as e:
                logger.error("Failed to process triage result: %s", e)
                logger.warning(
                    "Using raw findings unchanged due to triage processing error"
                )
                triage_dismissed = None
                triage_downgraded = None
                pre_triage_findings = None
                triage_meta = {}
        else:
            logger.warning("Triage returned no result — using raw findings unchanged")

    # ── Assemble Report ───────────────────────────────────────────────────────
    logger.info("Assembling Report")

    severity_order = SEVERITY_ORDER
    for f in all_findings:
        if f.ai_spm_severity not in severity_order:
            logger.warning(
                "Finding %s has invalid severity '%s'. Treating as 'Low'.",
                f.id,
                f.ai_spm_severity,
            )
            f.ai_spm_severity = "Low"  # type: ignore
    all_findings.sort(key=lambda f: severity_order.get(f.ai_spm_severity, 5))

    category_summary = build_category_summary(all_findings)

    report = ScanReport(
        repo_url="",
        repo_name=repo_name,
        scan_timestamp=timestamp,
        framework_detected=primary_framework,
        agents_found=len(agents),
        findings=all_findings,
        category_summary=category_summary,
        agent_profiles=agent_profile_dicts,
        scan_metadata={
            "total_files_scanned": total_files_scanned,
            "python_files": len(python_files),
            "requirements_files": len(requirements_files),
            "frameworks_detected": frameworks_detected or [],
            "static_findings_count": len(static_findings),
            "ai_findings_count": len(ai_findings),
            "ai_summary": ai_summary,
            "triage_run": (triage_meta != {}),
            "triage_dismissed_count": len(triage_dismissed) if triage_dismissed else 0,
            "triage_downgraded_count": (
                len(triage_downgraded) if triage_downgraded else 0
            ),
            "tools_used": (
                static_result.tools_used
                + (
                    [f"ai-reasoning:{get_model_name()}"]
                    if llm_model is not None
                    else []
                )
                + (
                    [f"triage:{get_model_name()}"]
                    if (triage_dismissed is not None or triage_downgraded is not None)
                    and llm_model
                    else []
                )
            ),
            # asdict, not the dataclasses themselves — the SARIF formatter
            # json.dumps() scan_metadata without an asdict pass of its own.
            "tools_skipped": [
                dataclasses.asdict(s) for s in static_result.tools_skipped
            ],
            "agentic_trace": agentic_trace,
        },
        pre_triage_findings=pre_triage_findings,
        triage_dismissed=triage_dismissed,
        triage_downgraded=triage_downgraded,
    )

    return report
