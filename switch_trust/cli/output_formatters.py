"""
Output format abstraction for switch-trust CLI.

Provides pluggable formatters for scan and eval results.
To add a new format: subclass ScanOutputFormatter / EvalOutputFormatter,
register in the SCAN_OUTPUT_FORMATTERS / EVAL_OUTPUT_FORMATTERS dicts.
"""

from __future__ import annotations

import dataclasses
import json
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from switch_trust.cli.version import VERSION
from switch_trust.eval.common.utils import now_utc, strip_nulls


class OutputFormat(str, Enum):
    JSON = "json"
    SARIF = "sarif"


# ── Scan output formatters ──────────────────────────────────────────────────

# Agent discovery (the InventoryParser pass) is out of scope for the CLI, so
# these ScanReport fields are structurally always 0 / empty on this path —
# `run_core` is never handed any AgentProfile objects. Emitting them advertises
# a measurement the CLI never makes and creates a "framework detected but 0
# agents" contradiction, so drop them from CLI output rather than ship a
# permanent `agents_found: 0`. The shared schema keeps the fields for the
# inventory sensor path, which does populate them; the SARIF formatter already
# omits them.
_CLI_UNPOPULATED_SCAN_FIELDS = ("agents_found", "agent_profiles")


class ScanOutputFormatter(ABC):
    @abstractmethod
    def format(self, report: Any) -> str: ...

    @property
    @abstractmethod
    def extension(self) -> str: ...


class JsonScanOutputFormatter(ScanOutputFormatter):
    @property
    def extension(self) -> str:
        return "json"

    def format(self, report: Any) -> str:
        data = dataclasses.asdict(report)
        for key in _CLI_UNPOPULATED_SCAN_FIELDS:
            data.pop(key, None)
        return json.dumps(data, indent=2)


class SarifScanOutputFormatter(ScanOutputFormatter):
    @property
    def extension(self) -> str:
        return "sarif"

    def format(self, report: Any) -> str:
        return json.dumps(self._to_sarif(report), indent=2)

    def _severity_to_level(self, severity: str) -> str:
        s = severity.lower()
        if s in ("critical", "high"):
            return "error"
        if s == "medium":
            return "warning"
        return "note"

    @staticmethod
    def _normalize_text_paths(text: str) -> str:
        return re.sub(r"(?<![/\w])Users/", "/Users/", text)

    @staticmethod
    def _normalize_uri(path: str) -> str:
        if not path:
            return path
        if path.startswith("/"):
            return path
        if len(path) > 1 and path[0].isalpha() and path[1] == ":":
            return path
        if path.startswith("file:"):
            return path
        parts = path.split("/")
        if len(parts) >= 2 and parts[0].lower() == "users":
            return "/" + path
        return path

    def _to_sarif(self, report: Any) -> dict:
        rules: list[dict] = []
        rule_index: dict[str, int] = {}
        results: list[dict] = []

        for finding in report.findings:
            rule_id = f"{finding.category}/{finding.subcategory}"
            if rule_id not in rule_index:
                rule_index[rule_id] = len(rules)
                rule: dict[str, Any] = {
                    "id": rule_id,
                    "shortDescription": {"text": finding.subcategory},
                    "properties": {"tags": [finding.category]},
                }
                if finding.remediation:
                    rule["help"] = {"text": finding.remediation}
                rules.append(rule)

            # A location's URI must be a file path. Evidence carries one; an
            # affected component's `name` does not — it is a free-text label the
            # reasoning model writes, and for AI findings `path` is never set, so
            # the old `comp.path or comp.name` fallback published things like
            # "request_return", "check_order" and "agent.py:app" as artifact
            # URIs. Which label it chose varied run to run, so the same
            # vulnerability landed under a different location on each scan while
            # the correct file sat in `relatedLocations` all along.
            #
            # Preference order is therefore evidence file, then a component that
            # carries a real `path` (static findings set one). A component name
            # is never used: no location beats a wrong one, and the names are
            # preserved under `properties.affectedComponents`.
            locations = []
            for ev in finding.evidence or []:
                if not ev.file:
                    continue
                loc: dict[str, Any] = {
                    "physicalLocation": {
                        "artifactLocation": {"uri": self._normalize_uri(ev.file)},
                    },
                }
                if ev.line:
                    loc["physicalLocation"]["region"] = {"startLine": ev.line}
                    if ev.column:
                        loc["physicalLocation"]["region"]["startColumn"] = ev.column
                locations.append(loc)

            if not locations:
                for comp in finding.affected_components or []:
                    if not comp.path:
                        continue
                    locations.append(
                        {
                            "physicalLocation": {
                                "artifactLocation": {
                                    "uri": self._normalize_uri(comp.path)
                                },
                            },
                        }
                    )

            related_locations = []
            for idx, ev in enumerate(finding.evidence or []):
                if not ev.file:
                    continue
                rel: dict[str, Any] = {
                    "id": idx,
                    "physicalLocation": {
                        "artifactLocation": {"uri": self._normalize_uri(ev.file)},
                    },
                }
                if ev.line:
                    rel["physicalLocation"]["region"] = {"startLine": ev.line}
                if ev.code_snippet:
                    rel["physicalLocation"]["region"] = rel["physicalLocation"].get(
                        "region", {}
                    )
                    rel["physicalLocation"]["region"]["snippet"] = {
                        "text": ev.code_snippet
                    }
                if ev.context:
                    rel["message"] = {"text": ev.context}
                related_locations.append(rel)

            message_parts = []
            if finding.impact:
                message_parts.append(f"Impact: {finding.impact}")
            if not message_parts:
                message_parts.append(finding.title or finding.description)

            result: dict[str, Any] = {
                "ruleId": rule_id,
                "ruleIndex": rule_index[rule_id],
                "level": self._severity_to_level(finding.ai_spm_severity),
                "message": {
                    "text": self._normalize_text_paths(" ".join(message_parts))
                },
            }

            if finding.id:
                result["guid"] = finding.id

            if locations:
                result["locations"] = locations

            if related_locations:
                result["relatedLocations"] = related_locations

            props: dict[str, Any] = {}
            if finding.cvss_scores:
                props["cvssv4"] = {
                    "baseScore": finding.cvss_scores.base_score,
                    "vector": finding.cvss_scores.vector,
                }
            if finding.source:
                props["source"] = finding.source
            if finding.ai_spm_severity:
                props["severity"] = finding.ai_spm_severity
            # Kept because these no longer appear as location URIs. For an AI
            # finding this is the agent, tool or function the model named, which
            # is useful context even though it is not a path.
            component_names = [
                comp.name for comp in finding.affected_components or [] if comp.name
            ]
            if component_names:
                props["affectedComponents"] = component_names
            if finding.hallucination_flag:
                props["hallucinationFlag"] = True
            if props:
                result["properties"] = props

            results.append(result)

        invocation: dict[str, Any] = {"executionSuccessful": True}
        if report.scan_timestamp:
            invocation["startTimeUtc"] = report.scan_timestamp

        run: dict[str, Any] = {
            "tool": {
                "driver": {
                    "name": "switch-trust-scan",
                    "version": VERSION,
                    "informationUri": "https://github.com/sandbox-quantum/switch-trust",
                    "rules": rules,
                },
            },
            "invocations": [invocation],
            "results": results,
        }

        artifacts = []
        seen_uris: set[str] = set()
        for r in results:
            for loc in r.get("locations", []):
                uri = (
                    loc.get("physicalLocation", {})
                    .get("artifactLocation", {})
                    .get("uri", "")
                )
                if uri and uri not in seen_uris:
                    seen_uris.add(uri)
                    artifacts.append({"location": {"uri": uri}})
        if artifacts:
            run["artifacts"] = artifacts

        run_props: dict[str, Any] = {}
        if report.repo_name:
            run_props["repoName"] = report.repo_name
        if report.scan_metadata:
            run_props["scanMetadata"] = report.scan_metadata
        if run_props:
            run["properties"] = run_props

        return {
            "$schema": (
                "https://docs.oasis-open.org/sarif/sarif/v2.1.0"
                "/errata01/os/schemas/sarif-schema-2.1.0.json"
            ),
            "version": "2.1.0",
            "runs": [run],
        }


# ── Eval output formatters ──────────────────────────────────────────────────

EVAL_SCHEMA_VERSION = "2.0"


class EvalOutputFormatter(ABC):
    @abstractmethod
    def format(self, runs: list, config_path: str) -> str: ...

    @property
    @abstractmethod
    def extension(self) -> str: ...


class JsonEvalOutputFormatter(EvalOutputFormatter):
    @property
    def extension(self) -> str:
        return "json"

    def format(self, runs: list, config_path: str) -> str:
        output = prepare_eval_output(runs, config_path)
        return json.dumps(output, indent=2, default=str)


class SarifEvalOutputFormatter(EvalOutputFormatter):
    @property
    def extension(self) -> str:
        return "sarif"

    def format(self, runs: list, config_path: str) -> str:
        return json.dumps(self._to_sarif(runs, config_path), indent=2, default=str)

    def _score_to_level(self, score: float) -> str:
        if score >= 0.7:
            return "error"
        if score >= 0.3:
            return "warning"
        return "note"

    def _to_sarif(self, runs: list, config_path: str) -> dict:
        rules: list[dict] = []
        rule_index: dict[str, int] = {}
        results: list[dict] = []

        for run in runs:
            eval_name = run.evaluation.get("name", "unknown")
            eval_type = run.evaluation.get("type", "unknown")
            model_name = run.model.get("name", "unknown")

            if eval_name not in rule_index:
                rule_index[eval_name] = len(rules)
                rules.append(
                    {
                        "id": eval_name,
                        "shortDescription": {"text": eval_name},
                        "properties": {"tags": ["ai-red-teaming", eval_type]},
                    }
                )

            achieved = 0.0
            max_score = 0.0
            total = 0
            summary_props: dict[str, Any] = {}
            if run.summary:
                s = (
                    run.summary
                    if isinstance(run.summary, dict)
                    else run.summary.to_dict()
                )
                achieved = s.get("achieved_score", 0.0)
                max_score = s.get("max_score", 0.0)
                total = s.get("total_evaluations", 0)
                summary_props = {
                    "status": s.get("status", ""),
                    "totalEvaluations": total,
                    "finishedEvaluations": s.get("finished_evaluations", 0),
                    "errorEvaluations": s.get("error_evaluations", 0),
                }

            if achieved > 0:
                avg_score = achieved / total if total > 0 else 0.0
                results.append(
                    {
                        "ruleId": eval_name,
                        "ruleIndex": rule_index[eval_name],
                        "level": self._score_to_level(avg_score),
                        "message": {
                            "text": (
                                f"Model '{model_name}' scored {achieved:.1f}/{max_score:.1f} "
                                f"({avg_score:.0%}) on evaluation '{eval_name}' "
                                f"across {total} prompts."
                            ),
                        },
                        "locations": [
                            {
                                "logicalLocations": [
                                    {
                                        "name": model_name,
                                        "kind": "module",
                                        "fullyQualifiedName": run.model.get(
                                            "id", model_name
                                        ),
                                    }
                                ],
                            }
                        ],
                        "properties": {
                            "modelEvaluationId": run.model_evaluation_id,
                            "modelEvaluationName": run.model_evaluation_name,
                            "achievedScore": achieved,
                            "maxScore": max_score,
                            "averageScore": round(avg_score, 4),
                            "totalPrompts": total,
                            **summary_props,
                        },
                    }
                )

        return {
            "$schema": (
                "https://docs.oasis-open.org/sarif/sarif/v2.1.0"
                "/errata01/os/schemas/sarif-schema-2.1.0.json"
            ),
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "switch-trust-eval",
                            "version": VERSION,
                            "informationUri": "https://github.com/sandbox-quantum/switch-trust",
                            "rules": rules,
                        },
                    },
                    "invocations": [{"executionSuccessful": True}],
                    "results": results,
                    "properties": {
                        "configFile": config_path,
                    },
                }
            ],
        }


# ── Helpers ──────────────────────────────────────────────────────────────────


def _strip_session(session: dict | None) -> dict | None:
    if session is None:
        return None
    session.pop("id", None)
    session.pop("timestamp", None)
    session.pop("metadata", None)
    for msg in session.get("messages", []):
        msg.pop("id", None)
        msg.pop("timestamp", None)
        msg.pop("metadata", None)
    return session


def _strip_result(result: dict) -> dict:
    result.pop("status", None)
    result["session"] = _strip_session(result.get("session"))
    return result


def prepare_eval_output(runs: list, config_path: str) -> dict:
    # Deliberately lazy: runner imports get_eval_output_formatter from this
    # module at import time, so hoisting this would close the cycle.
    from switch_trust.cli.runner import _aggregate_summary  # noqa: PLC0415

    overall = _aggregate_summary(runs)
    raw: dict[str, Any] = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "config_file": config_path,
        "timestamp": now_utc().isoformat(),
        "summary": overall.to_dict(),
        "runs": [r.to_dict() for r in runs],
    }
    for run in raw["runs"]:
        for result in run.get("results", []):
            _strip_result(result)
    return strip_nulls(raw)


# ── Registry ─────────────────────────────────────────────────────────────────


SCAN_OUTPUT_FORMATTERS: dict[OutputFormat, ScanOutputFormatter] = {
    OutputFormat.JSON: JsonScanOutputFormatter(),
    OutputFormat.SARIF: SarifScanOutputFormatter(),
}

EVAL_OUTPUT_FORMATTERS: dict[OutputFormat, EvalOutputFormatter] = {
    OutputFormat.JSON: JsonEvalOutputFormatter(),
    OutputFormat.SARIF: SarifEvalOutputFormatter(),
}


def get_scan_output_formatter(fmt: OutputFormat) -> ScanOutputFormatter:
    return SCAN_OUTPUT_FORMATTERS[fmt]


def get_eval_output_formatter(fmt: OutputFormat) -> EvalOutputFormatter:
    return EVAL_OUTPUT_FORMATTERS[fmt]
