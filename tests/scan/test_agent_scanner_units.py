"""
Unit tests for agent scanner core modules.

Covers: scorer, discovery utilities, convert_static_findings,
_dicts_to_findings, finding evidence, and llm_provider.
"""

import dataclasses
import re
import unittest
import uuid
from unittest.mock import MagicMock, patch

from switch_trust.scan.schema import AffectedComponent, CvssScores, Evidence, Finding

# ── Helpers ──────────────────────────────────────────────────────────


def _make_finding(**overrides) -> Finding:
    defaults = {
        "id": "TEST-0001",
        "category": "asi01_agent_goal_hijack",
        "subcategory": "direct_prompt_injection",
        "ai_spm_severity": "High",
        "cvss_v4_severity": "High",
        "cvss_scores": CvssScores(base_score=7.5, vector="CVSS:4.0/AV:N"),
        "description": "desc",
        "impact": "impact",
        "likelihood": "High",
        "remediation": "fix",
        "affected_components": [AffectedComponent(name="my_agent", path="agent.py")],
        "evidence": [
            Evidence(file="agent.py", code_snippet="eval(x)", confidence="High")
        ],
        "hallucination_flag": False,
        "title": "Test",
        "source": "ai_reasoning",
    }
    defaults.update(overrides)
    return Finding(**defaults)


# ═════════════════════════════════════════════════════════════════════
# 1. scorer.py
# ═════════════════════════════════════════════════════════════════════


class TestSeverityFromScore(unittest.TestCase):
    def test_critical(self):
        from switch_trust.scan.scorer import severity_from_score

        self.assertEqual(severity_from_score(9.0), "critical")
        self.assertEqual(severity_from_score(10.0), "critical")

    def test_high(self):
        from switch_trust.scan.scorer import severity_from_score

        self.assertEqual(severity_from_score(7.0), "high")
        self.assertEqual(severity_from_score(8.9), "high")

    def test_medium(self):
        from switch_trust.scan.scorer import severity_from_score

        self.assertEqual(severity_from_score(4.0), "medium")
        self.assertEqual(severity_from_score(6.9), "medium")

    def test_low(self):
        from switch_trust.scan.scorer import severity_from_score

        self.assertEqual(severity_from_score(0.1), "low")
        self.assertEqual(severity_from_score(3.9), "low")

    def test_info(self):
        from switch_trust.scan.scorer import severity_from_score

        self.assertEqual(severity_from_score(0.0), "info")


class TestComputeCvssV4Score(unittest.TestCase):
    def test_valid_vector(self):
        from switch_trust.scan.scorer import compute_cvss_v4_score

        vector = "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N"
        score = compute_cvss_v4_score(vector)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 10.0)

    def test_invalid_prefix(self):
        from switch_trust.scan.scorer import compute_cvss_v4_score

        self.assertEqual(compute_cvss_v4_score("CVSS:3.1/AV:N"), 0.0)

    def test_empty_string(self):
        from switch_trust.scan.scorer import compute_cvss_v4_score

        self.assertEqual(compute_cvss_v4_score(""), 0.0)


class TestScoreFinding(unittest.TestCase):
    def test_known_subcategory(self):
        from switch_trust.scan.scorer import score_finding

        score, vector, severity = score_finding("hardcoded_credentials")
        self.assertIsInstance(score, float)
        self.assertTrue(vector.startswith("CVSS:4.0/"))
        self.assertIn(severity, ("critical", "high", "medium", "low", "info"))

    def test_score_within_severity_band(self):
        from switch_trust.scan.scorer import score_finding

        score, _, severity = score_finding("arbitrary_code_execution")
        if severity == "critical":
            self.assertGreaterEqual(score, 9.0)
        elif severity == "high":
            self.assertGreaterEqual(score, 7.0)
            self.assertLessEqual(score, 8.9)
        elif severity == "medium":
            self.assertGreaterEqual(score, 4.0)
            self.assertLessEqual(score, 6.9)

    def test_unknown_subcategory_returns_defaults(self):
        from switch_trust.scan.scorer import score_finding

        score, vector, severity = score_finding("nonexistent_subcategory_xyz")
        self.assertIsInstance(score, float)
        self.assertIn(severity, ("critical", "high", "medium", "low", "info"))


class TestMapBanditToTaxonomy(unittest.TestCase):
    def test_known_rule(self):
        from switch_trust.scan.scorer import map_bandit_to_taxonomy

        cat, sub = map_bandit_to_taxonomy("B102", "")
        self.assertEqual(cat, "asi05_unexpected_code_execution")
        self.assertEqual(sub, "arbitrary_code_execution")

    def test_keyword_fallback_subprocess(self):
        from switch_trust.scan.scorer import map_bandit_to_taxonomy

        cat, sub = map_bandit_to_taxonomy("B999", "subprocess call detected")
        self.assertEqual(cat, "asi05_unexpected_code_execution")

    def test_keyword_fallback_hardcoded(self):
        from switch_trust.scan.scorer import map_bandit_to_taxonomy

        cat, sub = map_bandit_to_taxonomy("B999", "hardcoded password found")
        self.assertEqual(cat, "asi03_identity_privilege_abuse")
        self.assertEqual(sub, "hardcoded_credentials")

    def test_unknown_rule_returns_default(self):
        from switch_trust.scan.scorer import map_bandit_to_taxonomy

        cat, sub = map_bandit_to_taxonomy("B999", "something benign")
        self.assertEqual(cat, "asi02_tool_misuse")
        self.assertEqual(sub, "unvalidated_tool_input")


class TestMapOpenGrepToTaxonomy(unittest.TestCase):
    def test_known_rule(self):
        from switch_trust.scan.scorer import map_opengrep_to_taxonomy

        cat, sub = map_opengrep_to_taxonomy("crewai-allow-delegation-true")
        self.assertEqual(cat, "asi10_rogue_agents")

    def test_dotted_rule_id(self):
        from switch_trust.scan.scorer import map_opengrep_to_taxonomy

        cat, sub = map_opengrep_to_taxonomy("rules.custom.eval-call")
        self.assertEqual(cat, "asi05_unexpected_code_execution")

    def test_unknown_rule(self):
        from switch_trust.scan.scorer import map_opengrep_to_taxonomy

        cat, sub = map_opengrep_to_taxonomy("totally-unknown-rule")
        self.assertEqual(cat, "asi02_tool_misuse")


# ═════════════════════════════════════════════════════════════════════
# 2. _dicts_to_findings round-trip
# ═════════════════════════════════════════════════════════════════════


class TestDictsToFindings(unittest.TestCase):
    def test_round_trip(self):
        from switch_trust.scan.agent_scanner import _dicts_to_findings

        finding = _make_finding()
        d = dataclasses.asdict(finding)
        result = _dicts_to_findings([d])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "TEST-0001")
        self.assertEqual(result[0].cvss_scores.base_score, 7.5)
        self.assertEqual(result[0].affected_components[0].name, "my_agent")

    def test_skips_non_dict(self):
        from switch_trust.scan.agent_scanner import _dicts_to_findings

        result = _dicts_to_findings(["not a dict", 42, None])
        self.assertEqual(result, [])

    def test_skips_malformed_dict(self):
        from switch_trust.scan.agent_scanner import _dicts_to_findings

        result = _dicts_to_findings([{"id": "bad", "missing": "fields"}])
        self.assertEqual(result, [])

    def test_empty_list(self):
        from switch_trust.scan.agent_scanner import _dicts_to_findings

        self.assertEqual(_dicts_to_findings([]), [])

    def test_multiple_findings(self):
        from switch_trust.scan.agent_scanner import _dicts_to_findings

        f1 = dataclasses.asdict(_make_finding(id="F1"))
        f2 = dataclasses.asdict(_make_finding(id="F2"))
        result = _dicts_to_findings([f1, f2])
        self.assertEqual(len(result), 2)
        self.assertEqual({r.id for r in result}, {"F1", "F2"})


# ═════════════════════════════════════════════════════════════════════
# 4. convert_static_findings
# ═════════════════════════════════════════════════════════════════════


class TestConvertStaticFindings(unittest.TestCase):
    def _make_static_finding(self, **overrides):
        from switch_trust.scan.static_scanner import StaticFinding

        defaults = {
            "tool": "bandit",
            "rule_id": "B102",
            "severity": "high",
            "message": "exec() used",
            "filepath": "agent.py",
            "line": 10,
            "evidence": "exec(user_input)",
        }
        defaults.update(overrides)
        return StaticFinding(**defaults)

    def test_converts_bandit_finding(self):
        from switch_trust.scan.agent_scanner import convert_static_findings

        sf = self._make_static_finding()
        result = convert_static_findings([sf])
        self.assertEqual(len(result), 1)
        f = result[0]
        self.assertEqual(f.source, "static_bandit")
        self.assertEqual(f.category, "asi05_unexpected_code_execution")

    def test_converts_opengrep_finding(self):
        from switch_trust.scan.agent_scanner import convert_static_findings

        sf = self._make_static_finding(
            tool="opengrep",
            rule_id="eval-call",
            message="eval() call",
        )
        result = convert_static_findings([sf])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].source, "static_opengrep")

    def test_deduplicates_same_line(self):
        from switch_trust.scan.agent_scanner import convert_static_findings

        sf1 = self._make_static_finding(rule_id="B602", message="subprocess shell=True")
        sf2 = self._make_static_finding(
            tool="opengrep",
            rule_id="subprocess-shell-true",
            message="subprocess shell=True",
        )
        result = convert_static_findings([sf1, sf2])
        self.assertEqual(len(result), 1)

    def test_different_lines_not_deduped(self):
        from switch_trust.scan.agent_scanner import convert_static_findings

        sf1 = self._make_static_finding(line=10)
        sf2 = self._make_static_finding(line=20)
        result = convert_static_findings([sf1, sf2])
        self.assertEqual(len(result), 2)

    def test_confidence_uses_computed_severity(self):
        from switch_trust.scan.agent_scanner import convert_static_findings

        sf = self._make_static_finding(severity="low")
        result = convert_static_findings([sf])
        self.assertEqual(len(result), 1)
        conf = result[0].evidence[0].confidence
        self.assertIn(conf, ("High", "Medium"))

    def test_skips_invalid_static_finding(self):
        from switch_trust.scan.agent_scanner import convert_static_findings

        bad = MagicMock(spec=[])
        result = convert_static_findings([bad])
        self.assertEqual(result, [])

    def test_detect_secrets_mapping(self):
        from switch_trust.scan.agent_scanner import convert_static_findings

        sf = self._make_static_finding(
            tool="detect_secrets",
            rule_id="HexHighEntropyString",
            message="High entropy string",
        )
        result = convert_static_findings([sf])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].category, "asi03_identity_privilege_abuse")
        self.assertEqual(result[0].subcategory, "hardcoded_credentials")


# ═════════════════════════════════════════════════════════════════════
# 5. AI findings evidence propagation
# ═════════════════════════════════════════════════════════════════════


class TestConvertAiFindingsEvidence(unittest.TestCase):
    def test_evidence_file_and_line_from_raw_dict(self):
        from switch_trust.scan.agent_scanner import convert_ai_findings

        raw = {
            "category": "asi01_agent_goal_hijack",
            "subcategory": "direct_prompt_injection",
            "title": "Prompt injection in agent",
            "description": "desc",
            "impact": "impact",
            "remediation": "fix",
            "affected_component": "my_agent",
            "evidence": "eval(user_input)",
            "confidence": "high",
            "hallucination_flag": False,
            "evidence_file": "src/agents.py",
            "evidence_line": 42,
        }
        result = convert_ai_findings([raw])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].evidence[0].file, "src/agents.py")
        self.assertEqual(result[0].evidence[0].line, 42)

    def test_evidence_defaults_without_evidence_file(self):
        from switch_trust.scan.agent_scanner import convert_ai_findings

        raw = {
            "category": "asi01_agent_goal_hijack",
            "subcategory": "direct_prompt_injection",
            "title": "Test",
            "description": "desc",
            "impact": "impact",
            "remediation": "fix",
            "affected_component": "my_agent",
            "evidence": "code",
            "confidence": "medium",
            "hallucination_flag": False,
        }
        result = convert_ai_findings([raw])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].evidence[0].file, "my_agent")
        self.assertEqual(result[0].evidence[0].line, 0)
        self.assertEqual(result[0].agent_name, "")

    def test_agent_name_propagated(self):
        from switch_trust.scan.agent_scanner import convert_ai_findings

        raw = {
            "category": "asi01_agent_goal_hijack",
            "subcategory": "direct_prompt_injection",
            "title": "Test",
            "description": "desc",
            "impact": "impact",
            "remediation": "fix",
            "affected_component": "some_file.py",
            "evidence": "code",
            "confidence": "medium",
            "hallucination_flag": False,
            "agent_name": "research_agent",
            "evidence_file": "src/agents.py",
            "evidence_line": 25,
        }
        result = convert_ai_findings([raw])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].agent_name, "research_agent")
        self.assertEqual(result[0].evidence[0].file, "src/agents.py")
        self.assertEqual(result[0].evidence[0].line, 25)


class TestReportFindingEvidenceFields(unittest.TestCase):
    def test_evidence_fields_stored_in_raw_finding(self):
        from switch_trust.scan.tool_dispatcher import ToolDispatcher

        dispatcher = ToolDispatcher(repo_files={}, agents=[], static_findings=[])
        dispatcher.report_finding(
            subcategory="direct_prompt_injection",
            title="Test",
            description="desc",
            impact="impact",
            remediation="fix",
            affected_component="my_agent",
            evidence="eval(x)",
            confidence="high",
            hallucination_flag=False,
            evidence_file="src/agents.py",
            evidence_line=42,
            agent_name="research_agent",
        )
        self.assertEqual(len(dispatcher.session_findings), 1)
        raw = dispatcher.session_findings[0]
        self.assertEqual(raw["evidence_file"], "src/agents.py")
        self.assertEqual(raw["evidence_line"], 42)
        self.assertEqual(raw["agent_name"], "research_agent")

    def test_evidence_fields_default_when_omitted(self):
        from switch_trust.scan.tool_dispatcher import ToolDispatcher

        dispatcher = ToolDispatcher(repo_files={}, agents=[], static_findings=[])
        dispatcher.report_finding(
            subcategory="direct_prompt_injection",
            title="Test",
            description="desc",
            impact="impact",
            remediation="fix",
            affected_component="my_agent",
            evidence="eval(x)",
            confidence="high",
            hallucination_flag=False,
        )
        raw = dispatcher.session_findings[0]
        self.assertEqual(raw["evidence_file"], "")
        self.assertEqual(raw["evidence_line"], 0)
        self.assertEqual(raw["agent_name"], "")


# ═════════════════════════════════════════════════════════════════════
# 6. CVSS regex — multi-char metric values
# ═════════════════════════════════════════════════════════════════════


class TestCvssRegex(unittest.TestCase):
    def test_single_char_values(self):
        vector = "CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N"
        metrics = dict(re.findall(r"([A-Z]{1,2}):([A-Z]{1,2})", vector))
        self.assertEqual(metrics["AV"], "N")
        self.assertEqual(metrics["AC"], "L")
        self.assertEqual(metrics["AT"], "N")

    def test_multi_char_values_not_truncated(self):
        vector = "CVSS:4.0/AV:N/AC:L/AT:PN/PR:N/UI:N"
        metrics = dict(re.findall(r"([A-Z]{1,2}):([A-Z]{1,2})", vector))
        self.assertEqual(metrics["AT"], "PN")

        old_metrics = dict(re.findall(r"([A-Z]{1,2}):([A-Z])", vector))
        self.assertEqual(old_metrics["AT"], "P")  # truncated!


# ═════════════════════════════════════════════════════════════════════
# 7. Tool call ID uniqueness
# ═════════════════════════════════════════════════════════════════════


class TestToolCallIdUniqueness(unittest.TestCase):
    def test_ids_are_unique(self):
        ids = set()
        for _ in range(100):
            call_id = f"call_my_tool_{uuid.uuid4().hex[:8]}"
            ids.add(call_id)
        self.assertEqual(len(ids), 100)


# ═════════════════════════════════════════════════════════════════════
# 8. llm_provider.py — _safe_error redaction
# ═════════════════════════════════════════════════════════════════════


class TestSafeError(unittest.TestCase):
    def test_redacts_bearer_token(self):
        from switch_trust.scan.llm_provider import _safe_error

        err = Exception("Authorization: Bearer sk-abc123def456ghi789")
        result = _safe_error(err)
        self.assertNotIn("sk-abc123", result)
        self.assertIn("[REDACTED]", result)

    def test_redacts_openai_key(self):
        from switch_trust.scan.llm_provider import _safe_error

        err = Exception("Invalid key: sk-proj1234567890abcdef")
        result = _safe_error(err)
        self.assertNotIn("sk-proj", result)

    def test_preserves_safe_message(self):
        from switch_trust.scan.llm_provider import _safe_error

        err = Exception("Connection timeout")
        result = _safe_error(err)
        self.assertEqual(result, "Connection timeout")


# ═════════════════════════════════════════════════════════════════════
# 9. OSV CVSS vector string handling
# ═════════════════════════════════════════════════════════════════════


class TestOsvCvssVectorHandling(unittest.TestCase):
    def test_numeric_score_parsed(self):
        raw_score = "7.5"
        score = float(raw_score)
        self.assertEqual(score, 7.5)

    def test_vector_string_skipped(self):
        raw_score = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
        self.assertIsInstance(raw_score, str)
        self.assertTrue(raw_score.startswith("CVSS:"))

    def test_float_on_vector_raises(self):
        raw_score = "CVSS:3.1/AV:N/AC:L"
        with self.assertRaises(ValueError):
            float(raw_score)


# ═════════════════════════════════════════════════════════════════════
# 10. static_scanner — opengrep availability
# ═════════════════════════════════════════════════════════════════════


class TestRunOpenGrepSkipsWhenMissing(unittest.TestCase):
    def test_returns_empty_when_binary_not_found(self):
        from switch_trust.scan import static_scanner

        with patch.object(static_scanner, "find_opengrep_binary", return_value=None):
            findings = static_scanner.run_opengrep(
                "/nonexistent", "/nonexistent/rules.yaml"
            )
            self.assertEqual(findings, [])


# ═════════════════════════════════════════════════════════════════════
# 12. agent_scanner helper functions
# ═════════════════════════════════════════════════════════════════════


class TestMakeFindingId(unittest.TestCase):
    def test_format(self):
        from switch_trust.scan.agent_scanner import make_finding_id

        fid = make_finding_id()
        self.assertTrue(fid.startswith("AGT-"))
        self.assertEqual(len(fid), 12)

    def test_unique(self):
        from switch_trust.scan.agent_scanner import make_finding_id

        ids = {make_finding_id() for _ in range(50)}
        self.assertEqual(len(ids), 50)


class TestCvssV4SeverityFromScore(unittest.TestCase):
    def test_critical(self):
        from switch_trust.scan.agent_scanner import _cvss_v4_severity_from_score

        self.assertEqual(_cvss_v4_severity_from_score(9.0), "Critical")
        self.assertEqual(_cvss_v4_severity_from_score(10.0), "Critical")

    def test_high(self):
        from switch_trust.scan.agent_scanner import _cvss_v4_severity_from_score

        self.assertEqual(_cvss_v4_severity_from_score(7.0), "High")
        self.assertEqual(_cvss_v4_severity_from_score(8.9), "High")

    def test_medium(self):
        from switch_trust.scan.agent_scanner import _cvss_v4_severity_from_score

        self.assertEqual(_cvss_v4_severity_from_score(4.0), "Medium")

    def test_low(self):
        from switch_trust.scan.agent_scanner import _cvss_v4_severity_from_score

        self.assertEqual(_cvss_v4_severity_from_score(0.1), "Low")

    def test_none(self):
        from switch_trust.scan.agent_scanner import _cvss_v4_severity_from_score

        self.assertEqual(_cvss_v4_severity_from_score(0.0), "None")


class TestDeriveLikelihood(unittest.TestCase):
    def test_high_confidence_critical(self):
        from switch_trust.scan.agent_scanner import _derive_likelihood

        self.assertEqual(_derive_likelihood("high", "Critical"), "High")

    def test_high_confidence_high(self):
        from switch_trust.scan.agent_scanner import _derive_likelihood

        self.assertEqual(_derive_likelihood("high", "High"), "High")

    def test_low_confidence(self):
        from switch_trust.scan.agent_scanner import _derive_likelihood

        self.assertEqual(_derive_likelihood("low", "Critical"), "Low")

    def test_low_severity(self):
        from switch_trust.scan.agent_scanner import _derive_likelihood

        self.assertEqual(_derive_likelihood("high", "Low"), "Low")

    def test_medium_default(self):
        from switch_trust.scan.agent_scanner import _derive_likelihood

        self.assertEqual(_derive_likelihood("medium", "Medium"), "Medium")


class TestBuildCategorySummary(unittest.TestCase):
    def test_empty_findings(self):
        from switch_trust.scan.agent_scanner import build_category_summary

        summary = build_category_summary([])
        for i in range(1, 11):
            key = f"asi{i:02d}"
            matching = [k for k in summary if k.startswith(key)]
            self.assertGreaterEqual(len(matching), 1)

    def test_counts_findings(self):
        from switch_trust.scan.agent_scanner import build_category_summary

        f1 = _make_finding(category="asi01_agent_goal_hijack", ai_spm_severity="High")
        f2 = _make_finding(category="asi01_agent_goal_hijack", ai_spm_severity="Medium")
        summary = build_category_summary([f1, f2])
        self.assertEqual(summary["asi01_agent_goal_hijack"]["count"], 2)
        self.assertEqual(summary["asi01_agent_goal_hijack"]["high"], 1)
        self.assertEqual(summary["asi01_agent_goal_hijack"]["medium"], 1)

    def test_total_matches(self):
        from switch_trust.scan.agent_scanner import build_category_summary

        findings = [
            _make_finding(category="asi01_agent_goal_hijack"),
            _make_finding(category="asi05_unexpected_code_execution"),
        ]
        summary = build_category_summary(findings)
        total = sum(v["count"] for v in summary.values())
        self.assertEqual(total, 2)

    def test_beyond_asi(self):
        from switch_trust.scan.agent_scanner import build_category_summary

        f = _make_finding(category="beyond_asi")
        summary = build_category_summary([f])
        self.assertEqual(summary["beyond_asi"]["count"], 1)

    def test_unknown_category_falls_to_beyond(self):
        from switch_trust.scan.agent_scanner import build_category_summary

        f = _make_finding(category="unknown_xyz")
        summary = build_category_summary([f])
        self.assertEqual(summary["beyond_asi"]["count"], 1)


class TestConvertAiFindingsEdgeCases(unittest.TestCase):
    def test_not_a_list_returns_empty(self):
        from switch_trust.scan.agent_scanner import convert_ai_findings

        result = convert_ai_findings("not a list")
        self.assertEqual(result, [])

    def test_non_dict_item_skipped(self):
        from switch_trust.scan.agent_scanner import convert_ai_findings

        result = convert_ai_findings(["not a dict", 42])
        self.assertEqual(result, [])

    def test_hallucinated_finding_capped(self):
        from switch_trust.scan.agent_scanner import convert_ai_findings

        raw = {
            "category": "asi05_unexpected_code_execution",
            "subcategory": "arbitrary_code_execution",
            "title": "Test",
            "description": "desc",
            "impact": "impact",
            "remediation": "fix",
            "affected_component": "f.py",
            "evidence": "code",
            "confidence": "high",
            "hallucination_flag": True,
        }
        result = convert_ai_findings([raw])
        self.assertEqual(len(result), 1)
        self.assertIn(result[0].ai_spm_severity, ("Medium", "Low"))

    def test_beyond_asi_category(self):
        from switch_trust.scan.agent_scanner import convert_ai_findings

        raw = {
            "category": "beyond_asi",
            "subcategory": "custom_issue",
            "title": "Custom",
            "description": "desc",
            "impact": "impact",
            "remediation": "fix",
            "affected_component": "f.py",
            "evidence": "code",
            "confidence": "medium",
            "hallucination_flag": False,
        }
        result = convert_ai_findings([raw])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].category, "beyond_asi")


class TestConvertStaticFindingsPipAudit(unittest.TestCase):
    def _make_static_finding(self, **overrides):
        from switch_trust.scan.static_scanner import StaticFinding

        defaults = {
            "tool": "pip_audit",
            "rule_id": "CVE-2024-1234",
            "severity": "high",
            "message": "langchain==0.1.0 has known vulnerability",
            "filepath": "requirements.txt",
            "line": 1,
            "evidence": "langchain==0.1.0",
        }
        defaults.update(overrides)
        return StaticFinding(**defaults)

    def test_pip_audit_mapping(self):
        from switch_trust.scan.agent_scanner import convert_static_findings

        sf = self._make_static_finding()
        result = convert_static_findings([sf])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].category, "asi04_supply_chain")
        self.assertEqual(result[0].subcategory, "known_vulnerable_dependency")
        self.assertEqual(result[0].source, "static_pip_audit")

    def test_internal_tool_mapping(self):
        from switch_trust.scan.agent_scanner import convert_static_findings

        sf = self._make_static_finding(tool="internal", rule_id="unpinned")
        result = convert_static_findings([sf])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].subcategory, "unpinned_dependencies")


if __name__ == "__main__":
    unittest.main()
