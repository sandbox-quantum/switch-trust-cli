"""
Tests for model string parsing and the unified Finding schema.
"""

import dataclasses
import unittest

from switch_trust.scan.llm_provider import parse_model_string
from switch_trust.scan.schema import AffectedComponent, CvssScores, Evidence, Finding

# ── Model string parsing ────────────────────────────────────────────


class TestModelStringParsing(unittest.TestCase):
    def test_provider_colon_model(self):
        self.assertEqual(parse_model_string("openai:gpt-5.4"), ("openai", "gpt-5.4"))

    def test_gemini_alias_maps_to_google(self):
        self.assertEqual(
            parse_model_string("gemini:gemini-2.5-flash"),
            ("google", "gemini-2.5-flash"),
        )

    def test_bare_provider_returns_none_model(self):
        self.assertEqual(parse_model_string("openai"), ("openai", None))

    def test_ollama_parsing(self):
        self.assertEqual(parse_model_string("ollama:llama3"), ("ollama", "llama3"))

    def test_litellm_parsing(self):
        self.assertEqual(parse_model_string("litellm:gpt-4o"), ("litellm", "gpt-4o"))

    def test_anthropic_parsing(self):
        self.assertEqual(
            parse_model_string("anthropic:claude-haiku-4-20250414"),
            ("anthropic", "claude-haiku-4-20250414"),
        )


# ── Unified Finding schema ──────────────────────────────────────────


class TestUnifiedFindingSchema(unittest.TestCase):
    def test_finding_round_trip(self):
        finding = Finding(
            id="AGT-TEST0001",
            category="asi01_agent_goal_hijack",
            subcategory="direct_prompt_injection",
            ai_spm_severity="High",
            cvss_v4_severity="High",
            cvss_scores=CvssScores(
                base_score=7.5,
                vector="CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:H/VI:H/VA:H/SC:N/SI:N/SA:N",
            ),
            description="Test finding",
            impact="Test impact",
            likelihood="High",
            remediation="Fix it",
            affected_components=[
                AffectedComponent(name="agent.py", path="src/agent.py")
            ],
            evidence=[
                Evidence(
                    file="agent.py", code_snippet="eval(user_input)", confidence="High"
                )
            ],
            hallucination_flag=False,
            title="Test title",
            source="ai_reasoning",
        )

        d = dataclasses.asdict(finding)
        self.assertEqual(d["id"], "AGT-TEST0001")
        self.assertEqual(d["ai_spm_severity"], "High")
        self.assertEqual(d["cvss_scores"]["base_score"], 7.5)
        self.assertEqual(d["affected_components"][0]["name"], "agent.py")
        self.assertEqual(d["evidence"][0]["code_snippet"], "eval(user_input)")

        from switch_trust.scan.agent_scanner import _dicts_to_findings

        reconstructed = _dicts_to_findings([d])
        self.assertEqual(len(reconstructed), 1)
        r = reconstructed[0]
        self.assertEqual(r.id, finding.id)
        self.assertEqual(r.ai_spm_severity, finding.ai_spm_severity)
        self.assertEqual(r.cvss_scores.base_score, finding.cvss_scores.base_score)
        self.assertEqual(r.affected_components[0].name, "agent.py")
        self.assertEqual(r.evidence[0].code_snippet, "eval(user_input)")

    def test_finding_has_no_legacy_fields(self):
        fields = {f.name for f in Finding.__dataclass_fields__.values()}
        self.assertNotIn("finding_id", fields)
        self.assertNotIn("severity", fields)
        self.assertNotIn("cvss_score", fields)
        self.assertNotIn("cvss_vector", fields)
        self.assertNotIn("affected_component", fields)
        self.assertNotIn("confidence", fields)
        self.assertNotIn("compliance_mappings", fields)
        self.assertNotIn("owasp_classified", fields)


if __name__ == "__main__":
    unittest.main()
