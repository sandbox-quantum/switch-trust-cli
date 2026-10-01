"""
Tests for triage.py — triage agent logic.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from switch_trust.scan.triage import build_agent_context, run_triage


class TestBuildAgentContext(unittest.TestCase):
    def test_basic_context(self):
        profiles = [
            {
                "agent_id": "agent1",
                "role": "researcher",
                "goal": "find info",
                "tools": ["web_search", "file_read"],
            }
        ]
        ctx = build_agent_context(
            repo_name="test-repo",
            framework="crewai",
            agent_profiles=profiles,
            description="A test agent",
        )
        self.assertEqual(ctx["repo_name"], "test-repo")
        self.assertEqual(ctx["framework"], "crewai")
        self.assertEqual(ctx["description"], "A test agent")
        self.assertEqual(len(ctx["agent_profiles"]), 1)
        self.assertIn("agent1: web_search", ctx["tool_inventory"])
        self.assertIn("agent1: file_read", ctx["tool_inventory"])

    def test_empty_profiles(self):
        ctx = build_agent_context("repo", "unknown", [])
        self.assertEqual(ctx["agent_profiles"], [])
        self.assertEqual(ctx["tool_inventory"], [])

    def test_deduplicates_tools(self):
        profiles = [
            {"agent_id": "a1", "tools": ["search"]},
            {"agent_id": "a2", "tools": ["search"]},
        ]
        ctx = build_agent_context("repo", "fw", profiles)
        self.assertEqual(len(ctx["tool_inventory"]), 2)

    def test_compact_profiles_structure(self):
        profiles = [
            {
                "agent_id": "a1",
                "role": "r",
                "goal": "g",
                "backstory": "b",
                "tools": ["t"],
                "memory": True,
                "allow_delegation": False,
                "extra_field": "ignored",
            }
        ]
        ctx = build_agent_context("repo", "fw", profiles)
        compact = ctx["agent_profiles"][0]
        self.assertEqual(compact["agent_id"], "a1")
        self.assertEqual(compact["role"], "r")
        self.assertNotIn("extra_field", compact)


class TestRunTriage(unittest.TestCase):
    def test_returns_none_for_empty_findings(self):
        result = run_triage([], {})
        self.assertIsNone(result)

    @patch("switch_trust.scan.triage.complete_text")
    @patch(
        "switch_trust.scan.triage._load_triage_prompt",
        return_value="You are a triage agent.",
    )
    def test_successful_triage(self, mock_prompt, mock_complete):
        mock_complete.return_value = json.dumps(
            {
                "kept_finding_ids": [
                    {"id": "F1", "ai_spm_severity": "High"},
                ],
                "triage_dismissed": [{"id": "F2", "reason": "expected behavior"}],
                "triage_downgraded": [],
                "triage_summary": {
                    "total_input": 2,
                    "total_kept": 1,
                    "total_dismissed": 1,
                    "total_downgraded": 0,
                    "severity_distribution": {"high": 1},
                },
            }
        )

        findings = [
            {
                "id": "F1",
                "ai_spm_severity": "High",
                "source": "ai_reasoning",
                "subcategory": "test",
                "evidence": [{"confidence": "High"}],
            },
            {
                "id": "F2",
                "ai_spm_severity": "Low",
                "source": "ai_reasoning",
                "subcategory": "test",
                "evidence": [{"confidence": "Low"}],
            },
        ]
        ctx = {"repo_name": "test", "framework": "crewai", "agent_profiles": []}

        result = run_triage(findings, ctx, model=MagicMock())
        self.assertIsNotNone(result)
        self.assertEqual(len(result["kept_findings"]), 1)
        self.assertEqual(result["kept_findings"][0]["id"], "F1")
        self.assertEqual(len(result["triage_dismissed"]), 1)

    @patch("switch_trust.scan.triage.complete_text", return_value="not valid json {{")
    @patch("switch_trust.scan.triage._load_triage_prompt", return_value="prompt")
    def test_json_parse_error_returns_identity(self, mock_prompt, mock_complete):
        findings = [{"id": "F1", "ai_spm_severity": "High", "subcategory": "test"}]
        result = run_triage(findings, {}, model=MagicMock())
        self.assertIsNotNone(result)
        self.assertEqual(len(result["kept_findings"]), 1)
        self.assertTrue(result["triage_summary"]["parse_error"])

    @patch("switch_trust.scan.triage.complete_text", return_value=None)
    @patch("switch_trust.scan.triage._load_triage_prompt", return_value="prompt")
    def test_provider_returns_none(self, mock_prompt, mock_complete):
        findings = [{"id": "F1", "ai_spm_severity": "High", "subcategory": "test"}]
        result = run_triage(findings, {}, model=MagicMock())
        self.assertIsNone(result)

    @patch("switch_trust.scan.triage.complete_text")
    @patch("switch_trust.scan.triage._load_triage_prompt", return_value="prompt")
    def test_strips_code_fences(self, mock_prompt, mock_complete):
        inner = json.dumps(
            {
                "kept_finding_ids": [{"id": "F1", "ai_spm_severity": "High"}],
                "triage_dismissed": [],
                "triage_downgraded": [],
                "triage_summary": {
                    "total_input": 1,
                    "total_kept": 1,
                    "total_dismissed": 0,
                    "total_downgraded": 0,
                    "severity_distribution": {},
                },
            }
        )
        mock_complete.return_value = f"```json\n{inner}\n```"

        findings = [{"id": "F1", "ai_spm_severity": "High", "subcategory": "test"}]
        result = run_triage(findings, {}, model=MagicMock())
        self.assertIsNotNone(result)
        self.assertEqual(len(result["kept_findings"]), 1)

    @patch("switch_trust.scan.triage.complete_text")
    @patch("switch_trust.scan.triage._load_triage_prompt", return_value="prompt")
    def test_unaccounted_findings_readded(self, mock_prompt, mock_complete):
        mock_complete.return_value = json.dumps(
            {
                "kept_finding_ids": [{"id": "F1", "ai_spm_severity": "High"}],
                "triage_dismissed": [],
                "triage_downgraded": [],
                "triage_summary": {},
            }
        )

        findings = [
            {"id": "F1", "ai_spm_severity": "High", "subcategory": "test"},
            {"id": "F2", "ai_spm_severity": "Low", "subcategory": "test"},
        ]
        result = run_triage(findings, {}, model=MagicMock())
        self.assertIsNotNone(result)
        kept_ids = {f["id"] for f in result["kept_findings"]}
        self.assertIn("F2", kept_ids)

    @patch("switch_trust.scan.triage.complete_text")
    @patch("switch_trust.scan.triage._load_triage_prompt", return_value="prompt")
    def test_cve_dedup_and_expansion(self, mock_prompt, mock_complete):
        mock_complete.return_value = json.dumps(
            {
                "kept_finding_ids": [{"id": "CVE1", "ai_spm_severity": "High"}],
                "triage_dismissed": [],
                "triage_downgraded": [],
                "triage_summary": {},
            }
        )

        findings = [
            {
                "id": "CVE1",
                "ai_spm_severity": "High",
                "subcategory": "known_vulnerable_dependency",
                "source": "static_pip_audit",
                "affected_components": [{"name": "requests"}],
            },
            {
                "id": "CVE2",
                "ai_spm_severity": "Medium",
                "subcategory": "known_vulnerable_dependency",
                "source": "static_pip_audit",
                "affected_components": [{"name": "requests"}],
            },
        ]
        result = run_triage(findings, {}, model=MagicMock())
        self.assertIsNotNone(result)
        kept_ids = {f["id"] for f in result["kept_findings"]}
        self.assertIn("CVE1", kept_ids)
        self.assertIn("CVE2", kept_ids)

    @patch("switch_trust.scan.triage._load_triage_prompt")
    def test_prompt_not_found_returns_none(self, mock_prompt):
        mock_prompt.side_effect = FileNotFoundError("not found")

        findings = [{"id": "F1", "ai_spm_severity": "High", "subcategory": "test"}]
        result = run_triage(findings, {}, model=MagicMock())
        self.assertIsNone(result)

    @patch("switch_trust.scan.triage.complete_text")
    @patch("switch_trust.scan.triage._load_triage_prompt", return_value="prompt")
    def test_missing_keys_backward_compat(self, mock_prompt, mock_complete):
        mock_complete.return_value = json.dumps(
            {
                "kept_findings": [{"id": "F1", "ai_spm_severity": "High"}],
                "triage_dismissed": [],
                "triage_downgraded": [],
                "triage_summary": {},
            }
        )

        findings = [{"id": "F1", "ai_spm_severity": "High", "subcategory": "test"}]
        result = run_triage(findings, {}, model=MagicMock())
        self.assertIsNotNone(result)
        self.assertEqual(len(result["kept_findings"]), 1)


if __name__ == "__main__":
    unittest.main()
