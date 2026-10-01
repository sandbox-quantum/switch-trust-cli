"""
Tests for static_scanner.py — static analysis tool runners.
"""

import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from switch_trust.scan.static_scanner import (
    BANDIT_SEVERITY_MAP,
    OPENGREP_SEVERITY_MAP,
    StaticFinding,
    check_unpinned_dependencies,
    run_bandit,
    run_detect_secrets,
    run_opengrep,
    run_static_scan,
)


class TestStaticFinding(unittest.TestCase):
    def test_creation(self):
        f = StaticFinding(
            tool="bandit",
            rule_id="B102",
            severity="high",
            message="exec used",
            filepath="agent.py",
            line=10,
        )
        self.assertEqual(f.tool, "bandit")
        self.assertEqual(f.rule_id, "B102")
        self.assertEqual(f.line, 10)

    def test_defaults(self):
        f = StaticFinding(
            tool="test",
            rule_id="T1",
            severity="low",
            message="msg",
            filepath="f.py",
        )
        self.assertEqual(f.line, 0)
        self.assertEqual(f.evidence, "")


class TestSeverityMaps(unittest.TestCase):
    def test_bandit_map(self):
        self.assertEqual(BANDIT_SEVERITY_MAP["HIGH"], "high")
        self.assertEqual(BANDIT_SEVERITY_MAP["MEDIUM"], "medium")
        self.assertEqual(BANDIT_SEVERITY_MAP["LOW"], "low")

    def test_opengrep_map(self):
        self.assertEqual(OPENGREP_SEVERITY_MAP["ERROR"], "high")
        self.assertEqual(OPENGREP_SEVERITY_MAP["WARNING"], "medium")


class TestRunBandit(unittest.TestCase):
    # Bandit is a core dependency, but the scanner still guards on
    # `_module_available` in case it can't be imported at runtime, so it is
    # patched throughout: these tests cover parsing, not installation.
    @patch("switch_trust.scan.static_scanner._module_available", return_value=True)
    @patch("switch_trust.scan.static_scanner.subprocess.run")
    def test_parses_json_output(self, mock_run, mock_available):
        mock_run.return_value = MagicMock(
            stdout=json.dumps(
                {
                    "results": [
                        {
                            "test_id": "B102",
                            "issue_severity": "HIGH",
                            "issue_text": "exec() used",
                            "filename": "/tmp/scan/agent.py",
                            "line_number": 10,
                            "code": "exec(user_input)",
                            "issue_cwe": {"id": "CWE-78"},
                        }
                    ]
                }
            ),
            returncode=1,
        )
        findings = run_bandit("/tmp/scan")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].tool, "bandit")
        self.assertEqual(findings[0].rule_id, "B102")
        self.assertEqual(findings[0].severity, "high")

    @patch("switch_trust.scan.static_scanner._module_available", return_value=True)
    @patch("switch_trust.scan.static_scanner.subprocess.run")
    def test_empty_output(self, mock_run, mock_available):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        findings = run_bandit("/tmp/scan")
        self.assertEqual(findings, [])

    @patch("switch_trust.scan.static_scanner._module_available", return_value=True)
    @patch("switch_trust.scan.static_scanner.subprocess.run")
    def test_handles_exception(self, mock_run, mock_available):
        mock_run.side_effect = FileNotFoundError("bandit not found")
        findings = run_bandit("/tmp/scan")
        self.assertEqual(findings, [])

    @patch("switch_trust.scan.static_scanner._module_available", return_value=False)
    @patch("switch_trust.scan.static_scanner.subprocess.run")
    def test_skips_when_module_not_installed(self, mock_run, mock_available):
        findings = run_bandit("/tmp/scan")
        self.assertEqual(findings, [])
        mock_run.assert_not_called()


class TestRunOpengrep(unittest.TestCase):
    @patch("switch_trust.scan.static_scanner.find_opengrep_binary", return_value=None)
    def test_skips_when_binary_not_found(self, mock_find):
        findings = run_opengrep("/tmp", "/tmp/rules.yaml")
        self.assertEqual(findings, [])

    @patch("switch_trust.scan.static_scanner.subprocess.run")
    @patch(
        "switch_trust.scan.static_scanner.find_opengrep_binary",
        return_value="/usr/bin/opengrep",
    )
    def test_parses_json_output(self, mock_find, mock_run):
        mock_run.return_value = MagicMock(
            stdout=json.dumps(
                {
                    "results": [
                        {
                            "check_id": "eval-call",
                            "extra": {
                                "severity": "ERROR",
                                "message": "eval() call",
                                "lines": "eval(x)",
                            },
                            "path": "agent.py",
                            "start": {"line": 5},
                        }
                    ]
                }
            ),
            returncode=0,
        )
        findings = run_opengrep("/tmp/scan", "/tmp/rules.yaml")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].tool, "opengrep")

    @patch("switch_trust.scan.static_scanner.subprocess.run")
    @patch(
        "switch_trust.scan.static_scanner.find_opengrep_binary",
        return_value="/usr/bin/opengrep",
    )
    def test_handles_empty_output(self, mock_find, mock_run):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        findings = run_opengrep("/tmp", "/tmp/rules.yaml")
        self.assertEqual(findings, [])


class TestRunDetectSecrets(unittest.TestCase):
    # detect-secrets is a core dependency, but the scanner still guards on
    # `_module_available` in case it can't be imported at runtime, so it is
    # patched throughout: these tests cover parsing, not installation.
    @patch("switch_trust.scan.static_scanner._module_available", return_value=True)
    @patch("switch_trust.scan.static_scanner.subprocess.run")
    def test_parses_output(self, mock_run, mock_available):
        mock_run.return_value = MagicMock(
            stdout=json.dumps(
                {
                    "results": {
                        "agent.py": [
                            {
                                "type": "Hex High Entropy String",
                                "line_number": 3,
                            }
                        ]
                    }
                }
            ),
            returncode=0,
        )
        findings = run_detect_secrets("/tmp/scan")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].tool, "detect_secrets")

    @patch("switch_trust.scan.static_scanner._module_available", return_value=True)
    @patch("switch_trust.scan.static_scanner.subprocess.run")
    def test_empty_results(self, mock_run, mock_available):
        mock_run.return_value = MagicMock(
            stdout=json.dumps({"results": {}}),
            returncode=0,
        )
        findings = run_detect_secrets("/tmp/scan")
        self.assertEqual(findings, [])

    @patch("switch_trust.scan.static_scanner._module_available", return_value=True)
    @patch("switch_trust.scan.static_scanner.subprocess.run")
    def test_handles_exception(self, mock_run, mock_available):
        mock_run.side_effect = FileNotFoundError("not found")
        findings = run_detect_secrets("/tmp/scan")
        self.assertEqual(findings, [])

    @patch("switch_trust.scan.static_scanner._module_available", return_value=False)
    @patch("switch_trust.scan.static_scanner.subprocess.run")
    def test_skips_when_module_not_installed(self, mock_run, mock_available):
        findings = run_detect_secrets("/tmp/scan")
        self.assertEqual(findings, [])
        mock_run.assert_not_called()


class TestCheckUnpinnedDependencies(unittest.TestCase):
    def test_detects_unpinned_ai_package(self):
        content = "openai>=1.0\ncrewai\nflask==2.0\n"
        findings = check_unpinned_dependencies(content, "requirements.txt")
        unpinned = [f for f in findings if f.rule_id == "unpinned-ai-dependency"]
        self.assertEqual(len(unpinned), 2)

    def test_all_pinned(self):
        content = "openai==1.0.0\ncrewai==0.5.0\n"
        findings = check_unpinned_dependencies(content, "requirements.txt")
        unpinned = [f for f in findings if f.rule_id == "unpinned-ai-dependency"]
        self.assertEqual(len(unpinned), 0)

    def test_empty_content(self):
        findings = check_unpinned_dependencies("", "requirements.txt")
        self.assertEqual(findings, [])


class TestRunStaticScan(unittest.TestCase):
    def _files(self):
        from switch_trust.schema import RepoFile

        return (
            RepoFile(path="test.py", content="x = 1\n", size=6),
            RepoFile(
                path="requirements.txt",
                content="flask==2.0\nopenai\n",
                size=20,
            ),
        )

    def test_with_real_temp_files(self):
        py_file, req_file = self._files()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_static_scan([py_file], [req_file], tmpdir)

        self.assertIsInstance(result.findings, list)
        # Bandit, detect-secrets and pip-audit may fail to import at runtime,
        # so a tool is either used or declared skipped — never silently absent.
        accounted = set(result.tools_used) | {s.tool for s in result.tools_skipped}
        self.assertIn("bandit", accounted)
        self.assertIn("detect-secrets", accounted)

    @patch("switch_trust.scan.static_scanner._module_available", return_value=False)
    def test_missing_tools_are_declared(self, mock_available):
        from switch_trust.schema import RepoFile

        py_file, _ = self._files()
        # Unpinned on purpose: the OSV fallback only queries `==`-pinned
        # packages, which keeps this test off the network.
        req_file = RepoFile(path="requirements.txt", content="openai\n", size=7)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_static_scan([py_file], [req_file], tmpdir)

        skipped = {s.tool for s in result.tools_skipped}
        self.assertIn("bandit", skipped)
        self.assertIn("detect-secrets", skipped)
        self.assertIn("pip-audit", skipped)
        self.assertNotIn("bandit", result.tools_used)
        self.assertNotIn("detect-secrets", result.tools_used)
        self.assertNotIn("pip-audit", result.tools_used)
        # pip-audit degrades to the OSV.dev API rather than dropping out.
        self.assertIn("osv", result.tools_used)


if __name__ == "__main__":
    unittest.main()
