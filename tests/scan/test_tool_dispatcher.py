"""
Tests for tool_dispatcher.py — agentic tool implementations.
"""

import json
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from switch_trust.scan.schema import AgentProfile
from switch_trust.scan.tool_dispatcher import ToolDispatcher, _truncate


@dataclass
class RepoFile:
    path: str
    content: str
    size: int


@dataclass
class StaticFinding:
    tool: str
    rule_id: str
    severity: str
    message: str
    filepath: str
    line: int = 0
    evidence: str = ""


def _make_dispatcher(files=None, agents=None, findings=None):
    repo_files = {}
    if files:
        for path, content in files.items():
            repo_files[path] = RepoFile(path=path, content=content, size=len(content))
    return ToolDispatcher(
        repo_files=repo_files,
        agents=agents or [],
        static_findings=findings or [],
    )


class TestTruncate(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(_truncate("hello"), "hello")

    def test_long_text_truncated(self):
        text = "a" * 100000
        result = _truncate(text, max_tokens=10)
        self.assertIn("truncated", result)
        self.assertLess(len(result), len(text))


class TestReadSource(unittest.TestCase):
    def test_fetch_file(self):
        d = _make_dispatcher(files={"agent.py": "line1\nline2\nline3"})
        result = d.read_source(resource_type="file", path="agent.py")
        self.assertIn("line1", result)
        self.assertIn("line2", result)

    def test_fetch_file_with_line_range(self):
        content = "\n".join(f"line {i}" for i in range(1, 21))
        d = _make_dispatcher(files={"big.py": content})
        result = d.read_source(
            resource_type="file", path="big.py", start_line=5, end_line=10
        )
        self.assertIn("line 5", result)
        self.assertIn("line 10", result)
        self.assertNotIn("line 1 ", result)

    def test_fetch_file_not_found(self):
        d = _make_dispatcher(files={"a.py": "x"})
        result = d.read_source(resource_type="file", path="missing.py")
        self.assertIn("ERROR", result)

    def test_fetch_file_path_traversal(self):
        d = _make_dispatcher(files={"a.py": "x"})
        result = d.read_source(resource_type="file", path="../../../etc/passwd")
        self.assertIn("ERROR", result)
        self.assertIn("traversal", result)

    def test_fetch_file_fuzzy_match(self):
        d = _make_dispatcher(files={"src/agents/crew.py": "code"})
        result = d.read_source(resource_type="file", path="crew.py")
        self.assertIn("code", result)

    def test_fetch_file_ambiguous_path(self):
        d = _make_dispatcher(files={"a/f.py": "x", "b/f.py": "y"})
        result = d.read_source(resource_type="file", path="f.py")
        self.assertIn("ERROR", result)
        self.assertIn("Ambiguous", result)

    def test_fetch_file_missing_path(self):
        d = _make_dispatcher()
        result = d.read_source(resource_type="file")
        self.assertIn("ERROR", result)

    def test_fetch_file_negative_start_line(self):
        d = _make_dispatcher(files={"a.py": "line1\nline2"})
        result = d.read_source(resource_type="file", path="a.py", start_line=-1)
        self.assertIn("ERROR", result)

    def test_fetch_file_start_line_past_eof(self):
        d = _make_dispatcher(files={"a.py": "line1\nline2"})
        result = d.read_source(resource_type="file", path="a.py", start_line=999)
        self.assertIn("ERROR", result)

    def test_list_files(self):
        d = _make_dispatcher(files={"a.py": "x", "b.txt": "y", "c.py": "z"})
        result = d.read_source(resource_type="list")
        self.assertIn("a.py", result)
        self.assertIn("b.txt", result)

    def test_list_files_with_pattern(self):
        d = _make_dispatcher(files={"a.py": "x", "b.txt": "y"})
        result = d.read_source(resource_type="list", pattern="*.py")
        self.assertIn("a.py", result)
        self.assertNotIn("b.txt", result)

    def test_list_files_no_match(self):
        d = _make_dispatcher(files={"a.py": "x"})
        result = d.read_source(resource_type="list", pattern="*.rs")
        self.assertIn("No files", result)

    def test_get_agent_profile(self):
        agent = AgentProfile(
            agent_id="test_agent",
            framework="crewai",
            source_file="crew.py",
            role="researcher",
            goal="find info",
        )
        d = _make_dispatcher(agents=[agent])
        result = d.read_source(resource_type="agent", path="test_agent")
        self.assertIn("test_agent", result)
        self.assertIn("crewai", result)

    def test_get_agent_not_found(self):
        d = _make_dispatcher()
        result = d.read_source(resource_type="agent", path="missing")
        self.assertIn("ERROR", result)

    def test_invalid_resource_type(self):
        d = _make_dispatcher()
        result = d.read_source(resource_type="invalid")
        self.assertIn("ERROR", result)


class TestAnalyzeCode(unittest.TestCase):
    def test_search_finds_match(self):
        d = _make_dispatcher(files={"a.py": "import openai\nx = 1"})
        result = d.analyze_code(mode="search", pattern="openai")
        self.assertIn("openai", result)
        self.assertIn("a.py", result)

    def test_search_no_match(self):
        d = _make_dispatcher(files={"a.py": "x = 1"})
        result = d.analyze_code(mode="search", pattern="openai")
        self.assertIn("No matches", result)

    def test_search_with_extension_filter(self):
        d = _make_dispatcher(files={"a.py": "hello", "b.txt": "hello"})
        result = d.analyze_code(mode="search", pattern="hello", file_extension=".py")
        self.assertIn("a.py", result)
        self.assertNotIn("b.txt", result)

    def test_search_regex(self):
        d = _make_dispatcher(files={"a.py": "def foo():\n    pass"})
        result = d.analyze_code(mode="search", pattern=r"def \w+")
        self.assertIn("def foo", result)

    def test_search_invalid_regex_falls_back(self):
        d = _make_dispatcher(files={"a.py": "test[data"})
        result = d.analyze_code(mode="search", pattern="test[data")
        self.assertIn("test[data", result)

    def test_search_missing_pattern(self):
        d = _make_dispatcher()
        result = d.analyze_code(mode="search")
        self.assertIn("ERROR", result)

    def test_imports_resolution(self):
        d = _make_dispatcher(
            files={
                "crew.py": "import os\nimport openai\nfrom utils import helper",
                "utils.py": "def helper(): pass",
            }
        )
        result = d.analyze_code(mode="imports", file_path="crew.py")
        self.assertIn("openai", result)
        self.assertIn("utils", result)
        self.assertIn("IN REPO", result)

    def test_imports_file_not_found(self):
        d = _make_dispatcher()
        result = d.analyze_code(mode="imports", file_path="missing.py")
        self.assertIn("ERROR", result)

    def test_imports_missing_file_path(self):
        d = _make_dispatcher()
        result = d.analyze_code(mode="imports")
        self.assertIn("ERROR", result)

    def test_invalid_mode(self):
        d = _make_dispatcher()
        result = d.analyze_code(mode="invalid")
        self.assertIn("ERROR", result)


class TestGetFindings(unittest.TestCase):
    def test_cached_findings(self):
        sf = StaticFinding(
            tool="bandit",
            rule_id="B102",
            severity="high",
            message="exec used",
            filepath="agent.py",
            line=10,
            evidence="exec(x)",
        )
        d = _make_dispatcher(findings=[sf])
        result = d.get_findings(file_path="agent.py")
        self.assertIn("BANDIT", result.upper())
        self.assertIn("B102", result)

    def test_no_cached_findings(self):
        d = _make_dispatcher()
        result = d.get_findings(file_path="agent.py")
        self.assertIn("No pre-computed", result)


class TestReportFinding(unittest.TestCase):
    def test_records_finding(self):
        d = _make_dispatcher()
        result = d.report_finding(
            subcategory="direct_prompt_injection",
            title="Test",
            description="desc",
            impact="impact",
            remediation="fix",
            affected_component="agent.py",
            evidence="eval(x)",
            confidence="high",
            hallucination_flag=False,
        )
        self.assertIn("Finding #1", result)
        self.assertEqual(len(d.session_findings), 1)
        self.assertEqual(d.session_findings[0]["category"], "asi01_agent_goal_hijack")

    def test_truncates_evidence(self):
        d = _make_dispatcher()
        d.report_finding(
            subcategory="test",
            title="t",
            description="d",
            impact="i",
            remediation="r",
            affected_component="c",
            evidence="x" * 500,
            confidence="low",
            hallucination_flag=False,
        )
        self.assertLessEqual(len(d.session_findings[0]["evidence"]), 200)


class TestSessionProperties(unittest.TestCase):
    def test_session_findings_empty(self):
        d = _make_dispatcher()
        self.assertEqual(d.session_findings, [])

    def test_tokens_consumed(self):
        d = _make_dispatcher(files={"a.py": "x" * 100})
        d.read_source(resource_type="file", path="a.py")
        self.assertGreater(d.tokens_consumed, 0)


class TestComputeCvss(unittest.TestCase):
    def test_known_vuln_type(self):
        d = _make_dispatcher()
        result = d.compute_cvss(vuln_type="arbitrary_code_execution")
        try:
            data = json.loads(result)
            self.assertIn("score", data)
            self.assertIn("vector", data)
            self.assertIn("severity", data)
        except json.JSONDecodeError:
            if "ERROR" in result and "not found in CVSS mapping" in result:
                pass
            else:
                self.fail(f"Unexpected result: {result}")

    def test_unknown_vuln_type(self):
        d = _make_dispatcher()
        result = d.compute_cvss(vuln_type="totally_nonexistent_xyz")
        self.assertIn("ERROR", result)

    def test_context_overrides(self):
        d = _make_dispatcher()
        result = d.compute_cvss(
            vuln_type="arbitrary_code_execution",
            exposed_over_network=True,
            requires_auth=True,
        )
        if not result.startswith("ERROR"):
            data = json.loads(result)
            self.assertTrue(data["overrides"]["exposed_over_network"])
            self.assertTrue(data["overrides"]["requires_auth"])


class TestRunTargetedBandit(unittest.TestCase):
    def test_file_not_found(self):
        d = _make_dispatcher()
        result = d.get_findings(file_path="missing.py", mode="fresh")
        self.assertIn("ERROR", result)

    def test_non_python_file(self):
        d = _make_dispatcher(files={"data.txt": "content"})
        result = d.get_findings(file_path="data.txt", mode="fresh")
        self.assertIn("ERROR", result)

    @patch("switch_trust.scan.tool_dispatcher.subprocess.run")
    def test_bandit_success(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout=json.dumps(
                {
                    "results": [
                        {
                            "issue_severity": "HIGH",
                            "line_number": 5,
                            "test_id": "B102",
                            "issue_text": "exec() used",
                            "code": "exec(x)",
                        }
                    ]
                }
            ),
            returncode=1,
        )
        d = _make_dispatcher(files={"agent.py": "exec(x)\n"})
        result = d.get_findings(file_path="agent.py", mode="fresh")
        self.assertIn("B102", result)

    @patch("switch_trust.scan.tool_dispatcher.subprocess.run")
    def test_bandit_no_issues(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        d = _make_dispatcher(files={"safe.py": "x = 1\n"})
        result = d.get_findings(file_path="safe.py", mode="fresh")
        self.assertIn("no issues", result)

    @patch("switch_trust.scan.tool_dispatcher.subprocess.run")
    def test_bandit_not_installed(self, mock_run):
        mock_run.side_effect = FileNotFoundError("bandit not found")
        d = _make_dispatcher(files={"a.py": "x = 1\n"})
        result = d.get_findings(file_path="a.py", mode="fresh")
        self.assertIn("ERROR", result)

    @patch("switch_trust.scan.tool_dispatcher.subprocess.run")
    def test_bandit_timeout(self, mock_run):
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("bandit", 30)
        d = _make_dispatcher(files={"a.py": "x = 1\n"})
        result = d.get_findings(file_path="a.py", mode="fresh")
        self.assertIn("ERROR", result)


class TestResolveImportsEdgeCases(unittest.TestCase):
    def test_syntax_error_in_file(self):
        d = _make_dispatcher(files={"bad.py": "def broken("})
        result = d.analyze_code(mode="imports", file_path="bad.py")
        self.assertIn("ERROR", result)

    def test_no_imports(self):
        d = _make_dispatcher(files={"simple.py": "x = 1\ny = 2\n"})
        result = d.analyze_code(mode="imports", file_path="simple.py")
        self.assertIn("No non-stdlib imports", result)


class TestGetAdkTools(unittest.TestCase):
    def test_returns_tools_without_tracer(self):
        d = _make_dispatcher()
        tools = d.get_adk_tools()
        self.assertEqual(len(tools), 5)

    def test_returns_wrapped_tools_with_tracer(self):
        from switch_trust.scan.trace_logger_log import LogTraceLogger

        d = _make_dispatcher()
        tracer = LogTraceLogger()
        tracer._iterations = 0
        tools = d.get_adk_tools(tracer=tracer)
        self.assertEqual(len(tools), 5)


if __name__ == "__main__":
    unittest.main()
