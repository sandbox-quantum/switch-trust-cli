"""
tool_dispatcher.py — Tool Implementations for the Agentic Reasoning Loop
=========================================================================
Routes tool calls from the AI reasoning agent to their implementations.

DESIGN PRINCIPLES:
  - All tools are READ-ONLY. No tool mutates state, writes files, or makes
    network calls. The repository files are already in memory from Layer 1.
  - report_finding() is the only "stateful" tool — it accumulates findings
    into the session list. This is intentional and auditable.
  - Every tool returns a string. The string becomes the tool message in the
    conversation and is visible in the trace log.
  - Tool errors return an "ERROR: ..." string rather than raising exceptions,
    so the agent can decide how to handle failures gracefully.

SAFETY:
  - fetch_file() enforces path validation — no path traversal.
  - run_targeted_bandit() is rate-limited by TOOL_CALL_LIMITS in tools.py.
  - The tool allowlist is enforced at the SDK level, not here.
"""

import ast
import fnmatch
import functools
import json
import logging
import os
import re
import subprocess
import tempfile
import typing

import yaml
from cvss import CVSS4

from switch_trust.schema import RepoFile
from switch_trust.scan.schema import AgentProfile, RawFinding
from switch_trust.scan.static_scanner import StaticFinding
from switch_trust.scan.taxonomy import (
    category_for_subcategory,
    resolve_subcategory,
)
from switch_trust.secret_anonymizer import anonymize_secrets

logger = logging.getLogger(__name__)


# Lazy-loaded CVSS mapping — loaded once on first compute_cvss() call
_CVSS_MAPPING: dict | None = None
_CVSS_MAPPING_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "agentic_cvss_mapping.yaml"
)


def _load_cvss_mapping() -> dict:
    """Load agentic_cvss_mapping.yaml once and cache it."""
    global _CVSS_MAPPING
    if _CVSS_MAPPING is not None:
        return _CVSS_MAPPING
    try:
        with open(_CVSS_MAPPING_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        _CVSS_MAPPING = data.get("agentic_cvss_mapping", {})
        return _CVSS_MAPPING
    except Exception as e:
        logger.warning("Could not load CVSS mapping: %s", e)
        _CVSS_MAPPING = {}
        return _CVSS_MAPPING


# ── Token approximation ───────────────────────────────────────────────────────
MAX_TOKENS_PER_FILE = 4000  # per fetch_file() call
CHARS_PER_TOKEN = 4  # rough approximation


def _truncate(text: str, max_tokens: int = MAX_TOKENS_PER_FILE) -> str:
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return (
        text[:max_chars]
        + f"\n\n[... truncated at {max_tokens} tokens. Use start_line/end_line to read more ...]"  # noqa: B950
    )


# ── Session state — populated by ToolDispatcher.__init__ ─────────────────────


class ToolDispatcher:
    """
    Stateful dispatcher for a single scan session.

    Holds references to the in-memory repository data so tool implementations
    can work without any I/O. Tracks per-tool call counts for rate limiting.

    Usage:
        dispatcher = ToolDispatcher(
            repo_files={"crew.py": RepoFile(...), ...},
            agents=[AgentProfile(...), ...],
            static_findings=[StaticFinding(...), ...],
        )
        result = dispatcher.dispatch("fetch_file", {"path": "crew.py"})
    """

    def __init__(
        self,
        repo_files: dict[str, RepoFile],
        agents: list[AgentProfile],
        static_findings: list[StaticFinding],
    ):
        self._files = repo_files  # {relative_path: RepoFile}
        self._agents = {a.agent_id: a for a in agents}
        self._findings = static_findings
        self._session_findings: list[RawFinding] = []
        self._call_counts: dict[str, int] = {}  # tool_name → call count
        self._tokens_consumed: int = 0  # total tokens fetched via fetch_file

        # Build a flattened search index: (path, line_number, line_text)
        self._search_index: list[tuple[str, int, str]] = []
        for path, rf in self._files.items():
            for i, line in enumerate(rf.content.splitlines(), start=1):
                self._search_index.append((path, i, line))

    @property
    def session_findings(self) -> list[RawFinding]:
        """All findings reported via report_finding() during this session."""
        return self._session_findings

    @property
    def tokens_consumed(self) -> int:
        """Total approximate tokens fetched via fetch_file() during this session."""
        return self._tokens_consumed

    # ── Tool methods (ADK-facing interface) ──────────────────────────────────

    def read_source(
        self,
        resource_type: str,
        path: str = "",
        pattern: str = "",
        start_line: int = 0,
        end_line: int = 0,
    ) -> str:
        """Read source data from the scanned repository.

        Use resource_type to select what to read:
          - 'file': Fetch file content (supports line ranges). Returns up to 4000 tokens.
          - 'agent': Return the full extracted AgentProfile for an agent, untruncated.
          - 'list': List all available files, optionally filtered by a glob pattern.

        Args:
            resource_type: What to read — 'file', 'agent', or 'list'.
            path: For 'file': relative file path. For 'agent': the agent_id.
            pattern: For 'list': optional glob pattern (e.g. '*.py').
            start_line: For 'file': first line to return (1-indexed, optional).
            end_line: For 'file': last line to return (optional).

        Returns:
            File content, agent profile, or file listing. ERROR string on failure.
        """
        self._call_counts["read_source"] = self._call_counts.get("read_source", 0) + 1
        if resource_type == "file":
            if not path:
                return "ERROR: 'path' is required when resource_type='file'."
            self._call_counts["fetch_file"] = self._call_counts.get("fetch_file", 0) + 1
            return self._fetch_file(
                path,
                start_line if start_line else None,
                end_line if end_line else None,
            )
        elif resource_type == "agent":
            if not path:
                return (
                    "ERROR: 'path' (agent_id) is required when resource_type='agent'."
                )
            return self._get_agent_profile(path)
        elif resource_type == "list":
            return self._list_files(pattern or None)
        return (
            f"ERROR: Invalid resource_type '{resource_type}'. "
            "Use 'file', 'agent', or 'list'."
        )

    def analyze_code(
        self,
        mode: str,
        pattern: str = "",
        file_path: str = "",
        file_extension: str = "",
        max_results: int = 20,
    ) -> str:
        """Analyze code patterns in the repository.

        Use mode to select the analysis type:
          - 'search': Search all files for lines matching a regex or substring.
          - 'imports': List all imports in a Python file and identify repo-local ones.

        Args:
            mode: 'search' for pattern matching, 'imports' for import resolution.
            pattern: For 'search': regex or substring (case-insensitive).
            file_path: For 'imports': relative path of the Python file.
            file_extension: For 'search': optional filter (e.g. '.py').
            max_results: For 'search': max matching lines (default 20).

        Returns:
            Matching results or import list. ERROR string on failure.
        """
        self._call_counts["analyze_code"] = self._call_counts.get("analyze_code", 0) + 1
        if mode == "search":
            if not pattern:
                return "ERROR: 'pattern' is required when mode='search'."
            return self._search_codebase(pattern, file_extension or None, max_results)
        elif mode == "imports":
            if not file_path:
                return "ERROR: 'file_path' is required when mode='imports'."
            return self._resolve_imports(file_path)
        return f"ERROR: Invalid mode '{mode}'. Use 'search' or 'imports'."

    def get_findings(
        self,
        file_path: str,
        mode: str = "cached",
    ) -> str:
        """Retrieve security findings for a specific file.

        Use mode to select the source:
          - 'cached': Return pre-computed findings from Bandit, OpenGrep,
            and detect-secrets (fast, no cost).
          - 'fresh': Run Bandit now (slower, rate-limited to 5 calls/session).

        Args:
            file_path: Relative path of the file.
            mode: 'cached' for pre-computed, 'fresh' to run Bandit. Default: 'cached'.

        Returns:
            Security findings for the file. ERROR string on failure.
        """
        self._call_counts["get_findings"] = self._call_counts.get("get_findings", 0) + 1
        if mode == "fresh":
            return self._run_targeted_bandit(file_path)
        return self._get_static_findings_for_file(file_path)

    # ── Tool implementations ─────────────────────────────────────────────────

    def _fetch_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        """
        Return the content of a repository file.
        Supports optional line range for targeted reads of large files.
        """
        # Path validation — prevent traversal and reject negative line numbers.
        # Use normpath + split to check individual path components rather than
        # a substring match, which is bypassable with lstrip tricks.
        normalized = os.path.normpath(path)
        if any(part == ".." for part in normalized.replace("\\", "/").split("/")):
            return f"ERROR: Path traversal not permitted: {path!r}"

        # Exact match first
        rf = self._files.get(path) or self._files.get(normalized)

        # Fuzzy match: if path is a filename, find it in any directory
        if rf is None:
            basename = os.path.basename(path)
            matches = [p for p in self._files if os.path.basename(p) == basename]
            if len(matches) == 1:
                rf = self._files[matches[0]]
                path = matches[0]  # update path for display
            elif len(matches) > 1:
                return (
                    f"ERROR: Ambiguous path '{path}' matches multiple files:\n"
                    + "\n".join(f"  {m}" for m in sorted(matches))
                    + "\nPlease specify the full path."
                )

        if rf is None:
            available = sorted(self._files.keys())[:20]
            return (
                f"ERROR: File not found: '{path}'\n"
                f"Available files (first 20):\n"
                + "\n".join(f"  {p}" for p in available)
            )

        lines = rf.content.splitlines()
        total = len(lines)

        if start_line is not None or end_line is not None:
            # Reject negative line numbers — they would silently clamp to 1
            # via max(1, ...) and confuse the agent about what was returned.
            if start_line is not None and start_line < 1:
                return (
                    f"ERROR: start_line must be >= 1, got {start_line}. "
                    f"Use a value between 1 and {total}."
                )
            if end_line is not None and end_line < 1:
                return (
                    f"ERROR: end_line must be >= 1, got {end_line}. "
                    f"Use a value between 1 and {total}."
                )
            # Fix #27: Validate line number bounds before slicing.
            # Out-of-range requests return an empty result that confuses the agent.
            if start_line is not None and start_line > total:
                return (
                    f"ERROR: start_line {start_line} exceeds file length ({total} lines) "
                    f"for '{path}'. Use a value between 1 and {total}."
                )
            if end_line is not None and end_line > total:
                # Clamp silently — requesting lines past EOF is a benign mistake.
                end_line = total

            s = max(1, start_line or 1) - 1  # convert to 0-indexed
            e = min(total, end_line or total)
            selected = lines[s:e]
            header = f"# {path} (lines {s + 1}–{e} of {total})\n"
            content = header + "\n".join(
                f"{s + 1 + i:4d}  {line}" for i, line in enumerate(selected)
            )
        else:
            header = f"# {path} ({total} lines)\n"
            content = header + "\n".join(
                f"{i + 1:4d}  {line}" for i, line in enumerate(lines)
            )

        result = _truncate(content)
        result = anonymize_secrets(result)
        self._tokens_consumed += len(result) // CHARS_PER_TOKEN
        return result

    def _get_agent_profile(self, agent_id: str) -> str:
        """Return the full AgentProfile for an agent, without truncation."""
        agent = self._agents.get(agent_id)
        if agent is None:
            available = sorted(self._agents.keys())
            return (
                f"ERROR: Agent '{agent_id}' not found.\nAvailable agents: {available}"
            )
        parts = [
            f"AgentProfile: {agent.agent_id}",
            f"  Framework:        {agent.framework}",
            f"  Source file:      {agent.source_file}",
            f"  Role:             {agent.role or 'not specified'}",
            f"  Goal:             {agent.goal or 'not specified'}",
            f"  Backstory:        {agent.backstory or 'not specified'}",
            f"  LLM:              {agent.llm or 'not specified'}",
            f"  Memory:           {agent.memory}",
            f"  Allow delegation: {agent.allow_delegation}",
            f"  Verbose:          {agent.verbose}",
            f"  Tools:            {agent.tools or []}",
            f"  Tool details:     {agent.tool_details or []}",
            f"  System prompt:    {agent.system_prompt or 'not specified'}",
            f"\n--- Full raw_code ---\n{anonymize_secrets(agent.raw_code)}",
        ]
        return "\n".join(parts)

    def _resolve_imports(self, file_path: str) -> str:
        """
        Parse a Python file's import statements and identify which imported
        modules are also present in the repository.
        """
        rf = self._files.get(file_path)
        if rf is None:
            return f"ERROR: File not found: '{file_path}'"

        try:
            tree = ast.parse(rf.content)
        except SyntaxError as e:
            return f"ERROR: Could not parse {file_path}: {e}"

        stdlib_skip = {
            "os",
            "sys",
            "re",
            "json",
            "time",
            "datetime",
            "pathlib",
            "typing",
            "dataclasses",
            "abc",
            "uuid",
            "tempfile",
            "subprocess",
            "logging",
            "argparse",
            "base64",
            "urllib",
            "collections",
            "itertools",
            "functools",
        }

        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.append(node.module.split(".")[0])

        results = []
        for mod in sorted(set(imported)):
            if mod in stdlib_skip:
                continue
            # Look for matching file in repo
            candidates = [
                p for p in self._files if os.path.basename(p).replace(".py", "") == mod
            ]
            if candidates:
                results.append(f"  {mod}  →  {candidates[0]}  [IN REPO — can fetch]")
            else:
                results.append(f"  {mod}  →  (external package or not fetched)")

        if not results:
            return f"No non-stdlib imports found in {file_path}"

        return f"Imports in {file_path}:\n" + "\n".join(results)

    def _search_codebase(
        self,
        pattern: str,
        file_extension: str | None = None,
        max_results: int = 20,
    ) -> str:
        """
        Search all fetched files for lines matching a pattern.
        Returns file:line matches with surrounding context.
        """
        matches = []
        regex = None
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # Fall back to literal substring match
            regex = None

        for file_path, line_no, line_text in self._search_index:
            if file_extension and not file_path.endswith(file_extension):
                continue
            hit = (
                (regex.search(line_text) is not None)
                if regex
                else (pattern.lower() in line_text.lower())
            )
            if hit:
                matches.append(f"  {file_path}:{line_no}:  {line_text.rstrip()}")
                if len(matches) >= max_results:
                    break

        if not matches:
            return f"No matches found for pattern '{pattern}'"

        header = f"Search results for '{pattern}' ({len(matches)} matches"
        if len(matches) == max_results:
            header += f", limit reached — narrow your pattern for more precision"  # noqa: F541
        header += "):"
        return header + "\n" + "\n".join(matches)

    def _run_targeted_bandit(self, file_path: str) -> str:
        """
        Run Bandit on a specific file from the in-memory repository.
        Writes the file to a temp dir, runs Bandit, returns results.
        """
        rf = self._files.get(file_path)
        if rf is None:
            return f"ERROR: File not found: '{file_path}'"

        if not file_path.endswith(".py"):
            return f"ERROR: Bandit only works on Python files: '{file_path}'"

        try:
            with tempfile.TemporaryDirectory() as tmp:
                safe_name = os.path.basename(file_path).replace("/", "_")
                tmp_path = os.path.join(tmp, safe_name)
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(rf.content)

                result = subprocess.run(
                    ["bandit", tmp_path, "-f", "json", "-q", "--severity-level", "low"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                stdout = result.stdout.strip()

                # Fix #8: Guard against pathologically large Bandit output.
                # A well-formed Python file should never produce more than ~500 KB
                # of Bandit JSON. If it does, truncate before parsing to prevent
                # memory exhaustion on adversarially crafted files.
                _MAX_BANDIT_OUTPUT = 500_000  # bytes
                if len(stdout) > _MAX_BANDIT_OUTPUT:
                    stdout = stdout[:_MAX_BANDIT_OUTPUT]
                    # The truncated JSON will fail to parse below — handle gracefully.

                if not stdout:
                    return f"Bandit found no issues in {file_path}"

                data = json.loads(stdout)
                issues = data.get("results", [])
                if not issues:
                    return f"Bandit found no issues in {file_path}"

                lines = [f"Bandit results for {file_path} ({len(issues)} issues):"]
                for issue in issues:
                    lines.append(
                        f"  [{issue.get('issue_severity', '?')}] "
                        f"Line {issue.get('line_number', '?')}: "
                        f"{issue.get('test_id', '?')} — {issue.get('issue_text', '')}"
                    )
                    if issue.get("code"):
                        lines.append(f"    Code: {issue['code'][:100].strip()}")
                return "\n".join(lines)

        except FileNotFoundError:
            return "ERROR: Bandit is not installed or not on PATH"
        except subprocess.TimeoutExpired:
            return f"ERROR: Bandit timed out scanning {file_path}"
        except Exception as e:
            return f"ERROR: Bandit scan failed: {e}"

    def _get_static_findings_for_file(self, file_path: str) -> str:
        """Return pre-computed static findings attributed to a specific file."""
        basename = os.path.basename(file_path)
        matching = [
            f
            for f in self._findings
            if (
                hasattr(f, "filepath")
                and (
                    f.filepath == file_path or os.path.basename(f.filepath) == basename
                )
            )
        ]
        if not matching:
            return f"No pre-computed static findings for '{file_path}'"

        lines = [f"Static findings for {file_path} ({len(matching)} findings):"]
        for f in matching:
            lines.append(
                f"  [{f.severity.upper()}] {f.tool.upper()} rule {f.rule_id} "
                f"line {f.line}: {f.message}"
            )
            if f.evidence:
                lines.append(
                    f"    Evidence: {anonymize_secrets(f.evidence[:120].strip())}"
                )
        return "\n".join(lines)

    def _list_files(self, pattern: str | None = None) -> str:
        """List all fetched file paths, optionally filtered by glob pattern."""
        paths = sorted(self._files.keys())
        if pattern:
            paths = [
                p
                for p in paths
                if fnmatch.fnmatch(p, pattern)
                or fnmatch.fnmatch(os.path.basename(p), pattern)
            ]
        if not paths:
            return (
                f"No files match pattern '{pattern}'"
                if pattern
                else "No files in repository."
            )
        return f"Files ({len(paths)}):\n" + "\n".join(f"  {p}" for p in paths)

    def compute_cvss(
        self,
        vuln_type: str,
        exposed_over_network: bool = False,
        requires_auth: bool = False,
        user_interaction: bool = False,
        affects_remote_system: bool = False,
    ) -> str:
        """Compute a deterministic CVSS v4 base score for a confirmed vulnerability.

        ALWAYS call this before calling report_finding(). Use the score and
        vector it returns in your finding. Do not estimate severity yourself.
        vuln_type must exactly match a key in the OWASP ASI taxonomy
        (e.g. 'hardcoded_credentials', 'arbitrary_code_execution').

        Args:
            vuln_type: Subcategory key from the OWASP ASI taxonomy.
            exposed_over_network: True if the component is network-reachable.
            requires_auth: True if exploitation requires authentication.
            user_interaction: True if a user must act for exploitation.
            affects_remote_system: True if impact extends to downstream systems.

        Returns:
            JSON string with vuln_type, vector, score, and severity.
        """
        self._call_counts["compute_cvss"] = self._call_counts.get("compute_cvss", 0) + 1
        mapping = _load_cvss_mapping()

        if vuln_type not in mapping:
            available = ", ".join(sorted(mapping.keys())[:10])
            return (
                f"ERROR: vuln_type '{vuln_type}' not found in CVSS mapping.\n"
                f"Available types (first 10): {available}\n"
                f"Check config/agentic_cvss_mapping.yaml for the full list."
            )

        base_vector = mapping[vuln_type].get("vector", "")
        if not base_vector:
            return f"ERROR: No base CVSS vector found for '{vuln_type}'"

        # Parse vector components into a mutable dict
        metrics = dict(re.findall(r"([A-Z]{1,2}):([A-Z]{1,2})", base_vector))

        # Apply context flag overrides
        if exposed_over_network:
            metrics["AV"] = "N"
        if requires_auth:
            metrics["PR"] = "L"
        if user_interaction:
            metrics["UI"] = "P"
        if affects_remote_system:
            metrics["SC"] = "H"
            metrics["SI"] = "H"
            metrics["SA"] = "H"

        # Reconstruct vector string
        new_vector = "CVSS:4.0/" + "/".join(f"{k}:{v}" for k, v in metrics.items())

        # Compute score using the cvss library
        try:
            cvss_obj = CVSS4(new_vector)
            score = round(cvss_obj.base_score, 1)
            severity = cvss_obj.severities()[
                0
            ]  # 'Critical' | 'High' | 'Medium' | 'Low' | 'None'
        except Exception as e:
            return f"ERROR: CVSS computation failed for vector '{new_vector}': {e}"

        result = {
            "vuln_type": vuln_type,
            "vector": new_vector,
            "score": score,
            "severity": severity,
            "overrides": {
                "exposed_over_network": exposed_over_network,
                "requires_auth": requires_auth,
                "user_interaction": user_interaction,
                "affects_remote_system": affects_remote_system,
            },
        }
        return json.dumps(result)

    def report_finding(
        self,
        subcategory: str,
        title: str,
        description: str,
        impact: str,
        remediation: str,
        affected_component: str,
        evidence: str,
        confidence: str,
        hallucination_flag: bool,
        evidence_file: str = "",
        evidence_line: int = 0,
        agent_name: str = "",
    ) -> str:
        """Record a confirmed security finding.

        Call compute_cvss() first to get the correct score and vector, then
        call this function to record the finding. Do not call this with
        speculative findings — only report what you can directly evidence.

        Args:
            subcategory: Specific vulnerability type, exactly as spelled in the
                taxonomy (e.g. 'direct_prompt_injection'). The ASI category is
                derived from it — do not pass one.
            title: Short human-readable title.
            description: Technical description of the issue.
            impact: What an attacker could achieve.
            remediation: Concrete fix guidance.
            affected_component: File, agent ID, or tool name.
            evidence: Direct code snippet or pattern (max 200 chars).
            confidence: One of 'high', 'medium', 'low'.
            hallucination_flag: True if you suspect but cannot fully prove the issue.
            evidence_file: Relative file path where the vulnerability was found.
            evidence_line: Line number in the file (use 0 if unknown).
            agent_name: The Agent name from the AGENT PROFILES.

        Returns:
            Confirmation string with the finding number.
        """
        self._call_counts["report_finding"] = (
            self._call_counts.get("report_finding", 0) + 1
        )
        # Fold case/separator variants onto the canonical key. An unrecognised
        # value is kept verbatim and classifies as `beyond_asi` downstream — the
        # documented path for a finding outside the framework — but it is logged,
        # because that path used to be silent and so was indistinguishable from
        # a typo the model could have corrected.
        canonical = resolve_subcategory(subcategory)
        if canonical is None:
            logger.warning(
                "report_finding: subcategory %r is not in the taxonomy; "
                "classifying as beyond_asi",
                subcategory,
            )
        else:
            subcategory = canonical
        category = category_for_subcategory(subcategory)
        finding = {
            "category": category,
            "subcategory": subcategory,
            "title": title,
            "description": description,
            "impact": impact,
            "remediation": remediation,
            "affected_component": affected_component,
            "evidence": anonymize_secrets(evidence[:200]),
            "confidence": confidence,
            "hallucination_flag": bool(hallucination_flag),
            "evidence_file": evidence_file,
            "evidence_line": int(evidence_line),
            "agent_name": agent_name,
        }
        self._session_findings.append(finding)
        return (
            f"Finding #{len(self._session_findings)} recorded: "
            f"[{category}/{subcategory}] {title} (confidence={confidence})"
        )

    def get_adk_tools(self, tracer=None) -> list:
        """Return ADK-compatible tool callables for this dispatcher.

        When tracer is provided, wraps each tool call with tracer.record_call()
        so the JSONL trace file records the agent's investigation.

        Resolves deferred type annotations (from __future__ import annotations)
        so ADK's function parameter parser gets real types, not strings.
        """
        tools = [
            self.read_source,
            self.analyze_code,
            self.get_findings,
            self.compute_cvss,
            self.report_finding,
        ]
        for tool in tools:
            fn = tool.__func__ if hasattr(tool, "__func__") else tool
            try:
                fn.__annotations__ = typing.get_type_hints(fn)
            except Exception:
                pass

        if tracer is None:
            return tools

        def _wrap(tool_name: str, fn):
            @functools.wraps(fn)
            def traced(*args, **kwargs):
                with tracer.record_call(tool_name, kwargs, tracer._iterations) as ctx:
                    result = fn(*args, **kwargs)
                    ctx.set_result(result)
                return result

            return traced

        return [
            _wrap("read_source", self.read_source),
            _wrap("analyze_code", self.analyze_code),
            _wrap("get_findings", self.get_findings),
            _wrap("compute_cvss", self.compute_cvss),
            _wrap("report_finding", self.report_finding),
        ]
