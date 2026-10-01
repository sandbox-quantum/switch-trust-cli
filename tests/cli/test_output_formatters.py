"""Tests for output_formatters module."""

from __future__ import annotations

import dataclasses
import json
import unittest

from switch_trust.cli.output_formatters import (
    EVAL_SCHEMA_VERSION,
    JsonEvalOutputFormatter,
    JsonScanOutputFormatter,
    OutputFormat,
    SarifEvalOutputFormatter,
    SarifScanOutputFormatter,
    get_eval_output_formatter,
    get_scan_output_formatter,
    prepare_eval_output,
)
from switch_trust.cli.runner import CliRunResult
from switch_trust.eval.core.eval.evaluation import (
    EvaluationResult,
    EvaluationStatus,
    EvaluationSummary,
)
from switch_trust.scan.schema import (
    CvssScores,
    Finding,
    ScanReport,
)
from switch_trust.schema import AffectedComponent, Evidence

# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_scan_report() -> ScanReport:
    return ScanReport(
        schema_version="2.0",
        repo_name="test-repo",
        scan_timestamp="2025-01-15T12:00:00Z",
        framework_detected="crewai",
        agents_found=1,
        findings=[
            Finding(
                id="f-001",
                category="ASI-04",
                subcategory="Prompt Injection",
                ai_spm_severity="high",
                cvss_v4_severity="high",
                cvss_scores=CvssScores(base_score=7.5, vector="AV:N/AC:L"),
                description="Prompt injection via user input.",
                impact="Agent behavior redirection.",
                likelihood="high",
                remediation="Add input validation.",
                affected_components=[
                    AffectedComponent(name="agent.py", path="src/agent.py"),
                ],
                evidence=[
                    Evidence(
                        file="src/agent.py",
                        line=42,
                        code_snippet='prompt = f"{user_input}"',
                        confidence="HIGH",
                        context="User input in system prompt",
                    ),
                ],
                title="Prompt Injection in Agent",
                source="ai_reasoning",
            ),
            Finding(
                id="f-002",
                category="ASI-04",
                subcategory="Prompt Injection",
                ai_spm_severity="medium",
                cvss_v4_severity="medium",
                cvss_scores=CvssScores(base_score=5.0, vector=""),
                description="Secondary injection vector.",
                impact="Partial control.",
                likelihood="medium",
                remediation="Sanitize inputs.",
                affected_components=[],
                evidence=[Evidence(file="other.py", line=10)],
                title="Secondary Injection",
                source="static_bandit",
            ),
        ],
        scan_metadata={"python_files": 5, "total_files_scanned": 10},
    )


def _make_scan_report_empty() -> ScanReport:
    return ScanReport(
        schema_version="2.0",
        repo_name="clean-repo",
        scan_timestamp="2025-01-15T12:00:00Z",
        framework_detected="langchain",
        agents_found=0,
    )


def _make_eval_run() -> CliRunResult:
    return CliRunResult(
        model_evaluation_id="me-001",
        model_evaluation_name="GPT-4 / Jailbreak",
        model={"id": "m-001", "name": "GPT-4", "type": "openai"},
        evaluation={"id": "e-001", "name": "Jailbreak", "type": "adversarial"},
        summary=EvaluationSummary(
            status=EvaluationStatus.FINISHED,
            total_evaluations=3,
            finished_evaluations=3,
            error_evaluations=0,
            max_score=3.0,
            achieved_score=1.8,
        ),
        results=[
            EvaluationResult(score=0.8, status=EvaluationStatus.FINISHED),
            EvaluationResult(score=0.0, status=EvaluationStatus.FINISHED),
            EvaluationResult(score=1.0, status=EvaluationStatus.FINISHED),
        ],
    )


def _make_eval_run_all_zero() -> CliRunResult:
    return CliRunResult(
        model_evaluation_id="me-002",
        model_evaluation_name="Claude / Safety",
        model={"id": "m-002", "name": "Claude", "type": "anthropic"},
        evaluation={"id": "e-002", "name": "Safety", "type": "safety"},
        summary=EvaluationSummary(
            status=EvaluationStatus.FINISHED,
            total_evaluations=2,
            finished_evaluations=2,
            error_evaluations=0,
            max_score=2.0,
            achieved_score=0.0,
        ),
        results=[
            EvaluationResult(score=0.0, status=EvaluationStatus.FINISHED),
            EvaluationResult(score=0.0, status=EvaluationStatus.FINISHED),
        ],
    )


# ── Registry ─────────────────────────────────────────────────────────────────


class TestRegistry(unittest.TestCase):
    def test_get_scan_json(self):
        fmt = get_scan_output_formatter(OutputFormat.JSON)
        self.assertIsInstance(fmt, JsonScanOutputFormatter)

    def test_get_scan_sarif(self):
        fmt = get_scan_output_formatter(OutputFormat.SARIF)
        self.assertIsInstance(fmt, SarifScanOutputFormatter)

    def test_get_eval_json(self):
        fmt = get_eval_output_formatter(OutputFormat.JSON)
        self.assertIsInstance(fmt, JsonEvalOutputFormatter)

    def test_get_eval_sarif(self):
        fmt = get_eval_output_formatter(OutputFormat.SARIF)
        self.assertIsInstance(fmt, SarifEvalOutputFormatter)

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            OutputFormat("xml")

    def test_extensions(self):
        self.assertEqual(get_scan_output_formatter(OutputFormat.JSON).extension, "json")
        self.assertEqual(
            get_scan_output_formatter(OutputFormat.SARIF).extension, "sarif"
        )
        self.assertEqual(get_eval_output_formatter(OutputFormat.JSON).extension, "json")
        self.assertEqual(
            get_eval_output_formatter(OutputFormat.SARIF).extension, "sarif"
        )


# ── Scan JSON ────────────────────────────────────────────────────────────────


class TestScanJson(unittest.TestCase):
    def setUp(self):
        self.scan_report = _make_scan_report()
        self.scan_report_empty = _make_scan_report_empty()

    def test_output_matches_dataclasses_asdict(self):
        formatter = JsonScanOutputFormatter()
        result = json.loads(formatter.format(self.scan_report))
        expected = json.loads(
            json.dumps(dataclasses.asdict(self.scan_report), indent=2),
        )
        # Agent-discovery fields are dropped from CLI JSON output (out of scope
        # for the CLI path); everything else must match the raw asdict.
        for key in ("agents_found", "agent_profiles"):
            expected.pop(key, None)
        self.assertEqual(result, expected)

    def test_agent_discovery_fields_dropped(self):
        formatter = JsonScanOutputFormatter()
        parsed = json.loads(formatter.format(self.scan_report))
        self.assertNotIn("agents_found", parsed)
        self.assertNotIn("agent_profiles", parsed)

    def test_valid_json(self):
        formatter = JsonScanOutputFormatter()
        output = formatter.format(self.scan_report)
        parsed = json.loads(output)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["schema_version"], "2.0")

    def test_empty_findings(self):
        formatter = JsonScanOutputFormatter()
        parsed = json.loads(formatter.format(self.scan_report_empty))
        self.assertEqual(parsed["findings"], [])

    def test_findings_preserved(self):
        formatter = JsonScanOutputFormatter()
        parsed = json.loads(formatter.format(self.scan_report))
        self.assertEqual(len(parsed["findings"]), 2)
        self.assertEqual(parsed["findings"][0]["id"], "f-001")
        self.assertEqual(parsed["findings"][1]["id"], "f-002")


# ── Scan SARIF ───────────────────────────────────────────────────────────────


class TestScanSarif(unittest.TestCase):
    def setUp(self):
        self.scan_report = _make_scan_report()
        self.scan_report_empty = _make_scan_report_empty()

    def test_sarif_envelope(self):
        formatter = SarifScanOutputFormatter()
        sarif = json.loads(formatter.format(self.scan_report))
        self.assertEqual(sarif["version"], "2.1.0")
        self.assertIn("$schema", sarif)
        self.assertEqual(len(sarif["runs"]), 1)

    def test_tool_info(self):
        formatter = SarifScanOutputFormatter()
        sarif = json.loads(formatter.format(self.scan_report))
        driver = sarif["runs"][0]["tool"]["driver"]
        self.assertEqual(driver["name"], "switch-trust-scan")
        self.assertIn("version", driver)
        self.assertIn("informationUri", driver)

    def test_rules_deduplicated(self):
        formatter = SarifScanOutputFormatter()
        sarif = json.loads(formatter.format(self.scan_report))
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["id"], "ASI-04/Prompt Injection")

    def test_result_count(self):
        formatter = SarifScanOutputFormatter()
        sarif = json.loads(formatter.format(self.scan_report))
        results = sarif["runs"][0]["results"]
        self.assertEqual(len(results), 2)

    def test_severity_mapping(self):
        formatter = SarifScanOutputFormatter()
        sarif = json.loads(formatter.format(self.scan_report))
        results = sarif["runs"][0]["results"]
        self.assertEqual(results[0]["level"], "error")  # high
        self.assertEqual(results[1]["level"], "warning")  # medium

    def test_severity_mapping_all_levels(self):
        formatter = SarifScanOutputFormatter()
        self.assertEqual(formatter._severity_to_level("critical"), "error")
        self.assertEqual(formatter._severity_to_level("high"), "error")
        self.assertEqual(formatter._severity_to_level("medium"), "warning")
        self.assertEqual(formatter._severity_to_level("low"), "note")
        self.assertEqual(formatter._severity_to_level("CRITICAL"), "error")

    def test_locations_from_affected_components(self):
        formatter = SarifScanOutputFormatter()
        sarif = json.loads(formatter.format(self.scan_report))
        result = sarif["runs"][0]["results"][0]
        self.assertEqual(len(result["locations"]), 1)
        uri = result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        self.assertEqual(uri, "src/agent.py")

    def test_locations_fallback_to_evidence(self):
        formatter = SarifScanOutputFormatter()
        sarif = json.loads(formatter.format(self.scan_report))
        result = sarif["runs"][0]["results"][1]
        self.assertEqual(len(result["locations"]), 1)
        uri = result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        self.assertEqual(uri, "other.py")

    def test_related_locations(self):
        formatter = SarifScanOutputFormatter()
        sarif = json.loads(formatter.format(self.scan_report))
        result = sarif["runs"][0]["results"][0]
        self.assertEqual(len(result["relatedLocations"]), 1)
        rel = result["relatedLocations"][0]
        self.assertEqual(rel["physicalLocation"]["region"]["startLine"], 42)
        self.assertIn("snippet", rel["physicalLocation"]["region"])
        self.assertEqual(rel["message"]["text"], "User input in system prompt")

    def test_cvssv4_in_properties(self):
        formatter = SarifScanOutputFormatter()
        sarif = json.loads(formatter.format(self.scan_report))
        props = sarif["runs"][0]["results"][0]["properties"]
        self.assertEqual(props["cvssv4"]["baseScore"], 7.5)
        self.assertEqual(props["cvssv4"]["vector"], "AV:N/AC:L")

    def test_guid_set(self):
        formatter = SarifScanOutputFormatter()
        sarif = json.loads(formatter.format(self.scan_report))
        self.assertEqual(sarif["runs"][0]["results"][0]["guid"], "f-001")

    def test_empty_findings_produces_empty_results(self):
        formatter = SarifScanOutputFormatter()
        sarif = json.loads(formatter.format(self.scan_report_empty))
        self.assertEqual(sarif["runs"][0]["results"], [])
        self.assertEqual(sarif["runs"][0]["tool"]["driver"]["rules"], [])

    def test_artifacts_collected(self):
        formatter = SarifScanOutputFormatter()
        sarif = json.loads(formatter.format(self.scan_report))
        artifacts = sarif["runs"][0]["artifacts"]
        uris = [a["location"]["uri"] for a in artifacts]
        self.assertIn("src/agent.py", uris)
        self.assertIn("other.py", uris)

    def test_run_properties(self):
        formatter = SarifScanOutputFormatter()
        sarif = json.loads(formatter.format(self.scan_report))
        props = sarif["runs"][0]["properties"]
        self.assertNotIn("frameworkDetected", props)
        self.assertEqual(props["repoName"], "test-repo")

    def test_invocation_timestamp(self):
        formatter = SarifScanOutputFormatter()
        sarif = json.loads(formatter.format(self.scan_report))
        inv = sarif["runs"][0]["invocations"][0]
        self.assertTrue(inv["executionSuccessful"])
        self.assertEqual(inv["startTimeUtc"], "2025-01-15T12:00:00Z")

    def test_hallucination_flag(self):
        report = ScanReport(
            findings=[
                Finding(
                    id="f-h",
                    category="C",
                    subcategory="S",
                    ai_spm_severity="low",
                    cvss_v4_severity="low",
                    cvss_scores=CvssScores(),
                    description="d",
                    impact="",
                    likelihood="low",
                    remediation="",
                    affected_components=[],
                    evidence=[],
                    hallucination_flag=True,
                    source="ai_reasoning",
                ),
            ],
        )
        formatter = SarifScanOutputFormatter()
        sarif = json.loads(formatter.format(report))
        self.assertTrue(
            sarif["runs"][0]["results"][0]["properties"]["hallucinationFlag"]
        )

    def test_message_uses_impact_without_title(self):
        formatter = SarifScanOutputFormatter()
        sarif = json.loads(formatter.format(self.scan_report))
        msg = sarif["runs"][0]["results"][0]["message"]["text"]
        self.assertNotIn("\n", msg)
        self.assertTrue(msg.startswith("Impact:"))
        self.assertNotIn("Prompt Injection in Agent", msg)

    def test_message_falls_back_to_title_when_no_impact(self):
        report = ScanReport(
            findings=[
                Finding(
                    id="f-t",
                    category="C",
                    subcategory="S",
                    ai_spm_severity="low",
                    cvss_v4_severity="low",
                    cvss_scores=CvssScores(),
                    description="desc",
                    impact="",
                    likelihood="low",
                    remediation="",
                    affected_components=[],
                    evidence=[],
                    title="Some Title",
                    source="test",
                ),
            ],
        )
        formatter = SarifScanOutputFormatter()
        sarif = json.loads(formatter.format(report))
        self.assertEqual(
            sarif["runs"][0]["results"][0]["message"]["text"], "Some Title"
        )

    def test_paths_have_leading_slash(self):
        report = ScanReport(
            findings=[
                Finding(
                    id="f-p",
                    category="C",
                    subcategory="S",
                    ai_spm_severity="low",
                    cvss_v4_severity="low",
                    cvss_scores=CvssScores(),
                    description="d",
                    impact="i",
                    likelihood="low",
                    remediation="",
                    affected_components=[
                        AffectedComponent(
                            name="agent.py",
                            path="Users/timo/sb/project/agent.py",
                        ),
                    ],
                    evidence=[
                        Evidence(
                            file="Users/timo/sb/project/agent.py",
                            line=10,
                        ),
                    ],
                    source="test",
                ),
            ],
        )
        formatter = SarifScanOutputFormatter()
        sarif = json.loads(formatter.format(report))
        result = sarif["runs"][0]["results"][0]
        uri = result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        self.assertTrue(uri.startswith("/"), f"URI missing leading slash: {uri}")
        rel_uri = result["relatedLocations"][0]["physicalLocation"]["artifactLocation"][
            "uri"
        ]
        self.assertTrue(
            rel_uri.startswith("/"), f"Related URI missing leading slash: {rel_uri}"
        )

    def test_location_uri_comes_from_evidence_not_component_name(self):
        # Regression: the primary location used `comp.path or comp.name`, and an
        # AI finding's component has no path — so the model's free-text label
        # ("request_return", "check_order", "agent.py:app") was published as the
        # artifact URI, varying between runs while the real file sat unused in
        # relatedLocations.
        report = _make_scan_report()
        finding = report.findings[0]
        finding.affected_components = [AffectedComponent(name="request_return")]
        finding.evidence = [Evidence(file="agent.py", line=42)]

        sarif = json.loads(SarifScanOutputFormatter().format(report))
        result = sarif["runs"][0]["results"][0]
        uri = result["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        self.assertEqual(uri, "agent.py")
        self.assertEqual(
            result["locations"][0]["physicalLocation"]["region"]["startLine"], 42
        )

    def test_component_names_are_preserved_as_a_property(self):
        report = _make_scan_report()
        report.findings[0].affected_components = [
            AffectedComponent(name="request_return")
        ]
        sarif = json.loads(SarifScanOutputFormatter().format(report))
        props = sarif["runs"][0]["results"][0]["properties"]
        self.assertEqual(props["affectedComponents"], ["request_return"])

    def test_component_path_is_used_when_there_is_no_evidence_file(self):
        # Static findings set a real path, so that remains a valid fallback.
        report = _make_scan_report()
        finding = report.findings[0]
        finding.affected_components = [
            AffectedComponent(name="agent.py", path="src/agent.py")
        ]
        finding.evidence = []
        sarif = json.loads(SarifScanOutputFormatter().format(report))
        uri = sarif["runs"][0]["results"][0]["locations"][0]["physicalLocation"][
            "artifactLocation"
        ]["uri"]
        self.assertEqual(uri, "src/agent.py")

    def test_no_location_rather_than_a_name_shaped_one(self):
        # With neither an evidence file nor a component path there is nothing
        # path-like to report. Emitting the name would put a non-path back into
        # a URI field, which is the bug this replaced.
        report = _make_scan_report()
        finding = report.findings[0]
        finding.affected_components = [AffectedComponent(name="request_return")]
        finding.evidence = []
        sarif = json.loads(SarifScanOutputFormatter().format(report))
        result = sarif["runs"][0]["results"][0]
        self.assertNotIn("locations", result)
        self.assertEqual(result["properties"]["affectedComponents"], ["request_return"])

    def test_relative_paths_unchanged(self):
        formatter = SarifScanOutputFormatter()
        self.assertEqual(formatter._normalize_uri("src/agent.py"), "src/agent.py")
        self.assertEqual(formatter._normalize_uri("/abs/path.py"), "/abs/path.py")
        self.assertEqual(
            formatter._normalize_uri("Users/timo/project/f.py"),
            "/Users/timo/project/f.py",
        )


# ── Eval JSON ────────────────────────────────────────────────────────────────


class TestEvalJson(unittest.TestCase):
    def setUp(self):
        self.eval_run = _make_eval_run()

    def test_schema_version_present(self):
        formatter = JsonEvalOutputFormatter()
        parsed = json.loads(formatter.format([self.eval_run], "/config.json"))
        self.assertEqual(parsed["schema_version"], EVAL_SCHEMA_VERSION)

    def test_keys_match_old_format_plus_schema_version(self):
        formatter = JsonEvalOutputFormatter()
        parsed = json.loads(formatter.format([self.eval_run], "/config.json"))
        expected_keys = {
            "schema_version",
            "config_file",
            "timestamp",
            "summary",
            "runs",
        }
        self.assertEqual(set(parsed.keys()), expected_keys)

    def test_runs_preserved(self):
        formatter = JsonEvalOutputFormatter()
        parsed = json.loads(formatter.format([self.eval_run], "/config.json"))
        self.assertEqual(len(parsed["runs"]), 1)
        self.assertEqual(parsed["runs"][0]["model_evaluation_id"], "me-001")

    def test_nulls_stripped(self):
        formatter = JsonEvalOutputFormatter()
        output = formatter.format([self.eval_run], "/config.json")
        parsed = json.loads(output)
        for result in parsed["runs"][0]["results"]:
            self.assertNotIn("error_message", result)

    def test_session_metadata_stripped(self):
        formatter = JsonEvalOutputFormatter()
        output = formatter.format([self.eval_run], "/config.json")
        parsed = json.loads(output)
        for result in parsed["runs"][0]["results"]:
            self.assertNotIn("status", result)

    def test_config_path(self):
        formatter = JsonEvalOutputFormatter()
        parsed = json.loads(formatter.format([self.eval_run], "/my/config.json"))
        self.assertEqual(parsed["config_file"], "/my/config.json")


# ── Eval SARIF ───────────────────────────────────────────────────────────────


class TestEvalSarif(unittest.TestCase):
    def setUp(self):
        self.eval_run = _make_eval_run()
        self.eval_run_all_zero = _make_eval_run_all_zero()

    def test_sarif_envelope(self):
        formatter = SarifEvalOutputFormatter()
        sarif = json.loads(formatter.format([self.eval_run], "/config.json"))
        self.assertEqual(sarif["version"], "2.1.0")
        self.assertEqual(len(sarif["runs"]), 1)

    def test_tool_info(self):
        formatter = SarifEvalOutputFormatter()
        sarif = json.loads(formatter.format([self.eval_run], "/config.json"))
        driver = sarif["runs"][0]["tool"]["driver"]
        self.assertEqual(driver["name"], "switch-trust-eval")

    def test_one_result_per_run(self):
        formatter = SarifEvalOutputFormatter()
        sarif = json.loads(formatter.format([self.eval_run], "/config.json"))
        results = sarif["runs"][0]["results"]
        self.assertEqual(len(results), 1)

    def test_all_zero_scores_produces_no_results(self):
        formatter = SarifEvalOutputFormatter()
        sarif = json.loads(formatter.format([self.eval_run_all_zero], "/config.json"))
        self.assertEqual(sarif["runs"][0]["results"], [])

    def test_score_to_level_mapping(self):
        formatter = SarifEvalOutputFormatter()
        self.assertEqual(formatter._score_to_level(1.0), "error")
        self.assertEqual(formatter._score_to_level(0.7), "error")
        self.assertEqual(formatter._score_to_level(0.5), "warning")
        self.assertEqual(formatter._score_to_level(0.3), "warning")
        self.assertEqual(formatter._score_to_level(0.1), "note")

    def test_logical_location(self):
        formatter = SarifEvalOutputFormatter()
        sarif = json.loads(formatter.format([self.eval_run], "/config.json"))
        loc = sarif["runs"][0]["results"][0]["locations"][0]
        self.assertEqual(loc["logicalLocations"][0]["name"], "GPT-4")
        self.assertEqual(loc["logicalLocations"][0]["fullyQualifiedName"], "m-001")

    def test_result_properties_contain_aggregate_scores(self):
        formatter = SarifEvalOutputFormatter()
        sarif = json.loads(formatter.format([self.eval_run], "/config.json"))
        props = sarif["runs"][0]["results"][0]["properties"]
        self.assertEqual(props["achievedScore"], 1.8)
        self.assertEqual(props["maxScore"], 3.0)
        self.assertEqual(props["totalPrompts"], 3)
        self.assertAlmostEqual(props["averageScore"], 0.6)

    def test_run_properties(self):
        formatter = SarifEvalOutputFormatter()
        sarif = json.loads(formatter.format([self.eval_run], "/config.json"))
        props = sarif["runs"][0]["properties"]
        self.assertEqual(props["configFile"], "/config.json")
        self.assertNotIn("modelEvaluationId", props)

    def test_result_properties_contain_run_metadata(self):
        formatter = SarifEvalOutputFormatter()
        sarif = json.loads(formatter.format([self.eval_run], "/config.json"))
        props = sarif["runs"][0]["results"][0]["properties"]
        self.assertEqual(props["modelEvaluationId"], "me-001")
        self.assertEqual(props["modelEvaluationName"], "GPT-4 / Jailbreak")
        self.assertEqual(props["totalEvaluations"], 3)
        self.assertEqual(props["status"], "finished")

    def test_message_text_shows_score_without_interpretation(self):
        formatter = SarifEvalOutputFormatter()
        sarif = json.loads(formatter.format([self.eval_run], "/config.json"))
        msg = sarif["runs"][0]["results"][0]["message"]["text"]
        self.assertIn("GPT-4", msg)
        self.assertIn("1.8/3.0", msg)
        self.assertIn("3 prompts", msg)
        self.assertNotIn("attack", msg)
        self.assertNotIn("successful", msg)

    def test_short_description_uses_eval_name(self):
        formatter = SarifEvalOutputFormatter()
        sarif = json.loads(formatter.format([self.eval_run], "/config.json"))
        rule = sarif["runs"][0]["tool"]["driver"]["rules"][0]
        self.assertEqual(rule["shortDescription"]["text"], "Jailbreak")

    def test_single_run_multiple_evals(self):
        formatter = SarifEvalOutputFormatter()
        sarif = json.loads(
            formatter.format([self.eval_run, self.eval_run_all_zero], "/config.json"),
        )
        self.assertEqual(len(sarif["runs"]), 1)
        rules = sarif["runs"][0]["tool"]["driver"]["rules"]
        self.assertEqual(len(rules), 2)
        self.assertEqual(rules[0]["id"], "Jailbreak")
        self.assertEqual(rules[1]["id"], "Safety")
        results = sarif["runs"][0]["results"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["ruleId"], "Jailbreak")


# ── prepare_eval_output (backward compat) ────────────────────────────────────


class TestPrepareEvalOutput(unittest.TestCase):
    def setUp(self):
        self.eval_run = _make_eval_run()

    def test_schema_version_added(self):
        output = prepare_eval_output([self.eval_run], "/config.json")
        self.assertEqual(output["schema_version"], EVAL_SCHEMA_VERSION)

    def test_status_stripped_from_results(self):
        output = prepare_eval_output([self.eval_run], "/config.json")
        for result in output["runs"][0]["results"]:
            self.assertNotIn("status", result)

    def test_nulls_removed(self):
        output = prepare_eval_output([self.eval_run], "/config.json")
        for result in output["runs"][0]["results"]:
            for _key, val in result.items():
                self.assertIsNotNone(val)
