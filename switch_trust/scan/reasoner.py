"""
reasoner.py — Agentic Reasoning Loop (v2)
==========================================
Implements the agentic scanning loop using Google ADK.

ADK handles the internal reasoning loop — tool selection, invocation,
and termination. We supply the agent definition, tools, and system prompt.
ADK's Runner executes locally (no Vertex AI deployment required).

SESSION STORAGE:
  Uses InMemorySessionService — sessions are ephemeral and discarded when
  the process exits. This is appropriate because findings are accumulated
  in-memory via the ToolDispatcher (not from session state) and no
  downstream code reads sessions back after a scan completes.

MULTI-PROVIDER SUPPORT:
  Model selection is handled via llm_provider.make_model() which reads
  the SCANNER_MODEL env var (provider:model format):
    google            → bare Gemini model string (e.g. "gemini-3.5-flash")
    all others        → LiteLlm("provider/model") via llm_provider

CONTROL PLANE (ASI compliance for our own scanner):
  MAX_ITERATIONS     = 40   — hard cap on tool call rounds (ASI08)
  MAX_FILES_FETCHED  = 50   — limit on files the agent can read (ASI08)
  LOOP_TIMEOUT_SECS  = 600  — wall-clock timeout (ASI08)

OUTPUT:
  Identical format to v1: (findings_list, summary_string, trace_dict)
  The rest of the pipeline (triage, report assembly) is unchanged.

Canonical import path: from agent_scanner.reasoner import run_agentic_reasoning
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import time
import uuid
from contextlib import suppress
from typing import Any

from switch_trust.schema import RepoFile
from switch_trust.scan.schema import AgentProfile, RawFinding
from switch_trust.scan.static_scanner import StaticFinding
from switch_trust.scan.taxonomy import AGENT_TAXONOMY
from switch_trust.scan.tool_dispatcher import ToolDispatcher
from switch_trust.scan.trace_logger_file import FileTraceLogger
from switch_trust.scan.trace_logger_log import LogTraceLogger
from google.adk.agents import LlmAgent
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
from switch_trust.scan.exception_classification import (
    FailureDomain,
    classify_exception,
    is_auth_error,
    is_model_error,
    is_network_error,
    is_not_entitled,
    is_rate_limited,
    is_resource_exhausted,
    is_timeout_error,
    is_upstream_error,
)
from switch_trust.scan.llm_provider import _safe_error, make_model
from switch_trust.secret_anonymizer import anonymize_secrets

logger = logging.getLogger(__name__)

# ── Operational limits ────────────────────────────────────────────────────────
# All three limits are configurable via environment variables.
# The values below are the production defaults — override per-deployment
# without code changes or redeployment.
#
#   ADK_MAX_ITERATIONS    — max tool-call rounds before forced stop      (default: 40)
#   ADK_MAX_FILES_FETCHED — max distinct files the agent may read        (default: 50)
#   ADK_MAX_FETCH_TOKENS  — token budget for all fetch_file content      (default: 200000)
#   ADK_LOOP_TIMEOUT_SECS — wall-clock timeout for the full ADK loop     (default: 600)
MAX_ITERATIONS = int(os.getenv("ADK_MAX_ITERATIONS", "80"))
MAX_FILES_FETCHED = int(os.getenv("ADK_MAX_FILES_FETCHED", "50"))
MAX_FETCH_TOKENS = int(os.getenv("ADK_MAX_FETCH_TOKENS", "200000"))
LOOP_TIMEOUT_SECS = int(os.getenv("ADK_LOOP_TIMEOUT_SECS", "600"))

# The `exit_reason` values that mean "we stopped early because one of our own
# budgets ran out", as opposed to "we finished" or "we broke". Kept next to the
# guards in `_run_adk_async` that produce them, not in the caller that
# interprets them, so a new guard cannot silently fall outside the set.
BUDGET_EXIT_REASONS = frozenset(
    {
        "timeout",
        "max_iterations",
        "max_files",
        "max_tokens",
        "max_bandit_calls",
        "max_cvss_calls",
    }
)


# ── System prompt ─────────────────────────────────────────────────────────────


def _build_system_prompt() -> str:
    taxonomy_lines = []
    for _cat_key, cat_data in AGENT_TAXONOMY.items():
        taxonomy_lines.append(
            f"\n{cat_data['asi_code']} — {cat_data['asi_title']}: {cat_data['description']}"
        )
        for subcat_key, subcat_data in cat_data["subcategories"].items():
            taxonomy_lines.append(
                f"  [{subcat_data['severity'].upper()}] {subcat_key}: "
                f"{subcat_data['title']} — {subcat_data['description']}"
            )
    taxonomy_ref = "\n".join(taxonomy_lines)

    return f"""You are an application security researcher specializing in AI
agents. You investigate source code and reason about how an agent actually
behaves — how untrusted data flows into it, what its tools can do, how it is
exposed, and where a control is missing — then report every well-evidenced
vulnerability you can substantiate against the taxonomy below.

You are NOT a grep script. The most important agentic vulnerabilities are
behavioral and data-flow properties, not single-line syntactic patterns:
untrusted input reaching the agent, tool output fed back to the model,
state-changing tools with no confirmation, an unauthenticated entry point, an
unbounded run, missing guardrails or monitoring. Reason about the code; do not
merely match tokens.

TAXONOMY — every finding maps to one of these ASI subcategories. Use the exact
subcategory key (e.g. `indirect_prompt_injection`) as the `subcategory` argument:
{taxonomy_ref}

PROCEDURE:

1. Read every file. Call read_source(resource_type="file", path=<path>) for
   EVERY file in the AGENT PROFILES (and any repo-local file they import that is
   relevant). Use read_source(resource_type="agent", path=<agent_id>) for the
   full agent profile when useful.

2. Understand what the static layer already found. Call get_findings() on each
   file first, so you do not re-report a finding the static analyzers already
   surfaced (Bandit/OpenGrep/detect-secrets). Your job is the reasoning layer:
   the behavioral and cross-cutting issues static rules miss.

3. Investigate against the taxonomy. For each agent and tool, reason through the
   ASI categories and ask the kinds of questions below. Use
   analyze_code(mode="search") to trace data flow (e.g. where a request body,
   query param, or tool argument is used). Investigate — do not assume clean.

4. Report each finding you can evidence. Call compute_cvss(vuln_type=<subcategory>)
   then report_finding() for it. One call per finding; do not batch.

5. When the investigation is complete, output ONLY:
   {{"summary": "N findings reported", "investigation_notes": "<brief notes>"}}

WHAT TO REASON ABOUT (illustrative, not exhaustive — apply judgment, and use any
taxonomy subcategory that fits, not only these):

- ASI01 goal hijack: Does untrusted input (an HTTP body, request param, message,
  or file) reach the agent's run/invocation or its prompt? That is
  direct_prompt_injection even when the input is passed as a run argument rather
  than concatenated into a template (e.g. `Runner.run(agent, req.input)`). Is
  tool OUTPUT (search results, error strings that echo user input, retrieved
  documents) fed back into the model without sanitization? That is
  indirect_prompt_injection / goal_manipulation_via_rag.
- ASI02 tool misuse: Do tools validate their inputs? Can a tool touch the
  filesystem, network, or shell beyond what its purpose requires
  (excessive_tool_permissions, unvalidated_tool_input, path_traversal)?
- ASI03 identity/privilege: Is the agent exposed through an endpoint with no
  authentication (missing_auth_on_endpoint)? Hardcoded credentials? An agent
  running with more privilege than it needs?
- ASI05 code execution: eval/exec/compile, shell=True/os.system, unsafe
  deserialization, or model output used to generate/execute code.
- ASI06 memory/context: Shared mutable state across sessions, unsanitized
  persistent memory, RAG poisoning.
- ASI08 cascading failures: Is the agent run without a turn/iteration bound
  (`Runner.run`/loop with no max_turns/max_iterations/recursion_limit →
  unbounded_agent_loop)? Missing circuit breaker or blast-radius limit?
- ASI09 human/agent trust: Do state-changing or destructive actions (returns,
  refunds, writes, deletes, purchases) proceed with no human confirmation or
  human-in-the-loop (missing_action_confirmation, no_human_in_the_loop)? Is
  sensitive data returned in output?
- ASI10 rogue agents: Missing behavioral guardrails, unchecked delegation to
  sub-agents, no monitoring/kill switch.

REPORTING RULES:
- Call compute_cvss(vuln_type=<subcategory>) before EVERY report_finding().
- `subcategory` MUST be copied EXACTLY from the taxonomy keys above (e.g.
  `indirect_prompt_injection`). Do not reword, expand or abbreviate it. There is
  no `category` argument — the ASI category is derived from the subcategory.
- Set evidence to the EXACT code (the offending line or minimal snippet), not a
  description. Set evidence_file to the file path and evidence_line to the line.
- Set agent_name to the agent display name from AGENT PROFILES.
- Set confidence to how strongly the code evidences the issue: "high" when the
  code plainly shows it, "medium"/"low" when it depends on runtime context or
  configuration you cannot fully see. Set hallucination_flag=true when you infer
  an issue you cannot fully prove from the code in front of you.
- Do NOT report the same issue twice, and do NOT re-report a finding already
  surfaced by static analysis (you checked via get_findings()).
- Report every distinct, evidenced issue — do not stop at one per category, and
  do not withhold a real finding because it is behavioral rather than syntactic.

EXAMPLE:
  agent.py line 180: `result = await Runner.run(agent, req.input)` — the FastAPI
  request body `req.input` is passed straight to the agent with no validation or
  turn bound.
  Step 1: compute_cvss(vuln_type="direct_prompt_injection", exposed_over_network=true)
  Step 2: report_finding(
    subcategory="direct_prompt_injection",
    title="Untrusted request body passed directly to the agent",
    description="The /run endpoint feeds req.input into Runner.run(agent, ...) "
                "with no sanitization, so an attacker controls the agent's input.",
    impact="An attacker can hijack the agent's goal or exfiltrate data via the "
           "HTTP endpoint.",
    remediation="Validate and constrain user input; apply input guardrails "
                "before invoking the agent.",
    affected_component="agent.py",
    evidence="result = await Runner.run(agent, req.input)",
    confidence="high", hallucination_flag=false,
    evidence_file="agent.py", evidence_line=180,
    agent_name="bookstore_agent")

LIMITS: {MAX_ITERATIONS} rounds, {MAX_FILES_FETCHED} files, {MAX_FETCH_TOKENS} tokens.
"""


# ── Initial investigation brief ───────────────────────────────────────────────


def _build_initial_context(
    agents: list[AgentProfile],
    python_files: list[RepoFile],
    static_count: int,
) -> str:
    parts = ["=== INVESTIGATION BRIEF ===\n"]
    parts.append(f"Repository contains {len(python_files)} Python file(s).")
    parts.append(
        f"Pre-computed static scan found {static_count} findings "
        f"(available via get_static_findings_for_file).\n"
    )

    parts.append(f"=== AGENT PROFILES SUMMARY ({len(agents)} agent(s)) ===")
    sorted_agents = sorted(agents, key=lambda a: a.agent_id)
    for a in sorted_agents[:15]:
        graph_info = (
            f" | graph_nodes={a.graph_nodes}" if getattr(a, "graph_nodes", None) else ""
        )
        bounded = (
            f" | bounded={a.has_recursion_limit}"
            if getattr(a, "has_recursion_limit", None) is not None
            else ""
        )
        parts.append(
            f"\nAgent: {a.role or a.agent_id} | ID: {a.agent_id}"
            f" | Framework: {a.framework} | File: {a.source_file}"
            f"{graph_info}{bounded}\n"
            f"  Tools: {a.tools or []}\n"
            f"  LLM: {a.llm or '?'} | Memory: {a.memory} | Delegation: {a.allow_delegation}\n"
            f"  Code preview (use get_agent_profile for full content):\n"
            f"  {anonymize_secrets(a.raw_code[:300].strip())}"
        )

    parts.append("\n=== AVAILABLE FILES ===")
    parts.append("Use read_source(resource_type='list') to see all fetched files.")
    parts.append("Use read_source(resource_type='file', path=...) to read any file.")
    parts.append(
        "\n\nBegin the PROCEDURE now: fetch the full source code for each agent "
        "file listed above, review the existing static findings, then investigate "
        "against the taxonomy and report every issue you can evidence."
    )
    return "\n".join(parts)


# ── ADK agent builder ─────────────────────────────────────────────────────────


def _build_adk_agent(adk_model: Any, adk_tools: list[Any]) -> Any:
    """Build and return a Google ADK LlmAgent instance."""
    return LlmAgent(
        name="agent_security_scanner",
        model=adk_model,
        description=(
            "AI agent security researcher that scans codebases for "
            "OWASP ASI01-ASI10 vulnerabilities."
        ),
        instruction=_build_system_prompt(),
        tools=adk_tools,
    )


# ── Session service factory ───────────────────────────────────────────────────


def _create_session_service() -> Any:
    """Create and return an in-memory ADK SessionService."""
    return InMemorySessionService()


# ── Async ADK runner ──────────────────────────────────────────────────────────


async def _run_adk_async(
    agent,
    initial_message: str,
    dispatcher: ToolDispatcher,
    session_id: str,
    tracer,
) -> tuple[str, str]:
    """
    Run the ADK agent via InMemorySessionService-backed Runner.

    Returns (final_text_response, exit_reason).
    Findings are accumulated via dispatcher.session_findings as side-effects
    of report_finding() tool calls made by the agent.
    """
    APP_NAME = "agent_security_scanner"
    USER_ID = "scanner"

    session_service = _create_session_service()
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )
    logger.info("Session created: %s", session.id)

    user_content = genai_types.Content(
        role="user",
        parts=[genai_types.Part.from_text(text=initial_message)],
    )

    final_text = ""
    exit_reason = "completed"
    iteration = 0
    loop_start = time.monotonic()

    event_stream = runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=user_content,
    )
    try:
        async for event in event_stream:
            # ── Control plane guards ──────────────────────────────────────

            elapsed = time.monotonic() - loop_start
            if elapsed > LOOP_TIMEOUT_SECS:
                logger.warning(
                    "Timeout after %.1fs (limit=%ds)", elapsed, LOOP_TIMEOUT_SECS
                )
                exit_reason = "timeout"
                break

            if hasattr(event, "actions") and event.actions:
                iteration += 1
                tracer.set_iteration(iteration)
                logger.debug("Iteration %d/%d", iteration, MAX_ITERATIONS)
                if iteration >= MAX_ITERATIONS:
                    logger.warning("Max iterations (%d) reached", MAX_ITERATIONS)
                    exit_reason = "max_iterations"
                    break

            if dispatcher._call_counts.get("fetch_file", 0) > MAX_FILES_FETCHED:
                logger.warning("File fetch limit (%d) reached", MAX_FILES_FETCHED)
                exit_reason = "max_files"
                break

            if dispatcher.tokens_consumed > MAX_FETCH_TOKENS:
                logger.warning(
                    "Token budget exhausted (%d/%d)",
                    dispatcher.tokens_consumed,
                    MAX_FETCH_TOKENS,
                )
                exit_reason = "max_tokens"
                break

            if dispatcher._call_counts.get("run_targeted_bandit", 0) > 5:
                logger.warning("Bandit call limit (5) reached")
                exit_reason = "max_bandit_calls"
                break

            if dispatcher._call_counts.get("compute_cvss", 0) > 50:
                logger.warning("compute_cvss call limit (50) reached")
                exit_reason = "max_cvss_calls"
                break

            if hasattr(event, "content") and event.content:
                for part in event.content.parts or []:
                    if hasattr(part, "text") and part.text:
                        final_text += part.text
    finally:
        # ADK's Runner wraps this generator's `yield` in an OTEL span
        # (google/adk/runners.py::_run_with_trace). Every `break` above
        # leaves it suspended mid-span rather than exhausted. Left alone, the
        # event loop's implicit shutdown_asyncgens() closes it later — in a
        # different contextvars.Context than the one that attached the span,
        # which makes the span's detach raise "Token ... was created in a
        # different Context". Closing it here, in this same coroutine, closes
        # the span in the Context that opened it.
        with suppress(Exception):
            await event_stream.aclose()

    return final_text, exit_reason


def _classify_reasoning_failure(exc: Exception, domain: FailureDomain) -> str | None:
    """Which bounded `error_reason` the trace carries out for `exc`.

    The eight predicates are an allowlist of *specific*, customer-actionable
    causes, checked most-specific first: entitlement ahead of auth (a 403 is
    a plan problem, not a wrong key), auth ahead of model (a 401 is never a
    bad model id), and the four retryable shapes after the three config ones.

    `domain` is the floor under them. Every predicate can decline — most
    need a status, and `is_upstream_error` deliberately will not sniff
    messages for one — so without a terminal branch a failure none of them
    named carried no reason at all, `scan_cli._check_llm_credentials` found
    nothing to look up, and `switch-trust scan` exited 0 on a report whose
    entire AI reasoning layer was missing. That is the same false green the
    named reasons exist to close, reached from the one direction none of
    them covers, and it is not exotic: a statusless 5xx gets there whenever
    litellm wraps a Gemini error without a `.response` for `http_status_of`
    to read.

    `provider` is the half worth claiming, and `classify_exception` reaches
    it for exactly those statusless SDK failures, by module prefix. It takes
    "upstream" rather than a reason of its own because the posture is what
    the caller acts on — a third party across a network boundary failed, so
    retry — and exit 63 already says that. The remaining domains fall
    through: a `user` failure no predicate recognised, and an `internal` or
    `unknown` one, still return `None`.
    """
    if is_not_entitled(exc):
        # Valid credential, insufficient entitlement — distinct from a
        # rejected key (401), and a different fix (plan/quota, not the
        # token). Checked ahead of `is_auth_error`, which still claims it
        # for the console's coarser "key invalid" diagnostic.
        return "not_entitled"
    if is_auth_error(exc):
        return "auth"
    if is_model_error(exc):
        return "model"
    if is_timeout_error(exc):
        return "timeout"
    if is_upstream_error(exc):
        return "upstream"
    if is_resource_exhausted(exc):
        return "resource_exhausted"
    if is_rate_limited(exc):
        return "rate_limited"
    if is_network_error(exc):
        return "network"
    if domain is FailureDomain.PROVIDER:
        return "upstream"
    return None


# ── Main entry point ──────────────────────────────────────────────────────────


def run_agentic_reasoning(
    agents: list[AgentProfile],
    python_files: list[RepoFile],
    static_findings: list[StaticFinding],
    output_path: str | None = None,
) -> tuple[list[RawFinding], str, dict[str, Any]]:
    """
    Main entry point for v2 agentic reasoning using Google ADK.

    The ADK model is resolved via llm_provider.make_model() using the
    SCANNER_MODEL env var.

    Session storage uses InMemorySessionService (ephemeral, per-scan).

    Returns:
        (findings_list, summary_string, trace_metadata_dict)
    """
    if not agents and not python_files:
        logger.info("No agents or files found — skipping agentic scan")
        return [], "No agent profiles or files found to analyze.", {}

    # Validate that each agent has the three fields the dispatcher and investigation
    # brief depend on. Check each field value individually — a field is "missing"
    # only when its value is None or an empty string, not when its string value
    # differs from the field name (which was the original bug: the old code did
    # `required_fields - set(values)` which always returned the full set because
    # "researcher" != "agent_id", etc.).
    _REQUIRED = ("agent_id", "framework", "source_file")
    invalid_agents = []
    for agent in agents:
        missing = [f for f in _REQUIRED if not getattr(agent, f, None)]
        if missing:
            logger.warning(
                "[reasoner] Agent missing required fields: %s — skipping", missing
            )
            invalid_agents.append(agent)

    if invalid_agents:
        agents = [a for a in agents if a not in invalid_agents]
        logger.warning(
            "[reasoner] Excluded %d agents with missing required fields. "
            "Proceeding with %d valid agents.",
            len(invalid_agents),
            len(agents),
        )

    if not agents and not python_files:
        logger.info("No valid agents or files after validation — skipping agentic scan")
        return [], "No valid agent profiles or files found to analyze.", {}

    try:
        adk_model = make_model(scanner="agent", phase="reasoner")
        logger.info("ADK model: %s", adk_model)
    except (ImportError, ValueError) as e:
        logger.error(
            "ADK setup failed — cannot proceed without ADK (%s)", type(e).__name__
        )
        return [], f"ADK setup failed — {e}", {"error": str(e)}

    # Set up dispatcher and tracer.
    # IMPORTANT: tracer must be created before build_adk_tools() is called so
    # the tool closures can close over it. Previously the order was reversed,
    # which meant the tracer was never passed in and the trace file showed
    # calls=0 despite the agent making dozens of tool calls.
    repo_files_by_path = {rf.path: rf for rf in python_files}
    dispatcher = ToolDispatcher(
        repo_files=repo_files_by_path,
        agents=agents,
        static_findings=static_findings,
    )

    session_id = f"AGT-{uuid.uuid4().hex[:8].upper()}"
    tracer = (
        FileTraceLogger(session_id=session_id, output_path=output_path)
        if output_path
        else LogTraceLogger()
    )
    tracer.start(provider_model=str(adk_model))

    # Build ADK tools with the tracer now wired in — every tool call will be
    # recorded in the JSONL trace file and counted in dispatcher._call_counts.
    adk_tools = dispatcher.get_adk_tools(tracer=tracer)

    # Build ADK agent
    try:
        adk_agent = _build_adk_agent(adk_model, adk_tools)
    except ImportError as e:
        logger.error("ADK agent build failed (%s)", type(e).__name__)
        return [], f"ADK agent build failed — {e}", {"error": str(e)}

    # Build initial investigation brief
    initial_ctx = _build_initial_context(agents, python_files, len(static_findings))

    logger.info(
        "Starting ADK scan | session: %s | agents: %d | files: %d",
        session_id,
        len(agents),
        len(python_files),
    )

    # Execute ADK reasoning loop.
    #
    # Always run the coroutine in a dedicated worker thread with a hard
    # deadline. asyncio.run() works in that thread whether or not the caller
    # already has a running event loop (Jupyter/FastAPI as well as the plain
    # CLI path), and future.result(timeout=...) bounds the whole loop even
    # when the model call never returns a control-plane event — the in-loop
    # guard in _run_adk_async only fires between events, so a stalled LLM call
    # would otherwise hang forever on the CLI path. On timeout we shut the
    # pool down without waiting so we don't block on the stuck worker thread.
    exit_reason = "completed"
    final_text = ""
    # Classified where the exception still exists, then carried out in the trace
    # (`error_domain`, and `error_reason` — "not_entitled"/"auth"/"model"/
    # "timeout"/"upstream"/"resource_exhausted"/"rate_limited"/"network" — for
    # the specific customer-actionable config failures; see
    # `_classify_reasoning_failure` for why a provider failure none of those
    # name still gets one). The exception itself is discarded here for
    # telemetry safety, so the caller cannot re-classify it — see
    # `_agent_scan_summary`.
    error_domain: FailureDomain | None = None
    error_reason: str | None = None

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(
            asyncio.run,
            _run_adk_async(adk_agent, initial_ctx, dispatcher, session_id, tracer),
        )
        final_text, exit_reason = future.result(timeout=LOOP_TIMEOUT_SECS + 30)
    except concurrent.futures.TimeoutError:
        logger.error("ADK execution timed out after %ds", LOOP_TIMEOUT_SECS + 30)
        exit_reason = "timeout"
    except Exception as e:
        err_type = type(e).__name__
        safe_msg = _safe_error(e)
        logger.error("ADK execution failed (%s)", err_type)
        logger.debug("ADK execution failed (full):\n%s", safe_msg, exc_info=True)
        exit_reason = "error"
        error_domain = classify_exception(e)
        error_reason = _classify_reasoning_failure(e, error_domain)
    finally:
        pool.shutdown(wait=False)

    # Collect results
    findings = dispatcher.session_findings
    summary = "Agentic investigation complete."

    if final_text:
        clean = final_text.strip()
        if clean.startswith("```"):
            lines = clean.splitlines()
            clean = "\n".join(lines[1:]).rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(clean)
            if isinstance(parsed, dict):
                summary = parsed.get("summary", summary)
        except (json.JSONDecodeError, TypeError):
            if len(final_text.strip()) > 20:
                summary = final_text.strip()

    tracer.finish(findings_count=len(findings), exit_reason=exit_reason)
    trace_meta = tracer.as_dict()
    trace_meta.update(
        {
            "exit_reason": exit_reason,
            "adk_model": str(adk_model),
            "session_id": session_id,
            "tokens_consumed": dispatcher.tokens_consumed,
        }
    )
    if error_domain is not None:
        # Bounded enum value, safe to carry out (unlike str(exc)); the stage
        # summary reads it back to attribute the failure.
        trace_meta["error_domain"] = error_domain.value
    if error_reason is not None:
        # "not_entitled"/"auth"/"model"/"timeout"/"upstream"/
        # "resource_exhausted"/"rate_limited"/"network" — lets the caller
        # surface a precise "not permitted" / "API key invalid" / "model
        # invalid" / "timed out" / "backend failed" / "rate limited (429)" /
        # "rate limited" / "could not reach provider" line to the customer.
        # Bounded, like error_domain.
        trace_meta["error_reason"] = error_reason

    logger.info(
        "ADK scan complete — findings: %d | exit: %s | session: %s",
        len(findings),
        exit_reason,
        session_id,
    )

    return findings, summary, trace_meta
