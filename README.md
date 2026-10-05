> [!CAUTION]
> **Deprecation note**: `flintai-cli` is no longer maintained and has been superseded by [`switch-trust-cli`](https://github.com/sandbox-quantum/switch-trust-cli).\
> Install: `pip install switch-trust-cli`.\
> Usage: Replace `flintai <command>` with `switch-trust <command>`.

<br/>
<br/>
<br/>

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/sandbox-quantum/flintai-cli/main/images/flint-ai-wordmark-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/sandbox-quantum/flintai-cli/main/images/flint-ai-wordmark.svg">
  <img src="https://raw.githubusercontent.com/sandbox-quantum/flintai-cli/main/images/flint-ai-wordmark.svg" alt="Flint AI" width="350">
</picture>

[![PyPI version](https://img.shields.io/pypi/v/flintai-cli?color=A1C9D2&logo=pypi&logoColor=white)](https://pypi.org/project/flintai-cli/) [![Python](https://img.shields.io/badge/python-3.11+-A1C9D2?logo=python&logoColor=white)](https://www.python.org/downloads/) [![Documentation](https://img.shields.io/badge/docs-docs.flintai.dev-FF895E)](https://docs.flintai.dev) [![Website](https://img.shields.io/badge/website-flintai.dev-FF895E)](https://flintai.dev)

</div>

**Ship AI agents with confidence**

One CLI to analyze agent code and runtime behavior, any framework.

| | **Flint AI Scan** | **Flint AI Eval** |
|---|---|---|
| **Command** | `flintai scan` | `flintai eval` |
| **What** | AI-powered security analysis of your agent's code (whitebox testing) | Runtime behavioral evaluation with adversarial prompts (blackbox testing) |
| **Output** | Security findings mapped to OWASP top 10 with CVSS severity scores  | Evaluation scores (0-100%) mapped to OWASP top 10  |

**Why Flint AI?**
- **AI-powered analysis** — Contextual code understanding, not just pattern matching
- **OWASP ASI mapped** — Findings aligned to Top 10 for Agentic Applications
- **100% free** — First results in minutes


## Try it now - 5 minute Quickstart

> **Requirements**
> - Python 3.11 or later
> - [OpenGrep](https://github.com/opengrep/opengrep#linux--macos) (required for Flint AI Scan)
> - A running agent accessible via HTTP (required for Flint AI Eval)
>
> **Supported frameworks:** Google ADK, Google GenAI, Anthropic, OpenAI, OpenAI Agents SDK, LangGraph, CrewAI, AutoGen, HuggingFace Transformers, HuggingFace smolagents

### Step 1: Install Flint AI

Using a virtual environment is recommended to avoid dependency conflicts:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Flint AI CLI:
```bash
pip install flintai-cli
```

<details>
<summary>Optional <code>full</code> extra for toxicity, local & garak evaluations</summary>

Some evaluations rely on heavy ML backends that are kept out of the default
install: toxicity detection, local evaluation of models retrieved from Huggingface,
and garak probes and detectors. Install them with the optional `full`
extra when you need those features:
```bash
pip install "flintai-cli[full]"
```
</details>

### Step 2: Configure your LLM provider

Flint AI uses AI to analyze agent code and score reliability. Run the interactive setup:

```bash
flintai init
```

You'll be prompted to select a provider (Gemini, OpenAI, Anthropic, or LiteLLM), choose a model, and enter your API key.

<details>
<summary>Where to get API keys</summary>

- **Google Gemini**: [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (free tier available)
- **OpenAI**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Anthropic**: [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
- **LiteLLM**: Supports 100+ providers. See [docs.litellm.ai](https://docs.litellm.ai/docs/)

</details>

> Run into issues? See [installation troubleshooting](https://docs.flintai.dev/flintai/cli/troubleshooting/common-issues#installation)

### Step 3: Try the example agents

To demonstrate the CLIs capabilities, we've shipped this tool with two example agents. You can get them [here](https://github.com/sandbox-quantum/flintai-cli/tree/main/examples).

**Both agents work with both `flintai scan` and `flintai eval`**:

| Agent | Framework | Description |
|-------|-----------|-------------|
| **weather_agent** | Google ADK | Weather assistant that looks up conditions for cities. Should refuse off-topic requests. |
| **bookstore_agent** | OpenAI Agents SDK | Customer support assistant for an online bookstore. Searches books, checks orders, and processes returns. |

The included `examples/config.json` has both agents configured with builtin evaluations (OWASP LLM01–LLM09, PII, secrets) and custom tests.

---

`flintai scan` finds security issues in the code without running the agent. We'll scan the bookstore agent to see what issues Flint AI can find:
```bash
flintai scan examples/bookstore_agent/
```

<img src="https://raw.githubusercontent.com/sandbox-quantum/flintai-cli/main/images/scan-findings.png" alt="Scan results showing security findings" width="550">

*Example: Scan found 2 security issues - High severity missing authentication and Medium severity unbounded execution loop*

---

`flintai eval` tests runtime behavior, so the agent needs to be running. Start the bookstore agent:

```bash
# Start the bookstore agent (serves on http://localhost:8010)
uvx --with openai-agents,fastapi --from uvicorn uvicorn examples.bookstore_agent.agent:app --port 8010 --host 0.0.0.0
```

In a new terminal, run evaluations:

```bash
flintai eval run --model model-bookstore-agent --config examples/config.json
```

### Step 4: Test your own agents

See our documentation to configure, scan and evaluate your agents:
- `flintai scan`
  - [Scan your own agent](https://docs.flintai.dev/flintai/cli/scan/getting-started) — Apply Flint AI Scan to your codebase
  - [Understand scan results](https://docs.flintai.dev/flintai/cli/scan/scan-results) — Interpret findings and severity scores
- `flintai eval`
  - [Evaluate your own agent](https://docs.flintai.dev/flintai/cli/eval/getting-started) — Configure and test your agent's behavior
  - [Configuration](https://docs.flintai.dev/flintai/cli/eval/eval-configuration) — In-depth documentation of our configuration
  - [Understand eval results](https://docs.flintai.dev/flintai/cli/eval/eval-results) — What the scores means and how to improve

**Ship with confidence.** Validate behavior, catch risks, prove readiness.


## Commands

### `init`

Setup wizard that configures Flint AI for first use. Creates the `~/.flintai` directory with a `.env` file (LLM provider, API key, runtime settings) and a `config.json` skeleton.

Runs automatically on first use in non-CI environments. You can re-run it at any time to reconfigure.

```bash
flintai init
```

### `scan`

AI-powered security analysis of agent source code. Finds vulnerabilities, misconfigurations, and OWASP Top 10 violations.

```bash
# Scan a directory
flintai scan /path/to/agent/code

# Scan a single file
flintai scan agent.py

# Specify output file
flintai scan /path/to/code --output results.json
```

[Full scan guide](https://docs.flintai.dev/flintai/cli/scan/getting-started)

### `eval`

Test agent behavior at runtime. Get a evaluation score proving production-readiness.

```bash
# List all available configuration
flintai eval evaluations list

# List your agents and models
flintai eval models list

# Attach an evalation to your agent
flintai eval model-evaluations attach \
  --model my-agent \
  --eval eval-llm01-adversarial

# Run all evaluations for an agent
flintai eval run --model my-agent
```

The `flintai eval` command requires configuration. See [Configuration](https://docs.flintai.dev/flintai/cli/eval/eval-configuration) to:
1. Define your models (agents to test)
2. View available evaluations
3. Attach evaluations to models

[Full eval guide](https://docs.flintai.dev/flintai/cli/eval/getting-started)

## Documentation

**Complete guides and reference:**
- [Getting started](https://docs.flintai.dev)
- [Command reference](https://docs.flintai.dev/flintai/cli/reference/commands)
- [Configuration](https://docs.flintai.dev/flintai/cli/eval/eval-configuration)
- [Environment variables](https://docs.flintai.dev/flintai/cli/reference/env-vars)
- [Built-in evaluations](https://docs.flintai.dev/flintai/cli/reference/builtin-evaluations)
- [Data privacy](https://docs.flintai.dev/flintai/cli/reference/data-privacy)
- [FAQ](https://docs.flintai.dev/flintai/cli/resources/faq)

## Data privacy

Flint AI runs on your machine, but several features can call external LLM providers. This can be configured via `GENERATOR_MODEL`
(located in `~/.flintai/.env`, created by `flintai init`). You can set this to a remote managed LLM (i.e. `gemini`, `openai`, `anthropic`)
or a locally hosted LLM (i.e. `litellm` or `ollama`).

[Read more](https://docs.flintai.dev/flintai/cli/reference/data-privacy).

### Telemetry

Flint AI CLI collects **anonymous usage analytics** to help us understand how the tool is used and improve it. Telemetry is **opt-in** — you are asked for consent during `flintai init`, and no data is sent without your explicit agreement.

**What we collect:**

| Data | Example | Purpose |
|------|---------|---------|
| Command name | `scan`, `eval run` | Understand which features are used |
| CLI version | `1.1.1` | Track adoption of new releases |
| Execution duration | `12.3s` | Identify performance issues |
| Error type (on crash) | `ConnectionError` | Prioritize bug fixes |
| CI environment flag | `true` / `false` | Understand CI vs interactive usage |
| Anonymous client ID | `a1b2c3d4-...` (random UUID) | Count unique installations |

**What we never collect:** source code, file paths, prompts, API keys, model outputs, usernames, hostnames, IP addresses, or any personally identifiable information.

**Opting out:** Set `FLINTAI_TELEMETRY_CONSENT=false` in `~/.flintai/.env`, or select "N" when prompted during `flintai init`. You can change this at any time.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Free to use - [full license](LICENSE).

## Contact

- Website: [https://flintai.dev](https://flintai.dev)
- Email: [info@flintai.dev](mailto:info@flintai.dev)
