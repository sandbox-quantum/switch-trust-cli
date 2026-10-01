"""Subcommand for `switch-trust init`."""

from __future__ import annotations

import argparse
import json
import os
import stat

from rich.prompt import Prompt

from switch_trust.cli.console import console, select
from switch_trust.cli.utils import (
    CLIENT_ID_ENV_VAR,
    TELEMETRY_CONSENT_ENV_VAR,
    generate_client_id,
    get_switch_trust_config_path,
    get_switch_trust_dir,
    get_switch_trust_env_path,
    prompt_telemetry_consent,
)

_PROVIDER_DEFAULTS: dict[str, dict[str, str | None]] = {
    "gemini": {
        "model": "gemini-3.8-flash",
        "api_key_var": "GEMINI_API_KEY",
    },
    "openai": {
        "model": "gpt-5.6-sol",
        "api_key_var": "OPENAI_API_KEY",
    },
    "anthropic": {
        "model": "claude-sonnet-5",
        "api_key_var": "ANTHROPIC_API_KEY",
    },
    "litellm": {
        "model": "gpt-5.6-sol",
        "api_key_var": None,
    },
}

_LITELLM_BACKENDS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def register(
    subparsers: argparse._SubParsersAction,
) -> None:
    subparsers.add_parser(
        "init",
        help="Initialize switch-trust configuration",
    )


def run_init() -> None:
    switch_trust_dir = get_switch_trust_dir()

    console.print("[bold cyan]Switch Trust Setup[/bold cyan]")
    console.print()

    if switch_trust_dir.exists():
        proceed = select(
            "Existing configuration will be overwritten. Do you want to proceed?",
            options=["yes", "no"],
            default_index=0,
        )
        if proceed.lower() == "no":
            console.print("[dim]Aborted.[/dim]")
            return
        console.print()

    provider = select(
        "Switch Trust CLI needs an LLM for certain functionality. Select your model provider:",
        options=["gemini", "openai", "anthropic", "litellm"],
        default_index=0,
    )
    console.print()

    defaults = _PROVIDER_DEFAULTS[provider]

    if provider == "litellm":
        model_name = Prompt.ask(
            "LiteLLM model string (e.g. openai/gpt-5.4, gemini/gemini-2.5-flash)",
            default="openai/gpt-5.4",
        )
        console.print()

        backend = model_name.split("/")[0] if "/" in model_name else None
        api_key_var = _LITELLM_BACKENDS.get(backend, None) if backend else None
        if not api_key_var:
            api_key_var = Prompt.ask("API key env var name")
            console.print()

        generator_model = f"litellm:{model_name}"
    else:
        model_name = Prompt.ask(
            "Model name",
            default=defaults["model"],
        )
        console.print()
        api_key_var = defaults["api_key_var"]
        generator_model = f"{provider}:{model_name}"

    assert api_key_var is not None
    api_key = Prompt.ask(
        f"Enter your {api_key_var}",
        password=True,
    )
    console.print()

    telemetry_consent = prompt_telemetry_consent()

    client_id = generate_client_id()
    env_lines = [
        f"GENERATOR_MODEL={generator_model}",
        f"{api_key_var}={api_key}",
        "EXECUTOR_MAX_WORKERS=20",
        f"{CLIENT_ID_ENV_VAR}={client_id}",
        f"{TELEMETRY_CONSENT_ENV_VAR}={telemetry_consent}",
    ]

    default_config = {
        "models": [],
        "evaluations": [],
        "detectors": [],
        "message_collections": [],
        "model_evaluations": [],
    }

    old_umask = os.umask(0o077)
    try:
        switch_trust_dir.mkdir(parents=True, exist_ok=True)

        env_path = get_switch_trust_env_path()
        env_path.write_text("\n".join(env_lines) + "\n")
        env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

        config_path = get_switch_trust_config_path()
        config_path.write_text(
            json.dumps(default_config, indent=2) + "\n",
        )
    finally:
        os.umask(old_umask)

    console.print(f"[green]Created {env_path}[/green]")
    console.print(
        f"[green]Created {config_path}[/green]",
    )
    console.print()
    console.print(
        "[bold green]Switch Trust initialized successfully![/bold green]",
    )
    console.print()
