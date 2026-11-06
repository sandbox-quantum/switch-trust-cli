"""Subcommand for `switch-trust init`."""

import logging
import os
import stat
import sys
import uuid
from pathlib import Path

from switch_trust.cli.console import console, select

logger = logging.getLogger(__name__)

CLIENT_ID_ENV_VAR = "SWITCH_TRUST_CLIENT_ID"
TELEMETRY_CONSENT_ENV_VAR = "SWITCH_TRUST_TELEMETRY_CONSENT"


def get_switch_trust_dir() -> Path:
    return Path.home() / ".switch-trust"


def get_switch_trust_env_path() -> Path:
    """Return the ``.env`` to use, preferring a project-local one.

    If the current working directory contains a ``.env`` it takes precedence;
    otherwise fall back to the global ``~/.switch-trust/.env``.
    """
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        return cwd_env
    return get_switch_trust_dir() / ".env"


def get_switch_trust_config_path() -> Path:
    return get_switch_trust_dir() / "config.json"


_CI_ENV_VARS = (
    "CI",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "CIRCLECI",
    "JENKINS_URL",
    "TRAVIS",
    "BUILDKITE",
    "CODEBUILD_BUILD_ID",
    "TF_BUILD",
    "BITBUCKET_PIPELINE",
    "TEAMCITY_VERSION",
)


def is_ci() -> bool:
    return any(os.environ.get(v) for v in _CI_ENV_VARS)


def get_client_id() -> str | None:
    return os.environ.get(CLIENT_ID_ENV_VAR)


def generate_client_id() -> str:
    return str(uuid.uuid4())


def ensure_client_id() -> None:
    if os.environ.get(CLIENT_ID_ENV_VAR):
        return

    client_id = generate_client_id()
    os.environ[CLIENT_ID_ENV_VAR] = client_id

    env_path = get_switch_trust_env_path()
    if env_path.exists():
        with open(env_path, "a") as f:
            f.write(f"{CLIENT_ID_ENV_VAR}={client_id}\n")
        env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        return

    logger.warning("dot env file missing, client id not persisted")


def get_telemetry_consent() -> bool:
    return (os.environ.get(TELEMETRY_CONSENT_ENV_VAR) or "").lower() == "true"


def stdin_is_interactive() -> bool:
    """True only for a terminal: piped, closed or missing stdin can't answer a prompt."""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except ValueError:
        return False


def prompt_telemetry_consent() -> str | None:
    """Ask for analytics consent, or return ``None`` without asking.

    Only a terminal can answer: piped or closed stdin would otherwise accept the
    default or crash, so callers must treat ``None`` as "no consent, nothing to
    persist" and let a later interactive run ask.
    """
    if not stdin_is_interactive():
        return None

    console.print(
        "[dim]Switch Trust can collect pseudonymous usage analytics to improve the product.[/dim]"
    )
    console.print("[dim]No code, prompts, or keys are ever collected.[/dim]")
    console.print("[dim]You can change this at any time in your .env config.[/dim]")
    console.print()
    choice = select(
        "Do you consent to sharing pseudonymous usage analytics?",
        options=["Y", "N"],
        default_index=0,
    )
    console.print()

    return "true" if choice.lower() == "y" else "false"


def ensure_telemetry_consent() -> None:
    if os.environ.get(TELEMETRY_CONSENT_ENV_VAR):
        return

    if is_ci():
        os.environ[TELEMETRY_CONSENT_ENV_VAR] = "false"
        return

    value = prompt_telemetry_consent()
    if value is None:
        logger.info("stdin is not a terminal; telemetry off for this run")
        os.environ[TELEMETRY_CONSENT_ENV_VAR] = "false"
        return
    os.environ[TELEMETRY_CONSENT_ENV_VAR] = value

    env_path = get_switch_trust_env_path()
    if env_path.exists():
        with open(env_path, "a") as f:
            f.write(f"{TELEMETRY_CONSENT_ENV_VAR}={value}\n")
        env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
