"""Subcommand for `switch-trust init`."""

import logging
import os
import stat
import uuid
from pathlib import Path

from switch_trust.cli.console import console, select

logger = logging.getLogger(__name__)

CLIENT_ID_ENV_VAR = "SWITCH_TRUST_CLIENT_ID"
TELEMETRY_CONSENT_ENV_VAR = "SWITCH_TRUST_TELEMETRY_CONSENT"

# Pre-rebrand env var names, still honored as a fallback so existing installs
# keep working. The current name wins; the legacy name is only read when the
# current one is unset, and its use prints a one-time deprecation note.
_LEGACY_ENV_VARS = {
    CLIENT_ID_ENV_VAR: "FLINTAI_CLIENT_ID",
    TELEMETRY_CONSENT_ENV_VAR: "FLINTAI_TELEMETRY_CONSENT",
}

_deprecation_notices_shown: set[str] = set()


def resolve_env_var(name: str) -> str | None:
    """Return ``name`` from the environment, falling back to its legacy alias.

    The current name takes precedence. When it is unset but the deprecated alias
    is present, the alias value is used and a one-time note (per alias) is
    printed pointing at the replacement.
    """
    value = os.environ.get(name)
    if value is not None:
        return value

    legacy = _LEGACY_ENV_VARS.get(name)
    if legacy is None:
        return None

    legacy_value = os.environ.get(legacy)
    if legacy_value is not None and legacy not in _deprecation_notices_shown:
        _deprecation_notices_shown.add(legacy)
        console.print(
            f"[yellow]{legacy} is deprecated; please use {name} instead.[/yellow]"
        )
    return legacy_value


def legacy_env_vars_used() -> bool:
    """True if any deprecated env var was resolved this run (a note was shown)."""
    return bool(_deprecation_notices_shown)


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
    return resolve_env_var(CLIENT_ID_ENV_VAR)


def generate_client_id() -> str:
    return str(uuid.uuid4())


def ensure_client_id() -> None:
    if resolve_env_var(CLIENT_ID_ENV_VAR):
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
    return (resolve_env_var(TELEMETRY_CONSENT_ENV_VAR) or "").lower() == "true"


def prompt_telemetry_consent() -> str:
    console.print(
        "[dim]Switch Trust collects anonymous usage analytics to improve the product.[/dim]"
    )
    console.print(
        "[dim]No code, prompts, keys, or personal data is ever collected.[/dim]"
    )
    console.print("[dim]You can change this at any time in your .env config.[/dim]")
    console.print()
    choice = select(
        "Do you consent to sharing anonymous usage analytics?",
        options=["Y", "N"],
        default_index=0,
    )
    console.print()

    return "true" if choice.lower() == "y" else "false"


def ensure_telemetry_consent() -> None:
    if resolve_env_var(TELEMETRY_CONSENT_ENV_VAR):
        return

    if is_ci():
        os.environ[TELEMETRY_CONSENT_ENV_VAR] = "false"
        return

    value = prompt_telemetry_consent()
    os.environ[TELEMETRY_CONSENT_ENV_VAR] = value

    env_path = get_switch_trust_env_path()
    if env_path.exists():
        with open(env_path, "a") as f:
            f.write(f"{TELEMETRY_CONSENT_ENV_VAR}={value}\n")
        env_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
