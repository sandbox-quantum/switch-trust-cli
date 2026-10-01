import json
import logging
import os
import re
from datetime import UTC, datetime
from uuid import uuid4

from dataclasses_json import config


def extract_json(text: str) -> dict:
    """Extract a JSON object from model output.

    Handles markdown fences, leading/trailing prose, and thinking text that
    some models prepend to JSON output — Gemini in particular tends to wrap its
    JSON in a ```json fence, which a bare json.loads cannot parse.
    """
    cleaned = text.strip()

    # Strip markdown fences (```json ... ``` or ``` ... ```)
    if "```" in cleaned:
        fence = re.search(
            r"```(?:json)?\s*\n?(.*?)```",
            cleaned,
            re.DOTALL,
        )
        if fence:
            cleaned = fence.group(1).strip()

    # Fix double braces (models sometimes echo {{ }})
    if "{{" in cleaned:
        cleaned = cleaned.replace("{{", "{").replace("}}", "}")

    # Try parsing directly first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Find the first { ... } block in the text
    start = cleaned.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        break

    raise json.JSONDecodeError(
        "No valid JSON object found in model output",
        cleaned,
        0,
    )


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging with a standard format."""
    logging.basicConfig(
        level=level,
        format=("%(asctime)s %(levelname)s %(name)s: %(message)s"),
    )


def generate_id() -> str:
    return str(uuid4())


def now_utc() -> datetime:
    return datetime.now(UTC)


datetime_config = config(
    encoder=datetime.isoformat,
    decoder=datetime.fromisoformat,
)


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def resolve_env(value: str | None) -> str | None:
    """Substitute ``${VAR_NAME}`` references with environment variable values.

    Plain strings without ``${…}`` are returned unchanged.
    Raises ``ValueError`` if a referenced variable is not set.
    """
    if value is None:
        return None

    def _replace(match: re.Match[str]) -> str:
        var = match.group(1)
        resolved = os.environ.get(var)
        if resolved is None:
            raise ValueError(f"Environment variable {var!r} is not set")
        return resolved

    return _ENV_PATTERN.sub(_replace, value)


def resolve_env_dict(
    d: dict[str, str],
) -> dict[str, str]:
    """Apply :func:`resolve_env` to every value in *d*."""
    return {k: resolve_env(v) for k, v in d.items()}


def strip_nulls(obj: object) -> object:
    """Recursively remove None-valued keys from dicts."""
    if isinstance(obj, dict):
        return {k: strip_nulls(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [strip_nulls(item) for item in obj]
    return obj
