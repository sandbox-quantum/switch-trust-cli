"""
Global generator/judge model used for LLM-as-judge detection,
metric evaluations, adversarial probing, and evaluation
generation.

Configured via the GENERATOR_MODEL environment variable:
    GENERATOR_MODEL=type:model_name

Examples:
    GENERATOR_MODEL=gemini:gemini-2.5-flash-lite
    GENERATOR_MODEL=openai:gpt-4o-mini
    GENERATOR_MODEL=anthropic:claude-haiku-4-20250414
    GENERATOR_MODEL=litellm:gpt-4o
    GENERATOR_MODEL=ollama:llama3

For the ``gemini`` type, the client authenticates against the Gemini Developer
API by default (via ``GOOGLE_API_KEY``). Set GOOGLE_VERTEX_CREDENTIALS_JSON to a
service-account key (raw JSON or base64-encoded JSON) to route through Vertex AI
instead; GOOGLE_CLOUD_LOCATION is then required and GOOGLE_CLOUD_PROJECT defaults
to the ``project_id`` inside the key.

Config handling follows one rule: the environment is read and validated in
exactly one place — ``GeneratorConfig.from_env()`` — which returns a validated
object the rest of this module builds from. The factories (``_create_inner`` /
``_create_gemini_client``) take that typed config and never touch ``os.environ``,
so a malformed value is caught at the funnel (and, in the worker, at startup via
``set_generator_config``) rather than mid-run.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from switch_trust.eval.common.schema import Message
from switch_trust.eval.core.models.model import Model, ModelResponse
from switch_trust.eval.core.models.model_retry import RetryModel

logger = logging.getLogger(__name__)


@dataclass
class VertexConfig:
    """Vertex AI wiring for a ``gemini`` generator client.

    Present only when GOOGLE_VERTEX_CREDENTIALS_JSON is set; a Developer-API
    gemini generator leaves ``GeneratorConfig.vertex`` as ``None``. Built through
    ``from_key`` so the service-account JSON is decoded and structurally
    validated — and the project/location resolved — before it is ever used.
    """

    # Parsed, structurally validated service-account key. Secret (holds the
    # private key), so it is never echoed in errors or logs.
    service_account: dict
    project: str
    location: str

    @staticmethod
    def from_key(raw: str, project: str, location: str) -> VertexConfig:
        """Build+validate from the raw key and the (possibly empty) project/location.

        Project defaults to the key's ``project_id`` when not given; location has
        no default and is required, since a Vertex client cannot be built without
        it. Raises ``ValueError`` on any missing/malformed piece.
        """
        info = _parse_service_account(raw, source="GOOGLE_VERTEX_CREDENTIALS_JSON")
        project = project or str(info.get("project_id") or "")
        if not location:
            raise ValueError(
                "Vertex AI generator requires GOOGLE_CLOUD_LOCATION alongside "
                "GOOGLE_VERTEX_CREDENTIALS_JSON"
            )
        if not project:
            raise ValueError(
                "Vertex generator requires a project (GOOGLE_CLOUD_PROJECT, or a "
                "project_id in the service-account key)"
            )
        return VertexConfig(service_account=info, project=project, location=location)


@dataclass
class GeneratorConfig:
    """Validated generator configuration — the single environment funnel.

    ``from_env`` is the only place the environment is read; it parses and
    validates the shape, so everything downstream works with a typed object and
    focuses on *what* to build, not *how* to fetch config.
    """

    model_type: str
    model_name: str
    # gemini-only: the Vertex client wiring, or None for the Developer-API path.
    vertex: VertexConfig | None = None
    # gemini-only: whether a GOOGLE_API_KEY was present at parse time. Only its
    # presence is captured — the key itself is read from the environment by the
    # google-genai client at build time. Used by ``validate`` to insist on a
    # usable Developer-API credential when Vertex is not configured.
    gemini_api_key_present: bool = False

    @staticmethod
    def from_env() -> GeneratorConfig:
        """Parse and validate the generator configuration from the environment."""
        spec = os.environ.get("GENERATOR_MODEL", "").strip()
        if not spec:
            raise ValueError(
                "GENERATOR_MODEL must be set (format type:model_name, "
                "e.g. gemini:gemini-2.5-flash)"
            )
        if ":" not in spec:
            raise ValueError(
                f"Invalid GENERATOR_MODEL format: {spec!r}. Expected type:model_name"
            )
        model_type, model_name = (part.strip() for part in spec.split(":", 1))
        if not model_type or not model_name:
            raise ValueError(
                f"Invalid GENERATOR_MODEL format: {spec!r}. Expected type:model_name"
            )

        vertex: VertexConfig | None = None
        api_key_present = False
        if model_type == "gemini":
            raw = os.environ.get("GOOGLE_VERTEX_CREDENTIALS_JSON", "").strip()
            if raw:
                vertex = VertexConfig.from_key(
                    raw,
                    project=os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip(),
                    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip(),
                )
            else:
                api_key_present = bool(os.environ.get("GOOGLE_API_KEY", "").strip())

        return GeneratorConfig(
            model_type=model_type,
            model_name=model_name,
            vertex=vertex,
            gemini_api_key_present=api_key_present,
        )

    def validate(self) -> None:
        """Fail-fast startup check, beyond the shape checks ``from_env`` did.

        Requires a usable gemini credential combination (Vertex creds or
        GOOGLE_API_KEY — ADC alone is accepted at runtime but not treated as
        "configured" here) and exercises model construction, which makes no API
        call but does validate provider wiring and the service-account key's
        private-key material. Called once by the worker before it goes ready so a
        misconfigured generator aborts the process rather than the first job.
        """
        if (
            self.model_type == "gemini"
            and self.vertex is None
            and not self.gemini_api_key_present
        ):
            raise ValueError(
                "gemini generator requires either GOOGLE_VERTEX_CREDENTIALS_JSON + "
                "GOOGLE_CLOUD_LOCATION (Vertex AI) or GOOGLE_API_KEY "
                "(Gemini Developer API)"
            )
        # Build the model once to exercise wiring / credential structure; discard.
        _create_inner(self)


# Installed once at worker startup so every model build reuses the config that
# already passed through the from_env() validation funnel instead of re-reading
# the environment. Left None (the CLI path), get_generator_model() falls back to
# GeneratorConfig.from_env().
_config: GeneratorConfig | None = None


def set_generator_config(config: GeneratorConfig) -> None:
    """Install the validated config get_generator_model() builds from.

    The worker calls this once at startup with the object returned by
    ``GeneratorConfig.from_env()`` (after ``validate()``), so the many lazy
    ``get_generator_model()`` call sites reuse one validated config rather than
    each re-parsing the environment.
    """
    global _config
    _config = config


class GeneratorModel(Model):
    """Decorator that flags generator/judge calls made without a schema.

    The generator model is used for LLM-as-judge detection and synthetic
    prompt generation, which parse the response as JSON. Callers should pass an
    ``output_schema`` so the underlying provider constrains the output; a call
    without one falls back to unconstrained text and is prone to parse
    failures, so we log a warning to surface the omission.
    """

    _model: Model

    def __init__(self, model: Model) -> None:
        self._model = model

    async def _generate(
        self,
        messages: list[Message],
        *,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        if output_schema is None:
            logger.warning(
                "Generator model called without an output_schema; structured "
                "output is not enforced and response parsing may fail"
            )
        return await self._model.generate(
            messages, output_schema=output_schema, **kwargs
        )


def get_generator_model() -> Model:
    """Create a fresh generator model instance.

    Always creates a new instance because async SDK clients are bound to the
    event loop they're first used on. Caching across loops causes 'Future
    attached to a different loop' errors. The *config* is reused when installed
    via ``set_generator_config`` (worker) or parsed on demand (CLI); only the
    model/client is rebuilt.
    """
    return GeneratorModel(_create_generator_model())


def _create_generator_model() -> Model:
    config = _config if _config is not None else GeneratorConfig.from_env()
    inner = _create_inner(config)
    logger.debug(
        "Creating generator model: %s:%s", config.model_type, config.model_name
    )
    return RetryModel(inner, max_retries=5)


def _create_inner(config: GeneratorConfig) -> Model:
    model_type, model_name = config.model_type, config.model_name
    if model_type == "gemini":
        from switch_trust.eval.core.models.model_gemini import (  # noqa: PLC0415 - patched at source in tests
            SAFETY_OFF,
            GeminiModel,
        )

        # The generator is the attacker/judge; it must not be safety-blocked, or
        # adversarial turns and judge JSON come back empty (especially on Vertex).
        return GeminiModel(
            _create_gemini_client(config.vertex),
            model_name,
            safety_settings=SAFETY_OFF,
        )

    elif model_type == "openai":
        from openai import AsyncOpenAI  # noqa: PLC0415 - patched at source in tests

        from switch_trust.eval.core.models.model_openai import (  # noqa: PLC0415 - patched at source in tests
            OpenAIModel,
        )

        return OpenAIModel(AsyncOpenAI(), model_name)

    elif model_type == "anthropic":
        from anthropic import (  # noqa: PLC0415 - patched at source in tests
            AsyncAnthropic,
        )

        from switch_trust.eval.core.models.model_anthropic import (  # noqa: PLC0415 - patched at source in tests
            AnthropicModel,
        )

        return AnthropicModel(AsyncAnthropic(), model_name)

    elif model_type == "litellm":
        from switch_trust.eval.core.models.model_litellm import (  # noqa: PLC0415 - patched at source in tests
            LiteLLMModel,
        )

        return LiteLLMModel(model_name)

    elif model_type == "ollama":
        from switch_trust.eval.core.models.model_ollama import (  # noqa: PLC0415 - patched at source in tests
            OllamaModel,
        )

        return OllamaModel(model_name)

    else:
        raise ValueError(
            f"Unknown generator model type: {model_type!r}. "
            "Supported: gemini, openai, anthropic, litellm, ollama"
        )


def _create_gemini_client(vertex: VertexConfig | None):
    """Build the google-genai client for the generator model.

    With ``vertex`` unset this is the unchanged default: ``Client()`` picks up
    the Gemini Developer API (GOOGLE_API_KEY) or a fully env-driven Vertex/ADC
    setup. With it set, the client is built in Vertex AI mode from the (already
    parsed and validated) service-account key, so a key handed to us as JSON is
    usable without writing it to a file or relying on ADC discovery.

    Imports stay lazy (no module-scope google dependency), so the return type is
    left undeclared rather than pulling ``google.genai.Client`` to the top.
    """
    from google.genai import (  # noqa: PLC0415 - lazy import keeps google.genai out of module-scope deps
        Client,
    )

    if vertex is None:
        return Client()

    from google.oauth2 import (  # noqa: PLC0415 - lazy import keeps google.oauth2 out of module-scope deps
        service_account,
    )

    credentials = service_account.Credentials.from_service_account_info(
        vertex.service_account,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    logger.debug("Creating Vertex AI Gemini client (project=%s)", vertex.project)
    return Client(
        vertexai=True,
        project=vertex.project,
        location=vertex.location,
        credentials=credentials,
    )


def _parse_service_account(raw: str, source: str = "service-account key") -> dict:
    """Decode + structurally validate a service-account key.

    Accepts raw JSON or base64-encoded JSON (base64 so the multi-line key
    survives shell env files), then checks it is actually a service-account key
    with the fields Vertex auth needs — a tight structural check rather than a
    bare parse — and that ``private_key`` is a structurally intact PEM (see
    ``_validate_private_key_pem``), so a newline-mangled key fails here with an
    actionable message rather than as a terse google-auth error later. ``source``
    names the origin of ``raw`` in error messages so the function is not coupled
    to any single caller/env var. The errors never echo ``raw`` — it is secret
    material (contains a private key).
    """
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        try:
            decoded = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"{source} is neither valid JSON nor base64") from exc
        try:
            info = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source} base64-decoded to invalid JSON") from exc

    if not isinstance(info, dict):
        raise ValueError(f"{source} must be a JSON object")
    if info.get("type") != "service_account":
        raise ValueError(
            f'{source} must be a service-account key (its "type" must be '
            '"service_account")'
        )
    missing = [
        field
        for field in ("client_email", "private_key", "token_uri")
        if not info.get(field)
    ]
    if missing:
        raise ValueError(
            f"{source} is missing required service-account fields: {', '.join(missing)}"
        )
    _validate_private_key_pem(str(info["private_key"]), source)
    return info


def _validate_private_key_pem(private_key: str, source: str) -> None:
    """Check the ``private_key`` field is a structurally intact PEM block.

    The structural checks above only confirm ``private_key`` is *present*; the
    actual PEM is parsed much later by google-auth
    (``from_service_account_info``), which fails with a terse "invalid private
    key" that gives no hint at the real cause. That cause is almost always the
    key's newlines being mangled while the secret is seeded by hand: a JSON key
    piped through a shell/YAML/``jq -r`` step has its ``\\n`` either collapsed
    (the PEM becomes one line) or double-escaped (``\\\\n``), and either way the
    JSON still parses and this field is still non-empty. Catch that here, at the
    config funnel, with a message that names the fix instead of leaving it to a
    google-auth stack trace mid-startup.

    A cheap structural check, deliberately no crypto dependency (this lives in
    the OSS-exported ``core/``): require the PEM header/footer markers and real
    newlines, and reject the literal ``\\n`` escape sequence. The message never
    echoes the key material.
    """
    if r"\n" in private_key:
        raise ValueError(
            f"{source} private_key contains the literal escape sequence '\\n' "
            "rather than real newlines — its newlines were double-escaped during "
            "seeding. Re-seed from the untouched key file (e.g. "
            "--secret-string file://key.json), or store the whole key as "
            "single-line base64."
        )
    if "-----BEGIN" not in private_key or "-----END" not in private_key:
        raise ValueError(
            f"{source} private_key is not a PEM block (missing BEGIN/END "
            "markers) — the key material was corrupted during seeding. Re-seed "
            "from the untouched key file, or store the whole key as single-line "
            "base64."
        )
    if "\n" not in private_key.strip("\n"):
        raise ValueError(
            f"{source} private_key has no internal newlines — the PEM was "
            "collapsed onto one line during seeding, which google-auth rejects "
            "as an invalid private key. Re-seed from the untouched key file "
            "(e.g. --secret-string file://key.json), or store the whole key as "
            "single-line base64 so the newlines survive."
        )
