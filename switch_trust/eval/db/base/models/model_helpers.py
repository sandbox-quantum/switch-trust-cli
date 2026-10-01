from collections.abc import Callable

import aiohttp

from switch_trust.eval.common.utils import resolve_env, resolve_env_dict
from switch_trust.eval.core.models.model import Model
from switch_trust.eval.core.models.model_per_endpoint_limiter import (
    PerEndpointLimiter,
)
from switch_trust.eval.core.models.model_retry import RetryModel
from switch_trust.eval.db.base.models.model_types import DbModel, ModelType

# HTTP-based agent types. SDK-based providers (Anthropic, OpenAI, Gemini,
# LiteLLM, HuggingFace, Ollama, OpenAI-compatible) own their own rate limiting
# and don't need our per-endpoint semaphore.
_LIMITED_HTTP_TYPES = {
    ModelType.ADK,
    ModelType.OPENAI_AGENT,
    ModelType.ANTHROPIC_AGENT,
    ModelType.GENERIC_HTTP,
    ModelType.LANGSERVE,
    ModelType.VERTEX_AGENT_RUNTIME,
}


def create_model(
    db_model: DbModel,
    max_retries: int = 5,
    resolve_env_vars: bool = True,
    connector_factory: Callable[[], aiohttp.BaseConnector] | None = None,
    per_endpoint_limit: int = 8,
    initial_jitter_max: float = 1.0,
) -> Model:
    """Create a Model instance from a DbModel, wrapped in a RetryModel for
    transient error handling. HTTP-based types are additionally wrapped in a
    PerEndpointLimiter so bursts against one tenant endpoint don't blow past
    its QPS quota.

    connector_factory, when set, is passed to the HTTP-based models so their
    aiohttp sessions enforce the eval worker's request-time SSRF guard
    (platform/ssrf.py). None keeps the default connector for CLI/local use.
    """
    inner = _create_inner_model(
        db_model,
        resolve_env_vars=resolve_env_vars,
        connector_factory=connector_factory,
    )
    if db_model.type in _LIMITED_HTTP_TYPES and db_model.host:
        inner = PerEndpointLimiter(
            inner,
            endpoint_url=db_model.host,
            limit=per_endpoint_limit,
            jitter_max=initial_jitter_max,
        )
    return RetryModel(inner, max_retries=max_retries)


def _create_inner_model(
    db_model: DbModel,
    resolve_env_vars: bool = True,
    connector_factory: Callable[[], aiohttp.BaseConnector] | None = None,
) -> Model:
    temp = db_model.temperature
    if resolve_env_vars:
        key = resolve_env(db_model.key)
        headers = resolve_env_dict(db_model.headers)
    else:
        key = db_model.key
        headers = dict(db_model.headers)

    if db_model.type == ModelType.ANTHROPIC:
        from anthropic import (  # noqa: PLC0415 - patched at source in tests
            AsyncAnthropic,
        )

        from switch_trust.eval.core.models.model_anthropic import (  # noqa: PLC0415 - patched at source in tests
            AnthropicModel,
        )

        client_kwargs = {}
        if key:
            client_kwargs["api_key"] = key

        return AnthropicModel(
            AsyncAnthropic(**client_kwargs),
            db_model.model_name,
            temperature=temp,
        )

    elif db_model.type == ModelType.OPENAI:
        from openai import AsyncOpenAI  # noqa: PLC0415 - patched at source in tests

        from switch_trust.eval.core.models.model_openai import (  # noqa: PLC0415 - patched at source in tests
            OpenAIModel,
        )

        client_kwargs = {}
        if key:
            client_kwargs["api_key"] = key

        return OpenAIModel(
            AsyncOpenAI(**client_kwargs),
            db_model.model_name,
            temperature=temp,
        )

    elif db_model.type == ModelType.GEMINI:
        from google.genai import Client  # noqa: PLC0415 - patched at source in tests

        from switch_trust.eval.core.models.model_gemini import (  # noqa: PLC0415 - patched at source in tests
            GeminiModel,
        )

        client_kwargs = {}
        if key:
            client_kwargs["api_key"] = key

        return GeminiModel(
            Client(**client_kwargs),
            db_model.model_name,
            temperature=temp,
        )

    elif db_model.type == ModelType.LITELLM:
        from switch_trust.eval.core.models.model_litellm import (  # noqa: PLC0415 - patched at source in tests
            LiteLLMModel,
        )

        return LiteLLMModel(
            db_model.model_name,
            temperature=temp,
        )

    elif db_model.type == ModelType.HUGGINGFACE:
        from switch_trust.eval.core.models.model_huggingface import (  # noqa: PLC0415 - patched at source in tests
            HuggingFaceModel,
        )

        return HuggingFaceModel(
            db_model.model_name,
            token=key or None,
            temperature=temp,
        )

    elif db_model.type == ModelType.OLLAMA:
        from switch_trust.eval.core.models.model_ollama import (  # noqa: PLC0415 - patched at source in tests
            OllamaModel,
        )

        return OllamaModel(
            db_model.model_name,
            host=db_model.host or "http://localhost:11434",
            temperature=temp,
        )

    elif db_model.type == ModelType.ADK:
        from switch_trust.eval.core.models.model_adk import (  # noqa: PLC0415 - patched at source in tests
            ADKModel,
        )

        return ADKModel(
            app_name=db_model.model_name,
            host=db_model.host or "http://localhost:8000",
            immediate_result=db_model.immediate_result,
            headers=headers or None,
            connector_factory=connector_factory,
        )

    elif db_model.type == ModelType.OPENAI_AGENT:
        from switch_trust.eval.core.models.model_openai_agent import (  # noqa: PLC0415 - patched at source in tests
            OpenAIAgentModel,
        )

        return OpenAIAgentModel(
            host=db_model.host or "http://localhost:8000",
            endpoint=db_model.endpoint or "/run",
            connector_factory=connector_factory,
        )

    elif db_model.type == ModelType.ANTHROPIC_AGENT:
        from switch_trust.eval.core.models.model_anthropic_agent import (  # noqa: PLC0415 - patched at source in tests
            AnthropicAgentModel,
        )

        return AnthropicAgentModel(
            host=db_model.host or "http://localhost:8000",
            endpoint=db_model.endpoint or "/run",
            connector_factory=connector_factory,
        )

    elif db_model.type == ModelType.OPENAI_COMPATIBLE:
        from switch_trust.eval.core.models.model_openai_compatible import (  # noqa: PLC0415 - patched at source in tests
            OpenAICompatibleModel,
        )

        return OpenAICompatibleModel(
            base_url=db_model.host or "http://localhost:8000/v1",
            model=db_model.model_name,
            api_key=key or "EMPTY",
            temperature=temp,
            headers=headers or None,
        )

    elif db_model.type == ModelType.GENERIC_HTTP:
        from switch_trust.eval.core.models.model_generic_http import (  # noqa: PLC0415 - patched at source in tests
            GenericHttpModel,
        )

        url = db_model.host or "http://localhost:8000"
        if db_model.endpoint:
            url = f"{url.rstrip('/')}{db_model.endpoint}"

        return GenericHttpModel(
            url=url,
            headers=headers or None,
            input_path=db_model.input_path or "input",
            output_path=db_model.output_path or "output",
            connector_factory=connector_factory,
        )

    elif db_model.type == ModelType.VERTEX_AGENT_RUNTIME:
        from switch_trust.eval.core.models.model_vertex_agent_runtime import (  # noqa: PLC0415 - patched at source in tests
            DEFAULT_CLASS_METHOD,
            DEFAULT_USER_ID,
            VertexAgentRuntimeModel,
        )

        # The engine URL already carries the :streamQuery suffix, so `endpoint`
        # is not appended here the way it is for the generic HTTP types.
        return VertexAgentRuntimeModel(
            url=db_model.host or "",
            credential=key or None,
            headers=headers or None,
            user_id=db_model.user_id or DEFAULT_USER_ID,
            class_method=db_model.class_method or DEFAULT_CLASS_METHOD,
            immediate_result=db_model.immediate_result,
            connector_factory=connector_factory,
        )

    elif db_model.type == ModelType.LANGSERVE:
        from switch_trust.eval.core.models.model_langserve import (  # noqa: PLC0415 - patched at source in tests
            LangServeModel,
        )

        return LangServeModel(
            base_url=db_model.host or "http://localhost:8000",
            chain_path=db_model.endpoint or "",
            headers=headers or None,
            connector_factory=connector_factory,
        )

    else:
        raise ValueError(f"unknown model type: {db_model.type}")
