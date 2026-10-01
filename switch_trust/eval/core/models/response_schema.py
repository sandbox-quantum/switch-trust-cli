"""Structured-output schema plumbing shared across model adapters.

Judge/generator callers pass a Pydantic ``BaseModel`` subclass as the
``output_schema`` argument to :meth:`Model.generate`. Each provider adapter
translates it to that provider's native structured-output mechanism:

- OpenAI / OpenAI-compatible (vLLM, Azure, ...) → ``json_schema`` response format
- Ollama → the native ``format`` JSON-schema parameter
- Gemini → ``response_mime_type`` + ``response_schema``
- LiteLLM → ``response_format`` (the Pydantic class, translated per provider)
- Anthropic → a single forced tool whose ``input_schema`` is the target schema

:func:`parse_model_response` deserialises the model output back into the schema,
with a best-effort fallback for the model-under-test and any provider that could
not hard-enforce the shape.
"""

from __future__ import annotations

import logging
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from switch_trust.eval.core.models.model import (
    ModelResponse,
    extract_text_from_message,
)


class ModelResponseParseError(ValueError):
    """A model/judge response could not be parsed into the target schema.

    A ``ValueError`` subclass, so any existing ``except ValueError`` handling
    keeps working unchanged; the distinct type lets the worker's error
    categorization (see ``platform/worker.py::_classify_error_category``)
    distinguish this from other permanent failures for dashboard/log
    discoverability.
    """


logger = logging.getLogger(__name__)

# TypeVar rather than PEP 695 `[T: BaseModel]` syntax: eval/core is exported to
# the flintai-cli wheel (requires-python >= 3.11), and PEP 695 type parameters
# are a syntax error before 3.12.
T = TypeVar("T", bound=BaseModel)

# Name of the forced tool used by providers that emit structured output via
# tool calling (Anthropic).
STRUCTURED_OUTPUT_TOOL_NAME = "emit_structured_output"


def to_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Return the plain JSON Schema for a Pydantic model."""
    return schema.model_json_schema()


def to_strict_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Return a JSON Schema tightened for OpenAI-style strict decoding.

    Strict ``json_schema`` response formats require every object to set
    ``additionalProperties: false`` and to list all of its properties in
    ``required``. Pydantic emits ``required`` but not ``additionalProperties``,
    so we walk the schema and add what the strict validator expects.
    """
    return _tighten(schema.model_json_schema())


def _tighten(node: Any) -> Any:
    if isinstance(node, dict):
        tightened = {key: _tighten(value) for key, value in node.items()}
        if tightened.get("type") == "object" and "properties" in tightened:
            tightened["additionalProperties"] = False
            tightened["required"] = list(tightened["properties"].keys())
        return tightened
    if isinstance(node, list):
        return [_tighten(item) for item in node]
    return node


def to_openai_response_format(schema: type[BaseModel]) -> dict[str, Any]:
    """Build an OpenAI / vLLM ``response_format`` for ``schema``."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": to_strict_json_schema(schema),
            "strict": True,
        },
    }


def to_anthropic_tool(
    schema: type[BaseModel],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a forced-tool definition and ``tool_choice`` for ``schema``.

    Anthropic has no cross-version ``response_format``, so forcing a single
    tool whose ``input_schema`` is the target schema makes the model emit the
    payload as the tool input, which the adapter reads back as JSON.
    """
    tool = {
        "name": STRUCTURED_OUTPUT_TOOL_NAME,
        "description": (
            "Record the final structured result. Always call this tool with "
            "the result and nothing else."
        ),
        "input_schema": to_json_schema(schema),
    }
    tool_choice = {"type": "tool", "name": STRUCTURED_OUTPUT_TOOL_NAME}
    return tool, tool_choice


def parse_model_response(response: ModelResponse, schema: type[T]) -> T:  # noqa: UP047
    """Deserialise a model response into ``schema``.

    Providers that hard-enforce the schema return exactly the JSON payload;
    those that cannot may wrap it in prose or markdown fences, so we fall back
    to extracting the first ``{...}`` block. Raises ``ValueError`` if nothing
    parses.
    """
    if response.message is None:
        raise ModelResponseParseError("model returned no message to parse")
    return parse_json_text(extract_text_from_message(response.message), schema)


def parse_json_text(text: str, schema: type[T]) -> T:  # noqa: UP047
    """Parse ``text`` into ``schema``, tolerating fences and surrounding prose."""
    stripped = text.strip()
    try:
        return schema.model_validate_json(stripped)
    except ValidationError as first_error:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match is not None:
            try:
                return schema.model_validate_json(match.group())
            except ValidationError:
                pass
        logger.debug(
            "Failed to parse %s from model output: %r",
            schema.__name__,
            text,
        )
        raise ModelResponseParseError(
            f"model output did not match schema {schema.__name__}"
        ) from first_error
