from typing import Any

from google.genai import Client
from google.genai import types as genai_types
from pydantic import BaseModel

from switch_trust.eval.common import converter_genai
from switch_trust.eval.common.schema import Message, Role
from switch_trust.eval.core.models.model import (
    Model,
    ModelResponse,
    ResponseStatus,
)

# Safety settings that disable content blocking across every active harm
# category. Red-teaming needs the attacker/judge model to see and produce
# adversarial content without Vertex's filters returning an empty candidate — a
# blocked judge yields no JSON to parse, which surfaces as a spurious eval error
# rather than a real result. HARM_CATEGORY_CIVIC_INTEGRITY is deprecated and
# omitted. Note this is only appropriate for the generator; a model *under test*
# should keep its own safety behavior so blocking can be measured.
SAFETY_OFF: list[genai_types.SafetySetting] = [
    genai_types.SafetySetting(category=category, threshold="OFF")
    for category in (
        genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    )
]


class GeminiModel(Model):
    _client: Client
    _model: str
    _temperature: float
    _safety_settings: list[genai_types.SafetySetting] | None

    def __init__(
        self,
        client: Client,
        model: str,
        temperature: float = 0.0,
        safety_settings: list[genai_types.SafetySetting] | None = None,
    ):
        self._client = client
        self._model = model
        self._temperature = temperature
        self._safety_settings = safety_settings

    async def _generate(
        self,
        messages: list[Message],
        *,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        system_parts: list[dict[str, Any]] = []
        contents: list[dict[str, Any]] = []
        for msg in messages:
            if msg.content.role == Role.SYSTEM:
                system_parts.extend(
                    converter_genai.from_content(
                        msg.content,
                    )["parts"]
                )
            else:
                contents.append(
                    converter_genai.from_content(msg.content),
                )

        config = kwargs.pop("config", None)
        if config is None:
            config = genai_types.GenerateContentConfig(
                temperature=self._temperature,
            )
        elif config.temperature is None:
            config.temperature = self._temperature
        if system_parts:
            system_instruction = genai_types.Content(
                parts=[genai_types.Part(**p) for p in system_parts],
            )
            config.system_instruction = system_instruction
        if self._safety_settings is not None and config.safety_settings is None:
            config.safety_settings = self._safety_settings
        if output_schema is not None and config.response_schema is None:
            config.response_mime_type = "application/json"
            config.response_schema = output_schema

        generate_kwargs: dict[str, Any] = {
            "model": self._model,
            "contents": contents,
            **kwargs,
        }
        if config is not None:
            generate_kwargs["config"] = config

        response = await self._client.aio.models.generate_content(
            **generate_kwargs,
        )
        if (
            response.candidates is not None
            and len(response.candidates) == 1
            and response.candidates[0].content is not None
        ):
            message = converter_genai.to_message(
                response.candidates[0].content,
            )
            return ModelResponse(message)
        return ModelResponse(
            None,
            _classify_block_reason(response),
        )


def _classify_block_reason(response) -> ResponseStatus:
    """Determine why a Gemini response was blocked."""
    # Check prompt-level blocking
    reason = getattr(
        response,
        "prompt_feedback",
        None,
    )
    if reason is not None:
        block_reason = getattr(reason, "block_reason", None)
        if block_reason is not None:
            reason_str = str(block_reason).upper()
            if "SAFETY" in reason_str:
                return ResponseStatus.BLOCKED_SAFETY
            if "PROHIBITED" in reason_str:
                return ResponseStatus.BLOCKED_PROHIBITED

    # Check candidate-level finish reason
    candidates = getattr(response, "candidates", None)
    if candidates:
        finish = getattr(candidates[0], "finish_reason", None)
        if finish is not None:
            finish_str = str(finish).upper()
            if "SAFETY" in finish_str:
                return ResponseStatus.BLOCKED_SAFETY
            if "RECITATION" in finish_str:
                return ResponseStatus.BLOCKED_RECITATION
            if "PROHIBITED" in finish_str:
                return ResponseStatus.BLOCKED_PROHIBITED

    return ResponseStatus.BLOCKED_SAFETY
