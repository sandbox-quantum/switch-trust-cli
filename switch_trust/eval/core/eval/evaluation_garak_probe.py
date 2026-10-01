from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dataclasses_json import dataclass_json

from switch_trust.eval.common.schema import Content, Message, PartType, Role, Session
from switch_trust.eval.core.detectors.detector_garak import GarakDetector
from switch_trust.eval.core.eval.evaluation import Evaluation, EvaluationStatus
from switch_trust.eval.core.eval.evaluation_multi import MultiEvaluation
from switch_trust.eval.core.eval.evaluation_single import SingleEvaluation
from switch_trust.eval.core.eval.evaluation_single_prompt import (
    SinglePromptEvaluation,
)
from switch_trust.eval.core.models.model import Model, ResponseStatus
from switch_trust.eval.core.models.model_sync_wrapper import SyncModelWrapper
from switch_trust.eval.core.optional_deps import require_garak

if TYPE_CHECKING:
    from garak.attempt import Conversation
    from garak.attempt import Message as GarakMessage

logger = logging.getLogger(__name__)

# Cached at first use so the garak Generator base class (part of the optional
# ``full`` extra) is only imported when a dynamic garak probe actually runs.
_generator_adapter_cls: type | None = None


# -- Helpers -------------------------------------------------------


def _ensure_garak_config() -> None:
    """Initialize garak's global config if not already loaded.

    Garak probes expect config values (verbose, generations, seed,
    etc.) that are normally set by garak's CLI harness.
    ``load_base_config`` reads garak's bundled ``garak.core.yaml``
    to set all defaults.  We also provide a dummy ``reportfile``
    since some probes write progress to it.
    """
    config = require_garak()._config
    if not config.loaded:
        config.load_base_config()
    if config.transient.reportfile is None:
        config.transient.reportfile = io.StringIO()


def _garak_prompt_to_message(prompt) -> Message:
    garak = require_garak()
    if isinstance(prompt, str):
        text = prompt
    elif isinstance(prompt, garak.attempt.Message):
        text = prompt.text
    elif isinstance(prompt, garak.attempt.Conversation):
        parts = [turn.content.text for turn in prompt.turns]
        text = "\n".join(parts)
    else:
        text = str(prompt)
    return Message(content=Content.text(Role.USER, text))


def _has_static_prompts(probe: Any) -> bool:
    """Check whether a garak probe has a static prompts list."""
    return (
        hasattr(probe, "prompts")
        and probe.prompts is not None
        and len(probe.prompts) > 0
    )


def _conversation_to_messages(
    conversation: Conversation,
) -> list[Message]:
    """Convert a garak Conversation to a list of AIRed Messages."""
    role_map = {
        "system": Role.SYSTEM,
        "user": Role.USER,
        "assistant": Role.ASSISTANT,
    }
    messages: list[Message] = []
    for turn in conversation.turns:
        role = role_map.get(turn.role, Role.USER)
        text = turn.content.text if turn.content else ""
        messages.append(
            Message(content=Content.text(role, text)),
        )
    return messages


def _extract_text(message: Message) -> str:
    """Extract plain text from an AIRed Message."""
    text_parts = [
        part.text
        for part in message.content.parts
        if part.part_type == PartType.TEXT and part.text
    ]
    return "\n".join(text_parts)


def _attempts_to_session(attempts: list) -> Session:
    """Convert garak Attempts into an AIRed Session."""
    messages: list[Message] = []
    role_map = {
        "system": Role.SYSTEM,
        "user": Role.USER,
        "assistant": Role.ASSISTANT,
    }
    for attempt in attempts:
        for conversation in attempt.conversations:
            for turn in conversation.turns:
                role = role_map.get(turn.role, Role.USER)
                text = turn.content.text if turn.content else ""
                messages.append(
                    Message(content=Content.text(role, text)),
                )
    return Session(messages=messages)


# -- Generator adapter ---------------------------------------------


def _get_generator_adapter_cls() -> type:
    """Build (once) the garak Generator adapter class.

    The class subclasses garak's ``Generator``, which lives in the optional
    ``full`` extra, so it can only be defined once garak is importable.  The
    result is memoized in ``_generator_adapter_cls``.
    """
    global _generator_adapter_cls
    if _generator_adapter_cls is not None:
        return _generator_adapter_cls

    generator_base = require_garak().generators.base.Generator

    class GarakGeneratorAdapter(generator_base):
        """Wraps an AIRed Model as a garak Generator.

        Garak probes call ``generator.generate(conversation)`` which
        delegates to ``_call_model``.  This adapter translates between
        garak's Conversation/Message types and AIRed's Message type,
        forwards the call to the wrapped SyncModelWrapper, and returns
        the result as a garak Message.

        Uses SyncModelWrapper because garak's probe system is entirely
        synchronous.
        """

        def __init__(self, sync_model: SyncModelWrapper):
            self._sync_model = sync_model
            self.name = "aired-adapter"
            self.generations = 1
            self.supports_multiple_generations = False
            self.seed = None
            self.fullname = "aired-adapter"

        def _call_model(
            self,
            prompt: Conversation,
            generations_this_call: int = 1,
        ) -> list[GarakMessage | None]:
            messages = _conversation_to_messages(prompt)
            response = self._sync_model.generate(messages)

            if response.status != ResponseStatus.OK:
                return [None]
            if response.message is None:
                return [None]

            text = _extract_text(response.message)
            return [require_garak().attempt.Message(text=text)]

    _generator_adapter_cls = GarakGeneratorAdapter
    return _generator_adapter_cls


# -- Evaluation classes --------------------------------------------


@dataclass
class GarakMultiTurnEvaluation(SingleEvaluation):
    """Evaluates a multi-turn garak probe against a model.

    Delegates the entire interaction to the garak probe's
    ``.probe()`` method.  This supports probes that dynamically
    generate prompts, use red-team models, or adapt based on
    target responses (e.g. ``atkgen.Tox``).

    The probe manages the full conversation flow.  Results are
    scored using the probe's primary detector.
    """

    probe_name: str | None = None

    def __init__(self, probe_name: str):
        super().__init__()
        self.probe_name = probe_name

    async def init(self):
        if not self.probe_name:
            raise ValueError("probe_name is required")
        self.status = EvaluationStatus.INITIALIZED

    async def execute_internal(self, model: Model) -> float:
        if not self.probe_name:
            raise ValueError("probe_name is required")

        return await asyncio.to_thread(
            self._execute_sync,
            model,
        )

    def _execute_sync(self, model: Model) -> float:
        sync_model = SyncModelWrapper(model)
        try:
            plugins = require_garak()._plugins
            _ensure_garak_config()
            probe = plugins.load_plugin(self.probe_name)
            detector_name = f"detectors.{probe.primary_detector}"
            detector = plugins.load_plugin(detector_name)

            adapter = _get_generator_adapter_cls()(sync_model)
            attempts = probe.probe(adapter)

            if not attempts:
                return 1.0

            self.session = _attempts_to_session(attempts)

            all_scores: list[float] = []
            for attempt in attempts:
                results = detector.detect(attempt)
                scores = [s for s in results if s is not None]
                if scores:
                    all_scores.append(max(scores))

            if not all_scores:
                score = 1.0
            else:
                worst_hit = max(all_scores)
                score = 1.0 - worst_hit
            logger.debug(
                "GarakMultiTurnEvaluation(%s): score=%.2f", self.probe_name, score
            )
            return score
        finally:
            sync_model.close()


@dataclass_json
@dataclass
class GarakProbeEvaluation(MultiEvaluation):
    """Evaluates a garak probe against a model.

    Automatically detects whether the probe provides static prompts
    (single-prompt evaluation, run in parallel) or requires dynamic
    interaction (multi-turn evaluation) and delegates accordingly.
    """

    probe_name: str | None = None

    def __init__(self, probe_name: str):
        super().__init__()
        self.probe_name = probe_name

    async def get_children(self) -> list[Evaluation]:
        if not self.probe_name:
            raise ValueError("probe_name is required")

        _ensure_garak_config()
        probe = require_garak()._plugins.load_plugin(self.probe_name)

        if _has_static_prompts(probe):
            detector_name = probe.primary_detector
            detector = GarakDetector(
                f"detectors.{detector_name}",
            )
            children = [
                SinglePromptEvaluation(
                    prompt=_garak_prompt_to_message(p),
                    detector=detector,
                )
                for p in probe.prompts
            ]
            logger.debug(
                "GarakProbeEvaluation(%s): %d children (%s)",
                self.probe_name,
                len(children),
                "static",
            )
            return children

        children = [
            GarakMultiTurnEvaluation(
                probe_name=self.probe_name,
            ),
        ]
        logger.debug(
            "GarakProbeEvaluation(%s): %d children (%s)",
            self.probe_name,
            len(children),
            "multi-turn",
        )
        return children
