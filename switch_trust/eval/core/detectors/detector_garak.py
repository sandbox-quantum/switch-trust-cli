from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING

from switch_trust.eval.common.schema import PartType, Role
from switch_trust.eval.core.detectors.detector import Detector, DetectorResult
from switch_trust.eval.core.models.model import ModelResponse
from switch_trust.eval.core.optional_deps import require_garak

if TYPE_CHECKING:
    from garak.attempt import Conversation

logger = logging.getLogger(__name__)

# Process-wide cache of loaded garak plugins, keyed by detector_name. Some
# detectors (e.g. the packagehallucination family) pull a multi-hundred-MB
# third-party package registry into memory on first use; a fresh GarakDetector
# is built per job/probe (see evaluation_garak_probe.py), so without this
# cache every job reloads and re-materializes that dataset from scratch,
# multiplying memory with concurrent and successive jobs instead of paying
# the cost once per worker process. Plugins are read-only after their first
# `detect()` call, so sharing one instance across jobs/threads is safe.
_plugin_cache: dict[str, object] = {}
_plugin_cache_lock = threading.Lock()


class GarakDetector(Detector):
    """A detector backed by a garak detector plugin."""

    def __init__(self, detector_name: str) -> None:
        self._detector_name = detector_name
        self._detector = None

    def _ensure_loaded(self):
        if self._detector is not None:
            return self._detector

        cached = _plugin_cache.get(self._detector_name)
        if cached is None:
            # detect() runs on a thread-pool thread (via asyncio.to_thread), so
            # concurrent jobs can race here; double-checked locking keeps the
            # plugin load (and its dataset fetch) to once per process.
            with _plugin_cache_lock:
                cached = _plugin_cache.get(self._detector_name)
                if cached is None:
                    logger.debug("Loading garak detector: %s", self._detector_name)
                    cached = require_garak()._plugins.load_plugin(
                        self._detector_name,
                    )
                    _plugin_cache[self._detector_name] = cached
        self._detector = cached
        return self._detector

    async def detect(self, response: ModelResponse) -> DetectorResult:
        return await asyncio.to_thread(
            self._detect_sync,
            response,
        )

    def _detect_sync(self, response: ModelResponse) -> DetectorResult:
        conversation = _create_conversation(response)
        attempt = require_garak().attempt.Attempt(prompt=conversation)
        results = self._ensure_loaded().detect(attempt)
        scores = [r for r in results if r is not None]
        if not scores:
            logger.debug("GarakDetector(%s): score=%.2f", self._detector_name, 1.0)
            return DetectorResult(score=1.0)
        max_hit = max(scores)
        logger.debug(
            "GarakDetector(%s): score=%.2f", self._detector_name, 1.0 - max_hit
        )
        return DetectorResult(score=1.0 - max_hit)


def _extract_text(response: ModelResponse) -> str:
    if response.message is None:
        return ""
    parts = response.message.content.parts
    text_parts = [
        part.text
        for part in parts
        if part.part_type == PartType.TEXT and part.text is not None
    ]
    return "\n".join(text_parts)


def _map_role(role: Role) -> str:
    if role == Role.ASSISTANT:
        return "assistant"
    if role == Role.USER:
        return "user"
    if role == Role.SYSTEM:
        return "system"
    return str(role.value)


def _create_conversation(response: ModelResponse) -> Conversation:
    garak = require_garak()
    turns = []
    if response.message is not None:
        role = _map_role(response.message.content.role)
        text = _extract_text(response)
        turns.append(
            garak.attempt.Turn(
                role=role,
                content=garak.attempt.Message(text=text),
            )
        )
    return garak.attempt.Conversation(turns)
