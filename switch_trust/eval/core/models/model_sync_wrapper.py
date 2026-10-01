"""Sync wrapper around an async Model for use in sync contexts.

Used by garak integration which requires synchronous model calls.
Creates its own event loop to bridge async-to-sync.
"""

import asyncio
import logging
from typing import Any

from switch_trust.eval.core.models.model import Model, ModelContent, ModelResponse

logger = logging.getLogger(__name__)


class SyncModelWrapper:
    """Wraps an async Model for use in sync code (e.g. garak)."""

    def __init__(self, model: Model):
        self._model = model
        self._loop = asyncio.new_event_loop()

    def generate(
        self,
        contents: ModelContent,
        **kwargs: Any,
    ) -> ModelResponse:
        logger.debug("SyncModelWrapper: bridging async model call")
        return self._loop.run_until_complete(
            self._model.generate(contents, **kwargs),
        )

    def close(self):
        self._loop.close()

    def __del__(self):
        if not self._loop.is_closed():
            self._loop.close()
