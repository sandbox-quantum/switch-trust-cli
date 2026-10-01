"""Per-endpoint concurrency cap and initial jitter for HTTP-backed models.

Multiple :class:`Model` instances are built for the same tenant agent endpoint
(the worker rebuilds a fresh ``Model`` per job) so the semaphore is held on the
class rather than the instance: every instance sharing a URL shares one budget.

asyncio is single-threaded, so no lock is needed around the dict lookup — the
check-then-insert in :meth:`_get_semaphore` never awaits between the two ops.

Bucketing is by ``endpoint_url`` as passed by the factory, which uses
``db_model.host``. For HTTP model types whose real request URL is ``host +
endpoint`` (GenericHttp, LangServe), calls to distinct paths on the same host
share one budget — that matches "one tenant server" as a rate-limit unit.
"""

from __future__ import annotations

import asyncio
import random
from typing import TYPE_CHECKING, Any, ClassVar

from switch_trust.eval.core.models.model import Model, ModelResponse

if TYPE_CHECKING:
    from pydantic import BaseModel

    from switch_trust.eval.common.schema import Message


class PerEndpointLimiter(Model):
    """Bounds concurrent calls per endpoint URL and jitters the initial burst.

    Wraps an inner :class:`Model`. Delegates to the inner model's public
    :meth:`Model.generate` (not ``_generate``) so the outer ``generate``
    wrapper's ``TokenUsage`` accumulator dedupe (see ``core/models/model.py``
    ``TokenUsageAccumulator.record_once``) still runs at every layer.
    """

    _semaphores: ClassVar[dict[str, asyncio.Semaphore]] = {}

    def __init__(
        self,
        model: Model,
        endpoint_url: str,
        limit: int = 8,
        jitter_max: float = 1.0,
    ):
        self._model = model
        self._endpoint_url = endpoint_url
        self._limit = limit
        self._jitter_max = jitter_max

    def _get_semaphore(self) -> asyncio.Semaphore:
        sem = PerEndpointLimiter._semaphores.get(self._endpoint_url)
        if sem is None:
            sem = asyncio.Semaphore(self._limit)
            PerEndpointLimiter._semaphores[self._endpoint_url] = sem
        return sem

    async def _generate(
        self,
        messages: list[Message],
        *,
        output_schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        if self._jitter_max > 0:
            await asyncio.sleep(random.uniform(0, self._jitter_max))
        async with self._get_semaphore():
            return await self._model.generate(
                messages, output_schema=output_schema, **kwargs
            )

    @classmethod
    def _reset_state_for_testing(cls) -> None:
        # asyncio.Semaphore binds to the loop it was created on, and
        # IsolatedAsyncioTestCase spins up a fresh loop per test. Without
        # clearing the class-level dict, later tests would await on a
        # semaphore attached to a dead loop and hang.
        cls._semaphores.clear()
