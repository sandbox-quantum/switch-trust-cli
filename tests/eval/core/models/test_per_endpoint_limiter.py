import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from switch_trust.eval.common.schema import Content, Message, Part, PartType, Role
from switch_trust.eval.core.models.model import ModelResponse
from switch_trust.eval.core.models.model_per_endpoint_limiter import (
    PerEndpointLimiter,
)


def _make_message():
    return Message(
        content=Content(
            role=Role.USER,
            parts=[Part(part_type=PartType.TEXT, text="hello")],
        ),
    )


def _make_response():
    return ModelResponse(message=None)


class TestPerEndpointLimiter(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        PerEndpointLimiter._reset_state_for_testing()

    def tearDown(self) -> None:
        PerEndpointLimiter._reset_state_for_testing()

    async def test_semaphore_bounds_concurrent_calls(self):
        gate = asyncio.Event()
        in_flight = 0
        max_in_flight = 0

        async def inner_generate(*_args, **_kwargs):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await gate.wait()
            in_flight -= 1
            return _make_response()

        inner = MagicMock()
        inner.generate = AsyncMock(side_effect=inner_generate)
        limiter = PerEndpointLimiter(
            inner, endpoint_url="http://a", limit=3, jitter_max=0.0
        )

        tasks = [
            asyncio.create_task(limiter.generate(_make_message())) for _ in range(20)
        ]
        # Let tasks settle at the semaphore.
        await asyncio.sleep(0.05)
        self.assertLessEqual(max_in_flight, 3)
        gate.set()
        await asyncio.gather(*tasks)
        self.assertEqual(inner.generate.call_count, 20)

    async def test_shared_semaphore_across_instances_same_url(self):
        inner_a = MagicMock()
        inner_a.generate = AsyncMock(return_value=_make_response())
        inner_b = MagicMock()
        inner_b.generate = AsyncMock(return_value=_make_response())

        a = PerEndpointLimiter(inner_a, endpoint_url="http://a", jitter_max=0.0)
        b = PerEndpointLimiter(inner_b, endpoint_url="http://a", jitter_max=0.0)
        await a.generate(_make_message())
        await b.generate(_make_message())
        self.assertEqual(len(PerEndpointLimiter._semaphores), 1)
        self.assertIs(a._get_semaphore(), b._get_semaphore())

    async def test_distinct_semaphores_for_distinct_urls(self):
        inner = MagicMock()
        inner.generate = AsyncMock(return_value=_make_response())
        a = PerEndpointLimiter(inner, endpoint_url="http://a", jitter_max=0.0)
        b = PerEndpointLimiter(inner, endpoint_url="http://b", jitter_max=0.0)
        await a.generate(_make_message())
        await b.generate(_make_message())
        self.assertEqual(len(PerEndpointLimiter._semaphores), 2)
        self.assertIsNot(a._get_semaphore(), b._get_semaphore())

    @patch(
        "switch_trust.eval.core.models.model_per_endpoint_limiter.random.uniform",
        return_value=0.5,
    )
    @patch(
        "switch_trust.eval.core.models.model_per_endpoint_limiter.asyncio.sleep",
        new_callable=AsyncMock,
    )
    async def test_initial_jitter_applied_before_acquire(self, mock_sleep, _):
        inner = MagicMock()
        inner.generate = AsyncMock(return_value=_make_response())
        limiter = PerEndpointLimiter(
            inner, endpoint_url="http://a", limit=1, jitter_max=1.0
        )
        await limiter.generate(_make_message())
        mock_sleep.assert_called_once_with(0.5)

    @patch(
        "switch_trust.eval.core.models.model_per_endpoint_limiter.asyncio.sleep",
        new_callable=AsyncMock,
    )
    async def test_no_jitter_when_zero(self, mock_sleep):
        inner = MagicMock()
        inner.generate = AsyncMock(return_value=_make_response())
        limiter = PerEndpointLimiter(
            inner, endpoint_url="http://a", limit=1, jitter_max=0.0
        )
        await limiter.generate(_make_message())
        mock_sleep.assert_not_called()

    async def test_wraps_public_generate_not_underscore_generate(self):
        # The token-usage dedupe contract in Model.record_once requires
        # decorators to call the wrapped model's public generate(), not
        # _generate(). Assert we do.
        inner = MagicMock()
        inner.generate = AsyncMock(return_value=_make_response())
        inner._generate = AsyncMock(return_value=_make_response())
        limiter = PerEndpointLimiter(inner, endpoint_url="http://a", jitter_max=0.0)
        await limiter.generate(_make_message())
        inner.generate.assert_awaited_once()
        inner._generate.assert_not_awaited()
