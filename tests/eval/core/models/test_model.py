import asyncio
import socket
import unittest

from switch_trust.eval.common.schema import Content, Message, Part, PartType, Role
from switch_trust.eval.core.models.model import (
    EndpointScopedModel,
    Model,
    ModelErrorAccumulator,
    ModelErrorKey,
    ModelResponse,
    ResponseStatus,
    flatten_messages,
    record_model_error,
    track_model_errors,
)
from switch_trust.eval.core.models.model_errors import (
    BLOCKED_STATUS,
    HOST_NOT_FOUND,
    ErrorKind,
    SsrfBlockedError,
    mark_generator_failure,
)
from switch_trust.eval.core.models.model_per_endpoint_limiter import (
    PerEndpointLimiter,
)
from switch_trust.eval.core.models.model_retry import RetryModel
from tests.eval.core.models.test_model_errors import dns_error


class StubModel(Model):
    """Concrete Model for testing the base class behavior."""

    def __init__(
        self,
        response: ModelResponse | None = None,
        error: Exception | None = None,
    ):
        self._response = response
        self._error = error

    async def _generate(self, messages, **kwargs):
        if self._error:
            raise self._error
        return self._response


class TestModelGenerate(unittest.TestCase):
    def test_generate_with_string_input(self):
        expected = ModelResponse(
            message=Message(
                content=Content.text(Role.ASSISTANT, "reply"),
            ),
        )
        model = StubModel(response=expected)
        result = asyncio.run(model.generate("hello"))
        self.assertEqual(result.status, ResponseStatus.OK)
        self.assertEqual(
            result.message.content.parts[0].text,
            "reply",
        )

    def test_generate_with_message_input(self):
        expected = ModelResponse(
            message=Message(
                content=Content.text(Role.ASSISTANT, "reply"),
            ),
        )
        model = StubModel(response=expected)
        msg = Message(content=Content.text(Role.USER, "hi"))
        result = asyncio.run(model.generate(msg))
        self.assertEqual(result.status, ResponseStatus.OK)

    def test_generate_with_message_list(self):
        expected = ModelResponse(
            message=Message(
                content=Content.text(Role.ASSISTANT, "reply"),
            ),
        )
        model = StubModel(response=expected)
        msgs = [Message(content=Content.text(Role.USER, "hi"))]
        result = asyncio.run(model.generate(msgs))
        self.assertEqual(result.status, ResponseStatus.OK)

    def test_generate_error_logs_and_reraises(self):
        error = ValueError("model failure")
        model = StubModel(error=error)
        with self.assertRaises(ValueError) as ctx:
            asyncio.run(model.generate("test"))
        self.assertIn("model failure", str(ctx.exception))

    def test_generate_non_ok_status(self):
        response = ModelResponse(
            message=Message(
                content=Content.text(Role.ASSISTANT, "blocked"),
            ),
            status=ResponseStatus.BLOCKED_SAFETY,
        )
        model = StubModel(response=response)
        result = asyncio.run(model.generate("test"))
        self.assertEqual(result.status, ResponseStatus.BLOCKED_SAFETY)

    def test_generate_none_message(self):
        response = ModelResponse(message=None)
        model = StubModel(response=response)
        result = asyncio.run(model.generate("test"))
        self.assertIsNone(result.message)


class TestModelGenerateLogLevel(unittest.IsolatedAsyncioTestCase):
    async def test_tenant_caused_failure_logs_as_warning(self):
        model = StubModel(error=RuntimeError("agent down"))
        with self.assertLogs(
            "switch_trust.eval.core.models.model", level="WARNING"
        ) as cm:
            with self.assertRaises(RuntimeError):
                await model.generate("hello")
        self.assertFalse(any(r.levelname == "ERROR" for r in cm.records))

    async def test_generator_marked_failure_logs_as_error(self):
        exc = RuntimeError("judge exploded")
        mark_generator_failure(exc)
        model = StubModel(error=exc)
        with self.assertLogs(
            "switch_trust.eval.core.models.model", level="ERROR"
        ) as cm:
            with self.assertRaises(RuntimeError):
                await model.generate("hello")
        self.assertTrue(any(r.levelname == "ERROR" for r in cm.records))


class _StatusError(Exception):
    """A stand-in for ModelHttpError (or aiohttp.ClientResponseError): any
    exception carrying a `.status` attribute."""

    def __init__(self, status: int | None):
        self.status = status
        super().__init__(f"status={status}")


class TestModelErrorAccumulatorRecordOnce(unittest.TestCase):
    def test_record_once_counts_the_same_exception_a_single_time(self):
        exc = _StatusError(429)
        acc = ModelErrorAccumulator()

        acc.record_once(exc, "VertexAgentRuntimeModel", 429)
        acc.record_once(exc, "VertexAgentRuntimeModel", 429)
        acc.record_once(exc, "VertexAgentRuntimeModel", 429)

        self.assertEqual(
            acc.counts(),
            {
                ModelErrorKey(
                    "VertexAgentRuntimeModel", "429", None, ErrorKind.UNKNOWN, None
                ): 1
            },
        )

    def test_record_once_counts_distinct_exceptions_separately(self):
        # Each exc must be a distinct, still-referenced object: CPython can
        # reuse the id() of an unreferenced object immediately, which would
        # make two *different* exceptions collide under identity-based dedup
        # -- a test-only hazard, not a production one, since a real exception
        # stays referenced by its own traceback/`except` frame for the
        # duration it takes to propagate.
        exc_a, exc_b, exc_c = _StatusError(429), _StatusError(429), _StatusError(500)
        acc = ModelErrorAccumulator()
        acc.record_once(exc_a, "VertexAgentRuntimeModel", 429)
        acc.record_once(exc_b, "VertexAgentRuntimeModel", 429)
        acc.record_once(exc_c, "GenericHttpModel", 500)

        self.assertEqual(
            acc.counts(),
            {
                ModelErrorKey(
                    "VertexAgentRuntimeModel", "429", None, ErrorKind.UNKNOWN, None
                ): 2,
                ModelErrorKey(
                    "GenericHttpModel", "500", None, ErrorKind.UNKNOWN, None
                ): 1,
            },
        )

    def test_none_status_recorded_as_a_bucket(self):
        acc = ModelErrorAccumulator()
        acc.record_once(TimeoutError(), "GenericHttpModel", None)
        self.assertEqual(
            acc.counts(),
            {
                ModelErrorKey(
                    "GenericHttpModel", "None", None, ErrorKind.UNKNOWN, None
                ): 1
            },
        )

    def test_error_status_distinguishes_otherwise_identical_keys(self):
        # Same model_type/status (429), different error_status -- e.g. a real
        # quota error vs. an ordinary rate limit under the same HTTP code.
        # This is the exact signal the motivating incident's log line
        # couldn't distinguish.
        exc_a, exc_b = _StatusError(429), _StatusError(429)
        acc = ModelErrorAccumulator()
        acc.record_once(exc_a, "VertexAgentRuntimeModel", 429, "RESOURCE_EXHAUSTED")
        acc.record_once(exc_b, "VertexAgentRuntimeModel", 429, None)

        self.assertEqual(
            acc.counts(),
            {
                ModelErrorKey(
                    "VertexAgentRuntimeModel",
                    "429",
                    "RESOURCE_EXHAUSTED",
                    ErrorKind.UNKNOWN,
                    None,
                ): 1,
                ModelErrorKey(
                    "VertexAgentRuntimeModel", "429", None, ErrorKind.UNKNOWN, None
                ): 1,
            },
        )


class _StatusCodeError(Exception):
    """A stand-in for the OpenAI/Anthropic SDK errors: the HTTP code is on
    `.status_code`, and there is no `.status` at all."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"status_code={status_code}")


class _HttpLikeError(Exception):
    """A stand-in for ModelHttpError: carries both `.status` and
    `.error_status`, as opposed to `_StatusError` which only has `.status`."""

    def __init__(self, status: int, error_status: str | None):
        self.status = status
        self.error_status = error_status
        super().__init__(f"status={status}, error_status={error_status}")


class _GeminiLikeError(Exception):
    """A stand-in for google.genai.errors.APIError: the HTTP code is on
    `.code`, the google.rpc string is on `.status`, and there is no
    `.error_status` — the inverse of `_HttpLikeError`'s layout."""

    def __init__(self, code: int, status: str):
        self.code = code
        self.status = status
        super().__init__(f"{code} {status}")


class TestModelErrorTracking(unittest.IsolatedAsyncioTestCase):
    async def test_records_host_not_found_as_its_status(self):
        model = StubModel(error=dns_error(socket.EAI_NONAME))

        with track_model_errors() as error_acc:
            with self.assertRaises(OSError):
                await model.generate("hello")

        self.assertEqual(
            error_acc.counts(),
            {
                ModelErrorKey(
                    "StubModel", HOST_NOT_FOUND, None, ErrorKind.UNREACHABLE, None
                ): 1
            },
        )

    async def test_records_ssrf_block_as_its_status(self):
        model = StubModel(error=SsrfBlockedError("internal"))

        with track_model_errors() as error_acc:
            with self.assertRaises(SsrfBlockedError):
                await model.generate("hello")

        self.assertEqual(
            error_acc.counts(),
            {
                ModelErrorKey(
                    "StubModel", BLOCKED_STATUS, None, ErrorKind.BLOCKED, None
                ): 1
            },
        )

    async def test_records_error_from_generate(self):
        model = StubModel(error=_StatusError(429))

        with track_model_errors() as error_acc:
            with self.assertRaises(_StatusError):
                await model.generate("hello")

        self.assertEqual(
            error_acc.counts(),
            {
                ModelErrorKey(
                    "StubModel", "429", None, ErrorKind.QUOTA_EXCEEDED, None
                ): 1
            },
        )

    async def test_records_error_status_from_generate(self):
        model = StubModel(error=_HttpLikeError(429, "RESOURCE_EXHAUSTED"))

        with track_model_errors() as error_acc:
            with self.assertRaises(_HttpLikeError):
                await model.generate("hello")

        self.assertEqual(
            error_acc.counts(),
            {
                ModelErrorKey(
                    "StubModel",
                    "429",
                    "RESOURCE_EXHAUSTED",
                    ErrorKind.QUOTA_EXCEEDED,
                    None,
                ): 1
            },
        )

    async def test_records_a_status_code_attribute(self):
        """The OpenAI/Anthropic SDK shape: the code is on `status_code`.

        Reading only `.status` recorded the literal "None" for every failure
        from those two providers, and the consumers sub-split on this value —
        `eval_cli._check_model_credentials` reads a statusless `unreachable`
        as a transient network failure (60, retry) rather than as the wrong
        endpoint or model name it was (24).
        """
        model = StubModel(error=_StatusCodeError(404))

        with track_model_errors() as error_acc:
            with self.assertRaises(_StatusCodeError):
                await model.generate("hello")

        self.assertEqual(
            error_acc.counts(),
            {ModelErrorKey("StubModel", "404", None, ErrorKind.UNREACHABLE, None): 1},
        )

    async def test_records_both_halves_of_a_gemini_error(self):
        """The Gemini shape: the code is on `.code` and the google.rpc string
        on `.status`, so the two fields come from two different readers.

        `classify_model_error` reads both — the string is what separates a
        real quota error from an ordinary rate limit under the same 429 — so
        recording either half its own way puts a key in the accumulator that
        disagrees with the kind stored beside it. Reading `.error_status`,
        which this SDK never sets, dropped the string from the run summary
        and from the `error_status` metric attribute entirely.
        """
        model = StubModel(error=_GeminiLikeError(429, "RESOURCE_EXHAUSTED"))

        with track_model_errors() as error_acc:
            with self.assertRaises(_GeminiLikeError):
                await model.generate("hello")

        self.assertEqual(
            error_acc.counts(),
            {
                ModelErrorKey(
                    "StubModel",
                    "429",
                    "RESOURCE_EXHAUSTED",
                    ErrorKind.QUOTA_EXCEEDED,
                    None,
                ): 1
            },
        )

    async def test_a_statusless_failure_still_records_none(self):
        """A bare connection refusal carries no code anywhere, and the "None"
        it stringifies to is what `_NETWORK_FAILURE_STATUSES` matches on."""
        model = StubModel(error=ConnectionRefusedError("refused"))

        with track_model_errors() as error_acc:
            with self.assertRaises(ConnectionRefusedError):
                await model.generate("hello")

        statuses = {key.status for key in error_acc.counts()}
        self.assertEqual(statuses, {"None"})

    async def test_success_leaves_accumulator_empty(self):
        response = ModelResponse(
            message=Message(content=Content.text(Role.ASSISTANT, "reply")),
        )
        model = StubModel(response=response)

        with track_model_errors() as error_acc:
            await model.generate("hello")

        self.assertEqual(error_acc.counts(), {})

    async def test_calls_outside_scope_are_not_recorded(self):
        model = StubModel(error=_StatusError(429))

        # No track_model_errors() scope active: generate() must still raise
        # normally even though there's nowhere to record the failure.
        with self.assertRaises(_StatusError):
            await model.generate("hello")

    async def test_scopes_do_not_leak_into_each_other(self):
        with track_model_errors() as first:
            with self.assertRaises(_StatusError):
                await StubModel(error=_StatusError(429)).generate("hello")
        with track_model_errors() as second:
            pass

        self.assertEqual(
            first.counts(),
            {
                ModelErrorKey(
                    "StubModel", "429", None, ErrorKind.QUOTA_EXCEEDED, None
                ): 1
            },
        )
        self.assertEqual(second.counts(), {})

    async def test_retry_wrapped_failure_counted_once_not_twice_by_inner_class(self):
        # RetryModel exhausts retries and re-raises the same exception object
        # through its own Model.generate() layer. Deduped by id(exc), so this
        # must record the inner StubModel's failure once, not once per layer.
        model = RetryModel(StubModel(error=_StatusError(500)), max_retries=0)

        with track_model_errors() as error_acc:
            with self.assertRaises(_StatusError):
                await model.generate("hello")

        self.assertEqual(
            error_acc.counts(),
            {ModelErrorKey("StubModel", "500", None, ErrorKind.UNREACHABLE, None): 1},
        )


class TestEndpointAttribution(unittest.IsolatedAsyncioTestCase):
    """CHOO-2934 follow-up: a failure must name the endpoint it went to, so two
    agents failing in one run stay distinguishable."""

    def setUp(self) -> None:
        PerEndpointLimiter._reset_state_for_testing()

    async def test_limiter_names_the_endpoint_on_the_surviving_record(self):
        # The limiter and the inner model both reach the choke point with the
        # same exception; dedupe keeps the inner one, which is the recording
        # that has to carry the URL.
        model = PerEndpointLimiter(
            StubModel(error=_StatusError(500)), "https://agent.test", jitter_max=0
        )

        with track_model_errors() as error_acc:
            with self.assertRaises(_StatusError):
                await model.generate("hello")

        self.assertEqual(
            error_acc.counts(),
            {
                ModelErrorKey(
                    "StubModel",
                    "500",
                    None,
                    ErrorKind.UNREACHABLE,
                    "https://agent.test",
                ): 1
            },
        )

    async def test_two_endpoints_do_not_merge_into_one_bucket(self):
        a = PerEndpointLimiter(
            StubModel(error=_StatusError(500)), "https://a.test", jitter_max=0
        )
        b = PerEndpointLimiter(
            StubModel(error=_StatusError(500)), "https://b.test", jitter_max=0
        )

        with track_model_errors() as error_acc:
            for model in (a, b):
                with self.assertRaises(_StatusError):
                    await model.generate("hello")

        self.assertEqual(
            {key.endpoint_url for key in error_acc.counts()},
            {"https://a.test", "https://b.test"},
        )

    async def test_scoped_model_names_the_endpoint(self):
        model = EndpointScopedModel(
            StubModel(error=_StatusError(500)), "https://vllm.test"
        )

        with track_model_errors() as error_acc:
            with self.assertRaises(_StatusError):
                await model.generate("hello")

        [key] = error_acc.counts()
        self.assertEqual(key.model_type, "StubModel")
        self.assertEqual(key.endpoint_url, "https://vllm.test")

    async def test_unwrapped_model_records_no_endpoint(self):
        # The judge is SDK-backed, so it is never wrapped. A null URL is what
        # keeps our failure separable from the tenant agent's.
        with track_model_errors() as error_acc:
            with self.assertRaises(_StatusError):
                await StubModel(error=_StatusError(500)).generate("hello")

        [key] = error_acc.counts()
        self.assertIsNone(key.endpoint_url)

    async def test_scope_does_not_leak_past_the_limited_call(self):
        limited = PerEndpointLimiter(
            StubModel(error=_StatusError(500)), "https://a.test", jitter_max=0
        )

        with track_model_errors() as error_acc:
            with self.assertRaises(_StatusError):
                await limited.generate("hello")
            with self.assertRaises(_StatusError):
                await StubModel(error=_StatusError(429)).generate("hello")

        by_kind = {key.kind: key.endpoint_url for key in error_acc.counts()}
        self.assertEqual(by_kind[ErrorKind.UNREACHABLE], "https://a.test")
        self.assertIsNone(by_kind[ErrorKind.QUOTA_EXCEEDED])

    async def test_dominant_kind_survives_the_endpoint_split(self):
        # Splitting by URL multiplies the key space, so the CHOO-2833 path has
        # to keep summing across endpoints rather than per bucket.
        agent = PerEndpointLimiter(
            StubModel(error=_StatusError(401)), "https://a.test", jitter_max=0
        )
        other = PerEndpointLimiter(
            StubModel(error=_StatusError(401)), "https://b.test", jitter_max=0
        )

        with track_model_errors() as error_acc:
            for model in (agent, other):
                with self.assertRaises(_StatusError):
                    await model.generate("hello")
            record_model_error(_StatusError(500), "judge", 500)

        self.assertEqual(error_acc.dominant_kind(), ErrorKind.AUTH_FAILED)


class TestLastErrorCapture(unittest.TestCase):
    """The stored message is persisted and read back by the tenant, so it is
    redacted and bounded at the source rather than downstream."""

    def test_secret_in_a_provider_error_does_not_survive(self):
        acc = ModelErrorAccumulator()
        acc.record_once(
            Exception("rejected: Bearer abc123DEF456ghi789"), "GenericHttpModel", 401
        )

        [message] = acc.last_errors().values()
        self.assertNotIn("abc123DEF456ghi789", message)
        self.assertIn("[REDACTED]", message)

    def test_oversized_message_is_truncated(self):
        acc = ModelErrorAccumulator()
        acc.record_once(Exception("x" * 5000), "GenericHttpModel", 500)

        [message] = acc.last_errors().values()
        self.assertEqual(len(message), 500)

    def test_message_names_the_exception_type(self):
        acc = ModelErrorAccumulator()
        acc.record_once(TimeoutError("took too long"), "GenericHttpModel", None)

        [message] = acc.last_errors().values()
        self.assertEqual(message, "TimeoutError: took too long")

    def test_no_failures_means_no_messages(self):
        self.assertEqual(ModelErrorAccumulator().last_errors(), {})


class TestRecordModelError(unittest.TestCase):
    def test_records_onto_the_open_scope(self):
        exc = _StatusError(429)
        mark_generator_failure(exc)

        with track_model_errors() as error_acc:
            record_model_error(exc, "judge", exc.status)

        self.assertEqual(
            error_acc.counts(),
            {ModelErrorKey("judge", "429", None, ErrorKind.INTERNAL_ERROR, None): 1},
        )

    def test_noop_outside_a_scope(self):
        # Nothing to assert on besides "doesn't raise" -- there's no
        # accumulator to have recorded onto.
        record_model_error(_StatusError(429), "judge")


class TestFlattenMessages(unittest.TestCase):
    def test_single_message(self):
        msgs = [Message(content=Content.text(Role.USER, "hello world"))]
        self.assertEqual(flatten_messages(msgs), "hello world")

    def test_multiple_messages(self):
        msgs = [
            Message(content=Content.text(Role.USER, "Hello")),
            Message(content=Content.text(Role.ASSISTANT, "Hi!")),
            Message(content=Content.text(Role.USER, "How are you?")),
        ]
        result = flatten_messages(msgs)
        self.assertIn("USER: Hello", result)
        self.assertIn("ASSISTANT: Hi!", result)
        self.assertIn("USER: How are you?", result)

    def test_single_message_multiple_parts(self):
        msg = Message(
            content=Content(
                role=Role.USER,
                parts=[
                    Part(part_type=PartType.TEXT, text="part one"),
                    Part(part_type=PartType.TEXT, text="part two"),
                ],
            ),
        )
        result = flatten_messages([msg])
        self.assertEqual(result, "part one part two")


class TestFailureLoggedOncePerException(unittest.IsolatedAsyncioTestCase):
    """A failure is logged by the layer that saw it first, not by every layer.

    Decorator models re-raise the same exception object through each wrapper's
    own Model.generate(), so a single failed call used to emit one log line per
    layer. CHOO-3512.
    """

    async def test_decorator_layers_do_not_relog(self) -> None:
        inner = _RaisingModel(lambda: ConnectionError("agent unreachable"))
        wrapped = _real_stack(inner)

        with self.assertLogs(
            "switch_trust.eval.core.models.model", level="WARNING"
        ) as captured:
            with self.assertRaises(ConnectionError):
                await wrapped.generate("hi")

        # Innermost wins: the concrete class that failed, not a wrapper.
        assert len(captured.records) == 1
        assert "_RaisingModel" in captured.records[0].getMessage()

    async def test_a_second_distinct_failure_still_logs(self) -> None:
        # Dedupe is per exception object -- the same semantics as
        # ModelErrorAccumulator.record_once -- so two real failures, which
        # raise two different objects, must still produce two lines.
        inner = _RaisingModel(lambda: ConnectionError("agent unreachable"))
        with self.assertLogs(
            "switch_trust.eval.core.models.model", level="WARNING"
        ) as captured:
            for _ in range(2):
                with self.assertRaises(ConnectionError):
                    await inner.generate("hi")
        assert len(captured.records) == 2

    async def test_non_ok_status_logs_once_across_layers(self) -> None:
        with self.assertLogs(
            "switch_trust.eval.core.models.model", level="WARNING"
        ) as captured:
            response = await _real_stack(_StatusModel()).generate("hi")

        assert response.status == ResponseStatus.EMPTY_RESPONSE
        assert len(captured.records) == 1
        assert "_StatusModel" in captured.records[0].getMessage()


class _RaisingModel(Model):
    """Raises a freshly built exception per call, as a real adapter does."""

    def __init__(self, make_exc) -> None:
        super().__init__()
        self._make_exc = make_exc

    async def _generate(self, messages, *, output_schema=None, **kwargs):
        raise self._make_exc()


class _StatusModel(Model):
    """Returns a non-OK response, as an adapter does for an empty reply."""

    async def _generate(self, messages, *, output_schema=None, **kwargs):
        return ModelResponse(message=None, status=ResponseStatus.EMPTY_RESPONSE)


def _real_stack(inner: Model) -> Model:
    """The worker's decorator stack, without retries or jitter to keep it fast."""
    return RetryModel(
        PerEndpointLimiter(inner, "http://agent.test", jitter_max=0), max_retries=0
    )


if __name__ == "__main__":
    unittest.main()
