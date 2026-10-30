import logging
import socket
import unittest
from unittest.mock import MagicMock

import aiohttp
import anthropic
import httpx
import openai

from switch_trust.eval.core.models.model_errors import (
    ErrorKind,
    SsrfBlockedError,
    classify_model_error,
    generator_scope,
    in_generator_scope,
    is_generator_failure,
    is_host_not_found,
    log_level,
    log_level_for_run_failure,
    mark_generator_failure,
    status_of,
)


class _StatusError(Exception):
    def __init__(self, status: int):
        self.status = status
        super().__init__(f"status={status}")


class _HttpLikeError(Exception):
    """Stand-in for ModelHttpError: carries `.status` and `.error_status`."""

    def __init__(self, status: int, error_status: str | None):
        self.status = status
        self.error_status = error_status
        super().__init__(f"{status}/{error_status}")


class _GeminiLikeError(Exception):
    """Stand-in for google.genai.errors.APIError: the numeric HTTP status is
    `.code`, and the google.rpc status string is `.status` -- there is no
    `.error_status`."""

    def __init__(self, code: int, status: str):
        self.code = code
        self.status = status
        super().__init__(f"{code} {status}")


def dns_error(errno: int) -> aiohttp.ClientConnectorDNSError:
    """What aiohttp raises when the resolver's getaddrinfo() fails."""
    key = MagicMock(host="dead.trycloudflare.com", port=443, ssl=True)
    return aiohttp.ClientConnectorDNSError(key, socket.gaierror(errno, "lookup failed"))


class TestIsHostNotFound(unittest.TestCase):
    def test_nxdomain_is_host_not_found(self):
        assert is_host_not_found(dns_error(socket.EAI_NONAME)) is True

    def test_transient_lookup_failure_is_not(self):
        assert is_host_not_found(dns_error(socket.EAI_AGAIN)) is False

    def test_plain_connection_error_is_not(self):
        assert is_host_not_found(ConnectionRefusedError("refused")) is False


class TestStatusOf(unittest.TestCase):
    def test_reads_the_first_integer_attribute_present(self):
        self.assertEqual(status_of(_StatusError(503)), 503)

    def test_none_when_no_status_attribute(self):
        self.assertIsNone(status_of(ValueError("nope")))

    def test_ignores_a_non_integer_status(self):
        exc = Exception()
        exc.status = "429"
        self.assertIsNone(status_of(exc))


class TestClassifyModelError(unittest.TestCase):
    def test_quota_from_status(self):
        self.assertIs(classify_model_error(_StatusError(429)), ErrorKind.QUOTA_EXCEEDED)

    def test_auth_from_status(self):
        self.assertIs(classify_model_error(_StatusError(401)), ErrorKind.AUTH_FAILED)
        self.assertIs(classify_model_error(_StatusError(403)), ErrorKind.AUTH_FAILED)

    def test_unreachable_from_server_error_and_not_found(self):
        self.assertIs(classify_model_error(_StatusError(503)), ErrorKind.UNREACHABLE)
        self.assertIs(classify_model_error(_StatusError(404)), ErrorKind.UNREACHABLE)

    def test_other_4xx_is_a_config_problem(self):
        # The endpoint answered and rejected the request shape.
        self.assertIs(classify_model_error(_StatusError(400)), ErrorKind.INVALID_CONFIG)

    def test_timeout_wins_over_the_oserror_branch(self):
        # TimeoutError subclasses OSError, which would otherwise read as
        # unreachable.
        self.assertIs(classify_model_error(TimeoutError()), ErrorKind.TIMEOUT)

    def test_connection_error_without_a_status(self):
        self.assertIs(
            classify_model_error(ConnectionError("refused")), ErrorKind.UNREACHABLE
        )

    def test_ssrf_guard_refusal_is_blocked(self):
        assert classify_model_error(SsrfBlockedError("internal")) == ErrorKind.BLOCKED

    def test_ssrf_refusal_is_not_transient(self):
        # RetryModel and the worker retry OSError-family exceptions; a blocked
        # request must fail once.
        assert not isinstance(SsrfBlockedError("x"), OSError)

    def test_unclassifiable_is_unknown(self):
        self.assertIs(classify_model_error(ValueError("huh")), ErrorKind.UNKNOWN)

    def test_error_status_separates_quota_from_a_plain_rate_limit(self):
        # Both arrive as 429; only the google.rpc code says which it is. Here
        # they agree, but DEADLINE_EXCEEDED below shows the override working.
        self.assertIs(
            classify_model_error(_HttpLikeError(429, "RESOURCE_EXHAUSTED")),
            ErrorKind.QUOTA_EXCEEDED,
        )

    def test_error_status_overrides_the_http_status(self):
        # A 504 alone reads as unreachable; DEADLINE_EXCEEDED says timeout.
        self.assertIs(
            classify_model_error(_HttpLikeError(500, "DEADLINE_EXCEEDED")),
            ErrorKind.TIMEOUT,
        )

    def test_unmapped_error_status_falls_back_to_the_http_status(self):
        self.assertIs(
            classify_model_error(_HttpLikeError(401, "INTERNAL")),
            ErrorKind.AUTH_FAILED,
        )

    def test_gemini_style_status_attribute_is_read_as_error_status(self):
        # google.genai.errors.APIError has no .error_status; the google.rpc
        # code lives under .status instead (a string, not the numeric code,
        # which is .code).
        self.assertIs(
            classify_model_error(_GeminiLikeError(429, "RESOURCE_EXHAUSTED")),
            ErrorKind.QUOTA_EXCEEDED,
        )

    def test_gemini_unauthenticated_classifies_as_auth_failed(self):
        self.assertIs(
            classify_model_error(_GeminiLikeError(401, "UNAUTHENTICATED")),
            ErrorKind.AUTH_FAILED,
        )

    def test_gemini_invalid_api_key_classifies_as_invalid_config(self):
        # Google reports an invalid API key as INVALID_ARGUMENT at the top
        # level (the more specific API_KEY_INVALID reason is nested deeper
        # and not read), so this lands on invalid_config, not auth_failed.
        self.assertIs(
            classify_model_error(_GeminiLikeError(400, "INVALID_ARGUMENT")),
            ErrorKind.INVALID_CONFIG,
        )


class TestClassifySdkTransportErrors(unittest.TestCase):
    """Transport failures as the OpenAI/Anthropic SDKs actually raise them.

    Built from the installed SDKs rather than mimicked by a local stand-in:
    the whole difficulty is the shape of the real exception tree, and a
    hand-written copy would keep passing after the real one moved.
    """

    def setUp(self):
        self.request = httpx.Request("POST", "https://api.example.com/v1/chat")

    def test_openai_timeout_is_a_timeout(self):
        exc = openai.APITimeoutError(request=self.request)
        self.assertIs(classify_model_error(exc), ErrorKind.TIMEOUT)

    def test_anthropic_timeout_is_a_timeout(self):
        exc = anthropic.APITimeoutError(request=self.request)
        self.assertIs(classify_model_error(exc), ErrorKind.TIMEOUT)

    def test_openai_connection_error_is_unreachable(self):
        exc = openai.APIConnectionError(request=self.request)
        self.assertIs(classify_model_error(exc), ErrorKind.UNREACHABLE)

    def test_anthropic_connection_error_is_unreachable(self):
        exc = anthropic.APIConnectionError(request=self.request)
        self.assertIs(classify_model_error(exc), ErrorKind.UNREACHABLE)

    def test_every_other_branch_misses_these(self):
        # The premise of the name table. If a future SDK release makes these
        # derive from TimeoutError/OSError or carry a status, the table is no
        # longer what classifies them and can go.
        for exc in (
            openai.APITimeoutError(request=self.request),
            anthropic.APITimeoutError(request=self.request),
        ):
            with self.subTest(sdk=type(exc).__module__):
                self.assertNotIsInstance(exc, TimeoutError)
                self.assertNotIsInstance(exc, OSError)
                self.assertIsNone(status_of(exc))

    def test_a_status_carrying_sdk_error_is_still_read_from_its_status(self):
        # `APIStatusError` subclasses do carry one, and it is the better
        # signal — the name table must not shadow it.
        exc = openai.AuthenticationError(
            "bad key",
            response=httpx.Response(401, request=self.request),
            body=None,
        )
        self.assertIs(classify_model_error(exc), ErrorKind.AUTH_FAILED)

    def test_a_judge_timeout_is_still_ours(self):
        # The generator stamp wins: our judge timing out is an internal error,
        # not the tenant's agent being slow.
        exc = openai.APITimeoutError(request=self.request)
        mark_generator_failure(exc)
        self.assertIs(classify_model_error(exc), ErrorKind.INTERNAL_ERROR)


class TestLogLevel(unittest.TestCase):
    def test_internal_error_alerts(self):
        self.assertEqual(log_level(ErrorKind.INTERNAL_ERROR), logging.ERROR)

    def test_every_other_kind_is_a_warning(self):
        for kind in ErrorKind:
            if kind == ErrorKind.INTERNAL_ERROR:
                continue
            self.assertEqual(log_level(kind), logging.WARNING)


class TestLogLevelForRunFailure(unittest.TestCase):
    def test_confidently_tenant_caused_kinds_are_warnings(self):
        for kind in (
            ErrorKind.TIMEOUT,
            ErrorKind.BLOCKED,
            ErrorKind.AUTH_FAILED,
            ErrorKind.QUOTA_EXCEEDED,
            ErrorKind.UNREACHABLE,
            ErrorKind.INVALID_CONFIG,
            ErrorKind.PLAN_LIMIT_EXCEEDED,
        ):
            self.assertEqual(log_level_for_run_failure(kind), logging.WARNING)

    def test_internal_error_alerts(self):
        self.assertEqual(
            log_level_for_run_failure(ErrorKind.INTERNAL_ERROR), logging.ERROR
        )

    def test_unknown_alerts(self):
        # Unlike log_level(), this scope also wraps our own detector/scoring
        # code, so an unclassifiable failure defaults to ERROR, not WARNING.
        self.assertEqual(log_level_for_run_failure(ErrorKind.UNKNOWN), logging.ERROR)


class TestGeneratorAttribution(unittest.TestCase):
    def test_scope_is_off_by_default(self):
        self.assertFalse(in_generator_scope())

    def test_scope_resets_on_exit(self):
        with generator_scope():
            self.assertTrue(in_generator_scope())
        self.assertFalse(in_generator_scope())

    def test_a_marked_failure_is_ours_whatever_its_status(self):
        # Our judge key hitting its quota must not be reported to the customer
        # as their agent being rate-limited.
        exc = _StatusError(429)
        mark_generator_failure(exc)
        self.assertTrue(is_generator_failure(exc))
        self.assertIs(classify_model_error(exc), ErrorKind.INTERNAL_ERROR)

    def test_an_unmarked_failure_is_the_tenants(self):
        self.assertIs(classify_model_error(_StatusError(429)), ErrorKind.QUOTA_EXCEEDED)


if __name__ == "__main__":
    unittest.main()
