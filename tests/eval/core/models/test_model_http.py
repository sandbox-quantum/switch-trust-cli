import unittest
from unittest.mock import AsyncMock, MagicMock

from switch_trust.eval.core.models.model_http import (
    ModelHttpError,
    _extract_error_status,
    _same_project_as_url,
    raise_for_status,
)

VERTEX_URL = (
    "https://us-central1-aiplatform.googleapis.com/v1/projects/740550221362/"
    "locations/us-central1/reasoningEngines/123:streamQuery"
)


def _mock_response(status: int, body: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.url = "http://example.com" if status < 400 else VERTEX_URL
    resp.charset = "utf-8"
    resp.content.read = AsyncMock(return_value=body.encode("utf-8"))
    return resp


class TestExtractErrorStatus(unittest.TestCase):
    def test_known_status_extracted(self):
        body = (
            '{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "x"}}'
        )
        self.assertEqual(_extract_error_status(body), "RESOURCE_EXHAUSTED")

    def test_unrecognized_status_dropped(self):
        body = (
            '{"error": {"status": "SOMETHING_A_TENANT_SERVER_MADE_UP", "message": "x"}}'
        )
        self.assertIsNone(_extract_error_status(body))

    def test_non_json_body(self):
        self.assertIsNone(_extract_error_status("not json at all"))

    def test_json_without_error_envelope(self):
        self.assertIsNone(_extract_error_status('{"detail": "bad request"}'))


class TestSameProjectAsUrl(unittest.TestCase):
    def test_matching_project(self):
        body = '{"error": {"message": "... for consumer \'project_number:740550221362\'."}}'
        self.assertTrue(_same_project_as_url(body, VERTEX_URL))

    def test_different_project(self):
        body = '{"error": {"message": "... for consumer \'project_number:999999\'."}}'
        self.assertFalse(_same_project_as_url(body, VERTEX_URL))

    def test_url_without_project_number(self):
        self.assertIsNone(_same_project_as_url("anything", "http://example.com/run"))


class TestRaiseForStatus(unittest.IsolatedAsyncioTestCase):
    async def test_success_never_reads_body(self):
        resp = _mock_response(200)
        await raise_for_status(resp)
        resp.content.read.assert_not_called()

    async def test_error_raises_model_http_error(self):
        body = (
            '{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", '
            '"message": "quota exceeded for consumer \'project_number:740550221362\'"}}'
        )
        resp = _mock_response(429, body)
        with self.assertRaises(ModelHttpError) as ctx:
            await raise_for_status(resp)
        exc = ctx.exception
        self.assertEqual(exc.status, 429)
        self.assertEqual(exc.url, VERTEX_URL)
        self.assertEqual(exc.error_status, "RESOURCE_EXHAUSTED")
        self.assertTrue(exc.same_project_as_url)

    async def test_query_and_fragment_stripped_from_url(self):
        resp = _mock_response(429, "")
        resp.url = VERTEX_URL + "?api_key=super-secret#frag"
        with self.assertRaises(ModelHttpError) as ctx:
            await raise_for_status(resp)
        self.assertEqual(ctx.exception.url, VERTEX_URL)

    async def test_duck_types_status_attribute(self):
        """model_retry.py's _is_transient reads .status/.status_code/.code via
        getattr -- confirm ModelHttpError satisfies that without any change
        there."""
        resp = _mock_response(503, "")
        with self.assertRaises(ModelHttpError) as ctx:
            await raise_for_status(resp)
        self.assertEqual(getattr(ctx.exception, "status", None), 503)


class TestNoBodyContentLeak(unittest.TestCase):
    """The safety property this module exists for: no response-body content
    (echoed prompts, tenant server internals) may ever reach the exception's
    string form or its attributes."""

    def test_canary_never_appears_in_str_or_attrs(self):
        canary = "SECRET_TENANT_CONTENT_1234"
        body = (
            '{"error": {"status": "INTERNAL", '
            f'"message": "failed while processing: {canary}"}}'
        )
        exc = ModelHttpError(
            500,
            VERTEX_URL,
            _extract_error_status(body),
            _same_project_as_url(body, VERTEX_URL),
        )
        self.assertNotIn(canary, str(exc))
        for value in vars(exc).values():
            self.assertNotIn(canary, str(value))

    def test_arbitrary_html_error_page_leaks_nothing(self):
        body = (
            "<html><body>Internal Server Error: /Users/tenant/app.py:42</body></html>"
        )
        exc = ModelHttpError(
            500,
            "http://tenant-server.example.com/run",
            _extract_error_status(body),
            _same_project_as_url(body, "http://tenant-server.example.com/run"),
        )
        self.assertNotIn("app.py", str(exc))
        self.assertIsNone(exc.error_status)


if __name__ == "__main__":
    unittest.main()
