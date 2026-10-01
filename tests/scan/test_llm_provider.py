"""
Tests for llm_provider.py — LLM configuration and provider creation.
"""

import os
import unittest
from unittest.mock import patch

from switch_trust.scan import ADKModel
from switch_trust.scan.llm_provider import (
    DEFAULT_MODEL,
    PROVIDER_GOOGLE,
    _resolve_model_string,
    _safe_error,
    make_model,
)


class TestResolveModelString(unittest.TestCase):
    @patch.dict(os.environ, {}, clear=True)
    def test_defaults_to_google_gemini(self):
        provider, model = _resolve_model_string(None)
        self.assertEqual(provider, PROVIDER_GOOGLE)
        self.assertEqual(model, DEFAULT_MODEL)

    def test_explicit_string(self):
        provider, model = _resolve_model_string("openai:gpt-4o")
        self.assertEqual(provider, "openai")
        self.assertEqual(model, "gpt-4o")

    def test_bare_provider_gets_default_model(self):
        provider, model = _resolve_model_string("anthropic")
        self.assertEqual(provider, "anthropic")
        self.assertEqual(model, DEFAULT_MODEL)

    @patch.dict(os.environ, {"SCANNER_MODEL": "google:gemini-2.5-pro"})
    def test_env_var_fallback(self):
        provider, model = _resolve_model_string(None)
        self.assertEqual(provider, "google")
        self.assertEqual(model, "gemini-2.5-pro")

    def test_explicit_overrides_env(self):
        with patch.dict(os.environ, {"SCANNER_MODEL": "google:gemini-flash"}):
            provider, model = _resolve_model_string("openai:gpt-4o")
            self.assertEqual(provider, "openai")
            self.assertEqual(model, "gpt-4o")

    @patch.dict(os.environ, {}, clear=True)
    def test_empty_string_defaults(self):
        provider, model = _resolve_model_string("")
        self.assertEqual(provider, PROVIDER_GOOGLE)
        self.assertEqual(model, DEFAULT_MODEL)


class TestMakeModel(unittest.TestCase):
    def test_google_returns_bare_string(self):
        result = make_model("gemini:gemini-2.5-flash")
        self.assertEqual(result, "gemini-2.5-flash")

    def test_non_google_returns_adk(self):
        try:
            import litellm  # noqa: F401
        except ImportError:
            self.skipTest("litellm not installed")
        result = make_model("openai:gpt-4o")
        self.assertIsInstance(result, ADKModel)
        self.assertIn("openai/gpt-4o", str(result))


class TestSafeErrorExtended(unittest.TestCase):
    def test_redacts_google_api_key(self):
        err = Exception("Invalid key: AIzaSyB1234567890abcdefghijklmnopqrstuvwx")
        result = _safe_error(err)
        self.assertNotIn("AIzaSy", result)

    def test_redacts_long_hex_strings(self):
        err = Exception("Token: " + "a" * 45)
        result = _safe_error(err)
        self.assertIn("[REDACTED]", result)

    def test_keeps_short_strings(self):
        err = Exception("timeout after 30s")
        result = _safe_error(err)
        self.assertEqual(result, "timeout after 30s")


if __name__ == "__main__":
    unittest.main()
