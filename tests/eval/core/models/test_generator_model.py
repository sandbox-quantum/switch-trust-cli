import base64
import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel

from switch_trust.eval.core.models import generator_model
from switch_trust.eval.core.models.generator_model import (
    GeneratorConfig,
    VertexConfig,
)
from switch_trust.eval.core.models.model_per_endpoint_limiter import (
    PerEndpointLimiter,
)
from switch_trust.eval.core.models.model_retry import RetryModel

_SA_KEY = {
    "type": "service_account",
    "project_id": "key-project",
    "private_key": "-----BEGIN PRIVATE KEY-----\nfake\n-----END PRIVATE KEY-----\n",
    "client_email": "sa@key-project.iam.gserviceaccount.com",
    "token_uri": "https://oauth2.googleapis.com/token",
}


def _config(model_type: str, model_name: str, **kwargs) -> GeneratorConfig:
    return GeneratorConfig(model_type=model_type, model_name=model_name, **kwargs)


class TestGeneratorConfigFromEnv(unittest.TestCase):
    """The single environment funnel: parse + validate shape in one place."""

    def _from_env(self, env: dict[str, str]) -> GeneratorConfig:
        # clear=True so ambient credentials can't leak into the parse.
        with patch.dict(os.environ, env, clear=True):
            return GeneratorConfig.from_env()

    def test_missing_generator_model_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._from_env({})
        self.assertIn("GENERATOR_MODEL must be set", str(ctx.exception))

    def test_invalid_format_no_colon_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._from_env({"GENERATOR_MODEL": "nocolon"})
        self.assertIn("Invalid GENERATOR_MODEL format", str(ctx.exception))

    def test_empty_half_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._from_env({"GENERATOR_MODEL": "gemini:"})
        self.assertIn("Invalid GENERATOR_MODEL format", str(ctx.exception))

    def test_strips_whitespace(self):
        config = self._from_env({"GENERATOR_MODEL": "openai: gpt-4o "})
        self.assertEqual((config.model_type, config.model_name), ("openai", "gpt-4o"))

    def test_non_gemini_has_no_gemini_fields(self):
        config = self._from_env({"GENERATOR_MODEL": "openai:gpt-4o"})
        self.assertIsNone(config.vertex)
        self.assertFalse(config.gemini_api_key_present)

    def test_gemini_developer_api_records_key_presence(self):
        config = self._from_env(
            {"GENERATOR_MODEL": "gemini:gemini-2.5-flash", "GOOGLE_API_KEY": "k"}
        )
        self.assertIsNone(config.vertex)
        self.assertTrue(config.gemini_api_key_present)

    def test_gemini_vertex_parses_key_and_project_location(self):
        config = self._from_env(
            {
                "GENERATOR_MODEL": "gemini:gemini-2.5-flash",
                "GOOGLE_VERTEX_CREDENTIALS_JSON": json.dumps(_SA_KEY),
                "GOOGLE_CLOUD_PROJECT": "env-project",
                "GOOGLE_CLOUD_LOCATION": "us-central1",
            }
        )
        assert config.vertex is not None
        self.assertEqual(config.vertex.service_account, _SA_KEY)
        self.assertEqual(config.vertex.project, "env-project")
        self.assertEqual(config.vertex.location, "us-central1")

    def test_gemini_vertex_project_defaults_to_key(self):
        config = self._from_env(
            {
                "GENERATOR_MODEL": "gemini:gemini-2.5-flash",
                "GOOGLE_VERTEX_CREDENTIALS_JSON": json.dumps(_SA_KEY),
                "GOOGLE_CLOUD_LOCATION": "us-central1",
            }
        )
        assert config.vertex is not None
        self.assertEqual(config.vertex.project, "key-project")

    def test_gemini_vertex_base64_key_is_decoded(self):
        encoded = base64.b64encode(json.dumps(_SA_KEY).encode()).decode()
        config = self._from_env(
            {
                "GENERATOR_MODEL": "gemini:gemini-2.5-flash",
                "GOOGLE_VERTEX_CREDENTIALS_JSON": encoded,
                "GOOGLE_CLOUD_LOCATION": "us-central1",
            }
        )
        assert config.vertex is not None
        self.assertEqual(config.vertex.service_account, _SA_KEY)

    def test_gemini_vertex_missing_location_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self._from_env(
                {
                    "GENERATOR_MODEL": "gemini:gemini-2.5-flash",
                    "GOOGLE_VERTEX_CREDENTIALS_JSON": json.dumps(_SA_KEY),
                }
            )
        self.assertIn("GOOGLE_CLOUD_LOCATION", str(ctx.exception))

    def test_gemini_vertex_invalid_key_does_not_leak_secret(self):
        secret = "not-json-and-not-base64-$$$"
        with self.assertRaises(ValueError) as ctx:
            self._from_env(
                {
                    "GENERATOR_MODEL": "gemini:gemini-2.5-flash",
                    "GOOGLE_VERTEX_CREDENTIALS_JSON": secret,
                    "GOOGLE_CLOUD_LOCATION": "us-central1",
                }
            )
        self.assertNotIn(secret, str(ctx.exception))


class TestGeminiCredentialPresence(unittest.TestCase):
    """`gemini_credential_present`: can `Client()` be built at all?

    Deliberately wider than `gemini_api_key_present`, which asks the narrower
    "was a Developer-API credential *configured*" that `validate()` holds the
    worker to. Every combination google-genai itself accepts has to answer
    True here, or the CLI's exit-20 refusal would reject runs that work.
    """

    def _from_env(self, env: dict[str, str]) -> GeneratorConfig:
        with patch.dict(os.environ, env, clear=True):
            return GeneratorConfig.from_env()

    def test_nothing_configured_is_absent(self):
        config = self._from_env({"GENERATOR_MODEL": "gemini:gemini-2.5-flash"})
        self.assertFalse(config.gemini_credential_present)

    def test_an_empty_key_is_absent(self):
        """The reported case: `GEMINI_API_KEY=` set but empty."""
        config = self._from_env(
            {"GENERATOR_MODEL": "gemini:gemini-2.5-flash", "GEMINI_API_KEY": ""}
        )
        self.assertFalse(config.gemini_credential_present)

    def test_google_api_key_is_present(self):
        config = self._from_env(
            {"GENERATOR_MODEL": "gemini:gemini-2.5-flash", "GOOGLE_API_KEY": "k"}
        )
        self.assertTrue(config.gemini_credential_present)

    def test_gemini_api_key_alone_is_present(self):
        """google-genai reads either name, so neither may be refused."""
        config = self._from_env(
            {"GENERATOR_MODEL": "gemini:gemini-2.5-flash", "GEMINI_API_KEY": "k"}
        )
        self.assertTrue(config.gemini_credential_present)
        # The narrower field still says no: this is not a *configured*
        # Developer-API credential by `validate()`'s stricter standard.
        self.assertFalse(config.gemini_api_key_present)

    def test_env_driven_vertex_needs_no_key(self):
        """ADC carries its own credential; refusing it would be wrong."""
        for value in ("true", "TRUE", "1"):
            with self.subTest(value=value):
                config = self._from_env(
                    {
                        "GENERATOR_MODEL": "gemini:gemini-2.5-flash",
                        "GOOGLE_GENAI_USE_VERTEXAI": value,
                    }
                )
                self.assertTrue(config.gemini_credential_present)

    def test_a_falsy_vertex_flag_does_not_count(self):
        config = self._from_env(
            {
                "GENERATOR_MODEL": "gemini:gemini-2.5-flash",
                "GOOGLE_GENAI_USE_VERTEXAI": "false",
            }
        )
        self.assertFalse(config.gemini_credential_present)

    def test_a_vertex_service_account_is_present(self):
        config = self._from_env(
            {
                "GENERATOR_MODEL": "gemini:gemini-2.5-flash",
                "GOOGLE_VERTEX_CREDENTIALS_JSON": json.dumps(_SA_KEY),
                "GOOGLE_CLOUD_LOCATION": "us-central1",
            }
        )
        self.assertTrue(config.gemini_credential_present)

    def test_non_gemini_is_never_refused(self):
        config = self._from_env({"GENERATOR_MODEL": "openai:gpt-4o"})
        self.assertTrue(config.gemini_credential_present)


class TestGeneratorConfigValidate(unittest.TestCase):
    """Startup validation: usable credential + exercised construction."""

    def test_gemini_without_any_credentials_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _config("gemini", "gemini-2.5-flash").validate()
        msg = str(ctx.exception)
        self.assertIn("GOOGLE_VERTEX_CREDENTIALS_JSON", msg)
        self.assertIn("GOOGLE_API_KEY", msg)

    def test_gemini_with_api_key_exercises_construction(self):
        config = _config("gemini", "gemini-2.5-flash", gemini_api_key_present=True)
        with patch.object(generator_model, "_create_inner") as mock_inner:
            config.validate()
        mock_inner.assert_called_once_with(config)

    def test_gemini_with_vertex_exercises_construction(self):
        vertex = VertexConfig(
            service_account=_SA_KEY, project="p", location="us-central1"
        )
        config = _config("gemini", "gemini-2.5-flash", vertex=vertex)
        with patch.object(generator_model, "_create_inner") as mock_inner:
            config.validate()
        mock_inner.assert_called_once_with(config)

    def test_non_gemini_skips_credential_check(self):
        config = _config("openai", "gpt-4o")
        with patch.object(generator_model, "_create_inner") as mock_inner:
            config.validate()
        mock_inner.assert_called_once_with(config)


class _Score(BaseModel):
    score: float


class TestGetGeneratorConfig(unittest.TestCase):
    def setUp(self):
        # Module-level installed config; keep tests independent of each other
        # and of any prior set_generator_config call.
        self._reset = patch.object(generator_model, "_config", None)
        self._reset.start()
        self.addCleanup(self._reset.stop)

    def test_none_when_nothing_installed(self):
        self.assertIsNone(generator_model.get_generator_config())

    def test_returns_installed_config(self):
        installed = _config("gemini", "gemini-3.8-flash")
        generator_model.set_generator_config(installed)
        self.assertIs(generator_model.get_generator_config(), installed)


class TestGetGeneratorModel(unittest.TestCase):
    def setUp(self):
        # get_generator_model reads a module-level installed config; keep tests
        # independent of each other and of any prior set_generator_config call.
        self._reset = patch.object(generator_model, "_config", None)
        self._reset.start()
        self.addCleanup(self._reset.stop)

    @patch.dict(os.environ, {"GENERATOR_MODEL": "openai:gpt-4o"}, clear=True)
    @patch.object(generator_model, "_create_inner")
    def test_falls_back_to_from_env_when_uninstalled(self, mock_inner: MagicMock):
        mock_inner.return_value = MagicMock()
        result = generator_model.get_generator_model()
        self.assertIsInstance(result, generator_model.GeneratorModel)
        # Built from the config parsed out of the environment.
        (built_config,) = mock_inner.call_args.args
        self.assertEqual(built_config.model_type, "openai")
        self.assertEqual(built_config.model_name, "gpt-4o")

    @patch.object(generator_model, "_create_inner")
    def test_uses_installed_config_without_reading_env(self, mock_inner: MagicMock):
        mock_inner.return_value = MagicMock()
        installed = _config("anthropic", "claude-haiku-4-20250414")
        generator_model.set_generator_config(installed)

        # No GENERATOR_MODEL in the environment: the installed config wins.
        with patch.dict(os.environ, {}, clear=True):
            generator_model.get_generator_model()

        mock_inner.assert_called_once_with(installed)

    @patch.object(generator_model, "_create_inner")
    def test_unbounded_by_default(self, mock_inner: MagicMock):
        # 0 is the CLI's behaviour, and generator_model ships in flintai-cli,
        # so the default must add no limiter at all.
        mock_inner.return_value = MagicMock()
        generator_model.set_generator_config(_config("openai", "gpt-4o"))

        model = generator_model.get_generator_model()

        assert not isinstance(model._model._model, PerEndpointLimiter)

    @patch.object(generator_model, "_create_inner")
    def test_concurrency_limit_wraps_the_generator(self, mock_inner: MagicMock):
        # Without this cap, slots x concurrency x replicas lands hundreds of
        # simultaneous judge calls on one shared Vertex project.
        mock_inner.return_value = MagicMock()
        generator_model.set_generator_config(
            _config("gemini", "gemini-2.5-flash-lite", concurrency_limit=32)
        )

        model = generator_model.get_generator_model()

        limiter = model._model._model
        assert isinstance(limiter, PerEndpointLimiter)
        assert limiter._limit == 32
        # Keyed per generator model, so two differently-configured processes
        # cannot accidentally share one budget.
        assert limiter._endpoint_url == "generator:gemini:gemini-2.5-flash-lite"
        # No jitter: it exists to spread an opening burst across a small tenant
        # server, and this guards a per-minute cloud quota instead.
        assert limiter._jitter_max == 0.0

    @patch.object(generator_model, "_create_inner")
    def test_limiter_sits_inside_the_retry(self, mock_inner: MagicMock):
        # Retries are exactly what a 429 produces, so they have to queue on the
        # same budget rather than route around it.
        mock_inner.return_value = MagicMock()
        generator_model.set_generator_config(
            _config("openai", "gpt-4o", concurrency_limit=4)
        )

        model = generator_model.get_generator_model()

        assert isinstance(model._model, RetryModel)
        assert isinstance(model._model._model, PerEndpointLimiter)


class TestGeneratorModel(unittest.IsolatedAsyncioTestCase):
    async def test_warns_without_output_schema(self):
        inner = MagicMock()
        inner.generate = AsyncMock(return_value=MagicMock())
        model = generator_model.GeneratorModel(inner)

        with self.assertLogs(
            "switch_trust.eval.core.models.generator_model", level="WARNING"
        ) as logs:
            await model.generate("hi")
        self.assertTrue(any("output_schema" in line for line in logs.output))
        inner.generate.assert_awaited_once()

    async def test_no_warning_with_output_schema(self):
        inner = MagicMock()
        inner.generate = AsyncMock(return_value=MagicMock())
        model = generator_model.GeneratorModel(inner)

        with self.assertNoLogs(
            "switch_trust.eval.core.models.generator_model", level="WARNING"
        ):
            await model.generate("hi", output_schema=_Score)
        # The schema is forwarded to the wrapped model.
        _, kwargs = inner.generate.await_args
        self.assertIs(kwargs["output_schema"], _Score)


class TestCreateInnerRefusesAMissingGeminiCredential(unittest.TestCase):
    """A judge with no key is refused with a type, not a bare SDK ValueError.

    The CLI maps the type to exit 20; `Client()`'s own ValueError carries
    nothing to classify from and lands on the catch-all 1.
    """

    def test_it_raises_before_the_client_is_built(self):
        config = _config("gemini", "gemini-2.5-flash", gemini_credential_present=False)
        with patch("google.genai.Client") as mock_client:
            with self.assertRaises(generator_model.GeneratorCredentialError):
                generator_model._create_inner(config)
        mock_client.assert_not_called()

    def test_the_message_names_every_way_to_fix_it(self):
        config = _config("gemini", "gemini-2.5-flash", gemini_credential_present=False)
        with self.assertRaises(generator_model.GeneratorCredentialError) as ctx:
            generator_model._create_inner(config)
        msg = str(ctx.exception)
        self.assertIn("GOOGLE_API_KEY", msg)
        self.assertIn("GEMINI_API_KEY", msg)
        self.assertIn("GOOGLE_VERTEX_CREDENTIALS_JSON", msg)

    def test_it_is_a_value_error(self):
        """So the worker's `validate()` and every `except ValueError` are
        unaffected by the new type."""
        self.assertTrue(
            issubclass(generator_model.GeneratorCredentialError, ValueError)
        )


class TestCreateInner(unittest.TestCase):
    def test_gemini(self):
        with patch(
            "switch_trust.eval.core.models.model_gemini.GeminiModel"
        ) as mock_model:
            with patch("google.genai.Client") as mock_client:
                result = generator_model._create_inner(
                    _config("gemini", "gemini-2.5-flash-lite")
                )
                mock_client.assert_called_once()
                mock_model.assert_called_once()
                self.assertIsNotNone(result)

    def test_openai(self):
        with patch(
            "switch_trust.eval.core.models.model_openai.OpenAIModel"
        ) as mock_model:
            with patch("openai.AsyncOpenAI") as mock_client:
                result = generator_model._create_inner(_config("openai", "gpt-4o"))
                mock_client.assert_called_once()
                mock_model.assert_called_once()
                self.assertIsNotNone(result)

    def test_anthropic(self):
        with patch(
            "switch_trust.eval.core.models.model_anthropic.AnthropicModel"
        ) as mock_model:
            with patch("anthropic.AsyncAnthropic") as mock_client:
                result = generator_model._create_inner(
                    _config("anthropic", "claude-haiku-4-20250414")
                )
                mock_client.assert_called_once()
                mock_model.assert_called_once()
                self.assertIsNotNone(result)

    def test_litellm(self):
        with patch(
            "switch_trust.eval.core.models.model_litellm.LiteLLMModel"
        ) as mock_model:
            result = generator_model._create_inner(_config("litellm", "gpt-4o"))
            mock_model.assert_called_once_with("gpt-4o")
            self.assertIsNotNone(result)

    def test_ollama(self):
        with patch(
            "switch_trust.eval.core.models.model_ollama.OllamaModel"
        ) as mock_model:
            result = generator_model._create_inner(_config("ollama", "llama3"))
            mock_model.assert_called_once_with("llama3")
            self.assertIsNotNone(result)

    def test_unknown_type(self):
        with self.assertRaises(ValueError) as ctx:
            generator_model._create_inner(_config("foobar", "model"))
        self.assertIn("Unknown generator model type", str(ctx.exception))


class TestCreateGeminiClient(unittest.TestCase):
    """Client construction is pure now: Developer API for vertex=None, Vertex
    when a VertexConfig is supplied. No environment reads."""

    def test_none_uses_default_client(self):
        with patch("google.genai.Client") as mock_client:
            with patch(
                "google.oauth2.service_account.Credentials.from_service_account_info"
            ) as mock_creds:
                generator_model._create_gemini_client(None)
        mock_client.assert_called_once_with()
        mock_creds.assert_not_called()

    def test_vertex_builds_vertex_client(self):
        vertex = VertexConfig(
            service_account=_SA_KEY, project="env-project", location="us-central1"
        )
        with patch("google.genai.Client") as mock_client:
            with patch(
                "google.oauth2.service_account.Credentials.from_service_account_info"
            ) as mock_creds:
                mock_creds.return_value = "CREDS"
                generator_model._create_gemini_client(vertex)

        self.assertEqual(mock_creds.call_args.args[0], _SA_KEY)
        mock_client.assert_called_once_with(
            vertexai=True,
            project="env-project",
            location="us-central1",
            credentials="CREDS",
        )


class TestParseServiceAccount(unittest.TestCase):
    def test_valid_key(self):
        self.assertEqual(
            generator_model._parse_service_account(json.dumps(_SA_KEY)), _SA_KEY
        )

    def test_base64_key(self):
        encoded = base64.b64encode(json.dumps(_SA_KEY).encode()).decode()
        self.assertEqual(generator_model._parse_service_account(encoded), _SA_KEY)

    def test_non_object_raises(self):
        with self.assertRaises(ValueError) as ctx:
            generator_model._parse_service_account("123")
        self.assertIn("JSON object", str(ctx.exception))

    def test_wrong_type_raises(self):
        key = {**_SA_KEY, "type": "authorized_user"}
        with self.assertRaises(ValueError) as ctx:
            generator_model._parse_service_account(json.dumps(key))
        self.assertIn("service-account key", str(ctx.exception))

    def test_missing_fields_raises(self):
        key = {"type": "service_account", "project_id": "p"}
        with self.assertRaises(ValueError) as ctx:
            generator_model._parse_service_account(json.dumps(key))
        msg = str(ctx.exception)
        self.assertIn("missing required", msg)
        self.assertIn("client_email", msg)

    def test_source_label_is_used_in_errors(self):
        # Decoupled from any one caller: the message names the supplied source,
        # not a hardcoded env var.
        with self.assertRaises(ValueError) as ctx:
            generator_model._parse_service_account(
                "not-json-$$$", source="MY_CUSTOM_SOURCE"
            )
        self.assertIn("MY_CUSTOM_SOURCE", str(ctx.exception))

    def test_default_source_is_generic(self):
        with self.assertRaises(ValueError) as ctx:
            generator_model._parse_service_account("not-json-$$$")
        self.assertIn("service-account key", str(ctx.exception))

    def test_collapsed_private_key_newlines_raises(self):
        # The classic hand-seeding mangle: json parses, fields present, but the
        # PEM was flattened onto one line — google-auth would reject it later as
        # a terse "invalid private key". Catch it here with an actionable message.
        key = {
            **_SA_KEY,
            "private_key": "-----BEGIN PRIVATE KEY-----fake-----END PRIVATE KEY-----",
        }
        with self.assertRaises(ValueError) as ctx:
            generator_model._parse_service_account(json.dumps(key))
        msg = str(ctx.exception)
        self.assertIn("collapsed", msg)
        self.assertNotIn("fake", msg)  # never echoes the key material

    def test_double_escaped_private_key_newlines_raises(self):
        # Newlines double-escaped during seeding: the parsed value holds the
        # literal two-char sequence backslash-n instead of real newlines.
        key = {
            **_SA_KEY,
            "private_key": "-----BEGIN PRIVATE KEY-----\\nfake\\n-----END PRIVATE KEY-----",
        }
        with self.assertRaises(ValueError) as ctx:
            generator_model._parse_service_account(json.dumps(key))
        msg = str(ctx.exception)
        self.assertIn("double-escaped", msg)
        self.assertNotIn("fake", msg)

    def test_private_key_without_pem_markers_raises(self):
        key = {**_SA_KEY, "private_key": "not a pem at all\nsecret-bytes\n"}
        with self.assertRaises(ValueError) as ctx:
            generator_model._parse_service_account(json.dumps(key))
        msg = str(ctx.exception)
        self.assertIn("BEGIN/END", msg)
        self.assertNotIn("secret-bytes", msg)


if __name__ == "__main__":
    unittest.main()
