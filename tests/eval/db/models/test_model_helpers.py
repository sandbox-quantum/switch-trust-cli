import unittest
from unittest.mock import MagicMock, patch

from switch_trust.eval.core.models.model_per_endpoint_limiter import (
    PerEndpointLimiter,
)
from switch_trust.eval.core.models.model_retry import RetryModel
from switch_trust.eval.db.base.models.model_helpers import (
    _create_inner_model,
    create_model,
)
from switch_trust.eval.db.base.models.model_types import DbModel, ModelType


def _db_model(**overrides) -> DbModel:
    defaults = {
        "type": ModelType.OPENAI,
        "name": "test-model",
        "model_name": "gpt-4",
    }
    defaults.update(overrides)
    return DbModel(**defaults)


class TestCreateModel(unittest.TestCase):
    @patch(
        "switch_trust.eval.db.base.models.model_helpers._create_inner_model",
    )
    def test_returns_retry_model(self, mock_create_inner):
        mock_create_inner.return_value = MagicMock()
        result = create_model(_db_model())
        self.assertIsInstance(result, RetryModel)

    @patch(
        "switch_trust.eval.db.base.models.model_helpers._create_inner_model",
    )
    def test_passes_max_retries(self, mock_create_inner):
        mock_create_inner.return_value = MagicMock()
        result = create_model(_db_model(), max_retries=3)
        self.assertEqual(result._max_retries, 3)

    @patch(
        "switch_trust.eval.db.base.models.model_helpers._create_inner_model",
    )
    def test_wraps_http_type_in_limiter(self, mock_create_inner):
        mock_create_inner.return_value = MagicMock()
        for http_type in (
            ModelType.ADK,
            ModelType.OPENAI_AGENT,
            ModelType.ANTHROPIC_AGENT,
            ModelType.GENERIC_HTTP,
            ModelType.LANGSERVE,
            ModelType.VERTEX_AGENT_RUNTIME,
        ):
            with self.subTest(model_type=http_type):
                PerEndpointLimiter._reset_state_for_testing()
                result = create_model(
                    _db_model(type=http_type, host="http://tenant.example"),
                )
                self.assertIsInstance(result, RetryModel)
                self.assertIsInstance(result._model, PerEndpointLimiter)
                self.assertEqual(result._model._endpoint_url, "http://tenant.example")

    @patch(
        "switch_trust.eval.db.base.models.model_helpers._create_inner_model",
    )
    def test_no_limiter_for_sdk_type(self, mock_create_inner):
        inner = MagicMock()
        mock_create_inner.return_value = inner
        for sdk_type in (
            ModelType.OPENAI,
            ModelType.ANTHROPIC,
            ModelType.GEMINI,
            ModelType.LITELLM,
            ModelType.HUGGINGFACE,
            ModelType.OLLAMA,
            ModelType.OPENAI_COMPATIBLE,
        ):
            with self.subTest(model_type=sdk_type):
                result = create_model(
                    _db_model(type=sdk_type, host="http://irrelevant"),
                )
                self.assertIsInstance(result, RetryModel)
                self.assertIs(result._model, inner)

    @patch(
        "switch_trust.eval.db.base.models.model_helpers._create_inner_model",
    )
    def test_no_limiter_when_http_host_empty(self, mock_create_inner):
        inner = MagicMock()
        mock_create_inner.return_value = inner
        result = create_model(_db_model(type=ModelType.GENERIC_HTTP, host=None))
        self.assertIsInstance(result, RetryModel)
        self.assertIs(result._model, inner)

    @patch(
        "switch_trust.eval.db.base.models.model_helpers._create_inner_model",
    )
    def test_passes_limiter_kwargs(self, mock_create_inner):
        mock_create_inner.return_value = MagicMock()
        PerEndpointLimiter._reset_state_for_testing()
        result = create_model(
            _db_model(type=ModelType.VERTEX_AGENT_RUNTIME, host="http://x"),
            per_endpoint_limit=4,
            initial_jitter_max=0.5,
        )
        limiter = result._model
        self.assertEqual(limiter._limit, 4)
        self.assertEqual(limiter._jitter_max, 0.5)


class TestCreateInnerModelAnthropic(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.models.model_anthropic.AnthropicModel",
    )
    @patch("anthropic.AsyncAnthropic")
    def test_creates_anthropic_model(
        self,
        MockClient,
        MockModel,
    ):
        db = _db_model(type=ModelType.ANTHROPIC, model_name="claude-3")
        _create_inner_model(db)
        MockClient.assert_called_once_with()
        MockModel.assert_called_once_with(
            MockClient.return_value,
            "claude-3",
            temperature=0.0,
        )

    @patch(
        "switch_trust.eval.core.models.model_anthropic.AnthropicModel",
    )
    @patch("anthropic.AsyncAnthropic")
    def test_anthropic_with_api_key(
        self,
        MockClient,
        MockModel,
    ):
        db = _db_model(
            type=ModelType.ANTHROPIC,
            model_name="claude-3",
            key="sk-ant-123",
        )
        _create_inner_model(db)
        MockClient.assert_called_once_with(api_key="sk-ant-123")

    @patch(
        "switch_trust.eval.core.models.model_anthropic.AnthropicModel",
    )
    @patch("anthropic.AsyncAnthropic")
    def test_anthropic_without_api_key(
        self,
        MockClient,
        MockModel,
    ):
        db = _db_model(
            type=ModelType.ANTHROPIC,
            model_name="claude-3",
        )
        _create_inner_model(db)
        MockClient.assert_called_once_with()


class TestCreateInnerModelOpenAI(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.models.model_openai.OpenAIModel",
    )
    @patch("openai.AsyncOpenAI")
    def test_creates_openai_model(
        self,
        MockClient,
        MockModel,
    ):
        db = _db_model(type=ModelType.OPENAI, model_name="gpt-4")
        _create_inner_model(db)
        MockClient.assert_called_once_with()
        MockModel.assert_called_once_with(
            MockClient.return_value,
            "gpt-4",
            temperature=0.0,
        )

    @patch(
        "switch_trust.eval.core.models.model_openai.OpenAIModel",
    )
    @patch("openai.AsyncOpenAI")
    def test_openai_with_api_key(
        self,
        MockClient,
        MockModel,
    ):
        db = _db_model(
            type=ModelType.OPENAI,
            model_name="gpt-4",
            key="sk-openai-123",
        )
        _create_inner_model(db)
        MockClient.assert_called_once_with(api_key="sk-openai-123")

    @patch(
        "switch_trust.eval.core.models.model_openai.OpenAIModel",
    )
    @patch("openai.AsyncOpenAI")
    def test_openai_without_api_key(
        self,
        MockClient,
        MockModel,
    ):
        db = _db_model(type=ModelType.OPENAI, model_name="gpt-4")
        _create_inner_model(db)
        MockClient.assert_called_once_with()


class TestCreateInnerModelGemini(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.models.model_gemini.GeminiModel",
    )
    @patch("google.genai.Client")
    def test_creates_gemini_model(
        self,
        MockClient,
        MockModel,
    ):
        db = _db_model(type=ModelType.GEMINI, model_name="gemini-pro")
        _create_inner_model(db)
        MockClient.assert_called_once_with()
        MockModel.assert_called_once_with(
            MockClient.return_value,
            "gemini-pro",
            temperature=0.0,
        )

    @patch(
        "switch_trust.eval.core.models.model_gemini.GeminiModel",
    )
    @patch("google.genai.Client")
    def test_gemini_with_api_key(
        self,
        MockClient,
        MockModel,
    ):
        db = _db_model(
            type=ModelType.GEMINI,
            model_name="gemini-pro",
            key="gemini-key",
        )
        _create_inner_model(db)
        MockClient.assert_called_once_with(api_key="gemini-key")

    @patch(
        "switch_trust.eval.core.models.model_gemini.GeminiModel",
    )
    @patch("google.genai.Client")
    def test_gemini_without_api_key(
        self,
        MockClient,
        MockModel,
    ):
        db = _db_model(type=ModelType.GEMINI, model_name="gemini-pro")
        _create_inner_model(db)
        MockClient.assert_called_once_with()


class TestCreateInnerModelLiteLLM(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.models.model_litellm.LiteLLMModel",
    )
    def test_creates_litellm_model(self, MockModel):
        db = _db_model(
            type=ModelType.LITELLM,
            model_name="claude-3-opus",
            temperature=0.5,
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            "claude-3-opus",
            temperature=0.5,
        )


class TestCreateInnerModelHuggingFace(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.models.model_huggingface.HuggingFaceModel",
    )
    def test_creates_huggingface_model(self, MockModel):
        db = _db_model(
            type=ModelType.HUGGINGFACE,
            model_name="meta-llama/Llama-2",
            key="hf-token",
            temperature=0.7,
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            "meta-llama/Llama-2",
            token="hf-token",
            temperature=0.7,
        )

    @patch(
        "switch_trust.eval.core.models.model_huggingface.HuggingFaceModel",
    )
    def test_huggingface_without_key(self, MockModel):
        db = _db_model(
            type=ModelType.HUGGINGFACE,
            model_name="meta-llama/Llama-2",
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            "meta-llama/Llama-2",
            token=None,
            temperature=0.0,
        )


class TestCreateInnerModelOllama(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.models.model_ollama.OllamaModel",
    )
    def test_creates_ollama_model(self, MockModel):
        db = _db_model(
            type=ModelType.OLLAMA,
            model_name="llama2",
            host="http://my-host:11434",
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            "llama2",
            host="http://my-host:11434",
            temperature=0.0,
        )

    @patch(
        "switch_trust.eval.core.models.model_ollama.OllamaModel",
    )
    def test_ollama_default_host(self, MockModel):
        db = _db_model(
            type=ModelType.OLLAMA,
            model_name="llama2",
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            "llama2",
            host="http://localhost:11434",
            temperature=0.0,
        )


class TestCreateInnerModelADK(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.models.model_adk.ADKModel",
    )
    def test_creates_adk_model(self, MockModel):
        db = _db_model(
            type=ModelType.ADK,
            model_name="my-app",
            host="http://adk-host:9000",
            immediate_result=True,
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            app_name="my-app",
            host="http://adk-host:9000",
            immediate_result=True,
            headers=None,
            connector_factory=None,
        )

    @patch(
        "switch_trust.eval.core.models.model_adk.ADKModel",
    )
    def test_adk_default_host(self, MockModel):
        db = _db_model(
            type=ModelType.ADK,
            model_name="my-app",
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            app_name="my-app",
            host="http://localhost:8000",
            immediate_result=False,
            headers=None,
            connector_factory=None,
        )

    @patch(
        "switch_trust.eval.core.models.model_adk.ADKModel",
    )
    def test_adk_headers_passed_through(self, MockModel):
        db = _db_model(
            type=ModelType.ADK,
            model_name="my-app",
            host="http://adk-host:9000",
            headers={"Authorization": "Bearer test-token"},
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            app_name="my-app",
            host="http://adk-host:9000",
            immediate_result=False,
            headers={"Authorization": "Bearer test-token"},
            connector_factory=None,
        )


class TestCreateInnerModelOpenAIAgent(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.models.model_openai_agent.OpenAIAgentModel",
    )
    def test_creates_openai_agent_model(self, MockModel):
        db = _db_model(
            type=ModelType.OPENAI_AGENT,
            model_name="agent-1",
            host="http://agent-host:5000",
            endpoint="/chat",
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            host="http://agent-host:5000",
            endpoint="/chat",
            connector_factory=None,
        )

    @patch(
        "switch_trust.eval.core.models.model_openai_agent.OpenAIAgentModel",
    )
    def test_openai_agent_defaults(self, MockModel):
        db = _db_model(
            type=ModelType.OPENAI_AGENT,
            model_name="agent-1",
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            host="http://localhost:8000",
            endpoint="/run",
            connector_factory=None,
        )


class TestCreateInnerModelAnthropicAgent(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.models.model_anthropic_agent.AnthropicAgentModel",
    )
    def test_creates_anthropic_agent_model(self, MockModel):
        db = _db_model(
            type=ModelType.ANTHROPIC_AGENT,
            model_name="agent-1",
            host="http://agent-host:5000",
            endpoint="/invoke",
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            host="http://agent-host:5000",
            endpoint="/invoke",
            connector_factory=None,
        )

    @patch(
        "switch_trust.eval.core.models.model_anthropic_agent.AnthropicAgentModel",
    )
    def test_anthropic_agent_defaults(self, MockModel):
        db = _db_model(
            type=ModelType.ANTHROPIC_AGENT,
            model_name="agent-1",
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            host="http://localhost:8000",
            endpoint="/run",
            connector_factory=None,
        )


class TestCreateInnerModelOpenAICompatible(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.models.model_openai_compatible.OpenAICompatibleModel",
    )
    def test_creates_openai_compatible_model(self, MockModel):
        db = _db_model(
            type=ModelType.OPENAI_COMPATIBLE,
            model_name="local-llm",
            host="http://vllm:8080/v1",
            key="my-key",
            temperature=0.3,
            headers={"X-Custom": "value"},
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            base_url="http://vllm:8080/v1",
            model="local-llm",
            api_key="my-key",
            temperature=0.3,
            headers={"X-Custom": "value"},
        )

    @patch(
        "switch_trust.eval.core.models.model_openai_compatible.OpenAICompatibleModel",
    )
    def test_openai_compatible_defaults(self, MockModel):
        db = _db_model(
            type=ModelType.OPENAI_COMPATIBLE,
            model_name="local-llm",
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            base_url="http://localhost:8000/v1",
            model="local-llm",
            api_key="EMPTY",
            temperature=0.0,
            headers=None,
        )


class TestCreateInnerModelGenericHTTP(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.models.model_generic_http.GenericHttpModel",
    )
    def test_creates_generic_http_model(self, MockModel):
        db = _db_model(
            type=ModelType.GENERIC_HTTP,
            model_name="http-model",
            host="http://my-api:9000",
            endpoint="/predict",
            headers={"Authorization": "Bearer xyz"},
            input_path="request.text",
            output_path="response.text",
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            url="http://my-api:9000/predict",
            headers={"Authorization": "Bearer xyz"},
            input_path="request.text",
            output_path="response.text",
            connector_factory=None,
        )

    @patch(
        "switch_trust.eval.core.models.model_generic_http.GenericHttpModel",
    )
    def test_generic_http_defaults(self, MockModel):
        db = _db_model(
            type=ModelType.GENERIC_HTTP,
            model_name="http-model",
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            url="http://localhost:8000",
            headers=None,
            input_path="input",
            output_path="output",
            connector_factory=None,
        )

    @patch(
        "switch_trust.eval.core.models.model_generic_http.GenericHttpModel",
    )
    def test_generic_http_strips_trailing_slash(self, MockModel):
        db = _db_model(
            type=ModelType.GENERIC_HTTP,
            model_name="http-model",
            host="http://my-api:9000/",
            endpoint="/predict",
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            url="http://my-api:9000/predict",
            headers=None,
            input_path="input",
            output_path="output",
            connector_factory=None,
        )


class TestCreateInnerModelLangServe(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.models.model_langserve.LangServeModel",
    )
    def test_creates_langserve_model(self, MockModel):
        db = _db_model(
            type=ModelType.LANGSERVE,
            model_name="langserve-model",
            host="http://langserve:8000",
            endpoint="/my-chain",
            headers={"X-Token": "abc"},
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            base_url="http://langserve:8000",
            chain_path="/my-chain",
            headers={"X-Token": "abc"},
            connector_factory=None,
        )

    @patch(
        "switch_trust.eval.core.models.model_langserve.LangServeModel",
    )
    def test_langserve_defaults(self, MockModel):
        db = _db_model(
            type=ModelType.LANGSERVE,
            model_name="langserve-model",
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            base_url="http://localhost:8000",
            chain_path="",
            headers=None,
            connector_factory=None,
        )


class TestCreateInnerModelUnknown(unittest.TestCase):
    def test_unknown_type_raises_value_error(self):
        db = _db_model()
        db.type = "unknown_type"
        with self.assertRaises(ValueError, msg="unknown model type"):
            _create_inner_model(db)


class TestEnvVarResolution(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.models.model_anthropic.AnthropicModel",
    )
    @patch("anthropic.AsyncAnthropic")
    @patch.dict(
        "os.environ",
        {"ANTHROPIC_API_KEY": "resolved-key"},
    )
    def test_env_var_key_resolved_for_anthropic(
        self,
        MockClient,
        MockModel,
    ):
        db = _db_model(
            type=ModelType.ANTHROPIC,
            model_name="claude-3",
            key="${ANTHROPIC_API_KEY}",
        )
        _create_inner_model(db)
        MockClient.assert_called_once_with(
            api_key="resolved-key",
        )

    @patch(
        "switch_trust.eval.core.models.model_openai.OpenAIModel",
    )
    @patch("openai.AsyncOpenAI")
    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "sk-resolved"},
    )
    def test_env_var_key_resolved_for_openai(
        self,
        MockClient,
        MockModel,
    ):
        db = _db_model(
            type=ModelType.OPENAI,
            model_name="gpt-4",
            key="${OPENAI_API_KEY}",
        )
        _create_inner_model(db)
        MockClient.assert_called_once_with(
            api_key="sk-resolved",
        )

    @patch(
        "switch_trust.eval.core.models.model_generic_http.GenericHttpModel",
    )
    @patch.dict(
        "os.environ",
        {"AUTH_TOKEN": "my-secret"},
    )
    def test_env_var_in_headers_resolved(
        self,
        MockModel,
    ):
        db = _db_model(
            type=ModelType.GENERIC_HTTP,
            model_name="http-model",
            headers={
                "Authorization": "Bearer ${AUTH_TOKEN}",
            },
        )
        _create_inner_model(db)
        MockModel.assert_called_once_with(
            url="http://localhost:8000",
            headers={
                "Authorization": "Bearer my-secret",
            },
            input_path="input",
            output_path="output",
            connector_factory=None,
        )

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_env_var_raises(self):
        db = _db_model(
            type=ModelType.ANTHROPIC,
            model_name="claude-3",
            key="${NONEXISTENT_KEY}",
        )
        with self.assertRaises(ValueError):
            _create_inner_model(db)


if __name__ == "__main__":
    unittest.main()
