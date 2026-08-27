import sys
import types as module_types

import pytest

from agent.budget import BudgetManager
from agent.models import (
    GeminiClient,
    GenerateContentRateLimiter,
    ModelConfig,
    ProviderInfrastructureError,
    ScriptedLLMClient,
)
from agent.state import TokenUsage


class FakeUsage:
    prompt_token_count = 7
    candidates_token_count = 9
    total_token_count = 20


class FakeResponse:
    text = "ok"
    usage_metadata = FakeUsage()


class FakeCount:
    total_tokens = 7


class FakeThinkingConfig:
    def __init__(self, thinking_budget):
        self.thinking_budget = thinking_budget


class FakeGenerateContentConfig:
    def __init__(self, temperature, max_output_tokens, thinking_config):
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.thinking_config = thinking_config


class FakeTypes:
    GenerateContentConfig = FakeGenerateContentConfig
    ThinkingConfig = FakeThinkingConfig


class FakeModels:
    def __init__(self, responses=None, errors=None, count_tokens=7):
        self.responses = list(responses or [FakeResponse()])
        self.errors = list(errors or [])
        self.count_tokens_value = count_tokens
        self.events = []
        self.configs = []

    def count_tokens(self, model, contents):
        self.events.append(("count_tokens", model, contents))
        return module_types.SimpleNamespace(total_tokens=self.count_tokens_value)

    def generate_content(self, model, contents, config):
        self.events.append(("generate_content", model, contents))
        self.configs.append(config)
        if self.errors:
            raise self.errors.pop(0)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, models):
        self.models = models


class FakeLimiter:
    def __init__(self, waits=None):
        self.waits = list(waits or [0.0])

    def wait(self):
        return self.waits.pop(0) if self.waits else 0.0


class FakeHTTPError(Exception):
    def __init__(self, status_code, retry_after=None):
        super().__init__(f"{status_code} error")
        self.status_code = status_code
        self.retry_after = retry_after


def make_client(models, **config_kwargs):
    client = GeminiClient.__new__(GeminiClient)
    client.config = ModelConfig(name="gemini-2.5-flash", temperature=0, **config_kwargs)
    client._client = FakeClient(models)
    client._types = FakeTypes
    client._rate_limiter = FakeLimiter()
    return client


def test_pre_call_budget_admission_rejects_oversized_call():
    budget = BudgetManager(total_budget=10)
    decision = budget.admit("x" * 100, generation_budget=20)
    assert not decision.admitted
    assert decision.reason == "insufficient_token_budget"


def test_token_accounting_records_estimated_usage():
    budget = BudgetManager(total_budget=200)
    call = ScriptedLLMClient(["patched"]).generate("executor", "prompt", budget, 20)
    assert call.admitted
    assert budget.used.total_tokens == call.usage.total_tokens
    assert budget.used.token_count_estimated is True


def test_current_google_genai_client_integration(monkeypatch):
    created = {}

    class ClientFactory:
        def __init__(self, api_key):
            created["api_key"] = api_key
            self.models = FakeModels()

    google_module = module_types.ModuleType("google")
    genai_module = module_types.ModuleType("google.genai")
    genai_module.Client = ClientFactory
    genai_module.types = FakeTypes
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

    client = GeminiClient(ModelConfig(requests_per_minute=123))
    assert created["api_key"] == "fake-key"
    assert isinstance(client._client, ClientFactory)


def test_gemini_count_tokens_before_generate_and_thinking_budget_zero():
    models = FakeModels()
    client = make_client(models)
    call = client.generate("executor", "prompt", BudgetManager(100), generation_budget=17)
    assert call.admitted
    assert [event[0] for event in models.events] == ["count_tokens", "generate_content"]
    config = models.configs[0]
    assert config.max_output_tokens == 17
    assert config.temperature == 0
    assert config.thinking_config.thinking_budget == 0
    assert call.thinking_budget == 0
    assert call.thinking_config_applied is True
    assert call.prompt_token_count_estimated is False
    assert call.usage.input_tokens == 7
    assert call.usage.output_tokens == 13
    assert call.usage.total_tokens == 20


def test_gemini_dynamic_max_output_tokens_uses_remaining_budget():
    models = FakeModels(count_tokens=80)
    budget = BudgetManager(100)
    budget.record(TokenUsage(total_tokens=10))
    client = make_client(models, min_output_tokens=5)
    call = client.generate("executor", "prompt", budget, generation_budget=50)
    assert call.admitted
    assert call.max_output_tokens == 10
    assert models.configs[0].max_output_tokens == 10


def test_gemini_budget_exhaustion_when_counted_prompt_leaves_too_little_output():
    models = FakeModels(count_tokens=90)
    client = make_client(models, min_output_tokens=32)
    call = client.generate("planner", "prompt", BudgetManager(100), generation_budget=50)
    assert not call.admitted
    assert call.skipped_reason == "insufficient_token_budget"
    assert call.max_output_tokens == 0
    assert [event[0] for event in models.events] == ["count_tokens"]


def test_gemini_records_fallback_when_provider_counting_fails():
    class CountingFails(FakeModels):
        def count_tokens(self, model, contents):
            self.events.append(("count_tokens", model, contents))
            raise RuntimeError("counting unavailable")

    models = CountingFails()
    client = make_client(models)
    call = client.generate("critic", "prompt", BudgetManager(100), generation_budget=10)
    assert call.admitted
    assert call.prompt_token_count_estimated is True


def test_global_request_pacing_waits_between_generate_calls():
    now = {"value": 0.0}
    sleeps = []

    def monotonic():
        return now["value"]

    def sleep(seconds):
        sleeps.append(seconds)
        now["value"] += seconds

    limiter = GenerateContentRateLimiter(4, monotonic=monotonic, sleep=sleep)
    assert limiter.wait() == 0
    assert limiter.wait() == 15
    assert sleeps == [15]


def test_429_retry_does_not_consume_budget_until_success(monkeypatch):
    monkeypatch.setattr("agent.models.time.sleep", lambda seconds: None)
    models = FakeModels(errors=[FakeHTTPError(429, retry_after=2)])
    client = make_client(models)
    budget = BudgetManager(100)
    call = client.generate("executor", "prompt", budget, generation_budget=20)
    assert call.admitted
    assert call.provider_attempts == 2
    assert call.transient_retries == 1
    assert call.rate_limit_retries == 1
    assert call.rate_limit_wait_seconds == 2
    assert budget.used.total_tokens == 20


def test_503_retry(monkeypatch):
    monkeypatch.setattr("agent.models.time.sleep", lambda seconds: None)
    monkeypatch.setattr("agent.models.random.uniform", lambda start, end: 0)
    models = FakeModels(errors=[FakeHTTPError(503)])
    client = make_client(models)
    call = client.generate("critic", "prompt", BudgetManager(100), generation_budget=20)
    assert call.provider_attempts == 2
    assert call.transient_retries == 1
    assert call.rate_limit_retries == 0


def test_retry_limit_raises_without_consuming_tokens(monkeypatch):
    monkeypatch.setattr("agent.models.time.sleep", lambda seconds: None)
    monkeypatch.setattr("agent.models.random.uniform", lambda start, end: 0)
    models = FakeModels(errors=[FakeHTTPError(503), FakeHTTPError(503)])
    client = make_client(models, max_transient_retries=1)
    budget = BudgetManager(100)
    with pytest.raises(ProviderInfrastructureError) as exc_info:
        client.generate("executor", "prompt", budget, generation_budget=20)
    assert exc_info.value.provider_attempts == 2
    assert exc_info.value.transient_retries == 1
    assert budget.used.total_tokens == 0


def test_permanent_errors_are_not_retried():
    models = FakeModels(errors=[FakeHTTPError(401)])
    client = make_client(models)
    with pytest.raises(ProviderInfrastructureError) as exc_info:
        client.generate("planner", "prompt", BudgetManager(100), generation_budget=20)
    assert exc_info.value.provider_attempts == 1
    assert exc_info.value.transient_retries == 0
