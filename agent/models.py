from __future__ import annotations

import os
import random
import re
import time
from dataclasses import dataclass
from typing import Protocol

from agent.budget import BudgetManager, estimate_tokens
from agent.state import LLMCallRecord, TokenUsage


@dataclass(frozen=True)
class ModelConfig:
    name: str = "gemini-2.5-flash"
    temperature: float = 0.0
    provider: str = "google"
    thinking_budget: int | None = 0
    min_output_tokens: int = 32
    requests_per_minute: int = 4
    max_transient_retries: int = 6


class LLMClient(Protocol):
    def generate(self, role: str, prompt: str, budget: BudgetManager, generation_budget: int) -> LLMCallRecord:
        ...


class ProviderInfrastructureError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        provider_attempts: int = 0,
        transient_retries: int = 0,
        rate_limit_retries: int = 0,
        rate_limit_wait_seconds: float = 0.0,
        provider_wall_time_seconds: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.provider_attempts = provider_attempts
        self.transient_retries = transient_retries
        self.rate_limit_retries = rate_limit_retries
        self.rate_limit_wait_seconds = rate_limit_wait_seconds
        self.provider_wall_time_seconds = provider_wall_time_seconds


class GenerateContentRateLimiter:
    def __init__(
        self,
        requests_per_minute: int,
        *,
        monotonic=time.monotonic,
        sleep=time.sleep,
    ) -> None:
        self.requests_per_minute = max(1, requests_per_minute)
        self._min_interval = 60.0 / self.requests_per_minute
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_request_at: float | None = None

    def wait(self) -> float:
        now = self._monotonic()
        if self._last_request_at is None:
            self._last_request_at = now
            return 0.0
        elapsed = now - self._last_request_at
        wait_seconds = max(0.0, self._min_interval - elapsed)
        if wait_seconds > 0:
            self._sleep(wait_seconds)
            now = self._monotonic()
        self._last_request_at = now
        return wait_seconds


class GeminiClient:
    _rate_limiters: dict[int, GenerateContentRateLimiter] = {}

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for real Gemini runs.")
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self._rate_limiter = self._process_rate_limiter(self.config.requests_per_minute)

    @classmethod
    def _process_rate_limiter(cls, requests_per_minute: int) -> GenerateContentRateLimiter:
        rpm = max(1, requests_per_minute)
        if rpm not in cls._rate_limiters:
            cls._rate_limiters[rpm] = GenerateContentRateLimiter(rpm)
        return cls._rate_limiters[rpm]

    def generate(self, role: str, prompt: str, budget: BudgetManager, generation_budget: int) -> LLMCallRecord:
        prompt_tokens, prompt_count_estimated = self._count_prompt_tokens(prompt)
        remaining = budget.remaining
        available_output = remaining - prompt_tokens
        if available_output < self.config.min_output_tokens:
            return LLMCallRecord(
                role=role,
                admitted=False,
                prompt_tokens_estimate=prompt_tokens,
                prompt_token_count_estimated=prompt_count_estimated,
                configured_generation_budget=generation_budget,
                generation_budget=0,
                max_output_tokens=0,
                thinking_budget=self.config.thinking_budget,
                thinking_config_note=self._thinking_config_note(),
                skipped_reason="insufficient_token_budget",
            )
        max_output_tokens = min(generation_budget, available_output)
        response, operation = self._generate_content_with_retries(
            prompt=prompt,
            max_output_tokens=max_output_tokens,
        )
        text = getattr(response, "text", "") or ""
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta:
            input_tokens = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
            total_tokens = int(getattr(usage_meta, "total_token_count", 0) or 0)
            candidate_tokens = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
            output_tokens = max(0, total_tokens - input_tokens) if total_tokens else candidate_tokens
            usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens or input_tokens + output_tokens,
                token_count_estimated=False,
            )
        else:
            usage = TokenUsage(
                input_tokens=prompt_tokens,
                output_tokens=estimate_tokens(text),
                total_tokens=prompt_tokens + estimate_tokens(text),
                token_count_estimated=True,
            )
        budget.record(usage)
        return LLMCallRecord(
            role=role,
            admitted=True,
            prompt_tokens_estimate=prompt_tokens,
            prompt_token_count_estimated=prompt_count_estimated,
            configured_generation_budget=generation_budget,
            generation_budget=max_output_tokens,
            max_output_tokens=max_output_tokens,
            thinking_budget=self.config.thinking_budget,
            thinking_config_applied=True,
            usage=usage,
            runtime_seconds=operation["provider_wall_time_seconds"],
            provider_attempts=operation["provider_attempts"],
            transient_retries=operation["transient_retries"],
            rate_limit_retries=operation["rate_limit_retries"],
            rate_limit_wait_seconds=operation["rate_limit_wait_seconds"],
            provider_wall_time_seconds=operation["provider_wall_time_seconds"],
            raw_output=text,
            finish_reason=_native_finish_reason(response),
        )

    def _count_prompt_tokens(self, prompt: str) -> tuple[int, bool]:
        try:
            result = self._client.models.count_tokens(model=self.config.name, contents=prompt)
            return int(getattr(result, "total_tokens")), False
        except Exception:
            return estimate_tokens(prompt), True

    def _generation_config(self, max_output_tokens: int):
        return self._types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=self._types.ThinkingConfig(thinking_budget=self.config.thinking_budget),
        )

    def _thinking_config_note(self) -> str | None:
        if self.config.thinking_budget is None:
            return "thinking budget not configured"
        return None

    def _generate_content_with_retries(self, prompt: str, max_output_tokens: int):
        provider_attempts = 0
        transient_retries = 0
        rate_limit_retries = 0
        rate_limit_wait_seconds = 0.0
        provider_wall_time_seconds = 0.0
        last_error: Exception | None = None
        config = self._generation_config(max_output_tokens)
        for retry_index in range(self.config.max_transient_retries + 1):
            wait_seconds = self._rate_limiter.wait()
            rate_limit_wait_seconds += wait_seconds
            start = time.perf_counter()
            provider_attempts += 1
            try:
                response = self._client.models.generate_content(
                    model=self.config.name,
                    contents=prompt,
                    config=config,
                )
                provider_wall_time_seconds += time.perf_counter() - start
                return response, {
                    "provider_attempts": provider_attempts,
                    "transient_retries": transient_retries,
                    "rate_limit_retries": rate_limit_retries,
                    "rate_limit_wait_seconds": rate_limit_wait_seconds,
                    "provider_wall_time_seconds": provider_wall_time_seconds,
                }
            except Exception as exc:
                provider_wall_time_seconds += time.perf_counter() - start
                last_error = exc
                status_code = _status_code(exc)
                if status_code not in TRANSIENT_STATUS_CODES:
                    raise ProviderInfrastructureError(
                        f"Permanent provider error during generate_content: {exc}",
                        provider_attempts=provider_attempts,
                        transient_retries=transient_retries,
                        rate_limit_retries=rate_limit_retries,
                        rate_limit_wait_seconds=rate_limit_wait_seconds,
                        provider_wall_time_seconds=provider_wall_time_seconds,
                    ) from exc
                if retry_index >= self.config.max_transient_retries:
                    break
                transient_retries += 1
                if status_code == 429:
                    rate_limit_retries += 1
                backoff = _retry_after_seconds(exc)
                if backoff is None:
                    backoff = min(60.0, (2**retry_index) + random.uniform(0.0, 1.0))
                rate_limit_wait_seconds += backoff
                time.sleep(backoff)
        raise ProviderInfrastructureError(
            f"Provider transient error retries exhausted: {last_error}",
            provider_attempts=provider_attempts,
            transient_retries=transient_retries,
            rate_limit_retries=rate_limit_retries,
            rate_limit_wait_seconds=rate_limit_wait_seconds,
            provider_wall_time_seconds=provider_wall_time_seconds,
        ) from last_error


class ScriptedLLMClient:
    """Deterministic test double used for CI and mocked end-to-end tests."""

    def __init__(self, outputs: list[str] | None = None) -> None:
        self.outputs = list(outputs or [])

    def generate(self, role: str, prompt: str, budget: BudgetManager, generation_budget: int) -> LLMCallRecord:
        decision = budget.admit(prompt, generation_budget)
        if not decision.admitted:
            return LLMCallRecord(
                role=role,
                admitted=False,
                prompt_tokens_estimate=decision.prompt_tokens_estimate,
                prompt_token_count_estimated=True,
                configured_generation_budget=generation_budget,
                generation_budget=generation_budget,
                max_output_tokens=generation_budget,
                provider_attempts=0,
                skipped_reason=decision.reason,
            )
        text = self.outputs.pop(0) if self.outputs else "ACCEPT\n"
        usage = TokenUsage(
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(text),
            total_tokens=estimate_tokens(prompt) + estimate_tokens(text),
            token_count_estimated=True,
        )
        budget.record(usage)
        return LLMCallRecord(
            role=role,
            admitted=True,
            prompt_tokens_estimate=decision.prompt_tokens_estimate,
            prompt_token_count_estimated=True,
            configured_generation_budget=generation_budget,
            generation_budget=generation_budget,
            max_output_tokens=generation_budget,
            usage=usage,
            provider_attempts=1,
            raw_output=text,
        )


TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _native_finish_reason(response) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    if reason is None:
        return None
    name = getattr(reason, "name", None)
    return str(name or reason)


def _status_code(exc: Exception) -> int | None:
    for attr in ["code", "status_code"]:
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    match = re.search(r"\b(408|429|500|502|503|504|400|401|403|404)\b", str(exc))
    return int(match.group(1)) if match else None


def _retry_after_seconds(exc: Exception) -> float | None:
    for attr in ["retry_after", "retry_delay"]:
        value = getattr(exc, attr, None)
        seconds = _seconds_from_retry_value(value)
        if seconds is not None:
            return seconds
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    for key in ["retry-after", "Retry-After"]:
        if key in headers:
            return _seconds_from_retry_value(headers[key])
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s?", str(exc))
    if match:
        return float(match.group(1))
    return None


def _seconds_from_retry_value(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    seconds = getattr(value, "seconds", None)
    nanos = getattr(value, "nanos", 0)
    if seconds is not None:
        return float(seconds) + float(nanos or 0) / 1_000_000_000
    text = str(value).strip()
    match = re.match(r"(\d+(?:\.\d+)?)s?$", text)
    if match:
        return float(match.group(1))
    return None
