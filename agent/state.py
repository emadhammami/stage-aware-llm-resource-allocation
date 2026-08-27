from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

MethodName = Literal["single_shot", "pec", "pevc", "evidence_gated"]
ErrorCategory = Literal[
    "none",
    "patch_error",
    "syntax_error",
    "runtime_error",
    "assertion_failure",
    "timeout",
    "import_error",
    "other",
]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    token_count_estimated: bool = False


class LLMCallRecord(BaseModel):
    role: str
    admitted: bool
    prompt_tokens_estimate: int
    prompt_token_count_estimated: bool = False
    configured_generation_budget: int | None = None
    generation_budget: int
    max_output_tokens: int | None = None
    thinking_budget: int | None = None
    thinking_config_applied: bool = False
    thinking_config_note: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    runtime_seconds: float = 0.0
    provider_attempts: int = 0
    transient_retries: int = 0
    rate_limit_retries: int = 0
    rate_limit_wait_seconds: float = 0.0
    provider_wall_time_seconds: float = 0.0
    raw_output: str = ""
    finish_reason: str | None = None
    skipped_reason: str | None = None


class PlannerOutput(BaseModel):
    hypothesis: str
    target_function: str | None = None


class ExecutorOutput(BaseModel):
    proposed_code: str
    rationale: str = ""


class CriticOutput(BaseModel):
    accepted: bool
    rationale: str = ""


class PatchRecord(BaseModel):
    applied: bool
    syntax_valid: bool
    affected_function: str | None = None
    original_snippet: str = ""
    proposed_snippet: str = ""
    error: str | None = None


class ValidationResult(BaseModel):
    success: bool = False
    tests_passed: int = 0
    tests_failed: int = 0
    tests_total: int = 0
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    timed_out: bool = False
    runtime_seconds: float = 0.0
    error_category: ErrorCategory = "none"
    failing_test_info: str = ""


class RepairState(BaseModel):
    experiment_id: str
    task_id: str
    method: MethodName
    token_budget: int
    repetition: int = 1
    is_pilot: bool = False
    run_status: Literal["completed", "infrastructure_error"] = "completed"
    max_executor_attempts: int = 1
    planner_generation_budget: int = 384
    executor_generation_budget: int = 768
    critic_generation_budget: int = 384
    started_at_utc: str = Field(default_factory=utc_now_iso)
    ended_at_utc: str | None = None

    original_code: str = ""
    planner: PlannerOutput | None = None
    executor_outputs: list[ExecutorOutput] = Field(default_factory=list)
    critic: CriticOutput | None = None
    patch: PatchRecord | None = None
    validations: list[ValidationResult] = Field(default_factory=list)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    node_telemetry: list[dict[str, Any]] = Field(default_factory=list)

    retry_used: bool = False
    early_exit: bool = False
    budget_exhausted: bool = False
    budget_violation: bool = False
    provider_attempts: int = 0
    transient_retries: int = 0
    rate_limit_retries: int = 0
    rate_limit_wait_seconds: float = 0.0
    provider_wall_time_seconds: float = 0.0
    infrastructure_error: str | None = None
    final_error_category: ErrorCategory = "none"

    @property
    def tokens_used(self) -> TokenUsage:
        usage = TokenUsage()
        estimated = False
        for call in self.llm_calls:
            usage.input_tokens += call.usage.input_tokens
            usage.output_tokens += call.usage.output_tokens
            usage.total_tokens += call.usage.total_tokens
            estimated = estimated or call.usage.token_count_estimated
        usage.token_count_estimated = estimated
        return usage

    @property
    def latest_validation(self) -> ValidationResult | None:
        return self.validations[-1] if self.validations else None

    def add_event(self, node: str, **data: Any) -> None:
        self.node_telemetry.append(
            {
                "timestamp_utc": utc_now_iso(),
                "node": node,
                "tokens_used": self.tokens_used.total_tokens,
                **data,
            }
        )
