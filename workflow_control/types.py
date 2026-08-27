from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class Policy(StrEnum):
    LEGACY_STATIC = "legacy_static"
    GREEDY = "greedy"
    FIXED_RESERVATION = "fixed_reservation"
    PROPOSED = "proposed"


class Provenance(StrEnum):
    PROVIDER_EXACT = "provider_exact"
    TOKENIZER_EXACT = "tokenizer_exact"
    HISTORICAL_EXACT = "historical_exact"
    DETERMINISTIC_ESTIMATE = "deterministic_estimate"


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    minimum_output: int
    soft_output: int
    hard_output: int
    legacy_output: int

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_output <= self.soft_output <= self.hard_output:
            raise ValueError(f"invalid output bounds for {self.stage_id}")
        if not self.minimum_output <= self.legacy_output <= self.hard_output:
            raise ValueError(f"invalid legacy output bound for {self.stage_id}")


@dataclass(frozen=True)
class PromptEstimate:
    stage_id: str
    predicted_prompt_tokens: int
    provenance: str
    exact_prompt_tokens: int | None = None
    rendered_prompt_sha256: str | None = None

    @property
    def effective_tokens(self) -> int:
        return (
            self.exact_prompt_tokens
            if self.exact_prompt_tokens is not None
            else self.predicted_prompt_tokens
        )

    @property
    def prediction_error_tokens(self) -> int | None:
        if self.exact_prompt_tokens is None:
            return None
        return self.exact_prompt_tokens - self.predicted_prompt_tokens


@dataclass(frozen=True)
class InitialBudgetEstimate:
    estimator_version: str
    model_id: str
    task_id: str
    b_min: int
    b_soft: int
    calibration_rho: float
    b0: int
    route_assumptions: dict[str, Any]
    stages: tuple[dict[str, Any], ...]
    tokenization_provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int
    output_tokens: int
    finish_reason: str | None
    mapped_finish_class: str
    latency_seconds: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    model_revision: str | None
    tokenizer_revision: str | None
    transformers_version: str | None
    torch_version: str | None
    cuda_version: str | None
    dtype: str
    device: str
    decoding_parameters: dict[str, Any]
    chat_template_sha256: str
    rendered_prompt_sha256: str | None = None
    tokenizer_vocabulary_size: int | None = None
    safetensors_backend: str | None = None
    loader_workaround: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
