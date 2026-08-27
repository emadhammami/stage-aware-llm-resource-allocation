from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class WorkflowStage(StrEnum):
    PLANNER = "planner"
    EXECUTOR_1 = "executor_1"
    EXECUTOR_2 = "executor_2"
    CRITIC = "critic"


class PolicyName(StrEnum):
    LEGACY_STATIC = "legacy_static"
    GREEDY = "greedy"
    FIXED_RESERVED = "fixed_reserved"
    ADAPTIVE_STAGE_AWARE = "adaptive_stage_aware"


class Provenance(StrEnum):
    OBSERVED_EXACT = "observed_exact"
    RECONSTRUCTED_EXACT = "reconstructed_exact"
    DETERMINISTIC_ESTIMATE = "deterministic_estimate"
    MISSING = "missing"
    COUNTERFACTUAL_UNKNOWN = "counterfactual_unknown"


class DenialReason(StrEnum):
    GLOBAL_MINIMUM_INFEASIBLE = "global_minimum_infeasible"
    CURRENT_PROMPT_EXCEEDS_BUDGET = "current_prompt_exceeds_budget"
    MINIMUM_OUTPUT_SHORTFALL = "minimum_output_shortfall"
    PROTECTED_RESERVATION_CONFLICT = "protected_reservation_conflict"
    FUTURE_PROMPT_FLOOR_INFEASIBLE = "future_prompt_floor_infeasible"
    RESERVATION_SHORTFALL = "reservation_shortfall"
    HISTORICAL_DATA_MISSING = "historical_data_missing"
    NOT_REACHABLE = "not_reachable"
    NOT_APPLICABLE = "not_applicable"


class ReservationLifecycle(StrEnum):
    CREATED = "created"
    CLAIMED = "claimed"
    RELEASED = "released"
    STRANDED = "stranded"


@dataclass(frozen=True)
class StageSpec:
    stage: WorkflowStage
    minimum_output: int
    soft_output: int
    hard_output: int
    legacy_output: int

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_output <= self.soft_output <= self.hard_output:
            raise ValueError(f"invalid output bounds for {self.stage}")


@dataclass(frozen=True)
class PromptRequirement:
    stage: WorkflowStage
    tokens: int | None
    provenance: Provenance
    conditional: bool = False

    @property
    def known(self) -> bool:
        return self.tokens is not None and self.provenance not in {
            Provenance.MISSING,
            Provenance.COUNTERFACTUAL_UNKNOWN,
        }


@dataclass(frozen=True)
class ReservationRequest:
    stage: WorkflowStage
    amount: int
    prompt_tokens: int | None
    prompt_provenance: Provenance
    conditional: bool
    priority: int


@dataclass(frozen=True)
class AllocationContext:
    policy: PolicyName
    total_budget: int
    consumed_tokens: int
    current_stage: WorkflowStage
    current_prompt_tokens: int | None
    current_prompt_provenance: Provenance
    stage_spec: StageSpec
    stage_specs: tuple[StageSpec, ...]
    future_requirements: tuple[PromptRequirement, ...] = ()
    fixed_reservation_fraction: float = 0.25
    material_action_threshold: int = 32

    @property
    def free_before_prompt(self) -> int:
        return max(0, self.total_budget - self.consumed_tokens)


@dataclass(frozen=True)
class AllocationDecision:
    policy: PolicyName
    stage: WorkflowStage
    admitted: bool
    allocated_output: int
    free_before_prompt: int
    free_after_prompt: int
    desired_reservations: tuple[ReservationRequest, ...] = ()
    denial_reason: DenialReason | None = None
    constraint_reason: DenialReason | None = None
    discretionary_unallocated: int = 0
    explanation: str = ""

    @property
    def protected_tokens(self) -> int:
        return sum(reservation.amount for reservation in self.desired_reservations)


@dataclass(frozen=True)
class HistoricalCall:
    stage: WorkflowStage
    role: str
    prompt_tokens: int | None
    prompt_provenance: Provenance
    output_tokens: int | None
    total_tokens: int | None
    usage_provenance: Provenance
    original_output_cap: int | None
    admitted: bool
    skipped_reason: str | None = None


@dataclass(frozen=True)
class HistoricalRun:
    experiment_id: str
    task_id: str
    method: str
    source_budget: int
    repetition: int
    calls: tuple[HistoricalCall, ...]
    validation_successes: tuple[bool, ...]
    row: dict[str, Any] = field(compare=False, repr=False)


@dataclass
class ReservationRecord:
    reservation_id: str
    stage: WorkflowStage
    amount: int
    initial_amount: int
    created_event: int
    prompt_provenance: Provenance
    conditional: bool
    lifecycle: ReservationLifecycle = ReservationLifecycle.CREATED
    resolved_event: int | None = None
    resolution_note: str | None = None
    resize_count: int = 0
    total_increase: int = 0
    total_decrease: int = 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["stage"] = self.stage.value
        data["prompt_provenance"] = self.prompt_provenance.value
        data["lifecycle"] = self.lifecycle.value
        return data


@dataclass
class AllocationEvent:
    schema_version: str
    experiment_id: str
    task_id: str
    policy: PolicyName
    target_budget: int
    event_index: int
    event_type: str
    stage: WorkflowStage | None
    attempt: int | None
    admitted: bool | None
    denial_reason: DenialReason | None
    constraint_reason: DenialReason | None
    prompt_tokens: int | None
    prompt_provenance: Provenance
    minimum_output: int
    soft_output: int
    hard_output: int
    requested_output: int
    allocation_envelope: int
    historical_output_tokens: int | None
    historical_output_compatible: bool | None
    original_output_cap: int | None
    allocated_output: int
    consumed_before: int
    protected_before: int
    protected_after: int
    reservation_created_amount: int
    reservation_resized_amount: int
    reservation_claimed_amount: int
    reservation_stranded_amount: int
    reservation_shortfall: int
    returned_capacity: int
    released_capacity: int
    reallocation_available: int
    reallocation_source: tuple[str, ...]
    beneficiary_stage: WorkflowStage | None
    reachable_stages: tuple[WorkflowStage, ...]
    structural_infeasible: bool
    historical_cap_binding_proxy: float | None
    reservation_prediction_error: int | None
    known_consumed: int
    cumulative_known_consumed: int
    remaining_before: int
    remaining_after: int
    material_action: bool
    material_reasons: tuple[str, ...]
    reservations_created: tuple[str, ...] = ()
    reservations_claimed: tuple[str, ...] = ()
    reservations_released: tuple[str, ...] = ()
    reservations_resized: tuple[str, ...] = ()
    reservations_stranded: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["policy"] = self.policy.value
        data["stage"] = self.stage.value if self.stage else None
        data["beneficiary_stage"] = (
            self.beneficiary_stage.value if self.beneficiary_stage else None
        )
        data["denial_reason"] = self.denial_reason.value if self.denial_reason else None
        data["constraint_reason"] = (
            self.constraint_reason.value if self.constraint_reason else None
        )
        data["prompt_provenance"] = self.prompt_provenance.value
        for key in (
            "material_reasons",
            "reservations_created",
            "reservations_claimed",
            "reservations_released",
            "reservations_resized",
            "reservations_stranded",
            "reallocation_source",
        ):
            data[key] = "|".join(data[key])
        data["reachable_stages"] = "|".join(stage.value for stage in self.reachable_stages)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AllocationEvent:
        values = dict(data)
        values["policy"] = PolicyName(values["policy"])
        values["stage"] = WorkflowStage(values["stage"]) if values.get("stage") else None
        values["beneficiary_stage"] = (
            WorkflowStage(values["beneficiary_stage"])
            if values.get("beneficiary_stage")
            else None
        )
        values["denial_reason"] = (
            DenialReason(values["denial_reason"]) if values.get("denial_reason") else None
        )
        values["constraint_reason"] = (
            DenialReason(values["constraint_reason"])
            if values.get("constraint_reason")
            else None
        )
        values["prompt_provenance"] = Provenance(values["prompt_provenance"])
        for key in (
            "material_reasons",
            "reservations_created",
            "reservations_claimed",
            "reservations_released",
            "reservations_resized",
            "reservations_stranded",
            "reallocation_source",
        ):
            value = values.get(key, ())
            values[key] = tuple(value.split("|")) if isinstance(value, str) and value else tuple(value)
        reachable = values.get("reachable_stages", ())
        if isinstance(reachable, str):
            reachable = tuple(filter(None, reachable.split("|")))
        values["reachable_stages"] = tuple(WorkflowStage(stage) for stage in reachable)
        return cls(**values)


@dataclass
class ReplayResult:
    experiment_id: str
    task_id: str
    policy: PolicyName
    target_budget: int
    historical_route: str
    historical_call_count: int
    processed_call_count: int
    admitted_call_count: int
    historical_output_compatible_calls: int
    known_consumed_tokens: int
    allocated_output_tokens: int
    returned_capacity: int
    released_capacity: int
    protected_capacity_peak: int
    material_action: bool
    material_action_count: int
    disposition: str
    terminal_reason: str
    critic_historically_present: bool
    critic_admitted: bool
    finish_reason_available: bool
    events: list[AllocationEvent]
    reservations: list[ReservationRecord]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("events")
        data.pop("reservations")
        data["policy"] = self.policy.value
        return data
