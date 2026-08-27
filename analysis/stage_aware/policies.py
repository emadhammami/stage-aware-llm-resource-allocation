from __future__ import annotations

import math
from collections.abc import Callable

from analysis.stage_aware.types import (
    AllocationContext,
    AllocationDecision,
    DenialReason,
    PolicyName,
    PromptRequirement,
    ReservationRequest,
    WorkflowStage,
)

PolicyFunction = Callable[[AllocationContext], AllocationDecision]


def _base_feasibility(context: AllocationContext) -> AllocationDecision | None:
    free = context.free_before_prompt
    if context.current_prompt_tokens is None:
        return AllocationDecision(
            policy=context.policy,
            stage=context.current_stage,
            admitted=False,
            allocated_output=0,
            free_before_prompt=free,
            free_after_prompt=free,
            denial_reason=DenialReason.HISTORICAL_DATA_MISSING,
            explanation="current prompt requirement is unavailable",
        )
    if context.current_prompt_tokens > free:
        return AllocationDecision(
            policy=context.policy,
            stage=context.current_stage,
            admitted=False,
            allocated_output=0,
            free_before_prompt=free,
            free_after_prompt=0,
            denial_reason=DenialReason.CURRENT_PROMPT_EXCEEDS_BUDGET,
            explanation="current prompt alone exceeds remaining global budget",
        )
    available = free - context.current_prompt_tokens
    if available < context.stage_spec.minimum_output:
        reason = (
            DenialReason.GLOBAL_MINIMUM_INFEASIBLE
            if context.consumed_tokens == 0
            else DenialReason.MINIMUM_OUTPUT_SHORTFALL
        )
        return AllocationDecision(
            policy=context.policy,
            stage=context.current_stage,
            admitted=False,
            allocated_output=0,
            free_before_prompt=free,
            free_after_prompt=available,
            denial_reason=reason,
            explanation="remaining capacity cannot fund current prompt plus minimum output",
        )
    return None


def legacy_static(context: AllocationContext) -> AllocationDecision:
    denied = _base_feasibility(context)
    if denied:
        return denied
    available = context.free_before_prompt - int(context.current_prompt_tokens or 0)
    allocation = min(context.stage_spec.legacy_output, available)
    return AllocationDecision(
        policy=context.policy,
        stage=context.current_stage,
        admitted=True,
        allocated_output=allocation,
        free_before_prompt=context.free_before_prompt,
        free_after_prompt=available,
        explanation="legacy configured cap clipped only by unspent global budget",
    )


def greedy(context: AllocationContext) -> AllocationDecision:
    denied = _base_feasibility(context)
    if denied:
        return denied
    available = context.free_before_prompt - int(context.current_prompt_tokens or 0)
    allocation = min(context.stage_spec.hard_output, available)
    return AllocationDecision(
        policy=context.policy,
        stage=context.current_stage,
        admitted=True,
        allocated_output=allocation,
        free_before_prompt=context.free_before_prompt,
        free_after_prompt=available,
        explanation="current stage receives all available capacity up to its hard cap",
    )


def _priority(requirement: PromptRequirement) -> int:
    order = {
        WorkflowStage.CRITIC: 0,
        WorkflowStage.EXECUTOR_1: 1,
        WorkflowStage.EXECUTOR_2: 2,
        WorkflowStage.PLANNER: 3,
    }
    return order[requirement.stage]


def fixed_reserved(context: AllocationContext) -> AllocationDecision:
    denied = _base_feasibility(context)
    if denied:
        return denied
    available = context.free_before_prompt - int(context.current_prompt_tokens or 0)
    critic = next(
        (
            requirement
            for requirement in context.future_requirements
            if requirement.stage == WorkflowStage.CRITIC
        ),
        None,
    )
    reservations: tuple[ReservationRequest, ...] = ()
    constraint = None
    if critic is not None:
        target = int(math.floor(context.total_budget * context.fixed_reservation_fraction))
        protectable = max(0, available - context.stage_spec.minimum_output)
        protected = min(target, protectable)
        if protected > 0:
            reservations = (
                ReservationRequest(
                    stage=WorkflowStage.CRITIC,
                    amount=protected,
                    prompt_tokens=critic.tokens,
                    prompt_provenance=critic.provenance,
                    conditional=critic.conditional,
                    priority=0,
                ),
            )
        if protected < target:
            constraint = DenialReason.RESERVATION_SHORTFALL
    allocation = min(context.stage_spec.hard_output, available - sum(r.amount for r in reservations))
    return AllocationDecision(
        policy=context.policy,
        stage=context.current_stage,
        admitted=True,
        allocated_output=allocation,
        free_before_prompt=context.free_before_prompt,
        free_after_prompt=available,
        desired_reservations=reservations,
        constraint_reason=constraint,
        discretionary_unallocated=max(0, available - allocation - sum(r.amount for r in reservations)),
        explanation="25% of total budget is protected for a reachable Critic after current minimum",
    )


def adaptive_stage_aware(context: AllocationContext) -> AllocationDecision:
    denied = _base_feasibility(context)
    if denied:
        return denied
    available = context.free_before_prompt - int(context.current_prompt_tokens or 0)
    reserve_capacity = available - context.stage_spec.minimum_output
    requests: list[ReservationRequest] = []
    constraint = None
    known_future = sorted(
        (requirement for requirement in context.future_requirements if requirement.known),
        key=_priority,
    )
    stage_specs = {spec.stage: spec for spec in context.stage_specs}
    if len(known_future) != len(context.future_requirements):
        constraint = DenialReason.FUTURE_PROMPT_FLOOR_INFEASIBLE
    for requirement in known_future:
        spec = stage_specs[requirement.stage]
        needed = int(requirement.tokens or 0) + spec.minimum_output
        amount = min(needed, reserve_capacity)
        if amount > 0:
            requests.append(
                ReservationRequest(
                    stage=requirement.stage,
                    amount=amount,
                    prompt_tokens=requirement.tokens,
                    prompt_provenance=requirement.provenance,
                    conditional=requirement.conditional,
                    priority=_priority(requirement),
                )
            )
        reserve_capacity -= amount
        if amount < needed:
            constraint = DenialReason.RESERVATION_SHORTFALL

    protected = sum(request.amount for request in requests)
    discretionary = max(0, available - context.stage_spec.minimum_output - protected)
    if not context.future_requirements:
        allocation = min(context.stage_spec.hard_output, available)
    else:
        current_gap = context.stage_spec.soft_output - context.stage_spec.minimum_output
        future_gap = sum(
            stage_specs[requirement.stage].soft_output
            - stage_specs[requirement.stage].minimum_output
            for requirement in known_future
        )
        share_denominator = max(1, current_gap + future_gap)
        current_extra = min(current_gap, math.floor(discretionary * current_gap / share_denominator))
        allocation = context.stage_spec.minimum_output + current_extra
    return AllocationDecision(
        policy=context.policy,
        stage=context.current_stage,
        admitted=True,
        allocated_output=allocation,
        free_before_prompt=context.free_before_prompt,
        free_after_prompt=available,
        desired_reservations=tuple(requests),
        constraint_reason=constraint,
        discretionary_unallocated=max(0, available - allocation - protected),
        explanation=(
            "current minimum is funded first; reachable minima are protected in Critic, "
            "Executor 1, Executor 2 priority; discretionary capacity is gap-weighted"
        ),
    )


POLICIES: dict[PolicyName, PolicyFunction] = {
    PolicyName.LEGACY_STATIC: legacy_static,
    PolicyName.GREEDY: greedy,
    PolicyName.FIXED_RESERVED: fixed_reserved,
    PolicyName.ADAPTIVE_STAGE_AWARE: adaptive_stage_aware,
}


def decide(context: AllocationContext) -> AllocationDecision:
    return POLICIES[context.policy](context)
