from __future__ import annotations

import pytest

from analysis.stage_aware.config import load_config
from analysis.stage_aware.policies import decide
from analysis.stage_aware.types import (
    AllocationContext,
    DenialReason,
    PolicyName,
    PromptRequirement,
    Provenance,
    WorkflowStage,
)


@pytest.fixture(scope="module")
def config():
    return load_config("research/stage_aware/phase0_config.yaml")


def make_context(config, policy: PolicyName, **overrides) -> AllocationContext:
    stage = overrides.pop("current_stage", WorkflowStage.PLANNER)
    values = {
        "policy": policy,
        "total_budget": 1000,
        "consumed_tokens": 0,
        "current_stage": stage,
        "current_prompt_tokens": 100,
        "current_prompt_provenance": Provenance.OBSERVED_EXACT,
        "stage_spec": config.stage_specs[stage],
        "stage_specs": tuple(config.stage_specs.values()),
        "future_requirements": (),
        "fixed_reservation_fraction": 0.25,
        "material_action_threshold": 32,
    }
    values.update(overrides)
    return AllocationContext(**values)


def test_legacy_static_uses_legacy_cap(config):
    decision = decide(make_context(config, PolicyName.LEGACY_STATIC))
    assert decision.admitted
    assert decision.allocated_output == 384
    assert not decision.desired_reservations


def test_greedy_uses_hard_cap(config):
    decision = decide(make_context(config, PolicyName.GREEDY))
    assert decision.admitted
    assert decision.allocated_output == 768


def test_fixed_reserved_protects_exact_quarter_for_reachable_critic(config):
    future = (
        PromptRequirement(
            WorkflowStage.CRITIC,
            tokens=100,
            provenance=Provenance.OBSERVED_EXACT,
            conditional=True,
        ),
    )
    decision = decide(
        make_context(config, PolicyName.FIXED_RESERVED, future_requirements=future)
    )
    assert decision.admitted
    assert decision.protected_tokens == 250
    assert decision.allocated_output == 650
    assert decision.desired_reservations[0].stage == WorkflowStage.CRITIC


def test_fixed_reserve_never_displaces_current_minimum(config):
    future = (
        PromptRequirement(
            WorkflowStage.CRITIC,
            tokens=400,
            provenance=Provenance.OBSERVED_EXACT,
            conditional=True,
        ),
    )
    decision = decide(
        make_context(
            config,
            PolicyName.FIXED_RESERVED,
            consumed_tokens=850,
            current_prompt_tokens=100,
            future_requirements=future,
        )
    )
    assert decision.admitted
    assert decision.allocated_output == 32
    assert decision.protected_tokens == 18
    assert decision.constraint_reason == DenialReason.RESERVATION_SHORTFALL


def test_adaptive_reservation_priority_is_critic_then_executors(config):
    future = tuple(
        PromptRequirement(stage, 100, Provenance.OBSERVED_EXACT, conditional=True)
        for stage in (
            WorkflowStage.EXECUTOR_2,
            WorkflowStage.EXECUTOR_1,
            WorkflowStage.CRITIC,
        )
    )
    decision = decide(
        make_context(config, PolicyName.ADAPTIVE_STAGE_AWARE, future_requirements=future)
    )
    assert decision.admitted
    assert [request.stage for request in decision.desired_reservations] == [
        WorkflowStage.CRITIC,
        WorkflowStage.EXECUTOR_1,
        WorkflowStage.EXECUTOR_2,
    ]
    assert decision.allocated_output >= config.stage_specs[WorkflowStage.PLANNER].minimum_output
    assert decision.allocated_output <= config.stage_specs[WorkflowStage.PLANNER].hard_output


@pytest.mark.parametrize("policy", list(PolicyName))
def test_every_policy_denies_missing_current_prompt(config, policy):
    decision = decide(
        make_context(config, policy, current_prompt_tokens=None)
    )
    assert not decision.admitted
    assert decision.denial_reason == DenialReason.HISTORICAL_DATA_MISSING
    assert decision.allocated_output == 0


@pytest.mark.parametrize("policy", list(PolicyName))
def test_every_policy_denies_below_global_minimum(config, policy):
    decision = decide(
        make_context(config, policy, total_budget=120, current_prompt_tokens=100)
    )
    assert not decision.admitted
    assert decision.denial_reason == DenialReason.GLOBAL_MINIMUM_INFEASIBLE
