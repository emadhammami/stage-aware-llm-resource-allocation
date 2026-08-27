from __future__ import annotations

from analysis.stage_aware.config import load_config
from analysis.stage_aware.replay import assert_replay_invariants, replay_run
from analysis.stage_aware.types import (
    AllocationEvent,
    HistoricalCall,
    HistoricalRun,
    PolicyName,
    Provenance,
    ReservationLifecycle,
    WorkflowStage,
)


def call(stage, prompt, output, cap):
    return HistoricalCall(
        stage=stage,
        role=stage.value.split("_")[0],
        prompt_tokens=prompt,
        prompt_provenance=Provenance.OBSERVED_EXACT,
        output_tokens=output,
        total_tokens=prompt + output,
        usage_provenance=Provenance.OBSERVED_EXACT,
        original_output_cap=cap,
        admitted=True,
    )


def direct_run() -> HistoricalRun:
    return HistoricalRun(
        experiment_id="synthetic",
        task_id="unit",
        method="evidence_gated",
        source_budget=4000,
        repetition=1,
        calls=(
            call(WorkflowStage.PLANNER, 100, 50, 384),
            call(WorkflowStage.EXECUTOR_1, 120, 100, 768),
            call(WorkflowStage.CRITIC, 100, 40, 384),
        ),
        validation_successes=(True,),
        row={},
    )


def retry_run(second_validation_success: bool) -> HistoricalRun:
    calls = [
        call(WorkflowStage.PLANNER, 100, 50, 384),
        call(WorkflowStage.EXECUTOR_1, 120, 100, 768),
        call(WorkflowStage.EXECUTOR_2, 140, 90, 768),
    ]
    if second_validation_success:
        calls.append(call(WorkflowStage.CRITIC, 110, 40, 384))
    return HistoricalRun(
        experiment_id=f"retry-{second_validation_success}",
        task_id="unit",
        method="evidence_gated",
        source_budget=4000,
        repetition=1,
        calls=tuple(calls),
        validation_successes=(False, second_validation_success),
        row={},
    )


def test_adaptive_replay_conserves_budget_and_resolves_reservations():
    config = load_config("research/stage_aware/phase0_config.yaml")
    result = replay_run(
        direct_run(),
        PolicyName.ADAPTIVE_STAGE_AWARE,
        2000,
        config,
        {stage: 100 for stage in WorkflowStage},
    )
    assert result.disposition == "historical_cap_compatible"
    assert result.known_consumed_tokens == 510
    assert result.material_action
    assert result.critic_admitted
    assert all(record.lifecycle != ReservationLifecycle.CREATED for record in result.reservations)
    assert any(record.lifecycle == ReservationLifecycle.CLAIMED for record in result.reservations)
    assert_replay_invariants(result)


def test_passing_first_validation_claims_critic_and_releases_retry_reservation():
    config = load_config("research/stage_aware/phase0_config.yaml")
    result = replay_run(
        direct_run(),
        PolicyName.ADAPTIVE_STAGE_AWARE,
        8000,
        config,
        {stage: 100 for stage in WorkflowStage},
    )
    critic = next(event for event in result.events if event.stage == WorkflowStage.CRITIC)
    assert critic.reservations_claimed
    assert critic.reservations_released
    assert any("executor_2" in value for value in critic.reservations_released)


def test_failing_first_validation_makes_retry_reachable_and_claimed():
    config = load_config("research/stage_aware/phase0_config.yaml")
    result = replay_run(
        retry_run(False),
        PolicyName.ADAPTIVE_STAGE_AWARE,
        8000,
        config,
        {stage: 100 for stage in WorkflowStage},
    )
    executor_1 = next(event for event in result.events if event.stage == WorkflowStage.EXECUTOR_1)
    executor_2 = next(event for event in result.events if event.stage == WorkflowStage.EXECUTOR_2)
    assert WorkflowStage.EXECUTOR_2 in executor_1.reachable_stages
    assert executor_2.reservations_claimed


def test_successful_second_validation_claims_critic():
    config = load_config("research/stage_aware/phase0_config.yaml")
    result = replay_run(
        retry_run(True),
        PolicyName.ADAPTIVE_STAGE_AWARE,
        8000,
        config,
        {stage: 100 for stage in WorkflowStage},
    )
    critic = next(event for event in result.events if event.stage == WorkflowStage.CRITIC)
    assert critic.reservations_claimed
    assert result.critic_admitted


def test_failed_second_validation_releases_unreachable_critic():
    config = load_config("research/stage_aware/phase0_config.yaml")
    result = replay_run(
        retry_run(False),
        PolicyName.ADAPTIVE_STAGE_AWARE,
        8000,
        config,
        {stage: 100 for stage in WorkflowStage},
    )
    terminal = result.events[-1]
    assert terminal.event_type == "reservation_reconciliation"
    assert terminal.reservations_released
    assert any("critic" in value for value in terminal.reservations_released)


def test_returned_allocation_is_available_to_the_next_historical_stage():
    config = load_config("research/stage_aware/phase0_config.yaml")
    result = replay_run(
        direct_run(),
        PolicyName.GREEDY,
        8000,
        config,
        {stage: 100 for stage in WorkflowStage},
    )
    planner = result.events[0]
    assert planner.returned_capacity > 0
    assert planner.reallocation_available == planner.returned_capacity
    assert planner.beneficiary_stage == WorkflowStage.EXECUTOR_1
    assert planner.reallocation_source == ("returned:planner",)


def test_allocation_event_serialization_round_trip():
    config = load_config("research/stage_aware/phase0_config.yaml")
    result = replay_run(
        direct_run(),
        PolicyName.FIXED_RESERVED,
        2000,
        config,
        {stage: 100 for stage in WorkflowStage},
    )
    serialized = result.events[0].to_dict()
    assert AllocationEvent.from_dict(serialized).to_dict() == serialized


def test_cap_incompatibility_stops_before_inventing_downstream_state():
    config = load_config("research/stage_aware/phase0_config.yaml")
    oversized = HistoricalRun(
        **{
            **direct_run().__dict__,
            "calls": (
                call(WorkflowStage.PLANNER, 100, 700, 768),
                *direct_run().calls[1:],
            ),
        }
    )
    result = replay_run(
        oversized,
        PolicyName.LEGACY_STATIC,
        4000,
        config,
        {stage: 100 for stage in WorkflowStage},
    )
    assert result.disposition == "cap_incompatible_counterfactual_unknown"
    assert result.processed_call_count == 1
    assert result.known_consumed_tokens == 100
    assert result.events[0].historical_output_compatible is False


def test_global_budget_denial_records_no_synthetic_usage():
    config = load_config("research/stage_aware/phase0_config.yaml")
    result = replay_run(
        direct_run(),
        PolicyName.LEGACY_STATIC,
        100,
        config,
        {stage: 100 for stage in WorkflowStage},
    )
    assert result.disposition == "admission_denied"
    assert result.known_consumed_tokens == 0
    assert result.events[0].admitted is False


def test_exact_minimum_boundary_admits_and_one_token_below_denies():
    config = load_config("research/stage_aware/phase0_config.yaml")
    one_call = HistoricalRun(
        experiment_id="boundary",
        task_id="unit",
        method="evidence_gated",
        source_budget=4000,
        calls=(call(WorkflowStage.PLANNER, 100, 10, 384),),
        validation_successes=(),
        repetition=1,
        row={},
    )
    admitted = replay_run(
        one_call,
        PolicyName.LEGACY_STATIC,
        132,
        config,
        {stage: 100 for stage in WorkflowStage},
    )
    denied = replay_run(
        one_call,
        PolicyName.LEGACY_STATIC,
        131,
        config,
        {stage: 100 for stage in WorkflowStage},
    )
    assert admitted.events[0].admitted is True
    assert admitted.events[0].allocated_output == 32
    assert denied.events[0].admitted is False


def test_estimated_and_missing_telemetry_remain_explicit():
    config = load_config("research/stage_aware/phase0_config.yaml")
    estimated = HistoricalCall(
        stage=WorkflowStage.PLANNER,
        role="planner",
        prompt_tokens=100,
        prompt_provenance=Provenance.DETERMINISTIC_ESTIMATE,
        output_tokens=None,
        total_tokens=None,
        usage_provenance=Provenance.MISSING,
        original_output_cap=384,
        admitted=True,
    )
    run = HistoricalRun(
        experiment_id="missing",
        task_id="unit",
        method="evidence_gated",
        source_budget=4000,
        repetition=1,
        calls=(estimated,),
        validation_successes=(),
        row={},
    )
    result = replay_run(
        run,
        PolicyName.LEGACY_STATIC,
        4000,
        config,
        {stage: 100 for stage in WorkflowStage},
    )
    assert result.disposition == "historical_data_missing"
    assert result.events[0].prompt_provenance == Provenance.DETERMINISTIC_ESTIMATE
