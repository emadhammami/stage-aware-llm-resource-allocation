from __future__ import annotations

from dataclasses import replace

from analysis.stage_aware.config import Phase0Config
from analysis.stage_aware.policies import decide
from analysis.stage_aware.types import (
    AllocationContext,
    AllocationEvent,
    DenialReason,
    HistoricalRun,
    PolicyName,
    PromptRequirement,
    Provenance,
    ReplayResult,
    ReservationLifecycle,
    ReservationRecord,
    WorkflowStage,
)


def _future_stages(stage: WorkflowStage) -> tuple[tuple[WorkflowStage, bool], ...]:
    if stage == WorkflowStage.PLANNER:
        return (
            (WorkflowStage.EXECUTOR_1, False),
            (WorkflowStage.CRITIC, True),
            (WorkflowStage.EXECUTOR_2, True),
        )
    if stage == WorkflowStage.EXECUTOR_1:
        return ((WorkflowStage.CRITIC, True), (WorkflowStage.EXECUTOR_2, True))
    if stage == WorkflowStage.EXECUTOR_2:
        return ((WorkflowStage.CRITIC, True),)
    return ()


def _future_requirements(
    run: HistoricalRun,
    call_index: int,
    stage: WorkflowStage,
    prompt_fallbacks: dict[WorkflowStage, int],
) -> tuple[PromptRequirement, ...]:
    later_calls = run.calls[call_index + 1 :]
    requirements: list[PromptRequirement] = []
    for future_stage, conditional in _future_stages(stage):
        observed = next((call for call in later_calls if call.stage == future_stage), None)
        if observed and observed.prompt_tokens is not None:
            requirements.append(
                PromptRequirement(
                    stage=future_stage,
                    tokens=observed.prompt_tokens,
                    provenance=observed.prompt_provenance,
                    conditional=conditional,
                )
            )
        elif future_stage in prompt_fallbacks:
            requirements.append(
                PromptRequirement(
                    stage=future_stage,
                    tokens=prompt_fallbacks[future_stage],
                    provenance=Provenance.DETERMINISTIC_ESTIMATE,
                    conditional=conditional,
                )
            )
        else:
            requirements.append(
                PromptRequirement(
                    stage=future_stage,
                    tokens=None,
                    provenance=Provenance.MISSING,
                    conditional=conditional,
                )
            )
    return tuple(requirements)


def _context(
    *,
    config: Phase0Config,
    policy: PolicyName,
    target_budget: int,
    consumed: int,
    stage: WorkflowStage,
    prompt_tokens: int | None,
    prompt_provenance: Provenance,
    future: tuple[PromptRequirement, ...],
) -> AllocationContext:
    return AllocationContext(
        policy=policy,
        total_budget=target_budget,
        consumed_tokens=consumed,
        current_stage=stage,
        current_prompt_tokens=prompt_tokens,
        current_prompt_provenance=prompt_provenance,
        stage_spec=config.stage_specs[stage],
        stage_specs=tuple(config.stage_specs.values()),
        future_requirements=future,
        fixed_reservation_fraction=config.fixed_reservation_fraction,
        material_action_threshold=config.material_action_threshold,
    )


def _reconcile_reservations(
    *,
    event_index: int,
    desired: tuple,
    active: dict[WorkflowStage, ReservationRecord],
    all_records: list[ReservationRecord],
    counters: dict[WorkflowStage, int],
) -> tuple[list[str], int, list[str], int, list[str], int]:
    desired_by_stage = {request.stage: request for request in desired}
    released: list[str] = []
    created: list[str] = []
    released_capacity = 0
    resized: list[str] = []
    resized_amount = 0
    created_capacity = 0
    for stage, record in list(active.items()):
        request = desired_by_stage.get(stage)
        if request is not None:
            if request.amount != record.amount:
                delta = request.amount - record.amount
                record.resize_count += 1
                record.total_increase += max(0, delta)
                record.total_decrease += max(0, -delta)
                record.amount = request.amount
                resized.append(record.reservation_id)
                resized_amount += abs(delta)
                released_capacity += max(0, -delta)
            continue
        record.lifecycle = ReservationLifecycle.RELEASED
        record.resolved_event = event_index
        record.resolution_note = "reservation target changed or stage no longer protected"
        released.append(record.reservation_id)
        released_capacity += record.amount
        del active[stage]
    for stage, request in desired_by_stage.items():
        if stage in active or request.amount <= 0:
            continue
        counters[stage] = counters.get(stage, 0) + 1
        reservation_id = f"{stage.value}:r{counters[stage]}"
        record = ReservationRecord(
            reservation_id=reservation_id,
            stage=stage,
            amount=request.amount,
            initial_amount=request.amount,
            created_event=event_index,
            prompt_provenance=request.prompt_provenance,
            conditional=request.conditional,
        )
        active[stage] = record
        all_records.append(record)
        created.append(reservation_id)
        created_capacity += request.amount
    return (
        created,
        created_capacity,
        released,
        released_capacity,
        resized,
        resized_amount,
    )


def _material_reasons(
    *,
    policy: PolicyName,
    admitted: bool,
    allocated_output: int,
    baseline_admitted: bool,
    baseline_output: int,
    protected_tokens: int,
    claimed_capacity: int,
    released_capacity: int,
    resized_capacity: int,
    returned_capacity: int,
    threshold: int,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if admitted != baseline_admitted:
        reasons.append("admission_changed")
    if admitted and baseline_admitted and abs(allocated_output - baseline_output) >= threshold:
        reasons.append("allocation_changed")
    if protected_tokens >= threshold and allocated_output + protected_tokens <= baseline_output:
        reasons.append("protected_current_capacity")
    if policy in {PolicyName.FIXED_RESERVED, PolicyName.ADAPTIVE_STAGE_AWARE}:
        if claimed_capacity >= threshold:
            reasons.append("reservation_claimed")
        if released_capacity >= threshold:
            reasons.append("reservation_released")
        if resized_capacity >= threshold:
            reasons.append("reservation_resized")
        if returned_capacity >= threshold:
            reasons.append("capacity_returned")
    return tuple(dict.fromkeys(reasons))


def replay_run(
    run: HistoricalRun,
    policy: PolicyName,
    target_budget: int,
    config: Phase0Config,
    prompt_fallbacks: dict[WorkflowStage, int],
) -> ReplayResult:
    consumed = 0
    events: list[AllocationEvent] = []
    active: dict[WorkflowStage, ReservationRecord] = {}
    reservations: list[ReservationRecord] = []
    counters: dict[WorkflowStage, int] = {}
    processed = 0
    admitted_count = 0
    compatible_count = 0
    allocated_total = 0
    returned_total = 0
    released_total = 0
    protected_peak = 0
    terminal_reason = "historical_route_complete"
    disposition = "historical_cap_compatible"
    critic_admitted = False

    for call_index, call in enumerate(run.calls):
        event_index = len(events)
        remaining_before = max(0, target_budget - consumed)
        protected_before = sum(record.amount for record in active.values())
        claimed: list[str] = []
        claimed_capacity = 0
        if call.stage in active:
            claimed_record = active.pop(call.stage)
            claimed_record.lifecycle = ReservationLifecycle.CLAIMED
            claimed_record.resolved_event = event_index
            claimed_record.resolution_note = "protected stage became current"
            claimed.append(claimed_record.reservation_id)
            claimed_capacity = claimed_record.amount

        future = _future_requirements(run, call_index, call.stage, prompt_fallbacks)
        context = _context(
            config=config,
            policy=policy,
            target_budget=target_budget,
            consumed=consumed,
            stage=call.stage,
            prompt_tokens=call.prompt_tokens,
            prompt_provenance=call.prompt_provenance,
            future=future,
        )
        decision = decide(context)
        baseline = decide(replace(context, policy=PolicyName.LEGACY_STATIC))
        (
            created,
            created_capacity,
            released,
            released_capacity,
            resized,
            resized_capacity,
        ) = _reconcile_reservations(
            event_index=event_index,
            desired=decision.desired_reservations if decision.admitted else (),
            active=active,
            all_records=reservations,
            counters=counters,
        )
        released_total += released_capacity
        protected_after = sum(record.amount for record in active.values())
        protected_peak = max(protected_peak, protected_before, protected_after)

        historical_compatible: bool | None = None
        known_consumed = 0
        returned_capacity = 0
        notes = decision.explanation
        if not decision.admitted:
            disposition = "admission_denied"
            terminal_reason = (
                decision.denial_reason.value
                if decision.denial_reason
                else DenialReason.NOT_APPLICABLE.value
            )
        elif call.output_tokens is None or call.total_tokens is None:
            disposition = "historical_data_missing"
            terminal_reason = DenialReason.HISTORICAL_DATA_MISSING.value
            historical_compatible = None
        elif decision.allocated_output < call.output_tokens:
            disposition = "cap_incompatible_counterfactual_unknown"
            terminal_reason = "historical_output_exceeds_allocation"
            historical_compatible = False
            known_consumed = int(call.prompt_tokens or 0)
            notes += "; only the prompt-cost lower bound remains known after cap divergence"
        else:
            historical_compatible = True
            known_consumed = call.total_tokens
            returned_capacity = decision.allocated_output - call.output_tokens
            compatible_count += 1
            notes += "; compatible means output fits cap, not that a missing finish reason proves non-binding"

        admitted_count += int(decision.admitted)
        allocated_total += decision.allocated_output
        returned_total += returned_capacity
        material_reasons = _material_reasons(
            policy=policy,
            admitted=decision.admitted,
            allocated_output=decision.allocated_output,
            baseline_admitted=baseline.admitted,
            baseline_output=baseline.allocated_output,
            protected_tokens=decision.protected_tokens,
            claimed_capacity=claimed_capacity,
            released_capacity=released_capacity,
            resized_capacity=resized_capacity,
            returned_capacity=returned_capacity,
            threshold=config.material_action_threshold,
        )
        consumed += known_consumed
        minimum_output = config.stage_specs[call.stage].minimum_output
        requested_output = (
            config.stage_specs[call.stage].legacy_output
            if policy == PolicyName.LEGACY_STATIC
            else config.stage_specs[call.stage].hard_output
        )
        reservation_shortfall = 0
        if decision.constraint_reason == DenialReason.RESERVATION_SHORTFALL:
            if policy == PolicyName.FIXED_RESERVED:
                reservation_shortfall = max(
                    0,
                    int(target_budget * config.fixed_reservation_fraction)
                    - decision.protected_tokens,
                )
            else:
                required_future = sum(
                    int(requirement.tokens or 0)
                    + config.stage_specs[requirement.stage].minimum_output
                    for requirement in future
                    if requirement.known
                )
                reservation_shortfall = max(0, required_future - decision.protected_tokens)
        reallocation_sources = tuple(
            source
            for source, amount in (
                (f"returned:{call.stage.value}", returned_capacity),
                ("released_reservation", released_capacity),
            )
            if amount > 0
        )
        next_stage = (
            run.calls[call_index + 1].stage
            if call_index + 1 < len(run.calls) and historical_compatible is True
            else None
        )
        events.append(
            AllocationEvent(
                schema_version=config.schema_version,
                experiment_id=run.experiment_id,
                task_id=run.task_id,
                policy=policy,
                target_budget=target_budget,
                event_index=event_index,
                event_type="allocation_decision",
                stage=call.stage,
                attempt=(
                    2 if call.stage == WorkflowStage.EXECUTOR_2 else 1
                ),
                admitted=decision.admitted,
                denial_reason=decision.denial_reason,
                constraint_reason=decision.constraint_reason,
                prompt_tokens=call.prompt_tokens,
                prompt_provenance=call.prompt_provenance,
                minimum_output=minimum_output,
                soft_output=config.stage_specs[call.stage].soft_output,
                hard_output=config.stage_specs[call.stage].hard_output,
                requested_output=requested_output,
                allocation_envelope=int(call.prompt_tokens or 0) + decision.allocated_output,
                historical_output_tokens=call.output_tokens,
                historical_output_compatible=historical_compatible,
                original_output_cap=call.original_output_cap,
                allocated_output=decision.allocated_output,
                consumed_before=consumed - known_consumed,
                protected_before=protected_before,
                protected_after=protected_after,
                reservation_created_amount=created_capacity,
                reservation_resized_amount=resized_capacity,
                reservation_claimed_amount=claimed_capacity,
                reservation_stranded_amount=0,
                reservation_shortfall=reservation_shortfall,
                returned_capacity=returned_capacity,
                released_capacity=released_capacity,
                reallocation_available=returned_capacity + released_capacity,
                reallocation_source=reallocation_sources,
                beneficiary_stage=next_stage,
                reachable_stages=tuple(requirement.stage for requirement in future),
                structural_infeasible=(
                    not decision.admitted
                    and decision.denial_reason
                    in {
                        DenialReason.GLOBAL_MINIMUM_INFEASIBLE,
                        DenialReason.CURRENT_PROMPT_EXCEEDS_BUDGET,
                        DenialReason.MINIMUM_OUTPUT_SHORTFALL,
                    }
                ),
                historical_cap_binding_proxy=(
                    call.output_tokens / call.original_output_cap
                    if call.output_tokens is not None and call.original_output_cap
                    else None
                ),
                reservation_prediction_error=(
                    claimed_capacity - (int(call.prompt_tokens or 0) + minimum_output)
                    if claimed_capacity
                    else None
                ),
                known_consumed=known_consumed,
                cumulative_known_consumed=consumed,
                remaining_before=remaining_before,
                remaining_after=max(0, target_budget - consumed),
                material_action=bool(material_reasons),
                material_reasons=material_reasons,
                reservations_created=tuple(created),
                reservations_claimed=tuple(claimed),
                reservations_released=tuple(released),
                reservations_resized=tuple(resized),
                notes=notes,
            )
        )
        processed += 1
        if call.stage == WorkflowStage.CRITIC and decision.admitted:
            critic_admitted = True
        if disposition != "historical_cap_compatible":
            break

    if active:
        final_event_index = len(events)
        release_at_end = disposition == "historical_cap_compatible" and processed == len(run.calls)
        released_at_end: list[str] = []
        stranded: list[str] = []
        released_capacity = 0
        for record in list(active.values()):
            if release_at_end:
                record.lifecycle = ReservationLifecycle.RELEASED
                record.resolution_note = "historical route ended; protected stage is unreachable"
                released_at_end.append(record.reservation_id)
                released_capacity += record.amount
            else:
                record.lifecycle = ReservationLifecycle.STRANDED
                record.resolution_note = "replay stopped before future reachability could be established"
                stranded.append(record.reservation_id)
            record.resolved_event = final_event_index
        active.clear()
        released_total += released_capacity
        events.append(
            AllocationEvent(
                schema_version=config.schema_version,
                experiment_id=run.experiment_id,
                task_id=run.task_id,
                policy=policy,
                target_budget=target_budget,
                event_index=final_event_index,
                event_type="reservation_reconciliation",
                stage=None,
                attempt=None,
                admitted=None,
                denial_reason=None,
                constraint_reason=None,
                prompt_tokens=None,
                prompt_provenance=Provenance.RECONSTRUCTED_EXACT,
                minimum_output=0,
                soft_output=0,
                hard_output=0,
                requested_output=0,
                allocation_envelope=0,
                historical_output_tokens=None,
                historical_output_compatible=None,
                original_output_cap=None,
                allocated_output=0,
                consumed_before=consumed,
                protected_before=sum(
                    record.amount
                    for record in reservations
                    if record.resolved_event == final_event_index
                ),
                protected_after=0,
                reservation_created_amount=0,
                reservation_resized_amount=0,
                reservation_claimed_amount=0,
                reservation_stranded_amount=sum(
                    record.amount
                    for record in reservations
                    if record.resolved_event == final_event_index
                    and record.lifecycle == ReservationLifecycle.STRANDED
                ),
                reservation_shortfall=0,
                returned_capacity=0,
                released_capacity=released_capacity,
                reallocation_available=released_capacity,
                reallocation_source=("released_reservation",) if released_capacity else (),
                beneficiary_stage=None,
                reachable_stages=(),
                structural_infeasible=False,
                historical_cap_binding_proxy=None,
                reservation_prediction_error=None,
                known_consumed=0,
                cumulative_known_consumed=consumed,
                remaining_before=max(0, target_budget - consumed),
                remaining_after=max(0, target_budget - consumed),
                material_action=(
                    policy in {PolicyName.FIXED_RESERVED, PolicyName.ADAPTIVE_STAGE_AWARE}
                    and released_capacity >= config.material_action_threshold
                ),
                material_reasons=("reservation_released",) if released_capacity else (),
                reservations_released=tuple(released_at_end),
                reservations_stranded=tuple(stranded),
                notes=(
                    "terminal reservation release"
                    if release_at_end
                    else "terminal reservation state is counterfactual-unknown"
                ),
            )
        )

    result = ReplayResult(
        experiment_id=run.experiment_id,
        task_id=run.task_id,
        policy=policy,
        target_budget=target_budget,
        historical_route=">".join(call.stage.value for call in run.calls),
        historical_call_count=len(run.calls),
        processed_call_count=processed,
        admitted_call_count=admitted_count,
        historical_output_compatible_calls=compatible_count,
        known_consumed_tokens=consumed,
        allocated_output_tokens=allocated_total,
        returned_capacity=returned_total,
        released_capacity=released_total,
        protected_capacity_peak=protected_peak,
        material_action=any(event.material_action for event in events),
        material_action_count=sum(event.material_action for event in events),
        disposition=disposition,
        terminal_reason=terminal_reason,
        critic_historically_present=any(
            call.stage == WorkflowStage.CRITIC for call in run.calls
        ),
        critic_admitted=critic_admitted,
        finish_reason_available=False,
        events=events,
        reservations=reservations,
    )
    assert_replay_invariants(result)
    return result


def assert_replay_invariants(result: ReplayResult) -> None:
    decision_events = [event for event in result.events if event.event_type == "allocation_decision"]
    previous_consumed = 0
    for event in result.events:
        if not 0 <= event.cumulative_known_consumed <= result.target_budget:
            raise AssertionError("known token consumption exceeds global budget")
        if event.remaining_after != result.target_budget - event.cumulative_known_consumed:
            raise AssertionError("remaining-budget conservation failed")
        if event.cumulative_known_consumed < previous_consumed:
            raise AssertionError("known consumption is not monotone")
        if min(
            event.allocated_output,
            event.protected_before,
            event.protected_after,
            event.returned_capacity,
            event.released_capacity,
            event.known_consumed,
            event.reservation_created_amount,
            event.reservation_resized_amount,
            event.reservation_claimed_amount,
            event.reservation_stranded_amount,
            event.reservation_shortfall,
            event.reallocation_available,
        ) < 0:
            raise AssertionError("negative ledger quantity")
        if event.event_type == "allocation_decision" and event.admitted:
            prompt = int(event.prompt_tokens or 0)
            if prompt + event.allocated_output + event.protected_after > event.remaining_before:
                raise AssertionError("allocation plus protection exceeds available budget")
        previous_consumed = event.cumulative_known_consumed
    if len(decision_events) != result.processed_call_count:
        raise AssertionError("processed call count disagrees with decision ledger")
    if any(record.lifecycle == ReservationLifecycle.CREATED for record in result.reservations):
        raise AssertionError("unresolved reservation at replay termination")
    ids = [record.reservation_id for record in result.reservations]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate reservation identifier")


def replay_canonical_cohort(
    runs: tuple[HistoricalRun, ...],
    config: Phase0Config,
    prompt_fallbacks: dict[WorkflowStage, int],
) -> list[ReplayResult]:
    return [
        replay_run(run, policy, budget, config, prompt_fallbacks)
        for run in runs
        for budget in config.target_budgets
        for policy in config.policies
    ]
