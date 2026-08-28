from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from workflow_control.routes import (
    RouteNode,
    maximal_remaining_path,
    optional_stages,
    topological_depths,
)
from workflow_control.types import Policy, PromptEstimate, StageSpec


@dataclass
class Reservation:
    reservation_id: str
    stage_id: str
    amount: int
    predicted_prompt_tokens: int
    status: str = "CREATE"


@dataclass(frozen=True)
class Allocation:
    call_id: str
    stage_id: str
    admitted: bool
    budget_before: int
    exact_current_prompt: int
    predicted_future_prompts: dict[str, int]
    current_min: int
    current_soft: int
    current_hard: int
    protected_continuation: int
    selected_max_output: int
    uncommitted_capacity: int
    structural_shortfall: bool
    shortfall_cause: str | None
    required_minimum: int
    reallocation_sources: tuple[str, ...] = ()


@dataclass
class ControllerState:
    schema_version: str
    task_id: str
    model_id: str
    policy: Policy
    b0: int
    consumed: int = 0
    completed_stages: set[str] = field(default_factory=set)
    unreachable_stages: set[str] = field(default_factory=set)
    reservations: dict[str, Reservation] = field(default_factory=dict)
    debited_call_ids: set[str] = field(default_factory=set)
    prepared: dict[str, Allocation] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    released_sources: list[str] = field(default_factory=list)
    shortfall_count: int = 0
    budget_violation_count: int = 0


class BudgetController:
    """Exact-debit controller with virtual reservations and append-only events."""

    SCHEMA_VERSION = "1.0.0"

    def __init__(
        self,
        *,
        task_id: str,
        model_id: str,
        policy: Policy,
        b0: int,
        route: RouteNode,
        stage_specs: Mapping[str, StageSpec],
        prompt_estimates: Mapping[str, PromptEstimate],
        fixed_reservation_fraction: float = 0.25,
    ) -> None:
        if b0 < 0:
            raise ValueError("B0 must be non-negative")
        self.route = route
        self.stage_specs = dict(stage_specs)
        self.prompt_estimates = dict(prompt_estimates)
        self.fixed_reservation_fraction = fixed_reservation_fraction
        self._topological_depths = topological_depths(route)
        self._optional_stages = optional_stages(route)
        self.state = ControllerState(
            schema_version=self.SCHEMA_VERSION,
            task_id=task_id,
            model_id=model_id,
            policy=policy,
            b0=b0,
        )

    @property
    def remaining(self) -> int:
        return self.state.b0 - self.state.consumed

    def _emit(self, event_type: str, **payload: Any) -> None:
        self.state.events.append(
            {
                "schema_version": self.SCHEMA_VERSION,
                "event_index": len(self.state.events),
                "event_type": event_type,
                "task_id": self.state.task_id,
                "model_id": self.state.model_id,
                "policy": self.state.policy.value,
                **payload,
            }
        )

    def _minimum_requirements(self) -> dict[str, int]:
        return {
            stage: estimate.effective_tokens + self.stage_specs[stage].minimum_output
            for stage, estimate in self.prompt_estimates.items()
        }

    def _create_or_resize_reservation(self, stage_id: str, amount: int) -> None:
        reservation_id = f"minimum:{stage_id}"
        estimate = self.prompt_estimates[stage_id]
        existing = self.state.reservations.get(reservation_id)
        if existing is None:
            reservation = Reservation(
                reservation_id=reservation_id,
                stage_id=stage_id,
                amount=amount,
                predicted_prompt_tokens=estimate.predicted_prompt_tokens,
            )
            self.state.reservations[reservation_id] = reservation
            self._emit(
                "RESERVATION_CREATE",
                reservation_action="CREATE",
                reservation_id=reservation_id,
                stage_id=stage_id,
                amount=amount,
                prompt_provenance=estimate.provenance,
            )
        elif existing.status == "CREATE" and existing.amount != amount:
            before = existing.amount
            existing.amount = amount
            self._emit(
                "RESERVATION_RESIZE",
                reservation_action="RESIZE",
                reservation_id=reservation_id,
                stage_id=stage_id,
                amount_before=before,
                amount_after=amount,
            )


    def prepare_call(self, call_id: str, stage_id: str, exact_prompt_tokens: int) -> Allocation:
        if call_id in self.state.debited_call_ids:
            raise ValueError(f"call {call_id} was already debited")
        if call_id in self.state.prepared:
            return self.state.prepared[call_id]
        if stage_id not in self.stage_specs:
            raise KeyError(stage_id)
        if exact_prompt_tokens < 0:
            raise ValueError("exact prompt tokens must be non-negative")

        prediction = self.prompt_estimates[stage_id]
        self.prompt_estimates[stage_id] = PromptEstimate(
            stage_id=stage_id,
            predicted_prompt_tokens=prediction.predicted_prompt_tokens,
            exact_prompt_tokens=exact_prompt_tokens,
            provenance=prediction.provenance,
            rendered_prompt_sha256=prediction.rendered_prompt_sha256,
        )
        reservation_id = f"minimum:{stage_id}"
        current_reservation = self.state.reservations.get(reservation_id)
        if current_reservation is not None and current_reservation.status == "CREATE":
            current_reservation.status = "CLAIM"
            self._emit(
                "RESERVATION_CLAIM",
                reservation_action="CLAIM",
                reservation_id=reservation_id,
                stage_id=stage_id,
                amount=current_reservation.amount,
            )

        spec = self.stage_specs[stage_id]
        budget_before = self.remaining
        requirements = self._minimum_requirements()
        excluded = self.state.completed_stages | self.state.unreachable_stages | {stage_id}
        continuation_path = maximal_remaining_path(self.route, requirements, excluded)
        predicted_future = {
            future: self.prompt_estimates[future].effective_tokens
            for future in sorted(continuation_path)
        }

        current_required = exact_prompt_tokens + spec.minimum_output
        if current_required > budget_before:
            allocation = Allocation(
                call_id=call_id,
                stage_id=stage_id,
                admitted=False,
                budget_before=budget_before,
                exact_current_prompt=exact_prompt_tokens,
                predicted_future_prompts=predicted_future,
                current_min=spec.minimum_output,
                current_soft=spec.soft_output,
                current_hard=spec.hard_output,
                protected_continuation=0,
                selected_max_output=0,
                uncommitted_capacity=max(0, budget_before - exact_prompt_tokens),
                structural_shortfall=True,
                shortfall_cause="current_minimum_infeasible",
                required_minimum=current_required,
                reallocation_sources=tuple(self.state.released_sources),
            )
            self.state.shortfall_count += 1
            self._emit(
                "STRUCTURAL_SHORTFALL",
                stage_id=stage_id,
                call_id=call_id,
                predicted_prompt_tokens=prediction.predicted_prompt_tokens,
                exact_prompt_tokens=exact_prompt_tokens,
                prediction_error_tokens=exact_prompt_tokens
                - prediction.predicted_prompt_tokens,
                remaining_budget=budget_before,
                required_minimum=current_required,
                cause=allocation.shortfall_cause,
            )
            self.state.prepared[call_id] = allocation
            return allocation

        protected = 0
        desired_protected = 0
        reservation_shortfall = False

        if self.state.policy == Policy.PROPOSED:
            desired_protected = sum(requirements[future] for future in continuation_path)
            reserve_capacity = budget_before - current_required
            reservation_shortfall = desired_protected > reserve_capacity

            remaining_paths = tuple(path - excluded for path in self.route.paths())
            if remaining_paths:
                unavoidable = set(remaining_paths[0])
                for path in remaining_paths[1:]:
                    unavoidable.intersection_update(path)
            else:
                unavoidable = set()

            terminal_stages = {"critic", "terminal_verifier", "verifier", "verify"}
            priority = sorted(
                continuation_path,
                key=lambda future: (
                    (
                        0
                        if future in unavoidable
                        else 1
                        if future in terminal_stages
                        else 2
                    ),
                    self._topological_depths.get(
                        future, len(self._topological_depths)
                    ),
                    1 if future in self._optional_stages else 0,
                    future,
                ),
            )

            reserve_left = reserve_capacity
            for future in priority:
                desired = requirements[future]
                funded = desired if not reservation_shortfall else min(desired, reserve_left)
                existing = self.state.reservations.get(f"minimum:{future}")
                if funded > 0 or (
                    existing is not None and existing.status == "CREATE"
                ):
                    self._create_or_resize_reservation(future, funded)
                protected += funded
                reserve_left -= funded

            if reservation_shortfall:
                self.state.shortfall_count += 1
                self._emit(
                    "RESERVATION_SHORTFALL",
                    stage_id=stage_id,
                    call_id=call_id,
                    cause="reservation_shortfall_best_effort",
                    full_branch_protection=False,
                    required_reserve=desired_protected,
                    funded_reserve=protected,
                    reservation_shortfall_tokens=desired_protected - protected,
                    remaining_budget=budget_before,
                    exact_prompt_tokens=exact_prompt_tokens,
                    current_minimum=spec.minimum_output,
                 )
        elif self.state.policy == Policy.FIXED_RESERVATION and continuation_path:
            has_terminal = any(
                stage in continuation_path
                for stage in ("critic", "terminal_verifier", "verifier", "verify")
            )
            if has_terminal:
                target = int(self.state.b0 * self.fixed_reservation_fraction)
                protected = min(
                    target,
                    max(0, budget_before - exact_prompt_tokens - spec.minimum_output),
                )

        full_required = current_required + (
            desired_protected if self.state.policy == Policy.PROPOSED else protected
        )
        available_output = budget_before - exact_prompt_tokens - protected

        if reservation_shortfall:
            selected = spec.minimum_output
            shortfall_cause = "reservation_shortfall_best_effort"
        else:
            if self.state.policy == Policy.LEGACY_STATIC:
                target = spec.legacy_output
            elif self.state.policy in {Policy.GREEDY, Policy.FIXED_RESERVATION}:
                target = spec.hard_output
            else:
                target = spec.soft_output
            selected = min(target, available_output)
            shortfall_cause = None

        uncommitted = max(
            0, budget_before - exact_prompt_tokens - protected - selected
        )
        allocation = Allocation(
            call_id=call_id,
            stage_id=stage_id,
            admitted=True,
            budget_before=budget_before,
            exact_current_prompt=exact_prompt_tokens,
            predicted_future_prompts=predicted_future,
            current_min=spec.minimum_output,
            current_soft=spec.soft_output,
            current_hard=spec.hard_output,
            protected_continuation=protected,
            selected_max_output=selected,
            uncommitted_capacity=uncommitted,
            structural_shortfall=False,
            shortfall_cause=shortfall_cause,
            required_minimum=full_required,
            reallocation_sources=tuple(self.state.released_sources),
        )
        allocation_event = asdict(allocation)
        allocation_event["full_branch_protection"] = (
            not reservation_shortfall
            if self.state.policy == Policy.PROPOSED
            else None
        )
        self._emit(
            "ALLOCATION",
            call_id=call_id,
            stage_id=stage_id,
            allocation=allocation_event,
        )
        if self.state.released_sources:
            self._emit(
                "CAPACITY_REALLOCATE",
                capacity_action="REALLOCATE",
                source_ids=list(self.state.released_sources),
                destination_stage=stage_id,
            )
            self.state.released_sources.clear()
        self.state.prepared[call_id] = allocation
        return allocation
    def finish_call(
        self,
        call_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        provider: Mapping[str, Any] | None = None,
    ) -> None:
        if call_id in self.state.debited_call_ids:
            raise ValueError(f"duplicate provider debit for {call_id}")
        allocation = self.state.prepared.get(call_id)
        if allocation is None or not allocation.admitted:
            raise ValueError(f"call {call_id} has no admitted allocation")
        if input_tokens != allocation.exact_current_prompt:
            raise ValueError("provider input count differs from exact preflight count")
        if output_tokens > allocation.selected_max_output:
            raise ValueError("provider output exceeds selected maximum")
        realized = input_tokens + output_tokens
        if realized > self.remaining:
            self.state.budget_violation_count += 1
            raise ValueError("provider debit would violate B0")
        self.state.consumed += realized
        self.state.debited_call_ids.add(call_id)
        self.state.completed_stages.add(allocation.stage_id)
        returned = allocation.selected_max_output - output_tokens
        source_id = f"allowance:{call_id}"
        self.state.released_sources.append(source_id)
        self._emit(
            "PROVIDER_DEBIT",
            call_id=call_id,
            stage_id=allocation.stage_id,
            provider={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": realized,
                **dict(provider or {}),
            },
            total_consumed=self.state.consumed,
            b0=self.state.b0,
        )
        self._emit(
            "CAPACITY_RETURN",
            capacity_action="RETURN",
            source_id=source_id,
            amount=returned,
            destination_stage=None,
        )

    def materialize_future_prompt(self, stage_id: str, exact_tokens: int) -> int:
        previous = self.prompt_estimates[stage_id]
        error = exact_tokens - previous.predicted_prompt_tokens
        self.prompt_estimates[stage_id] = PromptEstimate(
            stage_id=stage_id,
            predicted_prompt_tokens=previous.predicted_prompt_tokens,
            exact_prompt_tokens=exact_tokens,
            provenance=previous.provenance,
            rendered_prompt_sha256=previous.rendered_prompt_sha256,
        )
        reservation = self.state.reservations.get(f"minimum:{stage_id}")
        if reservation is not None and reservation.status == "CREATE":
            before = reservation.amount
            reservation.amount += error
            self._emit(
                "RESERVATION_RESIZE",
                reservation_action="RESIZE",
                reservation_id=reservation.reservation_id,
                stage_id=stage_id,
                amount_before=before,
                amount_after=reservation.amount,
                predicted_prompt_tokens=previous.predicted_prompt_tokens,
                exact_prompt_tokens=exact_tokens,
                prediction_error_tokens=error,
                prompt_provenance=previous.provenance,
            )
        return error

    def release_unreachable(self, reachable_stages: set[str]) -> int:
        released = 0
        self.state.unreachable_stages.update(set(self.stage_specs) - reachable_stages)
        for reservation in self.state.reservations.values():
            if reservation.status == "CREATE" and reservation.stage_id not in reachable_stages:
                reservation.status = "RELEASE"
                released += reservation.amount
                self.state.released_sources.append(reservation.reservation_id)
                self._emit(
                    "RESERVATION_RELEASE",
                    reservation_action="RELEASE",
                    reservation_id=reservation.reservation_id,
                    stage_id=reservation.stage_id,
                    amount=reservation.amount,
                )
        return released

    def finalize(self, *, normal_completion: bool = True) -> dict[str, Any]:
        active = [
            reservation
            for reservation in self.state.reservations.values()
            if reservation.status == "CREATE"
        ]
        for reservation in active:
            if normal_completion:
                reservation.status = "RELEASE"
                self._emit(
                    "RESERVATION_RELEASE",
                    reservation_action="RELEASE",
                    reservation_id=reservation.reservation_id,
                    stage_id=reservation.stage_id,
                    amount=reservation.amount,
                )
            else:
                reservation.status = "STRAND"
                self._emit(
                    "RESERVATION_STRAND",
                    reservation_action="STRAND",
                    reservation_id=reservation.reservation_id,
                    stage_id=reservation.stage_id,
                    amount=reservation.amount,
                )
        unresolved = [
            reservation.reservation_id
            for reservation in self.state.reservations.values()
            if reservation.status == "CREATE"
        ]
        summary = {
            "normal_completion": normal_completion,
            "total_consumed": self.state.consumed,
            "B0": self.state.b0,
            "unused_capacity": self.remaining,
            "unresolved_reservations": unresolved,
            "budget_violation_count": self.state.budget_violation_count,
            "shortfall_count": self.state.shortfall_count,
        }
        self._emit("RUN_END", run_end=summary)
        return summary

    def dumps(self) -> str:
        data = asdict(self.state)
        data["policy"] = self.state.policy.value
        data["completed_stages"] = sorted(self.state.completed_stages)
        data["unreachable_stages"] = sorted(self.state.unreachable_stages)
        data["debited_call_ids"] = sorted(self.state.debited_call_ids)
        return json.dumps(data, sort_keys=True)

    def restore(self, payload: str) -> None:
        data = json.loads(payload)
        if data["task_id"] != self.state.task_id or data["model_id"] != self.state.model_id:
            raise ValueError("resume identity mismatch")
        if data["policy"] != self.state.policy.value or data["b0"] != self.state.b0:
            raise ValueError("resume configuration mismatch")
        data["policy"] = Policy(data["policy"])
        data["completed_stages"] = set(data["completed_stages"])
        data["unreachable_stages"] = set(data.get("unreachable_stages", []))
        data["debited_call_ids"] = set(data["debited_call_ids"])
        data["reservations"] = {
            key: Reservation(**value) for key, value in data["reservations"].items()
        }
        data["prepared"] = {
            key: Allocation(**value) for key, value in data["prepared"].items()
        }
        self.state = ControllerState(**data)
