from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from analysis.stage_aware.types import HistoricalRun, ReplayResult, WorkflowStage

TARGET_STAGES = (WorkflowStage.EXECUTOR_2, WorkflowStage.CRITIC)


def starvation_rows(
    runs: tuple[HistoricalRun, ...],
    results: list[ReplayResult],
    feasibility_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    run_by_id = {run.experiment_id: run for run in runs}
    floor_by_id = {
        row["experiment_id"]: int(row["historical_route_floor"])
        for row in feasibility_rows
        if row["historical_route_floor"] is not None
    }
    rows: list[dict[str, Any]] = []
    for result in results:
        run = run_by_id[result.experiment_id]
        event_by_stage = {
            event.stage: event
            for event in result.events
            if event.event_type == "allocation_decision" and event.stage
        }
        for stage in TARGET_STAGES:
            eligible = any(call.stage == stage for call in run.calls)
            route_floor = floor_by_id[result.experiment_id]
            structurally_feasible = route_floor <= result.target_budget
            event = event_by_stage.get(stage)
            if not eligible:
                status = "not_historically_eligible"
                reason = "not_reachable"
            elif not structurally_feasible:
                status = "structural_infeasible"
                reason = "global_minimum_infeasible"
            elif event is None:
                status = "counterfactual_unknown_before_stage"
                reason = "prior_cap_divergence"
            elif event.admitted is False:
                status = "eligible_policy_starvation"
                reason = event.denial_reason.value if event.denial_reason else "not_applicable"
            else:
                status = "eligible_admitted"
                reason = "not_applicable"
            rows.append(
                {
                    "experiment_id": result.experiment_id,
                    "task_id": result.task_id,
                    "policy": result.policy.value,
                    "target_budget": result.target_budget,
                    "historical_route": result.historical_route,
                    "stage": stage.value,
                    "historically_eligible": eligible,
                    "historical_route_floor": route_floor,
                    "structurally_feasible": structurally_feasible,
                    "status": status,
                    "primary_reason": reason,
                }
            )
    return rows


def starvation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["policy"], int(row["target_budget"]), row["stage"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (policy, budget, stage), group in sorted(groups.items()):
        counts = Counter(row["status"] for row in group)
        eligible = [row for row in group if row["historically_eligible"]]
        feasible = [row for row in eligible if row["structurally_feasible"]]
        known_feasible = [
            row
            for row in feasible
            if row["status"] != "counterfactual_unknown_before_stage"
        ]
        starved = counts["eligible_policy_starvation"]
        summaries.append(
            {
                "policy": policy,
                "target_budget": budget,
                "stage": stage,
                "runs": len(group),
                "historically_eligible_runs": len(eligible),
                "structurally_infeasible_runs": counts["structural_infeasible"],
                "structurally_feasible_eligible_runs": len(feasible),
                "counterfactual_unknown_before_stage": counts[
                    "counterfactual_unknown_before_stage"
                ],
                "known_structurally_feasible_eligible_runs": len(known_feasible),
                "eligible_policy_starvation_runs": starved,
                "eligible_policy_starvation_fraction": (
                    starved / len(known_feasible) if known_feasible else None
                ),
                "eligible_admitted_runs": counts["eligible_admitted"],
            }
        )
    return summaries
