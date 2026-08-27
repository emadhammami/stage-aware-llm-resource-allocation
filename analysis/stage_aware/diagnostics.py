from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from analysis.stage_aware.artifact import ArtifactBundle
from analysis.stage_aware.config import Phase0Config
from analysis.stage_aware.types import ReplayResult, WorkflowStage


def stage_call_cost_rows(bundle: ArtifactBundle) -> list[dict[str, Any]]:
    canonical_ids = {run.experiment_id for run in bundle.canonical_runs}
    rows: list[dict[str, Any]] = []
    for run in bundle.runs:
        for attempt, call in enumerate(run.calls, start=1):
            total = int(call.total_tokens or 0)
            prompt = int(call.prompt_tokens or 0)
            output = int(call.output_tokens or 0)
            cap = call.original_output_cap
            rows.append(
                {
                    "experiment_id": run.experiment_id,
                    "task_id": run.task_id,
                    "method": run.method,
                    "source_budget": run.source_budget,
                    "canonical_replay_cohort": run.experiment_id in canonical_ids,
                    "stage": call.stage.value,
                    "stage_attempt": 2 if call.stage == WorkflowStage.EXECUTOR_2 else 1,
                    "sequence_index": attempt - 1,
                    "historical_admitted": call.admitted,
                    "prompt_tokens": call.prompt_tokens,
                    "prompt_provenance": call.prompt_provenance.value,
                    "output_tokens": call.output_tokens,
                    "usage_provenance": call.usage_provenance.value,
                    "total_tokens": call.total_tokens,
                    "prompt_share": prompt / total if total else None,
                    "output_share": output / total if total else None,
                    "configured_output_cap": cap,
                    "output_cap_ratio": output / cap if cap else None,
                    "finish_reason_available": False,
                }
            )
    return rows


def stage_cost_summary(call_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for cohort, selector in (
        ("publication_all", lambda row: True),
        ("canonical_evidence_gated_4000", lambda row: row["canonical_replay_cohort"]),
    ):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in call_rows:
            if selector(row):
                if row["historical_admitted"]:
                    groups[row["stage"]].append(row)
        for stage, rows in sorted(groups.items()):
            prompts = [int(row["prompt_tokens"]) for row in rows]
            outputs = [int(row["output_tokens"]) for row in rows]
            totals = [int(row["total_tokens"]) for row in rows]
            summaries.append(
                {
                    "cohort": cohort,
                    "stage": stage,
                    "calls": len(rows),
                    "distinct_tasks": len({row["task_id"] for row in rows}),
                    "mean_prompt_tokens": statistics.fmean(prompts),
                    "median_prompt_tokens": statistics.median(prompts),
                    "minimum_prompt_tokens": min(prompts),
                    "maximum_prompt_tokens": max(prompts),
                    "mean_output_tokens": statistics.fmean(outputs),
                    "median_output_tokens": statistics.median(outputs),
                    "minimum_output_tokens": min(outputs),
                    "maximum_output_tokens": max(outputs),
                    "pooled_prompt_share": sum(prompts) / sum(totals),
                    "pooled_output_share": sum(outputs) / sum(totals),
                }
            )
    return summaries


def cap_binding_summary(
    call_rows: list[dict[str, Any]], config: Phase0Config
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for cohort, selector in (
        ("publication_all", lambda row: True),
        ("canonical_evidence_gated_4000", lambda row: row["canonical_replay_cohort"]),
    ):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in call_rows:
            if selector(row):
                if row["historical_admitted"] and row["output_cap_ratio"] is not None:
                    groups[row["stage"]].append(row)
        for stage, rows in sorted(groups.items()):
            ratios = [float(row["output_cap_ratio"]) for row in rows]
            summary: dict[str, Any] = {
                "cohort": cohort,
                "stage": stage,
                "calls": len(rows),
                "mean_output_cap_ratio": statistics.fmean(ratios),
                "median_output_cap_ratio": statistics.median(ratios),
                "maximum_output_cap_ratio": max(ratios),
                "finish_reason_present": 0,
                "finish_reason_missing": len(rows),
                "interpretation": "cap_binding_proxy_only",
            }
            for threshold in config.near_cap_thresholds:
                label = int(round(threshold * 100))
                count = sum(ratio >= 1.0 - threshold for ratio in ratios)
                summary[f"within_{label}pct_count"] = count
                summary[f"within_{label}pct_fraction"] = count / len(ratios)
            summaries.append(summary)
    return summaries


def policy_action_rate_rows(results: list[ReplayResult]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, str], list] = defaultdict(list)
    for result in results:
        for event in result.events:
            if event.event_type == "allocation_decision" and event.stage:
                groups[
                    (
                        result.policy.value,
                        result.target_budget,
                        event.stage.value,
                        result.historical_route,
                    )
                ].append(event)
    return [
        {
            "policy": policy,
            "target_budget": budget,
            "stage": stage,
            "historical_route": route,
            "decision_events": len(events),
            "material_action_events": sum(event.material_action for event in events),
            "material_action_fraction": sum(event.material_action for event in events)
            / len(events),
        }
        for (policy, budget, stage, route), events in sorted(groups.items())
    ]


def reservation_summary(results: list[ReplayResult]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[ReplayResult]] = defaultdict(list)
    for result in results:
        groups[(result.policy.value, result.target_budget)].append(result)
    rows: list[dict[str, Any]] = []
    for (policy, budget), group in sorted(groups.items()):
        records = [record for result in group for record in result.reservations]
        events = [event for result in group for event in result.events]
        lifecycle = Counter(record.lifecycle.value for record in records)
        prediction_errors = [
            event.reservation_prediction_error
            for event in events
            if event.reservation_prediction_error is not None
        ]
        returned = sum(event.returned_capacity for event in events)
        released = sum(event.released_capacity for event in events)
        rows.append(
            {
                "policy": policy,
                "target_budget": budget,
                "runs": len(group),
                "reservations_created": len(records),
                "reservations_resized": sum(record.resize_count for record in records),
                "reservations_claimed": lifecycle["claimed"],
                "reservations_released": lifecycle["released"],
                "reservations_stranded": lifecycle["stranded"],
                "reservation_shortfall_events": sum(
                    event.reservation_shortfall > 0 for event in events
                ),
                "reservation_shortfall_tokens": sum(
                    event.reservation_shortfall for event in events
                ),
                "mean_claim_prediction_error": (
                    statistics.fmean(prediction_errors) if prediction_errors else None
                ),
                "total_returned_capacity": returned,
                "total_released_capacity": released,
                "total_reallocation_opportunity": returned + released,
                "mean_reallocation_opportunity_per_run": (returned + released) / len(group),
            }
        )
    return rows


def provenance_summary(
    bundle: ArtifactBundle, results: list[ReplayResult]
) -> list[dict[str, Any]]:
    source_calls = [call for run in bundle.runs for call in run.calls]
    decision_events = [
        event
        for result in results
        for event in result.events
        if event.event_type == "allocation_decision"
    ]
    reservation_records = [record for result in results for record in result.reservations]
    counterfactual_unknown = sum(
        result.disposition == "cap_incompatible_counterfactual_unknown"
        for result in results
    )
    return [
        {
            "scope": "source_prompt_counts",
            "observed_exact": sum(
                call.prompt_provenance.value == "observed_exact" for call in source_calls
            ),
            "reconstructed_exact": 0,
            "deterministic_estimate": sum(
                call.prompt_provenance.value == "deterministic_estimate"
                for call in source_calls
            ),
            "missing": sum(call.prompt_tokens is None for call in source_calls),
            "counterfactual_unknown": 0,
        },
        {
            "scope": "source_usage_counts",
            "observed_exact": sum(
                call.usage_provenance.value == "observed_exact" for call in source_calls
            ),
            "reconstructed_exact": 0,
            "deterministic_estimate": sum(
                call.usage_provenance.value == "deterministic_estimate"
                for call in source_calls
            ),
            "missing": sum(call.total_tokens is None for call in source_calls),
            "counterfactual_unknown": 0,
        },
        {
            "scope": "replay_current_prompt_counts",
            "observed_exact": sum(
                event.prompt_provenance.value == "observed_exact" for event in decision_events
            ),
            "reconstructed_exact": 0,
            "deterministic_estimate": sum(
                event.prompt_provenance.value == "deterministic_estimate"
                for event in decision_events
            ),
            "missing": sum(event.prompt_tokens is None for event in decision_events),
            "counterfactual_unknown": 0,
        },
        {
            "scope": "replay_remaining_budget_values",
            "observed_exact": 0,
            "reconstructed_exact": len(decision_events),
            "deterministic_estimate": 0,
            "missing": 0,
            "counterfactual_unknown": 0,
        },
        {
            "scope": "reservation_future_prompt_requirements",
            "observed_exact": sum(
                record.prompt_provenance.value == "observed_exact"
                for record in reservation_records
            ),
            "reconstructed_exact": 0,
            "deterministic_estimate": sum(
                record.prompt_provenance.value == "deterministic_estimate"
                for record in reservation_records
            ),
            "missing": sum(
                record.prompt_provenance.value == "missing" for record in reservation_records
            ),
            "counterfactual_unknown": 0,
        },
        {
            "scope": "policy_replay_run_disposition",
            "observed_exact": 0,
            "reconstructed_exact": len(results) - counterfactual_unknown,
            "deterministic_estimate": 0,
            "missing": 0,
            "counterfactual_unknown": counterfactual_unknown,
        },
    ]


def go_no_go_summary(
    results: list[ReplayResult], feasibility_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    floor_by_id = {
        row["experiment_id"]: int(row["historical_route_floor"])
        for row in feasibility_rows
        if row["historical_route_floor"] is not None
    }
    groups: dict[tuple[str, int], list[ReplayResult]] = defaultdict(list)
    for result in results:
        if result.target_budget in {2000, 4000}:
            groups[(result.policy.value, result.target_budget)].append(result)
    rows: list[dict[str, Any]] = []
    for (policy, budget), group in sorted(groups.items()):
        feasible = [
            result for result in group if floor_by_id[result.experiment_id] <= budget
        ]
        actions = sum(result.material_action for result in feasible)
        rows.append(
            {
                "policy": policy,
                "target_budget": budget,
                "structurally_feasible_runs": len(feasible),
                "material_action_runs": actions,
                "material_action_fraction": actions / len(feasible) if feasible else None,
                "preregistered_threshold": 0.20,
                "threshold_passed": actions / len(feasible) >= 0.20 if feasible else False,
            }
        )
    return rows
