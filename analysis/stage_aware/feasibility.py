from __future__ import annotations

import math
import statistics
from typing import Any

from analysis.stage_aware.config import Phase0Config
from analysis.stage_aware.types import HistoricalRun, Provenance, WorkflowStage


def _prompt_for_stage(
    run: HistoricalRun,
    stage: WorkflowStage,
    fallbacks: dict[WorkflowStage, int],
) -> tuple[int | None, Provenance]:
    call = next((call for call in run.calls if call.stage == stage), None)
    if call and call.prompt_tokens is not None:
        return call.prompt_tokens, call.prompt_provenance
    if stage in fallbacks:
        return fallbacks[stage], Provenance.DETERMINISTIC_ESTIMATE
    return None, Provenance.MISSING


def structural_feasibility_rows(
    runs: tuple[HistoricalRun, ...],
    config: Phase0Config,
    prompt_fallbacks: dict[WorkflowStage, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        observed_prompts = [call.prompt_tokens for call in run.calls]
        historical_floor = None
        historical_provenance = Provenance.MISSING
        if all(value is not None for value in observed_prompts):
            historical_floor = sum(int(value or 0) for value in observed_prompts) + sum(
                config.stage_specs[call.stage].minimum_output for call in run.calls
            )
            historical_provenance = (
                Provenance.OBSERVED_EXACT
                if all(call.prompt_provenance == Provenance.OBSERVED_EXACT for call in run.calls)
                else Provenance.DETERMINISTIC_ESTIMATE
            )

        direct_stages = (
            WorkflowStage.PLANNER,
            WorkflowStage.EXECUTOR_1,
            WorkflowStage.CRITIC,
        )
        direct_values = [_prompt_for_stage(run, stage, {}) for stage in direct_stages]
        direct_floor = None
        direct_provenance = Provenance.MISSING
        if all(value is not None for value, _ in direct_values):
            direct_floor = sum(int(value or 0) for value, _ in direct_values) + sum(
                config.stage_specs[stage].minimum_output for stage in direct_stages
            )
            direct_provenance = (
                Provenance.OBSERVED_EXACT
                if all(provenance == Provenance.OBSERVED_EXACT for _, provenance in direct_values)
                else Provenance.DETERMINISTIC_ESTIMATE
            )

        full_stages = (
            WorkflowStage.PLANNER,
            WorkflowStage.EXECUTOR_1,
            WorkflowStage.EXECUTOR_2,
            WorkflowStage.CRITIC,
        )
        full_values = [
            _prompt_for_stage(run, stage, prompt_fallbacks) for stage in full_stages
        ]
        full_floor = None
        full_provenance = Provenance.MISSING
        if all(value is not None for value, _ in full_values):
            full_floor = sum(int(value or 0) for value, _ in full_values) + sum(
                config.stage_specs[stage].minimum_output for stage in full_stages
            )
            full_provenance = (
                Provenance.OBSERVED_EXACT
                if all(provenance == Provenance.OBSERVED_EXACT for _, provenance in full_values)
                else Provenance.DETERMINISTIC_ESTIMATE
            )

        row: dict[str, Any] = {
            "experiment_id": run.experiment_id,
            "task_id": run.task_id,
            "observed_route": ">".join(call.stage.value for call in run.calls),
            "historical_route_floor": historical_floor,
            "historical_route_floor_provenance": historical_provenance.value,
            "direct_completion_floor": direct_floor,
            "direct_completion_floor_provenance": direct_provenance.value,
            "full_retry_success_floor": full_floor,
            "full_retry_success_floor_provenance": full_provenance.value,
        }
        for budget in config.target_budgets:
            for metric in (
                "historical_route_floor",
                "direct_completion_floor",
                "full_retry_success_floor",
            ):
                value = row[metric]
                row[f"{metric}_feasible_at_{budget}"] = (
                    value is not None and int(value) <= budget
                )
        rows.append(row)
    return rows


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def structural_feasibility_summary(
    rows: list[dict[str, Any]], config: Phase0Config
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    metrics = (
        ("historical_route_floor", "historical observed route"),
        ("direct_completion_floor", "Planner-Executor1-Critic exact when observed"),
        ("full_retry_success_floor", "full retry-success topology; estimated where absent"),
    )
    for metric, label in metrics:
        available = [float(row[metric]) for row in rows if row[metric] is not None]
        for budget in config.target_budgets:
            feasible = sum(value <= budget for value in available)
            summary.append(
                {
                    "metric": metric,
                    "label": label,
                    "target_budget": budget,
                    "available_runs": len(available),
                    "feasible_runs": feasible,
                    "feasible_fraction": feasible / len(available) if available else None,
                    "minimum_floor": min(available) if available else None,
                    "median_floor": statistics.median(available) if available else None,
                    "p95_floor": _percentile(available, 0.95) if available else None,
                    "maximum_floor": max(available) if available else None,
                }
            )
    return summary


def structural_feasibility_by_budget(
    runs: tuple[HistoricalRun, ...],
    config: Phase0Config,
    prompt_fallbacks: dict[WorkflowStage, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        paths: list[tuple[str, list[tuple[WorkflowStage, int, Provenance]]]] = []
        historical = [
            (call.stage, call.prompt_tokens, call.prompt_provenance)
            for call in run.calls
            if call.prompt_tokens is not None
        ]
        if len(historical) == len(run.calls):
            paths.append(("historical_route_floor", historical))
        direct_stages = [
            WorkflowStage.PLANNER,
            WorkflowStage.EXECUTOR_1,
            WorkflowStage.CRITIC,
        ]
        direct = [
            (stage, *_prompt_for_stage(run, stage, {})) for stage in direct_stages
        ]
        if all(prompt is not None for _, prompt, _ in direct):
            direct_complete = [
                (stage, int(prompt), provenance)
                for stage, prompt, provenance in direct
                if prompt is not None
            ]
            paths.append(("direct_completion_floor", direct_complete))
        retry_stages = [
            WorkflowStage.PLANNER,
            WorkflowStage.EXECUTOR_1,
            WorkflowStage.EXECUTOR_2,
            WorkflowStage.CRITIC,
        ]
        retry = [
            (stage, *_prompt_for_stage(run, stage, prompt_fallbacks))
            for stage in retry_stages
        ]
        if all(prompt is not None for _, prompt, _ in retry):
            retry_complete = [
                (stage, int(prompt), provenance)
                for stage, prompt, provenance in retry
                if prompt is not None
            ]
            paths.append(("full_retry_success_floor", retry_complete))
        for path_name, path in paths:
            floor = sum(
                int(prompt or 0) + config.stage_specs[stage].minimum_output
                for stage, prompt, _ in path
            )
            provenance = (
                Provenance.OBSERVED_EXACT
                if all(item_provenance == Provenance.OBSERVED_EXACT for _, _, item_provenance in path)
                else Provenance.DETERMINISTIC_ESTIMATE
            )
            for budget in config.target_budgets:
                cumulative = 0
                failure_stage = None
                for stage, prompt, _ in path:
                    cumulative += int(prompt or 0) + config.stage_specs[stage].minimum_output
                    if cumulative > budget and failure_stage is None:
                        failure_stage = stage
                rows.append(
                    {
                        "experiment_id": run.experiment_id,
                        "task_id": run.task_id,
                        "path_metric": path_name,
                        "path": ">".join(stage.value for stage, _, _ in path),
                        "provenance": provenance.value,
                        "target_budget": budget,
                        "minimum_path_cost": floor,
                        "structurally_feasible": floor <= budget,
                        "infeasibility_reason": (
                            "not_applicable" if floor <= budget else "global_minimum_infeasible"
                        ),
                        "minimum_floor_failure_stage": (
                            failure_stage.value if failure_stage else "not_applicable"
                        ),
                    }
                )
    return rows
