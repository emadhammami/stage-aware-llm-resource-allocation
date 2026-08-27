from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from analysis.stage_aware.types import ReplayResult


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write empty table: {path}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def _mean(values: list[int | float]) -> float:
    return statistics.fmean(values) if values else 0.0


def policy_summary(results: list[ReplayResult]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[ReplayResult]] = defaultdict(list)
    for result in results:
        groups[(result.policy.value, result.target_budget)].append(result)
    rows: list[dict[str, Any]] = []
    for (policy, budget), group in sorted(groups.items()):
        critic_eligible = [result for result in group if result.critic_historically_present]
        rows.append(
            {
                "policy": policy,
                "target_budget": budget,
                "runs": len(group),
                "historical_cap_compatible_fraction": sum(
                    result.disposition == "historical_cap_compatible" for result in group
                )
                / len(group),
                "admission_denied_fraction": sum(
                    result.disposition == "admission_denied" for result in group
                )
                / len(group),
                "cap_incompatible_fraction": sum(
                    result.disposition == "cap_incompatible_counterfactual_unknown"
                    for result in group
                )
                / len(group),
                "material_action_fraction": sum(result.material_action for result in group)
                / len(group),
                "mean_material_action_count": _mean(
                    [result.material_action_count for result in group]
                ),
                "reservation_run_fraction": sum(bool(result.reservations) for result in group)
                / len(group),
                "mean_known_consumed_tokens": _mean(
                    [result.known_consumed_tokens for result in group]
                ),
                "mean_allocated_output_tokens": _mean(
                    [result.allocated_output_tokens for result in group]
                ),
                "mean_returned_capacity": _mean(
                    [result.returned_capacity for result in group]
                ),
                "mean_released_capacity": _mean(
                    [result.released_capacity for result in group]
                ),
                "mean_peak_protected_capacity": _mean(
                    [result.protected_capacity_peak for result in group]
                ),
                "critic_historical_runs": len(critic_eligible),
                "critic_admission_fraction": (
                    sum(result.critic_admitted for result in critic_eligible)
                    / len(critic_eligible)
                    if critic_eligible
                    else None
                ),
                "finish_reason_available_fraction": sum(
                    result.finish_reason_available for result in group
                )
                / len(group),
            }
        )
    return rows


def stage_summary(results: list[ReplayResult]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list] = defaultdict(list)
    for result in results:
        for event in result.events:
            if event.event_type == "allocation_decision" and event.stage is not None:
                groups[(result.policy.value, result.target_budget, event.stage.value)].append(event)
    rows: list[dict[str, Any]] = []
    for (policy, budget, stage), events in sorted(groups.items()):
        comparable = [
            event
            for event in events
            if event.historical_output_compatible is not None
        ]
        rows.append(
            {
                "policy": policy,
                "target_budget": budget,
                "stage": stage,
                "decision_events": len(events),
                "admission_fraction": sum(event.admitted is True for event in events) / len(events),
                "historical_output_compatible_fraction": (
                    sum(event.historical_output_compatible is True for event in comparable)
                    / len(comparable)
                    if comparable
                    else None
                ),
                "material_action_fraction": sum(event.material_action for event in events)
                / len(events),
                "mean_allocated_output": _mean([event.allocated_output for event in events]),
                "mean_protected_before": _mean([event.protected_before for event in events]),
                "mean_protected_after": _mean([event.protected_after for event in events]),
                "mean_returned_capacity": _mean([event.returned_capacity for event in events]),
                "mean_known_consumed": _mean([event.known_consumed for event in events]),
            }
        )
    return rows


def disposition_summary(results: list[ReplayResult]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[ReplayResult]] = defaultdict(list)
    for result in results:
        groups[(result.policy.value, result.target_budget)].append(result)
    rows: list[dict[str, Any]] = []
    for (policy, budget), group in sorted(groups.items()):
        counts = Counter(result.disposition for result in group)
        for disposition, count in sorted(counts.items()):
            rows.append(
                {
                    "policy": policy,
                    "target_budget": budget,
                    "disposition": disposition,
                    "runs": count,
                    "fraction": count / len(group),
                }
            )
    return rows


def denial_reason_summary(results: list[ReplayResult]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, int, str, str]] = Counter()
    for result in results:
        for event in result.events:
            if event.denial_reason:
                counter[
                    (
                        result.policy.value,
                        result.target_budget,
                        event.stage.value if event.stage else "none",
                        event.denial_reason.value,
                    )
                ] += 1
    if not counter:
        return [
            {
                "policy": "none",
                "target_budget": 0,
                "stage": "none",
                "denial_reason": "none",
                "events": 0,
            }
        ]
    return [
        {
            "policy": policy,
            "target_budget": budget,
            "stage": stage,
            "denial_reason": reason,
            "events": count,
        }
        for (policy, budget, stage, reason), count in sorted(counter.items())
    ]


def reservation_rows(results: list[ReplayResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        for record in result.reservations:
            row = {
                "experiment_id": result.experiment_id,
                "task_id": result.task_id,
                "policy": result.policy.value,
                "target_budget": result.target_budget,
                **record.to_dict(),
            }
            rows.append(row)
    if rows:
        return rows
    return [
        {
            "experiment_id": "none",
            "task_id": "none",
            "policy": "none",
            "target_budget": 0,
            "reservation_id": "none",
            "stage": "none",
            "amount": 0,
            "created_event": 0,
            "prompt_provenance": "missing",
            "conditional": False,
            "lifecycle": "released",
            "resolved_event": 0,
            "resolution_note": "no reservations",
        }
    ]


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    destination = Path(path)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
