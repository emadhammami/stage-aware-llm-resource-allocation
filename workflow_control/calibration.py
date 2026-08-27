from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CalibrationRun:
    run_id: str
    actual_cost: int
    b_min: int
    b_soft: int


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def calibrate_rho(runs: Iterable[CalibrationRun]) -> dict[str, object]:
    raw: list[float] = []
    zero_denominator: list[str] = []
    for run in runs:
        denominator = run.b_soft - run.b_min
        if denominator == 0:
            zero_denominator.append(run.run_id)
            continue
        if denominator < 0:
            raise ValueError(f"B_soft < B_min for {run.run_id}")
        raw.append((run.actual_cost - run.b_min) / denominator)
    if not raw:
        raise ValueError("no historically replayable runs have a positive denominator")
    clamped = [max(0.0, value) for value in raw]
    statistics = {
        "count": len(raw),
        "zero_denominator_count": len(zero_denominator),
        "raw_min": min(raw),
        "raw_median": percentile(raw, 0.50),
        "raw_p75": percentile(raw, 0.75),
        "raw_p90": percentile(raw, 0.90),
        "raw_p95": percentile(raw, 0.95),
        "raw_max": max(raw),
        "negative_count": sum(value < 0 for value in raw),
        "clamped_p90": percentile(clamped, 0.90),
    }
    return {
        "statistics": statistics,
        "rho_star": statistics["clamped_p90"],
        "negative_handling": "clamp_to_zero_before_frozen_percentile",
        "high_value_handling": "unclamped",
        "zero_denominator_run_ids": zero_denominator,
    }
