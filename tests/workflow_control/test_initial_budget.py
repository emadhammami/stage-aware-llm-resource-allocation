from __future__ import annotations

import math

from workflow_control.calibration import CalibrationRun, calibrate_rho
from workflow_control.initial_budget import InitialBudgetEstimator
from workflow_control.routes import Exclusive, Sequence, Stage, maximal_requirement
from workflow_control.types import PromptEstimate, StageSpec


def _spec(stage: str, minimum: int = 5, soft: int = 10) -> StageSpec:
    return StageSpec(stage, minimum, soft, 20, 10)


def test_sequential_recursion_and_shared_stage_once() -> None:
    route = Sequence((Stage("a"), Stage("b"), Stage("a")))
    assert maximal_requirement(route, {"a": 11, "b": 7}) == 18


def test_exclusive_branch_uses_max_not_sum() -> None:
    route = Sequence((Stage("start"), Exclusive((Stage("left"), Stage("right"))), Stage("end")))
    requirements = {"start": 2, "left": 5, "right": 9, "end": 3}
    assert maximal_requirement(route, requirements) == 14


def test_estimator_is_deterministic_and_soft_is_not_below_minimum() -> None:
    route = Sequence((Stage("a"), Exclusive((Stage("b"), Stage("c"))), Stage("d")))
    specs = {stage: _spec(stage) for stage in "abcd"}
    prompts = {
        stage: PromptEstimate(stage, index + 1, "fixture")
        for index, stage in enumerate("abcd")
    }
    estimator = InitialBudgetEstimator(0.25)
    first = estimator.estimate(
        model_id="fixture-model",
        task_id="fixture-task",
        route=route,
        stage_specs=specs,
        prompt_estimates=prompts,
        tokenization_provenance={"kind": "fixture"},
        route_assumptions={},
    )
    second = estimator.estimate(
        model_id="fixture-model",
        task_id="fixture-task",
        route=route,
        stage_specs=specs,
        prompt_estimates=prompts,
        tokenization_provenance={"kind": "fixture"},
        route_assumptions={},
    )
    assert first == second
    assert first.b_soft >= first.b_min
    assert first.b0 == math.ceil(first.b_min + 0.25 * (first.b_soft - first.b_min))


def test_historical_calibration_is_deterministic_and_reports_raw_values() -> None:
    runs = [
        CalibrationRun("a", actual_cost=8, b_min=10, b_soft=20),
        CalibrationRun("b", actual_cost=25, b_min=10, b_soft=20),
        CalibrationRun("zero", actual_cost=3, b_min=3, b_soft=3),
    ]
    first = calibrate_rho(runs)
    assert first == calibrate_rho(runs)
    assert first["statistics"]["raw_min"] == -0.2
    assert first["statistics"]["raw_max"] == 1.5
    assert first["statistics"]["negative_count"] == 1
    assert first["statistics"]["zero_denominator_count"] == 1
    assert first["rho_star"] >= 0
