from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Mapping

from workflow_control.routes import RouteNode, maximal_path, maximal_requirement
from workflow_control.types import InitialBudgetEstimate, PromptEstimate, StageSpec


class InitialBudgetEstimator:
    """Compute a policy-independent starting quota from a frozen route and rho."""

    VERSION = "1.0.0"

    def __init__(self, calibration_rho: float) -> None:
        if calibration_rho < 0:
            raise ValueError("calibration_rho must be non-negative")
        self.calibration_rho = calibration_rho

    def estimate(
        self,
        *,
        model_id: str,
        task_id: str,
        route: RouteNode,
        stage_specs: Mapping[str, StageSpec],
        prompt_estimates: Mapping[str, PromptEstimate],
        tokenization_provenance: Mapping[str, Any],
        route_assumptions: Mapping[str, Any],
    ) -> InitialBudgetEstimate:
        stage_ids = {stage for path in route.paths() for stage in path}
        if stage_ids != set(stage_specs) or stage_ids != set(prompt_estimates):
            raise ValueError("route, stage specifications, and prompt estimates must agree")
        minimum = {
            stage: prompt_estimates[stage].effective_tokens + stage_specs[stage].minimum_output
            for stage in stage_ids
        }
        soft = {
            stage: prompt_estimates[stage].effective_tokens + stage_specs[stage].soft_output
            for stage in stage_ids
        }
        b_min = maximal_requirement(route, minimum)
        b_soft = maximal_requirement(route, soft)
        if b_soft < b_min:
            raise AssertionError("soft-feasible budget is below minimum-feasible budget")
        b0 = math.ceil(b_min + self.calibration_rho * (b_soft - b_min))
        min_path = maximal_path(route, minimum)
        soft_path = maximal_path(route, soft)
        stage_rows = []
        for stage in sorted(stage_ids):
            prompt = prompt_estimates[stage]
            stage_rows.append(
                {
                    "stage_id": stage,
                    "predicted_prompt_tokens": prompt.predicted_prompt_tokens,
                    "exact_prompt_tokens": prompt.exact_prompt_tokens,
                    "prediction_error_tokens": prompt.prediction_error_tokens,
                    "prompt_provenance": prompt.provenance,
                    "minimum_output": stage_specs[stage].minimum_output,
                    "soft_output": stage_specs[stage].soft_output,
                    "rendered_prompt_sha256": prompt.rendered_prompt_sha256,
                }
            )
        assumptions = dict(route_assumptions)
        assumptions.update(
            {
                "minimum_maximal_path": sorted(min_path),
                "soft_maximal_path": sorted(soft_path),
                "branch_operator": "max",
                "shared_stage_accounting": "once_per_path",
            }
        )
        return InitialBudgetEstimate(
            estimator_version=self.VERSION,
            model_id=model_id,
            task_id=task_id,
            b_min=b_min,
            b_soft=b_soft,
            calibration_rho=self.calibration_rho,
            b0=b0,
            route_assumptions=assumptions,
            stages=tuple(stage_rows),
            tokenization_provenance=dict(tokenization_provenance),
        )


def estimate_fingerprint(estimate: InitialBudgetEstimate) -> dict[str, Any]:
    data = asdict(estimate)
    data.pop("b0")
    return data
