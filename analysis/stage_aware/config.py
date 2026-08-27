from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from analysis.stage_aware.types import PolicyName, StageSpec, WorkflowStage


@dataclass(frozen=True)
class Phase0Config:
    schema_version: str
    publication_commit: str
    artifact_sha256: str
    source_artifact: str
    target_budgets: tuple[int, ...]
    near_cap_thresholds: tuple[float, ...]
    policies: tuple[PolicyName, ...]
    stage_specs: dict[WorkflowStage, StageSpec]
    fixed_reservation_fraction: float
    material_action_threshold: int
    canonical_method: str
    canonical_source_budget: int


def load_config(path: str | Path) -> Phase0Config:
    with Path(path).open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    stage_specs = {
        WorkflowStage(name): StageSpec(
            stage=WorkflowStage(name),
            minimum_output=int(values["minimum_output"]),
            soft_output=int(values["soft_output"]),
            hard_output=int(values["hard_output"]),
            legacy_output=int(values["legacy_output"]),
        )
        for name, values in raw["stages"].items()
    }
    return Phase0Config(
        schema_version=str(raw["schema_version"]),
        publication_commit=str(raw["publication_commit"]),
        artifact_sha256=str(raw["artifact_sha256"]),
        source_artifact=str(raw["source_artifact"]),
        target_budgets=tuple(int(value) for value in raw["target_budgets"]),
        near_cap_thresholds=tuple(float(value) for value in raw["near_cap_thresholds"]),
        policies=tuple(PolicyName(value) for value in raw["policies"]),
        stage_specs=stage_specs,
        fixed_reservation_fraction=float(raw["fixed_reservation_fraction"]),
        material_action_threshold=int(raw["material_action_threshold"]),
        canonical_method=str(raw["canonical_replay"]["method"]),
        canonical_source_budget=int(raw["canonical_replay"]["source_budget"]),
    )
