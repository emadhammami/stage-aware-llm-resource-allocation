from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from analysis.stage_aware.run_phase0 import run


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_is_deterministic_and_has_no_quality_counterfactual(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    artifact = Path("final_core_results.zip")
    config = Path("research/stage_aware/phase0_config.yaml")
    source_before = digest(artifact)
    run(artifact, config, first)
    run(artifact, config, second)
    for relative in (
        "allocation_ledger.csv",
        "policy_summary.csv",
        "structural_feasibility_summary.csv",
        "figures/figure_3_policy_action_rates.png",
        "figures/figure_3_policy_action_rates.pdf",
    ):
        assert digest(first / relative) == digest(second / relative)
    assert digest(artifact) == source_before
    manifest = (first / "phase0_manifest.json").read_text(encoding="utf-8")
    assert '"counterfactual_quality_estimated": false' in manifest
    assert '"adaptive_material_action_gate_passed": true' in manifest

    with (first / "allocation_ledger.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        first_row = next(reader)
    required_schema = {
        "schema_version",
        "experiment_id",
        "policy",
        "target_budget",
        "stage",
        "allocated_output",
        "reservation_created_amount",
        "returned_capacity",
        "reallocation_available",
        "structural_infeasible",
        "prompt_provenance",
    }
    assert required_schema <= set(first_row)
