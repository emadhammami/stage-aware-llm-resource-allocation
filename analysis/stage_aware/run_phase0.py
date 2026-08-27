from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from analysis.stage_aware.artifact import load_publication_artifact, sha256_file
from analysis.stage_aware.config import load_config
from analysis.stage_aware.diagnostics import (
    cap_binding_summary,
    go_no_go_summary,
    policy_action_rate_rows,
    provenance_summary,
    reservation_summary,
    stage_call_cost_rows,
    stage_cost_summary,
)
from analysis.stage_aware.feasibility import (
    structural_feasibility_by_budget,
    structural_feasibility_rows,
    structural_feasibility_summary,
)
from analysis.stage_aware.figures import generate_all_figures
from analysis.stage_aware.replay import replay_canonical_cohort
from analysis.stage_aware.starvation import starvation_rows, starvation_summary
from analysis.stage_aware.summarize import (
    denial_reason_summary,
    disposition_summary,
    policy_summary,
    reservation_rows,
    stage_summary,
    write_csv,
    write_manifest,
)


def _hash_outputs(paths: list[Path]) -> dict[str, str]:
    return {
        path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def run(artifact_path: Path, config_path: Path, output_directory: Path) -> None:
    config = load_config(config_path)
    source_hash_before = sha256_file(artifact_path)
    bundle = load_publication_artifact(artifact_path, config)
    results = replay_canonical_cohort(
        bundle.canonical_runs, config, bundle.prompt_fallbacks
    )
    feasibility_runs = structural_feasibility_rows(
        bundle.canonical_runs, config, bundle.prompt_fallbacks
    )
    feasibility_summary_rows = structural_feasibility_summary(feasibility_runs, config)
    feasibility_budget_rows = structural_feasibility_by_budget(
        bundle.canonical_runs, config, bundle.prompt_fallbacks
    )
    policy_rows = policy_summary(results)
    stage_rows = stage_summary(results)
    disposition_rows = disposition_summary(results)
    call_cost_rows = stage_call_cost_rows(bundle)
    stage_cost_rows = stage_cost_summary(call_cost_rows)
    cap_rows = cap_binding_summary(call_cost_rows, config)
    action_rows = policy_action_rate_rows(results)
    reservation_summary_rows = reservation_summary(results)
    provenance_rows = provenance_summary(bundle, results)
    starvation_run_rows = starvation_rows(
        bundle.canonical_runs, results, feasibility_runs
    )
    starvation_summary_rows = starvation_summary(starvation_run_rows)
    gate_rows = go_no_go_summary(results, feasibility_runs)

    output_directory.mkdir(parents=True, exist_ok=True)
    table_paths = {
        "artifact_data_quality.csv": list(bundle.quality_metrics),
        "structural_feasibility_runs.csv": feasibility_runs,
        "structural_feasibility_summary.csv": feasibility_summary_rows,
        "structural_feasibility_by_budget.csv": feasibility_budget_rows,
        "stage_call_costs.csv": call_cost_rows,
        "stage_cost_summary.csv": stage_cost_rows,
        "cap_binding_summary.csv": cap_rows,
        "policy_action_rates.csv": action_rows,
        "reservation_summary.csv": reservation_summary_rows,
        "provenance_summary.csv": provenance_rows,
        "starvation_opportunities.csv": starvation_run_rows,
        "starvation_summary.csv": starvation_summary_rows,
        "go_no_go_summary.csv": gate_rows,
        "replay_run_summary.csv": [result.to_dict() for result in results],
        "allocation_ledger.csv": [
            event.to_dict() for result in results for event in result.events
        ],
        "reservation_ledger.csv": reservation_rows(results),
        "policy_summary.csv": policy_rows,
        "stage_summary.csv": stage_rows,
        "disposition_summary.csv": disposition_rows,
        "denial_reason_summary.csv": denial_reason_summary(results),
    }
    generated: list[Path] = []
    for filename, rows in table_paths.items():
        path = output_directory / filename
        write_csv(path, rows)
        generated.append(path)
    generated.extend(
        generate_all_figures(
            output_directory=output_directory,
            call_rows=call_cost_rows,
            feasibility_rows=feasibility_summary_rows,
            policy_rows=policy_rows,
            stage_rows=stage_rows,
            disposition_rows=disposition_rows,
            cap_binding_rows=cap_rows,
            starvation_rows=starvation_summary_rows,
        )
    )
    source_hash_after = sha256_file(artifact_path)
    if source_hash_before != source_hash_after:
        raise AssertionError("publication artifact changed during read-only replay")
    go_rows = [
        row
        for row in gate_rows
        if row["policy"] == "adaptive_stage_aware" and row["target_budget"] in {2000, 4000}
    ]
    material_gate = all(float(row["material_action_fraction"]) >= 0.20 for row in go_rows)
    manifest: dict[str, Any] = {
        "analysis_kind": "offline_telemetry_replay",
        "schema_version": config.schema_version,
        "canonical_publication_runs_parsed": len(bundle.runs),
        "canonical_policy_replay_runs": len(results),
        "canonical_replay_condition": {
            "method": config.canonical_method,
            "source_budget": config.canonical_source_budget,
            "runs": len(bundle.canonical_runs),
        },
        "counterfactual_quality_estimated": False,
        "finish_reason_available": False,
        "go_no_go": {
            "threshold": 0.20,
            "budgets": [2000, 4000],
            "adaptive_material_action_gate_passed": material_gate,
        },
        "publication_artifact": {
            "path": artifact_path.as_posix(),
            "sha256_before": source_hash_before,
            "sha256_after": source_hash_after,
            "immutable": source_hash_before == source_hash_after,
        },
        "publication_commit": config.publication_commit,
        "target_budgets": list(config.target_budgets),
        "policies": [policy.value for policy in config.policies],
        "generated_file_sha256": _hash_outputs(generated),
    }
    write_manifest(output_directory / "phase0_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the no-inference Phase 0 replay audit")
    parser.add_argument("--artifact", type=Path, default=Path("final_core_results.zip"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("research/stage_aware/phase0_config.yaml"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/stage_aware_phase0")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.artifact, args.config, args.output)


if __name__ == "__main__":
    main()
