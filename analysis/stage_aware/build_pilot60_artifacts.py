from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import yaml

from analysis.stage_aware.artifact import load_publication_artifact
from analysis.stage_aware.config import load_config
from benchmark.hotpotqa import HotpotQAAdapter
from benchmark.quixbugs import QuixBugsBenchmark
from workflow_control.backends import NativeChatCounter
from workflow_control.calibration import CalibrationRun, calibrate_rho
from workflow_control.initial_budget import InitialBudgetEstimator
from workflow_control.specs import (
    CODE_ROUTE,
    CODE_STAGE_SPECS,
    HOTPOT_ROUTE,
    HOTPOT_STAGE_SPECS,
    code_prompt_estimates,
    hotpot_prompt_estimates,
)

FROZEN_RHO_STAR = 0.2963554987212276


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def historical_calibration(
    artifact_path: str | Path,
    phase0_config_path: str | Path,
) -> dict[str, Any]:
    phase0 = load_config(phase0_config_path)
    artifact = load_publication_artifact(artifact_path, phase0)
    calibration_runs: list[CalibrationRun] = []
    for run in artifact.runs:
        admitted = [call for call in run.calls if call.admitted]
        if any(call.prompt_tokens is None or call.total_tokens is None for call in admitted):
            continue
        b_min = sum(
            int(call.prompt_tokens or 0) + phase0.stage_specs[call.stage].minimum_output
            for call in admitted
        )
        b_soft = sum(
            int(call.prompt_tokens or 0) + phase0.stage_specs[call.stage].soft_output
            for call in admitted
        )
        actual = sum(int(call.total_tokens or 0) for call in admitted)
        calibration_runs.append(CalibrationRun(run.experiment_id, actual, b_min, b_soft))
    result = calibrate_rho(calibration_runs)
    return {
        "schema_version": "1.0.0",
        "calibration_id": "publication-resource-p90-v1",
        "source": {
            "artifact": Path(artifact_path).name,
            "artifact_sha256": artifact.checksum,
            "publication_commit": phase0.publication_commit,
            "total_historical_runs": len(artifact.runs),
            "historically_replayable_runs": len(calibration_runs),
            "resource_fields": ["provider_input_tokens", "provider_output_tokens"],
            "excluded_fields": [
                "correctness",
                "hidden_tests",
                "success",
                "reference_answer",
                "final_quality",
            ],
        },
        "definition": {
            "route_assumption": "historical_realized_admitted_route",
            "B_min_r": "sum(observed prompt tokens + frozen stage minimum output)",
            "B_soft_r": "sum(observed prompt tokens + frozen stage soft output)",
            "rho_r": "(actual provider cost - B_min_r) / (B_soft_r - B_min_r)",
            "frozen_quantile": "P90",
        },
        **result,
        "validation": {
            "p90_sensible": 0 <= float(str(result["rho_star"])) <= 1,
            "transfer_to_local_models": "frozen rho; recompute model-native B_min/B_soft",
        },
    }


def load_frozen_calibration(
    path: str | Path, *, expected_rho_star: float
) -> dict[str, Any]:
    calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    observed = float(calibration["rho_star"])
    if observed != FROZEN_RHO_STAR or observed != expected_rho_star:
        raise ValueError(
            f"frozen rho_star mismatch: expected {FROZEN_RHO_STAR}, observed {observed}"
        )
    return calibration


def _native_counter(chat_component: Any, *, thinking_disabled: bool) -> Callable[[str], int]:
    counter = NativeChatCounter(chat_component, thinking_disabled=thinking_disabled)

    def count(prompt: str) -> int:
        return counter.count([{"role": "user", "content": prompt}])[0]

    return count


def _load_local_tokenizers(
    root: Path, models: list[dict[str, Any]]
) -> dict[str, tuple[Callable[[str], int], dict[str, Any]]]:
    from transformers import AutoProcessor, AutoTokenizer

    counters: dict[str, tuple[Callable[[str], int], dict[str, Any]]] = {}
    for model in models:
        model_id = model["model_id"]
        revision = model["tokenizer_revision"]
        escaped = model_id.replace("/", "--")
        location = root / ".model_cache" / f"models--{escaped}" / "snapshots" / revision
        if not location.exists():
            raise FileNotFoundError(
                f"pinned tokenizer snapshot is unavailable for {model_id} at {location}"
            )
        if model["family"] == "gemma4":
            chat_component = AutoProcessor.from_pretrained(
                str(location.relative_to(root)), local_files_only=True
            )
            tokenizer = chat_component.tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                str(location.relative_to(root)), local_files_only=True
            )
            chat_component = tokenizer
        thinking_disabled = model["family"] in {"gemma4", "qwen3"}
        native = NativeChatCounter(chat_component, thinking_disabled=thinking_disabled)
        counters[model_id] = (
            _native_counter(chat_component, thinking_disabled=thinking_disabled),
            {
                "kind": "native_chat_template_tokenizer",
                "model_native": True,
                "model_id": model_id,
                "model_revision": model["revision"],
                "tokenizer_revision": revision,
                "tokenizer_vocabulary_size": len(tokenizer),
                "chat_template_sha256": native.template_hash,
                "thinking": model.get("thinking", "not_applicable"),
            },
        )
    return counters


def build_manifest(root: Path, configuration: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    adapter = HotpotQAAdapter(root / configuration["hotpotqa_dataset"])
    selected = adapter.deterministic_pilot_ids(configuration["selection_seed"])
    if selected != configuration["tasks"]["hotpotqa"]:
        raise ValueError("frozen HotpotQA ids differ from deterministic selection")
    benchmark = QuixBugsBenchmark(root / "configs/benchmark.yaml")
    counters = _load_local_tokenizers(root, configuration["models"])
    estimator = InitialBudgetEstimator(float(calibration["rho_star"]))
    estimates: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for task_id in configuration["tasks"]["quixbugs"]:
        task_rows.append({"task_id": f"quixbugs:{task_id}", "benchmark": "quixbugs"})
        code = benchmark.load_buggy_code(task_id)
        for model in configuration["models"]:
            model_id = model["model_id"]
            counter, provenance = counters[model_id]
            estimate = estimator.estimate(
                model_id=model_id,
                task_id=f"quixbugs:{task_id}",
                route=CODE_ROUTE,
                stage_specs=CODE_STAGE_SPECS,
                prompt_estimates=code_prompt_estimates(
                    task_id=task_id,
                    code=code,
                    counter=counter,
                    provenance=provenance["kind"],
                ),
                tokenization_provenance=provenance,
                route_assumptions={
                    "workflow": "plan -> execute -> optional retry -> verify",
                    "optional_retry": "maximal reachable continuation",
                },
            )
            estimates.append(estimate.to_dict())
    for question_type in ("bridge", "comparison"):
        native_id = selected[question_type]
        task_id = f"hotpotqa:{native_id}"
        task_rows.append(
            {"task_id": task_id, "benchmark": "hotpotqa", "question_type": question_type}
        )
        hotpot_task = adapter.get(native_id)
        for model in configuration["models"]:
            model_id = model["model_id"]
            counter, provenance = counters[model_id]
            estimate = estimator.estimate(
                model_id=model_id,
                task_id=task_id,
                route=HOTPOT_ROUTE,
                stage_specs=HOTPOT_STAGE_SPECS,
                prompt_estimates=hotpot_prompt_estimates(
                    task=hotpot_task,
                    counter=counter,
                    provenance=provenance["kind"],
                ),
                tokenization_provenance=provenance,
                route_assumptions={
                    "workflow": "plan -> local retrieval -> answer -> verify -> optional revise -> terminal verify",
                    "initial_retrieval_documents": 2,
                    "expanded_retrieval_documents": 3,
                },
            )
            estimates.append(estimate.to_dict())
    estimate_by_cell = {(row["task_id"], row["model_id"]): row for row in estimates}
    frozen_shared = {
        "selection_seed": configuration["selection_seed"],
        "rho_star": calibration["rho_star"],
        "fixed_reservation_fraction": configuration["fixed_reservation_fraction"],
        "repetition": 1,
        "prompt_and_route_version": "pilot60-v1",
    }
    runs = []
    for task in task_rows:
        for model in configuration["models"]:
            cell = estimate_by_cell[(task["task_id"], model["model_id"])]
            for policy in configuration["policies"]:
                run_configuration = {
                    **frozen_shared,
                    "task_id": task["task_id"],
                    "model_id": model["model_id"],
                    "B0": cell["b0"],
                }
                runs.append(
                    {
                        "run_id": f"{task['task_id']}::{model['family']}::{policy}::r1",
                        **run_configuration,
                        "policy": policy,
                        "scientific_config_fingerprint": canonical_json_hash(run_configuration),
                        "full_config_fingerprint": canonical_json_hash(
                            {**run_configuration, "policy": policy}
                        ),
                    }
                )
    if len(runs) != 60:
        raise AssertionError(f"pilot manifest must contain 60 runs, got {len(runs)}")
    return {
        "schema_version": "1.0.0",
        "manifest_id": "fllm-2026-pilot60-open-models-v2",
        "execution_authorized": False,
        "matrix": {"tasks": 5, "policies": 4, "models": 3, "repetitions": 1, "runs": 60},
        "tasks": task_rows,
        "models": configuration["models"],
        "policies": configuration["policies"],
        "calibration_id": calibration["calibration_id"],
        "dataset": {
            "name": "HotpotQA dev distractor v1",
            "source_url": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
            "archive_url": "http://web.archive.org/web/20260310132809id_/http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
            "sha256": adapter.dataset_sha256(),
            "selection_seed": configuration["selection_seed"],
        },
        "initial_budget_estimates": estimates,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", default="research/stage_aware/pilot60_config.yaml")
    args = parser.parse_args()
    root = args.root.resolve()
    configuration = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    calibration = load_frozen_calibration(
        root / configuration["initial_budget_calibration"],
        expected_rho_star=float(configuration["rho_star"]),
    )
    manifest = build_manifest(root, configuration, calibration)
    (root / "pilot60_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"used frozen rho_star={calibration['rho_star']} without recalibration")
    print(f"wrote manifest with {len(manifest['runs'])} planned runs")


if __name__ == "__main__":
    main()
