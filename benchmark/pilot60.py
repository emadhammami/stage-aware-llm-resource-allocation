from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from benchmark.hotpotqa import HotpotQAAdapter
from benchmark.quixbugs import QuixBugsBenchmark
from workflow_control.backends import (
    GeminiScientificBackend,
    HuggingFaceBackend,
    LocalModelConfig,
    ModelBackend,
)
from workflow_control.controller import BudgetController
from workflow_control.runtime import (
    StructuralShortfallError,
    run_code_workflow,
    run_hotpot_workflow,
)
from workflow_control.specs import CODE_ROUTE, CODE_STAGE_SPECS, HOTPOT_ROUTE, HOTPOT_STAGE_SPECS
from workflow_control.telemetry import AppendOnlyTelemetry
from workflow_control.types import Policy, PromptEstimate

FROZEN_RHO_STAR = 0.2963554987212276
PILOT_MODELS = {
    "Qwen/Qwen3-8B": {
        "family": "qwen3",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "family": "llama",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
    },
    "google/gemma-4-E4B-it": {
        "family": "gemma4",
        "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
    },
}
PILOT_MODEL_IDS = set(PILOT_MODELS)
PILOT_POLICIES = {"legacy_static", "greedy", "fixed_reservation", "proposed"}


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("matrix") != {
        "tasks": 5,
        "policies": 4,
        "models": 3,
        "repetitions": 1,
        "runs": 60,
    }:
        raise ValueError("manifest matrix is not the frozen 5 x 4 x 3 x 1 design")
    runs = manifest["runs"]
    if len(runs) != 60 or len({run["run_id"] for run in runs}) != 60:
        raise ValueError("manifest must contain 60 unique run ids")
    models = manifest["models"]
    if len(models) != 3:
        raise ValueError("manifest must contain exactly three model definitions")
    model_ids = {model["model_id"] for model in models}
    run_model_ids = {run["model_id"] for run in runs}
    if model_ids != PILOT_MODEL_IDS or run_model_ids != PILOT_MODEL_IDS:
        raise ValueError("manifest must contain exactly Gemma, Qwen, and Llama")
    for model in models:
        expected = PILOT_MODELS[model["model_id"]]
        if model["family"] != expected["family"]:
            raise ValueError(f"incorrect family for {model['model_id']}")
        if model["revision"] != expected["revision"]:
            raise ValueError(f"incorrect pinned revision for {model['model_id']}")
        if model["tokenizer_revision"] != expected["revision"]:
            raise ValueError(f"incorrect tokenizer revision for {model['model_id']}")

    tasks = {task["task_id"] for task in manifest["tasks"]}
    if len(tasks) != 5 or {run["task_id"] for run in runs} != tasks:
        raise ValueError("manifest must contain exactly five frozen tasks")
    if set(manifest["policies"]) != PILOT_POLICIES:
        raise ValueError("manifest policies differ from the frozen A-D set")
    if Counter(run["model_id"] for run in runs) != Counter(
        {model_id: 20 for model_id in PILOT_MODEL_IDS}
    ):
        raise ValueError("manifest must contain 20 runs per model")
    if Counter(run["policy"] for run in runs) != Counter(
        {policy: 15 for policy in PILOT_POLICIES}
    ):
        raise ValueError("manifest must contain 15 runs per policy")
    if Counter(run["task_id"] for run in runs) != Counter({task_id: 12 for task_id in tasks}):
        raise ValueError("manifest must contain 12 runs per task")
    if any(run["repetition"] != 1 for run in runs):
        raise ValueError("manifest repetitions must all equal one")

    estimates = {
        (estimate["task_id"], estimate["model_id"]): estimate
        for estimate in manifest["initial_budget_estimates"]
    }
    expected_cells = {(task_id, model_id) for task_id in tasks for model_id in PILOT_MODEL_IDS}
    if len(manifest["initial_budget_estimates"]) != 15 or set(estimates) != expected_cells:
        raise ValueError("manifest must contain exactly 15 task/model budget estimates")
    if any(float(estimate["calibration_rho"]) != FROZEN_RHO_STAR for estimate in estimates.values()):
        raise ValueError("manifest budget estimates do not use the frozen rho_star")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((run["task_id"], run["model_id"]), []).append(run)
    if set(grouped) != expected_cells:
        raise ValueError("manifest run cells differ from the frozen task/model product")
    for cell_key, cell in grouped.items():
        if len(cell) != 4 or len({run["B0"] for run in cell}) != 1:
            raise ValueError("A-D do not share one B0 in a task/model cell")
        if {run["policy"] for run in cell} != PILOT_POLICIES:
            raise ValueError("task/model cell does not contain all four policies")
        if cell[0]["B0"] != estimates[cell_key]["b0"]:
            raise ValueError("run B0 differs from its task/model budget estimate")
        if len({run["scientific_config_fingerprint"] for run in cell}) != 1:
            raise ValueError("scientific configuration differs across policies")


def _backend(model: dict[str, Any]) -> ModelBackend:
    if model["family"] == "gemini":
        return GeminiScientificBackend()
    return HuggingFaceBackend(
        LocalModelConfig(
            model_id=model["model_id"],
            revision=model.get("revision"),
            tokenizer_revision=model.get("tokenizer_revision"),
            local_files_only=True,
            device=model["device"],
            dtype=model["dtype"],
        ),
        family=model["family"],
    )


def _estimate_index(manifest: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (row["task_id"], row["model_id"]): row
        for row in manifest["initial_budget_estimates"]
    }


def _controller(run: dict[str, Any], estimate: dict[str, Any]) -> BudgetController:
    is_code = run["task_id"].startswith("quixbugs:")
    route = CODE_ROUTE if is_code else HOTPOT_ROUTE
    specifications = CODE_STAGE_SPECS if is_code else HOTPOT_STAGE_SPECS
    prompts = {
        row["stage_id"]: PromptEstimate(
            stage_id=row["stage_id"],
            predicted_prompt_tokens=row["predicted_prompt_tokens"],
            exact_prompt_tokens=row["exact_prompt_tokens"],
            provenance=row["prompt_provenance"],
            rendered_prompt_sha256=row["rendered_prompt_sha256"],
        )
        for row in estimate["stages"]
    }
    return BudgetController(
        task_id=run["task_id"],
        model_id=run["model_id"],
        policy=Policy(run["policy"]),
        b0=run["B0"],
        route=route,
        stage_specs=specifications,
        prompt_estimates=prompts,
    )


def execute(manifest: dict[str, Any], root: Path, output: Path) -> None:
    if os.environ.get("PILOT60_EXECUTION_AUTHORIZED") != "YES":
        raise RuntimeError("real execution is locked until explicit authorization")
    telemetry = AppendOnlyTelemetry(output)
    completed = {
        event["run_id"]
        for event in telemetry.read()
        if event.get("event_type") == "PILOT_RUN_END"
    }
    estimates = _estimate_index(manifest)
    quixbugs = QuixBugsBenchmark(root / "configs/benchmark.yaml")
    hotpot = HotpotQAAdapter(root / "data/hotpotqa/hotpot_dev_distractor_v1.json")
    models = {row["model_id"]: row for row in manifest["models"]}
    for model_id in models:
        planned = [run for run in manifest["runs"] if run["model_id"] == model_id]
        if all(run["run_id"] in completed for run in planned):
            continue
        backend = _backend(models[model_id])
        local = backend if isinstance(backend, HuggingFaceBackend) else None
        if local is not None:
            local.load()
        try:
            for run in planned:
                if run["run_id"] in completed:
                    continue
                estimate = estimates[(run["task_id"], model_id)]
                controller = _controller(run, estimate)
                telemetry.append(
                    {
                        "schema_version": "1.0.0",
                        "event_type": "INITIAL_BUDGET",
                        "run_id": run["run_id"],
                        "initial_budget": {
                            "estimator_version": estimate["estimator_version"],
                            "B_min": estimate["b_min"],
                            "B_soft": estimate["b_soft"],
                            "rho_star": estimate["calibration_rho"],
                            "B0": estimate["b0"],
                            "model_native_tokenization_provenance": estimate[
                                "tokenization_provenance"
                            ],
                        },
                    }
                )
                try:
                    if run["task_id"].startswith("quixbugs:"):
                        result = run_code_workflow(
                            backend=backend,
                            controller=controller,
                            benchmark=quixbugs,
                            task_id=run["task_id"].partition(":")[2],
                        )
                    else:
                        result = run_hotpot_workflow(
                            backend=backend,
                            controller=controller,
                            task=hotpot.get(run["task_id"].partition(":")[2]),
                        )
                except StructuralShortfallError as error:
                    result = {
                        "task_id": run["task_id"],
                        "terminal_status": "structural_shortfall",
                        "classification": "scientific",
                        "shortfall": error.to_dict(),
                        "run_end": controller.finalize(normal_completion=False),
                    }
                for event in controller.state.events:
                    telemetry.append({**event, "run_id": run["run_id"]})
                telemetry.append(
                    {
                        "schema_version": "1.0.0",
                        "event_type": "PILOT_RUN_END",
                        "run_id": run["run_id"],
                        "result": result,
                    }
                )
        finally:
            if local is not None:
                local.unload()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or execute the frozen pilot60 manifest")
    parser.add_argument("--manifest", type=Path, default=Path("pilot60_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/stage_aware_pilot60/events.jsonl"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    if not args.execute:
        print("manifest valid: 60 runs; execution not requested")
        return
    execute(manifest, root, args.output)


if __name__ == "__main__":
    main()
