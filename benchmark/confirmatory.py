from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, cast

from benchmark.hotpotqa import HotpotQAAdapter, HotpotTask
from benchmark.outcomes import score_hotpot_result, score_quixbugs_result
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

PROTOCOL_ID = "confirmatory-v1"
FROZEN_RHO_STAR = 0.2963554987212276
RUN_END_EVENT = "CONFIRMATORY_RUN_END"
EXECUTION_ENV = "CONFIRMATORY_EXECUTION_AUTHORIZED"


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("protocol") != PROTOCOL_ID:
        raise ValueError(f"manifest protocol must be {PROTOCOL_ID!r}")

    models = manifest.get("models")
    tasks = manifest.get("tasks")
    policies = manifest.get("policies")
    estimates = manifest.get("initial_budget_estimates")
    runs = manifest.get("runs")
    collections = (models, tasks, policies, estimates, runs)
    if not all(isinstance(value, list) and len(value) > 0 for value in collections):
        raise ValueError(
            "manifest models, tasks, policies, initial_budget_estimates, and runs "
            "must be non-empty lists"
        )

    models = cast(list[dict[str, Any]], models)
    tasks = cast(list[dict[str, Any]], tasks)
    policies = cast(list[str], policies)
    estimates = cast(list[dict[str, Any]], estimates)
    runs = cast(list[dict[str, Any]], runs)
    model_ids = [str(row["model_id"]) for row in models]
    task_ids = [str(row["task_id"]) for row in tasks]
    run_ids = [str(row["run_id"]) for row in runs]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("manifest model ids must be unique")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("manifest task ids must be unique")
    if any(
        not (task_id.startswith("quixbugs:") or task_id.startswith("hotpotqa:"))
        for task_id in task_ids
    ):
        raise ValueError("manifest task ids must use quixbugs: or hotpotqa: prefixes")
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("manifest run ids must be unique")

    valid_policies = {policy.value for policy in Policy}
    policy_names = [str(policy) for policy in policies]
    policy_set = set(policy_names)
    if len(policy_names) != len(policy_set):
        raise ValueError("manifest policies must be unique")
    if not policy_set <= valid_policies:
        raise ValueError("manifest contains an unknown allocation policy")

    known_models = set(model_ids)
    known_tasks = set(task_ids)
    estimate_index: dict[tuple[str, str], dict[str, Any]] = {}
    for estimate in estimates:
        key = (str(estimate["task_id"]), str(estimate["model_id"]))
        if key in estimate_index:
            raise ValueError(f"duplicate initial budget estimate for {key}")
        if key[0] not in known_tasks or key[1] not in known_models:
            raise ValueError("initial budget estimate references an unknown task or model")
        if float(estimate["calibration_rho"]) != FROZEN_RHO_STAR:
            raise ValueError("manifest budget estimates do not use the frozen rho_star")
        stages = estimate.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("each initial budget estimate must contain stage prompt estimates")
        stage_ids = [str(stage["stage_id"]) for stage in stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError(f"duplicate stage estimate in {key}")
        expected = (
            set(CODE_STAGE_SPECS)
            if key[0].startswith("quixbugs:")
            else set(HOTPOT_STAGE_SPECS)
        )
        if set(stage_ids) != expected:
            raise ValueError(
                f"stage estimates do not match the workflow specification for {key[0]}"
            )
        estimate_index[key] = estimate

    for run in runs:
        task_id = str(run["task_id"])
        model_id = str(run["model_id"])
        policy = str(run["policy"])
        if task_id not in known_tasks or model_id not in known_models:
            raise ValueError("run references an unknown task or model")
        if policy not in policy_set:
            raise ValueError("run policy is not declared in manifest policies")
        if (task_id, model_id) not in estimate_index:
            raise ValueError("run has no matching task/model prompt estimate")
        if int(run["B0"]) <= 0:
            raise ValueError("run B0 must be positive")


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
        (str(row["task_id"]), str(row["model_id"])): row
        for row in manifest["initial_budget_estimates"]
    }


def _controller(run: dict[str, Any], estimate: dict[str, Any]) -> BudgetController:
    is_code = str(run["task_id"]).startswith("quixbugs:")
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
        b0=int(run["B0"]),
        route=route,
        stage_specs=specifications,
        prompt_estimates=prompts,
    )


def _attach_outcome(
    *,
    task_id: str,
    result: dict[str, Any],
    hotpot_task: HotpotTask | None,
) -> dict[str, Any]:
    scored = dict(result)
    if task_id.startswith("quixbugs:"):
        scored["outcome"] = score_quixbugs_result(scored)
    else:
        if hotpot_task is None:
            raise ValueError(
                "HotpotQA outcome scoring requires the task only after workflow execution"
            )
        scored["outcome"] = score_hotpot_result(scored, hotpot_task)
    return scored


def execute(manifest: dict[str, Any], root: Path, output: Path) -> None:
    _validate_manifest(manifest)
    if os.environ.get(EXECUTION_ENV) != "YES":
        raise RuntimeError("confirmatory execution is locked until explicit authorization")

    telemetry = AppendOnlyTelemetry(output)
    completed = {
        event["run_id"]
        for event in telemetry.read()
        if event.get("event_type") == RUN_END_EVENT
    }
    estimates = _estimate_index(manifest)
    quixbugs = QuixBugsBenchmark(root / "configs/benchmark.yaml")
    hotpot = HotpotQAAdapter(root / "data/hotpotqa/hotpot_dev_distractor_v1.json")
    models = {row["model_id"]: row for row in manifest["models"]}

    for model_id, model in models.items():
        planned = [run for run in manifest["runs"] if run["model_id"] == model_id]
        if all(run["run_id"] in completed for run in planned):
            continue
        backend = _backend(model)
        local = backend if isinstance(backend, HuggingFaceBackend) else None
        if local is not None:
            local.load()
        try:
            for run in planned:
                if run["run_id"] in completed:
                    continue

                task_id = str(run["task_id"])
                estimate = estimates[(task_id, model_id)]
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
                            "reference_B0": estimate["b0"],
                            "run_B0": int(run["B0"]),
                            "model_native_tokenization_provenance": estimate[
                                "tokenization_provenance"
                            ],
                        },
                    }
                )

                hotpot_task: HotpotTask | None = None
                try:
                    if task_id.startswith("quixbugs:"):
                        result = run_code_workflow(
                            backend=backend,
                            controller=controller,
                            benchmark=quixbugs,
                            task_id=task_id.partition(":")[2],
                        )
                    else:
                        hotpot_task = hotpot.get(task_id.partition(":")[2])
                        result = run_hotpot_workflow(
                            backend=backend,
                            controller=controller,
                            task=hotpot_task,
                        )
                except StructuralShortfallError as error:
                    result = {
                        "task_id": task_id,
                        "terminal_status": "structural_shortfall",
                        "classification": "scientific",
                        "shortfall": error.to_dict(),
                        "run_end": controller.finalize(normal_completion=False),
                    }

                result = _attach_outcome(
                    task_id=task_id,
                    result=result,
                    hotpot_task=hotpot_task,
                )

                for event in controller.state.events:
                    telemetry.append({**event, "run_id": run["run_id"]})
                telemetry.append(
                    {
                        "schema_version": "1.0.0",
                        "event_type": RUN_END_EVENT,
                        "run_id": run["run_id"],
                        "result": result,
                    }
                )
        finally:
            if local is not None:
                local.unload()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate or execute a confirmatory-v1 manifest"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("confirmatory_manifest_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stage_aware_confirmatory_v1/events.jsonl"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).parents[1]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    if not args.execute:
        print(f"manifest valid: {len(manifest['runs'])} runs; execution not requested")
        return
    execute(manifest, root, args.output)


if __name__ == "__main__":
    main()
