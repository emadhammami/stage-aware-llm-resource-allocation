from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import yaml

RUN_END_EVENT = "CONFIRMATORY_RUN_END"
INITIAL_BUDGET_EVENT = "INITIAL_BUDGET"
EXPECTED_MANIFEST_CANONICAL_SHA256 = "A351BBA587CB34D879E5DDBD670E7D091EB94AFB43B6FE120A967F77DCB34AD5"
EXPECTED_ANALYSIS_PLAN_CANONICAL_SHA256 = "DB2F07326C257CF01CFBF34ECFBAEEAEB5DADC6B66572DFFC046C1A0DC49050E"
EXPECTED_RAW_EVENTS_SHA256 = "1C17817145CD64276D88096EE4559097E95A729E50B306DBA30C8075C0DD3E71"
PRIMARY_METRICS = (
    "exact_match",
    "token_f1",
    "reliable_correct",
    "verification_supported_correct",
)
QUIXBUGS_METRICS = (
    "functional_correct",
    "end_to_end_functional_correct",
    "reliable_correct",
)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    task_id: str
    benchmark: str
    model_id: str
    policy: str
    budget_condition: str
    repetition: int
    primary_analysis: bool
    result: dict[str, Any]
    attempt_events: tuple[dict[str, Any], ...]


def _benchmark(task_id: str) -> str:
    if task_id.startswith("hotpotqa:"):
        return "hotpotqa"
    if task_id.startswith("quixbugs:"):
        return "quixbugs"
    raise ValueError(f"unsupported task id: {task_id}")


def load_completed_runs(
    manifest: dict[str, Any], events: Iterable[dict[str, Any]]
) -> list[RunRecord]:
    run_defs = {str(row["run_id"]): row for row in manifest["runs"]}
    if len(run_defs) != len(manifest["runs"]):
        raise ValueError("manifest run ids are not unique")

    active: dict[str, list[dict[str, Any]]] = {}
    completed: dict[str, RunRecord] = {}
    for event in events:
        run_id_raw = event.get("run_id")
        if run_id_raw is None:
            continue
        run_id = str(run_id_raw)
        event_type = event.get("event_type")

        if event_type == INITIAL_BUDGET_EVENT:
            active[run_id] = [event]
            continue

        if run_id in active:
            active[run_id].append(event)

        if event_type != RUN_END_EVENT:
            continue
        if run_id not in run_defs:
            raise ValueError(f"run end not present in manifest: {run_id}")
        if run_id in completed:
            raise ValueError(f"duplicate confirmatory run end: {run_id}")
        if run_id not in active:
            raise ValueError(f"run end has no attempt start: {run_id}")

        definition = run_defs[run_id]
        result = event.get("result")
        if not isinstance(result, dict):
            raise ValueError(f"run end result is not a mapping: {run_id}")
        completed[run_id] = RunRecord(
            run_id=run_id,
            task_id=str(definition["task_id"]),
            benchmark=_benchmark(str(definition["task_id"])),
            model_id=str(definition["model_id"]),
            policy=str(definition["policy"]),
            budget_condition=str(definition["budget_condition"]),
            repetition=int(definition["repetition"]),
            primary_analysis=bool(definition.get("primary_analysis", False)),
            result=result,
            attempt_events=tuple(active[run_id]),
        )
        active.pop(run_id, None)

    expected = set(run_defs)
    observed = set(completed)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"completed run ids differ from manifest; missing={missing[:5]} extra={extra[:5]}"
        )
    return [completed[str(row["run_id"])] for row in manifest["runs"]]


def _outcome_value(record: RunRecord, metric: str) -> float:
    outcome = record.result.get("outcome")
    if not isinstance(outcome, dict) or metric not in outcome:
        raise ValueError(f"{record.run_id} lacks outcome metric {metric}")
    value = outcome[metric]
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"{record.run_id} metric {metric} is not numeric")


def _run_end(record: RunRecord) -> dict[str, Any]:
    run_end = record.result.get("run_end")
    if not isinstance(run_end, dict):
        raise ValueError(f"{record.run_id} lacks run_end")
    return run_end


def _resource_values(record: RunRecord) -> dict[str, float]:
    run_end = _run_end(record)
    stage_rows = record.result.get("stages")
    stages = stage_rows if isinstance(stage_rows, list) else []
    stage_ids = {
        str(stage["stage_id"])
        for stage in stages
        if isinstance(stage, dict) and "stage_id" in stage
    }
    event_types = Counter(str(event.get("event_type")) for event in record.attempt_events)

    if record.benchmark == "hotpotqa":
        downstream_completed = float("verifier" in stage_ids)
    else:
        downstream_completed = float("critic" in stage_ids)

    return {
        "total_consumed_tokens": float(run_end["total_consumed"]),
        "unused_capacity": float(run_end["unused_capacity"]),
        "provider_calls": float(event_types["PROVIDER_DEBIT"]),
        "downstream_stage_completion": downstream_completed,
        "structural_shortfall": float(event_types["STRUCTURAL_SHORTFALL"] > 0),
        "reservation_shortfall": float(event_types["RESERVATION_SHORTFALL"] > 0),
    }


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("cannot average an empty collection")
    return float(sum(items) / len(items))


def aggregate_repetitions(
    records: Iterable[RunRecord], metric_getter: Callable[[RunRecord], float]
) -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in records:
        grouped[(record.task_id, record.model_id, record.policy)].append(
            float(metric_getter(record))
        )
    return {key: _mean(values) for key, values in grouped.items()}


def paired_bootstrap(
    task_ids: list[str],
    model_ids: list[str],
    legacy: dict[tuple[str, str], float],
    proposed: dict[tuple[str, str], float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if not task_ids or not model_ids:
        raise ValueError("bootstrap requires tasks and models")
    for task_id in task_ids:
        for model_id in model_ids:
            if (task_id, model_id) not in legacy or (task_id, model_id) not in proposed:
                raise ValueError(f"unpaired primary cell: {(task_id, model_id)}")

    deltas = np.asarray(
        [
            [proposed[(task_id, model_id)] - legacy[(task_id, model_id)] for task_id in task_ids]
            for model_id in model_ids
        ],
        dtype=np.float64,
    )
    point_per_model = deltas.mean(axis=1)
    point_macro = float(point_per_model.mean())

    rng = np.random.default_rng(seed)
    n_tasks = len(task_ids)
    boot_per_model = np.empty((resamples, len(model_ids)), dtype=np.float64)
    for start in range(0, resamples, 1000):
        stop = min(start + 1000, resamples)
        index = rng.integers(0, n_tasks, size=(stop - start, n_tasks))
        sampled = deltas[:, index].transpose(1, 0, 2)
        boot_per_model[start:stop] = sampled.mean(axis=2)
    boot_macro = boot_per_model.mean(axis=1)

    def ci(values: np.ndarray) -> list[float]:
        low, high = np.quantile(values, [0.025, 0.975], method="linear")
        return [float(low), float(high)]

    return {
        "per_model_delta": {
            model_id: {
                "point": float(point_per_model[i]),
                "ci95": ci(boot_per_model[:, i]),
            }
            for i, model_id in enumerate(model_ids)
        },
        "macro_delta": {"point": point_macro, "ci95": ci(boot_macro)},
    }


def _policy_raw(
    task_ids: list[str],
    model_ids: list[str],
    values: dict[tuple[str, str], float],
) -> dict[str, Any]:
    per_model = {
        model_id: _mean(values[(task_id, model_id)] for task_id in task_ids)
        for model_id in model_ids
    }
    return {
        "per_model": per_model,
        "macro": _mean(per_model.values()),
    }



def validate_primary_shape(
    records: list[RunRecord],
    manifest: dict[str, Any],
    plan: dict[str, Any],
) -> None:
    primary = [record for record in records if record.primary_analysis]
    comparison = plan["primary_comparison"]
    expected_models = int(comparison["models"])
    expected_repetitions = int(comparison["repetitions"])
    expected_runs = int(manifest["matrix_summary"]["primary_runs"])
    expected_policies = {
        str(comparison["baseline_policy"]),
        str(comparison["candidate_policy"]),
    }
    if len(primary) != expected_runs:
        raise ValueError(
            f"primary run count differs from frozen manifest: {len(primary)} != {expected_runs}"
        )
    if {record.benchmark for record in primary} != {str(comparison["benchmark"])}:
        raise ValueError("primary benchmark differs from frozen plan")
    model_ids = {record.model_id for record in primary}
    if len(model_ids) != expected_models:
        raise ValueError("primary model count differs from frozen plan")
    if {record.policy for record in primary} != expected_policies:
        raise ValueError("primary policies differ from frozen plan")
    if {record.budget_condition for record in primary} != {"transition"}:
        raise ValueError("primary analysis must use transition budget only")

    cells: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for record in primary:
        cells[(record.task_id, record.model_id, record.policy)].append(record.repetition)
    for key, repetitions in cells.items():
        if sorted(repetitions) != list(range(1, expected_repetitions + 1)):
            raise ValueError(f"primary repetitions differ from frozen plan for {key}: {repetitions}")

    task_ids = {record.task_id for record in primary}
    denominator = expected_models * len(expected_policies) * expected_repetitions
    if expected_runs % denominator != 0:
        raise ValueError("frozen primary run count is not divisible by its design factors")
    expected_tasks = expected_runs // denominator
    if len(task_ids) != expected_tasks:
        raise ValueError(
            f"primary task count differs from frozen design: {len(task_ids)} != {expected_tasks}"
        )
    expected_cells = expected_tasks * expected_models * len(expected_policies)
    if len(cells) != expected_cells:
        raise ValueError("primary task/model/policy cells are incomplete")


def primary_metric_summary(
    records: list[RunRecord],
    metric: str,
    *,
    resamples: int,
    seed: int,
    noninferiority_margin: float,
    is_guardrail: bool,
) -> dict[str, Any]:
    primary = [record for record in records if record.primary_analysis]
    if any(record.benchmark != "hotpotqa" for record in primary):
        raise ValueError("primary analysis contains non-HotpotQA runs")
    if {record.policy for record in primary} != {"legacy_static", "proposed"}:
        raise ValueError("primary analysis policies differ from frozen pair")
    if {record.budget_condition for record in primary} != {"transition"}:
        raise ValueError("primary analysis must use transition budget only")

    aggregated = aggregate_repetitions(primary, lambda r: _outcome_value(r, metric))
    task_ids = sorted({record.task_id for record in primary})
    model_ids = list(dict.fromkeys(record.model_id for record in primary))

    legacy = {
        (task_id, model_id): aggregated[(task_id, model_id, "legacy_static")]
        for task_id in task_ids
        for model_id in model_ids
    }
    proposed = {
        (task_id, model_id): aggregated[(task_id, model_id, "proposed")]
        for task_id in task_ids
        for model_id in model_ids
    }
    legacy_raw = _policy_raw(task_ids, model_ids, legacy)
    proposed_raw = _policy_raw(task_ids, model_ids, proposed)
    bootstrap = paired_bootstrap(
        task_ids,
        model_ids,
        legacy,
        proposed,
        resamples=resamples,
        seed=seed,
    )

    per_model: dict[str, Any] = {}
    for model_id in model_ids:
        lval = legacy_raw["per_model"][model_id]
        pval = proposed_raw["per_model"][model_id]
        delta = bootstrap["per_model_delta"][model_id]
        per_model[model_id] = {
            "legacy": lval,
            "rrr": pval,
            "delta": pval - lval,
            "relative_difference": ((pval - lval) / lval if lval != 0 else None),
            "delta_ci95": delta["ci95"],
        }

    lmacro = legacy_raw["macro"]
    pmacro = proposed_raw["macro"]
    macro_ci = bootstrap["macro_delta"]["ci95"]
    return {
        "metric": metric,
        "task_count": len(task_ids),
        "model_count": len(model_ids),
        "repetitions_aggregated_within_task": True,
        "per_model": per_model,
        "macro": {
            "legacy": lmacro,
            "rrr": pmacro,
            "delta": pmacro - lmacro,
            "relative_difference": ((pmacro - lmacro) / lmacro if lmacro != 0 else None),
            "delta_ci95": macro_ci,
            "guardrail_decision": (
                {
                    "noninferiority_margin": noninferiority_margin,
                    "noninferior": bool(macro_ci[0] > -noninferiority_margin),
                    "superior": bool(macro_ci[0] > 0.0),
                }
                if is_guardrail
                else None
            ),
        },
    }


def primary_resource_summary(records: list[RunRecord]) -> dict[str, Any]:
    primary = [record for record in records if record.primary_analysis]
    metrics = (
        "total_consumed_tokens",
        "unused_capacity",
        "provider_calls",
        "downstream_stage_completion",
        "structural_shortfall",
        "reservation_shortfall",
    )
    output: dict[str, Any] = {}
    for metric in metrics:
        aggregated = aggregate_repetitions(primary, lambda r, m=metric: _resource_values(r)[m])
        task_ids = sorted({record.task_id for record in primary})
        model_ids = list(dict.fromkeys(record.model_id for record in primary))
        per_model: dict[str, Any] = {}
        for model_id in model_ids:
            legacy = _mean(
                aggregated[(task_id, model_id, "legacy_static")] for task_id in task_ids
            )
            proposed = _mean(
                aggregated[(task_id, model_id, "proposed")] for task_id in task_ids
            )
            per_model[model_id] = {
                "legacy": legacy,
                "rrr": proposed,
                "delta": proposed - legacy,
            }
        output[metric] = {
            "per_model": per_model,
            "macro": {
                "legacy": _mean(row["legacy"] for row in per_model.values()),
                "rrr": _mean(row["rrr"] for row in per_model.values()),
            },
        }
        output[metric]["macro"]["delta"] = (
            output[metric]["macro"]["rrr"] - output[metric]["macro"]["legacy"]
        )
    return output


def descriptive_quality(records: list[RunRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        groups[
            (
                record.benchmark,
                record.model_id,
                record.policy,
                record.budget_condition,
            )
        ].append(record)

    for key in sorted(groups):
        benchmark, model_id, policy, condition = key
        group = groups[key]
        metrics = PRIMARY_METRICS if benchmark == "hotpotqa" else QUIXBUGS_METRICS
        for metric in metrics:
            aggregated = aggregate_repetitions(group, lambda r, m=metric: _outcome_value(r, m))
            rows.append(
                {
                    "benchmark": benchmark,
                    "model_id": model_id,
                    "policy": policy,
                    "budget_condition": condition,
                    "metric": metric,
                    "task_count": len({r.task_id for r in group}),
                    "value": _mean(aggregated.values()),
                }
            )
    return rows


def analyze(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    if plan["analysis_plan_id"] != "confirmatory-analysis-v1":
        raise ValueError("unexpected analysis plan")
    records = load_completed_runs(manifest, events)
    validate_primary_shape(records, manifest, plan)
    quality = plan["quality_guardrail"]
    resamples = int(quality["bootstrap_resamples"])
    seed = int(quality["bootstrap_seed"])
    margin = float(quality["noninferiority_margin_absolute"])

    return {
        "schema_version": "1.0.0",
        "analysis_plan_id": plan["analysis_plan_id"],
        "completed_runs": len(records),
        "primary_quality": {
            metric: primary_metric_summary(
                records,
                metric,
                resamples=resamples,
                seed=seed,
                noninferiority_margin=margin,
                is_guardrail=(metric == str(quality["primary_metric"])),
            )
            for metric in PRIMARY_METRICS
        },
        "primary_resources": primary_resource_summary(records),
        "descriptive_quality": descriptive_quality(records),
        "analysis_semantics": {
            "completed_attempt_rule": (
                "For each CONFIRMATORY_RUN_END, use the event segment beginning at "
                "the most recent INITIAL_BUDGET for that run_id. Incomplete resumed "
                "prefixes are retained in raw telemetry but excluded from analysis."
            ),
            "repetition_rule": "aggregate repetitions within task/model/policy before task-level inference",
            "primary_bootstrap": (
                "resample HotpotQA task IDs with replacement; preserve cross-model pairing; "
                "compute per-model effects then macro-average across the three models"
            ),
            "bootstrap_quantile_method": "numpy.quantile method=linear at 0.025 and 0.975",
            "downstream_stage_completion": {
                "hotpotqa": "verifier stage present in completed attempt result",
                "quixbugs": "critic stage present in completed attempt result",
            },
            "shortfall_rate": "fraction of runs with at least one corresponding shortfall event in completed attempt",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen confirmatory-v1 analyzer")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/stage_aware/confirmatory_manifest_v1.json"),
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("research/stage_aware/confirmatory_analysis_plan_v1.yaml"),
    )
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    manifest_canonical = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest().upper()
    plan_canonical = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest().upper()
    events_bytes = args.events.read_bytes()
    events_sha256 = hashlib.sha256(events_bytes).hexdigest().upper()
    if manifest_canonical != EXPECTED_MANIFEST_CANONICAL_SHA256:
        raise ValueError("manifest canonical SHA-256 differs from frozen input")
    if plan_canonical != EXPECTED_ANALYSIS_PLAN_CANONICAL_SHA256:
        raise ValueError("analysis plan canonical SHA-256 differs from frozen input")
    if events_sha256 != EXPECTED_RAW_EVENTS_SHA256:
        raise ValueError("raw events SHA-256 differs from frozen input")
    events = [
        json.loads(line)
        for line in events_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    summary = analyze(manifest, plan, events)
    summary["source_artifacts"] = {
        "manifest_canonical_sha256": manifest_canonical,
        "analysis_plan_canonical_sha256": plan_canonical,
        "raw_events_sha256": events_sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"analysis complete: {summary['completed_runs']} runs")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
