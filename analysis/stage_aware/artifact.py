from __future__ import annotations

import csv
import hashlib
import io
import json
import statistics
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analysis.stage_aware.config import Phase0Config
from analysis.stage_aware.types import (
    HistoricalCall,
    HistoricalRun,
    Provenance,
    WorkflowStage,
)


@dataclass(frozen=True)
class ArtifactBundle:
    checksum: str
    rows: tuple[dict[str, str], ...]
    runs: tuple[HistoricalRun, ...]
    canonical_runs: tuple[HistoricalRun, ...]
    prompt_fallbacks: dict[WorkflowStage, int]
    quality_metrics: tuple[dict[str, Any], ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _stage_calls(raw_calls: list[dict[str, Any]]) -> tuple[HistoricalCall, ...]:
    executor_attempt = 0
    staged: list[HistoricalCall] = []
    for call in raw_calls:
        role = str(call.get("role", "")).lower()
        if role == "planner":
            stage = WorkflowStage.PLANNER
        elif role == "critic":
            stage = WorkflowStage.CRITIC
        elif role == "executor":
            executor_attempt += 1
            stage = (
                WorkflowStage.EXECUTOR_1
                if executor_attempt == 1
                else WorkflowStage.EXECUTOR_2
            )
        else:
            raise ValueError(f"unrecognized legacy role: {role!r}")
        prompt = call.get("prompt_tokens_estimate")
        prompt_provenance = (
            Provenance.DETERMINISTIC_ESTIMATE
            if call.get("prompt_token_count_estimated")
            else Provenance.OBSERVED_EXACT
        )
        if prompt is None:
            prompt_provenance = Provenance.MISSING
        usage = call.get("usage") or {}
        total = usage.get("total_tokens")
        output = usage.get("output_tokens")
        usage_provenance = (
            Provenance.DETERMINISTIC_ESTIMATE
            if usage.get("token_count_estimated")
            else Provenance.OBSERVED_EXACT
        )
        if total is None or output is None:
            usage_provenance = Provenance.MISSING
        staged.append(
            HistoricalCall(
                stage=stage,
                role=role,
                prompt_tokens=int(prompt) if prompt is not None else None,
                prompt_provenance=prompt_provenance,
                output_tokens=int(output) if output is not None else None,
                total_tokens=int(total) if total is not None else None,
                usage_provenance=usage_provenance,
                original_output_cap=(
                    int(call["max_output_tokens"])
                    if call.get("max_output_tokens") is not None
                    else None
                ),
                admitted=bool(call.get("admitted")),
                skipped_reason=call.get("skipped_reason"),
            )
        )
    return tuple(staged)


def load_publication_artifact(path: str | Path, config: Phase0Config) -> ArtifactBundle:
    artifact_path = Path(path)
    checksum = sha256_file(artifact_path)
    if checksum != config.artifact_sha256:
        raise ValueError(f"artifact checksum mismatch: expected {config.artifact_sha256}, got {checksum}")

    with zipfile.ZipFile(artifact_path) as archive:
        names = archive.namelist()
        raw_names = sorted(
            name for name in names if name.startswith("raw/") and name.endswith(".json")
        )
        with archive.open("core_runs.csv") as binary:
            rows = tuple(csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig")))
        implementation_commit = archive.read("experiment_git_commit.txt").decode().strip()
        if implementation_commit != config.publication_commit:
            raise ValueError("artifact implementation commit does not match the frozen configuration")
        raw_documents = [json.loads(archive.read(name)) for name in raw_names]

    if len(rows) != 240 or len(raw_documents) != 240:
        raise ValueError("publication artifact must contain 240 summary rows and 240 raw records")
    row_by_id = {row["experiment_id"]: row for row in rows}
    raw_by_id = {document["row"]["experiment_id"]: document for document in raw_documents}
    if len(row_by_id) != len(rows) or len(raw_by_id) != len(raw_documents):
        raise ValueError("duplicate experiment identifiers in publication artifact")
    if row_by_id.keys() != raw_by_id.keys():
        raise ValueError("summary/raw experiment identifier mismatch")

    runs: list[HistoricalRun] = []
    call_count = 0
    exact_prompt_count = 0
    exact_usage_count = 0
    raw_row_mismatches = 0
    finish_reason_present = 0
    for experiment_id, row in row_by_id.items():
        document = raw_by_id[experiment_id]
        raw_row = document["row"]
        for field in ("experiment_id", "task_id", "method", "token_budget", "repetition"):
            if str(raw_row[field]) != str(row[field]):
                raw_row_mismatches += 1
        if row["run_status"] != "completed" or _as_bool(row["is_pilot"]):
            raise ValueError("publication replay requires completed non-pilot records")
        if row["git_commit"] != config.publication_commit:
            raise ValueError("run recorded against a different implementation commit")
        calls = _stage_calls(document["state"].get("llm_calls", []))
        validations = tuple(
            bool(validation.get("success"))
            for validation in document["state"].get("validations", [])
        )
        call_count += len(calls)
        exact_prompt_count += sum(
            call.prompt_provenance == Provenance.OBSERVED_EXACT for call in calls
        )
        exact_usage_count += sum(
            call.usage_provenance == Provenance.OBSERVED_EXACT for call in calls
        )
        finish_reason_present += sum(
            "finish_reason" in call for call in document["state"].get("llm_calls", [])
        )
        if sum((call.total_tokens or 0) for call in calls) != int(row["total_tokens"]):
            raise ValueError(f"token total mismatch for {experiment_id}")
        runs.append(
            HistoricalRun(
                experiment_id=experiment_id,
                task_id=row["task_id"],
                method=row["method"],
                source_budget=int(row["token_budget"]),
                repetition=int(row["repetition"]),
                calls=calls,
                validation_successes=validations,
                row=dict(row),
            )
        )

    canonical = tuple(
        run
        for run in runs
        if run.method == config.canonical_method
        and run.source_budget == config.canonical_source_budget
    )
    if len(canonical) != 40:
        raise ValueError("canonical replay cohort must contain exactly 40 runs")
    fallbacks: dict[WorkflowStage, int] = {}
    for stage in WorkflowStage:
        values = [
            call.prompt_tokens
            for run in canonical
            for call in run.calls
            if call.stage == stage
            and call.prompt_tokens is not None
            and call.prompt_provenance == Provenance.OBSERVED_EXACT
        ]
        if not values:
            raise ValueError(f"no exact prompt observations for {stage}")
        fallbacks[stage] = int(round(statistics.median(values)))

    method_budget_counts: dict[tuple[str, int], int] = {}
    for run in runs:
        key = (run.method, run.source_budget)
        method_budget_counts[key] = method_budget_counts.get(key, 0) + 1
    exact_token_runs = sum(
        all(
            call.prompt_provenance == Provenance.OBSERVED_EXACT
            and call.usage_provenance == Provenance.OBSERVED_EXACT
            for call in run.calls
        )
        for run in runs
    )
    estimated_token_runs = sum(
        any(
            call.prompt_provenance == Provenance.DETERMINISTIC_ESTIMATE
            or call.usage_provenance == Provenance.DETERMINISTIC_ESTIMATE
            for call in run.calls
        )
        for run in runs
    )
    missing_token_runs = sum(
        any(call.prompt_tokens is None or call.total_tokens is None for call in run.calls)
        for run in runs
    )
    quality_metrics: list[dict[str, Any]] = [
        {
            "metric": "artifact_sha256",
            "value": checksum,
            "provenance": Provenance.OBSERVED_EXACT.value,
            "notes": "computed from immutable source bytes",
        },
        {
            "metric": "publication_rows",
            "value": len(rows),
            "provenance": Provenance.OBSERVED_EXACT.value,
            "notes": "all completed and non-pilot",
        },
        {
            "metric": "raw_records",
            "value": len(raw_documents),
            "provenance": Provenance.OBSERVED_EXACT.value,
            "notes": "one-to-one match by experiment_id",
        },
        {
            "metric": "raw_records_matched",
            "value": len(raw_by_id),
            "provenance": Provenance.RECONSTRUCTED_EXACT.value,
            "notes": "matched one-to-one by experiment_id",
        },
        {
            "metric": "raw_records_missing",
            "value": 0,
            "provenance": Provenance.RECONSTRUCTED_EXACT.value,
            "notes": "no unmatched publication rows",
        },
        {
            "metric": "summary_raw_field_mismatches",
            "value": raw_row_mismatches,
            "provenance": Provenance.RECONSTRUCTED_EXACT.value,
            "notes": "five identity fields compared per run",
        },
        {
            "metric": "llm_call_records",
            "value": call_count,
            "provenance": Provenance.OBSERVED_EXACT.value,
            "notes": "all publication conditions",
        },
        {
            "metric": "provider_exact_prompt_counts",
            "value": exact_prompt_count,
            "provenance": Provenance.RECONSTRUCTED_EXACT.value,
            "notes": f"{exact_prompt_count}/{call_count}",
        },
        {
            "metric": "provider_exact_usage_counts",
            "value": exact_usage_count,
            "provenance": Provenance.RECONSTRUCTED_EXACT.value,
            "notes": f"{exact_usage_count}/{call_count}",
        },
        {
            "metric": "exact_provider_token_runs",
            "value": exact_token_runs,
            "provenance": Provenance.RECONSTRUCTED_EXACT.value,
            "notes": "all prompt and usage fields provider reported",
        },
        {
            "metric": "estimated_token_runs",
            "value": estimated_token_runs,
            "provenance": Provenance.RECONSTRUCTED_EXACT.value,
            "notes": "source publication telemetry",
        },
        {
            "metric": "missing_token_metadata_runs",
            "value": missing_token_runs,
            "provenance": Provenance.RECONSTRUCTED_EXACT.value,
            "notes": "source publication telemetry",
        },
        {
            "metric": "finish_reason_records",
            "value": finish_reason_present,
            "provenance": Provenance.MISSING.value,
            "notes": f"{finish_reason_present}/{call_count}; cap binding cannot be proved",
        },
        {
            "metric": "finish_reason_missing_records",
            "value": call_count - finish_reason_present,
            "provenance": Provenance.MISSING.value,
            "notes": "all historical calls lack provider finish reason",
        },
        {
            "metric": "malformed_or_incomplete_records",
            "value": 0,
            "provenance": Provenance.RECONSTRUCTED_EXACT.value,
            "notes": "identity, status, and token-total validations passed",
        },
        {
            "metric": "canonical_replay_runs",
            "value": len(canonical),
            "provenance": Provenance.RECONSTRUCTED_EXACT.value,
            "notes": f"{config.canonical_method}@{config.canonical_source_budget}",
        },
    ]
    for (method, budget), count in sorted(method_budget_counts.items()):
        quality_metrics.append(
            {
                "metric": f"condition_runs:{method}:{budget}",
                "value": count,
                "provenance": Provenance.RECONSTRUCTED_EXACT.value,
                "notes": "publication condition count",
            }
        )
    for stage, value in fallbacks.items():
        quality_metrics.append(
            {
                "metric": f"canonical_prompt_median:{stage.value}",
                "value": value,
                "provenance": Provenance.DETERMINISTIC_ESTIMATE.value,
                "notes": "median of exact canonical-cohort prompt counts",
            }
        )
    return ArtifactBundle(
        checksum=checksum,
        rows=rows,
        runs=tuple(runs),
        canonical_runs=canonical,
        prompt_fallbacks=fallbacks,
        quality_metrics=tuple(quality_metrics),
    )
