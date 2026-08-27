from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from agent.state import RepairState
from benchmark.quixbugs import runtime_fingerprint

CSV_FIELDS = [
    "experiment_id",
    "timestamp_utc",
    "git_commit",
    "benchmark_commit",
    "task_id",
    "method",
    "token_budget",
    "repetition",
    "model",
    "temperature",
    "is_pilot",
    "run_status",
    "candidate_correct",
    "workflow_success",
    "tests_passed",
    "tests_failed",
    "tests_total",
    "critic_invoked",
    "critic_accepted",
    "false_accept",
    "false_reject",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "token_count_estimated",
    "llm_calls",
    "provider_attempts",
    "transient_retries",
    "rate_limit_retries",
    "rate_limit_wait_seconds",
    "provider_wall_time_seconds",
    "planner_calls",
    "executor_attempts",
    "critic_calls",
    "validation_attempts",
    "validation_failures",
    "retry_used",
    "early_exit",
    "budget_exhausted",
    "budget_violation",
    "budget_limit",
    "budget_used",
    "budget_remaining",
    "patch_applied",
    "syntax_valid",
    "final_error_category",
    "runtime_seconds",
    "llm_runtime_seconds",
    "validation_runtime_seconds",
    "infrastructure_error",
]


def completed_keys(csv_path: str | Path = "results/runs.csv") -> set[tuple[str, str, int, int]]:
    path = Path(csv_path)
    if not path.exists():
        return set()
    keys: set[tuple[str, str, int, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("run_status", "completed") != "completed":
                continue
            keys.add(
                (
                    row["task_id"],
                    row["method"],
                    int(row["token_budget"]),
                    int(row["repetition"]),
                )
            )
    return keys


def state_to_row(
    state: RepairState,
    git_commit: str,
    benchmark_commit: str,
    model: str,
    temperature: float,
) -> dict[str, Any]:
    usage = state.tokens_used
    latest = state.latest_validation
    candidate_correct = bool(latest and latest.success)
    critic_invoked = state.critic is not None
    critic_accepted = state.critic.accepted if state.critic else None
    if state.method == "single_shot":
        workflow_success = candidate_correct
        false_accept = None
        false_reject = None
    else:
        workflow_success = bool(candidate_correct and critic_accepted)
        false_accept = bool(critic_accepted and not candidate_correct)
        false_reject = bool(critic_accepted is False and candidate_correct)
    runtime = 0.0
    if state.ended_at_utc and state.started_at_utc:
        runtime = max(0.0, sum(v.runtime_seconds for v in state.validations) + sum(c.runtime_seconds for c in state.llm_calls))
    return {
        "experiment_id": state.experiment_id,
        "timestamp_utc": state.ended_at_utc,
        "git_commit": git_commit,
        "benchmark_commit": benchmark_commit,
        "task_id": state.task_id,
        "method": state.method,
        "token_budget": state.token_budget,
        "repetition": state.repetition,
        "model": model,
        "temperature": temperature,
        "is_pilot": state.is_pilot,
        "run_status": state.run_status,
        "candidate_correct": candidate_correct,
        "workflow_success": workflow_success,
        "tests_passed": latest.tests_passed if latest else 0,
        "tests_failed": latest.tests_failed if latest else 0,
        "tests_total": latest.tests_total if latest else 0,
        "critic_invoked": critic_invoked,
        "critic_accepted": critic_accepted,
        "false_accept": false_accept,
        "false_reject": false_reject,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "token_count_estimated": usage.token_count_estimated,
        "llm_calls": len([c for c in state.llm_calls if c.admitted]),
        "provider_attempts": state.provider_attempts + sum(c.provider_attempts for c in state.llm_calls),
        "transient_retries": state.transient_retries + sum(c.transient_retries for c in state.llm_calls),
        "rate_limit_retries": state.rate_limit_retries + sum(c.rate_limit_retries for c in state.llm_calls),
        "rate_limit_wait_seconds": state.rate_limit_wait_seconds
        + sum(c.rate_limit_wait_seconds for c in state.llm_calls),
        "provider_wall_time_seconds": state.provider_wall_time_seconds
        + sum(c.provider_wall_time_seconds for c in state.llm_calls),
        "planner_calls": len([c for c in state.llm_calls if c.role == "planner" and c.admitted]),
        "executor_attempts": len(state.executor_outputs),
        "critic_calls": len([c for c in state.llm_calls if c.role == "critic" and c.admitted]),
        "validation_attempts": len(state.validations),
        "validation_failures": len([v for v in state.validations if not v.success]),
        "retry_used": state.retry_used,
        "early_exit": state.early_exit,
        "budget_exhausted": state.budget_exhausted,
        "budget_violation": state.budget_violation,
        "budget_limit": state.token_budget,
        "budget_used": usage.total_tokens,
        "budget_remaining": max(0, state.token_budget - usage.total_tokens),
        "patch_applied": state.patch.applied if state.patch else False,
        "syntax_valid": state.patch.syntax_valid if state.patch else False,
        "final_error_category": state.final_error_category,
        "runtime_seconds": runtime,
        "llm_runtime_seconds": sum(c.runtime_seconds for c in state.llm_calls),
        "validation_runtime_seconds": sum(v.runtime_seconds for v in state.validations),
        "infrastructure_error": state.infrastructure_error,
    }


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=path.parent) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def append_csv_row(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def persist_result(
    state: RepairState,
    git_commit: str,
    benchmark_commit: str,
    model: str,
    temperature: float,
    results_root: str | Path = "results",
) -> Path:
    row = state_to_row(state, git_commit, benchmark_commit, model, temperature)
    root = Path(results_root)
    raw_name = f"{state.task_id}__{state.method}__{state.token_budget}__run{state.repetition}.json"
    raw_path = root / "raw" / raw_name
    atomic_write_json(
        raw_path,
        {
            "row": row,
            "state": state.model_dump(),
            "runtime_fingerprint": runtime_fingerprint(),
        },
    )
    append_csv_row(root / "runs.csv", row)
    return raw_path
