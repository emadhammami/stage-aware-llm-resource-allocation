from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

from agent.models import GeminiClient, LLMClient, ProviderInfrastructureError
from agent.state import MethodName, RepairState, utc_now_iso
from agent.variants import run_evidence_gated, run_pec, run_pevc, run_single_shot
from benchmark.config import ExperimentConfig
from benchmark.quixbugs import QuixBugsBenchmark
from benchmark.results import persist_result

VARIANTS = {
    "single_shot": run_single_shot,
    "pec": run_pec,
    "pevc": run_pevc,
    "evidence_gated": run_evidence_gated,
}


def git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout.strip() or "unknown"


def run_one(
    task_id: str,
    method: MethodName,
    budget: int,
    repetition: int = 1,
    is_pilot: bool = False,
    llm: LLMClient | None = None,
    benchmark: QuixBugsBenchmark | None = None,
    experiment_config: ExperimentConfig | None = None,
    persist: bool = True,
    results_root: str | Path = "results",
) -> RepairState:
    bench = benchmark or QuixBugsBenchmark()
    config = experiment_config or ExperimentConfig.load()
    model_config = config.model
    original_code = bench.load_buggy_code(task_id)
    state = RepairState(
        experiment_id=str(uuid.uuid4()),
        task_id=task_id,
        method=method,
        token_budget=budget,
        repetition=repetition,
        is_pilot=is_pilot,
        original_code=original_code,
        max_executor_attempts=config.executor_attempts(method),
        planner_generation_budget=config.generation_budget("planner"),
        executor_generation_budget=config.generation_budget("executor"),
        critic_generation_budget=config.generation_budget("critic"),
    )
    client = llm or GeminiClient(model_config)
    try:
        final_state = VARIANTS[method](state, client, bench)
    except ProviderInfrastructureError as exc:
        state.run_status = "infrastructure_error"
        state.infrastructure_error = str(exc)
        state.provider_attempts += exc.provider_attempts
        state.transient_retries += exc.transient_retries
        state.rate_limit_retries += exc.rate_limit_retries
        state.rate_limit_wait_seconds += exc.rate_limit_wait_seconds
        state.provider_wall_time_seconds += exc.provider_wall_time_seconds
        state.ended_at_utc = utc_now_iso()
        state.add_event("infrastructure_error", error=str(exc))
        final_state = state
    if persist:
        persist_result(
            final_state,
            git_commit=git_commit(),
            benchmark_commit=bench.benchmark_commit(),
            model=model_config.name,
            temperature=model_config.temperature,
            results_root=Path(results_root),
        )
    return final_state
