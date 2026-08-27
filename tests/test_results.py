from pathlib import Path

from agent.models import ProviderInfrastructureError
from agent.state import CriticOutput, PatchRecord, RepairState, ValidationResult
from benchmark.config import ExperimentConfig
from benchmark.matrix import execute_plan
from benchmark.results import completed_keys, persist_result, state_to_row
from benchmark.runner import run_one


def test_false_acceptance_calculation():
    state = RepairState(experiment_id="x", task_id="gcd", method="pec", token_budget=8000)
    state.critic = CriticOutput(accepted=True)
    state.validations.append(ValidationResult(success=False, tests_failed=1, tests_total=1))
    row = state_to_row(state, "git", "bench", "model", 0)
    assert row["false_accept"] is True
    assert row["candidate_correct"] is False
    assert row["workflow_success"] is False


def test_false_reject_calculation():
    state = RepairState(experiment_id="x", task_id="gcd", method="pevc", token_budget=8000)
    state.critic = CriticOutput(accepted=False)
    state.validations.append(ValidationResult(success=True, tests_passed=1, tests_total=1))
    row = state_to_row(state, "git", "bench", "model", 0)
    assert row["false_reject"] is True
    assert row["candidate_correct"] is True
    assert row["workflow_success"] is False


def test_result_serialization_and_resume_skip(tmp_path: Path):
    state = RepairState(experiment_id="x", task_id="gcd", method="single_shot", token_budget=8000)
    state.patch = PatchRecord(applied=True, syntax_valid=True, affected_function="gcd")
    state.validations.append(ValidationResult(success=True, tests_passed=1, tests_total=1))
    state.llm_calls = []
    state.ended_at_utc = "2026-08-07T00:00:00+00:00"
    persist_result(state, "git", "bench", "model", 0, results_root=tmp_path)
    assert (tmp_path / "raw" / "gcd__single_shot__8000__run1.json").exists()
    assert ("gcd", "single_shot", 8000, 1) in completed_keys(tmp_path / "runs.csv")


def test_budget_exhausted_completed_run_is_skipped(tmp_path: Path):
    state = RepairState(experiment_id="x", task_id="gcd", method="evidence_gated", token_budget=2000)
    state.early_exit = True
    state.budget_exhausted = True
    state.ended_at_utc = "2026-08-07T00:00:00+00:00"
    persist_result(state, "git", "bench", "model", 0, results_root=tmp_path)
    assert ("gcd", "evidence_gated", 2000, 1) in completed_keys(tmp_path / "runs.csv")


def test_infrastructure_error_is_not_skipped(tmp_path: Path):
    state = RepairState(experiment_id="x", task_id="gcd", method="evidence_gated", token_budget=2000)
    state.run_status = "infrastructure_error"
    state.ended_at_utc = "2026-08-07T00:00:00+00:00"
    persist_result(state, "git", "bench", "model", 0, results_root=tmp_path)
    assert ("gcd", "evidence_gated", 2000, 1) not in completed_keys(tmp_path / "runs.csv")


def test_run_one_persists_infrastructure_error(tmp_path: Path, fake_quixbugs):
    class FailingLLM:
        def generate(self, role, prompt, budget, generation_budget):
            raise ProviderInfrastructureError(
                "429 RESOURCE_EXHAUSTED",
                provider_attempts=2,
                transient_retries=1,
                rate_limit_retries=1,
                rate_limit_wait_seconds=10,
                provider_wall_time_seconds=0.5,
            )

    state = run_one(
        "gcd",
        "single_shot",
        8000,
        llm=FailingLLM(),
        benchmark=fake_quixbugs,
        results_root=tmp_path,
    )
    assert state.run_status == "infrastructure_error"
    assert state.provider_attempts == 2
    assert state.transient_retries == 1
    assert state.rate_limit_retries == 1
    assert state.rate_limit_wait_seconds == 10
    assert (tmp_path / "raw" / "gcd__single_shot__8000__run1.json").exists()
    assert ("gcd", "single_shot", 8000, 1) not in completed_keys(tmp_path / "runs.csv")


def test_matrix_continues_after_infrastructure_error(tmp_path: Path, fake_quixbugs):
    calls = []

    def fake_run_one_fn(**kwargs):
        calls.append(kwargs["task_id"])
        state = RepairState(
            experiment_id=kwargs["task_id"],
            task_id=kwargs["task_id"],
            method=kwargs["method"],
            token_budget=kwargs["budget"],
        )
        if kwargs["task_id"] == "a":
            state.run_status = "infrastructure_error"
        return state

    execute_plan(
        [("a", "single_shot", 8000), ("b", "single_shot", 8000)],
        repetition=1,
        is_pilot=True,
        force=False,
        benchmark=fake_quixbugs,
        config=ExperimentConfig.load(),
        completed_csv=str(tmp_path / "runs.csv"),
        run_one_fn=fake_run_one_fn,
    )
    assert calls == ["a", "b"]


def test_resume_skips_completed_but_retries_infrastructure_error(tmp_path: Path, fake_quixbugs):
    completed = RepairState(experiment_id="a", task_id="a", method="single_shot", token_budget=8000)
    completed.ended_at_utc = "2026-08-07T00:00:00+00:00"
    infra = RepairState(experiment_id="b", task_id="b", method="single_shot", token_budget=8000)
    infra.run_status = "infrastructure_error"
    infra.ended_at_utc = "2026-08-07T00:00:00+00:00"
    persist_result(completed, "git", "bench", "model", 0, results_root=tmp_path)
    persist_result(infra, "git", "bench", "model", 0, results_root=tmp_path)

    calls = []

    def fake_run_one_fn(**kwargs):
        calls.append(kwargs["task_id"])
        return None

    execute_plan(
        [("a", "single_shot", 8000), ("b", "single_shot", 8000)],
        repetition=1,
        is_pilot=True,
        force=False,
        benchmark=fake_quixbugs,
        config=ExperimentConfig.load(),
        completed_csv=str(tmp_path / "runs.csv"),
        run_one_fn=fake_run_one_fn,
    )
    assert calls == ["b"]
