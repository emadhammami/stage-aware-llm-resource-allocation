from __future__ import annotations

from agent.budget import BudgetManager
from agent.critic import run_critic
from agent.executor import run_executor
from agent.models import LLMClient
from agent.planner import run_planner
from agent.routing import evidence_gate_should_invoke_critic, evidence_gate_should_retry
from agent.state import RepairState, utc_now_iso
from agent.validation import validate_candidate
from benchmark.quixbugs import QuixBugsBenchmark


def run_evidence_gated(state: RepairState, llm: LLMClient, benchmark: QuixBugsBenchmark) -> RepairState:
    budget = BudgetManager(state.token_budget)
    run_planner(state, llm, budget, generation_budget=state.planner_generation_budget)
    if state.early_exit:
        state.ended_at_utc = utc_now_iso()
        return state

    run_executor(state, llm, budget, generation_budget=state.executor_generation_budget)
    if not state.early_exit:
        validate_candidate(state, benchmark)

    if not state.early_exit and evidence_gate_should_retry(state, state.max_executor_attempts):
        state.retry_used = True
        run_executor(
            state,
            llm,
            budget,
            prior_evidence=state.latest_validation,
            generation_budget=state.executor_generation_budget,
        )
        if not state.early_exit:
            validate_candidate(state, benchmark)

    if not state.early_exit and evidence_gate_should_invoke_critic(state):
        run_critic(
            state,
            llm,
            budget,
            evidence=state.latest_validation,
            generation_budget=state.critic_generation_budget,
        )
    elif not state.latest_validation or not state.latest_validation.success:
        state.add_event("evidence_gate", critic_skipped=True, reason="tests_failed")

    state.ended_at_utc = utc_now_iso()
    return state
