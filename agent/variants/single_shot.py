from __future__ import annotations

from agent.budget import BudgetManager
from agent.executor import run_executor
from agent.models import LLMClient
from agent.state import RepairState, utc_now_iso
from agent.validation import validate_candidate
from benchmark.quixbugs import QuixBugsBenchmark


def run_single_shot(state: RepairState, llm: LLMClient, benchmark: QuixBugsBenchmark) -> RepairState:
    budget = BudgetManager(state.token_budget)
    # No planner and no critic by protocol; executor sees the full buggy code once.
    state.planner = None
    run_executor(state, llm, budget, generation_budget=state.executor_generation_budget)
    if not state.early_exit:
        validate_candidate(state, benchmark)
    state.ended_at_utc = state.ended_at_utc or utc_now_iso()
    return state
