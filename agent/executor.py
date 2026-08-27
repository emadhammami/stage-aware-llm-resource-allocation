from __future__ import annotations

from agent.budget import BudgetManager
from agent.models import LLMClient
from agent.prompts import executor_prompt, retry_prompt
from agent.state import ExecutorOutput, RepairState, ValidationResult


def clean_code(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("python"):
            stripped = stripped[len("python") :].lstrip()
    return stripped.strip()


def run_executor(
    state: RepairState,
    llm: LLMClient,
    budget: BudgetManager,
    prior_evidence: ValidationResult | None = None,
    generation_budget: int = 768,
) -> None:
    hypothesis = state.planner.hypothesis if state.planner else "Unknown defect."
    if prior_evidence is None:
        prompt = executor_prompt(state.task_id, state.original_code, hypothesis, budget.remaining)
    else:
        prior_patch = state.executor_outputs[-1].proposed_code if state.executor_outputs else ""
        prompt = retry_prompt(
            state.task_id,
            state.original_code,
            prior_patch,
            prior_evidence,
            hypothesis,
            budget.remaining,
        )
    call = llm.generate("executor", prompt, budget, generation_budget=generation_budget)
    state.llm_calls.append(call)
    if not call.admitted:
        state.early_exit = True
        state.budget_exhausted = True
        state.add_event("executor", admitted=False)
        return
    state.budget_violation = state.budget_violation or budget.violated
    state.executor_outputs.append(ExecutorOutput(proposed_code=clean_code(call.raw_output)))
    state.add_event("executor", admitted=True, attempt=len(state.executor_outputs))
