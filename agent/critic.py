from __future__ import annotations

from agent.budget import BudgetManager
from agent.models import LLMClient
from agent.prompts import critic_prompt
from agent.state import CriticOutput, RepairState, ValidationResult


def parse_critic_output(text: str) -> CriticOutput:
    first_line = (text.strip().splitlines() or [""])[0].strip().upper()
    accepted = first_line.startswith("ACCEPT")
    return CriticOutput(accepted=accepted, rationale=text.strip())


def run_critic(
    state: RepairState,
    llm: LLMClient,
    budget: BudgetManager,
    evidence: ValidationResult | None,
    generation_budget: int = 384,
) -> None:
    hypothesis = state.planner.hypothesis if state.planner else "Unknown defect."
    patch = state.executor_outputs[-1].proposed_code if state.executor_outputs else ""
    prompt = critic_prompt(state.task_id, state.original_code, hypothesis, patch, evidence)
    call = llm.generate("critic", prompt, budget, generation_budget=generation_budget)
    state.llm_calls.append(call)
    if not call.admitted:
        state.early_exit = True
        state.budget_exhausted = True
        state.add_event("critic", admitted=False)
        return
    state.budget_violation = state.budget_violation or budget.violated
    state.critic = parse_critic_output(call.raw_output)
    state.add_event("critic", admitted=True, accepted=state.critic.accepted)
