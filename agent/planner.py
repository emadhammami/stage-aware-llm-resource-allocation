from __future__ import annotations

import re

from agent.budget import BudgetManager
from agent.models import LLMClient
from agent.prompts import planner_prompt
from agent.state import PlannerOutput, RepairState


def parse_planner_output(text: str) -> PlannerOutput:
    target = None
    target_match = re.search(r"TARGET_FUNCTION:\s*([A-Za-z_][A-Za-z0-9_]*)", text)
    if target_match:
        target = target_match.group(1)
    hypothesis_match = re.search(r"HYPOTHESIS:\s*(.*)", text, flags=re.DOTALL)
    hypothesis = hypothesis_match.group(1).strip() if hypothesis_match else text.strip()
    return PlannerOutput(hypothesis=hypothesis or "Unknown defect.", target_function=target)


def run_planner(
    state: RepairState,
    llm: LLMClient,
    budget: BudgetManager,
    generation_budget: int = 384,
) -> None:
    prompt = planner_prompt(state.task_id, state.original_code)
    call = llm.generate("planner", prompt, budget, generation_budget=generation_budget)
    state.llm_calls.append(call)
    if not call.admitted:
        state.early_exit = True
        state.budget_exhausted = True
        state.add_event("planner", admitted=False)
        return
    state.budget_violation = state.budget_violation or budget.violated
    state.planner = parse_planner_output(call.raw_output)
    state.add_event("planner", admitted=True, target_function=state.planner.target_function)
