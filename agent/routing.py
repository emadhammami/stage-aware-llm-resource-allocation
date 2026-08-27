from __future__ import annotations

from agent.state import RepairState


def evidence_gate_should_invoke_critic(state: RepairState) -> bool:
    validation = state.latest_validation
    return bool(validation and validation.success)


def evidence_gate_should_retry(state: RepairState, max_executor_attempts: int = 2) -> bool:
    validation = state.latest_validation
    if validation is None or validation.success:
        return False
    return (
        len(state.executor_outputs) < max_executor_attempts
        and not state.budget_exhausted
        and not state.budget_violation
    )
