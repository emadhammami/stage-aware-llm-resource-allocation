from agent.routing import evidence_gate_should_invoke_critic, evidence_gate_should_retry
from agent.state import ExecutorOutput, RepairState, ValidationResult


def state_with_validation(success: bool) -> RepairState:
    state = RepairState(
        experiment_id="x",
        task_id="gcd",
        method="evidence_gated",
        token_budget=8000,
        original_code="def gcd(a,b): pass",
    )
    state.executor_outputs.append(ExecutorOutput(proposed_code="def gcd(a,b): return a"))
    state.validations.append(ValidationResult(success=success))
    return state


def test_evidence_gate_skips_critic_on_failed_tests():
    state = state_with_validation(False)
    assert not evidence_gate_should_invoke_critic(state)
    assert evidence_gate_should_retry(state)


def test_evidence_gate_invokes_critic_on_passing_tests():
    state = state_with_validation(True)
    assert evidence_gate_should_invoke_critic(state)
    assert not evidence_gate_should_retry(state)

