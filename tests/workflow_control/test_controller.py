from __future__ import annotations

import json

import pytest

from workflow_control.controller import BudgetController
from workflow_control.routes import Sequence, Stage
from workflow_control.telemetry import AppendOnlyTelemetry
from workflow_control.types import Policy, PromptEstimate, StageSpec

ROUTE = Sequence((Stage("plan"), Stage("execute"), Stage("verify")))
SPECS = {
    stage: StageSpec(stage, 10, 30, 60, 30) for stage in ("plan", "execute", "verify")
}
PROMPTS = {
    "plan": PromptEstimate("plan", 20, "fixture"),
    "execute": PromptEstimate("execute", 30, "fixture"),
    "verify": PromptEstimate("verify", 20, "fixture"),
}


def _controller(policy: Policy = Policy.PROPOSED, b0: int = 240) -> BudgetController:
    return BudgetController(
        task_id="task",
        model_id="model",
        policy=policy,
        b0=b0,
        route=ROUTE,
        stage_specs=SPECS,
        prompt_estimates=PROMPTS,
    )


def test_exact_prompt_replaces_prediction_resize_release_return_and_reallocation() -> None:
    controller = _controller()
    allocation = controller.prepare_call("c1", "plan", 22)
    assert allocation.admitted
    assert allocation.exact_current_prompt == 22
    assert allocation.protected_continuation == 70
    assert controller.materialize_future_prompt("execute", 36) == 6
    assert controller.release_unreachable({"plan", "execute"}) == 30
    controller.finish_call("c1", input_tokens=22, output_tokens=8)
    next_allocation = controller.prepare_call("c2", "execute", 36)
    assert next_allocation.admitted
    assert next_allocation.reallocation_sources
    actions = [event["event_type"] for event in controller.state.events]
    assert "RESERVATION_RESIZE" in actions
    assert "RESERVATION_RELEASE" in actions
    assert "CAPACITY_RETURN" in actions
    assert "CAPACITY_REALLOCATE" in actions


def test_no_precharge_and_hard_conservation() -> None:
    controller = _controller()
    allocation = controller.prepare_call("c1", "plan", 20)
    assert controller.state.consumed == 0
    controller.finish_call("c1", input_tokens=20, output_tokens=allocation.selected_max_output)
    assert controller.state.consumed <= controller.state.b0
    assert controller.remaining >= 0


def test_fixed_policy_protects_frozen_quarter_not_prompt_aware_minimum() -> None:
    controller = _controller(policy=Policy.FIXED_RESERVATION)
    allocation = controller.prepare_call("c1", "plan", 20)
    assert allocation.protected_continuation == 60


def test_shortfall_is_detected_before_invocation() -> None:
    controller = _controller(b0=30)
    allocation = controller.prepare_call("c1", "plan", 25)
    assert not allocation.admitted
    assert allocation.structural_shortfall
    assert controller.state.consumed == 0
    with pytest.raises(ValueError, match="no admitted allocation"):
        controller.finish_call("c1", input_tokens=25, output_tokens=0)


def test_resume_and_duplicate_debit_protection() -> None:
    first = _controller()
    first.prepare_call("c1", "plan", 20)
    first.finish_call("c1", input_tokens=20, output_tokens=5)
    restored = _controller()
    restored.restore(first.dumps())
    assert restored.state.consumed == 25
    with pytest.raises(ValueError, match="duplicate provider debit"):
        restored.finish_call("c1", input_tokens=20, output_tokens=5)


def test_append_only_telemetry_resume_does_not_duplicate(tmp_path) -> None:
    telemetry = AppendOnlyTelemetry(tmp_path / "events.jsonl")
    events = [
        {"event_index": 0, "event_type": "A"},
        {"event_index": 1, "event_type": "B"},
    ]
    assert telemetry.append_new(events) == 2
    assert telemetry.append_new(events) == 0
    assert len(telemetry.read()) == 2
    assert json.loads((tmp_path / "events.jsonl").read_text().splitlines()[0]) == events[0]


@pytest.mark.parametrize("policy", list(Policy))
def test_all_policies_respect_same_supplied_b0(policy: Policy) -> None:
    assert _controller(policy=policy).state.b0 == 240
