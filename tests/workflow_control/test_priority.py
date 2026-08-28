from workflow_control.controller import BudgetController
from workflow_control.routes import Sequence, Stage, optional_stages, topological_depths
from workflow_control.specs import CODE_ROUTE, HOTPOT_ROUTE
from workflow_control.types import Policy, PromptEstimate, StageSpec


def test_frozen_route_priority_metadata() -> None:
    assert topological_depths(CODE_ROUTE) == {
        "planner": 0,
        "executor_1": 1,
        "executor_2": 2,
        "critic": 3,
    }
    assert optional_stages(CODE_ROUTE) == {"executor_2"}
    assert topological_depths(HOTPOT_ROUTE) == {
        "plan": 0,
        "answer": 1,
        "verifier": 2,
        "revise": 3,
        "terminal_verifier": 4,
    }
    assert optional_stages(HOTPOT_ROUTE) == {"revise", "terminal_verifier"}


def test_best_effort_priority_uses_topological_depth_before_stage_id() -> None:
    route = Sequence((Stage("current"), Stage("zeta"), Stage("alpha")))
    specs = {
        stage: StageSpec(stage, 10, 10, 10, 10)
        for stage in ("current", "zeta", "alpha")
    }
    prompts = {
        stage: PromptEstimate(stage, 0, "test")
        for stage in specs
    }
    controller = BudgetController(
        task_id="priority-test",
        model_id="test-model",
        policy=Policy.PROPOSED,
        b0=25,
        route=route,
        stage_specs=specs,
        prompt_estimates=prompts,
    )
    allocation = controller.prepare_call("current:1", "current", 0)
    assert allocation.admitted
    assert allocation.shortfall_cause == "reservation_shortfall_best_effort"
    assert allocation.selected_max_output == 10
    assert controller.state.reservations["minimum:zeta"].amount == 10
    assert controller.state.reservations["minimum:alpha"].amount == 5
