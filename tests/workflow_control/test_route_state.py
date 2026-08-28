from workflow_control.controller import BudgetController
from workflow_control.specs import CODE_ROUTE, CODE_STAGE_SPECS
from workflow_control.types import Policy, PromptEstimate


def _code_controller() -> BudgetController:
    prompts = {
        stage_id: PromptEstimate(
            stage_id=stage_id,
            predicted_prompt_tokens=10,
            provenance="test",
        )
        for stage_id in CODE_STAGE_SPECS
    }
    return BudgetController(
        task_id="route-state-test",
        model_id="test-model",
        policy=Policy.PROPOSED,
        b0=1000,
        route=CODE_ROUTE,
        stage_specs=CODE_STAGE_SPECS,
        prompt_estimates=prompts,
    )


def test_released_optional_code_branch_cannot_reenter_or_resume() -> None:
    controller = _code_controller()
    planner = controller.prepare_call("planner:1", "planner", 10)
    assert planner.admitted
    controller.finish_call("planner:1", input_tokens=10, output_tokens=planner.current_min)
    executor = controller.prepare_call("executor_1:1", "executor_1", 10)
    assert executor.admitted
    controller.finish_call("executor_1:1", input_tokens=10, output_tokens=executor.current_min)
    released = controller.release_unreachable({"planner", "executor_1", "critic"})
    assert released > 0
    assert controller.state.unreachable_stages == {"executor_2"}
    critic = controller.prepare_call("critic:1", "critic", 10)
    assert critic.admitted
    assert "executor_2" not in critic.predicted_future_prompts
    assert critic.protected_continuation == 0
    assert controller.state.reservations["minimum:executor_2"].status == "RELEASE"
    restored = _code_controller()
    restored.restore(controller.dumps())
    assert restored.state.unreachable_stages == {"executor_2"}
