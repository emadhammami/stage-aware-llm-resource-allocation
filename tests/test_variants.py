from agent.models import ScriptedLLMClient
from benchmark.runner import run_one

GOOD_PATCH = "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n"
BAD_PATCH = "def gcd(a, b):\n    return 0\n"


def test_single_shot_uses_one_executor_attempt(fake_quixbugs):
    state = run_one(
        "gcd",
        "single_shot",
        8000,
        llm=ScriptedLLMClient([GOOD_PATCH]),
        benchmark=fake_quixbugs,
        persist=False,
    )
    assert len(state.executor_outputs) == 1
    assert state.latest_validation.success
    assert state.critic is None


def test_evidence_gated_retries_without_critic_on_failed_first_patch(fake_quixbugs):
    state = run_one(
        "gcd",
        "evidence_gated",
        8000,
        llm=ScriptedLLMClient(
            [
                "TARGET_FUNCTION: gcd\nHYPOTHESIS: recursive call uses wrong arguments",
                BAD_PATCH,
                GOOD_PATCH,
                "ACCEPT\npasses tests and preserves intent",
            ]
        ),
        benchmark=fake_quixbugs,
        persist=False,
    )
    assert state.retry_used
    assert len(state.executor_outputs) == 2
    assert state.latest_validation.success
    assert state.critic.accepted
