from agent.prompts import critic_prompt, executor_prompt, planner_prompt, retry_prompt
from agent.state import ValidationResult


def test_executor_prompts_require_complete_function_definition():
    prompt = executor_prompt("gcd", "def gcd(a, b):\n    return a\n", "wrong recursion", 100)
    retry = retry_prompt(
        "gcd",
        "def gcd(a, b):\n    return a\n",
        "def gcd(a, b):\n    return 0\n",
        ValidationResult(success=False, failing_test_info="E assert 0 == 3"),
        "wrong recursion",
        100,
    )
    for text in [prompt, retry]:
        assert "Return exactly one complete corrected Python function definition." in text
        assert "Preserve the target function name and a compatible signature." in text
        assert "function body or full function" not in text
        assert "markdown fences or explanation" in text


def test_llm_prompts_do_not_name_benchmark():
    prompts = [
        planner_prompt("gcd", "def gcd(a, b):\n    return a\n"),
        executor_prompt("gcd", "def gcd(a, b):\n    return a\n", "wrong recursion", 100),
        retry_prompt(
            "gcd",
            "def gcd(a, b):\n    return a\n",
            "def gcd(a, b):\n    return 0\n",
            ValidationResult(success=False, failing_test_info="E assert 0 == 3"),
            "wrong recursion",
            100,
        ),
        critic_prompt(
            "gcd",
            "def gcd(a, b):\n    return a\n",
            "wrong recursion",
            "def gcd(a, b):\n    return 0\n",
            None,
        ),
    ]
    assert all("QuixBugs" not in prompt for prompt in prompts)
    assert "memorized benchmark answers" in prompts[0]
