from __future__ import annotations

from dataclasses import dataclass

from agent.state import TokenUsage


def estimate_tokens(text: str) -> int:
    """Small deterministic estimate used only when provider metadata is unavailable."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


@dataclass(frozen=True)
class BudgetDecision:
    admitted: bool
    remaining_budget: int
    prompt_tokens_estimate: int
    generation_budget: int
    reason: str | None = None


class BudgetManager:
    def __init__(self, total_budget: int, default_generation_budget: int = 512) -> None:
        self.total_budget = total_budget
        self.default_generation_budget = default_generation_budget
        self.used = TokenUsage()

    @property
    def remaining(self) -> int:
        return max(0, self.total_budget - self.used.total_tokens)

    def admit(self, prompt: str, generation_budget: int | None = None) -> BudgetDecision:
        requested_generation = generation_budget or self.default_generation_budget
        prompt_estimate = estimate_tokens(prompt)
        required = prompt_estimate + requested_generation
        if required > self.remaining:
            return BudgetDecision(
                admitted=False,
                remaining_budget=self.remaining,
                prompt_tokens_estimate=prompt_estimate,
                generation_budget=requested_generation,
                reason="insufficient_token_budget",
            )
        return BudgetDecision(
            admitted=True,
            remaining_budget=self.remaining,
            prompt_tokens_estimate=prompt_estimate,
            generation_budget=requested_generation,
        )

    def record(self, usage: TokenUsage) -> None:
        self.used.input_tokens += usage.input_tokens
        self.used.output_tokens += usage.output_tokens
        self.used.total_tokens += usage.total_tokens
        self.used.token_count_estimated = self.used.token_count_estimated or usage.token_count_estimated

    @property
    def violated(self) -> bool:
        return self.used.total_tokens > self.total_budget

