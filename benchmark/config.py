from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agent.models import ModelConfig
from agent.state import MethodName


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path = "configs/experiments.yaml") -> "ExperimentConfig":
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    @property
    def model(self) -> ModelConfig:
        model = self.raw["model"]
        return ModelConfig(
            name=model["name"],
            temperature=float(model["temperature"]),
            provider=model.get("provider", "google"),
            thinking_budget=model.get("thinking_budget"),
            min_output_tokens=int(model.get("min_output_tokens", 32)),
            requests_per_minute=int(self.raw.get("provider", {}).get("requests_per_minute", 4)),
            max_transient_retries=int(self.raw.get("provider", {}).get("max_transient_retries", 6)),
        )

    @property
    def required_budgets(self) -> list[int]:
        return [int(value) for value in self.raw["budgets"]["required"]]

    @property
    def main_comparison_budget(self) -> int:
        return int(self.raw["budgets"]["main_comparison_budget"])

    @property
    def pilot_tasks(self) -> list[str]:
        return list(self.raw["pilot"]["tasks"])

    @property
    def pilot_method(self) -> MethodName:
        return self.raw["pilot"]["method"]

    @property
    def pilot_budget(self) -> int:
        return int(self.raw["pilot"]["budget"])

    def executor_attempts(self, method: MethodName) -> int:
        return int(self.raw["attempt_policy"][method]["executor_attempts"])

    def generation_budget(self, role: str) -> int:
        return int(self.raw["generation_budgets"][role])
