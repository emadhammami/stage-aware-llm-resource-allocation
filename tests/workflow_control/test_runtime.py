from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from benchmark.hotpotqa import HotpotQAAdapter
from workflow_control.controller import BudgetController
from workflow_control.runtime import run_hotpot_workflow
from workflow_control.specs import HOTPOT_ROUTE, HOTPOT_STAGE_SPECS
from workflow_control.types import ModelMetadata, Policy, PromptEstimate, ProviderUsage


class ScriptedBackend:
    model_id = "scripted"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs

    def count_prompt(self, messages):
        rendered = messages[0]["content"]
        count = max(1, len(rendered) // 20)
        return count, rendered, hashlib.sha256(rendered.encode()).hexdigest()

    def generate(self, messages, max_output_tokens):
        text = self.outputs.pop(0)
        input_tokens = self.count_prompt(messages)[0]
        output_tokens = min(max_output_tokens, max(1, len(text) // 4))
        usage = ProviderUsage(input_tokens, output_tokens, "stop", "natural", 0.0)
        metadata = ModelMetadata(
            model_id=self.model_id,
            model_revision="fixture",
            tokenizer_revision="fixture",
            transformers_version=None,
            torch_version=None,
            cuda_version=None,
            dtype="fixture",
            device="fixture",
            decoding_parameters={"do_sample": False},
            chat_template_sha256="0" * 64,
        )
        return text, usage, metadata


@pytest.mark.integration
def test_scripted_hotpot_workflow_takes_sufficient_branch_without_gold() -> None:
    root = Path(__file__).parents[2]
    task = HotpotQAAdapter(
        root / "data/hotpotqa/hotpot_dev_distractor_v1.json"
    ).get("5ade15ae5542990dbb2f7f4c")
    prompts = {
        stage: PromptEstimate(stage, 50, "fixture") for stage in HOTPOT_STAGE_SPECS
    }
    controller = BudgetController(
        task_id=task.task_id,
        model_id="scripted",
        policy=Policy.PROPOSED,
        b0=3000,
        route=HOTPOT_ROUTE,
        stage_specs=HOTPOT_STAGE_SPECS,
        prompt_estimates=prompts,
    )
    backend = ScriptedBackend(["intent one; intent two", "FINAL_ANSWER: fixture\nEVIDENCE: fixture", "VERDICT: SUFFICIENT\nREASON: supported"])
    result = run_hotpot_workflow(backend=backend, controller=controller, task=task)
    assert len(result["stages"]) == 3
    assert result["candidate_answer"] == "fixture"
    assert result["run_end"]["budget_violation_count"] == 0
    assert any(event["event_type"] == "RESERVATION_RELEASE" for event in controller.state.events)
