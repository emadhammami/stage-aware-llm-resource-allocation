from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Mapping

from agent.prompts import critic_prompt, executor_prompt, planner_prompt, retry_prompt
from agent.state import ValidationResult
from benchmark.hotpotqa import DeterministicRetriever, HotpotTask
from workflow_control.prompts import (
    hotpot_answer_prompt,
    hotpot_plan_prompt,
    hotpot_revision_prompt,
    hotpot_verify_prompt,
)
from workflow_control.routes import Exclusive, Sequence, Stage
from workflow_control.types import PromptEstimate, StageSpec

CODE_ROUTE = Sequence(
    (
        Stage("planner"),
        Stage("executor_1"),
        Exclusive((Sequence(()), Stage("executor_2"))),
        Stage("critic"),
    )
)

HOTPOT_ROUTE = Sequence(
    (
        Stage("plan"),
        Stage("answer"),
        Stage("verifier"),
        Exclusive((Sequence(()), Sequence((Stage("revise"), Stage("terminal_verifier"))))),
    )
)


CODE_STAGE_SPECS = {
    "planner": StageSpec("planner", 32, 384, 768, 384),
    "executor_1": StageSpec("executor_1", 32, 768, 1536, 768),
    "executor_2": StageSpec("executor_2", 32, 768, 1536, 768),
    "critic": StageSpec("critic", 32, 384, 768, 384),
}

HOTPOT_STAGE_SPECS = {
    "plan": StageSpec("plan", 32, 256, 512, 256),
    "answer": StageSpec("answer", 32, 512, 1024, 512),
    "verifier": StageSpec("verifier", 32, 256, 512, 256),
    "revise": StageSpec("revise", 32, 512, 1024, 512),
    "terminal_verifier": StageSpec("terminal_verifier", 32, 256, 512, 256),
}


TokenCount = Callable[[str], int]


def _estimate(stage_id: str, prompt: str, counter: TokenCount, provenance: str) -> PromptEstimate:
    return PromptEstimate(
        stage_id=stage_id,
        predicted_prompt_tokens=counter(prompt),
        provenance=provenance,
        rendered_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    )


def code_prompt_estimates(
    *,
    task_id: str,
    code: str,
    counter: TokenCount,
    provenance: str,
) -> Mapping[str, PromptEstimate]:
    hypothesis = "<bounded hypothesis: at most 96 model-native tokens>"
    patch = "<bounded corrected function: at most 512 model-native tokens>"
    evidence = ValidationResult(
        success=False,
        error_category="assertion_failure",
        failing_test_info="<bounded local validation evidence: at most 192 model-native tokens>",
        stderr="<bounded stderr: at most 96 model-native tokens>",
    )
    prompts = {
        "planner": planner_prompt(task_id, code),
        "executor_1": executor_prompt(task_id, code, hypothesis, 0),
        "executor_2": retry_prompt(task_id, code, patch, evidence, hypothesis, 0),
        "critic": critic_prompt(task_id, code, hypothesis, patch, evidence),
    }
    return {
        stage: _estimate(stage, prompt, counter, provenance) for stage, prompt in prompts.items()
    }


def hotpot_prompt_estimates(
    *,
    task: HotpotTask,
    counter: TokenCount,
    provenance: str,
    initial_documents: int = 2,
) -> Mapping[str, PromptEstimate]:
    ranked = DeterministicRetriever(task.documents).ranked(task.question)
    initial = tuple(ranked[:initial_documents])
    expanded = tuple(ranked[: initial_documents + 2])
    answer = "FINAL_ANSWER: <bounded short answer>\nEVIDENCE: <bounded evidence citation or justification>"
    prompts = {
        "plan": hotpot_plan_prompt(task),
        "answer": hotpot_answer_prompt(task, initial),
        "verifier": hotpot_verify_prompt(task, initial, answer),
        "revise": hotpot_revision_prompt(task, expanded, answer),
        "terminal_verifier": hotpot_verify_prompt(task, expanded, answer),
    }
    return {
        stage: _estimate(stage, prompt, counter, provenance) for stage, prompt in prompts.items()
    }
