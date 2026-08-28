from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from benchmark.hotpotqa import DeterministicRetriever, HotpotQAAdapter
from benchmark.hotpotqa_eval import exact_match_score, normalize_answer, token_f1_score
from workflow_control.backends import _map_finish_reason
from workflow_control.controller import BudgetController
from workflow_control.parsing import parse_final_answer, parse_verdict
from workflow_control.runtime import run_hotpot_workflow
from workflow_control.specs import HOTPOT_ROUTE, HOTPOT_STAGE_SPECS
from workflow_control.types import ModelMetadata, Policy, PromptEstimate, ProviderUsage

ROOT = Path(__file__).parents[2]
TASK_ID = "5ade15ae5542990dbb2f7f4c"


class ScriptedBackend:
    model_id = "scripted"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)

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


def _task():
    return HotpotQAAdapter(ROOT / "data/hotpotqa/hotpot_dev_distractor_v1.json").get(TASK_ID)


def _controller(task) -> BudgetController:
    prompts = {
        stage: PromptEstimate(stage, 50, "fixture") for stage in HOTPOT_STAGE_SPECS
    }
    return BudgetController(
        task_id=task.task_id,
        model_id="scripted",
        policy=Policy.PROPOSED,
        b0=5000,
        route=HOTPOT_ROUTE,
        stage_specs=HOTPOT_STAGE_SPECS,
        prompt_estimates=prompts,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("FINAL_ANSWER: October\nEVIDENCE: supported", "October"),
        ("final_answer: no\nevidence: supported", "no"),
        ("ANSWER: October", None),
        ("FINAL_ANSWER: October", None),
        ("FINAL_ANSWER: October\nEVIDENCE:", None),
        ("FINAL_ANSWER: October\nEVIDENCE: x\nextra", None),
    ],
)
def test_parse_final_answer_is_strict(text: str, expected: str | None) -> None:
    assert parse_final_answer(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("VERDICT: SUFFICIENT\nREASON: supported", "SUFFICIENT"),
        ("verdict: insufficient\nreason: missing evidence", "INSUFFICIENT"),
        ("SUFFICIENT", None),
        ("VERDICT: SUFFICIENT", None),
        ("VERDICT: SUFFICIENT\nREASON:", None),
        ("VERDICT: MAYBE\nREASON: uncertain", None),
    ],
)
def test_parse_verdict_is_strict(text: str, expected: str | None) -> None:
    assert parse_verdict(text) == expected


def test_hotpot_official_style_answer_metrics() -> None:
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"
    assert exact_match_score("The Eiffel Tower", "Eiffel Tower") == 1.0
    assert token_f1_score("Eiffel Tower Paris", "Eiffel Tower") == pytest.approx(0.8)
    assert token_f1_score("yes", "no") == 0.0


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("stop", "natural"),
        ("eos", "natural"),
        ("end_turn", "natural"),
        ("length", "length"),
        ("max_tokens", "length"),
        ("safety", "safety"),
        ("provider_error", "error"),
        (None, "unknown"),
    ],
)
def test_finish_reason_mapping(reason: str | None, expected: str) -> None:
    assert _map_finish_reason(reason) == expected


@pytest.mark.integration
def test_retriever_recovery_returns_two_unseen_documents() -> None:
    task = _task()
    retriever = DeterministicRetriever(task.documents)
    ranked = retriever.ranked(task.question)
    seen = {ranked[0].title, ranked[1].title}
    recovered = retriever.next_unseen_many(task.question, seen, limit=2)
    assert len(recovered) == 2
    assert all(document.title not in seen for document in recovered)


@pytest.mark.integration
def test_hotpot_sufficient_branch_uses_structured_contract() -> None:
    task = _task()
    backend = ScriptedBackend(
        [
            "intent one; intent two",
            "FINAL_ANSWER: October\nEVIDENCE: fixture",
            "VERDICT: SUFFICIENT\nREASON: supported",
        ]
    )
    result = run_hotpot_workflow(backend=backend, controller=_controller(task), task=task)
    assert result["candidate_answer"] == "October"
    assert result["candidate_answer_raw"].startswith("FINAL_ANSWER:")
    assert result["answer_parse_ok"] is True
    assert result["final_verification_parse_ok"] is True
    assert result["final_verification_verdict"] == "SUFFICIENT"
    assert result["recovery_documents"] == 0
    assert len(result["stages"]) == 3


@pytest.mark.integration
def test_hotpot_insufficient_branch_recovers_two_documents() -> None:
    task = _task()
    backend = ScriptedBackend(
        [
            "intent one; intent two",
            "FINAL_ANSWER: first\nEVIDENCE: initial",
            "VERDICT: INSUFFICIENT\nREASON: more evidence required",
            "FINAL_ANSWER: revised\nEVIDENCE: expanded",
            "VERDICT: SUFFICIENT\nREASON: now supported",
        ]
    )
    result = run_hotpot_workflow(backend=backend, controller=_controller(task), task=task)
    assert result["candidate_answer"] == "revised"
    assert result["initial_verifier_verdict"] == "INSUFFICIENT"
    assert result["final_verification_stage"] == "terminal_verifier"
    assert result["final_verification_verdict"] == "SUFFICIENT"
    assert result["recovery_documents"] == 2
    assert len(result["stages"]) == 5


@pytest.mark.integration
def test_malformed_answer_is_scientific_invalid_response() -> None:
    task = _task()
    backend = ScriptedBackend(["intent one; intent two", "ANSWER: October"])
    result = run_hotpot_workflow(backend=backend, controller=_controller(task), task=task)
    assert result["terminal_status"] == "invalid_response"
    assert result["classification"] == "scientific"
    assert result["parse_failure_stage"] == "answer"
    assert result["candidate_answer"] is None
    assert result["answer_parse_ok"] is False
    assert result["run_end"]["normal_completion"] is False


@pytest.mark.integration
def test_malformed_verifier_is_scientific_invalid_response() -> None:
    task = _task()
    backend = ScriptedBackend(
        [
            "intent one; intent two",
            "FINAL_ANSWER: October\nEVIDENCE: fixture",
            "SUFFICIENT",
        ]
    )
    result = run_hotpot_workflow(backend=backend, controller=_controller(task), task=task)
    assert result["terminal_status"] == "invalid_response"
    assert result["parse_failure_stage"] == "verifier"
    assert result["candidate_answer"] == "October"
    assert result["final_verification_parse_ok"] is False
    assert result["run_end"]["normal_completion"] is False


@pytest.mark.integration
def test_malformed_terminal_verifier_is_scientific_invalid_response() -> None:
    task = _task()
    backend = ScriptedBackend(
        [
            "intent one; intent two",
            "FINAL_ANSWER: first\nEVIDENCE: initial",
            "VERDICT: INSUFFICIENT\nREASON: more evidence required",
            "FINAL_ANSWER: revised\nEVIDENCE: expanded",
            "SUFFICIENT",
        ]
    )
    result = run_hotpot_workflow(backend=backend, controller=_controller(task), task=task)
    assert result["terminal_status"] == "invalid_response"
    assert result["parse_failure_stage"] == "terminal_verifier"
    assert result["candidate_answer"] == "revised"
    assert result["recovery_documents"] == 2
    assert result["run_end"]["normal_completion"] is False
