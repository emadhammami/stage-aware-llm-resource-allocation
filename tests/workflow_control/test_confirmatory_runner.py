from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmark import confirmatory
from workflow_control.types import ModelMetadata, ProviderUsage

ROOT = Path(__file__).parents[2]
TASK_ID = "hotpotqa:5ade15ae5542990dbb2f7f4c"
MODEL_ID = "fixture/model"


class ScriptedBackend:
    model_id = MODEL_ID

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


def _estimate(task_id: str = TASK_ID) -> dict[str, Any]:
    stage_ids = (
        ["planner", "executor_1", "executor_2", "critic"]
        if task_id.startswith("quixbugs:")
        else ["plan", "answer", "verifier", "revise", "terminal_verifier"]
    )
    return {
        "task_id": task_id,
        "model_id": MODEL_ID,
        "estimator_version": "fixture-v1",
        "b_min": 1000,
        "b_soft": 2000,
        "b0": 1500,
        "calibration_rho": confirmatory.FROZEN_RHO_STAR,
        "tokenization_provenance": "fixture",
        "stages": [
            {
                "stage_id": stage_id,
                "predicted_prompt_tokens": 50,
                "exact_prompt_tokens": 50,
                "prompt_provenance": "fixture",
                "rendered_prompt_sha256": "0" * 64,
            }
            for stage_id in stage_ids
        ],
    }


def _manifest() -> dict[str, Any]:
    return {
        "protocol": confirmatory.PROTOCOL_ID,
        "models": [
            {
                "model_id": MODEL_ID,
                "family": "fixture",
                "device": "cpu",
                "dtype": "float32",
            }
        ],
        "tasks": [{"task_id": TASK_ID}],
        "policies": ["proposed"],
        "initial_budget_estimates": [_estimate()],
        "runs": [
            {
                "run_id": "fixture-hotpot-proposed",
                "task_id": TASK_ID,
                "model_id": MODEL_ID,
                "policy": "proposed",
                "B0": 5000,
            }
        ],
    }


def test_validate_manifest_accepts_minimal_confirmatory_manifest() -> None:
    confirmatory._validate_manifest(_manifest())


def test_validate_manifest_rejects_nonfrozen_rho() -> None:
    manifest = _manifest()
    manifest["initial_budget_estimates"][0]["calibration_rho"] = 0.5
    with pytest.raises(ValueError, match="frozen rho_star"):
        confirmatory._validate_manifest(manifest)


def test_validate_manifest_rejects_duplicate_run_id() -> None:
    manifest = _manifest()
    manifest["runs"].append(dict(manifest["runs"][0]))
    with pytest.raises(ValueError, match="run ids must be unique"):
        confirmatory._validate_manifest(manifest)


def test_validate_manifest_rejects_missing_workflow_stage_estimate() -> None:
    manifest = _manifest()
    manifest["initial_budget_estimates"][0]["stages"].pop()
    with pytest.raises(ValueError, match="workflow specification"):
        confirmatory._validate_manifest(manifest)


@pytest.mark.integration
def test_execute_scores_after_hotpot_workflow_and_writes_confirmatory_run_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    backend = ScriptedBackend(
        [
            "intent one; intent two",
            "FINAL_ANSWER: October\nEVIDENCE: fixture",
            "VERDICT: SUFFICIENT\nREASON: supported",
        ]
    )
    monkeypatch.setattr(confirmatory, "_backend", lambda model: backend)
    monkeypatch.setenv(confirmatory.EXECUTION_ENV, "YES")

    output = tmp_path / "events.jsonl"
    confirmatory.execute(_manifest(), ROOT, output)

    events = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    run_end = [event for event in events if event.get("event_type") == confirmatory.RUN_END_EVENT]
    assert len(run_end) == 1
    result = run_end[0]["result"]
    assert result["candidate_answer"] == "October"
    assert result["outcome"]["exact_match"] == 1.0
    assert result["outcome"]["end_to_end_exact_match"] == 1.0
    assert result["outcome"]["reliable_correct"] is True
    assert "answer" not in result["outcome"]
    assert "supporting_facts" not in result["outcome"]

    confirmatory.execute(_manifest(), ROOT, output)
    events_after_resume = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(
        event.get("event_type") == confirmatory.RUN_END_EVENT for event in events_after_resume
    ) == 1


def test_execute_is_locked_without_explicit_authorization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(confirmatory.EXECUTION_ENV, raising=False)
    with pytest.raises(RuntimeError, match="locked"):
        confirmatory.execute(_manifest(), ROOT, tmp_path / "events.jsonl")


@pytest.mark.integration
def test_execute_scores_hotpot_structural_shortfall_as_abnormal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    manifest["runs"][0]["B0"] = 1
    backend = ScriptedBackend([])
    monkeypatch.setattr(confirmatory, "_backend", lambda model: backend)
    monkeypatch.setenv(confirmatory.EXECUTION_ENV, "YES")

    output = tmp_path / "events.jsonl"
    confirmatory.execute(manifest, ROOT, output)

    events = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    run_end = [
        event
        for event in events
        if event.get("event_type") == confirmatory.RUN_END_EVENT
    ]
    assert len(run_end) == 1
    result = run_end[0]["result"]
    assert result["terminal_status"] == "structural_shortfall"
    assert result["run_end"]["normal_completion"] is False
    outcome = result["outcome"]
    assert outcome["exact_match"] == 0.0
    assert outcome["token_f1"] == 0.0
    assert outcome["end_to_end_exact_match"] == 0.0
    assert outcome["end_to_end_token_f1"] == 0.0
    assert outcome["reliable_correct"] is False


@pytest.mark.integration
def test_execute_scores_hotpot_only_after_workflow_returns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    backend = ScriptedBackend([])
    order: list[str] = []

    def fake_workflow(**kwargs):
        order.append("workflow")
        controller = kwargs["controller"]
        return {
            "task_id": kwargs["task"].task_id,
            "candidate_answer": "October",
            "answer_parse_ok": True,
            "final_verification_parse_ok": True,
            "final_verification_verdict": "SUFFICIENT",
            "stages": [],
            "run_end": controller.finalize(),
        }

    def fake_score(result, task):
        assert order == ["workflow"]
        assert result["run_end"]["normal_completion"] is True
        order.append("score")
        return {"scored_after_workflow": True}

    assert "evaluation_gold(" not in (ROOT / "workflow_control/runtime.py").read_text(encoding="utf-8")
    monkeypatch.setattr(confirmatory, "_backend", lambda model: backend)
    monkeypatch.setattr(confirmatory, "run_hotpot_workflow", fake_workflow)
    monkeypatch.setattr(confirmatory, "score_hotpot_result", fake_score)
    monkeypatch.setenv(confirmatory.EXECUTION_ENV, "YES")

    output = tmp_path / "events.jsonl"
    confirmatory.execute(manifest, ROOT, output)

    assert order == ["workflow", "score"]
    events = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    run_end = next(
        event
        for event in events
        if event.get("event_type") == confirmatory.RUN_END_EVENT
    )
    assert run_end["result"]["outcome"] == {"scored_after_workflow": True}
