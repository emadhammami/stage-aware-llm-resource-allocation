from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from benchmark.hotpotqa import HotpotQAAdapter
from benchmark.pilot60 import FROZEN_RHO_STAR, PILOT_MODELS, PILOT_POLICIES, _backend
from workflow_control.backends import HuggingFaceBackend
from workflow_control.specs import hotpot_prompt_estimates

ROOT = Path(__file__).parents[2]


@pytest.mark.integration
def test_hotpot_selection_and_manifest_are_frozen() -> None:
    adapter = HotpotQAAdapter(ROOT / "data/hotpotqa/hotpot_dev_distractor_v1.json")
    assert adapter.deterministic_pilot_ids() == {
        "bridge": "5ade15ae5542990dbb2f7f4c",
        "comparison": "5ae3790f5542992e3233c415",
    }
    manifest = json.loads((ROOT / "pilot60_manifest.json").read_text())
    assert len(manifest["runs"]) == 60
    assert len({run["run_id"] for run in manifest["runs"]}) == 60
    expected_models = {
        "google/gemma-4-E4B-it",
        "Qwen/Qwen3-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
    }
    assert {model["model_id"] for model in manifest["models"]} == expected_models
    assert {run["model_id"] for run in manifest["runs"]} == expected_models
    assert Counter(run["model_id"] for run in manifest["runs"]) == Counter(
        {model_id: 20 for model_id in expected_models}
    )
    assert Counter(run["policy"] for run in manifest["runs"]) == Counter(
        {policy: 15 for policy in PILOT_POLICIES}
    )
    assert set(Counter(run["task_id"] for run in manifest["runs"]).values()) == {12}
    assert {
        model["model_id"]: {
            "family": model["family"],
            "revision": model["revision"],
        }
        for model in manifest["models"]
    } == PILOT_MODELS
    assert all(
        model["tokenizer_revision"] == PILOT_MODELS[model["model_id"]]["revision"]
        for model in manifest["models"]
    )
    assert len(manifest["initial_budget_estimates"]) == 15
    assert {
        estimate["calibration_rho"] for estimate in manifest["initial_budget_estimates"]
    } == {FROZEN_RHO_STAR}
    serialized = json.dumps(manifest)
    assert "gemini" not in serialized.lower()
    assert "mistral" not in serialized.lower()


@pytest.mark.integration
def test_hotpot_gold_is_not_in_workflow_view_or_budget_estimates() -> None:
    adapter = HotpotQAAdapter(ROOT / "data/hotpotqa/hotpot_dev_distractor_v1.json")
    task = adapter.get("5ade15ae5542990dbb2f7f4c")
    public = json.dumps(task.workflow_view(), sort_keys=True)
    gold = task.evaluation_gold()
    assert "answer" not in task.workflow_view()
    assert "supporting_facts" not in task.workflow_view()
    assert "supporting_facts" not in public
    assert gold["answer"]
    first = hotpot_prompt_estimates(
        task=task,
        counter=lambda text: len(text),
        provenance="fixture",
    )
    changed = replace(
        task,
        _gold_answer="deliberately different",
        _gold_supporting_facts=(("not visible", 999),),
    )
    second = hotpot_prompt_estimates(
        task=changed,
        counter=lambda text: len(text),
        provenance="fixture",
    )
    assert first == second


def test_same_task_model_b0_and_config_across_policies() -> None:
    manifest = json.loads((ROOT / "pilot60_manifest.json").read_text())
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for run in manifest["runs"]:
        grouped.setdefault((run["task_id"], run["model_id"]), []).append(run)
    assert len(grouped) == 15
    for runs in grouped.values():
        assert len(runs) == 4
        assert len({run["B0"] for run in runs}) == 1
        assert len({run["scientific_config_fingerprint"] for run in runs}) == 1
        assert len({run["policy"] for run in runs}) == 4


def test_three_families_share_one_local_backend_and_workflow_configuration() -> None:
    configuration = yaml.safe_load(
        (ROOT / "research/stage_aware/pilot60_config.yaml").read_text(encoding="utf-8")
    )
    models = configuration["models"]
    assert {model["family"] for model in models} == {"gemma4", "qwen3", "llama"}
    assert all(isinstance(_backend(model), HuggingFaceBackend) for model in models)
    assert {model["do_sample"] for model in models} == {False}
    assert {model["batch_size"] for model in models} == {1}
    assert {model["num_beams"] for model in models} == {1}
    assert {model["dtype"] for model in models} == {"bfloat16"}
    assert {model["device"] for model in models} == {"cuda"}
