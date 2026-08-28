from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark import pilot60
from workflow_control.runtime import StructuralShortfallError


def _raise_shortfall(**kwargs: object) -> dict[str, object]:
    raise StructuralShortfallError(
        call_id="fixture:1",
        stage_id="fixture",
        remaining=10,
        required_minimum=20,
    )


def _raise_unexpected(**kwargs: object) -> dict[str, object]:
    raise RuntimeError("unexpected fixture failure")


def test_execute_records_structural_shortfall_and_continues(tmp_path, monkeypatch) -> None:
    root = Path(__file__).parents[2]
    manifest = json.loads((root / "pilot60_manifest.json").read_text(encoding="utf-8"))
    output = tmp_path / "events.jsonl"
    monkeypatch.setenv("PILOT60_EXECUTION_AUTHORIZED", "YES")
    monkeypatch.setattr(pilot60, "_backend", lambda model: object())
    monkeypatch.setattr(pilot60, "run_code_workflow", _raise_shortfall)
    monkeypatch.setattr(pilot60, "run_hotpot_workflow", _raise_shortfall)

    pilot60.execute(manifest, root, output)

    events = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    run_ends = [event for event in events if event.get("event_type") == "PILOT_RUN_END"]
    assert len(run_ends) == 60
    assert all(event["result"]["terminal_status"] == "structural_shortfall" for event in run_ends)
    assert all(event["result"]["classification"] == "scientific" for event in run_ends)
    assert all(event["result"]["shortfall"]["required_minimum"] == 20 for event in run_ends)


def test_execute_does_not_swallow_unexpected_runtime_errors(tmp_path, monkeypatch) -> None:
    root = Path(__file__).parents[2]
    manifest = json.loads((root / "pilot60_manifest.json").read_text(encoding="utf-8"))
    output = tmp_path / "events.jsonl"
    monkeypatch.setenv("PILOT60_EXECUTION_AUTHORIZED", "YES")
    monkeypatch.setattr(pilot60, "_backend", lambda model: object())
    monkeypatch.setattr(pilot60, "run_code_workflow", _raise_unexpected)
    monkeypatch.setattr(pilot60, "run_hotpot_workflow", _raise_unexpected)

    with pytest.raises(RuntimeError, match="unexpected fixture failure"):
        pilot60.execute(manifest, root, output)
