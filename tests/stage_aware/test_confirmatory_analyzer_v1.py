from __future__ import annotations

import copy

import pytest

from analysis.stage_aware import confirmatory_analyzer_v1 as analyzer


def _result(
    *,
    outcome: dict[str, float | bool] | None = None,
    total_consumed: int = 100,
    b0: int = 200,
    unused_capacity: int = 100,
    stages: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "outcome": outcome
        or {
            "exact_match": 1.0,
            "token_f1": 1.0,
            "reliable_correct": True,
            "verification_supported_correct": True,
        },
        "stages": stages or [],
        "run_end": {
            "normal_completion": True,
            "total_consumed": total_consumed,
            "B0": b0,
            "unused_capacity": unused_capacity,
            "unresolved_reservations": [],
            "budget_violation_count": 0,
            "shortfall_count": 0,
        },
    }


def _record(
    *,
    run_id: str,
    task_id: str,
    model_id: str,
    policy: str,
    repetition: int = 1,
    outcome: dict[str, float | bool] | None = None,
    events: tuple[dict[str, object], ...] = (),
    stages: list[dict[str, object]] | None = None,
) -> analyzer.RunRecord:
    return analyzer.RunRecord(
        run_id=run_id,
        task_id=task_id,
        benchmark="hotpotqa" if task_id.startswith("hotpotqa:") else "quixbugs",
        model_id=model_id,
        policy=policy,
        budget_condition="transition",
        repetition=repetition,
        primary_analysis=True,
        result=_result(outcome=outcome, stages=stages),
        attempt_events=events,
    )


def test_resume_uses_only_latest_attempt_segment() -> None:
    run_id = "hotpotqa:t1::m1::legacy_static::transition::r1"
    manifest = {
        "runs": [
            {
                "run_id": run_id,
                "task_id": "hotpotqa:t1",
                "model_id": "m1",
                "policy": "legacy_static",
                "budget_condition": "transition",
                "repetition": 1,
                "primary_analysis": True,
            }
        ]
    }
    events = [
        {"event_type": "INITIAL_BUDGET", "run_id": run_id, "attempt": "old"},
        {"event_type": "PROVIDER_DEBIT", "run_id": run_id, "attempt": "old"},
        {"event_type": "INITIAL_BUDGET", "run_id": run_id, "attempt": "new"},
        {"event_type": "PROVIDER_DEBIT", "run_id": run_id, "attempt": "new"},
        {
            "event_type": "CONFIRMATORY_RUN_END",
            "run_id": run_id,
            "result": _result(),
        },
    ]

    records = analyzer.load_completed_runs(manifest, events)

    assert len(records) == 1
    assert [event.get("attempt") for event in records[0].attempt_events] == [
        "new",
        "new",
        None,
    ]


def test_duplicate_confirmatory_run_end_is_rejected() -> None:
    run_id = "hotpotqa:t1::m1::legacy_static::transition::r1"
    manifest = {
        "runs": [
            {
                "run_id": run_id,
                "task_id": "hotpotqa:t1",
                "model_id": "m1",
                "policy": "legacy_static",
                "budget_condition": "transition",
                "repetition": 1,
                "primary_analysis": True,
            }
        ]
    }
    end = {
        "event_type": "CONFIRMATORY_RUN_END",
        "run_id": run_id,
        "result": _result(),
    }
    events = [
        {"event_type": "INITIAL_BUDGET", "run_id": run_id},
        end,
        {"event_type": "INITIAL_BUDGET", "run_id": run_id},
        copy.deepcopy(end),
    ]

    with pytest.raises(ValueError, match="duplicate confirmatory run end"):
        analyzer.load_completed_runs(manifest, events)


def test_repetitions_are_aggregated_within_task_model_policy() -> None:
    records = [
        _record(
            run_id=f"r{i}",
            task_id="hotpotqa:t1",
            model_id="m1",
            policy="legacy_static",
            repetition=i,
            outcome={"reliable_correct": value},
        )
        for i, value in enumerate([True, False, True], start=1)
    ]

    aggregated = analyzer.aggregate_repetitions(
        records, lambda record: analyzer._outcome_value(record, "reliable_correct")
    )

    assert aggregated[("hotpotqa:t1", "m1", "legacy_static")] == pytest.approx(2 / 3)


def test_paired_bootstrap_is_deterministic_and_macro_uses_model_effects() -> None:
    task_ids = ["t1", "t2", "t3", "t4"]
    model_ids = ["m1", "m2"]
    legacy = {(task, model): 0.0 for task in task_ids for model in model_ids}
    proposed = {
        ("t1", "m1"): 0.1,
        ("t2", "m1"): 0.1,
        ("t3", "m1"): 0.1,
        ("t4", "m1"): 0.1,
        ("t1", "m2"): -0.05,
        ("t2", "m2"): -0.05,
        ("t3", "m2"): -0.05,
        ("t4", "m2"): -0.05,
    }

    first = analyzer.paired_bootstrap(
        task_ids, model_ids, legacy, proposed, resamples=1000, seed=20260828
    )
    second = analyzer.paired_bootstrap(
        task_ids, model_ids, legacy, proposed, resamples=1000, seed=20260828
    )

    assert first == second
    assert first["per_model_delta"]["m1"]["point"] == pytest.approx(0.1)
    assert first["per_model_delta"]["m2"]["point"] == pytest.approx(-0.05)
    assert first["macro_delta"]["point"] == pytest.approx(0.025)


def test_negative_effect_can_be_noninferior_without_being_superior(monkeypatch) -> None:
    records = [
        _record(
            run_id=f"{policy}-{rep}",
            task_id="hotpotqa:t1",
            model_id="m1",
            policy=policy,
            repetition=rep,
            outcome={"reliable_correct": 0.50 if policy == "legacy_static" else 0.48},
        )
        for policy in ("legacy_static", "proposed")
        for rep in (1, 2, 3)
    ]

    def fake_bootstrap(*args, **kwargs):
        return {
            "per_model_delta": {"m1": {"point": -0.02, "ci95": [-0.04, -0.01]}},
            "macro_delta": {"point": -0.02, "ci95": [-0.04, -0.01]},
        }

    monkeypatch.setattr(analyzer, "paired_bootstrap", fake_bootstrap)
    summary = analyzer.primary_metric_summary(
        records,
        "reliable_correct",
        resamples=10000,
        seed=20260828,
        noninferiority_margin=0.05,
        is_guardrail=True,
    )

    assert summary["macro"]["delta"] == pytest.approx(-0.02)
    assert summary["macro"]["guardrail_decision"]["noninferior"] is True
    assert summary["macro"]["guardrail_decision"]["superior"] is False


def test_non_guardrail_metric_has_no_noninferiority_decision(monkeypatch) -> None:
    records = [
        _record(
            run_id=f"{policy}-{rep}",
            task_id="hotpotqa:t1",
            model_id="m1",
            policy=policy,
            repetition=rep,
            outcome={"token_f1": 0.70 if policy == "legacy_static" else 0.71},
        )
        for policy in ("legacy_static", "proposed")
        for rep in (1, 2, 3)
    ]

    def fake_bootstrap(*args, **kwargs):
        return {
            "per_model_delta": {"m1": {"point": 0.01, "ci95": [-0.01, 0.03]}},
            "macro_delta": {"point": 0.01, "ci95": [-0.01, 0.03]},
        }

    monkeypatch.setattr(analyzer, "paired_bootstrap", fake_bootstrap)
    summary = analyzer.primary_metric_summary(
        records,
        "token_f1",
        resamples=10000,
        seed=20260828,
        noninferiority_margin=0.05,
        is_guardrail=False,
    )

    assert summary["macro"]["guardrail_decision"] is None


def test_resources_use_completed_attempt_events_and_run_end() -> None:
    record = _record(
        run_id="r1",
        task_id="hotpotqa:t1",
        model_id="m1",
        policy="proposed",
        events=(
            {"event_type": "INITIAL_BUDGET"},
            {"event_type": "PROVIDER_DEBIT"},
            {"event_type": "PROVIDER_DEBIT"},
            {"event_type": "RESERVATION_SHORTFALL"},
            {"event_type": "CONFIRMATORY_RUN_END"},
        ),
        stages=[{"stage_id": "plan"}, {"stage_id": "answer"}, {"stage_id": "verifier"}],
    )

    values = analyzer._resource_values(record)

    assert values["total_consumed_tokens"] == 100
    assert values["unused_capacity"] == 100
    assert values["provider_calls"] == 2
    assert values["downstream_stage_completion"] == 1
    assert values["structural_shortfall"] == 0
    assert values["reservation_shortfall"] == 1


def test_validate_primary_shape_requires_complete_repetition_cells() -> None:
    records = [
        _record(
            run_id=f"{task}-{model}-{policy}-{rep}",
            task_id=f"hotpotqa:{task}",
            model_id=model,
            policy=policy,
            repetition=rep,
        )
        for task in ("t1", "t2")
        for model in ("m1", "m2")
        for policy in ("legacy_static", "proposed")
        for rep in (1, 2, 3)
    ]
    manifest = {"matrix_summary": {"primary_runs": 24}}
    plan = {
        "primary_comparison": {
            "benchmark": "hotpotqa",
            "baseline_policy": "legacy_static",
            "candidate_policy": "proposed",
            "models": 2,
            "repetitions": 3,
        }
    }

    analyzer.validate_primary_shape(records, manifest, plan)

    with pytest.raises(ValueError, match="primary run count"):
        analyzer.validate_primary_shape(records[:-1], manifest, plan)
