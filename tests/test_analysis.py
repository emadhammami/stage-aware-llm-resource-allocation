import pandas as pd

from analysis.aggregate import aggregate, load_runs


def test_aggregation_metrics():
    df = pd.DataFrame(
        [
            {"method": "evidence_gated", "token_budget": 8000, "repaired": True, "total_tokens": 1000, "critic_accepted": True, "false_accept": False, "llm_calls": 3, "early_exit": False, "budget_exceeded": False, "runtime_seconds": 1.0},
            {"method": "evidence_gated", "token_budget": 8000, "repaired": False, "total_tokens": 2000, "critic_accepted": True, "false_accept": True, "llm_calls": 2, "early_exit": False, "budget_exceeded": False, "runtime_seconds": 2.0},
        ]
    )
    df = df.rename(columns={"repaired": "workflow_success", "budget_exceeded": "budget_violation"})
    df["candidate_correct"] = df["workflow_success"]
    df["budget_exhausted"] = False
    result = aggregate(df, ["method", "token_budget"])
    row = result.iloc[0]
    assert row["repair_rate_pct"] == 50
    assert round(row["repairs_per_100k_tokens"], 2) == 33.33
    assert row["false_acceptance_rate_pct"] == 50


def test_csv_boolean_normalization_preserves_false_values(tmp_path):
    path = tmp_path / "runs.csv"
    pd.DataFrame(
        [
            {
                "method": "evidence_gated",
                "token_budget": 8000,
                "workflow_success": "True",
                "candidate_correct": "True",
                "total_tokens": 1000,
                "critic_accepted": "True",
                "false_accept": "False",
                "llm_calls": 3,
                "early_exit": "False",
                "budget_exhausted": "False",
                "budget_violation": "False",
                "runtime_seconds": 1.0,
                "is_pilot": "False",
            },
            {
                "method": "evidence_gated",
                "token_budget": 8000,
                "workflow_success": "False",
                "candidate_correct": "False",
                "total_tokens": 2000,
                "critic_accepted": "False",
                "false_accept": "False",
                "llm_calls": 2,
                "early_exit": "True",
                "budget_exhausted": "True",
                "budget_violation": "False",
                "runtime_seconds": 2.0,
                "is_pilot": "False",
            },
        ]
    ).to_csv(path, index=False)
    result = aggregate(load_runs(path), ["method", "token_budget"]).iloc[0]
    assert result["repair_rate_pct"] == 50
    assert result["early_exit_rate_pct"] == 50
    assert result["budget_exhaustion_rate_pct"] == 50
    assert result["budget_violation_rate_pct"] == 0


def test_analysis_excludes_infrastructure_errors_by_default(tmp_path):
    path = tmp_path / "runs.csv"
    pd.DataFrame(
        [
            {
                "method": "single_shot",
                "token_budget": 8000,
                "workflow_success": "True",
                "candidate_correct": "True",
                "total_tokens": 100,
                "critic_accepted": "",
                "false_accept": "",
                "llm_calls": 1,
                "early_exit": "False",
                "budget_exhausted": "False",
                "budget_violation": "False",
                "runtime_seconds": 1,
                "is_pilot": "False",
                "run_status": "completed",
            },
            {
                "method": "single_shot",
                "token_budget": 8000,
                "workflow_success": "False",
                "candidate_correct": "False",
                "total_tokens": 0,
                "critic_accepted": "",
                "false_accept": "",
                "llm_calls": 0,
                "early_exit": "False",
                "budget_exhausted": "False",
                "budget_violation": "False",
                "runtime_seconds": 0,
                "is_pilot": "False",
                "run_status": "infrastructure_error",
            },
        ]
    ).to_csv(path, index=False)
    loaded = load_runs(path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["run_status"] == "completed"
