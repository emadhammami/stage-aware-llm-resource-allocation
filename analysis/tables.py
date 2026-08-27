from __future__ import annotations

import argparse
from pathlib import Path

from analysis.aggregate import aggregate, load_runs
from benchmark.config import ExperimentConfig

METHOD_LABELS = {
    "single_shot": "Single-shot",
    "pec": "PEC",
    "pevc": "PEVC",
    "evidence_gated": "Evidence-Gated",
}


def write_table(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".md").write_text(df.to_markdown(index=False), encoding="utf-8")
    path.with_suffix(".tex").write_text(df.to_latex(index=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results/runs.csv")
    parser.add_argument("--out", default="results/summary")
    parser.add_argument("--include-pilot", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    df = load_runs(args.runs, include_pilot=args.include_pilot)
    config = ExperimentConfig.load()
    main_budget = config.main_comparison_budget

    table_a = aggregate(df[df["token_budget"] == main_budget], ["method"])
    table_a["Method"] = table_a["method"].map(METHOD_LABELS)
    table_a = table_a[
        [
            "Method",
            "repair_rate_pct",
            "candidate_correct_rate_pct",
            "mean_tokens",
            "median_tokens",
            "repairs_per_100k_tokens",
            "false_acceptance_rate_pct",
            "mean_llm_calls",
        ]
    ].rename(
        columns={
            "repair_rate_pct": "Repair Rate (%)",
            "candidate_correct_rate_pct": "Candidate Correct Rate (%)",
            "mean_tokens": "Mean Tokens",
            "median_tokens": "Median Tokens",
            "repairs_per_100k_tokens": "Repairs / 100K Tokens",
            "false_acceptance_rate_pct": "False Acceptance Rate (%)",
            "mean_llm_calls": "Mean LLM Calls",
        }
    )
    write_table(table_a, out / "table_a_main_comparison")

    table_b = aggregate(df[df["method"] == "evidence_gated"], ["token_budget"])
    table_b = table_b[
        [
            "token_budget",
            "repair_rate_pct",
            "mean_tokens",
            "repairs_per_100k_tokens",
            "early_exit_rate_pct",
            "budget_exhaustion_rate_pct",
            "budget_violation_rate_pct",
        ]
    ].rename(
        columns={
            "token_budget": "Budget",
            "repair_rate_pct": "Repair Rate (%)",
            "mean_tokens": "Mean Tokens",
            "repairs_per_100k_tokens": "Repairs / 100K Tokens",
            "early_exit_rate_pct": "Early Exit Rate (%)",
            "budget_exhaustion_rate_pct": "Budget Exhaustion Rate (%)",
            "budget_violation_rate_pct": "Budget Violation Rate (%)",
        }
    )
    write_table(table_b, out / "table_b_budget_sensitivity")
    print(f"wrote tables to {out}")


if __name__ == "__main__":
    main()
