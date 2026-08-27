from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def normalize_bool(series: pd.Series) -> pd.Series:
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f", "", "none", "nan", "<na>"}

    def convert(value):
        if pd.isna(value):
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in true_values:
            return True
        if text in false_values:
            return False
        raise ValueError(f"Cannot normalize boolean value: {value!r}")

    return series.map(convert)


def load_runs(path: str | Path = "results/runs.csv", include_pilot: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "run_status" in df.columns:
        df = df[df["run_status"].fillna("completed") == "completed"]
    if "is_pilot" in df.columns and not include_pilot:
        df = df[normalize_bool(df["is_pilot"]) != True]  # noqa: E712
    return df


def false_acceptance_rate(group: pd.DataFrame) -> float:
    accepted = group[normalize_bool(group["critic_accepted"]) == True]  # noqa: E712
    if accepted.empty:
        return float("nan")
    return normalize_bool(accepted["false_accept"]).mean() * 100


def aggregate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(by, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        total_tokens = group["total_tokens"].sum()
        workflow_success = normalize_bool(group["workflow_success"])
        candidate_correct = normalize_bool(group["candidate_correct"])
        repairs = workflow_success.sum()
        rows.append(
            {
                **dict(zip(by, key_tuple, strict=True)),
                "evaluated_tasks": len(group),
                "repair_rate_pct": workflow_success.mean() * 100,
                "candidate_correct_rate_pct": candidate_correct.mean() * 100,
                "mean_tokens": group["total_tokens"].mean(),
                "median_tokens": group["total_tokens"].median(),
                "repairs_per_100k_tokens": (repairs / total_tokens * 100000) if total_tokens else 0,
                "false_acceptance_rate_pct": false_acceptance_rate(group),
                "mean_llm_calls": group["llm_calls"].mean(),
                "early_exit_rate_pct": normalize_bool(group["early_exit"]).mean() * 100,
                "budget_exhaustion_rate_pct": normalize_bool(group["budget_exhausted"]).mean() * 100,
                "budget_violation_rate_pct": normalize_bool(group["budget_violation"]).mean() * 100,
                "mean_runtime_seconds": group["runtime_seconds"].mean(),
                "median_runtime_seconds": group["runtime_seconds"].median(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results/runs.csv")
    parser.add_argument("--include-pilot", action="store_true")
    args = parser.parse_args()
    df = load_runs(args.runs, include_pilot=args.include_pilot)
    print(aggregate(df, ["method", "token_budget"]).to_string(index=False))


if __name__ == "__main__":
    main()
