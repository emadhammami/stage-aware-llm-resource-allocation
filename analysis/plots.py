from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from analysis.aggregate import aggregate, load_runs
from benchmark.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results/runs.csv")
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--include-pilot", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_runs(args.runs, include_pilot=args.include_pilot)
    config = ExperimentConfig.load()

    v3 = aggregate(df[df["method"] == "evidence_gated"], ["token_budget"]).sort_values("token_budget")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(v3["token_budget"], v3["repair_rate_pct"], marker="o")
    ax.set_xlabel("Token budget")
    ax.set_ylabel("Repair success rate (%)")
    ax.set_title("Evidence-Gated Repair Rate vs Budget")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "v3_repair_rate_vs_budget.png", dpi=200)
    fig.savefig(out / "v3_repair_rate_vs_budget.pdf")
    plt.close(fig)

    main_budget = config.main_comparison_budget
    main_df = aggregate(df[df["token_budget"] == main_budget], ["method"])
    fig, ax1 = plt.subplots(figsize=(6, 3.4))
    x = range(len(main_df))
    ax1.bar([i - 0.18 for i in x], main_df["repair_rate_pct"], width=0.36, label="Repair rate (%)")
    ax2 = ax1.twinx()
    ax2.bar([i + 0.18 for i in x], main_df["repairs_per_100k_tokens"], width=0.36, color="#5b8c5a", label="Repairs / 100K tokens")
    ax1.set_xticks(list(x), main_df["method"], rotation=20, ha="right")
    ax1.set_ylabel("Repair rate (%)")
    ax2.set_ylabel("Repairs / 100K tokens")
    ax1.set_title(f"Main Method Comparison at {main_budget} Tokens")
    fig.tight_layout()
    fig.savefig(out / "main_method_comparison.png", dpi=200)
    fig.savefig(out / "main_method_comparison.pdf")
    plt.close(fig)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
