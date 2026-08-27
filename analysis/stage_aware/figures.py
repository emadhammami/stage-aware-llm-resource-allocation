from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "stage_aware_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

POLICY_LABELS = {
    "legacy_static": "Legacy static",
    "greedy": "Greedy",
    "fixed_reserved": "Fixed reserve",
    "adaptive_stage_aware": "Adaptive stage-aware",
}
COLORS = {
    "legacy_static": "#4C566A",
    "greedy": "#D08770",
    "fixed_reserved": "#5E81AC",
    "adaptive_stage_aware": "#2E8B57",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    paths = [directory / f"{stem}.png", directory / f"{stem}.pdf"]
    for path in paths:
        metadata = (
            {"Software": "stage-aware Phase 0 replay"}
            if path.suffix == ".png"
            else {
                "Creator": "stage-aware Phase 0 replay",
                "Producer": "matplotlib",
                "CreationDate": None,
                "ModDate": None,
            }
        )
        fig.savefig(path, metadata=metadata)
    plt.close(fig)
    return paths


def figure_prompt_output_share(
    call_rows: list[dict[str, Any]], directory: Path
) -> list[Path]:
    stages = ["planner", "executor_1", "executor_2", "critic"]
    labels = ["Planner", "Executor 1", "Executor 2", "Critic"]
    grouped = {
        stage: [row for row in call_rows if row["stage"] == stage]
        for stage in stages
    }
    prompt_means = [
        float(np.mean([row["prompt_tokens"] for row in grouped[stage]])) for stage in stages
    ]
    output_means = [
        float(np.mean([row["output_tokens"] for row in grouped[stage]])) for stage in stages
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    x = np.arange(len(stages))
    axes[0].bar(x, prompt_means, color="#5E81AC", label="Prompt/input")
    axes[0].bar(x, output_means, bottom=prompt_means, color="#D08770", label="Generated output")
    axes[0].set_xticks(x, labels, rotation=18)
    axes[0].set_ylabel("Mean provider-reported tokens")
    axes[0].set_title("Mean token composition")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(loc="upper right")
    positions = np.arange(len(stages)) * 3
    prompt_values = [[row["prompt_tokens"] for row in grouped[stage]] for stage in stages]
    output_values = [[row["output_tokens"] for row in grouped[stage]] for stage in stages]
    prompt_plot = axes[1].boxplot(
        prompt_values,
        positions=positions - 0.45,
        widths=0.75,
        patch_artist=True,
        showfliers=False,
    )
    output_plot = axes[1].boxplot(
        output_values,
        positions=positions + 0.45,
        widths=0.75,
        patch_artist=True,
        showfliers=False,
    )
    for patch in prompt_plot["boxes"]:
        patch.set_facecolor("#5E81AC")
    for patch in output_plot["boxes"]:
        patch.set_facecolor("#D08770")
    axes[1].set_xticks(positions, labels, rotation=18)
    axes[1].set_ylabel("Provider-reported tokens per call")
    axes[1].set_title("Across-call dispersion (outliers hidden)")
    axes[1].grid(axis="y", alpha=0.22)
    fig.suptitle("Prompt versus output cost by workflow stage", y=1.02)
    fig.tight_layout()
    return _save(fig, directory, "figure_1_prompt_output_share")


def figure_feasibility(rows: list[dict[str, Any]], directory: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    labels = {
        "historical_route_floor": "Observed route (exact)",
        "direct_completion_floor": "Direct completion (exact subset)",
        "full_retry_success_floor": "Full retry-success (mixed provenance)",
    }
    colors = ["#4C566A", "#5E81AC", "#2E8B57"]
    for (metric, label), color in zip(labels.items(), colors, strict=True):
        subset = sorted(
            (row for row in rows if row["metric"] == metric),
            key=lambda row: int(row["target_budget"]),
        )
        ax.plot(
            [int(row["target_budget"]) for row in subset],
            [float(row["feasible_fraction"]) for row in subset],
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
    ax.set(xlabel="Global token budget", ylabel="Structurally feasible fraction", ylim=(-0.03, 1.03))
    ax.set_xticks(sorted({int(row["target_budget"]) for row in rows}))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="lower right")
    ax.set_title("Prompt-plus-minimum-output structural feasibility")
    return _save(fig, directory, "figure_2_structural_feasibility")


def figure_action_rates(rows: list[dict[str, Any]], directory: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    policies = list(POLICY_LABELS)
    budgets = sorted({int(row["target_budget"]) for row in rows})
    width = 0.19
    x = np.arange(len(budgets))
    lookup = {(row["policy"], int(row["target_budget"])): row for row in rows}
    for index, policy in enumerate(policies):
        values = [float(lookup[(policy, budget)]["material_action_fraction"]) for budget in budgets]
        ax.bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=POLICY_LABELS[policy],
            color=COLORS[policy],
        )
    ax.axhline(0.20, color="#BF616A", linestyle="--", linewidth=1.2, label="20% gate")
    ax.set_xticks(x, [str(budget) for budget in budgets])
    ax.set(xlabel="Global token budget", ylabel="Runs with a material action", ylim=(0, 1.05))
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=2, loc="upper center")
    ax.set_title("Predeclared material-action rate (≥32 tokens)")
    return _save(fig, directory, "figure_3_policy_action_rates")


def figure_cap_binding_proxy(rows: list[dict[str, Any]], directory: Path) -> list[Path]:
    stages = ["planner", "executor_1", "executor_2", "critic"]
    labels = ["Planner", "Executor 1", "Executor 2", "Critic"]
    subset = {
        row["stage"]: row
        for row in rows
        if row["cohort"] == "publication_all"
    }
    x = np.arange(len(stages))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    for index, (threshold, color) in enumerate(
        ((1, "#4C566A"), (5, "#5E81AC"), (10, "#88C0D0"))
    ):
        values = [float(subset[stage][f"within_{threshold}pct_fraction"]) for stage in stages]
        ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=f"Within {threshold}% of cap",
            color=color,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Fraction of historical calls")
    ax.set_ylim(0, 1.03)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper right")
    ax.set_title("Historical cap-binding proxy (finish reason unavailable)")
    return _save(fig, directory, "figure_4_cap_binding_proxy")


def figure_starvation(rows: list[dict[str, Any]], directory: Path) -> list[Path]:
    policies = list(POLICY_LABELS)
    budgets = sorted({int(row["target_budget"]) for row in rows})
    stages = [("executor_2", "Retry / Executor 2"), ("critic", "Critic")]
    lookup = {
        (row["policy"], int(row["target_budget"]), row["stage"]): row for row in rows
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), constrained_layout=True)
    image_handle = None
    for ax, (stage, title) in zip(axes, stages, strict=True):
        values = np.array(
            [
                [
                    float(
                        lookup[(policy, budget, stage)][
                            "eligible_policy_starvation_fraction"
                        ]
                        or 0.0
                    )
                    for budget in budgets
                ]
                for policy in policies
            ]
        )
        image_handle = ax.imshow(values, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        for row_index, policy in enumerate(policies):
            for column_index, budget in enumerate(budgets):
                record = lookup[(policy, budget, stage)]
                numerator = int(record["eligible_policy_starvation_runs"])
                denominator = int(record["known_structurally_feasible_eligible_runs"])
                unknown = int(record["counterfactual_unknown_before_stage"])
                label = f"{numerator}/{denominator}"
                if unknown:
                    label += f"\n?{unknown}"
                ax.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if values[row_index, column_index] > 0.55 else "black",
                )
        ax.set_xticks(range(len(budgets)), [str(value) for value in budgets])
        ax.set_yticks(range(len(policies)), [POLICY_LABELS[policy] for policy in policies])
        ax.set_xlabel("Global token budget")
        ax.set_title(title)
    if image_handle is not None:
        fig.colorbar(
            image_handle,
            ax=axes,
            label="Eligible policy-starvation fraction\n(structurally infeasible cases excluded)",
            shrink=0.86,
        )
    fig.suptitle("Predicted eligible-stage starvation opportunity", y=1.04)
    return _save(fig, directory, "figure_5_eligible_stage_starvation")


def figure_capacity_flows(rows: list[dict[str, Any]], directory: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharex=True)
    policies = ["fixed_reserved", "adaptive_stage_aware"]
    budgets = sorted({int(row["target_budget"]) for row in rows})
    lookup = {(row["policy"], int(row["target_budget"])): row for row in rows}
    for policy in policies:
        axes[0].plot(
            budgets,
            [float(lookup[(policy, budget)]["mean_peak_protected_capacity"]) for budget in budgets],
            marker="o",
            label=POLICY_LABELS[policy],
            color=COLORS[policy],
        )
        axes[1].plot(
            budgets,
            [float(lookup[(policy, budget)]["mean_returned_capacity"]) for budget in budgets],
            marker="o",
            label=POLICY_LABELS[policy],
            color=COLORS[policy],
        )
    axes[0].set_title("Peak protected capacity")
    axes[1].set_title("Historical-compatible returned capacity")
    for ax in axes:
        ax.set_xlabel("Global token budget")
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Mean tokens per run")
    axes[1].legend(loc="upper left")
    fig.suptitle("Reservation and return flows", y=1.02)
    return _save(fig, directory, "figure_6_capacity_flows")


def figure_stage_allocations(rows: list[dict[str, Any]], directory: Path) -> list[Path]:
    stages = ["planner", "executor_1", "executor_2", "critic"]
    policies = list(POLICY_LABELS)
    budgets = sorted({int(row["target_budget"]) for row in rows})
    lookup = {
        (row["policy"], int(row["target_budget"]), row["stage"]): row for row in rows
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), sharex=True)
    for ax, stage in zip(axes.flat, stages, strict=True):
        for policy in policies:
            values = [
                float(lookup[(policy, budget, stage)]["mean_allocated_output"])
                if (policy, budget, stage) in lookup
                else np.nan
                for budget in budgets
            ]
            ax.plot(budgets, values, marker="o", label=POLICY_LABELS[policy], color=COLORS[policy])
        ax.set_title(stage.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.22)
    axes[1, 0].set_xlabel("Global token budget")
    axes[1, 1].set_xlabel("Global token budget")
    axes[0, 0].set_ylabel("Mean output allocation")
    axes[1, 0].set_ylabel("Mean output allocation")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Stage-level allocation envelopes", y=1.01)
    fig.tight_layout()
    return _save(fig, directory, "figure_s1_stage_allocations")


def figure_dispositions(rows: list[dict[str, Any]], directory: Path) -> list[Path]:
    policies = list(POLICY_LABELS)
    budgets = sorted({int(row["target_budget"]) for row in rows})
    dispositions = [
        "historical_cap_compatible",
        "cap_incompatible_counterfactual_unknown",
        "admission_denied",
        "historical_data_missing",
    ]
    labels = {
        "historical_cap_compatible": "Historical-cap compatible",
        "cap_incompatible_counterfactual_unknown": "Cap-incompatible / unknown",
        "admission_denied": "Admission denied",
        "historical_data_missing": "Historical data missing",
    }
    colors = ["#A3BE8C", "#EBCB8B", "#BF616A", "#B48EAD"]
    lookup = defaultdict(float)
    for row in rows:
        lookup[(row["policy"], int(row["target_budget"]), row["disposition"])] = float(
            row["fraction"]
        )
    fig, axes = plt.subplots(1, len(policies), figsize=(9.2, 3.2), sharey=True)
    for ax, policy in zip(axes, policies, strict=True):
        bottom = np.zeros(len(budgets))
        for disposition, color in zip(dispositions, colors, strict=True):
            values = np.array([lookup[(policy, budget, disposition)] for budget in budgets])
            ax.bar([str(value) for value in budgets], values, bottom=bottom, color=color, label=labels[disposition])
            bottom += values
        ax.set_title(POLICY_LABELS[policy])
        ax.set_xlabel("Budget")
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Run fraction")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle("Offline replay disposition (no outcome claims)", y=1.03)
    return _save(fig, directory, "figure_7_replay_dispositions")


def generate_all_figures(
    *,
    output_directory: str | Path,
    call_rows: list[dict[str, Any]],
    feasibility_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    disposition_rows: list[dict[str, Any]],
    cap_binding_rows: list[dict[str, Any]],
    starvation_rows: list[dict[str, Any]],
) -> list[Path]:
    _style()
    directory = Path(output_directory) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.extend(figure_prompt_output_share(call_rows, directory))
    paths.extend(figure_feasibility(feasibility_rows, directory))
    paths.extend(figure_action_rates(policy_rows, directory))
    paths.extend(figure_cap_binding_proxy(cap_binding_rows, directory))
    paths.extend(figure_starvation(starvation_rows, directory))
    paths.extend(figure_capacity_flows(policy_rows, directory))
    paths.extend(figure_dispositions(disposition_rows, directory))
    return paths
