import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "results" / "figures" / "r3tune_publication"
FIGDIR.mkdir(parents=True, exist_ok=True)

CONFIRMATORY = ROOT / "results" / "stage_aware_confirmatory_v1" / "confirmatory_analysis_v1.json"
EXPLORATORY = ROOT / "results" / "stage_aware_confirmatory_v1" / "exploratory_operating_envelope_v1.json"

EXPECTED_CONFIRMATORY_SHA256 = "3F21FDA9559D32A03B5A0CE7BA0DC4D8CC38B82D64454EC87D88A4D7E43B74A4"
EXPECTED_EXPLORATORY_SHA256 = "7EFA1061657D0063BC6D7DC4ADDB5A807ED34456CC21EB50FF31F6B59C5F59C6"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest().upper()


def load_frozen(path: Path, expected_sha: str) -> dict:
    raw = path.read_bytes()
    actual = sha256_bytes(raw)
    if actual != expected_sha:
        raise RuntimeError(f"Unexpected source SHA256 for {path}: {actual} != {expected_sha}")
    return json.loads(raw.decode("utf-8-sig"))


# ---------------------------------------------------------------------
# Figure 1: Qwen operating envelope
# Exact original geometry/style/data; only:
#   - non-GUI Agg backend
#   - IEEE-safe TrueType PDF fonts
#   - visible method label RRR -> R3Tune
# ---------------------------------------------------------------------
def make_figure1() -> tuple[Path, Path]:
    data = load_frozen(EXPLORATORY, EXPECTED_EXPLORATORY_SHA256)
    assert data["status"] == "POSTHOC_EXPLORATORY_NOT_CONFIRMATORY_NOT_CAUSAL"
    rows = data["settings"]

    model = "Qwen/Qwen3-8B"
    regimes = ["constrained", "transition", "relaxed"]
    regime_labels = ["Constrained", "Transition", "Relaxed"]
    benchmarks = ["hotpotqa", "quixbugs"]
    lookup = {(r["model"], r["benchmark"], r["budget_condition"]): r for r in rows}

    for b in benchmarks:
        for c in regimes:
            assert (model, b, c) in lookup

    def series(benchmark, key):
        return [100.0 * float(lookup[(model, benchmark, c)][key]) for c in regimes]

    def delta_series(benchmark, metric):
        return [100.0 * float(lookup[(model, benchmark, c)]["metrics"][metric]["delta_rrr_minus_legacy"]) for c in regimes]

    x = list(range(3))
    hot_def = series("hotpotqa", "deficit_prevalence")
    qx_def = series("quixbugs", "deficit_prevalence")
    hot_floor = series("hotpotqa", "rrr_floor_saturation_intensity")
    qx_floor = series("quixbugs", "rrr_floor_saturation_intensity")
    hot_len = series("hotpotqa", "rrr_length_finish_fraction")
    qx_len = series("quixbugs", "rrr_length_finish_fraction")
    hot_q = delta_series("hotpotqa", "reliable_correct")
    qx_q = delta_series("quixbugs", "reliable_correct")
    hot_d = delta_series("hotpotqa", "downstream_stage_completion")
    qx_d = delta_series("quixbugs", "downstream_stage_completion")

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.45), constrained_layout=True)

    def setup(ax, ylabel):
        ax.set_xticks(x)
        ax.set_xticklabels(regime_labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linewidth=0.5, alpha=0.3)

    def panel(ax, label):
        ax.text(0.0, 1.02, label, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=9)

    ax = axes[0, 0]
    ax.plot(x, hot_def, marker="o", linestyle="-", linewidth=1.4, label="HotpotQA")
    ax.plot(x, qx_def, marker="s", linestyle="--", linewidth=1.4, label="QuixBugs")
    setup(ax, "Deficit prevalence (%)")
    ax.set_ylim(-5, 105)
    ax.legend(frameon=False, loc="upper right")
    panel(ax, "(a)")

    ax = axes[0, 1]
    ax.plot(x, hot_floor, marker="o", linestyle="-", linewidth=1.2, label="HotpotQA: floor")
    ax.plot(x, hot_len, marker="^", linestyle="-.", linewidth=1.2, label="HotpotQA: length")
    ax.plot(x, qx_floor, marker="s", linestyle="--", linewidth=1.2, label="QuixBugs: floor")
    ax.plot(x, qx_len, marker="D", linestyle=":", linewidth=1.2, label="QuixBugs: length")
    setup(ax, "R3Tune event rate (%)")
    ax.set_ylim(-5, 105)
    ax.legend(frameon=False, loc="upper right", ncols=1)
    panel(ax, "(b)")

    ax = axes[1, 0]
    ax.axhline(0.0, linestyle="--", linewidth=0.9)
    ax.plot(x, hot_q, marker="o", linestyle="-", linewidth=1.4, label="HotpotQA")
    ax.plot(x, qx_q, marker="s", linestyle="--", linewidth=1.4, label="QuixBugs")
    setup(ax, r"$\Delta$ reliable correctness (pp)")
    ax.legend(frameon=False, loc="lower right")
    panel(ax, "(c)")

    ax = axes[1, 1]
    ax.axhline(0.0, linestyle="--", linewidth=0.9)
    ax.plot(x, hot_d, marker="o", linestyle="-", linewidth=1.4, label="HotpotQA")
    ax.plot(x, qx_d, marker="s", linestyle="--", linewidth=1.4, label="QuixBugs")
    setup(ax, r"$\Delta$ downstream completion (pp)")
    ax.legend(frameon=False, loc="lower right")
    panel(ax, "(d)")

    pdf = FIGDIR / "figure1_qwen_operating_envelope.pdf"
    png = FIGDIR / "figure1_qwen_operating_envelope.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


# ---------------------------------------------------------------------
# Figure 2: primary confirmatory quality
# Exact original geometry/style/data; only R3Tune labels + Type42.
# ---------------------------------------------------------------------
def make_figure2() -> tuple[Path, Path]:
    data = load_frozen(CONFIRMATORY, EXPECTED_CONFIRMATORY_SHA256)
    metric = data["primary_quality"]["reliable_correct"]

    model_order = [
        ("Macro", None),
        ("Qwen3-8B", "Qwen/Qwen3-8B"),
        ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),
        ("Gemma-4-E4B", "google/gemma-4-E4B-it"),
    ]

    labels, legacy_vals, rrr_vals, delta_vals, ci_low_err, ci_high_err = [], [], [], [], [], []

    for label, model_id in model_order:
        row = metric["macro"] if model_id is None else metric["per_model"][model_id]
        labels.append(label)
        legacy = 100.0 * float(row["legacy"])
        rrr = 100.0 * float(row["rrr"])
        delta = 100.0 * float(row["delta"])
        legacy_vals.append(legacy)
        rrr_vals.append(rrr)
        delta_vals.append(delta)
        ci = row.get("delta_ci95", [row["delta"], row["delta"]])
        ci_low = 100.0 * float(ci[0])
        ci_high = 100.0 * float(ci[1])
        ci_low_err.append(max(0.0, delta - ci_low))
        ci_high_err.append(max(0.0, ci_high - delta))

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(7.1, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.45)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    x = np.arange(len(labels))
    width = 0.34

    ax1.bar(x - width / 2, legacy_vals, width, label="Legacy static", edgecolor="black", linewidth=0.6)
    ax1.bar(x + width / 2, rrr_vals, width, label="R3Tune", edgecolor="black", linewidth=0.6)
    ax1.set_ylabel("Reliable correct (%)")
    ax1.set_xticks(x, ["Macro", "Qwen3-8B", "Llama-3.1\n8B", "Gemma-4\nE4B"])
    ax1.set_ylim(0, max(max(legacy_vals + rrr_vals) + 8.0, 40.0))
    ax1.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper left", frameon=False, ncols=2, handlelength=1.5)
    ax1.text(0.0, 1.03, "(a)", transform=ax1.transAxes, ha="left", va="bottom")

    y = np.arange(len(labels))
    xerr = np.vstack([ci_low_err, ci_high_err])
    ax2.errorbar(delta_vals, y, xerr=xerr, fmt="o", capsize=3, linewidth=1.0, markersize=4)
    ax2.axvline(0.0, linewidth=1.0)
    ax2.axvline(-5.0, linestyle="--", linewidth=1.0)
    ax2.set_yticks(y, labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("R3Tune - Legacy (pp)")
    ax2.grid(axis="x", linewidth=0.5, alpha=0.4)
    ax2.set_axisbelow(True)

    xmin = min([-6.0] + [d - lo - 0.6 for d, lo in zip(delta_vals, ci_low_err)])
    xmax = max([2.0] + [d + hi + 0.6 for d, hi in zip(delta_vals, ci_high_err)])
    if math.isclose(xmin, xmax):
        xmin -= 1.0
        xmax += 1.0
    ax2.set_xlim(xmin, xmax)

    label_x = -0.35
    for xv, yv in zip(delta_vals, y):
        ax2.text(label_x, yv, f"{xv:+.1f}", ha="right", va="center", fontsize=7)

    ax2.text(0.0, 1.03, "(b)", transform=ax2.transAxes, ha="left", va="bottom")

    pdf = FIGDIR / "figure2_primary_confirmatory_cross_model.pdf"
    png = FIGDIR / "figure2_primary_confirmatory_cross_model.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


# ---------------------------------------------------------------------
# Figure 3: primary resources
# Exact original geometry/style/data; only R3Tune labels + Type42.
# ---------------------------------------------------------------------
MODEL_ORDER_3 = [
    ("Macro", None),
    ("Qwen3-8B", "Qwen/Qwen3-8B"),
    ("Llama-3.1\n8B", "meta-llama/Llama-3.1-8B-Instruct"),
    ("Gemma-4\nE4B", "google/gemma-4-E4B-it"),
]


def row_for(metric: dict, model_id):
    return metric["macro"] if model_id is None else metric["per_model"][model_id]


def values(metric: dict):
    legacy, rrr, delta = [], [], []
    for _, model_id in MODEL_ORDER_3:
        row = row_for(metric, model_id)
        legacy.append(float(row["legacy"]))
        rrr.append(float(row["rrr"]))
        delta.append(float(row["delta"]))
    return legacy, rrr, delta


def annotate_delta_bars(ax, xs, vals, fmt="{:+.1f}"):
    lo, hi = ax.get_ylim()
    pad = 0.025 * (hi - lo)
    for x, v in zip(xs, vals):
        if v > 0:
            ax.text(x, v + pad, fmt.format(v), ha="center", va="bottom", fontsize=7)
        elif v < 0:
            ax.text(x, v - pad, fmt.format(v), ha="center", va="top", fontsize=7)
        else:
            ax.text(x, pad, fmt.format(v), ha="center", va="bottom", fontsize=7)


def make_figure3() -> tuple[Path, Path]:
    data = load_frozen(CONFIRMATORY, EXPECTED_CONFIRMATORY_SHA256)
    resources = data["primary_resources"]

    tok_l, tok_r, tok_d = values(resources["total_consumed_tokens"])
    _, _, down_d = values(resources["downstream_stage_completion"])
    _, _, struct_d = values(resources["structural_shortfall"])
    down_d = [100.0 * v for v in down_d]
    struct_d = [100.0 * v for v in struct_d]

    labels = [x[0] for x in MODEL_ORDER_3]
    x = np.arange(len(labels))
    width = 0.34

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.bar(x - width/2, tok_l, width, label="Legacy static", edgecolor="black", linewidth=0.6)
    ax1.bar(x + width/2, tok_r, width, label="R3Tune", edgecolor="black", linewidth=0.6)
    ax1.set_ylabel("Mean consumed tokens")
    ax1.set_xticks(x, labels)
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.set_ylim(0, 2000)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False, ncols=2, handlelength=1.5)
    ax1.text(0.0, 1.03, "(a)", transform=ax1.transAxes, ha="left", va="bottom")

    ax2.bar(x, tok_d, edgecolor="black", linewidth=0.6)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_ylim(-6.0, 6.0)
    ax2.set_ylabel("R3Tune - Legacy (tokens)")
    ax2.set_xticks(x, labels)
    ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax2.set_axisbelow(True)
    annotate_delta_bars(ax2, x, tok_d, fmt="{:+.1f}")
    ax2.text(0.0, 1.03, "(b)", transform=ax2.transAxes, ha="left", va="bottom")

    ax3.bar(x, down_d, edgecolor="black", linewidth=0.6)
    ax3.axhline(0.0, color="black", linewidth=0.8)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_ylabel("Delta downstream completion (pp)")
    ax3.set_xticks(x, labels)
    ax3.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax3.set_axisbelow(True)
    annotate_delta_bars(ax3, x, down_d, fmt="{:+.1f}")
    ax3.text(0.0, 1.03, "(c)", transform=ax3.transAxes, ha="left", va="bottom")

    ax4.bar(x, struct_d, edgecolor="black", linewidth=0.6)
    ax4.axhline(0.0, color="black", linewidth=0.8)
    ax4.set_ylim(-3.0, 3.0)
    ax4.set_ylabel("Delta structural shortfall (pp)")
    ax4.set_xticks(x, labels)
    ax4.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax4.set_axisbelow(True)
    annotate_delta_bars(ax4, x, struct_d, fmt="{:+.1f}")
    ax4.text(0.0, 1.03, "(d)", transform=ax4.transAxes, ha="left", va="bottom")

    fig.subplots_adjust(wspace=0.32, hspace=0.42)

    pdf = FIGDIR / "figure3_primary_resources_cross_model.pdf"
    png = FIGDIR / "figure3_primary_resources_cross_model.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def main():
    outputs = []
    outputs.extend(make_figure1())
    outputs.extend(make_figure2())
    outputs.extend(make_figure3())

    print("CONFIRMATORY_SOURCE_SHA256=" + sha256_bytes(CONFIRMATORY.read_bytes()))
    print("EXPLORATORY_SOURCE_SHA256=" + sha256_bytes(EXPLORATORY.read_bytes()))
    for path in outputs:
        print(path.relative_to(ROOT), sha256_bytes(path.read_bytes()))
    print("R3TUNE_IEEE_FIGURES_READY=TRUE")


if __name__ == "__main__":
    main()
