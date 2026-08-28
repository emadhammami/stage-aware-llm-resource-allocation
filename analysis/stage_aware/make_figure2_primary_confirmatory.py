from pathlib import Path
import json
import hashlib
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_ANALYSIS_SHA256 = "3F21FDA9559D32A03B5A0CE7BA0DC4D8CC38B82D64454EC87D88A4D7E43B74A4"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest().upper()


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    analysis_path = root / "results" / "stage_aware_confirmatory_v1" / "confirmatory_analysis_v1.json"
    out_dir = root / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / "figure2_primary_confirmatory_cross_model.pdf"
    png_path = out_dir / "figure2_primary_confirmatory_cross_model.png"

    raw = analysis_path.read_bytes()
    analysis_sha = sha256_bytes(raw)
    if analysis_sha != EXPECTED_ANALYSIS_SHA256:
        raise RuntimeError(
            f"Unexpected analysis SHA256: {analysis_sha} != {EXPECTED_ANALYSIS_SHA256}"
        )

    data = json.loads(raw.decode("utf-8"))
    metric = data["primary_quality"]["reliable_correct"]

    model_order = [
        ("Macro", None),
        ("Qwen3-8B", "Qwen/Qwen3-8B"),
        ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),
        ("Gemma-4-E4B", "google/gemma-4-E4B-it"),
    ]

    labels = []
    legacy_vals = []
    rrr_vals = []
    delta_vals = []
    ci_low_err = []
    ci_high_err = []

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

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
        }
    )

    fig = plt.figure(figsize=(7.1, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.45)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Panel (a): raw values
    x = np.arange(len(labels))
    width = 0.34

    bars_legacy = ax1.bar(
        x - width / 2,
        legacy_vals,
        width,
        label="Legacy static",
        edgecolor="black",
        linewidth=0.6,
    )
    bars_rrr = ax1.bar(
        x + width / 2,
        rrr_vals,
        width,
        label="RRR",
        edgecolor="black",
        linewidth=0.6,
    )

    ax1.set_ylabel("Reliable correct (%)")
    ax1.set_xticks(x, ["Macro", "Qwen3-8B", "Llama-3.1\n8B", "Gemma-4\nE4B"])
    ax1.set_ylim(0, max(max(legacy_vals + rrr_vals) + 8.0, 40.0))
    ax1.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper left", frameon=False, ncols=2, handlelength=1.5)


    ax1.text(0.0, 1.03, "(a)", transform=ax1.transAxes, ha="left", va="bottom")

    # Panel (b): delta forest plot
    y = np.arange(len(labels))
    xerr = np.vstack([ci_low_err, ci_high_err])

    ax2.errorbar(
        delta_vals,
        y,
        xerr=xerr,
        fmt="o",
        capsize=3,
        linewidth=1.0,
        markersize=4,
    )

    ax2.axvline(0.0, linewidth=1.0)
    ax2.axvline(-5.0, linestyle="--", linewidth=1.0)

    ax2.set_yticks(y, labels)
    ax2.invert_yaxis()
    ax2.set_xlabel("RRR - Legacy (pp)")
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
        ax2.text(
            label_x,
            yv,
            f"{xv:+.1f}",
            ha="right",
            va="center",
            fontsize=7,
        )

    ax2.text(0.0, 1.03, "(b)", transform=ax2.transAxes, ha="left", va="bottom")

    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("FIGURE2_SOURCE_SHA256=" + analysis_sha)
    print("FIGURE2_PDF=" + str(pdf_path.relative_to(root)))
    print("FIGURE2_PNG=" + str(png_path.relative_to(root)))
    print("FIGURE2_PDF_SHA256=" + sha256_bytes(pdf_path.read_bytes()))
    print("FIGURE2_PNG_SHA256=" + sha256_bytes(png_path.read_bytes()))
    print("FIGURE2_READY=TRUE")


if __name__ == "__main__":
    main()
