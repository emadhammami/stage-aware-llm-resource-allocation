from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

EXPECTED_ANALYSIS_SHA256 = "3F21FDA9559D32A03B5A0CE7BA0DC4D8CC38B82D64454EC87D88A4D7E43B74A4"

MODEL_ORDER = [
    ("Macro", None),
    ("Qwen3-8B", "Qwen/Qwen3-8B"),
    ("Llama-3.1\n8B", "meta-llama/Llama-3.1-8B-Instruct"),
    ("Gemma-4\nE4B", "google/gemma-4-E4B-it"),
]

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest().upper()

def row_for(metric: dict, model_id):
    return metric["macro"] if model_id is None else metric["per_model"][model_id]

def values(metric: dict):
    legacy, rrr, delta = [], [], []
    for _, model_id in MODEL_ORDER:
        row = row_for(metric, model_id)
        legacy.append(float(row["legacy"]))
        rrr.append(float(row["rrr"]))
        delta.append(float(row["delta"]))
    return legacy, rrr, delta

def symmetric_limit(vals, floor=1.0):
    m = max([abs(v) for v in vals] + [floor])
    return 1.25 * m

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

def make_figure3(root: Path, data: dict):
    resources = data["primary_resources"]

    tok_l, tok_r, tok_d = values(resources["total_consumed_tokens"])
    _, _, down_d = values(resources["downstream_stage_completion"])
    _, _, struct_d = values(resources["structural_shortfall"])

    down_d = [100.0 * v for v in down_d]
    struct_d = [100.0 * v for v in struct_d]

    labels = [x[0] for x in MODEL_ORDER]
    x = np.arange(len(labels))
    width = 0.34

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
    })

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # (a) Raw mean token consumption
    ax1.bar(x - width/2, tok_l, width, label="Legacy static", edgecolor="black", linewidth=0.6)
    ax1.bar(x + width/2, tok_r, width, label="RRR", edgecolor="black", linewidth=0.6)
    ax1.set_ylabel("Mean consumed tokens")
    ax1.set_xticks(x, labels)
    ax1.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.set_ylim(0, 2000)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False, ncols=2, handlelength=1.5)
    ax1.text(0.0, 1.03, "(a)", transform=ax1.transAxes, ha="left", va="bottom")

    # (b) Token delta
    ax2.bar(x, tok_d, edgecolor="black", linewidth=0.6)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    lim = symmetric_limit(tok_d, floor=2.0)
    ax2.set_ylim(-6.0, 6.0)
    ax2.set_ylabel("RRR - Legacy (tokens)")
    ax2.set_xticks(x, labels)
    ax2.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax2.set_axisbelow(True)
    annotate_delta_bars(ax2, x, tok_d, fmt="{:+.1f}")
    ax2.text(0.0, 1.03, "(b)", transform=ax2.transAxes, ha="left", va="bottom")

    # (c) Downstream completion delta
    ax3.bar(x, down_d, edgecolor="black", linewidth=0.6)
    ax3.axhline(0.0, color="black", linewidth=0.8)
    lim = symmetric_limit(down_d, floor=1.0)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_ylabel("Delta downstream completion (pp)")
    ax3.set_xticks(x, labels)
    ax3.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax3.set_axisbelow(True)
    annotate_delta_bars(ax3, x, down_d, fmt="{:+.1f}")
    ax3.text(0.0, 1.03, "(c)", transform=ax3.transAxes, ha="left", va="bottom")

    # (d) Structural shortfall delta
    ax4.bar(x, struct_d, edgecolor="black", linewidth=0.6)
    ax4.axhline(0.0, color="black", linewidth=0.8)
    lim = symmetric_limit(struct_d, floor=1.0)
    ax4.set_ylim(-3.0, 3.0)
    ax4.set_ylabel("Delta structural shortfall (pp)")
    ax4.set_xticks(x, labels)
    ax4.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax4.set_axisbelow(True)
    annotate_delta_bars(ax4, x, struct_d, fmt="{:+.1f}")
    ax4.text(0.0, 1.03, "(d)", transform=ax4.transAxes, ha="left", va="bottom")

    fig.subplots_adjust(wspace=0.32, hspace=0.42)

    out = root / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / "figure3_primary_resources_cross_model.pdf"
    png = out / "figure3_primary_resources_cross_model.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png

def rounded_box(ax, xy, w, h, text, fontsize=8):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor="none", edgecolor="black", linewidth=1.0
    )
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)
    return box

def arrow(ax, start, end, linestyle="-"):
    a = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10,
        linewidth=1.0, linestyle=linestyle
    )
    ax.add_patch(a)
    return a

def make_figure4(root: Path):
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(7.1, 2.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    rounded_box(ax, (0.03, 0.58), 0.17, 0.22, "Remaining budget\n$B_t$")
    rounded_box(ax, (0.27, 0.58), 0.22, 0.22, "RRR controller\nreserve future stages")
    rounded_box(ax, (0.55, 0.58), 0.17, 0.22, "Current stage\nexecutes")
    rounded_box(ax, (0.80, 0.58), 0.17, 0.22, "Next-stage\nbudget")

    arrow(ax, (0.20, 0.69), (0.27, 0.69))
    arrow(ax, (0.49, 0.69), (0.55, 0.69))
    arrow(ax, (0.72, 0.69), (0.80, 0.69))

    rounded_box(ax, (0.30, 0.18), 0.17, 0.18, "Protected\nfuture reserve")
    arrow(ax, (0.38, 0.58), (0.385, 0.36))

    rounded_box(ax, (0.56, 0.18), 0.17, 0.18, "Unused current-stage\ncapacity")
    arrow(ax, (0.635, 0.58), (0.645, 0.36))

    arrow(ax, (0.73, 0.27), (0.84, 0.58), linestyle="--")
    ax.text(0.755, 0.39, "release +\nreallocate", ha="center", va="center", fontsize=7)

    arrow(ax, (0.47, 0.27), (0.80, 0.61), linestyle="--")
    ax.text(0.62, 0.38, "preserved capacity", ha="center", va="center", fontsize=7)

    ax.text(
        0.03, 0.06,
        "Invariant: current-stage allocation + protected future reserve must remain within the remaining budget.",
        ha="left", va="center", fontsize=8
    )

    out = root / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    pdf = out / "figure4_rrr_mechanism_schematic.pdf"
    png = out / "figure4_rrr_mechanism_schematic.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf, png

def main():
    root = Path(__file__).resolve().parents[2]
    analysis_path = root / "results" / "stage_aware_confirmatory_v1" / "confirmatory_analysis_v1.json"
    raw = analysis_path.read_bytes()
    analysis_sha = sha256_bytes(raw)
    if analysis_sha != EXPECTED_ANALYSIS_SHA256:
        raise RuntimeError(f"Unexpected analysis SHA256: {analysis_sha}")
    data = json.loads(raw.decode("utf-8"))

    outputs = []
    outputs.extend(make_figure3(root, data))
    outputs.extend(make_figure4(root))

    print("SOURCE_ANALYSIS_SHA256=" + analysis_sha)
    for path in outputs:
        print(path.name + "_SHA256=" + sha256_bytes(path.read_bytes()))
    print("REMAINING_FIGURES_READY=TRUE")

if __name__ == "__main__":
    main()
