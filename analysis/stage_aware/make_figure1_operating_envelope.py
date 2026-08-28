
from pathlib import Path
import json
import hashlib
import matplotlib.pyplot as plt

SRC = Path(r"results/stage_aware_confirmatory_v1/exploratory_operating_envelope_v1.json")
OUTDIR = Path(r"results/figures")
EXPECTED_SHA = "7EFA1061657D0063BC6D7DC4ADDB5A807ED34456CC21EB50FF31F6B59C5F59C6"

blob = SRC.read_bytes()
assert hashlib.sha256(blob).hexdigest().upper() == EXPECTED_SHA
data = json.loads(blob)
assert data["status"] == "POSTHOC_EXPLORATORY_NOT_CONFIRMATORY_NOT_CAUSAL"
rows = data["settings"]

MODEL = "Qwen/Qwen3-8B"
REGIMES = ["constrained", "transition", "relaxed"]
REGIME_LABELS = ["Constrained", "Transition", "Relaxed"]
BENCHMARKS = ["hotpotqa", "quixbugs"]
lookup = {(r["model"], r["benchmark"], r["budget_condition"]): r for r in rows}

for b in BENCHMARKS:
    for c in REGIMES:
        assert (MODEL, b, c) in lookup

def series(benchmark, key):
    return [100.0 * float(lookup[(MODEL, benchmark, c)][key]) for c in REGIMES]

def delta_series(benchmark, metric):
    return [100.0 * float(lookup[(MODEL, benchmark, c)]["metrics"][metric]["delta_rrr_minus_legacy"]) for c in REGIMES]

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
    ax.set_xticklabels(REGIME_LABELS)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)

def panel(ax, label):
    ax.text(0.0, 1.02, label, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=9)

def annotate_nonzero(ax, xs, ys, offsets=None, threshold=0.000001):
    if offsets is None:
        offsets = [5] * len(xs)
    for xi, yi, off in zip(xs, ys, offsets):
        if abs(yi) <= threshold:
            continue
        ax.annotate(f"{yi:.1f}", (xi, yi), xytext=(0, off), textcoords="offset points",
                    ha="center", va="bottom" if off >= 0 else "top", fontsize=6.5)

# (a) deficit prevalence
ax = axes[0, 0]
ax.plot(x, hot_def, marker="o", linestyle="-", linewidth=1.4, label="HotpotQA")
ax.plot(x, qx_def, marker="s", linestyle="--", linewidth=1.4, label="QuixBugs")
setup(ax, "Deficit prevalence (%)")
ax.set_ylim(-5, 105)
ax.legend(frameon=False, loc="upper right")
panel(ax, "(a)")

# (b) mechanism
ax = axes[0, 1]
ax.plot(x, hot_floor, marker="o", linestyle="-", linewidth=1.2, label="HotpotQA: floor")
ax.plot(x, hot_len, marker="^", linestyle="-.", linewidth=1.2, label="HotpotQA: length")
ax.plot(x, qx_floor, marker="s", linestyle="--", linewidth=1.2, label="QuixBugs: floor")
ax.plot(x, qx_len, marker="D", linestyle=":", linewidth=1.2, label="QuixBugs: length")
setup(ax, "RRR event rate (%)")
ax.set_ylim(-5, 105)
ax.legend(frameon=False, loc="upper right", ncols=1)
panel(ax, "(b)")

# (c) reliable correctness delta
ax = axes[1, 0]
ax.axhline(0.0, linestyle="--", linewidth=0.9)
ax.plot(x, hot_q, marker="o", linestyle="-", linewidth=1.4, label="HotpotQA")
ax.plot(x, qx_q, marker="s", linestyle="--", linewidth=1.4, label="QuixBugs")
setup(ax, r"$\Delta$ reliable correctness (pp)")
ax.legend(frameon=False, loc="lower right")
panel(ax, "(c)")

# (d) downstream delta
ax = axes[1, 1]
ax.axhline(0.0, linestyle="--", linewidth=0.9)
ax.plot(x, hot_d, marker="o", linestyle="-", linewidth=1.4, label="HotpotQA")
ax.plot(x, qx_d, marker="s", linestyle="--", linewidth=1.4, label="QuixBugs")
setup(ax, r"$\Delta$ downstream completion (pp)")
ax.legend(frameon=False, loc="lower right")
panel(ax, "(d)")

OUTDIR.mkdir(parents=True, exist_ok=True)
pdf = OUTDIR / "figure1_qwen_operating_envelope.pdf"
png = OUTDIR / "figure1_qwen_operating_envelope.png"
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, dpi=600, bbox_inches="tight")
plt.close(fig)

print("FIGURE1_SOURCE_SHA256=" + hashlib.sha256(blob).hexdigest().upper())
print("FIGURE1_PDF=" + str(pdf))
print("FIGURE1_PNG=" + str(png))
print("FIGURE1_PDF_SHA256=" + hashlib.sha256(pdf.read_bytes()).hexdigest().upper())
print("FIGURE1_PNG_SHA256=" + hashlib.sha256(png.read_bytes()).hexdigest().upper())
print("FIGURE1_READY=TRUE")
