from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "model_architecture.png"


def add_box(ax, x, y, w, h, title, body, fc, ec="#1f2937"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.8,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        x + w / 2,
        y + h * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#374151",
        linespacing=1.4,
    )


def add_arrow(ax, start, end, color="#475569"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.0,
        color=color,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


fig, ax = plt.subplots(figsize=(16, 9), dpi=220)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("#fffaf2")
ax.set_facecolor("#fffaf2")

ax.text(
    0.05,
    0.95,
    "AML Mule Detection System Architecture",
    fontsize=24,
    fontweight="bold",
    color="#111827",
    ha="left",
)
ax.text(
    0.05,
    0.915,
    "Two-pass ensemble pipeline with feature engineering, OOF network learning, and calibrated submission scoring",
    fontsize=11.5,
    color="#4b5563",
    ha="left",
)

boxes = {
    "data": (0.05, 0.67, 0.22, 0.17, "Input Data", "Accounts, customers,\ntransactions, branch,\nproduct, labels, test IDs", "#fde68a"),
    "feat": (0.32, 0.67, 0.24, 0.17, "Feature Engineering", "Base + advanced + geo/IP\ncounterparty + branch-label\nfeatures -> all_features_v4", "#bfdbfe"),
    "pass1": (0.62, 0.67, 0.16, 0.17, "Train Pass 1", "LightGBM\nXGBoost\nCatBoost", "#c7f9cc"),
    "oof": (0.81, 0.67, 0.14, 0.17, "OOF Scores", "Out-of-fold predictions\nfor training accounts", "#fecdd3"),
    "net": (0.18, 0.37, 0.24, 0.17, "Network Features", "OOF-based graph and\nneighbor-risk features", "#fbcfe8"),
    "pass2": (0.47, 0.37, 0.18, 0.17, "Train Pass 2", "Retrain final\nbase learners using\nupdated features", "#bbf7d0"),
    "ens": (0.70, 0.37, 0.22, 0.17, "Ensemble + Calibration", "Weighted blend\nscore transform\nthreshold selection", "#ddd6fe"),
    "out": (0.36, 0.10, 0.28, 0.17, "Final Outputs", "submission.csv\nmetrics.json\nfeature importance\nSHAP + XAI report", "#fed7aa"),
}

for _, (x, y, w, h, title, body, fc) in boxes.items():
    add_box(ax, x, y, w, h, title, body, fc)

add_arrow(ax, (0.27, 0.755), (0.32, 0.755))
add_arrow(ax, (0.56, 0.755), (0.62, 0.755))
add_arrow(ax, (0.78, 0.755), (0.81, 0.755))
add_arrow(ax, (0.84, 0.67), (0.42, 0.45))
add_arrow(ax, (0.42, 0.455), (0.47, 0.455))
add_arrow(ax, (0.65, 0.455), (0.70, 0.455))
add_arrow(ax, (0.81, 0.37), (0.64, 0.19))
add_arrow(ax, (0.32, 0.67), (0.25, 0.54))

ax.text(
    0.07,
    0.57,
    "Stage A",
    fontsize=11,
    fontweight="bold",
    color="#92400e",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffedd5", edgecolor="none"),
)
ax.text(
    0.39,
    0.57,
    "Stage B",
    fontsize=11,
    fontweight="bold",
    color="#1d4ed8",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="#dbeafe", edgecolor="none"),
)
ax.text(
    0.72,
    0.57,
    "Stage C",
    fontsize=11,
    fontweight="bold",
    color="#7c3aed",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="#ede9fe", edgecolor="none"),
)

ax.text(
    0.05,
    0.03,
    "Final production path: data -> feature engineering -> pass-1 OOF -> network features -> pass-2 ensemble -> calibrated submission",
    fontsize=10,
    color="#6b7280",
    ha="left",
)

plt.tight_layout()
plt.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved architecture diagram to {OUT_PATH}")
