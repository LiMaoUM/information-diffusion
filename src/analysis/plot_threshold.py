"""Figure A3: user ideology shares across classification thresholds.

Replaces the earlier version generated in src/analysis/analysis.ipynb, which
classified users with a strict ">" cutoff. Every model in the paper, and the
shares reported in the text, use ">=" (see labels_at in
robustness_tables_huber.py), so the figure disagreed with the text by about
2.5 points on the Bluesky center share.

Run: uv run --with numpy --with pandas --with matplotlib \
       python src/analysis/plot_threshold.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from noise_propagation import load_portions  # noqa: E402

OUT = ROOT / "figures" / "threshold.png"
TITLES = {"bsky": "BlueSky", "ts": "TruthSocial"}
SERIES = [("left", "Left", "#1f77b4"), ("center", "Center", "#ff7f0e"), ("right", "Right", "#2ca02c")]


def shares(portions, thresholds):
    """Share of users in each category at each threshold, using the >= rule."""
    out = {}
    for plat, users in portions.items():
        rows = {k: [] for k, _, _ in SERIES}
        for thr in thresholds:
            c = Counter()
            for p in users.values():
                left, right = p.get("left") or 0, p.get("right") or 0
                c["left" if left >= thr else "right" if right >= thr else "center"] += 1
            n = sum(c.values())
            for k, _, _ in SERIES:
                rows[k].append(c[k] / n)
        out[plat] = rows
        print(f"{plat}: n={n}, at 0.6 " + ", ".join(
            f"{k}={rows[k][list(thresholds).index(0.6) if 0.6 in list(thresholds) else 5]:.3%}"
            for k, _, _ in SERIES))
    return out


def main():
    thresholds = np.round(np.arange(0.5, 0.71, 0.02), 2)
    data = shares(load_portions(), thresholds)
    fig, axes = plt.subplots(2, 1, figsize=(10, 11), dpi=300, sharex=True)
    for ax, plat in zip(axes, ["bsky", "ts"]):
        for key, label, color in SERIES:
            ax.plot(thresholds, data[plat][key], marker="o", color=color, label=label, lw=2)
        ax.set_title(TITLES[plat], fontsize=18, fontweight="bold")
        ax.set_ylabel("Proportion", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.4)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12)
    axes[1].set_xlabel("Threshold", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
