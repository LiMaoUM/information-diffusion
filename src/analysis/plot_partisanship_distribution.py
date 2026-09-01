"""Rebuild Figure 7, root-post partisanship by platform.

Two defects in the published version: the value labels are annotated to the
right of each bar, so the longest bar's label fell outside the axes box, and a
colour legend repeated what the y-axis already said, sitting inside the plot
area. Panel titles also used the internal codes BSKY and TS.

The panels share one x-axis here. The two maxima are 46,000 and 37,639, close
enough that sharing costs almost no resolution and removes the risk of reading
one panel's bar length against the other's.

Run:
  uv run --with numpy --with pandas --with matplotlib \
      python src/analysis/plot_partisanship_distribution.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FRAME = ROOT / "data" / "combined_reply_stats_06.csv"
OUT = ROOT / "paper" / "Revision" / "figures" / "party_dis.png"

PLATFORMS = [("bsky", "Bluesky"), ("ts", "Truth Social")]
ORDER = ["left", "center", "right"]
LABELS = {"left": "Left", "center": "Center", "right": "Right"}
COLORS = {"left": "#2E6DA4", "center": "#B8CCE4", "right": "#E08214"}


def counts():
    d = pd.read_csv(FRAME, usecols=["platform", "partisanship", "index"],
                    dtype={"index": str}, low_memory=False)
    d["partisanship"] = d["partisanship"].replace({"error": "center"})
    return d.groupby(["platform", "partisanship"])["index"].nunique().unstack(fill_value=0)


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 13, "axes.titlesize": 18, "axes.labelsize": 15,
        "xtick.labelsize": 12.5, "ytick.labelsize": 13.5,
    })
    per = counts()
    top = per.to_numpy().max()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True, sharex=True)
    for ax, (key, title) in zip(axes, PLATFORMS):
        vals = [per.loc[key, s] for s in ORDER]
        ax.barh(range(len(ORDER)), vals,
                color=[COLORS[s] for s in ORDER], height=0.62)
        ax.set_yticks(range(len(ORDER)), [LABELS[s] for s in ORDER])
        ax.set_title(title)
        ax.set_xlabel("Number of Root Posts")
        for j, v in enumerate(vals):
            ax.annotate(f"{v:,}", xy=(v, j), xytext=(6, 0),
                        textcoords="offset points", va="center", ha="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # once, not per-axis: with sharey a second call flips it back
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Stance of root-post author")
    axes[0].set_xlim(0, top * 1.10)
    plt.tight_layout()

    def label_end(ax):
        r = fig.canvas.get_renderer()
        return max(ax.transData.inverted().transform(
            [(t.get_window_extent(r).x1, 0)])[0, 0] for t in ax.texts)

    for _ in range(6):                       # widen until every label fits
        fig.canvas.draw()
        need = max(label_end(a) for a in axes) * 1.02
        if need <= axes[0].get_xlim()[1]:
            break
        axes[0].set_xlim(0, need)            # shared axis: setting one sets both

    fig.canvas.draw()
    for ax, (key, _) in zip(axes, PLATFORMS):
        assert label_end(ax) <= ax.get_xlim()[1], f"{key}: label outside the axes"
    print("QA: all value labels inside the axes box")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(per.loc[[p for p, _ in PLATFORMS], ORDER].to_string())


if __name__ == "__main__":
    main()
