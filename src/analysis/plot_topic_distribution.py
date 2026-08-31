"""Regenerate Figure 1, the per-platform distribution of root posts across topics.

Reproduces the published figure from the modeling frame rather than from a
notebook cell. The only substantive change is x-axis headroom: value labels
are annotated to the right of each bar, so without it the label on the longest
bar in each panel falls outside the axes box.

Run:
  uv run --with numpy --with pandas --with matplotlib --with seaborn \
      python src/analysis/plot_topic_distribution.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
FRAME = ROOT / "data" / "combined_reply_stats_06.csv"
OUT = ROOT / "paper" / "Revision" / "figures" / "topic_dis.png"

PLATFORMS = [("bsky", "BlueSky"), ("ts", "Truth Social")]
HEADROOM = 1.10   # initial guess; widened below to whatever the labels actually need


def counts():
    d = pd.read_csv(FRAME, usecols=["platform", "topic_label", "index"],
                    dtype={"index": str}, low_memory=False)
    d = d.dropna(subset=["topic_label"])
    per = d.groupby(["topic_label", "platform"])["index"].nunique().unstack(fill_value=0)
    order = per.sum(axis=1).sort_values(ascending=False).index
    return per.reindex(order)


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 13,
        "axes.titlesize": 20,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })
    per = counts()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, (key, title) in zip(axes, PLATFORMS):
        vals = per[key]
        sns.barplot(x=vals.values, y=vals.index, ax=ax, palette="tab20", hue=vals.index,
                    legend=False, dodge=False)
        ax.set_title(title)
        ax.set_xlabel("Number of Unique Root Posts")
        ax.set_ylabel("Topic" if key == "bsky" else "")
        # headroom first, so the annotations below land inside the axes
        ax.set_xlim(0, vals.max() * HEADROOM)
        for j, v in enumerate(vals.values):
            ax.annotate(f"{v}", xy=(v, j), xytext=(5, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=13)

    plt.tight_layout()

    def label_extent(ax):
        """Rightmost data coordinate reached by any value label in this panel."""
        r = fig.canvas.get_renderer()
        return max(ax.transData.inverted().transform(
            [(t.get_window_extent(r).x1, 0)])[0, 0] for t in ax.texts)

    # Widen each panel until its longest value label sits inside the box. The
    # label width is fixed in points, so it depends on the axes width in a way
    # that is only known after layout; measure and expand instead of guessing.
    for _ in range(6):
        fig.canvas.draw()
        grew = False
        for ax in axes:
            need = label_extent(ax) * 1.02
            if need > ax.get_xlim()[1]:
                ax.set_xlim(0, need)
                grew = True
        if not grew:
            break

    fig.canvas.draw()
    for ax, (key, _) in zip(axes, PLATFORMS):
        hi = ax.get_xlim()[1]
        end = label_extent(ax)
        assert end <= hi, f"{key}: a label ends at {end:.0f} past xlim {hi:.0f}"
    print("QA: all value labels inside the axes box")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
