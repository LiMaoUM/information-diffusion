"""Draft spec-curve (forest) figure for repost reconstruction robustness.

Spec: results/repost_robustness/figure-spec.md. Final version goes through the
nature-figure workflow after spec confirmation.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "results" / "repost_robustness"

# reply-cascade reference (same estimator, src/df.csv, computed 2026-07-18)
REPLY_B3 = {"breadth": 0.1703, "depth": -0.1817}

ROWS = [
    ("code", "api", "published rule (code replica)"),
    ("first", "api", "first prior followee"),
    ("first", "rev", "first prior followee, reversed order"),
    ("last", "api", "last prior followee"),
    ("last", "rev", "last prior followee, reversed order"),
]

r = pd.read_csv(OUT / "b3_by_rule.csv")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, y in zip(axes, ["breadth", "depth"]):
    sub = r[r["y"] == y]
    labels, ypos = [], []
    for k, (rule, order, label) in enumerate(ROWS):
        row = sub[(sub["rule"] == rule) & (sub["order"] == order)].iloc[0]
        ax.errorbar(
            row["b3"], k,
            xerr=[[row["b3"] - row["lo"]], [row["hi"] - row["b3"]]],
            fmt="o", color="#0072B2", ms=5, capsize=2.5, lw=1.4,
        )
        labels.append(label)
        ypos.append(k)
    rand = sub[sub["rule"] == "random"]
    k = len(ROWS)
    ax.barh(k, rand["b3"].max() - rand["b3"].min(), left=rand["b3"].min(),
            height=0.55, color="#0072B2", alpha=0.25, edgecolor="none")
    labels.append("random rule ensemble (40 draws)")
    ypos.append(k)

    k += 1
    ax.plot(REPLY_B3[y], k, marker="D", color="#D55E00", ms=6, ls="none")
    labels.append("reply cascades (reference)")
    ypos.append(k)

    ax.axvline(0, color="#666666", lw=0.8, zorder=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel(r"platform $\times$ log size interaction $\beta_3$"
                  f"  (log {y})", fontsize=9)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_title("A" if y == "breadth" else "B", loc="left",
                 fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

fig.suptitle("Repost cascade scaling: platform interaction under alternative "
             "reconstruction rules (DRAFT)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "spec_curve_draft.png", dpi=300, bbox_inches="tight")
print("wrote", OUT / "spec_curve_draft.png")
