"""Rebuild the partial-residual diagnostic (appendix "Spline Transformations").

The published figure was three separate screenshots pasted vertically, with
mismatched panel sizes and axis ranges, one clipped title, and panel letters
that did not match the caption. It also carried no smoother: the notebook
function was called partial_residual_plot_with_lowess, but the LOWESS call was
never written, so the curvature the appendix text describes was not shown.

This version draws all three panels in one row on a shared y-axis and overlays,
for each predictor, the linear term actually fitted (a straight line through
the origin with slope beta) against a LOWESS curve of the partial residuals.
Where the two separate, a linear term is inadequate and a spline is warranted,
which is exactly the claim the appendix makes.

Run:
  uv run --with numpy --with pandas --with statsmodels --with patsy \
      --with matplotlib python src/analysis/plot_partial_residuals.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from statsmodels.nonparametric.smoothers_lowess import lowess

ROOT = Path(__file__).resolve().parents[2]
FRAME = ROOT / "data" / "combined_reply_stats_06.csv"
OUT = ROOT / "paper" / "Revision" / "figures" / "partial.png"

# The baseline the appendix diagnoses: every ratio entered linearly.
FORMULA = ("log_depth ~ log_size * platform"
           " + log_size * author_left_ratio"
           " + log_size * author_right_ratio"
           " + log_size * author_alignment_ratio")

# Panel order follows the caption: (A) left, (B) right, (C) alignment.
PANELS = [
    ("author_left_ratio", "Left-leaning share of authors"),
    ("author_right_ratio", "Right-leaning share of authors"),
    ("author_alignment_ratio", "Alignment ratio"),
]

SCATTER_N = 20000     # subsample only what is drawn; every point enters the fit
RNG = np.random.default_rng(0)


def fit():
    cols = ["log_depth", "log_size", "platform", "author_left_ratio",
            "author_right_ratio", "author_alignment_ratio"]
    d = pd.read_csv(FRAME, usecols=cols, low_memory=False)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
    y, X = dmatrices(FORMULA, data=d, return_type="dataframe")
    # Huber's proposal 2 scale, matching the main models: the default MAD scale
    # degenerates here because 61% of cascades sit at the origin.
    res = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit(
        maxiter=200, scale_est=sm.robust.scale.HuberScale())
    return res, X, len(d)


def main():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 9,
    })
    res, X, n = fit()
    resid = np.asarray(res.resid).ravel()

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    ls = X["log_size"].to_numpy(float)
    for ax, letter, (var, label) in zip(axes, "ABC", PANELS):
        # Each ratio enters both on its own and interacted with log_size, so the
        # variable's contribution is beta*x + beta_int*log_size*x. Plotting only
        # the main effect, as the notebook did, evaluates the term at
        # log_size = 0, where 61% of cascades sit with no replies at all.
        beta = res.params.iloc[X.columns.get_loc(var)]
        ik = [c for c in X.columns if c.startswith("log_size:") and c.endswith(var)]
        beta_int = res.params.iloc[X.columns.get_loc(ik[0])] if ik else 0.0
        x = X[var].to_numpy(float)
        pr = resid + beta * x + beta_int * ls * x
        beta = beta + beta_int * ls.mean()   # slope a straight term would trace

        idx = RNG.choice(len(x), min(SCATTER_N, len(x)), replace=False)
        ax.scatter(x[idx], pr[idx], s=3, alpha=0.07, color="#4878A8",
                   linewidths=0, rasterized=True)

        grid = np.linspace(x.min(), x.max(), 100)
        ax.plot(grid, beta * grid, color="#B0B0B0", lw=1.8, ls="--", zorder=3,
                label="linear term at mean cascade size")
        sm_xy = lowess(pr, x, frac=0.25, it=0, delta=0.01 * np.ptp(x))
        ax.plot(sm_xy[:, 0], sm_xy[:, 1], color="#C44E52", lw=2.2, zorder=4,
                label="LOWESS of partial residuals")

        ax.axhline(0, color="0.35", lw=0.7, zorder=2)
        ax.set_xlabel(label)
        ax.set_title(f"{letter}", loc="left", fontweight="bold")
        ax.set_xlim(-0.03, 1.03)

    axes[0].set_ylabel("Partial residual")
    axes[0].set_ylim(-0.9, 1.1)
    axes[0].legend(loc="upper left", frameon=False, handlelength=1.6)

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}  (n = {n:,})")

    # Report the gap between the linear term and the smoother, so the appendix
    # text can state which predictors actually need a spline.
    for var, label in PANELS:
        b = res.params.iloc[X.columns.get_loc(var)]
        ik = [c for c in X.columns if c.startswith("log_size:") and c.endswith(var)]
        bi = res.params.iloc[X.columns.get_loc(ik[0])] if ik else 0.0
        x = X[var].to_numpy(float)
        pr = resid + b * x + bi * ls * x
        beta = b + bi * ls.mean()
        s = lowess(pr, x, frac=0.25, it=0, delta=0.01 * np.ptp(x))
        dev = s[:, 1] - beta * s[:, 0]
        print(f"{var:24s} beta={beta:+.4f}  max|LOWESS - linear|={np.abs(dev).max():.4f}"
              f"  at x={s[np.abs(dev).argmax(), 0]:.2f}")


if __name__ == "__main__":
    main()
