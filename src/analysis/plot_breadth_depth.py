"""Figure 3: breadth and depth against cascade size, by platform.

Replaces the earlier version generated in src/story.py, which fitted each
platform separately with statsmodels' default MAD scale. That scale degenerates
here, because 61.3% of reply cascades are single posts sitting at the origin, so
the printed standard errors collapsed to zero and the slopes disagreed with the
regression table.

The lines here come from the same pooled models the paper reports, so the
printed slopes match them exactly:
  reply panels  log y ~ log size * platform, Huber loss with Huber's proposal 2
                scale, on the modeling frame behind Table 2 (N = 123,189);
  repost panels the same specification by OLS with HC3 errors, as in the
                repost-reconstruction appendix, where the Huber scale
                degenerates on lattice-valued repost trees.
Per-platform slopes are read off the pooled fit: Bluesky is the size
coefficient, Truth Social adds the platform-by-size interaction, and its
standard error comes from the covariance of the two terms.

Run: uv run --with numpy --with pandas --with statsmodels --with matplotlib \
       python src/analysis/plot_breadth_depth.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "figures" / "breadth_depth.png"
COLORS = {"bsky": "#5F9EA0", "ts": "#FF6347"}
LABELS = {"bsky": "BlueSky", "ts": "TruthSocial"}


def pooled_fit(df, ycol, robust):
    """Pooled fit of log y on log size by platform, as reported in the paper."""
    f = f"{ycol} ~ log_size * platform"
    if robust:
        res = smf.rlm(f, data=df, M=sm.robust.norms.HuberT()).fit(
            maxiter=350, scale_est=sm.robust.scale.HuberScale())
    else:
        res = smf.ols(f, data=df).fit(cov_type="HC3")
    b0 = res.params["Intercept"]
    b_size = res.params["log_size"]
    k_plat = [i for i in res.params.index if i.startswith("platform[T.")][0]
    k_int = [i for i in res.params.index if "log_size:" in i][0]
    cov = res.cov_params()
    se_ts = float(np.sqrt(cov.loc["log_size", "log_size"] + cov.loc[k_int, k_int]
                          + 2 * cov.loc["log_size", k_int]))
    return {
        "bsky": (b0, b_size, float(res.bse["log_size"])),
        "ts": (b0 + res.params[k_plat], b_size + res.params[k_int], se_ts),
        "interaction": (float(res.params[k_int]), float(res.bse[k_int])),
    }


def reply_frame():
    d = pd.read_csv(DATA / "combined_reply_stats_06.csv", low_memory=False)
    return d[["platform", "log_size", "log_breadth", "log_depth"]].dropna()


def repost_frame():
    parts = []
    for plat, name in [("bsky", "bsky_repost_stats.csv"), ("ts", "ts_repost_stats.csv")]:
        d = pd.read_csv(DATA / name, low_memory=False)
        d = d[(d["size"] > 1) & (d["breadth"] > 0) & (d["max_depth"] >= 0)].copy()
        d["platform"] = plat
        d["log_size"] = np.log10(d["size"])
        d["log_breadth"] = np.log10(d["breadth"])
        d["log_depth"] = np.log10(d["max_depth"] + 1)
        parts.append(d[["platform", "log_size", "log_breadth", "log_depth"]])
    return pd.concat(parts, ignore_index=True).dropna()


def panel(ax, df, ycol, ylabel, letter, robust, legend=False):
    fit = pooled_fit(df, ycol, robust)
    rows = []
    for plat in ["ts", "bsky"]:   # denser cloud first, so both stay visible
        g = df[df["platform"] == plat]
        ax.scatter(g["log_size"], g[ycol], s=8, alpha=0.08, color=COLORS[plat],
                   edgecolors="none", rasterized=True)
    for plat in ["bsky", "ts"]:
        g = df[df["platform"] == plat]
        a, b, se = fit[plat]
        xs = np.linspace(g["log_size"].min(), g["log_size"].max(), 100)
        ax.plot(xs, a + b * xs, color=COLORS[plat], lw=2.5, label=LABELS[plat])
        rows.append((LABELS[plat], b, se))
    ax.text(0.03, 0.97, "\n".join(f"{n} Slope: {b:.4f} ± {se:.4f}" for n, b, se in rows),
            transform=ax.transAxes, va="top", ha="left", fontsize=13)
    ax.text(0.97, 0.97, f"({letter})", transform=ax.transAxes, va="top", ha="right",
            fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.set_xlabel("Cascade Size", fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=12)
    if legend:
        ax.legend(loc="center left", fontsize=13, frameon=False)
    print(f"  ({letter}) " + "; ".join(f"{n} {b:+.4f} ({se:.4f})" for n, b, se in rows)
          + f"; interaction {fit['interaction'][0]:+.4f} ({fit['interaction'][1]:.4f})")


def main():
    reply, repost = reply_frame(), repost_frame()
    print(f"reply N={len(reply)} {reply.platform.value_counts().to_dict()}")
    print(f"repost N={len(repost)} {repost.platform.value_counts().to_dict()}")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6), dpi=300)
    panel(axes[0, 0], reply, "log_breadth", "Cascade Max Breadth", "A", robust=True, legend=True)
    panel(axes[0, 1], reply, "log_depth", "Cascade Depth", "B", robust=True)
    panel(axes[1, 0], repost, "log_breadth", "Cascade Max Breadth", "C", robust=False)
    panel(axes[1, 1], repost, "log_depth", "Cascade Depth", "D", robust=False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
