"""Does each ratio actually need a spline? Two checks the appendix relies on.

1. Partial residuals using the variable's FULL contribution (main effect plus
   its log_size interaction), not just the main effect. The notebook plotted
   only the main effect, which is the term at log_size = 0, where 61% of
   cascades sit with no replies at all.
2. Linear vs bs(df=3) refit per variable, comparing MAE and MedAE, which is
   the criterion the notebook actually used to decide.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
from statsmodels.nonparametric.smoothers_lowess import lowess

ROOT = Path(__file__).resolve().parents[2]
FRAME = ROOT / "data" / "combined_reply_stats_06.csv"
OUT = ROOT / "results" / "reply_robustness" / "spline_evidence.csv"
VARS = ["author_left_ratio", "author_right_ratio", "author_alignment_ratio"]
BASE = ("log_depth ~ log_size * platform + log_size * author_left_ratio"
        " + log_size * author_right_ratio + log_size * author_alignment_ratio")


def load():
    cols = ["log_depth", "log_size", "platform"] + VARS
    d = pd.read_csv(FRAME, usecols=cols, low_memory=False)
    return d.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)


def rlm(formula, d):
    y, X = dmatrices(formula, data=d, return_type="dataframe")
    return sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit(
        maxiter=200, scale_est=sm.robust.scale.HuberScale()), X


def main():
    d = load()
    print(f"n = {len(d):,}\n")

    res, X = rlm(BASE, d)
    resid = np.asarray(res.resid).ravel()
    ls = X["log_size"].to_numpy(float)

    print("Curvature of the FULL contribution (main effect + log_size interaction):")
    for v in VARS:
        b = res.params.iloc[X.columns.get_loc(v)]
        ik = [c for c in X.columns if c.startswith("log_size:") and c.endswith(v)]
        bi = res.params.iloc[X.columns.get_loc(ik[0])] if ik else 0.0
        x = X[v].to_numpy(float)
        contrib = b * x + bi * ls * x
        pr = resid + contrib
        s = lowess(pr, x, frac=0.25, it=0, delta=0.01 * np.ptp(x))
        # the linear contribution at mean log_size, which is what a straight
        # term in x would trace through this cloud
        lin = (b + bi * ls.mean()) * s[:, 0]
        dev = np.abs(s[:, 1] - lin)
        interior = (s[:, 0] > 0.05) & (s[:, 0] < 0.95)
        print(f"  {v:24s} max|dev|={dev.max():.4f}  "
              f"interior max|dev|={dev[interior].max():.4f}")

    print("\nLinear vs spline refit (the criterion the notebook used):")
    rows = []
    for v in VARS:
        others = [o for o in VARS if o != v]
        stem = "log_depth ~ log_size * platform" + "".join(f" + log_size * {o}" for o in others)
        for tag, term in [("linear", v), ("spline", f"bs({v}, df=3)")]:
            r, _ = rlm(f"{stem} + log_size * {term}", d)
            e = np.abs(np.asarray(r.resid).ravel())
            rows.append(dict(variable=v, form=tag, mae=e.mean(), medae=np.median(e),
                             rmse=float(np.sqrt((e ** 2).mean()))))
            print(f"  {v:24s} {tag:6s} MAE={rows[-1]['mae']:.5f} "
                  f"MedAE={rows[-1]['medae']:.5f}", flush=True)
    t = pd.DataFrame(rows)
    for v in VARS:
        a = t[(t.variable == v) & (t.form == "linear")].iloc[0]
        b = t[(t.variable == v) & (t.form == "spline")].iloc[0]
        print(f"  -> {v:24s} spline improves MAE by {100*(1-b.mae/a.mae):5.2f}%, "
              f"MedAE by {100*(1-b.medae/a.medae):5.2f}%  "
              f"({'spline' if (b.mae <= .95*a.mae or b.medae <= .95*a.medae) else 'linear'} by the 5% rule)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
