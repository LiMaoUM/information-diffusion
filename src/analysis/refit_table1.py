"""Refit the main results table under a non-degenerate scale estimator (C12).

Table 1 as submitted reports a Huber RLM fitted with the default MAD scale.
Because 61% of reply cascades are single posts sitting exactly at the origin,
that scale estimate collapses toward zero, which makes the standard errors and
significance stars uninterpretable. This refits every model with Huber's
proposal 2 scale, which does not degenerate here, and reports both so the two
can be compared.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "reply_robustness"
FRAME = ROOT / "data" / "combined_reply_stats_06.csv"

MODELS = {
    "1 (Baseline)": "{y} ~ log_size * platform",
    "2 (Influencers)": "{y} ~ log_size * outlier",
    "3a (Alignment)": "{y} ~ log_size * platform + log_size * bs(author_alignment_ratio,df=3)",
    "3b (Composition)": "{y} ~ log_size * platform + author_right_ratio * log_size"
                        " + bs(author_left_ratio,df=3) * log_size",
    "3c (Combined)": "{y} ~ log_size * platform + bs(author_left_ratio,df=3) * log_size"
                     " + log_size * bs(author_alignment_ratio,df=3)"
                     " + author_right_ratio * log_size",
    "4 (Topic)": "{y} ~ log_size * platform + log_size * topic_label",
}


def fit(d, formula, y, scale):
    kw = {} if scale == "MAD" else {"scale_est": sm.robust.scale.HuberScale()}
    f = smf.rlm(formula.format(y=y), data=d, M=sm.robust.norms.HuberT()).fit(
        maxiter=350, **kw)
    # interaction of interest: log_size x platform (or x outlier for Model 2)
    keys = [i for i in f.params.index
            if "log_size" in i and (":" in i)
            and ("platform" in i or "outlier" in i) and "bs(" not in i]
    k = keys[0]
    resid = f.resid
    tss = float(np.sum((d[y] - d[y].mean()) ** 2))
    return dict(term=k, coef=float(f.params[k]), se=float(f.bse[k]),
                pval=float(f.pvalues[k]),
                main=float(f.params["log_size"]),
                main_se=float(f.bse["log_size"]),
                pseudo_r2=1 - float(np.sum(resid ** 2)) / tss,
                rmse=float(np.sqrt(np.mean(resid ** 2))),
                scale=float(f.scale), n=len(d))


def main():
    d = pd.read_csv(FRAME, dtype={"index": str}, low_memory=False)
    d = d.replace([np.inf, -np.inf], np.nan)
    rows = []
    for y in ["log_breadth", "log_depth"]:
        for name, formula in MODELS.items():
            need = ["log_size", y] + (
                ["outlier"] if "outlier" in formula else ["platform"])
            for v in ["author_left_ratio", "author_right_ratio",
                      "author_alignment_ratio", "topic_label"]:
                if v in formula:
                    need.append(v)
            sub = d.dropna(subset=need)
            for scale in ["MAD", "Huber"]:
                try:
                    r = fit(sub, formula, y, scale)
                except Exception as e:
                    r = dict(term="FAILED", coef=np.nan, se=np.nan, pval=np.nan,
                             main=np.nan, main_se=np.nan, pseudo_r2=np.nan,
                             rmse=np.nan, scale=np.nan, n=len(sub))
                    print(f"  {name} {y} {scale}: {e}")
                rows.append(dict(outcome=y, model=name, scale_est=scale, **r))
                print(f"{y:12s} {name:18s} {scale:5s} "
                      f"coef={r['coef']:+.4f} se={r['se']:.4f} "
                      f"scale={r['scale']:.2e} R2={r['pseudo_r2']:.4f}")
    res = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "table1_refit.csv", index=False)


if __name__ == "__main__":
    main()
