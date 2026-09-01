"""Does the platform-by-size interaction depend on WHERE the splines go?

The appendix justifies a spline on the left-user ratio and a linear term on the
right-user ratio. The partial-residual diagnostic points the other way, so this
refits Model 3c with the right-user ratio splined as well and reports whether
beta_3, the quantity every conclusion rests on, moves.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
FRAME = ROOT / "data" / "combined_reply_stats_06.csv"
OUT = ROOT / "results" / "reply_robustness" / "spline_placement.csv"

SPECS = {
    "as published (spline L + align, linear R)":
        "{y} ~ log_size * platform + bs(author_left_ratio,df=3) * log_size"
        " + log_size * bs(author_alignment_ratio,df=3)"
        " + author_right_ratio * log_size",
    "spline on all three":
        "{y} ~ log_size * platform + bs(author_left_ratio,df=3) * log_size"
        " + log_size * bs(author_alignment_ratio,df=3)"
        " + bs(author_right_ratio,df=3) * log_size",
    "linear on all three":
        "{y} ~ log_size * platform + author_left_ratio * log_size"
        " + log_size * author_alignment_ratio"
        " + author_right_ratio * log_size",
    "baseline (no ideology)": "{y} ~ log_size * platform",
}


def main():
    cols = ["log_depth", "log_breadth", "log_size", "platform",
            "author_left_ratio", "author_right_ratio", "author_alignment_ratio"]
    d = pd.read_csv(FRAME, usecols=cols, low_memory=False)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)

    rows = []
    for y in ["log_breadth", "log_depth"]:
        for name, f in SPECS.items():
            r = smf.rlm(f.format(y=y), data=d, M=sm.robust.norms.HuberT()).fit(
                maxiter=200, scale_est=sm.robust.scale.HuberScale())
            k = [i for i in r.params.index
                 if "log_size" in i and "platform" in i and "bs(" not in i][0]
            e = np.abs(np.asarray(r.resid).ravel())
            rows.append(dict(outcome=y, spec=name, b3=float(r.params[k]),
                             mae=e.mean(), n=len(d)))
            print(f"{y:12s} {name:42s} b3={rows[-1]['b3']:+.4f} MAE={e.mean():.5f}",
                  flush=True)

    t = pd.DataFrame(rows)
    for y in ["log_breadth", "log_depth"]:
        s = t[t.outcome == y].set_index("spec")
        base = s.loc["baseline (no ideology)", "b3"]
        print(f"\n{y}: baseline b3 = {base:+.4f}")
        for name in SPECS:
            if name == "baseline (no ideology)":
                continue
            print(f"   {name:42s} b3={s.loc[name,'b3']:+.4f}  "
                  f"absorbed {100*(1-abs(s.loc[name,'b3']/base)):.1f}%")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
