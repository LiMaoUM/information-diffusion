"""Re-estimate the threshold and encoding robustness tables under the same
scale estimator used for the main table (Huber's proposal 2).

Both tables previously used the default MAD scale, which degenerates on these
data, so they were inconsistent with the main specification.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from noise_propagation import tree_ratios, load_portions   # noqa: E402

FRAME = ROOT / "data" / "combined_reply_stats_06.csv"
CACHE = ROOT / "data" / "interim"
OUT = ROOT / "results" / "reply_robustness"
BASE = "{y} ~ log_size * platform"
M3C = ("{y} ~ log_size * platform + bs({comp},df=3) * log_size"
       " + log_size * bs(align,df=3) + {comp2} * log_size")


def labels_at(portions, thr):
    lab = {}
    for plat, d in portions.items():
        for u, p in d.items():
            lab[(plat, u)] = ("left" if (p.get("left") or 0) >= thr
                              else "right" if (p.get("right") or 0) >= thr else "center")
    return lab


def fit(d, formula, y):
    dd = d.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[c for c in ["log_size", "platform", "comp", "comp2", "align", y] if c in d.columns])
    f = smf.rlm(formula.format(y=y, comp="comp", comp2="comp2"), data=dd,
                M=sm.robust.norms.HuberT()).fit(maxiter=350,
                                                scale_est=sm.robust.scale.HuberScale())
    k = [i for i in f.params.index if "log_size" in i and "platform" in i and "bs(" not in i][0]
    resid = f.resid
    tss = float(np.sum((dd[y] - dd[y].mean()) ** 2))
    ksz = [i for i in f.params.index if i == "log_size"]
    return dict(coef=float(f.params[k]),
                b_size=float(f.params[ksz[0]]) if ksz else float("nan"),
                pseudo_r2=1 - float(np.sum(resid ** 2)) / tss,
                mae=float(np.mean(np.abs(resid))),
                rmse=float(np.sqrt(np.mean(resid ** 2))), n=len(dd))


def main():
    nodes = pd.read_parquet(CACHE / "trees_nodes.parquet")
    edges = pd.read_parquet(CACHE / "trees_edges.parquet")
    frame = pd.read_csv(FRAME, dtype={"index": str}, low_memory=False)
    portions = load_portions()

    rows = []
    for thr in [0.5, 0.6, 0.7]:
        rat = tree_ratios(nodes, edges, labels_at(portions, thr))
        m = frame.merge(rat, left_on=["platform", "index"], right_index=True, how="inner")
        m["comp"], m["comp2"], m["align"] = m["r_left"], m["r_right"], m["r_align"].fillna(m["author_alignment_ratio"])
        for y in ["log_breadth", "log_depth"]:
            r = fit(m, M3C, y)
            rows.append(dict(table="threshold", spec=f"{thr}", y=y, **r))
            print(f"threshold {thr} {y}: coef={r['coef']:+.4f} R2={r['pseudo_r2']:.3f} "
                  f"MAE={r['mae']:.3f} RMSE={r['rmse']:.3f} n={r['n']}", flush=True)

    # encoding comparison at the default threshold: L/R versus majority/minority
    rat = tree_ratios(nodes, edges, labels_at(portions, 0.6))
    m = frame.merge(rat, left_on=["platform", "index"], right_index=True, how="inner")
    m["align"] = m["r_align"].fillna(m["author_alignment_ratio"])
    for enc in ["LR", "MajMin"]:
        if enc == "LR":
            m["comp"], m["comp2"] = m["r_left"], m["r_right"]
        else:   # majority = platform-dominant ideology
            maj = np.where(m["platform"] == "ts", m["r_right"], m["r_left"])
            mino = np.where(m["platform"] == "ts", m["r_left"], m["r_right"])
            m["comp"], m["comp2"] = maj, mino
        for y in ["log_breadth", "log_depth"]:
            r = fit(m, M3C, y)
            rows.append(dict(table="encoding", spec=enc, y=y, **r))
            print(f"encoding {enc} {y}: coef={r['coef']:+.4f} R2={r['pseudo_r2']:.3f} "
                  f"MAE={r['mae']:.3f} RMSE={r['rmse']:.3f} n={r['n']}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "robustness_tables_huber.csv", index=False)


if __name__ == "__main__":
    main()
