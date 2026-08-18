"""Center-category robustness for Model 3c (R&R item C5).

The user-level "center" label is a residual: users whose posts reach neither
the left nor the right threshold. This refits Model 3c with composition and
alignment recomputed over partisan users only, so that "center" enters as
unlabeled rather than as a third ideological position.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = ROOT / "results" / "reply_robustness"
FRAME = DATA / "combined_reply_stats_06.csv"
M3C = ("{y} ~ log_size * platform + bs(author_left_ratio,df=3) * log_size"
       " + log_size * bs(author_alignment_ratio,df=3)"
       " + author_right_ratio * log_size")
BASE = "{y} ~ log_size * platform"


def fit(frame, formula, y):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    need = ["log_size", "platform", y]
    if "left_ratio" in formula:
        need += ["author_left_ratio", "author_right_ratio", "author_alignment_ratio"]
    d = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=need)
    # Huber proposal-2 scale: the default MAD scale degenerates to ~0 because
    # 61% of cascades have size 1 and sit exactly at the origin.
    f = smf.rlm(formula.format(y=y), data=d, M=sm.robust.norms.HuberT()).fit(
        maxiter=350, scale_est=sm.robust.scale.HuberScale())
    k = [i for i in f.params.index if "log_size" in i and "platform" in i and "bs(" not in i][0]
    return float(f.params[k]), len(d)


def main():
    nodes = pd.read_parquet(DATA / "interim" / "trees_nodes.parquet")
    edges = pd.read_parquet(DATA / "interim" / "trees_edges.parquet")
    frame = pd.read_csv(FRAME, dtype={"index": str}, low_memory=False)

    lab = {}
    for plat, path in [("bsky", "bsky_author_ideology_portion.json"),
                       ("ts", "ts_author_ideology_portion_new.json")]:
        for u, p in json.load(open(DATA / path)).items():
            v = ("left" if (p.get("left") or 0) >= 0.6
                 else "right" if (p.get("right") or 0) >= 0.6 else "center")
            lab[f"{plat}|{u}"] = v
    lab_s = pd.Series(lab)

    nkey = nodes["platform"] + "|" + nodes["author"]
    nl = nkey.map(lab_s).fillna("center")
    n = pd.DataFrame({"platform": nodes.platform, "tree_id": nodes.tree_id,
                      "lab": nl.values})
    part = n[n.lab != "center"]
    comp = pd.DataFrame({
        "r_left_p": part.assign(x=part.lab == "left").groupby(["platform", "tree_id"])["x"].mean(),
        "n_partisan": part.groupby(["platform", "tree_id"])["lab"].size(),
    })

    ep = (edges["platform"] + "|" + edges["parent_author"]).map(lab_s).fillna("center")
    ec = (edges["platform"] + "|" + edges["child_author"]).map(lab_s).fillna("center")
    e = pd.DataFrame({"platform": edges.platform, "tree_id": edges.tree_id,
                      "p": ep.values, "c": ec.values})
    e = e[(e.p != "center") & (e.c != "center")]
    align = e.assign(same=e.p == e.c).groupby(["platform", "tree_id"])["same"].mean().rename("r_align_p")

    rat = comp.join(align, how="left")
    frame = frame[frame["size"] > 1]
    j = frame.merge(rat, left_on=["platform", "index"], right_index=True, how="inner")
    print(f"matched {len(j)}/{len(frame)}; trees with >=1 partisan-partisan edge: "
          f"{j['r_align_p'].notna().sum()}")

    rows = []
    for label, cols in {
        "published (center as third position)":
            ("author_left_ratio", "author_right_ratio", "author_alignment_ratio"),
        "partisan-only (center unlabeled)": ("__L", "__R", "__A"),
    }.items():
        t = j.copy()
        if cols[0] == "__L":
            t["author_left_ratio"] = t["r_left_p"]
            t["author_right_ratio"] = 1 - t["r_left_p"]
            t["author_alignment_ratio"] = t["r_align_p"]
            t = t.dropna(subset=["author_left_ratio", "author_alignment_ratio"])
        for y in ["log_breadth", "log_depth"]:
            b0, _ = fit(t, BASE, y)
            b3, nn = fit(t, M3C, y)
            rows.append(dict(check="C5_center", variant=label, y=y, baseline_b3=b0,
                             model3c_b3=b3, n=nn,
                             attenuation=1 - abs(b3) / abs(b0)))
    r = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "c5_center.csv", index=False)
    print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
