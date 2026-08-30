"""Defenses for the validation sample using only the data we have (R&R C4).

Three checks that address "200 items is too few" without collecting more:

1. Annotator choice. The per-class rates are computed against the 171 items
   where both annotators agree. This recomputes them against Annotator 1 alone
   and Annotator 2 alone, so the reader can see the conclusions do not depend
   on which annotator, or on restricting to consensus.
2. Aggregation gain. User ideology is a threshold over many posts, so the
   error rate that matters is the user-level one, which is far below the
   post-level rate the validation measures. Given the observed posts-per-user
   distribution and the measured post-level confusion, this simulates the
   implied user-level error, showing the noise analysis is conservative.
3. Direction of the bias. The paper's claim is that the classifier
   over-assigns each platform's minority. That is a directional claim and can
   be tested even where individual cells are imprecise: a one-sided bootstrap
   of the difference between minority and majority precision.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "ideology_validation"
CATS = ["left", "center", "right"]
B = 5000
RNG = np.random.default_rng(20260830)


def load():
    d = pd.read_csv(ROOT / "src" / "val_ideology.csv")
    for c in ["stance", "Ideology", "Ideology2"]:
        d[c] = d[c].astype(str).str.strip().str.lower()
    bsky = json.load(open(ROOT / "data" / "bsky_post_to_label.json"))
    d["platform"] = d["post"].map(lambda p: "bsky" if p in bsky else "ts")
    return d


def precision(truth, pred, cls):
    sel = pred == cls
    return (truth[sel] == cls).mean() if sel.sum() else np.nan


# ---------------------------------------------------- 1. annotator choice

def annotator_choice(d):
    rows = []
    refs = {"consensus": d[d.Ideology == d.Ideology2],
            "annotator 1": d, "annotator 2": d}
    cols = {"consensus": "Ideology", "annotator 1": "Ideology", "annotator 2": "Ideology2"}
    for name, sub in refs.items():
        truth = sub[cols[name]].values
        pred = sub["stance"].values
        for scope in ["all", "bsky", "ts"]:
            m = np.ones(len(sub), bool) if scope == "all" else (sub.platform == scope).values
            for cls in CATS:
                rows.append(dict(reference=name, scope=scope, cls=cls,
                                 n=int(m.sum()),
                                 precision=precision(truth[m], pred[m], cls)))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "defense_annotator_choice.csv", index=False)
    return r


# ---------------------------------------------------- 2. aggregation gain

def aggregation_gain(d):
    """A user's ideology is a 0.6 threshold over that user's posts, so the
    error rate that matters is the user-level one. Given the measured
    post-level confusion and the observed posts-per-user distribution, this
    simulates the user-level confusion the pipeline actually produces."""
    cons = d[d.Ideology == d.Ideology2]
    conf = {}
    for plat in ["bsky", "ts"]:
        g = cons[cons.platform == plat]
        t = pd.crosstab(g["Ideology"], g["stance"]).reindex(
            index=CATS, columns=CATS).fillna(0.0)
        # P(model label | true label): rows are the truth
        conf[plat] = t.div(t.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0).values

    nodes = pd.read_parquet(ROOT / "data" / "interim" / "trees_nodes.parquet")
    rows = []
    for plat in ["bsky", "ts"]:
        counts = nodes[nodes.platform == plat].groupby("author").size().values
        counts = counts[counts > 0]
        med = int(np.median(counts))
        # simulate: for a user with n posts and true label t, draw n post
        # labels from P(model | t), then apply the 0.6 threshold rule
        sim = {}
        for ti, true in enumerate(CATS):
            p = conf[plat][ti]
            if p.sum() == 0:
                continue
            n_draw = RNG.choice(counts, size=20000, replace=True)
            correct = 0
            for n in np.unique(n_draw):
                k = int((n_draw == n).sum())
                draws = RNG.choice(3, size=(k, int(n)), p=p / p.sum())
                frac_left = (draws == 0).mean(axis=1)
                frac_right = (draws == 2).mean(axis=1)
                lab = np.where(frac_left >= 0.6, 0,
                               np.where(frac_right >= 0.6, 2, 1))
                correct += int((lab == ti).sum())
            sim[true] = correct / len(n_draw)
        post_acc = float(np.mean([conf[plat][i, i] for i in range(3)]))
        rows.append(dict(platform=plat, n_users=len(counts),
                         median_posts_per_user=med,
                         mean_posts_per_user=round(float(counts.mean()), 1),
                         post_level_acc=round(post_acc, 3),
                         user_level_acc=round(float(np.mean(list(sim.values()))), 3),
                         **{f"user_acc_{k}": round(v, 3) for k, v in sim.items()}))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "defense_aggregation.csv", index=False)
    return r


# ---------------------------------------------------- 3. direction of bias

def direction_of_bias(d):
    cons = d[d.Ideology == d.Ideology2]
    minority = {"bsky": "right", "ts": "left"}
    majority = {"bsky": "center", "ts": "right"}
    rows = []
    for plat in ["bsky", "ts"]:
        g = cons[cons.platform == plat].reset_index(drop=True)
        truth, pred = g["Ideology"].values, g["stance"].values
        obs = (precision(truth, pred, minority[plat])
               - precision(truth, pred, majority[plat]))
        diffs = []
        for _ in range(B):
            i = RNG.integers(0, len(g), len(g))
            v = (precision(truth[i], pred[i], minority[plat])
                 - precision(truth[i], pred[i], majority[plat]))
            if not np.isnan(v):
                diffs.append(v)
        diffs = np.array(diffs)
        rows.append(dict(platform=plat, minority=minority[plat],
                         majority=majority[plat], diff=obs,
                         lo=np.percentile(diffs, 2.5),
                         hi=np.percentile(diffs, 97.5),
                         p_one_sided=float((diffs >= 0).mean())))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "defense_direction.csv", index=False)
    return r


if __name__ == "__main__":
    d = load()
    print("=== 1. Does the choice of reference annotator matter? ===")
    a = annotator_choice(d)
    piv = a.pivot_table(index=["scope", "cls"], columns="reference", values="precision")
    print(piv.round(3).to_string())
    print()
    print("=== 2. Does aggregating posts into user labels reduce error? ===")
    print(aggregation_gain(d).to_string(index=False))
    print()
    print("=== 3. Is the minority-over-assignment directional claim supported? ===")
    print(direction_of_bias(d).round(4).to_string(index=False))
