"""Ideology label-noise propagation for Model 3c (R&R item C6).

Reconstructs each reply tree's node authors and parent-child author edges
from the raw thread files, validates the reconstruction against the stored
modeling frame, then perturbs user-level ideology labels according to the
measured confusion matrices and refits the paper's Model 3c to see whether
the attenuation conclusion survives.

Usage:
  uv run python noise_propagation.py prepare
  uv run python noise_propagation.py validate
  uv run python noise_propagation.py run [--k 100] [--smoke]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = DATA / "interim"
OUT = ROOT / "results" / "noise_propagation"

FRAME = DATA / "combined_reply_stats_06.csv"
FORMULA = ("{y} ~ log_size * platform + bs(author_left_ratio,df=3) * log_size"
           " + log_size * bs(author_alignment_ratio,df=3)"
           " + author_right_ratio * log_size")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ prepare

def walk_bsky(node, tree_id, rows, edges, parent_author):
    post = node.get("post") or {}
    author = (post.get("author") or {}).get("did")
    if author is None:
        return
    rows.append((tree_id, author))
    if parent_author is not None:
        edges.append((tree_id, parent_author, author))
    for r in node.get("replies") or []:
        walk_bsky(r, tree_id, rows, edges, author)


def prepare():
    frame = pd.read_csv(FRAME, usecols=["index", "platform"], dtype={"index": str})
    keep = {p: set(frame[frame.platform == p]["index"]) for p in ["bsky", "ts"]}

    sys.setrecursionlimit(100000)
    rows, edges = [], []
    log("parsing bsky threads ...")
    for item in json.load(open(DATA / "bsky_threads.json")):
        th = item.get("thread") or {}
        root_id = (th.get("post") or {}).get("uri")
        if root_id in keep["bsky"]:
            walk_bsky(th, root_id, rows, edges, None)
    nb = pd.DataFrame(rows, columns=["tree_id", "author"]); nb["platform"] = "bsky"
    eb = pd.DataFrame(edges, columns=["tree_id", "parent_author", "child_author"]); eb["platform"] = "bsky"
    log(f"bsky: {nb.tree_id.nunique()} trees, {len(nb)} nodes, {len(eb)} edges")

    log("parsing ts threads (8.7 GB) ...")
    posts = json.load(open(DATA / "ts_threads_withReblogs.json"))
    meta = {p["_id"]: ((p.get("account") or {}).get("id"), p.get("in_reply_to_id"))
            for p in posts}
    root_memo = {}

    def root_of(pid):
        path = []
        while True:
            if pid in root_memo:
                r = root_memo[pid]; break
            path.append(pid)
            parent = meta.get(pid, (None, None))[1]
            if parent is None or parent not in meta or parent in path:
                r = pid; break
            pid = parent
        for q in path:
            root_memo[q] = r
        return r

    rows, edges = [], []
    for pid, (author, parent) in meta.items():
        if author is None:
            continue
        r = root_of(pid)
        if r not in keep["ts"]:
            continue
        rows.append((r, author))
        if parent in meta and meta[parent][0] is not None and pid != r:
            edges.append((r, meta[parent][0], author))
    nt = pd.DataFrame(rows, columns=["tree_id", "author"]); nt["platform"] = "ts"
    et = pd.DataFrame(edges, columns=["tree_id", "parent_author", "child_author"]); et["platform"] = "ts"
    log(f"ts: {nt.tree_id.nunique()} trees, {len(nt)} nodes, {len(et)} edges")

    CACHE.mkdir(parents=True, exist_ok=True)
    pd.concat([nb, nt]).to_parquet(CACHE / "trees_nodes.parquet")
    pd.concat([eb, et]).to_parquet(CACHE / "trees_edges.parquet")
    log("cache written")


# ------------------------------------------------- labels, ratios, validate

def load_portions():
    return {
        "bsky": json.load(open(DATA / "bsky_author_ideology_portion.json")),
        "ts": json.load(open(DATA / "ts_author_ideology_portion_new.json")),
    }


def base_labels(portions, thr):
    lab = {}
    for plat, d in portions.items():
        for u, p in d.items():
            if (p.get("left") or 0) >= thr:
                lab[(plat, u)] = "left"
            elif (p.get("right") or 0) >= thr:
                lab[(plat, u)] = "right"
            else:
                lab[(plat, u)] = "center"
    return lab


_KEYS = {}


def _key_cols(nodes, edges):
    """Precompute join keys once; reused across all draws."""
    if not _KEYS:
        _KEYS["n"] = nodes["platform"] + "|" + nodes["author"]
        _KEYS["ep"] = edges["platform"] + "|" + edges["parent_author"]
        _KEYS["ec"] = edges["platform"] + "|" + edges["child_author"]
        _KEYS["ng"] = nodes[["platform", "tree_id"]]
        _KEYS["eg"] = edges[["platform", "tree_id"]]
    return _KEYS


def tree_ratios(nodes, edges, labels):
    """labels: dict {(platform, user): label}. Vectorized via key Series."""
    k = _key_cols(nodes, edges)
    lab_s = pd.Series({f"{p}|{u}": v for (p, u), v in labels.items()})
    nlab = k["n"].map(lab_s).fillna("center")
    comp = (
        pd.DataFrame({
            "platform": k["ng"]["platform"], "tree_id": k["ng"]["tree_id"],
            "is_left": (nlab == "left").values, "is_right": (nlab == "right").values,
        })
        .groupby(["platform", "tree_id"])[["is_left", "is_right"]].mean()
        .rename(columns={"is_left": "r_left", "is_right": "r_right"})
    )
    same = (k["ep"].map(lab_s).fillna("center").values
            == k["ec"].map(lab_s).fillna("center").values)
    align = (
        pd.DataFrame({
            "platform": k["eg"]["platform"], "tree_id": k["eg"]["tree_id"],
            "same": same,
        })
        .groupby(["platform", "tree_id"])["same"].mean().rename("r_align")
    )
    return comp.join(align, how="left")


def validate():
    nodes = pd.read_parquet(CACHE / "trees_nodes.parquet")
    edges = pd.read_parquet(CACHE / "trees_edges.parquet")
    frame = pd.read_csv(FRAME, dtype={"index": str}, low_memory=False)
    portions = load_portions()
    for thr in [0.6, 0.8]:
        rat = tree_ratios(nodes, edges, base_labels(portions, thr))
        m = frame.merge(rat, left_on=["platform", "index"], right_index=True)
        print(f"\nthreshold {thr}: matched {len(m)}/{len(frame)} frame rows")
        for a, b in [("author_left_ratio", "r_left"), ("author_right_ratio", "r_right"),
                     ("author_alignment_ratio", "r_align")]:
            sub = m[[a, b]].dropna()
            r = sub[a].corr(sub[b])
            print(f"  {a}: corr {r:.3f}, mean abs diff {(sub[a]-sub[b]).abs().mean():.4f}")


# ----------------------------------------------------------------- simulate

def validation_items():
    """The 200 canonical validation items with platform, for bootstrapping
    the confusion matrix itself."""
    d = pd.read_csv(ROOT / "src" / "val_ideology.csv")
    for c in ["stance", "Ideology", "Ideology2"]:
        d[c] = d[c].astype(str).str.strip().str.lower()
    bsky = json.load(open(DATA / "bsky_post_to_label.json"))
    d["platform"] = d["post"].map(lambda p: "bsky" if p in bsky else "ts")
    return d[d["Ideology"] == d["Ideology2"]][["platform", "Ideology", "stance"]]


def validation_items_a1():
    """All 200 items scored against Annotator 1 alone. Consensus items are the
    ones both annotators found easy, so a single-annotator reference yields a
    harsher error rate; this scenario tests the conclusion against it."""
    d = pd.read_csv(ROOT / "src" / "val_ideology.csv")
    for c in ["stance", "Ideology"]:
        d[c] = d[c].astype(str).str.strip().str.lower()
    bsky = json.load(open(DATA / "bsky_post_to_label.json"))
    d["platform"] = d["post"].map(lambda p: "bsky" if p in bsky else "ts")
    return d[["platform", "Ideology", "stance"]]


def conf_from_items(items):
    """Column-normalized P(human | model) per platform from a set of items."""
    cats = ["left", "center", "right"]
    out = {}
    items = items.reset_index(drop=True)  # resamples carry duplicate labels
    for plat in ["bsky", "ts"]:
        g = items[items.platform == plat]
        cm = pd.crosstab(g["Ideology"], g["stance"]).reindex(
            index=cats, columns=cats).fillna(0.0)
        col = cm.sum(axis=0).replace(0, np.nan)
        cm = cm.div(col, axis=1)
        # a model class never seen in this resample keeps its label
        for c in cats:
            if cm[c].isna().all():
                cm[c] = [1.0 if r == c else 0.0 for r in cats]
        out[plat] = cm.fillna(0.0)
    return out


def confusions():
    """Column-normalized P(human | model) per scope from validation output."""
    out = {}
    for scope in ["all", "bsky", "ts"]:
        cm = pd.read_csv(ROOT / "results" / "ideology_validation" / f"confusion_{scope}.csv", index_col=0)
        cm.index = [i.replace("h_", "") for i in cm.index]
        cm.columns = [c.replace("m_", "") for c in cm.columns]
        out[scope] = cm.div(cm.sum(axis=0), axis=1)  # columns sum to 1
    return out


_G = {}  # populated by run() before forking; workers inherit via fork


def _one_draw(args):
    scen, draw = args
    seed_base = {"per_platform": 0, "pooled": 1000, "bootstrap_matrix": 5000,
                 "annotator1": 9000}[scen]
    drng = np.random.default_rng(seed_base + draw)
    users, base_arr = _G["users"], _G["base_arr"]
    cats = ["left", "center", "right"]
    if scen == "annotator1":
        boot_conf = _G["conf_a1"]
    if scen == "bootstrap_matrix":
        items = _G["items"]
        idx = drng.integers(0, len(items), len(items))
        boot_conf = conf_from_items(items.iloc[idx])
    pick = {}
    for plat in ["bsky", "ts"]:
        if scen in ("bootstrap_matrix", "annotator1"):
            cm = boot_conf[plat]
        else:
            cm = _G["conf"][plat if scen == "per_platform" else "all"]
        idx = [i for i, u in enumerate(users) if u[0] == plat]
        labs = base_arr[idx]
        new = labs.copy()
        for m_lab in cats:
            mask = labs == m_lab
            if mask.sum() == 0:
                continue
            p = cm[m_lab].reindex(cats).fillna(0).values
            p = p / p.sum()
            new[mask] = drng.choice(cats, size=mask.sum(), p=p)
        for j, i in enumerate(idx):
            pick[users[i]] = new[j]
    pert = _G["apply_ratios"](_G["frame"], pick)
    out = []
    for y in ["log_breadth", "log_depth"]:
        b3, n = fit_b3(pert, y)
        out.append({"scenario": scen, "draw": draw, "y": y, "b3": b3, "n": n})
    return out


def fit_b3(frame, y):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    d = frame.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[y, "log_size", "author_left_ratio", "author_right_ratio",
                "author_alignment_ratio", "platform"])
    fit = smf.rlm(FORMULA.format(y=y), data=d,
                  M=sm.robust.norms.HuberT()).fit(maxiter=350)
    key = [k for k in fit.params.index
           if "log_size" in k and "platform" in k and "bs(" not in k]
    return float(fit.params[key[0]]), len(d)


def run(k=100, smoke=False, scenarios=None):
    OUT.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_parquet(CACHE / "trees_nodes.parquet")
    edges = pd.read_parquet(CACHE / "trees_edges.parquet")
    frame = pd.read_csv(FRAME, dtype={"index": str}, low_memory=False)
    portions = load_portions()
    base = base_labels(portions, 0.6)
    conf = confusions()
    rng = np.random.default_rng(11)
    if smoke:
        k = 2

    def apply_ratios(fr, labels):
        rat = tree_ratios(nodes, edges, labels)
        m = fr.merge(rat, left_on=["platform", "index"], right_index=True, how="inner")
        m["author_left_ratio"] = m["r_left"]
        m["author_right_ratio"] = m["r_right"]
        m["author_alignment_ratio"] = m["r_align"].fillna(m["author_alignment_ratio"])
        return m

    results = []
    # references
    for y in ["log_breadth", "log_depth"]:
        b3, n = fit_b3(frame, y)
        results.append({"scenario": "paper_frame", "draw": -1, "y": y, "b3": b3, "n": n})
    recon = apply_ratios(frame, base)
    for y in ["log_breadth", "log_depth"]:
        b3, n = fit_b3(recon, y)
        results.append({"scenario": "reconstructed_unperturbed", "draw": -1, "y": y, "b3": b3, "n": n})
    log(f"references done: {results}")

    _G.update(frame=frame, conf=conf, items=validation_items(),
              conf_a1=conf_from_items(validation_items_a1()),
              users=list(base.keys()),
              base_arr=np.array([base[u] for u in base]),
              apply_ratios=apply_ratios)

    import multiprocessing as mp
    scen_list = scenarios or ["per_platform", "pooled", "bootstrap_matrix",
                              "annotator1"]
    jobs = [(scen, d) for scen in scen_list for d in range(k)]
    nproc = 1 if smoke else min(12, mp.cpu_count() - 2)
    if nproc == 1:
        for j, job in enumerate(jobs):
            results.extend(_one_draw(job))
            log(f"draw {j + 1}/{len(jobs)}")
    else:
        with mp.get_context("fork").Pool(nproc) as pool:
            for j, out in enumerate(pool.imap_unordered(_one_draw, jobs)):
                results.extend(out)
                if j % 10 == 0:
                    log(f"{j + 1}/{len(jobs)} draws done")

    res = pd.DataFrame(results)
    name = "b3_draws_smoke.csv" if smoke else (
        "b3_draws_extra.csv" if scenarios else "b3_draws.csv")
    res.to_csv(OUT / name, index=False)
    for y in ["log_breadth", "log_depth"]:
        ref = res[(res.scenario == "reconstructed_unperturbed") & (res.y == y)]["b3"].iloc[0]
        for scen in scen_list:
            d = res[(res.scenario == scen) & (res.y == y)]["b3"]
            log(f"HEADLINE {y} [{scen}]: unperturbed {ref:.4f}, perturbed "
                f"median {d.median():.4f}, range [{d.min():.4f}, {d.max():.4f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "validate", "run"])
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--scenarios", default=None,
                    help="comma-separated subset of scenarios to run")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "validate":
        validate()
    else:
        run(k=a.k, smoke=a.smoke,
            scenarios=a.scenarios.split(",") if a.scenarios else None)
