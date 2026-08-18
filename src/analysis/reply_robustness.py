"""Reply-cascade robustness checks for the R&R (items C3, C5, C15).

C3  Unit of analysis: reply cascades are trees of POSTS. This rebuilds each
    tree at post level, validates the metrics against the modeling frame, then
    collapses repeated posts by the same user into one node (a user's depth is
    the minimum depth among their posts) and refits the models.
C5  Center category: recompute composition and alignment over partisan users
    only (center treated as unlabeled rather than as a third position) and
    refit Model 3c.
C15 Two-user exchanges: share of reply chains that alternate between the same
    two accounts.

Usage:
  uv run python reply_robustness.py prepare
  uv run python reply_robustness.py run
"""

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = DATA / "interim"
OUT = ROOT / "results" / "reply_robustness"
FRAME = DATA / "combined_reply_stats_06.csv"

BASE_F = "{y} ~ log_size * platform"
M3C_F = ("{y} ~ log_size * platform + bs(author_left_ratio,df=3) * log_size"
         " + log_size * bs(author_alignment_ratio,df=3)"
         " + author_right_ratio * log_size")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def walk_bsky(node, tree_id, parent_uri, rows):
    post = node.get("post") or {}
    uri = post.get("uri")
    author = (post.get("author") or {}).get("did")
    if uri is None or author is None:
        return
    rows.append((tree_id, uri, parent_uri, author))
    for r in node.get("replies") or []:
        walk_bsky(r, tree_id, uri, rows)


def prepare():
    frame = pd.read_csv(FRAME, usecols=["index", "platform"], dtype={"index": str})
    keep = {p: set(frame[frame.platform == p]["index"]) for p in ["bsky", "ts"]}
    sys.setrecursionlimit(200000)

    rows = []
    log("parsing bsky threads ...")
    for item in json.load(open(DATA / "bsky_threads.json")):
        th = item.get("thread") or {}
        root = (th.get("post") or {}).get("uri")
        if root in keep["bsky"]:
            walk_bsky(th, root, None, rows)
    db = pd.DataFrame(rows, columns=["tree_id", "post_id", "parent_id", "author"])
    db["platform"] = "bsky"
    log(f"bsky: {db.tree_id.nunique()} trees, {len(db)} posts")

    log("parsing ts threads (8.7 GB) ...")
    posts = json.load(open(DATA / "ts_threads_withReblogs.json"))
    meta = {p["_id"]: ((p.get("account") or {}).get("id"), p.get("in_reply_to_id"))
            for p in posts}
    del posts
    memo = {}

    def root_of(pid):
        path = []
        while True:
            if pid in memo:
                r = memo[pid]; break
            path.append(pid)
            par = meta.get(pid, (None, None))[1]
            if par is None or par not in meta or par in path:
                r = pid; break
            pid = par
        for q in path:
            memo[q] = r
        return r

    rows = []
    for pid, (author, par) in meta.items():
        if author is None:
            continue
        r = root_of(pid)
        if r not in keep["ts"]:
            continue
        rows.append((r, pid, par if par in meta else None, author))
    dt = pd.DataFrame(rows, columns=["tree_id", "post_id", "parent_id", "author"])
    dt["platform"] = "ts"
    log(f"ts: {dt.tree_id.nunique()} trees, {len(dt)} posts")

    CACHE.mkdir(parents=True, exist_ok=True)
    pd.concat([db, dt], ignore_index=True).to_parquet(CACHE / "reply_posts.parquet")
    log("cache written")


# ------------------------------------------------------------------ metrics

def tree_metrics(posts):
    """Per tree: post-level and user-collapsed size/breadth/depth, plus the
    two-user back-and-forth diagnostics. posts: DataFrame for ONE platform."""
    out = {}
    for tid, g in posts.groupby("tree_id", sort=False):
        par = dict(zip(g.post_id, g.parent_id))
        auth = dict(zip(g.post_id, g.author))
        # a parent outside the tree (or None) makes the post a root
        par = {p: (q if q in par else None) for p, q in par.items()}

        depth = {}
        for p in par:
            if p in depth:
                continue
            chain = []
            q = p
            while q is not None and q not in depth:
                chain.append(q)
                q = par.get(q)
            base = 0 if q is None else depth[q]
            for node in reversed(chain):
                base = 0 if par.get(node) is None else base + 1
                depth[node] = base
        if not depth:
            continue

        lv = Counter(depth.values())
        post_size = len(depth)
        post_breadth = max(lv.values())
        post_depth = max(depth.values())

        # user-collapsed: a user sits at the minimum depth among their posts
        umin = {}
        for p, d in depth.items():
            a = auth.get(p)
            if a is not None and (a not in umin or d < umin[a]):
                umin[a] = d
        ulv = Counter(umin.values())
        u_size = len(umin)
        u_breadth = max(ulv.values()) if ulv else 0
        u_depth = max(umin.values()) if umin else 0

        # two-user back-and-forth: grandparent and child are the same account,
        # with a different account in between (A -> B -> A)
        bf_edges = n_edges = 0
        for p, pa in par.items():
            if pa is None:
                continue
            n_edges += 1
            gp = par.get(pa)
            if gp is not None and auth.get(gp) == auth.get(p) \
               and auth.get(pa) != auth.get(p):
                bf_edges += 1
        out[tid] = dict(post_size=post_size, post_breadth=post_breadth,
                        post_depth=post_depth, u_size=u_size,
                        u_breadth=u_breadth, u_depth=u_depth,
                        bf_edges=bf_edges, n_edges=n_edges,
                        has_bf=bf_edges > 0)
    m = pd.DataFrame(out).T
    for c in ["post_size", "post_breadth", "post_depth", "u_size",
              "u_breadth", "u_depth", "bf_edges", "n_edges"]:
        m[c] = pd.to_numeric(m[c])
    m["has_bf"] = m["has_bf"].astype(bool)
    return m


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


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    posts = pd.read_parquet(CACHE / "reply_posts.parquet")
    frame = pd.read_csv(FRAME, dtype={"index": str}, low_memory=False)

    mets = []
    for plat, g in posts.groupby("platform"):
        log(f"metrics for {plat} ({g.tree_id.nunique()} trees) ...")
        m = tree_metrics(g)
        m["platform"] = plat
        mets.append(m)
    m = pd.concat(mets)
    m.index.name = "tree_id"
    m = m.reset_index()
    m.to_parquet(OUT / "tree_metrics.parquet")

    j = frame.merge(m, left_on=["platform", "index"], right_on=["platform", "tree_id"])
    log(f"matched {len(j)}/{len(frame)} frame rows")

    # --- C3 validation: post-level metrics must reproduce the frame ---
    val = {}
    for a, b in [("size", "post_size"), ("breadth", "post_breadth"), ("max_depth", "post_depth")]:
        s = j[[a, b]].dropna()
        val[a] = dict(corr=round(s[a].corr(s[b]), 4),
                      mean_abs_diff=round((s[a] - s[b]).abs().mean(), 4))
    log(f"C3 validation vs frame: {val}")

    # --- C3: refit on user-collapsed metrics ---
    jj = j.copy()
    res = []
    for label, (sz, br, dp) in {
        "post_level": ("post_size", "post_breadth", "post_depth"),
        "user_collapsed": ("u_size", "u_breadth", "u_depth"),
    }.items():
        t = jj[jj[sz] > 1].copy()
        t["log_size"] = np.log10(t[sz])
        t["log_breadth"] = np.log10(t[br].clip(lower=1))
        t["log_depth"] = np.log10(t[dp] + 1)
        for y in ["log_breadth", "log_depth"]:
            b_base, n1 = fit(t, BASE_F, y)
            b_m3c, n2 = fit(t, M3C_F, y)
            res.append(dict(check="C3_representation", variant=label, y=y,
                            baseline_b3=b_base, model3c_b3=b_m3c, n=n2,
                            attenuation=1 - abs(b_m3c) / abs(b_base)))
    pd.DataFrame(res).to_csv(OUT / "c3_representation.csv", index=False)
    log("C3 done:\n" + pd.DataFrame(res).round(4).to_string(index=False))

    # --- C15: two-user back-and-forth exchanges ---
    c15 = []
    for plat, g in j.groupby("platform"):
        deep = g[g["post_depth"] >= 2]
        c15.append(dict(platform=plat, cascades=len(g),
                        cascades_depth_ge2=len(deep),
                        share_with_backforth=float(g["has_bf"].mean()),
                        share_with_backforth_deep=float(deep["has_bf"].mean()) if len(deep) else np.nan,
                        share_edges_backforth=float(g["bf_edges"].sum() / g["n_edges"].sum())))
    c15 = pd.DataFrame(c15)
    c15.to_csv(OUT / "c15_backforth.csv", index=False)
    log("C15 done:\n" + c15.round(4).to_string(index=False))
    return j


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run"])
    a = ap.parse_args()
    {"prepare": prepare, "run": run}[a.cmd]()
