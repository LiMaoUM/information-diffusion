"""Repost-cascade reconstruction robustness suite (ICWSM 2027 R&R, roadmap P1-1).

Neither platform's data carries per-repost timestamps (repost list items are
account/profile objects), so the only temporal signal is the API list order.
Every reconstruction rule therefore selects each reposter's parent from the
same candidate set: the root, plus other reposters the user follows. This
script re-derives cascade metrics under alternative selection rules and
re-estimates the platform-by-size interaction (b3) per rule.

Rules
  code       replica of the shipped build_repost_graph: root-priority if the
             reposter follows the original author, else first followee found in
             full list order (any position, cycle-checked), else root fallback
  first      earliest prior (j < i in list order) followee; root fallback
  last       latest prior followee; root fallback
  random     uniform draw among prior followees; root fallback (K draws)

first/last/random also run with the repost list reversed (order=rev), since
the APIs' order direction is undocumented.

Usage
  uv run python repost_robustness.py prepare [--smoke N]   # parse + cache
  uv run python repost_robustness.py run [--smoke N] [--k 20]
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = DATA / "interim"
OUT = ROOT / "results" / "repost_robustness"

RULES = ["code", "first", "last", "random"]
ORDERED_RULES = ["first", "last", "random"]  # run under both order readings


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- parse/cache

def parse_bsky(smoke=None):
    log("loading bsky reposts json ...")
    posts = json.load(open(DATA / "bsky_reposts_new.json"))
    cascades = []
    for p in posts:
        reposts = p.get("reposts") or []
        if not reposts:
            continue
        cascades.append(
            (p["_id"], p["author"]["did"], [r["did"] for r in reposts])
        )
        if smoke and len(cascades) >= smoke:
            break
    log(f"bsky: {len(cascades)} cascades with >=1 repost")
    return cascades


def parse_ts(smoke=None):
    log("loading ts threads json (8.7 GB, takes a while) ...")
    posts = json.load(open(DATA / "ts_threads_withReblogs.json"))
    cascades = []
    for p in posts:
        reposts = p.get("reblogList") or []
        if not reposts:
            continue
        cascades.append(
            (p["_id"], p["account"]["id"], [r["id"] for r in reposts])
        )
        if smoke and len(cascades) >= smoke:
            break
    log(f"ts: {len(cascades)} cascades with >=1 repost")
    return cascades


def load_follows(platform, universe):
    """Followee sets restricted to users in `universe` (authors seen in
    cascades). Membership tests never need anyone else."""
    path = DATA / ("bsky_followings.json" if platform == "bsky" else "ts_user_following_map.json")
    log(f"{platform}: loading follow map {path.name} ...")
    raw = json.load(open(path))
    follows = {}
    for user, followees in raw.items():
        if user in universe:
            follows[user] = universe.intersection(followees)
    log(f"{platform}: follow sets for {len(follows)} of {len(universe)} relevant users")
    return follows


def prepare(platform, smoke=None):
    cascades = parse_bsky(smoke) if platform == "bsky" else parse_ts(smoke)
    universe = set()
    for _, root_author, reposters in cascades:
        universe.add(root_author)
        universe.update(reposters)
    follows = load_follows(platform, universe)

    CACHE.mkdir(parents=True, exist_ok=True)
    tag = f"_smoke{smoke}" if smoke else ""
    pd.DataFrame(
        cascades, columns=["post_id", "root_author", "reposters"]
    ).to_parquet(CACHE / f"repost_cache_{platform}{tag}.parquet")
    with open(CACHE / f"follow_cache_{platform}{tag}.json", "w") as f:
        json.dump({u: sorted(s) for u, s in follows.items()}, f)
    log(f"{platform}: cache written ({tag or 'full'})")


def load_cache(platform, smoke=None):
    tag = f"_smoke{smoke}" if smoke else ""
    df = pd.read_parquet(CACHE / f"repost_cache_{platform}{tag}.parquet")
    follows = {
        u: set(v)
        for u, v in json.load(open(CACHE / f"follow_cache_{platform}{tag}.json")).items()
    }
    cascades = list(df.itertuples(index=False, name=None))
    return cascades, follows


# ------------------------------------------------------------------ the rules

def candidates_prior(reposters, follows):
    """For each position i: indices j < i whose author u_i follows, and
    whether u_i follows the root author (computed by caller)."""
    seen_positions = defaultdict(list)  # author -> positions so far
    out = []
    for i, u in enumerate(reposters):
        fset = follows.get(u, ())
        cand = []
        for a in fset:
            cand.extend(seen_positions.get(a, ()))
        out.append(sorted(cand))
        seen_positions[u].append(i)
    return out


def parents_ordered(reposters, root_author, follows, mode, rng=None):
    """Rules first/last/random. Parent index: -1 = root."""
    cands = candidates_prior(reposters, follows)
    parents = np.full(len(reposters), -1, dtype=np.int64)
    n_fallback = n_unique = 0
    for i, cand in enumerate(cands):
        if not cand:
            n_fallback += 1  # root: either followed root or pure fallback
            continue
        if len(cand) == 1:
            n_unique += 1
        if mode == "first":
            parents[i] = cand[0]
        elif mode == "last":
            parents[i] = cand[-1]
        else:
            parents[i] = cand[rng.randrange(len(cand))]
    stats = {
        "n": len(reposters),
        "root_attached": int(n_fallback),
        "unique_candidate": int(n_unique),
        "mean_cands": float(np.mean([len(c) for c in cands])) if cands else 0.0,
    }
    return parents, stats


def parents_code(reposters, root_author, follows):
    """Faithful replica of InformationCascadeGraph.build_repost_graph,
    including its two quirks:
    (a) after a node links under author A, linked_users[A] is overwritten with
        the CHILD's id, so later followers of A chain under the previous
        attacher rather than becoming siblings;
    (b) unlinked_nodes.remove() during iteration skips the next element, which
        then falls back to the root.
    Root-priority: reposters following the root author attach to the root."""
    n = len(reposters)
    parents = np.full(n, -2, dtype=np.int64)  # -2 = unlinked
    author_positions = defaultdict(list)
    for i, u in enumerate(reposters):
        author_positions[u].append(i)

    linked_users = {root_author: -1}  # author -> node index (-1 = root)
    unlinked_users = {}
    has_edge = np.zeros(n, dtype=bool)
    unlinked = []
    for i, u in enumerate(reposters):
        if root_author in follows.get(u, ()):
            parents[i] = -1
            has_edge[i] = True
            linked_users[u] = i
        else:
            unlinked_users[u] = i
            unlinked.append((u, i))

    def is_ancestor(a, x):
        while x is not None and x >= 0:
            if x == a:
                return True
            p = parents[x]
            x = p if p != -2 else None
        return False

    pos = 0
    while pos < len(unlinked):
        node, i = unlinked[pos]
        fset = follows.get(node, ())
        # first occurrence, in list order, of any author the node follows
        cand = sorted(
            j for a in fset if a in author_positions
            for j in author_positions[a] if reposters[j] != node
        )
        linked_here = False
        for j in cand:
            v = reposters[j]
            target = linked_users.get(v)
            if has_edge[i] and target is not None and is_ancestor(i, target):
                continue  # original cycle check
            parent = linked_users[v] if v in linked_users else unlinked_users[v]
            parents[i] = parent
            has_edge[i] = True
            if parent >= 0:
                has_edge[parent] = True
            linked_users[v] = i  # quirk (a)
            linked_here = True
            break
        if linked_here:
            unlinked.pop(pos)
            pos += 1  # quirk (b): the next element is skipped
        else:
            pos += 1

    parents[parents == -2] = -1
    return parents


# ----------------------------------------------------------------- metrics

def tree_metrics(parents):
    """size, max breadth, max depth, directed structural virality for a tree
    given parent indices (-1 = root). O(n) via depth counting; for a directed
    tree, sum of ancestor distances of node v is d_v (d_v + 1) / 2 and the
    ordered reachable pair count is sum d_v."""
    n = len(parents) + 1  # + root
    depth = np.zeros(len(parents), dtype=np.int64)
    for i in range(len(parents)):
        # iterative depth with path memo
        stack = []
        j = i
        while j >= 0 and depth[j] == 0:
            stack.append(j)
            j = parents[j]
        base = 0 if j < 0 else depth[j]
        for k in reversed(stack):
            base += 1
            depth[k] = base
    if len(depth):
        counts = Counter(depth)
        breadth = max(counts.values())
        max_depth = int(depth.max())
        d = depth.astype(np.float64)
        total_dist = float((d * (d + 1) / 2).sum())
        pairs = float(d.sum())
        sv = total_dist / pairs if pairs else 0.0
    else:
        breadth, max_depth, sv = 0, 0, 0.0
    return {"size": n, "breadth": breadth, "depth": max_depth, "sv": sv}


# ----------------------------------------------------------------- run

def run_platform(platform, smoke=None, k=20, seed=7):
    cascades, follows = load_cache(platform, smoke)
    rows = []
    diag_rows = []
    rng = random.Random(seed)
    t0 = time.time()
    for ci, (post_id, root_author, reposters) in enumerate(cascades):
        reposters = list(reposters)
        for order in ["api", "rev"]:
            rl = reposters if order == "api" else list(reversed(reposters))
            for rule in ["first", "last"]:
                parents, stats = parents_ordered(rl, root_author, follows, rule)
                m = tree_metrics(parents)
                rows.append({"post_id": post_id, "platform": platform,
                             "rule": rule, "order": order, "draw": 0, **m})
                if rule == "first" and order == "api":
                    diag_rows.append({"post_id": post_id, "platform": platform, **stats})
            for draw in range(k):
                parents, _ = parents_ordered(rl, root_author, follows, "random", rng)
                m = tree_metrics(parents)
                rows.append({"post_id": post_id, "platform": platform,
                             "rule": "random", "order": order, "draw": draw, **m})
        parents = parents_code(reposters, root_author, follows)
        m = tree_metrics(parents)
        rows.append({"post_id": post_id, "platform": platform,
                     "rule": "code", "order": "api", "draw": 0, **m})
        if ci % 2000 == 0 and ci:
            log(f"{platform}: {ci}/{len(cascades)} cascades, {time.time()-t0:.0f}s")
    log(f"{platform}: done, {len(cascades)} cascades in {time.time()-t0:.0f}s")
    return pd.DataFrame(rows), pd.DataFrame(diag_rows)


def fit_b3(df, y):
    """Huber RLM of log10(y) ~ log10(size) * platform; returns b3 and CI."""
    import statsmodels.api as sm

    d = df[(df["size"] > 1) & (df[y] > 0)].copy()
    d["ly"] = np.log10(d[y])
    d["ls"] = np.log10(d["size"])
    d["ts"] = (d["platform"] == "ts").astype(float)
    X = sm.add_constant(
        np.column_stack([d["ls"], d["ts"], d["ls"] * d["ts"]])
    )
    fit = sm.RLM(d["ly"], X, M=sm.robust.norms.HuberT()).fit()
    b3, se = fit.params[3], fit.bse[3]
    return {"b3": b3, "lo": b3 - 1.96 * se, "hi": b3 + 1.96 * se, "n": len(d)}


def run(smoke=None, k=20):
    OUT.mkdir(parents=True, exist_ok=True)
    metrics, diags = [], []
    for platform in ["bsky", "ts"]:
        m, dg = run_platform(platform, smoke, k)
        metrics.append(m)
        diags.append(dg)
    metrics = pd.concat(metrics, ignore_index=True)
    diags = pd.concat(diags, ignore_index=True)
    tag = f"_smoke{smoke}" if smoke else ""
    metrics.to_parquet(OUT / f"cascade_metrics{tag}.parquet")
    diags.to_parquet(OUT / f"diagnostics{tag}.parquet")

    results = []
    for y in ["breadth", "depth"]:
        for (rule, order), grp in metrics[metrics["rule"] != "random"].groupby(["rule", "order"]):
            results.append({"y": y, "rule": rule, "order": order, "draw": None,
                            **fit_b3(grp, y)})
        for (order, draw), grp in metrics[metrics["rule"] == "random"].groupby(["order", "draw"]):
            results.append({"y": y, "rule": "random", "order": order,
                            "draw": draw, **fit_b3(grp, y)})
    res = pd.DataFrame(results)
    res.to_csv(OUT / f"b3_by_rule{tag}.csv", index=False)

    # headline
    for y in ["breadth", "depth"]:
        r = res[res["y"] == y]
        log(f"HEADLINE {y}: b3 range across rules [{r['b3'].min():.4f}, {r['b3'].max():.4f}]"
            f" (code rule: {r[r['rule'] == 'code']['b3'].iloc[0]:.4f})")
    # diagnostics summary
    ds = diags.groupby("platform").apply(
        lambda g: pd.Series({
            "cascades": len(g),
            "reposts": g["n"].sum(),
            "root_attached_share": g["root_attached"].sum() / g["n"].sum(),
            "unique_candidate_share": g["unique_candidate"].sum() / g["n"].sum(),
            "mean_candidates": (g["mean_cands"] * g["n"]).sum() / g["n"].sum(),
        }), include_groups=False)
    ds.to_csv(OUT / f"diagnostics_summary{tag}.csv")
    log("diagnostics:\n" + ds.to_string())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run"])
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--platform", choices=["bsky", "ts"], default=None)
    args = ap.parse_args()
    if args.cmd == "prepare":
        for p in [args.platform] if args.platform else ["bsky", "ts"]:
            prepare(p, args.smoke)
    else:
        run(args.smoke, args.k)
