"""Follow-network timing sensitivity for repost reconstruction (R&R item C2).

The follower network is a post-collection snapshot with no edge creation
dates, so some follow edges used in the reconstruction may have been created
after the repost. Such an edge can only pull a repost away from the root and
attach it to an interior parent. This deletes follow edges at random at rates
of 5 to 30 percent, simulating edges that did not yet exist, and re-estimates
the platform-by-size interaction.
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repost_robustness import candidates_prior, select_parents, tree_metrics, fit_b3  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "interim"
OUT = ROOT / "results" / "repost_robustness"
RATES = [0.0, 0.05, 0.10, 0.20, 0.30]
SEEDS = [1, 2, 3]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def thin(follows, rate, rng):
    if rate == 0:
        return follows
    out = {}
    for u, s in follows.items():
        keep = [v for v in s if rng.random() >= rate]
        out[u] = set(keep)
    return out


def main():
    log("loading caches once ...")
    cache = {}
    for plat in ["bsky", "ts"]:
        df = pd.read_parquet(CACHE / f"repost_cache_{plat}.parquet")
        follows = {u: set(v) for u, v in
                   json.load(open(CACHE / f"follow_cache_{plat}.json")).items()}
        cache[plat] = (list(df.itertuples(index=False, name=None)), follows)
        log(f"  {plat}: {len(cache[plat][0])} cascades, {len(follows)} follow sets")

    rows = []
    for rate in RATES:
        for seed in (SEEDS if rate > 0 else [0]):
            mets = []
            for plat in ["bsky", "ts"]:
                cascades, follows = cache[plat]
                rng = random.Random(seed * 100 + int(rate * 100))
                f2 = thin(follows, rate, rng)
                for post_id, root_author, reposters in cascades:
                    rl = list(reposters)
                    cands = candidates_prior(rl, f2)
                    m = tree_metrics(select_parents(cands, "last"))
                    mets.append({"platform": plat, **m})
            m = pd.DataFrame(mets)
            for y in ["breadth", "depth"]:
                r = fit_b3(m, y)
                rows.append(dict(rate=rate, seed=seed, y=y, b3=r["b3"],
                                 lo=r["lo"], hi=r["hi"], n=r["n"]))
            log(f"rate {rate} seed {seed} done")
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "c2_follow_timing.csv", index=False)
    print(r.groupby(["y", "rate"])["b3"].agg(["mean", "min", "max"]).round(4).to_string())


if __name__ == "__main__":
    main()
