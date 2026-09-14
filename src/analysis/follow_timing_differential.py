"""Follow-timing error that differs by platform (R3: could the error differ
systematically between Truth Social and Bluesky?).

follow_timing_check.py deletes the same share of follow edges on both
platforms. Here one platform loses 5 percent and the other 30 percent, both
ways round, under the same most-recent-reposter rule, so the platform
difference is re-estimated when the timing error is deliberately unequal.
"""
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repost_robustness import candidates_prior, select_parents, tree_metrics, fit_b3  # noqa: E402
from follow_timing_check import thin, CACHE, OUT  # noqa: E402

CONFIGS = [{"bsky": 0.05, "ts": 0.30}, {"bsky": 0.30, "ts": 0.05}]
SEEDS = [1, 2]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    cache = {}
    for plat in ["bsky", "ts"]:
        df = pd.read_parquet(CACHE / f"repost_cache_{plat}.parquet")
        follows = {u: set(v) for u, v in json.load(open(CACHE / f"follow_cache_{plat}.json")).items()}
        cache[plat] = (list(df.itertuples(index=False, name=None)), follows)
    log("caches loaded")
    rows = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            mets = []
            for plat in ["bsky", "ts"]:
                cascades, follows = cache[plat]
                rate = cfg[plat]
                f2 = thin(follows, rate, random.Random(seed * 100 + int(rate * 100)))
                for _, _, reposters in cascades:
                    mets.append({"platform": plat,
                                 **tree_metrics(select_parents(candidates_prior(list(reposters), f2), "last"))})
            m = pd.DataFrame(mets)
            for y in ["breadth", "depth"]:
                r = fit_b3(m, y)
                rows.append(dict(bsky_rate=cfg["bsky"], ts_rate=cfg["ts"], seed=seed, y=y,
                                 b3=r["b3"], lo=r["lo"], hi=r["hi"], n=r["n"]))
                log(f"bsky {cfg['bsky']:.2f} ts {cfg['ts']:.2f} seed {seed} {y}: b3={r['b3']:+.4f}")
    pd.DataFrame(rows).to_csv(OUT / "c2_follow_timing_differential.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()
