"""Post-level versus user-collapsed cascades on ONE matched set of threads.

The earlier version filtered each representation on its own size column, so
post-level ran on 47,106 threads and user-collapsed on 42,947. That confounds
the change of unit with a change of sample. Collapsing can only reduce node
count, so u_size > 1 implies post_size > 1: the intersection is exactly the
42,947 threads that still have more than one node after collapsing, and both
representations are estimated on that same set here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reply_robustness import FRAME, OUT, CACHE, fit, BASE_F, M3C_F   # noqa: E402

REPS = {"post_level": ("post_size", "post_breadth", "post_depth"),
        "user_collapsed": ("u_size", "u_breadth", "u_depth")}


def main():
    m = pd.read_parquet(OUT / "tree_metrics.parquet")
    frame = pd.read_csv(FRAME, dtype={"index": str}, low_memory=False)
    j = frame.merge(m, left_on=["platform", "index"], right_on=["platform", "tree_id"])

    assert (j.loc[j.u_size > 1, "post_size"] > 1).all(), "u_size>1 should imply post_size>1"
    matched = j[j.u_size > 1].copy()
    print(f"matched threads: {len(matched):,} "
          f"(post_size>1 alone would be {int((j.post_size > 1).sum()):,})")

    rows = []
    for label, (sz, br, dp) in REPS.items():
        t = matched.copy()
        t["log_size"] = np.log10(t[sz])
        t["log_breadth"] = np.log10(t[br].clip(lower=1))
        t["log_depth"] = np.log10(t[dp] + 1)
        for y in ["log_breadth", "log_depth"]:
            b_base, _ = fit(t, BASE_F, y)
            b_m3c, n = fit(t, M3C_F, y)
            rows.append(dict(check="C3_matched", variant=label, y=y,
                             baseline_b3=b_base, model3c_b3=b_m3c, n=n,
                             attenuation=1 - abs(b_m3c) / abs(b_base)))
            print(f"  {label:15s} {y:12s} baseline={b_base:+.4f} "
                  f"3c={b_m3c:+.4f} n={n:,}", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "c3_matched_representation.csv", index=False)
    for y in ["log_breadth", "log_depth"]:
        a = t[(t.y == y) & (t.variant == "post_level")].iloc[0]
        b = t[(t.y == y) & (t.variant == "user_collapsed")].iloc[0]
        print(f"\n{y}: baseline {a.baseline_b3:+.4f} post-level -> "
              f"{b.baseline_b3:+.4f} collapsed, on the same {int(a.n):,} threads")
    print(f"\nwrote {OUT / 'c3_matched_representation.csv'}")


if __name__ == "__main__":
    main()
