"""Do two-person threads drive the reply-cascade platform difference? (R1 comment 6)

Among reply threads with a reply to a reply, 28.4% on Bluesky and 6.5% on
Truth Social involve only two accounts. Truth Social threads are much larger,
so raw shares are not comparable; this refits the baseline and the model with
both ideology measures after dropping threads with exactly two participants,
and compares the platform-by-size interaction with the full sample.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reply_robustness import FRAME, OUT, fit, BASE_F, M3C_F   # noqa: E402


def main():
    m = pd.read_parquet(OUT / "tree_metrics.parquet")
    frame = pd.read_csv(FRAME, dtype={"index": str}, low_memory=False)
    j = frame.merge(m[["platform", "tree_id", "u_size"]],
                    left_on=["platform", "index"], right_on=["platform", "tree_id"])
    rows = []
    for label, t in [("all matched", j), ("without two-participant threads", j[j.u_size != 2])]:
        for y in ["log_breadth", "log_depth"]:
            b0, _ = fit(t, BASE_F, y)
            b3, n = fit(t, M3C_F, y)
            rows.append(dict(sample=label, y=y, baseline=b0, model3c=b3, n=n,
                             accounted=1 - abs(b3) / abs(b0)))
            print(f"{label:32s} {y:12s} baseline {b0:+.4f}  with ideology {b3:+.4f}  "
                  f"accounted {100*(1-abs(b3)/abs(b0)):.0f}%  n={n:,}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "two_participant_check.csv", index=False)


if __name__ == "__main__":
    main()
