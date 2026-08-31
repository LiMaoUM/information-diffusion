"""Panel A (composition only, no alignment) of the encoding-comparison table,
under Huber's proposal 2 scale, to match Panel A/B and the main table."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from robustness_tables_huber import (ROOT, CACHE, FRAME, OUT, fit,
                                     labels_at, tree_ratios, load_portions)

M3A = "{y} ~ log_size * platform + bs({comp},df=3) * log_size + {comp2} * log_size"


def main():
    nodes = pd.read_parquet(CACHE / "trees_nodes.parquet")
    edges = pd.read_parquet(CACHE / "trees_edges.parquet")
    frame = pd.read_csv(FRAME, dtype={"index": str}, low_memory=False)
    rat = tree_ratios(nodes, edges, labels_at(load_portions(), 0.6))
    m = frame.merge(rat, left_on=["platform", "index"], right_index=True, how="inner")

    rows = []
    for enc in ["LR", "MajMin"]:
        if enc == "LR":
            m["comp"], m["comp2"] = m["r_left"], m["r_right"]
        else:
            m["comp"] = np.where(m["platform"] == "ts", m["r_right"], m["r_left"])
            m["comp2"] = np.where(m["platform"] == "ts", m["r_left"], m["r_right"])
        for y in ["log_breadth", "log_depth"]:
            r = fit(m, M3A, y)
            rows.append(dict(table="encoding_panelA", spec=enc, y=y, **r))
            print(f"panelA {enc} {y}: coef={r['coef']:+.4f} R2={r['pseudo_r2']:.3f} "
                  f"MAE={r['mae']:.3f} RMSE={r['rmse']:.3f} n={r['n']}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "encoding_panelA_huber.csv", index=False)


if __name__ == "__main__":
    main()
