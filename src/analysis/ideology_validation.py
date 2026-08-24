"""Ideology-label validation metrics (R&R items C4, C5, C7).

Canonical source: src/val_ideology.csv (200 items: post text, model label
`stance`, and two independent human annotators `Ideology` and `Ideology2`).
Labels are normalized to lowercase left/center/right. Platform is recovered by
matching post text against the per-platform LLM label stores.

Reports agreement, class-specific and platform-specific precision and recall
against human consensus, and bootstrap confidence intervals for every rate, so
that the stability of each estimate is visible rather than assumed.

Outputs to results/ideology_validation/.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "ideology_validation"
OUT.mkdir(parents=True, exist_ok=True)
CATS = ["left", "center", "right"]
B = 2000
RNG = np.random.default_rng(20260824)


def load():
    d = pd.read_csv(ROOT / "src" / "val_ideology.csv")
    for c in ["stance", "Ideology", "Ideology2"]:
        d[c] = d[c].astype(str).str.strip().str.lower()
    bsky = json.load(open(ROOT / "data" / "bsky_post_to_label.json"))
    d["platform"] = d["post"].map(lambda p: "bsky" if p in bsky else "ts")
    return d


def prf(truth, pred, cls):
    tp = ((pred == cls) & (truth == cls)).sum()
    fp = ((pred == cls) & (truth != cls)).sum()
    fn = ((pred != cls) & (truth == cls)).sum()
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    return p, r


def boot_ci(truth, pred, cls, stat):
    """Percentile bootstrap over items."""
    n = len(truth)
    if n == 0:
        return (np.nan, np.nan)
    vals = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        v = prf(truth.values[idx], pred.values[idx], cls)[0 if stat == "precision" else 1]
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan)
    return tuple(np.percentile(vals, [2.5, 97.5]))


def main():
    d = load()
    rows, per_class = [], []

    for scope, m in [("all", d), ("bsky", d[d.platform == "bsky"]),
                     ("ts", d[d.platform == "ts"])]:
        cons = m[m["Ideology"] == m["Ideology2"]]
        # bootstrap the agreement statistics too
        kap = []
        for _ in range(B):
            idx = RNG.integers(0, len(m), len(m))
            s = m.iloc[idx]
            try:
                kap.append(cohen_kappa_score(s["Ideology"], s["Ideology2"]))
            except ValueError:
                pass
        rows.append({
            "scope": scope, "n": len(m), "n_consensus": len(cons),
            "kappa_h1_h2": cohen_kappa_score(m["Ideology"], m["Ideology2"]),
            "kappa_h1_h2_lo": np.percentile(kap, 2.5),
            "kappa_h1_h2_hi": np.percentile(kap, 97.5),
            "kappa_model_h1": cohen_kappa_score(m["stance"], m["Ideology"]),
            "kappa_model_h2": cohen_kappa_score(m["stance"], m["Ideology2"]),
            "accuracy_vs_consensus": (cons["stance"] == cons["Ideology"]).mean(),
        })
        for cls in CATS:
            p, r = prf(cons["Ideology"], cons["stance"], cls)
            plo, phi = boot_ci(cons["Ideology"], cons["stance"], cls, "precision")
            rlo, rhi = boot_ci(cons["Ideology"], cons["stance"], cls, "recall")
            per_class.append({
                "scope": scope, "class": cls, "support": int((cons["Ideology"] == cls).sum()),
                "precision": p, "prec_lo": plo, "prec_hi": phi,
                "recall": r, "rec_lo": rlo, "rec_hi": rhi,
            })
        cm = confusion_matrix(cons["Ideology"], cons["stance"], labels=CATS)
        pd.DataFrame(cm, index=[f"h_{c}" for c in CATS],
                     columns=[f"m_{c}" for c in CATS]).to_csv(OUT / f"confusion_{scope}.csv")

    k = pd.DataFrame(rows).round(3)
    pc = pd.DataFrame(per_class).round(3)
    k.to_csv(OUT / "kappas.csv", index=False)
    pc.to_csv(OUT / "per_class_metrics.csv", index=False)
    print(k.to_string(index=False))
    print()
    print(pc.to_string(index=False))


if __name__ == "__main__":
    main()
