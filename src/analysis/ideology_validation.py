"""Ideology-label validation metrics from data/val_chen.csv (R&R item P1-3).

The canonical 200-item stance validation: post text, model label (stance),
and two human annotators (Ideology, Ideology2). Labels normalized to
lowercase left/center/right. Platform recovered by matching post text
against the per-platform LLM label stores.

Outputs results/ideology_validation/: kappas, per-class and per-platform
precision/recall, confusion matrices (model vs human consensus).
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "ideology_validation"
OUT.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(ROOT / "data" / "val_chen.csv")
for c in ["stance", "Ideology", "Ideology2"]:
    d[c] = d[c].astype(str).str.strip().str.lower()

bsky = json.load(open(ROOT / "data" / "bsky_post_to_label.json"))
d["platform"] = d["post"].map(lambda p: "bsky" if p in bsky else "ts")

rows = []
reports = []
for scope, m in [("all", d), ("bsky", d[d.platform == "bsky"]), ("ts", d[d.platform == "ts"])]:
    cons = m[m["Ideology"] == m["Ideology2"]]
    rows.append({
        "scope": scope, "n": len(m), "n_consensus": len(cons),
        "kappa_h1_h2": cohen_kappa_score(m["Ideology"], m["Ideology2"]),
        "kappa_model_h1": cohen_kappa_score(m["stance"], m["Ideology"]),
        "kappa_model_h2": cohen_kappa_score(m["stance"], m["Ideology2"]),
    })
    rep = classification_report(cons["Ideology"], cons["stance"],
                                digits=3, zero_division=0, output_dict=True)
    for cls in ["left", "center", "right"]:
        reports.append({"scope": scope, "class": cls, **rep[cls]})
    cm = confusion_matrix(cons["Ideology"], cons["stance"],
                          labels=["left", "center", "right"])
    pd.DataFrame(cm, index=["h_left", "h_center", "h_right"],
                 columns=["m_left", "m_center", "m_right"]).to_csv(
        OUT / f"confusion_{scope}.csv")

pd.DataFrame(rows).round(3).to_csv(OUT / "kappas.csv", index=False)
pd.DataFrame(reports).round(3).to_csv(OUT / "per_class_metrics.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(pd.DataFrame(reports).round(3).to_string(index=False))
