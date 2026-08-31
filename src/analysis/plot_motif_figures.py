"""Regenerate the chain and star motif z-score figures (R&R item C18).

Reproduces the figures from the analysis notebook, with one change: the motif
family labels (Aligned, End shift, Mid shift, Diverse for chains; the
alignment and root groupings for stars) move out of the plot area into a
header band above the axes. In the submitted version they were drawn as
semi-transparent watermarks inside the panel, where they sat on top of the
data points; a reviewer reported labels overlapping content.

Data come from the motif pipeline's z-score files, which carry the observed
count, null mean, and null standard deviation for all 54 motifs.

Spec: results/motif_figure_spec.md
"""

import io
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "motif" / "src"
OUT = ROOT / "paper" / "Revision" / "figures"
IDEO = ["left", "center", "right"]
IDEO_ORDER = {"left": 0, "center": 1, "right": 2}
TS_COLOR, BS_COLOR, GRAY = "#CD3800", "#4F94CD", "#BDBDBD"


def load_stats(path):
    stats = {}
    for line in open(path):
        m = re.match(r"Motif (\d+) Real: ([\d.e+-]+), Mean: ([\d.e+-]+), Std: ([\d.e+-]+)", line)
        if m:
            stats[int(m.group(1))] = tuple(float(x) for x in m.groups()[1:])
    return stats


def motif_label(i):
    shape, j = ("Star", i) if i < 27 else ("Chain", i - 27)
    a, b, c = j // 9, (j // 3) % 3, j % 3
    return f"{shape}: {IDEO[a]}-{IDEO[b]}-{IDEO[c]}"


def node_color(n):
    s = n.split("#")[0]
    return {"right": "#D98B73", "left": "#76AECB", "center": "#C9D895"}.get(s, "#8a9197")


def glyph(ideologies_str, shape):
    ide = ideologies_str.lower().split("-")
    ids = [f"{ide[i]}#{i}" for i in range(3)]
    G = nx.DiGraph()
    if shape == "Chain":
        G.add_edge(ids[0], ids[1]); G.add_edge(ids[1], ids[2])
        pos = {ids[i]: (0, -i) for i in range(3)}
        figsize, fs, w, arrow = (0.4, 1.2), 14, 4.2, 5
    else:
        G.add_edge(ids[0], ids[1]); G.add_edge(ids[0], ids[2])
        pos = {ids[0]: (0, 0.2), ids[1]: (-0.8, -1), ids[2]: (0.8, -1)}
        figsize, fs, w, arrow = (0.8, 1.0), 10, 2.5, 6
    fig, ax = plt.subplots(figsize=figsize, dpi=600)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=ids,
                          node_color=[node_color(n) for n in ids], node_size=200, ax=ax)
    nx.draw_networkx_labels(G, pos=pos,
                           labels={n: {"left": "L", "right": "R", "center": "C"}[n.split("#")[0]] for n in ids},
                           font_size=fs, font_color="white", font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos=pos, edgelist=list(G.edges()), width=w,
                          edge_color="grey", arrows=True, arrowsize=arrow, ax=ax)
    ax.set_axis_off()
    if shape == "Star":
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 0.5)
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True,
                bbox_inches="tight", pad_inches=0.15 if shape == "Star" else 0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def classify_chain(label):
    nodes = label.split(": ")[1].split("-")
    u = len(set(nodes))
    if u == 1:
        return "aligned"
    if u == 3:
        return "all-different"
    common = [k for k, v in Counter(nodes).items() if v == 2][0]
    return "one-different(middle)" if [i for i, v in enumerate(nodes) if v != common][0] == 1 \
        else "one-different(end)"


def draw(df, spans, band_colors, shape, outfile, figsize, zoom, glyph_pad):
    """Shared renderer. spans: list of (start, end, display label, color key)."""
    fig = plt.figure(figsize=figsize, dpi=800)
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[0.10, 1, 0.26], hspace=0.0)
    hax = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=hax)
    iax = fig.add_subplot(gs[2, 0], sharex=hax)
    x = np.arange(len(df))

    for s, e, _, key in spans:
        ax.axvspan(s - 0.5, e - 0.5, color=band_colors[key], alpha=0.30, zorder=0)
    for s, e, _, _ in spans[1:]:
        ax.axvline(s - 0.5, color="grey", linewidth=1.0, zorder=0.5)

    for i, (_, r) in enumerate(df.iterrows()):
        ax.plot([i, i], [r["Z_TS"], r["Z_BSKY"]], color="gray", alpha=0.5, lw=3.0, zorder=1)
    tl, bl = df["Z_TS"].abs() < 3, df["Z_BSKY"].abs() < 3
    ax.scatter(x[~tl], df.loc[~tl, "Z_TS"], facecolors="white", edgecolors=TS_COLOR,
               s=60, linewidth=2.2, zorder=2, marker="o")
    ax.scatter(x[tl], df.loc[tl, "Z_TS"], facecolors="white", edgecolors=GRAY,
               s=60, linewidth=2.2, zorder=2, marker="o")
    ax.scatter(x[~bl], df.loc[~bl, "Z_BSKY"], facecolors="white", edgecolors=BS_COLOR,
               s=60, linewidth=2.2, zorder=2, marker="^")
    ax.scatter(x[bl], df.loc[bl, "Z_BSKY"], facecolors="white", edgecolors=GRAY,
               s=60, linewidth=2.2, zorder=2, marker="^")

    ax.axhline(0, color="black", linewidth=1.3, zorder=1)
    ax.set_ylabel("Z-score", fontsize=12)
    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.grid(axis="y", color="lightgray", linestyle="--", linewidth=0.7, alpha=0.5)
    for sp in ["top", "right", "bottom"]:
        ax.spines[sp].set_visible(False)
    ax.set_xticks([])
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=TS_COLOR, markeredgewidth=2.2, markersize=8, label="Truth Social"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="white",
               markeredgecolor=BS_COLOR, markeredgewidth=2.2, markersize=8, label="Bluesky"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
               markeredgecolor=GRAY, markeredgewidth=2.2, markersize=8, label="$|Z| < 3$"),
    ], loc="upper right", frameon=False, fontsize=9, labelspacing=0.35)

    # header band: family labels live here, never over the data
    for s, e, text, key in spans:
        hax.axvspan(s - 0.5, e - 0.5, color=band_colors[key], alpha=0.55, zorder=0)
        hax.text((s + e - 1) / 2, 0.5, text, ha="center", va="center",
                 fontsize=9, color="#333333", weight="semibold")
    hax.set_ylim(0, 1)
    hax.set_xticks([]); hax.set_yticks([])
    for sp in hax.spines.values():
        sp.set_visible(False)

    for i, motif in enumerate(df["Motif"]):
        img = glyph(motif.split(": ")[1], shape)
        iax.add_artist(AnnotationBbox(OffsetImage(img, zoom=zoom), (i, glyph_pad),
                                      xycoords="data", box_alignment=(0.5, 1.0), frameon=False))
    iax.set_ylim(0, 1)
    iax.set_xticks([]); iax.set_yticks([])
    for sp in iax.spines.values():
        sp.set_visible(False)

    fig.savefig(OUT / outfile, dpi=800, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / outfile)


def main():
    ts = load_stats(SRC / "ts_motif_nodes_zscore_06_new.txt")
    bs = load_stats(SRC / "bsky_motif_nodes_zscores_06.txt")
    labels = {i: motif_label(i) for i in range(54)}

    # ---------------- chains ----------------
    rows = []
    for i in range(27, 54):
        tr, tm, tsd = ts[i]; br, bm, bsd = bs[i]
        rows.append((labels[i], (tr - tm) / tsd, (br - bm) / bsd))
    df = pd.DataFrame(rows, columns=["Motif", "Z_TS", "Z_BSKY"])
    df["Category"] = df["Motif"].apply(classify_chain)
    order = {"aligned": 0, "one-different(end)": 1, "one-different(middle)": 2, "all-different": 3}
    df["cat_order"] = df["Category"].map(order)
    df["sort_key"] = df["Z_BSKY"] - df["Z_TS"]
    df = df.sort_values(["cat_order", "sort_key"], ascending=[True, False]).reset_index(drop=True)
    names = {"aligned": "Aligned", "one-different(end)": "End shift",
             "one-different(middle)": "Mid shift", "all-different": "Diverse"}
    band = {"aligned": "#f8f8f8", "one-different(end)": "#f2e6ff",
            "one-different(middle)": "#d6ebff", "all-different": "#fdf0e8"}
    spans = []
    for cat, g in df.groupby("Category", sort=False):
        spans.append((g.index.min(), g.index.max() + 1, names[cat], cat))
    spans.sort()
    draw(df, spans, band, "Chain", "motif_z.png", (11, 4.6), 0.05, 1.0)

    # ---------------- stars (symmetric leaves merged) ----------------
    agg = defaultdict(lambda: defaultdict(list))
    for i in range(27):
        shape, rest = labels[i].split(": ")
        c, l1, l2 = rest.split("-")
        leaves = sorted([l1, l2], key=lambda v: IDEO_ORDER[v])
        key = f"{shape}: {c}-{leaves[0]}-{leaves[1]}"
        for nm, val in zip(("tr", "tm", "tsd"), ts[i]):
            agg[key][nm].append(val)
        for nm, val in zip(("br", "bm", "bsd"), bs[i]):
            agg[key][nm].append(val)
    rows = []
    for key, v in agg.items():
        tsd = float(np.sqrt(np.sum(np.square(v["tsd"]))))
        bsd = float(np.sqrt(np.sum(np.square(v["bsd"]))))
        rows.append((key, (np.sum(v["tr"]) - np.sum(v["tm"])) / tsd,
                     (np.sum(v["br"]) - np.sum(v["bm"])) / bsd))
    ds = pd.DataFrame(rows, columns=["Motif", "Z_TS", "Z_BSKY"])
    ds["Rest"] = ds["Motif"].apply(lambda x: x.split(": ")[1])
    ds["full"] = ds["Rest"].apply(lambda r: 0 if len(set(r.split("-"))) == 1 else 1)
    ds["root"] = ds["Rest"].apply(lambda r: IDEO_ORDER[r.split("-")[0]])
    ds["l1"] = ds["Rest"].apply(lambda r: IDEO_ORDER[r.split("-")[1]])
    ds["l2"] = ds["Rest"].apply(lambda r: IDEO_ORDER[r.split("-")[2]])
    ds = ds.sort_values(["full", "root", "l1", "l2"]).reset_index(drop=True)
    sband = {"full": "#f8f8f8", 0: "#e7f1ff", 1: "#f2e6ff", 2: "#fdf0e8"}
    spans = []
    fa = ds.index[ds["full"] == 0]
    if len(fa):
        spans.append((fa.min(), fa.max() + 1, "Aligned", "full"))
    for ro, nm in zip([0, 1, 2], ["Left-rooted", "Center-rooted", "Right-rooted"]):
        idx = ds.index[(ds["full"] == 1) & (ds["root"] == ro)]
        if len(idx):
            spans.append((idx.min(), idx.max() + 1, nm, ro))
    spans.sort()
    draw(ds, spans, sband, "Star", "star_motif_z.png", (11, 4.6), 0.055, 1.0)


if __name__ == "__main__":
    main()
