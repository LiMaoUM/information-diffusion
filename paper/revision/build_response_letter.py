"""Build the point-by-point response letter as a single table.

Rows follow reviewer_response_tracker.csv, with each response rewritten in past
tense and a pointer to where the change lands in the manuscript.

Run:  python3 build_response_letter.py && pandoc response_letter.md \
          -o response_letter.pdf --pdf-engine=xelatex
"""
import textwrap

W = [6, 12, 29, 60, 14]   # relative column widths, in characters

ROWS = [
 ("C1", "SPC-1, R1-R3",
  "Repost cascades are reconstructed, not observed, so the cross-platform similarity may be an artifact of the method.",
  "Rebuilt all 244,129 repost cascades under five alternative linking rules plus a 40-draw random-rule ensemble. The platform interaction stays within [-0.012, +0.031] for breadth and [-0.083, +0.028] for depth under every rule, against +0.170 and -0.182 for replies. These cascades are called reconstructed throughout.",
  "App. D, Fig. 9"),
 ("C2", "R3, SPC-1",
  "A user may have followed someone only after reposting, and that error could differ by platform.",
  "Both follow networks were collected at the same time, so any timing gap applies equally to the two platforms. Deleting 5 to 30 percent of follow edges at random moves the breadth interaction from 0.031 to 0.005 and depth from 0.004 to 0.013, an order of magnitude below the reply values.",
  "App. D"),
 ("C3", "SPC-2, R2, R3",
  "Unclear whether cascade nodes are users or posts, and how repeated users in one thread are handled.",
  "Reply nodes are posts; ideology is aggregated per user, so a user who replies twice appears twice carrying the same label. We rebuilt every cascade under a user-collapsed alternative. Depth holds; breadth shrinks, so part of Truth Social's extra width is repeat posting rather than more distinct participants. Both are reported.",
  "Methods; App. E"),
 ("C4", "SPC-3, R1, R2",
  "Ideology validation is thin: 200 items, no class-specific or platform-specific performance.",
  "Added per-class and per-platform precision and recall with bootstrap intervals against the 171 consensus items. Accuracy 0.86; kappa 0.77 between annotators, 0.61 and 0.74 model to human. Minority-class precision is 0.64 on both platforms against 0.98 and 0.93 for each platform's largest class.",
  "App. B, Table 3"),
 ("C5", "SPC-3, R2",
  "The center category may mix genuinely centrist, mixed, and uncertain users.",
  "Center is now defined explicitly as a residual category, and we report the models recomputed over partisan users only.",
  "Methods; App. B"),
 ("C6", "SPC-3, R1",
  "Show the findings survive plausible ideology label noise.",
  "Propagated the measured confusion matrix through the full pipeline, 100 draws per scenario across four error processes including a worst case built on the stricter annotator. The median run retains 86 to 96 percent of the attenuation, the worst case 70 percent.",
  "App. B"),
 ("C7", "R1",
  "The left share on Truth Social and right share on Bluesky look too high against prior work.",
  "Our frame is Biden and Trump conversation during the campaign, not the platform population. A prevalence correction lowers the Bluesky right share from 24.2 to 17.9 percent and the Truth Social left share from 11.5 to 1.9 percent. A gap remains for Bluesky, which we attribute to the sampling frame rather than to label error.",
  "App. B"),
 ("C8", "SPC-4, R2, R3",
  "Alignment is computed from the same reply edges it is meant to explain, so causal language is too strong.",
  "Causal language is replaced by statistical accounting throughout, and the simultaneity is stated directly in the Limitations subsection.",
  "Results; Limitations"),
 ("C9", "SPC-5, R1",
  "Readable descriptive statistics are missing.",
  "Added cascade counts, post counts, and the cascade size distribution by platform and cascade type.",
  "App. C, Table 4"),
 ("C10", "R1",
  "Report raw motif counts, not only z-scores.",
  "Added observed counts for all 54 ideology-labeled three-node motifs on both platforms.",
  "App. G, Table 5"),
 ("C11", "R1",
  "Motifs overlap; does the null model account for that?",
  "It does: randomized graphs are counted with the identical enumeration, so overlap inflates observed and null counts alike. We now state the procedure precisely, 100 randomizations by within-cascade double-edge swaps with 5m attempts, and note the consequence that neighboring z-scores are correlated and should be read as a pattern across families.",
  "Methods; App. G"),
 ("C12", "SPC-6, R2",
  "Main text says Huber loss, appendix says OLS with HC3. Which is it?",
  "Huber. The stale appendix paragraph is removed. Checking this surfaced a further problem: the default scale estimate degenerates here because 61 percent of cascades sit at the origin, so every model is refitted with Huber's proposal 2 scale and Table 1 now carries interpretable standard errors. A sensitivity table reports both estimators.",
  "Methods; App. I, Table 6"),
 ("C13", "SPC-7, R1-R3",
  "One month of Biden and Trump content in an unusual period: what generalizes, and what do the two shapes imply?",
  "The Discussion is rewritten so implications come first, followed by a Limitations subsection stating what a candidate-centered, one-month frame does and does not support.",
  "Discussion"),
 ("C14", "R1",
  "Excluding influencers only on Truth Social is poorly justified; why not the top n percent on both?",
  "The follower distributions are not comparable at the top. The largest Truth Social account has 56 times the followers of that platform's own 99.9th percentile account; on Bluesky the ratio is 5. The top eight Truth Social accounts hold 16 percent of all followers against 6 percent on Bluesky, and the eighth largest Truth Social account exceeds the largest Bluesky account. A symmetric top-n rule would delete ordinary Bluesky accounts to mirror a structure only Truth Social has.",
  "App. A"),
 ("C15", "R1",
  "How many reply chains are two users going back and forth?",
  "Among cascades of depth two or more, 68.8 percent on Bluesky and 70.0 percent on Truth Social contain an A to B to A exchange, carrying 20.0 and 15.6 percent of reply edges. The rate is similar on both platforms, so the depth difference is not explained by dyadic exchange.",
  "App. F"),
 ("C16", "R1",
  "More topic model detail; BERTopic often leaves half of short posts as outliers.",
  "Added the corpus size (125,623 root posts), the outlier reduction step and its settings, the resulting outlier share, and the eleven retained topics.",
  "App. H"),
 ("C17", "R1",
  "The alignment results are hard to follow.",
  "The subsection is rewritten with a plain definition and a worked example, and notation is defined at first use.",
  "Methods"),
 ("C18", "SPC-5, R1",
  "Figures are messy: overlapping labels, unclear quantities.",
  "Rebuilt the motif figures with family labels moved out of the data area, and the topic, partisanship and partial-residual figures, where value labels had been falling outside the axes. All are 300 dpi with no overlapping elements.",
  "Figs. 1, 4, 5, 7, 11"),
 ("C19", "R1",
  "Citations are dated; add recent information-spreading work.",
  "Added recent work on cascades and newer platforms, including the Bluesky diffusion literature.",
  "Related Work"),
 ("C20", "R1",
  "Unexplained notation and stray paragraph indents.",
  "Notation and formatting pass.",
  "Throughout"),
 ("C21", "R1",
  "Another US-centric paper.",
  "Acknowledged in the Limitations subsection, with non-US comparison named as future work.",
  "Limitations"),
]

HEAD = """---
title: "Response to Reviewers"
subtitle: "Paper 1217, Depth, Breadth, and Bias: Structural Diffusion of Political Content on Divergent Platforms"
geometry: margin=0.7in, landscape
fontsize: 9pt
---

We thank the reviewers and the SPC for a careful and specific set of comments. Every point
is answered below, with what we did and where to find it. Additions are highlighted in blue
in the revised manuscript.

Four changes are substantive rather than editorial. The repost reconstruction is now
validated against an envelope of alternative linking rules, which is what the similarity
claim rests on (C1, C2). The ideology labels carry class-specific and platform-specific
validation and a noise-propagation analysis (C4, C6). Cascades are reported under both a
post-level and a user-collapsed definition (C3). And we corrected a degenerate scale
estimate in the regression models, then refitted every specification (C12).

Two of these did not simply confirm what we had. The user-collapsed representation shows
that part of the raw breadth gap comes from repeat posting rather than from more distinct
participants, and the center-category check shows that some of the statistical accounting
runs through partisan versus non-partisan participation. We report both and have narrowed
the claims accordingly.

"""


def rule(sep="-"):
    return "+" + "+".join(w * sep for w in W) + "+"


def row(cells):
    cols = [textwrap.wrap(c, w - 2, break_on_hyphens=False) or [""] for c, w in zip(cells, W)]
    out = []
    for i in range(max(len(c) for c in cols)):
        line = "|"
        for col, w in zip(cols, W):
            line += " " + (col[i] if i < len(col) else "").ljust(w - 2) + " |"
        out.append(line)
    return "\n".join(out)


def main():
    body = [rule(), row(["#", "Raised by", "Reviewer point", "What we did", "Where"]), rule("=")]
    for r in ROWS:
        body += [row(list(r)), rule()]
    text = HEAD + "\n".join(body) + "\n"
    for c in "–—":
        assert c not in text, "en or em dash in the letter"
    for r in ROWS:
        assert len(r[0]) <= W[0] - 2, f"ID {r[0]} does not fit the column"
    open("response_letter.md", "w").write(text)
    print(f"wrote response_letter.md ({len(ROWS)} items)")


if __name__ == "__main__":
    main()
