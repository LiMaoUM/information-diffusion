# Revision Roadmap: ICWSM 2027 R&R (paper 1217)

Deadline: **2026-09-15**. One shot, final accept/reject. Changes must be highlighted in color, with a separate response document.

## Overview

- Decision: Revise and Resubmit
- Scores: R1 weak reject (expertise 5, the reviewer to satisfy), R2 borderline, R3 weak accept
- Total distinct comments: 24 (7 SPC roadmap items, which subsume most reviewer points)
- By type: 8 Major, 9 Minor, 4 Editorial, 6 Positive themes
- Estimated effort: **Substantial** (new analyses required: repost robustness, ideology sensitivity, motif counts). Roughly 3 to 5 weeks of work against a 2-month window.

## The one scope decision to make first (blocks everything else)

**Keep repost cascades with robustness checks, or narrow the paper to reply cascades?** (SPC item 1, R1-1, R2-2.1, R3-2). Everything downstream (framing, figures, motifs) depends on this.

Recommendation: **keep reposts, add robustness checks**. The repost/reply asymmetry is what R2 calls the strongest contribution; cutting reposts removes the headline finding. The robustness checks are computable from data already in the repo (`data/bsky_follows.json`, `data/ts_user_following_map.json`, repost tables, `src/analysis/cascade_analysis.py`). Cut reposts only if the checks show the similarity result is fragile.

## P1: Must fix (SPC roadmap items; editor-mandated)

| # | Comment | Raised by | Section | Suggested action |
|---|---------|-----------|---------|------------------|
| P1-1 | Repost reconstruction may drive the cross-platform similarity result; validate or narrow scope; stop calling reconstructed cascades "empirical" | SPC, R1-1, R2-2.1, R3-2 | 3.2 Methods, Appendix Repost Cascades | Re-run cascade construction under alternative rules in `cascade_analysis.py`: (a) earliest-posting followee as parent (current), (b) most recent active followee, (c) random eligible followee, (d) star baseline where every reposter attaches to the root. Show scaling slopes and CCDFs are stable across (a) to (c) and distinct from (d). Address R3's follow-timing point: quantify or bound reposts where the follow edge may postdate the repost; discuss platform asymmetry of this error. Rename to "inferred" or "reconstructed" cascades throughout. |
| P1-2 | Define cascade unit of analysis; how are repeated users in one thread handled | SPC, R2-2, R3-3 | 3.2, Appendix Reply Cascades | State explicitly for each cascade type whether nodes are posts or users. If reply trees are post trees, say so and note users can appear multiple times; if user-collapsed, re-derive depth/breadth on post trees as a robustness check, since collapsing changes depth, breadth, and motifs. Add a schematic figure if space allows. |
| P1-3 | Ideology label validity: class-specific and platform-specific metrics, center category, robustness to label noise; surprising left presence on TS and right presence on Bluesky | SPC, R1-2, R2-4 | 3.x ideology labeling, Appendix validation | Expand validation beyond the 200 labeled replies: report per-class and per-platform precision/recall/F1 from `data/eval_*_chen.xlsx` and `eval_*_Li.xlsx` plus the llama3-70b eval samples; add labels if needed to get class-level power. Define "center" explicitly (centrist vs mixed vs uncertain) and run main models excluding center or reassigning it. Add a label-noise sensitivity analysis (flip x% of labels, show conclusions hold). Engage the Bovet team Bluesky paper directly and explain the discrepancy (their sample vs our Biden/Trump-mention sample). |
| P1-4 | Causal language: alignment is computed from the same reply edges it explains; endogeneity | SPC, R2-1, R3-1 | Results 4.2, Discussion, Abstract | Systematic language pass: "explains" becomes "is associated with" / "statistically accounts for". Add an explicit endogeneity paragraph (deep vs star cascades mechanically offer different cross-ideology contact opportunities). Optional strengthening: use early-cascade alignment (first k replies) to predict final structure, which breaks the simultaneity. |
| P1-5 | Descriptive reporting: posts, period, cascades by size, motif counts, participation, topic model outputs; motif dependence | SPC, R1-2/3/5, R2 | Data 3.1, Results, Appendix | Add a descriptive statistics table (posts, users, cascades, cascade size bins per platform per cascade type; counts readable, not only CCDFs). Add raw motif counts alongside z-scores (already in `src/motif/src/*_motif_nodes_counts*.txt`). Explain that the null model preserves the dependence structure (overlapping motifs occur in both observed and randomized graphs, so z-scores compare like with like), and state the randomization scheme precisely. |
| P1-6 | Huber loss (main text) vs HC3 OLS (appendix) inconsistency | SPC, R2 | 3.2 line ~258, Appendix line ~765 | The code (`src/modeling/model_R.R`) fits `robustbase::lmrob` (KS2014) and `MASS::rlm` with Huber psi. Rewrite the appendix paragraph to describe the actual estimator; keep HC3 only if it was genuinely used for a secondary OLS specification, and then say for which model. |
| P1-7 | Implications and limitations: one month, Biden/Trump, 2024 US election; what do broad/shallow vs narrow/deep structures mean | SPC, R2-3, R3-4, R1-9 | Discussion, Limitations | Rewrite Discussion: state what is claimed to generalize (structural signatures of ideologically sorted vs mixed platforms) vs what is context-bound (levels, composition). Spell out theoretical implications (broadcast vs deliberation styles of political talk) and practical ones (moderation, ranking, measurement). Own the US-centric scope explicitly. |

## P2: Should fix

| # | Comment | Raised by | Section | Suggested action |
|---|---------|-----------|---------|------------------|
| P2-1 | Influencer analysis only on TS is poorly justified; "no natural gap" is weak | R1-8 | Results 4.2 (H1) | Re-run influencer exclusion using top n% most-followed accounts on both platforms (n = 1, 5, 10) and show robustness; keep the original threshold as one specification. |
| P2-2 | Alignment results hard to follow | R1-7 | Results 4.2 | Rewrite the alignment subsection with a worked example and a definition box for alignment ratio; define all notation at first use. |
| P2-3 | Back-and-forth chains: how many reply chains are two-user ping-pong | R1-6, SPC | Results, Appendix | Compute the share of reply chains that alternate between the same two accounts (a small script over the reply edge tables); report per platform. This also speaks to what "depth" means on Bluesky. |
| P2-4 | Topic modeling details; BERTopic outlier share on short posts | R1-4, SPC | Methods, Appendix | Report corpus sizes, outlier-topic share, topic count, representative topics per platform (artifacts in `models/` and `data/*_df_id_topic.csv`); describe the reduce-outliers strategy if used. |
| P2-5 | Figures messy, overlapping labels, unclear quantities | SPC, R1 (presentation) | All figures | Figure QA pass on every figure (labels, overlap, axis titles, legend, colorblind-safe); regenerate at consistent DPI. Route through the figure-spec template before regenerating. |
| P2-6 | Literature is old; add recent information-spreading work | R1 (citations) | Related Work | Add 2023 to 2026 cascade/diffusion papers (including decentralized/alt-platform work and the Bovet Bluesky paper). |

## P3: Consider

| # | Comment | Raised by | Suggested action |
|---|---------|-----------|------------------|
| P3-1 | Unexplained notation | R1 | Notation table or define-at-first-use pass. |
| P3-2 | LaTeX paragraph indents where there should not be any | R1 | Fix stray blank lines after equations/figures in the tex. |
| P3-3 | US-centricity | R1-9 | Acknowledge in limitations; note non-US extension as future work. No analysis change. |
| P3-4 | Political engagement of typical users | R1-2 (aside) | One sentence noting the Biden/Trump filter selects politically engaged activity by design. |

## Positive comments (acknowledge in response letter)

- Cross-platform comparison of understudied platforms is novel and valuable (R1, R2, R3, SPC)
- Repost-as-amplification vs reply-as-conversation distinction (all)
- Repost-similar / reply-divergent asymmetry is clear and interesting; "strongest contribution" (R2)
- Well written, well organized, professional, avoids overstating (R1, R2, R3)
- Motif analysis interesting for platform comparison (R1, R2)
- Framing in existing literature is excellent (R1)

## Cross-reviewer patterns (the acceptance-critical trio)

1. **Repost reconstruction validity**: raised by all four. The revision lives or dies here.
2. **Ideology label validity**: R1 (expertise 5) says "currently I am not convinced". Class/platform-specific validation plus the Bovet reconciliation is what moves R1 from weak reject.
3. **Endogeneity of alignment**: R2 and R3 raise it independently; SPC adopts it. Language reframing is cheap; the early-cascade-alignment analysis is the strengthening option.

## Suggested revision order

1. Decide repost scope (see above), then immediately start the two compute-heavy items in parallel: repost reconstruction robustness (P1-1) and ideology validation/sensitivity (P1-3). These gate the narrative.
2. While those run: cascade definition clarification (P1-2), regression fix (P1-6), descriptive tables (P1-5), back-and-forth stats (P2-3), influencer top n% (P2-1).
3. Reframing pass (P1-4), alignment exposition (P2-2), Discussion/limitations rewrite (P1-7), literature refresh (P2-6).
4. Figure QA (P2-5) and editorial items (P3) last, then color-highlighted diff and response letter.
