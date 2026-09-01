---
title: "Response to Reviewers"
subtitle: "Paper 1217, Depth, Breadth, and Bias: Structural Diffusion of Political Content on Divergent Platforms"
geometry: margin=0.7in, landscape
fontsize: 9pt
---

We thank the reviewers and the SPC for a careful and specific set of comments. Every point
is answered below, with what we did and where to find it. Additions are highlighted in blue
in the revised manuscript.

Four changes are substantive rather than editorial: the repost reconstruction is validated
against an envelope of alternative linking rules (C1, C2); the ideology labels carry
class-specific and platform-specific validation plus a noise-propagation analysis (C4, C6);
cascades are reported under both a post-level and a user-collapsed definition (C3); and a
degenerate scale estimate in the regression models is corrected, with every specification
refitted (C12).

The central finding survived every check. Ideological composition and alignment absorb 81 to
94 percent of the baseline platform interaction across two samples and two scale estimators,
never below 68 percent under any spline placement, and never below 70 percent in the worst of
four label-noise processes. It also holds across ideology thresholds from 0.5 to 0.7, under
both composition encodings, and under every repost reconstruction rule.

Two checks qualified a magnitude rather than the conclusion, and we report both rather than
only the stronger specification. The user-collapsed representation shows that part of the
raw breadth gap comes from repeat posting rather than from more distinct participants, and
the center-category check shows that some of the statistical accounting runs through
partisan versus non-partisan participation. We have narrowed the claims accordingly.

+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| #    | Raised by  | Reviewer point              | What we did                                                | Where        |
+======+============+=============================+============================================================+==============+
| C1   | SPC-1,     | Repost cascades are         | Rebuilt all 244,129 repost cascades under five alternative | App. D, Fig. |
|      | R1-R3      | reconstructed, not          | linking rules plus a 40-draw random-rule ensemble. The     | 9            |
|      |            | observed, so the            | platform interaction stays within [-0.012, +0.031] for     |              |
|      |            | cross-platform similarity   | breadth and [-0.083, +0.028] for depth under every rule,   |              |
|      |            | may be an artifact of the   | against +0.170 and -0.182 for replies. These cascades are  |              |
|      |            | method.                     | called reconstructed throughout.                           |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C2   | R3, SPC-1  | A user may have followed    | Both follow networks were collected at the same time, so   | App. D       |
|      |            | someone only after          | any timing gap applies equally to the two platforms.       |              |
|      |            | reposting, and that error   | Deleting 5 to 30 percent of follow edges at random moves   |              |
|      |            | could differ by platform.   | the breadth interaction from 0.031 to 0.005 and depth from |              |
|      |            |                             | 0.004 to 0.013, an order of magnitude below the reply      |              |
|      |            |                             | values.                                                    |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C3   | SPC-2, R2, | Unclear whether cascade     | Reply nodes are posts; ideology is aggregated per user, so | Methods;     |
|      | R3         | nodes are users or posts,   | a user who replies twice appears twice carrying the same   | App. E       |
|      |            | and how repeated users in   | label. We rebuilt every cascade under a user-collapsed     |              |
|      |            | one thread are handled.     | alternative. Depth holds; breadth shrinks, so part of      |              |
|      |            |                             | Truth Social's extra width is repeat posting rather than   |              |
|      |            |                             | more distinct participants. Both are reported.             |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C4   | SPC-3, R1, | Ideology validation is      | Added per-class and per-platform precision and recall with | App. B,      |
|      | R2         | thin: 200 items, no         | bootstrap intervals against the 171 consensus items.       | Table 3      |
|      |            | class-specific or           | Accuracy 0.86; kappa 0.77 between annotators, 0.61 and     |              |
|      |            | platform-specific           | 0.74 model to human. Minority-class precision is 0.64 on   |              |
|      |            | performance.                | both platforms against 0.98 and 0.93 for each platform's   |              |
|      |            |                             | largest class.                                             |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C5   | SPC-3, R2  | The center category may mix | Center is now defined explicitly as a residual category,   | Methods;     |
|      |            | genuinely centrist, mixed,  | and we report the models recomputed over partisan users    | App. B       |
|      |            | and uncertain users.        | only.                                                      |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C6   | SPC-3, R1  | Show the findings survive   | Propagated the measured confusion matrix through the full  | App. B       |
|      |            | plausible ideology label    | pipeline, 100 draws per scenario across four error         |              |
|      |            | noise.                      | processes including a worst case built on the stricter     |              |
|      |            |                             | annotator. The median run retains 86 to 96 percent of the  |              |
|      |            |                             | attenuation, the worst case 70 percent.                    |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C7   | R1         | The left share on Truth     | Our frame is Biden and Trump conversation during the       | App. B       |
|      |            | Social and right share on   | campaign, not the platform population. A prevalence        |              |
|      |            | Bluesky look too high       | correction lowers the Bluesky right share from 24.2 to     |              |
|      |            | against prior work.         | 17.9 percent and the Truth Social left share from 11.5 to  |              |
|      |            |                             | 1.9 percent. A gap remains for Bluesky, which we attribute |              |
|      |            |                             | to the sampling frame rather than to label error.          |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C8   | SPC-4, R2, | Alignment is computed from  | Causal language is replaced by statistical accounting      | Results;     |
|      | R3         | the same reply edges it is  | throughout, and the simultaneity is stated directly in the | Limitations  |
|      |            | meant to explain, so causal | Limitations subsection.                                    |              |
|      |            | language is too strong.     |                                                            |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C9   | SPC-5, R1  | Readable descriptive        | Added cascade counts, post counts, and the cascade size    | App. C,      |
|      |            | statistics are missing.     | distribution by platform and cascade type. Compiling it    | Table 4;     |
|      |            |                             | surfaced something we had not reported: most cascades are  | App. I,      |
|      |            |                             | a single post, and the share differs sharply by platform,  | Table 6      |
|      |            |                             | 71.3 percent of Bluesky reply cascades against 43.3        |              |
|      |            |                             | percent on Truth Social. Those cascades sit at the origin  |              |
|      |            |                             | of the scaling plot, so we refitted every model on the     |              |
|      |            |                             | 47,670 cascades with more than one post. The baseline      |              |
|      |            |                             | interaction is smaller there, as expected once those       |              |
|      |            |                             | points are removed, and Model 3c still absorbs 81 to 92    |              |
|      |            |                             | percent of it.                                             |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C10  | R1         | Report raw motif counts,    | Added observed counts for all 54 ideology-labeled          | App. G,      |
|      |            | not only z-scores.          | three-node motifs on both platforms.                       | Table 5      |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C11  | R1         | Motifs overlap; does the    | It does: randomized graphs are counted with the identical  | Methods;     |
|      |            | null model account for      | enumeration, so overlap inflates observed and null counts  | App. G       |
|      |            | that?                       | alike. We now state the procedure precisely, 100           |              |
|      |            |                             | randomizations by within-cascade double-edge swaps with 5m |              |
|      |            |                             | attempts, and note the consequence that neighboring        |              |
|      |            |                             | z-scores are correlated and should be read as a pattern    |              |
|      |            |                             | across families.                                           |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C12  | SPC-6, R2  | Main text says Huber loss,  | Huber. The stale appendix paragraph is removed. Checking   | Methods;     |
|      |            | appendix says OLS with HC3. | this surfaced a further problem: the default scale         | App. I,      |
|      |            | Which is it?                | estimate degenerates here because 61 percent of cascades   | Table 6      |
|      |            |                             | sit at the origin, so every model is refitted with Huber's |              |
|      |            |                             | proposal 2 scale and Table 1 now carries interpretable     |              |
|      |            |                             | standard errors. A sensitivity table reports both          |              |
|      |            |                             | estimators.                                                |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C13  | SPC-7,     | One month of Biden and      | The Discussion is rewritten so implications come first,    | Discussion   |
|      | R1-R3      | Trump content in an unusual | followed by a Limitations subsection stating what a        |              |
|      |            | period: what generalizes,   | candidate-centered, one-month frame does and does not      |              |
|      |            | and what do the two shapes  | support.                                                   |              |
|      |            | imply?                      |                                                            |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C14  | R1         | Excluding influencers only  | The follower distributions are not comparable at the top.  | App. A       |
|      |            | on Truth Social is poorly   | The largest Truth Social account has 56 times the          |              |
|      |            | justified; why not the top  | followers of that platform's own 99.9th percentile         |              |
|      |            | n percent on both?          | account; on Bluesky the ratio is 5. The top eight Truth    |              |
|      |            |                             | Social accounts hold 16 percent of all followers against 6 |              |
|      |            |                             | percent on Bluesky. A symmetric top-n rule would delete    |              |
|      |            |                             | ordinary Bluesky accounts to mirror a structure only Truth |              |
|      |            |                             | Social has.                                                |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C15  | R1         | How many reply chains are   | Among cascades of depth two or more, 68.8 percent on       | App. F       |
|      |            | two users going back and    | Bluesky and 70.0 percent on Truth Social contain an A to B |              |
|      |            | forth?                      | to A exchange, carrying 20.0 and 15.6 percent of reply     |              |
|      |            |                             | edges. The rate is similar on both platforms, so the depth |              |
|      |            |                             | difference is not explained by dyadic exchange.            |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C16  | R1         | More topic model detail;    | Added the corpus size (125,623 root posts), the outlier    | App. H       |
|      |            | BERTopic often leaves half  | reduction step and its settings, the resulting outlier     |              |
|      |            | of short posts as outliers. | share, and the eleven retained topics.                     |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C17  | R1         | The alignment results are   | The subsection is rewritten with a plain definition and a  | Methods      |
|      |            | hard to follow.             | worked example, and notation is defined at first use.      |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C18  | SPC-5, R1  | Figures are messy:          | Rebuilt the motif figures with family labels moved out of  | Figs. 1, 4,  |
|      |            | overlapping labels, unclear | the data area, and the topic, partisanship and             | 5, 7, 11     |
|      |            | quantities.                 | partial-residual figures, where value labels had been      |              |
|      |            |                             | falling outside the axes. All are 300 dpi with no          |              |
|      |            |                             | overlapping elements.                                      |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C19  | R1         | Citations are dated; add    | Added recent work on cascades and newer platforms,         | Related Work |
|      |            | recent                      | including the Bluesky diffusion literature.                |              |
|      |            | information-spreading work. |                                                            |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C20  | R1         | Unexplained notation and    | Notation and formatting pass.                              | Throughout   |
|      |            | stray paragraph indents.    |                                                            |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C21  | R1         | Another US-centric paper.   | Acknowledged in the Limitations subsection, with non-US    | Limitations  |
|      |            |                             | comparison named as future work.                           |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
