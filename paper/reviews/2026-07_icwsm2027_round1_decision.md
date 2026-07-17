# ICWSM 2027 Round 1 Decision: Revise and Resubmit

- Paper: 1217, "Depth, Breadth, and Bias: Structural Diffusion of Political Content on Divergent Platforms"
- Received: 2026-07-17
- Revision deadline: **2026-09-15**, same submission system ("ICWSM 27 May submissions")
- Next decision is final accept/reject, no further revision rounds
- Required: revised paper with changes highlighted in color, plus a separate response document

Scores: R1 weak reject (-1, confidence 4, expertise 5), R2 borderline (0, conf 3, exp 2), R3 weak accept (+1, conf 3, exp 3), plus SPC meta-review with explicit roadmap.

---

## Verbatim reviews

### SPC / Associate Editor (reviewer 4), meta-review and roadmap

The reviewers agree that this paper addresses a timely and relevant question for ICWSM: how political information diffusion differs across Truth Social and Bluesky during the 2024 U.S. election period. They view the cross-platform comparison as valuable, especially because both platforms remain relatively understudied in comparative social media research. Reviewers also consistently appreciated the paper's distinction between repost cascades as amplification and reply cascades as conversation, and they found the main empirical pattern interesting: repost cascades appear structurally similar across platforms, while reply cascades differ, with Truth Social tending toward broader and shallower conversations and Bluesky toward narrower and deeper ones. The paper was also generally viewed as well written, appropriately cautious in parts, and relevant to the computational social science community.

At the same time, the reviews identify several important methodological and interpretive concerns that need to be addressed before the contribution can be fully evaluated. The most serious concerns involve the construction and interpretation of cascades, the validity of ideological labels, and the strength of the paper's explanatory claims.

First, all reviewers raised concerns about the inferred construction of repost cascades. Because repost paths are not directly observed and must be reconstructed using follower relationships and timing, the results may depend strongly on the assumptions used in the reconstruction procedure. Reviewers were particularly concerned that the apparent similarity between platforms in repost cascade structure could be an artifact of the reconstruction method. A revised version should either provide substantial validation and robustness checks for the repost cascade reconstruction procedure or narrow the paper's scope to focus more centrally on reply cascades, where the observed interaction structure is clearer. The paper should also avoid describing reconstructed repost cascades as purely "empirical" unless this terminology is carefully qualified.

Second, the definition of cascade nodes and edges requires clarification. Reviewers were uncertain whether cascade nodes represent users, posts, replies, or some combination of these. This is especially important for reply cascades, where the same user may appear multiple times in a thread. Collapsing repeated posts by the same user into a single user node could substantially alter measured depth, breadth, and motif structure. A revision should explicitly define the unit of analysis for each cascade type and explain how repeated participation by the same user is handled.

Third, the paper's ideology-based analyses need stronger validation and more cautious interpretation. Several reviewers were concerned that the ideological labels may be noisy, especially given the surprising presence of many left-leaning users on Truth Social and right-leaning users on Bluesky. Since ideology is central to the paper's explanation of platform differences, the authors should provide stronger evidence that the labeling procedure is valid. This could include more detailed human validation, platform-specific and class-specific performance metrics, sensitivity analyses under alternative labeling assumptions, and a clearer treatment of the "center" category, which may conflate genuinely centrist users, mixed-position users, and uncertain classifications.

Fourth, the paper should weaken or better justify causal language around ideological composition and interaction alignment. Reviewers noted that alignment is computed from the same reply edges that form the cascades whose structure the paper seeks to explain. As a result, alignment may not be a prior cause of cascade shape; it may partly describe the cascade after it has already formed. The safer claim is that ideology and alignment are associated with platform differences in cascade structure, unless the authors can provide additional analysis supporting a causal interpretation.

Fifth, the revision should improve methodological transparency and presentation. Reviewers asked for clearer reporting of dataset size, collection period, numbers of cascades by size, motif counts, topic modeling details, and the number of back-and-forth reply chains involving the same two users. The paper should also reconcile an apparent inconsistency between the description of robust regression with Huber loss and the appendix's reference to linear regression with HC3 robust standard errors. Figures should be cleaned up where labels overlap or where the plotted quantities are difficult to interpret.

Finally, the authors should strengthen the discussion of implications and limitations. The paper studies a one-month period focused on Biden/Trump-related content during an unusual U.S. election context. This does not invalidate the contribution, but the revised paper should more clearly explain what can and cannot be generalized from these results. The implications of broader/shallow versus narrower/deeper reply structures should also be made more explicit, both theoretically and practically.

For a revise-and-resubmit version, the authors should prioritize the following changes:

1. Validate or revise the repost cascade analysis. Provide robustness checks showing how results change under alternative repost reconstruction assumptions, including assumptions about follower-network timing and inferred exposure paths. If this cannot be done convincingly, consider reducing the emphasis on repost cascades or focusing the main contribution on reply cascades.
2. Clarify cascade definitions. Explicitly state whether nodes are users, posts, or interactions; how repeated users in the same thread are represented; and how these choices affect depth, breadth, and motif analyses.
3. Strengthen ideology-label validation. Report more detailed validation results, including class-specific and platform-specific performance. Clarify the meaning of the center category and test whether the main findings are robust to plausible ideology-label noise.
4. Reframe explanatory claims. Replace causal language suggesting that ideology and alignment "explain" platform differences with more cautious language about association, unless additional analyses can support a stronger claim. Discuss the potential endogeneity between alignment ratio and cascade structure.
5. Improve reporting of data and robustness. Add readable descriptive statistics on posts, cascades, cascade sizes, motif frequencies, user participation patterns, and topic model outputs. Include motif counts and explain how dependence among motifs is handled.
6. Resolve methodological inconsistencies. Ensure that the regression estimator, loss function, and standard error procedure are described consistently across the main text and appendix.
7. Sharpen implications and limitations. Explain why the observed structural differences matter, what they suggest about platformed political discussion, and how limited the conclusions are to the 2024 U.S. election context and Biden/Trump-related discourse.

### Reviewer 1 (weak reject, confidence 4, methodological expertise 5)

Summary: compares reply and repost cascade structure on Truth Social and Bluesky in the month before the 2024 US election; asks whether platforms with similar affordances but different ideological user groups differ in breadth/depth of reply and repost trees; finds limited differences once controls are included, with Truth Social broader/shallower and Bluesky narrower/deeper.

Reasons to accept:
1. Cross-platform analysis is novel and an important contribution to information spreading literature.
2. Comprehensive and largely robust analysis, though some details are missing.
3. The platforms are understudied in comparative work; major novelty warranting publication.
4. Excellent framing in existing literature, avoids overstating results, professional and balanced tone.
5. Cascade motif analysis is interesting, especially for comparing platforms.

Reasons to reject:
1. Repost cascades are reconstructed deterministically (Section 3.2); no way to gauge realism. Concern that similar repost structure across platforms is an artifact of reconstruction. Wants substantial validation and robustness checks on the reconstruction mechanism. "Empirical" is the wrong word for modelled cascades. Suggests either (1) cutting repost content and focusing on reply trees, or (2) adding substantial robustness checks.
2. Ideology labelling: finds far more left users on Truth Social and right users on Bluesky than the literature suggests (cites Bovet team Bluesky paper, Zurich). Questions labelling validity and all downstream results. Also notes most users are not politically engaged. Wants convincing validation.

General comments:
1. Figures need cleaning up / further explanation.
2. Likes Figures 4 and 5, but wants counts of each motif present in the dataset, crucial for assessing validity.
3. Motifs are not independent (a chain of length 4 contains two 3-chains); does the randomisation account for this, and how does it affect results?
4. More info on topic modelling; BERTopic often throws 50%+ of short posts into the outlier group.
5. Far more info about the data: how many posts, over how long; CCDFs do not show how many cascades of size N exist; provide readable counts.
6. How many reply chains are back-and-forth conversations between the same two accounts?
7. Does not fully follow the alignment results; wants clearer explanation.
8. Justification for looking at influencers only on TS is poor (lack of natural gap is not a justification); why not top n% of accounts on both platforms?
9. Dislikes US-centricity (not a reason to reject).

Presentation: fair bit of unexplained notation; messy figures with overlapping labels; LaTeX paragraph indents where there should not be any.
Citations: appropriate and comprehensive but old; update with recent information-spreading papers.

### Reviewer 2 (borderline, confidence 3, methodological expertise 2)

Summary: compares political diffusion on Truth Social and Bluesky, distinguishing repost (amplification) from reply (conversation) cascades; headline asymmetry conditional on cascade size; attributes the difference to ideological composition and interaction alignment, with ideology-labelled three-node motifs for local interpretation.

Reasons to accept:
1. Timely question, relevant to ICWSM.
2. Clear and interesting main result; amplification vs conversation distinction valuable; the repost-similar/reply-different finding is the strongest contribution.
3. Clearly written, generally rigorous, appropriately cautious about causality in the discussion.

Reasons to reject:
1. "Ideology explains the platform gap" claim is too strong: composition and alignment are measured from the same completed reply cascades whose depth and breadth they explain; partly endogenous; alignment may describe the cascade after it formed. Safer claim: associated, not causal.
2. Cascade construction ambiguous: paper says nodes are users, but reply trees are built from posts/replies; collapsing repeated replies by the same user into one node can change depth, breadth, motif structure. Clarify.
2.1 Repost path inference depends on assumptions about follower relationships and timing; justify further.
3. One month, Biden/Trump only, unusual campaign period; limited generalizability.
4. LLM-based stance validation uses only 200 human-labeled replies; no class-specific or platform-specific performance; the center category conflates centrist, mixed, and uncertain users; label noise could directly affect results.

Comment: methods describe robust regression with Huber loss, appendix says linear regression with HC3 robust standard errors; different estimators; double check.

### Reviewer 3 (weak accept, confidence 3, methodological expertise 3)

Summary: analyzes cascade structure for US presidential campaign content across Truth Social and Bluesky (posts mentioning Trump or Biden); repost cascades similar across platforms, reply cascades differ substantially; argues ideological composition and interaction alignment jointly account for most platform-level differences.

Reasons to accept:
1. Timely and interesting topic for computational social science.
2. Explicit repost/reply distinction is useful.
3. Structural evaluation approach interesting and reasonable.
4. Well written and organized.

Reasons to reject:
1. Concerns about repost cascade construction affecting results.
2. Some cascade definition aspects unclear.
3. Alignment ratio and cascade structure may influence each other.

Detailed comments:
1. Ideological alignment: alignment ratio is computed from the reply edges constituting the cascades, so causality can run both ways; deep vs star-like cascades provide different opportunities for cross-ideological interaction; weaken causal interpretation or strengthen the analysis.
2. Repost cascades: timing of post collection vs follower-network collection may affect inference. If u followed v only after reposting v's post, inferring flow from v to u is inappropriate. Discuss this inference error and whether it differs systematically between platforms.
3. Cascade definition: can the same user appear as multiple nodes in one cascade, or are nodes strictly users with only one appearance counted? Clarify.
4. Implications and limitations: practical and theoretical implications unclear; unclear whether results are specific to the 2024 US election context; discuss more carefully.

Presentation: good. Citations: OK. Ethical concerns: none.
