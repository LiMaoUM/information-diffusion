# Response to Reviewers

Paper 1217: "Depth, Breadth, and Bias: Structural Diffusion of Political Content on Divergent Platforms"

Dear Editors and Reviewers,

Thank you for reviews that were detailed, fair, and unusually useful. Three concerns were shared across the reports: whether our reconstructed repost cascades could be producing the cross-platform similarity we report, whether our ideology labels are accurate enough to carry the paper's explanation, and whether our language claimed more than an association. We treated each as a question to answer with evidence, and each is now settled in the manuscript by an analysis added in this revision.

The main changes are:

1. **Repost reconstruction is validated.** We rebuilt all 244,129 repost cascades under five alternative linking rules plus a 40 draw random rule ensemble, and the cross platform similarity holds under every one of them. Our original rule turns out to be the most conservative case.
2. **Ideology labels are validated per class and per platform, and the results are shown to survive label noise.** We report class specific and platform specific accuracy, and a simulation that perturbs labels at the measured error rates leaves the central result intact in all 100 draws.
3. **Cascade definitions are stated explicitly**, and we report the analysis under both the post level and the user collapsed representation.
4. **Causal language is removed** and replaced with statistical accounting, together with an explicit discussion of why alignment cannot be treated as prior to cascade shape.
5. **New reporting**: descriptive statistics, raw motif counts, topic model details, two user exchange rates, and a specification sensitivity table.
6. **One error in the submitted version is corrected**: the inconsistent description of the regression estimator.

All additions are highlighted in blue in the revised manuscript.

We should say plainly that two of these analyses did not simply confirm what we had. The user collapsed representation shows that part of the raw breadth gap comes from repeat posting rather than from more distinct participants, and the center category check shows that some of the statistical accounting runs through partisan versus non partisan participation. We report both rather than only the stronger specification, and we have narrowed the claims accordingly.

---

## Response to the SPC meta-review

### SPC-1. Validate the repost cascade reconstruction, or narrow the paper's scope. Avoid calling reconstructed cascades "empirical."

We validated it, and we now call them reconstructed cascades throughout.

The key observation is that every admissible reconstruction rule chooses each repost's parent from the same candidate set: the original post, plus earlier reposters whom the reposting user follows. The rule only decides which candidate is selected. We therefore re-derived every repost cascade under the rule used in the paper, the earliest eligible reposter, the most recent eligible reposter, and a uniformly random eligible reposter drawn 40 times, each under both readings of the repost list order, over all 244,129 repost cascades and 4.03 million reposts.

No rule produces a platform by size interaction outside [-0.036, +0.038] for breadth or [-0.083, +0.028] for depth. The same estimator applied to reply cascades gives +0.170 and -0.182, five to eight times larger than any repost estimate under any rule. The rule used in the paper produced the largest apparent cross platform difference of any rule we tested, so every alternative moves the repost result further toward similarity, not away from it.

Two diagnostics explain why the rule has so little room to matter: 55 percent of Bluesky reposts and 59 percent of Truth Social reposts have no eligible prior reposter at all and attach to the root under every rule, and a further 21 percent and 12 percent have exactly one candidate.

**Changes**: new appendix section "Robustness of Repost Cascade Reconstruction" with Figure 5; terminology changed throughout; a pointer added in Results.

### SPC-2. Clarify cascade definitions and the treatment of repeated users.

Cascade nodes are messages, not accounts. A user who replies three times in a thread contributes three nodes. We state this at the start of the cascade construction appendix and in the definition of size, where the submitted text incorrectly described size as a count of unique users; the implementation counts nodes, and the corrected definition is now the one in the paper.

Because this choice affects depth and breadth, we rebuilt every reply cascade from the raw thread data and recomputed all metrics under both representations. The post level reconstruction reproduces the analysis file (correlations of 0.999 for size, 0.999 for breadth, 0.988 for depth), confirming the reported analyses are post level. Under user collapsing, the depth divergence persists and is again absorbed by ideology, but the breadth divergence is much smaller. We interpret and report this directly: part of the raw breadth gap reflects the same Truth Social users posting repeatedly within a thread rather than a wider set of participants. The depth contrast, which the discussion rests on, holds under both representations.

**Changes**: cascade construction appendix; corrected size definition; new appendix section "Unit of Analysis: Posts Versus Users."

### SPC-3. Strengthen ideology label validation; clarify the center category; test robustness to label noise.

We report all three.

*Per class and per platform performance.* Table 3 gives precision and recall for left, center, and right, pooled and separately by platform, against the 171 of 200 validation items where both annotators agree, each with a percentile bootstrap confidence interval over 2,000 resamples. Overall accuracy against consensus is 0.86, and inter annotator agreement is 0.77 with a bootstrap interval of 0.70 to 0.85. The errors are not uniform, and they run in the direction the reviewers suspected: the classifier over assigns each platform's ideological minority, with precision of 0.64 for right on Bluesky and 0.64 for left on Truth Social, against 0.93 and 0.91 for the corresponding majority classes.

*On the size of the validation sample.* We report bootstrap intervals for every rate so that the precision of each estimate is visible: the pooled rates and both majority classes are tightly estimated, while the two minority cells rest on 8 and 17 items and are correspondingly wide, from 0.33 to 0.91 for right on Bluesky. We then asked what the sample we have does and does not license, and added three checks.

First, the claim we make from this table is directional, that the classifier over assigns each platform's minority, and a directional claim can be tested even where individual cells are imprecise. Bootstrapping the difference between minority and majority precision within each platform gives -0.34 on Bluesky (one sided p = 0.017) and -0.29 on Truth Social (p = 0.004). The pattern is supported on both platforms.

Second, our rates are computed against the items where both annotators agree, and those are the easy ones, which flatters the classifier. Recomputing against each annotator alone gives lower precision throughout, most sharply for right on Bluesky (0.33 against Annotator 1 versus 0.64 against consensus), but the ordering is unchanged under all three references. We therefore use the single annotator rates as a worst case in the label noise analysis.

Third, the validation measures post level accuracy while the analysis uses user level labels formed by thresholding a user's posts. Simulating that threshold with the measured post level confusion and the observed posts per user raises accuracy from 0.85 to 0.88 on Bluesky and 0.81 to 0.87 on Truth Social, so perturbing user labels at post level rates overstates the error the pipeline carries.

Most directly, we no longer rely on a single point estimate of the error rates at all. The label noise analysis includes a nested bootstrap that resamples the validation set itself before perturbing labels, so the uncertainty in Table 3, including the wide interval on the thin cell, is carried through into the substantive result. Across 100 such draws the median draw still absorbs 96 percent of the baseline breadth gap and 94 percent of the depth gap, and the least favorable draw absorbs 72 and 70 percent. We also reran the analysis using the harsher error rates measured against a single annotator, where precision for right leaning labels on Bluesky is 0.33 rather than 0.64; there the least favorable draw absorbs 77 percent of each. Neither the imprecision of the validation sample nor a stricter reference standard restores the platform gap.

*On the labeling scheme.* The submitted text described a five category scheme. The five category model output is collapsed to three categories for all analyses, and the validation uses the collapsed scheme; the text now says so.

*The center category.* It is a residual: users whose posts reach neither the left nor the right threshold, mixing moderates, inconsistent posters, and unresolved cases. We now define it that way in the text, and we refit the models with composition and alignment computed over partisan users only. Ideology still absorbs 61 percent of the baseline breadth divergence and 64 percent of the depth divergence, against 99 percent and 77 percent under the published specification on the same sample. The conclusion is weaker without the center category but not reversed, and we report both.

*Label noise.* We perturbed every user's label by drawing from the measured confusion distribution and refit Model 3c, 100 times. Under platform specific error rates the platform by size interaction stays near zero in every draw, with at least 85 percent of the baseline divergence still absorbed in the least favorable draw and about 96 percent in the median draw. Under a deliberately harsher pooled matrix, roughly 70 percent is still absorbed. Because noise in a regressor attenuates its measured explanatory power, these are lower bounds on what error free labels would show.

**Changes**: rewritten validation appendix with Table 3; new sections "Robustness to Ideology Label Noise," "The Center Category," and "Ideological Composition Compared With Prior Estimates."

### SPC-4. Replace causal language with association; discuss endogeneity between alignment and cascade structure.

Agreed, and changed. We now describe Model 3c as statistically accounting for the platform difference rather than explaining it, and we added a paragraph in Results stating the problem in the reviewers' own terms: the alignment ratio is computed from the same reply edges whose arrangement produces depth and breadth, so it is not measured prior to the cascade. We note that influence plausibly runs in both directions, since a deep chain offers repeated opportunities for a reply to cross ideological lines while a wide star offers few, and we explicitly decline the claim that changing alignment would change cascade shape. We name the design that could separate the directions, predicting eventual shape from the ideology of a cascade's earliest replies, as future work rather than implying we have done it.

**Changes**: Results Section 4.2, Discussion, and the summary paragraph.

### SPC-5. Improve reporting of data, cascades by size, motif counts, and topic model details.

Added: a descriptive statistics table (cascade counts, message counts, and size distribution by platform and cascade type); a full table of raw counts for all 54 ideology labeled motifs on both platforms; and topic model details including corpus size, the outlier reduction step, and the resulting outlier share.

On motif dependence, which Reviewer 1 raised: overlapping motifs do not bias the z-scores, because the randomized graphs are counted with the identical enumeration, so overlap inflates observed and null counts alike. We now describe the null model precisely, including that rewiring happens within each cascade and preserves cascade sizes, out degrees, and each participant's ideology. We also state the real consequence of the dependence, which is that neighboring motif z-scores are correlated and should be read as a pattern across families rather than as independent evidence.

**Changes**: Table 2, Table 4, new sections "Motif Frequencies and the Null Model" and "Topic Model Details."

### SPC-6. Resolve the inconsistency between robust regression with Huber loss and OLS with HC3 standard errors.

The reviewers are right that these are different estimators and that the paper described both. The models reported in the paper are robust regressions with the Huber loss; the appendix paragraph describing OLS with HC3 was left over from an earlier specification and has been removed. The only analysis in the paper that uses OLS with HC3 is the new repost reconstruction check, for a reason we now state: repost cascades are small enough that log breadth and log depth take few distinct values, which makes the robust regression's scale estimate degenerate.

While checking this we found a related issue and have acted on it rather than leaving it for a reader to find. Because 61 percent of reply cascades consist of a root with no replies, and these sit exactly at the origin of the scaling plot, the default median absolute deviation scale collapses toward zero on the reply data. Every model in the submitted Table 1 was fitted that way, with scale estimates on the order of 1e-16, so the standard errors and significance stars in that table could not be interpreted.

We have therefore re-estimated Table 1 with Huber's proposal 2 scale, which does not degenerate here, and the table now reports asymptotic standard errors from that fit. Point estimates shift modestly, most visibly for Model 3c on breadth, from 0.023 to 0.071 against a baseline of 0.419. The substantive conclusion holds: composition and alignment together still remove roughly five sixths of the baseline platform gap, and a sensitivity table (Table 5) shows ideology absorbs between 81 and 94 percent of it across two samples and two scale estimators. We now state the estimator and the covariance basis explicitly rather than referring to robust standard errors.

Re-estimating the table surfaced two further errors in the submitted version, which we have corrected. The footnote to Model 2 stated that the interaction was with an outlier indicator; the model is in fact a three category platform variable (Bluesky, Truth Social non influencer, Truth Social influencer) and the reported entry is the Truth Social non influencer contrast against Bluesky. Model 4 for breadth was reported as 0.4286 while its own fit statistics correspond to 0.4086, and the Model 4 depth row did not reproduce from the stated specification. All Model 4 entries are now re estimated on the same sample as the other models. Neither correction affects the conclusion that topic does not account for the platform gap.

**Changes**: corrected appendix paragraph; new section "Sensitivity of the Modeling Specification" with Table 5.

### SPC-7. Sharpen implications and limitations.

The Discussion now says what the two conversation shapes mean. A wide, shallow cascade is an audience responding in parallel to one speaker, where many people are heard but rarely by each other; a narrow, deep cascade is fewer people taking turns, where positions must be restated or defended. Neither is inherently healthier, but they distribute attention differently, and the finding with the longest reach is that this difference arises between platforms whose reply mechanics are nearly identical. We also state a methodological implication: comparisons based on volume or on cascade size alone would record Truth Social as simply more active and would miss the divergence entirely, since it appears only after conditioning on size.

The limitations now separate what we claim travels from what does not. Absolute levels of depth and breadth, the ideological composition of each platform, and anything outside candidate centered political talk are bound to this setting. What we take to generalize is narrower: that identical reply mechanics can host systematically different conversational shapes, and that the difference tracks who participates and whether they talk across ideological lines.

**Changes**: Discussion and Limitations.

---

## Response to Reviewer 1

We appreciate the care of this review, and in particular that it identified the two assumptions the paper actually rested on. Both are now tested rather than assumed.

**R1-1. Repost reconstruction may be producing the similarity; "empirical" is the wrong word; either cut reposts or add substantial robustness checks.**
We added the checks rather than cutting the analysis, and the terminology is corrected. See SPC-1: five alternative rules plus a 40 draw random ensemble over the full data, with the platform interaction staying near zero throughout, and the paper's own rule turning out to be the least favorable to our claim. We also report how much freedom any rule has: more than half of all reposts attach to the root under every rule.

**R1-2. Ideology labelling: too many left users on Truth Social and right users on Bluesky relative to the literature.**
This was the right thing to be suspicious about, and the measurement confirms part of it. Our classifier does over assign each platform's minority: precision is 0.64 for right on Bluesky and 0.64 for left on Truth Social, against roughly 0.9 for each platform's majority class. Applying a prevalence correction that inverts the measured confusion matrix lowers our Bluesky right share from 24.2 percent to 17.9 percent and the Truth Social left share from 11.5 percent to 1.9 percent.

We want to be straightforward about what this does and does not settle. On Truth Social the correction is large and brings the left share close to what the reviewer would expect. On Bluesky a substantial gap with the 4.8 percent that Quelle and Bovet report for the whole platform remains after correcting for label error, so measurement error alone does not explain it. We attribute the remainder to the sampling frame: their estimate covers all users and is based on shared news domains, while ours covers accounts posting in Biden or Trump conversations during a campaign month and is based on post text. A candidate mention sample selects for politically engaged users on both sides, including those who arrive specifically to argue with the platform's majority. We now say in the paper that our ideological shares describe a conversation sample and should not be read as platform-wide estimates. We have added the published version of that paper and a more recent Bluesky study to the references. Finally, the label noise simulation shows the structural conclusions survive errors of the measured size.

**R1-3. Figures need cleaning up and further explanation.**
The new figure follows the journal's sizing with axis labels and a self contained caption. We are completing a pass over the existing figures for overlapping labels and unreadable quantities for the camera ready version; we would welcome specifics on which panels were hardest to read.

**R1-4. How many of each motif are actually present?**
Table 4 now reports the raw observed count of all 54 ideology labeled motifs on both platforms. This matters exactly as the reviewer suggests: on Truth Social, Left to Right chain motifs are strongly overrepresented against the null model while remaining rare in absolute terms, and the counts make that visible.

**R1-5. Motifs are not independent; does the randomisation account for this?**
It does, in the sense that matters for the z-scores: the randomized graphs are counted with the identical overlapping instance enumeration, so a chain of four contributing two three chains inflates observed and null counts alike. We now describe the null model precisely rather than in passing. The dependence does have a real consequence, which we now state: neighboring motif z-scores are correlated, so we read them as patterns across motif families rather than as 54 independent tests.

**R1-6. More information on topic modelling; BERTopic usually leaves half of short posts as outliers.**
Reported. The model covers 125,623 root posts. HDBSCAN does assign many short posts to the outlier class initially, and we applied BERTopic's outlier reduction step, after which 21 documents (0.02 percent) remain unassigned. We now state this explicitly along with the trade off it involves, since reassigned documents carry less certain labels. We also note that topic enters only as a categorical moderator and that the topic hypothesis is not supported, so this choice does not carry the paper's conclusions.

**R1-7. Far more information about the data; the CCDFs do not show how many cascades of size N exist.**
Table 2 now gives, for each platform and cascade type, the number of cascades, the number of messages, and the share of cascades in each size band.

**R1-8. How many reply chains are back and forth between the same two accounts?**
We measured it. Counting replies whose author also wrote the grandparent post, 9.3 percent of Bluesky cascades and 20.8 percent of Truth Social cascades contain such an exchange, which mostly reflects that Truth Social cascades are larger. Conditioning on cascades deep enough to allow the pattern, the shares are nearly identical: 68.8 percent on Bluesky and 70.0 percent on Truth Social, accounting for 20.0 and 15.6 percent of reply edges. Dyadic exchange is therefore common on both platforms to a very similar degree, so the depth difference is not explained by one platform hosting more two user conversations.

**R1-9. The alignment results are hard to follow.**
We added a worked example where alignment is defined: a four reply thread with three same stance edges and one crossing edge has an alignment ratio of 0.75, a ratio of 1 means nobody replies across ideological lines, and a ratio near 0 means almost everybody does. We also state that alignment is a property of the conversation rather than of a participant.

**R1-10. The justification for looking at influencers only on Truth Social is poor; why not the top n percent on both platforms?**
The reviewer is right that "no natural gap" understated the reasoning, and we replaced it with the measurement it rested on. The top Truth Social account has 56 times the followers of that platform's own 99.9th percentile account, and the top eight accounts hold 16.4 percent of all follower ties; on Bluesky the top account sits 5 times above the 99.9th percentile and the top eight hold 6.1 percent. The eighth largest Truth Social account is larger than the largest Bluesky account in our data. A symmetric top n percent rule would not equalize the comparison: on Bluesky it would remove accounts continuous with the rest of the distribution, while on Truth Social any n large enough to capture the separated group would also sweep in ordinary accounts. We treat influencer dominance as a property of Truth Social because in these data it is one, and the appendix now reports these statistics next to the follower distribution.

**R1-11. Another US centric paper.**
A fair observation, and now stated as a limitation rather than left implicit. We note that both platforms are US centered, that the alignment of partisanship with platform choice is itself a feature of this ecosystem, and that applying the same design outside the United States is the natural next step.

**R1-12. Unexplained notation; stray paragraph indents; the literature is somewhat old.**
We have defined notation at first use in the revised sections and fixed the stray indentation. On references, we updated the Bluesky ideology study to its published version and added more recent work on Bluesky politics. We would welcome pointers to specific recent work the reviewer has in mind, since this is a fast moving area and their expertise here is evident.

---

## Response to Reviewer 2

**R2-1. The "ideology explains the platform gap" claim is too strong, because alignment is measured from the same completed cascades.**
Agreed, and the claim is now stated as association. See SPC-4. The endogeneity is described explicitly, including the specific mechanism the reviewer implies: alignment may partly describe a cascade after it has formed rather than shaping it.

**R2-2. The cascade construction is ambiguous; are nodes users or posts?**
Nodes are posts. See SPC-2, including the correction of the size definition in the submitted appendix and the new analysis under both representations.

**R2-2.1. Repost path inference depends on assumptions about follower relationships and timing.**
See SPC-1 for the reconstruction rule, and the response to Reviewer 3's second point for the follow network timing.

**R2-3. One month and only Biden or Trump discussion limits generalizability.**
Accepted, and the limitations section now separates what we claim generalizes from what does not. See SPC-7.

**R2-4. Only 200 human labeled replies, with insufficient class specific and platform specific performance; the center category is mixed.**
All three are addressed in SPC-3: per class and per platform metrics are now reported, the center category is defined as a residual and tested by refitting without it, and a label noise simulation shows the conclusions survive errors at the measured rates. We agree that 200 items is a small sample for per class per platform estimates, and we have tried to be precise about which statements it supports. Every rate now carries a bootstrap interval, the directional claim is tested separately and holds on both platforms, the rates are shown to be robust to which annotator is used as the reference, and the label noise analysis carries the uncertainty in the validation table through into the substantive result via a nested bootstrap. Where the sample is thin we say so rather than presenting all cells as equally well established.

**R2-5. The methods describe robust regression with Huber loss while the appendix says linear regression with HC3.**
Corrected, and the check turned up a related issue in the scale estimator that we also report. See SPC-6.

---

## Response to Reviewer 3

**R3-1. Alignment ratio and cascade structure may influence each other.**
Agreed, and this is now stated in the paper using the reviewer's own example: deep cascades and star like cascades offer different opportunities for cross ideological interaction, so the arrow can run from structure to alignment as well as the reverse. We weakened the interpretation accordingly rather than claiming a direction we cannot establish. See SPC-4.

**R3-2. A user may have followed someone only after reposting, and this error may differ between platforms.**
This is a real limitation of our data and we now say so directly. Neither platform's API exposes follow creation dates, so the follower network is a post collection snapshot and we cannot verify that a follow predates a repost. We can bound the consequence in two ways. First, the error has a known direction: a follow edge that did not yet exist can only move a repost away from the root and onto an interior parent, and that displacement is already spanned by the alternative rules in the reconstruction analysis, whose results are stable. Second, we ran a direct sensitivity check, deleting follow edges at random at rates from 5 to 30 percent to simulate edges that did not yet exist at repost time, and re-estimating the platform interaction. Across that range the breadth interaction moves from 0.031 to 0.005 and the depth interaction from 0.004 to 0.013, both an order of magnitude smaller than the reply cascade values of +0.170 and -0.182. Even a substantial rate of spurious follow edges would therefore not disturb the conclusion. On platform asymmetry, the reviewer is right that it could differ: the two follow networks were collected at different times relative to the posting window, and Truth Social users in our sample have larger followee sets, so it has more candidate edges that could be affected. We state this in the appendix rather than assuming symmetry.

**R3-3. In reply cascades the same user may appear multiple times; how is this handled?**
Each appearance is a separate node. See SPC-2 for the definition and the new analysis under the user collapsed alternative, which we ran precisely because this choice affects depth and breadth.

**R3-4. The practical and theoretical implications are unclear, as is whether results are specific to the 2024 US election.**
Addressed in SPC-7. The Discussion now states what the two conversational shapes mean for how attention is distributed and for how platform comparisons should be measured, and the limitations state plainly which findings are bound to this election context.

---

We are grateful for the reviews. The requirement to validate the reconstruction and the labels made this a better paper, and in two cases it changed what we claim.
