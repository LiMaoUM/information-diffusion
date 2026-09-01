---
title: "Response to Reviewers"
subtitle: "Paper 1217, Depth, Breadth, and Bias: Structural Diffusion of Political Content on Divergent Platforms"
geometry: margin=0.7in, landscape
fontsize: 9pt
---

We thank the reviewers and the SPC for their careful and constructive comments. We found the feedback very helpful in identifying several places where the original manuscript needed clearer definitions, additional validation, and more cautious interpretation. We have revised the manuscript substantially in response. All additions and major revisions are highlighted in blue in the revised manuscript.

The main changes include additional robustness checks for repost-cascade reconstruction, expanded validation of the ideology labels, clarification of the post-level unit of analysis, sensitivity analyses using alternative cascade and ideology specifications, and a revised regression specification after identifying a problem with the original scale estimator. We also expanded the descriptive reporting, motif analysis, and discussion of the scope and limitations of the findings.

Across these sensitivity analyses, estimates sometimes change substantially in magnitude, but none overturns the central conclusions. The depth difference persists across alternative samples and cascade representations, while the user-collapsed analysis shows that part of the post-level breadth difference reflects repeat participation. The association with ideological composition and interaction alignment remains substantial across alternative ideology thresholds, label-noise simulations, and composition specifications. Some checks also narrow the interpretation of particular findings. For example, collapsing repeated posts by the same user reduces the breadth difference, indicating that repeat participation contributes to the post-level breadth contrast, while the partisan-only analysis shows that the residual center category contributes to part of the statistical attenuation. We have revised the manuscript to make these qualifications explicit.

The first seven items are the seven points raised in the SPC decision, each answered once. Where a
reviewer raised the same issue it is listed in the Raised by column and answered there rather than
repeated. Items C8 onward are the remaining reviewer-specific comments.

+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| #    | Raised by  | Reviewer point              | Response                                                   | Where        |
+======+============+=============================+============================================================+==============+
| C1   | SPC-1,     | Repost cascades are         | We addressed this in two ways. First, we rebuilt all       | App. D, Fig. |
|      | R1-R3      | reconstructed rather than   | 244,129 repost cascades under alternative linking rules,   | 9            |
|      |            | directly observed, so the   | together with a 40-draw random-linking specification.      |              |
|      |            | apparent cross-platform     | Across these, the platform-by-size interaction stays       |              |
|      |            | similarity could depend on  | within [-0.012, +0.031] for breadth and [-0.083, +0.028]   |              |
|      |            | the reconstruction method   | for depth, against +0.170 and -0.182 for the corresponding |              |
|      |            | and on follower edges that  | reply cascades. Second, because follower edges are         |              |
|      |            | may postdate the repost.    | observed only in a post-collection snapshot without edge   |              |
|      |            |                             | creation dates, we simulated timing error by randomly      |              |
|      |            |                             | removing 5% to 30% of follower edges and reconstructing    |              |
|      |            |                             | the cascades; the resulting platform interactions remain   |              |
|      |            |                             | much smaller than the reply-cascade values. Both follower  |              |
|      |            |                             | networks were collected during the same period, although   |              |
|      |            |                             | we do not assume the timing error is identical across      |              |
|      |            |                             | platforms. We also now refer to these cascades as          |              |
|      |            |                             | reconstructed throughout.                                  |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C2   | SPC-2, R2, | It was unclear whether      | We have clarified that reply-cascade nodes are posts, not  | Methods;     |
|      | R3         | cascade nodes represent     | unique users. A user who replies multiple times therefore  | App. E       |
|      |            | users or posts, and how     | contributes multiple nodes carrying the same user-level    |              |
|      |            | repeated users are treated. | ideology label. We also reconstructed the reply cascades   |              |
|      |            |                             | after collapsing repeated appearances by the same user.    |              |
|      |            |                             | The depth difference remains under this representation,    |              |
|      |            |                             | whereas the breadth difference becomes substantially       |              |
|      |            |                             | smaller. We now state explicitly that part of the          |              |
|      |            |                             | post-level breadth contrast reflects repeat participation  |              |
|      |            |                             | within threads.                                            |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C3   | SPC-3, R1, | The ideology validation was | We addressed these concerns together because they all bear | App. B,      |
|      | R2         | limited, the Center         | on the measurement of ideology. We now report              | Table 3      |
|      |            | category may combine        | class-specific and platform-specific precision and recall  |              |
|      |            | moderate with uncertain     | with bootstrap intervals on the 171 cases where both       |              |
|      |            | classifications, and the    | annotators agreed. Accuracy against consensus is 0.86,     |              |
|      |            | manuscript should show      | inter-annotator kappa is 0.77, and model-human kappa is    |              |
|      |            | robustness to plausible     | 0.61 and 0.74. We also define the Center category          |              |
|      |            | label error.                | explicitly as a residual and recompute composition and     |              |
|      |            |                             | alignment using partisan users only, where the association |              |
|      |            |                             | remains substantial although smaller. Finally, we          |              |
|      |            |                             | propagate label error through Model 3c under four error    |              |
|      |            |                             | processes, with 100 draws per scenario. Median attenuation |              |
|      |            |                             | ranges from 70 to 86 percent, against 84 and 86 percent    |              |
|      |            |                             | with unperturbed labels, and even the least favorable draw |              |
|      |            |                             | retains 66 percent. The pooled-error scenario produces the |              |
|      |            |                             | largest reduction; the platform-specific and bootstrap     |              |
|      |            |                             | scenarios, closest to the error we measured, stay nearest  |              |
|      |            |                             | the unperturbed estimates.                                 |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C4   | SPC-4, R2, | Alignment is calculated     | We agree and have revised the manuscript throughout to     | Results;     |
|      | R3         | from the same reply edges   | avoid a causal interpretation of the alignment results.    | Limitations  |
|      |            | that determine cascade      | Alignment and cascade geometry are jointly realized        |              |
|      |            | structure, making causal    | features of the same conversation, so the observed         |              |
|      |            | language inappropriate.     | association does not establish a direction of influence.   |              |
|      |            |                             | The Results now state this limitation directly, and the    |              |
|      |            |                             | Discussion consistently describes composition and          |              |
|      |            |                             | alignment as statistically associated with the platform    |              |
|      |            |                             | difference rather than as causal mechanisms.               |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C5   | SPC-5, R1  | Descriptive reporting is    | We expanded the reporting throughout. Cascade counts, post | App. C, G,   |
|      |            | thin: cascade counts by     | counts and size distributions are now given as n (%) by    | H; Tables 4, |
|      |            | size, raw motif counts,     | platform and cascade type; this also made clear that       | 5            |
|      |            | whether the null model      | root-only cascades are common and far more frequent on     |              |
|      |            | handles motif overlap, and  | Bluesky, and because they carry no variation in breadth or |              |
|      |            | topic-model detail.         | depth we refit the models excluding them, where the        |              |
|      |            |                             | baseline platform difference is smaller but present and    |              |
|      |            |                             | the combined ideology specification still accounts for     |              |
|      |            |                             | most of it. We added observed counts for all 54            |              |
|      |            |                             | ideology-labeled three-node motifs on both platforms. We   |              |
|      |            |                             | describe the randomization and counting procedure          |              |
|      |            |                             | explicitly: the same overlapping-instance enumeration is   |              |
|      |            |                             | applied to observed and randomized graphs alike, so        |              |
|      |            |                             | overlap is treated consistently, and because motifs share  |              |
|      |            |                             | substructure we interpret the results as patterns across   |              |
|      |            |                             | motif families rather than as independent evidence. We     |              |
|      |            |                             | also report the topic-model corpus size (125,623 root      |              |
|      |            |                             | posts), the outlier reduction procedure and settings, the  |              |
|      |            |                             | remaining outlier share, and the eleven topic categories.  |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C6   | SPC-6, R2  | The main text describes     | We corrected this inconsistency. The main reply-cascade    | Methods;     |
|      |            | robust regression with      | models use the Huber loss. Reviewing the procedure, we     | App. I,      |
|      |            | Huber loss, whereas the     | found the default median-absolute-deviation scale          | Table 6      |
|      |            | appendix referred to OLS    | degenerates because of the large mass of root-only         |              |
|      |            | with HC3 standard errors.   | cascades, so the main models are refit with Huber's        |              |
|      |            |                             | Proposal 2 scale. Table 1 now reports the corresponding    |              |
|      |            |                             | estimates and standard errors, and Appendix I reports      |              |
|      |            |                             | sensitivity results across samples and scale estimators.   |              |
|      |            |                             | OLS with HC3 errors is used only for the                   |              |
|      |            |                             | repost-reconstruction robustness analysis, where the Huber |              |
|      |            |                             | scale estimate also degenerates.                           |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C7   | SPC-7,     | The paper examines one      | We revised the Discussion to state both the substantive    | Discussion;  |
|      | R1-R3      | month of Biden- and         | implications and the limits of the comparison, and         | Limitations  |
|      |            | Trump-related discussion    | gathered the limitations into a single section. We         |              |
|      |            | during an unusual election  | distinguish the structural interpretation of broad and     |              |
|      |            | period, in a single         | shallow from narrow and deep reply cascades from any       |              |
|      |            | national context. It should | normative evaluation of those forms, and make clear that   |              |
|      |            | clarify what generalizes    | the observed ideological composition and absolute cascade  |              |
|      |            | and what the two cascade    | statistics are specific to the candidate-centered sampling |              |
|      |            | shapes imply.               | frame and period studied here. We also state explicitly    |              |
|      |            |                             | that both platforms are U.S.-centered and the sampled      |              |
|      |            |                             | conversation is American, so the patterns describe a       |              |
|      |            |                             | single national context, and that comparable analyses      |              |
|      |            |                             | elsewhere are needed to establish how broadly they         |              |
|      |            |                             | generalize.                                                |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C8   | R1         | The estimated left share on | These estimates are not directly comparable to prior       | App. B;      |
|      |            | Truth Social and right      | platform-wide prevalence estimates because the studies     | Limitations  |
|      |            | share on Bluesky appear     | differ in both sampling frame and how ideology is          |              |
|      |            | high relative to previous   | measured. Our labels summarize expressed political         |              |
|      |            | estimates.                  | position within the sampled Biden and Trump discourse      |              |
|      |            |                             | rather than a context-independent ideology, and a position |              |
|      |            |                             | that appears left-leaning in a Trump-centered conversation |              |
|      |            |                             | need not imply a consistently left-wing position across    |              |
|      |            |                             | candidates or issues. We therefore do not treat exact      |              |
|      |            |                             | agreement with a particular platform-wide percentage as    |              |
|      |            |                             | the validation criterion. Instead, we evaluate whether the |              |
|      |            |                             | labels are interpretable and sufficiently reliable for the |              |
|      |            |                             | comparative analysis in which they are used. We report     |              |
|      |            |                             | human validation by class and platform, quantify the       |              |
|      |            |                             | direction and uncertainty of classification error, apply a |              |
|      |            |                             | prevalence correction, and propagate measured label error  |              |
|      |            |                             | through the downstream models. The correction reduces the  |              |
|      |            |                             | estimated Bluesky right share from 24.2% to 17.9% and the  |              |
|      |            |                             | Truth Social left share from 11.5% to 1.9%.                |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C9   | R1         | The rationale for examining | We expanded the justification for treating the unusually   | App. A       |
|      |            | highly followed users only  | high-follower Truth Social accounts separately. The upper  |              |
|      |            | on Truth Social was not     | tail is far more concentrated on Truth Social: the largest |              |
|      |            | sufficiently developed, and | account has 56 times the follower count of that platform's |              |
|      |            | a symmetric top-n rule      | 99.9th percentile account, against a ratio of 5 on         |              |
|      |            | might be preferable.        | Bluesky. We therefore treat this as a platform-specific    |              |
|      |            |                             | concentration rather than imposing the same numerical      |              |
|      |            |                             | cutoff on Bluesky, where the upper tail shows no           |              |
|      |            |                             | comparable separation.                                     |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C10  | R1         | How much of the observed    | We added a direct analysis of A -> B -> A exchanges. Among | App. F       |
|      |            | reply depth consists of     | cascades with depth of at least two, 68.8% on Bluesky and  |              |
|      |            | back-and-forth interaction  | 70.0% on Truth Social contain at least one such exchange.  |              |
|      |            | between the same two users? | These exchanges account for 20.0% and 15.6% of reply       |              |
|      |            |                             | edges, respectively. The similar prevalence across the two |              |
|      |            |                             | platforms suggests that the observed depth difference is   |              |
|      |            |                             | not primarily due to one platform containing more dyadic   |              |
|      |            |                             | back-and-forth conversations.                              |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C11  | R1         | The alignment results were  | We revised the alignment subsection to provide a clearer   | Methods      |
|      |            | difficult to follow.        | definition, define the notation at first use, and include  |              |
|      |            |                             | a worked example showing how the measure is calculated     |              |
|      |            |                             | within a reply tree.                                       |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C12  | R1         | Several figures were        | We revised the figures to improve readability, including   | Figs. 1, 4,  |
|      |            | difficult to read because   | moving motif-family labels outside the plotting area and   | 5, 7, 11     |
|      |            | of overlapping labels and   | correcting labels and annotations that extended beyond the |              |
|      |            | unclear plotted quantities. | axes. The revised figures are rendered at 300 dpi and no   |              |
|      |            |                             | longer contain overlapping elements.                       |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C13  | R1         | The literature review would | We updated the Related Work section with more recent work  | Related Work |
|      |            | benefit from more recent    | on cascade structure, information diffusion, and newer     |              |
|      |            | information-spreading work. | social media platforms, including recent work on Bluesky.  |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C14  | R1         | Some notation was           | We reviewed the manuscript for undefined notation,         | Throughout   |
|      |            | unexplained and formatting  | inconsistent terminology, and formatting issues and        |              |
|      |            | was inconsistent.           | corrected these throughout.                                |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+

\noindent\begin{minipage}{\textwidth}
We thank the reviewers and the SPC again for their careful and constructive feedback. Regardless of the
final decision, we feel that this revision has made the project substantially stronger and more robust.
The comments pushed us to clarify several parts of the analysis, add sensitivity checks that we had not
originally included, and narrow some of the interpretations where appropriate. We very much appreciate
the time the reviewers spent on the paper and the opportunity to revise it.
\end{minipage}
