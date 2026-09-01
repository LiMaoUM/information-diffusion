---
title: "Response to Reviewers"
subtitle: "Paper 1217, Depth, Breadth, and Bias: Structural Diffusion of Political Content on Divergent Platforms"
geometry: margin=0.7in, landscape
fontsize: 9pt
---

We thank the reviewers and the SPC for their careful and constructive comments. We found the feedback very helpful in identifying several places where the original manuscript needed clearer definitions, additional validation, and more cautious interpretation. We have revised the manuscript substantially in response. All additions and major revisions are highlighted in blue in the revised manuscript.

The main changes include additional robustness checks for repost-cascade reconstruction, expanded validation of the ideology labels, clarification of the post-level unit of analysis, sensitivity analyses using alternative cascade and ideology specifications, and a revised regression specification after identifying a problem with the original scale estimator. We also expanded the descriptive reporting, motif analysis, and discussion of the scope and limitations of the findings.

Across these sensitivity analyses, estimates sometimes change in magnitude, but the substantive conclusions remain consistent. In particular, the cross-platform difference in reply-cascade structure persists under alternative samples and cascade representations, and ideological composition and interaction alignment remain strongly associated with the remaining platform difference under alternative ideology thresholds, label-noise simulations, and composition specifications. Some checks also narrow the interpretation of particular findings. For example, collapsing repeated posts by the same user reduces the breadth difference, indicating that repeat participation contributes to the post-level breadth contrast, while the partisan-only analysis shows that the residual center category contributes to part of the statistical attenuation. We have revised the manuscript to make these qualifications explicit.

+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| #    | Raised by  | Reviewer point              | Response                                                   | Where        |
+======+============+=============================+============================================================+==============+
| C1   | SPC-1,     | Repost cascades are         | We agree that the reconstructed repost cascades require    | App. D, Fig. |
|      | R1-R3      | reconstructed rather than   | additional validation. We therefore rebuilt all 244,129    | 9            |
|      |            | directly observed, so the   | repost cascades under alternative linking rules, together  |              |
|      |            | apparent cross-platform     | with a 40-draw random-linking specification. Across these  |              |
|      |            | similarity could depend on  | specifications, the platform-by-size interaction stays     |              |
|      |            | the reconstruction method.  | within [-0.012, +0.031] for breadth and [-0.083, +0.028]   |              |
|      |            |                             | for depth, against +0.170 and -0.182 for the corresponding |              |
|      |            |                             | reply cascades. We also now refer to these cascades as     |              |
|      |            |                             | reconstructed throughout the manuscript.                   |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C2   | R3, SPC-1  | A follower relationship     | This is an important limitation because the follower       | App. D       |
|      |            | observed after collection   | networks are post-collection snapshots without edge        |              |
|      |            | may not have existed at the | creation dates. To examine the sensitivity of the results  |              |
|      |            | time of reposting.          | to this uncertainty, we randomly removed 5% to 30% of      |              |
|      |            |                             | follower edges and reconstructed the cascades. The         |              |
|      |            |                             | resulting platform interactions remain much smaller than   |              |
|      |            |                             | those observed for reply cascades. Both follower networks  |              |
|      |            |                             | were collected during the same period, although we do not  |              |
|      |            |                             | assume that timing error is necessarily identical across   |              |
|      |            |                             | platforms.                                                 |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C3   | SPC-2, R2, | It was unclear whether      | We have clarified that reply-cascade nodes are posts, not  | Methods;     |
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
| C4   | SPC-3, R1, | The ideology validation     | We expanded the validation analysis to report              | App. B,      |
|      | R2         | sample was limited and did  | class-specific and platform-specific precision and recall, | Table 3      |
|      |            | not report class- or        | with bootstrap confidence intervals, using the 171 cases   |              |
|      |            | platform-specific           | on which the two human annotators agreed. Overall accuracy |              |
|      |            | performance.                | against the consensus labels is 0.86. Inter-annotator      |              |
|      |            |                             | agreement is kappa = 0.77, and model-human agreement is    |              |
|      |            |                             | kappa = 0.61 and 0.74. We also report the lower precision  |              |
|      |            |                             | observed for the ideological minority class on each        |              |
|      |            |                             | platform and its associated uncertainty.                   |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C5   | SPC-3, R2  | The center category may     | We agree that the center category should not be treated as | Methods;     |
|      |            | combine genuinely moderate  | a homogeneous group of ideological moderates. We now       | App. B       |
|      |            | users with mixed or         | define it explicitly as a residual category that includes  |              |
|      |            | uncertain classifications.  | users who do not reach either partisan threshold. We also  |              |
|      |            |                             | recomputed composition and alignment using partisan users  |              |
|      |            |                             | only. The association with ideological structure becomes   |              |
|      |            |                             | smaller but remains substantial, and we have narrowed the  |              |
|      |            |                             | interpretation of the center category accordingly.         |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C6   | SPC-3, R1  | The manuscript should show  | We propagated the observed classification error through    | App. B       |
|      |            | whether the findings are    | the full analysis rather than treating the ideology labels |              |
|      |            | robust to plausible         | as error-free. For each error specification, user labels   |              |
|      |            | ideology-label error.       | were perturbed, cascade-level composition and alignment    |              |
|      |            |                             | were recomputed, and Model 3c was refit over 100 draws. We |              |
|      |            |                             | considered platform-specific errors, pooled errors,        |              |
|      |            |                             | uncertainty in the estimated confusion matrix, and the     |              |
|      |            |                             | less favorable performance measured against a single       |              |
|      |            |                             | annotator. Across these simulations the median run retains |              |
|      |            |                             | 86 to 96 percent of the attenuation and the least          |              |
|      |            |                             | favorable specification retains 70 percent, so the         |              |
|      |            |                             | substantive conclusion is unchanged.                       |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C7   | R1         | The estimated left share on | We agree that the ideological composition in our sample    | App. B       |
|      |            | Truth Social and right      | differs from platform-wide estimates reported in previous  |              |
|      |            | share on Bluesky appear     | work. We now emphasize that our sampling frame consists of |              |
|      |            | high relative to previous   | users participating in Biden- or Trump-related             |              |
|      |            | estimates.                  | conversations during one campaign month rather than a      |              |
|      |            |                             | representative sample of either platform. We also applied  |              |
|      |            |                             | a prevalence correction based on the measured confusion    |              |
|      |            |                             | matrix. This reduces the estimated Bluesky right share     |              |
|      |            |                             | from 24.2% to 17.9% and the Truth Social left share from   |              |
|      |            |                             | 11.5% to 1.9%. A remaining difference for Bluesky may      |              |
|      |            |                             | partly reflect the candidate-centered sampling frame, and  |              |
|      |            |                             | we now state this cautiously in the manuscript.            |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C8   | SPC-4, R2, | Alignment is calculated     | We agree and have revised the manuscript throughout to     | Results;     |
|      | R3         | from the same reply edges   | avoid a causal interpretation of the alignment results.    | Discussion   |
|      |            | that determine cascade      | Alignment and cascade geometry are jointly realized        |              |
|      |            | structure, making causal    | features of the same conversation, so the observed         |              |
|      |            | language inappropriate.     | association does not establish a direction of influence.   |              |
|      |            |                             | The Results now state this limitation directly, and the    |              |
|      |            |                             | Discussion consistently describes composition and          |              |
|      |            |                             | alignment as statistically associated with the platform    |              |
|      |            |                             | difference rather than as causal mechanisms.               |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C9   | SPC-5, R1  | The paper needs more        | We added a descriptive table reporting cascade counts,     | App. C,      |
|      |            | readable descriptive        | post counts, and cascade-size distributions by platform    | Table 4;     |
|      |            | statistics, including the   | and cascade type. This also made clear that root-only      | App. I,      |
|      |            | number of cascades at       | cascades are common and substantially more frequent on     | Table 6      |
|      |            | different sizes.            | Bluesky than on Truth Social. Because these observations   |              |
|      |            |                             | contain no variation in breadth or depth, we additionally  |              |
|      |            |                             | refit the models after excluding size-1 cascades. The      |              |
|      |            |                             | baseline platform difference becomes smaller, but remains  |              |
|      |            |                             | present, and the combined ideology specification continues |              |
|      |            |                             | to account for most of the remaining difference.           |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C10  | R1         | Raw motif counts are needed | We added the observed counts for all 54 ideology-labeled   | App. G,      |
|      |            | in addition to standardized | three-node motifs on both platforms so that the            | Table 5      |
|      |            | motif scores.               | standardized motif results can be interpreted alongside    |              |
|      |            |                             | their absolute frequency.                                  |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C11  | R1         | Motifs overlap, so the      | We now describe the motif randomization and counting       | Methods;     |
|      |            | manuscript should explain   | procedure explicitly. The same overlapping-instance        | App. G       |
|      |            | whether the null model      | enumeration is applied to both the observed and randomized |              |
|      |            | accommodates this           | graphs, so overlap is treated consistently in both. We     |              |
|      |            | dependence.                 | also clarify that neighboring motif statistics are         |              |
|      |            |                             | correlated because motifs share substructure. Accordingly, |              |
|      |            |                             | we interpret the motif results as patterns across motif    |              |
|      |            |                             | families rather than as independent pieces of evidence.    |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C12  | SPC-6, R2  | The main text describes     | We corrected this inconsistency. The main reply-cascade    | Methods;     |
|      |            | robust regression with      | models use the Huber loss. In reviewing the estimation     | App. I,      |
|      |            | Huber loss, whereas the     | procedure, we also found that the default                  | Table 6      |
|      |            | appendix referred to OLS    | median-absolute-deviation scale estimate degenerates       |              |
|      |            | with HC3 standard errors.   | because of the large mass of root-only cascades. We        |              |
|      |            |                             | therefore refit the main models using Huber's Proposal 2   |              |
|      |            |                             | scale, which does not degenerate in these data. Table 1    |              |
|      |            |                             | now reports the corresponding estimates and standard       |              |
|      |            |                             | errors, and Appendix I reports sensitivity results across  |              |
|      |            |                             | samples and scale estimators. OLS with HC3 errors is used  |              |
|      |            |                             | only for the repost-reconstruction robustness analysis,    |              |
|      |            |                             | where the Huber scale estimate also degenerates.           |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C13  | SPC-7,     | The paper examines one      | We revised the Discussion to state more clearly both the   | Discussion   |
|      | R1-R3      | month of Biden- and         | substantive implications and the limits of the comparison. |              |
|      |            | Trump-related discussion    | In particular, we distinguish the structural               |              |
|      |            | during an unusual election  | interpretation of broad/shallow and narrow/deep reply      |              |
|      |            | period. The paper should    | cascades from any normative evaluation of those forms. We  |              |
|      |            | clarify what can be         | also make clear that the observed ideological composition  |              |
|      |            | generalized and what the    | and absolute cascade statistics are specific to the        |              |
|      |            | two cascade shapes imply.   | candidate-centered sampling frame and period studied here. |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C14  | R1         | The rationale for examining | We expanded the justification for treating the unusually   | App. A       |
|      |            | highly followed users only  | high-follower Truth Social accounts separately. The upper  |              |
|      |            | on Truth Social was not     | tail of the follower distribution is substantially more    |              |
|      |            | sufficiently developed, and | concentrated on Truth Social: the largest account has 56   |              |
|      |            | a symmetric top-n rule      | times the follower count of the platform's 99.9th          |              |
|      |            | might be preferable.        | percentile account, compared with a ratio of 5 on Bluesky, |              |
|      |            |                             | and the top eight accounts account for a much larger share |              |
|      |            |                             | of total followers. We therefore treat the Truth Social    |              |
|      |            |                             | group as a platform-specific concentration rather than     |              |
|      |            |                             | imposing the same numerical cutoff on Bluesky, where there |              |
|      |            |                             | is no comparable separation in the upper tail.             |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C15  | R1         | How much of the observed    | We added a direct analysis of A -> B -> A exchanges. Among | App. F       |
|      |            | reply depth consists of     | cascades with depth of at least two, 68.8% on Bluesky and  |              |
|      |            | back-and-forth interaction  | 70.0% on Truth Social contain at least one such exchange.  |              |
|      |            | between the same two users? | These exchanges account for 20.0% and 15.6% of reply       |              |
|      |            |                             | edges, respectively. The similar prevalence across the two |              |
|      |            |                             | platforms suggests that the observed depth difference is   |              |
|      |            |                             | not primarily due to one platform containing more dyadic   |              |
|      |            |                             | back-and-forth conversations.                              |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C16  | R1         | More detail is needed about | We expanded the topic-model description to report the      | App. H       |
|      |            | the BERTopic procedure and  | corpus size (125,623 root posts), the HDBSCAN outlier      |              |
|      |            | the treatment of outliers.  | reduction procedure and settings, the remaining outlier    |              |
|      |            |                             | share, and the eleven topic categories used in the         |              |
|      |            |                             | analysis.                                                  |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C17  | R1         | The alignment results were  | We revised the alignment subsection to provide a clearer   | Methods      |
|      |            | difficult to follow.        | definition, define the notation at first use, and include  |              |
|      |            |                             | a worked example showing how the measure is calculated     |              |
|      |            |                             | within a reply tree.                                       |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C18  | SPC-5, R1  | Several figures were        | We revised the figures to improve readability, including   | Figs. 1, 4,  |
|      |            | difficult to read because   | moving motif-family labels outside the plotting area and   | 5, 7, 11     |
|      |            | of overlapping labels and   | correcting labels and annotations that extended beyond the |              |
|      |            | unclear plotted quantities. | axes. The revised figures are rendered at 300 dpi and no   |              |
|      |            |                             | longer contain overlapping elements.                       |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C19  | R1         | The literature review would | We updated the Related Work section with more recent work  | Related Work |
|      |            | benefit from more recent    | on cascade structure, information diffusion, and newer     |              |
|      |            | information-spreading work. | social media platforms, including recent work on Bluesky.  |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C20  | R1         | Some notation was           | We reviewed the manuscript for undefined notation,         | Throughout   |
|      |            | unexplained and formatting  | inconsistent terminology, and formatting issues and        |              |
|      |            | was inconsistent.           | corrected these throughout.                                |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
| C21  | R1         | The analysis is limited to  | We agree that this is an important limitation of the       | Discussion   |
|      |            | a U.S. political context.   | present study. The revised Discussion states explicitly    |              |
|      |            |                             | that the findings are based on U.S. candidate-centered     |              |
|      |            |                             | political discussion during one election period and that   |              |
|      |            |                             | comparable analyses in other national contexts are needed  |              |
|      |            |                             | to establish how broadly the structural patterns           |              |
|      |            |                             | generalize.                                                |              |
+------+------------+-----------------------------+------------------------------------------------------------+--------------+
