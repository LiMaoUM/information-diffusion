# Response to Reviewers (skeleton)

Paper 1217: "Depth, Breadth, and Bias: Structural Diffusion of Political Content on Divergent Platforms"

Dear Editors and Reviewers,

We thank the SPC and the three reviewers for their careful and constructive reviews. We were encouraged that all reviewers found the cross-platform question timely, the repost/reply distinction valuable, and the central asymmetry interesting. The revision addresses every point raised. In summary, the major changes are:

1. [PLACEHOLDER: repost robustness suite or scope narrowing, per decision]
2. [PLACEHOLDER: expanded ideology-label validation and sensitivity analysis]
3. [PLACEHOLDER: explicit cascade definitions and unit-of-analysis clarification]
4. [PLACEHOLDER: reframed associational language and endogeneity discussion]
5. [PLACEHOLDER: new descriptive statistics and motif count tables]
6. [PLACEHOLDER: corrected estimator description; figure and editorial cleanup]

All changes are highlighted in [color] in the revised manuscript. Line numbers below refer to the revised version.

---

## Response to the SPC meta-review

### SPC-1: Validate or revise the repost cascade analysis; qualify "empirical"
**Response**: [PLACEHOLDER: describe alternative reconstruction rules tested, stability of scaling results, follow-timing error bound, terminology change to "reconstructed/inferred cascades"]
**Changes made**: [Section 3.2 rewritten; new Appendix X; Figures X]

### SPC-2: Clarify cascade definitions
**Response**: [PLACEHOLDER: nodes are posts/users per cascade type; treatment of repeated users; robustness under the alternative representation]
**Changes made**: [Section 3.2; new schematic Figure X]

### SPC-3: Strengthen ideology-label validation
**Response**: [PLACEHOLDER: per-class per-platform metrics, center category definition, label-noise sensitivity, reconciliation with prior Bluesky ideology estimates]
**Changes made**: [Section 3.x; Appendix X, Tables X]

### SPC-4: Reframe explanatory claims; endogeneity of alignment
**Response**: [PLACEHOLDER: associational language throughout; endogeneity paragraph; optional early-cascade alignment analysis]
**Changes made**: [Abstract, Sections 4.2 and 5]

### SPC-5: Improve reporting of data and robustness
**Response**: [PLACEHOLDER: descriptive statistics table, cascade size counts, motif counts, null-model dependence explanation]
**Changes made**: [Table X, Appendix X]

### SPC-6: Resolve regression estimator inconsistency
**Response**: The main analysis uses robust regression (M-estimation with Huber psi; lmrob with the KS2014 setting for confirmation). The appendix paragraph describing OLS with HC3 standard errors was a leftover from an earlier specification and has been corrected to describe the estimator actually used. [ADJUST if HC3 was genuinely used for a secondary model.]
**Changes made**: [Appendix section X]

### SPC-7: Sharpen implications and limitations
**Response**: [PLACEHOLDER: what generalizes vs what is context-bound; theoretical and practical meaning of broad/shallow vs narrow/deep]
**Changes made**: [Section 5, Limitations]

---

## Response to Reviewer 1

We thank the reviewer for the detailed and expert review, and for noting the novelty of the cross-platform analysis, the balanced tone, and the value of the motif analysis.

### R1-1: Repost reconstruction realism and robustness; "empirical" terminology
**Response**: [see SPC-1; PLACEHOLDER for specifics]
**Changes made**: [...]

### R1-2: Ideology labelling validity; discrepancy with prior literature (Bovet et al.)
**Response**: [see SPC-3; PLACEHOLDER: explicit comparison with the Bovet team's Bluesky estimates and why our Biden/Trump-mention sample differs from platform-wide samples]
**Changes made**: [...]

### R1-3: Figures need cleaning up
**Response**: [PLACEHOLDER: list regenerated figures]
**Changes made**: [...]

### R1-4: Raw motif counts
**Response**: [PLACEHOLDER: motif count table added]
**Changes made**: [Table X]

### R1-5: Motif non-independence in the randomization
**Response**: [PLACEHOLDER: null model preserves overlapping-motif dependence; z-scores compare observed vs randomized under identical counting]
**Changes made**: [...]

### R1-6: Topic modeling details and BERTopic outlier share
**Response**: [PLACEHOLDER]
**Changes made**: [...]

### R1-7: Dataset descriptives; readable cascade counts
**Response**: [see SPC-5]
**Changes made**: [...]

### R1-8: Share of two-user back-and-forth reply chains
**Response**: [PLACEHOLDER: computed; X% on Truth Social, Y% on Bluesky; implications for depth interpretation]
**Changes made**: [...]

### R1-9: Alignment results exposition
**Response**: [PLACEHOLDER: rewritten with worked example and definition box]
**Changes made**: [...]

### R1-10: Influencer analysis justification; top n% on both platforms
**Response**: [PLACEHOLDER: robustness with top 1/5/10% on both platforms]
**Changes made**: [...]

### R1-11: US-centric focus
**Response**: [PLACEHOLDER: acknowledged in limitations; future work]
**Changes made**: [...]

### R1-12: Notation, LaTeX indentation, older literature
**Response**: [PLACEHOLDER: notation pass, formatting fixes, added 2023 to 2026 references]
**Changes made**: [...]

---

## Response to Reviewer 2

We thank the reviewer for highlighting the amplification vs conversation distinction and the repost/reply asymmetry as the paper's strongest contribution.

### R2-1: Endogeneity of ideology and alignment; causal language too strong
**Response**: [see SPC-4]
**Changes made**: [...]

### R2-2: Cascade node definition; repeated users
**Response**: [see SPC-2]
**Changes made**: [...]

### R2-3: Repost path inference assumptions
**Response**: [see SPC-1]
**Changes made**: [...]

### R2-4: Generalizability of the one-month Biden/Trump window
**Response**: [see SPC-7]
**Changes made**: [...]

### R2-5: Stance validation sample size; class and platform-specific performance; center category
**Response**: [see SPC-3; PLACEHOLDER: expanded validation set size and per-class results]
**Changes made**: [...]

### R2-6: Huber vs HC3 inconsistency
**Response**: [see SPC-6]
**Changes made**: [...]

---

## Response to Reviewer 3

We thank the reviewer for the positive assessment and for the constructive points on alignment, reconstruction timing, and definitions.

### R3-1: Alignment ratio and cascade structure may influence each other
**Response**: [see SPC-4; PLACEHOLDER: note the reviewer's deep vs star opportunity-structure point explicitly in the endogeneity paragraph]
**Changes made**: [...]

### R3-2: Follow edge may postdate the repost; systematic platform differences in this error
**Response**: [PLACEHOLDER: quantify or bound; discuss collection timing for each platform's follow network]
**Changes made**: [...]

### R3-3: Repeated users in reply cascades
**Response**: [see SPC-2]
**Changes made**: [...]

### R3-4: Implications and limitations
**Response**: [see SPC-7]
**Changes made**: [...]
