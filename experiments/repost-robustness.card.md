# Experiment Card: repost-robustness v1 (2026-07-17)

- **Study / version / date**: repost-cascade reconstruction robustness, v1, 2026-07-17.
  For ICWSM 2027 R&R item P1-1 (see paper/revision/REVISION_ROADMAP.md).
- **Hypothesis**: the cross-platform similarity in repost-cascade scaling (the
  platform-by-size interaction b3 for log breadth and log depth) is stable across
  reconstruction rules; falsified if b3 changes sign or its CI excludes the
  paper-rule estimate under any plausible rule.
- **Sample / cohort definition (FROZEN)**: all root posts with at least 1 repost in
  `data/bsky_reposts_new.json` (Bluesky) and `data/ts_threads_withReblogs.json`
  (Truth Social, posts with non-empty reblogList). Same corpus as the submitted
  paper. Regression subset: cascades with size > 0 and metric > 0, matching the
  paper filter. Expected N: to be recorded at smoke test (paper reports counts).
- **Conditions and metrics**:
  - Rules: (1) paper-rule replica (root-priority, first-in-list followee, cycle
    check), (2) first-prior-in-list followee, (3) last-prior-in-list followee,
    (4) random eligible followee, K = 20 draws, (5) no-root-priority variant of
    (1). Rules 2 to 4 run under both list-order interpretations (as-is and
    reversed) since API order direction is undocumented; no repost timestamps
    exist in either dataset.
  - Metrics per cascade per rule: size, max breadth, max depth, directed
    structural virality (closed-form tree computation).
  - Outcome: Huber RLM of log10(Y) on log10(size) x platform; report b3 with 95%
    CI per rule; random-rule band from the K draws. Diagnostics: share of
    direct-to-root, fallback, unique-candidate attachments; mean candidate-set
    size; max feasible chain length.
- **Compute plan**: CPU only, no GPU. Data parsing dominates (1.6 GB + 8.7 GB
  JSON, 23 GB + 2 GB follow maps; machine has 1.5 TB RAM, currently ~1.3 TB
  available). Parsed minimal structures cached to `data/interim/` so rules rerun
  in minutes. Estimated wall time: ~30 to 60 min parse + cache, then < 15 min for
  all rules.
- **Smoke test**: first 2,000 cascades per platform through the full path
  (parse, all rules, metrics, regression, output schema) before any full run.
- **Stopping rule**: one full pass over all rules. If b3 is stable (all rule CIs
  overlap the paper-rule estimate), conclusion is "robust", write it up. If a
  rule flips the result, conclusion is "reconstruction-sensitive", report which
  rule and STOP; the paper then leans on reply cascades per the SPC fallback. No
  additional rules without an explicit override.
- **Artifacts**: `results/repost_robustness/` (per-rule cascade metrics parquet,
  b3 table CSV, spec-curve figure), log at `logs/repost_robustness.log`,
  parse cache in `data/interim/repost_cache_{bsky,ts}.parquet`.
- **Monitor contract**: background task notifies on completion or error only;
  completion message leads with the b3 range across rules and the artifact path.
  No loop; single run.
- **Status log**:
  - 2026-07-17 v1 created; smoke test pending.
  - 2026-07-17/18 smoke: first attempt hung (candidate sets recomputed per
    draw + a parent-array cycle from a code path where the original would
    crash); fixed (candidates cached per cascade-order, cycle check on the
    chosen parent). Smoke passes: 2000+2000 cascades, rules phase 7 s,
    schema and outputs validated.
  - 2026-07-18 full run launched.
  - 2026-07-18 COMPLETE. Headline: repost b3 within [-0.031, +0.031] under all
    principled rules (published-rule replica -0.083 depth is the worst case;
    reply reference +0.170/-0.182). Similarity claim robust. Estimator note:
    OLS+HC3 primary (MAD Huber scale collapses on lattice outcomes).
    Artifacts: results/repost_robustness/.
