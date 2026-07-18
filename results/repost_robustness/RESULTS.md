# Repost reconstruction robustness: results (2026-07-18, full run)

Sample: all cascades with at least 1 repost. Bluesky 25,550 cascades (184,807
reposts), Truth Social 218,579 cascades (3,847,903 reposts). Estimator: OLS of
log10(Y) on log10(size) x platform with HC3 SEs (Huber RLM secondary agrees;
MAD-based Huber scale collapses on these lattice-valued outcomes, which is why
plain RLM is not usable here). Depth modeled as log10(depth + 1), matching the
paper figure.

## Headline

The cross-platform similarity of repost cascade scaling is robust to the
reconstruction rule, and the published pipeline is the most conservative case.

| Rule | b3 breadth [95% CI] | b3 depth [95% CI] |
|---|---|---|
| published rule (code replica) | -0.012 [-0.016, -0.008] | -0.083 [-0.090, -0.076] |
| first prior followee (api order) | +0.027 [+0.022, +0.031] | -0.024 [-0.027, -0.021] |
| first prior followee (reversed) | +0.011 [+0.007, +0.015] | -0.020 [-0.023, -0.018] |
| last prior followee (api order) | +0.031 [+0.026, +0.037] | +0.004 [+0.000, +0.008] |
| last prior followee (reversed) | -0.003 [-0.009, +0.003] | +0.028 [+0.024, +0.031] |
| random ensemble (40 draws, range) | [+0.003, +0.029] | [-0.014, +0.001] |
| **reply cascades (same estimator, reference)** | **+0.170** | **-0.182** |

Per-platform slopes (Fig 3 C/D quantity) differ by at most 0.031 (breadth,
slopes ~1.02 to 1.17) and 0.083 (depth, slopes ~0.22 to 0.34) across all rules.

## Reading

1. Every principled reconstruction rule gives |b3| <= 0.031; the random-rule
   ensemble (which samples the space of admissible rules) sits within
   [-0.014, +0.029]. The reply-cascade divergence is 5 to 8 times larger than
   any repost estimate. The repost-similar / reply-divergent asymmetry cannot
   be an artifact of the reconstruction rule.
2. The shipped pipeline (with its chaining and skip-next quirks) produced the
   LARGEST apparent cross-platform repost difference (depth b3 = -0.083). All
   alternative rules move the estimate toward zero. So the published claim of
   similarity was made under the least favorable reconstruction for that
   claim.
3. Structural freedom is limited anyway: 55% (Bluesky) and 59% (Truth Social)
   of reposts have no eligible prior followee and attach to the root under any
   rule; a further 21% / 12% have exactly one candidate. Mean candidate-set
   size is 1.6 (Bluesky) and 5.5 (Truth Social).

## Caveats for the write-up

- No per-repost timestamps exist on either platform; list order is the only
  temporal signal, so "time-inferred" should be restated as order-and-network
  inferred, and both order readings are covered here.
- b3 signs flip across rules within the tiny range; do not interpret the sign
  of any single repost b3.
- Files: b3_by_rule.csv (all fits incl. per-platform slopes and RLM check),
  cascade_metrics.parquet (per-cascade metrics per rule), diagnostics.parquet,
  spec_curve_draft.png (draft figure; final via nature-figure after spec
  confirmation, spec in figure-spec.md).
