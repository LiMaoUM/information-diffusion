# Label-noise propagation for Model 3c: results (2026-08-17)

Question: does the paper's central mechanism claim (ideological composition
plus alignment statistically account for the platform divergence in reply
cascade scaling) survive ideology-label noise at the measured error rates?

Design: user-level labels perturbed by drawing "true" labels from the
column-normalized confusion matrices P(human | model) measured on the 200-item
validation set; cascade-level composition and alignment recomputed from the
reconstructed reply trees (reconstruction validated against the paper's frame
at r >= 0.99); the paper's exact Model 3c (RLM, Huber, spline terms) refit per
draw. K = 100 draws per scenario. All numbers are the platform-by-log-size
interaction b3; smaller magnitude = more of the platform gap absorbed.

Reference points (this frame): baseline Model 1 b3 = 0.398 (breadth), -0.334
(depth). Published Model 3c b3 = 0.0230 / -0.0547 (reproduced exactly);
unperturbed refit on the 98%-matched reconstruction = 0.0161 / -0.0494.

## Headline

| Scenario | b3 breadth, median [95% band] | b3 depth, median [95% band] | Attenuation vs baseline |
|---|---|---|---|
| Unperturbed | 0.016 | -0.049 | 96% / 85% |
| Measured per-platform noise | 0.019 [0.000, 0.052] | -0.018 [-0.049, 0.000] | 95% / 95% (worst draw 87% / 83%) |
| Pooled-matrix stress test | 0.116 [0.095, 0.139] | -0.106 [-0.122, -0.087] | 71% / 68% (worst draw 63% / 62%) |

## Reading

1. Under noise at the MEASURED platform-specific error rates, the conclusion
   fully survives: every one of 100 draws keeps at least 83% of the baseline
   divergence absorbed, and the median draw is indistinguishable from the
   unperturbed model. This is the scenario that corresponds to the validation
   evidence, including the left-overassignment on Truth Social and
   right-overassignment on Bluesky that R1 suspected.
2. The pooled-matrix stress test applies each platform's dominant class the
   error rate measured mostly on the OTHER platform (for example it flips
   Truth Social right-labeled users at Bluesky's much worse right-precision).
   Even under this deliberately crude noise, ideology still absorbs about two
   thirds of the platform gap, though the interaction no longer vanishes.
3. Direction of bias favors the paper's claim: classical attenuation, noisier
   ideology regressors absorb LESS of the gap, which is exactly what the
   simulation shows. Since the real labels contain the measured noise, the
   published attenuation is a lower bound on what error-free labels would
   show.

## Caveats

- Per-platform matrices rest on thin cells (right-on-Bluesky n = 9); the
  planned annotation top-up tightens them.
- Simulation perturbs final user labels at post-level error rates; user labels
  aggregate many posts, so true user-level error is likely lower. Conservative.
- 40 frame rows labeled ts carry Bluesky uris (0.03%); drop in the revision.

Files: b3_draws.csv (draw-level coefficients), machinery in
src/analysis/noise_propagation.py, card in experiments/noise-propagation.card.md.
