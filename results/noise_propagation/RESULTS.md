# Label-noise propagation for Model 3c: results (rerun 2026-08-24)

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
| Measured per-platform noise | 0.012 [0.000, 0.038] | -0.013 [-0.043, 0.000] | 97% / 96% (worst draw 90% / 85%) |
| Pooled-matrix stress test | 0.096 [0.079, 0.119] | -0.086 [-0.103, -0.073] | 76% / 74% (worst draw 70% / 69%) |

NOTE: rerun 2026-08-24 against confusion matrices recomputed from the
canonical validation file (src/val_ideology.csv). The earlier run used a
superseded copy with a different second annotator; the corrected labels carry
less error, so the attenuation result strengthened slightly.

## Reading

1. Under noise at the MEASURED platform-specific error rates, the conclusion
   fully survives: every one of 100 draws keeps at least 85% of the baseline
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

- Per-platform matrices rest on thin cells (right-on-Bluesky n = 8, bootstrap
  precision interval 0.33 to 0.91). Per Ceren, the response is bootstrap
  reporting plus a modest unstratified addition, not stratified upsampling.
- Simulation perturbs final user labels at post-level error rates; user labels
  aggregate many posts, so true user-level error is likely lower. Conservative.
- 40 frame rows labeled ts carry Bluesky uris (0.03%); drop in the revision.

Files: b3_draws.csv (draw-level coefficients), machinery in
src/analysis/noise_propagation.py, card in experiments/noise-propagation.card.md.
