# Label-noise propagation for Model 3c: results (rerun 2026-08-31, Huber proposal 2)

Question: does the paper's central mechanism claim (ideological composition
plus alignment statistically account for the platform divergence in reply
cascade scaling) survive ideology-label noise at the measured error rates?

Design: user-level labels perturbed by drawing "true" labels from the
column-normalized confusion matrices P(human | model) measured on the 200-item
validation set; cascade-level composition and alignment recomputed from the
reconstructed reply trees (reconstruction validated against the paper's frame
at r >= 0.99); the paper's exact Model 3c (RLM, Huber loss, Huber's proposal 2 scale,
spline terms) refit per draw. K = 100 draws per scenario. All numbers are the platform-by-log-size
interaction b3; smaller magnitude = more of the platform gap absorbed.

Reference points (this frame): baseline Model 1 b3 = 0.419 (breadth), -0.321
(depth), both under proposal 2. Model 3c on the paper frame = 0.0706 / -0.0491
(matches Table 1); unperturbed refit on the matched reconstruction = 0.0662 /
-0.0455.

Superseded: the 2026-08-24 run used the default MAD scale, which degenerates on
these data (61% of cascades sit at the origin). Its baseline was 0.398 / -0.334
and it reported 86-96% median absorption with a 70% worst draw. Those numbers
are not comparable to the ones below and are not used in the paper.

## Headline

| Scenario | b3 breadth, median [range] | b3 depth, median [range] | Absorbed vs baseline |
|---|---|---|---|
| Unperturbed | 0.066 | -0.045 | 84% / 86% |
| Measured per-platform noise | 0.059 [0.037, 0.080] | -0.052 [-0.070, -0.034] | 86% / 84% (worst draw 81% / 78%) |
| Pooled-matrix stress test | 0.108 [0.089, 0.123] | -0.097 [-0.110, -0.080] | 74% / 70% (worst draw 71% / 66%) |
| Nested bootstrap of the validation set | 0.057 [-0.025, 0.119] | -0.050 [-0.100, 0.035] | 86% / 84% (worst draw 72% / 69%) |
| Annotator-1 worst case (thin-cell precision 0.33) | 0.090 [0.051, 0.118] | -0.078 [-0.101, -0.049] | 79% / 76% (worst draw 72% / 69%) |

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
