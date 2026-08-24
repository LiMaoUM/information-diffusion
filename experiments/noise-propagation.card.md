# Experiment Card: noise-propagation v1 (2026-08-17)

- **Study / version / date**: ideology label-noise propagation, v1, 2026-08-17.
  R&R item C6 (tracker) / P1-3 (roadmap).
- **Hypothesis**: the Model 3c conclusion (platform-by-size interaction near
  zero once composition and alignment are included) survives label noise at
  the measured error rates; falsified if the perturbed Model 3c interaction
  distribution moves substantially toward the baseline divergence (less than
  80 percent attenuation in more than 5 percent of draws).
- **Sample / cohort definition (FROZEN)**: reply cascades in
  `data/combined_reply_stats_06.csv` (the notebook's Model 3 frame), trees
  matched to reconstructed author/edge lists from `data/bsky_threads.json`
  and `data/ts_threads_withReblogs.json`. Match rate reported; reconstruction
  validated against the frame's stored author_* columns before any
  perturbation (require correlation >= 0.95, else stop and diagnose).
- **Conditions and metrics**: scenarios = per-platform confusion matrix and
  pooled matrix, both column-normalized P(human | model) from
  `results/ideology_validation/confusion_*.csv`; K = 100 draws each; outcomes
  log_breadth and log_depth with the notebook's exact Model 3c formula (RLM,
  Huber). Metric: platform-by-log-size coefficient per draw, compared with
  the unperturbed refit and the baseline model's interaction.
- **Compute plan**: CPU only; one-time parse of thread JSONs (~4 GB, minutes,
  cached to data/interim); ~400 RLM fits at seconds each. Expected < 30 min.
- **Smoke test**: prepare on both platforms, validate correlations, K = 2
  draws end to end before the full K = 100.
- **Stopping rule**: one full pass. Robust or not, report the distribution
  and stop; no scenario shopping beyond the two pre-registered matrices.
- **Artifacts**: `results/noise_propagation/` (draw-level coefficients CSV,
  summary MD), cache `data/interim/trees_{nodes,edges}.parquet`, log
  `logs/noise_propagation.log`.
- **Monitor contract**: background completion notification only; completion
  message leads with the attenuation percentage range and artifact path.
- **Status log**:
  - 2026-08-17 v1 created.
  - 2026-08-17 COMPLETE (after 3 false starts: frame index dtype mangling,
    closure pickling, background cwd). Headline: per-platform measured noise
    keeps >=83% attenuation in all 100 draws (median ~95%); pooled stress
    test keeps ~62-71%. Conclusion robust under measured error rates.
    Artifacts: results/noise_propagation/.
  - 2026-08-24 RERUN against corrected confusion matrices (Mao identified
    src/val_ideology.csv as canonical; the prior run used a superseded copy).
    Result strengthened: worst draw now keeps 85% attenuation (was 83%),
    median ~96%; pooled stress test ~70% (was ~two thirds).
