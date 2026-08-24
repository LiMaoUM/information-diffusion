# CLAUDE.md (project)

## What this is

ICWSM 2026 paper project comparing information-diffusion cascade structure on
Truth Social vs. Bluesky. See README.md for layout and pipeline.

## Ops facts

- Python env: `uv venv` + `requirements.txt`. The old `.venv` here had no pip
  (see freeze_err.log pattern); prefer recreating with uv over reusing it.
- `data/` is ~45 GB of raw, PII-bearing social media JSON (post text, handles,
  follow graphs, coder-named eval spreadsheets). Git-ignored. Never commit,
  publish, or paste its contents.
- `src/matching.ipynb` reads from the sibling repo `../echo-chamber/data/`,
  which is not part of this repository.
- Motif counting: CUDA, build/run via `src/motif/src/motif.sh`
  (`motif_count_node.cu`). Binaries are git-ignored, rebuild as needed.
- Canonical reply-stat threshold appears to be the `_06` / `_06_new2` variants
  of files in `data/` and `src/motif/data/` (older `_05`..`_09` variants are
  historical sweeps).

## Version-control policy

- Large regenerable artifacts (edge-list CSVs, jsonl, parquet, edgelists,
  compiled binaries) are git-ignored. Do not force-add them.
- Git history before 2026-07-17 still contains ~250 MB of data blobs
  (90 MB edge CSVs x3, jsonl). Purging requires git filter-repo plus a force
  push; only with Mao's explicit go-ahead.
- Notebooks are committed WITH outputs (comparison_reply_combined.ipynb is
  ~50 MB, right at GitHub's warning threshold). Do not clear outputs without
  asking; they are the record of results. If it exceeds 100 MB, GitHub will
  reject the push.

## Repost reconstruction facts (discovered 2026-07-17, matter for the R&R)

- NO per-repost timestamps exist on either platform: `reposts` (bsky) and
  `reblogList` (TS) items are account/profile objects; their `createdAt` is
  account creation. The appendix's "timestamped repost data, t_j < t_i" does
  not match the data; list order is the only temporal signal.
- The shipped `build_repost_graph` (cascade_analysis.py) differs from the
  appendix text AND has two quirks that shaped published numbers:
  (a) after a node links under author A, `linked_users[A]` is overwritten with
  the child's id, so later followers of A chain under the previous attacher;
  (b) `unlinked_nodes.remove()` during iteration skips the next element,
  sending it to root fallback. Also root-priority: reposters following the
  root author always attach to the root (the appendix describes root as
  fallback only).
- Follow maps: use `data/bsky_followings.json` (id-only, 1.5 GB), not the
  23 GB `bsky_follows.json` (profile objects). TS: `ts_user_following_map.json`.
- Robustness suite: `src/analysis/repost_robustness.py` (card in
  `experiments/repost-robustness.card.md`); caches in `data/interim/`,
  outputs in `results/repost_robustness/`.

## Validation data (settled 2026-08-24)

- The canonical stance validation is `src/val_ideology.csv`, NOT
  `data/val_chen.csv`. The two share the same 200 posts and the same
  Annotator 1, but Annotator 2 differs on 15 items. Using the wrong file
  inflates human-human kappa to 0.89 and makes minority-class precision look
  much worse than it is.
- With the correct file the published appendix kappas are essentially
  confirmed (0.77 / 0.61 / 0.74 recomputed vs 0.78 / 0.64 / 0.73 published),
  so those numbers do NOT need correcting in the revision.
- Per Ceren (2026-08-17): do not stratify-upsample the rare classes, since a
  reviewer can call that cheating. Report the numbers we have with bootstrap
  intervals; if they fluctuate, that is the argument for collecting a modest
  amount more, not for reweighting what we have.

## Paper status (2026-07)

- ICWSM 2027, paper 1217, **R&R received 2026-07-17, revision due 2026-09-15**
  (final accept/reject, color-highlighted changes + response document).
- Reviews: `paper/reviews/2026-07_icwsm2027_round1_decision.md`.
  Roadmap + response skeleton: `paper/revision/`.
- Submitted version: `paper/AnonymousSubmission/LaTeX/anonymous-submission-latex-2026.tex`.
  `RestructuredLatex/` is an alternate draft; `CameraReady/` is still blank template.
- Known factual issue to fix in revision: main text claims Huber-loss robust
  regression (line ~258), appendix claims OLS + HC3 (line ~765); actual code
  `src/modeling/model_R.R` uses lmrob (KS2014) and rlm with Huber psi.
