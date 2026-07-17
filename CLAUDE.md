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

## Paper status (2026-07)

- Anonymous submission complete: `paper/AnonymousSubmission/LaTeX/anonymous-submission-latex-2026.tex`.
- `RestructuredLatex/icwsm2026-restructured.tex` is an alternate draft with
  extended appendices; which one is final is undecided.
- `CameraReady/LaTeX/` is still the blank AAAI template, no paper content yet.
